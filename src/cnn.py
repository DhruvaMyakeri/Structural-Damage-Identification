"""Transfer learning with MobileNetV2 and InceptionV3.

The paper froze an ImageNet network and trained a new fully connected layer
with a softmax. Cross-entropy in PyTorch is that softmax plus the log loss.
Torchvision's ImageNet weights expect mean/std normalisation, not the paper's
TensorFlow map of pixel/128 - 1. Using the matching preprocess is what makes
the downloaded weights meaningful.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.data import open_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = open_image(self.paths[index])
        return self.transform(image), int(self.labels[index])


def build_transform(image_size: int, augment: bool) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size))]
    if augment:
        # Flip and a small shift. The paper also tried zoom, and showed that
        # strong zoom can invent pictures that are not real structures.
        steps.append(transforms.RandomHorizontalFlip())
        steps.append(transforms.RandomAffine(degrees=0, translate=(0.12, 0.12)))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def make_loader(paths, labels, image_size: int, batch_size: int, augment: bool, shuffle: bool) -> DataLoader:
    dataset = PathDataset(paths, labels, build_transform(image_size, augment))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_mobilenet() -> tuple[nn.Module, int]:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    return model, 224


def build_inception() -> tuple[nn.Module, int]:
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            parameter.requires_grad = False
    return model, 299


def unfreeze_last_mobilenet_blocks(model: nn.Module, blocks: int = 4) -> None:
    """Train the last few convolutional blocks as well as the new layer.

    The paper's frozen Inception run was limited by bias, so they trained
    extra layers. Those extra parameters can then overfit. This switch is the
    same idea on MobileNetV2, which is the network we can fine-tune quickly.
    """
    children = list(model.features.children())
    for child in children[-blocks:]:
        for parameter in child.parameters():
            parameter.requires_grad = True


def _logits(output) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    return output


@torch.no_grad()
def predict_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truths: list[np.ndarray] = []
    guesses: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(images)
        pred = _logits(output).argmax(dim=1)
        truths.append(labels.numpy())
        guesses.append(pred.cpu().numpy())
    return np.concatenate(truths), np.concatenate(guesses)


def accuracy_of(model, loader, device) -> float:
    truth, guess = predict_labels(model, loader, device)
    return float((truth == guess).mean())


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    log: Callable[[str], None] = print,
) -> dict:
    """Minimise cross-entropy. Keep the weights with the best validation accuracy."""
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history = {"train_loss": [], "val_accuracy": []}
    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(_logits(model(images)), labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total_loss += float(loss.detach()) * labels.size(0)
            seen += labels.size(0)
        train_loss = total_loss / max(seen, 1)
        val_acc = accuracy_of(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        log(f"  epoch {epoch}/{epochs} loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    history["best_val_accuracy"] = best_val
    return history
