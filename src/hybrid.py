"""An SVM on a middle layer of pretrained InceptionV3.

The paper's point: ImageNet's last layer is tuned for a thousand everyday
categories, and buildings are rare in that set, so an earlier layer may be a
better feature source. Their TensorFlow graph had a layer numbered 288.
Torchvision exposes named blocks instead. Mixed_6e is a late-middle block
(the network then still has Mixed_7a, Mixed_7b, and Mixed_7c), which is the
same idea: general visual features, pooled to one vector per image.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.cnn import _logits, build_inception


@torch.no_grad()
def inception_mixed6e_features(
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model, _ = build_inception()
    model.to(device)
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["pooled"] = output.mean(dim=(2, 3))

    handle = model.Mixed_6e.register_forward_hook(hook)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    try:
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                _logits(model(images))
            features.append(captured["pooled"].float().cpu().numpy())
            labels.append(batch_labels.numpy())
    finally:
        handle.remove()
    return np.concatenate(features), np.concatenate(labels)


def parameter_count_note(model: nn.Module) -> str:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return f"trainable {trainable:,} of {total:,}"
