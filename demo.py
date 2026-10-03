"""Predict damaged or undamaged for one photo, or for a few test photos.

Examples:

    python demo.py
    python demo.py --image path\\to\\photo.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from src.cnn import build_inception, build_mobilenet, build_transform, unfreeze_last_mobilenet_blocks
from src.data import image_name, list_split_zip, open_image, pixels_to_vectors, stratified_take
from src.hybrid import inception_mixed6e_features

ROOT = Path(__file__).resolve().parent
NAMES = {0: "undamaged", 1: "damaged"}


def load_best():
    payload = json.loads((ROOT / "results" / "best.json").read_text(encoding="utf-8"))
    return payload


def _cnn_model(payload: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if payload["arch"] == "inception_v3":
        model, image_size = build_inception()
    else:
        model, image_size = build_mobilenet()
    if "unfrozen" in payload["name"].lower():
        unfreeze_last_mobilenet_blocks(model)
    state = torch.load(ROOT / payload["weights"], map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, image_size, device


def predict_cnn_batch(paths, payload: dict) -> list[tuple[str, float]]:
    model, image_size, device = _cnn_model(payload)
    transform = build_transform(image_size, augment=False)
    results = []
    for path in paths:
        batch = transform(open_image(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(batch)
            if hasattr(logits, "logits"):
                logits = logits.logits
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        label = int(probabilities.argmax())
        results.append((NAMES[label], float(probabilities[label])))
    return results


def predict_hybrid_batch(paths, payload: dict) -> list[tuple[str, float]]:
    """One Inception pass per photo, then the saved RBF SVM on Mixed_6e.

    The number printed with the label is the SVM margin, not a probability.
    A positive margin votes for damaged (class 1) under scikit-learn's
    decision_function for this two-class fit.
    """
    from torch.utils.data import DataLoader

    from src.cnn import PathDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reuse the feature helper by pointing it at an in-memory list of paths.
    labels = np.zeros(len(paths), dtype=np.int64)
    loader = DataLoader(PathDataset(paths, labels, build_transform(299, False)), batch_size=8, shuffle=False)
    features, _ = inception_mixed6e_features(loader, device)
    model = joblib.load(ROOT / payload["joblib"])
    guesses = model.predict(features)
    margins = model.decision_function(features)
    results = []
    for guess, margin in zip(guesses, margins):
        label = int(guess)
        # decision_function is positive for class 1. Flip the sign when the
        # predicted class is 0 so the printed margin is in favour of the call.
        signed = float(margin) if label == 1 else float(-margin)
        results.append((NAMES[label], signed))
    return results


def predict_classical_batch(paths, payload: dict) -> list[tuple[str, float | None]]:
    model = joblib.load(ROOT / payload["joblib"])
    features = pixels_to_vectors(list(paths), size=32)
    guesses = model.predict(features)
    probabilities = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    results = []
    for index, guess in enumerate(guesses):
        label = int(guess)
        score = None if probabilities is None else float(probabilities[index, label])
        results.append((NAMES[label], score))
    return results


def sample_test_images(n: int = 4) -> list[tuple]:
    paths, labels = list_split_zip("test")
    paths, labels = stratified_take(paths, labels, max(1, n // 2), seed=7)
    return list(zip(paths, labels.tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a structural image as damaged or undamaged.")
    parser.add_argument("--image", type=Path, help="Path to a .jpg or .png. If omitted, four test photos are used.")
    args = parser.parse_args()
    payload = load_best()
    print(f"Model: {payload['name']}")
    print(
        "Chosen by validation accuracy "
        f"({100 * payload['val_accuracy']:.2f}%), "
        f"test accuracy {100 * payload['test_accuracy']:.2f}%"
    )

    if args.image:
        items = [(args.image, None)]
    else:
        items = sample_test_images(4)

    paths = [path for path, _ in items]
    if payload["arch"] in {"mobilenet_v2", "inception_v3"}:
        scored = predict_cnn_batch(paths, payload)
        detail = "softmax confidence"
    elif payload["arch"] == "hybrid_svm":
        scored = predict_hybrid_batch(paths, payload)
        detail = "SVM margin for the predicted class"
    else:
        scored = predict_classical_batch(paths, payload)
        detail = "class probability"

    for (path, truth), (guess, score) in zip(items, scored):
        extra = f", {detail} {score:.2f}" if score is not None else ""
        truth_text = f", true label {NAMES[truth]}" if truth is not None else ""
        print(f"{image_name(path)}: {guess}{extra}{truth_text}")


if __name__ == "__main__":
    main()
