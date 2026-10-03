"""Figures for the write-up and the slides. Numbers come from metrics.json."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def accuracy_bars(models: list[dict], path: Path) -> None:
    _style()
    names = [item["short_name"] for item in models]
    train = [100 * item["train_accuracy"] for item in models]
    test = [100 * item["test_accuracy"] for item in models]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.bar(x - width / 2, train, width, label="Train", color="#4A5560")
    ax.bar(x + width / 2, test, width, label="Test", color="#C45C26")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, names, rotation=18, ha="right")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="#8A9099", linewidth=0.8, linestyle="--")
    ax.legend(frameon=False)
    ax.set_title("Train accuracy against held-out test accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def learning_curve(curve: dict, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(curve["sizes"], [100 * v for v in curve["train_accuracy"]], marker="o", color="#4A5560", label="Train")
    ax.plot(curve["sizes"], [100 * v for v in curve["val_accuracy"]], marker="o", color="#C45C26", label="Validation")
    ax.set_xlabel("Training images")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 105)
    ax.set_title("Logistic regression as the training set grows")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def epoch_curves(histories: dict, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    colors = ["#1F4E5F", "#C45C26", "#5E7A6B", "#4A5560"]
    for color, (name, history) in zip(colors, histories.items()):
        epochs = np.arange(1, len(history["val_accuracy"]) + 1)
        ax.plot(epochs, [100 * v for v in history["val_accuracy"]], marker="o", color=color, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Validation accuracy while the new layer trains")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def gamma_curve(sweep: dict, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(sweep["gammas"], [100 * v for v in sweep["val_accuracy"]], marker="o", color="#C45C26")
    ax.set_xscale("log")
    ax.set_xlabel("RBF gamma")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Choosing gamma for the Inception feature SVM")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def confusion_figure(matrix: list[int], path: Path, title: str) -> None:
    _style()
    tn, fp, fn, tp = matrix
    grid = np.array([[tn, fp], [fn, tp]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(grid, cmap="Oranges")
    ax.set_xticks([0, 1], ["Pred undamaged", "Pred damaged"])
    ax.set_yticks([0, 1], ["True undamaged", "True damaged"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{int(grid[row, col])}", ha="center", va="center", color="#1A1A1A", fontsize=12)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def misclassified_grid(records: list[dict], path: Path) -> None:
    if not records:
        return
    columns = min(3, len(records))
    rows = int(np.ceil(len(records) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.1, rows * 3.3))
    axes = np.atleast_1d(axes).reshape(rows, columns)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, record in zip(axes.ravel(), records):
        image = Image.open(record["path"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(record["caption"], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
