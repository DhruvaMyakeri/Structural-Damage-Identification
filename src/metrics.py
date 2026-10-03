"""Accuracy and the confusion matrix, laid out the way the paper wrote it.

Rows are the true label, columns are the prediction, order [undamaged, damaged].
The paper's matrix [448, 86, 117, 349] uses that order.
"""

from __future__ import annotations

import numpy as np


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    accuracy = float((y_true == y_pred).mean())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision_damaged": float(precision),
        "recall_damaged": float(recall),
        "f1_damaged": float(f1),
        "confusion_test": [tn, fp, fn, tp],
    }
