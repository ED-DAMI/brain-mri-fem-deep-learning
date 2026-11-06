"""Binary segmentation metrics."""

from __future__ import annotations

import numpy as np


def dice(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Return the Sørensen-Dice score."""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    intersection = np.count_nonzero(pred & truth)
    denominator = pred.sum() + truth.sum()
    return float((2 * intersection + eps) / (denominator + eps))


def iou(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Return intersection over union."""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    union = np.count_nonzero(pred | truth)
    return float((np.count_nonzero(pred & truth) + eps) / (union + eps))


def precision_recall(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    """Return binary precision and recall."""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    tp = np.count_nonzero(pred & truth)
    return float((tp + eps) / (pred.sum() + eps)), float((tp + eps) / (truth.sum() + eps))
