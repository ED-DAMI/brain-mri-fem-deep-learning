"""Dependency-light preprocessing primitives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Load a NIfTI volume, preserving affine and voxel spacing."""
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    return data, image.affine.copy(), tuple(float(x) for x in image.header.get_zooms()[:3])


def normalize_volume(volume: np.ndarray, method: str = "zscore_nonzero") -> np.ndarray:
    """Normalize an MRI volume using non-zero brain voxels."""
    values = np.asarray(volume, dtype=np.float32)
    mask = values != 0
    result = np.zeros_like(values)
    if not np.any(mask):
        return result
    if method == "zscore_nonzero":
        mean = float(values[mask].mean())
        std = float(values[mask].std())
        result[mask] = (values[mask] - mean) / max(std, np.finfo(np.float32).eps)
    elif method == "minmax_percentile":
        low, high = np.percentile(values[mask], [1, 99])
        clipped = np.clip(values, low, high)
        result[mask] = (clipped[mask] - low) / max(high - low, np.finfo(np.float32).eps)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return result


def patient_splits(
    patient_ids: list[str], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> dict[str, list[str]]:
    """Create deterministic, disjoint patient-level train/validation/test splits."""
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must contain three values summing to one")
    rng = np.random.default_rng(seed)
    ids = list(dict.fromkeys(patient_ids))
    rng.shuffle(ids)
    n = len(ids)
    first = int(n * ratios[0])
    second = first + int(n * ratios[1])
    return {"train": ids[:first], "val": ids[first:second], "test": ids[second:]}


def save_splits(splits: dict[str, list[str]], path: str | Path) -> None:
    """Persist split assignments as JSON."""
    Path(path).write_text(json.dumps(splits, indent=2), encoding="utf-8")
