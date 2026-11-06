"""Marching-cubes reconstruction in physical coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.measure import marching_cubes


def mask_to_surface(mask: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> dict[str, Any]:
    """Extract vertices and triangular faces from a binary mask."""
    binary = np.asarray(mask, dtype=np.float32)
    if binary.ndim != 3:
        raise ValueError("mask must be three-dimensional")
    if binary.min() == binary.max():
        raise ValueError("mask must contain both background and foreground")
    vertices, faces, normals, values = marching_cubes(binary, level=0.5, spacing=spacing)
    return {"vertices": vertices, "faces": faces, "normals": normals, "values": values}
