"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fem import solve_tetrahedral_elasticity
from .reconstruction import mask_to_surface
from .utils import configure_logging, seed_everything


def _demo(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = np.indices((32, 32, 32)).astype(float)
    center = np.array([16, 16, 16])[:, None, None, None]
    mask = ((grid - center) ** 2).sum(axis=0) < 8**2
    surface = mask_to_surface(mask)
    np.savez(output_dir / "surface.npz", vertices=surface["vertices"], faces=surface["faces"])
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    result = solve_tetrahedral_elasticity(nodes, np.array([[0, 1, 2, 3]]), np.array([0, 1, 2]))
    (output_dir / "fem.json").write_text(json.dumps({"max_displacement": float(np.abs(result["displacement"]).max())}), encoding="utf-8")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Brain MRI segmentation and FEM pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo")
    demo.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    args = parser.parse_args()
    seed_everything(42)
    if args.command == "demo":
        _demo(args.output_dir)
