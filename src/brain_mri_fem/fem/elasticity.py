"""Linear isotropic tetrahedral elasticity with a dense educational baseline."""

from __future__ import annotations

import numpy as np


def _tetra_b_matrix(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the constant strain-displacement matrix and element volume."""
    matrix = np.column_stack((np.ones(4), points))
    volume = abs(float(np.linalg.det(matrix))) / 6.0
    if volume <= np.finfo(float).eps:
        raise ValueError("Degenerate tetrahedron")
    gradients = np.linalg.inv(matrix)[1:, :].T
    b = np.zeros((6, 12))
    for node in range(4):
        x, y, z = gradients[node]
        column = 3 * node
        b[0, column] = x
        b[1, column + 1] = y
        b[2, column + 2] = z
        b[3, column : column + 2] = (y, x)
        b[4, column + 1 : column + 3] = (z, y)
        b[5, column : column + 3 : 2] = (z, x)
    return b, volume


def _elasticity_matrix(young: float, poisson: float) -> np.ndarray:
    if young <= 0 or not -1 < poisson < 0.5:
        raise ValueError("Invalid Young modulus or Poisson ratio")
    factor = young / ((1 + poisson) * (1 - 2 * poisson))
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = poisson
    np.fill_diagonal(matrix[:3, :3], 1 - poisson)
    matrix[3:, 3:] = np.eye(3) * (1 - 2 * poisson) / 2
    return factor * matrix


def solve_tetrahedral_elasticity(
    nodes: np.ndarray,
    elements: np.ndarray,
    fixed_nodes: np.ndarray,
    forces: np.ndarray | None = None,
    young_modulus: float = 1000.0,
    poisson_ratio: float = 0.3,
) -> dict[str, np.ndarray]:
    """Assemble and solve K u=f for linear 4-node tetrahedral elements."""
    coordinates = np.asarray(nodes, dtype=float)
    tetrahedra = np.asarray(elements, dtype=int)
    fixed = np.asarray(fixed_nodes, dtype=int)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or tetrahedra.shape[1] != 4:
        raise ValueError("nodes must be (N,3) and elements must be (M,4)")
    size = 3 * len(coordinates)
    stiffness = np.zeros((size, size))
    load = np.zeros(size) if forces is None else np.asarray(forces, dtype=float).reshape(size).copy()
    constitutive = _elasticity_matrix(young_modulus, poisson_ratio)
    for element in tetrahedra:
        b, volume = _tetra_b_matrix(coordinates[element])
        local = volume * b.T @ constitutive @ b
        indices = np.repeat(3 * element, 3) + np.tile(np.arange(3), 4)
        stiffness[np.ix_(indices, indices)] += local
    constrained = np.concatenate([3 * fixed, 3 * fixed + 1, 3 * fixed + 2])
    constrained = np.unique(constrained)
    free = np.setdiff1d(np.arange(size), constrained)
    if len(free) == 0:
        raise ValueError("At least one free degree of freedom is required")
    displacement = np.zeros(size)
    displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], load[free])
    return {"displacement": displacement.reshape((-1, 3)), "stiffness": stiffness, "load": load.reshape((-1, 3))}
