import numpy as np

from brain_mri_fem.fem import solve_tetrahedral_elasticity


def test_tetrahedral_solver_returns_finite_displacement():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    result = solve_tetrahedral_elasticity(nodes, np.array([[0, 1, 2, 3]]), np.array([0, 1, 2]))
    assert result["displacement"].shape == (4, 3)
    assert np.isfinite(result["displacement"]).all()
