"""Unit tests for CEM optimizer (Chapter 7)."""

from __future__ import annotations

import numpy as np

from zoosim.optimizers.cem import CEMConfig, cem_optimize


def test_cem_optimizes_quadratic() -> None:
    """CEM should find the maximum of a simple quadratic."""
    rng = np.random.default_rng(0)
    action_dim = 2
    target = np.array([0.3, -0.2], dtype=np.float64)

    def objective(actions: np.ndarray) -> np.ndarray:
        # Maximize -||a - target||^2, so optimum is at 'target'.
        diffs = actions - target[None, :]
        return -np.sum(diffs**2, axis=1)

    config = CEMConfig(
        n_samples=128,
        elite_frac=0.2,
        n_iters=8,
        init_std=0.5,
        min_std=0.05,
        alpha=0.5,
        seed=42,
        a_max=1.0,
    )

    best_action, history = cem_optimize(
        objective=objective,
        action_dim=action_dim,
        config=config,
    )

    assert history["mean_history"].shape == (config.n_iters, action_dim)
    assert history["std_history"].shape == (config.n_iters, action_dim)
    assert history["best_values"].shape == (config.n_iters,)

    # Best action should be close to the true optimum.
    distance = float(np.linalg.norm(best_action - target))
    assert distance < 0.15, f"CEM did not converge near optimum, distance={distance:.3f}"


def test_cem_trust_region_projection() -> None:
    """Trust region should keep actions within specified radius."""
    action_dim = 3
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    radius = 0.1

    def objective(actions: np.ndarray) -> np.ndarray:
        # Objective that prefers actions far from the origin, but trust region
        # should keep them close.
        return np.linalg.norm(actions, axis=1)

    config = CEMConfig(
        n_samples=64,
        elite_frac=0.25,
        n_iters=5,
        init_std=1.0,
        min_std=0.05,
        alpha=0.7,
        seed=123,
        a_max=5.0,
    )

    best_action, _ = cem_optimize(
        objective=objective,
        action_dim=action_dim,
        config=config,
        init_mean=center,
        trust_region_center=center,
        trust_region_radius=radius,
    )

    distance = float(np.linalg.norm(best_action - center))
    assert distance <= radius + 1e-6, (
        f"Best action {best_action} lies outside trust region (radius={radius}, "
        f"distance={distance:.4f})"
    )

