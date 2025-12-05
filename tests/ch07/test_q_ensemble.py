"""Unit tests for QEnsembleRegressor (Chapter 7).

These tests verify that:
- The ensemble can learn a simple synthetic Q(x,a) mapping.
- Training reduces mean-squared error compared to untrained predictions.
- The predict() API returns mean and standard deviation with correct shapes.
"""

from __future__ import annotations

import numpy as np

from zoosim.policies.q_ensemble import QEnsembleConfig, QEnsembleRegressor


def test_q_ensemble_learns_simple_function() -> None:
    """Q-ensemble should reduce MSE on a simple linear target."""
    rng = np.random.default_rng(0)

    state_dim = 3
    action_dim = 2
    n_samples = 256

    states = rng.normal(size=(n_samples, state_dim)).astype(np.float32)
    actions = rng.uniform(low=-0.5, high=0.5, size=(n_samples, action_dim)).astype(
        np.float32
    )

    # Linear ground-truth Q(x,a) with small noise.
    w_x = np.array([0.7, -0.4, 0.2], dtype=np.float32)
    w_a = np.array([0.5, -0.3], dtype=np.float32)
    rewards = (
        states @ w_x + actions @ w_a + 0.05 * rng.standard_normal(n_samples)
    ).astype(np.float32)

    config = QEnsembleConfig(
        n_ensembles=3,
        hidden_sizes=(16, 16),
        learning_rate=5e-3,
        device="cpu",
        seed=123,
    )
    model = QEnsembleRegressor(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
    )

    mean_before, _ = model.predict(states, actions)
    mse_before = float(np.mean((mean_before - rewards) ** 2))

    model.update_batch(states, actions, rewards, n_epochs=10, batch_size=64)

    mean_after, std_after = model.predict(states, actions)
    mse_after = float(np.mean((mean_after - rewards) ** 2))

    # Training should reduce MSE by a healthy margin.
    assert mse_after < 0.7 * mse_before, (
        f"Training did not reduce MSE enough: before={mse_before:.4f}, "
        f"after={mse_after:.4f}"
    )

    # Standard deviation should be non-negative and finite.
    assert np.all(std_after >= 0.0)
    assert np.all(np.isfinite(std_after))


def test_q_ensemble_predict_shapes() -> None:
    """predict() should return correct shapes for mean/std and individual preds."""
    rng = np.random.default_rng(1)
    state_dim = 4
    action_dim = 3
    n_samples = 20

    states = rng.normal(size=(n_samples, state_dim)).astype(np.float32)
    actions = rng.uniform(low=-1.0, high=1.0, size=(n_samples, action_dim)).astype(
        np.float32
    )
    rewards = rng.normal(size=n_samples).astype(np.float32)

    config = QEnsembleConfig(
        n_ensembles=4,
        hidden_sizes=(8,),
        learning_rate=1e-3,
        device="cpu",
        seed=7,
    )
    model = QEnsembleRegressor(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
    )
    model.update_batch(states, actions, rewards, n_epochs=2, batch_size=8)

    individual, mean, std = model.predict(states, actions, return_individual=True)

    assert individual.shape == (config.n_ensembles, n_samples)
    assert mean.shape == (n_samples,)
    assert std.shape == (n_samples,)

