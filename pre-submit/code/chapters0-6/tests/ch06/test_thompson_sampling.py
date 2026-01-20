"""Tests for Thompson Sampling contextual bandit.

Tests:
1. Initialization with Gaussian prior
2. Action selection produces valid template IDs
3. Posterior update increases precision
4. Posterior mean converges to true parameters (synthetic data)
5. Reproducibility with fixed seed
"""

import numpy as np
import pytest

from zoosim.policies.templates import BoostTemplate, create_standard_templates
from zoosim.policies.thompson_sampling import (
    LinearThompsonSampling,
    ThompsonSamplingConfig,
)


@pytest.fixture
def mock_templates():
    """Create minimal template library for testing."""
    templates = [
        BoostTemplate(id=0, name="Template0", description="Test", boost_fn=lambda p: 0.0),
        BoostTemplate(id=1, name="Template1", description="Test", boost_fn=lambda p: 1.0),
        BoostTemplate(id=2, name="Template2", description="Test", boost_fn=lambda p: 2.0),
    ]
    return templates


def test_initialization(mock_templates):
    """Test Thompson Sampling initialization."""
    M = len(mock_templates)
    d = 10
    config = ThompsonSamplingConfig(lambda_reg=1.0, sigma_noise=1.0, seed=42)

    policy = LinearThompsonSampling(mock_templates, feature_dim=d, config=config)

    # Check attributes
    assert policy.M == M
    assert policy.d == d
    assert policy.config == config

    # Check prior initialization
    assert policy.theta_hat.shape == (M, d)
    assert np.allclose(policy.theta_hat, 0.0), "Prior mean should be zero"

    assert policy.Sigma_inv.shape == (M, d, d)
    # Check prior precision: Σ_a^{-1} = λI
    for a in range(M):
        expected_prior = config.lambda_reg * np.eye(d)
        assert np.allclose(policy.Sigma_inv[a], expected_prior), f"Prior precision for action {a} incorrect"

    # Check selection counts
    assert policy.n_samples.shape == (M,)
    assert np.all(policy.n_samples == 0), "Initial selection counts should be zero"


def test_select_action_valid_range(mock_templates):
    """Test that action selection returns valid template IDs."""
    M = len(mock_templates)
    d = 5
    policy = LinearThompsonSampling(mock_templates, feature_dim=d)

    # Select actions for random features
    for _ in range(100):
        features = np.random.randn(d)
        action = policy.select_action(features)

        assert isinstance(action, int)
        assert 0 <= action < M, f"Action {action} outside valid range [0, {M})"


def test_update_increases_precision(mock_templates):
    """Test that posterior update increases precision (reduces uncertainty)."""
    M = len(mock_templates)
    d = 5
    policy = LinearThompsonSampling(mock_templates, feature_dim=d)

    # Get initial precision trace (measure of total uncertainty)
    initial_traces = [np.trace(policy.Sigma_inv[a]) for a in range(M)]

    # Perform updates for action 0
    for _ in range(10):
        features = np.random.randn(d)
        reward = np.random.randn()
        policy.update(action=0, features=features, reward=reward)

    # Get updated precision trace
    updated_traces = [np.trace(policy.Sigma_inv[a]) for a in range(M)]

    # Action 0 should have increased precision
    assert updated_traces[0] > initial_traces[0], "Precision should increase after updates"

    # Other actions should have unchanged precision
    for a in range(1, M):
        assert np.isclose(updated_traces[a], initial_traces[a]), f"Action {a} precision should not change"


def test_posterior_mean_convergence():
    """Test that posterior mean converges to true parameters on synthetic data."""
    # Synthetic linear bandit with known parameters
    M = 3
    d = 5
    np.random.seed(42)

    # True parameters
    theta_star = np.random.randn(M, d)

    # Mock templates
    templates = [
        BoostTemplate(id=i, name=f"Template{i}", description="Test", boost_fn=lambda p: 0.0)
        for i in range(M)
    ]

    config = ThompsonSamplingConfig(lambda_reg=0.1, sigma_noise=0.1, seed=42)
    policy = LinearThompsonSampling(templates, feature_dim=d, config=config)

    # Collect data: 100 episodes per action
    for a in range(M):
        for _ in range(100):
            features = np.random.randn(d)
            # True reward: r = θ_a^* · φ + noise
            true_reward = theta_star[a] @ features + config.sigma_noise * np.random.randn()
            policy.update(action=a, features=features, reward=true_reward)

    # Check convergence: ||θ̂_a - θ_a^*|| < threshold
    for a in range(M):
        error = np.linalg.norm(policy.theta_hat[a] - theta_star[a])
        assert error < 0.5, f"Action {a} posterior mean error {error:.3f} too large (should converge)"


def test_reproducibility_with_fixed_seed(mock_templates):
    """Test that TS produces identical trajectory with fixed seed."""
    M = len(mock_templates)
    d = 5

    # Run 1
    config1 = ThompsonSamplingConfig(seed=42)
    policy1 = LinearThompsonSampling(mock_templates, feature_dim=d, config=config1)
    np.random.seed(42)  # Also fix feature sampling
    actions1 = []
    for _ in range(50):
        features = np.random.randn(d)
        action = policy1.select_action(features)
        reward = np.random.randn()
        policy1.update(action, features, reward)
        actions1.append(action)

    # Run 2 (same seed)
    config2 = ThompsonSamplingConfig(seed=42)
    policy2 = LinearThompsonSampling(mock_templates, feature_dim=d, config=config2)
    np.random.seed(42)
    actions2 = []
    for _ in range(50):
        features = np.random.randn(d)
        action = policy2.select_action(features)
        reward = np.random.randn()
        policy2.update(action, features, reward)
        actions2.append(action)

    # Check identical actions
    assert actions1 == actions2, "TS should be reproducible with fixed seed"

    # Check identical posterior means
    for a in range(M):
        assert np.allclose(policy1.theta_hat[a], policy2.theta_hat[a]), f"Posterior mean for action {a} differs"


def test_diagnostics(mock_templates):
    """Test diagnostic information retrieval."""
    M = len(mock_templates)
    d = 5
    policy = LinearThompsonSampling(mock_templates, feature_dim=d)

    # Perform some updates
    for _ in range(20):
        features = np.random.randn(d)
        action = policy.select_action(features)
        reward = np.random.randn()
        policy.update(action, features, reward)

    # Get diagnostics
    diagnostics = policy.get_diagnostics()

    # Check keys
    assert "selection_counts" in diagnostics
    assert "selection_frequencies" in diagnostics
    assert "theta_norms" in diagnostics
    assert "uncertainty" in diagnostics

    # Check shapes
    assert diagnostics["selection_counts"].shape == (M,)
    assert diagnostics["selection_frequencies"].shape == (M,)
    assert diagnostics["theta_norms"].shape == (M,)
    assert diagnostics["uncertainty"].shape == (M,)

    # Check selection frequencies sum to 1
    assert np.isclose(diagnostics["selection_frequencies"].sum(), 1.0)

    # Check selection counts sum to 20
    assert diagnostics["selection_counts"].sum() == 20
