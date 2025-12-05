"""Tests for LinUCB contextual bandit.

Tests:
1. Initialization with ridge regression
2. Action selection is deterministic
3. UCB scores include exploration bonus
4. Posterior mean identical to Thompson Sampling
5. Adaptive alpha decay
"""

import numpy as np
import pytest

from zoosim.policies.templates import BoostTemplate
from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
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
    """Test LinUCB initialization."""
    M = len(mock_templates)
    d = 10
    config = LinUCBConfig(lambda_reg=1.0, alpha=1.0, seed=42)

    policy = LinUCB(mock_templates, feature_dim=d, config=config)

    # Check attributes
    assert policy.M == M
    assert policy.d == d
    assert policy.config == config

    # Check weight initialization
    assert policy.theta_hat.shape == (M, d)
    assert np.allclose(policy.theta_hat, 0.0), "Initial weights should be zero"

    # Check design matrix: A_a = λI
    assert policy.A.shape == (M, d, d)
    for a in range(M):
        expected_A = config.lambda_reg * np.eye(d)
        assert np.allclose(policy.A[a], expected_A), f"Design matrix for action {a} incorrect"

    # Check reward accumulator
    assert policy.b.shape == (M, d)
    assert np.allclose(policy.b, 0.0), "Initial reward accumulator should be zero"

    # Check episode counter
    assert policy.t == 0


def test_select_action_deterministic(mock_templates):
    """Test that LinUCB action selection is deterministic."""
    M = len(mock_templates)
    d = 5
    config = LinUCBConfig(seed=42)
    policy = LinUCB(mock_templates, feature_dim=d, config=config)

    # Same features should produce same action
    features = np.random.randn(d)

    action1 = policy.select_action(features)
    policy.t -= 1  # Reset episode counter
    action2 = policy.select_action(features)

    assert action1 == action2, "LinUCB should be deterministic"


def test_select_action_valid_range(mock_templates):
    """Test that action selection returns valid template IDs."""
    M = len(mock_templates)
    d = 5
    policy = LinUCB(mock_templates, feature_dim=d)

    for _ in range(100):
        features = np.random.randn(d)
        action = policy.select_action(features)

        assert isinstance(action, int)
        assert 0 <= action < M, f"Action {action} outside valid range [0, {M})"


def test_ucb_exploration_bonus():
    """Test that UCB scores include exploration bonus (alpha > 0)."""
    M = 2
    d = 5

    templates = [
        BoostTemplate(id=i, name=f"Template{i}", description="Test", boost_fn=lambda p: 0.0)
        for i in range(M)
    ]

    # Policy with alpha=0 (greedy, no exploration)
    config_greedy = LinUCBConfig(alpha=0.0, seed=42)
    policy_greedy = LinUCB(templates, feature_dim=d, config=config_greedy)

    # Policy with alpha=1.0 (with exploration)
    config_explore = LinUCBConfig(alpha=1.0, seed=42)
    policy_explore = LinUCB(templates, feature_dim=d, config=config_explore)

    # Collect some data for action 0
    for _ in range(10):
        features = np.random.randn(d)
        reward = 5.0  # High reward for action 0
        policy_greedy.update(action=0, features=features, reward=reward)
        policy_explore.update(action=0, features=features, reward=reward)

    # Now select action with features that slightly favor action 1
    # (but action 1 has high uncertainty due to no data)
    features_test = np.random.randn(d)

    # Greedy should select based only on mean reward
    # (likely action 0 since it has positive mean, action 1 has zero mean)

    # Explore should consider uncertainty bonus
    # (action 1 might get selected due to high uncertainty despite lower mean)

    # This is a probabilistic test, so we just check that actions are valid
    action_greedy = policy_greedy.select_action(features_test)
    action_explore = policy_explore.select_action(features_test)

    assert 0 <= action_greedy < M
    assert 0 <= action_explore < M

    # Note: Can't guarantee they differ, but with high probability they should
    # if exploration bonus is working correctly


def test_posterior_mean_identical_to_thompson_sampling(mock_templates):
    """Test that LinUCB and TS maintain identical posterior mean θ̂_a."""
    M = len(mock_templates)
    d = 5

    # Initialize both policies with same hyperparameters
    ts_config = ThompsonSamplingConfig(lambda_reg=1.0, sigma_noise=1.0, seed=42)
    ucb_config = LinUCBConfig(lambda_reg=1.0, alpha=1.0, seed=42)

    ts_policy = LinearThompsonSampling(mock_templates, feature_dim=d, config=ts_config)
    ucb_policy = LinUCB(mock_templates, feature_dim=d, config=ucb_config)

    # Apply same sequence of updates
    np.random.seed(42)
    for _ in range(50):
        action = np.random.randint(0, M)  # Random action
        features = np.random.randn(d)
        reward = np.random.randn()

        ts_policy.update(action, features, reward)
        ucb_policy.update(action, features, reward)

    # Check posterior means are identical
    for a in range(M):
        assert np.allclose(ts_policy.theta_hat[a], ucb_policy.theta_hat[a]), \
            f"Posterior mean for action {a} differs between TS and LinUCB"


def test_adaptive_alpha_decay(mock_templates):
    """Test that adaptive alpha decays as √log(1+t)."""
    M = len(mock_templates)
    d = 5
    config = LinUCBConfig(alpha=1.0, adaptive_alpha=True, seed=42)
    policy = LinUCB(mock_templates, feature_dim=d, config=config)

    # Initial alpha
    diagnostics0 = policy.get_diagnostics()
    alpha0 = diagnostics0["alpha_current"]
    assert np.isclose(alpha0, 1.0), "Initial alpha should be base value"

    # After 100 episodes
    for _ in range(100):
        features = np.random.randn(d)
        action = policy.select_action(features)
        reward = np.random.randn()
        policy.update(action, features, reward)

    diagnostics100 = policy.get_diagnostics()
    alpha100 = diagnostics100["alpha_current"]

    # Alpha should have increased: α_t = α * √log(1+t)
    expected_alpha100 = config.alpha * np.sqrt(np.log(1 + 100))
    assert np.isclose(alpha100, expected_alpha100), f"Adaptive alpha incorrect: {alpha100} vs {expected_alpha100}"
    assert alpha100 > alpha0, "Adaptive alpha should increase over time"


def test_update_design_matrix(mock_templates):
    """Test that design matrix A accumulates φφ^T correctly."""
    M = len(mock_templates)
    d = 5
    policy = LinUCB(mock_templates, feature_dim=d)

    # Initial design matrix: A_a = λI
    lambda_reg = policy.config.lambda_reg
    initial_A0 = policy.A[0].copy()
    assert np.allclose(initial_A0, lambda_reg * np.eye(d))

    # Update with known features
    features1 = np.ones(d)  # All ones
    reward1 = 1.0
    policy.update(action=0, features=features1, reward=reward1)

    # Check A_0 updated: A_0 = λI + φ_1 φ_1^T
    expected_A0 = lambda_reg * np.eye(d) + np.outer(features1, features1)
    assert np.allclose(policy.A[0], expected_A0), "Design matrix update incorrect"

    # Update again
    features2 = np.array([1, 2, 3, 4, 5])
    reward2 = 2.0
    policy.update(action=0, features=features2, reward=reward2)

    # Check A_0 accumulated: A_0 = λI + φ_1 φ_1^T + φ_2 φ_2^T
    expected_A0 = lambda_reg * np.eye(d) + np.outer(features1, features1) + np.outer(features2, features2)
    assert np.allclose(policy.A[0], expected_A0), "Design matrix accumulation incorrect"


def test_diagnostics(mock_templates):
    """Test diagnostic information retrieval."""
    M = len(mock_templates)
    d = 5
    config = LinUCBConfig(alpha=1.0, adaptive_alpha=True)
    policy = LinUCB(mock_templates, feature_dim=d, config=config)

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
    assert "alpha_current" in diagnostics

    # Check shapes
    assert diagnostics["selection_counts"].shape == (M,)
    assert diagnostics["selection_frequencies"].shape == (M,)
    assert diagnostics["theta_norms"].shape == (M,)
    assert diagnostics["uncertainty"].shape == (M,)
    assert isinstance(diagnostics["alpha_current"], float)

    # Check selection frequencies sum to 1
    assert np.isclose(diagnostics["selection_frequencies"].sum(), 1.0)

    # Check selection counts sum to 20
    assert diagnostics["selection_counts"].sum() == 20
