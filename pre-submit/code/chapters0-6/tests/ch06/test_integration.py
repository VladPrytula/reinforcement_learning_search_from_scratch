"""Integration tests for Chapter 6: Template bandits on simulator.

Tests:
1. LinUCB training loop with mock environment
2. Thompson Sampling training loop
3. Template selection frequencies converge
4. Reproducibility across multiple runs
"""

import numpy as np
import pytest

from zoosim.policies.templates import BoostTemplate, create_standard_templates, compute_catalog_stats
from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
from zoosim.policies.thompson_sampling import LinearThompsonSampling, ThompsonSamplingConfig


class MockSearchEnv:
    """Minimal mock environment for testing bandit training loops."""

    def __init__(self, d=10, seed=42):
        """Initialize mock environment.

        Args:
            d: Feature dimension
            seed: Random seed
        """
        self.d = d
        self.rng = np.random.default_rng(seed)

        # True reward parameters (simulated optimal template)
        # Template 2 is best for most contexts
        self.theta_star = np.zeros(d)
        self.theta_star[0] = 2.0  # Feature 0 strongly predicts template 2 reward

    def reset(self):
        """Reset environment, return features."""
        features = self.rng.standard_normal(self.d)
        return features

    def step(self, template_id):
        """Execute template, return reward.

        Simulated reward structure:
        - Template 2 is optimal (high base reward)
        - Template 0 has medium reward
        - Template 1 has low reward
        """
        # Base rewards per template
        base_rewards = {0: 5.0, 1: 2.0, 2: 10.0}
        base = base_rewards.get(template_id, 0.0)

        # Add noise
        noise = self.rng.normal(0, 1.0)

        reward = base + noise
        return reward


def test_linucb_training_loop():
    """Test that LinUCB learns to prefer optimal template."""
    # Setup
    templates = [
        BoostTemplate(id=0, name="Template0", description="Medium", boost_fn=lambda p: 0.0),
        BoostTemplate(id=1, name="Template1", description="Low", boost_fn=lambda p: 0.0),
        BoostTemplate(id=2, name="Template2", description="High", boost_fn=lambda p: 0.0),
    ]
    d = 10
    env = MockSearchEnv(d=d, seed=42)
    config = LinUCBConfig(lambda_reg=1.0, alpha=1.0, seed=42)
    policy = LinUCB(templates, feature_dim=d, config=config)

    # Training loop
    T = 500  # Episodes
    for t in range(T):
        features = env.reset()
        action = policy.select_action(features)
        reward = env.step(action)
        policy.update(action, features, reward)

    # Check that template 2 (optimal) is selected most frequently
    diagnostics = policy.get_diagnostics()
    selection_freqs = diagnostics["selection_frequencies"]

    # Template 2 should have highest selection frequency
    assert selection_freqs[2] == np.max(selection_freqs), \
        f"Template 2 should be selected most frequently, got freqs: {selection_freqs}"

    # Template 2 should be selected often enough to indicate clear preference.
    # In this mock environment features are pure noise, so the best we can
    # expect is a strong but not overwhelming preference.
    assert selection_freqs[2] > 0.4, \
        f"Template 2 selected only {selection_freqs[2]:.2%}, expected >40%"


def test_thompson_sampling_training_loop():
    """Test that Thompson Sampling learns to prefer optimal template."""
    templates = [
        BoostTemplate(id=0, name="Template0", description="Medium", boost_fn=lambda p: 0.0),
        BoostTemplate(id=1, name="Template1", description="Low", boost_fn=lambda p: 0.0),
        BoostTemplate(id=2, name="Template2", description="High", boost_fn=lambda p: 0.0),
    ]
    d = 10
    env = MockSearchEnv(d=d, seed=42)
    config = ThompsonSamplingConfig(lambda_reg=1.0, sigma_noise=1.0, seed=42)
    policy = LinearThompsonSampling(templates, feature_dim=d, config=config)

    # Training loop
    T = 500
    for t in range(T):
        features = env.reset()
        action = policy.select_action(features)
        reward = env.step(action)
        policy.update(action, features, reward)

    # Check convergence
    diagnostics = policy.get_diagnostics()
    selection_freqs = diagnostics["selection_frequencies"]

    # Template 2 should have highest selection frequency
    assert selection_freqs[2] == np.max(selection_freqs)

    # Template 2 should be selected often enough to indicate clear preference.
    # Thompson Sampling remains stochastic, so we use a slightly softer threshold.
    assert selection_freqs[2] > 0.4, \
        f"Template 2 selected only {selection_freqs[2]:.2%}, expected >40%"


def test_linucb_reproducibility():
    """Test that LinUCB produces identical trajectory with fixed seed."""
    templates = [
        BoostTemplate(id=i, name=f"Template{i}", description="Test", boost_fn=lambda p: 0.0)
        for i in range(3)
    ]
    d = 10

    # Run 1
    env1 = MockSearchEnv(d=d, seed=42)
    config1 = LinUCBConfig(seed=42)
    policy1 = LinUCB(templates, feature_dim=d, config=config1)
    actions1 = []
    for _ in range(100):
        features = env1.reset()
        action = policy1.select_action(features)
        reward = env1.step(action)
        policy1.update(action, features, reward)
        actions1.append(action)

    # Run 2 (same seed)
    env2 = MockSearchEnv(d=d, seed=42)
    config2 = LinUCBConfig(seed=42)
    policy2 = LinUCB(templates, feature_dim=d, config=config2)
    actions2 = []
    for _ in range(100):
        features = env2.reset()
        action = policy2.select_action(features)
        reward = env2.step(action)
        policy2.update(action, features, reward)
        actions2.append(action)

    # Check identical actions
    assert actions1 == actions2, "LinUCB should be reproducible with fixed seed"


def test_exploration_decay():
    """Test that exploration decreases over time (selection diversity → concentration)."""
    templates = [
        BoostTemplate(id=i, name=f"Template{i}", description="Test", boost_fn=lambda p: 0.0)
        for i in range(3)
    ]
    d = 10
    env = MockSearchEnv(d=d, seed=42)
    config = LinUCBConfig(lambda_reg=1.0, alpha=1.0, adaptive_alpha=False)
    policy = LinUCB(templates, feature_dim=d, config=config)

    # Track selection entropy over time (measure of exploration)
    # Entropy = -Σ p_a log(p_a), high entropy = uniform exploration, low = concentrated

    def entropy(freqs):
        """Compute entropy of selection distribution."""
        freqs = freqs[freqs > 0]  # Avoid log(0)
        return -np.sum(freqs * np.log(freqs))

    window_size = 100
    entropies = []

    for window in range(5):  # 5 windows of 100 episodes each
        window_actions = []
        for _ in range(window_size):
            features = env.reset()
            action = policy.select_action(features)
            reward = env.step(action)
            policy.update(action, features, reward)
            window_actions.append(action)

        # Compute entropy for this window
        window_freqs = np.bincount(window_actions, minlength=len(templates)) / window_size
        window_entropy = entropy(window_freqs)
        entropies.append(window_entropy)

    # Entropy should decrease over time (exploration → exploitation)
    assert entropies[0] > entropies[-1], \
        f"Entropy should decrease: {entropies[0]:.3f} → {entropies[-1]:.3f}"

    # Final entropy should be lower than log(3) ≈ 1.099 (uniform distribution)
    assert entropies[-1] < np.log(len(templates)), \
        f"Final entropy {entropies[-1]:.3f} should be < log({len(templates)}) = {np.log(len(templates)):.3f}"
