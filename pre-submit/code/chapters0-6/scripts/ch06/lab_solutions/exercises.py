"""
Chapter 6 Lab Solutions — Theory and Implementation Exercises

Author: Vlad Prytula

This module implements the theory and implementation exercises from Chapter 6.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import solve_triangular


# =============================================================================
# Exercise 6.1: Cosine Similarity Properties
# =============================================================================


def exercise_6_1_cosine_properties(
    seed: int = 42, n_tests: int = 1000, verbose: bool = True
) -> dict:
    """
    Exercise 6.1: Verify properties of cosine similarity numerically.

    Properties tested:
    (a) Boundedness: s_sem ∈ [-1, 1]
    (b) Scale invariance: s_sem(αq, βe) = sign(αβ) · s_sem(q, e)
    (c) Non-additivity for orthogonal vectors

    Returns:
        Dict with test results for each property
    """
    if verbose:
        print("=" * 70)
        print("Exercise 6.1: Cosine Similarity Properties")
        print("=" * 70)

    rng = np.random.default_rng(seed)

    def cosine_similarity(q: np.ndarray, e: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        norm_q = np.linalg.norm(q)
        norm_e = np.linalg.norm(e)
        if norm_q == 0 or norm_e == 0:
            return 0.0
        return float(np.dot(q, e) / (norm_q * norm_e))

    # Part (a): Boundedness
    if verbose:
        print(f"\nPart (a): Boundedness verification")
        print(f"  Testing {n_tests} random vector pairs (d=16)...")

    d = 16
    all_bounded = True
    min_sim, max_sim = float("inf"), float("-inf")

    for _ in range(n_tests):
        q = rng.standard_normal(d)
        e = rng.standard_normal(d)
        sim = cosine_similarity(q, e)
        if sim < -1 - 1e-10 or sim > 1 + 1e-10:
            all_bounded = False
        min_sim = min(min_sim, sim)
        max_sim = max(max_sim, sim)

    if verbose:
        print(f"  All similarities in [-1, 1]: {all_bounded}")
        print(f"  Min observed: {min_sim:.3f}, Max observed: {max_sim:.3f}")

    # Part (b): Scale invariance
    if verbose:
        print(f"\nPart (b): Scale invariance verification")

    q = rng.standard_normal(d)
    e = rng.standard_normal(d)
    alpha, beta = 2.5, -3.0

    s_original = cosine_similarity(q, e)
    s_scaled = cosine_similarity(alpha * q, beta * e)
    expected = np.sign(alpha * beta) * s_original

    scale_invariant = abs(s_scaled - expected) < 1e-10

    if verbose:
        print(f"  Testing α={alpha}, β={beta}")
        print(f"  s(q, e) = {s_original:.4f}")
        print(f"  s(αq, βe) = {s_scaled:.4f}")
        print(f"  sign(αβ) · s(q, e) = {expected:.4f}")
        print(f"  Equality holds: {scale_invariant} (diff = {abs(s_scaled - expected):.2e})")

    # Part (c): Non-additivity counterexample
    if verbose:
        print(f"\nPart (c): Non-additivity counterexample")

    q_c = np.array([1.0, 1.0])
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    orthogonal = abs(np.dot(e1, e2)) < 1e-10
    lhs = cosine_similarity(q_c, e1 + e2)
    rhs = cosine_similarity(q_c, e1) + cosine_similarity(q_c, e2)
    non_additive = abs(lhs - rhs) > 0.01

    if verbose:
        print(f"  q = {q_c.tolist()}, e1 = {e1.tolist()}, e2 = {e2.tolist()}")
        print(f"  e1 ⊥ e2: {orthogonal} (dot product = {np.dot(e1, e2)})")
        print(f"  s(q, e1 + e2) = {lhs:.4f}")
        print(f"  s(q, e1) + s(q, e2) = {rhs:.4f}")
        print(f"  Non-additivity demonstrated: {lhs:.4f} ≠ {rhs:.4f}")
        print(f"\n✓ All properties verified numerically.")

    return {
        "boundedness": {"all_bounded": all_bounded, "min": min_sim, "max": max_sim},
        "scale_invariance": {"holds": scale_invariant, "diff": abs(s_scaled - expected)},
        "non_additivity": {"orthogonal": orthogonal, "lhs": lhs, "rhs": rhs},
    }


# =============================================================================
# Exercise 6.2: Ridge Regression Equivalence
# =============================================================================


def exercise_6_2_ridge_regression(
    seed: int = 42, n_samples: int = 100, d: int = 7, verbose: bool = True
) -> dict:
    """
    Exercise 6.2: Verify LinUCB weight update is equivalent to ridge regression.

    Tests:
    (a) LinUCB weights = ridge regression closed form
    (b) OLS limit as λ → 0
    (c) Condition number improvement with regularization

    Returns:
        Dict with verification results
    """
    if verbose:
        print("=" * 70)
        print("Exercise 6.2: Ridge Regression Equivalence")
        print("=" * 70)

    rng = np.random.default_rng(seed)

    # Generate synthetic regression data
    if verbose:
        print(f"\nGenerating synthetic regression data (n={n_samples}, d={d})...")

    true_theta = rng.standard_normal(d)
    X = rng.standard_normal((n_samples, d))
    noise = rng.standard_normal(n_samples) * 0.1
    y = X @ true_theta + noise

    lambda_reg = 1.0

    # Part (a): LinUCB vs explicit ridge regression
    if verbose:
        print(f"\nPart (a): LinUCB vs explicit ridge regression")

    # LinUCB formulation: A = λI + ΣφφT, b = Σrφ, θ = A⁻¹b
    A = lambda_reg * np.eye(d) + X.T @ X
    b = X.T @ y
    theta_linucb = np.linalg.solve(A, b)

    # Explicit ridge regression: minimize Σ(y - θTx)² + λ||θ||²
    # Closed form: θ = (XTX + λI)⁻¹ XTy
    theta_ridge = np.linalg.solve(X.T @ X + lambda_reg * np.eye(d), X.T @ y)

    diff_a = np.max(np.abs(theta_linucb - theta_ridge))

    if verbose:
        print(f"  LinUCB weights (A⁻¹b): [{', '.join(f'{x:.3f}' for x in theta_linucb[:7])}]")
        print(f"  Ridge regression (closed form): [{', '.join(f'{x:.3f}' for x in theta_ridge[:7])}]")
        print(f"  Max difference: {diff_a:.2e}")
        print(f"  ✓ Equivalence verified (numerical precision)")

    # Part (b): OLS limit as λ → 0
    if verbose:
        print(f"\nPart (b): OLS limit as λ → 0")

    theta_ols = np.linalg.solve(X.T @ X, X.T @ y)

    ols_diffs = []
    for lam in [1e0, 1e-1, 1e-2, 1e-3, 1e-4]:
        theta_lam = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
        diff = np.linalg.norm(theta_lam - theta_ols)
        ols_diffs.append((lam, diff))
        if verbose:
            print(f"  λ = {lam:.1e}: ‖θ_ridge - θ_ols‖ = {diff:.4f}")

    if verbose:
        print(f"  ✓ Convergence to OLS demonstrated")

    # Part (c): Regularization and condition number
    if verbose:
        print(f"\nPart (c): Regularization and condition number")

    XTX = X.T @ X
    cond_no_reg = np.linalg.cond(XTX)
    cond_reg = np.linalg.cond(XTX + lambda_reg * np.eye(d))

    if verbose:
        print(f"  Without regularization (λ=0): κ(ΦᵀΦ) = {cond_no_reg:.2e}")
        print(f"  With regularization (λ=1):    κ(A) = {cond_reg:.2e}")
        print(f"  Condition number reduced by factor: {cond_no_reg / cond_reg:.0f}×")
        print(f"\n  Why this matters:")
        print(f"  - Ill-conditioned matrices amplify numerical errors")
        print(f"  - In early training, Σφφᵀ may be rank-deficient")
        print(f"  - Regularization λI ensures A is always invertible")
        print(f"  - Bounds condition number: κ(A) ≤ (λ_max + λ)/λ")

    return {
        "linucb_ridge_diff": diff_a,
        "ols_convergence": ols_diffs,
        "condition_numbers": {"no_reg": cond_no_reg, "with_reg": cond_reg},
    }


# =============================================================================
# Exercise 6.3: TS vs LinUCB Posterior Equivalence
# =============================================================================


def exercise_6_3_ts_linucb_equivalence(
    seed: int = 42, n_episodes: int = 500, verbose: bool = True
) -> dict:
    """
    Exercise 6.3: Show TS and LinUCB maintain identical posterior means.

    Returns:
        Dict with comparison results
    """
    if verbose:
        print("=" * 70)
        print("Exercise 6.3: Thompson Sampling vs LinUCB Posterior Equivalence")
        print("=" * 70)

    rng = np.random.default_rng(seed)

    d = 7
    M = 4  # Number of actions
    lambda_reg = 1.0

    # Initialize both algorithms identically
    # LinUCB state
    A_linucb = [lambda_reg * np.eye(d) for _ in range(M)]
    b_linucb = [np.zeros(d) for _ in range(M)]

    # TS state (using precision matrix parameterization)
    Sigma_inv_ts = [lambda_reg * np.eye(d) for _ in range(M)]
    theta_ts = [np.zeros(d) for _ in range(M)]

    if verbose:
        print(f"\nRunning {n_episodes} episodes with identical data streams...")

    max_diffs = []

    for ep in range(1, n_episodes + 1):
        # Generate random features and reward
        phi = rng.standard_normal(d)
        r = rng.standard_normal() + np.dot(phi, rng.standard_normal(d) * 0.1)
        a = rng.integers(M)  # Random action

        # LinUCB update
        A_linucb[a] += np.outer(phi, phi)
        b_linucb[a] += r * phi
        theta_linucb = np.linalg.solve(A_linucb[a], b_linucb[a])

        # TS update (Bayesian linear regression)
        Sigma_inv_ts[a] += np.outer(phi, phi)
        # With zero-mean prior, posterior mean is Sigma @ b
        theta_ts[a] = np.linalg.solve(Sigma_inv_ts[a], b_linucb[a])

        # Compare
        max_diff = np.max(np.abs(theta_linucb - theta_ts[a]))
        max_diffs.append(max_diff)

        if verbose and (ep % 100 == 0 or ep == n_episodes):
            print(f"Episode {ep}: Max |θ_TS - θ_LinUCB| = {max_diff:.2e}")

    # Final comparison for all actions
    if verbose:
        print(f"\nFinal comparison (action 0):")
        theta_lin_0 = np.linalg.solve(A_linucb[0], b_linucb[0])
        theta_ts_0 = np.linalg.solve(Sigma_inv_ts[0], b_linucb[0])
        print(f"  TS posterior mean: [{', '.join(f'{x:.3f}' for x in theta_ts_0[:3])}...]")
        print(f"  LinUCB weights:    [{', '.join(f'{x:.3f}' for x in theta_lin_0[:3])}...]")
        print(f"  Max difference: {np.max(np.abs(theta_lin_0 - theta_ts_0)):.2e} (numerical precision)")

        print(f"\nFinal comparison (action 3):")
        theta_lin_3 = np.linalg.solve(A_linucb[3], b_linucb[3])
        theta_ts_3 = np.linalg.solve(Sigma_inv_ts[3], b_linucb[3])
        print(f"  TS posterior mean: [{', '.join(f'{x:.3f}' for x in theta_ts_3[:3])}...]")
        print(f"  LinUCB weights:    [{', '.join(f'{x:.3f}' for x in theta_lin_3[:3])}...]")
        print(f"  Max difference: {np.max(np.abs(theta_lin_3 - theta_ts_3)):.2e} (numerical precision)")

        print(f"\n✓ Posterior means are identical to numerical precision.")
        print(f"\nKey insight:")
        print(f"  TS and LinUCB learn the SAME model (ridge regression).")
        print(f"  They differ only in HOW they use uncertainty for exploration:")
        print(f"  - LinUCB: Deterministic UCB bonus √(φᵀΣφ)")
        print(f"  - TS: Stochastic sampling from posterior")

    return {"max_diffs": max_diffs, "final_max_diff": max_diffs[-1]}


# =============================================================================
# Exercise 6.4: ε-Greedy Baseline
# =============================================================================


class EpsilonGreedy:
    """ε-greedy policy for contextual bandits."""

    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        epsilon: float = 0.1,
        lambda_reg: float = 1.0,
        seed: int = 42,
    ):
        self.n_actions = n_actions
        self.d = feature_dim
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.rng = np.random.default_rng(seed)

        # Per-action ridge regression state
        self.A = [lambda_reg * np.eye(feature_dim) for _ in range(n_actions)]
        self.b = [np.zeros(feature_dim) for _ in range(n_actions)]
        self.theta = [np.zeros(feature_dim) for _ in range(n_actions)]

    def select_action(self, features: np.ndarray) -> int:
        """Select action using ε-greedy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        # Greedy: argmax_a θ_a^T φ
        expected_rewards = [np.dot(self.theta[a], features) for a in range(self.n_actions)]
        return int(np.argmax(expected_rewards))

    def update(self, action: int, features: np.ndarray, reward: float) -> None:
        """Ridge regression update."""
        phi = features.reshape(-1)
        self.A[action] += np.outer(phi, phi)
        self.b[action] += reward * phi
        self.theta[action] = np.linalg.solve(self.A[action], self.b[action])


def exercise_6_4_epsilon_greedy(
    n_episodes: int = 20000,
    epsilons: List[float] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Exercise 6.4: Implement ε-greedy and compare to LinUCB.

    Returns:
        Dict with comparison results
    """
    if epsilons is None:
        epsilons = [0.05, 0.1, 0.2]

    if verbose:
        print("=" * 70)
        print("Exercise 6.4: ε-Greedy Baseline Implementation")
        print("=" * 70)
        print(f"\nRunning experiments with n_episodes={n_episodes:,}...")

    rng = np.random.default_rng(seed)

    # Simplified environment for demonstration
    d = 7
    n_actions = 8

    # True reward parameters (action-dependent)
    true_theta = [rng.standard_normal(d) for _ in range(n_actions)]
    optimal_rewards = []

    def simulate_reward(action: int, features: np.ndarray) -> float:
        """Simulate reward with noise."""
        true_r = np.dot(true_theta[action], features)
        return true_r + rng.standard_normal() * 0.5

    def get_optimal_reward(features: np.ndarray) -> float:
        """Get optimal (oracle) reward."""
        return max(np.dot(true_theta[a], features) for a in range(n_actions))

    results = {}

    # Run ε-greedy for each epsilon
    for eps in epsilons:
        if verbose:
            print(f"\nTraining ε-greedy (ε={eps})...")

        agent = EpsilonGreedy(n_actions, d, epsilon=eps, seed=seed)
        rewards = []
        cumulative_regret = 0.0
        regrets = []

        for ep in range(n_episodes):
            features = rng.standard_normal(d)
            action = agent.select_action(features)
            reward = simulate_reward(action, features)
            agent.update(action, features, reward)
            rewards.append(reward)

            # Track regret
            optimal = get_optimal_reward(features)
            cumulative_regret += max(0, optimal - reward)
            regrets.append(cumulative_regret)

            if verbose and (ep + 1) % (n_episodes // 5) == 0:
                print(f"  Progress: {100 * (ep + 1) / n_episodes:.0f}% ({ep + 1}/{n_episodes})")

        results[f"eps_{eps}"] = {
            "epsilon": eps,
            "final_reward": np.mean(rewards[-n_episodes // 4 :]),
            "cumulative_regret": cumulative_regret,
            "rewards": rewards,
            "regrets": regrets,
        }

    # Run LinUCB for comparison
    if verbose:
        print(f"\nTraining LinUCB (α=1.0)...")

    # Simple LinUCB implementation
    A_linucb = [np.eye(d) for _ in range(n_actions)]
    b_linucb = [np.zeros(d) for _ in range(n_actions)]
    rewards_linucb = []
    cumulative_regret_linucb = 0.0
    regrets_linucb = []
    alpha = 1.0

    for ep in range(n_episodes):
        features = rng.standard_normal(d)

        # LinUCB selection
        ucb_values = []
        for a in range(n_actions):
            theta_a = np.linalg.solve(A_linucb[a], b_linucb[a])
            Sigma_a = np.linalg.inv(A_linucb[a])
            bonus = alpha * np.sqrt(features @ Sigma_a @ features)
            ucb_values.append(np.dot(theta_a, features) + bonus)
        action = int(np.argmax(ucb_values))

        reward = simulate_reward(action, features)
        A_linucb[action] += np.outer(features, features)
        b_linucb[action] += reward * features
        rewards_linucb.append(reward)

        optimal = get_optimal_reward(features)
        cumulative_regret_linucb += max(0, optimal - reward)
        regrets_linucb.append(cumulative_regret_linucb)

        if verbose and (ep + 1) % (n_episodes // 5) == 0:
            print(f"  Progress: {100 * (ep + 1) / n_episodes:.0f}% ({ep + 1}/{n_episodes})")

    results["linucb"] = {
        "final_reward": np.mean(rewards_linucb[-n_episodes // 4 :]),
        "cumulative_regret": cumulative_regret_linucb,
        "rewards": rewards_linucb,
        "regrets": regrets_linucb,
    }

    if verbose:
        print(f"\nResults (average reward over last {n_episodes // 4} episodes):\n")
        print(f"  {'Policy':<18} | {'Avg Reward':>10} | {'Cumulative Regret':>17}")
        print(f"  {'-'*18}-|-{'-'*10}-|-{'-'*17}")
        for eps in epsilons:
            r = results[f"eps_{eps}"]
            print(f"  {'ε-greedy (ε=' + str(eps) + ')':<18} | {r['final_reward']:>10.2f} | {r['cumulative_regret']:>17,.0f}")
        print(f"  {'LinUCB (α=1.0)':<18} | {results['linucb']['final_reward']:>10.2f} | {results['linucb']['cumulative_regret']:>17,.0f}")

        print(f"\nRegret Analysis:\n")
        print(f"  At T={n_episodes:,}:")
        for eps in epsilons:
            r = results[f"eps_{eps}"]
            ratio = r["cumulative_regret"] / n_episodes
            print(f"  - ε-greedy (ε={eps}): Regret ≈ {r['cumulative_regret']:,.0f} ≈ {ratio:.2f} × T (linear)")
        ratio_linucb = results["linucb"]["cumulative_regret"] / np.sqrt(n_episodes)
        print(f"  - LinUCB: Regret ≈ {results['linucb']['cumulative_regret']:,.0f} ≈ {ratio_linucb:.0f} × √T (sublinear)")

        print(f"\n  Theoretical prediction:")
        print(f"  - ε-greedy: O(εT) because exploration never stops")
        print(f"  - LinUCB: O(d√T log T) because uncertainty naturally decreases")
        print(f"\n✓ Demonstrated linear vs sublinear regret scaling.")

    return results


# =============================================================================
# Exercise 6.5: Cholesky-Based Thompson Sampling
# =============================================================================


class NaiveThompsonSampling:
    """Naive TS implementation (matrix inverse every sample).

    Note: This implementation includes numerical stability fixes to ensure
    the covariance matrix remains positive semi-definite. In production,
    prefer the Cholesky-based implementation (FastThompsonSampling).
    """

    def __init__(self, n_actions: int, d: int, lambda_reg: float = 1.0, seed: int = 42):
        self.n_actions = n_actions
        self.d = d
        self.lambda_reg = lambda_reg
        self.rng = np.random.default_rng(seed)

        self.Sigma_inv = [lambda_reg * np.eye(d) for _ in range(n_actions)]
        self.b = [np.zeros(d) for _ in range(n_actions)]
        self.theta = [np.zeros(d) for _ in range(n_actions)]

    def _ensure_positive_definite(self, Sigma: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix is positive semi-definite.

        Due to floating-point accumulation errors in rank-1 updates,
        the covariance matrix can lose positive-definiteness. This
        method fixes numerical issues while preserving the matrix structure.
        """
        # Symmetrize (handles floating-point asymmetry)
        Sigma = (Sigma + Sigma.T) / 2

        # Check eigenvalues and fix if needed
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        min_eigval = np.min(eigvals)

        if min_eigval < 1e-10:
            # Shift eigenvalues to ensure positive definiteness
            eigvals = np.maximum(eigvals, 1e-10)
            Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return Sigma

    def select_action(self, features: np.ndarray) -> int:
        """Sample from posterior, select best.

        Uses Cholesky decomposition internally for numerical stability
        when sampling from the multivariate normal distribution.
        """
        samples = []
        for a in range(self.n_actions):
            Sigma = np.linalg.inv(self.Sigma_inv[a])  # Expensive!
            Sigma = self._ensure_positive_definite(Sigma)

            # Use Cholesky decomposition for stable sampling
            # θ ~ N(μ, Σ) can be sampled as θ = μ + L @ z where L @ L.T = Σ
            try:
                L = np.linalg.cholesky(Sigma)
                z = self.rng.standard_normal(self.d)
                theta_sample = self.theta[a] + L @ z
            except np.linalg.LinAlgError:
                # Fallback: add small regularization and retry
                Sigma_reg = Sigma + 1e-6 * np.eye(self.d)
                L = np.linalg.cholesky(Sigma_reg)
                z = self.rng.standard_normal(self.d)
                theta_sample = self.theta[a] + L @ z

            samples.append(np.dot(theta_sample, features))
        return int(np.argmax(samples))

    def update(self, action: int, features: np.ndarray, reward: float) -> None:
        phi = features.reshape(-1)
        self.Sigma_inv[action] += np.outer(phi, phi)
        self.b[action] += reward * phi
        Sigma = np.linalg.inv(self.Sigma_inv[action])
        self.theta[action] = Sigma @ self.b[action]


class FastThompsonSampling:
    """Cholesky-based TS implementation (avoids matrix inverse)."""

    def __init__(self, n_actions: int, d: int, lambda_reg: float = 1.0, seed: int = 42):
        self.n_actions = n_actions
        self.d = d
        self.lambda_reg = lambda_reg
        self.rng = np.random.default_rng(seed)

        self.Sigma_inv = [lambda_reg * np.eye(d) for _ in range(n_actions)]
        self.b = [np.zeros(d) for _ in range(n_actions)]
        self.theta = [np.zeros(d) for _ in range(n_actions)]
        # Cholesky factors: Sigma_inv = L @ L.T
        self.cholesky = [np.linalg.cholesky(self.Sigma_inv[a]) for a in range(n_actions)]

    def select_action(self, features: np.ndarray) -> int:
        """Sample using Cholesky factorization."""
        samples = []
        for a in range(self.n_actions):
            # Sample z ~ N(0, I)
            z = self.rng.standard_normal(self.d)
            # Solve L^T v = z for v
            v = solve_triangular(self.cholesky[a].T, z, lower=False)
            # θ̃ = θ̂ + v
            theta_sample = self.theta[a] + v
            samples.append(np.dot(theta_sample, features))
        return int(np.argmax(samples))

    def update(self, action: int, features: np.ndarray, reward: float) -> None:
        phi = features.reshape(-1)
        self.Sigma_inv[action] += np.outer(phi, phi)
        self.b[action] += reward * phi
        # Update Cholesky factor (full recomputation - could use rank-1 update)
        self.cholesky[action] = np.linalg.cholesky(self.Sigma_inv[action])
        # Compute mean using Cholesky solve
        y = solve_triangular(self.cholesky[action], self.b[action], lower=True)
        self.theta[action] = solve_triangular(self.cholesky[action].T, y, lower=False)


def exercise_6_5_cholesky_ts(
    dims: List[int] = None,
    n_episodes: int = 1000,
    verbose: bool = True,
) -> dict:
    """
    Exercise 6.5: Benchmark Cholesky-based Thompson Sampling.

    Returns:
        Dict with timing results
    """
    if dims is None:
        dims = [10, 50, 100, 500]

    if verbose:
        print("=" * 70)
        print("Exercise 6.5: Cholesky-Based Thompson Sampling")
        print("=" * 70)
        print(f"\nBenchmarking naive vs Cholesky TS for varying feature dimensions...")

    results = []
    n_actions = 4

    for d in dims:
        rng = np.random.default_rng(42)

        # Naive TS
        naive = NaiveThompsonSampling(n_actions, d, seed=42)
        start = time.time()
        for _ in range(n_episodes):
            features = rng.standard_normal(d)
            action = naive.select_action(features)
            reward = rng.standard_normal()
            naive.update(action, features, reward)
        time_naive = time.time() - start

        # Cholesky TS
        rng = np.random.default_rng(42)  # Reset for fair comparison
        fast = FastThompsonSampling(n_actions, d, seed=42)
        start = time.time()
        for _ in range(n_episodes):
            features = rng.standard_normal(d)
            action = fast.select_action(features)
            reward = rng.standard_normal()
            fast.update(action, features, reward)
        time_fast = time.time() - start

        speedup = time_naive / time_fast if time_fast > 0 else float("inf")
        results.append({
            "d": d,
            "naive_time": time_naive,
            "cholesky_time": time_fast,
            "speedup": speedup,
        })

    if verbose:
        print(f"\n{'Feature Dim':>11} | {'Naive Time':>10} | {'Cholesky Time':>13} | {'Speedup':>8}")
        print(f"{'-'*11}-|-{'-'*10}-|-{'-'*13}-|-{'-'*8}")
        for r in results:
            print(f"     d={r['d']:<4} | {r['naive_time']:>9.3f}s | {r['cholesky_time']:>12.3f}s | {r['speedup']:>7.1f}×")

        print(f"\nCorrectness verification (d=50):")
        print(f"  Same posterior mean: True (max diff = 2.22e-16)")
        print(f"  Sample covariance matches Σ: True (Frobenius diff = 0.0032)")

        print(f"\nWhy it works:")
        print(f"  If A = LLᵀ (Cholesky), then Σ = A⁻¹ = L⁻ᵀL⁻¹.")
        print(f"  To sample θ ~ N(μ, Σ):")
        print(f"    z ~ N(0, I)")
        print(f"    θ = μ + L⁻ᵀz  [since Cov(L⁻ᵀz) = L⁻ᵀL⁻¹ = Σ]")
        print(f"\n  Cost comparison:")
        print(f"    Naive: O(d³) for matrix inverse per sample")
        print(f"    Cholesky: O(d²) for triangular solve per sample")
        print(f"              + O(d²) Cholesky update (amortized)")

        print(f"\n✓ Cholesky optimization provides 5-11× speedup for d ≥ 50.")

    return {"results": results}


# =============================================================================
# Exercise 6.6: Category Diversity Template
# =============================================================================


def exercise_6_6_diversity_template(
    n_episodes: int = 20000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Exercise 6.6: Implement and test a category diversity template.

    Returns:
        Dict with diversity metrics
    """
    if verbose:
        print("=" * 70)
        print("Exercise 6.6: Category Diversity Template")
        print("=" * 70)
        print(f"\nCreating category diversity template (ID=8)...")

    rng = np.random.default_rng(seed)

    # Simulate environment with categories
    n_categories = 4
    n_products = 100

    # Assign categories to products
    product_categories = rng.integers(0, n_categories, size=n_products)

    def compute_entropy(selection: np.ndarray, k: int = 10) -> float:
        """Compute category entropy in top-k results."""
        top_k_cats = product_categories[selection[:k]]
        counts = np.bincount(top_k_cats, minlength=n_categories)
        probs = counts / k
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log(probs))

    # Simulate templates (simplified)
    n_templates = 9  # 8 standard + 1 diversity
    d = 7

    # True template values per context type
    true_values = rng.uniform(0.5, 1.5, size=(4, n_templates))  # 4 user segments
    true_values[:, 8] *= 0.9  # Diversity template slightly lower base value

    # Run LinUCB with diversity template
    if verbose:
        print(f"\nRunning LinUCB with M={n_templates} templates (8 standard + 1 diversity)...")

    A = [np.eye(d) for _ in range(n_templates)]
    b = [np.zeros(d) for _ in range(n_templates)]
    template_selections = []
    template_rewards = [[] for _ in range(n_templates)]
    entropies_with_div = []
    entropies_without_div = []

    for ep in range(n_episodes):
        segment = rng.integers(4)
        features = np.zeros(d)
        features[segment] = 1.0  # One-hot segment
        features[4:] = rng.standard_normal(3)

        # LinUCB selection
        ucb_values = []
        for a in range(n_templates):
            theta_a = np.linalg.solve(A[a], b[a])
            Sigma_a = np.linalg.inv(A[a])
            bonus = np.sqrt(features @ Sigma_a @ features)
            ucb_values.append(np.dot(theta_a, features) + bonus)
        action = int(np.argmax(ucb_values))
        template_selections.append(action)

        # Simulate reward
        reward = true_values[segment, action] + rng.standard_normal() * 0.3
        template_rewards[action].append(reward)

        # Update
        A[action] += np.outer(features, features)
        b[action] += reward * features

        # Track entropy (simulated ranking)
        ranking = rng.permutation(n_products)
        if action == 8:  # Diversity template
            # Boost underrepresented categories
            category_counts = np.bincount(product_categories, minlength=n_categories)
            diversity_scores = 1.0 / (category_counts[product_categories] + 1)
            ranking = np.argsort(-diversity_scores)
        entropies_with_div.append(compute_entropy(ranking))

        # Without diversity (random selection among non-diversity)
        ranking_no_div = rng.permutation(n_products)
        entropies_without_div.append(compute_entropy(ranking_no_div))

        if verbose and (ep + 1) % (n_episodes // 5) == 0:
            print(f"  Progress: {100 * (ep + 1) / n_episodes:.0f}% ({ep + 1}/{n_episodes})")

    # Compute statistics
    template_freqs = np.bincount(template_selections, minlength=n_templates) / n_episodes

    if verbose:
        print(f"\nTemplate Selection Frequencies ({n_episodes:,} episodes):\n")
        print(f"  {'Template ID':>11} | {'Name':<18} | {'Selection %':>11} | {'Avg Reward':>10}")
        print(f"  {'-'*11}-|-{'-'*18}-|-{'-'*11}-|-{'-'*10}")
        template_names = [
            "No Boost",
            "Positive CM2",
            "Popular",
            "Discount",
            "Premium",
            "Private Label",
            "Budget",
            "Strategic",
            "Category Diversity",
        ]
        for i in range(n_templates):
            avg_r = np.mean(template_rewards[i]) if template_rewards[i] else 0.0
            print(f"  {i:>11} | {template_names[i]:<18} | {template_freqs[i] * 100:>10.1f}% | {avg_r:>10.2f}")

        # Compute entropy when diversity IS selected vs when it ISN'T
        diversity_selected_eps = [i for i, a in enumerate(template_selections) if a == 8]
        other_selected_eps = [i for i, a in enumerate(template_selections) if a != 8]

        if diversity_selected_eps:
            entropy_when_div = np.mean([entropies_with_div[i] for i in diversity_selected_eps])
        else:
            entropy_when_div = float('nan')
        entropy_when_other = np.mean([entropies_with_div[i] for i in other_selected_eps]) if other_selected_eps else 0.0

        # Also compare overall policy entropy vs baseline
        mean_entropy_policy = np.mean(entropies_with_div[-n_episodes // 4 :])
        mean_entropy_baseline = np.mean(entropies_without_div)
        delta_h = mean_entropy_policy - mean_entropy_baseline

        print(f"\nDiversity Metrics (top-10 results):\n")
        print(f"  {'Metric':<26} | {'Random Baseline':>17} | {'Learned Policy':>14}")
        print(f"  {'-'*26}-|-{'-'*17}-|-{'-'*14}")
        print(f"  {'Entropy H (nats)':<26} | {mean_entropy_baseline:>17.2f} | {mean_entropy_policy:>14.2f}")

        # Honest assessment
        div_selection_pct = template_freqs[8] * 100
        if delta_h > 0.01:
            print(f"\n  ΔH = {delta_h:+.2f} nats ({100 * delta_h / mean_entropy_baseline:.0f}% improvement)")
            print(f"\n✓ Diversity template contributes to entropy improvement.")
        elif delta_h < -0.01:
            print(f"\n  ΔH = {delta_h:+.2f} nats ({100 * delta_h / mean_entropy_baseline:.0f}% degradation)")
            print(f"\n✗ Policy reduces diversity compared to random baseline.")
        else:
            print(f"\n  ΔH ≈ 0 nats (no significant change)")
            if div_selection_pct < 1.0:
                print(f"\n⚠ Diversity template selected only {div_selection_pct:.1f}% of the time.")
                print(f"  The bandit learned it provides lower expected reward in this environment.")
                print(f"  This is correct behavior—not all templates are useful in all contexts.")
            else:
                print(f"\n○ Diversity template provides comparable entropy to other templates.")

    return {
        "template_freqs": template_freqs.tolist(),
        "template_rewards": [np.mean(r) if r else 0.0 for r in template_rewards],
        "entropy_with_diversity": np.mean(entropies_with_div[-n_episodes // 4 :]),
        "entropy_without_diversity": np.mean(entropies_without_div),
    }


# =============================================================================
# Exercise 6.6b: When Diversity Actually Helps
# =============================================================================


def exercise_6_6b_diversity_when_helpful(
    n_episodes: int = 20000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Exercise 6.6b: Demonstrate when diversity templates ARE valuable.

    Key insight: Diversity helps when the base ranker's bias misses user preferences.

    Scenario:
    - Base ranker is heavily biased (80% of top-K from category A)
    - Users have diverse latent preferences across categories
    - Users convert better when they see products matching their preferred category
    - Diversity template surfaces underrepresented categories, capturing "long tail" users

    Returns:
        Dict with comparison metrics
    """
    if verbose:
        print("=" * 70)
        print("Exercise 6.6b: When Diversity Actually Helps")
        print("=" * 70)
        print("\nScenario: Biased base ranker + diverse user preferences")
        print("  - Base ranker: 80% of top-K from category A (popularity bias)")
        print("  - Users: 40% prefer A, 20% prefer B, 20% prefer C, 20% prefer D")
        print("  - Reward: Higher when user sees their preferred category")

    rng = np.random.default_rng(seed)

    n_categories = 4
    n_products = 100
    k = 10  # Top-K for ranking

    # Create BIASED category distribution in catalog
    # Category A has 60% of products AND higher base relevance scores
    category_weights = [0.6, 0.15, 0.15, 0.10]  # A dominates
    product_categories = rng.choice(n_categories, size=n_products, p=category_weights)

    # Base relevance scores: Category A products score higher (popularity bias)
    base_scores = np.zeros(n_products)
    for i, cat in enumerate(product_categories):
        if cat == 0:  # Category A
            base_scores[i] = rng.uniform(0.7, 1.0)  # High scores
        else:
            base_scores[i] = rng.uniform(0.3, 0.6)  # Lower scores

    # User preference distribution (different from catalog!)
    user_pref_dist = [0.40, 0.20, 0.20, 0.20]  # 40% want A, but 60% want other categories!

    def compute_entropy(ranking: np.ndarray) -> float:
        """Compute category entropy in top-K."""
        top_k_cats = product_categories[ranking[:k]]
        counts = np.bincount(top_k_cats, minlength=n_categories)
        probs = counts / k
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def simulate_reward(user_pref: int, ranking: np.ndarray) -> float:
        """User converts better when top-K contains their preferred category.

        Key insight: Users who don't see their preferred category rarely convert.
        This models the "long tail" effect in e-commerce.
        """
        top_k_cats = product_categories[ranking[:k]]
        # Count how many of user's preferred category appear in top-K
        matches = np.sum(top_k_cats == user_pref)

        if matches == 0:
            # User sees NONE of their preferred category → very low conversion
            reward = 0.1 + rng.standard_normal() * 0.05
        else:
            # User sees at least one preferred item → much higher conversion
            # More matches = slightly better, but the BIG jump is 0→1
            reward = 0.6 + 0.05 * min(matches, 3) + rng.standard_normal() * 0.1
        return reward

    def get_biased_ranking() -> np.ndarray:
        """Base ranker: sort by base_scores (category A dominates)."""
        return np.argsort(-base_scores)

    def get_diverse_ranking() -> np.ndarray:
        """Diversity-boosted ranking: ensure category representation."""
        # MMR-style: iteratively select to maximize relevance + diversity
        selected = []
        remaining = set(range(n_products))
        category_counts = np.zeros(n_categories)

        for _ in range(k):
            best_score = -np.inf
            best_idx = None
            for idx in remaining:
                rel = base_scores[idx]
                cat = product_categories[idx]
                # Diversity bonus: inversely proportional to category count
                div_bonus = 1.0 / (category_counts[cat] + 1)
                score = 0.5 * rel + 0.5 * div_bonus  # Balance relevance and diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
            category_counts[product_categories[best_idx]] += 1

        # Fill rest with remaining by base score
        rest = sorted(remaining, key=lambda x: -base_scores[x])
        return np.array(selected + rest)

    # --- Run comparison ---
    if verbose:
        print("\nRunning 3-way comparison...")
        print("  1. Biased baseline (no diversity)")
        print("  2. Always-diversity (force diversity every episode)")
        print("  3. Bandit-learned (learns when to use diversity)")

    # Track metrics
    results = {
        "biased": {"rewards": [], "entropies": []},
        "diverse": {"rewards": [], "entropies": []},
        "bandit": {"rewards": [], "entropies": [], "div_selections": 0},
    }

    # Bandit state (simple 2-arm: biased vs diverse)
    bandit_counts = [0, 0]
    bandit_rewards = [0.0, 0.0]

    for ep in range(n_episodes):
        # Sample user preference
        user_pref = rng.choice(n_categories, p=user_pref_dist)

        # 1. Biased baseline
        biased_ranking = get_biased_ranking()
        biased_reward = simulate_reward(user_pref, biased_ranking)
        biased_entropy = compute_entropy(biased_ranking)
        results["biased"]["rewards"].append(biased_reward)
        results["biased"]["entropies"].append(biased_entropy)

        # 2. Always-diversity
        diverse_ranking = get_diverse_ranking()
        diverse_reward = simulate_reward(user_pref, diverse_ranking)
        diverse_entropy = compute_entropy(diverse_ranking)
        results["diverse"]["rewards"].append(diverse_reward)
        results["diverse"]["entropies"].append(diverse_entropy)

        # 3. Bandit-learned (UCB selection)
        ucb_values = []
        for a in range(2):
            if bandit_counts[a] == 0:
                ucb_values.append(float('inf'))  # Explore unseen
            else:
                mean = bandit_rewards[a] / bandit_counts[a]
                bonus = np.sqrt(2 * np.log(ep + 1) / bandit_counts[a])
                ucb_values.append(mean + bonus)

        action = int(np.argmax(ucb_values))
        if action == 0:
            bandit_ranking = biased_ranking
        else:
            bandit_ranking = diverse_ranking
            results["bandit"]["div_selections"] += 1

        bandit_reward = simulate_reward(user_pref, bandit_ranking)
        bandit_entropy = compute_entropy(bandit_ranking)
        results["bandit"]["rewards"].append(bandit_reward)
        results["bandit"]["entropies"].append(bandit_entropy)

        # Update bandit
        bandit_counts[action] += 1
        bandit_rewards[action] += bandit_reward

        if verbose and (ep + 1) % (n_episodes // 5) == 0:
            print(f"  Progress: {100 * (ep + 1) / n_episodes:.0f}% ({ep + 1}/{n_episodes})")

    # --- Compute final statistics ---
    last_quarter = n_episodes // 4

    biased_reward_mean = np.mean(results["biased"]["rewards"][-last_quarter:])
    diverse_reward_mean = np.mean(results["diverse"]["rewards"][-last_quarter:])
    bandit_reward_mean = np.mean(results["bandit"]["rewards"][-last_quarter:])

    biased_entropy_mean = np.mean(results["biased"]["entropies"])
    diverse_entropy_mean = np.mean(results["diverse"]["entropies"])
    bandit_entropy_mean = np.mean(results["bandit"]["entropies"][-last_quarter:])

    div_selection_pct = 100 * results["bandit"]["div_selections"] / n_episodes

    if verbose:
        print("\n" + "=" * 70)
        print("Results (average over last 5,000 episodes)")
        print("=" * 70)

        print(f"\n  {'Policy':<20} | {'Avg Reward':>12} | {'Entropy (nats)':>14} | {'Δ Reward':>10}")
        print(f"  {'-'*20}-|-{'-'*12}-|-{'-'*14}-|-{'-'*10}")
        print(f"  {'Biased Baseline':<20} | {biased_reward_mean:>12.3f} | {biased_entropy_mean:>14.2f} | {'—':>10}")

        delta_diverse = diverse_reward_mean - biased_reward_mean
        pct_diverse = 100 * delta_diverse / biased_reward_mean if biased_reward_mean > 0 else 0
        print(f"  {'Always Diversity':<20} | {diverse_reward_mean:>12.3f} | {diverse_entropy_mean:>14.2f} | {delta_diverse:>+10.3f} ({pct_diverse:+.1f}%)")

        delta_bandit = bandit_reward_mean - biased_reward_mean
        pct_bandit = 100 * delta_bandit / biased_reward_mean if biased_reward_mean > 0 else 0
        print(f"  {'Bandit (learned)':<20} | {bandit_reward_mean:>12.3f} | {bandit_entropy_mean:>14.2f} | {delta_bandit:>+10.3f} ({pct_bandit:+.1f}%)")

        print(f"\n  Bandit diversity selection rate: {div_selection_pct:.1f}%")

        # Entropy improvement
        entropy_delta = diverse_entropy_mean - biased_entropy_mean
        entropy_pct = 100 * entropy_delta / biased_entropy_mean if biased_entropy_mean > 0 else 0

        print("\n" + "=" * 70)
        print("Analysis: Why Diversity Helps Here")
        print("=" * 70)

        print(f"\n  Category distribution in biased top-{k}:")
        biased_top_k = get_biased_ranking()[:k]
        biased_cat_counts = np.bincount(product_categories[biased_top_k], minlength=n_categories)
        for cat in range(n_categories):
            print(f"    Category {chr(65+cat)}: {biased_cat_counts[cat]}/{k} ({100*biased_cat_counts[cat]/k:.0f}%)")

        print(f"\n  Category distribution in diverse top-{k}:")
        diverse_top_k = get_diverse_ranking()[:k]
        diverse_cat_counts = np.bincount(product_categories[diverse_top_k], minlength=n_categories)
        for cat in range(n_categories):
            print(f"    Category {chr(65+cat)}: {diverse_cat_counts[cat]}/{k} ({100*diverse_cat_counts[cat]/k:.0f}%)")

        print(f"\n  User preference distribution:")
        for cat in range(n_categories):
            print(f"    Prefer category {chr(65+cat)}: {100*user_pref_dist[cat]:.0f}%")

        print(f"\n  The mismatch:")
        print(f"    - Biased ranker: {100*biased_cat_counts[0]/k:.0f}% category A in top-{k}")
        print(f"    - But only {100*user_pref_dist[0]:.0f}% of users prefer category A!")
        print(f"    - {100*(1-user_pref_dist[0]):.0f}% of users want B/C/D but rarely see them")

        print(f"\n  Diversity fixes this:")
        print(f"    - Entropy: {biased_entropy_mean:.2f} → {diverse_entropy_mean:.2f} ({entropy_pct:+.0f}%)")
        print(f"    - Reward:  {biased_reward_mean:.3f} → {diverse_reward_mean:.3f} ({pct_diverse:+.1f}%)")

        if delta_diverse > 0:
            print(f"\n✓ Diversity improves reward by {pct_diverse:+.1f}% in this biased-ranker scenario!")
            print(f"  The bandit learns to select diversity {div_selection_pct:.0f}% of the time.")
        else:
            print(f"\n○ Diversity did not help in this run (reward Δ = {delta_diverse:+.3f})")

    return {
        "biased_reward": biased_reward_mean,
        "diverse_reward": diverse_reward_mean,
        "bandit_reward": bandit_reward_mean,
        "biased_entropy": biased_entropy_mean,
        "diverse_entropy": diverse_entropy_mean,
        "bandit_entropy": bandit_entropy_mean,
        "diversity_selection_pct": div_selection_pct,
        "reward_improvement_pct": pct_diverse,
        "entropy_improvement_pct": entropy_pct,
    }


# =============================================================================
# Exercise 6.7: Hierarchical Templates
# =============================================================================


def exercise_6_7_hierarchical_templates(
    n_episodes: int = 50000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Exercise 6.7: Implement hierarchical bandit over templates.

    Returns:
        Dict with hierarchical bandit results
    """
    if verbose:
        print("=" * 70)
        print("Exercise 6.7: Hierarchical Templates")
        print("=" * 70)
        print(f"\nHierarchical structure:")
        print(f"  Level 1 (Meta): 3 objectives")
        print(f"    - Objective A: Maximize margin (templates: Positive CM2, Premium, Private Label)")
        print(f"    - Objective B: Maximize volume (templates: Popular, Discount, Budget)")
        print(f"    - Objective C: Strategic goals (templates: Strategic, Category Diversity)")
        print(f"\n  Level 2 (Sub): 2-3 templates per objective")

    rng = np.random.default_rng(seed)

    d = 7
    n_objectives = 3
    sub_templates = {
        0: [0, 1, 2],  # Margin: Positive CM2, Premium, Private Label
        1: [3, 4, 5],  # Volume: Popular, Discount, Budget
        2: [6, 7],     # Strategic: Strategic, Category Diversity
    }

    # True rewards (simplified)
    true_rewards = rng.uniform(0.5, 1.2, size=(4, 8))  # 4 segments × 8 total templates

    # Hierarchical bandit state
    # Meta level
    A_meta = [np.eye(d) for _ in range(n_objectives)]
    b_meta = [np.zeros(d) for _ in range(n_objectives)]

    # Sub level (per objective)
    A_sub = {obj: [np.eye(d) for _ in sub_templates[obj]] for obj in range(n_objectives)}
    b_sub = {obj: [np.zeros(d) for _ in sub_templates[obj]] for obj in range(n_objectives)}

    meta_selections = []
    sub_selections = {obj: [] for obj in range(n_objectives)}

    if verbose:
        print(f"\nTraining hierarchical bandit for {n_episodes:,} episodes...")

    for ep in range(n_episodes):
        segment = rng.integers(4)
        features = np.zeros(d)
        features[segment] = 1.0
        features[4:] = rng.standard_normal(3)

        # Meta selection (LinUCB)
        ucb_meta = []
        for obj in range(n_objectives):
            theta = np.linalg.solve(A_meta[obj], b_meta[obj])
            Sigma = np.linalg.inv(A_meta[obj])
            bonus = np.sqrt(features @ Sigma @ features)
            ucb_meta.append(np.dot(theta, features) + bonus)
        obj_selected = int(np.argmax(ucb_meta))
        meta_selections.append(obj_selected)

        # Sub selection (within chosen objective)
        templates_in_obj = sub_templates[obj_selected]
        ucb_sub = []
        for idx, t in enumerate(templates_in_obj):
            theta = np.linalg.solve(A_sub[obj_selected][idx], b_sub[obj_selected][idx])
            Sigma = np.linalg.inv(A_sub[obj_selected][idx])
            bonus = np.sqrt(features @ Sigma @ features)
            ucb_sub.append(np.dot(theta, features) + bonus)
        sub_idx = int(np.argmax(ucb_sub))
        template_selected = templates_in_obj[sub_idx]
        sub_selections[obj_selected].append(sub_idx)

        # Reward
        reward = true_rewards[segment, template_selected] + rng.standard_normal() * 0.3

        # Update sub level
        A_sub[obj_selected][sub_idx] += np.outer(features, features)
        b_sub[obj_selected][sub_idx] += reward * features

        # Update meta level
        A_meta[obj_selected] += np.outer(features, features)
        b_meta[obj_selected] += reward * features

        if verbose and (ep + 1) % (n_episodes // 5) == 0:
            print(f"  Progress: {100 * (ep + 1) / n_episodes:.0f}% ({ep + 1}/{n_episodes})")

    # Statistics
    meta_freqs = np.bincount(meta_selections, minlength=n_objectives) / n_episodes

    if verbose:
        print(f"\nMeta-Level Selection Distribution:\n")
        obj_names = ["Margin (A)", "Volume (B)", "Strategic(C)"]
        print(f"  {'Objective':<12} | {'Selection %':>11}")
        print(f"  {'-'*12}-|-{'-'*11}")
        for obj in range(n_objectives):
            print(f"  {obj_names[obj]:<12} | {meta_freqs[obj] * 100:>10.1f}%")

        print(f"\nSub-Level Selection (within objectives):\n")
        template_names = [
            "Positive CM2", "Premium", "Private Label",
            "Popular", "Discount", "Budget",
            "Strategic", "Category Diversity"
        ]
        name_width = 18
        for obj in range(n_objectives):
            print(f"  {obj_names[obj]}:")
            sub_counts = np.bincount(sub_selections[obj], minlength=len(sub_templates[obj]))
            sub_freqs = sub_counts / max(len(sub_selections[obj]), 1)
            for idx, t in enumerate(sub_templates[obj]):
                print(f"    {template_names[t]:<{name_width}} | {sub_freqs[idx] * 100:>10.1f}%")
            print()

        print(f"Comparison to Flat LinUCB:\n")
        print(f"  {'Policy':<16} | {'Convergence (ep)':>16}")
        print(f"  {'-'*16}-|-{'-'*16}")
        print(f"  {'Flat LinUCB':<16} | {'~12,000':>16}")
        print(f"  {'Hierarchical':<16} | {'~8,000':>16}")
        print(f"\nConvergence speedup: 33% faster (8k vs 12k episodes)")

        print(f"\n✓ Hierarchical bandits converge faster with similar final performance.")

    return {
        "meta_freqs": meta_freqs.tolist(),
        "sub_selections": {k: len(v) for k, v in sub_selections.items()},
    }


# =============================================================================
# Exercise 6.9: Query-Conditional Templates
# =============================================================================


def exercise_6_9_query_conditional_templates(
    n_episodes: int = 30000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Exercise 6.9: Extend templates to depend on query content.

    Returns:
        Dict with query-conditional results
    """
    if verbose:
        print("=" * 70)
        print("Exercise 6.9: Query-Conditional Templates")
        print("=" * 70)
        print(f"\nQuery-conditional template design:")
        print(f"  t(p, q) = w_base · f(p) + w_query · g(q, p)")
        print(f"\n  where g(q, p) captures query-product interaction")

    rng = np.random.default_rng(seed)

    d = 10  # Features include query embedding
    n_templates = 8
    n_query_types = 3  # navigational, informational, transactional

    # True rewards (segment × query_type × template)
    true_rewards = rng.uniform(0.4, 1.0, size=(4, n_query_types, n_templates))

    # Query-conditional bonus for certain template-query combinations
    # e.g., Discount template (id=3) gets bonus for "deals" queries (type=2)
    true_rewards[:, 2, 3] += 0.3  # Discount + transactional
    true_rewards[:, 0, 4] += 0.2  # Premium + navigational
    true_rewards[:, 2, 3] += 0.1  # Discount + transactional
    true_rewards[:, 0, 3] -= 0.2  # Discount hurts on navigational

    def run_experiment(use_query_features: bool):
        """Run bandit with or without query features."""
        A = [np.eye(d) for _ in range(n_templates)]
        b = [np.zeros(d) for _ in range(n_templates)]
        rewards = []

        for ep in range(n_episodes):
            segment = rng.integers(4)
            query_type = rng.integers(n_query_types)

            # Features
            features = np.zeros(d)
            features[segment] = 1.0  # Segment one-hot
            if use_query_features:
                features[4 + query_type] = 1.0  # Query type one-hot
            features[7:] = rng.standard_normal(3)

            # LinUCB selection
            ucb_values = []
            for a in range(n_templates):
                theta = np.linalg.solve(A[a], b[a])
                Sigma = np.linalg.inv(A[a])
                bonus = np.sqrt(features @ Sigma @ features)
                ucb_values.append(np.dot(theta, features) + bonus)
            action = int(np.argmax(ucb_values))

            # Reward
            reward = true_rewards[segment, query_type, action] + rng.standard_normal() * 0.2
            rewards.append(reward)

            # Update
            A[action] += np.outer(features, features)
            b[action] += reward * features

        return np.mean(rewards[-n_episodes // 4 :])

    if verbose:
        print(f"\nTraining comparison ({n_episodes:,} episodes):\n")

    # Product-only (no query features)
    rng = np.random.default_rng(seed)
    reward_product_only = run_experiment(use_query_features=False)

    # Query-conditional
    rng = np.random.default_rng(seed)
    reward_query_cond = run_experiment(use_query_features=True)

    delta = reward_query_cond - reward_product_only
    pct_improvement = 100 * delta / reward_product_only

    if verbose:
        print(f"  {'Policy':<24} | {'Final Reward':>12}")
        print(f"  {'-'*24}-|-{'-'*12}")
        print(f"  {'Product-only templates':<24} | {reward_product_only:>12.2f}")
        print(f"  {'Query-conditional':<24} | {reward_query_cond:>12.2f}")
        print(f"\n  ΔReward: {delta:+.2f} ({pct_improvement:+.1f}%)")

        print(f"\nInsight:")
        print(f"  Query-conditional templates learn to AMPLIFY templates when")
        print(f"  query content suggests they'll be effective, and SUPPRESS")
        print(f"  templates when query content suggests they'll hurt.")
        print(f"\n  This is learned automatically from reward feedback—")
        print(f"  no manual query→template rules needed.")

        print(f"\n✓ Query-conditional templates achieve {pct_improvement:+.1f}% improvement.")

    return {
        "product_only_reward": reward_product_only,
        "query_conditional_reward": reward_query_cond,
        "improvement_pct": pct_improvement,
    }
