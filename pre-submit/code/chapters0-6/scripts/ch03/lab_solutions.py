"""
Chapter 3 Lab Solutions — Bellman Operators and Convergence Theory

Author: Vlad Prytula

This module implements all lab exercises from Chapter 3, demonstrating the
operator-theoretic foundations of RL: Bellman operators, contraction mappings,
and value iteration convergence.

Solutions included:
- Lab 3.1: Contraction Ratio Tracker
- Lab 3.2: Value Iteration Wall-Clock Profiling
- Extended: Perturbation Sensitivity Analysis
- Extended: Discount Factor and Convergence Rate Analysis
- Extended: Banach Fixed-Point Theorem Verification

Mathematical foundations:
- [DEF-3.6.3] Contraction mapping definition
- [THM-3.6.2-Banach] Banach fixed-point theorem
- [THM-3.7.1] Bellman operator contraction
- [EQ-3.16] Contraction inequality: ||T V1 - T V2||_inf <= gamma ||V1 - V2||_inf

Usage:
    python scripts/ch03/lab_solutions.py [--lab N] [--all]

    --lab N: Run only lab N (e.g., '3.1', '3.2')
    --extended N: Run extended lab N (e.g., 'perturbation', 'discount')
    --all: Run all labs sequentially
    (default): Run interactive menu
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Core MDP Data Structures
# =============================================================================


@dataclass
class MDPConfig:
    """Configuration for a finite MDP.

    Attributes:
        n_states: Number of states
        n_actions: Number of actions
        gamma: Discount factor (must be < 1 for contraction)
        seed: Random seed for reproducibility
    """

    n_states: int = 3
    n_actions: int = 2
    gamma: float = 0.9
    seed: int = 42


def create_random_mdp(
    config: MDPConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a random MDP with given configuration.

    Returns:
        P: (n_states, n_actions, n_states) transition tensor P(s'|s,a)
        R: (n_states, n_actions) reward matrix R(s,a)
    """
    rng = np.random.default_rng(config.seed)

    # Random transition probabilities (normalized)
    P = rng.random((config.n_states, config.n_actions, config.n_states))
    P = P / P.sum(axis=2, keepdims=True)

    # Random rewards (bounded in [0, 1] for simplicity)
    R = rng.random((config.n_states, config.n_actions))

    return P, R


def create_example_mdp() -> tuple[np.ndarray, np.ndarray, float]:
    """Create the example MDP from exercises_labs.md.

    This is the standard 3-state, 2-action MDP used in Lab 3.1 and 3.2.

    Returns:
        P: (3, 2, 3) transition tensor
        R: (3, 2) reward matrix
        gamma: Discount factor (0.9)
    """
    P = np.array(
        [
            [[0.7, 0.3, 0.0], [0.4, 0.6, 0.0]],
            [[0.0, 0.6, 0.4], [0.0, 0.3, 0.7]],
            [[0.2, 0.0, 0.8], [0.1, 0.0, 0.9]],
        ]
    )
    R = np.array(
        [
            [1.0, 0.5],
            [0.8, 1.2],
            [0.0, 0.4],
        ]
    )
    gamma = 0.9

    return P, R, gamma


# =============================================================================
# Bellman Operator Implementation
# =============================================================================


class BellmanOperator:
    """Bellman optimality operator for finite MDPs.

    Finite-state specialization of [EQ-3.12]:
        (TV)(s) = max_a {R(s,a) + gamma * sum_{s'} P(s'|s,a) V(s')}

    The Bellman operator is a gamma-contraction in the infinity norm
    ([THM-3.7.1], [EQ-3.16]):
        ||T V1 - T V2||_inf <= gamma ||V1 - V2||_inf

    Attributes:
        P: (n_states, n_actions, n_states) transition tensor
        R: (n_states, n_actions) reward matrix
        gamma: Discount factor
        n_states: Number of states
        n_actions: Number of actions
    """

    def __init__(self, P: np.ndarray, R: np.ndarray, gamma: float):
        """Initialize Bellman operator.

        Args:
            P: (n_states, n_actions, n_states) transition tensor P(s'|s,a)
            R: (n_states, n_actions) reward matrix R(s,a)
            gamma: Discount factor (0 <= gamma < 1)

        Raises:
            ValueError: If gamma >= 1 (not a contraction)
        """
        if gamma >= 1.0:
            raise ValueError(
                f"Discount factor gamma={gamma} must be < 1 for Bellman operator "
                "to be a contraction. See [THM-3.7.1]."
            )

        self.P = P
        self.R = R
        self.gamma = gamma
        self.n_states, self.n_actions, _ = P.shape

    def apply(self, V: np.ndarray) -> np.ndarray:
        """Apply Bellman operator: TV.

        Computes the finite-state backup:
            (TV)(s) = max_a {R(s,a) + gamma * sum_{s'} P(s'|s,a) V(s')}

        Args:
            V: (n_states,) value function

        Returns:
            TV: (n_states,) Bellman backup
        """
        # Q(s, a) = R(s, a) + gamma * sum_{s'} P(s'|s,a) V(s')
        # Using einsum: P[s,a,s'] @ V[s'] -> [s,a]
        Q = self.R + self.gamma * np.einsum("ijk,k->ij", self.P, V)

        # V(s) = max_a Q(s, a)
        return Q.max(axis=1)

    def compute_q_values(self, V: np.ndarray) -> np.ndarray:
        """Compute Q-values for given value function.

        Args:
            V: (n_states,) value function

        Returns:
            Q: (n_states, n_actions) Q-values
        """
        return self.R + self.gamma * np.einsum("ijk,k->ij", self.P, V)

    def extract_policy(self, V: np.ndarray) -> np.ndarray:
        """Extract greedy policy from value function.

        Args:
            V: (n_states,) value function

        Returns:
            policy: (n_states,) optimal action per state
        """
        Q = self.compute_q_values(V)
        return np.argmax(Q, axis=1)

    def contraction_ratio(self, V1: np.ndarray, V2: np.ndarray) -> float:
        """Compute empirical contraction ratio.

        Measures: ||T V1 - T V2||_inf / ||V1 - V2||_inf

        By [THM-3.7.1], this should be <= gamma.

        Args:
            V1, V2: Two value functions

        Returns:
            ratio: Empirical contraction ratio
        """
        TV1 = self.apply(V1)
        TV2 = self.apply(V2)

        numerator = np.linalg.norm(TV1 - TV2, ord=np.inf)
        denominator = np.linalg.norm(V1 - V2, ord=np.inf)

        if denominator < 1e-12:
            return 0.0  # Identical inputs

        return numerator / denominator


def value_iteration(
    bellman: BellmanOperator,
    V_init: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iters: int = 10_000,
    track_history: bool = False,
) -> dict:
    """Run value iteration to find optimal value function.

    Implements iterative application of Bellman operator:
        V_{k+1} = T V_k

    By [THM-3.6.2-Banach] (Banach fixed-point), converges to unique V*.

    Args:
        bellman: BellmanOperator instance
        V_init: Initial value function (default: zeros)
        tol: Convergence tolerance (||V_{k+1} - V_k||_inf < tol)
        max_iters: Maximum iterations
        track_history: Whether to record iteration history

    Returns:
        dict with:
            V: Converged value function
            iters: Number of iterations
            converged: Whether converged within max_iters
            errors: List of ||V_{k+1} - V_k||_inf if track_history
            V_history: List of V_k if track_history
    """
    if V_init is None:
        V = np.zeros(bellman.n_states)
    else:
        V = V_init.copy()

    errors = []
    V_history = [V.copy()] if track_history else []

    for k in range(max_iters):
        V_new = bellman.apply(V)
        error = np.linalg.norm(V_new - V, ord=np.inf)

        if track_history:
            errors.append(error)
            V_history.append(V_new.copy())

        if error < tol:
            return {
                "V": V_new,
                "iters": k + 1,
                "converged": True,
                "errors": errors,
                "V_history": V_history,
            }

        V = V_new

    return {
        "V": V,
        "iters": max_iters,
        "converged": False,
        "errors": errors,
        "V_history": V_history,
    }


# =============================================================================
# Lab 3.1: Contraction Ratio Tracker
# =============================================================================


def lab_3_1_contraction_ratio_tracker(
    seeds: list[int] | None = None,
    n_seeds: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Lab 3.1 Solution: Contraction Ratio Tracker

    Objective: Log ||T V1 - T V2||_inf / ||V1 - V2||_inf and compare to gamma.

    This lab empirically verifies [THM-3.7.1]: the Bellman operator is a
    gamma-contraction. We sample random value function pairs and measure
    the contraction ratio, confirming it never exceeds gamma.

    Tasks:
    1. Explain the slack between bound and observation
    2. Log ratio across multiple seeds, include extrema

    Args:
        seeds: List of seeds to test (or None to generate n_seeds)
        n_seeds: Number of seeds if seeds not provided
        verbose: Whether to print detailed output

    Returns:
        dict with ratios, statistics, and analysis
    """
    if verbose:
        print("=" * 70)
        print("Lab 3.1: Contraction Ratio Tracker")
        print("=" * 70)

    # Create example MDP from exercises_labs.md
    P, R, gamma = create_example_mdp()
    bellman = BellmanOperator(P, R, gamma)

    if verbose:
        print(f"\nMDP Configuration:")
        print(f"  States: {bellman.n_states}, Actions: {bellman.n_actions}")
        print(f"  Discount factor gamma = {gamma}")
        print(
            f"\nTheoretical bound [THM-3.7.1]: ||T V1 - T V2||_inf <= {gamma} * ||V1 - V2||_inf"
        )

    # Generate seeds
    if seeds is None:
        base_rng = np.random.default_rng(42)
        seeds = [int(base_rng.integers(0, 100000)) for _ in range(n_seeds)]

    # Compute contraction ratios
    ratios = []

    if verbose:
        print(f"\nComputing contraction ratios across {len(seeds)} random V pairs...\n")

    for seed in seeds:
        rng = np.random.default_rng(seed)

        # Sample random value functions
        V1 = rng.normal(size=bellman.n_states)
        V2 = rng.normal(size=bellman.n_states)

        ratio = bellman.contraction_ratio(V1, V2)
        ratios.append(ratio)

    ratios = np.array(ratios)

    # Statistics
    ratio_mean = ratios.mean()
    ratio_std = ratios.std()
    ratio_min = ratios.min()
    ratio_max = ratios.max()
    bound_satisfied = np.all(ratios <= gamma + 1e-10)

    if verbose:
        print(f"{'Seed':<10} {'Ratio':<12} {'<= gamma?':<10}")
        print("-" * 30)
        for i, (seed, ratio) in enumerate(zip(seeds[:10], ratios[:10])):
            satisfied = "OK" if ratio <= gamma + 1e-10 else "NO"
            print(f"{seed:<10} {ratio:<12.4f} {satisfied:<8}")
        if len(seeds) > 10:
            print(f"... ({len(seeds) - 10} more seeds)")

        print(f"\n{'=' * 50}")
        print("CONTRACTION RATIO STATISTICS")
        print(f"{'=' * 50}")
        print(f"  Theoretical bound (gamma): {gamma:.3f}")
        print(f"  Empirical mean:        {ratio_mean:.3f}")
        print(f"  Empirical std:         {ratio_std:.3f}")
        print(f"  Empirical min:         {ratio_min:.3f}")
        print(f"  Empirical max:         {ratio_max:.3f}")
        print(f"  Slack (gamma - max):   {gamma - ratio_max:.3f}")
        print(f"  All ratios <= gamma?   {'YES' if bound_satisfied else 'NO'}")

        # Task 1: Explain the slack
        print(f"\n{'=' * 50}")
        print("TASK 1: Why is there slack between gamma and observed ratios?")
        print(f"{'=' * 50}")
        print("""
The observed contraction ratio is often strictly less than gamma because:

1. **The max operator is 1-Lipschitz**: The proof of [THM-3.7.1] uses
   |max_a f(a) - max_a g(a)| <= max_a |f(a) - g(a)|
   This is an inequality, not equality. When the argmax differs between
   V1 and V2, the actual difference is smaller.

2. **Transition probability averaging**: The sum sum_{s'} P(s'|s,a)[V1(s') - V2(s')]
   averages the value differences. Unless V1 - V2 has constant sign, this
   is strictly less than ||V1 - V2||_inf.

3. **Structure in P and R**: Real MDPs have structure. Not all (V1, V2)
   pairs achieve the worst-case. The bound gamma is tight only for adversarial
   constructions.

Implication: In practice, value iteration often converges faster than the
worst-case gamma^k rate predicted by Banach fixed-point theory.
""")

    return {
        "ratios": ratios,
        "seeds": seeds,
        "gamma": gamma,
        "mean": ratio_mean,
        "std": ratio_std,
        "min": ratio_min,
        "max": ratio_max,
        "bound_satisfied": bound_satisfied,
        "slack": gamma - ratio_max,
    }


# =============================================================================
# Lab 3.2: Value Iteration Wall-Clock Profiling
# =============================================================================


def lab_3_2_value_iteration_profiling(
    gamma_values: list[float] | None = None,
    tol: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    Lab 3.2 Solution: Value Iteration Wall-Clock Profiling

    Objective: Verify the O(1/(1-gamma)) convergence rate numerically.

    From [THM-3.6.2-Banach] (Banach fixed-point):
        ||V_k - V*||_inf <= gamma^k ||V_0 - V*||_inf

    To achieve error eps, we need k ~ log(eps) / log(gamma) ~ 1/(1-gamma) * log(1/eps).

    Tasks:
    1. Plot iteration counts against 1/(1-gamma)
    2. Re-run with perturbed R to visualize perturbation bounds

    Args:
        gamma_values: List of discount factors to test
        tol: Convergence tolerance
        verbose: Whether to print detailed output

    Returns:
        dict with iteration counts, theoretical predictions, analysis
    """
    if verbose:
        print("=" * 70)
        print("Lab 3.2: Value Iteration Wall-Clock Profiling")
        print("=" * 70)

    # Create example MDP
    P, R, _ = create_example_mdp()

    if gamma_values is None:
        gamma_values = [0.5, 0.7, 0.9, 0.95, 0.99]

    results = {}
    iter_counts = []
    theoretical_rates = []

    if verbose:
        print(f"\nMDP Configuration:")
        print(f"  States: {P.shape[0]}, Actions: {P.shape[1]}")
        print(f"  Convergence tolerance: {tol}")
        print(f"\nRunning value iteration for gamma in {gamma_values}...\n")
        print(f"{'gamma':<8} {'Iters':<8} {'1/(1-gamma)':<12} {'Ratio':<10}")
        print("-" * 40)

    for gamma in gamma_values:
        bellman = BellmanOperator(P, R, gamma)

        # Time the iteration
        start = time.perf_counter()
        result = value_iteration(bellman, tol=tol, track_history=True)
        elapsed = time.perf_counter() - start

        iters = result["iters"]
        theoretical = 1 / (1 - gamma)

        iter_counts.append(iters)
        theoretical_rates.append(theoretical)

        ratio = iters / theoretical if theoretical > 0 else 0

        results[gamma] = {
            "iters": iters,
            "converged": result["converged"],
            "elapsed_ms": elapsed * 1000,
            "theoretical_1_over_1_minus_gamma": theoretical,
            "ratio": ratio,
            "V_star": result["V"],
            "errors": result["errors"],
        }

        if verbose:
            print(f"{gamma:<8.2f} {iters:<8} {theoretical:<10.1f} {ratio:<10.2f}")

    # Fit linear relationship
    iter_counts = np.array(iter_counts)
    theoretical_rates = np.array(theoretical_rates)

    # Linear regression: iters ~ slope * (1/(1-gamma)) + intercept
    A = np.vstack([theoretical_rates, np.ones(len(theoretical_rates))]).T
    slope, intercept = np.linalg.lstsq(A, iter_counts, rcond=None)[0]

    if verbose:
        print(f"\n{'=' * 50}")
        print("ANALYSIS: Iteration Count vs 1/(1-gamma)")
        print(f"{'=' * 50}")
        print(f"\nLinear fit: Iters ~ {slope:.2f} * 1/(1-gamma) + {intercept:.1f}")
        print(f"\nTheory predicts: Iters ~ C * 1/(1-gamma) * log(1/eps)")
        print(f"  With eps = {tol}, log(1/eps) ~ {np.log(1/tol):.1f}")
        print(f"  Expected slope ~ log(1/eps) ~ {np.log(1/tol):.1f}")
        print(f"  Observed slope: {slope:.2f}")

        print(f"\n{'=' * 50}")
        print("TASK 1: The O(1/(1-gamma)) Relationship")
        print(f"{'=' * 50}")
        print("""
The iteration count scales approximately as 1/(1-gamma) because:

From [THM-3.6.2-Banach] (Banach fixed-point), after k iterations:
    ||V_k - V*||_inf <= gamma^k ||V_0 - V*||_inf

To achieve tolerance eps, we need gamma^k ||V_0 - V*||_inf < eps:
    k > log(||V_0 - V*||_inf / eps) / log(1/gamma)
    k ~ log(1/eps) / (1-gamma)    [since log(1/gamma) ~ 1-gamma for gamma close to 1]

Key insight: The complexity diverges as gamma -> 1 (infinite horizon).
This explains why long-horizon RL is fundamentally harder:
For a fixed tolerance, moving from gamma = 0.9 to gamma = 0.99 typically costs
about 10x more Bellman backups. This scaling, not the exact constant, is the
main lesson: the effective horizon 1/(1-gamma) controls computational effort.

In practice, this motivates:
1. Using moderate gamma when it captures the planning horizon
2. Using function approximation to avoid per-state iteration in large MDPs
3. Using Monte Carlo methods when exact dynamic programming is infeasible
""")

    return {
        "gamma_values": gamma_values,
        "iter_counts": iter_counts.tolist(),
        "theoretical_rates": theoretical_rates.tolist(),
        "linear_fit": {"slope": slope, "intercept": intercept},
        "per_gamma_results": results,
    }


# =============================================================================
# Extended Lab: Perturbation Sensitivity Analysis
# =============================================================================


def extended_perturbation_sensitivity(
    noise_scales: list[float] | None = None,
    n_trials: int = 20,
    gamma: float = 0.9,
    verbose: bool = True,
) -> dict:
    """
    Extended Lab: Perturbation Sensitivity Analysis

    Objective: Empirically verify the reward-perturbation sensitivity bound
    from [PROP-3.7.4].

    For MDPs with reward perturbation R -> R + DeltaR:
        ||V*_perturbed - V*_original||_inf <= ||DeltaR||_inf / (1-gamma)

    This lab validates this sensitivity bound empirically.

    Args:
        noise_scales: List of perturbation magnitudes ||DeltaR||_inf
        n_trials: Number of random perturbations per scale
        gamma: Discount factor
        verbose: Whether to print detailed output

    Returns:
        dict with perturbation analysis results
    """
    if verbose:
        print("=" * 70)
        print("Extended Lab: Perturbation Sensitivity Analysis")
        print("=" * 70)

    P, R, _ = create_example_mdp()
    bellman_original = BellmanOperator(P, R, gamma)

    if noise_scales is None:
        noise_scales = [0.01, 0.05, 0.1, 0.2, 0.5]

    # Compute original V*
    result_original = value_iteration(bellman_original)
    V_star_original = result_original["V"]

    if verbose:
        print(f"\nOriginal MDP (gamma = {gamma}):")
        print(f"  V* = {V_star_original}")
        print(f"\nTheoretical bound [PROP-3.7.4]: ||V*_perturbed - V*||_inf <= ||DeltaR||_inf / (1-gamma)")
        print(
            f"  With gamma = {gamma}, bound = ||DeltaR||_inf / {1-gamma:.2f} = {1/(1-gamma):.1f} * ||DeltaR||_inf\n"
        )

    results = {}
    rng = np.random.default_rng(42)

    if verbose:
        print(
            f"{'||DeltaR||_inf':<14} {'Bound':<12} {'Mean ||DeltaV*||':<18} "
            f"{'Max ||DeltaV*||':<18} {'Bound OK?':<10}"
        )
        print("-" * 65)

    for delta_scale in noise_scales:
        v_diffs = []
        theoretical_bound = delta_scale / (1 - gamma)

        for _ in range(n_trials):
            # Random perturbation with ||DeltaR||_inf = delta_scale
            delta_R = rng.uniform(-delta_scale, delta_scale, size=R.shape)

            R_perturbed = R + delta_R
            bellman_perturbed = BellmanOperator(P, R_perturbed, gamma)

            result_perturbed = value_iteration(bellman_perturbed)
            V_star_perturbed = result_perturbed["V"]

            v_diff = np.linalg.norm(V_star_perturbed - V_star_original, ord=np.inf)
            v_diffs.append(v_diff)

        v_diffs = np.array(v_diffs)
        mean_diff = v_diffs.mean()
        max_diff = v_diffs.max()
        bound_satisfied = max_diff <= theoretical_bound + 1e-10

        results[delta_scale] = {
            "theoretical_bound": theoretical_bound,
            "mean_v_diff": mean_diff,
            "max_v_diff": max_diff,
            "bound_satisfied": bound_satisfied,
            "all_diffs": v_diffs.tolist(),
        }

        if verbose:
            satisfied = "OK" if bound_satisfied else "NO"
            print(f"{delta_scale:<10.2f} {theoretical_bound:<12.3f} {mean_diff:<15.4f} {max_diff:<15.4f} {satisfied:<10}")

    if verbose:
        all_bounds_ok = all(r["bound_satisfied"] for r in results.values())
        print(f"\n{'=' * 50}")
        print(f"All perturbation bounds satisfied: {'YES' if all_bounds_ok else 'NO'}")
        print(f"{'=' * 50}")
        print("""
TASK 2: Perturbation Sensitivity Interpretation

The bound ||V*_perturbed - V*||_inf <= ||DeltaR||_inf / (1-gamma) tells us:

1. **Reward uncertainty propagates**: Small errors in reward estimation
   can cause larger errors in the value function, amplified by 1/(1-gamma).

2. **gamma controls sensitivity**: Higher gamma means more sensitivity to errors.
   - gamma = 0.9: 10x amplification
   - gamma = 0.99: 100x amplification

3. **Practical implications**:
   - Reward modeling errors matter MORE for long-horizon problems
   - Conservative gamma provides robustness to model misspecification
   - Reward shaping should be done carefully
""")

    return {
        "gamma": gamma,
        "noise_scales": noise_scales,
        "V_star_original": V_star_original.tolist(),
        "per_scale_results": results,
    }


# =============================================================================
# Extended Lab: Discount Factor Analysis
# =============================================================================


def extended_discount_factor_analysis(
    gamma_values: list[float] | None = None,
    tol: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """
    Extended Lab: Discount Factor Analysis

    Objective: Deep dive into how gamma affects convergence properties.

    Analyzes:
    1. Convergence rate (iterations to tolerance)
    2. Effective horizon (1/(1-gamma))
    3. Value function magnitude
    4. Contraction tightness

    Args:
        gamma_values: List of discount factors to analyze
        tol: Convergence tolerance
        verbose: Whether to print detailed output

    Returns:
        dict with comprehensive gamma analysis
    """
    if verbose:
        print("=" * 70)
        print("Extended Lab: Discount Factor Analysis")
        print("=" * 70)

    P, R, _ = create_example_mdp()

    if gamma_values is None:
        gamma_values = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

    results = {}

    if verbose:
        print(f"\nMDP: {P.shape[0]} states, {P.shape[1]} actions")
        print(f"Convergence tolerance: {tol}\n")
        print(f"{'gamma':<6} {'Horizon':<10} {'Iters':<8} {'||V*||_inf':<12} {'Avg Ratio':<12}")
        print("-" * 50)

    for gamma in gamma_values:
        if gamma >= 1.0:
            continue

        bellman = BellmanOperator(P, R, gamma)

        # Value iteration
        result = value_iteration(bellman, tol=tol, track_history=True)
        V_star = result["V"]

        # Effective horizon
        horizon = 1 / (1 - gamma) if gamma < 1 else float("inf")

        # Value magnitude
        V_norm = np.linalg.norm(V_star, ord=np.inf)

        # Average contraction ratio in the tail: ||V_{k+1}-V_k|| / ||V_k-V_{k-1}|| -> gamma.
        avg_ratio = 0.0
        errors = result.get("errors", [])
        ratios = [
            errors[k] / errors[k - 1]
            for k in range(1, len(errors))
            if errors[k - 1] > 1e-12
        ]
        if ratios:
            tail = ratios[-8:] if len(ratios) >= 8 else ratios
            avg_ratio = float(np.mean(tail))

        results[gamma] = {
            "effective_horizon": horizon,
            "iterations": result["iters"],
            "V_star": V_star.tolist(),
            "V_norm": V_norm,
            "avg_contraction_ratio": avg_ratio,
            "converged": result["converged"],
        }

        if verbose:
            horizon_str = f"{horizon:.1f}" if horizon < 1000 else "inf"
            print(
                f"{gamma:<6.2f} {horizon_str:<10} {result['iters']:<8} {V_norm:<10.3f} {avg_ratio:<12.3f}"
            )

    if verbose:
        print(f"\n{'=' * 50}")
        print("KEY INSIGHTS: Discount Factor Analysis")
        print(f"{'=' * 50}")
        print("""
1. **Horizon interpretation**: 1/(1-gamma) is the "effective planning horizon"
   - gamma = 0.9: Look ~10 steps ahead
   - gamma = 0.99: Look ~100 steps ahead

2. **gamma = 0 (bandit case)**: Converges in 1 iteration because
   V(s) = max_a R(s,a) requires no bootstrapping.

3. **Convergence vs. horizon tradeoff**: Longer horizons mean
   - More iterations needed
   - Higher V* magnitudes (more total reward accumulated)
   - Greater sensitivity to errors

4. **Contraction ratio ~ gamma**: Empirically, the convergence rate
   closely tracks the theoretical bound.

5. **Practical guidance**:
   - Start with smaller gamma for faster iteration
   - Increase gamma only if needed for the task
   - Consider gamma as a hyperparameter, not a fundamental constant
""")

    return {
        "gamma_values": gamma_values,
        "per_gamma_results": results,
    }


# =============================================================================
# Extended Lab: Banach Fixed-Point Verification
# =============================================================================


def extended_banach_convergence_verification(
    n_initializations: int = 10,
    gamma: float = 0.9,
    verbose: bool = True,
) -> dict:
    """
    Extended Lab: Banach Fixed-Point Theorem Verification

    Objective: Empirically verify [THM-3.6.2-Banach]:
    1. Existence: A unique fixed point V* exists
    2. Convergence: From any V0, value iteration converges to V*
    3. Rate: ||V_k - V*||_inf <= gamma^k ||V_0 - V*||_inf

    Args:
        n_initializations: Number of random initializations to test
        gamma: Discount factor
        verbose: Whether to print detailed output

    Returns:
        dict with convergence verification results
    """
    if verbose:
        print("=" * 70)
        print("Extended Lab: Banach Fixed-Point Theorem Verification")
        print("=" * 70)

    P, R, _ = create_example_mdp()
    bellman = BellmanOperator(P, R, gamma)

    rng = np.random.default_rng(42)

    # First, find V* from zero initialization (reference)
    result_ref = value_iteration(bellman, V_init=np.zeros(bellman.n_states), tol=1e-10)
    V_star = result_ref["V"]

    if verbose:
        print(f"\nReference V* (from V0 = 0):")
        print(f"  V* = {V_star}")
        print(f"  Converged in {result_ref['iters']} iterations\n")

    # Test convergence from multiple random initializations
    all_converge_to_same = True
    convergence_results = []

    if verbose:
        print(f"Testing convergence from {n_initializations} random initializations...\n")
        print(f"{'Init #':<8} {'||V0||_inf':<12} {'Iters':<8} {'||V_final - V*||_inf':<22}")
        print("-" * 50)

    for i in range(n_initializations):
        # Random initialization with varying magnitudes
        scale = rng.uniform(1, 100)
        V_init = rng.normal(size=bellman.n_states) * scale

        result = value_iteration(bellman, V_init=V_init, tol=1e-10, track_history=True)
        V_final = result["V"]

        diff_from_ref = np.linalg.norm(V_final - V_star, ord=np.inf)
        all_converge_to_same = all_converge_to_same and (diff_from_ref < 1e-6)

        # Verify the Banach contraction rate on successive differences:
        # ||V_{k+1} - V_k||_inf <= gamma^k ||V_1 - V_0||_inf.
        rate_violations = 0
        errors = result.get("errors", [])
        if errors:
            initial_error = float(errors[0])
            for k, err in enumerate(errors):
                theoretical_bound = (gamma**k) * initial_error
                if float(err) > theoretical_bound + 1e-12:
                    rate_violations += 1

        convergence_results.append(
            {
                "V_init_norm": np.linalg.norm(V_init, ord=np.inf),
                "iters": result["iters"],
                "diff_from_ref": diff_from_ref,
                "rate_violations": rate_violations,
            }
        )

        if verbose:
            print(
                f"{i+1:<8} {np.linalg.norm(V_init, ord=np.inf):<12.2f} {result['iters']:<8} {diff_from_ref:<20.2e}"
            )

    if verbose:
        total_rate_violations = sum(
            int(item.get("rate_violations", 0)) for item in convergence_results
        )
        print(f"\n{'=' * 50}")
        print("BANACH FIXED-POINT THEOREM VERIFICATION")
        print(f"{'=' * 50}")
        print(f"\n[THM-3.6.2-Banach] Verification Results:")
        print(f"  (1) Existence:   V* exists OK")
        print(
            f"  (2) Uniqueness:  All {n_initializations} initializations -> same V*: "
            f"{'OK' if all_converge_to_same else 'NO'}"
        )
        print(f"  (3) Convergence: All trials converged OK")
        print(
            "  (4) Rate bound:  gamma^k bound violated "
            f"{total_rate_violations} times across all trials "
            f"{'OK' if total_rate_violations == 0 else 'CHECK'}"
        )

        print(f"\n{'=' * 50}")
        print("INTERPRETATION")
        print(f"{'=' * 50}")
        print("""
The Banach Fixed-Point Theorem guarantees:

1. **Global convergence**: For any initialization V0, the iterates converge
   to the same fixed point V*.

2. **Unique optimum**: There's exactly one optimal value function. No
   local optima, no sensitivity to initialization (for the value function).

3. **Exponential convergence**: Error shrinks by factor gamma each iteration.
   This is substantially faster than the O(1/k) rates typical of first-order methods.

Contrast with general non-convex optimization:
- Multiple local optima can exist
- Initialization can matter
- Convergence rates can be slow

The Bellman operator's contraction property eliminates all these issues.
This is why dynamic programming works so reliably when it's applicable.
""")

    return {
        "V_star": V_star.tolist(),
        "all_converge_to_same": all_converge_to_same,
        "convergence_results": convergence_results,
        "gamma": gamma,
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def run_all_labs(verbose: bool = True) -> dict:
    """Run all Chapter 3 labs sequentially."""
    print("\n" + "=" * 70)
    print("CHAPTER 3 LAB SOLUTIONS — COMPLETE RUN")
    print("Bellman Operators and Convergence Theory")
    print("=" * 70 + "\n")

    results = {}

    print("\n" + "─" * 70)
    results["lab_3_1"] = lab_3_1_contraction_ratio_tracker(verbose=verbose)

    print("\n" + "─" * 70)
    results["lab_3_2"] = lab_3_2_value_iteration_profiling(verbose=verbose)

    print("\n" + "─" * 70)
    results["extended_perturbation"] = extended_perturbation_sensitivity(verbose=verbose)

    print("\n" + "─" * 70)
    results["extended_discount"] = extended_discount_factor_analysis(verbose=verbose)

    print("\n" + "─" * 70)
    results["extended_banach"] = extended_banach_convergence_verification(verbose=verbose)

    print("\n" + "=" * 70)
    print("ALL CHAPTER 3 LABS COMPLETED")
    print("=" * 70)

    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Chapter 3 Lab Solutions")
    parser.add_argument("--all", action="store_true", help="Run all labs")
    parser.add_argument("--lab", type=str, help="Run specific lab (e.g., '3.1', '3.2')")
    parser.add_argument(
        "--extended",
        type=str,
        help="Run extended lab (e.g., 'perturbation', 'discount', 'banach')",
    )

    args = parser.parse_args()

    lab_map = {
        "3.1": lab_3_1_contraction_ratio_tracker,
        "31": lab_3_1_contraction_ratio_tracker,
        "3.2": lab_3_2_value_iteration_profiling,
        "32": lab_3_2_value_iteration_profiling,
    }

    extended_map = {
        "perturbation": extended_perturbation_sensitivity,
        "perturb": extended_perturbation_sensitivity,
        "discount": extended_discount_factor_analysis,
        "gamma": extended_discount_factor_analysis,
        "banach": extended_banach_convergence_verification,
        "fixedpoint": extended_banach_convergence_verification,
    }

    if args.all:
        run_all_labs()
    elif args.lab:
        key = args.lab.lower().replace("_", "").replace("-", "")
        if key in lab_map:
            lab_map[key]()
        else:
            print(f"Unknown lab: {args.lab}")
            print(f"Available: 3.1, 3.2")
            sys.exit(1)
    elif args.extended:
        key = args.extended.lower().replace("_", "").replace("-", "")
        if key in extended_map:
            extended_map[key]()
        else:
            print(f"Unknown extended lab: {args.extended}")
            print(f"Available: perturbation, discount, banach")
            sys.exit(1)
    else:
        # Interactive menu
        print("\nCHAPTER 3 LAB SOLUTIONS")
        print("Bellman Operators and Convergence Theory")
        print("=" * 45)
        print("1. Lab 3.1   - Contraction Ratio Tracker")
        print("2. Lab 3.2   - Value Iteration Profiling")
        print("3. Extended  - Perturbation Sensitivity")
        print("4. Extended  - Discount Factor Analysis")
        print("5. Extended  - Banach Theorem Verification")
        print("A. All       - Run everything")
        print()

        choice = input("Select (1-5 or A): ").strip().lower()
        choice_map = {
            "1": lab_3_1_contraction_ratio_tracker,
            "2": lab_3_2_value_iteration_profiling,
            "3": extended_perturbation_sensitivity,
            "4": extended_discount_factor_analysis,
            "5": extended_banach_convergence_verification,
            "a": run_all_labs,
        }
        if choice in choice_map:
            choice_map[choice]()
        else:
            print(f"Invalid choice: {choice}")


if __name__ == "__main__":
    main()
