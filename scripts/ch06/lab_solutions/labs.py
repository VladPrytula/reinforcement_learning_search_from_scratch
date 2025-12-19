"""
Chapter 6 Lab Solutions — Experimental Labs

Author: Vlad Prytula

This module implements the experimental labs from Chapter 6, using the
production zoosim simulator. Labs demonstrate the simple→rich feature
arc that is the pedagogical core of the chapter.

The labs wrap the scripts in scripts/ch06/ to ensure consistency
between lab exercises and the chapter narrative.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add scripts/ch06 directory to path for imports
_CH06_DIR = Path(__file__).parent.parent
if str(_CH06_DIR) not in sys.path:
    sys.path.insert(0, str(_CH06_DIR))


def _ensure_zoosim_available() -> bool:
    """Check if zoosim is available for real simulator labs."""
    try:
        from zoosim.core.config import SimulatorConfig
        return True
    except ImportError:
        return False


# =============================================================================
# Lab 6.1: Simple-Feature Baseline (Real Simulator)
# =============================================================================


def lab_6_1_simple_feature_baseline(
    n_static: int = 2000,
    n_bandit: int = 20000,
    seed: int = 20250319,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Lab 6.1: Simple-Feature Baseline using the real zoosim simulator.

    Demonstrates that contextual bandits with simple features (segment + query)
    perform comparably to static baselines—neither catastrophic failure nor
    dramatic improvement.

    This establishes the baseline for comparison with rich features in Lab 6.2.

    Args:
        n_static: Episodes for static template evaluation
        n_bandit: Episodes for bandit training
        seed: Random seed for reproducibility
        verbose: Print progress and results

    Returns:
        Dict with experiment results matching the optimization script format
    """
    if verbose:
        print("=" * 70)
        print("Lab 6.1: Simple-Feature Baseline (Real Simulator)")
        print("=" * 70)

    if not _ensure_zoosim_available():
        if verbose:
            print("\n⚠️  zoosim not available. Running toy simulation fallback.")
        return _lab_6_1_toy_fallback(n_static, n_bandit, seed, verbose)

    from template_bandits_demo import run_template_bandits_experiment
    from zoosim.core.config import SimulatorConfig
    from zoosim.world import catalog as catalog_module

    if verbose:
        print(f"\nConfiguration:")
        print(f"  Static baseline episodes: {n_static:,}")
        print(f"  Bandit training episodes: {n_bandit:,}")
        print(f"  Feature mode: simple (8-dim: segment + query + bias)")
        print(f"  Seed: {seed}")

    # Build world
    cfg = SimulatorConfig(seed=seed)
    rng = np.random.default_rng(seed)
    products = catalog_module.generate_catalog(cfg.catalog, rng)

    if verbose:
        print(f"\nGenerated catalog with {len(products):,} products")
        print(f"User segments: {cfg.users.segments}")
        print(f"Query types: {cfg.queries.query_types}")

    # Run experiment with simple features
    results = run_template_bandits_experiment(
        cfg=cfg,
        products=products,
        n_static=n_static,
        n_bandit=n_bandit,
        feature_mode="simple",
        base_seed=seed,
        prior_weight=0,  # No warm-start for simple features
        lin_alpha=1.0,
        ts_sigma=1.0,
    )

    # Extract key metrics
    static_gmv = results["static_best"]["result"]["gmv"]
    lin_gmv = results["linucb"]["global"]["gmv"]
    ts_gmv = results["ts"]["global"]["gmv"]

    lin_pct = 100 * (lin_gmv / static_gmv - 1)
    ts_pct = 100 * (ts_gmv / static_gmv - 1)

    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS (Simple Features)")
        print(f"{'='*70}")
        print(f"\nBest static template: {results['static_best']['name']}")
        print(f"  GMV = {static_gmv:.2f}")
        print(f"\nLinUCB:")
        print(f"  GMV = {lin_gmv:.2f} ({lin_pct:+.1f}% vs static)")
        print(f"\nThompson Sampling:")
        print(f"  GMV = {ts_gmv:.2f} ({ts_pct:+.1f}% vs static)")

        print(f"\nTemplate selection frequencies:")
        templates = results["templates"]
        lin_freqs = results["linucb"]["diagnostics"]["template_freqs"]
        ts_freqs = results["ts"]["diagnostics"]["template_freqs"]
        print(f"  {'Template':<15} | {'LinUCB':>8} | {'TS':>8}")
        print(f"  {'-'*15}-|-{'-'*8}-|-{'-'*8}")
        for t, lf, tf in zip(templates, lin_freqs, ts_freqs):
            print(f"  {t['name']:<15} | {100*lf:>7.1f}% | {100*tf:>7.1f}%")

        print(f"\nConclusion:")
        print(f"  With simple features, bandits perform {'better' if lin_pct > 0 else 'worse'} than static.")
        print(f"  The gap is modest ({lin_pct:+.1f}% / {ts_pct:+.1f}%), suggesting simple features")
        print(f"  don't capture enough context variance for dramatic improvement.")

    return results


def _lab_6_1_toy_fallback(
    n_static: int,
    n_bandit: int,
    seed: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Toy simulation fallback when zoosim is not available."""
    rng = np.random.default_rng(seed)

    # Simplified simulation
    n_templates = 8
    template_names = ["Neutral", "High Margin", "CM2 Boost", "Popular",
                      "Premium", "Budget", "Discount", "Strategic"]

    # Flat reward structure (simple features can't differentiate)
    base_rewards = np.array([4.5, 5.0, 5.5, 5.2, 5.8, 4.8, 5.3, 4.6])

    # Static baseline
    static_gmv = {t: base_rewards[t] + rng.standard_normal() * 0.1
                  for t in range(n_templates)}
    best_static_id = max(static_gmv, key=lambda t: static_gmv[t])
    best_static_gmv = static_gmv[best_static_id]

    # Bandits achieve similar performance (can't exploit structure)
    lin_gmv = best_static_gmv * (1 + rng.uniform(-0.05, 0.05))
    ts_gmv = best_static_gmv * (1 + rng.uniform(-0.05, 0.05))

    if verbose:
        lin_pct = 100 * (lin_gmv / best_static_gmv - 1)
        ts_pct = 100 * (ts_gmv / best_static_gmv - 1)
        print(f"\n[TOY FALLBACK] Best static GMV: {best_static_gmv:.2f}")
        print(f"[TOY FALLBACK] LinUCB GMV: {lin_gmv:.2f} ({lin_pct:+.1f}%)")
        print(f"[TOY FALLBACK] TS GMV: {ts_gmv:.2f} ({ts_pct:+.1f}%)")

    return {
        "static_best": {"name": template_names[best_static_id], "result": {"gmv": best_static_gmv}},
        "linucb": {"global": {"gmv": lin_gmv}},
        "ts": {"global": {"gmv": ts_gmv}},
        "_fallback": True,
    }


# =============================================================================
# Lab 6.2: Rich-Feature Improvement (Real Simulator)
# =============================================================================


def lab_6_2_rich_feature_improvement(
    n_static: int = 2000,
    n_bandit: int = 20000,
    seed: int = 20250319,
    regularization: str = "blend",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Lab 6.2: Rich-Feature Improvement using the real zoosim simulator.

    Demonstrates that rich features (segment + query + user latents + aggregates)
    enable significant improvement over static baselines.

    Args:
        n_static: Episodes for static template evaluation
        n_bandit: Episodes for bandit training
        seed: Random seed for reproducibility
        regularization: "none", "blend", or "quantized" (default "blend")
        verbose: Print progress and results

    Returns:
        Dict with experiment results
    """
    if verbose:
        print("=" * 70)
        print("Lab 6.2: Rich-Feature Improvement (Real Simulator)")
        print("=" * 70)

    if not _ensure_zoosim_available():
        if verbose:
            print("\n⚠️  zoosim not available. Running toy simulation fallback.")
        return _lab_6_2_toy_fallback(n_static, n_bandit, seed, verbose)

    from template_bandits_demo import (
        RichRegularizationConfig,
        run_template_bandits_experiment,
    )
    from zoosim.core.config import SimulatorConfig
    from zoosim.world import catalog as catalog_module

    if verbose:
        print(f"\nConfiguration:")
        print(f"  Static baseline episodes: {n_static:,}")
        print(f"  Bandit training episodes: {n_bandit:,}")
        print(f"  Feature mode: rich (18-dim: segment + query + latents + aggregates + bias)")
        print(f"  Regularization: {regularization}")
        print(f"  Seed: {seed}")

    # Build world (same as Lab 6.1 for comparison)
    cfg = SimulatorConfig(seed=seed)
    rng = np.random.default_rng(seed)
    products = catalog_module.generate_catalog(cfg.catalog, rng)

    if verbose:
        print(f"\nGenerated catalog with {len(products):,} products")

    # Configure rich feature regularization
    if regularization == "none":
        rich_reg = None
    else:
        rich_reg = RichRegularizationConfig(
            mode=regularization,
            blend_weight=0.4,
            shrink=0.9,
            quant_step=0.25,
            clip_min=-3.5,
            clip_max=3.5,
        )

    # Hyperparameters tuned for rich features
    prior_weight = 6 if regularization == "blend" else 12
    lin_alpha = 1.0 if regularization != "quantized" else 0.85
    ts_sigma = 1.0 if regularization != "quantized" else 0.7

    if verbose:
        print(f"\nHyperparameters:")
        print(f"  Prior weight: {prior_weight}")
        print(f"  LinUCB α: {lin_alpha}")
        print(f"  TS σ: {ts_sigma}")

    # Run experiment with rich features
    results = run_template_bandits_experiment(
        cfg=cfg,
        products=products,
        n_static=n_static,
        n_bandit=n_bandit,
        feature_mode="rich",
        base_seed=seed,
        prior_weight=prior_weight,
        lin_alpha=lin_alpha,
        ts_sigma=ts_sigma,
        rich_regularization=rich_reg,
    )

    # Extract key metrics
    static_gmv = results["static_best"]["result"]["gmv"]
    lin_gmv = results["linucb"]["global"]["gmv"]
    ts_gmv = results["ts"]["global"]["gmv"]

    lin_pct = 100 * (lin_gmv / static_gmv - 1)
    ts_pct = 100 * (ts_gmv / static_gmv - 1)

    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS (Rich Features)")
        print(f"{'='*70}")
        print(f"\nBest static template: {results['static_best']['name']}")
        print(f"  GMV = {static_gmv:.2f}")
        print(f"\nLinUCB:")
        print(f"  GMV = {lin_gmv:.2f} ({lin_pct:+.1f}% vs static)")
        print(f"\nThompson Sampling:")
        print(f"  GMV = {ts_gmv:.2f} ({ts_pct:+.1f}% vs static)")

        # Per-segment breakdown
        print(f"\nPer-segment GMV breakdown:")
        static_segs = results["static_best"]["segments"]
        lin_segs = results["linucb"]["segments"]
        ts_segs = results["ts"]["segments"]

        print(f"  {'Segment':<15} | {'Static':>8} | {'LinUCB':>8} | {'TS':>8} | {'Lin Δ%':>8} | {'TS Δ%':>8}")
        print(f"  {'-'*15}-|-{'-'*8}-|-{'-'*8}-|-{'-'*8}-|-{'-'*8}-|-{'-'*8}")
        for seg in static_segs:
            s_gmv = static_segs[seg]["gmv"]
            l_gmv = lin_segs.get(seg, {}).get("gmv", s_gmv)
            t_gmv = ts_segs.get(seg, {}).get("gmv", s_gmv)
            l_pct = 100 * (l_gmv / s_gmv - 1) if s_gmv > 0 else 0
            t_pct = 100 * (t_gmv / s_gmv - 1) if s_gmv > 0 else 0
            print(f"  {seg:<15} | {s_gmv:>8.2f} | {l_gmv:>8.2f} | {t_gmv:>8.2f} | {l_pct:>+7.1f}% | {t_pct:>+7.1f}%")

        best_pct = max(lin_pct, ts_pct)
        best_algo = "LinUCB" if lin_pct > ts_pct else "Thompson Sampling"
        print(f"\nConclusion:")
        print(f"  Rich features enable {best_algo} to achieve {best_pct:+.1f}% GMV improvement!")
        print(f"  This demonstrates the value of capturing user preference signals")
        print(f"  and product aggregates in the context features.")

    return results


def _lab_6_2_toy_fallback(
    n_static: int,
    n_bandit: int,
    seed: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Toy simulation fallback when zoosim is not available."""
    rng = np.random.default_rng(seed)

    template_names = ["Neutral", "High Margin", "CM2 Boost", "Popular",
                      "Premium", "Budget", "Discount", "Strategic"]

    # With rich features, bandits can exploit user preferences
    base_rewards = np.array([4.5, 5.0, 5.5, 5.2, 5.8, 4.8, 5.3, 4.6])
    best_static_gmv = np.max(base_rewards) + rng.standard_normal() * 0.1

    # Rich features enable significant improvement
    lin_gmv = best_static_gmv * (1 + rng.uniform(0.20, 0.35))
    ts_gmv = best_static_gmv * (1 + rng.uniform(0.25, 0.35))

    if verbose:
        lin_pct = 100 * (lin_gmv / best_static_gmv - 1)
        ts_pct = 100 * (ts_gmv / best_static_gmv - 1)
        print(f"\n[TOY FALLBACK] Best static GMV: {best_static_gmv:.2f}")
        print(f"[TOY FALLBACK] LinUCB GMV: {lin_gmv:.2f} ({lin_pct:+.1f}%)")
        print(f"[TOY FALLBACK] TS GMV: {ts_gmv:.2f} ({ts_pct:+.1f}%)")

    return {
        "static_best": {"name": "CM2 Boost", "result": {"gmv": best_static_gmv}},
        "linucb": {"global": {"gmv": lin_gmv}},
        "ts": {"global": {"gmv": ts_gmv}},
        "_fallback": True,
    }


# =============================================================================
# Lab 6.3: Hyperparameter Sensitivity
# =============================================================================


def lab_6_3_hyperparameter_sensitivity(
    n_episodes: int = 10000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Lab 6.3: Hyperparameter sensitivity analysis.

    Explores the impact of regularization (λ) and exploration (α) on
    LinUCB performance with rich features.

    This lab uses a simplified simulation to enable fast grid search.
    For production results, use scripts/ch06/run_bandit_matrix.py.

    Args:
        n_episodes: Episodes per configuration
        seed: Random seed
        verbose: Print progress and results

    Returns:
        Dict with grid search results
    """
    if verbose:
        print("=" * 70)
        print("Lab 6.3: Hyperparameter Sensitivity Analysis")
        print("=" * 70)

    lambdas = [0.1, 1.0, 10.0]
    alphas = [0.5, 1.0, 2.0]

    if verbose:
        print(f"\nGrid: λ ∈ {lambdas} × α ∈ {alphas}")
        print(f"Episodes per config: {n_episodes:,}")

    rng = np.random.default_rng(seed)

    # Simplified linear bandit simulation
    d = 8
    n_arms = 8
    true_theta = [rng.standard_normal(d) * 0.5 for _ in range(n_arms)]

    results = {}

    for lam in lambdas:
        for alpha in alphas:
            # Run simplified LinUCB
            A = [lam * np.eye(d) for _ in range(n_arms)]
            b = [np.zeros(d) for _ in range(n_arms)]
            rewards = []

            for _ in range(n_episodes):
                x = rng.standard_normal(d)
                x = x / np.linalg.norm(x)

                # UCB action selection
                ucb_values = []
                for a in range(n_arms):
                    theta_hat = np.linalg.solve(A[a], b[a])
                    A_inv = np.linalg.inv(A[a])
                    bonus = alpha * np.sqrt(x @ A_inv @ x)
                    ucb_values.append(theta_hat @ x + bonus)
                action = int(np.argmax(ucb_values))

                # Observe reward
                r = true_theta[action] @ x + rng.standard_normal() * 0.5
                rewards.append(r)

                # Update
                A[action] += np.outer(x, x)
                b[action] += r * x

            avg_reward = np.mean(rewards[-n_episodes//4:])  # Last quarter
            results[(lam, alpha)] = avg_reward

    if verbose:
        print(f"\nResults (average reward, last {n_episodes//4:,} episodes):")
        print(f"\n{'':>8}", end="")
        for alpha in alphas:
            print(f" | α={alpha:<4}", end="")
        print()
        print("-" * (8 + 9 * len(alphas)))
        for lam in lambdas:
            print(f"λ={lam:<5}", end="")
            for alpha in alphas:
                print(f" | {results[(lam, alpha)]:>5.2f}", end="")
            print()

        # Find best
        best_config = max(results, key=lambda k: results[k])
        print(f"\nBest: λ={best_config[0]}, α={best_config[1]} → {results[best_config]:.2f}")

        print(f"\nInsights:")
        print(f"  - Higher λ provides stronger regularization (prevents overfitting)")
        print(f"  - Higher α increases exploration (helps with uncertain arms)")
        print(f"  - Optimal tradeoff depends on problem structure and horizon")

    return {
        "grid": {"lambdas": lambdas, "alphas": alphas},
        "results": {f"({lam},{alpha})": v for (lam, alpha), v in results.items()},
        "best": {"lambda": best_config[0], "alpha": best_config[1], "reward": results[best_config]},
    }


# =============================================================================
# Lab 6.4: Exploration Dynamics Visualization
# =============================================================================


def lab_6_4_exploration_dynamics(
    n_episodes: int = 5000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Lab 6.4: Visualize exploration dynamics over time.

    Shows how Thompson Sampling's uncertainty decreases and template
    selection converges as learning progresses.

    Args:
        n_episodes: Total episodes
        seed: Random seed
        verbose: Print progress and results

    Returns:
        Dict with trajectory data for plotting
    """
    if verbose:
        print("=" * 70)
        print("Lab 6.4: Exploration Dynamics Visualization")
        print("=" * 70)

    rng = np.random.default_rng(seed)

    # Simplified Thompson Sampling
    n_arms = 4
    d = 4
    true_theta = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    # Track metrics over time
    arm_selection_history = []
    uncertainty_history = []
    reward_history = []

    # Initialize
    Sigma_inv = [np.eye(d) for _ in range(n_arms)]
    b = [np.zeros(d) for _ in range(n_arms)]
    theta_hat = [np.zeros(d) for _ in range(n_arms)]

    checkpoints = [100, 500, 1000, 2000, 5000]

    for ep in range(1, n_episodes + 1):
        # Context (one-hot for simplicity)
        context_id = rng.integers(d)
        x = np.zeros(d)
        x[context_id] = 1.0

        # Thompson Sampling
        samples = []
        for a in range(n_arms):
            Sigma = np.linalg.inv(Sigma_inv[a])
            # Ensure positive definiteness
            Sigma = (Sigma + Sigma.T) / 2
            eigvals = np.linalg.eigvalsh(Sigma)
            if np.min(eigvals) < 1e-10:
                Sigma += (1e-10 - np.min(eigvals)) * np.eye(d)
            try:
                L = np.linalg.cholesky(Sigma)
                theta_sample = theta_hat[a] + L @ rng.standard_normal(d)
            except np.linalg.LinAlgError:
                theta_sample = theta_hat[a]
            samples.append(theta_sample @ x)
        action = int(np.argmax(samples))

        # Reward
        r = true_theta[action] @ x + rng.standard_normal() * 0.1
        reward_history.append(r)
        arm_selection_history.append(action)

        # Update
        Sigma_inv[action] += np.outer(x, x)
        b[action] += r * x
        theta_hat[action] = np.linalg.solve(Sigma_inv[action], b[action])

        # Track uncertainty (trace of covariance)
        total_uncertainty = sum(
            np.trace(np.linalg.inv(Sigma_inv[a])) for a in range(n_arms)
        )
        uncertainty_history.append(total_uncertainty)

        if verbose and ep in checkpoints:
            recent_rewards = reward_history[-min(500, ep):]
            selection_counts = np.bincount(
                arm_selection_history[-min(500, ep):], minlength=n_arms
            )
            print(f"\nEpisode {ep:,}:")
            print(f"  Avg reward (last 500): {np.mean(recent_rewards):.3f}")
            print(f"  Total uncertainty: {total_uncertainty:.3f}")
            print(f"  Selection freq: {selection_counts / selection_counts.sum()}")

    if verbose:
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"\nUncertainty reduction: {uncertainty_history[0]:.2f} → {uncertainty_history[-1]:.2f}")
        print(f"Reduction ratio: {uncertainty_history[0] / uncertainty_history[-1]:.1f}x")

        final_selections = np.bincount(arm_selection_history[-1000:], minlength=n_arms)
        print(f"\nFinal selection distribution (last 1000):")
        for a in range(n_arms):
            print(f"  Arm {a}: {100 * final_selections[a] / 1000:.1f}%")

    return {
        "n_episodes": n_episodes,
        "uncertainty_trajectory": uncertainty_history[::100],  # Subsample
        "final_selection_freq": (np.bincount(arm_selection_history[-1000:], minlength=n_arms) / 1000).tolist(),
        "avg_reward_final": float(np.mean(reward_history[-1000:])),
    }


# =============================================================================
# Lab 6.5: Multi-Seed Robustness
# =============================================================================


def lab_6_5_multi_seed_robustness(
    n_seeds: int = 5,
    n_episodes: int = 5000,
    base_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Lab 6.5: Assess robustness across random seeds.

    Runs the same experiment with multiple seeds to measure variance
    in final performance.

    Args:
        n_seeds: Number of seeds to test
        n_episodes: Episodes per run
        base_seed: Starting seed
        verbose: Print progress and results

    Returns:
        Dict with per-seed results and statistics
    """
    if verbose:
        print("=" * 70)
        print("Lab 6.5: Multi-Seed Robustness Analysis")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Seeds: {n_seeds}")
        print(f"  Episodes per seed: {n_episodes:,}")

    results_per_seed = []

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx * 1000

        rng = np.random.default_rng(seed)

        # Simplified experiment
        n_arms = 4
        d = 4
        true_theta = rng.standard_normal((n_arms, d)) * 0.5

        # LinUCB
        A = [np.eye(d) for _ in range(n_arms)]
        b = [np.zeros(d) for _ in range(n_arms)]
        rewards = []

        for _ in range(n_episodes):
            x = rng.standard_normal(d)
            x = x / np.linalg.norm(x)

            ucb_values = []
            for a in range(n_arms):
                theta_hat = np.linalg.solve(A[a], b[a])
                A_inv = np.linalg.inv(A[a])
                bonus = np.sqrt(x @ A_inv @ x)
                ucb_values.append(theta_hat @ x + bonus)
            action = int(np.argmax(ucb_values))

            r = true_theta[action] @ x + rng.standard_normal() * 0.3
            rewards.append(r)

            A[action] += np.outer(x, x)
            b[action] += r * x

        final_reward = np.mean(rewards[-n_episodes//4:])
        results_per_seed.append(final_reward)

        if verbose:
            print(f"  Seed {seed}: {final_reward:.3f}")

    mean_reward = np.mean(results_per_seed)
    std_reward = np.std(results_per_seed)
    cv = 100 * std_reward / mean_reward if mean_reward != 0 else 0

    if verbose:
        print(f"\n{'='*70}")
        print("Statistics")
        print(f"{'='*70}")
        print(f"  Mean: {mean_reward:.3f}")
        print(f"  Std:  {std_reward:.3f}")
        print(f"  CV:   {cv:.1f}%")
        print(f"  Range: [{min(results_per_seed):.3f}, {max(results_per_seed):.3f}]")

        print(f"\nConclusion:")
        if cv < 5:
            print(f"  Low variance (CV={cv:.1f}%) indicates robust algorithm behavior.")
        elif cv < 15:
            print(f"  Moderate variance (CV={cv:.1f}%) is typical for bandit algorithms.")
        else:
            print(f"  High variance (CV={cv:.1f}%) suggests sensitivity to initialization.")

    return {
        "n_seeds": n_seeds,
        "results": results_per_seed,
        "mean": float(mean_reward),
        "std": float(std_reward),
        "cv_percent": float(cv),
    }
