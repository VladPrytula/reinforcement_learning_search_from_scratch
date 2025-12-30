#!/usr/bin/env python3
"""Lab 14.1: Pareto Front Construction via Constraint Sweeps.

Demonstrates Chapter 14's epsilon-constraint method for tracing Pareto fronts:
1. Sweep constraint thresholds (CM2 floor, stability ceiling)
2. Train a CMDP policy for each threshold setting
3. Evaluate and plot the resulting trade-off curve

This implements Construction A from Section 14.2.3: the epsilon-constraint
sweep that traces the Pareto front including unsupported points.

Mathematical basis:
    - [THM-14.2.1]: epsilon-constraint yields Pareto optimal points
    - [ALG-14.5.1]: Primal-Dual CMDP Training
    - Appendix E: Comparison to vector-reward MORL

Usage:
    python scripts/ch14/lab_01_pareto_fronts.py

Output:
    - docs/book/ch14/data/pareto_sweep_results.csv
    - Pareto front visualization (matplotlib)

References:
    - Chapter 14: Multi-Objective RL and Fairness at Scale
    - [@miettinen:multi_objective_optimization:1999]: epsilon-constraint method
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Plotting imports (optional, for visualization)
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from zoosim.core.config import SimulatorConfig, load_default_config
from zoosim.envs import ZooplusSearchEnv
from zoosim.evaluation.fairness import (
    compute_batch_fairness,
    exposure_share_by_group,
    position_weights,
)
from zoosim.policies import (
    CMDPConfig,
    ConstraintSense,
    ConstraintSpec,
    PrimalDualCMDPAgent,
    create_standard_constraints,
)
from zoosim.world.catalog import Product


@dataclass
class SweepConfig:
    """Configuration for Pareto front sweep experiment."""

    # Constraint sweep ranges
    cm2_floors: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20]
    )
    stability_ceilings: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.3, 0.2, 0.1]
    )

    # Training parameters
    n_episodes_per_config: int = 100
    eval_episodes: int = 20

    # CMDP agent parameters
    obs_dim: int = 10
    action_dim: int = 10
    action_scale: float = 5.0

    # Seeds for reproducibility
    seed: int = 2025_1227


@dataclass
class SweepResult:
    """Results from one configuration in the sweep."""

    cm2_floor: float
    stability_ceiling: float
    mean_gmv: float
    mean_cm2: float
    mean_stability: float
    constraint_violation_rate: float
    final_lambda_cm2: float
    final_lambda_stability: float


def train_and_evaluate(
    cfg: SimulatorConfig,
    constraints: List[ConstraintSpec],
    sweep_cfg: SweepConfig,
    run_seed: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Train CMDP agent and evaluate on held-out episodes.

    Args:
        cfg: Simulator configuration
        constraints: Constraint specifications for this run
        sweep_cfg: Sweep experiment configuration
        run_seed: Random seed for this run

    Returns:
        Tuple of (training_metrics, eval_metrics)
    """
    # Create environment
    env = ZooplusSearchEnv(cfg, seed=run_seed)

    # Create CMDP agent
    cmdp_config = CMDPConfig(
        seed=run_seed,
        hidden_sizes=(64, 64),
        learning_rate=1e-3,
        lambda_lr=0.01,
        gamma=0.99,
    )

    agent = PrimalDualCMDPAgent(
        obs_dim=sweep_cfg.obs_dim,
        action_dim=sweep_cfg.action_dim,
        constraints=constraints,
        action_scale=sweep_cfg.action_scale,
        config=cmdp_config,
    )

    # Training loop
    training_metrics: Dict[str, List[float]] = {
        "gmv": [],
        "cm2": [],
        "stability": [],
        "violations": [],
    }

    for ep in range(sweep_cfg.n_episodes_per_config):
        state = env.reset()

        # Simple observation: use random features for demo
        # In production, extract from state dict
        obs = np.random.randn(sweep_cfg.obs_dim).astype(np.float32)

        action = agent.select_action(obs)
        _, reward, done, info = env.step(action)

        # Extract metrics for constraints
        constraint_metrics = {
            "cm2": info.get("reward_details", {}).get("cm2", 0.0),
            "delta_rank_at_k_vs_baseline": info.get("delta_rank_at_k_vs_baseline", 0.0),
        }

        agent.store_transition(reward, info, constraint_metrics)

        # Update at end of episode
        metrics = agent.update()

        # Track training progress
        training_metrics["gmv"].append(info.get("reward_details", {}).get("gmv", 0.0))
        training_metrics["cm2"].append(constraint_metrics["cm2"])
        training_metrics["stability"].append(constraint_metrics["delta_rank_at_k_vs_baseline"])
        training_metrics["violations"].append(1.0 if any(metrics.violations.values()) else 0.0)

    # Evaluation loop (no training updates)
    eval_metrics: Dict[str, List[float]] = {
        "gmv": [],
        "cm2": [],
        "stability": [],
    }

    for _ in range(sweep_cfg.eval_episodes):
        env_eval = ZooplusSearchEnv(cfg, seed=run_seed + 10000 + _)
        state = env_eval.reset()
        obs = np.random.randn(sweep_cfg.obs_dim).astype(np.float32)
        action = agent.select_action(obs)
        _, reward, done, info = env_eval.step(action)

        eval_metrics["gmv"].append(info.get("reward_details", {}).get("gmv", 0.0))
        eval_metrics["cm2"].append(info.get("reward_details", {}).get("cm2", 0.0))
        eval_metrics["stability"].append(info.get("delta_rank_at_k_vs_baseline", 0.0))

    # Aggregate results
    train_summary = {
        "mean_gmv": np.mean(training_metrics["gmv"][-20:]),
        "mean_cm2": np.mean(training_metrics["cm2"][-20:]),
        "mean_stability": np.mean(training_metrics["stability"][-20:]),
        "violation_rate": np.mean(training_metrics["violations"][-20:]),
        "final_lambdas": dict(agent.lambdas),
    }

    eval_summary = {
        "mean_gmv": np.mean(eval_metrics["gmv"]),
        "mean_cm2": np.mean(eval_metrics["cm2"]),
        "mean_stability": np.mean(eval_metrics["stability"]),
    }

    return train_summary, eval_summary


def run_pareto_sweep(sweep_cfg: SweepConfig) -> List[SweepResult]:
    """Run the full Pareto front sweep experiment.

    Args:
        sweep_cfg: Sweep configuration

    Returns:
        List of SweepResult for each configuration
    """
    cfg = load_default_config()
    results: List[SweepResult] = []

    print("=" * 60)
    print("Chapter 14 Lab 1: Pareto Front via epsilon-Constraint Sweeps")
    print("=" * 60)

    total_configs = len(sweep_cfg.cm2_floors) * len(sweep_cfg.stability_ceilings)
    config_idx = 0

    for cm2_floor in sweep_cfg.cm2_floors:
        for stability_ceiling in sweep_cfg.stability_ceilings:
            config_idx += 1
            print(f"\n[{config_idx}/{total_configs}] CM2 >= {cm2_floor:.2f}, "
                  f"Stability <= {stability_ceiling:.2f}")

            # Create constraints for this configuration
            constraints = []
            if cm2_floor > 0:
                constraints.append(
                    ConstraintSpec(
                        name="cm2_floor",
                        threshold=cm2_floor,
                        sense=ConstraintSense.GEQ,
                        metric_key="cm2",
                    )
                )
            if stability_ceiling < 1.0:
                constraints.append(
                    ConstraintSpec(
                        name="stability",
                        threshold=stability_ceiling,
                        sense=ConstraintSense.LEQ,
                        metric_key="delta_rank_at_k_vs_baseline",
                    )
                )

            # Train and evaluate
            run_seed = sweep_cfg.seed + config_idx * 1000
            train_summary, eval_summary = train_and_evaluate(
                cfg, constraints, sweep_cfg, run_seed
            )

            # Store result
            result = SweepResult(
                cm2_floor=cm2_floor,
                stability_ceiling=stability_ceiling,
                mean_gmv=eval_summary["mean_gmv"],
                mean_cm2=eval_summary["mean_cm2"],
                mean_stability=eval_summary["mean_stability"],
                constraint_violation_rate=train_summary["violation_rate"],
                final_lambda_cm2=train_summary["final_lambdas"].get("cm2_floor", 0.0),
                final_lambda_stability=train_summary["final_lambdas"].get("stability", 0.0),
            )
            results.append(result)

            print(f"  GMV: {result.mean_gmv:.3f}, CM2: {result.mean_cm2:.3f}, "
                  f"Stability: {result.mean_stability:.3f}, "
                  f"Violations: {result.constraint_violation_rate:.1%}")

    return results


def save_results(results: List[SweepResult], output_path: Path) -> None:
    """Save sweep results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cm2_floor", "stability_ceiling", "mean_gmv", "mean_cm2",
            "mean_stability", "violation_rate", "lambda_cm2", "lambda_stability"
        ])
        for r in results:
            writer.writerow([
                r.cm2_floor, r.stability_ceiling, r.mean_gmv, r.mean_cm2,
                r.mean_stability, r.constraint_violation_rate,
                r.final_lambda_cm2, r.final_lambda_stability
            ])

    print(f"\nResults saved to: {output_path}")


def plot_pareto_front(results: List[SweepResult], output_dir: Path) -> None:
    """Plot the Pareto front from sweep results."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: GMV vs CM2
    ax1 = axes[0]
    gmv_vals = [r.mean_gmv for r in results]
    cm2_vals = [r.mean_cm2 for r in results]
    colors = [r.cm2_floor for r in results]

    scatter1 = ax1.scatter(cm2_vals, gmv_vals, c=colors, cmap="viridis", s=100, alpha=0.7)
    ax1.set_xlabel("Mean CM2")
    ax1.set_ylabel("Mean GMV")
    ax1.set_title("Pareto Front: GMV vs CM2")
    plt.colorbar(scatter1, ax=ax1, label="CM2 Floor Constraint")

    # Plot 2: GMV vs Stability
    ax2 = axes[1]
    stability_vals = [r.mean_stability for r in results]
    colors2 = [r.stability_ceiling for r in results]

    scatter2 = ax2.scatter(stability_vals, gmv_vals, c=colors2, cmap="plasma", s=100, alpha=0.7)
    ax2.set_xlabel("Mean Delta-Rank (Stability)")
    ax2.set_ylabel("Mean GMV")
    ax2.set_title("Pareto Front: GMV vs Stability")
    plt.colorbar(scatter2, ax=ax2, label="Stability Ceiling Constraint")

    plt.tight_layout()

    plot_path = output_dir / "pareto_front.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.close()


def main():
    """Run the Pareto front sweep experiment."""
    # Configuration
    sweep_cfg = SweepConfig(
        n_episodes_per_config=50,  # Reduced for demo; increase for real experiments
        eval_episodes=10,
    )

    # Run sweep
    results = run_pareto_sweep(sweep_cfg)

    # Save results
    output_dir = Path("docs/book/ch14/data")
    save_results(results, output_dir / "pareto_sweep_results.csv")

    # Plot if matplotlib available
    plot_pareto_front(results, output_dir)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total configurations tested: {len(results)}")

    # Find Pareto-optimal points (non-dominated)
    pareto_points = []
    for r in results:
        dominated = False
        for r2 in results:
            if (r2.mean_gmv >= r.mean_gmv and r2.mean_cm2 >= r.mean_cm2 and
                r2.mean_stability <= r.mean_stability and
                (r2.mean_gmv > r.mean_gmv or r2.mean_cm2 > r.mean_cm2 or
                 r2.mean_stability < r.mean_stability)):
                dominated = True
                break
        if not dominated:
            pareto_points.append(r)

    print(f"Pareto-optimal configurations: {len(pareto_points)}")
    print("\nPareto-optimal points:")
    for p in pareto_points:
        print(f"  CM2>={p.cm2_floor:.2f}, Stab<={p.stability_ceiling:.2f}: "
              f"GMV={p.mean_gmv:.3f}, CM2={p.mean_cm2:.3f}, Stab={p.mean_stability:.3f}")


if __name__ == "__main__":
    main()
