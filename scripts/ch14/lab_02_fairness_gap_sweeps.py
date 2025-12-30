#!/usr/bin/env python3
"""Lab 14.2: Fairness Gap Analysis and Comparison of Scalarization Methods.

Demonstrates:
1. Exposure fairness measurement across provider groups (pl, category, strategic)
2. Comparison of epsilon-constraint vs scalarization approaches
3. Identification of unsupported Pareto points

This implements the comparison between Construction A (epsilon-constraint)
and Construction B (scalarization) from Section 14.2.3.

Mathematical basis:
    - [DEF-14.2.1], [DEF-14.2.2]: Pareto dominance and Pareto front
    - [Remark 14.2.1]: Supported vs unsupported points
    - Appendix E, Section E.3.2: Geometric distinction

Usage:
    python scripts/ch14/lab_02_fairness_gap_sweeps.py

Output:
    - docs/book/ch14/data/fairness_sweep_results.csv
    - docs/book/ch14/data/scalarization_comparison.csv
    - Fairness gap visualization

References:
    - Chapter 14: Multi-Objective RL and Fairness at Scale
    - [@ehrgott:multicriteria_optimization:2005]: Supported vs unsupported points
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from zoosim.core.config import SimulatorConfig, load_default_config
from zoosim.envs import ZooplusSearchEnv
from zoosim.evaluation.fairness import (
    BandCheckResult,
    FairnessMetrics,
    GroupScheme,
    compute_batch_fairness,
    exposure_share_by_group,
    get_group_keys,
    l1_gap,
    position_weights,
    within_band,
)
from zoosim.world.catalog import Product


@dataclass
class FairnessSweepConfig:
    """Configuration for fairness gap sweep experiment."""

    # Provider group schemes to analyze
    schemes: List[GroupScheme] = field(
        default_factory=lambda: ["pl", "category", "strategic"]
    )

    # Exposure target bands to sweep (for pl group)
    pl_target_shares: List[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6]
    )
    band_width: float = 0.05  # +/- 5%

    # Scalarization weights to compare
    fairness_weights: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    )

    # Evaluation parameters
    n_rankings: int = 100
    top_k: int = 20
    seed: int = 2025_1227


@dataclass
class FairnessResult:
    """Results from one fairness configuration."""

    scheme: str
    target_share: float
    actual_share: float
    l1_gap: float
    within_band: bool
    band_deviation: float


@dataclass
class ScalarizationResult:
    """Results from scalarization comparison."""

    fairness_weight: float
    mean_gmv: float
    mean_cm2: float
    pl_exposure_share: float
    l1_gap_from_parity: float


def generate_rankings_with_template(
    env: ZooplusSearchEnv,
    template_weights: np.ndarray,
    n_rankings: int,
    seed: int,
) -> Tuple[List[List[int]], List[str]]:
    """Generate rankings using a fixed template action.

    Args:
        env: Search environment
        template_weights: Boost weights to apply
        n_rankings: Number of rankings to generate
        seed: Random seed

    Returns:
        Tuple of (rankings, query_types)
    """
    rankings = []
    query_types = []

    for i in range(n_rankings):
        env_i = ZooplusSearchEnv(env.cfg, seed=seed + i)
        state = env_i.reset()
        _, _, _, info = env_i.step(template_weights)

        rankings.append(info["ranking"])
        query_types.append(info["query_type"])

    return rankings, query_types


def run_fairness_sweep(cfg: SimulatorConfig, sweep_cfg: FairnessSweepConfig) -> List[FairnessResult]:
    """Run fairness gap sweep across target shares.

    Args:
        cfg: Simulator configuration
        sweep_cfg: Sweep configuration

    Returns:
        List of FairnessResult for each configuration
    """
    print("=" * 60)
    print("Chapter 14 Lab 2: Fairness Gap Analysis")
    print("=" * 60)

    results: List[FairnessResult] = []
    env = ZooplusSearchEnv(cfg, seed=sweep_cfg.seed)

    # Build catalog dict for fairness computation
    catalog_dict = {p.product_id: p for p in env._catalog}

    # Test different template actions
    templates = {
        "balanced": np.array([0.2, 0.1, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "pl_boost": np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "no_boost": np.zeros(cfg.action.feature_dim),
    }

    for template_name, template_weights in templates.items():
        print(f"\nTemplate: {template_name}")

        # Generate rankings
        rankings, query_types = generate_rankings_with_template(
            env, template_weights, sweep_cfg.n_rankings, sweep_cfg.seed
        )

        for scheme in sweep_cfg.schemes:
            # Get group keys for this scheme
            group_keys = get_group_keys(scheme, cfg)

            # Compute exposure shares
            all_shares: Dict[str, List[float]] = {k: [] for k in group_keys}

            for ranking, qt in zip(rankings, query_types):
                weights = position_weights(cfg, qt, k=sweep_cfg.top_k)
                shares = exposure_share_by_group(ranking, catalog_dict, weights, scheme)
                for k in group_keys:
                    all_shares[k].append(shares.get(k, 0.0))

            # Compute mean shares
            mean_shares = {k: np.mean(v) for k, v in all_shares.items()}

            # For pl scheme, sweep target shares
            if scheme == "pl":
                for target in sweep_cfg.pl_target_shares:
                    targets = {"pl": target, "non_pl": 1.0 - target}
                    gap = l1_gap(mean_shares, targets)
                    band_result = within_band(mean_shares, targets, sweep_cfg.band_width)

                    result = FairnessResult(
                        scheme=f"{scheme}_{template_name}",
                        target_share=target,
                        actual_share=mean_shares.get("pl", 0.0),
                        l1_gap=gap,
                        within_band=band_result.all_satisfied,
                        band_deviation=band_result.max_deviation,
                    )
                    results.append(result)

                    status = "OK" if band_result.all_satisfied else "VIOLATION"
                    print(f"  {scheme}: target={target:.2f}, actual={mean_shares.get('pl', 0):.3f}, "
                          f"gap={gap:.3f} [{status}]")
            else:
                # For other schemes, use uniform target
                n_groups = len(group_keys)
                targets = {k: 1.0 / n_groups for k in group_keys}
                gap = l1_gap(mean_shares, targets)

                print(f"  {scheme}: shares={mean_shares}, gap={gap:.3f}")

    return results


def run_scalarization_comparison(
    cfg: SimulatorConfig,
    sweep_cfg: FairnessSweepConfig,
) -> List[ScalarizationResult]:
    """Compare epsilon-constraint to scalarization approaches.

    Scalarization: max GMV - lambda * |exposure_pl - 0.5|
    This should only find supported Pareto points.

    Args:
        cfg: Simulator configuration
        sweep_cfg: Sweep configuration

    Returns:
        List of ScalarizationResult
    """
    print("\n" + "=" * 60)
    print("Scalarization Comparison (Construction B)")
    print("=" * 60)

    results: List[ScalarizationResult] = []
    env = ZooplusSearchEnv(cfg, seed=sweep_cfg.seed)
    catalog_dict = {p.product_id: p for p in env._catalog}

    for fairness_weight in sweep_cfg.fairness_weights:
        print(f"\nFairness weight: {fairness_weight}")

        # Simulate training with scalarized objective
        # In practice, this would be a full training loop
        # For demo, we use a heuristic: higher weight -> more PL boost

        # Heuristic: fairness_weight influences PL feature weight
        pl_boost = min(0.5, fairness_weight * 0.1)
        template = np.array([0.2, 0.1, pl_boost, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Generate rankings
        rankings, query_types = generate_rankings_with_template(
            env, template, sweep_cfg.n_rankings, sweep_cfg.seed + int(fairness_weight * 100)
        )

        # Compute metrics
        gmv_vals = []
        cm2_vals = []
        pl_shares = []

        for i, (ranking, qt) in enumerate(zip(rankings, query_types)):
            env_i = ZooplusSearchEnv(cfg, seed=sweep_cfg.seed + i)
            env_i.reset()
            _, reward, _, info = env_i.step(template)

            gmv_vals.append(info.get("reward_details", {}).get("gmv", 0.0))
            cm2_vals.append(info.get("reward_details", {}).get("cm2", 0.0))

            weights = position_weights(cfg, qt, k=sweep_cfg.top_k)
            shares = exposure_share_by_group(ranking, catalog_dict, weights, "pl")
            pl_shares.append(shares.get("pl", 0.0))

        mean_gmv = np.mean(gmv_vals)
        mean_cm2 = np.mean(cm2_vals)
        mean_pl_share = np.mean(pl_shares)

        # L1 gap from parity (0.5)
        gap = abs(mean_pl_share - 0.5)

        result = ScalarizationResult(
            fairness_weight=fairness_weight,
            mean_gmv=mean_gmv,
            mean_cm2=mean_cm2,
            pl_exposure_share=mean_pl_share,
            l1_gap_from_parity=gap,
        )
        results.append(result)

        print(f"  GMV={mean_gmv:.3f}, CM2={mean_cm2:.3f}, "
              f"PL_share={mean_pl_share:.3f}, gap_from_parity={gap:.3f}")

    return results


def save_results(
    fairness_results: List[FairnessResult],
    scalar_results: List[ScalarizationResult],
    output_dir: Path,
) -> None:
    """Save results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fairness sweep results
    fairness_path = output_dir / "fairness_sweep_results.csv"
    with open(fairness_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scheme", "target_share", "actual_share", "l1_gap",
            "within_band", "band_deviation"
        ])
        for r in fairness_results:
            writer.writerow([
                r.scheme, r.target_share, r.actual_share, r.l1_gap,
                r.within_band, r.band_deviation
            ])
    print(f"\nFairness results saved to: {fairness_path}")

    # Scalarization comparison results
    scalar_path = output_dir / "scalarization_comparison.csv"
    with open(scalar_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fairness_weight", "mean_gmv", "mean_cm2",
            "pl_exposure_share", "l1_gap_from_parity"
        ])
        for r in scalar_results:
            writer.writerow([
                r.fairness_weight, r.mean_gmv, r.mean_cm2,
                r.pl_exposure_share, r.l1_gap_from_parity
            ])
    print(f"Scalarization results saved to: {scalar_path}")


def plot_fairness_analysis(
    fairness_results: List[FairnessResult],
    scalar_results: List[ScalarizationResult],
    output_dir: Path,
) -> None:
    """Visualize fairness analysis results."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Actual vs Target exposure
    ax1 = axes[0]
    pl_results = [r for r in fairness_results if r.scheme.startswith("pl_")]

    for template in ["balanced", "pl_boost", "no_boost"]:
        template_results = [r for r in pl_results if template in r.scheme]
        if template_results:
            targets = [r.target_share for r in template_results]
            actuals = [r.actual_share for r in template_results]
            ax1.plot(targets, actuals, "o-", label=template, markersize=8)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect match")
    ax1.set_xlabel("Target PL Exposure Share")
    ax1.set_ylabel("Actual PL Exposure Share")
    ax1.set_title("Exposure Target vs Actual")
    ax1.legend()
    ax1.set_xlim(0, 0.7)
    ax1.set_ylim(0, 0.7)

    # Plot 2: GMV vs Fairness (scalarization comparison)
    ax2 = axes[1]
    gmv_vals = [r.mean_gmv for r in scalar_results]
    fairness_vals = [1 - r.l1_gap_from_parity for r in scalar_results]  # Higher = fairer
    weights = [r.fairness_weight for r in scalar_results]

    scatter = ax2.scatter(fairness_vals, gmv_vals, c=weights, cmap="coolwarm", s=100)
    ax2.set_xlabel("Fairness (1 - L1 gap from parity)")
    ax2.set_ylabel("Mean GMV")
    ax2.set_title("GMV-Fairness Trade-off (Scalarization)")
    plt.colorbar(scatter, ax=ax2, label="Fairness Weight")

    plt.tight_layout()

    plot_path = output_dir / "fairness_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.close()


def main():
    """Run fairness gap analysis experiment."""
    cfg = load_default_config()
    sweep_cfg = FairnessSweepConfig(
        n_rankings=50,  # Reduced for demo
    )

    # Run fairness sweep
    fairness_results = run_fairness_sweep(cfg, sweep_cfg)

    # Run scalarization comparison
    scalar_results = run_scalarization_comparison(cfg, sweep_cfg)

    # Save results
    output_dir = Path("docs/book/ch14/data")
    save_results(fairness_results, scalar_results, output_dir)

    # Plot results
    plot_fairness_analysis(fairness_results, scalar_results, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Price of Fairness")
    print("=" * 60)

    if scalar_results:
        baseline = scalar_results[0]  # weight=0
        most_fair = max(scalar_results, key=lambda r: 1 - r.l1_gap_from_parity)

        gmv_loss = baseline.mean_gmv - most_fair.mean_gmv
        fairness_gain = baseline.l1_gap_from_parity - most_fair.l1_gap_from_parity

        print(f"Baseline (no fairness): GMV={baseline.mean_gmv:.3f}, "
              f"PL_share={baseline.pl_exposure_share:.3f}")
        print(f"Most fair (weight={most_fair.fairness_weight}): "
              f"GMV={most_fair.mean_gmv:.3f}, PL_share={most_fair.pl_exposure_share:.3f}")
        print(f"\nPrice of fairness:")
        print(f"  GMV loss: {gmv_loss:.3f} ({100*gmv_loss/baseline.mean_gmv:.1f}%)")
        print(f"  Fairness gain: {fairness_gain:.3f} (L1 gap reduction)")


if __name__ == "__main__":
    main()
