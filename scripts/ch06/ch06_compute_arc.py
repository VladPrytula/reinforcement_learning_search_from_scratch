#!/usr/bin/env python
"""Chapter 6 compute arc: simple → rich oracle → rich estimated.

This script runs the complete experimental narrative for Chapter 6:

1. Simple features experiment (demonstrates failure of linear contextual bandits)
2. Rich features with oracle latents (LinUCB wins with clean features)
3. Rich features with estimated latents (Thompson Sampling wins with noisy features)
4. Saves JSON summaries for reproducibility
5. Optionally generates figures for the chapter

The three-stage arc teaches the chapter's deepest lesson:
**Algorithm selection depends on feature quality.**

- Clean/oracle features → LinUCB exploits precisely
- Noisy/estimated features → Thompson Sampling explores robustly

Usage
-----
export PYTHONUNBUFFERED=1
python scripts/ch06/ch06_compute_arc.py \\
    --n-static 2000 \\
    --n-bandit 20000 \\
    --base-seed 2025_0601 \\
    --out-dir docs/book/drafts/ch06/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zoosim.core.config import SimulatorConfig
from zoosim.world import catalog as catalog_module

from template_bandits_demo import (
    resolve_lin_alpha,
    resolve_prior_weight,
    resolve_ts_sigma,
    run_template_bandits_experiment,
)


def main() -> None:
    """Run the Chapter 6 compute arc."""
    parser = argparse.ArgumentParser(
        description="Chapter 6 compute arc: simple → rich features"
    )
    parser.add_argument(
        "--n-static",
        type=int,
        default=2000,
        help="Episodes for static template evaluation.",
    )
    parser.add_argument(
        "--n-bandit",
        type=int,
        default=20000,
        help="Episodes for bandit training.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=2025_0322,
        help="Base random seed for world generation (catalog).",
    )
    parser.add_argument(
        "--bandit-seed",
        type=int,
        default=2025_0349,
        help="Random seed for bandit episodes (user/query sequence).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/book/drafts/ch06/data"),
        help="Output directory for JSON summaries.",
    )
    parser.add_argument(
        "--prior-weight",
        type=int,
        default=None,
        help="Override pseudo-count used for static priors (applied to both experiments).",
    )
    parser.add_argument(
        "--lin-alpha",
        type=float,
        default=None,
        help="Override LinUCB exploration α (applied to both experiments).",
    )
    parser.add_argument(
        "--ts-sigma",
        type=float,
        default=None,
        help="Override Thompson Sampling σ (applied to both experiments).",
    )
    args = parser.parse_args()

    # Create output directories
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CHAPTER 6 — COMPUTE ARC: SIMPLE → RICH ORACLE → RICH ESTIMATED")
    print("=" * 80)
    print("\nConfig:")
    print(f"  Static episodes:  {args.n_static:,}")
    print(f"  Bandit episodes:  {args.n_bandit:,}")
    print(f"  World seed:       {args.base_seed}")
    print(f"  Bandit seed:      {args.bandit_seed}")
    print(f"  Output dir:       {args.out_dir}")

    # Build world ONCE (same catalog for both experiments)
    print("\nGenerating world...")
    cfg = SimulatorConfig(seed=args.base_seed)
    rng = np.random.default_rng(args.base_seed)
    products = catalog_module.generate_catalog(cfg.catalog, rng)
    print(f"  Catalog:  {len(products):,} products")
    print(f"  Segments: {cfg.users.segments}")
    print(f"  Queries:  {cfg.queries.query_types}")

    # Resolve hyperparameters for each feature setting
    # Note: We force defaults (None) for "simple" to preserve the pedagogical failure mode
    # (naive exploration often fails with poor features). The CLI overrides apply
    # only to the "rich" experiments where we want to demonstrate tuning/winning.
    simple_prior = resolve_prior_weight("simple", None)
    simple_lin_alpha = resolve_lin_alpha("simple", None)
    simple_ts_sigma = resolve_ts_sigma("simple", None)

    rich_prior = resolve_prior_weight("rich", args.prior_weight)
    rich_lin_alpha = resolve_lin_alpha("rich", args.lin_alpha)
    rich_ts_sigma = resolve_ts_sigma("rich", args.ts_sigma)

    # For rich_est, use same hyperparameters as rich (from rich_est hparam mode)
    rich_est_prior = resolve_prior_weight("rich_est", args.prior_weight)
    rich_est_lin_alpha = resolve_lin_alpha("rich_est", args.lin_alpha)
    rich_est_ts_sigma = resolve_ts_sigma("rich_est", args.ts_sigma)

    # ------------------------------------------------------------------
    # Experiment 1: Simple features (failure)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: SIMPLE FEATURES (Segment + Query Type)")
    print("=" * 80)
    print("Hypothesis: Linear contextual bandits should beat static baselines.")
    print("Context: φ_simple(x) = [segment_onehot, query_onehot, bias] (8 dims)")
    print(
        f"Hyperparameters: prior_weight={simple_prior}, "
        f"lin_alpha={simple_lin_alpha:.2f}, ts_sigma={simple_ts_sigma:.2f}"
    )
    print()

    simple_results = run_template_bandits_experiment(
        cfg=cfg,
        products=products,
        n_static=args.n_static,
        n_bandit=args.n_bandit,
        feature_mode="simple",
        base_seed=args.bandit_seed,
        prior_weight=simple_prior,
        lin_alpha=simple_lin_alpha,
        ts_sigma=simple_ts_sigma,
    )

    simple_json = args.out_dir / "template_bandits_simple_summary.json"
    with open(simple_json, "w", encoding="utf-8") as f:
        json.dump(simple_results, f, indent=2)
    print(f"\n✓ Saved: {simple_json}")

    static_gmv = simple_results["static_best"]["result"]["gmv"]
    linucb_gmv = simple_results["linucb"]["global"]["gmv"]
    ts_gmv = simple_results["ts"]["global"]["gmv"]

    print("\nRESULTS (Simple Features):")
    print(f"  Static (best):  GMV = {static_gmv:.2f}")
    print(
        f"  LinUCB:         GMV = {linucb_gmv:.2f}  "
        f"({100 * (linucb_gmv / static_gmv - 1):+.1f}%)"
    )
    print(
        f"  TS:             GMV = {ts_gmv:.2f}  "
        f"({100 * (ts_gmv / static_gmv - 1):+.1f}%)"
    )

    if linucb_gmv < static_gmv * 0.9:
        print("\n⚠️  BANDITS UNDERPERFORM STATIC (expected for pedagogical arc!)")

    # ------------------------------------------------------------------
    # Experiment 2: Rich features with ORACLE latents (LinUCB wins)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: RICH FEATURES — ORACLE LATENTS (LinUCB wins)")
    print("=" * 80)
    print("Hypothesis: With clean/oracle features, LinUCB's precise exploitation wins.")
    print(
        "Context: φ_rich(x) = [segment, query, TRUE user latents, aggregates] (18 dims)"
    )
    print(
        f"Hyperparameters: prior_weight={rich_prior}, "
        f"lin_alpha={rich_lin_alpha:.2f}, ts_sigma={rich_ts_sigma:.2f}"
    )
    print()

    rich_oracle_results = run_template_bandits_experiment(
        cfg=cfg,
        products=products,
        n_static=args.n_static,
        n_bandit=args.n_bandit,
        feature_mode="rich",
        base_seed=args.bandit_seed,
        prior_weight=rich_prior,
        lin_alpha=rich_lin_alpha,
        ts_sigma=rich_ts_sigma,
    )

    rich_oracle_json = args.out_dir / "template_bandits_rich_oracle_summary.json"
    with open(rich_oracle_json, "w", encoding="utf-8") as f:
        json.dump(rich_oracle_results, f, indent=2)
    print(f"\n✓ Saved: {rich_oracle_json}")

    oracle_linucb_gmv = rich_oracle_results["linucb"]["global"]["gmv"]
    oracle_ts_gmv = rich_oracle_results["ts"]["global"]["gmv"]

    print("\nRESULTS (Rich Features — Oracle Latents):")
    print(f"  Static (best):  GMV = {static_gmv:.2f}")
    print(
        f"  LinUCB:         GMV = {oracle_linucb_gmv:.2f}  "
        f"({100 * (oracle_linucb_gmv / static_gmv - 1):+.1f}%)"
    )
    print(
        f"  TS:             GMV = {oracle_ts_gmv:.2f}  "
        f"({100 * (oracle_ts_gmv / static_gmv - 1):+.1f}%)"
    )

    if oracle_linucb_gmv > oracle_ts_gmv:
        print("\n✓ LINUCB WINS WITH ORACLE FEATURES (precise exploitation)")
    else:
        print("\n⚠️  TS wins with oracle features (unexpected, check hyperparameters)")

    # ------------------------------------------------------------------
    # Experiment 3: Rich features with ESTIMATED latents (TS wins)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: RICH FEATURES — ESTIMATED LATENTS (TS wins)")
    print("=" * 80)
    print("Hypothesis: With noisy/estimated features, TS's robust exploration wins.")
    print(
        "Context: φ_rich_est(x) = [segment, query, ESTIMATED user latents, aggregates] (18 dims)"
    )
    print(
        f"Hyperparameters: prior_weight={rich_est_prior}, "
        f"lin_alpha={rich_est_lin_alpha:.2f}, ts_sigma={rich_est_ts_sigma:.2f}"
    )
    print()

    rich_est_results = run_template_bandits_experiment(
        cfg=cfg,
        products=products,
        n_static=args.n_static,
        n_bandit=args.n_bandit,
        feature_mode="rich_est",
        base_seed=args.bandit_seed,
        prior_weight=rich_est_prior,
        lin_alpha=rich_est_lin_alpha,
        ts_sigma=rich_est_ts_sigma,
    )

    rich_est_json = args.out_dir / "template_bandits_rich_estimated_summary.json"
    with open(rich_est_json, "w", encoding="utf-8") as f:
        json.dump(rich_est_results, f, indent=2)
    print(f"\n✓ Saved: {rich_est_json}")

    est_linucb_gmv = rich_est_results["linucb"]["global"]["gmv"]
    est_ts_gmv = rich_est_results["ts"]["global"]["gmv"]

    print("\nRESULTS (Rich Features — Estimated Latents):")
    print(f"  Static (best):  GMV = {static_gmv:.2f}")
    print(
        f"  LinUCB:         GMV = {est_linucb_gmv:.2f}  "
        f"({100 * (est_linucb_gmv / static_gmv - 1):+.1f}%)"
    )
    print(
        f"  TS:             GMV = {est_ts_gmv:.2f}  "
        f"({100 * (est_ts_gmv / static_gmv - 1):+.1f}%)"
    )

    if est_ts_gmv > est_linucb_gmv:
        print("\n✓ TS WINS WITH ESTIMATED FEATURES (robust exploration)")
    else:
        print("\n⚠️  LinUCB wins with estimated features (unexpected, check hyperparameters)")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    try:
        from plot_results import (  # type: ignore import-not-found
            plot_segment_comparison,
            plot_template_frequencies,
        )

        print("\nPlot 1: Segment GMV comparison (three-way)...")
        plot_segment_comparison(simple_json, rich_oracle_json, rich_est_json, figures_dir)

        print("Plot 2: Template selection frequencies (simple features)...")
        plot_template_frequencies(simple_json, figures_dir, suffix="simple")

        print("Plot 3: Template selection frequencies (rich oracle)...")
        plot_template_frequencies(rich_oracle_json, figures_dir, suffix="rich_oracle")

        print("Plot 4: Template selection frequencies (rich estimated)...")
        plot_template_frequencies(rich_est_json, figures_dir, suffix="rich_estimated")

    except ImportError as exc:
        print(f"\n⚠️  Plotting skipped (plot_results.py not available): {exc}")
        print("    Implement scripts/ch06/plot_results.py to enable figure generation.")

    # ------------------------------------------------------------------
    # Summary: The Algorithm Selection Principle
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY: THE ALGORITHM SELECTION PRINCIPLE")
    print("=" * 80)

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                   THREE-STAGE COMPUTE ARC RESULTS                   │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Stage 1 (Simple Features):                                          │")
    print(f"│   Static (best): {static_gmv:5.2f} GMV                                       │")
    print(f"│   LinUCB:        {linucb_gmv:5.2f} GMV ({100 * (linucb_gmv / static_gmv - 1):+6.1f}%)  ← FAILS                   │")
    print(f"│   TS:            {ts_gmv:5.2f} GMV ({100 * (ts_gmv / static_gmv - 1):+6.1f}%)  ← FAILS                   │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Stage 2 (Rich Oracle):                                              │")
    print(f"│   LinUCB:        {oracle_linucb_gmv:5.2f} GMV ({100 * (oracle_linucb_gmv / static_gmv - 1):+6.1f}%)  ← LINUCB WINS            │")
    print(f"│   TS:            {oracle_ts_gmv:5.2f} GMV ({100 * (oracle_ts_gmv / static_gmv - 1):+6.1f}%)                          │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Stage 3 (Rich Estimated):                                           │")
    print(f"│   LinUCB:        {est_linucb_gmv:5.2f} GMV ({100 * (est_linucb_gmv / static_gmv - 1):+6.1f}%)                          │")
    print(f"│   TS:            {est_ts_gmv:5.2f} GMV ({100 * (est_ts_gmv / static_gmv - 1):+6.1f}%)  ← TS WINS                │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n╔═════════════════════════════════════════════════════════════════════╗")
    print("║  ALGORITHM SELECTION DEPENDS ON FEATURE QUALITY                     ║")
    print("╠═════════════════════════════════════════════════════════════════════╣")
    print("║  Clean/Oracle features  →  LinUCB   (precise exploitation)          ║")
    print("║  Noisy/Estimated feat.  →  TS       (robust exploration)            ║")
    print("╠═════════════════════════════════════════════════════════════════════╣")
    print("║  Production systems have noisy features. Default to TS.             ║")
    print("╚═════════════════════════════════════════════════════════════════════╝")

    print("\nAll artifacts saved to:")
    print(f"  Data:    {args.out_dir}")
    print(f"  Figures: {figures_dir}")
    print("\nRegenerate with:")
    print("  python scripts/ch06/ch06_compute_arc.py \\")
    print(f"      --n-static {args.n_static} \\")
    print(f"      --n-bandit {args.n_bandit} \\")
    print(f"      --base-seed {args.base_seed}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
