#!/usr/bin/env python
"""GPU-accelerated Chapter 6 compute arc (simple → rich features)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

from template_bandits_gpu import (
    resolve_lin_alpha,
    resolve_prior_weight,
    resolve_ts_sigma,
    run_template_bandits_experiment_gpu,
)
from zoosim.core.config import SimulatorConfig
from zoosim.world import catalog as catalog_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chapter 6 compute arc (GPU): simple → rich feature comparison."
    )
    parser.add_argument("--n-static", type=int, default=2_000)
    parser.add_argument("--n-bandit", type=int, default=20_000)
    parser.add_argument("--base-seed", type=int, default=2025_0601)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/book/ch06/data"),
    )
    parser.add_argument("--prior-weight", type=int, default=None)
    parser.add_argument("--lin-alpha", type=float, default=None)
    parser.add_argument("--ts-sigma", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device identifier (cuda/cpu/auto).",
    )
    parser.add_argument(
        "--show-volume",
        action="store_true",
        help="Include orders column in summary tables (matching Chapter 6 text).",
    )
    return parser.parse_args()


def _resolve(value: int | None, fallback: int) -> int:
    if value is None:
        return fallback
    if value <= 0:
        raise ValueError("Episode counts must be positive.")
    return value


def _pct(delta: float, base: float) -> float:
    if base == 0.0:
        return 0.0
    return 100.0 * delta / base


def _get_segment_result(
    segments: Dict[str, Dict[str, float]],
    name: str,
    fallback: Dict[str, float],
) -> Dict[str, float]:
    return segments.get(name, fallback)


def _print_static_table(results: Dict, show_volume: bool) -> None:
    print("\nStatic templates (per-episode averages):")
    if show_volume:
        print(
            f"{'ID':>2s}  {'Template':15s}  {'Reward':>10s}  "
            f"{'GMV':>10s}  {'CM2':>10s}  {'Orders':>10s}"
        )
    else:
        print(
            f"{'ID':>2s}  {'Template':15s}  {'Reward':>10s}  "
            f"{'GMV':>10s}  {'CM2':>10s}"
        )
    static_results = results["static_results"]
    for template in results["templates"]:
        tid = template["id"]
        name = template["name"]
        metrics = static_results[str(tid)] if str(tid) in static_results else static_results[tid]
        if show_volume:
            print(
                f"{tid:2d}  {name:15s}  {metrics['reward']:10.2f}  "
                f"{metrics['gmv']:10.2f}  {metrics['cm2']:10.2f}  "
                f"{metrics['orders']:10.2f}"
            )
        else:
            print(
                f"{tid:2d}  {name:15s}  {metrics['reward']:10.2f}  "
                f"{metrics['gmv']:10.2f}  {metrics['cm2']:10.2f}"
            )


def _print_policy_summary(results: Dict, show_volume: bool) -> None:
    best = results["static_best"]["result"]
    lin = results["linucb"]["global"]
    ts = results["ts"]["global"]
    best_name = results["static_best"]["name"]
    rows = [
        (f"Static-{best_name}", best),
        ("LinUCB", lin),
        ("ThompsonSampling", ts),
    ]
    print("\nSummary (per-episode averages):")
    header = (
        f"{'Policy':18s}  {'Reward':>10s}  {'GMV':>10s}  {'CM2':>10s}  "
        + ("{'Orders':>10s}  " if show_volume else "")
        + "{'ΔGMV vs static':>14s}"
    )
    # Manual header due to f-string literal braces
    if show_volume:
        print(
            f"{'Policy':18s}  {'Reward':>10s}  {'GMV':>10s}  {'CM2':>10s}  "
            f"{'Orders':>10s}  {'ΔGMV vs static':>14s}"
        )
    else:
        print(
            f"{'Policy':18s}  {'Reward':>10s}  {'GMV':>10s}  "
            f"{'CM2':>10s}  {'ΔGMV vs static':>14s}"
        )
    for label, metrics in rows:
        delta = _pct(metrics["gmv"] - best["gmv"], best["gmv"])
        if show_volume:
            print(
                f"{label:18s}  {metrics['reward']:10.2f}  {metrics['gmv']:10.2f}  "
                f"{metrics['cm2']:10.2f}  {metrics['orders']:10.2f}  "
                f"{delta:+13.2f}%"
            )
        else:
            print(
                f"{label:18s}  {metrics['reward']:10.2f}  {metrics['gmv']:10.2f}  "
                f"{metrics['cm2']:10.2f}  {delta:+13.2f}%"
            )


def _print_segment_summary(
    *,
    results: Dict,
    segment_order: List[str],
    show_volume: bool,
) -> None:
    static_segments = results["static_best"]["segments"]
    lin_segments = results["linucb"]["segments"]
    ts_segments = results["ts"]["segments"]
    static_global = results["static_best"]["result"]

    if show_volume:
        print("\nPer-segment GMV & Orders (static best vs bandits):")
        print(
            f"{'Segment':15s}  {'Static GMV':>10s}  {'LinUCB GMV':>10s}  "
            f"{'TS GMV':>10s}  {'Static Orders':>14s}  {'LinUCB Orders':>14s}  "
            f"{'TS Orders':>11s}  {'Lin GMV Δ%':>12s}  {'TS GMV Δ%':>11s}  "
            f"{'Lin Orders Δ%':>15s}  {'TS Orders Δ%':>14s}"
        )
    else:
        print("\nPer-segment GMV (static best vs bandits):")
        print(
            f"{'Segment':15s}  {'Static GMV':>10s}  {'LinUCB GMV':>10s}  "
            f"{'TS GMV':>10s}  {'LinUCB Δ%':>11s}  {'TS Δ%':>8s}"
        )

    for seg in segment_order:
        s = _get_segment_result(static_segments, seg, static_global)
        l = _get_segment_result(lin_segments, seg, static_global)
        t = _get_segment_result(ts_segments, seg, static_global)
        lin_delta = _pct(l["gmv"] - s["gmv"], s["gmv"])
        ts_delta = _pct(t["gmv"] - s["gmv"], s["gmv"])
        if show_volume:
            lin_orders_delta = _pct(l["orders"] - s["orders"], s["orders"]) if s["orders"] else 0.0
            ts_orders_delta = _pct(t["orders"] - s["orders"], s["orders"]) if s["orders"] else 0.0
            print(
                f"{seg:15s}  {s['gmv']:10.2f}  {l['gmv']:10.2f}  {t['gmv']:10.2f}  "
                f"{s['orders']:14.2f}  {l['orders']:14.2f}  {t['orders']:11.2f}  "
                f"{lin_delta:+12.2f}%  {ts_delta:+11.2f}%  "
                f"{lin_orders_delta:+15.2f}%  {ts_orders_delta:+14.2f}%"
            )
        else:
            print(
                f"{seg:15s}  {s['gmv']:10.2f}  {l['gmv']:10.2f}  {t['gmv']:10.2f}  "
                f"{lin_delta:+10.2f}%  {ts_delta:+7.2f}%"
            )


def _print_template_freqs(results: Dict) -> None:
    lin_freqs = results["linucb"]["diagnostics"]["template_freqs"]
    ts_freqs = results["ts"]["diagnostics"]["template_freqs"]
    template_names = [t["name"] for t in results["templates"]]

    print("\nTemplate selection frequencies — LinUCB:")
    for idx, (name, freq) in enumerate(zip(template_names, lin_freqs)):
        print(f"  {idx:2d} {name:15s}: {100.0 * freq:6.2f}%")
    print("\nTemplate selection frequencies — ThompsonSampling:")
    for idx, (name, freq) in enumerate(zip(template_names, ts_freqs)):
        print(f"  {idx:2d} {name:15s}: {100.0 * freq:6.2f}%")


def _print_full_summary(
    *,
    results: Dict,
    label: str,
    segment_order: List[str],
    show_volume: bool,
) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    _print_static_table(results, show_volume)
    _print_policy_summary(results, show_volume)
    _print_segment_summary(
        results=results,
        segment_order=segment_order,
        show_volume=show_volume,
    )
    _print_template_freqs(results)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    shared = args.n_static
    static_episodes = _resolve(args.n_static, shared)
    bandit_episodes = _resolve(args.n_bandit, args.n_bandit)

    print("=" * 80)
    print("CHAPTER 6 — GPU COMPUTE ARC: SIMPLE → RICH FEATURES")
    print("=" * 80)
    print(f"Static episodes: {static_episodes:,}")
    print(f"Bandit episodes: {bandit_episodes:,}")
    print(f"Base seed:      {args.base_seed}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Device:         {args.device}")

    cfg = SimulatorConfig(seed=args.base_seed)
    rng = np.random.default_rng(cfg.seed)
    products = catalog_module.generate_catalog(cfg.catalog, rng)
    print(f"\nCatalog ready: {len(products):,} products, top_k={cfg.top_k}")

    def _run(feature_mode: str, label: str) -> dict:
        prior = resolve_prior_weight(feature_mode, args.prior_weight)
        lin_alpha = resolve_lin_alpha(feature_mode, args.lin_alpha)
        ts_sigma = resolve_ts_sigma(feature_mode, args.ts_sigma)
        print(
            f"\n{label} — features={feature_mode}, "
            f"prior={prior}, lin_alpha={lin_alpha:.2f}, ts_sigma={ts_sigma:.2f}"
        )
        return run_template_bandits_experiment_gpu(
            cfg=cfg,
            products=products,
            n_static=static_episodes,
            n_bandit=bandit_episodes,
            feature_mode=feature_mode,
            base_seed=args.base_seed,
            prior_weight=prior,
            lin_alpha=lin_alpha,
            ts_sigma=ts_sigma,
            batch_size=args.batch_size,
            device=None if args.device == "auto" else args.device,
        )

    simple_results = _run("simple", "Experiment 1: Simple features (failure mode)")
    simple_static_gmv = simple_results["static_best"]["result"]["gmv"]
    simple_lin_gmv = simple_results["linucb"]["global"]["gmv"]
    simple_ts_gmv = simple_results["ts"]["global"]["gmv"]

    print("\nRESULTS (Simple Features):")
    print(f"  Static (best):  GMV = {simple_static_gmv:.2f}")
    print(
        f"  LinUCB:         GMV = {simple_lin_gmv:.2f}  "
        f"({100 * (simple_lin_gmv / max(simple_static_gmv, 1e-8) - 1):+.1f}%)"
    )
    print(
        f"  TS:             GMV = {simple_ts_gmv:.2f}  "
        f"({100 * (simple_ts_gmv / max(simple_static_gmv, 1e-8) - 1):+.1f}%)"
    )
    if simple_lin_gmv < simple_static_gmv * 0.9:
        print("\n⚠️  Bandits underperform static baselines (expected for failure mode).")

    rich_results = _run("rich", "Experiment 2: Rich features (oracle)")
    rich_lin_gmv = rich_results["linucb"]["global"]["gmv"]
    rich_ts_gmv = rich_results["ts"]["global"]["gmv"]

    print("\nRESULTS (Rich Features):")
    print(f"  Static (best):  GMV = {simple_static_gmv:.2f}  (shared world)")
    print(
        f"  LinUCB:         GMV = {rich_lin_gmv:.2f}  "
        f"({100 * (rich_lin_gmv / max(simple_static_gmv, 1e-8) - 1):+.1f}%)"
    )
    print(
        f"  TS:             GMV = {rich_ts_gmv:.2f}  "
        f"({100 * (rich_ts_gmv / max(simple_static_gmv, 1e-8) - 1):+.1f}%)"
    )

    linucb_improvement = rich_lin_gmv - simple_lin_gmv
    ts_improvement = rich_ts_gmv - simple_ts_gmv
    print("\nIMPROVEMENT (Rich vs Simple):")
    print(
        f"  LinUCB:  {linucb_improvement:+.2f} GMV  "
        f"({100 * linucb_improvement / max(simple_lin_gmv, 1e-8):+.1f}%)"
    )
    print(
        f"  TS:      {ts_improvement:+.2f} GMV  "
        f"({100 * ts_improvement / max(simple_ts_gmv, 1e-8):+.1f}%)"
    )
    if linucb_improvement > 0.0:
        print("\n✓ Rich features recover the Chapter 6 improvement signal.")

    simple_path = args.out_dir / "template_bandits_simple_gpu_summary.json"
    rich_path = args.out_dir / "template_bandits_rich_gpu_summary.json"
    with simple_path.open("w", encoding="utf-8") as f:
        json.dump(simple_results, f, indent=2)
    with rich_path.open("w", encoding="utf-8") as f:
        json.dump(rich_results, f, indent=2)

    print(f"\n✓ Saved: {simple_path}")
    print(f"✓ Saved: {rich_path}")

    _print_full_summary(
        results=simple_results,
        label="Experiment 1 Summary — Simple features",
        segment_order=cfg.users.segments,
        show_volume=args.show_volume,
    )
    _print_full_summary(
        results=rich_results,
        label="Experiment 2 Summary — Rich features",
        segment_order=cfg.users.segments,
        show_volume=args.show_volume,
    )

    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    try:
        plot_root = Path(__file__).resolve().parents[1] / "optimization"
        if str(plot_root) not in sys.path:
            sys.path.append(str(plot_root))
        from plot_results import (  # type: ignore import-not-found
            plot_segment_comparison,
            plot_template_frequencies,
        )

        print("\nPlot 1: Segment GMV comparison...")
        plot_segment_comparison(simple_path, rich_path, figures_dir)

        print("Plot 2: Template selection frequencies (simple features)...")
        plot_template_frequencies(simple_path, figures_dir, suffix="simple_gpu")

        print("Plot 3: Template selection frequencies (rich features)...")
        plot_template_frequencies(rich_path, figures_dir, suffix="rich_gpu")
    except ImportError as exc:
        print(f"\n⚠️  Plotting skipped (plot_results.py unavailable): {exc}")
        print("    Ensure scripts/ch06/plot_results.py is on PYTHONPATH.")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nData artifacts:    {args.out_dir}")
    print(f"Figure artifacts:  {figures_dir}")
    print("\nRegenerate with:")
    print("  python scripts/ch06/optimization_gpu/ch06_compute_arc_gpu.py \\")
    print(f"      --n-static {args.n_static} \\")
    print(f"      --n-bandit {args.n_bandit} \\")
    print(f"      --base-seed {args.base_seed}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
