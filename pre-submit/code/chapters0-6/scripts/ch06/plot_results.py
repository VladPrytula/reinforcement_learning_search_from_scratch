"""Plotting utilities for Chapter 6 template bandit experiments.

Generates publication-quality figures for the chapter:
- Three-way segment GMV comparison (simple → oracle → estimated)
- Template selection frequencies (LinUCB and Thompson Sampling)

The three-stage compute arc demonstrates the Algorithm Selection Principle:
  - Simple features: Both bandits fail
  - Rich + Oracle latents: LinUCB wins (precise exploitation)
  - Rich + Estimated latents: TS wins (robust exploration)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_segment_comparison(
    simple_json_path: Path,
    oracle_json_path: Path,
    estimated_json_path: Path,
    out_dir: Path,
) -> None:
    """Bar chart comparing segment-level GMV across three feature modes.

    Three-stage comparison demonstrating the Algorithm Selection Principle:
      - Simple features: Both bandits fail
      - Rich + Oracle latents: LinUCB wins
      - Rich + Estimated latents: TS wins

    Args:
        simple_json_path: Path to template_bandits_simple_summary.json.
        oracle_json_path: Path to template_bandits_rich_oracle_summary.json.
        estimated_json_path: Path to template_bandits_rich_estimated_summary.json.
        out_dir: Output directory for figure.
    """
    with open(simple_json_path, encoding="utf-8") as f:
        simple = json.load(f)
    with open(oracle_json_path, encoding="utf-8") as f:
        oracle = json.load(f)
    with open(estimated_json_path, encoding="utf-8") as f:
        estimated = json.load(f)

    segments = list(simple["static_best"]["segments"].keys())

    # Extract per-segment GMV for each policy and feature mode
    static_gmvs = [simple["static_best"]["segments"][s]["gmv"] for s in segments]

    linucb_simple = [simple["linucb"]["segments"][s]["gmv"] for s in segments]
    linucb_oracle = [oracle["linucb"]["segments"][s]["gmv"] for s in segments]
    linucb_estimated = [estimated["linucb"]["segments"][s]["gmv"] for s in segments]

    ts_simple = [simple["ts"]["segments"][s]["gmv"] for s in segments]
    ts_oracle = [oracle["ts"]["segments"][s]["gmv"] for s in segments]
    ts_estimated = [estimated["ts"]["segments"][s]["gmv"] for s in segments]

    x = np.arange(len(segments))
    width = 0.11  # Narrower bars to fit 7 bars per segment

    fig, ax = plt.subplots(figsize=(14, 7))

    # Static baseline (best static template from the reference run)
    static_name = simple["static_best"]["name"]
    ax.bar(
        x - 3 * width,
        static_gmvs,
        width,
        label=f"Static ({static_name})",
        color="gray",
        alpha=0.8,
    )

    # LinUCB progression (red shades: fail → win → fall back)
    ax.bar(
        x - 2 * width,
        linucb_simple,
        width,
        label="LinUCB (simple) - FAIL",
        color="#d62728",
        alpha=0.5,
    )
    ax.bar(
        x - width,
        linucb_oracle,
        width,
        label="LinUCB (oracle) - WIN",
        color="#2ca02c",
        alpha=0.8,
    )
    ax.bar(
        x,
        linucb_estimated,
        width,
        label="LinUCB (estimated)",
        color="#98df8a",
        alpha=0.7,
    )

    # TS progression (blue shades: fail → modest → win)
    ax.bar(
        x + width,
        ts_simple,
        width,
        label="TS (simple) - FAIL",
        color="#ff7f0e",
        alpha=0.5,
    )
    ax.bar(
        x + 2 * width,
        ts_oracle,
        width,
        label="TS (oracle)",
        color="#aec7e8",
        alpha=0.7,
    )
    ax.bar(
        x + 3 * width,
        ts_estimated,
        width,
        label="TS (estimated) - WIN",
        color="#1f77b4",
        alpha=0.8,
    )

    ax.set_xlabel("User Segment", fontsize=12)
    ax.set_ylabel("GMV (per episode)", fontsize=12)
    ax.set_title(
        "The Algorithm Selection Principle: Feature Quality Determines Winner",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=15, ha="right")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "segment_gmv_comparison_threeway.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_template_frequencies(
    results_json_path: Path,
    out_dir: Path,
    suffix: str = "",
) -> None:
    """Bar chart showing template selection frequencies.

    X-axis: Template names.
    Y-axis: Selection frequency (0-1).
    Bars: LinUCB, Thompson Sampling.

    Args:
        results_json_path: Path to summary JSON output.
        out_dir: Output directory.
        suffix: Optional suffix for filename (e.g., "simple", "rich").
    """
    with open(results_json_path, encoding="utf-8") as f:
        results = json.load(f)

    templates = results["templates"]
    template_names = [t["name"] for t in templates]

    linucb_freqs = results["linucb"]["diagnostics"]["template_freqs"]
    ts_freqs = results["ts"]["diagnostics"]["template_freqs"]

    x = np.arange(len(templates))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x - width / 2,
        linucb_freqs,
        width,
        label="LinUCB",
        color="#2ca02c",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        ts_freqs,
        width,
        label="Thompson Sampling",
        color="#1f77b4",
        alpha=0.7,
    )

    ax.set_xlabel("Template", fontsize=12)
    ax.set_ylabel("Selection Frequency", fontsize=12)
    feature_mode = results["config"]["feature_mode"]
    ax.set_title(
        f"Template Selection Frequencies ({feature_mode} features)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(template_names, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    filename = f"template_selection_{suffix}.png" if suffix else "template_selection.png"
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print(
            "Usage: python plot_results.py <simple_json> <oracle_json> <estimated_json> <out_dir>"
        )
        print("")
        print("  Three-stage compute arc plots demonstrating Algorithm Selection Principle:")
        print("    - Stage 1: Simple features (bandits fail)")
        print("    - Stage 2: Rich + Oracle latents (LinUCB wins)")
        print("    - Stage 3: Rich + Estimated latents (TS wins)")
        raise SystemExit(1)

    simple_json_arg = Path(sys.argv[1])
    oracle_json_arg = Path(sys.argv[2])
    estimated_json_arg = Path(sys.argv[3])
    out_dir_arg = Path(sys.argv[4])
    out_dir_arg.mkdir(parents=True, exist_ok=True)

    print("Generating Chapter 6 figures...")
    plot_segment_comparison(simple_json_arg, oracle_json_arg, estimated_json_arg, out_dir_arg)
    plot_template_frequencies(simple_json_arg, out_dir_arg, suffix="simple")
    plot_template_frequencies(oracle_json_arg, out_dir_arg, suffix="rich_oracle")
    plot_template_frequencies(estimated_json_arg, out_dir_arg, suffix="rich_estimated")
    print("Done.")
