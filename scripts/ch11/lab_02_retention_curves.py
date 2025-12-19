#!/usr/bin/env python
"""Lab 11.2 — Retention curves and monotonicity diagnostics."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zoosim.core.config import load_default_config
from zoosim.multi_episode.retention import return_probability


def compute_heatmap(
    cfg,
    max_clicks: int = 15,
    n_sat: int = 21,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clicks = np.arange(0, max_clicks + 1)
    sats = np.linspace(0.0, 1.0, n_sat)
    grid = np.zeros((len(sats), len(clicks)), dtype=float)

    for i, s in enumerate(sats):
        for j, c in enumerate(clicks):
            grid[i, j] = return_probability(clicks=c, satisfaction=s, config=cfg)

    return clicks, sats, grid


def validate_monotonicity(cfg) -> Dict[str, float]:
    ps = []
    for sat in np.linspace(0.0, 1.0, 11):
        prev = 0.0
        for clicks in range(16):
            p = return_probability(clicks=clicks, satisfaction=sat, config=cfg)
            if p < prev - 1e-8:
                raise AssertionError(
                    f"Monotonicity violated at clicks={clicks}, sat={sat}: {p} < {prev}"
                )
            prev = p
            ps.append(p)
    return {"min": float(min(ps)), "max": float(max(ps))}


def plot_retention_heatmap(clicks, sats, grid, out_prefix: str) -> None:
    plt.figure(figsize=(6, 4))
    X, Y = np.meshgrid(clicks, sats)
    im = plt.pcolormesh(X, Y, grid, shading="auto", cmap="viridis")
    plt.xlabel("Clicks")
    plt.ylabel("Satisfaction")
    plt.title("P(return) vs clicks, satisfaction")
    plt.colorbar(im, label="P(return)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_heatmap.png")


def plot_retention_vs_clicks(cfg, out_prefix: str) -> None:
    clicks = np.arange(0, 16)
    sats = [0.0, 0.3, 0.6, 1.0]

    plt.figure(figsize=(6, 4))
    for s in sats:
        ps = [
            return_probability(clicks=c, satisfaction=s, config=cfg)
            for c in clicks
        ]
        plt.plot(clicks, ps, label=f"satisfaction={s:.1f}")

    plt.xlabel("Clicks")
    plt.ylabel("P(return)")
    plt.title("Retention vs clicks at fixed satisfaction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_lines.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab 11.2 — Retention curves and monotonicity diagnostics.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="docs/book/ch11/figures/lab11_retention",
    )
    args = parser.parse_args()

    cfg = load_default_config()
    clicks, sats, grid = compute_heatmap(cfg)
    stats = validate_monotonicity(cfg)

    print("Retention probability statistics:", stats)
    plot_retention_heatmap(clicks, sats, grid, args.out_prefix)
    plot_retention_vs_clicks(cfg, args.out_prefix)


if __name__ == "__main__":
    main()

