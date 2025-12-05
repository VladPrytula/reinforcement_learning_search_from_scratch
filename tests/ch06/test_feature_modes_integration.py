"""Integration test: rich features improve over simple features.

This test validates the Chapter 6 narrative claim that richer
context features improve bandit performance. We run both feature
modes on the same world and assert directional improvement.

Because bandits are stochastic, we use multiple seeds and assert on
averaged improvement, not single-run exact values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from zoosim.core.config import SimulatorConfig
from zoosim.world import catalog as catalog_module

# Import core experiment function from the Chapter 6 demo script.
SCRIPT_DIR = Path(__file__).parent.parent.parent / "scripts" / "ch06"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from template_bandits_demo import run_template_bandits_experiment


@pytest.mark.slow
def test_rich_features_improve_over_simple() -> None:
    """Verify rich features reduce GMV gap vs static baseline.

    This runs a small-scale version of the Chapter 6 compute arc
    and asserts that rich features improve performance over simple
    features on average across several seeds.
    """
    n_static = 200
    n_bandit = 1_000

    seeds = [2025_0601 + i * 100 for i in range(3)]

    gaps_simple: list[float] = []
    gaps_rich: list[float] = []

    for idx, seed in enumerate(seeds, start=1):
        cfg = SimulatorConfig(seed=seed)
        rng = np.random.default_rng(seed)
        products = catalog_module.generate_catalog(cfg.catalog, rng)

        simple_results = run_template_bandits_experiment(
            cfg=cfg,
            products=products,
            n_static=n_static,
            n_bandit=n_bandit,
            feature_mode="simple",
            base_seed=seed,
        )

        rich_results = run_template_bandits_experiment(
            cfg=cfg,
            products=products,
            n_static=n_static,
            n_bandit=n_bandit,
            feature_mode="rich",
            base_seed=seed,
        )

        static_gmv = simple_results["static_best"]["result"]["gmv"]
        simple_linucb_gmv = simple_results["linucb"]["global"]["gmv"]
        rich_linucb_gmv = rich_results["linucb"]["global"]["gmv"]

        gap_simple = static_gmv - simple_linucb_gmv
        gap_rich = static_gmv - rich_linucb_gmv

        gaps_simple.append(gap_simple)
        gaps_rich.append(gap_rich)

        # Lightweight progress indicator so this slow test does not look stuck.
        # Visible when pytest is run with -s / no capture.
        print(
            f"[ch06/test_rich_features] seed {idx}/{len(seeds)} "
            f"done (simple_gap={gap_simple:.2f}, rich_gap={gap_rich:.2f})",
            flush=True,
        )

    mean_gap_simple = float(np.mean(gaps_simple))
    mean_gap_rich = float(np.mean(gaps_rich))

    assert mean_gap_rich < mean_gap_simple, (
        "Rich features should reduce the GMV gap vs static baseline.\n"
        f"  Simple-features gap: {mean_gap_simple:.2f} (bandits below static)\n"
        f"  Rich-features gap:   {mean_gap_rich:.2f} (should be lower)\n"
        f"  Improvement:         {mean_gap_simple - mean_gap_rich:.2f} GMV\n"
        f"Seeds tested:          {seeds}"
    )


def test_template_selection_tracking() -> None:
    """Verify template selection diagnostics are tracked correctly."""
    cfg = SimulatorConfig(seed=2025_0601)
    rng = np.random.default_rng(cfg.seed)
    products = catalog_module.generate_catalog(cfg.catalog, rng)

    results = run_template_bandits_experiment(
        cfg=cfg,
        products=products,
        n_static=100,
        n_bandit=500,
        feature_mode="simple",
        base_seed=2025_0601,
    )

    linucb_counts = results["linucb"]["diagnostics"]["template_counts"]
    linucb_freqs = results["linucb"]["diagnostics"]["template_freqs"]

    assert len(linucb_counts) == 8
    assert sum(linucb_counts) == 500
    assert abs(sum(linucb_freqs) - 1.0) < 1e-6

    ts_counts = results["ts"]["diagnostics"]["template_counts"]
    ts_freqs = results["ts"]["diagnostics"]["template_freqs"]

    assert len(ts_counts) == 8
    assert sum(ts_counts) == 500
    assert abs(sum(ts_freqs) - 1.0) < 1e-6
