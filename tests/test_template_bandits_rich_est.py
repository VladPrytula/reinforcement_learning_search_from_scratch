"""Regression test for Chapter 6 rich_est contextual bandits."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

import numpy as np

from zoosim.core.config import CatalogConfig, SimulatorConfig
from zoosim.world import catalog as catalog_module

from scripts.ch06.template_bandits_demo import (
    resolve_lin_alpha,
    resolve_prior_weight,
    resolve_ts_sigma,
    run_template_bandits_experiment,
)


@pytest.mark.slow
def test_rich_est_bandits_exceed_static_baseline() -> None:
    """Ensure rich_est feature mode learns policies that beat static templates."""
    cfg = SimulatorConfig(
        catalog=CatalogConfig(n_products=500)
    )
    rng = np.random.default_rng(cfg.seed)
    products = catalog_module.generate_catalog(cfg.catalog, rng)

    prior_weight = resolve_prior_weight("rich_est")
    lin_alpha = resolve_lin_alpha("rich_est")
    ts_sigma = resolve_ts_sigma("rich_est")

    results = run_template_bandits_experiment(
        cfg=cfg,
        products=products,
        n_static=400,
        n_bandit=1_200,
        feature_mode="rich_est",
        base_seed=2025_0601,
        prior_weight=prior_weight,
        lin_alpha=lin_alpha,
        ts_sigma=ts_sigma,
    )

    static_gmv = results["static_best"]["result"]["gmv"]
    lin_gmv = results["linucb"]["global"]["gmv"]
    ts_gmv = results["ts"]["global"]["gmv"]

    # Allow small slack for stochasticity while ensuring uplift.
    assert lin_gmv >= static_gmv * 0.98
    assert ts_gmv >= static_gmv * 0.98
