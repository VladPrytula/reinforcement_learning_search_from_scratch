"""Chapter 2 tests: click behavior and position bias.

This module checks that the production click model (`simulate_session`)
respects position bias in aggregate: higher-ranked positions receive
at least as many clicks as lower-ranked positions, all else equal.

Mathematical correspondence:
- Section 2.5 (PBM / DBN intuition)
- Definition 2.5.3 (Utility-Based Cascade Model)
"""

from __future__ import annotations

import numpy as np

from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import generate_catalog
from zoosim.world.users import sample_user
from zoosim.world.queries import sample_query
from zoosim.dynamics.behavior import simulate_session


def test_click_rate_decays_with_position() -> None:
    """Verify that click probability decreases with position in aggregate.

    We construct a small simulator configuration, fix the query type to
    ``\"category\"`` so that a single position-bias vector applies, and
    run many independent sessions with a trivial ranking (identity order).
    Because product attributes are i.i.d. across positions, any systematic
    difference in click probability by position should be driven by the
    position bias and cascade structure from Chapter 2.
    """
    cfg = SimulatorConfig()

    # Make the test lightweight while preserving behavior.
    cfg.catalog.n_products = 64
    cfg.top_k = 5

    # Simplify satisfaction dynamics so that examination probabilities
    # are predominantly controlled by position bias.
    beh = cfg.behavior
    beh.satisfaction_gain = 0.0
    beh.satisfaction_decay = 0.0
    beh.post_purchase_fatigue = 0.0
    beh.abandonment_threshold = -100.0
    beh.max_purchases = 999

    # Force all queries to use the "category" position-bias vector.
    cfg.queries.query_type_mix = [1.0, 0.0, 0.0]

    rng = np.random.default_rng(seed=123)
    catalog = generate_catalog(cfg.catalog, rng)

    n_sessions = 5_000
    click_counts = np.zeros(cfg.top_k, dtype=float)

    for _ in range(n_sessions):
        user = sample_user(config=cfg, rng=rng)
        query = sample_query(user=user, config=cfg, rng=rng)
        # Randomize ranking to average out product attributes across positions
        ranking = list(rng.permutation(len(catalog))[:cfg.top_k])
        outcome = simulate_session(
            user=user,
            query=query,
            ranking=ranking,
            catalog=catalog,
            config=cfg,
            rng=rng,
        )
        click_counts += np.array(outcome.clicks, dtype=float)

    click_rates = click_counts / n_sessions

    # Click probability at each position should be non-increasing in position,
    # up to small Monte Carlo fluctuations.
    tol = 0.03
    for k in range(cfg.top_k - 1):
        assert (
            click_rates[k] + tol >= click_rates[k + 1]
        ), f"Click rate increased from position {k} to {k+1}: {click_rates}"

    # Additionally, top position should be meaningfully stronger than the last.
    assert click_rates[0] > click_rates[-1] + tol


if __name__ == "__main__":  # pragma: no cover
    import pytest
    import sys

    sys.exit(pytest.main(["-q", __file__]))

