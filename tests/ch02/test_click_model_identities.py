"""Chapter 2 tests: executable click-model identities.

These tests promote two Lab 2.3–2.4 verifications into the test suite:
- PBM-style per-position marginals under a PBM-like specialization of the production model.
- DBN examination identity from #EQ-2.3 under a toy DBN simulator.

We also test that `BehaviorConfig.exam_sensitivity` has the intended effect on downstream
examination when satisfaction is positive (#EQ-2.6–#EQ-2.7).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from zoosim.core.config import SimulatorConfig
from zoosim.dynamics.behavior import simulate_session
from zoosim.world.catalog import generate_catalog
from zoosim.world.queries import sample_query
from zoosim.world.users import sample_user


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def test_pbm_like_config_matches_per_position_marginals() -> None:
    """Verify Proposition 2.5.4(a) numerically under a PBM-like specialization."""
    cfg = SimulatorConfig()
    cfg.top_k = 5
    cfg.catalog.n_products = 32
    cfg.queries.query_type_mix = [1.0, 0.0, 0.0]  # force "category"

    beh = cfg.behavior
    beh.alpha_price = 0.0
    beh.alpha_pl = 0.0
    beh.alpha_cat = 0.0
    beh.sigma_u = 0.0
    beh.exam_sensitivity = 0.0
    beh.satisfaction_gain = 0.0
    beh.satisfaction_decay = 0.0
    beh.post_purchase_fatigue = 0.0
    beh.abandonment_threshold = -100.0
    beh.max_purchases = 999
    beh.beta_buy = 0.0
    beh.beta0 = -100.0  # disable purchases

    rng = np.random.default_rng(seed=7)
    catalog = generate_catalog(cfg.catalog, rng)
    user = sample_user(config=cfg, rng=rng)
    query = sample_query(user=user, config=cfg, rng=rng)
    ranking = list(range(cfg.top_k))

    pos_bias_vec = beh.pos_bias[query.query_type]
    continuation_probs = np.array(
        [_sigmoid(pos_bias_vec[position]) for position in range(cfg.top_k)],
        dtype=float,
    )
    exam_marginals = np.cumprod(continuation_probs)

    relevance = np.zeros(cfg.top_k, dtype=float)
    for position in range(cfg.top_k):
        product = catalog[ranking[position]]
        match = float(
            torch.nn.functional.cosine_similarity(query.phi_emb, product.embedding, dim=0)
        )
        relevance[position] = _sigmoid(beh.alpha_rel * match)

    ctr_theory = exam_marginals * relevance

    n_sessions = 20_000
    click_counts = np.zeros(cfg.top_k, dtype=float)
    for _ in range(n_sessions):
        outcome = simulate_session(
            user=user,
            query=query,
            ranking=ranking,
            catalog=catalog,
            config=cfg,
            rng=rng,
        )
        click_counts += np.asarray(outcome.clicks, dtype=float)

    ctr_empirical = click_counts / n_sessions
    max_abs_error = float(np.max(np.abs(ctr_empirical - ctr_theory)))
    assert max_abs_error < 0.02, (
        "PBM-like per-position marginals deviated from theory: "
        f"max_abs_error={max_abs_error:.4f}, "
        f"ctr_empirical={ctr_empirical}, "
        f"ctr_theory={ctr_theory}"
    )


def test_dbn_examination_identity_matches_eq_2_3() -> None:
    """Verify the DBN examination identity #EQ-2.3 under a toy DBN simulator."""
    rng = np.random.default_rng(seed=11)

    relevance = np.array([0.70, 0.62, 0.55, 0.48, 0.42, 0.36, 0.30, 0.25], dtype=float)
    satisfaction = np.array([0.25, 0.22, 0.20, 0.18, 0.15, 0.13, 0.10, 0.08], dtype=float)
    n_positions = int(relevance.size)
    n_sessions = 50_000

    exam_counts = np.zeros(n_positions, dtype=float)
    for _ in range(n_sessions):
        for position in range(n_positions):
            exam_counts[position] += 1.0
            clicked = rng.random() < relevance[position]
            satisfied = clicked and (rng.random() < satisfaction[position])
            if satisfied:
                break

    exam_empirical = exam_counts / n_sessions

    rel_sat = relevance * satisfaction
    exam_theory = np.ones(n_positions, dtype=float)
    survival = 1.0
    for position in range(1, n_positions):
        survival *= 1.0 - rel_sat[position - 1]
        exam_theory[position] = survival

    max_abs_error = float(np.max(np.abs(exam_empirical - exam_theory)))
    assert max_abs_error < 0.01, (
        "DBN examination identity deviated from #EQ-2.3: "
        f"max_abs_error={max_abs_error:.4f}, "
        f"exam_empirical={exam_empirical}, "
        f"exam_theory={exam_theory}"
    )


def test_exam_sensitivity_increases_downstream_examination() -> None:
    """`exam_sensitivity > 0` should increase downstream examination when satisfaction is positive."""
    cfg = SimulatorConfig()
    cfg.top_k = 2
    cfg.catalog.n_products = 2
    cfg.queries.query_type_mix = [1.0, 0.0, 0.0]  # force "category"

    beh = cfg.behavior
    beh.alpha_rel = 20.0
    beh.alpha_price = 0.0
    beh.alpha_pl = 0.0
    beh.alpha_cat = 0.0
    beh.sigma_u = 0.0
    beh.satisfaction_gain = 1.0
    beh.satisfaction_decay = 0.0
    beh.post_purchase_fatigue = 0.0
    beh.abandonment_threshold = -100.0
    beh.max_purchases = 999
    beh.beta_buy = 0.0
    beh.beta0 = -100.0
    beh.pos_bias = {
        "category": [10.0, -2.0],
        "brand": [10.0, -2.0],
        "generic": [10.0, -2.0],
    }

    setup_rng = np.random.default_rng(seed=123)
    catalog = generate_catalog(cfg.catalog, setup_rng)
    user = sample_user(config=cfg, rng=setup_rng)
    query = sample_query(user=user, config=cfg, rng=setup_rng)
    for product in catalog:
        product.embedding = query.phi_emb.clone()

    ranking = [0, 1]
    n_sessions = 5_000

    def estimate_rates(exam_sensitivity: float, seed: int) -> tuple[float, float, float]:
        beh.exam_sensitivity = exam_sensitivity
        rng = np.random.default_rng(seed=seed)
        click_count_pos1 = 0
        click_count_pos2 = 0
        reached_pos2 = 0
        click_count_pos2_given_reached = 0
        for _ in range(n_sessions):
            outcome = simulate_session(
                user=user,
                query=query,
                ranking=ranking,
                catalog=catalog,
                config=cfg,
                rng=rng,
            )
            clicked_pos1 = int(outcome.clicks[0])
            clicked_pos2 = int(outcome.clicks[1])
            click_count_pos1 += clicked_pos1
            click_count_pos2 += clicked_pos2
            if clicked_pos1:
                reached_pos2 += 1
                click_count_pos2_given_reached += clicked_pos2

        click_rate_pos1 = click_count_pos1 / n_sessions
        click_rate_pos2 = click_count_pos2 / n_sessions
        click_rate_pos2_given_reached = click_count_pos2_given_reached / max(1, reached_pos2)
        return click_rate_pos1, click_rate_pos2, click_rate_pos2_given_reached

    (
        click_rate_pos1_no_sensitivity,
        click_rate_pos2_no_sensitivity,
        click_rate_pos2_given_reached_no_sensitivity,
    ) = estimate_rates(exam_sensitivity=0.0, seed=1)
    (
        click_rate_pos1_with_sensitivity,
        click_rate_pos2_with_sensitivity,
        click_rate_pos2_given_reached_with_sensitivity,
    ) = estimate_rates(exam_sensitivity=0.2, seed=2)

    assert abs(click_rate_pos1_with_sensitivity - click_rate_pos1_no_sensitivity) < 0.01, (
        "Expected position-1 click rate to be insensitive to exam_sensitivity (since satisfaction=0), "
        f"got click_rate_pos1_no_sensitivity={click_rate_pos1_no_sensitivity:.3f}, "
        f"click_rate_pos1_with_sensitivity={click_rate_pos1_with_sensitivity:.3f}"
    )

    assert click_rate_pos2_with_sensitivity > click_rate_pos2_no_sensitivity + 0.1, (
        "Expected downstream click rate to increase when exam_sensitivity > 0, "
        f"got click_rate_pos2_no_sensitivity={click_rate_pos2_no_sensitivity:.3f}, "
        f"click_rate_pos2_with_sensitivity={click_rate_pos2_with_sensitivity:.3f}"
    )

    assert (
        click_rate_pos2_given_reached_with_sensitivity
        > click_rate_pos2_given_reached_no_sensitivity + 0.1
    ), (
        "Expected downstream examination to increase when exam_sensitivity > 0 "
        "(approximated by P(click at pos2 | click at pos1) in this controlled setup), "
        f"got click_rate_pos2_given_reached_no_sensitivity={click_rate_pos2_given_reached_no_sensitivity:.3f}, "
        f"click_rate_pos2_given_reached_with_sensitivity={click_rate_pos2_given_reached_with_sensitivity:.3f}"
    )
