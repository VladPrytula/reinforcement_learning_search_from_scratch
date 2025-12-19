"""Core Chapter 5 unit tests.

These tests mirror the Chapter‑5 acceptance criteria:

- Hybrid lexical/semantic relevance:
  base_score matches its semantic + lexical decomposition when noise is disabled.
- Engineered boost features:
  compute_features has the correct dimension and standardize_features yields
  standardized features with mean ≈0 and std ≈1 for non‑constant coordinates.
- Reward equals weighted sum:
  compute_reward equals α·GMV + β·CM2 + γ·STRAT + δ·CLICKS on a simple pattern.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from zoosim.core.config import SimulatorConfig
from zoosim.dynamics.reward import compute_reward
from zoosim.ranking.features import compute_features, standardize_features
from zoosim.ranking.relevance import base_score
from zoosim.world.catalog import generate_catalog
from zoosim.world.queries import sample_query
from zoosim.world.users import sample_user


def test_base_score_matches_semantic_plus_lexical() -> None:
    """base_score equals w_sem * sem + w_lex * lex when noise is zero."""
    cfg = SimulatorConfig(seed=123)
    rng = np.random.default_rng(cfg.seed)

    # Disable noise for exact equality.
    cfg.relevance.noise_sigma = 0.0

    catalog = generate_catalog(cfg=cfg.catalog, rng=rng)
    user = sample_user(config=cfg, rng=rng)
    query = sample_query(user=user, config=cfg, rng=rng)
    product = catalog[0]

    score = base_score(query=query, product=product, config=cfg, rng=rng)

    sem = float(torch.nn.functional.cosine_similarity(query.phi_emb, product.embedding, dim=0))
    query_tokens = set(query.tokens)
    prod_tokens = set(product.category.split("_"))
    overlap = len(query_tokens & prod_tokens)
    lex = float(np.log1p(overlap))

    w_sem = float(cfg.relevance.w_sem)
    w_lex = float(cfg.relevance.w_lex)
    expected = w_sem * sem + w_lex * lex

    assert np.isclose(score, expected, atol=1e-10)


def test_features_dimension_and_standardization() -> None:
    """compute_features dimension matches config and standardization behaves as expected."""
    cfg = SimulatorConfig(seed=321)
    rng = np.random.default_rng(cfg.seed)

    catalog = generate_catalog(cfg=cfg.catalog, rng=rng)
    user = sample_user(config=cfg, rng=rng)
    query = sample_query(user=user, config=cfg, rng=rng)

    n_samples = 64
    indices = rng.integers(0, len(catalog), size=n_samples)

    raw_features: List[List[float]] = [
        compute_features(user=user, query=query, product=catalog[int(idx)], config=cfg)
        for idx in indices
    ]

    # Dimension check.
    feature_dim = len(raw_features[0])
    assert feature_dim == cfg.action.feature_dim

    # Standardization checks.
    standardized = standardize_features(raw_features, config=cfg)
    arr = np.asarray(standardized, dtype=float)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)

    # Non‑constant coordinates should have std close to 1; all means close to 0.
    non_constant = np.std(raw_features, axis=0) > 1e-12
    if np.any(non_constant):
        assert np.allclose(stds[non_constant], 1.0, atol=1e-6)
    assert np.allclose(means, 0.0, atol=1e-6)


def test_reward_equals_weighted_sum_of_components() -> None:
    """compute_reward equals α·GMV + β·CM2 + γ·STRAT + δ·CLICKS."""
    cfg = SimulatorConfig(seed=456)
    rng = np.random.default_rng(cfg.seed)

    catalog = generate_catalog(cfg=cfg.catalog, rng=rng)

    top_k = min(cfg.top_k, 5)
    ranking = list(range(top_k))
    clicks = [1, 1, 0, 1, 0][:top_k]
    buys = [1, 0, 0, 1, 0][:top_k]

    reward, breakdown = compute_reward(
        ranking=ranking,
        clicks=clicks,
        buys=buys,
        catalog=catalog,
        config=cfg,
    )

    rcfg = cfg.reward
    expected = (
        rcfg.alpha_gmv * breakdown.gmv
        + rcfg.beta_cm2 * breakdown.cm2
        + rcfg.gamma_strat * breakdown.strat
        + rcfg.delta_clicks * breakdown.clicks
    )

    assert np.isclose(reward, expected, atol=1e-10)

