#!/usr/bin/env python
"""Chapter 5 validation script.

Runs small, deterministic checks that mirror the Chapter‑5 acceptance criteria:

1. Hybrid base relevance: `base_score()` matches its semantic + lexical decomposition.
2. Feature engineering: `compute_features()` has the correct dimension and
   `standardize_features()` produces ~0 mean and ~1 std for non‑constant features.
3. Reward aggregation: `compute_reward()` equals the weighted sum of components.

Usage:
    python scripts/validate_ch05.py
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from zoosim.core.config import SimulatorConfig
from zoosim.dynamics.reward import RewardBreakdown, compute_reward
from zoosim.ranking.features import compute_features, standardize_features
from zoosim.ranking.relevance import base_score
from zoosim.world.catalog import Product, generate_catalog
from zoosim.world.queries import Query, sample_query
from zoosim.world.users import User, sample_user


def _validate_base_score_decomposition(config: SimulatorConfig, rng: np.random.Generator) -> None:
    """Check that base_score matches its semantic + lexical decomposition."""
    print("=" * 80)
    print("VALIDATION 1: BASE RELEVANCE DECOMPOSITION")
    print("=" * 80)

    # Disable noise so the decomposition is exact
    config.relevance.noise_sigma = 0.0

    catalog = generate_catalog(cfg=config.catalog, rng=rng)
    user: User = sample_user(config=config, rng=rng)
    query: Query = sample_query(user=user, config=config, rng=rng)

    # Pick a single product for a precise check
    product: Product = catalog[0]

    score = base_score(query=query, product=product, config=config, rng=rng)

    # Manual decomposition (mirrors zoosim/ranking/relevance.py)
    sem = float(torch.nn.functional.cosine_similarity(query.phi_emb, product.embedding, dim=0))
    query_tokens = set(query.tokens)
    prod_tokens = set(product.category.split("_"))
    overlap = len(query_tokens & prod_tokens)
    lex = float(np.log1p(overlap))

    w_sem = float(config.relevance.w_sem)
    w_lex = float(config.relevance.w_lex)
    expected = w_sem * sem + w_lex * lex

    print(f"Semantic component: {sem:.6f}")
    print(f"Lexical component:  {lex:.6f}")
    print(f"w_sem={w_sem:.3f}, w_lex={w_lex:.3f}")
    print(f"base_score(...) = {score:.6f}")
    print(f"Expected        = {expected:.6f}")

    assert np.isclose(score, expected, atol=1e-10), "base_score does not match semantic + lexical decomposition"
    print("✓ base_score matches semantic + lexical decomposition (noise_sigma=0.0)")


def _validate_features_and_standardization(config: SimulatorConfig, rng: np.random.Generator) -> None:
    """Check feature dimension and standardization properties."""
    print("\n" + "=" * 80)
    print("VALIDATION 2: FEATURES AND STANDARDIZATION")
    print("=" * 80)

    catalog = generate_catalog(cfg=config.catalog, rng=rng)
    user: User = sample_user(config=config, rng=rng)
    query: Query = sample_query(user=user, config=config, rng=rng)

    # Sample a small batch of products for feature statistics
    n_samples = 50
    indices = rng.integers(0, len(catalog), size=n_samples)

    raw_features: List[List[float]] = [
        compute_features(user=user, query=query, product=catalog[int(idx)], config=config)
        for idx in indices
    ]

    # Check feature dimension against config
    feature_dim = len(raw_features[0])
    print(f"Feature dimension (len(compute_features(...))): {feature_dim}")
    print(f"config.action.feature_dim:                     {config.action.feature_dim}")
    assert feature_dim == config.action.feature_dim, "Feature dimension mismatch with config.action.feature_dim"
    print("✓ Feature dimension matches config.action.feature_dim")

    # Standardize and check mean/std properties
    standardized = standardize_features(raw_features, config=config)
    arr = np.asarray(standardized, dtype=float)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)

    print("\nStandardized feature means (first 5 dims):", np.round(means[:5], 6))
    print("Standardized feature stds  (first 5 dims):", np.round(stds[:5], 6))

    # For non-constant features, std should be close to 1
    # We allow a small numerical tolerance.
    non_constant = np.std(raw_features, axis=0) > 1e-12
    if np.any(non_constant):
        assert np.allclose(stds[non_constant], 1.0, atol=1e-6), "Standardized stds deviate from 1 for non-constant features"
    assert np.allclose(means, 0.0, atol=1e-6), "Standardized means deviate from 0"

    print("✓ Standardized features have mean ≈ 0 and std ≈ 1 for non-constant dimensions")


def _validate_reward_aggregation(config: SimulatorConfig, rng: np.random.Generator) -> None:
    """Check that compute_reward matches the weighted sum of its components."""
    print("\n" + "=" * 80)
    print("VALIDATION 3: REWARD AGGREGATION")
    print("=" * 80)

    catalog = generate_catalog(cfg=config.catalog, rng=rng)

    # Construct a simple synthetic ranking / clicks / buys pattern.
    top_k = min(config.top_k, 5)
    ranking = list(range(top_k))

    # Deterministic click/buy pattern for reproducibility
    clicks = [1, 1, 0, 1, 0][:top_k]
    buys = [1, 0, 0, 1, 0][:top_k]

    reward, breakdown = compute_reward(
        ranking=ranking,
        clicks=clicks,
        buys=buys,
        catalog=catalog,
        config=config,
    )

    # Manual aggregation mirroring zoosim/dynamics/reward.py
    cfg = config.reward
    expected_reward = (
        cfg.alpha_gmv * breakdown.gmv
        + cfg.beta_cm2 * breakdown.cm2
        + cfg.gamma_strat * breakdown.strat
        + cfg.delta_clicks * breakdown.clicks
    )

    print(f"GMV:    {breakdown.gmv:.4f}")
    print(f"CM2:    {breakdown.cm2:.4f}")
    print(f"STRAT:  {breakdown.strat:.4f}")
    print(f"CLICKS: {breakdown.clicks}")
    print(f"\ncompute_reward(...) = {reward:.6f}")
    print(f"Expected            = {expected_reward:.6f}")

    assert np.isclose(reward, expected_reward, atol=1e-10), "compute_reward does not equal weighted sum of components"
    print("✓ Reward equals α·GMV + β·CM2 + γ·STRAT + δ·CLICKS")


def main() -> None:
    """Run all Chapter 5 validation checks."""
    config = SimulatorConfig(seed=2025)
    rng = np.random.default_rng(config.seed)

    print("\n" + "=" * 80)
    print("CHAPTER 5 VALIDATION: Relevance, Features, and Reward")
    print("=" * 80)

    _validate_base_score_decomposition(config, rng)
    _validate_features_and_standardization(config, rng)
    _validate_reward_aggregation(config, rng)

    print("\n" + "=" * 80)
    print("ALL CHAPTER 5 VALIDATION CHECKS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()

