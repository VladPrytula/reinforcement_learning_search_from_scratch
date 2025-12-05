#!/usr/bin/env python3
"""Chapter 5 demonstration and validation script.

Demonstrates the three core components from Chapter 5:
1. Base relevance (hybrid semantic + lexical)
2. Feature engineering (10-dimensional feature vector)
3. Reward aggregation (multi-objective scalarization)

Usage:
    python scripts/ch05_demo.py
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch

from zoosim.core.config import SimulatorConfig
from zoosim.dynamics.reward import RewardBreakdown, compute_reward
from zoosim.ranking.features import compute_features, standardize_features
from zoosim.ranking.relevance import base_score, batch_base_scores
from zoosim.world.catalog import Product, generate_catalog
from zoosim.world.queries import Query, sample_query
from zoosim.world.users import User, sample_user


def demo_relevance_model(config: SimulatorConfig, rng: np.random.Generator) -> None:
    """Demonstrate base relevance computation (Section 5.2)."""
    print("=" * 80)
    print("PART 1: BASE RELEVANCE MODEL (Section 5.2)")
    print("=" * 80)

    # Generate world
    catalog = generate_catalog(cfg=config.catalog, rng=rng)
    user = sample_user(config=config, rng=rng)
    query = sample_query(user=user, config=config, rng=rng)

    print(f"\nUser segment: {user.segment}")
    print(f"Query category: '{query.intent_category}' (type: {query.query_type})")

    # Compute base scores for all products
    base_scores = batch_base_scores(query=query, catalog=catalog, config=config, rng=rng)

    # Rank by base score
    ranked_indices = np.argsort(base_scores)[::-1][:20]

    print("\nTop 10 products by base relevance:")
    print(f"{'Rank':<6} {'PID':<8} {'Category':<12} {'Price':<8} {'CM2':<8} {'Score':<8}")
    print("-" * 60)

    for rank, idx in enumerate(ranked_indices[:10], start=1):
        prod = catalog[idx]
        score = base_scores[idx]
        print(
            f"{rank:<6} {prod.product_id:<8} {prod.category:<12} "
            f"${prod.price:<7.2f} ${prod.cm2:<7.2f} {score:<7.3f}"
        )

    # Validation checks
    print("\n[VALIDATION]")
    print(f"✓ Computed {len(base_scores)} base scores")
    print(f"✓ Score range: [{min(base_scores):.3f}, {max(base_scores):.3f}]")
    print(f"✓ Mean score: {np.mean(base_scores):.3f}")

    # Check relevance weights
    assert config.relevance.w_sem == 0.7, "Expected w_sem=0.7"
    assert config.relevance.w_lex == 0.3, "Expected w_lex=0.3"
    print(f"✓ Relevance weights: w_sem={config.relevance.w_sem}, w_lex={config.relevance.w_lex}")


def demo_feature_engineering(config: SimulatorConfig, rng: np.random.Generator) -> None:
    """Demonstrate feature computation and standardization (Section 5.3)."""
    print("\n")
    print("=" * 80)
    print("PART 2: FEATURE ENGINEERING (Section 5.3)")
    print("=" * 80)

    # Generate world
    catalog = generate_catalog(cfg=config.catalog, rng=rng)
    user = sample_user(config=config, rng=rng)
    query = sample_query(user=user, config=config, rng=rng)

    # Compute base scores and get top 20
    base_scores = batch_base_scores(query=query, catalog=catalog, config=config, rng=rng)
    ranked_indices = np.argsort(base_scores)[::-1][:20]

    # Compute raw features for top 20 products
    raw_features = [
        compute_features(user=user, query=query, product=catalog[idx], config=config)
        for idx in ranked_indices
    ]

    # Standardize features
    standardized_features = standardize_features(raw_features, config=config)

    # Display features for top product
    print(f"\nUser segment: {user.segment}")
    print(f"Query category: '{query.intent_category}' (type: {query.query_type})")
    print(f"\nFeatures for top product (Product {catalog[ranked_indices[0]].product_id}):")

    feature_names = [
        "cm2",
        "discount",
        "pl_flag",
        "personalization",
        "bestseller",
        "price",
        "cm2_litter",
        "disc_price_sens",
        "pl_affinity",
        "spec_bestseller",
    ]

    print(f"{'Feature':<20} {'Raw':<12} {'Standardized':<12}")
    print("-" * 50)
    for i, name in enumerate(feature_names):
        raw_val = raw_features[0][i]
        std_val = standardized_features[0][i]
        print(f"{name:<20} {raw_val:<12.3f} {std_val:<12.3f}")

    # Validation checks
    print("\n[VALIDATION]")
    print(f"✓ Feature dimension: {len(raw_features[0])} (expected: 10)")
    assert len(raw_features[0]) == 10, "Feature dimension mismatch"
    assert len(raw_features[0]) == config.action.feature_dim, "Config mismatch"

    # Check standardization properties
    std_array = np.array(standardized_features)
    means = std_array.mean(axis=0)
    stds = std_array.std(axis=0)

    print(f"✓ Standardized means (should be ~0): {means[:3]} ...")
    print(f"✓ Standardized stds (should be ~1 for non-constant features): {stds[:3]} ...")
    assert np.allclose(means, 0, atol=1e-10), "Standardization mean check failed"
    # Check stds for non-constant features only (some features like cm2_litter may be all zeros)
    non_constant_features = stds > 1e-6
    if non_constant_features.any():
        assert np.allclose(stds[non_constant_features], 1, atol=1e-10), "Standardization std check failed"
    print(f"✓ Constant features (std=0): {np.where(~non_constant_features)[0].tolist()}")


def demo_reward_aggregation(config: SimulatorConfig, rng: np.random.Generator) -> None:
    """Demonstrate reward computation (Section 5.5)."""
    print("\n")
    print("=" * 80)
    print("PART 3: REWARD AGGREGATION (Section 5.5)")
    print("=" * 80)

    # Generate world
    catalog = generate_catalog(cfg=config.catalog, rng=rng)

    # Simulate a user session (mock clicks and purchases)
    ranking = list(range(20))  # Top 20 products (0-19)
    clicks = [1, 1, 0, 1, 0, 0, 0, 1, 0, 0] + [0] * 10  # 4 clicks
    buys = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0] + [0] * 10  # 2 purchases

    print("\nSimulated session:")
    print(f"  Ranking: {ranking[:10]} ...")
    print(f"  Clicks: {clicks[:10]} ...")
    print(f"  Buys: {buys[:10]} ...")

    # Compute reward
    reward, breakdown = compute_reward(
        ranking=ranking, clicks=clicks, buys=buys, catalog=catalog, config=config
    )

    # Display breakdown
    print("\nReward breakdown:")
    print(f"  GMV: ${breakdown.gmv:.2f}")
    print(f"  CM2: ${breakdown.cm2:.2f}")
    print(f"  Strategic purchases: {breakdown.strat}")
    print(f"  Clicks: {breakdown.clicks}")

    # Show scalarization
    cfg = config.reward
    gmv_contrib = cfg.alpha_gmv * breakdown.gmv
    cm2_contrib = cfg.beta_cm2 * breakdown.cm2
    strat_contrib = cfg.gamma_strat * breakdown.strat
    click_contrib = cfg.delta_clicks * breakdown.clicks

    print("\nScalarization [EQ-5.7]:")
    print(f"  {cfg.alpha_gmv:.1f} × GMV = {gmv_contrib:.2f}")
    print(f"  {cfg.beta_cm2:.1f} × CM2 = {cm2_contrib:.2f}")
    print(f"  {cfg.gamma_strat:.1f} × Strategic = {strat_contrib:.2f}")
    print(f"  {cfg.delta_clicks:.1f} × Clicks = {click_contrib:.2f}")
    print(f"  {'─' * 30}")
    print(f"  Total reward: {reward:.2f}")

    # Validation checks
    print("\n[VALIDATION]")
    computed_sum = gmv_contrib + cm2_contrib + strat_contrib + click_contrib
    assert math.isclose(reward, computed_sum, abs_tol=1e-6), "Reward computation mismatch"
    print(f"✓ Reward sum verified: {reward:.2f}")

    # Check safety constraint [EQ-5.8]
    ratio = cfg.delta_clicks / cfg.alpha_gmv
    print(f"✓ Engagement safety ratio: {ratio:.3f} (must be in [0.01, 0.10])")
    assert 0.01 <= ratio <= 0.10, f"Safety constraint violated: {ratio}"


def demo_integrated_episode(config: SimulatorConfig, rng: np.random.Generator) -> None:
    """Demonstrate full episode: relevance → features → reward (Section 5.9)."""
    print("\n")
    print("=" * 80)
    print("PART 4: INTEGRATED EPISODE (Section 5.9)")
    print("=" * 80)

    # Generate world
    catalog = generate_catalog(cfg=config.catalog, rng=rng)
    user = sample_user(config=config, rng=rng)
    query = sample_query(user=user, config=config, rng=rng)

    print(f"\nUser segment: {user.segment}")
    print(f"Query category: '{query.intent_category}' (type: {query.query_type})")

    # Step 1: Base relevance
    print("\n[Step 1] Computing base relevance for all products...")
    base_scores = batch_base_scores(query=query, catalog=catalog, config=config, rng=rng)
    ranked_by_relevance = np.argsort(base_scores)[::-1][:20]
    print(f"✓ Ranked {len(catalog)} products, selected top 20")

    # Step 2: Feature extraction
    print("\n[Step 2] Extracting features for top 20 products...")
    raw_features = [
        compute_features(user=user, query=query, product=catalog[idx], config=config)
        for idx in ranked_by_relevance
    ]
    features = standardize_features(raw_features, config=config)
    print(f"✓ Computed {len(features)} feature vectors (dim={len(features[0])})")

    # Step 3: Agent action (placeholder: random boosts)
    print("\n[Step 3] Agent applies boosts (random baseline)...")
    boosts = rng.uniform(-0.1, 0.1, size=20)
    adjusted_scores = [base_scores[idx] + boosts[i] for i, idx in enumerate(ranked_by_relevance)]
    final_ranking = ranked_by_relevance[np.argsort(adjusted_scores)[::-1]]
    print(f"✓ Applied random boosts in [-0.1, +0.1]")

    # Step 4: User interaction (mock for now)
    print("\n[Step 4] User interaction (mock clicks/buys)...")
    clicks = [1, 1, 0, 1, 0] + [0] * 15  # 3 clicks
    buys = [1, 0, 0, 1, 0] + [0] * 15  # 2 purchases
    print(f"✓ Clicks: {sum(clicks)}, Purchases: {sum(buys)}")

    # Step 5: Reward computation
    print("\n[Step 5] Computing reward...")
    reward, breakdown = compute_reward(
        ranking=final_ranking, clicks=clicks, buys=buys, catalog=catalog, config=config
    )

    print("\nFinal reward breakdown:")
    print(f"  GMV: ${breakdown.gmv:.2f}")
    print(f"  CM2: ${breakdown.cm2:.2f}")
    print(f"  Strategic: {breakdown.strat}")
    print(f"  Clicks: {breakdown.clicks}")
    print(f"  ──────────────────")
    print(f"  Total: {reward:.2f}")

    print("\n[VALIDATION]")
    print("✓ Complete episode executed: relevance → features → boosts → reward")


def main() -> None:
    """Run all Chapter 5 demonstrations."""
    print("\n" + "=" * 80)
    print("CHAPTER 5 DEMONSTRATION: Relevance, Features, and Reward")
    print("=" * 80)

    # Setup
    config = SimulatorConfig(seed=2025)
    rng = np.random.default_rng(config.seed)

    # Run demonstrations
    demo_relevance_model(config, rng)
    demo_feature_engineering(config, rng)
    demo_reward_aggregation(config, rng)
    demo_integrated_episode(config, rng)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ All demonstrations completed successfully!")
    print("\nKey takeaways:")
    print("1. Base relevance: Hybrid semantic + lexical matching [EQ-5.3]")
    print("2. Features: 10-dimensional state representation with standardization [EQ-5.5]")
    print("3. Reward: Multi-objective scalarization with safety constraint [EQ-5.7]")
    print("\nNext steps:")
    print("- Chapter 6: Implement RL agents (LinUCB, Thompson Sampling)")
    print("- Chapter 7: Continuous action spaces (Q-learning for boosts)")
    print("- Chapter 8: Add constraints (CM2 floors, exposure guarantees)")
    print("\nFor exercises and labs, see:")
    print("  docs/book/drafts/ch05/exercises_labs.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
