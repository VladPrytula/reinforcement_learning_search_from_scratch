"""Boost feature engineering utilities.

This module provides canonical feature extraction functions for:
1. Product-level features (compute_features) - Ch5 relevance scoring
2. Context-level features (compute_context_*) - Ch6+ RL agents

All Ch6-8 scripts should import from this module to ensure consistency.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import Product
from zoosim.world.queries import Query
from zoosim.world.users import User


# ---------------------------------------------------------------------------
# Product-level features (Ch5 relevance scoring)
# ---------------------------------------------------------------------------


def compute_features(*, user: User, query: Query, product: Product, config: SimulatorConfig) -> List[float]:
    """Compute per-product boost features for relevance scoring.

    Used by Ch5 base ranker to compute item-level boosts.
    Returns 10-dimensional feature vector.

    Mathematical correspondence: [EQ-5.4] feature extraction φ(u, q, p)

    Args:
        user: User instance
        query: Query instance
        product: Product instance
        config: Simulator configuration

    Returns:
        10-dim feature vector for boost scoring
    """
    pers = float(torch.dot(user.theta_emb, product.embedding))
    specificity = config.queries.specificity.get(query.query_type, 0.5)

    features = [
        product.cm2,
        product.discount,
        float(product.is_pl),
        pers,
        product.bestseller,
        product.price,
        product.cm2 if product.category == "litter" else 0.0,
        product.discount * user.theta_price,
        float(product.is_pl) * user.theta_pl,
        specificity * product.bestseller,
    ]
    return features


def standardize_features(feature_matrix: Sequence[Sequence[float]], *, config: SimulatorConfig) -> List[List[float]]:
    """Standardize feature matrix (z-score normalization).

    Args:
        feature_matrix: Feature matrix to standardize
        config: Simulator configuration (unused, kept for API consistency)

    Returns:
        Standardized feature matrix (mean=0, std=1 per column)
    """
    array = np.asarray(feature_matrix, dtype=float)
    means = array.mean(axis=0)
    stds = array.std(axis=0)
    stds[stds == 0] = 1.0
    normalized = (array - means) / stds
    return normalized.tolist()


# ---------------------------------------------------------------------------
# Context-level features (Ch6+ RL agents)
# ---------------------------------------------------------------------------


def compute_context_features_simple(
    user: User, query: Query, config: SimulatorConfig
) -> np.ndarray:
    """Simple context features (segment + query type only).

    This is deliberately impoverished to demonstrate failure mode in Ch6 §6.4.
    Used to show that insufficient features lead to -30% GMV vs static policies.

    Mathematical correspondence: [EQ-6.8] φ_simple(x)

    Args:
        user: User instance
        query: Query instance
        config: Simulator configuration

    Returns:
        7-dim feature vector (4 segment + 3 query type one-hots)
    """
    segs = config.users.segments
    qtypes = config.queries.query_types

    seg_vec = np.zeros(len(segs), dtype=float)
    seg_vec[segs.index(user.segment)] = 1.0

    q_vec = np.zeros(len(qtypes), dtype=float)
    q_vec[qtypes.index(query.query_type)] = 1.0

    return np.concatenate([seg_vec, q_vec]).astype(np.float64)


def compute_context_features_rich(
    user: User,
    query: Query,
    catalog: List[Product],
    base_scores: np.ndarray,
    config: SimulatorConfig,
) -> np.ndarray:
    """Rich context features (oracle user latents + catalog aggregates).

    Computes aggregates over base ranker top-k to avoid circularity.
    Uses true simulator latents (oracle access) for theta_price, theta_pl.
    This is the "rich features" fix in Ch6 §6.7 that recovers +27% GMV.

    Mathematical correspondence: [EQ-6.12] φ_rich(x)

    Args:
        user: User instance (uses oracle theta_price, theta_pl)
        query: Query instance
        catalog: Full product catalog
        base_scores: Base relevance scores (before template boosts)
        config: Simulator configuration

    Returns:
        17-dim standardized feature vector:
            - seg_vec (4): user segment one-hot
            - q_vec (3): query type one-hot
            - user_prefs (2): theta_price, theta_pl (oracle)
            - aggregates (8): avg_price, std_price, avg_cm2, avg_discount,
                             frac_pl, frac_strategic, avg_bestseller, avg_relevance
    """
    # 1. One-hot encodings
    seg_vec = np.zeros(len(config.users.segments), dtype=float)
    seg_vec[config.users.segments.index(user.segment)] = 1.0

    q_vec = np.zeros(len(config.queries.query_types), dtype=float)
    q_vec[config.queries.query_types.index(query.query_type)] = 1.0

    # 2. User preferences (oracle access to true simulator latents)
    theta_price = float(user.theta_price)
    theta_pl = float(user.theta_pl)

    # 3. Compute top-k products from base ranker
    k = min(config.top_k, len(catalog))
    top_idx = np.argsort(-np.asarray(base_scores, dtype=float))[:k]
    top_products = [catalog[i] for i in top_idx]

    # 4. Aggregate product features over base top-k
    prices = np.array([p.price for p in top_products], dtype=float)
    avg_price = float(prices.mean())
    std_price = float(prices.std())
    avg_cm2 = float(np.mean([p.cm2 for p in top_products]))
    avg_discount = float(np.mean([p.discount for p in top_products]))
    frac_pl = float(np.mean([1.0 if p.is_pl else 0.0 for p in top_products]))
    frac_strategic = float(
        np.mean([1.0 if p.strategic_flag else 0.0 for p in top_products])
    )
    avg_bestseller = float(np.mean([p.bestseller for p in top_products]))
    avg_relevance = float(np.mean(np.asarray(base_scores, dtype=float)[top_idx]))

    # 5. Concatenate raw features
    raw = np.concatenate(
        [
            seg_vec,  # 4 dims
            q_vec,  # 3 dims
            [theta_price, theta_pl],  # 2 dims
            [
                avg_price,
                std_price,
                avg_cm2,
                avg_discount,
                frac_pl,
                frac_strategic,
                avg_bestseller,
                avg_relevance,
            ],  # 8 dims
        ]
    )  # Total: 17 dims

    # 6. Standardize (z-score normalization) for aggregates.
    # Segments / query types / user prefs are already well-scaled.
    means = np.array(
        [
            0, 0, 0, 0,  # segments (binary, do not shift)
            0, 0, 0,  # query types (binary, do not shift)
            0, 0,  # theta_price, theta_pl (already normalized)
            30.0, 15.0, 0.3, 0.1, 0.5, 0.2, 300.0, 5.0,  # aggregates
        ],
        dtype=float,
    )
    stds = np.array(
        [
            1, 1, 1, 1,  # segments
            1, 1, 1,  # query types
            1, 1,  # theta_* (already normalized)
            20.0, 10.0, 0.2, 0.1, 0.3, 0.2, 200.0, 2.0,  # aggregates
        ],
        dtype=float,
    )

    standardized = (raw - means) / stds
    return standardized.astype(np.float64)


def compute_context_features_rich_estimated(
    user: User,
    query: Query,
    catalog: List[Product],
    base_scores: np.ndarray,
    config: SimulatorConfig,
) -> np.ndarray:
    """Rich context features with estimated (not oracle) user latents.

    Simulates production setting where user preferences are only approximately
    known. Applies shrinkage + coarse quantization to theta_price, theta_pl.

    Feature structure matches compute_context_features_rich (17 dims).
    Used in labs/experiments for more realistic evaluations.

    Args:
        user: User instance (latents will be shrunk + quantized)
        query: Query instance
        catalog: Full product catalog
        base_scores: Base relevance scores (before template boosts)
        config: Simulator configuration

    Returns:
        17-dim standardized feature vector (same structure as rich, but with
        estimated theta_price_est, theta_pl_est instead of oracle values)
    """
    # Derive approximate preferences from simulator latents
    # 1) Shrink toward zero (under-confident model)
    # 2) Quantize onto coarse 0.5 grid
    # 3) Clip to plausible range
    raw_price = float(user.theta_price)
    raw_pl = float(user.theta_pl)

    shrunk_price = 0.7 * raw_price
    shrunk_pl = 0.7 * raw_pl

    theta_price_est = float(
        np.clip(np.round(shrunk_price * 2.0) / 2.0, -3.0, 3.0)
    )
    theta_pl_est = float(
        np.clip(np.round(shrunk_pl * 2.0) / 2.0, -3.0, 3.0)
    )

    # Reuse rich feature construction but swap in estimated prefs
    seg_vec = np.zeros(len(config.users.segments), dtype=float)
    seg_vec[config.users.segments.index(user.segment)] = 1.0

    q_vec = np.zeros(len(config.queries.query_types), dtype=float)
    q_vec[config.queries.query_types.index(query.query_type)] = 1.0

    k = min(config.top_k, len(catalog))
    top_idx = np.argsort(-np.asarray(base_scores, dtype=float))[:k]
    top_products = [catalog[i] for i in top_idx]

    prices = np.array([p.price for p in top_products], dtype=float)
    avg_price = float(prices.mean())
    std_price = float(prices.std())
    avg_cm2 = float(np.mean([p.cm2 for p in top_products]))
    avg_discount = float(np.mean([p.discount for p in top_products]))
    frac_pl = float(np.mean([1.0 if p.is_pl else 0.0 for p in top_products]))
    frac_strategic = float(
        np.mean([1.0 if p.strategic_flag else 0.0 for p in top_products])
    )
    avg_bestseller = float(np.mean([p.bestseller for p in top_products]))
    avg_relevance = float(np.mean(np.asarray(base_scores, dtype=float)[top_idx]))

    raw = np.concatenate(
        [
            seg_vec,
            q_vec,
            [theta_price_est, theta_pl_est],  # Estimated, not oracle
            [
                avg_price,
                std_price,
                avg_cm2,
                avg_discount,
                frac_pl,
                frac_strategic,
                avg_bestseller,
                avg_relevance,
            ],
        ]
    )

    # Same standardization as rich features
    means = np.array(
        [
            0, 0, 0, 0,
            0, 0, 0,
            0, 0,
            30.0, 15.0, 0.3, 0.1, 0.5, 0.2, 300.0, 5.0,
        ],
        dtype=float,
    )
    stds = np.array(
        [
            1, 1, 1, 1,
            1, 1, 1,
            1, 1,
            20.0, 10.0, 0.2, 0.1, 0.3, 0.2, 200.0, 2.0,
        ],
        dtype=float,
    )

    standardized = (raw - means) / stds
    return standardized.astype(np.float64)
