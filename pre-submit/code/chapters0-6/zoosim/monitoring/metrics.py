"""Standard business and stability metrics for e-commerce ranking.

Defines the canonical implementations for:
- Business Metrics: CTR, CVR, GMV, CM2.
- Stability Metrics: Delta-Rank@k (churn).
- System Metrics: Latency (placeholder).

References:
    - Chapter 10: Robustness & Guardrails
    - [DEF-10.4]: Delta-Rank@k Stability
    - [EQ-10.11]: CM2 definition (GMV - COGS - Logistics - Marketing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray


def compute_ctr(clicks: int, impressions: int) -> float:
    """Compute Click-Through Rate.

    Args:
        clicks: Number of clicks
        impressions: Number of views/impressions

    Returns:
        CTR in [0, 1]
    """
    if impressions <= 0:
        return 0.0
    return clicks / impressions


def compute_cvr(conversions: int, clicks: int) -> float:
    """Compute Conversion Rate (per click).

    Args:
        conversions: Number of purchases
        clicks: Number of clicks

    Returns:
        CVR in [0, 1]
    """
    if clicks <= 0:
        return 0.0
    return conversions / clicks


def compute_gmv(prices: NDArray[np.float64], conversions: NDArray[np.int_]) -> float:
    """Compute Gross Merchandise Value.

    Args:
        prices: Array of product prices
        conversions: Boolean/Int array of conversions (1=bought, 0=not)

    Returns:
        Total GMV sum
    """
    return float(np.sum(prices * conversions))


def compute_cm2(
    gmv: float, 
    costs: float, 
    logistics_cost: float = 0.0, 
    marketing_cost: float = 0.0
) -> float:
    """Compute Contribution Margin 2 (CM2).

    CM2 = GMV - COGS - Logistics - Marketing (Variable Costs).
    Often approximated as a % of GMV or exact margin if COGS known.

    Args:
        gmv: Gross Merchandise Value
        costs: Cost of Goods Sold (COGS)
        logistics_cost: Shipping/Handling
        marketing_cost: Direct marketing spend (e.g. CPA)

    Returns:
        CM2 value
    """
    return gmv - costs - logistics_cost - marketing_cost


def compute_delta_rank_at_k(
    ranking_prev: List[int], 
    ranking_curr: List[int], 
    k: int = 10
) -> float:
    """Compute Delta-Rank@k (stability/churn metric).

    Defined as the fraction of items in the top-k that changed
    compared to the previous ranking. 
    
    Formula: 1 - (|TopK(t) ∩ TopK(t-1)| / k)

    Args:
        ranking_prev: List of item IDs from previous step
        ranking_curr: List of item IDs from current step
        k: Depth to check

    Returns:
        Churn rate in [0, 1]. 0 = identical top-k set, 1 = completely different.
    """
    k = min(k, len(ranking_prev), len(ranking_curr))
    if k == 0:
        return 0.0
    
    set_prev = set(ranking_prev[:k])
    set_curr = set(ranking_curr[:k])
    
    intersection = len(set_prev.intersection(set_curr))
    churn = 1.0 - (intersection / k)
    return churn


@dataclass
class RankingMetrics:
    """Container for batch evaluation metrics."""
    ctr: float
    cvr: float
    gmv: float
    cm2: float
    avg_position_bias: float = 1.0
    diversity_score: float = 0.0

    @classmethod
    def empty(cls) -> "RankingMetrics":
        return cls(0.0, 0.0, 0.0, 0.0)
