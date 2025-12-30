"""Fairness and exposure metrics for provider-group parity.

This module implements exposure-based fairness metrics for Chapter 14:
- Position-weighted exposure computation using simulator's own position bias
- Provider-group aggregation (private-label, category, strategic)
- Gap metrics (L1, KL divergence) between actual and target shares
- Band checking for constraint satisfaction

Mathematical basis:
    - [DEF-14.2.1]: Pareto dominance for outcome vectors
    - [EQ-14.8]: Exposure band constraints
    - Appendix E: Connection to vector-reward MORL

References:
    - Chapter 14: Multi-Objective RL and Fairness at Scale
    - [@miettinen:multi_objective_optimization:1999]: Pareto fronts
    - [@altman:constrained_mdps:1999]: Constrained MDPs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import Product


# ---------------------------------------------------------------------------
# Position-weighted exposure
# ---------------------------------------------------------------------------


def position_weights(
    cfg: SimulatorConfig,
    query_type: str,
    k: Optional[int] = None,
) -> NDArray[np.float64]:
    """Compute position-based examination weights for a query type.

    Uses the simulator's own position bias model from BehaviorConfig.pos_bias.
    Higher weights = more exposure (more likely to be examined by user).

    Mathematical basis: Position bias models from Chapter 2 (PBM/DBN).

    Args:
        cfg: Simulator configuration containing pos_bias curves
        query_type: One of 'category', 'brand', 'generic'
        k: Number of positions (default: cfg.top_k)

    Returns:
        Array of shape (k,) with examination probabilities per position.
        Sum is NOT normalized to 1; use exposure_share_by_group for shares.

    Example:
        >>> cfg = SimulatorConfig()
        >>> w = position_weights(cfg, 'category', k=5)
        >>> w  # array([1.2, 0.9, 0.7, 0.5, 0.3])
    """
    if k is None:
        k = cfg.top_k

    pos_bias = cfg.behavior.pos_bias.get(query_type)
    if pos_bias is None:
        # Fall back to generic if query_type not found
        pos_bias = cfg.behavior.pos_bias.get("generic", [1.0] * k)

    # Pad or truncate to length k
    if len(pos_bias) < k:
        # Extend with the last value (or a small default)
        last_val = pos_bias[-1] if pos_bias else 0.1
        pos_bias = list(pos_bias) + [last_val] * (k - len(pos_bias))

    return np.array(pos_bias[:k], dtype=np.float64)


# ---------------------------------------------------------------------------
# Provider-group assignment
# ---------------------------------------------------------------------------

GroupScheme = Literal["pl", "category", "strategic"]


def group_key(product: Product, scheme: GroupScheme) -> str:
    """Assign a product to a provider group based on the grouping scheme.

    Schemes align with Chapter 14 terminology contract (Section 14.1.2):
    - 'pl': Private-label vs national-brand (from Product.is_pl)
    - 'category': Product category (from Product.category)
    - 'strategic': Strategic vs non-strategic (from Product.strategic_flag)

    Args:
        product: A Product instance from the catalog
        scheme: Grouping scheme to use

    Returns:
        String key identifying the group (e.g., 'pl', 'non_pl', 'dog_food')

    Raises:
        ValueError: If scheme is not recognized
    """
    if scheme == "pl":
        return "pl" if product.is_pl else "non_pl"
    elif scheme == "category":
        return product.category
    elif scheme == "strategic":
        return "strategic" if product.strategic_flag else "non_strategic"
    else:
        raise ValueError(f"Unknown grouping scheme: {scheme}")


def get_group_keys(scheme: GroupScheme, cfg: SimulatorConfig) -> List[str]:
    """Get all possible group keys for a scheme.

    Args:
        scheme: Grouping scheme
        cfg: Simulator configuration (for category list)

    Returns:
        List of all group keys for the scheme
    """
    if scheme == "pl":
        return ["pl", "non_pl"]
    elif scheme == "category":
        return list(cfg.catalog.categories)
    elif scheme == "strategic":
        return ["strategic", "non_strategic"]
    else:
        raise ValueError(f"Unknown grouping scheme: {scheme}")


# ---------------------------------------------------------------------------
# Exposure metrics
# ---------------------------------------------------------------------------


def exposure_by_group(
    ranking_topk: List[int],
    catalog: Dict[int, Product],
    weights: NDArray[np.float64],
    scheme: GroupScheme,
) -> Dict[str, float]:
    """Compute total exposure for each provider group in a ranking.

    Exposure is the sum of position weights for items in each group.

    Mathematical basis:
        exposure_g = sum_{i: product_i in group g} w_i
    where w_i is the position weight at rank i.

    Args:
        ranking_topk: List of product IDs in ranked order (top-k)
        catalog: Mapping from product_id to Product
        weights: Position weights of shape (k,) from position_weights()
        scheme: Grouping scheme for provider groups

    Returns:
        Dictionary mapping group_key -> total exposure (unnormalized)

    Example:
        >>> exposure_by_group([0, 1, 2], catalog, weights, 'pl')
        {'pl': 1.5, 'non_pl': 0.7}
    """
    exposure: Dict[str, float] = {}
    k = min(len(ranking_topk), len(weights))

    for i in range(k):
        pid = ranking_topk[i]
        product = catalog.get(pid)
        if product is None:
            continue
        gkey = group_key(product, scheme)
        exposure[gkey] = exposure.get(gkey, 0.0) + float(weights[i])

    return exposure


def exposure_share_by_group(
    ranking_topk: List[int],
    catalog: Dict[int, Product],
    weights: NDArray[np.float64],
    scheme: GroupScheme,
) -> Dict[str, float]:
    """Compute exposure shares (normalized to sum to 1) for each group.

    This is the metric used in fairness constraints: shares should be
    within target bands for fair exposure.

    Args:
        ranking_topk: List of product IDs in ranked order
        catalog: Mapping from product_id to Product
        weights: Position weights from position_weights()
        scheme: Grouping scheme

    Returns:
        Dictionary mapping group_key -> exposure share in [0, 1]
        All shares sum to 1.0 (within floating point tolerance)

    Example:
        >>> shares = exposure_share_by_group([0, 1, 2], catalog, weights, 'pl')
        >>> sum(shares.values())  # 1.0
    """
    exposure = exposure_by_group(ranking_topk, catalog, weights, scheme)
    total = sum(exposure.values())

    if total <= 0.0:
        return {k: 0.0 for k in exposure}

    return {k: v / total for k, v in exposure.items()}


# ---------------------------------------------------------------------------
# Gap metrics
# ---------------------------------------------------------------------------


def l1_gap(shares: Dict[str, float], targets: Dict[str, float]) -> float:
    """Compute L1 (total variation) gap between shares and targets.

    L1 gap = sum_g |share_g - target_g|

    This measures total deviation from target allocation. Lower is better.
    A gap of 0 means perfect allocation.

    Args:
        shares: Actual exposure shares by group
        targets: Target shares by group (should sum to 1)

    Returns:
        L1 gap in [0, 2] (0 = perfect match, 2 = completely opposite)

    Example:
        >>> l1_gap({'pl': 0.4, 'non_pl': 0.6}, {'pl': 0.5, 'non_pl': 0.5})
        0.2
    """
    gap = 0.0
    all_keys = set(shares.keys()) | set(targets.keys())

    for key in all_keys:
        s = shares.get(key, 0.0)
        t = targets.get(key, 0.0)
        gap += abs(s - t)

    return gap


def kl_divergence(
    shares: Dict[str, float],
    targets: Dict[str, float],
    epsilon: float = 1e-10,
) -> float:
    """Compute KL divergence from shares to targets.

    KL(shares || targets) = sum_g share_g * log(share_g / target_g)

    This is an asymmetric measure: penalizes shares that deviate from
    targets, especially when shares are high but targets are low.

    Args:
        shares: Actual exposure shares (the distribution we have)
        targets: Target shares (the reference distribution)
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence in [0, inf). 0 = identical distributions.

    Note:
        KL divergence is undefined if target_g = 0 but share_g > 0.
        We add epsilon to avoid this numerical issue.
    """
    kl = 0.0
    for key, s in shares.items():
        t = targets.get(key, epsilon)
        # Add epsilon for numerical stability
        s_safe = max(s, epsilon)
        t_safe = max(t, epsilon)
        if s_safe > epsilon:
            kl += s_safe * np.log(s_safe / t_safe)

    return float(kl)


# ---------------------------------------------------------------------------
# Band checking
# ---------------------------------------------------------------------------


@dataclass
class BandCheckResult:
    """Result of checking whether shares fall within target bands."""

    within_band: Dict[str, bool]  # Per-group: is share within band?
    deviations: Dict[str, float]  # Per-group: signed deviation from band
    max_deviation: float  # Maximum absolute deviation
    all_satisfied: bool  # Are all groups within their bands?


def within_band(
    shares: Dict[str, float],
    targets: Dict[str, float],
    band: float,
) -> BandCheckResult:
    """Check if exposure shares fall within target bands.

    For each group g, the share must satisfy:
        target_g - band <= share_g <= target_g + band

    This implements the exposure band constraints [EQ-14.8] from Chapter 14.

    Args:
        shares: Actual exposure shares by group
        targets: Target shares (center of bands) by group
        band: Half-width of the acceptable band (e.g., 0.05 for +/- 5%)

    Returns:
        BandCheckResult with per-group status and maximum deviation

    Example:
        >>> result = within_band(
        ...     {'pl': 0.42, 'non_pl': 0.58},
        ...     {'pl': 0.40, 'non_pl': 0.60},
        ...     band=0.05
        ... )
        >>> result.all_satisfied  # True (both within +/- 5%)
    """
    within: Dict[str, bool] = {}
    deviations: Dict[str, float] = {}

    all_keys = set(shares.keys()) | set(targets.keys())

    for key in all_keys:
        s = shares.get(key, 0.0)
        t = targets.get(key, 0.0)

        lo = t - band
        hi = t + band

        # Deviation: negative if below band, positive if above
        if s < lo:
            dev = s - lo
        elif s > hi:
            dev = s - hi
        else:
            dev = 0.0

        within[key] = (lo <= s <= hi)
        deviations[key] = dev

    max_dev = max(abs(d) for d in deviations.values()) if deviations else 0.0
    all_ok = all(within.values())

    return BandCheckResult(
        within_band=within,
        deviations=deviations,
        max_deviation=max_dev,
        all_satisfied=all_ok,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics for batch evaluation
# ---------------------------------------------------------------------------


@dataclass
class FairnessMetrics:
    """Aggregated fairness metrics for a batch of rankings."""

    scheme: GroupScheme
    mean_shares: Dict[str, float]  # Average exposure share per group
    l1_gap: float  # L1 distance from targets
    kl_divergence: float  # KL divergence from targets
    band_satisfaction_rate: float  # Fraction of rankings within band
    max_deviation: float  # Worst-case deviation from band


def compute_batch_fairness(
    rankings: List[List[int]],
    catalog: Dict[int, Product],
    cfg: SimulatorConfig,
    scheme: GroupScheme,
    targets: Dict[str, float],
    band: float,
    query_types: Optional[List[str]] = None,
) -> FairnessMetrics:
    """Compute fairness metrics over a batch of rankings.

    Args:
        rankings: List of rankings (each ranking is a list of product IDs)
        catalog: Product catalog
        cfg: Simulator configuration (for position bias)
        scheme: Provider-group scheme
        targets: Target exposure shares
        band: Acceptable band width
        query_types: Query type for each ranking (uses 'generic' if None)

    Returns:
        FairnessMetrics with aggregated statistics
    """
    if not rankings:
        return FairnessMetrics(
            scheme=scheme,
            mean_shares={k: 0.0 for k in targets},
            l1_gap=0.0,
            kl_divergence=0.0,
            band_satisfaction_rate=1.0,
            max_deviation=0.0,
        )

    n = len(rankings)
    if query_types is None:
        query_types = ["generic"] * n

    # Accumulate shares
    share_sums: Dict[str, float] = {k: 0.0 for k in targets}
    band_satisfied = 0
    max_dev = 0.0

    for i, ranking in enumerate(rankings):
        qt = query_types[i] if i < len(query_types) else "generic"
        weights = position_weights(cfg, qt, k=len(ranking))
        shares = exposure_share_by_group(ranking, catalog, weights, scheme)

        # Accumulate
        for k in share_sums:
            share_sums[k] += shares.get(k, 0.0)

        # Check band
        result = within_band(shares, targets, band)
        if result.all_satisfied:
            band_satisfied += 1
        max_dev = max(max_dev, result.max_deviation)

    # Average shares
    mean_shares = {k: v / n for k, v in share_sums.items()}

    return FairnessMetrics(
        scheme=scheme,
        mean_shares=mean_shares,
        l1_gap=l1_gap(mean_shares, targets),
        kl_divergence=kl_divergence(mean_shares, targets),
        band_satisfaction_rate=band_satisfied / n,
        max_deviation=max_dev,
    )


# ---------------------------------------------------------------------------
# Utility parity helpers (segment-level reporting)
# ---------------------------------------------------------------------------


def utility_by_segment(
    rewards: NDArray[np.float64],
    segments: List[str],
) -> Dict[str, float]:
    """Aggregate rewards by user segment.

    This is for utility parity reporting: checking that different user
    segments receive comparable utility (reward).

    Args:
        rewards: Array of per-episode rewards
        segments: User segment for each episode

    Returns:
        Dictionary mapping segment -> mean reward
    """
    segment_rewards: Dict[str, List[float]] = {}

    for r, seg in zip(rewards, segments):
        if seg not in segment_rewards:
            segment_rewards[seg] = []
        segment_rewards[seg].append(float(r))

    return {
        seg: np.mean(vals) if vals else 0.0
        for seg, vals in segment_rewards.items()
    }


def utility_parity_gap(utilities: Dict[str, float]) -> float:
    """Compute the gap between highest and lowest segment utilities.

    A gap of 0 means perfect utility parity across segments.

    Args:
        utilities: Mean utility per segment

    Returns:
        max(utilities) - min(utilities)
    """
    if not utilities:
        return 0.0

    vals = list(utilities.values())
    return max(vals) - min(vals)
