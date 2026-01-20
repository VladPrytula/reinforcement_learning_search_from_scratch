"""Evaluation + OPE scaffolding for Part III (Chapters 9-10, 14).

Modules:
    - ope: Off-Policy Evaluation (Chapter 9)
    - fairness: Exposure fairness metrics (Chapter 14)
"""

from zoosim.evaluation.fairness import (
    BandCheckResult,
    FairnessMetrics,
    GroupScheme,
    compute_batch_fairness,
    exposure_by_group,
    exposure_share_by_group,
    get_group_keys,
    group_key,
    kl_divergence,
    l1_gap,
    position_weights,
    utility_by_segment,
    utility_parity_gap,
    within_band,
)

__all__: list[str] = [
    # Fairness metrics (Chapter 14)
    "GroupScheme",
    "BandCheckResult",
    "FairnessMetrics",
    "position_weights",
    "group_key",
    "get_group_keys",
    "exposure_by_group",
    "exposure_share_by_group",
    "l1_gap",
    "kl_divergence",
    "within_band",
    "compute_batch_fairness",
    "utility_by_segment",
    "utility_parity_gap",
]
