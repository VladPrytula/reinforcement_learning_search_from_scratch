"""Chapter 2 scripts: Probability, Measure, and Click Models.

Lab solutions demonstrating the integration of measure-theoretic foundations
with simulator code paths, including:

- Lab 2.1: Segment Mix Sanity Check (SLLN verification)
- Lab 2.2: Query Measure and Base Score Integration
- Extended: PBM and DBN Click Model Verification
- Extended: IPS Estimator Verification
"""

from scripts.ch02.lab_solutions import (
    lab_2_1_segment_mix_sanity_check,
    lab_2_1_multi_seed_analysis,
    lab_2_1_degenerate_distribution,
    lab_2_2_base_score_integration,
    lab_2_2_user_sampling_verification,
    lab_2_2_score_histogram,
    extended_click_model_verification,
    extended_ips_verification,
)

__all__ = [
    "lab_2_1_segment_mix_sanity_check",
    "lab_2_1_multi_seed_analysis",
    "lab_2_1_degenerate_distribution",
    "lab_2_2_base_score_integration",
    "lab_2_2_user_sampling_verification",
    "lab_2_2_score_histogram",
    "extended_click_model_verification",
    "extended_ips_verification",
]
