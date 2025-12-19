"""
Chapter 3 Scripts — Bellman Operators and Convergence Theory

This module implements lab exercises for Chapter 3, covering:
- Bellman operator contraction properties
- Value iteration convergence analysis
- Regret theory fundamentals
- Lagrangian methods for constrained optimization
"""

from scripts.ch03.lab_solutions import (
    lab_3_1_contraction_ratio_tracker,
    lab_3_2_value_iteration_profiling,
    extended_perturbation_sensitivity,
    extended_discount_factor_analysis,
    extended_banach_convergence_verification,
    run_all_labs,
)

__all__ = [
    "lab_3_1_contraction_ratio_tracker",
    "lab_3_2_value_iteration_profiling",
    "extended_perturbation_sensitivity",
    "extended_discount_factor_analysis",
    "extended_banach_convergence_verification",
    "run_all_labs",
]
