"""
Chapter 6 Lab Solutions — Complete Runnable Code

Author: Vlad Prytula

This module implements all lab exercises from Chapter 6: Discrete Template Bandits,
demonstrating the seamless integration of contextual bandit theory and production-quality code.

Solutions included:
- Exercise 6.1: Cosine Similarity Properties
- Exercise 6.2: Ridge Regression Equivalence
- Exercise 6.3: TS vs LinUCB Posterior Equivalence
- Exercise 6.4: ε-Greedy Baseline Implementation
- Exercise 6.5: Cholesky-Based Thompson Sampling
- Exercise 6.6: Category Diversity Template (No Improvement Case)
- Exercise 6.6b: When Diversity Actually Helps
- Lab 6.1: Simple-Feature Failure Reproduction
- Lab 6.2: Rich-Feature Retry
- Lab 6.3: Hyperparameter Sensitivity
- Lab 6.4: Exploration Dynamics Visualization
- Lab 6.5: Multi-Seed Robustness
- Exercise 6.7: Hierarchical Templates
- Exercise 6.9: Query-Conditional Templates

Usage:
    python -m scripts.ch06.lab_solutions [--exercise N] [--all]

    --exercise N: Run only exercise N (e.g., '6.1', 'lab6.1')
    --all: Run all exercises sequentially
    (default): Run interactive menu
"""

from scripts.ch06.lab_solutions.exercises import (
    exercise_6_1_cosine_properties,
    exercise_6_2_ridge_regression,
    exercise_6_3_ts_linucb_equivalence,
    exercise_6_4_epsilon_greedy,
    exercise_6_5_cholesky_ts,
    exercise_6_6_diversity_template,
    exercise_6_6b_diversity_when_helpful,
    exercise_6_7_hierarchical_templates,
    exercise_6_9_query_conditional_templates,
)

from scripts.ch06.lab_solutions.labs import (
    lab_6_1_simple_feature_baseline,
    lab_6_2_rich_feature_improvement,
    lab_6_3_hyperparameter_sensitivity,
    lab_6_4_exploration_dynamics,
    lab_6_5_multi_seed_robustness,
)

__all__ = [
    # Theory exercises
    "exercise_6_1_cosine_properties",
    "exercise_6_2_ridge_regression",
    "exercise_6_3_ts_linucb_equivalence",
    # Implementation exercises
    "exercise_6_4_epsilon_greedy",
    "exercise_6_5_cholesky_ts",
    "exercise_6_6_diversity_template",
    "exercise_6_6b_diversity_when_helpful",
    # Experimental labs
    "lab_6_1_simple_feature_baseline",
    "lab_6_2_rich_feature_improvement",
    "lab_6_3_hyperparameter_sensitivity",
    "lab_6_4_exploration_dynamics",
    "lab_6_5_multi_seed_robustness",
    # Advanced exercises
    "exercise_6_7_hierarchical_templates",
    "exercise_6_9_query_conditional_templates",
]
