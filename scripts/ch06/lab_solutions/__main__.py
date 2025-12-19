"""
Chapter 6 Lab Solutions — Main Entry Point

Author: Vlad Prytula

Usage:
    python -m scripts.ch06.lab_solutions [--exercise N] [--all]

    --exercise N: Run only exercise N (e.g., '6.1', 'lab6.1', 'lab6.2')
    --all: Run all exercises sequentially
    (default): Run interactive menu
"""

from __future__ import annotations

import argparse
import sys

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


def run_all_exercises(verbose: bool = True) -> dict:
    """Run all exercises sequentially."""
    print("\n" + "=" * 70)
    print("CHAPTER 6 LAB SOLUTIONS — COMPLETE RUN")
    print("=" * 70 + "\n")

    results = {}

    # Theory exercises
    print("\n### THEORY EXERCISES ###\n")
    results["ex_6_1"] = exercise_6_1_cosine_properties(verbose=verbose)
    results["ex_6_2"] = exercise_6_2_ridge_regression(verbose=verbose)
    results["ex_6_3"] = exercise_6_3_ts_linucb_equivalence(verbose=verbose)

    # Implementation exercises
    print("\n### IMPLEMENTATION EXERCISES ###\n")
    results["ex_6_4"] = exercise_6_4_epsilon_greedy(n_episodes=10000, verbose=verbose)
    results["ex_6_5"] = exercise_6_5_cholesky_ts(dims=[10, 50, 100], verbose=verbose)
    results["ex_6_6"] = exercise_6_6_diversity_template(n_episodes=10000, verbose=verbose)

    # Experimental labs
    print("\n### EXPERIMENTAL LABS ###\n")
    results["lab_6_1"] = lab_6_1_simple_feature_baseline(n_static=1000, n_bandit=10000, verbose=verbose)
    results["lab_6_2"] = lab_6_2_rich_feature_improvement(n_static=1000, n_bandit=10000, verbose=verbose)
    results["lab_6_3"] = lab_6_3_hyperparameter_sensitivity(n_episodes=5000, verbose=verbose)
    results["lab_6_4"] = lab_6_4_exploration_dynamics(n_episodes=5000, verbose=verbose)
    results["lab_6_5"] = lab_6_5_multi_seed_robustness(n_seeds=5, n_episodes=5000, verbose=verbose)

    # Advanced exercises
    print("\n### ADVANCED EXERCISES ###\n")
    results["ex_6_7"] = exercise_6_7_hierarchical_templates(n_episodes=20000, verbose=verbose)
    results["ex_6_9"] = exercise_6_9_query_conditional_templates(n_episodes=15000, verbose=verbose)

    print("\n" + "=" * 70)
    print("ALL EXERCISES COMPLETED")
    print("=" * 70)

    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Chapter 6 Lab Solutions")
    parser.add_argument("--all", action="store_true", help="Run all exercises")
    parser.add_argument("--exercise", type=str,
                        help="Run specific exercise (e.g., '6.1', 'lab6.1', 'lab6.2')")

    args = parser.parse_args()

    exercise_map = {
        # Theory exercises
        "6.1": exercise_6_1_cosine_properties,
        "61": exercise_6_1_cosine_properties,
        "6.2": exercise_6_2_ridge_regression,
        "62": exercise_6_2_ridge_regression,
        "6.3": exercise_6_3_ts_linucb_equivalence,
        "63": exercise_6_3_ts_linucb_equivalence,
        # Implementation exercises
        "6.4": exercise_6_4_epsilon_greedy,
        "64": exercise_6_4_epsilon_greedy,
        "6.5": exercise_6_5_cholesky_ts,
        "65": exercise_6_5_cholesky_ts,
        "6.6": exercise_6_6_diversity_template,
        "66": exercise_6_6_diversity_template,
        "6.6b": exercise_6_6b_diversity_when_helpful,
        "66b": exercise_6_6b_diversity_when_helpful,
        # Labs
        "lab6.1": lab_6_1_simple_feature_baseline,
        "lab61": lab_6_1_simple_feature_baseline,
        "lab6.2": lab_6_2_rich_feature_improvement,
        "lab62": lab_6_2_rich_feature_improvement,
        "lab6.3": lab_6_3_hyperparameter_sensitivity,
        "lab63": lab_6_3_hyperparameter_sensitivity,
        "lab6.4": lab_6_4_exploration_dynamics,
        "lab64": lab_6_4_exploration_dynamics,
        "lab6.5": lab_6_5_multi_seed_robustness,
        "lab65": lab_6_5_multi_seed_robustness,
        # Advanced
        "6.7": exercise_6_7_hierarchical_templates,
        "67": exercise_6_7_hierarchical_templates,
        "6.9": exercise_6_9_query_conditional_templates,
        "69": exercise_6_9_query_conditional_templates,
    }

    if args.all:
        run_all_exercises()
    elif args.exercise:
        key = args.exercise.lower().replace("_", "").replace("-", "")
        if key in exercise_map:
            exercise_map[key]()
        else:
            print(f"Unknown exercise: {args.exercise}")
            print(f"Available: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, lab6.1, lab6.2, lab6.3, lab6.4, lab6.5, 6.7, 6.9")
            sys.exit(1)
    else:
        # Interactive menu
        print("\nCHAPTER 6 LAB SOLUTIONS")
        print("=" * 50)
        print("\nTheory Exercises:")
        print("  1.  Ex 6.1   - Cosine Similarity Properties")
        print("  2.  Ex 6.2   - Ridge Regression Equivalence")
        print("  3.  Ex 6.3   - TS vs LinUCB Equivalence")
        print("\nImplementation Exercises:")
        print("  4.  Ex 6.4   - ε-Greedy Baseline")
        print("  5.  Ex 6.5   - Cholesky Thompson Sampling")
        print("  6.  Ex 6.6   - Category Diversity (No Improvement Case)")
        print("  6b. Ex 6.6b  - When Diversity Actually Helps")
        print("\nExperimental Labs:")
        print("  7.  Lab 6.1  - Simple-Feature Failure")
        print("  8.  Lab 6.2  - Rich-Feature Retry")
        print("  9.  Lab 6.3  - Hyperparameter Sensitivity")
        print(" 10.  Lab 6.4  - Exploration Dynamics")
        print(" 11.  Lab 6.5  - Multi-Seed Robustness")
        print("\nAdvanced Exercises:")
        print(" 12.  Ex 6.7   - Hierarchical Templates")
        print(" 13.  Ex 6.9   - Query-Conditional Templates")
        print("\n  A.  All      - Run everything")
        print()

        choice = input("Select (1-13 or A): ").strip().lower()
        choice_map = {
            "1": exercise_6_1_cosine_properties,
            "2": exercise_6_2_ridge_regression,
            "3": exercise_6_3_ts_linucb_equivalence,
            "4": exercise_6_4_epsilon_greedy,
            "5": exercise_6_5_cholesky_ts,
            "6": exercise_6_6_diversity_template,
            "6b": exercise_6_6b_diversity_when_helpful,
            "7": lab_6_1_simple_feature_baseline,
            "8": lab_6_2_rich_feature_improvement,
            "9": lab_6_3_hyperparameter_sensitivity,
            "10": lab_6_4_exploration_dynamics,
            "11": lab_6_5_multi_seed_robustness,
            "12": exercise_6_7_hierarchical_templates,
            "13": exercise_6_9_query_conditional_templates,
            "a": run_all_exercises,
        }
        if choice in choice_map:
            choice_map[choice]()
        else:
            print(f"Invalid choice: {choice}")


if __name__ == "__main__":
    main()
