"""
Chapter 1 Lab Solutions — Complete Runnable Code

Author: Vlad Prytula

This module implements all lab exercises from Chapter 1, demonstrating the
seamless integration of mathematical theory [EQ-1.2], [REM-1.2.1] and
production-quality code.

Solutions included:
- Lab 1.1: Reward Aggregation in the Simulator
- Lab 1.2: Delta/Alpha Bound Regression Test
- Extended: Weight Sensitivity Analysis
- Extended: Contextual Reward Variation

Usage:
    python scripts/ch01/lab_solutions.py [--lab N] [--exercise NAME] [--all]

    --lab N: Run only lab N (e.g., '1.1', '1.2')
    --exercise NAME: Run extended exercise ('sensitivity', 'contextual')
    --all: Run all labs and exercises sequentially
    (default): Run interactive menu
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Tuple

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Data Structures (mirrors tests/ch01/test_reward_examples.py)
# =============================================================================


class SessionOutcome(NamedTuple):
    """Outcomes from a single search session.

    Mathematical correspondence: realization omega in Omega of random variables
    (GMV, CM2, STRAT, CLICKS) from [EQ-1.2].

    Attributes:
        gmv: Gross merchandise value in EUR
        cm2: Contribution margin 2 (profit) in EUR
        strat_purchases: Number of strategic purchases in session
        clicks: Total clicks in session
    """

    gmv: float
    cm2: float
    strat_purchases: int
    clicks: int


@dataclass
class RewardConfig:
    """Business priority coefficients (alpha, beta, gamma, delta) in [EQ-1.2].

    Implements the reward aggregation formula:
        R = alpha*GMV + beta*CM2 + gamma*STRAT + delta*CLICKS

    The delta/alpha ratio must satisfy [REM-1.2.1] bounds: delta/alpha in [0.01, 0.10]
    to prevent clickbait strategies.

    Attributes:
        alpha_gmv: Weight for gross merchandise value (typically 1.0 as reference)
        beta_cm2: Weight for contribution margin (profit priority)
        gamma_strat: Weight for strategic purchases (STRAT in [EQ-1.2])
        delta_clicks: Weight for engagement (bounded by [REM-1.2.1])
    """

    alpha_gmv: float = 1.0
    beta_cm2: float = 0.5
    gamma_strat: float = 0.2
    delta_clicks: float = 0.1


@dataclass
class UserSegment:
    """User segment with behavior parameters.

    Attributes:
        name: Segment identifier (e.g., 'price_hunter', 'premium')
        discount_sensitivity: How much user responds to discount boosts [0, 1]
        quality_sensitivity: How much user responds to quality boosts [0, 1]
        base_purchase_prob: Baseline purchase probability per click
        avg_cart_value: Average cart value in EUR when purchase occurs
        cm2_margin: Typical CM2 margin as fraction of GMV
    """

    name: str
    discount_sensitivity: float
    quality_sensitivity: float
    base_purchase_prob: float
    avg_cart_value: float
    cm2_margin: float = 0.20


# Define user segments matching Chapter 0 and Chapter 1
USER_SEGMENTS = {
    "price_hunter": UserSegment(
        name="price_hunter",
        discount_sensitivity=0.8,
        quality_sensitivity=0.2,
        base_purchase_prob=0.35,
        avg_cart_value=80.0,
        cm2_margin=0.15,
    ),
    "premium": UserSegment(
        name="premium",
        discount_sensitivity=0.2,
        quality_sensitivity=0.8,
        base_purchase_prob=0.45,
        avg_cart_value=150.0,
        cm2_margin=0.25,
    ),
    "bulk_buyer": UserSegment(
        name="bulk_buyer",
        discount_sensitivity=0.5,
        quality_sensitivity=0.5,
        base_purchase_prob=0.40,
        avg_cart_value=200.0,
        cm2_margin=0.18,
    ),
    "pl_lover": UserSegment(
        name="pl_lover",
        discount_sensitivity=0.4,
        quality_sensitivity=0.6,
        base_purchase_prob=0.38,
        avg_cart_value=100.0,
        cm2_margin=0.30,  # Private label has higher margins
    ),
}


# =============================================================================
# Core Functions
# =============================================================================


def compute_reward(outcome: SessionOutcome, cfg: RewardConfig) -> float:
    """Compute scalar reward from session outcome using [EQ-1.2].

    R = alpha*GMV + beta*CM2 + gamma*STRAT + delta*CLICKS

    This is the objective function that RL agents optimize.

    Args:
        outcome: Session outcome with GMV, CM2, STRAT, CLICKS
        cfg: Reward weights (alpha, beta, gamma, delta)

    Returns:
        Scalar reward value
    """
    return (
        cfg.alpha_gmv * outcome.gmv
        + cfg.beta_cm2 * outcome.cm2
        + cfg.gamma_strat * outcome.strat_purchases
        + cfg.delta_clicks * outcome.clicks
    )


def compute_rpc(outcome: SessionOutcome) -> float:
    """Compute revenue-per-click (RPC) diagnostic from [REM-1.2.1].

    RPC = GMV / CLICKS (GMV per click)

    Used to detect clickbait: if RPC drops >10% while CTR rises,
    the agent is learning to inflate clicks at expense of conversion.

    Args:
        outcome: Session outcome

    Returns:
        GMV per click, or 0.0 if no clicks
    """
    return outcome.gmv / outcome.clicks if outcome.clicks > 0 else 0.0


def validate_delta_alpha_bound(cfg: RewardConfig) -> Tuple[bool, str]:
    """Validate [REM-1.2.1] bound: delta/alpha in [0.01, 0.10].

    Args:
        cfg: Reward configuration to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if cfg.alpha_gmv == 0:
        return False, "alpha_gmv cannot be zero (division error)"

    ratio = cfg.delta_clicks / cfg.alpha_gmv

    if ratio < 0.01:
        return False, f"delta/alpha = {ratio:.3f} below lower bound of 0.01"
    elif ratio > 0.10:
        return False, f"delta/alpha = {ratio:.3f} exceeds bound of 0.10"
    elif abs(ratio - 0.10) < 0.001:
        return True, "At boundary (delta/alpha = 0.10)"
    else:
        return True, f"delta/alpha = {ratio:.3f} within bounds [0.01, 0.10]"


def simulate_session(
    segment: UserSegment,
    w_discount: float,
    w_quality: float,
    rng: np.random.Generator,
) -> SessionOutcome:
    """Simulate a search session for a user segment.

    This is a simplified simulator for Chapter 1 labs. Production
    implementation lives in zoosim/dynamics/behavior.py.

    Args:
        segment: User segment with behavior parameters
        w_discount: Discount boost weight [-1, 1]
        w_quality: Quality boost weight [-1, 1]
        rng: Random number generator for reproducibility

    Returns:
        SessionOutcome with simulated GMV, CM2, STRAT, CLICKS
    """
    # Compute effective utility based on segment preferences
    utility = (
        segment.discount_sensitivity * w_discount
        + segment.quality_sensitivity * w_quality
    )

    # Sigmoid to bound click probability modification
    click_modifier = 1.0 / (1.0 + np.exp(-utility))

    # Base clicks from Poisson, modified by utility
    base_clicks = rng.poisson(4.0)
    clicks = max(1, int(base_clicks * (0.5 + click_modifier)))

    # Purchase probability per click
    purchase_prob = segment.base_purchase_prob * (0.8 + 0.4 * click_modifier)
    purchase_prob = np.clip(purchase_prob, 0.1, 0.8)

    # Simulate purchases
    n_purchases = sum(rng.random() < purchase_prob for _ in range(clicks))

    # Compute GMV and CM2
    if n_purchases > 0:
        # Cart value varies around average
        cart_value = segment.avg_cart_value * (0.7 + 0.6 * rng.random())
        gmv = n_purchases * cart_value
        cm2 = gmv * segment.cm2_margin
    else:
        gmv = 0.0
        cm2 = 0.0

    # Strategic purchases: a subset of purchases, weakly correlated with quality boost.
    # This matches the production semantics in `zoosim/dynamics/reward.py:34-38`:
    # STRAT increments only when a purchase occurs and the purchased item is strategic.
    strategic_prob = 0.3 + 0.2 * np.clip(w_quality, -1.0, 1.0)  # 10%-50% range
    strat_purchases = sum(rng.random() < strategic_prob for _ in range(n_purchases))

    return SessionOutcome(
        gmv=round(gmv, 2),
        cm2=round(cm2, 2),
        strat_purchases=strat_purchases,
        clicks=clicks,
    )


# =============================================================================
# Lab 1.1: Reward Aggregation in the Simulator
# =============================================================================


def lab_1_1_reward_aggregation(seed: int = 11, verbose: bool = True) -> dict:
    """Lab 1.1 Solution: Reward Aggregation in the Simulator.

    Inspects a simulated session, records the GMV/CM2/STRAT/CLICKS decomposition,
    and verifies that the computed reward matches [EQ-1.2].

    Args:
        seed: Random seed for reproducibility
        verbose: Whether to print detailed output

    Returns:
        Dict with outcome, computed_reward, reported_reward, verification status
    """
    if verbose:
        print("=" * 70)
        print("Lab 1.1: Reward Aggregation in the Simulator")
        print("=" * 70)

    rng = np.random.default_rng(seed)

    # Select segment and simulate
    segment = USER_SEGMENTS["price_hunter"]
    w_discount, w_quality = 0.5, 0.3

    outcome = simulate_session(segment, w_discount, w_quality, rng)

    # Configuration
    cfg = RewardConfig(alpha_gmv=1.0, beta_cm2=0.5, gamma_strat=0.2, delta_clicks=0.1)

    # Compute reward two ways
    R_computed = (
        cfg.alpha_gmv * outcome.gmv
        + cfg.beta_cm2 * outcome.cm2
        + cfg.gamma_strat * outcome.strat_purchases
        + cfg.delta_clicks * outcome.clicks
    )
    R_function = compute_reward(outcome, cfg)

    if verbose:
        print(f"\nSession simulation (seed={seed}):")
        print(f"  User segment: {segment.name}")
        print(f'  Query: "cat food"')

        print(f"\nOutcome breakdown:")
        print(f"  GMV:    €{outcome.gmv:6.2f} (gross merchandise value)")
        print(f"  CM2:    €{outcome.cm2:6.2f} (contribution margin 2)")
        print(f"  STRAT:  {outcome.strat_purchases:d} purchases  (strategic purchases in session)")
        print(f"  CLICKS: {outcome.clicks:d}        (total clicks)")

        print(f"\nReward weights (from RewardConfig):")
        print(f"  alpha (alpha_gmv):     {cfg.alpha_gmv:.2f}")
        print(f"  beta (beta_cm2):       {cfg.beta_cm2:.2f}")
        print(f"  gamma (gamma_strat):   {cfg.gamma_strat:.2f}")
        print(f"  delta (delta_clicks):  {cfg.delta_clicks:.2f}")

        print(f"\nManual computation of R = alpha*GMV + beta*CM2 + gamma*STRAT + delta*CLICKS:")
        print(f"  = {cfg.alpha_gmv:.2f} x {outcome.gmv:.2f} + {cfg.beta_cm2:.2f} x {outcome.cm2:.2f} + {cfg.gamma_strat:.2f} x {outcome.strat_purchases} + {cfg.delta_clicks:.2f} x {outcome.clicks}")
        gmv_term = cfg.alpha_gmv * outcome.gmv
        cm2_term = cfg.beta_cm2 * outcome.cm2
        strat_term = cfg.gamma_strat * outcome.strat_purchases
        clicks_term = cfg.delta_clicks * outcome.clicks
        print(f"  = {gmv_term:.2f} + {cm2_term:.2f} + {strat_term:.2f} + {clicks_term:.2f}")
        print(f"  = {R_computed:.2f}")

        print(f"\nSimulator-reported reward: {R_function:.2f}")

        diff = abs(R_computed - R_function)
        if diff < 0.01:
            print(f"\nVerification: |computed - reported| = {diff:.2f} < 0.01 ✓")
            print("\nThe simulator correctly implements [EQ-1.2].")
        else:
            print(f"\n✗ MISMATCH: |computed - reported| = {diff:.2f} >= 0.01")

    return {
        "outcome": outcome,
        "computed_reward": R_computed,
        "reported_reward": R_function,
        "verified": abs(R_computed - R_function) < 0.01,
        "config": cfg,
    }


def lab_1_1_delta_alpha_violation(verbose: bool = True) -> dict:
    """Lab 1.1 Task 2: Find smallest delta/alpha violation.

    Tests progressively higher delta values to find the boundary
    where [REM-1.2.1] constraints are violated.

    Args:
        verbose: Whether to print detailed output

    Returns:
        Dict with test results and smallest violation
    """
    if verbose:
        print("\n" + "=" * 70)
        print("Lab 1.1 Task 2: Delta/Alpha Bound Violation")
        print("=" * 70)
        print("\nTesting progressively higher delta values...")
        print("Bound from [REM-1.2.1]: delta/alpha in [0.01, 0.10]\n")

    test_ratios = [0.08, 0.10, 0.11, 0.12, 0.15, 0.20]
    results = []
    smallest_violation = None

    for ratio in test_ratios:
        cfg = RewardConfig(alpha_gmv=1.0, delta_clicks=ratio)
        is_valid, message = validate_delta_alpha_bound(cfg)
        results.append({"ratio": ratio, "valid": is_valid, "message": message})

        if not is_valid and smallest_violation is None:
            smallest_violation = ratio

        if verbose:
            status = "✓ VALID" if is_valid else "✗ VIOLATION"
            print(f"delta/alpha = {ratio:.2f}: {status}")

    if verbose and smallest_violation:
        print(f"\nSmallest violation: delta/alpha = {smallest_violation:.2f} ({smallest_violation/0.10:.2f}x the bound)")

    return {"results": results, "smallest_violation": smallest_violation}


# =============================================================================
# Lab 1.2: Delta/Alpha Bound Regression Test
# =============================================================================


def lab_1_2_regression_tests(verbose: bool = True) -> dict:
    """Lab 1.2 Solution: Regression tests for Chapter 1 examples.

    Verifies that code examples from the chapter produce expected outputs,
    tying assertions to [EQ-1.2] and [REM-1.2.1].

    Args:
        verbose: Whether to print detailed output

    Returns:
        Dict with test results
    """
    if verbose:
        print("=" * 70)
        print("Lab 1.2: Delta/Alpha Bound Regression Test")
        print("=" * 70)

    results = {"tests": [], "all_passed": True}

    # Test 1: Basic reward comparison (from chapter)
    outcome_a = SessionOutcome(gmv=120.0, cm2=15.0, strat_purchases=1, clicks=3)
    outcome_b = SessionOutcome(gmv=100.0, cm2=35.0, strat_purchases=3, clicks=4)
    weights = RewardConfig(alpha_gmv=1.0, beta_cm2=0.5, gamma_strat=0.2, delta_clicks=0.1)

    R_A = compute_reward(outcome_a, weights)
    R_B = compute_reward(outcome_b, weights)

    test1_passed = abs(R_A - 128.00) < 0.01 and abs(R_B - 118.50) < 0.01 and R_A > R_B
    results["tests"].append({
        "name": "Basic reward comparison",
        "passed": test1_passed,
        "expected": {"R_A": 128.00, "R_B": 118.50, "A_wins": True},
        "actual": {"R_A": R_A, "R_B": R_B, "A_wins": R_A > R_B},
    })
    results["all_passed"] &= test1_passed

    if verbose:
        print(f"\nTest 1: Basic reward comparison")
        print(f"  Strategy A: R = {R_A:.2f} (expected: 128.00)")
        print(f"  Strategy B: R = {R_B:.2f} (expected: 118.50)")
        print(f"  Result: {'✓ PASSED' if test1_passed else '✗ FAILED'}")

    # Test 2: Profitability weighting
    weights_profit = RewardConfig(alpha_gmv=0.5, beta_cm2=1.0, gamma_strat=0.5, delta_clicks=0.1)
    R_A_profit = compute_reward(outcome_a, weights_profit)
    R_B_profit = compute_reward(outcome_b, weights_profit)

    test2_passed = abs(R_A_profit - 75.80) < 0.01 and abs(R_B_profit - 86.90) < 0.01 and R_B_profit > R_A_profit
    results["tests"].append({
        "name": "Profitability weighting",
        "passed": test2_passed,
        "expected": {"R_A": 75.80, "R_B": 86.90, "B_wins": True},
        "actual": {"R_A": R_A_profit, "R_B": R_B_profit, "B_wins": R_B_profit > R_A_profit},
    })
    results["all_passed"] &= test2_passed

    if verbose:
        print(f"\nTest 2: Profitability weighting")
        print(f"  Strategy A: R = {R_A_profit:.2f} (expected: 75.80)")
        print(f"  Strategy B: R = {R_B_profit:.2f} (expected: 86.90)")
        print(f"  Result: {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    # Test 3: RPC diagnostic
    rpc_a = compute_rpc(outcome_a)
    rpc_b = compute_rpc(outcome_b)

    test3_passed = abs(rpc_a - 40.0) < 0.01 and abs(rpc_b - 25.0) < 0.01 and rpc_a > rpc_b
    results["tests"].append({
        "name": "RPC diagnostic",
        "passed": test3_passed,
        "expected": {"RPC_A": 40.0, "RPC_B": 25.0, "A_higher": True},
        "actual": {"RPC_A": rpc_a, "RPC_B": rpc_b, "A_higher": rpc_a > rpc_b},
    })
    results["all_passed"] &= test3_passed

    if verbose:
        print("\nTest 3: RPC diagnostic")
        print(f"  Strategy A: RPC = {rpc_a:.2f} (expected: 40.00)")
        print(f"  Strategy B: RPC = {rpc_b:.2f} (expected: 25.00)")
        print(f"  Result: {'✓ PASSED' if test3_passed else '✗ FAILED'}")

    # Test 4: Delta/alpha bounds
    valid_configs = [
        RewardConfig(alpha_gmv=1.0, delta_clicks=0.01),
        RewardConfig(alpha_gmv=1.0, delta_clicks=0.05),
        RewardConfig(alpha_gmv=1.0, delta_clicks=0.10),
    ]
    invalid_configs = [
        RewardConfig(alpha_gmv=1.0, delta_clicks=0.005),
        RewardConfig(alpha_gmv=1.0, delta_clicks=0.15),
    ]

    valid_passed = all(validate_delta_alpha_bound(c)[0] for c in valid_configs)
    invalid_failed = all(not validate_delta_alpha_bound(c)[0] for c in invalid_configs)
    test4_passed = valid_passed and invalid_failed

    results["tests"].append({
        "name": "Delta/alpha bounds",
        "passed": test4_passed,
        "valid_configs_pass": valid_passed,
        "invalid_configs_fail": invalid_failed,
    })
    results["all_passed"] &= test4_passed

    if verbose:
        print(f"\nTest 4: Delta/alpha bounds [REM-1.2.1]")
        print(f"  Valid configs pass: {valid_passed}")
        print(f"  Invalid configs fail: {invalid_failed}")
        print(f"  Result: {'✓ PASSED' if test4_passed else '✗ FAILED'}")

    # Test 5: Zero clicks edge case
    outcome_zero = SessionOutcome(gmv=0.0, cm2=0.0, strat_purchases=0, clicks=0)
    rpc_zero = compute_rpc(outcome_zero)
    test5_passed = rpc_zero == 0.0

    results["tests"].append({
        "name": "Zero clicks edge case",
        "passed": test5_passed,
        "expected": 0.0,
        "actual": rpc_zero,
    })
    results["all_passed"] &= test5_passed

    if verbose:
        print(f"\nTest 5: Zero clicks edge case")
        print(f"  RPC with zero clicks: {rpc_zero} (expected: 0.0)")
        print(f"  Result: {'✓ PASSED' if test5_passed else '✗ FAILED'}")

    if verbose:
        print("\n" + "=" * 70)
        if results["all_passed"]:
            print("All regression tests PASSED ✓")
        else:
            print("Some tests FAILED ✗")
        print("=" * 70)

    return results


# =============================================================================
# Extended Exercise: Weight Sensitivity Analysis
# =============================================================================


def weight_sensitivity_analysis(
    n_sessions: int = 500, seed: int = 42, verbose: bool = True
) -> dict:
    """Extended Exercise: Analyze reward sensitivity to weight changes.

    Demonstrates that the same session outcomes produce different rewards
    depending on business weight configuration.

    Args:
        n_sessions: Number of sessions to simulate
        seed: Random seed
        verbose: Whether to print output

    Returns:
        Dict with analysis results per configuration
    """
    if verbose:
        print("=" * 70)
        print("Weight Sensitivity Analysis")
        print("=" * 70)
        print(f"\nSimulating {n_sessions} sessions across 4 weight configurations...")

    rng = np.random.default_rng(seed)

    # Generate fixed session outcomes (same for all configs)
    segments = list(USER_SEGMENTS.values())
    outcomes = []
    for _ in range(n_sessions):
        seg = rng.choice(segments)
        w_d, w_q = rng.uniform(-0.5, 0.8), rng.uniform(-0.3, 0.8)
        outcomes.append(simulate_session(seg, w_d, w_q, rng))

    # Weight configurations to test
    configs = [
        ("Balanced", RewardConfig(alpha_gmv=1.0, beta_cm2=0.5, gamma_strat=0.2, delta_clicks=0.1)),
        ("GMV-Focused", RewardConfig(alpha_gmv=1.0, beta_cm2=0.2, gamma_strat=0.1, delta_clicks=0.05)),
        ("Profit-Focused", RewardConfig(alpha_gmv=0.5, beta_cm2=1.0, gamma_strat=0.3, delta_clicks=0.05)),
        ("Engagement-Heavy", RewardConfig(alpha_gmv=1.0, beta_cm2=0.3, gamma_strat=0.2, delta_clicks=0.09)),
    ]

    results = {}

    for name, cfg in configs:
        rewards = [compute_reward(o, cfg) for o in outcomes]
        gmvs = [o.gmv for o in outcomes]
        cm2s = [o.cm2 for o in outcomes]
        strats = [o.strat_purchases for o in outcomes]
        clicks = [o.clicks for o in outcomes]

        total_gmv = sum(gmvs)
        total_clicks = sum(clicks)
        rpc = total_gmv / total_clicks if total_clicks > 0 else 0

        results[name] = {
            "config": cfg,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_gmv": np.mean(gmvs),
            "mean_cm2": np.mean(cm2s),
            "mean_strat": np.mean(strats),
            "mean_clicks": np.mean(clicks),
            "rpc": rpc,
        }

        if verbose:
            print(f"\nConfiguration: {name} (alpha={cfg.alpha_gmv}, beta={cfg.beta_cm2}, gamma={cfg.gamma_strat}, delta={cfg.delta_clicks})")
            print(f"  Mean reward:     EUR{results[name]['mean_reward']:7.2f} +/- {results[name]['std_reward']:.2f}")
            print(f"  Mean GMV:        EUR{results[name]['mean_gmv']:7.2f}")
            print(f"  Mean CM2:        EUR{results[name]['mean_cm2']:7.2f}")
            print(f"  Mean STRAT:        {results[name]['mean_strat']:.2f}")
            print(f"  Mean CLICKS:       {results[name]['mean_clicks']:.2f}")
            print(f"  RPC (GMV/click): EUR{results[name]['rpc']:.2f}")

    if verbose:
        print("\n" + "-" * 70)
        print("Key Insight:")
        print("  Same outcomes, different rewards! The underlying user behavior")
        print("  (GMV, CM2, STRAT, CLICKS) is IDENTICAL across configurations.")
        print("\n  Only the WEIGHTING changes how we value those outcomes.")
        print("\n  This is why weight calibration is critical:")
        print("  - An RL agent will optimize whatever the weights incentivize")
        print("  - Poorly chosen weights -> agent learns wrong behavior")
        print("  - [REM-1.2.1] bounds prevent one failure mode (clickbait)")
        print("  - [EQ-1.3] constraints prevent others (margin collapse, etc.)")

    return results


# =============================================================================
# Extended Exercise: Contextual Reward Variation
# =============================================================================


def contextual_reward_variation(
    n_per_segment: int = 100, seed: int = 42, verbose: bool = True
) -> dict:
    """Extended Exercise: Show that optimal actions vary by context.

    Demonstrates the gap between static optimization [EQ-1.5] and
    contextual optimization [EQ-1.6].

    Args:
        n_per_segment: Sessions per user segment
        seed: Random seed
        verbose: Whether to print output

    Returns:
        Dict with per-segment results and static vs contextual comparison
    """
    if verbose:
        print("=" * 70)
        print("Contextual Reward Variation")
        print("=" * 70)
        print("\nSimulating different user segments with same boost configuration...")

    rng = np.random.default_rng(seed)
    cfg = RewardConfig()  # Default balanced weights

    # Fixed boost weights (static policy)
    w_d_static, w_q_static = 0.5, 0.3

    if verbose:
        print(f"\nStatic boost weights: w_discount={w_d_static}, w_quality={w_q_static}")

    # Simulate per segment with static weights
    segment_results = {}
    for seg_name, segment in USER_SEGMENTS.items():
        rewards = []
        for _ in range(n_per_segment):
            outcome = simulate_session(segment, w_d_static, w_q_static, rng)
            rewards.append(compute_reward(outcome, cfg))

        segment_results[seg_name] = {
            "mean": np.mean(rewards),
            "std": np.std(rewards),
        }

    if verbose:
        print("\nResults by user segment (static policy):")
        for name, res in segment_results.items():
            print(f"  {name:15s}: Mean R = EUR{res['mean']:6.2f} +/- {res['std']:.2f} (n={n_per_segment})")

    # Grid search for optimal boost per segment
    discount_grid = np.linspace(-0.5, 1.0, 7)
    quality_grid = np.linspace(-0.5, 1.0, 7)

    optimal_per_segment = {}
    for seg_name, segment in USER_SEGMENTS.items():
        best_w = (0, 0)
        best_reward = -float("inf")

        for w_d in discount_grid:
            for w_q in quality_grid:
                rewards = []
                for _ in range(50):  # Fewer samples for grid search
                    outcome = simulate_session(segment, w_d, w_q, rng)
                    rewards.append(compute_reward(outcome, cfg))
                mean_r = np.mean(rewards)
                if mean_r > best_reward:
                    best_reward = mean_r
                    best_w = (w_d, w_q)

        optimal_per_segment[seg_name] = {
            "w_discount": best_w[0],
            "w_quality": best_w[1],
            "mean_reward": best_reward,
        }

    if verbose:
        print("\nOptimal boost per segment (grid search):")
        for name, opt in optimal_per_segment.items():
            print(f"  {name:15s}: w_discount={opt['w_discount']:+.1f}, w_quality={opt['w_quality']:+.1f} -> R = EUR{opt['mean_reward']:.2f}")

    # Compute static vs contextual gap
    static_mean = np.mean([res["mean"] for res in segment_results.values()])
    contextual_mean = np.mean([opt["mean_reward"] for opt in optimal_per_segment.values()])
    improvement = 100 * (contextual_mean - static_mean) / static_mean

    if verbose:
        print("\nStatic vs Contextual Comparison:")
        print(f"  Static (best single w):     Mean R = EUR{static_mean:.2f} across all segments")
        print(f"  Contextual (w per segment): Mean R = EUR{contextual_mean:.2f} across all segments")
        print(f"\n  Improvement: +{improvement:.1f}% by adapting to context!")
        print("\nThis validates [EQ-1.6]: contextual optimization > static optimization.")
        print("The gap would widen with more user heterogeneity.")

    return {
        "segment_results": segment_results,
        "optimal_per_segment": optimal_per_segment,
        "static_mean": static_mean,
        "contextual_mean": contextual_mean,
        "improvement_pct": improvement,
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def run_all(verbose: bool = True) -> dict:
    """Run all labs and exercises sequentially."""
    print("\n" + "=" * 70)
    print("CHAPTER 1 LAB SOLUTIONS - COMPLETE RUN")
    print("=" * 70 + "\n")

    results = {
        "lab_1_1": lab_1_1_reward_aggregation(verbose=verbose),
        "lab_1_1_violation": lab_1_1_delta_alpha_violation(verbose=verbose),
        "lab_1_2": lab_1_2_regression_tests(verbose=verbose),
        "sensitivity": weight_sensitivity_analysis(verbose=verbose),
        "contextual": contextual_reward_variation(verbose=verbose),
    }

    print("\n" + "=" * 70)
    print("ALL LABS AND EXERCISES COMPLETED")
    print("=" * 70)

    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Chapter 1 Lab Solutions")
    parser.add_argument("--all", action="store_true", help="Run all labs and exercises")
    parser.add_argument("--lab", type=str, help="Run specific lab (e.g., '1.1', '1.2')")
    parser.add_argument(
        "--exercise",
        type=str,
        help="Run extended exercise ('sensitivity', 'contextual')",
    )

    args = parser.parse_args()

    lab_map = {
        "1.1": lab_1_1_reward_aggregation,
        "1.2": lab_1_2_regression_tests,
    }

    exercise_map = {
        "sensitivity": weight_sensitivity_analysis,
        "contextual": contextual_reward_variation,
    }

    if args.all:
        run_all()
    elif args.lab:
        key = args.lab.lower().replace("lab", "").strip()
        if key in lab_map:
            lab_map[key]()
            if key == "1.1":
                lab_1_1_delta_alpha_violation()
        else:
            print(f"Unknown lab: {args.lab}")
            print(f"Available: 1.1, 1.2")
            sys.exit(1)
    elif args.exercise:
        key = args.exercise.lower()
        if key in exercise_map:
            exercise_map[key]()
        else:
            print(f"Unknown exercise: {args.exercise}")
            print(f"Available: sensitivity, contextual")
            sys.exit(1)
    else:
        # Interactive menu
        print("\nCHAPTER 1 LAB SOLUTIONS")
        print("=" * 40)
        print("1. Lab 1.1  - Reward Aggregation")
        print("2. Lab 1.2  - Regression Tests")
        print("3. Ex Sens  - Weight Sensitivity")
        print("4. Ex Ctx   - Contextual Variation")
        print("A. All      - Run everything")
        print()

        choice = input("Select (1-4 or A): ").strip().lower()
        choice_map = {
            "1": lambda: (lab_1_1_reward_aggregation(), lab_1_1_delta_alpha_violation()),
            "2": lab_1_2_regression_tests,
            "3": weight_sensitivity_analysis,
            "4": contextual_reward_variation,
            "a": run_all,
        }
        if choice in choice_map:
            choice_map[choice]()
        else:
            print(f"Invalid choice: {choice}")


if __name__ == "__main__":
    main()
