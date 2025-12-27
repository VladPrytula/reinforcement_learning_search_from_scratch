"""Test code examples from Chapter 1, Section 1.2 (Reward Function).

Validates that all code snippets compile and produce expected output.
"""

from dataclasses import dataclass
from typing import NamedTuple
import numpy as np


class SessionOutcome(NamedTuple):
    """Outcomes from a single search session.

    Mathematical correspondence: realization omega in Omega of random variables
    (GMV, CM2, STRAT, CLICKS).
    """
    gmv: float          # Gross merchandise value (€)
    cm2: float          # Contribution margin 2 (€)
    strat_purchases: int # Number of strategic purchases in session
    clicks: int         # Total clicks


@dataclass
class BusinessWeights:
    """Business priority coefficients (alpha, beta, gamma, delta) in #EQ-1.2."""
    alpha_gmv: float = 1.0
    beta_cm2: float = 0.5
    gamma_strat: float = 0.2
    delta_clicks: float = 0.1


def compute_reward(outcome: SessionOutcome, weights: BusinessWeights) -> float:
    """Implements #EQ-1.2: R = alpha*GMV + beta*CM2 + gamma*STRAT + delta*CLICKS.

    This is the **scalar objective** we will maximize via RL.

    See `zoosim/dynamics/reward.py:42-66` for the production implementation that
    aggregates GMV/CM2/strategic purchases/clicks using `RewardConfig`
    parameters defined in `zoosim/core/config.py:195`.
    """
    return (weights.alpha_gmv * outcome.gmv +
            weights.beta_cm2 * outcome.cm2 +
            weights.gamma_strat * outcome.strat_purchases +
            weights.delta_clicks * outcome.clicks)


def compute_rpc(outcome: SessionOutcome) -> float:
    """GMV per click (revenue per click, RPC).

    Diagnostic for clickbait detection: high CTR with low RPC indicates
    the agent is optimizing delta*CLICKS at expense of alpha*GMV.
    See [REM-1.2.1] for theory.
    """
    return outcome.gmv / outcome.clicks if outcome.clicks > 0 else 0.0


def test_basic_reward_comparison():
    """Test Strategy A vs Strategy B with default weights."""
    print("=" * 70)
    print("Test 1: Basic Reward Comparison")
    print("=" * 70)

    # Strategy A: Maximize GMV (show expensive products)
    outcome_A = SessionOutcome(gmv=120.0, cm2=15.0, strat_purchases=1, clicks=3)

    # Strategy B: Balance GMV and CM2 (show profitable products)
    outcome_B = SessionOutcome(gmv=100.0, cm2=35.0, strat_purchases=3, clicks=4)

    weights = BusinessWeights(alpha_gmv=1.0, beta_cm2=0.5, gamma_strat=0.2, delta_clicks=0.1)

    R_A = compute_reward(outcome_A, weights)
    R_B = compute_reward(outcome_B, weights)

    print(f"Strategy A (GMV-focused): R = {R_A:.2f}")
    print(f"Strategy B (Balanced):    R = {R_B:.2f}")
    print(f"Δ = {R_B - R_A:.2f} (Strategy {'B' if R_B > R_A else 'A'} wins!)")

    # Expected values from chapter (corrected arithmetic)
    assert abs(R_A - 128.00) < 0.01, f"R_A should be 128.00, got {R_A:.2f}"
    assert abs(R_B - 118.50) < 0.01, f"R_B should be 118.50, got {R_B:.2f}"
    assert R_A > R_B, "Strategy A should win with default weights"

    print("✓ Test passed: values match chapter output\n")


def test_profitability_weighting():
    """Test Strategy A vs B with profit-focused weights."""
    print("=" * 70)
    print("Test 2: Profitability Weighting")
    print("=" * 70)

    outcome_A = SessionOutcome(gmv=120.0, cm2=15.0, strat_purchases=1, clicks=3)
    outcome_B = SessionOutcome(gmv=100.0, cm2=35.0, strat_purchases=3, clicks=4)

    weights_profit = BusinessWeights(alpha_gmv=0.5, beta_cm2=1.0, gamma_strat=0.5, delta_clicks=0.1)
    R_A_profit = compute_reward(outcome_A, weights_profit)
    R_B_profit = compute_reward(outcome_B, weights_profit)

    print(f"With profitability weighting:")
    print(f"Strategy A: R = {R_A_profit:.2f}")
    print(f"Strategy B: R = {R_B_profit:.2f}")
    print(f"Δ = {R_B_profit - R_A_profit:.2f} (Strategy {'B' if R_B_profit > R_A_profit else 'A'} wins!)")

    # Expected values from chapter (corrected arithmetic)
    assert abs(R_A_profit - 75.80) < 0.01, f"R_A_profit should be 75.80, got {R_A_profit:.2f}"
    assert abs(R_B_profit - 86.90) < 0.01, f"R_B_profit should be 86.90, got {R_B_profit:.2f}"
    assert R_B_profit > R_A_profit, "Strategy B should win with profit weighting"

    print("✓ Test passed: values match chapter output\n")


def test_rpc_diagnostic():
    """Test revenue-per-click (RPC) diagnostic from [REM-1.2.1]."""
    print("=" * 70)
    print("Test 3: RPC Diagnostic (Clickbait Detection)")
    print("=" * 70)

    outcome_A = SessionOutcome(gmv=120.0, cm2=15.0, strat_purchases=1, clicks=3)
    outcome_B = SessionOutcome(gmv=100.0, cm2=35.0, strat_purchases=3, clicks=4)
    weights = BusinessWeights(alpha_gmv=1.0, beta_cm2=0.5, gamma_strat=0.2, delta_clicks=0.1)

    rpc_A = compute_rpc(outcome_A)
    rpc_B = compute_rpc(outcome_B)

    print("Revenue per click (GMV per click):")
    print(f"Strategy A: €{rpc_A:.2f}/click ({outcome_A.clicks} clicks → €{outcome_A.gmv:.0f} GMV)")
    print(f"Strategy B: €{rpc_B:.2f}/click ({outcome_B.clicks} clicks → €{outcome_B.gmv:.0f} GMV)")
    print(f"→ Strategy {'A' if rpc_A > rpc_B else 'B'} has higher-quality engagement")

    # Verify delta/alpha bound from [REM-1.2.1]
    delta_alpha_ratio = weights.delta_clicks / weights.alpha_gmv
    print(f"\n[Validation] delta/alpha = {delta_alpha_ratio:.3f}")
    print(f"             Bound check: {'✓' if delta_alpha_ratio <= 0.10 else '✗'} (must be ≤ 0.10)")

    # Expected values
    assert abs(rpc_A - 40.0) < 0.01, f"RPC_A should be 40.0, got {rpc_A:.2f}"
    assert abs(rpc_B - 25.0) < 0.01, f"RPC_B should be 25.0, got {rpc_B:.2f}"
    assert rpc_A > rpc_B, "Strategy A should have higher RPC (quality over quantity)"
    assert delta_alpha_ratio <= 0.10, f"delta/alpha = {delta_alpha_ratio:.3f} exceeds bound of 0.10"

    print("✓ Test passed: RPC diagnostic works correctly\n")


def test_delta_alpha_bounds():
    """Test validation of delta/alpha bounds."""
    print("=" * 70)
    print("Test 4: Delta/Alpha Bound Validation")
    print("=" * 70)

    # Test valid bounds
    valid_weights = [
        BusinessWeights(alpha_gmv=1.0, delta_clicks=0.01),  # Lower bound
        BusinessWeights(alpha_gmv=1.0, delta_clicks=0.05),  # Recommended
        BusinessWeights(alpha_gmv=1.0, delta_clicks=0.10),  # Upper bound
    ]

    for i, w in enumerate(valid_weights):
        ratio = w.delta_clicks / w.alpha_gmv
        assert 0.01 <= ratio <= 0.10, f"Valid weight {i} failed: {ratio:.3f}"
        print(f"✓ Valid: delta/alpha = {ratio:.3f}")

    # Test invalid bounds (should be caught in production)
    invalid_weights = [
        BusinessWeights(alpha_gmv=1.0, delta_clicks=0.005),  # Too low
        BusinessWeights(alpha_gmv=1.0, delta_clicks=0.15),   # Too high
        BusinessWeights(alpha_gmv=1.0, delta_clicks=0.50),   # Way too high
    ]

    for i, w in enumerate(invalid_weights):
        ratio = w.delta_clicks / w.alpha_gmv
        is_valid = 0.01 <= ratio <= 0.10
        print(f"✗ Invalid: delta/alpha = {ratio:.3f} (out of [0.01, 0.10])")
        assert not is_valid, f"Invalid weight {i} should fail validation"

    print("✓ Test passed: bound validation works correctly\n")


def test_rpc_edge_cases():
    """Test edge cases for RPC computation."""
    print("=" * 70)
    print("Test 5: RPC Edge Cases")
    print("=" * 70)

    # Zero clicks (should return 0.0, not divide-by-zero error)
    outcome_zero_clicks = SessionOutcome(gmv=0.0, cm2=0.0, strat_purchases=0, clicks=0)
    rpc_zero = compute_rpc(outcome_zero_clicks)
    assert rpc_zero == 0.0, f"RPC with zero clicks should be 0.0, got {rpc_zero}"
    print(f"✓ Zero clicks: RPC = {rpc_zero:.2f} (no divide-by-zero)")

    # High GMV, few clicks (high quality)
    outcome_high_quality = SessionOutcome(gmv=500.0, cm2=100.0, strat_purchases=2, clicks=2)
    rpc_high = compute_rpc(outcome_high_quality)
    assert rpc_high == 250.0, f"High-quality RPC should be 250.0, got {rpc_high:.2f}"
    print(f"✓ High quality: RPC = €{rpc_high:.2f}/click (€500 GMV / 2 clicks)")

    # Low GMV, many clicks (clickbait)
    outcome_clickbait = SessionOutcome(gmv=50.0, cm2=10.0, strat_purchases=1, clicks=20)
    rpc_low = compute_rpc(outcome_clickbait)
    assert rpc_low == 2.5, f"Clickbait RPC should be 2.5, got {rpc_low:.2f}"
    print(f"✓ Clickbait detected: RPC = €{rpc_low:.2f}/click (€50 GMV / 20 clicks)")
    print("  → This is the pathological case from [REM-1.2.1]")

    print("✓ Test passed: edge cases handled correctly\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Chapter 1, Section 1.2: Reward Function Code Validation")
    print("="*70 + "\n")

    test_basic_reward_comparison()
    test_profitability_weighting()
    test_rpc_diagnostic()
    test_delta_alpha_bounds()
    test_rpc_edge_cases()

    print("="*70)
    print("All tests passed! ✓")
    print("="*70)
    print("\nCode examples from ch01_foundations.md are verified correct.")
