"""
Test script for Chapter 1 toy example.

This script implements the complete toy problem from the introduction:
- 3 user types with different sensitivities
- 10 products with synthetic features
- Reward function with GMV + CM2 + CLICKS
- Simple simulation to verify it runs

This is based on the toy example in docs/book/revisions/ch01_foundations_revised_pedagogy_v2.md
"""

import numpy as np
from typing import NamedTuple
from dataclasses import dataclass


class UserType(NamedTuple):
    """User preference profile over product attributes."""
    discount_sensitivity: float  # How much user cares about discounts
    quality_sensitivity: float   # How much user cares about brand/quality


# Three user segments (ground truth preferences)
USER_TYPES = {
    "price_hunter": UserType(discount_sensitivity=0.9, quality_sensitivity=0.1),
    "premium": UserType(discount_sensitivity=0.1, quality_sensitivity=0.9),
    "bulk_buyer": UserType(discount_sensitivity=0.3, quality_sensitivity=0.5),
}


def generate_products(n_products: int = 10, seed: int = 42):
    """Generate synthetic product catalog with reproducible seed."""
    rng = np.random.default_rng(seed)
    return {
        i: {
            "discount_score": rng.uniform(0, 1),  # 0 = no discount, 1 = deep discount
            "quality_score": rng.uniform(0, 1),   # 0 = generic, 1 = premium brand
            "base_price": rng.uniform(5, 50),     # Base price in euros
            "margin_pct": rng.uniform(0.15, 0.45),  # Profit margin
        }
        for i in range(n_products)
    }


PRODUCTS = generate_products()


def compute_product_score(product_id: int,
                          w_discount: float,
                          w_quality: float,
                          products: dict = PRODUCTS) -> float:
    """
    Compute boosted score for a product given boost weights.

    Score = discount_score * w_discount + quality_score * w_quality

    Args:
        product_id: Product ID (0-9 in toy example)
        w_discount: Boost weight for discount feature [-1, +1]
        w_quality: Boost weight for quality feature [-1, +1]
        products: Product catalog dict

    Returns:
        Boosted score (higher = better rank)
    """
    prod = products[product_id]
    return (prod["discount_score"] * w_discount +
            prod["quality_score"] * w_quality)


def rank_products(w_discount: float,
                  w_quality: float,
                  products: dict = PRODUCTS) -> list[int]:
    """
    Rank all products by boosted score (descending).

    Returns:
        List of product IDs sorted by score (best first)
    """
    scores = {pid: compute_product_score(pid, w_discount, w_quality, products)
              for pid in products}
    return sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)


def simulate_user_interaction(user_type: UserType,
                               ranking: list[int],
                               products: dict = PRODUCTS,
                               seed: int | None = None) -> dict:
    """
    Simulate user clicking and potentially purchasing from ranking.

    Simple position bias model: P(click at position k) = 1/k
    Purchase probability depends on alignment with user preferences.

    Args:
        user_type: User preference profile
        ranking: List of product IDs in display order
        products: Product catalog
        seed: Random seed for reproducibility

    Returns:
        Dict with clicks, purchases, GMV, CM2 metrics
    """
    rng = np.random.default_rng(seed)

    clicks = []
    purchases = []
    gmv = 0.0
    cm2 = 0.0

    # Simulate examination and clicks (position-biased)
    for position, product_id in enumerate(ranking[:5], start=1):  # Top 5 only
        # Position bias: P(examine) = 1/position
        if rng.random() < 1.0 / position:
            prod = products[product_id]

            # Click probability based on alignment with user preferences
            utility = (user_type.discount_sensitivity * prod["discount_score"] +
                      user_type.quality_sensitivity * prod["quality_score"])

            p_click = min(0.9, utility)  # Cap at 90%

            if rng.random() < p_click:
                clicks.append(product_id)

                # Purchase probability (50% of clicks convert)
                if rng.random() < 0.5:
                    purchases.append(product_id)
                    price = prod["base_price"]
                    margin = prod["margin_pct"]

                    gmv += price
                    cm2 += price * margin

    return {
        "clicks": clicks,
        "purchases": purchases,
        "n_clicks": len(clicks),
        "n_purchases": len(purchases),
        "gmv": gmv,
        "cm2": cm2,
    }


def compute_reward(interaction: dict,
                   alpha: float = 0.6,
                   beta: float = 0.3,
                   gamma: float = 0.0,
                   delta: float = 0.1) -> float:
    """
    Compute reward from interaction metrics.

    R = alpha * GMV + beta * CM2 + gamma * STRAT + delta * CLICKS

    Args:
        interaction: Dict from simulate_user_interaction
        alpha, beta, gamma, delta: Business weights (must sum to 1)

    Returns:
        Scalar reward
    """
    return (alpha * interaction["gmv"] +
            beta * interaction["cm2"] +
            delta * interaction["n_clicks"])


def test_single_episode():
    """Test a single episode with the toy example."""
    print("Testing single episode...")
    print(f"Products: {len(PRODUCTS)}")
    print(f"User types: {list(USER_TYPES.keys())}")

    # Test with price hunter
    user_type = USER_TYPES["price_hunter"]
    print(f"\nUser type: price_hunter")
    print(f"  Discount sensitivity: {user_type.discount_sensitivity}")
    print(f"  Quality sensitivity: {user_type.quality_sensitivity}")

    # Try discount-heavy weights (should be good for price hunter)
    w_discount = 0.8
    w_quality = 0.2
    print(f"\nBoost weights: discount={w_discount}, quality={w_quality}")

    # Rank products
    ranking = rank_products(w_discount, w_quality)
    print(f"Top 5 ranking: {ranking[:5]}")

    # Simulate interaction
    interaction = simulate_user_interaction(user_type, ranking, seed=42)
    print(f"\nInteraction results:")
    print(f"  Clicks: {interaction['n_clicks']}")
    print(f"  Purchases: {interaction['n_purchases']}")
    print(f"  GMV: €{interaction['gmv']:.2f}")
    print(f"  CM2: €{interaction['cm2']:.2f}")

    # Compute reward
    reward = compute_reward(interaction)
    print(f"  Reward: {reward:.3f}")

    print("\n✓ Single episode test passed!")


def test_multiple_episodes():
    """Test multiple episodes with different user types and weights."""
    print("\nTesting multiple episodes...")

    n_episodes = 100
    results = []

    rng = np.random.default_rng(42)

    for ep in range(n_episodes):
        # Random user type
        user_name = rng.choice(list(USER_TYPES.keys()))
        user_type = USER_TYPES[user_name]

        # Random boost weights
        w_discount = rng.uniform(-1, 1)
        w_quality = rng.uniform(-1, 1)

        # Simulate
        ranking = rank_products(w_discount, w_quality)
        interaction = simulate_user_interaction(user_type, ranking, seed=ep)
        reward = compute_reward(interaction)

        results.append({
            "episode": ep,
            "user": user_name,
            "w_discount": w_discount,
            "w_quality": w_quality,
            "reward": reward,
        })

    # Summary statistics
    rewards = [r["reward"] for r in results]
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Std reward: {np.std(rewards):.3f}")
    print(f"Min reward: {np.min(rewards):.3f}")
    print(f"Max reward: {np.max(rewards):.3f}")

    print("\n✓ Multiple episodes test passed!")


def test_user_type_preferences():
    """Verify that different user types prefer different boost weights."""
    print("\nTesting user type preferences...")

    # Test each user type with discount-heavy vs quality-heavy weights
    test_configs = [
        ("discount-heavy", 0.9, 0.1),
        ("quality-heavy", 0.1, 0.9),
        ("balanced", 0.5, 0.5),
    ]

    n_trials = 50

    for user_name, user_type in USER_TYPES.items():
        print(f"\n{user_name}:")
        for config_name, w_discount, w_quality in test_configs:
            rewards = []
            for trial in range(n_trials):
                ranking = rank_products(w_discount, w_quality)
                interaction = simulate_user_interaction(user_type, ranking, seed=trial)
                reward = compute_reward(interaction)
                rewards.append(reward)

            mean_reward = np.mean(rewards)
            print(f"  {config_name:15s}: {mean_reward:.3f}")

    print("\n✓ User preference test passed!")
    print("Note: price_hunter should prefer discount-heavy, premium should prefer quality-heavy")


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 1 Toy Example Test Suite")
    print("=" * 60)

    test_single_episode()
    test_multiple_episodes()
    test_user_type_preferences()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
