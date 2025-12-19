"""
Complete solution to Chapter 1 toy problem.

Implements:
1. Bandit Q-learning (incremental Monte Carlo) with epsilon-greedy exploration
2. Baseline policies (random, static optimal, oracle)
3. Learning curves and convergence analysis
4. Verification of 90% optimality claim

Author: Vlad Prytula

Run: uv run python scripts/ch00/toy_problem_solution.py --all
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple, Callable
from collections import defaultdict


# ============================================================================
# Environment (from toy example)
# ============================================================================

class UserType(NamedTuple):
    """User preference profile over product attributes."""
    discount_sensitivity: float
    quality_sensitivity: float


USER_TYPES = {
    "price_hunter": UserType(discount_sensitivity=0.9, quality_sensitivity=0.1),
    "premium": UserType(discount_sensitivity=0.1, quality_sensitivity=0.9),
    "bulk_buyer": UserType(discount_sensitivity=0.3, quality_sensitivity=0.5),
}


def generate_products(n_products: int = 10, seed: int = 42):
    """Generate synthetic product catalog."""
    rng = np.random.default_rng(seed)
    return {
        i: {
            "discount_score": rng.uniform(0, 1),
            "quality_score": rng.uniform(0, 1),
            "base_price": rng.uniform(5, 50),
            "margin_pct": rng.uniform(0.15, 0.45),
        }
        for i in range(n_products)
    }


PRODUCTS = generate_products()


def rank_products(w_discount: float, w_quality: float, products: dict = PRODUCTS) -> list[int]:
    """Rank products by boosted score."""
    scores = {
        pid: (products[pid]["discount_score"] * w_discount +
              products[pid]["quality_score"] * w_quality)
        for pid in products
    }
    return sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)


def simulate_user_interaction(user_type: UserType,
                               ranking: list[int],
                               products: dict = PRODUCTS,
                               seed: int | None = None) -> dict:
    """Simulate user clicks and purchases with position bias."""
    rng = np.random.default_rng(seed)

    clicks = []
    purchases = []
    gmv = 0.0
    cm2 = 0.0

    for position, product_id in enumerate(ranking[:5], start=1):
        # Position bias: P(examine) = 1/position
        if rng.random() < 1.0 / position:
            prod = products[product_id]

            # Click probability based on user-product alignment
            utility = (user_type.discount_sensitivity * prod["discount_score"] +
                      user_type.quality_sensitivity * prod["quality_score"])
            p_click = min(0.9, utility)

            if rng.random() < p_click:
                clicks.append(product_id)

                # Purchase with 50% probability
                if rng.random() < 0.5:
                    purchases.append(product_id)
                    price = prod["base_price"]
                    margin = prod["margin_pct"]
                    gmv += price
                    cm2 += price * margin

    return {"clicks": clicks, "purchases": purchases, "n_clicks": len(clicks),
            "n_purchases": len(purchases), "gmv": gmv, "cm2": cm2}


def compute_reward(interaction: dict,
                   alpha: float = 0.6, beta: float = 0.3,
                   gamma: float = 0.0, delta: float = 0.1) -> float:
    """Compute reward: R = alpha*GMV + beta*CM2 + delta*CLICKS"""
    return (alpha * interaction["gmv"] + beta * interaction["cm2"] +
            delta * interaction["n_clicks"])


# ============================================================================
# Action Space Discretization
# ============================================================================

def discretize_action_space(n_bins: int = 5, a_min: float = -1.0, a_max: float = 1.0):
    """
    Create discrete action grid over [a_min, a_max]^2.

    Returns:
        List of (w_discount, w_quality) tuples
    """
    bins = np.linspace(a_min, a_max, n_bins)
    actions = [(w_d, w_q) for w_d in bins for w_q in bins]
    return actions


def action_to_index(w_discount: float, w_quality: float,
                    actions: list[tuple[float, float]]) -> int:
    """Find nearest action in discrete grid."""
    distances = [(w_discount - a[0])**2 + (w_quality - a[1])**2 for a in actions]
    return int(np.argmin(distances))


# ============================================================================
# Policies
# ============================================================================

class Policy:
    """Base class for policies."""

    def select_action(self, user_name: str, rng: np.random.Generator) -> tuple[float, float]:
        """Select (w_discount, w_quality) given context."""
        raise NotImplementedError

    def update(self, user_name: str, action: tuple[float, float], reward: float):
        """Update policy (for learning algorithms)."""
        pass


class RandomPolicy(Policy):
    """Uniform random policy over action space."""

    def __init__(self, a_min: float = -1.0, a_max: float = 1.0):
        self.a_min = a_min
        self.a_max = a_max

    def select_action(self, user_name: str, rng: np.random.Generator) -> tuple[float, float]:
        w_discount = rng.uniform(self.a_min, self.a_max)
        w_quality = rng.uniform(self.a_min, self.a_max)
        return (w_discount, w_quality)


class StaticOptimalPolicy(Policy):
    """
    Static policy with hand-tuned weights per user type.

    Based on ground truth preferences:
    - price_hunter: high discount weight
    - premium: high quality weight
    - bulk_buyer: balanced
    """

    def __init__(self):
        self.weights = {
            "price_hunter": (0.8, 0.2),   # Discount-heavy
            "premium": (0.2, 0.8),        # Quality-heavy
            "bulk_buyer": (0.4, 0.6),     # Balanced
        }

    def select_action(self, user_name: str, rng: np.random.Generator) -> tuple[float, float]:
        return self.weights[user_name]


class OraclePolicy(Policy):
    """
    Oracle that knows true Q-values and acts optimally.

    Implements grid search over actions to find best for each user type.
    """

    def __init__(self, actions: list[tuple[float, float]], n_eval: int = 100, seed: int = 42):
        """
        Pre-compute optimal actions for each user type via Monte Carlo.

        Args:
            actions: Discrete action grid
            n_eval: Number of trials per (user, action) pair
            seed: Random seed for reproducibility
        """
        self.actions = actions
        self.optimal_actions = {}

        print("Oracle: Computing optimal actions via grid search...")
        for user_name, user_type in USER_TYPES.items():
            best_action = None
            best_q = -np.inf

            for action in actions:
                # Estimate Q(user, action) via Monte Carlo
                rewards = []
                for trial in range(n_eval):
                    ranking = rank_products(action[0], action[1])
                    interaction = simulate_user_interaction(user_type, ranking, seed=seed + trial)
                    reward = compute_reward(interaction)
                    rewards.append(reward)

                q_est = np.mean(rewards)
                if q_est > best_q:
                    best_q = q_est
                    best_action = action

            self.optimal_actions[user_name] = best_action
            print(f"  {user_name}: w*={best_action}, Q*={best_q:.2f}")

    def select_action(self, user_name: str, rng: np.random.Generator) -> tuple[float, float]:
        return self.optimal_actions[user_name]


class TabularQLearning(Policy):
    """
    Contextual bandit Q-learning (incremental Monte Carlo) with epsilon-greedy exploration.

    Q-table: dict[(user_name, action_idx) -> Q-value]
    Learning rule: Q(s,a) += alpha * (r - Q(s,a))  [contextual bandit, no bootstrap]
    """

    def __init__(self,
                 actions: list[tuple[float, float]],
                 epsilon_init: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 learning_rate: float = 0.1):
        """
        Initialize Q-learning agent.

        Args:
            actions: Discrete action grid
            epsilon_init: Initial exploration rate
            epsilon_decay: Decay factor per episode
            epsilon_min: Minimum exploration rate
            learning_rate: Step size for Q-updates
        """
        self.actions = actions
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = learning_rate

        # Q-table: (user_name, action_idx) -> Q-value
        self.Q = defaultdict(float)

        # Visit counts for diagnostics
        self.N = defaultdict(int)

    def select_action(self, user_name: str, rng: np.random.Generator) -> tuple[float, float]:
        """Epsilon-greedy action selection."""
        if rng.random() < self.epsilon:
            # Explore: random action
            action_idx = rng.integers(0, len(self.actions))
        else:
            # Exploit: best known action
            q_values = [self.Q[(user_name, a_idx)] for a_idx in range(len(self.actions))]
            action_idx = int(np.argmax(q_values))

        self.N[(user_name, action_idx)] += 1
        return self.actions[action_idx]

    def update(self, user_name: str, action: tuple[float, float], reward: float):
        """Q-learning update (bandit version, no bootstrap)."""
        action_idx = action_to_index(action[0], action[1], self.actions)
        key = (user_name, action_idx)

        # Bandit Q-update: Q(s,a) += alpha * (r - Q(s,a))
        self.Q[key] += self.alpha * (reward - self.Q[key])

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(policy: Policy,
                    n_episodes: int = 200,
                    seed: int = 42) -> dict:
    """
    Evaluate policy performance over n_episodes.

    Returns:
        Dict with rewards, per-user rewards, and metrics
    """
    rng = np.random.default_rng(seed)
    rewards = []
    per_user_rewards = {name: [] for name in USER_TYPES}

    for ep in range(n_episodes):
        # Sample user type
        user_name = rng.choice(list(USER_TYPES.keys()))
        user_type = USER_TYPES[user_name]

        # Policy selects action
        action = policy.select_action(user_name, rng)

        # Simulate interaction
        ranking = rank_products(action[0], action[1])
        interaction = simulate_user_interaction(user_type, ranking, seed=seed + ep)
        reward = compute_reward(interaction)

        rewards.append(reward)
        per_user_rewards[user_name].append(reward)

    return {
        "rewards": rewards,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "per_user_mean": {name: np.mean(rews) for name, rews in per_user_rewards.items()},
    }


def run_learning_experiment(policy: Policy,
                            n_train: int = 1000,
                            eval_interval: int = 50,
                            n_eval: int = 100,
                            seed: int = 42) -> dict:
    """
    Run full learning experiment with periodic evaluation.

    Returns:
        Dict with training history and final evaluation
    """
    rng = np.random.default_rng(seed)

    eval_episodes = []
    eval_means = []
    eval_stds = []

    # Training loop
    for ep in range(n_train):
        # Sample user type
        user_name = rng.choice(list(USER_TYPES.keys()))
        user_type = USER_TYPES[user_name]

        # Select action
        action = policy.select_action(user_name, rng)

        # Simulate interaction
        ranking = rank_products(action[0], action[1])
        interaction = simulate_user_interaction(user_type, ranking, seed=seed + ep)
        reward = compute_reward(interaction)

        # Update policy
        policy.update(user_name, action, reward)

        # Periodic evaluation
        if (ep + 1) % eval_interval == 0:
            eval_result = evaluate_policy(policy, n_episodes=n_eval, seed=seed + 10000 + ep)
            eval_episodes.append(ep + 1)
            eval_means.append(eval_result["mean_reward"])
            eval_stds.append(eval_result["std_reward"])

    # Final evaluation
    final_eval = evaluate_policy(policy, n_episodes=n_eval, seed=seed + 20000)

    return {
        "eval_episodes": eval_episodes,
        "eval_means": eval_means,
        "eval_stds": eval_stds,
        "final_mean": final_eval["mean_reward"],
        "final_std": final_eval["std_reward"],
        "final_per_user": final_eval["per_user_mean"],
    }


# ============================================================================
# Main Experiments
# ============================================================================

def main():
    """Run complete toy problem solution."""
    print("=" * 70)
    print("Chapter 1 Toy Problem: Complete Solution")
    print("=" * 70)

    # Discretize action space
    actions = discretize_action_space(n_bins=5)
    print(f"\nAction space: {len(actions)} discrete actions (5x5 grid)")

    # Initialize policies
    print("\n" + "=" * 70)
    print("Baseline Policies")
    print("=" * 70)

    random_policy = RandomPolicy()
    static_policy = StaticOptimalPolicy()
    oracle_policy = OraclePolicy(actions, n_eval=100, seed=42)

    # Evaluate baselines
    print("\nEvaluating baselines (200 episodes each)...")
    baseline_results = {}

    for name, policy in [("Random", random_policy),
                          ("Static", static_policy),
                          ("Oracle", oracle_policy)]:
        result = evaluate_policy(policy, n_episodes=200, seed=42)
        baseline_results[name] = result["mean_reward"]
        print(f"{name:10s}: {result['mean_reward']:6.2f} ± {result['std_reward']:5.2f}")

    # Learning experiment
    print("\n" + "=" * 70)
    print("Q-Learning Agent")
    print("=" * 70)

    q_agent = TabularQLearning(actions, epsilon_init=1.0, epsilon_decay=0.995,
                               epsilon_min=0.05, learning_rate=0.1)

    print("\nRunning learning experiment (1000 episodes)...")
    learning_results = run_learning_experiment(
        q_agent, n_train=1000, eval_interval=50, n_eval=100, seed=42
    )

    print(f"\nFinal performance: {learning_results['final_mean']:.2f} ± {learning_results['final_std']:.2f}")
    print("\nPer-user performance:")
    for user_name, mean_r in learning_results['final_per_user'].items():
        print(f"  {user_name:15s}: {mean_r:.2f}")

    # Compute % of oracle
    oracle_reward = baseline_results["Oracle"]
    pct_optimal = 100 * learning_results['final_mean'] / oracle_reward
    print(f"\n% of Oracle: {pct_optimal:.1f}%")

    # Plot learning curves
    print("\n" + "=" * 70)
    print("Generating learning curves...")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Q-learning curve
    episodes = learning_results['eval_episodes']
    means = learning_results['eval_means']
    stds = learning_results['eval_stds']

    ax.plot(episodes, means, 'b-', linewidth=2, label='Q-Learning')
    ax.fill_between(episodes,
                     np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds),
                     alpha=0.2, color='blue')

    # Plot baselines
    ax.axhline(baseline_results['Oracle'], color='green', linestyle='--',
               linewidth=2, label=f'Oracle ({baseline_results["Oracle"]:.1f})')
    ax.axhline(baseline_results['Static'], color='orange', linestyle='--',
               linewidth=2, label=f'Static ({baseline_results["Static"]:.1f})')
    ax.axhline(baseline_results['Random'], color='red', linestyle='--',
               linewidth=2, label=f'Random ({baseline_results["Random"]:.1f})')

    # 90% oracle line
    ax.axhline(0.9 * baseline_results['Oracle'], color='green', linestyle=':',
               linewidth=1, alpha=0.5, label='90% Oracle')

    ax.set_xlabel('Training Episodes', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Chapter 1 Toy Problem: Learning Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('toy_problem_learning_curves.png', dpi=150)
    print("Saved: toy_problem_learning_curves.png")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Oracle (optimal):       {baseline_results['Oracle']:6.2f}")
    print(f"Q-Learning (final):     {learning_results['final_mean']:6.2f}  ({pct_optimal:.1f}% of oracle)")
    print(f"Static (hand-tuned):    {baseline_results['Static']:6.2f}")
    print(f"Random (baseline):      {baseline_results['Random']:6.2f}")

    if pct_optimal >= 90:
        print(f"\n✓ SUCCESS: Q-learning achieves ≥90% of oracle performance!")
    else:
        print(f"\n✗ Below target: {pct_optimal:.1f}% < 90%")
        print("  (Try: increase n_train, tune learning_rate, or refine action grid)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
