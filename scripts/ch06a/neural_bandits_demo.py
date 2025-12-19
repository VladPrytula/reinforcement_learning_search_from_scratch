#!/usr/bin/env python
"""Chapter 6A: NL-Bandit (Neural Q(x,a)) Demo.

Demonstrates:
1. Full neural Q-function for discrete templates.
2. Ensemble-based uncertainty quantification.
3. Comparison with Neural Linear.

Usage:
    python scripts/ch06a/neural_bandits_demo.py --ensemble-size 5
"""

import argparse
import numpy as np
import torch
from typing import List, Tuple
from zoosim.core.config import SimulatorConfig
from zoosim.world import catalog as catalog_module
from zoosim.world.users import sample_user
from zoosim.world.queries import sample_query
from zoosim.ranking.relevance import batch_base_scores
from zoosim.dynamics.reward import compute_reward
from zoosim.policies.templates import compute_catalog_stats, create_standard_templates
from zoosim.dynamics import behavior
from zoosim.policies.q_ensemble import QEnsembleRegressor

# --------------------------------------------------------------------------
# 1. Discrete NL-Bandit Policy
# --------------------------------------------------------------------------

class DiscreteNLBandit:
    """NL-Bandit for discrete templates using Q-ensemble.

    Wraps QEnsembleRegressor (designed for continuous actions) for discrete selection.
    Input to Q-net: [context, one_hot_action]
    """
    def __init__(
        self,
        templates: list,
        input_dim: int, # Dimension of context x
        hidden_dim: int = 64,
        ensemble_size: int = 5,
        ucb_beta: float = 2.0,
        learning_rate: float = 1e-3,
        seed: int = 42
    ):
        self.templates = templates
        self.n_actions = len(templates)
        self.ucb_beta = ucb_beta
        self.input_dim = input_dim

        # Q-ensemble input dim = context + action (one-hot)
        # We use state_dim = input_dim, action_dim = n_actions
        # QEnsembleRegressor expects concatenated input internally if we pass state, action separately
        from zoosim.policies.q_ensemble import QEnsembleConfig
        config = QEnsembleConfig(
            n_ensembles=ensemble_size,
            hidden_sizes=(hidden_dim, hidden_dim),
            learning_rate=learning_rate,
            seed=seed
        )
        self.q_ensemble = QEnsembleRegressor(
            state_dim=input_dim,
            action_dim=self.n_actions, # One-hot dimension
            config=config
        )
        
        # Replay buffer
        self.buffer_x = []
        self.buffer_a = [] # stored as one-hot
        self.buffer_r = []

    def select_action(self, context: np.ndarray) -> int:
        """Select action using UCB over ensemble."""
        # context shape (d,)
        # Prepare batch of contexts (repeated) and all possible actions (one-hot)
        
        # Contexts: (n_actions, d)
        contexts_batch = np.tile(context, (self.n_actions, 1))
        
        # Actions: (n_actions, n_actions) identity matrix represents one-hots
        actions_batch = np.eye(self.n_actions)
        
        # Query ensemble
        # returns mean, std of shape (n_actions,)
        means, stds = self.q_ensemble.predict(contexts_batch, actions_batch)
        
        # UCB
        ucb_scores = means + self.ucb_beta * stds
        return int(np.argmax(ucb_scores))

    def update(self, context: np.ndarray, action_idx: int, reward: float):
        """Add to buffer and train."""
        # Store
        self.buffer_x.append(context)
        
        action_onehot = np.zeros(self.n_actions)
        action_onehot[action_idx] = 1.0
        self.buffer_a.append(action_onehot)
        
        self.buffer_r.append(reward)
        
        # Train incrementally (simplification for demo)
        # In production, maybe train every N steps or use a fixed buffer size
        if len(self.buffer_x) % 10 == 0: # Train every 10 steps
             self._train()

    def _train(self, n_epochs: int = 1):
        x_arr = np.array(self.buffer_x)
        a_arr = np.array(self.buffer_a)
        r_arr = np.array(self.buffer_r)
        
        # Use only recent window or random sample in real implementation
        # Here we use all for small demo
        self.q_ensemble.update_batch(x_arr, a_arr, r_arr, n_epochs=n_epochs)

# --------------------------------------------------------------------------
# 2. Helpers (copied/shared)
# --------------------------------------------------------------------------

def get_raw_context(user, query, cfg) -> np.ndarray:
    # Segment one-hot
    seg_idx = cfg.users.segments.index(user.segment)
    seg_vec = np.zeros(len(cfg.users.segments))
    seg_vec[seg_idx] = 1.0
    
    # Query type one-hot
    q_idx = cfg.queries.query_types.index(query.query_type)
    q_vec = np.zeros(len(cfg.queries.query_types))
    q_vec[q_idx] = 1.0
    
    # Latents
    latents = np.array([user.theta_price, user.theta_pl])
    
    return np.concatenate([seg_vec, q_vec, latents])

def run_episode(env_components, policy, templates, rng):
    cfg, products = env_components
    user = sample_user(config=cfg, rng=rng)
    query = sample_query(user=user, config=cfg, rng=rng)
    
    context = get_raw_context(user, query, cfg)
    action_idx = policy.select_action(context)
    template = templates[action_idx]
    
    base_scores = batch_base_scores(query=query, catalog=products, config=cfg, rng=rng)
    boosts = template.apply(products)
    final_scores = np.array(base_scores) + np.array(boosts)
    ranking = np.argsort(final_scores)[::-1].tolist()
    
    outcome = behavior.simulate_session(user=user, query=query, ranking=ranking, catalog=products, config=cfg, rng=rng)
    r_val, _ = compute_reward(ranking=ranking, clicks=outcome.clicks, buys=outcome.buys, catalog=products, config=cfg)
    
    policy.update(context, action_idx, r_val)
    return r_val

# --------------------------------------------------------------------------
# 3. Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=2000)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--ucb-beta", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("="*60)
    print("NL-Bandit (Neural Q-Ensemble) Demo")
    print(f"Ensemble: {args.ensemble_size}, Beta: {args.ucb_beta}")
    print("="*60)

    cfg = SimulatorConfig(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    products = catalog_module.generate_catalog(cfg.catalog, rng)
    stats = compute_catalog_stats(products)
    templates = create_standard_templates(stats)
    
    # Dummy for dim
    dummy_u = sample_user(config=cfg, rng=rng)
    dummy_q = sample_query(user=dummy_u, config=cfg, rng=rng)
    dummy_ctx = get_raw_context(dummy_u, dummy_q, cfg)
    input_dim = len(dummy_ctx)

    policy = DiscreteNLBandit(
        templates, 
        input_dim=input_dim,
        ensemble_size=args.ensemble_size,
        ucb_beta=args.ucb_beta,
        seed=args.seed
    )

    rewards = []
    for i in range(args.n_episodes):
        r = run_episode((cfg, products), policy, templates, rng)
        rewards.append(r)
        
        if (i+1) % 500 == 0:
            print(f"  Episode {i+1}: Mean Reward (last 500) = {np.mean(rewards[-500:]):.3f}")

    print("\nFinal Mean Reward (last 500): {:.3f}".format(np.mean(rewards[-500:])))

if __name__ == "__main__":
    main()
