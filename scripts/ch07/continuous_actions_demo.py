#!/usr/bin/env python
"""Chapter 7: Continuous Action Optimization via Q-Ensemble and CEM.

This script demonstrates:
1.  Defining a continuous action space for ranking weights (a \u2208 \u211d^d).
2.  Learning a Q(x, a) model (Q-Ensemble) to predict session reward from context x and weights a.
3.  Using the Cross-Entropy Method (CEM) to optimize 'a' at runtime for each user session.
4.  Comparing performance against Static Best and Discrete Bandit baselines.

The loop simulates "Online RL":
- Collect batch of data using current policy (CEM on Q) + exploration.
- Update Q-ensemble with new (x, a, r) tuples.
- Evaluate performance.

Usage:
    python scripts/ch07/continuous_actions_demo.py --n-episodes 5000
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from zoosim.core.config import SimulatorConfig
from zoosim.dynamics import behavior, reward
from zoosim.optimizers.cem import CEMConfig, cem_optimize
from zoosim.policies.q_ensemble import QEnsembleConfig, QEnsembleRegressor
from zoosim.ranking import features, relevance
from zoosim.world import catalog as catalog_module
from zoosim.world import queries, users
from zoosim.world.catalog import Product
from zoosim.world.queries import Query
from zoosim.world.users import User

# ---------------------------------------------------------------------------
# Feature Engineering (Context x)
# ---------------------------------------------------------------------------

def encode_context(user: User, query: Query, cfg: SimulatorConfig) -> np.ndarray:
    """Encode user/query context into a flat vector x.
    
    Features:
    - User Segment (One-hot)
    - Query Type (One-hot)
    - Estimated User Price/PL Sensitivity (Quantized/Noisy) 
    
    We mimic 'Rich Estimated' features from Ch6.
    """
    # 1. Segment One-Hot
    segments = cfg.users.segments
    seg_vec = np.zeros(len(segments), dtype=np.float32)
    if user.segment in segments:
        seg_vec[segments.index(user.segment)] = 1.0
        
    # 2. Query Type One-Hot
    q_types = cfg.queries.query_types
    q_vec = np.zeros(len(q_types), dtype=np.float32)
    if query.query_type in q_types:
        q_vec[q_types.index(query.query_type)] = 1.0
        
    # 3. User Preferences (Estimated)
    # In production, we'd infer these. Here we take the ground truth and add noise/quantization
    # to simulate estimation error, as strictly using ground truth is "cheating" (Oracle).
    # We'll use a simple noisy proxy.
    est_price = float(np.round(user.theta_price * 2) / 2)  # Quantize to 0.5
    est_pl = float(np.round(user.theta_pl * 2) / 2)
    
    return np.concatenate([
        seg_vec, 
        q_vec, 
        np.array([est_price, est_pl], dtype=np.float32)
    ])


def get_context_dim(cfg: SimulatorConfig) -> int:
    """Return dimension of context vector x."""
    return len(cfg.users.segments) + len(cfg.queries.query_types) + 2


# ---------------------------------------------------------------------------
# Simulation & Evaluation
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    state: np.ndarray   # Context x
    action: np.ndarray  # Weights a
    reward: float       # Reward r
    gmv: float          # For reporting
    cm2: float          # For reporting

class Runner:
    def __init__(self, cfg: SimulatorConfig, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.catalog = catalog_module.generate_catalog(cfg.catalog, self.rng)
        
        # Pre-compute catalog features to speed up simulation? 
        # zoosim.envs.search_env computes them on the fly. We will do the same 
        # to keep it simple, but we need access to them for the action application.
        
    def run_episode(
        self, 
        action_weights: np.ndarray,
        record: bool = False
    ) -> Experience:
        """Run a single episode with a fixed weight vector 'action_weights'."""
        
        # 1. Sample User & Query
        user = users.sample_user(config=self.cfg, rng=self.rng)
        query = queries.sample_query(user=user, config=self.cfg, rng=self.rng)
        
        # 2. Encode Context (State)
        context_x = encode_context(user, query, self.cfg)
        
        # 3. Rank Products
        # We need to compute product features -> scores -> ranking
        # Score = Base + Features @ Action
        base_scores = np.array(relevance.batch_base_scores(
            query=query, catalog=self.catalog, config=self.cfg, rng=self.rng
        ))
        
        # Compute product features (N_products x D_action)
        # Note: In a real efficient loop we'd batch this or cache invariants.
        # For Ch7 demo, straightforward iteration is acceptable for N=1000s if needed,
        # but let's try to be reasonably efficient.
        # Optimization: Just compute for top candidate set from base ranker?
        # To be exact with the 'action', we should re-rank everything. 
        # But 'features.compute_features' is Python-heavy.
        # Let's stick to the 'Env' logic: compute all, standardize.
        
        prod_feats = [
            features.compute_features(user=user, query=query, product=p, config=self.cfg)
            for p in self.catalog
        ]
        feat_matrix = np.array(prod_feats, dtype=np.float32)
        
        if self.cfg.action.standardize_features:
            # Standardize using batch stats (approximate) or config stats?
            # The env uses 'features.standardize_features' which standardizes the *current batch*.
            # That's what we'll do to match Env behavior.
            mean = feat_matrix.mean(axis=0)
            std = feat_matrix.std(axis=0)
            std[std == 0] = 1.0
            feat_matrix = (feat_matrix - mean) / std

        # Apply Action (Weights)
        # Clip action to allowed range per config (simulating safe bounds)
        clipped_action = np.clip(action_weights, -self.cfg.action.a_max, self.cfg.action.a_max)
        
        boosts = feat_matrix @ clipped_action
        final_scores = base_scores + boosts
        
        # 4. Simulate Interaction
        ranking = np.argsort(-final_scores).tolist()
        outcome = behavior.simulate_session(
            user=user, query=query, ranking=ranking, catalog=self.catalog,
            config=self.cfg, rng=self.rng
        )
        
        # 5. Compute Reward
        r, breakdown = reward.compute_reward(
            ranking=ranking, clicks=outcome.clicks, buys=outcome.buys, 
            catalog=self.catalog, config=self.cfg
        )
        
        return Experience(
            state=context_x,
            action=clipped_action,
            reward=r,
            gmv=breakdown.gmv,
            cm2=breakdown.cm2
        )


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class RandomAgent:
    def __init__(self, action_dim: int, a_max: float, rng: np.random.Generator):
        self.d = action_dim
        self.a_max = a_max
        self.rng = rng
        
    def select_action(self, context: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-self.a_max, self.a_max, size=self.d)


class StaticTemplateAgent:
    """Agent that always applies a fixed weight vector (Template)."""
    def __init__(self, weights: List[float]):
        self.weights = np.array(weights, dtype=np.float32)
        
    def select_action(self, context: np.ndarray) -> np.ndarray:
        return self.weights


class CEMAgent:
    """Agent using Q-Ensemble + CEM for action selection."""
    def __init__(
        self, 
        q_model: QEnsembleRegressor, 
        action_dim: int, 
        a_max: float,
        cem_samples: int = 64,
        cem_iters: int = 5
    ):
        self.q_model = q_model
        self.action_dim = action_dim
        self.cem_config = CEMConfig(
            n_samples=cem_samples,
            n_iters=cem_iters,
            a_max=a_max,
            elite_frac=0.1,
            init_std=a_max / 2.0,  # Explore reasonable width
            min_std=0.01
        )
        
    def select_action(self, context: np.ndarray, ucb_beta: float = 0.0) -> np.ndarray:
        # Context shape (D_x,)
        # Objective for CEM: (N_samples, D_a) -> (N_samples,)
        
        def objective(actions: np.ndarray) -> np.ndarray:
            n = actions.shape[0]
            # Repeat context for batch prediction
            contexts = np.tile(context, (n, 1))
            
            # Predict Mean and Std
            means, stds = self.q_model.predict(contexts, actions)
            
            # UCB Objective: Mean + beta * Std
            return means + ucb_beta * stds
            
        best_action, _ = cem_optimize(
            objective=objective,
            action_dim=self.action_dim,
            config=self.cem_config
        )
        
        return best_action


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chapter 7 Demo: Continuous Actions")
    parser.add_argument("--n-episodes", type=int, default=3000, help="Total training episodes")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes per eval")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Chapter 7: Continuous Action Optimization (Q-Ensemble + CEM)")
    print("=" * 60)
    
    cfg = SimulatorConfig(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    
    runner = Runner(cfg, args.seed)
    
    # Dimensions
    state_dim = get_context_dim(cfg)
    action_dim = cfg.action.feature_dim  # e.g. 10
    
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")
    print(f"Action Bounds: [-{cfg.action.a_max}, +{cfg.action.a_max}]")
    
    # --- Initialize Models ---
    
    # 1. Q-Ensemble
    q_config = QEnsembleConfig(
        n_ensembles=5,
        hidden_sizes=(64, 64),
        learning_rate=0.005,
        device="cpu", # Use CPU for demo simplicity/compatibility
        seed=args.seed
    )
    q_model = QEnsembleRegressor(state_dim, action_dim, q_config)
    cem_agent = CEMAgent(q_model, action_dim, cfg.action.a_max)
    
    # 2. Baselines
    # "Premium" template from Ch6 (promotes expensive items)
    # We need to map the template concept to weights on features.
    # In Ch6, templates were functions t(p). In Ch7, we learn weights w.
    # We can approximate the 'Premium' template by setting the weight for 'price' to +a_max
    # and others to 0. The features list in `zoosim.ranking.features` has 'price' at index 5.
    # 0:cm2, 1:discount, 2:is_pl, 3:pers, 4:bestseller, 5:price, 6:litter_cm2, 7:disc*price, 8:pl*affinity, 9:spec*pop
    
    premium_weights = np.zeros(action_dim, dtype=np.float32)
    premium_weights[5] = cfg.action.a_max  # Max boost on Price
    premium_agent = StaticTemplateAgent(premium_weights)
    
    random_agent = RandomAgent(action_dim, cfg.action.a_max, rng)
    
    # --- Replay Buffer ---
    buffer = deque(maxlen=10000)
    
    # --- Training Loop ---
    
    print(f"\nStarting training loop ({args.n_episodes} episodes)...")
    print(f"{ 'Ep':>6} | {'Policy':>10} | {'Avg R':>7} | {'GMV':>7} | {'Q-Loss':>7}")
    print("-" * 60)
    
    # Warmup phase (Random exploration)
    n_warmup = 200
    for i in range(n_warmup):
        exp = runner.run_episode(random_agent.select_action(np.zeros(state_dim)))
        buffer.append(exp)
        
    # Train on warmup
    def train_step(epochs=10):
        if len(buffer) < args.batch_size: return 0.0
        
        # Batchify
        batch = list(buffer) # use all or sample? For simple demo, use all or recent sample
        # Let's sample random batch for SGD
        indices = rng.choice(len(buffer), size=min(len(buffer), 512), replace=False)
        states = np.array([buffer[i].state for i in indices])
        actions = np.array([buffer[i].action for i in indices])
        rewards = np.array([buffer[i].reward for i in indices])
        
        q_model.update_batch(states, actions, rewards, n_epochs=epochs, batch_size=args.batch_size)
        return 0.0 # Loss not easily returned by update_batch currently
        
    train_step(epochs=20)
    
    # Main Loop
    start_time = time.time()
    
    for ep in range(1, args.n_episodes + 1):
        # 1. Select Action (CEM with UCB exploration)
        # We need a dummy context to pick action? No, we need the REAL context.
        # But `runner.run_episode` encapsulates the user/query generation.
        # Refactor: Runner should expose "get_context" then "step".
        # For this demo, we'll generate the user/query *outside* run_episode-like logic
        # or modify run_episode to take a policy function.
        
        user = users.sample_user(config=cfg, rng=rng)
        query = queries.sample_query(user=user, config=cfg, rng=rng)
        ctx = encode_context(user, query, cfg)
        
        # UCB exploration: beta decays over time
        beta = 2.0 * np.exp(-3.0 * ep / args.n_episodes) # Decay from 2.0 to ~0.1
        
        action = cem_agent.select_action(ctx, ucb_beta=beta)
        
        # Add action noise? CEM+Ensemble handles exploration via variance, 
        # but explicit noise helps coverage early on.
        if rng.random() < 0.1:
            action += rng.normal(0, 0.1, size=action_dim)
            
        # Execute
        # We manually do the rest of run_episode here to keep context consistent
        base_scores = np.array(relevance.batch_base_scores(
            query=query, catalog=runner.catalog, config=cfg, rng=rng
        ))
        
        prod_feats = [
            features.compute_features(user=user, query=query, product=p, config=cfg)
            for p in runner.catalog
        ]
        feat_matrix = np.array(prod_feats, dtype=np.float32)
        
        if cfg.action.standardize_features:
            mean = feat_matrix.mean(axis=0)
            std = feat_matrix.std(axis=0)
            std[std == 0] = 1.0
            feat_matrix = (feat_matrix - mean) / std
            
        clipped_action = np.clip(action, -cfg.action.a_max, cfg.action.a_max)
        boosts = feat_matrix @ clipped_action
        final_scores = base_scores + boosts
        ranking = np.argsort(-final_scores).tolist()
        
        outcome = behavior.simulate_session(
            user=user, query=query, ranking=ranking, catalog=runner.catalog,
            config=cfg, rng=rng
        )
        r, bd = reward.compute_reward(
            ranking=ranking, clicks=outcome.clicks, buys=outcome.buys, 
            catalog=runner.catalog, config=cfg
        )
        
        # Store
        buffer.append(Experience(ctx, clipped_action, r, bd.gmv, bd.cm2))
        
        # Train
        if ep % 10 == 0:
            train_step(epochs=2)
            
        # Eval
        if ep % args.eval_interval == 0:
            # Evaluate CEM Agent (Greedy, beta=0)
            cem_rewards = []
            cem_gmvs = []
            for _ in range(args.eval_episodes):
                # We need to split generation and action selection again for eval
                # Or simply wrap the policy
                # Let's reuse the runner.run_episode logic if we pass the weights? 
                # No, weights depend on context.
                
                u_ev = users.sample_user(config=cfg, rng=rng)
                q_ev = queries.sample_query(user=u_ev, config=cfg, rng=rng)
                ctx_ev = encode_context(u_ev, q_ev, cfg)
                
                a_ev = cem_agent.select_action(ctx_ev, ucb_beta=0.0) # Greedy
                
                # ... execution copy-paste (should have refactored, alas)
                # Simplified Eval: just use runner.run_episode but we can't because action depends on context.
                # For simplicity in this block, we re-implement execution.
                bs_ev = np.array(relevance.batch_base_scores(query=q_ev, catalog=runner.catalog, config=cfg, rng=rng))
                pf_ev = [features.compute_features(user=u_ev, query=q_ev, product=p, config=cfg) for p in runner.catalog]
                fm_ev = np.array(pf_ev, dtype=np.float32)
                if cfg.action.standardize_features:
                    m = fm_ev.mean(0); s = fm_ev.std(0); s[s==0]=1.0
                    fm_ev = (fm_ev - m)/s
                
                act_ev = np.clip(a_ev, -cfg.action.a_max, cfg.action.a_max)
                rnk_ev = np.argsort(-(bs_ev + fm_ev @ act_ev)).tolist()
                out_ev = behavior.simulate_session(user=u_ev, query=q_ev, ranking=rnk_ev, catalog=runner.catalog, config=cfg, rng=rng)
                r_ev, bd_ev = reward.compute_reward(ranking=rnk_ev, clicks=out_ev.clicks, buys=out_ev.buys, catalog=runner.catalog, config=cfg)
                
                cem_rewards.append(r_ev)
                cem_gmvs.append(bd_ev.gmv)
                
            # Evaluate Static Baseline
            static_rewards = []
            static_gmvs = []
            for _ in range(args.eval_episodes):
                exp = runner.run_episode(premium_weights)
                static_rewards.append(exp.reward)
                static_gmvs.append(exp.gmv)
            
            print(f"{ep:6d} | {'CEM':>10} | {np.mean(cem_rewards):7.2f} | {np.mean(cem_gmvs):7.2f} | Beta={beta:.2f}")
            print(f"{ '':6s} | {'Static':>10} | {np.mean(static_rewards):7.2f} | {np.mean(static_gmvs):7.2f} |")

    print("-" * 60)
    print("Training Complete.")

if __name__ == "__main__":
    main()
