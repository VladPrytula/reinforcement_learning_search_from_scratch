#!/usr/bin/env python
"""Chapter 7: Discrete LinUCB (Ch6 Baseline) vs. Continuous Q-Learner (Ch7).

This script runs a head-to-head comparison between:
1.  **Discrete LinUCB**: Selects one of 8 fixed "Boost Templates" (e.g., 'High Margin', 'Popular').
    -   Limited flexibility (can only pick pre-defined strategies).
    -   Templates can be non-linear (e.g., "Price > 75th percentile").
2.  **Continuous Q-Ensemble**: Selects a weight vector w in R^d.
    -   Infinite flexibility (fine-grained control of every product attribute).
    -   Limited to linear combinations of available product features.

Methodology:
-   Both agents face the EXACT same sequence of Users and Queries (controlled via seed).
-   Both agents see the same "Rich Estimated" context features (user segment + query + history stats).
-   Discrete agent updates via LinUCB analytical solution.
-   Continuous agent updates via SGD on a Neural Network Ensemble (Q-function) + CEM for maximization.
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
from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
from zoosim.policies.q_ensemble import QEnsembleConfig, QEnsemblePolicy
from zoosim.policies.templates import BoostTemplate, compute_catalog_stats, create_standard_templates
from zoosim.ranking import features, relevance
from zoosim.world import catalog as catalog_module
from zoosim.world import queries, users
from zoosim.world.catalog import Product
from zoosim.world.queries import Query
from zoosim.world.users import User

# ---------------------------------------------------------------------------
# Shared Feature Engineering (Context x)
# We reuse the "Rich Estimated" logic from Ch6 to ensure fair information.
# ---------------------------------------------------------------------------

def get_context_features(
    user: User,
    query: Query,
    products: List[Product],
    cfg: SimulatorConfig,
    base_scores: np.ndarray,
) -> np.ndarray:
    """Rich context features with *estimated* user latents (Ch6 Standard)."""
    
    # 1. One-hot encodings
    seg_vec = np.zeros(len(cfg.users.segments), dtype=float)
    if user.segment in cfg.users.segments:
        seg_vec[cfg.users.segments.index(user.segment)] = 1.0

    q_vec = np.zeros(len(cfg.queries.query_types), dtype=float)
    if query.query_type in cfg.queries.query_types:
        q_vec[cfg.queries.query_types.index(query.query_type)] = 1.0

    # 2. User preferences (Estimated / Quantized to simulate production uncertainty)
    raw_price = float(user.theta_price)
    raw_pl = float(user.theta_pl)
    shrunk_price = 0.7 * raw_price
    shrunk_pl = 0.7 * raw_pl
    theta_price_est = float(np.clip(np.round(shrunk_price * 2.0) / 2.0, -3.0, 3.0))
    theta_pl_est = float(np.clip(np.round(shrunk_pl * 2.0) / 2.0, -3.0, 3.0))

    # 3. Aggregate product features over base top-k
    # This gives the agent context about the *result set* (e.g. "is this result set expensive?")
    k = min(cfg.top_k, len(products))
    top_idx = np.argsort(-np.asarray(base_scores, dtype=float))[:k]
    top_products = [products[i] for i in top_idx]

    prices = np.array([p.price for p in top_products], dtype=float)
    avg_price = float(prices.mean())
    std_price = float(prices.std())
    avg_cm2 = float(np.mean([p.cm2 for p in top_products]))
    avg_discount = float(np.mean([p.discount for p in top_products]))
    frac_pl = float(np.mean([1.0 if p.is_pl else 0.0 for p in top_products]))
    frac_strategic = float(np.mean([1.0 if p.strategic_flag else 0.0 for p in top_products]))
    avg_bestseller = float(np.mean([p.bestseller for p in top_products]))
    avg_relevance = float(np.mean(np.asarray(base_scores, dtype=float)[top_idx]))

    raw = np.concatenate(
        [
            seg_vec,
            q_vec,
            [theta_price_est, theta_pl_est],
            [
                avg_price,
                std_price,
                avg_cm2,
                avg_discount,
                frac_pl,
                frac_strategic,
                avg_bestseller,
                avg_relevance,
            ],
        ]
    )
    
    # Simple Standardization (Hardcoded means/stds from Ch6 analysis)
    means = np.array([0]*len(seg_vec) + [0]*len(q_vec) + [0, 0] + 
                     [30.0, 15.0, 0.3, 0.1, 0.5, 0.2, 300.0, 5.0], dtype=float)
    stds = np.array([1]*len(seg_vec) + [1]*len(q_vec) + [1, 1] + 
                    [20.0, 10.0, 0.2, 0.1, 0.3, 0.2, 200.0, 2.0], dtype=float)

    standardized = (raw - means) / stds
    return standardized.astype(np.float32) # Return float32 for PyTorch compatibility

# ---------------------------------------------------------------------------
# Simulation Helpers
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    reward: float
    gmv: float
    orders: float

def simulate_template(
    template: BoostTemplate,
    cfg: SimulatorConfig,
    user: User,
    query: Query,
    products: List[Product],
    rng: np.random.Generator,
    base_scores: np.ndarray
) -> EpisodeResult:
    """Apply a discrete template and simulate."""
    boosts = template.apply(products)
    blended = base_scores + boosts
    ranking = np.argsort(-blended).tolist()
    
    outcome = behavior.simulate_session(
        user=user, query=query, ranking=ranking, catalog=products, config=cfg, rng=rng
    )
    r, bd = reward.compute_reward(
        ranking=ranking, clicks=outcome.clicks, buys=outcome.buys, catalog=products, config=cfg
    )
    return EpisodeResult(reward=r, gmv=bd.gmv, orders=sum(outcome.buys))

def simulate_weights(
    weights: np.ndarray,
    product_features_matrix: np.ndarray,
    cfg: SimulatorConfig,
    user: User,
    query: Query,
    products: List[Product],
    rng: np.random.Generator,
    base_scores: np.ndarray
) -> EpisodeResult:
    """Apply continuous weights and simulate."""
    # Score = Base + Features @ Weights
    # Clip weights to Physics Limits (a_max)
    w_clipped = np.clip(weights, -cfg.action.a_max, cfg.action.a_max)
    
    boosts = product_features_matrix @ w_clipped
    blended = base_scores + boosts
    ranking = np.argsort(-blended).tolist()
    
    outcome = behavior.simulate_session(
        user=user, query=query, ranking=ranking, catalog=products, config=cfg, rng=rng
    )
    r, bd = reward.compute_reward(
        ranking=ranking, clicks=outcome.clicks, buys=outcome.buys, catalog=products, config=cfg
    )
    return EpisodeResult(reward=r, gmv=bd.gmv, orders=sum(outcome.buys))

# ---------------------------------------------------------------------------
# Main Comparison Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark: Discrete LinUCB vs Continuous Q-Learner")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Continuous agent batch size")
    parser.add_argument("--train-interval", type=int, default=10, help="Continuous agent train interval")
    args = parser.parse_args()

    print(f"\nRunning Benchmark: Discrete LinUCB vs Continuous Q-Learner")
    print(f"Episodes: {args.episodes}, Seed: {args.seed}")
    print("-" * 60)

    # 1. Setup Environment
    cfg = SimulatorConfig(seed=args.seed)
    
    # CRITICAL FIX: Match action magnitude between Discrete (defaults to 5.0) and Continuous.
    # Default cfg.action.a_max is 0.5, which handicaps the continuous agent by 10x.
    cfg.action.a_max = 5.0 
    
    rng = np.random.default_rng(args.seed)
    products = catalog_module.generate_catalog(cfg.catalog, rng)
    
    # 2. Setup Discrete Agent (LinUCB)
    stats = compute_catalog_stats(products)
    templates = create_standard_templates(stats, a_max=5.0)
    
    # Determine context dimension by running one dummy pass
    dummy_u = users.sample_user(config=cfg, rng=rng)
    dummy_q = queries.sample_query(user=dummy_u, config=cfg, rng=rng)
    dummy_base = np.zeros(len(products))
    dummy_ctx = get_context_features(dummy_u, dummy_q, products, cfg, dummy_base)
    context_dim = len(dummy_ctx)
    print(f"Context Feature Dimension: {context_dim}")
    
    linucb = LinUCB(
        templates=templates,
        feature_dim=context_dim,
        config=LinUCBConfig(lambda_reg=1.0, alpha=1.0, seed=args.seed)
    )

    # 3. Setup Continuous Agent (Q-Ensemble + CEM)
    # Product feature dim comes from zoosim.ranking.features
    # We run one dummy product feature calc
    dummy_pf = features.compute_features(user=dummy_u, query=dummy_q, product=products[0], config=cfg)
    action_dim = len(dummy_pf)
    print(f"Action (Weight) Dimension: {action_dim}")

    q_policy = QEnsemblePolicy(
        state_dim=context_dim,
        action_dim=action_dim,
        a_max=cfg.action.a_max,
        q_config=QEnsembleConfig(
            n_ensembles=3, # Smaller ensemble for speed in demo
            hidden_sizes=(64,), # Shallow net for speed
            learning_rate=0.005,
            seed=args.seed
        ),
        ucb_beta=1.0 # Exploration
    )
    
    # Replay Buffer for Continuous Agent
    # Tuple: (state, action, reward)
    replay_buffer = deque(maxlen=5000)
    
    # Metrics
    metrics = {
        "linucb": {"reward": [], "gmv": []},
        "continuous": {"reward": [], "gmv": []}
    }

    # 4. Main Loop
    start_time = time.time()
    
    for i in range(1, args.episodes + 1):
        # A. Common Environment State
        user = users.sample_user(config=cfg, rng=rng)
        query = queries.sample_query(user=user, config=cfg, rng=rng)
        
        base_scores = np.asarray(
            relevance.batch_base_scores(query=query, catalog=products, config=cfg, rng=rng),
            dtype=float
        )
        
        ctx = get_context_features(user, query, products, cfg, base_scores)
        
        # B. Discrete Agent Step
        # ----------------------
        # 1. Select
        a_disc_idx = linucb.select_action(ctx.astype(np.float64)) # LinUCB expects float64 usually
        template = templates[a_disc_idx]
        
        # 2. Act
        # We fork the RNG for simulation to ensures purely 'parallel' universes if desired, 
        # but here we want them to face the *same* user behavior if they showed the *same* ranking.
        # Ideally, we'd use the same RNG state for the session simulation. 
        # For simplicity, we pass the main 'rng'. Since the rankings will likely differ, 
        # the random calls inside 'simulate_session' (e.g. click/buy probabilities) will drift.
        # This adds variance but is acceptable for large N.
        res_disc = simulate_template(template, cfg, user, query, products, rng, base_scores)
        
        # 3. Learn
        linucb.update(action=a_disc_idx, features=ctx.astype(np.float64), reward=res_disc.reward)
        
        metrics["linucb"]["reward"].append(res_disc.reward)
        metrics["linucb"]["gmv"].append(res_disc.gmv)

        # C. Continuous Agent Step
        # ------------------------
        # 1. Select (CEM optimization over Q-function)
        # Decay exploration beta
        beta = 2.0 * np.exp(-4.0 * i / args.episodes)
        q_policy.ucb_beta = beta
        
        action_weights = q_policy.select_action(ctx) # returns float32
        
        # 2. Act
        # Need product feature matrix for this specific User/Query
        # Optimization: Compute only once? No, product features depend on User/Query (e.g. personalization).
        # This is the slow part.
        prod_feats = [
            features.compute_features(user=user, query=query, product=p, config=cfg)
            for p in products
        ]
        pf_matrix = np.array(prod_feats, dtype=np.float32)
        
        # Standardize features (as per Env logic)
        if cfg.action.standardize_features:
            m = pf_matrix.mean(axis=0)
            s = pf_matrix.std(axis=0)
            s[s == 0] = 1.0
            pf_matrix = (pf_matrix - m) / s
            
        res_cont = simulate_weights(action_weights, pf_matrix, cfg, user, query, products, rng, base_scores)
        
        # 3. Learn
        replay_buffer.append((ctx, action_weights, res_cont.reward))
        
        metrics["continuous"]["reward"].append(res_cont.reward)
        metrics["continuous"]["gmv"].append(res_cont.gmv)
        
        # Train Q-Network
        if i % args.train_interval == 0 and len(replay_buffer) >= args.batch_size:
            # Sample batch
            batch_indices = np.random.choice(len(replay_buffer), args.batch_size, replace=False)
            batch_states = np.array([replay_buffer[k][0] for k in batch_indices])
            batch_actions = np.array([replay_buffer[k][1] for k in batch_indices])
            batch_rewards = np.array([replay_buffer[k][2] for k in batch_indices])
            
            q_policy.q.update_batch(batch_states, batch_actions, batch_rewards, n_epochs=2, batch_size=args.batch_size)

        # Progress Log
        if i % 100 == 0:
            # Compute trailing avg (last 100)
            win = 100
            d_r = np.mean(metrics["linucb"]["reward"][-win:])
            c_r = np.mean(metrics["continuous"]["reward"][-win:])
            d_gmv = np.mean(metrics["linucb"]["gmv"][-win:])
            c_gmv = np.mean(metrics["continuous"]["gmv"][-win:])
            
            print(f"Ep {i:5d} | LinUCB R:{d_r:6.2f} GMV:{d_gmv:6.2f} | Cont R:{c_r:6.2f} GMV:{c_gmv:6.2f} | Beta:{beta:.2f}")

    # 5. Final Summary
    print("\n" + "="*60)
    print("FINAL RESULTS (Averages over last 1000 episodes)")
    print("="*60)
    
    win = min(1000, args.episodes)
    
    d_r = np.mean(metrics["linucb"]["reward"][-win:])
    c_r = np.mean(metrics["continuous"]["reward"][-win:])
    d_gmv = np.mean(metrics["linucb"]["gmv"][-win:])
    c_gmv = np.mean(metrics["continuous"]["gmv"][-win:])
    
    print(f"{ 'Agent':<15} | {'Reward':>10} | {'GMV':>10} | {'Δ vs LinUCB':>12}")
    print("-" * 60)
    print(f"{ 'Discrete LinUCB':<15} | {d_r:10.2f} | {d_gmv:10.2f} | {'-':>12}")
    
    delta_gmv = (c_gmv - d_gmv) / d_gmv * 100
    print(f"{ 'Continuous Q':<15} | {c_r:10.2f} | {c_gmv:10.2f} | {delta_gmv:+11.1f}%")
    print("-" * 60)
    
    if c_gmv > d_gmv:
        print("\nSUCCESS: Continuous Agent outperformed Discrete Baseline.")
    else:
        print("\nNOTE: Continuous Agent did not outperform. Check tuning or run longer.")
        print("Possible reasons: Harder exploration in continuous space, limited linear capacity.")

if __name__ == "__main__":
    main()
