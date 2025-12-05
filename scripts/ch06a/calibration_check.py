#!/usr/bin/env python
"""Calibration Check for Neural Bandits.

Diagnoses if uncertainty estimates (sigma) correlate with true prediction error.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from zoosim.policies.q_ensemble import QEnsembleRegressor, QEnsembleConfig
from zoosim.core.config import SimulatorConfig
from zoosim.world import catalog as catalog_module
from zoosim.world.users import sample_user
from zoosim.world.queries import sample_query
from zoosim.ranking.relevance import batch_base_scores
from zoosim.dynamics.reward import compute_reward
from zoosim.policies.templates import compute_catalog_stats, create_standard_templates
from zoosim.dynamics import behavior

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

def collect_data(n_episodes, cfg, products, templates, rng):
    data = []
    for _ in range(n_episodes):
        user = sample_user(config=cfg, rng=rng)
        query = sample_query(user=user, config=cfg, rng=rng)
        context = get_raw_context(user, query, cfg)
        
        # Random action
        action_idx = rng.integers(0, len(templates))
        template = templates[action_idx]
        
        # Reward
        base_scores = batch_base_scores(query=query, catalog=products, config=cfg, rng=rng)
        boosts = template.apply(products)
        final_scores = np.array(base_scores) + np.array(boosts)
        ranking = np.argsort(final_scores)[::-1].tolist()
        outcome = behavior.simulate_session(user=user, query=query, ranking=ranking, catalog=products, config=cfg, rng=rng)
        r_val, _ = compute_reward(ranking=ranking, clicks=outcome.clicks, buys=outcome.buys, catalog=products, config=cfg)
        
        data.append((context, action_idx, r_val))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    args = parser.parse_args()
    
    print("="*60)
    print("Calibration Check")
    print("="*60)
    
    cfg = SimulatorConfig(seed=42)
    rng = np.random.default_rng(42)
    products = catalog_module.generate_catalog(cfg.catalog, rng)
    stats = compute_catalog_stats(products)
    templates = create_standard_templates(stats)
    n_actions = len(templates)

    # 1. Collect Data
    print("Collecting training data...")
    train_data = collect_data(args.n_train, cfg, products, templates, rng)
    print("Collecting test data...")
    test_data = collect_data(args.n_test, cfg, products, templates, rng)
    
    # 2. Train Ensemble
    input_dim = len(train_data[0][0])
    print(f"Training ensemble (Input dim: {input_dim}, Actions: {n_actions})...")
    
    ensemble = QEnsembleRegressor(
        state_dim=input_dim,
        action_dim=n_actions,
        config=QEnsembleConfig(n_ensembles=5, hidden_sizes=(64,64))
    )
    
    X_train = np.array([d[0] for d in train_data])
    A_train = np.zeros((len(train_data), n_actions))
    for i, (_, a_idx, _) in enumerate(train_data):
        A_train[i, a_idx] = 1.0
    R_train = np.array([d[2] for d in train_data])
    
    ensemble.update_batch(X_train, A_train, R_train, n_epochs=50, batch_size=64)
    
    # 3. Evaluate Calibration
    print("Evaluating calibration...")
    X_test = np.array([d[0] for d in test_data])
    A_test = np.zeros((len(test_data), n_actions))
    for i, (_, a_idx, _) in enumerate(test_data):
        A_test[i, a_idx] = 1.0
    R_test = np.array([d[2] for d in test_data])
    
    means, stds = ensemble.predict(X_test, A_test)
    
    errors = np.abs(means - R_test)
    correlation = np.corrcoef(stds, errors)[0, 1]
    
    print(f"\nCalibration Correlation (sigma vs |error|): {correlation:.4f}")
    
    if correlation > 0.3:
        print("✓ Positive correlation: Uncertainty estimates are informative.")
    else:
        print("⚠️ Low correlation: Uncertainty may be miscalibrated.")

    # Ascii plot
    print("\nScatter plot (Sigma vs Error):")
    indices = np.argsort(stds)
    sorted_stds = stds[indices]
    sorted_errors = errors[indices]
    
    # Binning for display
    n_bins = 10
    bins = np.array_split(np.arange(len(stds)), n_bins)
    print(f"{ 'Sigma Range':<20} | {'Mean Error':<10}")
    print("-" * 35)
    for b in bins:
        if len(b) == 0: continue
        s_min = sorted_stds[b[0]]
        s_max = sorted_stds[b[-1]]
        mean_err = np.mean(sorted_errors[b])
        print(f"{s_min:.3f} - {s_max:.3f}      | {mean_err:.3f}")

if __name__ == "__main__":
    main()
