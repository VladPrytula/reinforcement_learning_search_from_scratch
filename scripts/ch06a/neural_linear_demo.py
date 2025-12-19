#!/usr/bin/env python
"""Chapter 6A: Neural Linear Bandits Demo.

Demonstrates:
1. Neural Feature Extractor (representation learning).
2. Neural Linear UCB (learned features + linear heads).
3. Comparison with Ch6 rich features.

Usage:
    python scripts/ch06a/neural_linear_demo.py --n-pretrain 5000 --n-bandit 10000
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional, Dict

from zoosim.core.config import SimulatorConfig
from zoosim.world import catalog as catalog_module
from zoosim.world.users import sample_user
from zoosim.world.queries import sample_query
from zoosim.ranking.relevance import batch_base_scores
from zoosim.dynamics.reward import compute_reward
from zoosim.policies.templates import compute_catalog_stats, create_standard_templates
from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
from zoosim.dynamics import behavior

# --------------------------------------------------------------------------
# 1. Neural Feature Extractor
# --------------------------------------------------------------------------

class NeuralFeatureExtractor(nn.Module):
    """Neural network for representation learning.

    Maps raw context x -> learned features f_ψ(x).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def pretrain_feature_extractor(
    logged_data: List[Tuple[np.ndarray, int, float]],
    input_dim: int,
    n_actions: int,
    hidden_dim: int = 64,
    output_dim: int = 20,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    seed: int = 42,
    verbose: bool = True
) -> NeuralFeatureExtractor:
    """Pretrain neural feature extractor on logged data."""
    torch.manual_seed(seed)
    
    # Prepare data
    X = torch.tensor(np.array([x for x, a, r in logged_data]), dtype=torch.float32)
    A = torch.tensor(np.array([a for x, a, r in logged_data]), dtype=torch.long)
    R = torch.tensor(np.array([r for x, a, r in logged_data]), dtype=torch.float32)

    dataset = TensorDataset(X, A, R)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Network + Heads
    feature_extractor = NeuralFeatureExtractor(input_dim, hidden_dim, output_dim)
    prediction_heads = nn.ModuleList([
        nn.Linear(output_dim, 1) for _ in range(n_actions)
    ])

    optimizer = optim.Adam(
        list(feature_extractor.parameters()) + list(prediction_heads.parameters()),
        lr=learning_rate
    )
    loss_fn = nn.MSELoss()

    if verbose:
        print(f"\nPretraining on {len(logged_data)} episodes...")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for x_batch, a_batch, r_batch in loader:
            features = feature_extractor(x_batch)
            # Select head for each action
            preds = torch.stack([
                prediction_heads[a](features[i]) 
                for i, a in enumerate(a_batch)
            ]).squeeze()
            
            loss = loss_fn(preds, r_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(loader):.4f}")

    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
        
    return feature_extractor

# --------------------------------------------------------------------------
# 2. Neural Linear UCB Policy
# --------------------------------------------------------------------------

class NeuralLinearUCB(LinUCB):
    """LinUCB using learned neural features."""
    def __init__(
        self,
        templates: list,
        feature_extractor: NeuralFeatureExtractor,
        raw_feature_dim: int, # Dimension of input to neural net
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        seed: int = 42
    ):
        self.feature_extractor = feature_extractor
        # Feature dim for LinUCB is the output of the neural net
        super().__init__(
            templates, 
            feature_dim=feature_extractor.output_dim, 
            config=LinUCBConfig(alpha=alpha, lambda_reg=lambda_reg, seed=seed)
        )
        self.raw_feature_dim = raw_feature_dim

    def select_action_from_raw(self, raw_features: np.ndarray) -> int:
        """Select action given raw context."""
        # 1. Extract neural features
        with torch.no_grad():
            x_tensor = torch.tensor(raw_features, dtype=torch.float32)
            neural_features = self.feature_extractor(x_tensor).numpy()
        
        # 2. Use LinUCB logic
        return self.select_action(neural_features)

    def update_from_raw(self, action: int, raw_features: np.ndarray, reward: float):
        """Update using raw context."""
        with torch.no_grad():
            x_tensor = torch.tensor(raw_features, dtype=torch.float32)
            neural_features = self.feature_extractor(x_tensor).numpy()
        
        self.update(action, neural_features, reward)

# --------------------------------------------------------------------------
# 3. Experiment Helpers
# --------------------------------------------------------------------------

def get_raw_context(user, query, cfg) -> np.ndarray:
    """
    Construct raw context vector.
    Simple concatenation of segment (one-hot), query type (one-hot), and user latents.
    """
    # Segment one-hot
    seg_idx = cfg.users.segments.index(user.segment)
    seg_vec = np.zeros(len(cfg.users.segments))
    seg_vec[seg_idx] = 1.0
    
    # Query type one-hot
    q_idx = cfg.queries.query_types.index(query.query_type)
    q_vec = np.zeros(len(cfg.queries.query_types))
    q_vec[q_idx] = 1.0
    
    # Raw Latents (assuming we have access for "raw" features in this experiment)
    # In a real setting, these might be embeddings or history features.
    latents = np.array([user.theta_price, user.theta_pl])
    
    return np.concatenate([seg_vec, q_vec, latents])

def run_episode(env_components, policy, templates, rng, get_context_fn):
    cfg, products = env_components
    
    # 1. Environment: User/Query
    user = sample_user(config=cfg, rng=rng)
    query = sample_query(user=user, config=cfg, rng=rng)
    
    # 2. Policy: Context -> Action
    raw_context = get_context_fn(user, query, cfg)
    
    if isinstance(policy, NeuralLinearUCB):
        action_idx = policy.select_action_from_raw(raw_context)
    elif hasattr(policy, 'select_action'): # Random or other
         # For random policy, we might not use context, but interface might expect it
         # If it's a simple policy, we handle it.
         if isinstance(policy, RandomPolicy):
             action_idx = policy.select_action(rng)
         else:
             # Assume LinUCB with manual features if not NeuralLinear
             # But wait, LinUCB expects 'features'
             # For comparison, if we pass LinUCB, we need feature extractor outside
             # Let's assume we handle specific classes or generic interface
             action_idx = policy.select_action(raw_context) # Generic
    else:
        action_idx = rng.integers(0, len(templates))

    template = templates[action_idx]
    
    # 3. Environment: Response
    base_scores = batch_base_scores(query=query, catalog=products, config=cfg, rng=rng)
    boosts = template.apply(products)
    final_scores = np.array(base_scores) + np.array(boosts)
    ranking = np.argsort(final_scores)[::-1].tolist()
    
    # Simulate session
    outcome = behavior.simulate_session(user=user, query=query, ranking=ranking, catalog=products, config=cfg, rng=rng)
    r_val, _ = compute_reward(ranking=ranking, clicks=outcome.clicks, buys=outcome.buys, catalog=products, config=cfg)
    
    # 4. Policy: Update
    if isinstance(policy, NeuralLinearUCB):
        policy.update_from_raw(action_idx, raw_context, r_val)
    elif hasattr(policy, 'update'):
        policy.update(action_idx, raw_context, r_val)
        
    return raw_context, action_idx, r_val

class RandomPolicy:
    def __init__(self, n_actions):
        self.n_actions = n_actions
    def select_action(self, rng):
        return rng.integers(0, self.n_actions)
    def update(self, *args):
        pass

# --------------------------------------------------------------------------
# 4. Main Execution
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pretrain", type=int, default=2000, help="Episodes for pretraining")
    parser.add_argument("--n-bandit", type=int, default=5000, help="Episodes for bandit eval")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("="*60)
    print("Neural Linear Bandits Demo")
    print(f"Pretrain: {args.n_pretrain}, Bandit: {args.n_bandit}")
    print("="*60)

    # Setup Environment
    cfg = SimulatorConfig(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    products = catalog_module.generate_catalog(cfg.catalog, rng)
    stats = compute_catalog_stats(products)
    templates = create_standard_templates(stats)
    n_actions = len(templates)
    env_components = (cfg, products)

    # 1. Collect Data for Pretraining
    print("\n[1/3] Collecting logged data with Random Policy...")
    logged_data = []
    rand_policy = RandomPolicy(n_actions)
    
    # Dummy context to check dim
    dummy_u = sample_user(config=cfg, rng=rng)
    dummy_q = sample_query(user=dummy_u, config=cfg, rng=rng)
    dummy_ctx = get_raw_context(dummy_u, dummy_q, cfg)
    input_dim = len(dummy_ctx)
    print(f"Raw Context Dimension: {input_dim}")

    for _ in range(args.n_pretrain):
        x, a, r = run_episode(env_components, rand_policy, templates, rng, get_raw_context)
        logged_data.append((x, a, r))

    # 2. Train Feature Extractor
    print("\n[2/3] Training Neural Feature Extractor...")
    feature_extractor = pretrain_feature_extractor(
        logged_data, 
        input_dim=input_dim, 
        n_actions=n_actions,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        seed=args.seed
    )

    # 3. Run Neural Linear Bandit
    print("\n[3/3] Running Neural Linear UCB...")
    nl_policy = NeuralLinearUCB(
        templates, 
        feature_extractor, 
        raw_feature_dim=input_dim,
        seed=args.seed
    )
    
    rewards = []
    for i in range(args.n_bandit):
        _, _, r = run_episode(env_components, nl_policy, templates, rng, get_raw_context)
        rewards.append(r)
        
        if (i+1) % 1000 == 0:
            print(f"  Episode {i+1}/{args.n_bandit}: Mean Reward (last 1k) = {np.mean(rewards[-1000:]):.3f}")

    print("\nDone! Final Mean Reward (last 1k): {:.3f}".format(np.mean(rewards[-1000:])))

if __name__ == "__main__":
    main()
