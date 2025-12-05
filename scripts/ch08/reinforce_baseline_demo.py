#!/usr/bin/env python
"""Chapter 8: REINFORCE with Baseline Demo.

Demonstrates variance reduction using a learned state-value baseline V(s).
This addresses the instability observed in vanilla REINFORCE (reinforce_demo.py).

Algorithm:
    A_t = G_t - V(s_t)
    Δθ = α * ∇logπ(a|s) * A_t
    Δw = β * ∇(G_t - V(s_t))^2

Usage:
    python scripts/ch08/reinforce_baseline_demo.py --episodes 5000
"""

import argparse
import numpy as np
import gymnasium as gym
from collections import deque

from zoosim.envs.gym_env import GymZooplusEnv
from zoosim.policies.reinforce_baseline import REINFORCEBaselineAgent, REINFORCEBaselineConfig
from zoosim.core.config import SimulatorConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--a_max", type=float, default=5.0)
    parser.add_argument("--rich-features", action="store_true", default=True)
    args = parser.parse_args()

    print("="*60)
    print(f"Chapter 8: REINFORCE + Baseline (Variance Reduction)")
    print(f"Episodes: {args.episodes} | Seed: {args.seed}")
    print(f"Policy LR: {args.lr} | Value LR: {args.value_lr}")
    print("="*60)

    # 1. Environment Setup
    cfg = SimulatorConfig(seed=args.seed)
    cfg.action.a_max = args.a_max
    env = GymZooplusEnv(cfg=cfg, seed=args.seed, rich_features=args.rich_features)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. Agent Setup
    config = REINFORCEBaselineConfig(
        learning_rate=args.lr,
        value_lr=args.value_lr,
        gamma=args.gamma,
        entropy_coef=args.entropy,
        seed=args.seed
    )
    
    agent = REINFORCEBaselineAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_scale=args.a_max,
        config=config
    )

    # 3. Training Loop
    recent_returns = deque(maxlen=100)
    
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.store_reward(reward)
            obs = next_obs
            done = term or trunc
            
        metrics = agent.update()
        recent_returns.append(metrics["return_mean"])
        
        if ep % 100 == 0:
            avg_return = np.mean(recent_returns)
            print(f"Ep {ep:5d} | Return: {avg_return:6.2f} | "
                  f"Entropy: {metrics['entropy']:5.2f} | "
                  f"ValLoss: {metrics['value_loss']:6.2f}")

    env.close()
    print("\nTraining Complete.")
    print(f"Final 100-ep Average Return: {np.mean(recent_returns):.2f}")

if __name__ == "__main__":
    main()
