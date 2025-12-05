#!/usr/bin/env python
"""Chapter 8: Policy Gradients (REINFORCE) Demo.

This script demonstrates:
1.  Using the `GymZooplusEnv` (Standard Gym API).
2.  Training a Gaussian Policy using REINFORCE [ALG-8.1].
3.  Analyzing learning curves and entropy collapse.

We solve the optimization problem:
    max_θ E[ Σ γ^t r(s_t, a_t) ]
by ascending the gradient of the expected return.

Usage:
    python scripts/ch08/reinforce_demo.py --episodes 5000
"""

import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

from zoosim.envs.gym_env import GymZooplusEnv
from zoosim.policies.reinforce import REINFORCEAgent, REINFORCEConfig
from zoosim.core.config import SimulatorConfig

def moving_average(data, window=100):
    return np.convolve(data, np.ones(window), 'valid') / window

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--a_max", type=float, default=5.0, help="Action magnitude limit (default: 5.0).")
    parser.add_argument("--rich-features", action="store_true", default=True, help="Use rich context features (default: True).")
    args = parser.parse_args()

    print("="*60)
    print(f"Chapter 8: REINFORCE on Zooplus Search")
    print(f"Episodes: {args.episodes} | Seed: {args.seed}")
    print(f"Features: {'Rich (Dim 17)' if args.rich_features else 'Standard'}")
    print(f"Action Limit: {args.a_max} | LR: {args.lr}")
    print("="*60)

    # 1. Setup Environment
    # Note: We stick to the default SimulatorConfig for consistency with previous chapters
    cfg = SimulatorConfig(seed=args.seed)
    # Ensure reasonable action bounds for continuous control
    cfg.action.a_max = args.a_max 
    
    env = GymZooplusEnv(cfg=cfg, seed=args.seed, rich_features=args.rich_features)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Observation Dim: {obs_dim}")
    print(f"Action Dim: {action_dim} (Continuous)")

    # 2. Setup Agent
    agent_config = REINFORCEConfig(
        learning_rate=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy,
        seed=args.seed
    )
    agent = REINFORCEAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_scale=cfg.action.a_max, # Not strictly used by internal Gaussian, but good context
        config=agent_config
    )

    # 3. Training Loop
    returns = []
    entropies = []
    losses = []
    
    # For tracking success
    recent_returns = deque(maxlen=100)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        
        # Episode loop
        while not done:
            action = agent.select_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            agent.store_reward(reward)
            obs = next_obs
            done = terminated or truncated
        
        # Update policy
        metrics = agent.update()
        
        returns.append(metrics["return_mean"])
        entropies.append(metrics["entropy"])
        losses.append(metrics["loss"])
        recent_returns.append(metrics["return_mean"])

        if ep % 100 == 0:
            avg_ret = np.mean(recent_returns)
            print(f"Ep {ep:5d} | Return: {avg_ret:6.2f} | Entropy: {metrics['entropy']:5.2f} | Loss: {metrics['loss']:6.2f}")

    env.close()

    # 4. Analysis / Plotting (ASCII for CLI)
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    print(f"Final 100-ep Average Return: {np.mean(recent_returns):.2f}")
    
    # Simple check for learning
    first_100 = np.mean(returns[:100])
    last_100 = np.mean(returns[-100:])
    print(f"Improvement: {first_100:.2f} -> {last_100:.2f}")
    
    if last_100 > first_100:
        print("\u2713 Agent learned to improve policy.")
    else:
        print("! Agent failed to improve (check hyperparameters or variance).")

if __name__ == "__main__":
    main()
