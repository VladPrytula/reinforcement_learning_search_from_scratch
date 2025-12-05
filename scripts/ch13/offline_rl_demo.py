#!/usr/bin/env python
"""Chapter 13: Offline RL (Conservative Q-Learning) Demo.

This script demonstrates learning from logged data (Offline RL).
1.  **Data Collection**: We run a Random Policy to collect a dataset of (s, a, r).
2.  **Training**: We train two agents on this fixed dataset:
    - Standard Q-Learning (Naive).
    - Conservative Q-Learning (CQL).
3.  **Evaluation**: We test both agents in the live environment.

Hypothesis: Standard Q-Learning will overestimate values for OOD actions and fail.
CQL will remain conservative and perform better.
"""

import argparse
import numpy as np
import torch
from collections import deque
import time

from zoosim.envs.gym_env import GymZooplusEnv
from zoosim.policies.offline.cql import CQLAgent
from zoosim.core.config import SimulatorConfig

def collect_dataset(env, episodes=1000):
    print(f"Collecting data from {episodes} episodes (Random Policy)...")
    dataset = []
    obs, _ = env.reset()
    for _ in range(episodes):
        # Random action
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        dataset.append((obs, action, reward, next_obs, terminated))
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    return dataset

def evaluate(agent, env, episodes=100):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs
            done = terminated or truncated
            if done:
                returns.append(reward)
    return np.mean(returns)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-episodes", type=int, default=2000)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("="*60)
    print("Chapter 13: Offline RL (CQL vs Naive Q)")
    print("="*60)

    # 1. Setup
    cfg = SimulatorConfig(seed=args.seed)
    cfg.action.a_max = 2.0 # Tighter bounds for this demo
    env = GymZooplusEnv(cfg=cfg, seed=args.seed, rich_features=True) # Use Rich features for better generalization
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 2. Collect Data
    dataset = collect_dataset(env, args.data_episodes)
    print(f"Dataset size: {len(dataset)}")
    
    # Prepare batches
    states = np.array([x[0] for x in dataset])
    actions = np.array([x[1] for x in dataset])
    rewards = np.array([x[2] for x in dataset])
    next_states = np.array([x[3] for x in dataset])
    dones = np.array([x[4] for x in dataset])

    # 3. Train Agents
    print("\nTraining Agents...")
    
    # Naive Q (Alpha=0)
    naive_agent = CQLAgent(obs_dim, action_dim, action_max=2.0, alpha=0.0, device=args.device)
    # CQL (Alpha=1.0)
    cql_agent = CQLAgent(obs_dim, action_dim, action_max=2.0, alpha=5.0, device=args.device)
    
    batch_size = 64
    
    for i in range(args.train_steps):
        idx = np.random.choice(len(dataset), batch_size)
        
        # Train Naive
        naive_agent.update(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])
        
        # Train CQL
        cql_agent.update(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])
        
        if i % 1000 == 0:
            print(f"Step {i}")

    # 4. Evaluate
    print("\nEvaluating...")
    random_score = evaluate(type('RandomAgent', (), {'select_action': lambda self, x: env.action_space.sample()})(), env)
    print(f"Random Policy: {random_score:.2f}")
    
    naive_score = evaluate(naive_agent, env)
    print(f"Naive Q-Learning: {naive_score:.2f}")
    
    cql_score = evaluate(cql_agent, env)
    print(f"CQL: {cql_score:.2f}")
    
    print("-" * 60)
    if cql_score > naive_score:
        print("SUCCESS: CQL outperformed Naive Q-Learning on offline data.")
    else:
        print("NOTE: CQL performance similar or worse (check hyperparams alpha).")

if __name__ == "__main__":
    main()
