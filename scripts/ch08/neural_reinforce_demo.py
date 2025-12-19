#!/usr/bin/env python
"""Chapter 8: Deep REINFORCE with End-to-End Feature Learning.

This script demonstrates "Modern" Deep RL:
1.  **Raw Inputs Only**: We disable the handcrafted "Rich Features".
    The agent sees only basic context (User Segment, Query Type, Category Embedding).
2.  **End-to-End Learning**: The Policy Network acts as a Neural Feature Extractor.
    It must *learn* to infer user preferences (e.g., "Price Hunter segment likes cheap items")
    directly from the reward signal, without explicit feature engineering.
3.  **Hardware Acceleration**: explicit support for MPS (Mac), CUDA (NVIDIA), and CPU.

Usage:
    python scripts/ch08/neural_reinforce_demo.py --episodes 5000 --device auto
"""

import argparse
import numpy as np
import torch
import gymnasium as gym
from collections import deque
import time

from zoosim.envs.gym_env import GymZooplusEnv
from zoosim.policies.reinforce import REINFORCEAgent, REINFORCEConfig
from zoosim.core.config import SimulatorConfig

def get_device(device_arg: str) -> str:
    """Resolve device argument to valid PyTorch device string."""
    if device_arg != "auto":
        return device_arg
    
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size for feature extraction")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda', 'mps'")
    parser.add_argument("--a_max", type=float, default=5.0)
    args = parser.parse_args()

    # 1. Hardware Setup
    device = get_device(args.device)
    print("="*60)
    print(f"Deep REINFORCE (End-to-End Learning)")
    print(f"Device: {device.upper()}")
    print(f"Episodes: {args.episodes} | Seed: {args.seed}")
    print(f"Network: Raw Input -> [{args.hidden_dim}, {args.hidden_dim}] -> Action")
    print("="*60)

    # 2. Setup Environment (Raw Features)
    # We explicitly disable rich features to force the net to learn them.
    cfg = SimulatorConfig(seed=args.seed)
    cfg.action.a_max = args.a_max 
    
    env = GymZooplusEnv(cfg=cfg, seed=args.seed, rich_features=False)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Observation Dim: {obs_dim} (Raw Context)")
    print(f"Action Dim: {action_dim}")

    # 3. Setup Deep Agent
    # We use a larger network to handle the feature extraction burden
    agent_config = REINFORCEConfig(
        hidden_sizes=(args.hidden_dim, args.hidden_dim), # Deeper/Wider net
        learning_rate=args.lr,
        gamma=0.99,
        entropy_coef=0.02, # Slightly higher entropy for raw exploration
        device=device,
        seed=args.seed
    )
    
    agent = REINFORCEAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_scale=args.a_max,
        config=agent_config
    )

    # 4. Training Loop
    recent_returns = deque(maxlen=100)
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(reward)
            obs = next_obs
            done = terminated or truncated
        
        metrics = agent.update()
        recent_returns.append(metrics["return_mean"])

        if ep % 200 == 0:
            elapsed = time.time() - start_time
            avg_ret = np.mean(recent_returns)
            print(f"Ep {ep:5d} | Return: {avg_ret:6.2f} | Loss: {metrics['loss']:6.2f} | {elapsed:.1f}s")

    print("\n" + "="*60)
    print(f"Final Average Return: {np.mean(recent_returns):.2f}")
    
    # Comparison Context
    print("-" * 60)
    print("Contextual Baselines (Approximate):")
    print("  Random Agent:      ~8.7")
    print("  Rich Features (Ch6/8): ~10.5 - 11.5")
    print("  Continuous Q (Ch7):    ~25.0")
    print("-" * 60)
    
    if np.mean(recent_returns) > 10.0:
        print("SUCCESS: Neural Network successfully learned to reconstruct useful features!")
    elif np.mean(recent_returns) > 9.0:
        print("PARTIAL: Learning exhibited but hasn't matched handcrafted features yet.")
    else:
        print("NOTE: Training unstable or insufficient. Deep RL from scratch is hard!")

if __name__ == "__main__":
    main()
