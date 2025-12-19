#!/usr/bin/env python
"""Chapter 12: SlateQ / Deep Learning to Rank Demo.

This script demonstrates a Deep RL approach to ranking (SlateQ-ish) where:
1.  The agent receives (Context, CandidateList).
2.  The agent uses a Deep Network to score each candidate: Q(s, i).
3.  The slate is formed by sorting the candidates by score.
4.  The network is trained to predict the slate reward via decomposition: Q(Slate) = Sum(Q(item)).

This "Deep Ranker" overcomes the limitations of REINFORCE (Ch8) by being value-based
(sample efficient) and handling the ranking structure explicitly.
"""

import argparse
import numpy as np
import torch
from collections import deque
import time

from zoosim.envs.slate_gym_env import SlateGymEnv
from zoosim.policies.slate_q import SlateQAgent
from zoosim.core.config import SimulatorConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # 1. Setup Environment
    print("="*60)
    print("Chapter 12: Neural Slate Ranking (Deep Q-Network)")
    print(f"Episodes: {args.episodes} | Seed: {args.seed}")
    print("="*60)

    cfg = SimulatorConfig(seed=args.seed)
    env = SlateGymEnv(cfg=cfg, seed=args.seed, max_candidates=50, item_feature_dim=10)
    
    # Observation shapes
    # Context: 10 (Segment+Query+Prefs)
    # Item: 10 (Features)
    context_dim = 10
    item_dim = 10
    
    # 2. Setup Agent
    agent = SlateQAgent(
        context_dim=context_dim,
        item_dim=item_dim,
        lr=args.lr,
        device=args.device,
        batch_size=64
    )

    # 3. Training Loop
    recent_returns = deque(maxlen=100)
    losses = deque(maxlen=100)
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        
        # Select Action (Scores for all 50 candidates)
        action_scores = agent.select_action(obs)
        
        # Step
        next_obs, reward, terminated, truncated, info = env.step(action_scores)
        
        # Store
        # We store the components needed for the update
        # Note: In bandit setting, 'next_obs' is not used for bootstrapping 
        # unless we are doing multi-step. Here we treat as bandit.
        agent.store_transition(
            obs['context'], 
            obs['candidate_features'], 
            action_scores, 
            reward
        )
        
        # Update
        metrics = agent.update()
        if "loss" in metrics:
            losses.append(metrics["loss"])
            
        recent_returns.append(reward)

        if ep % 100 == 0:
            elapsed = time.time() - start_time
            avg_ret = np.mean(recent_returns)
            avg_loss = np.mean(losses) if losses else 0.0
            print(f"Ep {ep:5d} | Return: {avg_ret:6.2f} | Loss: {avg_loss:6.4f} | Eps: {agent.epsilon:.2f}")

    print("\n" + "="*60)
    print(f"Final Average Return: {np.mean(recent_returns):.2f}")
    
    # Comparison (Approximate from Ch8 analysis)
    print("-" * 60)
    print("Baselines:")
    print("  Random Agent:      ~8.7")
    print("  REINFORCE (Ch8):   ~11.5 - 15.0")
    print("  Continuous Q (Ch7): ~25.0")
    print("-" * 60)
    
    if np.mean(recent_returns) > 15.0:
        print("SUCCESS: Neural Slate Ranker is competitive!")
    else:
        print("NOTE: Further tuning required (or longer training).")

if __name__ == "__main__":
    main()
