#!/usr/bin/env python
"""Part IV Closing: Global Algorithm Comparison.

This script runs a "Battle Royale" between all major algorithms covered in the book:
1.  **Baselines**: Static (Neutral), Random.
2.  **Bandits (Ch6)**: LinUCB (Discrete Templates).
3.  **Value-Based (Ch7)**: Continuous Q-Learning (Q-Ensemble + CEM).
4.  **Policy Gradient (Ch8)**: REINFORCE.
5.  **Deep Ranking (Ch12)**: SlateQ (Neural Ranker).
6.  **Offline RL (Ch13)**: CQL (Conservative Q-Learning).

Goal: Demonstrate the progression from simple heuristics to advanced Deep RL.
"""

import argparse
import numpy as np
import torch
import time
from collections import deque
from typing import List, Dict, Any

from zoosim.core.config import SimulatorConfig
from zoosim.envs.gym_env import GymZooplusEnv
from zoosim.envs.slate_gym_env import SlateGymEnv

# Agents
from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
from zoosim.policies.q_ensemble import QEnsemblePolicy, QEnsembleConfig
from zoosim.policies.reinforce import REINFORCEAgent, REINFORCEConfig
from zoosim.policies.slate_q import SlateQAgent
from zoosim.policies.offline.cql import CQLAgent
from zoosim.policies.templates import BoostTemplate

# ==============================================================================
# Wrappers & Adapters
# ==============================================================================

class AgentWrapper:
    """Common interface for evaluation."""
    def __init__(self, name):
        self.name = name
    
    def select_action(self, obs):
        raise NotImplementedError
    
    def update(self, obs, action, reward, next_obs, done):
        pass

class StaticAgent(AgentWrapper):
    def __init__(self, action_dim):
        super().__init__("Static (Neutral)")
        self.action = np.zeros(action_dim, dtype=np.float32)
    
    def select_action(self, obs):
        return self.action

class RandomAgent(AgentWrapper):
    def __init__(self, action_space):
        super().__init__("Random (Noise)")
        self.action_space = action_space
    
    def select_action(self, obs):
        return self.action_space.sample()

class LinUCBAdapter(AgentWrapper):
    """Adapts Discrete LinUCB to Continuous Env via Templates."""
    def __init__(self, feature_dim, templates):
        super().__init__("LinUCB (Ch6)")
        self.templates = templates
        self.agent = LinUCB(templates, feature_dim, config=LinUCBConfig(alpha=0.5))
        
    def select_action(self, obs):
        # obs is (feature_dim,)
        self.last_idx = self.agent.select_action(obs)
        # Return the weight vector corresponding to the template
        return np.array(self.templates[self.last_idx].weights, dtype=np.float32)
    
    def update(self, obs, action, reward, next_obs, done):
        # LinUCB only needs (feature, action_idx, reward)
        self.agent.update(self.last_idx, obs, reward)

class QEnsembleAdapter(AgentWrapper):
    """Adapts Continuous Q-Ensemble."""
    def __init__(self, state_dim, action_dim, a_max):
        super().__init__("Cont. Q-Learning (Ch7)")
        # Lower LR for stability in "Battle Royale"
        q_cfg = QEnsembleConfig(learning_rate=1e-3)
        self.agent = QEnsemblePolicy(state_dim, action_dim, a_max, q_config=q_cfg, ucb_beta=1.0)
        self.buffer = []
        self.batch_size = 64
        
    def select_action(self, obs):
        return self.agent.select_action(obs)
    
    def update(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward))
        if len(self.buffer) >= self.batch_size:
            # Train
            batch = self.buffer[-self.batch_size:]
            states = np.array([x[0] for x in batch])
            actions = np.array([x[1] for x in batch])
            rewards = np.array([x[2] for x in batch])
            self.agent.q.update_batch(states, actions, rewards, n_epochs=5)
            self.buffer = [] 

class REINFORCEAdapter(AgentWrapper):
    def __init__(self, obs_dim, action_dim, a_max):
        super().__init__("REINFORCE (Ch8)")
        # Reduced LR to prevent collapse
        cfg = REINFORCEConfig(learning_rate=3e-4, entropy_coef=0.05)
        self.agent = REINFORCEAgent(obs_dim, action_dim, a_max, config=cfg)
        
    def select_action(self, obs):
        return self.agent.select_action(obs)
    
    def update(self, obs, action, reward, next_obs, done):
        self.agent.store_reward(reward)
        if done:
            self.agent.update()

class SlateQAdapter(AgentWrapper):
    """Adapts SlateQ. Note: This requires SlateGymEnv."""
    def __init__(self, context_dim, item_dim):
        super().__init__("SlateQ (Ch12)")
        self.agent = SlateQAgent(context_dim, item_dim, lr=5e-4) # Lower LR
        self.last_scores = None
        self.last_context = None
        self.last_candidates = None

    def select_action(self, obs):
        # obs is Dict
        self.last_context = obs['context']
        self.last_candidates = obs['candidate_features']
        self.last_scores = self.agent.select_action(obs)
        return self.last_scores
    
    def update(self, obs, action, reward, next_obs, done):
        # SlateQ needs the stored transition components
        self.agent.store_transition(
            self.last_context, 
            self.last_candidates, 
            self.last_scores, 
            reward
        )
        self.agent.update()

class CQLAdapter(AgentWrapper):
    """Adapts CQL (Offline)."""
    def __init__(self, obs_dim, action_dim, a_max, dataset=None):
        super().__init__("CQL (Ch13 - Offline)")
        self.agent = CQLAgent(obs_dim, action_dim, a_max, alpha=1.0)
        
        if dataset:
            print(f"  [CQL] Pre-training on {len(dataset)} transitions...")
            self._train(dataset)
            
    def _train(self, dataset, steps=1000):
        states = np.array([x[0] for x in dataset])
        actions = np.array([x[1] for x in dataset])
        rewards = np.array([x[2] for x in dataset])
        next_states = np.array([x[3] for x in dataset])
        dones = np.array([x[4] for x in dataset])
        
        batch_size = 64
        for _ in range(steps):
            idx = np.random.choice(len(dataset), batch_size)
            self.agent.update(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])

    def select_action(self, obs):
        return self.agent.select_action(obs)

# ==============================================================================
# Comparison Engine
# ==============================================================================

def run_trial(agent_name, agent_wrapper, env, episodes=200, is_slate=False):
    returns = []
    start_time = time.time()
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            action = agent_wrapper.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent_wrapper.update(obs, action, reward, next_obs, done)
            
            obs = next_obs
            ep_reward += reward
            
        returns.append(ep_reward)
        
    elapsed = time.time() - start_time
    
    # Smooth variance for reporting
    window = min(50, len(returns))
    final_smooth = np.mean(returns[-window:])
    
    return {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "final_smooth": final_smooth,
        "time": elapsed
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000) # Increased to 1000
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    print("="*80)
    print("ZOOSIM GLOBAL ALGORITHM COMPARISON (PART IV CLOSING)")
    print("="*80)
    print(f"Comparing algorithms on Zooplus Search Task.")
    print(f"Episodes per Agent: {args.episodes}")
    print(f"Random Seed: {args.seed}")
    print("-" * 80)

    results = {}
    
    # --------------------------------------------------------------------------
    # 1. Setup Standard Environment (Continuous Weights)
    # --------------------------------------------------------------------------
    cfg = SimulatorConfig(seed=args.seed)
    cfg.action.a_max = 3.0
    env = GymZooplusEnv(cfg=cfg, seed=args.seed, rich_features=True)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --------------------------------------------------------------------------
    # Agent 0: Static (Neutral)
    # --------------------------------------------------------------------------
    print(f"> Running Static (Neutral) Agent...")
    agent = StaticAgent(action_dim)
    results["Static"] = run_trial("Static", agent, env, args.episodes)

    # --------------------------------------------------------------------------
    # Agent 1: Random
    # --------------------------------------------------------------------------
    print(f"> Running Random Agent (Noise)...")
    agent = RandomAgent(env.action_space)
    results["Random"] = run_trial("Random", agent, env, args.episodes)

    # --------------------------------------------------------------------------
    # Agent 2: LinUCB (Discrete Templates)
    # --------------------------------------------------------------------------
    print(f"> Running LinUCB (Ch6)...")
    # Define Templates (Simple heuristics mapped to feature weights)
    # Features: [cm2, discount, is_pl, pers, bestseller, price, litter, price_inter, pl_inter, spec_inter]
    # We map "concepts" to weight vectors.
    
    def make_tmpl(name, weights):
        # Create a valid BoostTemplate (dummy fn)
        t = BoostTemplate(
            id=0, 
            name=name, 
            description=f"Weights: {weights}", 
            boost_fn=lambda p: 0.0
        )
        # Attach weights for the Adapter to use
        t.weights = weights
        return t

    templates = [
        make_tmpl("Neutral", [0.0]*action_dim),
        # Boost CM2 (index 0)
        make_tmpl("Revenue", [3.0] + [0.0]*(action_dim-1)),
        # Boost Discount (index 1)
        make_tmpl("Conversion", [0.0, 3.0] + [0.0]*(action_dim-2)),
        # Boost PL (index 2)
        make_tmpl("OwnBrand", [0.0, 0.0, 3.0] + [0.0]*(action_dim-3))
    ]
    
    # Fix weights length safety
    for t in templates:
        if len(t.weights) < action_dim:
            t.weights += [0.0] * (action_dim - len(t.weights))
            
    agent = LinUCBAdapter(obs_dim, templates)
    results["LinUCB"] = run_trial("LinUCB", agent, env, args.episodes)

    # --------------------------------------------------------------------------
    # Agent 3: Continuous Q-Learning (Ch7)
    # --------------------------------------------------------------------------
    print(f"> Running Continuous Q-Learning (Ch7)...")
    agent = QEnsembleAdapter(obs_dim, action_dim, cfg.action.a_max)
    results["ContQ"] = run_trial("ContQ", agent, env, args.episodes)
    
    # --------------------------------------------------------------------------
    # Agent 4: REINFORCE (Ch8)
    # --------------------------------------------------------------------------
    print(f"> Running REINFORCE (Ch8)...")
    agent = REINFORCEAdapter(obs_dim, action_dim, cfg.action.a_max)
    results["REINFORCE"] = run_trial("REINFORCE", agent, env, args.episodes)

    # --------------------------------------------------------------------------
    # Agent 5: Offline CQL (Ch13)
    # --------------------------------------------------------------------------
    print(f"> Running CQL (Ch13 - Offline)...")
    # 1. Collect Data
    print("  Collecting offline data (500 eps)...")
    dataset = []
    tmp_obs, _ = env.reset()
    for _ in range(500):
        act = env.action_space.sample()
        nxt, rew, term, trunc, _ = env.step(act)
        dataset.append((tmp_obs, act, rew, nxt, term or trunc))
        tmp_obs = nxt if not (term or trunc) else env.reset()[0]
    
    agent = CQLAdapter(obs_dim, action_dim, cfg.action.a_max, dataset)
    results["CQL"] = run_trial("CQL", agent, env, args.episodes)

    # --------------------------------------------------------------------------
    # Agent 6: SlateQ (Ch12 - Deep Ranking)
    # --------------------------------------------------------------------------
    print(f"> Running SlateQ (Ch12 - Deep Ranking)...")
    # Needs Slate Environment
    slate_env = SlateGymEnv(cfg=cfg, seed=args.seed, max_candidates=50, item_feature_dim=10)
    # Obs dims
    ctx_dim = 8
    itm_dim = 10
    
    agent = SlateQAdapter(ctx_dim, itm_dim)
    results["SlateQ"] = run_trial("SlateQ", agent, slate_env, args.episodes, is_slate=True)


    # ==============================================================================
    # Final Summary
    # ==============================================================================
    print("\n" + "="*80)
    print(f"{ 'ALGORITHM':<20} | { 'AVG RETURN':<12} | { 'FINAL 50 (SMOOTH)':<18} | { 'TIME (s)':<8}")
    print("-" * 80)
    
    # Sort by final performance
    sorted_res = sorted(results.items(), key=lambda x: x[1]['final_smooth'], reverse=True)
    
    for name, metrics in sorted_res:
        print(f"{name:<20} | {metrics['mean_return']:12.2f} | {metrics['final_smooth']:18.2f} | {metrics['time']:8.2f}")
    
    print("-" * 80)
    print("INTERPRETATION:")
    print("1. STATIC (Neutral): The floor baseline. Any boosting should beat this if done right.")
    print("2. RANDOM (Noise): High variance. Can hit lucky streaks (high 'Final 50') but Avg is usually low.")
    print("   * If Random > Learners, it means 1000 eps is too short for convergence, or learners collapsed.")
    print("3. LinUCB: Restricted to 4 specific weight vectors. Can be beaten by Random/ContQ which search ")
    print("   the full space, IF the optimal weights are not in the template set.")
    print("4. ContQ / CQL: Should be the winners in long run. CQL starts strong due to pre-training.")
    print("5. REINFORCE / SlateQ: Deep RL from scratch is SLOW. Expect low scores in short runs.")
    print("="*80)

if __name__ == "__main__":
    main()