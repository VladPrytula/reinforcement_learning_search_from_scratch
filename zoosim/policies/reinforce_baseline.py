"""REINFORCE with Learned Baseline (Vanilla Policy Gradient).

This extends the basic REINFORCE by adding a state-dependent value function V(s)
to reduce variance.

Key differences from naive REINFORCE:
1.  Value Network (Critic) estimates V(s).
2.  Advantage calculation: A_t = G_t - V(s_t).
3.  Two-headed optimization (Policy and Value).

This aligns with the "Variance Reduction" section of Chapter 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from zoosim.policies.reinforce import REINFORCEConfig, GaussianPolicy

@dataclass
class REINFORCEBaselineConfig(REINFORCEConfig):
    """Config for REINFORCE with Baseline."""
    value_lr: float = 1e-3  # Learning rate for value network

class ValueNetwork(nn.Module):
    """Estimates State Value V(s)."""
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            prev_dim = size
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

class REINFORCEBaselineAgent:
    """REINFORCE with state-dependent baseline (V(s))."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        config: Optional[REINFORCEBaselineConfig] = None,
    ):
        self.config = config or REINFORCEBaselineConfig()
        self.action_scale = action_scale
        
        torch.manual_seed(self.config.seed)
        self.device = torch.device(self.config.device)

        # 1. Policy (Actor)
        self.policy = GaussianPolicy(
            obs_dim, action_dim, self.config.hidden_sizes
        ).to(self.device)
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )

        # 2. Value Function (Critic/Baseline)
        self.value_net = ValueNetwork(
            obs_dim, self.config.hidden_sizes
        ).to(self.device)
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=self.config.value_lr
        )

        self._reset_buffer()

    def _reset_buffer(self):
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.states = [] # Need states for V(s) updates

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # Store state for value update later
        self.states.append(obs_t) 
        
        dist = self.policy(obs_t)
        action_t = dist.sample()
        
        self.log_probs.append(dist.log_prob(action_t).sum())
        self.entropies.append(dist.entropy().sum())
        
        # Return numpy action
        return action_t.cpu().numpy()

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def update(self) -> Dict[str, float]:
        returns = []
        R = 0
        
        # 1. Compute Monte Carlo Returns G_t
        for r in reversed(self.rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)
            
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # 3. Compute Baseline V(s) and Advantages A_t
        states_t = torch.stack(self.states)
        values_t = self.value_net(states_t).squeeze()
        
        # Edge case: batch size 1 squeeze might result in 0-dim tensor
        if values_t.ndim == 0:
            values_t = values_t.unsqueeze(0)

        # Advantage: G_t - V(s_t)
        # Note: Detach values_t so we don't backprop advantage to value net via policy loss
        advantages = returns_t - values_t.detach()
        
        # Normalize advantages (Standard variance reduction trick)
        if self.config.normalize_returns and len(advantages) > 1:
             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. Update Policy (Actor)
        policy_loss = []
        for log_prob, adv in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * adv)
            
        entropy_loss = -self.config.entropy_coef * torch.stack(self.entropies).mean()
        total_policy_loss = torch.stack(policy_loss).sum() + entropy_loss 
        
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # 5. Update Value Network (Critic)
        # MSE Loss between V(s) and G_t
        value_loss = nn.MSELoss()(values_t, returns_t)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        metrics = {
            "loss": total_policy_loss.item(),
            "value_loss": value_loss.item(),
            "return_mean": np.sum(self.rewards),
            "entropy": torch.stack(self.entropies).mean().item(),
            "baseline_avg": values_t.mean().item()
        }
        
        self._reset_buffer()
        return metrics
