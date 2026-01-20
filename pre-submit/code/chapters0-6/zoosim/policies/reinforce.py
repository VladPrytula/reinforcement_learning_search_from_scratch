"""REINFORCE (Monte Carlo Policy Gradient) implementation.

Mathematical basis:
- [ALG-8.1] REINFORCE with Baseline
- [THM-8.1] Policy Gradient Theorem
- [EQ-8.3] Log-derivative trick

Implements a Gaussian policy for continuous actions, optimized via
Monte Carlo return estimates.

References:
    - Chapter 8, §8.2: REINFORCE Algorithm
    - Williams (1992): Simple Statistical Gradient-Following Algorithms
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Distribution

from zoosim.core.config import SimulatorConfig


@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE agent.

    Attributes:
        hidden_sizes: MLP hidden layer dimensions.
        learning_rate: Policy learning rate (alpha).
        gamma: Discount factor.
        entropy_coef: Entropy regularization weight.
        normalize_returns: Whether to normalize returns (batch norm).
        baseline_momentum: Momentum for running average baseline.
        seed: Random seed.
        device: 'cpu' or 'cuda'.
    """
    hidden_sizes: Tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    normalize_returns: bool = True
    baseline_momentum: float = 0.9
    seed: int = 42
    device: str = "cpu"


class GaussianPolicy(nn.Module):
    """Gaussian Policy Network \u03c0_\u03b8(a|s).

    Outputs mean \u03bc(s) and log_std (parameter) for independent Gaussian actions.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.Tanh())  # Tanh often better for policy nets
            prev_dim = size
        self.net = nn.Sequential(*layers)
        
        self.mu_head = nn.Linear(prev_dim, action_dim)
        # Learnable log_std, initialized to 0 (std=1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Distribution:
        x = self.net(obs)
        mu = self.mu_head(x)
        # Clamp log_std for stability
        std = torch.exp(torch.clamp(self.log_std, min=-2.0, max=1.0))
        return Normal(mu, std)


class REINFORCEAgent:
    """REINFORCE Agent (Monte Carlo Policy Gradient).

    Algorithm [ALG-8.1]:
    1. Collect trajectory \u03c4 = (s_0, a_0, r_0, ...)
    2. Compute returns G_t = \u2211 \u03b3^(k-t) r_k
    3. Compute baseline b_t (optional)
    4. Update \u03b8 \u2190 \u03b8 + \u03b1 \u2207 log \u03c0(a_t|s_t) (G_t - b_t)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: float = 1.0,
        config: Optional[REINFORCEConfig] = None,
    ):
        self.config = config or REINFORCEConfig()
        self.action_scale = action_scale
        
        torch.manual_seed(self.config.seed)
        self.device = torch.device(self.config.device)

        self.policy = GaussianPolicy(
            obs_dim, action_dim, self.config.hidden_sizes
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )
        
        # Running average baseline (simple scalar baseline per timestep could be added,
        # here we use a simple global moving average of returns for variance reduction).
        self.baseline = 0.0

        self._reset_buffer()

    def _reset_buffer(self):
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action from policy \u03c0_\u03b8(a|s)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        dist = self.policy(obs_t)
        action_t = dist.sample()
        
        # Store log_prob for update
        self.log_probs.append(dist.log_prob(action_t).sum())
        self.entropies.append(dist.entropy().sum())
        
        # Convert to numpy and scale
        # Note: The policy outputs unbounded Gaussian. We clip at the environment level,
        # but here we just pass the raw sample (scaled if needed).
        # Ideally, action_scale should match a_max.
        return action_t.cpu().numpy() # * self.action_scale 

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def update(self) -> Dict[str, float]:
        """Perform policy update at end of episode.
        
        Returns:
            metrics: Dict containing loss, avg_return, etc.
        """
        R = 0
        returns = []
        
        # Calculate returns G_t (Monte Carlo)
        for r in reversed(self.rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)
            
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Baseline subtraction (Variance reduction)
        # Update running average
        if len(returns) > 0:
            batch_mean = returns_t.mean().item()
            self.baseline = (
                self.config.baseline_momentum * self.baseline
                + (1 - self.config.baseline_momentum) * batch_mean
            )
            
        # Normalize returns (common practice for stability)
        if self.config.normalize_returns and len(returns) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        else:
            # Apply simple baseline
            returns_t = returns_t - self.baseline

        # Compute Loss: L = - \u2211 (log \u03c0 * A + \u03b2 * H)
        policy_loss = []
        for log_prob, G_t in zip(self.log_probs, returns_t):
            policy_loss.append(-log_prob * G_t)
            
        entropy_loss = -self.config.entropy_coef * torch.stack(self.entropies).mean()
        
        loss = torch.stack(policy_loss).sum() + entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        metrics = {
            "loss": loss.item(),
            "return_mean": np.sum(self.rewards),
            "entropy": torch.stack(self.entropies).mean().item(),
            "baseline": self.baseline
        }
        
        self._reset_buffer()
        return metrics
