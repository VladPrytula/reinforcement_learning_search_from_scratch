#!/usr/bin/env python
"""Chapter 8: Gaussian Policy Ablations.

This script supports Lab 8.3 by comparing several continuous-action
policy parameterizations on the Zooplus search environment:

- Gaussian (diagonal, learnable std)      -- baseline (matches REINFORCEAgent)
- Fixed-std Gaussian                      -- mean only, std fixed
- Squashed Gaussian (tanh)                -- SAC-style squashing to [-1, 1]
- Beta distribution (scaled to [-a_max,a_max])

Usage (from Lab 8.3):

    for policy in gaussian beta squashed fixed_std; do
      for seed in 2025 2026 2027; do
        python scripts/ch08/policy_ablation.py \
          --policy $policy \
          --episodes 3000 \
          --seed $seed \
          > results_${policy}_seed${seed}.txt
      done
    done

This script is intentionally self-contained: it reuses the same REINFORCE
update logic as `zoosim/policies/reinforce.py` but swaps in different
policy heads. It is designed for ablation experiments rather than
production deployment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta, Distribution, Normal, kl_divergence

from zoosim.core.config import SimulatorConfig
from zoosim.envs.gym_env import GymZooplusEnv
from zoosim.policies.reinforce import REINFORCEAgent, REINFORCEConfig


def build_mlp(input_dim: int, hidden_sizes: Tuple[int, ...]) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.Tanh())
        prev = h
    return nn.Sequential(*layers)


class FixedStdGaussianPolicy(nn.Module):
    """Gaussian policy with fixed standard deviation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...], fixed_std: float = 1.0):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes)
        self.mu_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.fixed_std = fixed_std

    def forward(self, obs: torch.Tensor) -> Distribution:
        x = self.net(obs)
        mu = self.mu_head(x)
        std = torch.full_like(mu, self.fixed_std)
        return Normal(mu, std)


class BetaPolicy(nn.Module):
    """Beta policy on [0,1], scaled to [-a_max, a_max].

    Note: We ignore the change-of-variables term from scaling when
    computing log_prob. This keeps the implementation simple and is
    sufficient for comparative ablations in Lab 8.3.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...], a_max: float):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes)
        self.alpha_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.beta_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.a_max = a_max

    def forward(self, obs: torch.Tensor) -> Tuple[Distribution, torch.Tensor]:
        x = self.net(obs)
        # α, β > 1 for unimodal distributions (numerically stable).
        alpha = torch.exp(self.alpha_head(x)) + 1.0
        beta = torch.exp(self.beta_head(x)) + 1.0
        dist = Beta(alpha, beta)
        return dist, alpha.new_tensor(self.a_max)

    def sample_and_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist, a_max = self.forward(obs)
        a_01 = dist.rsample()  # [0, 1]
        action = 2 * a_max * (a_01 - 0.5)  # scale to [-a_max, a_max]
        log_prob = dist.log_prob(a_01).sum(-1)
        return action, log_prob


class SquashedGaussianPolicy(nn.Module):
    """Gaussian with tanh squashing (SAC-style)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...], a_max: float):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes)
        self.mu_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.a_max = a_max

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mu = self.mu_head(x)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 1.0))

        # Unbounded Gaussian sample
        z = mu + std * torch.randn_like(mu)
        # Squash to [-1, 1]
        action = torch.tanh(z)
        # Scale to [-a_max, a_max]
        action_scaled = self.a_max * action

        # Change-of-variables log-prob
        base_dist = Normal(mu, std)
        log_prob = base_dist.log_prob(z).sum(-1)
        # |da/dz| = 1 - tanh^2(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)

        return action_scaled, log_prob, base_dist.entropy().sum(-1)


@dataclass
class AblationConfig:
    hidden_sizes: Tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    normalize_returns: bool = True
    baseline_momentum: float = 0.9
    seed: int = 42
    device: str = "cpu"


class AblationAgent:
    """REINFORCE-style agent with pluggable policy head."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        a_max: float,
        policy_kind: str,
        config: AblationConfig,
    ):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)

        self.policy_kind = policy_kind
        self.a_max = a_max

        if policy_kind == "fixed_std":
            self.policy = FixedStdGaussianPolicy(obs_dim, action_dim, config.hidden_sizes).to(self.device)
        elif policy_kind == "beta":
            self.policy = BetaPolicy(obs_dim, action_dim, config.hidden_sizes, a_max).to(self.device)
        elif policy_kind == "squashed":
            self.policy = SquashedGaussianPolicy(obs_dim, action_dim, config.hidden_sizes, a_max).to(self.device)
        else:
            raise ValueError(f"Unknown ablation policy_kind={policy_kind!r}")

        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.baseline = 0.0
        self._reset_buffer()

    def _reset_buffer(self) -> None:
        self.log_probs: list[torch.Tensor] = []
        self.entropies: list[torch.Tensor] = []
        self.rewards: list[float] = []

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        if self.policy_kind == "beta":
            action_t, log_prob = self.policy.sample_and_log_prob(obs_t)
            entropy = - (log_prob.detach())  # rough proxy; not exact entropy
        elif self.policy_kind == "squashed":
            action_t, log_prob, entropy = self.policy(obs_t)
        elif self.policy_kind == "fixed_std":
            dist = self.policy(obs_t)
            action_t = dist.sample()
            action_t = torch.clamp(action_t, -self.a_max, self.a_max)
            log_prob = dist.log_prob(action_t).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            raise RuntimeError("Unsupported policy_kind in AblationAgent")

        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

        return action_t.detach().cpu().numpy()

    def store_reward(self, reward: float) -> None:
        self.rewards.append(reward)

    def update(self) -> Dict[str, float]:
        R = 0.0
        returns: list[float] = []
        for r in reversed(self.rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if len(returns) > 0:
            batch_mean = returns_t.mean().item()
            self.baseline = (
                self.config.baseline_momentum * self.baseline
                + (1.0 - self.config.baseline_momentum) * batch_mean
            )

        if self.config.normalize_returns and len(returns) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        else:
            returns_t = returns_t - self.baseline

        policy_loss_terms: Iterable[torch.Tensor] = (
            -lp * G_t for lp, G_t in zip(self.log_probs, returns_t)
        )
        policy_loss = torch.stack(list(policy_loss_terms)).sum()
        entropy_term = torch.stack(self.entropies).mean()
        entropy_loss = -self.config.entropy_coef * entropy_term

        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        metrics = {
            "loss": float(loss.item()),
            "return_mean": float(np.sum(self.rewards)),
            "entropy": float(entropy_term.item()),
            "baseline": float(self.baseline),
        }

        self._reset_buffer()
        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=str, default="gaussian",
                        choices=["gaussian", "fixed_std", "beta", "squashed"],
                        help="Policy family to use.")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--a_max", type=float, default=5.0)
    parser.add_argument("--rich-features", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("Chapter 8: Gaussian Policy Ablations")
    print(f"Policy: {args.policy}")
    print(f"Episodes: {args.episodes} | Seed: {args.seed}")
    print(f"Action Limit: {args.a_max} | LR: {args.lr} | Entropy: {args.entropy}")
    print("=" * 70)

    cfg = SimulatorConfig(seed=args.seed)
    cfg.action.a_max = args.a_max

    env = GymZooplusEnv(cfg=cfg, seed=args.seed, rich_features=args.rich_features)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"Observation Dim: {obs_dim}")
    print(f"Action Dim: {action_dim}")

    if args.policy == "gaussian":
        # Baseline: reuse the production REINFORCEAgent.
        agent_config = REINFORCEConfig(
            learning_rate=args.lr,
            gamma=args.gamma,
            entropy_coef=args.entropy,
            seed=args.seed,
            device=args.device,
        )
        agent = REINFORCEAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_scale=cfg.action.a_max,
            config=agent_config,
        )
    else:
        ab_cfg = AblationConfig(
            hidden_sizes=(64, 64),
            learning_rate=args.lr,
            gamma=args.gamma,
            entropy_coef=args.entropy,
            baseline_momentum=0.9,
            normalize_returns=True,
            seed=args.seed,
            device=args.device,
        )
        agent = AblationAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            a_max=cfg.action.a_max,
            policy_kind=args.policy,
            config=ab_cfg,
        )

    returns: list[float] = []
    entropies: list[float] = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(float(reward))
            obs = next_obs
            done = terminated or truncated

        metrics = agent.update()
        returns.append(metrics["return_mean"])
        entropies.append(metrics["entropy"])

        if ep % 100 == 0:
            print(
                f"Ep {ep:5d} | Return: {metrics['return_mean']:7.2f} "
                f"| Entropy: {metrics['entropy']:5.2f} | Loss: {metrics['loss']:7.2f}"
            )

    env.close()

    first_100 = np.mean(returns[:100]) if len(returns) >= 100 else np.mean(returns)
    last_100 = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)

    print("\n" + "=" * 70)
    print("Final Summary")
    print("=" * 70)
    print(f"Policy: {args.policy}")
    print(f"Final 100-ep Average Return: {last_100:.2f}")
    print(f"Improvement: {first_100:.2f} -> {last_100:.2f}")
    print(f"Mean Entropy (all episodes): {np.mean(entropies):.2f}")


if __name__ == "__main__":
    main()

