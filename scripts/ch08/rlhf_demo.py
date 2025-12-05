#!/usr/bin/env python
"""Chapter 8: RLHF-Style Pipeline Demo for Search Ranking.

This script provides a minimal, runnable version of Lab 8.4:

1. Supervised Fine-Tuning (SFT)
   - Collect "expert" trajectories from a simple template policy.
   - Train a Gaussian policy to imitate expert actions via MLE.

2. Reward Modeling (Bradley–Terry)
   - Build trajectory-level embeddings by average-pooling states.
   - Train a RewardMLP so that σ(r_φ(τ1) - r_φ(τ2)) matches which
     trajectory has higher GMV.

3. RL Fine-Tuning with KL Penalty
   - Initialize a new policy from the SFT weights.
   - Optimize the learned reward minus a KL penalty to the SFT policy
     using a REINFORCE-style policy gradient on the learned objective.

This is intentionally a toy RLHF-style pipeline for didactic purposes:
it mirrors the structure of Lab 8.4 and the InstructGPT workflow but
does NOT represent a production RLHF implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.optim import Adam

from zoosim.core.config import SimulatorConfig
from zoosim.envs.gym_env import GymZooplusEnv


def get_device(device_arg: str) -> str:
    """Resolve device argument to a valid PyTorch device string."""
    if device_arg != "auto":
        return device_arg

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_mlp(input_dim: int, hidden_sizes: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.Tanh())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy π_θ(a|s) used for SFT and RL phases."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        mu = self.net(obs)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 1.0))
        return Normal(mu, std)


class RewardMLP(nn.Module):
    """Simple trajectory-level reward model r_φ(τ).

    We average-pool states along the trajectory to form a fixed-size
    embedding, then predict a scalar reward.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, traj_states: torch.Tensor) -> torch.Tensor:
        # traj_states: [T, state_dim]; average over time
        pooled = traj_states.mean(dim=0)
        return self.net(pooled).squeeze(-1)


@dataclass
class Episode:
    states: np.ndarray  # shape: [T, obs_dim]
    actions: np.ndarray  # shape: [T, action_dim]
    rewards: np.ndarray  # env rewards, shape: [T]

    @property
    def return_gmv(self) -> float:
        return float(self.rewards.sum())


def expert_template_action(obs: np.ndarray, a_max: float) -> np.ndarray:
    """Very simple "expert" policy mimicking discrete templates.

    We look at the first two coordinates as crude segment/query proxies
    and choose among three fixed templates:
    - Budget: boost cheaper items (negative price boost)
    - Premium: boost higher-priced items (positive price boost)
    - Neutral: near-zero boosts
    """
    dim = obs.shape[0]
    action_dim = min(10, dim)  # environment has 10 boost dims; be defensive

    # Three simple templates in R^{action_dim}
    budget = np.zeros(action_dim, dtype=np.float32)
    premium = np.zeros_like(budget)
    neutral = np.zeros_like(budget)

    # Interpret index 0,1 as crude "segment" / "query"
    seg = obs[0] if dim > 0 else 0.0
    qry = obs[1] if dim > 1 else 0.0

    # Budget template: discount-heavy, negative price
    budget[0] = +0.4 * a_max   # discount boost
    budget[1] = -0.3 * a_max   # price boost (cheaper)

    # Premium template: positive price, little discount
    premium[0] = +0.1 * a_max
    premium[1] = +0.4 * a_max

    # Neutral: tiny boosts
    neutral[:] = 0.05 * a_max

    # Simple routing based on seg/qry:
    if seg > 0 and qry <= 0:
        a = budget
    elif seg <= 0 and qry > 0:
        a = premium
    else:
        a = neutral

    return np.clip(a, -a_max, a_max)


def collect_expert_episodes(
    env: GymZooplusEnv,
    num_episodes: int,
    a_max: float,
) -> List[Episode]:
    episodes: List[Episode] = []
    for _ in range(num_episodes):
        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        rewards: List[float] = []

        obs, _ = env.reset()
        done = False
        while not done:
            a = expert_template_action(obs, a_max)
            next_obs, r, terminated, truncated, _ = env.step(a)

            states.append(obs.astype(np.float32))
            actions.append(a.astype(np.float32))
            rewards.append(float(r))

            obs = next_obs
            done = terminated or truncated

        episodes.append(
            Episode(
                states=np.stack(states, axis=0),
                actions=np.stack(actions, axis=0),
                rewards=np.asarray(rewards, dtype=np.float32),
            )
        )
    return episodes


def supervised_fine_tune(
    episodes: Sequence[Episode],
    obs_dim: int,
    action_dim: int,
    hidden_sizes: Tuple[int, ...],
    lr: float,
    device: str,
    epochs: int = 10,
    batch_size: int = 64,
) -> GaussianPolicy:
    policy = GaussianPolicy(obs_dim, action_dim, hidden_sizes).to(device)
    opt = Adam(policy.parameters(), lr=lr)

    # Flatten dataset (states, actions)
    all_states = np.concatenate([ep.states for ep in episodes], axis=0)
    all_actions = np.concatenate([ep.actions for ep in episodes], axis=0)
    N = all_states.shape[0]

    states_t = torch.as_tensor(all_states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(all_actions, dtype=torch.float32, device=device)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            s_batch = states_t[idx]
            a_batch = actions_t[idx]

            dist = policy(s_batch)
            log_prob = dist.log_prob(a_batch).sum(dim=-1)
            loss = -log_prob.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item()) * len(idx)

        print(f"[SFT] Epoch {epoch:02d} | NLL={epoch_loss / N:.4f}")

    return policy


def build_bt_pairs(
    episodes: Sequence[Episode],
    rng: np.random.Generator,
    num_pairs: int,
) -> List[Tuple[Episode, Episode, float]]:
    pairs: List[Tuple[Episode, Episode, float]] = []
    n = len(episodes)
    for _ in range(num_pairs):
        i, j = rng.integers(0, n), rng.integers(0, n)
        tau1, tau2 = episodes[i], episodes[j]
        r1, r2 = tau1.return_gmv, tau2.return_gmv
        label = 1.0 if r1 > r2 else 0.0
        pairs.append((tau1, tau2, label))
    return pairs


def train_reward_model(
    episodes: Sequence[Episode],
    obs_dim: int,
    hidden_dim: int,
    lr: float,
    num_pairs: int,
    epochs: int,
    seed: int,
    device: str,
) -> RewardMLP:
    rng = np.random.default_rng(seed)
    pairs = build_bt_pairs(episodes, rng, num_pairs)

    model = RewardMLP(obs_dim, hidden_dim).to(device)
    opt = Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (tau1, tau2, label) in pairs:
            s1 = torch.as_tensor(tau1.states, dtype=torch.float32, device=device)
            s2 = torch.as_tensor(tau2.states, dtype=torch.float32, device=device)
            r1 = model(s1)
            r2 = model(s2)

            logit = r1 - r2
            y = torch.tensor(label, dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(logit, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        print(f"[RM] Epoch {epoch:02d} | total_loss={total_loss:.4f}")

    return model


def rl_fine_tune_with_kl(
    env: GymZooplusEnv,
    policy_sft: GaussianPolicy,
    reward_model: RewardMLP,
    obs_dim: int,
    action_dim: int,
    hidden_sizes: Tuple[int, ...],
    device: str,
    episodes: int,
    gamma: float,
    entropy_coef: float,
    kl_coef: float,
) -> None:
    # Initialize RL policy from SFT weights
    policy_rl = GaussianPolicy(obs_dim, action_dim, hidden_sizes).to(device)
    policy_rl.load_state_dict(policy_sft.state_dict())

    policy_sft_frozen = GaussianPolicy(obs_dim, action_dim, hidden_sizes).to(device)
    policy_sft_frozen.load_state_dict(policy_sft.state_dict())
    policy_sft_frozen.eval()

    opt = Adam(policy_rl.parameters(), lr=3e-4)

    returns_env: List[float] = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False

        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        kls: List[torch.Tensor] = []
        env_rewards: List[float] = []

        while not done:
            states.append(obs.astype(np.float32))

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            dist_rl = policy_rl(obs_t)
            dist_sft = policy_sft_frozen(obs_t)

            a_t = dist_rl.sample()
            log_prob = dist_rl.log_prob(a_t).sum()
            entropy = dist_rl.entropy().sum()
            kl = kl_divergence(dist_rl, dist_sft).sum()

            action_np = a_t.detach().cpu().numpy()
            next_obs, r_env, terminated, truncated, _ = env.step(action_np)

            actions.append(action_np.astype(np.float32))
            log_probs.append(log_prob)
            entropies.append(entropy)
            kls.append(kl)
            env_rewards.append(float(r_env))

            obs = next_obs
            done = terminated or truncated

        returns_env.append(float(np.sum(env_rewards)))

        # Learned reward on trajectory
        states_t = torch.as_tensor(np.stack(states, axis=0), dtype=torch.float32, device=device)
        r_model = reward_model(states_t)

        kl_mean = torch.stack(kls).mean()
        entropy_mean = torch.stack(entropies).mean()

        # Simple scalar objective: J = r_model - kl_coef * KL
        J_hat = r_model - kl_coef * kl_mean

        # REINFORCE-style loss: all timesteps share same scalar "advantage"
        log_probs_t = torch.stack(log_probs)
        policy_loss = -(log_probs_t.sum() * J_hat.detach())
        entropy_loss = -entropy_coef * entropy_mean
        loss = policy_loss + entropy_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_rl.parameters(), 1.0)
        opt.step()

        if ep % 20 == 0:
            avg_ret_last = float(np.mean(returns_env[-20:]))
            print(
                f"[RL] Ep {ep:4d} | Avg Env Return (last 20): {avg_ret_last:7.2f} "
                f"| r_model={float(r_model.item()):6.2f} | KL={float(kl_mean.item()):5.3f}"
            )

    print("\n[RL] Final Avg Env Return (last 50 episodes): "
          f"{float(np.mean(returns_env[-50:])):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--sft-episodes", type=int, default=100)
    parser.add_argument("--rm-pairs", type=int, default=200)
    parser.add_argument("--rl-episodes", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--a_max", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda', 'mps'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)

    print("=" * 70)
    print("Chapter 8: RLHF-Style Demo (SFT + RM + RL)")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"SFT episodes: {args.sft_episodes} | RM pairs: {args.rm_pairs} | RL episodes: {args.rl_episodes}")
    print("=" * 70)

    cfg = SimulatorConfig(seed=args.seed)
    cfg.action.a_max = args.a_max
    env = GymZooplusEnv(cfg=cfg, seed=args.seed, rich_features=True)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Phase 1: collect expert trajectories and SFT
    print("\n[Phase 1] Collecting expert trajectories...")
    expert_eps = collect_expert_episodes(env, args.sft_episodes, args.a_max)
    avg_expert_return = float(np.mean([ep.return_gmv for ep in expert_eps]))
    print(f"[Phase 1] Expert mean return over {args.sft_episodes} episodes: {avg_expert_return:.2f}")

    print("\n[Phase 1] Supervised fine-tuning (SFT) on expert data...")
    hidden_sizes = (64, 64)
    policy_sft = supervised_fine_tune(
        expert_eps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        lr=1e-3,
        device=device,
        epochs=10,
    )

    # Phase 2: reward modeling (Bradley–Terry)
    print("\n[Phase 2] Training reward model (Bradley–Terry)...")
    reward_model = train_reward_model(
        expert_eps,
        obs_dim=obs_dim,
        hidden_dim=64,
        lr=1e-3,
        num_pairs=args.rm_pairs,
        epochs=10,
        seed=args.seed,
        device=device,
    )

    # Phase 3: RL fine-tuning with KL penalty
    print("\n[Phase 3] RL fine-tuning with KL penalty to SFT policy...")
    rl_fine_tune_with_kl(
        env,
        policy_sft=policy_sft,
        reward_model=reward_model,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        device=device,
        episodes=args.rl_episodes,
        gamma=args.gamma,
        entropy_coef=args.entropy,
        kl_coef=args.kl_coef,
    )

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
