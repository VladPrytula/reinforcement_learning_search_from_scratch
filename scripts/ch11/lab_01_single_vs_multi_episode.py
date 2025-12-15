#!/usr/bin/env python
"""Lab 11.1 — Single-step proxy vs multi-episode value.

This script compares policy rankings under:
  1) Single-session reward (GMV + δ·CLICKS) and
  2) Multi-episode discounted GMV using MultiSessionEnv.

It reports Spearman correlation between the two rankings.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

from zoosim.core.config import load_default_config
from zoosim.dynamics.reward import compute_reward
from zoosim.envs import ZooplusSearchEnv
from zoosim.multi_episode.session_env import MultiSessionEnv, SessionMDPState

from .utils.policies import cm2_heavy_policy, uniform_policy
from .utils.trajectory import collect_trajectory


PolicyFn = Callable[[SessionMDPState], np.ndarray]


def _single_step_value(
    env_seed: int,
    policy_action: np.ndarray,
    n_users: int,
) -> float:
    cfg = load_default_config()
    rewards: List[float] = []
    for i in range(n_users):
        env = ZooplusSearchEnv(cfg, seed=env_seed + i)
        _state = env.reset()
        # Single-session action ignores state; we use a fixed boost pattern.
        _next_state, reward, done, _info = env.step(policy_action)
        assert done is True
        rewards.append(float(reward))

    return float(np.mean(rewards))


def _multi_episode_value(
    env_seed: int,
    policy: PolicyFn,
    gamma: float,
    n_users: int,
) -> float:
    env = MultiSessionEnv(seed=env_seed)
    values: List[float] = []

    for _ in range(n_users):
        traj = collect_trajectory(env, policy, gamma=gamma)
        values.append(traj.discounted_value)

    return float(np.mean(values))


def compare_policies(
    env_seed: int = 11,
    n_users: int = 200,
    gamma: float = 0.95,
) -> Dict[str, float]:
    cfg = load_default_config()
    action_dim = cfg.action.feature_dim

    # Define a small policy family
    policies: Dict[str, PolicyFn] = {
        "uniform": uniform_policy(action_dim),
        "cm2_heavy": cm2_heavy_policy(action_dim),
    }

    # For single-step values we use fixed action vectors for each policy.
    single_step_actions: Dict[str, np.ndarray] = {
        "uniform": np.zeros(action_dim, dtype=float),
        "cm2_heavy": cm2_heavy_policy(action_dim)(SessionMDPState(0, "", "", [], 0.0, 0)),
    }

    single_vals: List[float] = []
    multi_vals: List[float] = []
    names: List[str] = []

    for name, policy in policies.items():
        single = _single_step_value(env_seed, single_step_actions[name], n_users)
        multi = _multi_episode_value(env_seed, policy, gamma, n_users)
        names.append(name)
        single_vals.append(single)
        multi_vals.append(multi)

    rho, _ = spearmanr(single_vals, multi_vals)

    # For debugging: print raw tables
    print("Policy\tSingleStep\tMultiEpisode")
    for name, s, m in zip(names, single_vals, multi_vals):
        print(f"{name}\t{s:.4f}\t{m:.4f}")
    print(f"\nSpearman rho(single, multi) = {rho:.3f}")

    return {
        "rho": float(rho),
        "single_min": float(np.min(single_vals)),
        "single_max": float(np.max(single_vals)),
        "multi_min": float(np.min(multi_vals)),
        "multi_max": float(np.max(multi_vals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab 11.1 — Single-step vs multi-episode value.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    args = parser.parse_args()

    stats = compare_policies(env_seed=args.seed, n_users=args.n_users, gamma=args.gamma)
    target = 0.80
    print(f"\nTarget Spearman rho >= {target:.2f}")
    if stats["rho"] >= target:
        print("ACCEPTED: policy ordering agreement criterion met.")
    else:
        print("WARNING: policy ordering agreement below target.")


if __name__ == "__main__":
    main()
