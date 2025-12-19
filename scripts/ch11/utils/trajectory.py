"""Trajectory helpers for Chapter 11 multi-episode experiments.

These utilities collect full user journeys under the MultiSessionEnv and
compute discounted values used in the Chapter 11 labs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from zoosim.multi_episode.session_env import MultiSessionEnv, SessionMDPState


@dataclass
class MultiEpisodeTrajectory:
    """Complete user journey from first session until churn."""

    user_segment: str
    n_sessions: int
    total_gmv: float
    total_clicks: int
    discounted_value: float
    session_rewards: List[float]
    session_clicks: List[int]


def collect_trajectory(
    env: MultiSessionEnv,
    policy: Callable[[SessionMDPState], np.ndarray],
    gamma: float = 0.95,
    max_sessions: int = 100,
) -> MultiEpisodeTrajectory:
    """Run a single user journey until churn (or max_sessions).

    Args:
        env: Multi-session environment.
        policy: Maps SessionMDPState to an action vector.
        gamma: Discount factor for long-run GMV/reward.
        max_sessions: Safety cap on the number of sessions.
    """
    state = env.reset()
    session_rewards: List[float] = []
    session_clicks: List[int] = []
    total_gmv = 0.0
    total_clicks = 0
    discounted_value = 0.0
    discount = 1.0

    for _ in range(max_sessions):
        action = policy(state)
        next_state, reward, done, info = env.step(action)

        session_rewards.append(float(reward))
        clicks = int(sum(info.get("clicks", [])))
        session_clicks.append(clicks)

        reward_details = info.get("reward_details")
        gmv = float(getattr(reward_details, "gmv", 0.0))
        total_gmv += gmv
        total_clicks += clicks

        discounted_value += discount * float(reward)
        discount *= gamma

        state = next_state
        if done:
            break

    return MultiEpisodeTrajectory(
        user_segment=state.user_segment,
        n_sessions=len(session_rewards),
        total_gmv=total_gmv,
        total_clicks=total_clicks,
        discounted_value=discounted_value,
        session_rewards=session_rewards,
        session_clicks=session_clicks,
    )


def collect_dataset(
    env: MultiSessionEnv,
    policy: Callable[[SessionMDPState], np.ndarray],
    n_users: int,
    gamma: float = 0.95,
    seed: int = 42,
) -> List[MultiEpisodeTrajectory]:
    """Collect trajectories for multiple users with deterministic seeding.

    A fresh environment instance is created per user to avoid coupling
    trajectories via shared RNG state.
    """
    rng = np.random.default_rng(seed)
    cfg = env.cfg

    trajectories: List[MultiEpisodeTrajectory] = []
    for _ in range(n_users):
        env_seed = int(rng.integers(0, 2**32 - 1))
        local_env = MultiSessionEnv(cfg=cfg, seed=env_seed)
        traj = collect_trajectory(local_env, policy, gamma=gamma)
        trajectories.append(traj)

    return trajectories

