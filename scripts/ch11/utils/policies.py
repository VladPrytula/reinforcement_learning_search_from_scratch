"""Simple test policies for Chapter 11 experiments.

These are deliberately lightweight heuristics; the goal is to probe the
multi-episode behavior of the environment, not to optimize performance.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from zoosim.multi_episode.session_env import SessionMDPState


def uniform_policy(action_dim: int = 10) -> Callable[[SessionMDPState], np.ndarray]:
    """Zero-boost baseline (no personalization)."""

    def policy(state: SessionMDPState) -> np.ndarray:  # noqa: ARG001
        return np.zeros(action_dim, dtype=float)

    return policy


def cm2_heavy_policy(action_dim: int = 10) -> Callable[[SessionMDPState], np.ndarray]:
    """Heuristic CM2-focused boost pattern.

    Mirrors the intuition from Chapter 6: emphasize margin and strategic
    categories using a fixed boost vector.
    """
    boost = np.array(
        [0.5, 0.0, 0.3, 0.0, 0.1, -0.1] + [0.0] * max(0, action_dim - 6),
        dtype=float,
    )

    def policy(state: SessionMDPState) -> np.ndarray:  # noqa: ARG001
        return boost

    return policy


def segment_aware_policy(
    segment_boosts: Dict[str, np.ndarray],
    default_dim: int = 10,
) -> Callable[[SessionMDPState], np.ndarray]:
    """Apply segment-specific boosts, falling back to zeros."""

    def policy(state: SessionMDPState) -> np.ndarray:
        boost = segment_boosts.get(state.user_segment)
        if boost is None:
            return np.zeros(default_dim, dtype=float)
        return np.asarray(boost, dtype=float)

    return policy


def random_policy(
    rng: np.random.Generator, action_dim: int = 10, a_max: float = 0.5
) -> Callable[[SessionMDPState], np.ndarray]:
    """Random boost in [-a_max, a_max]."""

    def policy(state: SessionMDPState) -> np.ndarray:  # noqa: ARG001
        return rng.uniform(-a_max, a_max, size=action_dim).astype(float)

    return policy

