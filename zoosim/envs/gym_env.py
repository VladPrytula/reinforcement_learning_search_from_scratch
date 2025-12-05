"""Gymnasium wrapper for ZooplusSearchEnv."""

from __future__ import annotations

from typing import Optional, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from zoosim.core import config as cfg_module
from zoosim.envs.search_env import ZooplusSearchEnv
from zoosim.ranking import relevance
from zoosim.ranking.features import compute_context_features_rich_estimated
from zoosim.world.catalog import Product
from zoosim.world.queries import Query
from zoosim.world.users import User


class GymZooplusEnv(gym.Env):
    """Expose the simulator via the Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(
        self, 
        cfg: Optional[cfg_module.SimulatorConfig] = None, 
        seed: Optional[int] = None,
        rich_features: bool = False
    ) -> None:
        self.cfg = cfg or cfg_module.load_default_config()
        self.core = ZooplusSearchEnv(self.cfg, seed)
        self.query_types = self.cfg.queries.query_types
        self.segments = self.cfg.users.segments
        self.rich_features = rich_features
        
        dim = self._obs_dim()
        self.observation_space = spaces.Box(
            low=-float('inf') if rich_features else 0.0,
            high=float('inf') if rich_features else 1.0,
            shape=(dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-self.cfg.action.a_max,
            high=self.cfg.action.a_max,
            shape=(self.cfg.action.feature_dim,),
            dtype=np.float32,
        )
        self._last_state = None

    def _obs_dim(self) -> int:
        if self.rich_features:
            return 17  # Fixed dimension for rich features
        return len(self.cfg.catalog.categories) + len(self.query_types) + len(self.segments)

    def _compute_rich_features(self) -> np.ndarray:
        """Compute rich context features using centralized implementation.

        Delegates to zoosim.ranking.features.compute_context_features_rich_estimated
        which provides the canonical 17-dim feature vector with estimated user latents.
        """
        # Access internal state directly (Hack for Leveling the Playing Field)
        user = self.core._user
        query = self.core._query
        products = self.core._catalog

        if user is None or query is None:
            return np.zeros(self._obs_dim(), dtype=np.float32)

        # Compute Base Scores
        base_scores = np.asarray(
            relevance.batch_base_scores(
                query=query, catalog=products, config=self.cfg, rng=self.core.rng
            ),
            dtype=float
        )

        # Use centralized feature computation (estimated user latents version)
        features = compute_context_features_rich_estimated(
            user=user,
            query=query,
            catalog=products,
            base_scores=base_scores,
            config=self.cfg,
        )

        return features.astype(np.float32)

    def _encode_state(self, state) -> np.ndarray:
        if self.rich_features:
            return self._compute_rich_features()
            
        cat = np.asarray(state["phi_cat"], dtype=np.float32)
        qtype_vec = np.zeros(len(self.query_types), dtype=np.float32)
        qtype_vec[self.query_types.index(state["query_type"])] = 1.0
        segment_vec = np.zeros(len(self.segments), dtype=np.float32)
        segment_vec[self.segments.index(state["user_segment"])] = 1.0
        return np.concatenate([cat, qtype_vec, segment_vec])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        if seed is not None:
            self.core = ZooplusSearchEnv(self.cfg, seed)
        raw_state = self.core.reset()
        obs = self._encode_state(raw_state)
        self._last_state = raw_state
        return obs, {"raw_state": raw_state}

    def step(self, action):  # type: ignore[override]
        _, reward, done, info = self.core.step(action)
        obs, reset_info = self.reset()
        info.update(reset_info)
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info


__all__ = ["GymZooplusEnv"]
