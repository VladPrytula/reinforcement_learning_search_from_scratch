"""Gymnasium environment for Slate/List-wise RL.

This environment differs from GymZooplusEnv in two key ways:
1.  **Observation**: Instead of just context, it returns (Context, Candidate_List).
    The Candidate_List is a tensor of shape (max_candidates, feature_dim).
2.  **Action**: Instead of global weight vector w, it accepts per-item scores.
    Shape (max_candidates,). The environment sorts by these scores.

This enables "Deep Learning to Rank" where the agent scores items individually
using a neural network, rather than just tuning weights for handcrafted features.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from zoosim.core import config as cfg_module
from zoosim.dynamics import behavior, reward
from zoosim.envs.search_env import ZooplusSearchEnv
from zoosim.ranking import relevance, features


class SlateGymEnv(gym.Env):
    """Environment for Slate/List-wise Ranking RL."""

    metadata = {"render_modes": []}

    def __init__(
        self, 
        cfg: Optional[cfg_module.SimulatorConfig] = None, 
        seed: Optional[int] = None,
        max_candidates: int = 50,
        item_feature_dim: int = 10
    ) -> None:
        self.cfg = cfg or cfg_module.load_default_config()
        self.core = ZooplusSearchEnv(self.cfg, seed)
        self.max_candidates = max_candidates
        self.item_feature_dim = item_feature_dim
        
        # Define spaces
        # Context: User (Segment OneHot) + Query (Type OneHot)
        # We keep it simple: Segment(3) + QueryType(3) + RawTheta(2) = 8
        self.context_dim = 8 
        
        self.observation_space = spaces.Dict({
            "context": spaces.Box(low=-5, high=5, shape=(self.context_dim,), dtype=np.float32),
            "candidate_features": spaces.Box(
                low=-10, high=10, 
                shape=(self.max_candidates, self.item_feature_dim), 
                dtype=np.float32
            ),
            "candidate_mask": spaces.Box(low=0, high=1, shape=(self.max_candidates,), dtype=np.bool_),
        })
        
        # Action: Score for each candidate
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.max_candidates,), dtype=np.float32
        )

        # Internal state
        self._current_candidates = []
        self._current_candidate_features = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.core = ZooplusSearchEnv(self.cfg, seed)
        
        raw_state = self.core.reset()
        
        # 1. Encode Context
        context = self._encode_context()
        
        # 2. Retrieve Candidates (Top-M by base score)
        self._retrieve_candidates()
        
        obs = {
            "context": context,
            "candidate_features": self._current_candidate_features,
            "candidate_mask": np.ones(self.max_candidates, dtype=bool) # Simplified: always full batch
        }
        
        return obs, {"raw_state": raw_state}

    def _encode_context(self) -> np.ndarray:
        """Encodes user/query context into a fixed vector."""
        user = self.core._user
        query = self.core._query
        
        # One-hot Segment (assumed 3 segments: PriceSensitive, QualityOriented, TimeStarved)
        segs = ["PriceSensitive", "QualityOriented", "TimeStarved"]
        seg_vec = np.zeros(len(segs), dtype=np.float32)
        if user.segment in segs:
            seg_vec[segs.index(user.segment)] = 1.0
            
        # One-hot QueryType
        qtypes = ["informational", "navigational", "transactional"]
        q_vec = np.zeros(len(qtypes), dtype=np.float32)
        if query.query_type in qtypes:
            q_vec[qtypes.index(query.query_type)] = 1.0
            
        # Continuous User preferences
        prefs = np.array([user.theta_price, user.theta_pl], dtype=np.float32)
        
        return np.concatenate([seg_vec, q_vec, prefs])

    def _retrieve_candidates(self):
        """Selects top-k items based on BM25/Base score to serve as candidates for re-ranking."""
        query = self.core._query
        catalog = self.core._catalog
        
        # Compute base scores
        base_scores = relevance.batch_base_scores(
            query=query, catalog=catalog, config=self.cfg, rng=self.core.rng
        )
        
        # Sort and take top M
        top_indices = np.argsort(-np.array(base_scores))[:self.max_candidates]
        self._current_candidates = [catalog[i] for i in top_indices]
        
        # Compute features for these candidates
        feats_list = []
        for prod in self._current_candidates:
            f = features.compute_features(
                user=self.core._user, 
                query=self.core._query, 
                product=prod, 
                config=self.cfg
            )
            feats_list.append(f)
            
        # Pad if necessary (though we usually have enough items)
        feats_array = np.array(feats_list, dtype=np.float32)
        
        if feats_array.shape[0] < self.max_candidates:
            # Padding logic would go here
            pass
            
        # Standardize if needed (simple z-score on this batch for stability)
        if feats_array.shape[0] > 1:
             feats_array = (feats_array - feats_array.mean(axis=0)) / (feats_array.std(axis=0) + 1e-6)
             
        self._current_candidate_features = feats_array

    def step(self, action: np.ndarray):
        """
        Args:
            action: Scores for the candidates (shape: max_candidates)
        """
        # 1. Sort candidates by action scores
        # action contains scores for self._current_candidates
        
        # Sort indices descending
        sorted_indices = np.argsort(-action[:len(self._current_candidates)])
        
        # Reorder candidates
        ranked_candidates = [self._current_candidates[i] for i in sorted_indices]
        
        # The core simulator expects a full ranking of the *entire* catalog
        # But we only re-ranked the top M. 
        # We will construct a full ranking by putting our ranked M first, then the rest.
        # This requires mapping back to catalog indices.
        
        # Fast way: simulate session only on the top-k of our re-ranked list.
        # The core behavior logic only looks at the top_k items shown to user.
        
        # Let's cheat slightly and bypass core._rank_products and call behavior directly
        # assuming the 'ranking' list is just indices into the catalog.
        
        # Find indices of our ranked candidates in the original catalog
        # This is slow (O(N*M)). Ideally catalog has ID map.
        # For now, we trust the objects are the same.
        # We can just pass the product objects if we modify behavior, but behavior takes indices.
        
        catalog_map = {p.product_id: i for i, p in enumerate(self.core._catalog)}
        final_ranking_indices = [catalog_map[p.product_id] for p in ranked_candidates]
        
        # Append the rest of the catalog (randomly or by base score) if needed, 
        # but usually top-k is small (5-10) and max_candidates is 50, so we are good.
        
        outcome = behavior.simulate_session(
            user=self.core._user,
            query=self.core._query,
            ranking=final_ranking_indices, # This list is what user sees
            catalog=self.core._catalog,
            config=self.cfg,
            rng=self.core.rng,
        )
        
        reward_value, breakdown = reward.compute_reward(
            ranking=final_ranking_indices,
            clicks=outcome.clicks,
            buys=outcome.buys,
            catalog=self.core._catalog,
            config=self.cfg,
        )
        
        info = {
            "clicks": outcome.clicks,
            "buys": outcome.buys,
            "satisfaction": outcome.satisfaction
        }
        
        # Reset for next step
        obs, _ = self.reset()
        
        return obs, float(reward_value), True, False, info

