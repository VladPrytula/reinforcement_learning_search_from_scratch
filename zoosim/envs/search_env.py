"""Environment API skeleton for Zooplus Search RL simulator."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from zoosim.core import config as cfg_module
from zoosim.dynamics import behavior, reward
from zoosim.ranking import features, relevance
from zoosim.world import catalog, queries, users


class ZooplusSearchEnv:
    def __init__(self, cfg: Optional[cfg_module.SimulatorConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = cfg or cfg_module.load_default_config()
        seed = seed if seed is not None else self.cfg.seed
        self.rng = np.random.default_rng(seed)
        self._catalog = catalog.generate_catalog(self.cfg.catalog, self.rng)
        self._user = None
        self._query = None

    def reset(self) -> Dict[str, Any]:
        self._user = users.sample_user(config=self.cfg, rng=self.rng)
        self._query = queries.sample_query(user=self._user, config=self.cfg, rng=self.rng)
        state = {
            "user_segment": self._user.segment,
            "query_category": self._query.intent_category,
            "query_type": self._query.query_type,
            "phi_cat": self._query.phi_cat,
        }
        return state

    def _rank_products(self, action: Sequence[float]) -> Tuple[list[int], list[list[float]]]:
        base_scores = np.asarray(
            relevance.batch_base_scores(
                query=self._query, catalog=self._catalog, config=self.cfg, rng=self.rng
            ),
            dtype=float,
        )
        feature_rows = [
            features.compute_features(user=self._user, query=self._query, product=prod, config=self.cfg)
            for prod in self._catalog
        ]
        if self.cfg.action.standardize_features:
            feature_rows = features.standardize_features(feature_rows, config=self.cfg)

        feature_matrix = np.asarray(feature_rows, dtype=float)
        action = np.clip(np.asarray(action, dtype=float), -self.cfg.action.a_max, self.cfg.action.a_max)
        blended_scores = base_scores + feature_matrix @ action
        ranking = np.argsort(-blended_scores).tolist()
        return ranking, feature_matrix.tolist()

    def step(self, action: Sequence[float]) -> Tuple[None, float, bool, Dict[str, Any]]:  # type: ignore[override]
        ranking, feature_matrix = self._rank_products(action)
        outcome = behavior.simulate_session(
            user=self._user,
            query=self._query,
            ranking=ranking,
            catalog=self._catalog,
            config=self.cfg,
            rng=self.rng,
        )
        reward_value, breakdown = reward.compute_reward(
            ranking=ranking,
            clicks=outcome.clicks,
            buys=outcome.buys,
            catalog=self._catalog,
            config=self.cfg,
        )
        info = {
            "reward_details": breakdown,
            "satisfaction": outcome.satisfaction,
            "ranking": ranking[: self.cfg.top_k],
            "clicks": outcome.clicks,
            "buys": outcome.buys,
            "features": feature_matrix[: self.cfg.top_k],
        }
        return None, reward_value, True, info
