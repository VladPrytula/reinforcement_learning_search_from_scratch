"""Environment API skeleton for Zooplus Search RL simulator.

Implements the core MDP interface for search ranking RL:
- reset(): Sample user + query, return initial state
- step(action): Apply boost action, simulate session, return reward and info

Chapter 14 additions:
- Baseline ranking computation (action=0) for stability measurement
- delta_rank_at_k_vs_baseline in info for CMDP constraints
- user_segment and query_type in step info for fairness reporting

References:
    - Chapter 3: MDP formulation
    - Chapter 5: Relevance and features
    - Chapter 14: CMDP constraints (stability, fairness)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from zoosim.core import config as cfg_module
from zoosim.dynamics import behavior, reward
from zoosim.monitoring.metrics import compute_delta_rank_at_k
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

    def _rank_products(
        self, action: Sequence[float]
    ) -> Tuple[List[int], List[int], List[List[float]]]:
        """Compute action-adjusted ranking and baseline ranking.

        The baseline ranking (action=0) is computed from the same base_scores
        without additional RNG consumption. Used for stability measurement
        in Chapter 14 CMDP constraints [EQ-14.6].

        Returns:
            ranking: Product IDs sorted by blended score (base + action*features)
            baseline_ranking: Product IDs sorted by base score only (action=0)
            feature_matrix: Features for all products (list of lists)
        """
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

        # Baseline ranking: sorting by base_scores alone (no action boost)
        # No additional RNG consumption - uses same base_scores already computed
        baseline_ranking = np.argsort(-base_scores).tolist()

        # Action-adjusted ranking
        action = np.clip(np.asarray(action, dtype=float), -self.cfg.action.a_max, self.cfg.action.a_max)
        blended_scores = base_scores + feature_matrix @ action
        ranking = np.argsort(-blended_scores).tolist()

        return ranking, baseline_ranking, feature_matrix.tolist()

    def step(self, action: Sequence[float]) -> Tuple[None, float, bool, Dict[str, Any]]:  # type: ignore[override]
        ranking, baseline_ranking, feature_matrix = self._rank_products(action)
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

        # Compute stability metric for Chapter 14 CMDP constraints [EQ-14.6]
        k = self.cfg.top_k
        delta_rank = compute_delta_rank_at_k(baseline_ranking, ranking, k=k)

        info = {
            "reward_details": breakdown,
            "satisfaction": outcome.satisfaction,
            "ranking": ranking[:k],
            "clicks": outcome.clicks,
            "buys": outcome.buys,
            "features": feature_matrix[:k],
            # Chapter 14 additions for CMDP constraints
            "baseline_ranking": baseline_ranking[:k],
            "delta_rank_at_k_vs_baseline": delta_rank,
            "user_segment": self._user.segment,
            "query_type": self._query.query_type,
        }
        return None, reward_value, True, info
