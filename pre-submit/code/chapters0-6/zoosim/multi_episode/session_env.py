"""Multi-episode (inter-session) environment wrapping single-step sessions.

This environment models whether the same user returns across sessions
based on engagement signals (clicks, satisfaction) from the prior session.

API mirrors a minimal Gym-like step/reset with `done=True` on churn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from zoosim.core.config import SimulatorConfig, load_default_config
from zoosim.dynamics import behavior, reward
from zoosim.multi_episode.retention import return_probability, sample_return
from zoosim.ranking import features, relevance
from zoosim.world import catalog, queries
from zoosim.world.users import sample_user


@dataclass
class SessionMDPState:
    t: int
    user_segment: str
    query_type: str
    phi_cat: Sequence[float]
    last_satisfaction: float
    last_clicks: int


class MultiSessionEnv:
    def __init__(self, cfg: Optional[SimulatorConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = cfg or load_default_config()
        self.rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        self._catalog = catalog.generate_catalog(self.cfg.catalog, self.rng)
        self._user = None
        self._query = None
        self._t = 0
        self._last_clicks = 0
        self._last_satisfaction = 0.0

    def _rank(self, action: Sequence[float]) -> Tuple[Sequence[int], Sequence[Sequence[float]]]:
        base_scores = np.asarray(
            relevance.batch_base_scores(
                query=self._query, catalog=self._catalog, config=self.cfg, rng=self.rng
            ),
            dtype=float,
        )
        feat_rows = [
            features.compute_features(user=self._user, query=self._query, product=prod, config=self.cfg)
            for prod in self._catalog
        ]
        if self.cfg.action.standardize_features:
            feat_rows = features.standardize_features(feat_rows, config=self.cfg)
        feat_mat = np.asarray(feat_rows, dtype=float)
        action = np.clip(np.asarray(action, dtype=float), -self.cfg.action.a_max, self.cfg.action.a_max)
        blended = base_scores + feat_mat @ action
        ranking = np.argsort(-blended).tolist()
        return ranking, feat_mat.tolist()

    def reset(self) -> SessionMDPState:
        self._t = 0
        self._user = sample_user(config=self.cfg, rng=self.rng)
        self._query = queries.sample_query(user=self._user, config=self.cfg, rng=self.rng)
        self._last_clicks = 0
        self._last_satisfaction = 0.0
        return SessionMDPState(
            t=self._t,
            user_segment=self._user.segment,
            query_type=self._query.query_type,
            phi_cat=self._query.phi_cat,
            last_satisfaction=self._last_satisfaction,
            last_clicks=self._last_clicks,
        )

    def step(self, action: Sequence[float]) -> Tuple[SessionMDPState, float, bool, Dict[str, Any]]:
        ranking, feat_mat = self._rank(action)
        outcome = behavior.simulate_session(
            user=self._user,
            query=self._query,
            ranking=ranking,
            catalog=self._catalog,
            config=self.cfg,
            rng=self.rng,
        )
        rew, breakdown = reward.compute_reward(
            ranking=ranking,
            clicks=outcome.clicks,
            buys=outcome.buys,
            catalog=self._catalog,
            config=self.cfg,
        )
        clicks_total = int(sum(outcome.clicks))
        self._last_clicks = clicks_total
        self._last_satisfaction = float(outcome.satisfaction)

        # Retention decision
        p_return = return_probability(clicks=clicks_total, satisfaction=outcome.satisfaction, config=self.cfg)
        returns = sample_return(clicks=clicks_total, satisfaction=outcome.satisfaction, config=self.cfg, rng=self.rng)

        info = {
            "reward_details": breakdown,
            "satisfaction": outcome.satisfaction,
            "ranking": ranking[: self.cfg.top_k],
            "clicks": outcome.clicks,
            "buys": outcome.buys,
            "features": feat_mat[: self.cfg.top_k],
            "p_return": p_return,
            "returned": returns,
        }

        # Transition
        self._t += 1
        if returns:
            # Same user, new query
            self._query = queries.sample_query(user=self._user, config=self.cfg, rng=self.rng)
            done = False
        else:
            done = True

        state = SessionMDPState(
            t=self._t,
            user_segment=self._user.segment,
            query_type=self._query.query_type if not done else "terminal",
            phi_cat=self._query.phi_cat if not done else [],
            last_satisfaction=self._last_satisfaction,
            last_clicks=self._last_clicks,
        )

        return state, float(rew), bool(done), info


__all__ = ["MultiSessionEnv", "SessionMDPState"]
