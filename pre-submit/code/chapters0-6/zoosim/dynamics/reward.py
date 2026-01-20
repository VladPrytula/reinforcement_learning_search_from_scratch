"""Reward aggregation and constraint tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from zoosim.core.config import RewardConfig, SimulatorConfig
from zoosim.world.catalog import Product


@dataclass
class RewardBreakdown:
    gmv: float
    cm2: float
    strat: float
    clicks: int


def _compute_components(
    ranking: Sequence[int],
    clicks: Sequence[int],
    buys: Sequence[int],
    catalog: Sequence[Product],
) -> RewardBreakdown:
    gmv = 0.0
    cm2 = 0.0
    strat = 0.0
    click_total = 0
    limit = min(len(ranking), len(buys))
    for idx in range(limit):
        pid = ranking[idx]
        click_total += int(clicks[idx]) if idx < len(clicks) else 0
        if buys[idx]:
            prod = catalog[pid]
            gmv += prod.price
            cm2 += prod.cm2
            strat += 1.0 if prod.strategic_flag else 0.0
    return RewardBreakdown(gmv=gmv, cm2=cm2, strat=strat, clicks=click_total)


def compute_reward(
    *,
    ranking: Sequence[int],
    clicks: Sequence[int],
    buys: Sequence[int],
    catalog: Sequence[Product],
    config: SimulatorConfig,
) -> Tuple[float, RewardBreakdown]:
    breakdown = _compute_components(ranking, clicks, buys, catalog)
    cfg: RewardConfig = config.reward
    # Enforce engagement weight bounds to reduce clickbait risk (see REM-1.2.1).
    # Bound: 0.01 <= delta_clicks / alpha_gmv <= 0.10
    alpha = float(cfg.alpha_gmv)
    ratio = float("inf") if alpha == 0.0 else float(cfg.delta_clicks) / alpha
    assert 0.01 <= ratio <= 0.10, (
        f"Engagement weight outside safe range [0.01, 0.10]: "
        f"delta/alpha = {ratio:.3f}. Adjust RewardConfig to avoid clickbait optimization."
    )
    reward = (
        cfg.alpha_gmv * breakdown.gmv
        + cfg.beta_cm2 * breakdown.cm2
        + cfg.gamma_strat * breakdown.strat
        + cfg.delta_clicks * breakdown.clicks
    )
    return reward, breakdown
