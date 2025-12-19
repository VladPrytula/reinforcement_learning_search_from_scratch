"""Inter-session retention (hazard) modeling.

Seed-deterministic functions to compute probability of a user returning
for another session based on engagement signals from the previous one.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from zoosim.core.config import RetentionConfig, SimulatorConfig


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def return_probability(
    *,
    clicks: int,
    satisfaction: float,
    config: SimulatorConfig,
) -> float:
    """Probability a user returns for the next session.

    Logistic model with baseline and engagement contributions.
    Deterministic given inputs and config.
    """
    rc: RetentionConfig = config.retention
    logit = _logit(rc.base_rate) + rc.click_weight * float(clicks) + rc.satisfaction_weight * float(
        satisfaction
    )
    prob = 1.0 / (1.0 + math.exp(-logit))
    return float(min(max(prob, 0.0), 1.0))


def sample_return(
    *, clicks: int, satisfaction: float, config: SimulatorConfig, rng: Optional[np.random.Generator] = None
) -> bool:
    """Sample whether the user returns, using the configured RNG.

    The sampling is seed-deterministic when `rng` is created from the
    simulator's seeded generator.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)
    p = return_probability(clicks=clicks, satisfaction=satisfaction, config=config)
    return bool(rng.random() < p)


__all__ = ["return_probability", "sample_return"]
