"""RL policy implementations for search ranking.

Modules:
- templates: Discrete boost templates (Chapter 6)
- lin_ucb: Linear Upper Confidence Bound contextual bandit (Chapter 6)
- thompson_sampling: Bayesian contextual bandit (Chapter 6)
- q_ensemble: Continuous-action Q(x,a) ensemble + CEM policy (Chapter 7)

Upcoming modules:
- constraints and fairness-aware policies (Chapters 8+)
"""

from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
from zoosim.policies.q_ensemble import (
    QEnsembleConfig,
    QEnsemblePolicy,
    QEnsembleRegressor,
)
from zoosim.policies.reinforce import REINFORCEAgent, REINFORCEConfig
from zoosim.policies.templates import (
    BoostTemplate,
    compute_catalog_stats,
    create_standard_templates,
)
from zoosim.policies.thompson_sampling import (
    LinearThompsonSampling,
    ThompsonSamplingConfig,
)

__all__ = [
    # Template bandits (Chapter 6)
    "BoostTemplate",
    "create_standard_templates",
    "compute_catalog_stats",
    "LinUCB",
    "LinUCBConfig",
    "LinearThompsonSampling",
    "ThompsonSamplingConfig",
    # Continuous-action Q-ensemble (Chapter 7)
    "QEnsembleConfig",
    "QEnsembleRegressor",
    "QEnsemblePolicy",
    # Policy Gradients (Chapter 8)
    "REINFORCEAgent",
    "REINFORCEConfig",
]
