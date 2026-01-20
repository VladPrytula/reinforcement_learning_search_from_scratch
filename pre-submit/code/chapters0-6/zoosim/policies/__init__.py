"""RL policy implementations for search ranking.

Modules:
- templates: Discrete boost templates (Chapter 6)
- lin_ucb: Linear Upper Confidence Bound contextual bandit (Chapter 6)
- thompson_sampling: Bayesian contextual bandit (Chapter 6)
- q_ensemble: Continuous-action Q(x,a) ensemble + CEM policy (Chapter 7)
- reinforce: REINFORCE policy gradient (Chapter 8)
- reinforce_baseline: REINFORCE with learned baseline (Chapter 8)
- mo_cmdp: Multi-Objective CMDP with primal-dual optimization (Chapter 14)
"""

from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
from zoosim.policies.mo_cmdp import (
    CMDPConfig,
    CMDPMetrics,
    ConstraintSense,
    ConstraintSpec,
    PrimalDualCMDPAgent,
    create_standard_constraints,
    extract_ranking_metrics,
    train_cmdp_episode,
)
from zoosim.policies.q_ensemble import (
    QEnsembleConfig,
    QEnsemblePolicy,
    QEnsembleRegressor,
)
from zoosim.policies.reinforce import REINFORCEAgent, REINFORCEConfig
from zoosim.policies.reinforce_baseline import (
    REINFORCEBaselineAgent,
    REINFORCEBaselineConfig,
)
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
    "REINFORCEBaselineAgent",
    "REINFORCEBaselineConfig",
    # Multi-Objective CMDP (Chapter 14)
    "ConstraintSense",
    "ConstraintSpec",
    "CMDPConfig",
    "CMDPMetrics",
    "PrimalDualCMDPAgent",
    "create_standard_constraints",
    "train_cmdp_episode",
    "extract_ranking_metrics",
]
