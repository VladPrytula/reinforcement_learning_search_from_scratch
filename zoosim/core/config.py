"""Configuration models and defaults for the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Catalog configuration
# ---------------------------------------------------------------------------


@dataclass
class CatalogConfig:
    n_products: int = 10_000
    categories: List[str] = field(
        default_factory=lambda: ["dog_food", "cat_food", "litter", "toys"]
    )
    category_mix: List[float] = field(default_factory=lambda: [0.35, 0.35, 0.15, 0.15])
    embedding_dim: int = 16
    price_params: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "dog_food": {"mu": 2.6, "sigma": 0.4},
            "cat_food": {"mu": 2.5, "sigma": 0.4},
            "litter": {"mu": 2.2, "sigma": 0.35},
            "toys": {"mu": 1.8, "sigma": 0.6},
        }
    )
    margin_slope: Dict[str, float] = field(
        default_factory=lambda: {
            "dog_food": 0.12,
            "cat_food": 0.11,
            "litter": -0.03,
            "toys": 0.20,
        }
    )
    margin_noise: float = 0.3
    pl_prob: Dict[str, float] = field(
        default_factory=lambda: {
            "dog_food": 0.35,
            "cat_food": 0.40,
            "litter": 0.50,
            "toys": 0.15,
        }
    )
    discount_params: Dict[str, float] = field(
        default_factory=lambda: {"p_zero": 0.7, "low": 0.05, "high": 0.3}
    )
    bestseller_mean: Dict[str, float] = field(
        default_factory=lambda: {
            "dog_food": 2.0,
            "cat_food": 2.2,
            "litter": 1.2,
            "toys": 0.8,
        }
    )
    bestseller_std: Dict[str, float] = field(
        default_factory=lambda: {
            "dog_food": 0.7,
            "cat_food": 0.7,
            "litter": 0.5,
            "toys": 0.6,
        }
    )
    emb_cluster_std: Dict[str, float] = field(
        default_factory=lambda: {
            "dog_food": 0.7,
            "cat_food": 0.7,
            "litter": 0.5,
            "toys": 0.9,
        }
    )
    strategic_categories: List[str] = field(default_factory=lambda: ["litter"])


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------


@dataclass
class SegmentParams:
    price_mean: float
    price_std: float
    pl_mean: float
    pl_std: float
    cat_conc: List[float]


@dataclass
class UserConfig:
    segments: List[str] = field(
        default_factory=lambda: ["price_hunter", "pl_lover", "premium", "litter_heavy"]
    )
    segment_mix: List[float] = field(default_factory=lambda: [0.35, 0.25, 0.15, 0.25])
    segment_params: Dict[str, SegmentParams] = field(
        default_factory=lambda: {
            "price_hunter": SegmentParams(
                price_mean=-3.5,
                price_std=0.3,
                pl_mean=-1.0,
                pl_std=0.2,
                cat_conc=[0.30, 0.30, 0.20, 0.20],
            ),
            "pl_lover": SegmentParams(
                price_mean=-1.8,
                price_std=0.2,
                pl_mean=3.2,
                pl_std=0.3,
                cat_conc=[0.20, 0.45, 0.20, 0.15],
            ),
            "premium": SegmentParams(
                price_mean=1.2,
                price_std=0.25,
                pl_mean=-2.0,
                pl_std=0.2,
                cat_conc=[0.50, 0.25, 0.05, 0.20],
            ),
            "litter_heavy": SegmentParams(
                price_mean=-1.0,
                price_std=0.2,
                pl_mean=1.0,
                pl_std=0.2,
                cat_conc=[0.05, 0.05, 0.85, 0.05],
            ),
        }
    )


# ---------------------------------------------------------------------------
# Query configuration
# ---------------------------------------------------------------------------


@dataclass
class QueryConfig:
    query_type_mix: List[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])
    query_types: List[str] = field(
        default_factory=lambda: ["category", "brand", "generic"]
    )
    token_vocab_size: int = 512
    specificity: Dict[str, float] = field(
        default_factory=lambda: {"brand": 1.0, "category": 0.7, "generic": 0.4}
    )


# ---------------------------------------------------------------------------
# Relevance configuration
# ---------------------------------------------------------------------------


@dataclass
class RelevanceConfig:
    w_sem: float = 0.7
    w_lex: float = 0.3
    noise_sigma: float = 0.05


# ---------------------------------------------------------------------------
# Behavior configuration
# ---------------------------------------------------------------------------


@dataclass
class BehaviorConfig:
    alpha_rel: float = 1.0
    alpha_price: float = 0.8
    alpha_pl: float = 1.2
    alpha_cat: float = 0.6
    sigma_u: float = 0.8
    beta_buy: float = 1.2
    beta0: float = 0.3
    w_price_penalty: float = 0.01
    satisfaction_decay: float = 0.2
    satisfaction_gain: float = 0.5
    abandonment_threshold: float = -2.0
    max_purchases: int = 3
    post_purchase_fatigue: float = 1.2
    pos_bias: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "category": [1.2, 0.9, 0.7, 0.5, 0.3] + [0.2] * 15,
            "brand": [1.5, 0.8, 0.5, 0.3, 0.2] + [0.1] * 15,
            "generic": [1.0, 0.8, 0.6, 0.4, 0.2] + [0.1] * 15,
        }
    )


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    alpha_gmv: float = 1.0
    beta_cm2: float = 0.4
    gamma_strat: float = 2.0
    delta_clicks: float = 0.1


# ---------------------------------------------------------------------------
# Retention configuration (multi‑episode MDP)
# ---------------------------------------------------------------------------


@dataclass
class RetentionConfig:
    """Parameters for inter‑session retention/hazard.

    base_rate: baseline probability of returning for the next session
    click_weight: additive contribution per click to the return logit
    satisfaction_weight: contribution of end‑of‑session satisfaction to the return logit
    """

    base_rate: float = 0.35
    click_weight: float = 0.20
    satisfaction_weight: float = 0.15


# ---------------------------------------------------------------------------
# Action / policy configuration
# ---------------------------------------------------------------------------


@dataclass
class ActionConfig:
    feature_dim: int = 10
    a_max: float = 5.0  # Default for RL training (Ch6-8); use 0.5 for conservative production
    lambda_rank: float = 0.0
    standardize_features: bool = True
    cm2_floor: Optional[float] = None
    exposure_floors: Dict[str, float] = field(default_factory=dict)
    logging_epsilon: float = 0.05
    templates: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "balanced": [0.2, 0.1, 0.2, 0.1, 0.1, 0.0],
            "cm2_heavy": [0.5, 0.0, 0.3, 0.0, 0.1, -0.1],
            "revenue_max": [0.1, 0.3, 0.0, 0.2, 0.2, -0.2],
            "fairness": [0.0, 0.2, 0.4, 0.1, 0.0, 0.0],
        }
    )


# ---------------------------------------------------------------------------
# Simulator config
# ---------------------------------------------------------------------------


@dataclass
class SimulatorConfig:
    seed: int = 2025_1108
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    users: UserConfig = field(default_factory=UserConfig)
    queries: QueryConfig = field(default_factory=QueryConfig)
    relevance: RelevanceConfig = field(default_factory=RelevanceConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    top_k: int = 20


def load_default_config() -> SimulatorConfig:
    """Return a fully populated default configuration."""

    return SimulatorConfig()


DEFAULT_CONFIG = load_default_config()
