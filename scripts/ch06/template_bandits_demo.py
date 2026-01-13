#!/usr/bin/env python
"""Chapter 6 demonstration: Template bandits vs static baselines.

This script instantiates the Zoosim world (catalog, users, queries) and
compares:

- Static templates: each BoostTemplate applied greedily every episode
- LinUCB contextual bandit over templates
- Thompson Sampling contextual bandit over templates

The reward is the scalar multi-objective reward from Chapter 5
(#EQ-1.2 / #EQ-5.7), and we log GMV / CM2 averages to see how much
uplift the template bandits achieve relative to static baselines.

Run with:

    uv run python scripts/ch06/template_bandits_demo.py
    uv run python scripts/ch06/template_bandits_demo.py --n-episodes 20000
    uv run python scripts/ch06/template_bandits_demo.py --n-static 4000 --n-bandit 2000

CLI parameters:
- ``--n-episodes``: total episode count used for both static and bandit sections (default 5,000 if neither per‑section flag is set).
- ``--n-static``: episode count for static template baselines only; overrides ``--n-episodes`` for the static part (Chapter 6 §6.5/§6.7 tables).
- ``--n-bandit``: episode count for LinUCB / Thompson Sampling runs; overrides ``--n-episodes`` for the bandit part.
- ``--features``: context feature mode for the bandits:
    - ``simple`` → 7-dim signals (segment + query type) plus a bias term (8 total), used in §6.5 to show the failure mode (bandits underperform the best static template in the reference run).
    - ``rich`` → 17-dim signals (segment + query + oracle user latents + base-top-K aggregates) plus a bias term (18 total), used in §6.7.4 to show the rich-feature fix (bandits outperform static in the reference run).
    - ``rich_est`` → same structure as ``rich`` but with *estimated* user latents, used in §6.7.5 to demonstrate robustness differences under feature noise.
- ``--show-volume``: if set, also prints average order volume per policy and segment, exposing the “GMV vs Orders” trade‑off discussed in §6.7 and the labs.

Notes:
- This is a lightweight demonstration, not a full acceptance test.
- Defaults use 5,000 episodes for both static and bandit sections; override via CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from zoosim.core.config import SimulatorConfig
from zoosim.dynamics import behavior, reward
from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
from zoosim.policies.templates import BoostTemplate, compute_catalog_stats, create_standard_templates
from zoosim.policies.thompson_sampling import LinearThompsonSampling, ThompsonSamplingConfig
from zoosim.world import catalog as catalog_module
from zoosim.world.catalog import Product
from zoosim.world.queries import Query, sample_query
from zoosim.world.users import User, sample_user

DEFAULT_EPISODES = 5_000
BIAS_FEATURE_VALUE = 1.0


def _append_bias(features: np.ndarray) -> np.ndarray:
    """Append bias feature used for template priors and intercepts."""
    if features.dtype != np.float64:
        features = features.astype(np.float64)
    bias = np.array([BIAS_FEATURE_VALUE], dtype=np.float64)
    return np.concatenate([features, bias])


def _quantize_preference(value: float, step: float, min_val: float, max_val: float) -> float:
    """Quantize preference signal onto a coarse grid while keeping bounds."""
    quantized = np.round(value / step) * step
    return float(np.clip(quantized, min_val, max_val))


@dataclass
class EpisodeResult:
    reward: float
    gmv: float
    cm2: float
    orders: float


@dataclass
class RichRegularizationConfig:
    """Controls how oracle latents are regularized before feeding bandits.

    mode:
        - "none": leave oracle θ untouched.
        - "blend": mix oracle θ with segment priors and apply shrink/clip.
        - "quantized": same as "blend" plus coarse quantization (rich_est style).
    """

    mode: str = "none"
    blend_weight: float = 0.4
    shrink: float = 0.9
    quant_step: float = 0.25
    clip_min: float = -3.5
    clip_max: float = 3.5


def context_features_simple(user: User, query: Query, cfg: SimulatorConfig) -> np.ndarray:
    """Simple context features (segment + query type only).

    This is deliberately impoverished to demonstrate failure mode.
    See Chapter 6 §6.4 for pedagogical motivation.

    Returns:
        φ_simple(x): Feature vector of shape (8,)
            - seg_vec (4): user segment one-hot
            - q_vec (3): query type one-hot
            - bias (1): intercept term for template priors
    """
    segs = cfg.users.segments
    qtypes = cfg.queries.query_types

    seg_vec = np.zeros(len(segs), dtype=float)
    seg_vec[segs.index(user.segment)] = 1.0

    q_vec = np.zeros(len(qtypes), dtype=float)
    q_vec[qtypes.index(query.query_type)] = 1.0

    base = np.concatenate([seg_vec, q_vec]).astype(np.float64)
    return _append_bias(base)


def _regularize_oracle_latents(
    theta_price: float,
    theta_pl: float,
    user: User,
    cfg: SimulatorConfig,
    config: RichRegularizationConfig | None,
) -> Tuple[float, float]:
    """Blend / shrink / quantize oracle latents to stabilize bandit regressors."""
    if config is None or config.mode == "none":
        return theta_price, theta_pl

    seg_defaults = cfg.users.segment_params[user.segment]

    if config.mode in {"blend", "quantized"}:
        theta_price = (
            (1.0 - config.blend_weight) * theta_price
            + config.blend_weight * seg_defaults.price_mean
        )
        theta_pl = (
            (1.0 - config.blend_weight) * theta_pl
            + config.blend_weight * seg_defaults.pl_mean
        )
        theta_price *= config.shrink
        theta_pl *= config.shrink

    if config.mode == "quantized":
        theta_price = _quantize_preference(
            theta_price,
            step=config.quant_step,
            min_val=config.clip_min,
            max_val=config.clip_max,
        )
        theta_pl = _quantize_preference(
            theta_pl,
            step=config.quant_step,
            min_val=config.clip_min,
            max_val=config.clip_max,
        )
    else:
        theta_price = float(np.clip(theta_price, config.clip_min, config.clip_max))
        theta_pl = float(np.clip(theta_pl, config.clip_min, config.clip_max))

    return theta_price, theta_pl


def context_features_rich(
    user: User,
    query: Query,
    products: List[Product],
    cfg: SimulatorConfig,
    base_scores: np.ndarray,
    regularization: RichRegularizationConfig | None = None,
) -> np.ndarray:
    """Rich context features for template bandits (oracle user latents).

    Computes aggregates over the base ranker top-k to avoid circularity.
    Returns a fixed-dimensional feature vector with simple standardization.

    Args:
        user: User instance (uses true simulator latents for preferences).
        query: Query instance.
        products: Full catalog.
        cfg: Simulator config.
        base_scores: Base relevance scores (before template boosts).

    Returns:
        φ_rich(x): Feature vector of shape (18,)
            - seg_vec (4): user segment one-hot
            - q_vec (3): query type one-hot
            - user_prefs (2): theta_price, theta_pl
            - aggregates (8): price stats, margin, discount, PL fraction, etc.
            - bias (1): intercept term for template priors
    """
    # 1. One-hot encodings
    seg_vec = np.zeros(len(cfg.users.segments), dtype=float)
    seg_vec[cfg.users.segments.index(user.segment)] = 1.0

    q_vec = np.zeros(len(cfg.queries.query_types), dtype=float)
    q_vec[cfg.queries.query_types.index(query.query_type)] = 1.0

    # 2. User preferences (optionally regularized to stabilize learning)
    theta_price = float(user.theta_price)
    theta_pl = float(user.theta_pl)
    theta_price, theta_pl = _regularize_oracle_latents(
        theta_price, theta_pl, user, cfg, regularization
    )

    # 3. Compute top-k products from base ranker
    k = min(cfg.top_k, len(products))
    top_idx = np.argsort(-np.asarray(base_scores, dtype=float))[:k]
    top_products = [products[i] for i in top_idx]

    # 4. Aggregate product features over base top-k
    prices = np.array([p.price for p in top_products], dtype=float)
    avg_price = float(prices.mean())
    std_price = float(prices.std())
    avg_cm2 = float(np.mean([p.cm2 for p in top_products]))
    avg_discount = float(np.mean([p.discount for p in top_products]))
    frac_pl = float(np.mean([1.0 if p.is_pl else 0.0 for p in top_products]))
    frac_strategic = float(
        np.mean([1.0 if p.strategic_flag else 0.0 for p in top_products])
    )
    avg_bestseller = float(np.mean([p.bestseller for p in top_products]))
    avg_relevance = float(np.mean(np.asarray(base_scores, dtype=float)[top_idx]))

    # 5. Concatenate raw features
    raw = np.concatenate(
        [
            seg_vec,  # 4 dims
            q_vec,  # 3 dims
            [theta_price, theta_pl],  # 2 dims
            [
                avg_price,
                std_price,
                avg_cm2,
                avg_discount,
                frac_pl,
                frac_strategic,
                avg_bestseller,
                avg_relevance,
            ],  # 8 dims
        ]
    )  # Total: 17 dims

    # 6. Standardize (z-score normalization) for aggregates.
    # Segments / query types / user prefs are already well-scaled.
    means = np.array(
        [
            0,
            0,
            0,
            0,  # segments (binary, do not shift)
            0,
            0,
            0,  # query types (binary, do not shift)
            0,
            0,  # theta_price, theta_pl (already normalized)
            30.0,
            15.0,
            0.3,
            0.1,
            0.5,
            0.2,
            300.0,
            5.0,  # aggregates
        ],
        dtype=float,
    )
    stds = np.array(
        [
            1,
            1,
            1,
            1,  # segments
            1,
            1,
            1,  # query types
            1,
            1,  # theta_* (already normalized)
            20.0,
            10.0,
            0.2,
            0.1,
            0.3,
            0.2,
            200.0,
            2.0,  # aggregates
        ],
        dtype=float,
    )

    standardized = (raw - means) / stds
    return _append_bias(standardized)


def context_features_rich_estimated(
    user: User,
    query: Query,
    products: List[Product],
    cfg: SimulatorConfig,
    base_scores: np.ndarray,
) -> np.ndarray:
    """Rich context features with *estimated* user latents.

    This variant simulates a production setting where user preferences are
    only approximately known. It applies a simple shrinkage + coarse
    quantization to the simulator's internal (theta_price, theta_pl)
    before feeding them into the feature vector.

    The rest of the feature construction (segment/query one-hots and
    catalog aggregates over base top-k) matches `context_features_rich`.
    """
    # Derive approximate preferences from simulator latents.
    # 1) Blend with segment priors (simulating production priors),
    # 2) Apply mild shrinkage (under-confident model),
    # 3) Quantize onto a coarse 0.25 grid, and
    # 4) Clip to a plausible range.
    raw_price = float(user.theta_price)
    raw_pl = float(user.theta_pl)

    seg_defaults = cfg.users.segment_params[user.segment]
    blended_price = 0.6 * raw_price + 0.4 * seg_defaults.price_mean
    blended_pl = 0.6 * raw_pl + 0.4 * seg_defaults.pl_mean

    shrunk_price = 0.9 * blended_price
    shrunk_pl = 0.9 * blended_pl

    theta_price_est = _quantize_preference(
        shrunk_price,
        step=0.25,
        min_val=-3.5,
        max_val=3.5,
    )
    theta_pl_est = _quantize_preference(
        shrunk_pl,
        step=0.25,
        min_val=-3.5,
        max_val=3.5,
    )

    # Reuse the rich feature construction but swap in estimated prefs.
    seg_vec = np.zeros(len(cfg.users.segments), dtype=float)
    seg_vec[cfg.users.segments.index(user.segment)] = 1.0

    q_vec = np.zeros(len(cfg.queries.query_types), dtype=float)
    q_vec[cfg.queries.query_types.index(query.query_type)] = 1.0

    k = min(cfg.top_k, len(products))
    top_idx = np.argsort(-np.asarray(base_scores, dtype=float))[:k]
    top_products = [products[i] for i in top_idx]

    prices = np.array([p.price for p in top_products], dtype=float)
    avg_price = float(prices.mean())
    std_price = float(prices.std())
    avg_cm2 = float(np.mean([p.cm2 for p in top_products]))
    avg_discount = float(np.mean([p.discount for p in top_products]))
    frac_pl = float(np.mean([1.0 if p.is_pl else 0.0 for p in top_products]))
    frac_strategic = float(
        np.mean([1.0 if p.strategic_flag else 0.0 for p in top_products])
    )
    avg_bestseller = float(np.mean([p.bestseller for p in top_products]))
    avg_relevance = float(np.mean(np.asarray(base_scores, dtype=float)[top_idx]))

    raw = np.concatenate(
        [
            seg_vec,
            q_vec,
            [theta_price_est, theta_pl_est],
            [
                avg_price,
                std_price,
                avg_cm2,
                avg_discount,
                frac_pl,
                frac_strategic,
                avg_bestseller,
                avg_relevance,
            ],
        ]
    )

    means = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            30.0,
            15.0,
            0.3,
            0.1,
            0.5,
            0.2,
            300.0,
            5.0,
        ],
        dtype=float,
    )
    stds = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            20.0,
            10.0,
            0.2,
            0.1,
            0.3,
            0.2,
            200.0,
            2.0,
        ],
        dtype=float,
    )

    standardized = (raw - means) / stds
    return _append_bias(standardized)


def _progress(prefix: str, step: int, total: int, extra: str = "") -> None:
    """Lightweight textual progress indicator (newline per update)."""
    pct = 100.0 * step / max(total, 1)
    message = f"{prefix}: {pct:5.1f}% ({step}/{total})"
    if extra:
        message += f" | {extra}"
    print(message)


def _simulate_episode_for_template(
    *,
    template: BoostTemplate,
    cfg: SimulatorConfig,
    user: User,
    query: Query,
    products,
    rng: np.random.Generator,
    base_scores: np.ndarray | None = None,
) -> EpisodeResult:
    """Simulate a single episode under a fixed template.

    The ranking logic mirrors zoosim.envs.search_env and multi_episode.session_env:
    - Compute base relevance scores
    - Add template boosts
    - Sort to obtain ranking
    - Simulate session and compute reward / breakdown
    """
    from zoosim.ranking import relevance  # Imported here to avoid cycles at import time

    if base_scores is None:
        base_scores = np.asarray(
            relevance.batch_base_scores(
                query=query,
                catalog=products,
                config=cfg,
                rng=rng,
            ),
            dtype=float,
        )
    else:
        base_scores = np.asarray(base_scores, dtype=float)

    boosts = template.apply(products)
    blended = base_scores + boosts
    ranking = np.argsort(-blended).tolist()

    outcome = behavior.simulate_session(
        user=user,
        query=query,
        ranking=ranking,
        catalog=products,
        config=cfg,
        rng=rng,
    )
    reward_value, breakdown = reward.compute_reward(
        ranking=ranking,
        clicks=outcome.clicks,
        buys=outcome.buys,
        catalog=products,
        config=cfg,
    )
    orders = float(sum(outcome.buys))
    return EpisodeResult(
        reward=float(reward_value),
        gmv=breakdown.gmv,
        cm2=breakdown.cm2,
        orders=orders,
    )


def _run_static_templates(
    *,
    cfg: SimulatorConfig,
    products,
    templates: List[BoostTemplate],
    n_episodes: int,
    seed: int,
) -> Dict[int, EpisodeResult]:
    """Evaluate each template as a static policy over n_episodes."""
    from zoosim.ranking import relevance

    rng = np.random.default_rng(seed)
    totals: Dict[int, Tuple[float, float, float, float]] = {
        t.id: (0.0, 0.0, 0.0, 0.0) for t in templates
    }

    progress_interval = max(1, n_episodes // 50)  # 2% increments

    for episode_idx in range(1, n_episodes + 1):
        user = sample_user(config=cfg, rng=rng)
        query = sample_query(user=user, config=cfg, rng=rng)

        base_scores = np.asarray(
            relevance.batch_base_scores(
                query=query,
                catalog=products,
                config=cfg,
                rng=rng,
            ),
            dtype=float,
        )

        for template in templates:
            key = template.id
            episode = _simulate_episode_for_template(
                template=template,
                cfg=cfg,
                user=user,
                query=query,
                products=products,
                rng=rng,
                base_scores=base_scores,
            )
            r_sum, g_sum, c_sum, o_sum = totals[key]
            totals[key] = (
                r_sum + episode.reward,
                g_sum + episode.gmv,
                c_sum + episode.cm2,
                o_sum + episode.orders,
            )
        if (
            episode_idx == 1
            or episode_idx % progress_interval == 0
            or episode_idx == n_episodes
        ):
            # Compute provisional best template average reward
            best_template_id = max(
                totals,
                key=lambda key: totals[key][0] / episode_idx if episode_idx > 0 else 0.0,
            )
            best_template_avg = totals[best_template_id][0] / episode_idx
            extra = (
                f"best template ID {best_template_id} avg reward {best_template_avg:,.2f}"
            )
            _progress("  Static templates progress", episode_idx, n_episodes, extra)

    print("")  # newline after progress

    results: Dict[int, EpisodeResult] = {}
    for template in templates:
        r_sum, g_sum, c_sum, o_sum = totals[template.id]
        results[template.id] = EpisodeResult(
            reward=r_sum / n_episodes,
            gmv=g_sum / n_episodes,
            cm2=c_sum / n_episodes,
            orders=o_sum / n_episodes,
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chapter 6 template bandits demonstration."
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Episode count for both static and bandit sections.",
    )
    parser.add_argument(
        "--n-static",
        type=int,
        default=None,
        help="Episodes for static template baselines (overrides --n-episodes).",
    )
    parser.add_argument(
        "--n-bandit",
        type=int,
        default=None,
        help="Episodes for LinUCB/TS runs (overrides --n-episodes).",
    )
    parser.add_argument(
        "--show-volume",
        action="store_true",
        help="Print order/volume metrics alongside Reward/GMV/CM2.",
    )
    parser.add_argument(
        "--features",
        type=str,
        choices=["simple", "rich", "rich_est"],
        default="simple",
        help=(
            "Feature mode for contextual bandits: "
            "'simple' (segment+query), 'rich' (oracle user latents + aggregates), "
            "or 'rich_est' (approximate user latents + aggregates). "
            "Default 'simple' emphasizes the failure mode."
        ),
    )
    parser.add_argument(
        "--prior-weight",
        type=int,
        default=None,
        help=(
            "Pseudo-count used to seed bandit posteriors with static template results. "
            "If omitted, an adaptive default is chosen per feature mode."
        ),
    )
    parser.add_argument(
        "--lin-alpha",
        type=float,
        default=None,
        help=(
            "Override LinUCB exploration parameter α. "
            "Adaptive default selected per feature mode when omitted."
        ),
    )
    parser.add_argument(
        "--ts-sigma",
        type=float,
        default=None,
        help=(
            "Override Thompson Sampling reward noise σ. "
            "Adaptive default selected per feature mode when omitted."
        ),
    )
    parser.add_argument(
        "--world-seed",
        type=int,
        default=None,
        help="Override SimulatorConfig seed for catalog/users/queries generation.",
    )
    parser.add_argument(
        "--bandit-base-seed",
        type=int,
        default=2025_0601,
        help=(
            "Base RNG seed for experiment episodes. Static templates use this seed, "
            "LinUCB uses +1, Thompson Sampling +2, and the static-best segment sweep +3."
        ),
    )
    parser.add_argument(
        "--hparam-mode",
        type=str,
        choices=["auto", "simple", "rich", "rich_est"],
        default="auto",
        help=(
            "Use heuristic hyperparameters from the selected feature mode. "
            "Default 'auto' matches the active --features value; set to 'rich_est' "
            "when running --features=rich but you want the regularized preset."
        ),
    )
    parser.add_argument(
        "--rich-regularization",
        type=str,
        choices=["none", "blend", "quantized"],
        default="none",
        help=(
            "When running --features=rich, optionally regularize oracle user latents "
            "before feeding them to the bandits: 'blend' mixes with segment priors "
            "and shrinkage; 'quantized' also coarse-quantizes (rich_est style)."
        ),
    )
    parser.add_argument(
        "--rich-blend-weight",
        type=float,
        default=0.4,
        help=(
            "Weight assigned to segment priors under --rich-regularization "
            "('blend' or 'quantized'). 0.4 reproduces the rich_est heuristics."
        ),
    )
    parser.add_argument(
        "--rich-shrink",
        type=float,
        default=0.9,
        help="Multiplicative shrink applied after blending oracle latents.",
    )
    parser.add_argument(
        "--rich-quant-step",
        type=float,
        default=0.25,
        help="Quantization grid size used when --rich-regularization=quantized.",
    )
    parser.add_argument(
        "--rich-clip",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-3.5, 3.5),
        help="Clip bounds for regularized oracle latents.",
    )
    return parser.parse_args()


def resolve_prior_weight(feature_mode: str, override: int | None = None) -> int:
    """Select pseudo-count used for static priors."""
    if override is not None:
        return max(0, override)
    if feature_mode == "rich_est":
        return 12
    if feature_mode == "rich":
        return 6
    return 0


def resolve_lin_alpha(feature_mode: str, override: float | None = None) -> float:
    """Select LinUCB exploration width."""
    if override is not None:
        return max(0.1, override)
    return 0.85 if feature_mode == "rich_est" else 1.0


def resolve_ts_sigma(feature_mode: str, override: float | None = None) -> float:
    """Select TS reward noise (controls exploration)."""
    if override is not None:
        return max(0.1, override)
    return 0.7 if feature_mode == "rich_est" else 1.0


def _run_bandit_policy(
    *,
    cfg: SimulatorConfig,
    products,
    templates: List[BoostTemplate],
    n_episodes: int,
    seed: int,
    policy_type: str = "linucb",
    feature_mode: str = "simple",
    static_results: Dict[int, EpisodeResult] | None = None,
    prior_weight: int = 0,
    lin_alpha: float = 1.0,
    ts_sigma: float = 1.0,
    rich_regularization: RichRegularizationConfig | None = None,
) -> Tuple[EpisodeResult, Dict[str, EpisodeResult], Dict[str, Any]]:
    """Run a contextual bandit over templates for n_episodes.

    Returns:
        global_result: EpisodeResult averaged over all episodes
        segment_results: Mapping user_segment -> EpisodeResult averaged
                         over episodes where that segment appeared.
    """
    from zoosim.ranking import relevance

    rng = np.random.default_rng(seed)

    if feature_mode == "simple":
        d = len(cfg.users.segments) + len(cfg.queries.query_types)
    elif feature_mode in ("rich", "rich_est"):
        d = (
            len(cfg.users.segments)
            + len(cfg.queries.query_types)
            + 2  # user preference proxies
            + 8  # aggregates from base top-k
        )
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    d += 1  # bias feature

    if policy_type == "linucb":
        policy = LinUCB(
            templates=templates,
            feature_dim=d,
            config=LinUCBConfig(
                lambda_reg=1.0,
                alpha=lin_alpha,
                adaptive_alpha=True,
                seed=seed,
            ),
        )
    elif policy_type == "ts":
        policy = LinearThompsonSampling(
            templates=templates,
            feature_dim=d,
            config=ThompsonSamplingConfig(
                lambda_reg=1.0,
                sigma_noise=ts_sigma,
                seed=seed,
            ),
        )
    else:
        raise ValueError(f"Unknown policy_type: {policy_type}")

    _seed_policy_with_static_prior(
        policy=policy,
        templates=templates,
        static_results=static_results,
        feature_dim=d,
        prior_weight=prior_weight,
    )

    reward_total = 0.0
    gmv_total = 0.0
    cm2_total = 0.0
    orders_total = 0.0

    # Per-segment aggregates: segment -> (sum_reward, sum_gmv, sum_cm2, sum_orders, count)
    seg_totals: Dict[str, Tuple[float, float, float, float, int]] = {}
    template_counts = np.zeros(len(templates), dtype=int)

    progress_interval = max(1, n_episodes // 50)

    for episode_idx in range(1, n_episodes + 1):
        user = sample_user(config=cfg, rng=rng)
        query = sample_query(user=user, config=cfg, rng=rng)

        base_scores = np.asarray(
            relevance.batch_base_scores(
                query=query,
                catalog=products,
                config=cfg,
                rng=rng,
            ),
            dtype=float,
        )

        if feature_mode == "simple":
            phi = context_features_simple(user, query, cfg)
        elif feature_mode == "rich":
            phi = context_features_rich(
                user,
                query,
                products,
                cfg,
                base_scores,
                regularization=rich_regularization,
            )
        elif feature_mode == "rich_est":
            phi = context_features_rich_estimated(
                user,
                query,
                products,
                cfg,
                base_scores,
            )
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        action = policy.select_action(phi.astype(np.float64))
        template_counts[action] += 1

        episode = _simulate_episode_for_template(
            template=templates[action],
            cfg=cfg,
            user=user,
            query=query,
            products=products,
            rng=rng,
            base_scores=base_scores,
        )

        policy.update(
            action=action,
            features=phi.astype(np.float64),
            reward=episode.reward,
        )

        reward_total += episode.reward
        gmv_total += episode.gmv
        cm2_total += episode.cm2
        orders_total += episode.orders

        seg_key = user.segment
        s_r, s_g, s_c, s_o, s_n = seg_totals.get(seg_key, (0.0, 0.0, 0.0, 0.0, 0))
        seg_totals[seg_key] = (
            s_r + episode.reward,
            s_g + episode.gmv,
            s_c + episode.cm2,
            s_o + episode.orders,
            s_n + 1,
        )
        if (
            episode_idx == 1
            or episode_idx % progress_interval == 0
            or episode_idx == n_episodes
        ):
            avg_reward = reward_total / episode_idx
            extra = f"avg reward {avg_reward:,.2f}"
            _progress(f"  {policy_type.upper()} progress", episode_idx, n_episodes, extra)

    print("")  # newline after progress

    global_result = EpisodeResult(
        reward=reward_total / n_episodes,
        gmv=gmv_total / n_episodes,
        cm2=cm2_total / n_episodes,
        orders=orders_total / n_episodes,
    )

    segment_results: Dict[str, EpisodeResult] = {}
    for seg, (s_r, s_g, s_c, s_o, s_n) in seg_totals.items():
        if s_n > 0:
            segment_results[seg] = EpisodeResult(
                reward=s_r / s_n,
                gmv=s_g / s_n,
                cm2=s_c / s_n,
                orders=s_o / s_n,
            )
    diagnostics = {
        "template_counts": template_counts.tolist(),
        "template_freqs": (template_counts / max(n_episodes, 1)).tolist(),
    }

    return global_result, segment_results, diagnostics


def _seed_policy_with_static_prior(
    *,
    policy,
    templates: List[BoostTemplate],
    static_results: Dict[int, EpisodeResult] | None,
    feature_dim: int,
    prior_weight: int,
) -> None:
    """Warm-start bandit posteriors using static template averages."""
    if not static_results or prior_weight <= 0:
        return

    bias_features = np.zeros(feature_dim, dtype=np.float64)
    bias_features[-1] = 1.0
    id_to_index = {template.id: idx for idx, template in enumerate(templates)}

    for template_id, metrics in static_results.items():
        action_idx = id_to_index.get(template_id)
        if action_idx is None:
            continue
        for _ in range(prior_weight):
            policy.update(
                action=action_idx,
                features=bias_features,
                reward=metrics.reward,
            )


def _run_best_static_with_segments(
    *,
    cfg: SimulatorConfig,
    products,
    template: BoostTemplate,
    n_episodes: int,
    seed: int,
) -> Tuple[EpisodeResult, Dict[str, EpisodeResult]]:
    """Run the best static template and aggregate per-segment statistics."""
    rng = np.random.default_rng(seed)

    reward_total = 0.0
    gmv_total = 0.0
    cm2_total = 0.0
    orders_total = 0.0

    seg_totals: Dict[str, Tuple[float, float, float, float, int]] = {}

    progress_interval = max(1, n_episodes // 50)

    for episode_idx in range(1, n_episodes + 1):
        user = sample_user(config=cfg, rng=rng)
        query = sample_query(user=user, config=cfg, rng=rng)

        episode = _simulate_episode_for_template(
            template=template,
            cfg=cfg,
            user=user,
            query=query,
            products=products,
            rng=rng,
            base_scores=None,
        )

        reward_total += episode.reward
        gmv_total += episode.gmv
        cm2_total += episode.cm2
        orders_total += episode.orders

        seg_key = user.segment
        s_r, s_g, s_c, s_o, s_n = seg_totals.get(seg_key, (0.0, 0.0, 0.0, 0.0, 0))
        seg_totals[seg_key] = (
            s_r + episode.reward,
            s_g + episode.gmv,
            s_c + episode.cm2,
            s_o + episode.orders,
            s_n + 1,
        )
        if (
            episode_idx == 1
            or episode_idx % progress_interval == 0
            or episode_idx == n_episodes
        ):
            avg_reward = reward_total / episode_idx
            extra = f"avg reward {avg_reward:,.2f}"
            _progress("  Static (best) segment stats", episode_idx, n_episodes, extra)

    print("")  # newline after progress

    global_result = EpisodeResult(
        reward=reward_total / n_episodes,
        gmv=gmv_total / n_episodes,
        cm2=cm2_total / n_episodes,
        orders=orders_total / n_episodes,
    )

    segment_results: Dict[str, EpisodeResult] = {}
    for seg, (s_r, s_g, s_c, s_o, s_n) in seg_totals.items():
        if s_n > 0:
            segment_results[seg] = EpisodeResult(
                reward=s_r / s_n,
                gmv=s_g / s_n,
                cm2=s_c / s_n,
                orders=s_o / s_n,
            )

    return global_result, segment_results


def run_template_bandits_experiment(
    *,
    cfg: SimulatorConfig,
    products: List[Product],
    n_static: int,
    n_bandit: int,
    feature_mode: str,
    base_seed: int,
    prior_weight: int = 0,
    lin_alpha: float = 1.0,
    ts_sigma: float = 1.0,
    rich_regularization: RichRegularizationConfig | None = None,
) -> Dict[str, Any]:
    """Core experiment: static baselines + LinUCB + Thompson Sampling.

    This is the single source of truth for Chapter 6 experiments.
    It is used by the CLI and by chapter-specific scripts.
    """
    stats = compute_catalog_stats(products)
    templates = create_standard_templates(stats, a_max=5.0)

    print(f"\nRunning static template baselines for {n_static} episodes...")
    static_results = _run_static_templates(
        cfg=cfg,
        products=products,
        templates=templates,
        n_episodes=n_static,
        seed=base_seed,
    )

    best_static_id = max(static_results, key=lambda tid: static_results[tid].reward)
    best_template = templates[best_static_id]
    best_static = static_results[best_static_id]

    print(
        f"\nBest static template: ID={best_static_id} ({best_template.name}) "
        f"with avg reward={best_static.reward:.2f}, GMV={best_static.gmv:.2f}, "
        f"CM2={best_static.cm2:.2f}"
    )

    print(
        f"\nRunning LinUCB with features={feature_mode} for "
        f"{n_bandit} episodes..."
    )
    linucb_global, linucb_seg, linucb_diag = _run_bandit_policy(
        cfg=cfg,
        products=products,
        templates=templates,
        n_episodes=n_bandit,
        seed=base_seed + 1,
        policy_type="linucb",
        feature_mode=feature_mode,
        static_results=static_results,
        prior_weight=prior_weight,
        lin_alpha=lin_alpha,
        rich_regularization=rich_regularization,
    )

    print(
        f"Running Thompson Sampling with features={feature_mode} for "
        f"{n_bandit} episodes..."
    )
    ts_global, ts_seg, ts_diag = _run_bandit_policy(
        cfg=cfg,
        products=products,
        templates=templates,
        n_episodes=n_bandit,
        seed=base_seed + 2,
        policy_type="ts",
        feature_mode=feature_mode,
        static_results=static_results,
        prior_weight=prior_weight,
        ts_sigma=ts_sigma,
        rich_regularization=rich_regularization,
    )

    static_best_global, static_best_seg = _run_best_static_with_segments(
        cfg=cfg,
        products=products,
        template=best_template,
        n_episodes=n_bandit,
        seed=base_seed + 3,
    )

    return {
        "config": {
            "seed": base_seed,
            "n_static": n_static,
            "n_bandit": n_bandit,
            "feature_mode": feature_mode,
            "feature_dim": (7 if feature_mode == "simple" else 17) + 1,
            "prior_weight": prior_weight,
            "lin_alpha": lin_alpha,
            "ts_sigma": ts_sigma,
            "rich_regularization": (
                None if rich_regularization is None else rich_regularization.mode
            ),
            "rich_regularization_params": (
                None
                if rich_regularization is None
                else {
                    "blend_weight": rich_regularization.blend_weight,
                    "shrink": rich_regularization.shrink,
                    "quant_step": rich_regularization.quant_step,
                    "clip": [rich_regularization.clip_min, rich_regularization.clip_max],
                }
            ),
        },
        "templates": [
            {"id": t.id, "name": t.name, "description": t.description}
            for t in templates
        ],
        "static_results": {
            tid: {
                "reward": res.reward,
                "gmv": res.gmv,
                "cm2": res.cm2,
                "orders": res.orders,
            }
            for tid, res in static_results.items()
        },
        "static_best": {
            "id": best_static_id,
            "name": best_template.name,
            "result": {
                "reward": best_static.reward,
                "gmv": best_static.gmv,
                "cm2": best_static.cm2,
                "orders": best_static.orders,
            },
            "segments": {
                seg: {
                    "reward": res.reward,
                    "gmv": res.gmv,
                    "cm2": res.cm2,
                    "orders": res.orders,
                }
                for seg, res in static_best_seg.items()
            },
        },
        "linucb": {
            "global": {
                "reward": linucb_global.reward,
                "gmv": linucb_global.gmv,
                "cm2": linucb_global.cm2,
                "orders": linucb_global.orders,
            },
            "segments": {
                seg: {
                    "reward": res.reward,
                    "gmv": res.gmv,
                    "cm2": res.cm2,
                    "orders": res.orders,
                }
                for seg, res in linucb_seg.items()
            },
            "diagnostics": linucb_diag,
        },
        "ts": {
            "global": {
                "reward": ts_global.reward,
                "gmv": ts_global.gmv,
                "cm2": ts_global.cm2,
                "orders": ts_global.orders,
            },
            "segments": {
                seg: {
                    "reward": res.reward,
                    "gmv": res.gmv,
                    "cm2": res.cm2,
                    "orders": res.orders,
                }
                for seg, res in ts_seg.items()
            },
            "diagnostics": ts_diag,
        },
    }


def main() -> None:
    """CLI entry point for template bandit experiments."""
    args = _parse_args()

    def _resolve(value: int | None, fallback: int) -> int:
        if value is None:
            return fallback
        if value <= 0:
            raise ValueError("Episode counts must be positive.")
        return value

    shared = args.n_episodes
    static_episodes = _resolve(
        args.n_static, _resolve(shared, DEFAULT_EPISODES)
    )
    bandit_episodes = _resolve(
        args.n_bandit, _resolve(shared, DEFAULT_EPISODES)
    )

    print("\n" + "=" * 80)
    print("CHAPTER 6 — TEMPLATE BANDITS DEMONSTRATION")
    print("=" * 80)
    print(
        f"\nExperiment horizons: static={static_episodes:,} episodes, "
        f"bandits={bandit_episodes:,} episodes."
    )
    print(f"Feature mode: {args.features!r}")

    # Configuration and world
    cfg = (
        SimulatorConfig(seed=args.world_seed)
        if args.world_seed is not None
        else SimulatorConfig()
    )
    rng = np.random.default_rng(cfg.seed)

    print(
        f"\nGenerating catalog with {cfg.catalog.n_products:,} products "
        f"(world seed={cfg.seed})..."
    )
    products = catalog_module.generate_catalog(cfg.catalog, rng)

    base_seed = args.bandit_base_seed

    hparam_mode = args.hparam_mode if args.hparam_mode != "auto" else args.features

    resolved_prior = resolve_prior_weight(hparam_mode, args.prior_weight)
    resolved_lin_alpha = resolve_lin_alpha(hparam_mode, args.lin_alpha)
    resolved_ts_sigma = resolve_ts_sigma(hparam_mode, args.ts_sigma)

    rich_clip = tuple(args.rich_clip)
    if args.features == "rich" and args.rich_regularization != "none":
        rich_reg = RichRegularizationConfig(
            mode=args.rich_regularization,
            blend_weight=args.rich_blend_weight,
            shrink=args.rich_shrink,
            quant_step=args.rich_quant_step,
            clip_min=rich_clip[0],
            clip_max=rich_clip[1],
        )
    else:
        rich_reg = None

    print(
        f"\nResolved bandit hyperparameters → prior_weight={resolved_prior}, "
        f"lin_alpha={resolved_lin_alpha:.2f}, ts_sigma={resolved_ts_sigma:.2f}"
    )
    if rich_reg is not None:
        print(
            "Rich feature regularization: "
            f"mode={rich_reg.mode}, blend_weight={rich_reg.blend_weight:.2f}, "
            f"shrink={rich_reg.shrink:.2f}, quant_step={rich_reg.quant_step:.2f}, "
            f"clip=[{rich_reg.clip_min:.2f}, {rich_reg.clip_max:.2f}]"
        )

    results = run_template_bandits_experiment(
        cfg=cfg,
        products=products,
        n_static=static_episodes,
        n_bandit=bandit_episodes,
        feature_mode=args.features,
        base_seed=base_seed,
        prior_weight=resolved_prior,
        lin_alpha=resolved_lin_alpha,
        ts_sigma=resolved_ts_sigma,
        rich_regularization=rich_reg,
    )
    results["config"]["world_seed"] = cfg.seed
    results["config"]["bandit_base_seed"] = base_seed

    templates_info = results["templates"]
    static_results = results["static_results"]
    static_best_info = results["static_best"]
    linucb_info = results["linucb"]
    ts_info = results["ts"]

    def pct(delta: float, base: float) -> float:
        return 100.0 * delta / base if base != 0.0 else 0.0

    print("\nStatic templates (per-episode averages):")
    if args.show_volume:
        print(
            f"{'ID':>2s}  {'Template':15s}  {'Reward':>10s}  "
            f"{'GMV':>10s}  {'CM2':>10s}  {'Orders':>10s}"
        )
    else:
        print(
            f"{'ID':>2s}  {'Template':15s}  {'Reward':>10s}  "
            f"{'GMV':>10s}  {'CM2':>10s}"
        )
    for t in templates_info:
        tid = t["id"]
        name = t["name"]
        res = static_results[tid]
        if args.show_volume:
            print(
                f"{tid:2d}  {name:15s}  "
                f"{res['reward']:10.2f}  {res['gmv']:10.2f}  "
                f"{res['cm2']:10.2f}  {res['orders']:10.2f}"
            )
        else:
            print(
                f"{tid:2d}  {name:15s}  "
                f"{res['reward']:10.2f}  {res['gmv']:10.2f}  "
                f"{res['cm2']:10.2f}"
            )

    best_static = static_best_info["result"]
    best_template_name = static_best_info["name"]

    def pct(delta: float, base: float) -> float:
        return 100.0 * delta / base if base != 0.0 else 0.0

    print("\nSummary (per-episode averages):")
    if args.show_volume:
        print(
            f"{'Policy':18s}  {'Reward':>10s}  {'GMV':>10s}  {'CM2':>10s}  {'Orders':>10s}  {'ΔGMV vs static':>14s}"
        )
    else:
        print(f"{'Policy':18s}  {'Reward':>10s}  {'GMV':>10s}  {'CM2':>10s}  {'ΔGMV vs static':>14s}")
    for name, res in [
        (f"Static-{best_template_name}", best_static),
        ("LinUCB", linucb_info["global"]),
        ("ThompsonSampling", ts_info["global"]),
    ]:
        delta_gmv = res["gmv"] - best_static["gmv"]
        if args.show_volume:
            print(
                f"{name:18s}  {res['reward']:10.2f}  {res['gmv']:10.2f}  "
                f"{res['cm2']:10.2f}  {res['orders']:10.2f}  "
                f"{pct(delta_gmv, best_static['gmv']):+13.2f}%"
            )
        else:
            print(
                f"{name:18s}  {res['reward']:10.2f}  {res['gmv']:10.2f}  "
                f"{res['cm2']:10.2f}  "
                f"{pct(delta_gmv, best_static['gmv']):+13.2f}%"
            )

    # Per-segment comparison
    if args.show_volume:
        print("\nPer-segment GMV & Orders (static best vs bandits):")
        print(
            f"{'Segment':15s}  {'Static GMV':>10s}  {'LinUCB GMV':>10s}  "
            f"{'TS GMV':>10s}  {'Static Orders':>14s}  {'LinUCB Orders':>14s}  "
            f"{'TS Orders':>11s}  {'Lin GMV Δ%':>12s}  {'TS GMV Δ%':>11s}  "
            f"{'Lin Orders Δ%':>15s}  {'TS Orders Δ%':>14s}"
        )
    else:
        print("\nPer-segment GMV (static best vs bandits):")
        print(
            f"{'Segment':15s}  {'Static GMV':>10s}  {'LinUCB GMV':>10s}  "
            f"{'TS GMV':>10s}  {'LinUCB Δ%':>11s}  {'TS Δ%':>8s}"
        )

    static_best_global = EpisodeResult(
        reward=best_static["reward"],
        gmv=best_static["gmv"],
        cm2=best_static["cm2"],
        orders=best_static["orders"],
    )
    static_seg_results = {
        seg: EpisodeResult(
            reward=metrics["reward"],
            gmv=metrics["gmv"],
            cm2=metrics["cm2"],
            orders=metrics["orders"],
        )
        for seg, metrics in static_best_info["segments"].items()
    }
    linucb_seg_results = {
        seg: EpisodeResult(
            reward=metrics["reward"],
            gmv=metrics["gmv"],
            cm2=metrics["cm2"],
            orders=metrics["orders"],
        )
        for seg, metrics in linucb_info["segments"].items()
    }
    ts_seg_results = {
        seg: EpisodeResult(
            reward=metrics["reward"],
            gmv=metrics["gmv"],
            cm2=metrics["cm2"],
            orders=metrics["orders"],
        )
        for seg, metrics in ts_info["segments"].items()
    }

    for seg in cfg.users.segments:
        s = static_seg_results.get(seg, static_best_global)
        l = linucb_seg_results.get(seg, static_best_global)
        t = ts_seg_results.get(seg, static_best_global)
        lin_gmv_delta = pct(l.gmv - s.gmv, s.gmv)
        ts_gmv_delta = pct(t.gmv - s.gmv, s.gmv)
        if args.show_volume:
            lin_orders_delta = pct(l.orders - s.orders, s.orders) if s.orders else 0.0
            ts_orders_delta = pct(t.orders - s.orders, s.orders) if s.orders else 0.0
            print(
                f"{seg:15s}  {s.gmv:10.2f}  {l.gmv:10.2f}  {t.gmv:10.2f}  "
                f"{s.orders:14.2f}  {l.orders:14.2f}  {t.orders:11.2f}  "
                f"{lin_gmv_delta:+12.2f}%  {ts_gmv_delta:+11.2f}%  "
                f"{lin_orders_delta:+15.2f}%  {ts_orders_delta:+14.2f}%"
            )
        else:
            print(
                f"{seg:15s}  {s.gmv:10.2f}  {l.gmv:10.2f}  {t.gmv:10.2f}  "
                f"{lin_gmv_delta:+10.2f}%  {ts_gmv_delta:+7.2f}%"
            )

    def _print_template_freqs(label: str, freqs: List[float]) -> None:
        print(f"\nTemplate selection frequencies — {label}:")
        for template, freq in zip(templates_info, freqs):
            print(
                f"  {template['id']:2d} {template['name']:15s}: "
                f"{100.0 * freq:6.2f}%"
            )

    _print_template_freqs("LinUCB", linucb_info["diagnostics"]["template_freqs"])
    _print_template_freqs(
        "ThompsonSampling", ts_info["diagnostics"]["template_freqs"]
    )

    print("\nNotes:")
    print(
        "- This demonstration can use either a simple context feature map "
        "(segment + query type) or a richer feature set with catalog aggregates."
    )
    print(
        "- In many runs, bandits with simple features underperform strong static "
        "templates; richer features typically reduce this gap, revealing when "
        "linear contextual bandits are appropriate."
    )
    print("\nFor theory and detailed diagnostics, see:")
    print("  - docs/book/ch06/discrete_template_bandits.md")
    print("  - docs/book/ch06/exercises_labs.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
