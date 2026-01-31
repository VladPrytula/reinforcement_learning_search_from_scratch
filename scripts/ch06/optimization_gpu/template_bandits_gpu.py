#!/usr/bin/env python
"""GPU-accelerated Chapter 6 template bandit experiments.

This module reimplements the Chapter 6 optimization workflow using PyTorch to
batch the expensive catalog scoring and session simulation steps. The original
`template_bandits_demo.py` walks through every episode sequentially, computing
relevance scores, rankings, and simulator rollouts one at a time. Here we
pre-sample large batches of user/query contexts, evaluate every template in
parallel on the GPU, and then feed the resulting per-template reward traces to
LinUCB and Thompson Sampling.

Design goals:
    - Preserve compatibility with the JSON artifacts expected by downstream
      plotting scripts (`run_bandit_matrix`, `plot_results`).
    - Keep SimulatorConfig / Zoosim modules as the single source of truth for
      catalog stats, templates, and reward weights.
    - Allow both CPU and GPU execution (automatically picks CUDA when
      available, otherwise falls back to torch.float32 tensors on CPU).
    - Provide multiple feature modes (`simple`, `rich`, `rich_est`) that match
      the Chapter 6 narrative, including the optional rich-regularization knob.

API surface:
    - `run_template_bandits_experiment_gpu` mirrors the signature and output of
      the CPU variant so higher-level scripts can swap between the two.
"""

from __future__ import annotations

import warnings
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from zoosim.core.config import SimulatorConfig
from zoosim.dynamics import reward as reward_module
from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
from zoosim.policies.templates import (
    BoostTemplate,
    compute_catalog_stats,
    create_standard_templates,
)
from zoosim.policies.thompson_sampling import (
    LinearThompsonSampling,
    ThompsonSamplingConfig,
)
from zoosim.world import catalog as catalog_module
from zoosim.world.catalog import Product


# ---------------------------------------------------------------------------
# Shared dataclasses (mirrors scripts/ch06/template_bandits_demo.py)
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    reward: float
    gmv: float
    cm2: float
    orders: float


@dataclass
class RichRegularizationConfig:
    mode: str = "none"
    blend_weight: float = 0.4
    shrink: float = 0.9
    quant_step: float = 0.25
    clip_min: float = -3.5
    clip_max: float = 3.5


# ---------------------------------------------------------------------------
# Internal tensor bundles
# ---------------------------------------------------------------------------


@dataclass
class SegmentTensorParams:
    price_mean: Tensor
    price_std: Tensor
    pl_mean: Tensor
    pl_std: Tensor
    cat_conc: Tensor


@dataclass
class GPUWorld:
    device: torch.device
    products: List[Product]
    product_prices: Tensor
    product_cm2: Tensor
    product_price_signal: Tensor
    product_is_pl: Tensor
    product_discount: Tensor
    product_bestseller: Tensor
    product_strategic: Tensor
    product_category_idx: Tensor
    product_embeddings: Tensor
    normalized_embeddings: Tensor
    template_boosts: Tensor
    lexical_matrix: Tensor
    pos_bias: Tensor
    segment_params: SegmentTensorParams
    segment_lookup: Dict[str, int]
    category_lookup: Dict[str, int]
    query_type_lookup: Dict[str, int]


@dataclass
class TemplateBatchMetrics:
    reward: Tensor
    gmv: Tensor
    cm2: Tensor
    orders: Tensor
    clicks: Tensor


def _validate_reward_sample(
    *,
    cfg: SimulatorConfig,
    world: GPUWorld,
    ranking_sample: Tensor,
    clicks_sample: Tensor,
    buys_sample: Tensor,
    reward_sample: float,
) -> None:
    """Re-run compute_reward on a real trace to enforce safety guards."""
    ranking = [int(idx) for idx in ranking_sample.tolist()]
    clicks = [int(round(float(val))) for val in clicks_sample.tolist()]
    buys = [int(round(float(val))) for val in buys_sample.tolist()]
    reward_value, _ = reward_module.compute_reward(
        ranking=ranking,
        clicks=clicks,
        buys=buys,
        catalog=world.products,
        config=cfg,
    )
    if not math.isclose(
        reward_value,
        reward_sample,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise AssertionError(
            "GPU reward tensor drifted from compute_reward result; "
            "check TemplateBatchMetrics construction."
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _as_device(device: Optional[str | torch.device]) -> torch.device:
    if device is None or (isinstance(device, str) and device == "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if isinstance(device, torch.device):
        requested = device
    else:
        requested = torch.device(device)

    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        warnings.warn(
            "MPS device requested but torch.backends.mps.is_available() is False; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return requested


def _stack_product_tensor(values: Iterable[float], device: torch.device) -> Tensor:
    return torch.tensor(list(values), dtype=torch.float32, device=device)


def _build_segment_tensors(cfg: SimulatorConfig, device: torch.device) -> SegmentTensorParams:
    segments = cfg.users.segments
    price_mean = torch.tensor(
        [cfg.users.segment_params[s].price_mean for s in segments],
        dtype=torch.float32,
        device=device,
    )
    price_std = torch.tensor(
        [cfg.users.segment_params[s].price_std for s in segments],
        dtype=torch.float32,
        device=device,
    )
    pl_mean = torch.tensor(
        [cfg.users.segment_params[s].pl_mean for s in segments],
        dtype=torch.float32,
        device=device,
    )
    pl_std = torch.tensor(
        [cfg.users.segment_params[s].pl_std for s in segments],
        dtype=torch.float32,
        device=device,
    )
    cat_conc = torch.tensor(
        [cfg.users.segment_params[s].cat_conc for s in segments],
        dtype=torch.float32,
        device=device,
    )
    return SegmentTensorParams(
        price_mean=price_mean,
        price_std=price_std,
        pl_mean=pl_mean,
        pl_std=pl_std,
        cat_conc=cat_conc,
    )


def _progress(prefix: str, step: int, total: int, extra: str = "") -> None:
    pct = 100.0 * step / max(total, 1)
    suffix = f" — {extra}" if extra else ""
    print(f"{prefix}: {step:,}/{total:,} ({pct:5.1f} %){suffix}", end="\r")
    if step >= total:
        print("")


def _build_lexical_matrix(
    *,
    cfg: SimulatorConfig,
    products: Sequence[Product],
    device: torch.device,
) -> Tensor:
    categories = cfg.catalog.categories
    cat_tokens = [set(cat.split("_")) for cat in categories]
    prod_tokens = [set(prod.category.split("_")) for prod in products]
    matrix = torch.zeros(
        len(categories),
        len(products),
        dtype=torch.float32,
        device=device,
    )
    for cat_idx, q_tokens in enumerate(cat_tokens):
        for prod_idx, p_tokens in enumerate(prod_tokens):
            overlap = len(q_tokens & p_tokens)
            matrix[cat_idx, prod_idx] = float(np.log1p(overlap))
    return matrix


def _build_pos_bias(cfg: SimulatorConfig, device: torch.device) -> Tensor:
    query_types = cfg.queries.query_types
    top_k = cfg.top_k
    matrix = torch.zeros(len(query_types), top_k, dtype=torch.float32, device=device)
    for idx, qtype in enumerate(query_types):
        bias = cfg.behavior.pos_bias.get(qtype, cfg.behavior.pos_bias["category"])
        padded = (bias + [bias[-1]] * top_k)[:top_k]
        matrix[idx] = torch.tensor(padded, dtype=torch.float32, device=device)
    return matrix


def _prepare_world(
    cfg: SimulatorConfig,
    products: List[Product],
    templates: List[BoostTemplate],
    device: torch.device,
) -> GPUWorld:
    category_lookup = {name: idx for idx, name in enumerate(cfg.catalog.categories)}
    query_type_lookup = {name: idx for idx, name in enumerate(cfg.queries.query_types)}
    segment_lookup = {name: idx for idx, name in enumerate(cfg.users.segments)}

    product_category_idx = torch.tensor(
        [category_lookup[p.category] for p in products],
        dtype=torch.long,
        device=device,
    )
    embeddings = torch.stack([p.embedding.to(device=device) for p in products]).to(
        dtype=torch.float32
    )
    normalized_embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    template_boosts = torch.stack(
        [
            torch.tensor(template.apply(products), dtype=torch.float32, device=device)
            for template in templates
        ]
    )

    world = GPUWorld(
        device=device,
        products=products,
        product_prices=_stack_product_tensor((p.price for p in products), device),
        product_cm2=_stack_product_tensor((p.cm2 for p in products), device),
        product_price_signal=_stack_product_tensor(
            (np.log1p(p.price) for p in products), device
        ),
        product_is_pl=_stack_product_tensor(
            (1.0 if p.is_pl else 0.0 for p in products), device
        ),
        product_discount=_stack_product_tensor((p.discount for p in products), device),
        product_bestseller=_stack_product_tensor(
            (p.bestseller for p in products), device
        ),
        product_strategic=_stack_product_tensor(
            (1.0 if p.strategic_flag else 0.0 for p in products), device
        ),
        product_category_idx=product_category_idx,
        product_embeddings=embeddings,
        normalized_embeddings=normalized_embeddings,
        template_boosts=template_boosts,
        lexical_matrix=_build_lexical_matrix(cfg=cfg, products=products, device=device),
        pos_bias=_build_pos_bias(cfg=cfg, device=device),
        segment_params=_build_segment_tensors(cfg, device),
        segment_lookup=segment_lookup,
        category_lookup=category_lookup,
        query_type_lookup=query_type_lookup,
    )
    return world


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_segments(
    *,
    cfg: SimulatorConfig,
    batch: int,
    world: GPUWorld,
    generator: torch.Generator,
) -> Tensor:
    probs = torch.tensor(
        cfg.users.segment_mix,
        dtype=torch.float32,
        device=world.device,
    )
    return torch.multinomial(probs, num_samples=batch, replacement=True, generator=generator)


def _sample_theta(
    *,
    seg_indices: Tensor,
    params: SegmentTensorParams,
    generator: torch.Generator,
    np_rng: np.random.Generator,
) -> Tuple[Tensor, Tensor, Tensor]:
    theta_price = torch.normal(
        mean=params.price_mean[seg_indices],
        std=params.price_std[seg_indices],
        generator=generator,
    )
    theta_pl = torch.normal(
        mean=params.pl_mean[seg_indices],
        std=params.pl_std[seg_indices],
        generator=generator,
    )
    conc = params.cat_conc[seg_indices]
    conc_np = conc.cpu().numpy()
    gamma_np = np_rng.gamma(shape=conc_np, scale=1.0)
    theta_cat = torch.tensor(
        gamma_np,
        dtype=torch.float32,
        device=theta_price.device,
    )
    theta_cat = theta_cat / theta_cat.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return theta_price, theta_pl, theta_cat


def _sample_theta_emb(
    *,
    batch: int,
    cfg: SimulatorConfig,
    world: GPUWorld,
    generator: torch.Generator,
) -> Tensor:
    return torch.randn(
        batch,
        cfg.catalog.embedding_dim,
        generator=generator,
        device=world.device,
        dtype=torch.float32,
    )


def _sample_query_types(
    *,
    cfg: SimulatorConfig,
    batch: int,
    world: GPUWorld,
    generator: torch.Generator,
) -> Tensor:
    probs = torch.tensor(
        cfg.queries.query_type_mix,
        dtype=torch.float32,
        device=world.device,
    )
    return torch.multinomial(probs, num_samples=batch, replacement=True, generator=generator)


def _sample_query_intents(
    theta_cat: Tensor,
    generator: torch.Generator,
) -> Tensor:
    cdf = theta_cat.cumsum(dim=-1)
    u = torch.rand(theta_cat.size(0), 1, device=theta_cat.device, generator=generator)
    intent = torch.searchsorted(cdf, u, right=True).squeeze(-1)
    return intent


def _sample_query_embeddings(
    *,
    theta_emb: Tensor,
    cfg: SimulatorConfig,
    world: GPUWorld,
    generator: torch.Generator,
) -> Tensor:
    noise = torch.randn(theta_emb.shape, generator=generator, device=world.device, dtype=theta_emb.dtype)
    return theta_emb + noise * 0.05


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _one_hot(indices: Tensor, num_classes: int) -> Tensor:
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).to(dtype=torch.float32)


def _quantize(
    values: Tensor,
    step: float,
    min_val: float,
    max_val: float,
) -> Tensor:
    rounded = torch.round(values / step) * step
    return torch.clamp(rounded, min_val, max_val)


def _apply_rich_regularization(
    theta_price: Tensor,
    theta_pl: Tensor,
    seg_indices: Tensor,
    cfg: SimulatorConfig,
    reg_cfg: Optional[RichRegularizationConfig],
    world: GPUWorld,
) -> Tuple[Tensor, Tensor]:
    if reg_cfg is None or reg_cfg.mode in (None, "none"):
        return theta_price, theta_pl

    seg_params = world.segment_params
    price_mean = seg_params.price_mean[seg_indices]
    pl_mean = seg_params.pl_mean[seg_indices]

    blend_w = reg_cfg.blend_weight
    blended_price = (1.0 - blend_w) * theta_price + blend_w * price_mean
    blended_pl = (1.0 - blend_w) * theta_pl + blend_w * pl_mean

    shrunk_price = blended_price * reg_cfg.shrink
    shrunk_pl = blended_pl * reg_cfg.shrink

    if reg_cfg.mode == "quantized":
        q_price = _quantize(
            shrunk_price,
            step=reg_cfg.quant_step,
            min_val=reg_cfg.clip_min,
            max_val=reg_cfg.clip_max,
        )
        q_pl = _quantize(
            shrunk_pl,
            step=reg_cfg.quant_step,
            min_val=reg_cfg.clip_min,
            max_val=reg_cfg.clip_max,
        )
        return q_price, q_pl

    return (
        torch.clamp(shrunk_price, reg_cfg.clip_min, reg_cfg.clip_max),
        torch.clamp(shrunk_pl, reg_cfg.clip_min, reg_cfg.clip_max),
    )


def _estimate_latents(
    theta_price: Tensor,
    theta_pl: Tensor,
    seg_indices: Tensor,
    world: GPUWorld,
) -> Tuple[Tensor, Tensor]:
    seg_params = world.segment_params
    price_mean = seg_params.price_mean[seg_indices]
    pl_mean = seg_params.pl_mean[seg_indices]

    blended_price = 0.6 * theta_price + 0.4 * price_mean
    blended_pl = 0.6 * theta_pl + 0.4 * pl_mean
    shrunk_price = 0.9 * blended_price
    shrunk_pl = 0.9 * blended_pl
    return (
        _quantize(shrunk_price, step=0.25, min_val=-3.5, max_val=3.5),
        _quantize(shrunk_pl, step=0.25, min_val=-3.5, max_val=3.5),
    )


def _compute_base_topk_aggregates(
    *,
    base_scores: Tensor,
    top_indices: Tensor,
    world: GPUWorld,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    prices = world.product_prices[top_indices]
    cm2 = world.product_cm2[top_indices]
    discounts = world.product_discount[top_indices]
    pl_flags = world.product_is_pl[top_indices]
    strategic = world.product_strategic[top_indices]
    bestseller = world.product_bestseller[top_indices]
    relevance = base_scores.gather(dim=-1, index=top_indices)

    avg_price = prices.mean(dim=-1)
    std_price = prices.std(dim=-1, unbiased=False)
    avg_cm2 = cm2.mean(dim=-1)
    avg_discount = discounts.mean(dim=-1)
    frac_pl = pl_flags.mean(dim=-1)
    frac_strategic = strategic.mean(dim=-1)
    avg_bestseller = bestseller.mean(dim=-1)
    avg_relevance = relevance.mean(dim=-1)
    return (
        avg_price,
        std_price,
        avg_cm2,
        avg_discount,
        frac_pl,
        frac_strategic,
        avg_bestseller,
        avg_relevance,
    )


def _build_features(
    *,
    feature_mode: str,
    cfg: SimulatorConfig,
    seg_indices: Tensor,
    query_type_idx: Tensor,
    theta_price: Tensor,
    theta_pl: Tensor,
    theta_cat: Tensor,
    base_scores: Tensor,
    base_top_indices: Tensor,
    world: GPUWorld,
    rich_regularization: Optional[RichRegularizationConfig],
) -> Tensor:
    n_segments = len(cfg.users.segments)
    n_queries = len(cfg.queries.query_types)
    seg_one_hot = _one_hot(seg_indices, n_segments)
    query_one_hot = _one_hot(query_type_idx, n_queries)
    bias = torch.ones(seg_one_hot.size(0), 1, device=world.device)

    if feature_mode == "simple":
        return torch.cat([seg_one_hot, query_one_hot, bias], dim=-1)

    aggregates = _compute_base_topk_aggregates(
        base_scores=base_scores,
        top_indices=base_top_indices,
        world=world,
    )
    (
        avg_price,
        std_price,
        avg_cm2,
        avg_discount,
        frac_pl,
        frac_strategic,
        avg_bestseller,
        avg_relevance,
    ) = aggregates

    if feature_mode == "rich":
        theta_price_adj, theta_pl_adj = _apply_rich_regularization(
            theta_price,
            theta_pl,
            seg_indices,
            cfg,
            rich_regularization,
            world,
        )
    elif feature_mode == "rich_est":
        theta_price_adj, theta_pl_adj = _estimate_latents(
            theta_price,
            theta_pl,
            seg_indices,
            world,
        )
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    raw = torch.cat(
        [
            seg_one_hot,
            query_one_hot,
            theta_price_adj.unsqueeze(-1),
            theta_pl_adj.unsqueeze(-1),
            avg_price.unsqueeze(-1),
            std_price.unsqueeze(-1),
            avg_cm2.unsqueeze(-1),
            avg_discount.unsqueeze(-1),
            frac_pl.unsqueeze(-1),
            frac_strategic.unsqueeze(-1),
            avg_bestseller.unsqueeze(-1),
            avg_relevance.unsqueeze(-1),
        ],
        dim=-1,
    )

    means = torch.tensor(
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
        dtype=torch.float32,
        device=world.device,
    )
    stds = torch.tensor(
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
        dtype=torch.float32,
        device=world.device,
    )
    standardized = (raw - means) / stds
    return torch.cat([standardized, bias], dim=-1)


# ---------------------------------------------------------------------------
# Relevance + session simulation
# ---------------------------------------------------------------------------


def _compute_base_scores(
    *,
    query_embeddings: Tensor,
    query_intent_idx: Tensor,
    world: GPUWorld,
    cfg: SimulatorConfig,
    generator: torch.Generator,
) -> Tensor:
    normalized_queries = torch.nn.functional.normalize(query_embeddings, dim=-1)
    semantic = normalized_queries @ world.normalized_embeddings.T
    lexical = world.lexical_matrix[query_intent_idx]
    noise = torch.randn(
        semantic.shape,
        generator=generator,
        device=world.device,
        dtype=semantic.dtype,
    ) * cfg.relevance.noise_sigma
    return cfg.relevance.w_sem * semantic + cfg.relevance.w_lex * lexical + noise


def _simulate_sessions(
    *,
    base_scores: Tensor,
    user_theta_price: Tensor,
    user_theta_pl: Tensor,
    user_theta_cat: Tensor,
    query_embeddings: Tensor,
    query_type_idx: Tensor,
    world: GPUWorld,
    cfg: SimulatorConfig,
    generator: torch.Generator,
) -> Tuple[TemplateBatchMetrics, Tensor]:
    batch = base_scores.size(0)
    templates = world.template_boosts.size(0)
    scores = base_scores.unsqueeze(1) + world.template_boosts.unsqueeze(0)
    top_scores, top_indices = torch.topk(scores, k=cfg.top_k, dim=-1)

    rankings = top_indices.view(batch * templates, cfg.top_k)
    theta_price = user_theta_price.repeat_interleave(templates)
    theta_pl = user_theta_pl.repeat_interleave(templates)
    theta_cat = user_theta_cat.repeat_interleave(templates, dim=0)
    query_emb = query_embeddings.repeat_interleave(templates, dim=0)
    query_types = query_type_idx.repeat_interleave(templates)

    total = rankings.size(0)
    top_k = rankings.size(1)
    device = world.device
    beh = cfg.behavior

    clicks = torch.zeros(total, top_k, dtype=torch.float32, device=device)
    buys = torch.zeros_like(clicks)
    satisfaction = torch.zeros(total, dtype=torch.float32, device=device)
    purchase_count = torch.zeros(total, dtype=torch.int32, device=device)
    active = torch.ones(total, dtype=torch.bool, device=device)
    purchase_limit = max(1, beh.max_purchases)

    for pos in range(top_k):
        pid = rankings[:, pos]
        price_signal = world.product_price_signal[pid]
        cat_idx = world.product_category_idx[pid]
        cat_weight = theta_cat[torch.arange(total, device=device), cat_idx]
        pl_flag = world.product_is_pl[pid]
        match = torch.nn.functional.cosine_similarity(
            query_emb,
            world.product_embeddings[pid],
            dim=-1,
        )
        noise = torch.randn(total, device=device, generator=generator) * beh.sigma_u
        utility = (
            beh.alpha_rel * match
            + beh.alpha_price * theta_price * price_signal
            + beh.alpha_pl * theta_pl * pl_flag
            + beh.alpha_cat * cat_weight
            + noise
        )

        pos_bias = world.pos_bias[query_types, pos]
        examine_prob = torch.sigmoid(pos_bias + beh.exam_sensitivity * satisfaction)
        examine_draw = torch.rand(total, device=device, generator=generator)
        examine_mask = (examine_draw < examine_prob) & active

        inactive_from_skip = active & (~examine_mask)
        if inactive_from_skip.any():
            active[inactive_from_skip] = False

        active_indices = examine_mask.nonzero(as_tuple=False).flatten()
        if active_indices.numel() == 0:
            continue

        util_active = utility[active_indices]
        click_prob = torch.sigmoid(util_active)
        click_draw = torch.rand(
            active_indices.numel(),
            device=device,
            generator=generator,
        )
        clicked_mask = click_draw < click_prob
        clicked_indices = active_indices[clicked_mask]
        non_clicked_indices = active_indices[~clicked_mask]

        if clicked_indices.numel() > 0:
            clicks[clicked_indices, pos] = 1.0
            satisfaction[clicked_indices] += beh.satisfaction_gain * util_active[clicked_mask]
        if non_clicked_indices.numel() > 0:
            satisfaction[non_clicked_indices] -= beh.satisfaction_decay

        if clicked_indices.numel() > 0:
            price_clicked = price_signal[clicked_indices]
            util_clicked = utility[clicked_indices]
            buy_logit = (
                beh.beta_buy * util_clicked
                + beh.beta0
                - beh.w_price_penalty * price_clicked
            )
            buy_prob = torch.sigmoid(buy_logit)
            buy_draw = torch.rand(
                clicked_indices.numel(),
                device=device,
                generator=generator,
            )
            buyers_mask = buy_draw < buy_prob
            buyers = clicked_indices[buyers_mask]
            if buyers.numel() > 0:
                buys[buyers, pos] = 1.0
                purchase_count[buyers] += 1
                satisfaction[buyers] -= max(0.0, beh.post_purchase_fatigue)

        active = examine_mask.clone()
        active[satisfaction < beh.abandonment_threshold] = False
        active[purchase_count >= purchase_limit] = False

    clicks = clicks.view(batch, templates, top_k)
    buys = buys.view(batch, templates, top_k)
    rank_view = top_indices

    gmv = (
        world.product_prices[rank_view] * buys
    ).sum(dim=-1)
    cm2 = (
        world.product_cm2[rank_view] * buys
    ).sum(dim=-1)
    strat = (
        world.product_strategic[rank_view] * buys
    ).sum(dim=-1)
    orders = buys.sum(dim=-1)
    total_clicks = clicks.sum(dim=-1)

    cfg_reward = cfg.reward
    reward = (
        cfg_reward.alpha_gmv * gmv
        + cfg_reward.beta_cm2 * cm2
        + cfg_reward.gamma_strat * strat
        + cfg_reward.delta_clicks * total_clicks
    )
    metrics = TemplateBatchMetrics(
        reward=reward,
        gmv=gmv,
        cm2=cm2,
        orders=orders,
        clicks=total_clicks,
    )
    if batch > 0:
        ranking_sample = top_indices[0, 0].detach().cpu()
        clicks_sample = clicks[0, 0].detach().cpu()
        buys_sample = buys[0, 0].detach().cpu()
        reward_sample = float(reward[0, 0].detach().cpu())
        _validate_reward_sample(
            cfg=cfg,
            world=world,
            ranking_sample=ranking_sample,
            clicks_sample=clicks_sample,
            buys_sample=buys_sample,
            reward_sample=reward_sample,
        )
    return metrics, top_indices


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def _simulate_episode_batch(
    *,
    cfg: SimulatorConfig,
    world: GPUWorld,
    batch_size: int,
    feature_mode: str,
    generator: torch.Generator,
    np_rng: np.random.Generator,
    rich_regularization: Optional[RichRegularizationConfig],
) -> Tuple[Tensor, TemplateBatchMetrics, Tensor]:
    seg_idx = _sample_segments(
        cfg=cfg,
        batch=batch_size,
        world=world,
        generator=generator,
    )
    theta_price, theta_pl, theta_cat = _sample_theta(
        seg_indices=seg_idx,
        params=world.segment_params,
        generator=generator,
        np_rng=np_rng,
    )
    theta_emb = _sample_theta_emb(
        batch=batch_size,
        cfg=cfg,
        world=world,
        generator=generator,
    )
    query_type_idx = _sample_query_types(
        cfg=cfg,
        batch=batch_size,
        world=world,
        generator=generator,
    )
    query_intent_idx = _sample_query_intents(theta_cat, generator)
    query_emb = _sample_query_embeddings(
        theta_emb=theta_emb,
        cfg=cfg,
        world=world,
        generator=generator,
    )

    base_scores = _compute_base_scores(
        query_embeddings=query_emb,
        query_intent_idx=query_intent_idx,
        world=world,
        cfg=cfg,
        generator=generator,
    )
    base_top_indices = torch.topk(base_scores, k=cfg.top_k, dim=-1).indices
    features = _build_features(
        feature_mode=feature_mode,
        cfg=cfg,
        seg_indices=seg_idx,
        query_type_idx=query_type_idx,
        theta_price=theta_price,
        theta_pl=theta_pl,
        theta_cat=theta_cat,
        base_scores=base_scores,
        base_top_indices=base_top_indices,
        world=world,
        rich_regularization=rich_regularization,
    )

    metrics, _ = _simulate_sessions(
        base_scores=base_scores,
        user_theta_price=theta_price,
        user_theta_pl=theta_pl,
        user_theta_cat=theta_cat,
        query_embeddings=query_emb,
        query_type_idx=query_type_idx,
        world=world,
        cfg=cfg,
        generator=generator,
    )
    return features, metrics, seg_idx


def _feature_dim(cfg: SimulatorConfig, feature_mode: str) -> int:
    """Mirror CPU logic for contextual feature dimensionality."""
    base = len(cfg.users.segments) + len(cfg.queries.query_types)
    if feature_mode == "simple":
        return base + 1  # bias term
    if feature_mode in ("rich", "rich_est"):
        return base + 10 + 1  # + user prefs (2) + aggregates (8) + bias
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def _compute_static_results(
    *,
    cfg: SimulatorConfig,
    world: GPUWorld,
    templates: List[BoostTemplate],
    n_episodes: int,
    batch_size: int,
    generator: torch.Generator,
    np_rng: np.random.Generator,
) -> Dict[int, EpisodeResult]:
    reward_sum = torch.zeros(len(templates), device=world.device)
    gmv_sum = torch.zeros_like(reward_sum)
    cm2_sum = torch.zeros_like(reward_sum)
    orders_sum = torch.zeros_like(reward_sum)

    produced = 0
    while produced < n_episodes:
        current = min(batch_size, n_episodes - produced)
        _, metrics, _ = _simulate_episode_batch(
            cfg=cfg,
            world=world,
            batch_size=current,
            feature_mode="simple",
            generator=generator,
            np_rng=np_rng,
            rich_regularization=None,
        )
        reward_sum += metrics.reward.sum(dim=0)
        gmv_sum += metrics.gmv.sum(dim=0)
        cm2_sum += metrics.cm2.sum(dim=0)
        orders_sum += metrics.orders.sum(dim=0)
        produced += current

    results: Dict[int, EpisodeResult] = {}
    for idx, template in enumerate(templates):
        results[template.id] = EpisodeResult(
            reward=float(reward_sum[idx].item() / max(n_episodes, 1)),
            gmv=float(gmv_sum[idx].item() / max(n_episodes, 1)),
            cm2=float(cm2_sum[idx].item() / max(n_episodes, 1)),
            orders=float(orders_sum[idx].item() / max(n_episodes, 1)),
        )
    return results


# ---------------------------------------------------------------------------
# Policy + aggregation helpers
# ---------------------------------------------------------------------------


def _episode_result_from_arrays(
    reward: np.ndarray,
    gmv: np.ndarray,
    cm2: np.ndarray,
    orders: np.ndarray,
) -> EpisodeResult:
    return EpisodeResult(
        reward=float(np.mean(reward)) if reward.size else 0.0,
        gmv=float(np.mean(gmv)) if gmv.size else 0.0,
        cm2=float(np.mean(cm2)) if cm2.size else 0.0,
        orders=float(np.mean(orders)) if orders.size else 0.0,
    )


def _segment_episode_results(
    *,
    segment_indices: np.ndarray,
    segment_names: Sequence[str],
    reward: np.ndarray,
    gmv: np.ndarray,
    cm2: np.ndarray,
    orders: np.ndarray,
) -> Dict[str, EpisodeResult]:
    totals = {
        name: [0.0, 0.0, 0.0, 0.0, 0]
        for name in segment_names
    }
    for idx, seg_id in enumerate(segment_indices):
        name = segment_names[int(seg_id)]
        bucket = totals[name]
        bucket[0] += float(reward[idx])
        bucket[1] += float(gmv[idx])
        bucket[2] += float(cm2[idx])
        bucket[3] += float(orders[idx])
        bucket[4] += 1

    results: Dict[str, EpisodeResult] = {}
    for name, (s_r, s_g, s_c, s_o, count) in totals.items():
        if count == 0:
            continue
        inv = 1.0 / count
        results[name] = EpisodeResult(
            reward=s_r * inv,
            gmv=s_g * inv,
            cm2=s_c * inv,
            orders=s_o * inv,
        )
    return results


def _seed_policy_with_static_prior(
    *,
    policy,
    templates: List[BoostTemplate],
    static_results: Dict[int, EpisodeResult],
    feature_dim: int,
    prior_weight: int,
) -> None:
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


def _compute_best_static_segments(
    *,
    cfg: SimulatorConfig,
    world: GPUWorld,
    template_idx: int,
    n_episodes: int,
    batch_size: int,
    generator: torch.Generator,
    np_rng: np.random.Generator,
) -> Dict[str, EpisodeResult]:
    """Replay the best static template to capture per-segment stats."""
    reward_batches: List[np.ndarray] = []
    gmv_batches: List[np.ndarray] = []
    cm2_batches: List[np.ndarray] = []
    orders_batches: List[np.ndarray] = []
    segment_batches: List[np.ndarray] = []

    produced = 0
    while produced < n_episodes:
        current = min(batch_size, n_episodes - produced)
        _, metrics, seg_idx = _simulate_episode_batch(
            cfg=cfg,
            world=world,
            batch_size=current,
            feature_mode="simple",
            generator=generator,
            np_rng=np_rng,
            rich_regularization=None,
        )
        reward_batches.append(
            metrics.reward[:, template_idx].detach().cpu().numpy()
        )
        gmv_batches.append(metrics.gmv[:, template_idx].detach().cpu().numpy())
        cm2_batches.append(metrics.cm2[:, template_idx].detach().cpu().numpy())
        orders_batches.append(metrics.orders[:, template_idx].detach().cpu().numpy())
        segment_batches.append(seg_idx.detach().cpu().numpy())
        produced += current

    reward = np.concatenate(reward_batches, axis=0)[:n_episodes]
    gmv = np.concatenate(gmv_batches, axis=0)[:n_episodes]
    cm2 = np.concatenate(cm2_batches, axis=0)[:n_episodes]
    orders = np.concatenate(orders_batches, axis=0)[:n_episodes]
    segments = np.concatenate(segment_batches, axis=0)[:n_episodes]
    return _segment_episode_results(
        segment_indices=segments,
        segment_names=cfg.users.segments,
        reward=reward,
        gmv=gmv,
        cm2=cm2,
        orders=orders,
    )


def _run_policy_interactive(
    *,
    policy,
    cfg: SimulatorConfig,
    world: GPUWorld,
    templates: Sequence[BoostTemplate],
    n_episodes: int,
    batch_size: int,
    feature_mode: str,
    generator: torch.Generator,
    np_rng: np.random.Generator,
    segment_names: Sequence[str],
    rich_regularization: Optional[RichRegularizationConfig],
    policy_name: str,
) -> Tuple[EpisodeResult, Dict[str, EpisodeResult], Dict[str, Any]]:
    """Sample episodes on the fly so actions influence observed rewards."""
    episode_reward = np.zeros(n_episodes, dtype=np.float64)
    episode_gmv = np.zeros_like(episode_reward)
    episode_cm2 = np.zeros_like(episode_reward)
    episode_orders = np.zeros_like(episode_reward)
    episode_segments = np.zeros(n_episodes, dtype=np.int64)
    template_counts = np.zeros(len(templates), dtype=np.int64)

    produced = 0
    progress_interval = max(1, n_episodes // 50)
    while produced < n_episodes:
        current = min(batch_size, n_episodes - produced)
        features, metrics, seg_idx = _simulate_episode_batch(
            cfg=cfg,
            world=world,
            batch_size=current,
            feature_mode=feature_mode,
            generator=generator,
            np_rng=np_rng,
            rich_regularization=rich_regularization,
        )
        features_np = features.detach().cpu().numpy().astype(np.float64, copy=False)
        reward_np = metrics.reward.detach().cpu().numpy()
        gmv_np = metrics.gmv.detach().cpu().numpy()
        cm2_np = metrics.cm2.detach().cpu().numpy()
        orders_np = metrics.orders.detach().cpu().numpy()
        segments_np = seg_idx.detach().cpu().numpy()

        for local_idx in range(current):
            idx = produced + local_idx
            phi = features_np[local_idx]
            action = policy.select_action(phi)
            template_counts[action] += 1
            reward_value = float(reward_np[local_idx, action])
            policy.update(action=action, features=phi, reward=reward_value)
            episode_reward[idx] = reward_value
            episode_gmv[idx] = float(gmv_np[local_idx, action])
            episode_cm2[idx] = float(cm2_np[local_idx, action])
            episode_orders[idx] = float(orders_np[local_idx, action])
            episode_segments[idx] = int(segments_np[local_idx])

        produced += current
        idx = produced
        if (
            idx == 0
            or idx % progress_interval == 0
            or idx == n_episodes
        ):
            avg_reward = episode_reward[:idx].mean() if idx > 0 else 0.0
            _progress(
                f"  {policy_name} progress",
                idx,
                n_episodes,
                f"avg reward {avg_reward:,.2f}",
            )

    global_result = _episode_result_from_arrays(
        episode_reward,
        episode_gmv,
        episode_cm2,
        episode_orders,
    )
    segment_results = _segment_episode_results(
        segment_indices=episode_segments,
        segment_names=segment_names,
        reward=episode_reward,
        gmv=episode_gmv,
        cm2=episode_cm2,
        orders=episode_orders,
    )
    diagnostics = {
        "template_counts": template_counts.tolist(),
        "template_freqs": (template_counts / max(n_episodes, 1)).tolist(),
    }
    return global_result, segment_results, diagnostics


def _result_dict(result: EpisodeResult) -> Dict[str, float]:
    return {
        "reward": result.reward,
        "gmv": result.gmv,
        "cm2": result.cm2,
        "orders": result.orders,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_template_bandits_experiment_gpu(
    *,
    cfg: SimulatorConfig,
    products: List[Product],
    n_static: int,
    n_bandit: int,
    feature_mode: str,
    base_seed: int,
    prior_weight: int,
    lin_alpha: float,
    ts_sigma: float,
    rich_regularization: Optional[RichRegularizationConfig] = None,
    batch_size: int = 1024,
    device: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    if n_static <= 0 or n_bandit <= 0:
        raise ValueError("Episode counts must be positive.")
    if feature_mode != "rich" and rich_regularization is not None:
        raise ValueError("rich_regularization is only supported when feature_mode='rich'.")
    device_obj = _as_device(device)
    rng = np.random.default_rng(cfg.seed)
    catalog = products or catalog_module.generate_catalog(cfg.catalog, rng)
    catalog_stats = compute_catalog_stats(catalog)
    templates = create_standard_templates(catalog_stats)
    world = _prepare_world(cfg, catalog, templates, device_obj)

    static_generator = torch.Generator(device=device_obj).manual_seed(base_seed)
    np_rng_static = np.random.default_rng(base_seed)
    static_results = _compute_static_results(
        cfg=cfg,
        world=world,
        templates=templates,
        n_episodes=n_static,
        batch_size=batch_size,
        generator=static_generator,
        np_rng=np_rng_static,
    )

    template_index = {template.id: idx for idx, template in enumerate(templates)}
    best_template_id = max(static_results.items(), key=lambda kv: kv[1].reward)[0]
    best_template = templates[template_index[best_template_id]]
    best_idx = template_index[best_template_id]

    static_best_generator = torch.Generator(device=device_obj).manual_seed(base_seed + 3)
    static_best_np = np.random.default_rng(base_seed + 3)
    static_best_segments = _compute_best_static_segments(
        cfg=cfg,
        world=world,
        template_idx=best_idx,
        n_episodes=n_bandit,
        batch_size=batch_size,
        generator=static_best_generator,
        np_rng=static_best_np,
    )

    feature_dim = _feature_dim(cfg, feature_mode)
    lin_policy = LinUCB(
        templates=templates,
        feature_dim=feature_dim,
        config=LinUCBConfig(
            lambda_reg=1.0,
            alpha=lin_alpha,
            adaptive_alpha=True,
            seed=base_seed + 1,
        ),
    )
    ts_policy = LinearThompsonSampling(
        templates=templates,
        feature_dim=feature_dim,
        config=ThompsonSamplingConfig(
            lambda_reg=1.0,
            sigma_noise=ts_sigma,
            seed=base_seed + 2,
        ),
    )

    _seed_policy_with_static_prior(
        policy=lin_policy,
        templates=templates,
        static_results=static_results,
        feature_dim=feature_dim,
        prior_weight=prior_weight,
    )
    _seed_policy_with_static_prior(
        policy=ts_policy,
        templates=templates,
        static_results=static_results,
        feature_dim=feature_dim,
        prior_weight=prior_weight,
    )

    lin_generator = torch.Generator(device=device_obj).manual_seed(base_seed + 1)
    lin_np_rng = np.random.default_rng(base_seed + 1)
    ts_generator = torch.Generator(device=device_obj).manual_seed(base_seed + 2)
    ts_np_rng = np.random.default_rng(base_seed + 2)

    lin_global, lin_segments, lin_diag = _run_policy_interactive(
        policy=lin_policy,
        cfg=cfg,
        world=world,
        templates=templates,
        n_episodes=n_bandit,
        batch_size=batch_size,
        feature_mode=feature_mode,
        generator=lin_generator,
        np_rng=lin_np_rng,
        segment_names=cfg.users.segments,
        rich_regularization=rich_regularization,
        policy_name="LinUCB",
    )
    ts_global, ts_segments, ts_diag = _run_policy_interactive(
        policy=ts_policy,
        cfg=cfg,
        world=world,
        templates=templates,
        n_episodes=n_bandit,
        batch_size=batch_size,
        feature_mode=feature_mode,
        generator=ts_generator,
        np_rng=ts_np_rng,
        segment_names=cfg.users.segments,
        rich_regularization=rich_regularization,
        policy_name="ThompsonSampling",
    )

    rich_params = (
        None
        if rich_regularization is None
        else {
            "blend_weight": rich_regularization.blend_weight,
            "shrink": rich_regularization.shrink,
            "quant_step": rich_regularization.quant_step,
            "clip": [rich_regularization.clip_min, rich_regularization.clip_max],
        }
    )

    return {
        "config": {
            "seed": base_seed,
            "n_static": n_static,
            "n_bandit": n_bandit,
            "batch_size": batch_size,
            "feature_mode": feature_mode,
            "feature_dim": feature_dim,
            "prior_weight": prior_weight,
            "lin_alpha": lin_alpha,
            "ts_sigma": ts_sigma,
            "device": str(device_obj),
            "rich_regularization": None
            if rich_regularization is None
            else rich_regularization.mode,
            "rich_regularization_params": rich_params,
        },
        "templates": [
            {"id": t.id, "name": t.name, "description": t.description}
            for t in templates
        ],
        "static_results": {
            template_id: _result_dict(result)
            for template_id, result in static_results.items()
        },
        "static_best": {
            "id": best_template.id,
            "name": best_template.name,
            "result": _result_dict(static_results[best_template.id]),
            "segments": {
                seg: _result_dict(res) for seg, res in static_best_segments.items()
            },
        },
        "linucb": {
            "global": _result_dict(lin_global),
            "segments": {seg: _result_dict(res) for seg, res in lin_segments.items()},
            "diagnostics": lin_diag,
        },
        "ts": {
            "global": _result_dict(ts_global),
            "segments": {seg: _result_dict(res) for seg, res in ts_segments.items()},
            "diagnostics": ts_diag,
        },
    }


# Convenience wrappers for CLI parity ------------------------------------------------


def resolve_prior_weight(feature_mode: str, override: Optional[int] = None) -> int:
    if override is not None:
        return max(0, override)
    if feature_mode == "rich_est":
        return 12
    if feature_mode == "rich":
        return 6
    return 0


def resolve_lin_alpha(feature_mode: str, override: Optional[float] = None) -> float:
    if override is not None:
        return max(0.1, override)
    return 0.85 if feature_mode == "rich_est" else 1.0


def resolve_ts_sigma(feature_mode: str, override: Optional[float] = None) -> float:
    if override is not None:
        return max(0.1, override)
    return 0.7 if feature_mode == "rich_est" else 1.0
