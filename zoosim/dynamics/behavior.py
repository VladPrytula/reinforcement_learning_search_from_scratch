"""User behavior & session dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import Product
from zoosim.world.queries import Query
from zoosim.world.users import User


@dataclass
class SessionOutcome:
    clicks: List[int]
    buys: List[int]
    satisfaction: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _category_weight(user: User, product: Product, cfg: SimulatorConfig) -> float:
    idx = cfg.catalog.categories.index(product.category)
    return float(user.theta_cat[idx])


def _semantic_match(query: Query, product: Product) -> float:
    return float(torch.nn.functional.cosine_similarity(query.phi_emb, product.embedding, dim=0))


def _latent_utility(
    user: User,
    query: Query,
    product: Product,
    cfg: SimulatorConfig,
    rng: np.random.Generator,
    price_signal: float | None = None,
) -> float:
    beh = cfg.behavior
    match = _semantic_match(query, product)
    if price_signal is None:
        price_signal = math.log1p(product.price)
    price_term = beh.alpha_price * user.theta_price * price_signal
    pl_term = beh.alpha_pl * user.theta_pl * float(product.is_pl)
    cat_term = beh.alpha_cat * _category_weight(user, product, cfg)
    rel_term = beh.alpha_rel * match
    noise = rng.normal(0.0, beh.sigma_u)
    return rel_term + price_term + pl_term + cat_term + noise


def _position_bias(query: Query, position: int, cfg: SimulatorConfig) -> float:
    bias_vec = cfg.behavior.pos_bias.get(query.query_type)
    if not bias_vec:
        bias_vec = cfg.behavior.pos_bias["category"]
    if position < len(bias_vec):
        return bias_vec[position]
    return bias_vec[-1]


def simulate_session(
    *,
    user: User,
    query: Query,
    ranking: Sequence[int],
    catalog: Sequence[Product],
    config: SimulatorConfig,
    rng,
) -> SessionOutcome:
    top_k = min(config.top_k, len(ranking))
    clicks = [0] * top_k
    buys = [0] * top_k
    satisfaction = 0.0
    beh = config.behavior
    purchase_limit = max(1, beh.max_purchases)
    purchase_count = 0

    for j in range(top_k):
        pid = ranking[j]
        product = catalog[pid]
        price_signal = math.log1p(product.price)

        utility = _latent_utility(user, query, product, config, rng, price_signal=price_signal)
        pos_bias = _position_bias(query, j, config)
        examine_prob = _sigmoid(pos_bias + beh.exam_sensitivity * satisfaction)
        if rng.random() > examine_prob:
            break

        click_prob = _sigmoid(utility)
        clicked = rng.random() < click_prob
        if clicked:
            clicks[j] = 1
            satisfaction += beh.satisfaction_gain * utility

            buy_logit = beh.beta_buy * utility + beh.beta0 - beh.w_price_penalty * price_signal
            if rng.random() < _sigmoid(buy_logit):
                buys[j] = 1
                purchase_count += 1
                fatigue = max(0.0, beh.post_purchase_fatigue)
                satisfaction -= fatigue
                if purchase_count >= purchase_limit:
                    break
                if satisfaction < beh.abandonment_threshold:
                    break
                continue
        else:
            satisfaction -= beh.satisfaction_decay

        if satisfaction < beh.abandonment_threshold:
            break

    return SessionOutcome(clicks=clicks, buys=buys, satisfaction=satisfaction)
