"""Synthetic product catalog generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

from zoosim.core.config import CatalogConfig


@dataclass
class Product:
    product_id: int
    category: str
    price: float
    cm2: float
    is_pl: bool
    discount: float
    bestseller: float
    embedding: Tensor
    strategic_flag: bool


def _sample_price(cfg: CatalogConfig, category: str, rng: np.random.Generator) -> float:
    params = cfg.price_params[category]
    price = rng.lognormal(mean=params["mu"], sigma=params["sigma"])
    return float(price)


def _sample_cm2(cfg: CatalogConfig, category: str, price: float, rng: np.random.Generator) -> float:
    slope = cfg.margin_slope[category]
    cm2 = slope * price + rng.normal(0.0, cfg.margin_noise)
    return float(cm2)


def _sample_discount(cfg: CatalogConfig, rng: np.random.Generator) -> float:
    if rng.random() < cfg.discount_params["p_zero"]:
        return 0.0
    return float(rng.uniform(cfg.discount_params["low"], cfg.discount_params["high"]))


def _torch_generator(rng: np.random.Generator) -> torch.Generator:
    seed = int(rng.integers(0, 2**31 - 1))
    return torch.Generator().manual_seed(seed)


def _sample_embedding(
    cfg: CatalogConfig,
    category: str,
    rng: np.random.Generator,
    centers: Dict[str, Tensor],
) -> Tensor:
    dim = cfg.embedding_dim
    cluster_std = cfg.emb_cluster_std[category]
    center = centers[category]
    torch_gen = _torch_generator(rng)
    noise = torch.randn(dim, generator=torch_gen) * cluster_std
    emb = (center + noise).to(dtype=torch.float32)
    return emb


def generate_catalog(cfg: CatalogConfig, rng: np.random.Generator) -> List[Product]:
    products: List[Product] = []
    category_choices = cfg.categories
    category_probs = cfg.category_mix

    # Pre-sample a semantic embedding centroid for each category.
    centers: Dict[str, Tensor] = {}
    for category in category_choices:
        torch_gen = _torch_generator(rng)
        centers[category] = torch.randn(cfg.embedding_dim, generator=torch_gen).to(
            dtype=torch.float32
        )

    for pid in range(cfg.n_products):
        category = rng.choice(category_choices, p=category_probs)
        price = _sample_price(cfg, category, rng)
        cm2 = _sample_cm2(cfg, category, price, rng)
        is_pl = bool(rng.random() < cfg.pl_prob[category])
        discount = _sample_discount(cfg, rng)
        bestseller = max(0.0, rng.normal(cfg.bestseller_mean[category], cfg.bestseller_std[category]))
        embedding = _sample_embedding(cfg, category, rng, centers)
        strategic = category in cfg.strategic_categories

        products.append(
            Product(
                product_id=pid,
                category=category,
                price=price,
                cm2=cm2,
                is_pl=is_pl,
                discount=discount,
                bestseller=bestseller,
                embedding=embedding,
                strategic_flag=strategic,
            )
        )

    return products
