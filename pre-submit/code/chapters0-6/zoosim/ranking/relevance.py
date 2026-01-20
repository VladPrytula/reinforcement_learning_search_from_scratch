"""Hybrid base relevance (semantic + lexical)."""

from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np
import torch
from torch import Tensor

from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import Product
from zoosim.world.queries import Query


def _semantic_component(query: Query, product: Product) -> float:
    return float(torch.nn.functional.cosine_similarity(query.phi_emb, product.embedding, dim=0))


def _lexical_component(query: Query, product: Product) -> float:
    query_tokens = set(query.tokens)
    prod_tokens = set(product.category.split("_"))
    overlap = len(query_tokens & prod_tokens)
    return math.log1p(overlap)


def base_score(*, query: Query, product: Product, config: SimulatorConfig, rng: np.random.Generator) -> float:
    sem = _semantic_component(query, product)
    lex = _lexical_component(query, product)
    rel_cfg = config.relevance
    noise = float(rng.normal(0.0, rel_cfg.noise_sigma))
    return rel_cfg.w_sem * sem + rel_cfg.w_lex * lex + noise


def batch_base_scores(*, query: Query, catalog: Iterable[Product], config: SimulatorConfig, rng: np.random.Generator) -> List[float]:
    return [base_score(query=query, product=prod, config=config, rng=rng) for prod in catalog]
