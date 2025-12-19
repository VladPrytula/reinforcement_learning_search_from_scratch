"""Query sampling (intent, type, embeddings, tokens)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import Tensor

from zoosim.core.config import SimulatorConfig
from zoosim.world.users import User


@dataclass
class Query:
    intent_category: str
    query_type: str
    phi_cat: np.ndarray
    phi_emb: Tensor
    tokens: List[str]


def _torch_generator(rng: np.random.Generator) -> torch.Generator:
    seed = int(rng.integers(0, 2**31 - 1))
    return torch.Generator().manual_seed(seed)


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=float)
    vec[index] = 1.0
    return vec


def _sample_tokens(category: str, query_type: str, cfg_vocab: int, rng: np.random.Generator) -> List[str]:
    base_tokens = category.split("_") + [query_type]
    extra = [f"tok_{rng.integers(0, cfg_vocab)}" for _ in range(2)]
    return base_tokens + extra


def sample_query(*, user: User, config: SimulatorConfig, rng: np.random.Generator) -> Query:
    categories = config.catalog.categories
    cat_index = rng.choice(len(categories), p=user.theta_cat)
    intent_category = categories[cat_index]

    query_types = config.queries.query_types
    qtype = rng.choice(query_types, p=config.queries.query_type_mix)

    phi_cat = _one_hot(cat_index, len(categories))
    torch_gen = _torch_generator(rng)
    noise = torch.randn(config.catalog.embedding_dim, generator=torch_gen) * 0.05
    phi_emb = (user.theta_emb + noise).to(dtype=torch.float32)

    tokens = _sample_tokens(intent_category, qtype, config.queries.token_vocab_size, rng)

    return Query(
        intent_category=intent_category,
        query_type=qtype,
        phi_cat=phi_cat,
        phi_emb=phi_emb,
        tokens=tokens,
    )
