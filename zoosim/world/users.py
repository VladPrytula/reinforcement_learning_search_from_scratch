"""User and segment sampling implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from zoosim.core.config import SimulatorConfig


@dataclass
class User:
    segment: str
    theta_price: float
    theta_pl: float
    theta_cat: np.ndarray
    theta_emb: Tensor


def _torch_generator(rng: np.random.Generator) -> torch.Generator:
    seed = int(rng.integers(0, 2**31 - 1))
    return torch.Generator().manual_seed(seed)


def sample_user(*, config: SimulatorConfig, rng: np.random.Generator) -> User:
    """Sample a user with segment-conditioned preference parameters.

    Assumes `config.users.segment_mix` is a valid probability vector over
    `config.users.segments` (validated by `zoosim.core.config.UserConfig`).
    """

    segments = config.users.segments
    probs = config.users.segment_mix
    segment = rng.choice(segments, p=probs)
    params = config.users.segment_params[segment]

    theta_price = float(rng.normal(params.price_mean, params.price_std))
    theta_pl = float(rng.normal(params.pl_mean, params.pl_std))
    theta_cat = rng.dirichlet(params.cat_conc)

    torch_gen = _torch_generator(rng)
    theta_emb = torch.randn(config.catalog.embedding_dim, generator=torch_gen)

    return User(
        segment=segment,
        theta_price=theta_price,
        theta_pl=theta_pl,
        theta_cat=theta_cat,
        theta_emb=theta_emb,
    )
