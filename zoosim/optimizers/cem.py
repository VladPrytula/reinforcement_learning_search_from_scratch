"""Cross-Entropy Method (CEM) optimizer for continuous actions.

Implements a simple CEM variant suitable for optimizing bounded boost
vectors in the Zoosim search simulator.

Given a black-box objective f(a) (e.g., Q(x,a) or Q(x,a)+βσ(x,a)), CEM
maintains a Gaussian search distribution over actions and iteratively
updates it based on the top-performing (elite) samples.

This is used in Chapter 7 to approximate:

    a* ≈ argmax_a Q(x, a)

under box constraints |a_i| ≤ a_max and optional trust-region constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


ObjectiveFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class CEMConfig:
    """Configuration for the CEM optimizer.

    Attributes:
        n_samples: Number of candidate actions per iteration.
        elite_frac: Fraction of top samples used as elites.
        n_iters: Number of CEM iterations.
        init_std: Initial standard deviation for each action dimension.
        min_std: Minimum standard deviation to avoid premature collapse.
        alpha: Smoothing factor for mean/std updates (0→use elites only, 1→no update).
        seed: Random seed for reproducibility.
        a_max: Symmetric action bound; actions are clipped to [-a_max, +a_max].
    """

    n_samples: int = 64
    elite_frac: float = 0.2
    n_iters: int = 5
    init_std: float = 0.5
    min_std: float = 0.05
    alpha: float = 0.7
    seed: int = 42
    a_max: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 < self.elite_frac <= 1.0):
            raise ValueError("elite_frac must be in (0, 1].")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if self.n_iters <= 0:
            raise ValueError("n_iters must be positive.")
        if self.init_std <= 0.0:
            raise ValueError("init_std must be positive.")
        if self.min_std <= 0.0:
            raise ValueError("min_std must be positive.")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if self.a_max <= 0.0:
            raise ValueError("a_max must be positive.")


def _project_to_trust_region(
    actions: NDArray[np.float64],
    center: NDArray[np.float64],
    radius: float,
) -> NDArray[np.float64]:
    """Project actions into an ℓ2 ball around center."""
    diffs = actions - center[None, :]
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    # Avoid division by zero; where norm <= radius, projection is identity.
    mask = norms > radius
    safe_norms = np.where(mask, norms, 1.0)
    scaled = center[None, :] + diffs * (radius / safe_norms)
    return np.where(mask, scaled, actions)


def cem_optimize(
    objective: ObjectiveFn,
    action_dim: int,
    config: CEMConfig,
    init_mean: Optional[NDArray[np.float64]] = None,
    trust_region_center: Optional[NDArray[np.float64]] = None,
    trust_region_radius: Optional[float] = None,
) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
    """Optimize objective over continuous actions using CEM.

    Args:
        objective: Function mapping actions (N, d) → values (N,).
        action_dim: Dimensionality d of action vectors.
        config: CEMConfig with optimization hyperparameters.
        init_mean: Optional initial mean for the search distribution.
        trust_region_center: Optional center for ℓ2 trust region.
        trust_region_radius: Optional radius for ℓ2 trust region.

    Returns:
        best_action: Best action found (argmax objective) of shape (d,).
        history: Dictionary with optimization traces:
            - 'mean_history': Array (n_iters, d) of means per iteration.
            - 'std_history': Array (n_iters, d) of stds per iteration.
            - 'best_values': Array (n_iters,) of best value per iteration.
    """
    if action_dim <= 0:
        raise ValueError("action_dim must be positive.")

    rng = np.random.default_rng(config.seed)

    if init_mean is not None:
        mean = np.asarray(init_mean, dtype=np.float64).reshape(action_dim)
    else:
        mean = np.zeros(action_dim, dtype=np.float64)

    std = np.full(action_dim, config.init_std, dtype=np.float64)

    if trust_region_center is not None and trust_region_radius is None:
        raise ValueError("trust_region_radius must be provided with trust_region_center.")
    if trust_region_radius is not None and trust_region_center is None:
        raise ValueError("trust_region_center must be provided with trust_region_radius.")

    if trust_region_center is not None:
        trust_center = np.asarray(trust_region_center, dtype=np.float64).reshape(
            action_dim
        )
        trust_radius = float(trust_region_radius)
    else:
        trust_center = None
        trust_radius = 0.0

    n_elite = max(1, int(config.elite_frac * config.n_samples))

    mean_history: List[NDArray[np.float64]] = []
    std_history: List[NDArray[np.float64]] = []
    best_values_list: List[float] = []

    best_action = mean.copy()
    best_value = -np.inf

    for _ in range(config.n_iters):
        samples = rng.normal(loc=mean, scale=std, size=(config.n_samples, action_dim))
        samples = np.clip(samples, -config.a_max, config.a_max)

        if trust_center is not None:
            samples = _project_to_trust_region(samples, trust_center, trust_radius)

        values = np.asarray(objective(samples), dtype=np.float64).reshape(config.n_samples)

        elite_idx = np.argsort(values)[-n_elite:]
        elite_samples = samples[elite_idx]
        elite_values = values[elite_idx]

        elite_mean = elite_samples.mean(axis=0)
        elite_std = elite_samples.std(axis=0)

        mean = config.alpha * mean + (1.0 - config.alpha) * elite_mean
        std = np.maximum(
            config.min_std, config.alpha * std + (1.0 - config.alpha) * elite_std
        )

        iter_best_idx = int(np.argmax(values))
        iter_best_value = float(values[iter_best_idx])
        iter_best_action = samples[iter_best_idx]

        if iter_best_value > best_value:
            best_value = iter_best_value
            best_action = iter_best_action

        mean_history.append(mean.copy())
        std_history.append(std.copy())
        best_values_list.append(iter_best_value)

    history = {
        "mean_history": np.stack(mean_history, axis=0),
        "std_history": np.stack(std_history, axis=0),
        "best_values": np.asarray(best_values_list, dtype=np.float64),
    }
    return best_action, history

