"""Chapter 2 tests: segment mix sampling.

This module verifies that the empirical user segment frequencies produced by
`zoosim.world.users.sample_user` match the theoretical probability vector
`cfg.users.segment_mix` from `SimulatorConfig`.

Mathematical correspondence:
- [DEF-2.2.2] (Probability measure on discrete space)
- Strong Law of Large Numbers (SLLN): empirical frequencies converge to ρ.
"""

from __future__ import annotations

import numpy as np

from zoosim.core.config import SimulatorConfig
from zoosim.world.users import sample_user


def test_segment_mix_matches_config() -> None:
    """Empirical segment mix should be close to config.users.segment_mix.

    We sample many users from `sample_user` and check that the maximum
    deviation between empirical frequencies and the configured probabilities
    is small. This is the executable counterpart of Chapter 2 Lab 2.1
    ("Segment Mix Sanity Check") and validates the discrete measure ρ.
    """
    cfg = SimulatorConfig()
    rng = np.random.default_rng(seed=21)

    segments = cfg.users.segments
    theoretical = np.asarray(cfg.users.segment_mix, dtype=float)
    counts = {seg: 0 for seg in segments}

    n_samples = 20_000
    for _ in range(n_samples):
        user = sample_user(config=cfg, rng=rng)
        counts[user.segment] += 1

    empirical = np.array([counts[seg] / n_samples for seg in segments], dtype=float)
    l_inf = float(np.max(np.abs(empirical - theoretical)))

    # With n=20_000 and max ρ≈0.35, CLT suggests typical deviations ≈ O(0.01).
    # We allow a small safety margin to keep the test robust but meaningful.
    assert l_inf < 0.03, f"L_inf deviation too large: {l_inf:.3f}"


if __name__ == "__main__":  # pragma: no cover
    import pytest
    import sys

    sys.exit(pytest.main(["-q", __file__]))

