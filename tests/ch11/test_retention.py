import numpy as np

from zoosim.core.config import load_default_config
from zoosim.multi_episode.retention import return_probability, sample_return


def test_return_probability_bounds() -> None:
    """Retention probability should always lie in [0, 1]."""
    cfg = load_default_config()

    for clicks in range(0, 20):
        for satisfaction in np.linspace(-1.0, 2.0, 13):
            p = return_probability(clicks=clicks, satisfaction=satisfaction, config=cfg)
            assert 0.0 <= p <= 1.0


def test_return_probability_monotone_in_clicks() -> None:
    """Retention probability should be non-decreasing in clicks."""
    cfg = load_default_config()

    for satisfaction in [0.0, 0.5, 1.0]:
        prev = 0.0
        for clicks in range(0, 16):
            p = return_probability(clicks=clicks, satisfaction=satisfaction, config=cfg)
            # Allow tiny numerical noise
            assert p >= prev - 1e-12
            prev = p


def test_return_probability_monotone_in_satisfaction() -> None:
    """Retention probability should be non-decreasing in satisfaction."""
    cfg = load_default_config()

    for clicks in [0, 3, 7]:
        prev = 0.0
        for satisfaction in np.linspace(0.0, 1.0, 11):
            p = return_probability(clicks=clicks, satisfaction=satisfaction, config=cfg)
            assert p >= prev - 1e-12
            prev = p


def test_sample_return_seed_deterministic() -> None:
    """Sampling should be deterministic for a fixed RNG seed."""
    cfg = load_default_config()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    seq1 = [
        sample_return(clicks=3, satisfaction=0.5, config=cfg, rng=rng1) for _ in range(20)
    ]
    seq2 = [
        sample_return(clicks=3, satisfaction=0.5, config=cfg, rng=rng2) for _ in range(20)
    ]

    assert seq1 == seq2

