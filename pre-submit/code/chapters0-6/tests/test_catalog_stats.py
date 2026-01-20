import numpy as np
import torch

from zoosim.core import config
from zoosim.world import catalog
from zoosim.world.queries import sample_query
from zoosim.world.users import sample_user


def test_catalog_price_and_margin_stats():
    cfg = config.load_default_config()
    rng = np.random.default_rng(42)
    products = catalog.generate_catalog(cfg.catalog, rng)

    assert all(p.price > 0 for p in products)

    litter_cm2 = [p.cm2 for p in products if p.category == "litter"]
    assert litter_cm2, "Need litter products"
    assert np.mean(litter_cm2) < 0, "Litter margins should be negative on average"

    toys_cm2 = [p.cm2 for p in products if p.category == "toys"]
    assert np.mean(toys_cm2) > 0


def test_deterministic_catalog():
    """Verify that same seed produces identical catalog."""
    cfg = config.load_default_config()
    prods1 = catalog.generate_catalog(cfg.catalog, np.random.default_rng(42))
    prods2 = catalog.generate_catalog(cfg.catalog, np.random.default_rng(42))

    for idx in (0, 10, 100, 999):
        assert prods1[idx].price == prods2[idx].price
        assert prods1[idx].cm2 == prods2[idx].cm2
        assert prods1[idx].category == prods2[idx].category
        assert prods1[idx].is_pl == prods2[idx].is_pl
        assert torch.equal(prods1[idx].embedding, prods2[idx].embedding)


def test_deterministic_user_sampling():
    """Verify deterministic user sampling sequences for a fixed seed."""
    cfg = config.load_default_config()
    rng1 = np.random.default_rng(2025)
    rng2 = np.random.default_rng(2025)

    for _ in range(100):
        u1 = sample_user(config=cfg, rng=rng1)
        u2 = sample_user(config=cfg, rng=rng2)
        assert u1.segment == u2.segment
        assert u1.theta_price == u2.theta_price
        assert u1.theta_pl == u2.theta_pl
        assert np.array_equal(u1.theta_cat, u2.theta_cat)
        assert torch.equal(u1.theta_emb, u2.theta_emb)


def test_deterministic_user_query_sampling():
    """Verify deterministic (user, query) sampling sequences for a fixed seed."""
    cfg = config.load_default_config()
    rng1 = np.random.default_rng(2025_1108)
    rng2 = np.random.default_rng(2025_1108)

    for _ in range(100):
        u1 = sample_user(config=cfg, rng=rng1)
        u2 = sample_user(config=cfg, rng=rng2)
        q1 = sample_query(user=u1, config=cfg, rng=rng1)
        q2 = sample_query(user=u2, config=cfg, rng=rng2)
        assert q1.intent_category == q2.intent_category
        assert q1.query_type == q2.query_type
        assert np.array_equal(q1.phi_cat, q2.phi_cat)
        assert torch.equal(q1.phi_emb, q2.phi_emb)
        assert q1.tokens == q2.tokens
