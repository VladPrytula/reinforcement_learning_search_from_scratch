import numpy as np
import torch

from zoosim.core import config
from zoosim.world import catalog


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

    for idx in (0, 10, 100):
        assert prods1[idx].price == prods2[idx].price
        assert prods1[idx].cm2 == prods2[idx].cm2
        assert prods1[idx].category == prods2[idx].category
        assert prods1[idx].is_pl == prods2[idx].is_pl
        assert torch.equal(prods1[idx].embedding, prods2[idx].embedding)
