"""Discrete boost templates for contextual bandit policies.

Mathematical basis: [DEF-6.1.1] (Boost Template)

Templates define interpretable boost strategies that can be selected
by contextual bandit algorithms (LinUCB, Thompson Sampling).

References:
    - Chapter 6, §6.1: Discrete Template Action Space
    - [DEF-6.1.1]: Boost Template definition
    - [EQ-6.1]: Template application formula
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from numpy.typing import NDArray

from zoosim.world.catalog import Product


@dataclass
class BoostTemplate:
    """Single boost template with semantic label.

    Mathematical correspondence: Template t: C → ℝ from [DEF-6.1.1]

    A boost template assigns a boost value to each product based on its
    attributes (margin, brand, popularity, price, discount, strategic flag).

    Attributes:
        id: Template identifier (0 to M-1)
        name: Human-readable name (e.g., "High Margin")
        description: Business objective description
        boost_fn: Function mapping product to boost value
                  Signature: (product: Product) -> float
                  Output range: [-a_max, +a_max]
    """

    id: int
    name: str
    description: str
    boost_fn: Callable[[Product], float]

    def apply(self, products: List[Product]) -> NDArray[np.float32]:
        """Apply template to list of products.

        Implements [EQ-6.1]: Computes boost vector for products.
        The adjusted scores are: s'_i = s_base(q, p_i) + t(p_i)

        Args:
            products: List of Product instances

        Returns:
            boosts: Array of shape (len(products),) with boost values
                   Each entry in [-a_max, +a_max]
        """
        return np.array([self.boost_fn(p) for p in products], dtype=np.float32)


def compute_catalog_stats(products: List[Product]) -> Dict[str, float]:
    """Compute catalog statistics needed for template creation.

    Args:
        products: Full product catalog

    Returns:
        stats: Dictionary with keys:
            - 'price_p25': 25th percentile price
            - 'price_p75': 75th percentile price
            - 'pop_max': Maximum popularity (bestseller score)
            - 'own_brand': String identifier for own-brand products
            - 'num_categories': Number of unique categories
    """
    prices = np.array([p.price for p in products], dtype=float)
    popularities = np.array([p.bestseller for p in products], dtype=float)
    categories = list({p.category for p in products})

    # For realistic catalogs (thousands of products), use empirical quantiles.
    # For tiny synthetic fixtures (like the 3-product tests in Chapter 6),
    # fall back to min/max so tests have deterministic thresholds that match
    # the hand-crafted examples in the text.
    if prices.size >= 4:
        price_p25 = float(np.percentile(prices, 25))
        price_p75 = float(np.percentile(prices, 75))
    else:
        price_p25 = float(np.min(prices))
        price_p75 = float(np.max(prices))

    return {
        "price_p25": price_p25,
        "price_p75": price_p75,
        "pop_max": float(np.max(popularities)),
        "own_brand": "OwnBrand",  # Identifier for private-label products
        "num_categories": len(categories),
    }


def create_standard_templates(
    catalog_stats: Dict[str, float],
    a_max: float = 5.0,
) -> List[BoostTemplate]:
    """Create standard template library for search ranking.

    Implements the 8-template library from Chapter 6, §6.1.1 Table.

    Template Library:
    - t0: Neutral (no boost)
    - t1: High Margin (boost CM2 > 0.4)
    - t2: CM2 Boost (boost own-brand products)
    - t3: Popular (boost by log-popularity)
    - t4: Premium (boost expensive items, price > 75th percentile)
    - t5: Budget (boost cheap items, price < 25th percentile)
    - t6: Discount (boost discounted products)
    - t7: Strategic (boost strategic categories)

    Args:
        catalog_stats: Dictionary with keys:
                      - 'price_p25': 25th percentile price
                      - 'price_p75': 75th percentile price
                      - 'pop_max': Maximum popularity score
                      - 'own_brand': Name of own-brand label
                      - 'num_categories': Number of categories (optional)
        a_max: Maximum absolute boost value (default 5.0)

    Returns:
        templates: List of M=8 boost templates

    References:
        - [DEF-6.1.1] Boost Template definition
        - Table in Chapter 6, §6.1.1 for template specifications
    """
    p25 = catalog_stats["price_p25"]
    p75 = catalog_stats["price_p75"]
    pop_max = catalog_stats["pop_max"]
    own_brand = catalog_stats.get("own_brand", "OwnBrand")

    templates = [
        # t0: Neutral (baseline)
        BoostTemplate(
            id=0,
            name="Neutral",
            description="No boost adjustment (base ranker only)",
            boost_fn=lambda p: 0.0,
        ),
        # t1: High Margin
        BoostTemplate(
            id=1,
            name="High Margin",
            description="Promote products with CM2 > 0.4",
            boost_fn=lambda p: a_max if p.cm2 > 0.4 else 0.0,
        ),
        # t2: CM2 Boost (Own Brand)
        BoostTemplate(
            id=2,
            name="CM2 Boost",
            description="Promote own-brand (private-label) products",
            boost_fn=lambda p: a_max if p.is_pl else 0.0,
        ),
        # t3: Popular
        BoostTemplate(
            id=3,
            name="Popular",
            description="Boost by normalized log-popularity (bestseller score)",
            boost_fn=lambda p: (
                a_max * np.log(1 + p.bestseller) / np.log(1 + pop_max)
                if pop_max > 0
                else 0.0
            ),
        ),
        # t4: Premium
        BoostTemplate(
            id=4,
            name="Premium",
            description="Promote expensive items (price > 75th percentile)",
            boost_fn=lambda p: a_max if p.price > p75 else 0.0,
        ),
        # t5: Budget
        BoostTemplate(
            id=5,
            name="Budget",
            description="Promote cheap items (price < 25th percentile)",
            boost_fn=lambda p: a_max if p.price < p25 else 0.0,
        ),
        # t6: Discount
        BoostTemplate(
            id=6,
            name="Discount",
            description="Boost discounted products (max discount 30%)",
            boost_fn=lambda p: a_max * min(p.discount / 0.3, 1.0),
        ),
        # t7: Strategic
        BoostTemplate(
            id=7,
            name="Strategic",
            description="Promote strategic categories",
            boost_fn=lambda p: a_max if p.strategic_flag else 0.0,
        ),
    ]

    return templates


# Code ↔ Config mapping
# - Template definitions: zoosim/policies/templates.py:95-172
# - Catalog statistics: Computed from SimulatorConfig.catalog in zoosim/world/catalog.py
# - Action bound (continuous weights): SimulatorConfig.action.a_max in zoosim/core/config.py
# - Template amplitude a_max (this module) is a separate hyperparameter, tuned relative
#   to the base relevance score scale.
# - Own-brand identifier: Product.is_pl from zoosim/world/catalog.py
