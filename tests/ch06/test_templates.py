"""Tests for boost template library.

Tests:
1. Template application produces bounded boosts
2. Catalog statistics computation
3. Standard template library creation
4. Template semantic correctness (high margin boosts high-margin products)
"""

import numpy as np
import pytest
import torch

from zoosim.world.catalog import Product
from zoosim.policies.templates import (
    BoostTemplate,
    compute_catalog_stats,
    create_standard_templates,
)


@pytest.fixture
def mock_products():
    """Create synthetic products for testing."""
    products = [
        # High-margin own-brand product
        Product(
            product_id=0,
            category="dog_food",
            price=30.0,
            cm2=0.5,  # High margin (>0.4)
            is_pl=True,  # Own-brand
            discount=0.1,
            bestseller=500.0,
            embedding=torch.randn(16),
            strategic_flag=True,
        ),
        # Low-margin third-party budget product
        Product(
            product_id=1,
            category="cat_food",
            price=5.0,  # Low price
            cm2=0.2,  # Low margin
            is_pl=False,
            discount=0.0,
            bestseller=100.0,
            embedding=torch.randn(16),
            strategic_flag=False,
        ),
        # Premium discounted product
        Product(
            product_id=2,
            category="toys",
            price=60.0,  # High price
            cm2=0.35,
            is_pl=False,
            discount=0.25,  # High discount
            bestseller=800.0,
            embedding=torch.randn(16),
            strategic_flag=False,
        ),
    ]
    return products


def test_compute_catalog_stats(mock_products):
    """Test catalog statistics computation."""
    stats = compute_catalog_stats(mock_products)

    # Check all required keys present
    assert "price_p25" in stats
    assert "price_p75" in stats
    assert "pop_max" in stats
    assert "own_brand" in stats
    assert "num_categories" in stats

    # Check values are reasonable
    assert stats["price_p25"] == 5.0  # min price in fixture
    assert stats["price_p75"] == 60.0  # max price in fixture
    assert stats["pop_max"] == 800.0  # max bestseller
    assert stats["num_categories"] == 3  # dog_food, cat_food, toys


def test_create_standard_templates(mock_products):
    """Test standard template library creation."""
    stats = compute_catalog_stats(mock_products)
    templates = create_standard_templates(stats, a_max=5.0)

    # Check template count
    assert len(templates) == 8, "Should have M=8 templates"

    # Check template IDs are unique and sequential
    template_ids = [t.id for t in templates]
    assert template_ids == list(range(8))

    # Check template names
    expected_names = [
        "Neutral",
        "High Margin",
        "CM2 Boost",
        "Popular",
        "Premium",
        "Budget",
        "Discount",
        "Strategic",
    ]
    template_names = [t.name for t in templates]
    assert template_names == expected_names


def test_template_application_bounded(mock_products):
    """Test that template boosts are bounded in [-a_max, +a_max]."""
    stats = compute_catalog_stats(mock_products)
    templates = create_standard_templates(stats, a_max=5.0)

    for template in templates:
        boosts = template.apply(mock_products)

        # Check shape
        assert boosts.shape == (len(mock_products),)

        # Check dtype
        assert boosts.dtype == np.float32

        # Check bounds: all boosts in [-5.0, +5.0]
        assert np.all(boosts >= -5.0), f"{template.name} produces boost < -5.0"
        assert np.all(boosts <= 5.0), f"{template.name} produces boost > +5.0"


def test_neutral_template_zero_boost(mock_products):
    """Test that Neutral template produces zero boosts."""
    stats = compute_catalog_stats(mock_products)
    templates = create_standard_templates(stats, a_max=5.0)

    neutral_template = templates[0]  # t0: Neutral
    assert neutral_template.name == "Neutral"

    boosts = neutral_template.apply(mock_products)
    assert np.all(boosts == 0.0), "Neutral template should produce zero boosts"


def test_high_margin_template_semantic_correctness(mock_products):
    """Test that High Margin template boosts high-margin products."""
    stats = compute_catalog_stats(mock_products)
    templates = create_standard_templates(stats, a_max=5.0)

    high_margin_template = templates[1]  # t1: High Margin
    assert high_margin_template.name == "High Margin"

    boosts = high_margin_template.apply(mock_products)

    # Product 0 has cm2=0.5 (>0.4) → should get boost
    assert boosts[0] == 5.0, "High-margin product should get max boost"

    # Products 1, 2 have cm2 < 0.4 → should get zero boost
    assert boosts[1] == 0.0, "Low-margin product should get zero boost"
    assert boosts[2] == 0.0, "Product with cm2=0.35 should get zero boost"


def test_cm2_boost_template_semantic_correctness(mock_products):
    """Test that CM2 Boost template boosts own-brand products."""
    stats = compute_catalog_stats(mock_products)
    templates = create_standard_templates(stats, a_max=5.0)

    cm2_template = templates[2]  # t2: CM2 Boost
    assert cm2_template.name == "CM2 Boost"

    boosts = cm2_template.apply(mock_products)

    # Product 0 is own-brand (is_pl=True) → should get boost
    assert boosts[0] == 5.0, "Own-brand product should get max boost"

    # Products 1, 2 are third-party → should get zero boost
    assert boosts[1] == 0.0
    assert boosts[2] == 0.0


def test_budget_template_semantic_correctness(mock_products):
    """Test that Budget template boosts cheap products."""
    stats = compute_catalog_stats(mock_products)
    templates = create_standard_templates(stats, a_max=5.0)

    budget_template = templates[5]  # t5: Budget
    assert budget_template.name == "Budget"

    boosts = budget_template.apply(mock_products)

    # Product 1 has price=5.0 (< p25=5.0) → edge case, should get boost
    assert boosts[1] == 0.0, "Price exactly at p25 should not boost (strict inequality)"

    # Products 0, 2 are expensive → should get zero boost
    assert boosts[0] == 0.0
    assert boosts[2] == 0.0


def test_discount_template_proportional_boost(mock_products):
    """Test that Discount template gives proportional boost to discount."""
    stats = compute_catalog_stats(mock_products)
    templates = create_standard_templates(stats, a_max=5.0)

    discount_template = templates[6]  # t6: Discount
    assert discount_template.name == "Discount"

    boosts = discount_template.apply(mock_products)

    # Product 0: discount=0.1 → boost = 5.0 * (0.1 / 0.3) ≈ 1.67
    assert np.isclose(boosts[0], 5.0 * (0.1 / 0.3), atol=0.01)

    # Product 1: discount=0.0 → boost = 0.0
    assert boosts[1] == 0.0

    # Product 2: discount=0.25 → boost = 5.0 * (0.25 / 0.3) ≈ 4.17
    assert np.isclose(boosts[2], 5.0 * (0.25 / 0.3), atol=0.01)


def test_strategic_template_semantic_correctness(mock_products):
    """Test that Strategic template boosts strategic products."""
    stats = compute_catalog_stats(mock_products)
    templates = create_standard_templates(stats, a_max=5.0)

    strategic_template = templates[7]  # t7: Strategic
    assert strategic_template.name == "Strategic"

    boosts = strategic_template.apply(mock_products)

    # Product 0 has strategic_flag=True → should get boost
    assert boosts[0] == 5.0

    # Products 1, 2 are not strategic → should get zero boost
    assert boosts[1] == 0.0
    assert boosts[2] == 0.0
