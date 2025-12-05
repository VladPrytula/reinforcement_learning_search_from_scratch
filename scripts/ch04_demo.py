#!/usr/bin/env python
"""Chapter 4 demonstration: Generative World Model

This script demonstrates the key concepts from Chapter 4:
- Catalog generation with lognormal prices and linear margins
- User segment sampling with preference parameters
- Query generation coupled to user affinities
- Deterministic reproducibility

Run with: python scripts/ch04_demo.py
"""

import numpy as np
from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import generate_catalog
from zoosim.world.users import sample_user
from zoosim.world.queries import sample_query


def demo_catalog_generation():
    """Demonstrate catalog generation and statistics."""
    print("=" * 60)
    print("CATALOG GENERATION")
    print("=" * 60)

    cfg = SimulatorConfig(seed=2025_1108)
    rng = np.random.default_rng(cfg.seed)

    print(f"\nGenerating {cfg.catalog.n_products:,} products...")
    catalog = generate_catalog(cfg.catalog, rng)

    # Category distribution
    print("\nCategory distribution:")
    for cat in cfg.catalog.categories:
        count = sum(1 for p in catalog if p.category == cat)
        prob = count / len(catalog)
        print(f"  {cat:12s}: {count:5,} products ({prob:.3f})")

    # Price statistics by category
    print("\nPrice statistics by category:")
    for cat in cfg.catalog.categories:
        prices = [p.price for p in catalog if p.category == cat]
        median = np.median(prices)
        mu = cfg.catalog.price_params[cat]["mu"]
        theoretical_median = np.exp(mu)
        print(f"  {cat:12s}: median=${median:6.2f} (theory: ${theoretical_median:6.2f})")

    # Margin statistics
    print("\nMargin (CM2) statistics:")
    for cat in cfg.catalog.categories:
        margins = [p.cm2 for p in catalog if p.category == cat]
        mean_margin = np.mean(margins)
        slope = cfg.catalog.margin_slope[cat]
        print(f"  {cat:12s}: mean CM2=${mean_margin:6.3f} (slope β={slope:+.3f})")

    # Strategic category verification
    litter_products = [p for p in catalog if p.category == "litter"]
    strategic_count = sum(1 for p in litter_products if p.strategic_flag)
    print(f"\nStrategic category (litter): {strategic_count}/{len(litter_products)} flagged")


def demo_user_segments():
    """Demonstrate user segment sampling and preferences."""
    print("\n" + "=" * 60)
    print("USER SEGMENT SAMPLING")
    print("=" * 60)

    cfg = SimulatorConfig(seed=2025_1108)
    rng = np.random.default_rng(cfg.seed)

    print(f"\nSampling 10,000 users...")
    users = [sample_user(config=cfg, rng=rng) for _ in range(10_000)]

    # Segment distribution
    print("\nSegment distribution:")
    segment_counts = {}
    for u in users:
        segment_counts[u.segment] = segment_counts.get(u.segment, 0) + 1

    for seg, expected_prob in zip(cfg.users.segments, cfg.users.segment_mix):
        count = segment_counts[seg]
        prob = count / len(users)
        print(f"  {seg:15s}: {count:5,} ({prob:.3f}, expected {expected_prob:.3f})")

    # Preference statistics by segment
    print("\nPreference parameters by segment:")
    for seg in cfg.users.segments:
        seg_users = [u for u in users if u.segment == seg]
        mean_price = np.mean([u.theta_price for u in seg_users])
        mean_pl = np.mean([u.theta_pl for u in seg_users])
        params = cfg.users.segment_params[seg]
        print(f"  {seg:15s}: θ_price={mean_price:+.2f} (theory: {params.price_mean:+.2f}), "
              f"θ_pl={mean_pl:+.2f} (theory: {params.pl_mean:+.2f})")


def demo_query_intent_coupling():
    """Demonstrate query intent coupling to user preferences."""
    print("\n" + "=" * 60)
    print("QUERY INTENT COUPLING")
    print("=" * 60)

    cfg = SimulatorConfig(seed=2025_1108)
    rng = np.random.default_rng(cfg.seed)

    print(f"\nSampling 5,000 (user, query) pairs...")
    user_query_pairs = []
    for _ in range(5_000):
        user = sample_user(config=cfg, rng=rng)
        query = sample_query(user=user, config=cfg, rng=rng)
        user_query_pairs.append((user, query))

    # Query type distribution
    print("\nQuery type distribution:")
    qtype_counts = {}
    for _, q in user_query_pairs:
        qtype_counts[q.query_type] = qtype_counts.get(q.query_type, 0) + 1

    for qtype, expected_prob in zip(cfg.queries.query_types, cfg.queries.query_type_mix):
        count = qtype_counts[qtype]
        prob = count / len(user_query_pairs)
        print(f"  {qtype:10s}: {count:5,} ({prob:.3f}, expected {expected_prob:.3f})")

    # Intent coupling: litter query rate by segment
    print("\nLitter query rate by user segment:")
    for seg in cfg.users.segments:
        seg_pairs = [(u, q) for u, q in user_query_pairs if u.segment == seg]
        litter_queries = [(u, q) for u, q in seg_pairs if q.intent_category == "litter"]
        rate = len(litter_queries) / len(seg_pairs) if seg_pairs else 0

        # Compute expected rate from Dirichlet concentration
        params = cfg.users.segment_params[seg]
        litter_idx = cfg.catalog.categories.index("litter")
        expected_rate = params.cat_conc[litter_idx] / sum(params.cat_conc)

        print(f"  {seg:15s}: {rate:.3f} (expected {expected_rate:.3f})")


def demo_determinism():
    """Demonstrate deterministic generation with same seed."""
    print("\n" + "=" * 60)
    print("DETERMINISM VERIFICATION")
    print("=" * 60)

    cfg = SimulatorConfig(seed=42)

    print("\nGenerating two catalogs with seed 42...")
    catalog1 = generate_catalog(cfg.catalog, np.random.default_rng(42))
    catalog2 = generate_catalog(cfg.catalog, np.random.default_rng(42))

    # Check identical products
    print("\nVerifying products are identical:")
    for idx in [0, 10, 100, 999]:
        p1 = catalog1[idx]
        p2 = catalog2[idx]

        price_match = p1.price == p2.price
        cm2_match = p1.cm2 == p2.cm2
        cat_match = p1.category == p2.category

        print(f"  Product {idx:4d}: price={price_match}, cm2={cm2_match}, category={cat_match}")

    print("\n✓ Same seed → identical catalog (reproducible)")

    # Different seeds produce different catalogs
    print("\nGenerating two catalogs with different seeds (42 vs 123)...")
    catalog_a = generate_catalog(cfg.catalog, np.random.default_rng(42))
    catalog_b = generate_catalog(cfg.catalog, np.random.default_rng(123))

    p_a = catalog_a[0]
    p_b = catalog_b[0]

    print(f"  Catalog A [seed=42]:  Product 0 price=${p_a.price:.2f}, category={p_a.category}")
    print(f"  Catalog B [seed=123]: Product 0 price=${p_b.price:.2f}, category={p_b.category}")
    print(f"  Different: {p_a.price != p_b.price or p_a.category != p_b.category}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("CHAPTER 4 — GENERATIVE WORLD MODEL DEMONSTRATION")
    print("=" * 60)

    demo_catalog_generation()
    demo_user_segments()
    demo_query_intent_coupling()
    demo_determinism()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  ✓ Catalog: Lognormal prices, linear margins, strategic categories")
    print("  ✓ Users: Segment-specific preferences (price, PL, category affinity)")
    print("  ✓ Queries: Intent coupled to user affinities")
    print("  ✓ Determinism: Same seed → identical world (reproducible RL)")
    print("\nSee Chapter 4 for mathematical foundations and theory-practice analysis.")
    print()


if __name__ == "__main__":
    main()
