"""Tests for Chapter 14 fairness and CMDP modules.

Tests cover:
1. Fairness metrics (exposure_by_group, l1_gap, within_band)
2. Constraint specification (ConstraintSpec, slack computation)
3. CMDP agent (shaped rewards, dual updates)
4. Integration with ZooplusSearchEnv

References:
    - Chapter 14: Multi-Objective RL and Fairness at Scale
    - [EQ-14.5]: CMDP formulation
    - [ALG-14.5.1]: Primal-Dual CMDP Training
"""

import numpy as np
import pytest

from zoosim.core.config import SimulatorConfig, load_default_config
from zoosim.envs import ZooplusSearchEnv
from zoosim.evaluation.fairness import (
    BandCheckResult,
    GroupScheme,
    exposure_by_group,
    exposure_share_by_group,
    get_group_keys,
    group_key,
    kl_divergence,
    l1_gap,
    position_weights,
    within_band,
)
from zoosim.policies import (
    CMDPConfig,
    ConstraintSense,
    ConstraintSpec,
    PrimalDualCMDPAgent,
    create_standard_constraints,
)
from zoosim.world.catalog import generate_catalog


# ---------------------------------------------------------------------------
# Fairness metrics tests
# ---------------------------------------------------------------------------


class TestPositionWeights:
    """Tests for position_weights function."""

    def test_position_weights_shape(self):
        """Weights have correct shape."""
        cfg = load_default_config()
        weights = position_weights(cfg, "category", k=10)
        assert weights.shape == (10,)

    def test_position_weights_monotonic(self):
        """Position weights decrease with rank (position bias)."""
        cfg = load_default_config()
        weights = position_weights(cfg, "category", k=5)
        # First position should have highest weight
        assert weights[0] >= weights[1]

    def test_position_weights_fallback(self):
        """Unknown query type falls back to generic."""
        cfg = load_default_config()
        weights = position_weights(cfg, "unknown_type", k=5)
        assert weights.shape == (5,)


class TestGroupKey:
    """Tests for group_key function."""

    def test_pl_scheme(self):
        """PL scheme returns correct keys."""
        cfg = load_default_config()
        rng = np.random.default_rng(42)
        catalog = generate_catalog(cfg.catalog, rng)

        # Find a PL product and a non-PL product
        pl_products = [p for p in catalog if p.is_pl]
        non_pl_products = [p for p in catalog if not p.is_pl]

        if pl_products:
            assert group_key(pl_products[0], "pl") == "pl"
        if non_pl_products:
            assert group_key(non_pl_products[0], "pl") == "non_pl"

    def test_category_scheme(self):
        """Category scheme returns product category."""
        cfg = load_default_config()
        rng = np.random.default_rng(42)
        catalog = generate_catalog(cfg.catalog, rng)

        for prod in catalog[:5]:
            key = group_key(prod, "category")
            assert key == prod.category
            assert key in cfg.catalog.categories

    def test_strategic_scheme(self):
        """Strategic scheme returns correct keys."""
        cfg = load_default_config()
        rng = np.random.default_rng(42)
        catalog = generate_catalog(cfg.catalog, rng)

        for prod in catalog[:5]:
            key = group_key(prod, "strategic")
            expected = "strategic" if prod.strategic_flag else "non_strategic"
            assert key == expected


class TestExposureMetrics:
    """Tests for exposure computation functions."""

    def test_exposure_shares_sum_to_one(self):
        """Exposure shares sum to 1."""
        cfg = load_default_config()
        rng = np.random.default_rng(42)
        catalog = generate_catalog(cfg.catalog, rng)
        catalog_dict = {p.product_id: p for p in catalog}

        # Create a simple ranking
        ranking = [p.product_id for p in catalog[:20]]
        weights = position_weights(cfg, "category", k=20)

        shares = exposure_share_by_group(ranking, catalog_dict, weights, "pl")

        total = sum(shares.values())
        assert abs(total - 1.0) < 1e-6, f"Shares sum to {total}, expected 1.0"

    def test_exposure_by_group_positive(self):
        """Exposure values are non-negative."""
        cfg = load_default_config()
        rng = np.random.default_rng(42)
        catalog = generate_catalog(cfg.catalog, rng)
        catalog_dict = {p.product_id: p for p in catalog}

        ranking = [p.product_id for p in catalog[:10]]
        weights = position_weights(cfg, "generic", k=10)

        exposure = exposure_by_group(ranking, catalog_dict, weights, "category")

        for group, exp in exposure.items():
            assert exp >= 0, f"Exposure for {group} is negative: {exp}"


class TestGapMetrics:
    """Tests for L1 gap and KL divergence."""

    def test_l1_gap_identical(self):
        """L1 gap is zero for identical distributions."""
        shares = {"a": 0.3, "b": 0.7}
        targets = {"a": 0.3, "b": 0.7}
        assert l1_gap(shares, targets) == 0.0

    def test_l1_gap_symmetric(self):
        """L1 gap is symmetric."""
        shares = {"a": 0.4, "b": 0.6}
        targets = {"a": 0.3, "b": 0.7}
        gap1 = l1_gap(shares, targets)
        gap2 = l1_gap(targets, shares)
        assert abs(gap1 - gap2) < 1e-10

    def test_l1_gap_range(self):
        """L1 gap is in [0, 2]."""
        shares = {"a": 1.0, "b": 0.0}
        targets = {"a": 0.0, "b": 1.0}
        gap = l1_gap(shares, targets)
        assert 0 <= gap <= 2.0

    def test_kl_divergence_identical(self):
        """KL divergence is zero for identical distributions."""
        shares = {"a": 0.5, "b": 0.5}
        targets = {"a": 0.5, "b": 0.5}
        kl = kl_divergence(shares, targets)
        assert kl < 1e-8


class TestWithinBand:
    """Tests for band checking function."""

    def test_within_band_satisfied(self):
        """Band check returns True when shares within band."""
        shares = {"a": 0.42, "b": 0.58}
        targets = {"a": 0.40, "b": 0.60}
        band = 0.05

        result = within_band(shares, targets, band)

        assert result.all_satisfied is True
        assert result.within_band["a"] is True
        assert result.within_band["b"] is True
        assert result.max_deviation < 0.05

    def test_within_band_violated(self):
        """Band check returns False when shares outside band."""
        shares = {"a": 0.30, "b": 0.70}
        targets = {"a": 0.50, "b": 0.50}
        band = 0.05

        result = within_band(shares, targets, band)

        assert result.all_satisfied is False


# ---------------------------------------------------------------------------
# Constraint specification tests
# ---------------------------------------------------------------------------


class TestConstraintSpec:
    """Tests for ConstraintSpec class."""

    def test_geq_slack_positive_when_satisfied(self):
        """GEQ constraint has positive slack when satisfied."""
        constraint = ConstraintSpec(
            name="cm2_floor",
            threshold=0.1,
            sense=ConstraintSense.GEQ,
            metric_key="cm2",
        )

        # Metric above threshold -> positive slack
        slack = constraint.compute_slack(0.15)
        assert abs(slack - 0.05) < 1e-9

    def test_geq_slack_negative_when_violated(self):
        """GEQ constraint has negative slack when violated."""
        constraint = ConstraintSpec(
            name="cm2_floor",
            threshold=0.1,
            sense=ConstraintSense.GEQ,
            metric_key="cm2",
        )

        # Metric below threshold -> negative slack
        slack = constraint.compute_slack(0.05)
        assert slack == -0.05

    def test_leq_slack_positive_when_satisfied(self):
        """LEQ constraint has positive slack when satisfied."""
        constraint = ConstraintSpec(
            name="stability",
            threshold=0.2,
            sense=ConstraintSense.LEQ,
            metric_key="delta_rank",
        )

        # Metric below threshold -> positive slack
        slack = constraint.compute_slack(0.1)
        assert slack == 0.1

    def test_leq_slack_negative_when_violated(self):
        """LEQ constraint has negative slack when violated."""
        constraint = ConstraintSpec(
            name="stability",
            threshold=0.2,
            sense=ConstraintSense.LEQ,
            metric_key="delta_rank",
        )

        # Metric above threshold -> negative slack
        slack = constraint.compute_slack(0.3)
        assert abs(slack - (-0.1)) < 1e-9

    def test_is_satisfied(self):
        """is_satisfied correctly checks constraint."""
        constraint = ConstraintSpec(
            name="test",
            threshold=0.5,
            sense=ConstraintSense.GEQ,
            metric_key="test",
        )

        assert constraint.is_satisfied(0.6) is True
        assert constraint.is_satisfied(0.5) is True
        assert constraint.is_satisfied(0.4) is False
        # With tolerance
        assert constraint.is_satisfied(0.45, tolerance=0.1) is True


class TestCreateStandardConstraints:
    """Tests for constraint factory function."""

    def test_cm2_floor_only(self):
        """Creates single CM2 floor constraint."""
        constraints = create_standard_constraints(cm2_floor=0.1)
        assert len(constraints) == 1
        assert constraints[0].name == "cm2_floor"
        assert constraints[0].threshold == 0.1
        assert constraints[0].sense == ConstraintSense.GEQ

    def test_stability_only(self):
        """Creates single stability constraint."""
        constraints = create_standard_constraints(stability_ceiling=0.2)
        assert len(constraints) == 1
        assert constraints[0].name == "stability"
        assert constraints[0].sense == ConstraintSense.LEQ

    def test_exposure_band(self):
        """Creates two constraints for exposure band."""
        constraints = create_standard_constraints(exposure_band=(0.3, 0.5))
        assert len(constraints) == 2
        # One GEQ (lower bound) and one LEQ (upper bound)
        senses = {c.sense for c in constraints}
        assert senses == {ConstraintSense.GEQ, ConstraintSense.LEQ}


# ---------------------------------------------------------------------------
# CMDP Agent tests
# ---------------------------------------------------------------------------


class TestPrimalDualCMDPAgent:
    """Tests for PrimalDualCMDPAgent class."""

    @pytest.fixture
    def simple_agent(self):
        """Create a simple CMDP agent for testing."""
        constraints = [
            ConstraintSpec(
                name="test_constraint",
                threshold=0.5,
                sense=ConstraintSense.GEQ,
                metric_key="test_metric",
            )
        ]
        config = CMDPConfig(seed=42, hidden_sizes=(32,))
        return PrimalDualCMDPAgent(
            obs_dim=5,
            action_dim=3,
            constraints=constraints,
            config=config,
        )

    def test_agent_initialization(self, simple_agent):
        """Agent initializes with correct structure."""
        assert len(simple_agent.constraints) == 1
        assert "test_constraint" in simple_agent.lambdas
        assert simple_agent.lambdas["test_constraint"] == 0.0  # lambda_init

    def test_select_action_shape(self, simple_agent):
        """Action has correct shape."""
        obs = np.random.randn(5).astype(np.float32)
        action = simple_agent.select_action(obs)
        assert action.shape == (3,)

    def test_shaped_reward_with_satisfied_constraint(self, simple_agent):
        """Shaped reward increases when constraint satisfied."""
        obs = np.random.randn(5).astype(np.float32)
        simple_agent.select_action(obs)

        raw_reward = 1.0
        info = {}
        constraint_metrics = {"test_metric": 0.7}  # Above threshold 0.5

        # With lambda=0, shaped reward equals raw reward
        shaped = simple_agent.store_transition(raw_reward, info, constraint_metrics)
        assert shaped == raw_reward  # lambda is 0 initially

    def test_dual_update_increases_lambda_on_violation(self, simple_agent):
        """Dual update increases lambda when constraint violated."""
        obs = np.random.randn(5).astype(np.float32)
        simple_agent.select_action(obs)

        # Violate constraint - use constraint NAME as key
        simple_agent.store_transition(
            reward=1.0,
            info={},
            constraint_metrics={"test_constraint": 0.3},  # Below threshold 0.5
        )

        initial_lambda = simple_agent.lambdas["test_constraint"]
        simple_agent.update()

        # Lambda should increase (violation -> negative slack -> subtract negative)
        assert simple_agent.lambdas["test_constraint"] > initial_lambda

    def test_dual_update_decreases_lambda_on_satisfaction(self, simple_agent):
        """Dual update decreases lambda when constraint satisfied with slack."""
        # First, set lambda to positive value
        simple_agent.lambdas["test_constraint"] = 1.0

        obs = np.random.randn(5).astype(np.float32)
        simple_agent.select_action(obs)

        # Satisfy constraint with large slack - use constraint NAME as key
        simple_agent.store_transition(
            reward=1.0,
            info={},
            constraint_metrics={"test_constraint": 1.0},  # Well above threshold 0.5
        )

        simple_agent.update()

        # Lambda should decrease (positive slack -> subtract positive)
        assert simple_agent.lambdas["test_constraint"] < 1.0

    def test_lambda_projection_to_nonnegative(self, simple_agent):
        """Lambda stays non-negative after update."""
        obs = np.random.randn(5).astype(np.float32)
        simple_agent.select_action(obs)

        # Satisfy constraint with very large slack - use constraint NAME
        simple_agent.store_transition(
            reward=1.0,
            info={},
            constraint_metrics={"test_constraint": 10.0},  # Way above threshold
        )

        simple_agent.update()

        # Lambda should be clipped to 0
        assert simple_agent.lambdas["test_constraint"] >= 0.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestEnvIntegration:
    """Integration tests with ZooplusSearchEnv."""

    def test_env_provides_stability_metric(self):
        """Env info contains delta_rank_at_k_vs_baseline."""
        cfg = load_default_config()
        env = ZooplusSearchEnv(cfg, seed=42)

        env.reset()
        action = np.zeros(cfg.action.feature_dim)
        _, _, _, info = env.step(action)

        assert "delta_rank_at_k_vs_baseline" in info
        assert "baseline_ranking" in info
        assert "user_segment" in info
        assert "query_type" in info

    def test_delta_rank_zero_for_zero_action(self):
        """Zero action produces zero delta_rank (no change from baseline)."""
        cfg = load_default_config()
        env = ZooplusSearchEnv(cfg, seed=42)

        env.reset()
        action = np.zeros(cfg.action.feature_dim)
        _, _, _, info = env.step(action)

        # With zero action, ranking should match baseline
        assert info["delta_rank_at_k_vs_baseline"] == 0.0

    def test_nonzero_action_may_change_ranking(self):
        """Non-zero action may produce non-zero delta_rank."""
        cfg = load_default_config()
        env = ZooplusSearchEnv(cfg, seed=42)

        env.reset()
        # Use a non-zero action
        action = np.ones(cfg.action.feature_dim) * 2.0
        _, _, _, info = env.step(action)

        # Delta rank could be zero or positive depending on the ranking
        assert info["delta_rank_at_k_vs_baseline"] >= 0.0
        assert info["delta_rank_at_k_vs_baseline"] <= 1.0


class TestCMDPWithEnv:
    """Tests for CMDP agent with real environment."""

    def test_cmdp_training_loop(self):
        """Basic training loop completes without error."""
        cfg = load_default_config()
        env = ZooplusSearchEnv(cfg, seed=42)

        constraints = create_standard_constraints(
            cm2_floor=0.0,  # Easy constraint
            stability_ceiling=1.0,  # Easy constraint
        )

        agent = PrimalDualCMDPAgent(
            obs_dim=cfg.action.feature_dim,
            action_dim=cfg.action.feature_dim,
            constraints=constraints,
            config=CMDPConfig(seed=42),
        )

        # Run a few episodes
        for ep in range(3):
            env.reset()
            obs = np.random.randn(cfg.action.feature_dim).astype(np.float32)
            action = agent.select_action(obs)
            _, reward, _, info = env.step(action)

            # Use constraint NAMES as keys (from create_standard_constraints)
            # reward_details is a RewardBreakdown dataclass, not a dict
            breakdown = info.get("reward_details")
            constraint_metrics = {
                "cm2_floor": breakdown.cm2 if breakdown else 0.0,
                "stability": info.get("delta_rank_at_k_vs_baseline", 0.0),
            }
            agent.store_transition(reward, info, constraint_metrics)
            metrics = agent.update()

            assert hasattr(metrics, "policy_loss")
            assert hasattr(metrics, "lambdas")
