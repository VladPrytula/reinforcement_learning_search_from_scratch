"""Multi-Objective Constrained MDP with Primal-Dual Optimization.

This module implements Chapter 14's constrained MDP formulation:
- ConstraintSpec: Declarative constraint specification
- PrimalDualCMDPAgent: REINFORCE-based agent with Lagrangian reward shaping

Mathematical basis:
    - [EQ-14.5]: CMDP formulation with slack constraints
    - [EQ-14.9]: Lagrangian relaxation
    - [EQ-14.12], [EQ-14.13]: Primal-dual updates
    - [ALG-14.5.1]: Primal-Dual CMDP Training algorithm

References:
    - Chapter 14: Multi-Objective RL and Fairness at Scale
    - Appendix C: Convex Optimization for Constrained MDPs
    - [@altman:constrained_mdps:1999]: Constrained MDPs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from zoosim.policies.reinforce_baseline import (
    REINFORCEBaselineAgent,
    REINFORCEBaselineConfig,
)


# ---------------------------------------------------------------------------
# Constraint Specification (Phase 4)
# ---------------------------------------------------------------------------


class ConstraintSense(Enum):
    """Constraint inequality direction.

    GEQ: metric >= threshold (e.g., CM2 floor)
    LEQ: metric <= threshold (e.g., stability ceiling)
    """

    GEQ = ">="  # Greater than or equal
    LEQ = "<="  # Less than or equal


@dataclass
class ConstraintSpec:
    """Specification for a single constraint in the CMDP.

    Following Chapter 14 [EQ-14.5], constraints are expressed as:
        g_i(theta) = (metric - threshold) >= 0  for GEQ sense
        g_i(theta) = (threshold - metric) >= 0  for LEQ sense

    The slack computation ensures positive slack means constraint satisfied.

    Attributes:
        name: Human-readable identifier (e.g., "cm2_floor", "stability")
        threshold: Target value (tau in the equations)
        sense: Inequality direction (GEQ or LEQ)
        metric_key: Key in info dict to extract metric value
        weight: Optional scaling factor for constraint importance

    Example:
        >>> cm2_floor = ConstraintSpec(
        ...     name="cm2_floor",
        ...     threshold=0.1,
        ...     sense=ConstraintSense.GEQ,
        ...     metric_key="reward_details",  # extract CM2 from breakdown
        ... )
    """

    name: str
    threshold: float
    sense: ConstraintSense
    metric_key: str
    weight: float = 1.0

    def compute_slack(self, metric_value: float) -> float:
        """Compute constraint slack (positive = satisfied, negative = violated).

        For GEQ constraints: slack = metric - threshold
        For LEQ constraints: slack = threshold - metric

        This follows [EQ-14.6]--[EQ-14.8] in Chapter 14.

        Args:
            metric_value: Current value of the constrained metric

        Returns:
            Slack value. Positive means constraint satisfied with margin,
            negative means constraint violated.
        """
        if self.sense == ConstraintSense.GEQ:
            return metric_value - self.threshold
        else:  # LEQ
            return self.threshold - metric_value

    def is_satisfied(self, metric_value: float, tolerance: float = 0.0) -> bool:
        """Check if constraint is satisfied within tolerance.

        Args:
            metric_value: Current value of the constrained metric
            tolerance: Acceptable slack below zero

        Returns:
            True if slack >= -tolerance
        """
        return self.compute_slack(metric_value) >= -tolerance


def create_standard_constraints(
    cm2_floor: Optional[float] = None,
    stability_ceiling: Optional[float] = None,
    exposure_band: Optional[Tuple[float, float]] = None,
    exposure_group: str = "pl",
) -> List[ConstraintSpec]:
    """Create standard constraint specifications for search ranking.

    This is a convenience factory for common constraint configurations
    used in Chapter 14 experiments.

    Args:
        cm2_floor: Minimum required CM2 (None = no constraint)
        stability_ceiling: Maximum delta_rank@k (None = no constraint)
        exposure_band: (min, max) exposure share for a group (None = no constraint)
        exposure_group: Group name for exposure constraint

    Returns:
        List of ConstraintSpec objects

    Example:
        >>> constraints = create_standard_constraints(
        ...     cm2_floor=0.1,
        ...     stability_ceiling=0.2,
        ...     exposure_band=(0.3, 0.5),
        ... )
    """
    constraints = []

    if cm2_floor is not None:
        constraints.append(
            ConstraintSpec(
                name="cm2_floor",
                threshold=cm2_floor,
                sense=ConstraintSense.GEQ,
                metric_key="cm2",
            )
        )

    if stability_ceiling is not None:
        constraints.append(
            ConstraintSpec(
                name="stability",
                threshold=stability_ceiling,
                sense=ConstraintSense.LEQ,
                metric_key="delta_rank_at_k_vs_baseline",
            )
        )

    if exposure_band is not None:
        lo, hi = exposure_band
        constraints.append(
            ConstraintSpec(
                name=f"exposure_{exposure_group}_lo",
                threshold=lo,
                sense=ConstraintSense.GEQ,
                metric_key=f"exposure_{exposure_group}",
            )
        )
        constraints.append(
            ConstraintSpec(
                name=f"exposure_{exposure_group}_hi",
                threshold=hi,
                sense=ConstraintSense.LEQ,
                metric_key=f"exposure_{exposure_group}",
            )
        )

    return constraints


# ---------------------------------------------------------------------------
# Primal-Dual CMDP Agent (Phase 5)
# ---------------------------------------------------------------------------


@dataclass
class CMDPConfig(REINFORCEBaselineConfig):
    """Configuration for Primal-Dual CMDP Agent.

    Extends REINFORCEBaselineConfig with dual-variable parameters.
    """

    # Dual variable (Lagrange multiplier) parameters
    lambda_lr: float = 0.01  # Learning rate for dual update [EQ-14.13]
    lambda_init: float = 0.0  # Initial multiplier value
    lambda_max: float = 100.0  # Maximum multiplier (prevents divergence)

    # Constraint satisfaction monitoring
    constraint_ema_alpha: float = 0.1  # EMA smoothing for violation tracking


@dataclass
class CMDPMetrics:
    """Training metrics for CMDP agent."""

    policy_loss: float
    value_loss: float
    shaped_return: float
    raw_return: float
    entropy: float
    lambdas: Dict[str, float]
    slacks: Dict[str, float]
    violations: Dict[str, bool]


class PrimalDualCMDPAgent:
    """REINFORCE with Lagrangian reward shaping for constrained MDPs.

    Implements [ALG-14.5.1] from Chapter 14:
    1. Sample trajectory with shaped reward r' = r + sum_i lambda_i * slack_i
    2. Primal step: Policy gradient on shaped reward
    3. Dual step: Adjust multipliers based on constraint violations

    Under Slater's condition (Appendix C), this converges to the
    constrained optimum with zero duality gap.

    Attributes:
        constraints: List of constraint specifications
        lambdas: Current Lagrange multipliers (dual variables)
        base_agent: Underlying REINFORCE agent for policy updates
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        constraints: List[ConstraintSpec],
        action_scale: float = 1.0,
        config: Optional[CMDPConfig] = None,
    ):
        """Initialize CMDP agent.

        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            constraints: List of constraint specifications
            action_scale: Scaling factor for actions
            config: Agent configuration
        """
        self.config = config or CMDPConfig()
        self.constraints = constraints
        self.action_scale = action_scale

        # Initialize Lagrange multipliers (dual variables)
        self.lambdas: Dict[str, float] = {
            c.name: self.config.lambda_init for c in constraints
        }

        # EMA of constraint violations for monitoring
        self._violation_ema: Dict[str, float] = {c.name: 0.0 for c in constraints}

        # Base policy gradient agent
        self.base_agent = REINFORCEBaselineAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_scale=action_scale,
            config=self.config,
        )

        # Episode buffer for constraint metrics
        self._constraint_metrics: List[Dict[str, float]] = []
        self._raw_rewards: List[float] = []

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action using current policy.

        Args:
            obs: Observation array

        Returns:
            Action array (scaled)
        """
        return self.base_agent.select_action(obs)

    def store_transition(
        self,
        reward: float,
        info: Dict,
        constraint_metrics: Optional[Dict[str, float]] = None,
    ) -> float:
        """Store transition and compute shaped reward.

        The shaped reward follows [EQ-14.11]:
            r' = r + sum_i lambda_i * slack_i

        Args:
            reward: Raw environment reward
            info: Info dict from env.step() (may contain constraint metrics)
            constraint_metrics: Explicit constraint metrics (overrides info)

        Returns:
            Shaped reward value
        """
        # Extract constraint metrics from info or explicit dict
        metrics = constraint_metrics or {}
        for c in self.constraints:
            if c.metric_key in info and c.name not in metrics:
                # Try to extract from info
                val = info.get(c.metric_key)
                if isinstance(val, (int, float)):
                    metrics[c.name] = float(val)
                elif isinstance(val, dict) and "cm2" in c.name.lower():
                    # Extract CM2 from reward_details breakdown
                    metrics[c.name] = val.get("cm2", 0.0)

        self._constraint_metrics.append(metrics)
        self._raw_rewards.append(reward)

        # Compute shaped reward: r' = r + sum_i lambda_i * slack_i
        shaped_reward = reward
        for c in self.constraints:
            if c.name in metrics:
                slack = c.compute_slack(metrics[c.name])
                shaped_reward += self.lambdas[c.name] * slack * c.weight

        # Store shaped reward in base agent
        self.base_agent.store_reward(shaped_reward)

        return shaped_reward

    def update(self) -> CMDPMetrics:
        """Perform primal-dual update.

        Primal step: Update policy using shaped rewards (via base_agent)
        Dual step: Adjust Lagrange multipliers based on violations [EQ-14.13]

        Returns:
            CMDPMetrics with training statistics
        """
        # Primal step: Policy gradient update on shaped rewards
        base_metrics = self.base_agent.update()

        # Aggregate constraint metrics over episode
        avg_metrics: Dict[str, float] = {}
        for c in self.constraints:
            vals = [m.get(c.name, 0.0) for m in self._constraint_metrics if c.name in m]
            if vals:
                avg_metrics[c.name] = np.mean(vals)

        # Compute slacks and violations
        slacks: Dict[str, float] = {}
        violations: Dict[str, bool] = {}
        for c in self.constraints:
            if c.name in avg_metrics:
                slack = c.compute_slack(avg_metrics[c.name])
                slacks[c.name] = slack
                violations[c.name] = slack < 0

                # Update violation EMA
                v = 1.0 if slack < 0 else 0.0
                self._violation_ema[c.name] = (
                    self.config.constraint_ema_alpha * v
                    + (1 - self.config.constraint_ema_alpha) * self._violation_ema[c.name]
                )

        # Dual step: Adjust multipliers [EQ-14.13]
        # lambda_new = max(0, lambda_old - eta * slack)
        # Note: We SUBTRACT slack because positive slack means satisfied
        for c in self.constraints:
            if c.name in slacks:
                new_lambda = self.lambdas[c.name] - self.config.lambda_lr * slacks[c.name]
                # Project to [0, lambda_max]
                self.lambdas[c.name] = np.clip(new_lambda, 0.0, self.config.lambda_max)

        # Compute metrics
        raw_return = sum(self._raw_rewards)
        shaped_return = base_metrics["return_mean"]

        metrics = CMDPMetrics(
            policy_loss=base_metrics["loss"],
            value_loss=base_metrics["value_loss"],
            shaped_return=shaped_return,
            raw_return=raw_return,
            entropy=base_metrics["entropy"],
            lambdas=dict(self.lambdas),
            slacks=slacks,
            violations=violations,
        )

        # Clear episode buffers
        self._constraint_metrics = []
        self._raw_rewards = []

        return metrics

    def get_violation_rates(self) -> Dict[str, float]:
        """Get EMA-smoothed violation rates for each constraint.

        Returns:
            Dict mapping constraint name to violation rate in [0, 1]
        """
        return dict(self._violation_ema)

    def all_constraints_satisfied(self, tolerance: float = 0.0) -> bool:
        """Check if all constraints are currently satisfied.

        Args:
            tolerance: Acceptable slack below zero

        Returns:
            True if all constraint violation EMAs are below tolerance
        """
        return all(v <= tolerance for v in self._violation_ema.values())


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def train_cmdp_episode(
    agent: PrimalDualCMDPAgent,
    env,
    extract_metrics: Optional[Callable[[Dict], Dict[str, float]]] = None,
) -> CMDPMetrics:
    """Run one episode of CMDP training.

    Args:
        agent: CMDP agent
        env: Environment with step() returning (obs, reward, done, info)
        extract_metrics: Optional function to extract constraint metrics from info

    Returns:
        CMDPMetrics from the episode update
    """
    obs = env.reset()

    # Handle dict observation (ZooplusSearchEnv returns dict from reset)
    if isinstance(obs, dict):
        # Convert dict to flat array if needed
        # For now, assume env provides observation in expected format
        pass

    done = False
    while not done:
        action = agent.select_action(obs if isinstance(obs, np.ndarray) else np.zeros(10))
        next_obs, reward, done, info = env.step(action)

        # Extract constraint metrics
        metrics = extract_metrics(info) if extract_metrics else {}

        # Store transition with shaped reward
        agent.store_transition(reward, info, metrics)

        obs = next_obs

    # Perform primal-dual update
    return agent.update()


def extract_ranking_metrics(info: Dict) -> Dict[str, float]:
    """Extract standard ranking constraint metrics from env info dict.

    Designed for ZooplusSearchEnv info structure.

    Args:
        info: Info dict from env.step()

    Returns:
        Dict with extracted metric values
    """
    metrics = {}

    # Delta-rank stability (Chapter 14)
    if "delta_rank_at_k_vs_baseline" in info:
        metrics["delta_rank_at_k_vs_baseline"] = info["delta_rank_at_k_vs_baseline"]

    # CM2 from reward breakdown
    if "reward_details" in info and isinstance(info["reward_details"], dict):
        breakdown = info["reward_details"]
        if "cm2" in breakdown:
            metrics["cm2"] = breakdown["cm2"]
        if "gmv" in breakdown:
            metrics["gmv"] = breakdown["gmv"]

    return metrics
