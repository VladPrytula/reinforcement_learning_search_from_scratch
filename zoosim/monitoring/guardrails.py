"""Safety guardrails and monitoring wrappers.

Mathematical basis:
- [DEF-10.1] Delta-rank stability
- [DEF-10.2] CM2 Floor constraint (Safety Layer)

References:
    - Chapter 10: Robustness to Drift and Guardrails
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
from numpy.typing import NDArray

from zoosim.monitoring.drift import DriftDetector, PageHinkley, PageHinkleyConfig


class PolicyProtocol(Protocol):
    """Protocol for policies compatible with SafetyMonitor."""

    def select_action(self, features: NDArray[np.float64]) -> int: ...
    def update(
        self, action: int, features: NDArray[np.float64], reward: float
    ) -> None: ...


@dataclass
class GuardrailConfig:
    """Configuration for safety guardrails."""
    
    # Stability
    enable_stability_check: bool = True
    max_action_switch_rate: float = 0.3  # Max % of action switches allowed in window
    stability_window: int = 50

    # Business Constraints
    enable_cm2_floor: bool = False
    min_cm2: float = 0.0
    
    # Drift
    enable_drift_detection: bool = True
    drift_patience: int = 5  # How many drift signals before triggering fallback
    probe_frequency: int = 100  # Episodes to wait before probing primary policy again


class SafetyMonitor:
    """Wraps a policy to enforce guardrails and handle drift.

    Features:
    1. Drift Detection: Monitors rewards. If drift detected, can switch to fallback.
    2. Stability: Tracks how often the policy changes its mind (action switching).
    3. Fallback: Reverts to a 'safe' policy (e.g., best static template) on failure.
    
    Usage:
        monitor = SafetyMonitor(primary_policy, safe_policy)
        action = monitor.select_action(features)
        monitor.update(action, features, reward)
    """

    def __init__(
        self,
        primary_policy: PolicyProtocol,
        safe_policy: PolicyProtocol,
        config: Optional[GuardrailConfig] = None,
        drift_detector: Optional[DriftDetector] = None,
    ):
        self.primary_policy = primary_policy
        self.safe_policy = safe_policy  # e.g. Best Fixed Template
        self.config = config or GuardrailConfig()
        
        # Default to Page-Hinkley for reward degradation detection
        self.drift_detector = drift_detector or PageHinkley(
            PageHinkleyConfig(delta=0.01, threshold=10.0)
        )

        self.in_fallback_mode = False
        self.drift_counter = 0
        self.fallback_counter = 0
        
        # Stability tracking
        self.action_history: List[int] = []
        
        # Diagnostics
        self.violations = {
            "drift_events": 0,
            "stability_breaches": 0,
            "cm2_breaches": 0
        }

    def select_action(self, features: NDArray[np.float64]) -> int:
        """Select action with guardrails applied.
        
        If in fallback mode, uses safe_policy.
        Otherwise, uses primary_policy.
        """
        if self.in_fallback_mode:
            return self.safe_policy.select_action(features)

        action = self.primary_policy.select_action(features)
        
        # Stability Check (Simplified: High frequency action switching)
        # In a real ranking system, this would be Delta-Rank@k. 
        # Here, we check if the policy is thrashing between templates.
        if self.config.enable_stability_check:
            self._check_stability(action)
            # Note: We don't override action for stability here, just log/monitor.
            # Hard constraints would go here.

        return action

    def update(
        self, action: int, features: NDArray[np.float64], reward: float
    ) -> None:
        """Update policy and monitor for drift."""
        
        # 1. Update the active policy
        if self.in_fallback_mode:
            self.safe_policy.update(action, features, reward)
            self.fallback_counter += 1
            
            # Automatic Recovery Probing
            if self.fallback_counter >= self.config.probe_frequency:
                print(f"[SafetyMonitor] 🔄 Probing Primary Policy (after {self.fallback_counter} steps)")
                self.reset_to_primary()
        else:
            self.primary_policy.update(action, features, reward)

        # 2. Monitor Drift (on Reward)
        # We assume 'reward' is something we want to maximize.
        # PageHinkley detects increases in mean (for error) or decreases (if configured).
        # Standard PageHinkley detects increases. If we want to detect reward DROP, 
        # we can feed it (-reward).
        if self.config.enable_drift_detection:
            drift = self.drift_detector.update(-reward)
            if drift:
                self.drift_counter += 1
                if self.drift_counter >= self.config.drift_patience:
                    self._trigger_fallback("Reward Drift Detected")
            else:
                self.drift_counter = max(0, self.drift_counter - 1)

    def _check_stability(self, current_action: int) -> None:
        """Monitor action stability."""
        self.action_history.append(current_action)
        if len(self.action_history) > self.config.stability_window:
            self.action_history.pop(0)
            
        # Calculate switch rate
        if len(self.action_history) > 1:
            switches = sum(1 for i in range(1, len(self.action_history)) 
                         if self.action_history[i] != self.action_history[i-1])
            rate = switches / (len(self.action_history) - 1)
            
            if rate > self.config.max_action_switch_rate:
                self.violations["stability_breaches"] += 1
                # Could trigger fallback here if desired

    def _trigger_fallback(self, reason: str) -> None:
        """Trigger safety fallback mode."""
        if not self.in_fallback_mode:
            print(f"[SafetyMonitor] 🚨 FALLBACK TRIGGERED: {reason}")
            self.in_fallback_mode = True
            self.violations["drift_events"] += 1
            # Reset detector to avoid loops
            self.drift_detector.reset()
            self.drift_counter = 0

    def reset_to_primary(self) -> None:
        """Manually reset to primary policy (e.g., after retraining)."""
        self.in_fallback_mode = False
        self.drift_detector.reset()
        self.fallback_counter = 0
        print("[SafetyMonitor] ✅ Restored Primary Policy")
