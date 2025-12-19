"""Unit tests for Chapter 10: Monitoring and Guardrails."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from zoosim.monitoring.drift import CUSUM, CUSUMConfig, PageHinkley, PageHinkleyConfig
from zoosim.monitoring.guardrails import SafetyMonitor, GuardrailConfig
from zoosim.policies.templates import BoostTemplate


def test_cusum_detection():
    """Test CUSUM drift detection on a simple shift."""
    detector = CUSUM(CUSUMConfig(min_instances=10, delta=0.5, lambda_factor=5.0))
    
    # Phase 1: Mean 0
    for _ in range(50):
        assert not detector.update(np.random.normal(0, 0.1))
        
    # Phase 2: Mean 2 (Should detect)
    detected = False
    for _ in range(50):
        if detector.update(np.random.normal(2, 0.1)):
            detected = True
            break
            
    assert detected, "CUSUM failed to detect mean shift from 0 to 2"
    assert detector.detected


def test_page_hinkley_detection():
    """Test Page-Hinkley detection."""
    detector = PageHinkley(PageHinkleyConfig(min_instances=10, delta=0.1, threshold=5.0))
    
    # Phase 1: Mean 0
    for _ in range(50):
        assert not detector.update(np.random.normal(0, 0.1))
        
    # Phase 2: Mean 1 (Should detect increase)
    detected = False
    for _ in range(50):
        if detector.update(np.random.normal(1, 0.1)):
            detected = True
            break
            
    assert detected, "Page-Hinkley failed to detect mean shift"


def test_safety_monitor_fallback():
    """Test SafetyMonitor fallback mechanism."""
    
    # Mock policies
    primary_policy = MagicMock()
    primary_policy.select_action.return_value = 0
    
    safe_policy = MagicMock()
    safe_policy.select_action.return_value = 1
    
    # Config: sensitive drift detection
    detector = PageHinkley(PageHinkleyConfig(min_instances=5, delta=0.1, threshold=1.0))
    
    monitor = SafetyMonitor(
        primary_policy=primary_policy,
        safe_policy=safe_policy,
        config=GuardrailConfig(drift_patience=1),
        drift_detector=detector
    )
    
    # 1. Normal operation
    features = np.zeros(5)
    action = monitor.select_action(features)
    assert action == 0
    assert not monitor.in_fallback_mode
    
    # 2. Induce Drift (simulate bad rewards)
    # PageHinkley checks for increase in value.
    # monitor.update passes -reward. So if reward drops (e.g. 1.0 -> 0.0), -reward increases (-1.0 -> 0.0).
    # Let's say baseline reward is 1.0. -reward is -1.0.
    for _ in range(20):
        monitor.update(0, features, 1.0)
        
    # Now reward drops to 0.0. -reward is 0.0 (increase of 1.0).
    # Threshold is 1.0. Should trigger.
    triggered = False
    for _ in range(20):
        monitor.update(0, features, 0.0)
        if monitor.in_fallback_mode:
            triggered = True
            break
            
    assert triggered, "SafetyMonitor failed to trigger fallback on reward drop"
    
    # 3. Fallback operation
    action = monitor.select_action(features)
    assert action == 1, "SafetyMonitor did not switch to safe policy action"


def test_stability_check():
    """Test stability guardrail (action switching)."""
    primary_policy = MagicMock()
    # Alternating actions
    primary_policy.select_action.side_effect = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    monitor = SafetyMonitor(
        primary_policy=primary_policy,
        safe_policy=MagicMock(),
        config=GuardrailConfig(enable_stability_check=True, max_action_switch_rate=0.5, stability_window=10)
    )
    
    features = np.zeros(5)
    for _ in range(10):
        monitor.select_action(features)
        
    # Should have violations
    assert monitor.violations["stability_breaches"] > 0
