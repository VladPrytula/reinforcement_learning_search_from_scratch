"""Drift detection algorithms for non-stationary environments.

Mathematical basis:
- [ALG-10.1] CUSUM (Cumulative Sum)
- [ALG-10.2] Page-Hinkley Test

References:
    - Chapter 10: Robustness to Drift and Guardrails
    - [EQ-10.1] Lyapunov stability condition
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np


class DriftDetector(abc.ABC):
    """Abstract base class for drift detection algorithms.

    Interface:
    - update(value): Process new observation
    - detected: Boolean property indicating if drift was detected
    - reset(): Reset internal state
    """

    @abc.abstractmethod
    def update(self, value: float) -> bool:
        """Update detector with new observation.

        Args:
            value: Observed value (e.g., reward, error, log-likelihood)

        Returns:
            detected: True if drift is detected, False otherwise
        """
        pass

    @property
    @abc.abstractmethod
    def detected(self) -> bool:
        """Return True if drift is currently detected."""
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
        pass


@dataclass
class CUSUMConfig:
    """Configuration for CUSUM detector.

    Attributes:
        min_instances: Minimum number of samples before detecting
        delta: Drift magnitude to detect (mean shift)
        lambda_factor: Sensitivity threshold (usually based on standard deviation)
    """

    min_instances: int = 30
    delta: float = 0.05
    lambda_factor: float = 50.0  # Threshold = lambda * sigma


class CUSUM(DriftDetector):
    r"""Cumulative Sum (CUSUM) drift detector.

    Detects shifts in the mean of a stream of values. Two-sided test.
    Maintains sum of deviations from the mean.

    Mathematical Basis: [ALG-10.1]
    S_t = max(0, S_{t-1} + (x_t - \mu - \delta))
    If S_t > h, alarm.
    """

    def __init__(self, config: Optional[CUSUMConfig] = None):
        self.config = config or CUSUMConfig()
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.sum_pos = 0.0
        self.sum_neg = 0.0
        self._detected = False
        self.max_sq_sum = 0.0  # For variance estimation (Welford)
        self.variance = 0.0

    @property
    def detected(self) -> bool:
        return self._detected

    def update(self, value: float) -> bool:
        self.n += 1

        # Update running mean and variance (Welford's algorithm)
        delta_mean = value - self.mean
        self.mean += delta_mean / self.n
        self.max_sq_sum += delta_mean * (value - self.mean)
        
        if self.n < 2:
            std = 0.0
        else:
            self.variance = self.max_sq_sum / (self.n - 1)
            std = np.sqrt(self.variance)

        if self.n < self.config.min_instances:
            return False

        # Two-sided CUSUM
        # Shift magnitude we care about
        magnitude = self.config.delta * std if std > 1e-9 else self.config.delta

        self.sum_pos = max(0.0, self.sum_pos + value - self.mean - magnitude)
        self.sum_neg = min(0.0, self.sum_neg + value - self.mean + magnitude)

        threshold = self.config.lambda_factor * std if std > 1e-9 else self.config.lambda_factor

        if self.sum_pos > threshold or self.sum_neg < -threshold:
            self._detected = True
        else:
            self._detected = False

        return self._detected


@dataclass
class PageHinkleyConfig:
    """Configuration for Page-Hinkley detector.

    Attributes:
        min_instances: Minimum number of samples
        delta: Minimum magnitude of change
        threshold: Detection threshold (lambda)
        alpha: Fading factor (for forgetting old data) - optional extension
    """

    min_instances: int = 30
    delta: float = 0.005
    threshold: float = 50.0
    alpha: float = 0.9999  # Forgetting factor


class PageHinkley(DriftDetector):
    r"""Page-Hinkley Test for abrupt change detection.

    Detects changes in the average of a signal.
    Good for detecting increases in error or decreases in reward.

    Mathematical Basis: [ALG-10.2]
    m_t = \sum (x_i - \bar{x} - \delta)
    M_t = min(m_t)
    Test: m_t - M_t > \lambda
    """

    def __init__(self, config: Optional[PageHinkleyConfig] = None):
        self.config = config or PageHinkleyConfig()
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.sum_diff = 0.0
        self.min_sum_diff = 0.0
        self._detected = False

    @property
    def detected(self) -> bool:
        return self._detected

    def update(self, value: float) -> bool:
        self.n += 1
        
        # Update mean
        self.mean = self.mean + (value - self.mean) / self.n

        # Cumulative difference
        self.sum_diff += (value - self.mean - self.config.delta)

        if self.sum_diff < self.min_sum_diff:
            self.min_sum_diff = self.sum_diff

        if self.n < self.config.min_instances:
            return False

        if self.sum_diff - self.min_sum_diff > self.config.threshold:
            self._detected = True
        else:
            self._detected = False
            
        return self._detected
