r"""LinUCB (Linear Upper Confidence Bound) for contextual bandits.

Mathematical basis:
- [ALG-6.2] LinUCB algorithm
- [EQ-6.15] UCB action selection rule
- [THM-6.2] Regret bound $O(d \sqrt{M T \log T})$

Implements frequentist upper confidence bound exploration with deterministic
action selection. Maintains ridge regression estimates and selects the action
with highest optimistic reward estimate.

References:
    - Chapter 6, §6.4-6.5: LinUCB theory and implementation
    - [EQ-6.13]: UCB formula with exploration bonus
    - [EQ-6.14]: Uncertainty quantification √(φ^T Σ_a φ)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from zoosim.policies.templates import BoostTemplate


@dataclass
class LinUCBConfig:
    """Configuration for LinUCB policy.

    Attributes:
        lambda_reg: Regularization strength (ridge regression)
                   Prevents overfitting, ensures A_a is invertible
                   Typical range: [0.1, 10.0]
        alpha: Exploration parameter (UCB width multiplier)
               Controls exploration-exploitation tradeoff
               Typical values: α ∈ [0.5, 2.0]
               Theory: α = √(d log T) for optimal regret
        adaptive_alpha: If True, use α_t = alpha * √log(1 + t)
                       Automatically decays exploration over time
        seed: Random seed (for feature hashing, not action selection)
             LinUCB is deterministic given same features/data
        enable_diagnostics: If True, record per-episode diagnostic traces
                            (UCB scores, means, uncertainties, actions)
    """

    lambda_reg: float = 1.0
    alpha: float = 1.0
    adaptive_alpha: bool = False
    seed: int = 42
    enable_diagnostics: bool = False


class LinUCB:
    """Linear Upper Confidence Bound algorithm for contextual bandits.

    Maintains ridge regression estimates θ̂_a for each template and selects
    action with highest upper confidence bound:

        a = argmax_a {θ̂_a^T φ(x) + α √(φ(x)^T A_a^{-1} φ(x))}

    This is a frequentist alternative to Thompson Sampling with deterministic
    action selection. Both maintain identical posterior mean θ̂_a but differ
    in how they use uncertainty for exploration.

    Mathematical correspondence: Implements [ALG-6.2]

    Attributes:
        config: LinUCBConfig with hyperparameters
        templates: List of M boost templates
        M: Number of templates (action space size)
        d: Feature dimension
        theta_hat: Ridge regression weights, shape (M, d)
                  θ̂_a = A_a^{-1} b_a (ridge regression solution)
        A: Design matrices A_a = λI + Σ φ φ^T, shape (M, d, d)
        b: Reward accumulators b_a = Σ r φ, shape (M, d)
        n_samples: Selection counts per template, shape (M,)
        t: Current episode number (for adaptive α)
    """

    def __init__(
        self,
        templates: List[BoostTemplate],
        feature_dim: int,
        config: Optional[LinUCBConfig] = None,
    ):
        """Initialize LinUCB policy.

        Args:
            templates: List of M boost templates ([DEF-6.1.1])
            feature_dim: Feature dimension d
            config: Optional configuration (uses defaults if None)
        """
        self.templates = templates
        self.M = len(templates)
        self.d = feature_dim
        self.config = config or LinUCBConfig()

        # Initialize statistics: [ALG-6.2] initialization
        # Weight estimates: θ̂_a ← 0 ∈ ℝ^d
        self.theta_hat = np.zeros((self.M, self.d), dtype=np.float64)

        # Design matrices: A_a ← λI (prior precision)
        self.A = np.array(
            [
                self.config.lambda_reg * np.eye(self.d, dtype=np.float64)
                for _ in range(self.M)
            ]
        )

        # Reward accumulators: b_a ← 0 ∈ ℝ^d
        self.b = np.zeros((self.M, self.d), dtype=np.float64)

        self.n_samples = np.zeros(self.M, dtype=int)
        self.t = 0  # Episode counter

        # Optional per-episode diagnostics (history traces)
        self.enable_diagnostics = bool(self.config.enable_diagnostics)
        if self.enable_diagnostics:
            self._diagnostics_history: Dict[str, list] = {
                "ucb_scores_history": [],
                "mean_rewards_history": [],
                "uncertainties_history": [],
                "selected_actions": [],
            }

    def select_action(self, features: NDArray[np.float64]) -> int:
        """Select template using UCB criterion.

        Implements [ALG-6.2] steps 3-4 and [EQ-6.15].

        For each template a, computes:
        - Mean estimate: θ̂_a^T φ  [exploitation]
        - Uncertainty bonus: α √(φ^T A_a^{-1} φ)  [exploration]
        - UCB score: mean + bonus

        Selects template with highest UCB score (deterministic).

        Args:
            features: Context features φ(x), shape (d,)

        Returns:
            action: Selected template ID in {0, ..., M-1}
        """
        self.t += 1

        # Compute adaptive exploration parameter
        alpha = self.config.alpha
        if self.config.adaptive_alpha:
            alpha *= np.sqrt(np.log(1 + self.t))

        # Compute UCB scores for all templates
        ucb_scores = np.zeros(self.M)
        mean_rewards = np.zeros(self.M)
        uncertainties = np.zeros(self.M)
        for a in range(self.M):
            # Mean estimate: θ̂_a^T φ
            mean_reward = self.theta_hat[a] @ features
            mean_rewards[a] = mean_reward

            # Uncertainty bonus: α √(φ^T A_a^{-1} φ)  [EQ-6.14]
            # Compute \\phi^T A_a^{-1} \\phi without forming A_a^{-1} explicitly.
            # Solve A_a u = \\phi => u = A_a^{-1} \\phi, then take \\phi^T u.
            a_inv_phi = np.linalg.solve(self.A[a], features)
            quad_form = float(features @ a_inv_phi)
            uncertainty = np.sqrt(max(quad_form, 0.0))
            uncertainties[a] = uncertainty

            # UCB score [EQ-6.15]
            ucb_scores[a] = mean_reward + alpha * uncertainty

        # Select action with highest UCB (deterministic)
        action = int(np.argmax(ucb_scores))

        if self.enable_diagnostics:
            # Store shallow copies to avoid accidental mutation
            self._diagnostics_history["ucb_scores_history"].append(ucb_scores.copy())
            self._diagnostics_history["mean_rewards_history"].append(mean_rewards.copy())
            self._diagnostics_history["uncertainties_history"].append(
                uncertainties.copy()
            )
            self._diagnostics_history["selected_actions"].append(action)

        return action

    def update(
        self,
        action: int,
        features: NDArray[np.float64],
        reward: float,
    ) -> None:
        """Update statistics after observing (action, features, reward).

        Implements [ALG-6.2] step 6: Ridge regression update.

        Update equations:
        - A_a ← A_a + φφ^T  (accumulate design matrix)
        - b_a ← b_a + rφ    (accumulate reward-weighted features)
        - θ̂_a ← A_a^{-1} b_a  (solve ridge regression)

        Note: This is identical to Thompson Sampling posterior update
        (both maintain same θ̂_a). Only action selection differs.

        Args:
            action: Selected template ID
            features: Context features φ(x), shape (d,)
            reward: Observed reward r ∈ ℝ
        """
        a = action
        phi = features

        # Update design matrix: A_a ← A_a + φφ^T
        self.A[a] += np.outer(phi, phi)

        # Update reward accumulator: b_a ← b_a + rφ
        self.b[a] += reward * phi

        # Update weight estimate: θ̂_a ← A_a^{-1} b_a
        # Using np.linalg.solve for numerical stability (avoid explicit inversion)
        self.theta_hat[a] = np.linalg.solve(self.A[a], self.b[a])

        # Track selection count
        self.n_samples[a] += 1

    def get_diagnostics(self) -> Dict[str, NDArray[np.float64] | float]:
        """Return diagnostic information for monitoring.

        Returns:
            diagnostics: Dictionary with keys:
                - 'selection_counts': Selection counts per template, shape (M,)
                - 'selection_frequencies': Selection frequencies, shape (M,)
                - 'theta_norms': ||θ̂_a||_2 for each template, shape (M,)
                - 'uncertainty': Trace(A_a^{-1}) (average uncertainty), shape (M,)
                - 'alpha_current': Current exploration parameter (scalar)
        """
        total_samples = self.n_samples.sum()
        selection_freqs = (
            self.n_samples / total_samples if total_samples > 0 else self.n_samples
        )
        theta_norms = np.linalg.norm(self.theta_hat, axis=1)
        uncertainties = np.array(
            [np.trace(np.linalg.inv(self.A[a])) for a in range(self.M)]
        )

        alpha_current = self.config.alpha
        if self.config.adaptive_alpha:
            alpha_current *= np.sqrt(np.log(1 + self.t)) if self.t > 0 else 1.0

        return {
            "selection_counts": self.n_samples.copy(),
            "selection_frequencies": selection_freqs,
            "theta_norms": theta_norms,
            "uncertainty": uncertainties,
            "alpha_current": float(alpha_current),
        }

    def get_diagnostic_history(self) -> Dict[str, list]:
        """Return full per-episode diagnostic traces if enabled.

        The history dictionary contains:
            - 'ucb_scores_history': List of arrays of shape (M,)
            - 'mean_rewards_history': List of arrays of shape (M,)
            - 'uncertainties_history': List of arrays of shape (M,)
            - 'selected_actions': List of ints (chosen template per episode)

        Raises:
            ValueError: If diagnostics history was not enabled at construction
                        time (config.enable_diagnostics=False).
        """
        if not self.enable_diagnostics:
            raise ValueError(
                "Diagnostics history not enabled; set "
                "LinUCBConfig.enable_diagnostics=True when constructing the policy."
            )
        return self._diagnostics_history


# Code ↔ Algorithm mapping
# - Line 125-150: select_action() computes UCB scores via [EQ-6.15], selects argmax
# - Line 152-180: update() performs ridge regression update (design matrix + weight solve)
# - Line 138-140: Adaptive α_t = α√log(1+t) option for automatic exploration decay
# - Line 145-147: Uncertainty computation √(φ^T A_a^{-1} φ) from [EQ-6.14]
#
# File: zoosim/policies/lin_ucb.py
