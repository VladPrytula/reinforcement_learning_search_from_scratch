"""Thompson Sampling for linear contextual bandits.

Mathematical basis:
- [DEF-6.2] Stochastic Contextual Bandit
- [ALG-6.1] Linear Thompson Sampling
- [THM-6.1] Regret bound O(d√(MT log T))

Implements Bayesian posterior sampling over template weights with Gaussian
conjugate prior. At each episode, samples from posterior distribution and
selects the template with highest sampled expected reward.

References:
    - Chapter 6, §6.2-6.3: Thompson Sampling theory and implementation
    - [EQ-6.6]: Gaussian posterior θ_a ~ N(θ̂_a, Σ_a)
    - [EQ-6.8]: Bayesian update equations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from zoosim.policies.templates import BoostTemplate


@dataclass
class ThompsonSamplingConfig:
    """Configuration for Thompson Sampling policy.

    Attributes:
        lambda_reg: Prior precision (regularization strength)
                   Higher values → stronger regularization, slower learning
                   Typical range: [0.1, 10.0]
        sigma_noise: Reward noise standard deviation
                    Estimated from data or set conservatively (e.g., 1.0)
                    Typical range: [0.5, 2.0]
        use_cholesky: Use Cholesky decomposition for numerical stability
                     Recommended for production (default True)
        seed: Random seed for reproducibility (per-policy RNG)
    """

    lambda_reg: float = 1.0
    sigma_noise: float = 1.0
    use_cholesky: bool = True
    seed: int = 42


class LinearThompsonSampling:
    """Thompson Sampling for linear contextual bandits with discrete templates.

    Maintains Gaussian posterior N(θ_a, Σ_a) for each template's weight vector.
    At each episode:
    1. Sample θ̃_a ~ N(θ̂_a, Σ_a) for all templates
    2. Select template a = argmax_a θ̃_a^T φ(x)
    3. Observe reward r, update posterior for selected template

    Mathematical correspondence: Implements [ALG-6.1]

    The posterior updates follow Bayesian linear regression with Gaussian
    conjugate prior. Posterior mean θ̂_a matches the ridge regression
    solution (see [EQ-6.11]–[EQ-6.13] and [EQ-6.8]).

    Attributes:
        config: ThompsonSamplingConfig with hyperparameters
        templates: List of M boost templates ([DEF-6.1])
        M: Number of templates (action space size)
        d: Feature dimension
        theta_hat: Posterior means, shape (M, d)
        Sigma_inv: Posterior precision matrices, shape (M, d, d)
                  Σ_a^{-1} = λI + σ^{-2} Σ_{t:a_t=a} φ_t φ_t^T
        b: Accumulated moment vectors, shape (M, d)
           b_a = σ^{-2} Σ_{t:a_t=a} r_t φ_t
        n_samples: Number of times each template was selected, shape (M,)
        cholesky_factors: Cholesky factors L_a with Σ_a^{-1} = L_a L_a^T
                         Only maintained if config.use_cholesky = True.
        rng: NumPy Generator used for posterior sampling (seeded per policy).
    """

    def __init__(
        self,
        templates: List[BoostTemplate],
        feature_dim: int,
        config: Optional[ThompsonSamplingConfig] = None,
    ):
        """Initialize Thompson Sampling policy.

        Args:
            templates: List of M boost templates
            feature_dim: Dimension d of feature vectors φ(x)
            config: Optional configuration (uses defaults if None)
        """
        self.templates = templates
        self.M = len(templates)
        self.d = feature_dim
        self.config = config or ThompsonSamplingConfig()

        # Per-policy RNG (do not touch global NumPy state)
        self.rng = np.random.default_rng(self.config.seed)

        # Initialize Gaussian priors: θ_a ~ N(0, λ^{-1} I)
        # Prior mean: 0 (no initial bias)
        self.theta_hat = np.zeros((self.M, self.d), dtype=np.float64)

        # Prior precision: Σ_a^{-1} = λI (regularization)
        self.Sigma_inv = np.array(
            [
                self.config.lambda_reg * np.eye(self.d, dtype=np.float64)
                for _ in range(self.M)
            ]
        )

        # Reward-weighted feature accumulators: b_a = σ^{-2} Σ r_t φ_t
        self.b = np.zeros((self.M, self.d), dtype=np.float64)

        # Track selection counts
        self.n_samples = np.zeros(self.M, dtype=int)

        # Numerical stability: Precompute Cholesky factors
        # Σ_a^{-1} = L_a L_a^T (Cholesky decomposition)
        if self.config.use_cholesky:
            self.cholesky_factors = [
                np.linalg.cholesky(self.Sigma_inv[a]) for a in range(self.M)
            ]
        else:
            self.cholesky_factors = None

    def select_action(self, features: NDArray[np.float64]) -> int:
        """Select template using Thompson Sampling.

        Implements [ALG-6.1] steps 3-4: Sample posteriors, select optimistic action.

        For each template a:
        1. Sample θ̃_a ~ N(θ̂_a, Σ_a) from posterior
        2. Compute expected reward θ̃_a^T φ(x)
        3. Return a = argmax_a θ̃_a^T φ(x)

        Args:
            features: Context feature vector φ(x), shape (d,)

        Returns:
            action: Selected template ID in {0, ..., M-1}
        """
        theta_samples: List[NDArray[np.float64]] = []

        for a in range(self.M):
            if self.config.use_cholesky and self.cholesky_factors is not None:
                # Sample using precision Cholesky: Σ_a^{-1} = L_a L_a^T
                # Draw z ~ N(0, I) and solve L_a^T v = z ⇒ v ~ N(0, Σ_a)
                L_a = self.cholesky_factors[a]
                z = self.rng.standard_normal(self.d)
                v = np.linalg.solve(L_a.T, z)
                theta_tilde = self.theta_hat[a] + v
            else:
                # Fallback: form Σ_a explicitly
                Sigma_a = np.linalg.inv(self.Sigma_inv[a])
                theta_tilde = self.rng.multivariate_normal(
                    mean=self.theta_hat[a], cov=Sigma_a
                )

            theta_samples.append(theta_tilde)

        # Select action with highest sampled expected reward
        expected_rewards = [theta_samples[a] @ features for a in range(self.M)]
        action = int(np.argmax(expected_rewards))

        return action

    def update(
        self,
        action: int,
        features: NDArray[np.float64],
        reward: float,
    ) -> None:
        """Update posterior after observing (action, features, reward).

        Implements [ALG-6.1] step 6: Bayesian update via [EQ-6.7]–[EQ-6.8].

        Update equations (Gaussian conjugate prior):
        - Σ_a^{-1} ← Σ_a^{-1} + σ^{-2} φφ^T          [EQ-6.7]
        - b_a ← b_a + σ^{-2} rφ
        - θ̂_a ← Σ_a b_a with Σ_a = (Σ_a^{-1})^{-1}  [EQ-6.8]

        This is equivalent to ridge regression:
        θ̂_a = argmin_θ {Σ_{t:a_t=a} (r_t - θ^T φ_t)^2 + λ||θ||^2}

        Args:
            action: Selected template ID
            features: Context features φ(x), shape (d,)
            reward: Observed reward r ∈ ℝ
        """
        a = action
        phi = features.reshape(-1, 1)  # Column vector (d, 1)

        sigma_sq = self.config.sigma_noise**2

        # Precision update: Σ_a^{-1} ← Σ_a^{-1} + σ^{-2} φφ^T  [EQ-6.7]
        self.Sigma_inv[a] += (1.0 / sigma_sq) * (phi @ phi.T)

        # Moment update: b_a ← b_a + σ^{-2} rφ
        self.b[a] += (1.0 / sigma_sq) * (features * reward)

        # Update posterior mean via linear solve: θ̂_a = Σ_a b_a with Σ_a = Σ_a^{-1}^{-1}
        # Avoid forming Σ_a explicitly; solve Σ_a^{-1} θ̂_a = b_a instead.
        self.theta_hat[a] = np.linalg.solve(self.Sigma_inv[a], self.b[a])

        # Update Cholesky factor (incremental update for efficiency)
        if self.config.use_cholesky and self.cholesky_factors is not None:
            self.cholesky_factors[a] = np.linalg.cholesky(self.Sigma_inv[a])

        # Track selection count
        self.n_samples[a] += 1

    def get_diagnostics(self) -> Dict[str, NDArray[np.float64]]:
        """Return diagnostic information for monitoring.

        Returns:
            diagnostics: Dictionary with keys:
                - 'selection_counts': Array (M,) with selection counts
                - 'selection_frequencies': Array (M,) with selection frequencies
                - 'theta_norms': Array (M,) with ||θ̂_a||_2 for each template
                - 'uncertainty': Array (M,) with trace(Σ_a) (total uncertainty)
        """
        total_samples = self.n_samples.sum()
        selection_freqs = (
            self.n_samples / total_samples if total_samples > 0 else self.n_samples
        )
        theta_norms = np.linalg.norm(self.theta_hat, axis=1)
        uncertainties = np.array(
            [np.trace(np.linalg.inv(self.Sigma_inv[a])) for a in range(self.M)]
        )

        return {
            "selection_counts": self.n_samples.copy(),
            "selection_frequencies": selection_freqs,
            "theta_norms": theta_norms,
            "uncertainty": uncertainties,
        }


# Code ↔ Algorithm mapping
# - Line 126-142: select_action() implements [ALG-6.1] steps 3-4 (posterior sampling)
# - Line 144-180: update() implements [ALG-6.1] step 6 (Bayesian update via [EQ-6.8])
# - Line 97-109: Initialization sets Gaussian prior θ_a ~ N(0, λ^{-1}I)
# - Line 111-118: Optional Cholesky precomputation for numerical stability
#
# File: zoosim/policies/thompson_sampling.py
