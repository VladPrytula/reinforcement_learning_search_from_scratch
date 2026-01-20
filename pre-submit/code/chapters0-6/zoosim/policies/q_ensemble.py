"""Q-ensemble regression for continuous boost actions.

Mathematical basis:
- Chapter 7 — Continuous Actions via Q(x,a) Regression (CH-7)
- Q-function approximation with ensemble uncertainty for UCB-style selection

This module implements a small neural network ensemble that approximates
the state–action value function

    Q(x, a) ≈ E[R | x, a]

for the Zoosim search environment, where:
- x are context/features (user, query, product aggregates)
- a are continuous boost weights applied to engineered features

The ensemble provides both a mean prediction and an uncertainty estimate
via the empirical standard deviation across members. This uncertainty can
then be used with an optimizer (e.g., CEM in zoosim.optimizers.cem) to
select actions using either greedy or UCB-style objectives.

Code ↔ Chapter
- CH-7: Continuous Actions via Q(x,a) Regression
- This file: zoosim/policies/q_ensemble.py
- Optimizer: zoosim/optimizers/cem.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.optim import Adam

from zoosim.optimizers.cem import CEMConfig, cem_optimize


@dataclass
class QEnsembleConfig:
    """Configuration for Q-ensemble regressor.

    Attributes:
        n_ensembles: Number of neural networks in the ensemble.
        hidden_sizes: Sizes of hidden layers for each MLP.
        learning_rate: Adam learning rate for all ensemble members.
        weight_decay: Optional L2 weight decay.
        device: Torch device string (e.g., "cpu", "cuda").
        seed: Base random seed for network initialization.
    """

    n_ensembles: int = 5
    hidden_sizes: Sequence[int] = (64, 64)
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 42


class QEnsembleRegressor:
    """Ensemble of MLPs approximating Q(x,a) with uncertainty.

    Each ensemble member is an MLP taking concatenated [x, a] as input and
    outputting a scalar reward prediction. The ensemble mean and standard
    deviation provide an approximate predictive distribution over rewards.

    Typical usage:
        q = QEnsembleRegressor(state_dim=d_x, action_dim=d_a)
        q.update_batch(states, actions, rewards, n_epochs=10)
        mean, std = q.predict(states_eval, actions_eval)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[QEnsembleConfig] = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.config = config or QEnsembleConfig()

        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive integers.")

        # Torch device and RNG setup
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)

        input_dim = self.state_dim + self.action_dim
        self._models: List[nn.Module] = []
        self._optimizers: List[Adam] = []

        for idx in range(self.config.n_ensembles):
            # Use different seeds per ensemble member for diversity.
            torch.manual_seed(self.config.seed + idx)
            model = self._build_mlp(input_dim, self.config.hidden_sizes, 1).to(
                self.device
            )
            optimizer = Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self._models.append(model)
            self._optimizers.append(optimizer)

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
    ) -> nn.Module:
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _to_tensor(
        self,
        array: NDArray[np.float64] | NDArray[np.float32],
    ) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def update_batch(
        self,
        states: NDArray[np.float64] | NDArray[np.float32],
        actions: NDArray[np.float64] | NDArray[np.float32],
        rewards: NDArray[np.float64] | NDArray[np.float32],
        n_epochs: int = 1,
        batch_size: int = 128,
    ) -> None:
        """Train ensemble on a batch of transitions.

        Args:
            states: Array of shape (N, state_dim)
            actions: Array of shape (N, action_dim)
            rewards: Array of shape (N,) or (N, 1)
            n_epochs: Number of passes over the batch.
            batch_size: Mini-batch size for SGD.
        """
        states_arr = np.asarray(states, dtype=np.float32)
        actions_arr = np.asarray(actions, dtype=np.float32)
        rewards_arr = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)

        if states_arr.shape[0] != actions_arr.shape[0] or states_arr.shape[0] != rewards_arr.shape[0]:
            raise ValueError("states, actions, and rewards must have the same length.")

        if states_arr.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected states with dimension {self.state_dim}, "
                f"got {states_arr.shape[1]}."
            )
        if actions_arr.shape[1] != self.action_dim:
            raise ValueError(
                f"Expected actions with dimension {self.action_dim}, "
                f"got {actions_arr.shape[1]}."
            )

        dataset_size = states_arr.shape[0]
        if dataset_size == 0:
            return

        states_t = self._to_tensor(states_arr)
        actions_t = self._to_tensor(actions_arr)
        rewards_t = self._to_tensor(rewards_arr)

        for _ in range(max(1, n_epochs)):
            permutation = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, batch_size):
                idx = permutation[start : start + batch_size]
                x_batch = torch.cat(
                    [states_t[idx], actions_t[idx]],
                    dim=1,
                )
                y_batch = rewards_t[idx]

                for model, optimizer in zip(self._models, self._optimizers):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    preds = model(x_batch)
                    loss = nn.functional.mse_loss(preds, y_batch)
                    loss.backward()
                    optimizer.step()

    def predict(
        self,
        states: NDArray[np.float64] | NDArray[np.float32],
        actions: NDArray[np.float64] | NDArray[np.float32],
        return_individual: bool = False,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]] | Tuple[
        NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
    ]:
        """Predict mean and ensemble uncertainty for batch of (x,a).

        Args:
            states: Array of shape (N, state_dim) or (state_dim,)
            actions: Array of shape (N, action_dim) or (action_dim,)
            return_individual: If True, also return per-ensemble predictions
                               of shape (n_ensembles, N).

        Returns:
            mean: Mean prediction over ensemble, shape (N,)
            std: Standard deviation over ensemble, shape (N,)
            individual (optional): Per-ensemble predictions, shape (n_ensembles, N)
        """
        states_arr = np.asarray(states, dtype=np.float32)
        actions_arr = np.asarray(actions, dtype=np.float32)

        if states_arr.ndim == 1:
            states_arr = states_arr[None, :]
        if actions_arr.ndim == 1:
            actions_arr = actions_arr[None, :]

        if states_arr.shape[0] != actions_arr.shape[0]:
            raise ValueError("states and actions must have same batch size.")

        if states_arr.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected states with dimension {self.state_dim}, "
                f"got {states_arr.shape[1]}."
            )
        if actions_arr.shape[1] != self.action_dim:
            raise ValueError(
                f"Expected actions with dimension {self.action_dim}, "
                f"got {actions_arr.shape[1]}."
            )

        states_t = self._to_tensor(states_arr)
        actions_t = self._to_tensor(actions_arr)

        with torch.no_grad():
            x = torch.cat([states_t, actions_t], dim=1)
            preds_list: List[torch.Tensor] = []
            for model in self._models:
                model.eval()
                preds = model(x).squeeze(-1)
                preds_list.append(preds)

            preds_stack = torch.stack(preds_list, dim=0)  # (n_ensembles, N)
            mean = preds_stack.mean(dim=0).cpu().numpy().astype(np.float32)
            std = preds_stack.std(dim=0, unbiased=False).cpu().numpy().astype(
                np.float32
            )

        if return_individual:
            individual = preds_stack.cpu().numpy().astype(np.float32)
            return individual, mean, std
        return mean, std


class QEnsemblePolicy:
    """Continuous-action policy using Q-ensemble + CEM.

    This is a thin wrapper that:
    - Uses QEnsembleRegressor to model Q(x,a)
    - Uses CEM to approximately solve argmax_a Q(x,a) or argmax_a Q(x,a) + βσ(x,a)

    It does not implement a full training loop or replay buffer; callers are
    expected to accumulate (x,a,r) data and train the underlying regressor.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        a_max: float,
        q_config: Optional[QEnsembleConfig] = None,
        cem_config: Optional[CEMConfig] = None,
        ucb_beta: float = 0.0,
    ) -> None:
        if a_max <= 0:
            raise ValueError("a_max must be positive.")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.a_max = float(a_max)
        self.ucb_beta = float(ucb_beta)

        self.q = QEnsembleRegressor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=q_config,
        )
        self.cem_config = cem_config or CEMConfig(a_max=self.a_max)

    def select_action(
        self,
        state: NDArray[np.float64] | NDArray[np.float32],
        trust_region_center: Optional[NDArray[np.float64]] = None,
        trust_region_radius: Optional[float] = None,
    ) -> NDArray[np.float32]:
        """Select continuous action using CEM over Q-ensemble.

        Args:
            state: Context vector x, shape (state_dim,)
            trust_region_center: Optional previous action a_prev for trust region.
            trust_region_radius: Optional trust region radius in ℓ2 norm.

        Returns:
            action: Selected action vector, shape (action_dim,)
        """
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.ndim != 1 or state_arr.shape[0] != self.state_dim:
            raise ValueError(
                f"Expected state of shape ({self.state_dim},), got {state_arr.shape}."
            )

        def objective(actions: NDArray[np.float64]) -> NDArray[np.float64]:
            # actions: (N, action_dim)
            batch_size = actions.shape[0]
            states_batch = np.repeat(state_arr[None, :], batch_size, axis=0)
            mean, std = self.q.predict(states_batch, actions.astype(np.float32))
            return mean + self.ucb_beta * std

        best_action, _ = cem_optimize(
            objective=objective,
            action_dim=self.action_dim,
            config=self.cem_config,
            init_mean=trust_region_center,
            trust_region_center=trust_region_center,
            trust_region_radius=trust_region_radius,
        )
        return best_action.astype(np.float32)

