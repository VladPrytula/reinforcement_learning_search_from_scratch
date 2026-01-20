"""Numerical regression tests for Chapter 3's GridWorld value iteration.

The implementation mirrors the listing in §3.9.1 and checks the contraction
bound from #EQ-3.18 as well as the convergence guarantees from [THM-3.7.1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class GridWorldConfig:
    size: int = 5
    gamma: float = 0.9
    goal_reward: float = 10.0


class GridWorldMDP:
    """Deterministic GridWorld used in Chapter 3's convergence lab."""

    def __init__(self, cfg: GridWorldConfig | None = None) -> None:
        self.cfg = cfg or GridWorldConfig()
        self.size = self.cfg.size
        self.gamma = self.cfg.gamma
        self.goal = (self.size - 1, self.size - 1)
        self.goal_reward = self.cfg.goal_reward
        self.n_states = self.size * self.size
        self.n_actions = 4  # up, down, left, right

        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.r = np.zeros((self.n_states, self.n_actions))

        for i in range(self.size):
            for j in range(self.size):
                s = self._state_index(i, j)
                if (i, j) == self.goal:
                    for a in range(self.n_actions):
                        self.P[s, a, s] = 1.0
                        self.r[s, a] = self.goal_reward
                    continue
                for a in range(self.n_actions):
                    i_next, j_next = self._next_state(i, j, a)
                    s_next = self._state_index(i_next, j_next)
                    self.P[s, a, s_next] = 1.0
                    self.r[s, a] = -1.0

    def _state_index(self, i: int, j: int) -> int:
        return i * self.size + j

    def _next_state(self, i: int, j: int, action: int) -> Tuple[int, int]:
        if action == 0:  # up
            return max(i - 1, 0), j
        if action == 1:  # down
            return min(i + 1, self.size - 1), j
        if action == 2:  # left
            return i, max(j - 1, 0)
        return i, min(j + 1, self.size - 1)  # right

    def bellman_operator(self, values: np.ndarray) -> np.ndarray:
        q_values = self.r + self.gamma * np.einsum("ijk,k->ij", self.P, values)
        return np.max(q_values, axis=1)

    def value_iteration(
        self,
        V_init: np.ndarray | None = None,
        *,
        max_iter: int = 256,
        tol: float = 1e-8,
    ) -> Tuple[np.ndarray, list[float]]:
        values = np.zeros(self.n_states) if V_init is None else V_init.copy()
        errors: list[float] = []
        for _ in range(max_iter):
            updated = self.bellman_operator(values)
            error = float(np.max(np.abs(updated - values)))
            errors.append(error)
            values = updated
            if error < tol:
                break
        return values, errors


def test_value_iteration_converges_and_rewards_goal() -> None:
    """Regression test for the deterministic GridWorld convergence listing."""

    mdp = GridWorldMDP()
    V_star, errors = mdp.value_iteration(tol=1e-10)

    assert errors[-1] < 1e-8, "Value iteration failed to converge within tolerance"

    start_state = mdp._state_index(0, 0)
    goal_state = mdp._state_index(*mdp.goal)
    # Start state should learn a substantial positive value while the goal state
    # approaches the geometric sum goal_reward / (1 - gamma) because it is absorbing.
    expected_goal = mdp.goal_reward / (1.0 - mdp.gamma)
    assert V_star[start_state] > 30.0
    assert abs(V_star[goal_state] - expected_goal) < 1e-5


def test_convergence_respects_gamma_bound() -> None:
    """Verify the contraction inequality from #EQ-3.18 numerically."""

    mdp = GridWorldMDP()
    V_init = np.zeros(mdp.n_states)
    _, errors = mdp.value_iteration(V_init, tol=1e-12)

    initial_gap = np.max(np.abs(mdp.bellman_operator(V_init) - V_init))
    for k, err in enumerate(errors[:25]):
        bound = (mdp.gamma**k / (1.0 - mdp.gamma)) * initial_gap
        assert err <= bound + 1e-9, "Empirical error exceeded theoretical bound"

    ratios = [
        errors[k] / errors[k - 1]
        for k in range(1, min(len(errors), 20))
        if errors[k - 1] > 1e-12
    ]
    tail = ratios[-8:]
    assert all(abs(r - mdp.gamma) < 0.05 for r in tail), "Error ratios did not approach gamma"
