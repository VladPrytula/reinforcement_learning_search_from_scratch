"""
Chapter 0 — Toy Problem (Regression Tests)

These tests enforce the “theorem must compile” contract for the book’s first
end-to-end experiment: the Chapter 0 toy bandit must remain reproducible under
a fixed seed, with stable representative output.
"""

from __future__ import annotations

import numpy as np

from scripts.ch00.toy_problem_solution import (
    CHAPTER0_ACTIONS,
    CHAPTER0_CONTEXTS,
    train_chapter0_q_table,
)


def test_chapter0_reproducible_policy_and_summary():
    q_table, history = train_chapter0_q_table(seed=42, n_train=3000, eps=0.1, learning_rate=0.1)

    final_avg = float(np.mean(history[-100:]))
    assert final_avg == pytest_approx(1.640, abs=1e-3)

    learned = {ctx: max(CHAPTER0_ACTIONS, key=lambda a: q_table[ctx][a]) for ctx in CHAPTER0_CONTEXTS}
    assert learned == {"price_hunter": (4, 1), "premium": (1, 4), "bulk_buyer": (2, 2)}

    q_vals = {ctx: q_table[ctx][learned[ctx]] for ctx in CHAPTER0_CONTEXTS}
    assert q_vals["price_hunter"] == pytest_approx(1.948, abs=1e-3)
    assert q_vals["premium"] == pytest_approx(2.289, abs=1e-3)
    assert q_vals["bulk_buyer"] == pytest_approx(0.942, abs=1e-3)


def pytest_approx(value: float, *, abs: float):  # noqa: A002
    """Local helper to avoid importing pytest at module import time."""
    import pytest

    return pytest.approx(value, abs=abs)

