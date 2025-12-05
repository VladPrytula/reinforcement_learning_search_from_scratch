#!/usr/bin/env python3
"""Utility script for Exercise 8.4 (docs/book/drafts/ch08/exercises_labs.md).

Computes the discounted algebraic Riccati solution for the 1D system
    s_{t+1} = s_t + a_t,  r_t = -(s_t^2 + a_t^2)
and prints the resulting gain K* and value coefficient P.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
from scipy.linalg import solve_discrete_are


def discounted_lqr_gain(gamma: float) -> Tuple[float, float]:
    """Return (P, K*) for the discounted 1D LQR used in Exercise 8.4."""
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0, 1], received {gamma}")

    a = np.array([[1.0]])
    b = np.array([[1.0]])
    q = np.array([[1.0]])
    r = np.array([[1.0]])

    a_bar = np.sqrt(gamma) * a
    b_bar = np.sqrt(gamma) * b
    p = solve_discrete_are(a_bar, b_bar, q, r)

    gain = -(gamma * b.T @ p @ a) / (r + gamma * b.T @ p @ b)
    return float(p.squeeze()), float(gain.squeeze())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for the Riccati solution (default: 0.99)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_value, gain = discounted_lqr_gain(args.gamma)
    print(f"gamma={args.gamma:.4f} -> P={p_value:.6f}, K*={gain:.6f}")


if __name__ == "__main__":
    main()
