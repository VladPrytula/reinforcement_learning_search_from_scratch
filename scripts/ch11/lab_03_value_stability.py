#!/usr/bin/env python
"""Lab 11.3 — Value estimation stability across seeds.
 .venv/bin/python -m scripts.ch11.lab_03_value_stability \
    --n-users 600 \
    --gamma 0.95 \
    --seeds 11 13 17 19 23 29 31 37 41 43 \
    | tee docs/book/ch11/data/lab11_value_stability_run1.txt
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from zoosim.multi_episode.session_env import MultiSessionEnv, SessionMDPState

from .utils.policies import uniform_policy
from .utils.trajectory import collect_trajectory


def estimate_policy_value(
    env_seed: int,
    n_users: int,
    gamma: float,
) -> float:
    env = MultiSessionEnv(seed=env_seed)
    values: List[float] = []
    policy = uniform_policy(action_dim=env.cfg.action.feature_dim)

    progress_step = max(1, n_users // 10)

    for idx in range(n_users):
        traj = collect_trajectory(env, policy, gamma=gamma)
        values.append(traj.discounted_value)

        if (idx + 1) % progress_step == 0:
            print(
                f"[Lab 11.3]     Seed {env_seed}: "
                f"completed {idx + 1}/{n_users} users"
            )

    return float(np.mean(values))


def stability_analysis(
    seeds: List[int],
    n_users: int,
    gamma: float,
) -> Dict[str, float]:
    estimates: List[float] = []
    total = len(seeds)
    for idx, s in enumerate(seeds, start=1):
        print(
            f"[Lab 11.3] Estimating value for seed {s} "
            f"({idx}/{total}, n_users={n_users}, gamma={gamma})..."
        )
        v = estimate_policy_value(s, n_users, gamma)
        print(f"[Lab 11.3]   Seed {s} estimate: {v:.4f}")
        estimates.append(v)

    arr = np.asarray(estimates, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = float(std / mean) if mean != 0.0 else 0.0
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    return {"mean": mean, "std": std, "cv": cv, "ci_low": ci_low, "ci_high": ci_high}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab 11.3 — Value estimation stability across seeds.",
    )
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
    )
    args = parser.parse_args()

    stats = stability_analysis(args.seeds, args.n_users, args.gamma)
    print("Value estimation statistics:")
    for k in ["mean", "std", "cv", "ci_low", "ci_high"]:
        print(f"  {k}: {stats[k]:.4f}")

    target_cv = 0.10
    print(f"\nTarget CV < {target_cv:.2f}")
    if stats["cv"] < target_cv:
        print("ACCEPTED: value stability criterion met.")
    else:
        print("WARNING: value CV above target; consider increasing n_users or adjusting policy.")


if __name__ == "__main__":
    main()
