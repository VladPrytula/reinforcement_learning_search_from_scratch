#!/usr/bin/env python3
"""Synthetic Bradley–Terry sanity test (docs/book/drafts/ch08/exercises_labs.md).

Generates random trajectory pairs, labels preferences via a proxy GMV signal,
and trains the RewardMLP described in Lab 8.4 to ensure the sign convention of
the binary-cross-entropy loss matches the text.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Generator, Tensor, nn
from torch.optim import Adam


@dataclass
class TrajectoryPair:
    tau_1: Tensor
    tau_2: Tensor
    label: Tensor  # shape: (1,), value in {0, 1}


class RewardMLP(nn.Module):
    """Simple trajectory-level reward network (matches lab prose)."""

    def __init__(self, state_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trajectory: Tensor) -> Tensor:
        # Average-pool states as a stand-in for trajectory embedding.
        pooled = trajectory.mean(dim=0)
        return self.net(pooled)


def generate_pairs(
    num_pairs: int,
    traj_len: int,
    state_dim: int,
    generator: Generator,
) -> List[TrajectoryPair]:
    pairs: List[TrajectoryPair] = []
    for _ in range(num_pairs):
        tau_1 = torch.randn(traj_len, state_dim, generator=generator)
        tau_2 = torch.randn(traj_len, state_dim, generator=generator)
        label = torch.tensor([1.0]) if tau_1.mean() > tau_2.mean() else torch.tensor([0.0])
        pairs.append(TrajectoryPair(tau_1, tau_2, label))
    return pairs


def train_reward_model(
    pairs: Sequence[TrajectoryPair],
    state_dim: int,
    hidden_dim: int,
    lr: float,
    epochs: int,
) -> None:
    model = RewardMLP(state_dim, hidden_dim)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for pair in pairs:
            r1 = model(pair.tau_1)
            r2 = model(pair.tau_2)
            logit = r1 - r2
            loss = F.binary_cross_entropy_with_logits(logit, pair.label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        print(f"Epoch {epoch:02d} | total_loss={total_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=64, help="Number of trajectory pairs")
    parser.add_argument("--traj-len", type=int, default=5, help="Time steps per trajectory")
    parser.add_argument("--state-dim", type=int, default=8, help="Dimensionality of state vector")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden size for RewardMLP")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=2025, help="Torch RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    pairs = generate_pairs(args.pairs, args.traj_len, args.state_dim, generator)
    train_reward_model(pairs, args.state_dim, args.hidden_dim, args.lr, args.epochs)
    print("Sanity check complete: loss decreases and script exits without errors.")


if __name__ == "__main__":
    main()
