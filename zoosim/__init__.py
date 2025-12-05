"""Simulation package for RL-optimized search boosting.

The package mirrors the book outline:

- :mod:`zoosim.core` → simulator configuration (Chapter 4)
- :mod:`zoosim.world` → catalog/users/queries (Chapter 4)
- :mod:`zoosim.ranking` → relevance + boost features (Chapter 5)
- :mod:`zoosim.dynamics` → behavior + rewards (Chapters 2 & 5)
- :mod:`zoosim.envs` → single-step environments (Chapters 4–5)
- :mod:`zoosim.multi_episode` → retention + inter-session MDP (Chapter 11)
- :mod:`zoosim.policies`, :mod:`zoosim.optimizers` → policy+optimizer scaffolding (Chapters 6–8)
- :mod:`zoosim.evaluation`, :mod:`zoosim.monitoring` → Part III stubs (Chapters 9–10)

Legacy imports (``from zoosim import config`` etc.) continue to work via
the re-exports below so existing labs/tests stay stable.
"""

from .core import config
from .dynamics import behavior, reward
from .envs import GymZooplusEnv, ZooplusSearchEnv
from .multi_episode import MultiSessionEnv, SessionMDPState, return_probability, sample_return
from .ranking import features, relevance
from .world import catalog, queries, users

__all__ = [
    "config",
    "catalog",
    "queries",
    "users",
    "features",
    "relevance",
    "behavior",
    "reward",
    "ZooplusSearchEnv",
    "GymZooplusEnv",
    "MultiSessionEnv",
    "SessionMDPState",
    "return_probability",
    "sample_return",
]
