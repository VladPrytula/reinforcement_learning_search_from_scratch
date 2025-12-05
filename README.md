# RL Search From Scratch

**Author:** Vlad Prytula  
**Status:** Active Development (Chapters 0–7 Complete)

## Project Overview

This repository accompanies a textbook-in-progress that builds a reinforcement learning system for e-commerce search ranking.

Our goal is to bridge the gap between rigorous mathematics and production engineering. Too often, these worlds exist in isolation: theory papers often skip implementation details, while engineering guides may overlook the foundational math. This project aims to do both, treating them as inseparable parts of the same discipline.

We build the system on three core principles:

1.  **Foundational Math**: We derive algorithms from first principles (measure theory, Bellman operators) to understand *why* they work.
2.  **Production-Quality Code**: We avoid "research scripts" in favor of structured, typed, and tested Python/PyTorch code that mirrors real-world systems.
3.  **Realistic Constraints**: We move beyond gridworlds to tackle the messy reality of search: optimizing GMV, managing user retention, and respecting business constraints.

---

## Project Status

We are currently working through **Part III (Continuous Control)**.

| Part | Focus | Chapters | Status |
| :--- | :--- | :--- | :--- |
| **I** | **Foundations** | 0–3 | ✅ **Complete** (Measure theory, PBMs, Bellman operators) |
| **II** | **The Simulator** | 4–6 | ✅ **Complete** (Generative world, discrete bandits, LinUCB) |
| **III** | **Continuous Control** | 7 | ✅ **Complete** (Q-Ensembles, CEM, Trust Regions) |
| **IV** | **Safety & Scale** | 8–10 | 🚧 *In Progress* (Constrained MDPs, OPE, Drift) |
| **V** | **Frontier** | 11–15 | 📅 *Planned* (Slate RL, Offline RL, Differentiable Ranking) |

---

## Quick Start

**Prerequisites:** Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# 1. Clone and enter the repo
git clone https://github.com/VladPrytula/reinforcement_learning_search_from_scratch
cd reinforcement_learning_search_from_scratch

# 2. Install dependencies (uv handles the virtual environment)
uv sync

# 3. Verify setup
uv run python scripts/ch00/toy_problem_solution.py
```

**That's it.** You should see Q-learning convergence output.

### Browse the Textbook

```bash
uv run mkdocs serve
# Open http://127.0.0.1:8000
```

### Run Tests

```bash
uv run pytest                    # All tests
uv run pytest -m "not slow"      # Skip slow integration tests
```

### Run Chapter Labs

```bash
uv run python scripts/ch00/toy_problem_solution.py   # Chapter 0: Toy Q-learning
uv run python scripts/ch01/lab_solutions.py --all    # Chapter 1: Reward aggregation
uv run python -m scripts.ch06.lab_solutions          # Chapter 6: Bandits (menu)
```

---

## Project Setup (Detailed)

We use [uv](https://docs.astral.sh/uv/) for dependency management. It handles virtual environments automatically.

| Step | Command | Notes |
|------|---------|-------|
| 1. Install uv | See [uv installation](https://docs.astral.sh/uv/getting-started/installation/) | One-time setup |
| 2. Install deps | `uv sync` | Creates `.venv/` and installs everything |
| 3. Run commands | `uv run <command>` | Activates venv automatically |

The simulator uses **NumPy** for scalar sampling and **PyTorch** for embeddings and neural policies.

**Verify in Python REPL:**
```bash
uv run python
```
```python
import numpy as np
from zoosim.core import config
from zoosim.world import catalog

cfg = config.load_default_config()
rng = np.random.default_rng(42)
products = catalog.generate_catalog(cfg.catalog, rng)
print(f"Generated {len(products)} products with embedding dim {products[0].embedding.shape}")
```

---

## Visualizations & Testing

- **Visualizations:** Uses pandas, matplotlib, and seaborn (included in deps). See Chapter 5 drafts for plotting code.
- **Testing:** Run `uv run pytest` to execute the regression suite.
    - **Chapter 6/7 Experiments:** Some tests run full simulations and are marked `@pytest.mark.slow`.
    - **Skip slow tests:** `uv run pytest -m "not slow"`
    - **Run specific demo:** `uv run pytest tests/ch06/test_feature_modes_integration.py -m slow -s`

---

## Gymnasium Wrapper

For standard RL loops, we provide a Gymnasium-compatible wrapper:

```python
from zoosim.envs import GymZooplusEnv
env = GymZooplusEnv()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```