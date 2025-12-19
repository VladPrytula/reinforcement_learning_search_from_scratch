# RL Search From Scratch

[![Read the Book](https://img.shields.io/badge/Read%20the%20Book-Online-brightgreen?style=for-the-badge)](https://vladprytula.github.io/reinforcement_learning_search_from_scratch/)

[![Made with Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Powered by PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5?logo=openaigym&logoColor=white)](https://gymnasium.farama.org/)
[![MkDocs](https://img.shields.io/badge/MkDocs-526CFE?logo=materialformkdocs&logoColor=white)](https://www.mkdocs.org/)
[![LaTeX](https://img.shields.io/badge/LaTeX-008080?logo=latex&logoColor=white)](https://www.latex-project.org/)

**Author:** Vlad Prytula
**Status:** Work in Progress (Chapters 0--10 drafted, Part V in development)

**Syllabus:** See the GitHub-facing study roadmap in [`docs/book/syllabus_github.md`](docs/book/syllabus_github.md).

## Project Overview

This repository accompanies a textbook-in-progress that builds a reinforcement learning system for e-commerce search ranking.

The project grew from years of building personalized search and recommendation systems at one of Europe's largest e-commerce platforms, combined with a background in applied mathematics (functional analysis, measure theory). I kept wishing for a resource that held both standards simultaneously—rigorous foundations *and* production-quality implementations—so I started writing one.

The goal is to treat mathematics and engineering as inseparable parts of the same discipline:

1.  **Foundational Math**: We derive algorithms from first principles (measure theory, Bellman operators) to understand *why* they work.
2.  **Clean, Modular Code**: Readable, reproducible, and maintainable Python/PyTorch---not a pile of scripts. The architecture is designed so the same blueprint can realistically scale to production.
3.  **Realistic Constraints**: We move beyond gridworlds to tackle the messy reality of search: optimizing GMV, managing user retention, and respecting business constraints.

I wrote it for the version of myself who was searching library shelves for exactly this—and I hope it proves useful to anyone else who's felt that same gap between textbook algorithms and production systems.

---

## The Journey: From Toy Bandits to Frontier RL

By the final chapter, you'll have built and understood every major RL paradigm---and seen them compete head-to-head on the same e-commerce search task:

```
Foundations          Bandits              Value-Based           Policy Gradient        Frontier
───────────────────────────────────────────────────────────────────────────────────────────────────
Measure Theory   →   LinUCB           →   Q-Ensemble + CEM  →   REINFORCE          →   SlateQ
Bellman Ops          Thompson Sampling    Continuous Actions     Neural PG              CQL (Offline)
Click Models         Template Selection   Trust Regions          Variance Reduction     Differentiable Ranking
```

**Where these techniques appear in the wild:**
- **Bandits**: Netflix artwork personalization ([RecSys 2018](https://dl.acm.org/doi/10.1145/3240323.3241729)), Spotify's BaRT shelf ranking ([engineering blog](https://www.music-tomorrow.com/blog/how-spotify-recommendation-system-works-complete-guide)), LinkedIn feed optimization ([Netflix ML Meetup](https://netflixtechblog.com/ml-platform-meetup-infra-for-contextual-bandits-and-reinforcement-learning-4a90305948ef))
- **Value-Based RL**: Alibaba real-time bidding ([KDD 2018](https://arxiv.org/abs/1803.00259)), Google data center cooling ([DeepMind 2016](https://deepmind.google/discover/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/))
- **Policy Gradients**: RLHF for LLMs ([InstructGPT/PPO](https://huggingface.co/blog/rlhf)), YouTube recommendations ([WSDM 2019, REINFORCE](https://arxiv.org/abs/1812.02353)), chip placement ([AlphaChip/Nature 2021](https://deepmind.google/blog/how-alphachip-transformed-computer-chip-design/)), robotics control (PPO/SAC are standard)
- **Offline RL**: Learning from logs without risky online exploration---critical for healthcare treatment planning, autonomous driving, and any setting where mistakes are expensive ([survey](https://arxiv.org/abs/2005.01643))
- **Slate RL & Differentiable Ranking**: YouTube ([SlateQ, IJCAI 2019](https://research.google/pubs/pub48200/)), Amazon page layout optimization ([Amazon Science](https://www.amazon.science/publications/automate-page-layout-optimization-an-offline-deep-q-learning-approach))

The `scripts/part4_global_comparison.py` "Battle Royale" runs all algorithms on `zoosim`, our realistic e-commerce simulator with position bias, multi-objective rewards, and business constraints (CM2 floors, exposure fairness).

**What you'll learn along the way:**
- Why a theoretically sound bandit can underperform a static baseline by 28%---and how fixing *features* (not algorithms) yields a 44-point GMV swing
- When continuous Q-learning beats discrete templates, and when it doesn't
- The honest 50% gap between REINFORCE and value methods (and why policy gradients still matter for LLMs)
- How to evaluate policies offline without running A/B tests (OPE: IPS, SNIPS, DR)
- Deploying RL safely with guardrails, drift detection, and automatic rollback

---

## Project Status

| Part | Focus | Chapters | Status |
| :--- | :--- | :--- | :--- |
| **I** | **Foundations** | 0--3 | 📝 Drafted (Motivation, measure theory, click models, Bellman operators) |
| **II** | **Simulator & Bandits** | 4--6 | 📝 Drafted (Generative world, features/rewards, LinUCB/Thompson Sampling) |
| **III** | **Continuous & Policy Gradients** | 7--8 | 📝 Drafted (Q-Ensemble + CEM, REINFORCE, theory-practice gaps) |
| **IV** | **Evaluation & Robustness** | 9--10 | 📝 Drafted (OPE: IPS/SNIPS/DR, drift detection, guardrails) |
| **V** | **Frontier** | 11--15 | 🚧 In Development (Multi-episode retention, Slate RL, Offline RL, Fairness) |

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
