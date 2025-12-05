# Book Outline

This is our map. Each chapter builds on the previous, but you can jump ahead if you're comfortable with the prerequisites.

For detailed objectives, labs, and acceptance criteria for each chapter, see [syllabus.md](syllabus.md).

---

## Part I — Foundations

The mathematical machinery you need before touching a single line of RL code.

| Ch | Title | Main File | Code |
|----|-------|-----------|------|
| 0 | [A Tiny Search Engine That Learns](ch00/ch00_motivation_first_experiment_revised.md) | Motivation and first experiment | `scripts/ch00/toy_problem_solution.py` |
| 1 | [Search Ranking as Optimization](ch01/ch01_foundations_revised_math+pedagogy_v3.md) | Reward design, contextual bandits, constraints | `zoosim/dynamics/reward.py` |
| 2 | [Probability, Measure, and Click Models](ch02/ch02_probability_measure_click_models.md) | PBM/DBN, position bias, stopping times | `zoosim/dynamics/behavior.py` |
| 3 | [Bellman Equations and Regret](ch03/ch03_bellman_and_regret.md) | Contractions, value functions, regret bounds | — |

---

## Part II — The Simulator

Where the math meets synthetic data. You'll build a complete search environment.

| Ch | Title | Main File | Code |
|----|-------|-----------|------|
| 4 | [Catalog, Users, Queries](ch04/ch04_generative_world_design.md) | Generative design with seeds | `zoosim/world/` |
| 5 | [Relevance, Features, Reward](ch05/ch05_relevance_features_reward.md) | Hybrid scoring, feature engineering | `zoosim/ranking/` |

---

## Part III — Policies

From discrete bandits to continuous optimization to policy gradients.

| Ch | Title | Main File | Code |
|----|-------|-----------|------|
| 6 | [Discrete Template Bandits](ch06/discrete_template_bandits.md) | LinUCB, Thompson Sampling | `zoosim/policies/` |
| 7 | [Continuous Actions via Q(x,a)](ch07/ch07_continuous_actions.md) | Value regression, uncertainty, trust regions | `zoosim/optimizers/` |
| 8 | [Policy Gradient Methods](ch08/chapter08_policy_gradients_complete.md) | REINFORCE, variance reduction | — |

---

## Part IV — Evaluation and Deployment

How to know if your policy works without breaking production.

| Ch | Title | Main File | Code |
|----|-------|-----------|------|
| 9 | [Off-Policy Evaluation](ch09/ch09_off_policy_evaluation.md) | IPS, SNIPS, DR, FQE | `zoosim/evaluation/` |
| 10 | Robustness and Guardrails | *(in progress)* | `zoosim/monitoring/` |
| 11 | Multi-Episode Retention | *(planned)* | `zoosim/multi_episode/` |

---

## Part V — Frontier Methods (Planned)

Where the field is heading. These chapters are on the roadmap.

- **Ch 12:** Slate RL & Differentiable Ranking
- **Ch 13:** Offline RL (CQL, IQL, TD3+BC)
- **Ch 14:** Multi-Objective RL & Fairness
- **Ch 15:** Non-Stationarity & Meta-Adaptation

---

## Quick Reference

**Each chapter folder contains:**
- Main content (`ch0X_*.md`)
- Exercises (`exercises_labs.md`)
- Lab solutions (`ch0X_lab_solutions.md`)
- Archive of earlier drafts (`archive/`)

**The simulator lives in `zoosim/`:**
- `core/` — Configuration
- `world/` — Catalog, users, queries
- `ranking/` — Relevance, features
- `dynamics/` — Click models, rewards
- `envs/` — Gymnasium interface
- `policies/` — Agents (bandits, Q-learning)
- `evaluation/` — OPE estimators
- `monitoring/` — Drift detection, guardrails

**To run tests:**
```bash
pytest -q
```

**To serve the book locally:**
```bash
mkdocs serve
```
