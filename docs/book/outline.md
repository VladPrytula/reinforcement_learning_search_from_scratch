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

## Appendices

Foundational mathematics supporting multiple chapters. Read as needed based on background.

| App | Title | Main File | Dependencies |
|-----|-------|-----------|--------------|
| A | [Bayesian Preference Models](appendix_a_bayesian_preference_models.md) | Hierarchical priors, shrinkage, bandit integration | Ch06 |
| B | [Control-Theoretic Background](appendix_b_control_theory.md) | LQR, HJB, deep RL timeline | Ch01, Ch03 |
| C | [Convex Optimization for Constrained MDPs](appendix_c_convex_optimization.md) | Lagrangian duality, Slater's condition | Ch01 §1.9; Ch10 (guardrails context); Ch14 (primal--dual constrained RL) |
| D | [Information-Theoretic Lower Bounds](appendix_d_information_theory.md) | KL divergence, Fano's inequality, bandit lower bounds | Ch01 §1.7.6; Ch06 (THM-6.0) |
| E | [Vector-Reward Multi-Objective RL](appendix_e_vector_morl.md) | Pareto Q-learning, coverage sets, supported vs unsupported points | Ch14 (multi-objective context) |

**When to read:**
- **Appendix A**: When implementing Thompson Sampling or LinUCB with rich features (Chapter 6), or when modeling user preferences in the simulator (Chapter 4)
- **Appendix B**: If you have control theory background (LQR, HJB) and want to see connections to RL, or when control-theoretic tools appear in Chapters 8, 10, and 11
- **Appendix C**: For the Lagrangian-duality foundations used in Chapter 1 (§1.9), for the conceptual background behind constraints and guardrails in Chapter 10, and before implementing primal--dual constrained RL in Chapter 14
- **Appendix D**: When you want to understand why $\Omega(\sqrt{KT})$ regret is unavoidable for bandits, or when Chapter 6 references the minimax lower bound
- **Appendix E**: When Chapter 14's "multi-objective" framing raises questions about true vector-reward MORL, Pareto Q-learning, or when CMDP/$\varepsilon$-constraint is insufficient

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
