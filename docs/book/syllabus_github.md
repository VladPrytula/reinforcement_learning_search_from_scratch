# Book Syllabus — GitHub Edition

This syllabus is the public-facing roadmap for the **RL Search From Scratch** book and codebase. It gives you a chapter-by-chapter path from the first toy experiment to frontier methods for ranking and reinforcement learning, with pointers into the repository so you always know *where the math lives in the code*.

Use it as:
- A self-study plan (roughly 8–12 weeks of part‑time work)
- A course outline if you’re teaching from the book
- A map from chapters to code, labs, and tests in this repo

For internal authoring notes and more granular acceptance criteria, see `docs/book/syllabus.md`.

---

## Part I — Foundations

### Chapter 0 — Motivation: A Tiny Search Engine That Learns

- **Goal:** Run a complete small RL experiment (toy bandit with boosts) and build intuition for exploration, reward design, and regret—before any measure theory shows up.
- **Code & tests:** `scripts/ch00/toy_problem_solution.py`, `tests/ch00/test_toy_example.py`.
- **What you’ll do:** Plot learning curves, compare to simple baselines, add a basic CM2 floor, and reason about the geometry of the action space.
- **By the end:** You can describe why the toy bandit succeeds or fails and explain why we need more formal tools in later chapters.

### Chapter 1 — Search Ranking as Optimization

- **Goal:** Formalize multi-objective reward and cast the search problem as a contextual bandit, connecting the Chapter 0 toy to real ranking systems.
- **Theory:** Contextual bandits vs MDPs; constraints (CM2, exposure, rank stability); engagement as a viability condition; key equations like `{#EQ-1.2}` and the multi-episode preview `{#EQ-1.2-prime}`.
- **Code:** Reward weights in `zoosim/core/config.py`, aggregation in `zoosim/dynamics/reward.py`.
- **What you’ll do:** Compute reward under different weights, check CVR diagnostics, and verify wiring of rank-stability penalties.
- **By the end:** You can write and justify a multi-objective reward for search and understand how it flows into the simulator.

### Chapter 2 — Probability, Measure, and Click Models (PBM/DBN)

- **Goal:** Build a probabilistic model of examination, clicks, and purchases, and define position bias and abandonment formally.
- **Theory:** PBM/DBN abstractions; stopping times; propensities and unbiased estimators.
- **Code:** Click/exam/buy paths in `zoosim/dynamics/behavior.py`; position-bias knobs in `zoosim/core/config.py`.
- **What you’ll do:** Plot CTR vs rank by query type, simulate abandonment vs utility, and connect theory to simulator behavior.
- **By the end:** You can explain position bias mathematically and interpret simulated CTR/abandonment curves.

### Chapter 3 — Stochastic Processes & Bellman Foundations

- **Goal:** Bridge single-step bandits to MDPs/CMDPs, introduce Bellman operators and contractions, and set up the theoretical backbone for later chapters.
- **Theory:** Value functions; contraction mappings; Lagrangian treatment of constraints; preview of off-policy evaluation (OPE) as model-based decision making.
- **Code:** No major new modules—this chapter supports the rest of the book conceptually.
- **What you’ll do:** Prove or sketch key contraction results on a toy MDP and sanity-check them numerically.
- **By the end:** You’re comfortable with Bellman operators and know how they relate to the RL algorithms used later.

---

## Part II — Simulator and Data

### Chapter 4 — Catalog, Users, Queries (Generative Design)

- **Goal:** Build a deterministic, realistic world generator with controllable seeds and priors for catalogs, users, and queries.
- **Code:** `zoosim/world/{catalog,users,queries}.py`, configuration in `zoosim/core/config.py`, demo script `scripts/ch04/ch04_demo.py`.
- **What you’ll do:** Generate catalogs, compute price/margin statistics, and verify deterministic generation given a fixed random seed.
- **By the end:** You can regenerate the same synthetic universe on demand and understand its key distributional properties.

### Chapter 5 — Relevance, Features, Reward

- **Goal:** Define relevance, engineer features, and wire up the reward function used across the book.
- **Code:** `zoosim/ranking/{relevance,features}.py`, `zoosim/dynamics/reward.py`, configuration-driven constants, demo script `scripts/ch05/ch05_demo.py`.
- **What you’ll do:** Visualize feature interactions (e.g., CM2 × category, discount × price_sensitivity), inspect reward breakdowns, and relate them to business metrics.
- **By the end:** You can trace how a query–product pair turns into features, predictions, and a scalar reward, and you know how to validate this with the provided tests.

### Chapter 6 — Discrete Template Bandits (LinUCB / Thompson Sampling)

- **Goal:** Build safe, interpretable bandit baselines that operate over discrete ranking templates, and understand when they succeed or fail.
- **Code:** `policies/{templates,lin_ucb,thompson_sampling}.py`, demo script `scripts/ch06/template_bandits_demo.py`.
- **What you’ll do:** Compare simple vs rich context features, run per-segment diagnostics, and analyze template selection behavior.
- **By the end:** You can quantify how much lift you get from contextual bandits over strong static baselines and where the theory–practice gaps show up.

### Chapter 6A — Neural Bandits (Optional Bridge)

- **Goal:** Learn representations with neural networks and connect linear bandits (Chapter 6) to continuous Q(x, a) methods (Chapter 7).
- **Theory:** Neural Linear architecture (feature extractor \(f_\psi\) plus linear heads); discrete neural Q(x, a) on templates; sample complexity trade-offs.
- **Code:** Planned modules for neural linear and NL-bandit variants built on top of the bandit and feature infrastructure.
- **What you’ll do:** Experiment with learned features vs engineered ones, calibrate uncertainty estimates, and explore overfitting vs data efficiency.
- **By the end (optional):** You understand when neural bandits are worth the extra complexity and how they prepare you for continuous-action RL.

### Chapter 7 — Continuous Actions via Q(x, a)

- **Goal:** Move from discrete templates to continuous boosts, learning Q(x, a) and optimizing bounded actions under uncertainty and trust regions.
- **Code:** `policies/q_ensemble.py`, `optimizers/cem.py`.
- **What you’ll do:** Compare UCB vs greedy selection, tune trust-region parameters (e.g., Δrank@k constraints), and study the trade-off between exploration and stability.
- **By the end:** You can train continuous-action Q-ensembles that outperform strong discrete baselines while respecting guardrails.

### Chapter 8 — Policy Gradient Methods (REINFORCE)

- **Goal:** Derive the Policy Gradient Theorem, implement REINFORCE with Gaussian policies, and compare policy gradients to value-based methods.
- **Code:** `zoosim/policies/reinforce.py`, training and visualization scripts in `scripts/ch08/{reinforce_demo.py,neural_reinforce_demo.py,visualize_theory_practice_gap.py}`.
- **What you’ll do:** Reproduce REINFORCE vs Q-learning learning curves, ablate entropy regularization and learning rates, and analyze common failure modes.
- **By the end:** You can explain why REINFORCE underperforms continuous Q-learning on this task, while still understanding where policy gradients shine (e.g., RLHF-style setups).

---

## Part III — Evaluation, Robustness, Deployment

### Chapter 9 — Off-Policy Evaluation (OPE)

- **Goal:** Learn how to evaluate new policies from logged data using IPS, SNIPS, DR, and related estimators, and validate them in the simulator.
- **Code:** `evaluation/ope.py` with logging ε-mix protocols.
- **What you’ll do:** Reproduce OPE vs on-policy consistency, explore estimator bias/variance, and sanity-check FQE-style approaches.
- **By the end:** You can pick and justify an OPE method for a given logging setup and understand its limitations.

### Chapter 10 — Robustness & Guardrails

- **Goal:** Build monitoring and guardrail systems so RL deployments stay safe under drift and unexpected behavior.
- **Code:** `monitoring/metrics.py`, drift detectors (e.g., CUSUM, Page–Hinkley), guardrail logic.
- **What you’ll do:** Inject synthetic drifts, trigger alarms, and exercise rollback mechanisms.
- **By the end:** You can design and evaluate a simple but realistic safety layer around a learning system.

### Chapter 11 — Multi-Episode Inter-Session MDP

- **Goal:** Extend the single-session view to a multi-episode MDP that captures retention and long-term engagement (mapping back to Chapter 1’s `{#EQ-1.2-prime}`).
- **Theory:** Hazard/survival modeling; episodic transitions \(s_{t+1} = f(s_t, \text{clicks}_t, \text{buys}_t,\) seasonality\()\).
- **Code:** `zoosim/multi_episode/session_env.py`, `zoosim/multi_episode/retention.py`.
- **What you’ll do:** Compare single-step proxies (e.g., δ·CLICKS) to discounted multi-episode value, and explore retention curves by segment.
- **By the end:** You can reason about policies over longer horizons and connect short-term metrics to long-term value.

---

## Part IV — Frontier Methods

### Chapter 12 — Slate RL & Differentiable Ranking

- **Goal:** Optimize full slates (top‑k rankings), incorporate diversity/novelty, and experiment with differentiable ranking layers under constraints.
- **Code:** `policies/slate_q.py`, `policies/diff_ranker.py` (SoftSort / NeuralSort / Gumbel–Sinkhorn wrappers), `evaluation/ope_slate.py`.
- **What you’ll do:** Study diversity@k vs GMV trade-offs, correct OPE for slate settings, and compare discrete and gradient-based slate methods.
- **By the end:** You understand the core ideas behind slate RL and differentiable ranking in modern recommendation systems.

### Chapter 13 — Offline RL for Boosts (Pessimistic)

- **Goal:** Train policies from logs without risky online exploration, with an emphasis on pessimistic (conservative) offline RL.
- **Code:** `policies/offline/{cql.py,iql.py,td3bc.py}` for bounded continuous actions.
- **What you’ll do:** Train on logged mixtures, evaluate with OPE, and sweep pessimism strengths to study safety vs performance trade-offs.
- **By the end:** You can apply offline RL to continuous boost policies and judge when an offline policy is safe to try online.

### Chapter 14 — Multi-Objective RL & Fairness at Scale

- **Goal:** Handle multiple objectives and fairness constraints simultaneously using CMDPs and Pareto policies.
- **Code:** `policies/mo_cmdp.py` (primal–dual optimization), fairness metrics in `evaluation/fairness.py`.
- **What you’ll do:** Plot Pareto fronts, run fairness gap sweeps across segments/brands, and reason about exposure and utility parity.
- **By the end:** You can design and evaluate policies that trade off GMV against fairness and other business constraints.

### Chapter 15 — Non‑Stationarity & Meta‑Adaptation

- **Goal:** Make bandits and RL policies robust to drift and non-stationarity, with practical adaptation playbooks.
- **Code:** `policies/adaptive_bandit.py` and related change-point and schedule-updating components.
- **What you’ll do:** Simulate synthetic shifts, study regret and recovery, and experiment with different adaptation strategies.
- **By the end:** You can describe and implement basic meta-adaptation strategies for changing environments.

---

## Part V — Optional Bayesian Appendix

### Appendix A — Bayesian Preference Models (Hierarchical, Optional)

- **Goal:** Model user-level price and private-label preferences via hierarchical Bayes and connect them to the feature engineering in Chapter 6.
- **Theory:** Hierarchical priors over segments and users; partial pooling and shrinkage; how posterior means/variances tie back to value-function framing from Chapter 3.
- **Code (planned):** `bayes/price_sensitivity_model.py` for hierarchical models fit on simulator logs, with hooks for exporting posterior-based features.
- **What you’ll do:** Fit models on simulated logs, compare posterior means to ground-truth simulator parameters, and plug Bayesian estimates into the bandit policies.
- **By the end (optional):** You see how Bayesian preference models can boost bandit performance and where they sit between simple and oracle features.

---

## Suggested Pacing

These are rough guidelines for a motivated reader working part‑time:

- **Part I (Ch. 0–3):** 1.5–2 weeks
- **Part II (Ch. 4–8):** 2.5–3 weeks
- **Part III (Ch. 9–11):** 2–2.5 weeks
- **Part IV (Ch. 12–15):** 3–4 weeks
- **Total (including frontier topics):** ~9–11.5 weeks; Parts III–IV can be parallelized in a group setting.

---

## Cross‑References and “Code ↔ X” Conventions

- Equations are tagged with anchors like `{#EQ-X.Y}` immediately after their statement, so they can be cited in prose (e.g., “By `{#EQ-1.2}`…”).
- Chapters use “Code ↔ X” callouts to link theory to implementation with concrete file paths (and line numbers where stable). For planned modules, the path is referenced even before the code is fully implemented.

---

## Quick Validation Commands

You can use the repo’s tests and scripts to sanity‑check your environment as you progress:

- **Environment:**
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install uv && uv pip install -r pyproject.toml && uv pip install -e .
  ```
- **Smoke tests:**
  ```bash
  uv run pytest -q
  ```
- **Example output (representative):**
  ```text
  2 passed in 3.2s
  ```

For visualization-heavy chapters (especially Chapter 5), reuse the plotting code in the text (e.g., with seed 42) to reproduce the scatter and histogram figures referenced in the book.

