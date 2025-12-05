# Book Syllabus — RL Search From Scratch (Vlad Prytula)

This syllabus lays out a chapter-by-chapter path from foundations and a working simulator (zoosim) to frontier methods for ranking and reinforcement learning. Each chapter lists objectives, theory, implementation hooks, labs, and acceptance criteria so progress is estimable and auditable.

## Part I — Foundations

0) Chapter 0 — Motivation: A Tiny Search Engine That Learns (draft exists: `docs/book/drafts/ch00_motivation_first_experiment.md`)
- Objectives: Run a complete small RL experiment (toy bandit with boosts); build intuition for exploration, reward design, and regret.
- Theory: Minimal and informal in Ch0; rigor deferred to Ch1–Ch3.
- Implementation: `scripts/ch00/toy_problem_solution.py`; sanity in `tests/ch00/test_toy_example.py`.
- Labs: Plot learning curves; compare to baselines; simple CM2 floor; action‑space geometry.
- Acceptance: Reproduce representative outputs in Ch0; explain limitations and the need for formalization.

-1) Chapter 1 — Search Ranking as Optimization (draft exists: `docs/book/drafts/ch01_foundations.md`)
- Objectives: Formalize multi-objective reward and the contextual bandit formulation for the Chapter 0 toy and real search; introduce #EQ-1.2 and the multi-episode preview #EQ-1.2-prime.
- Theory: Contextual bandit vs MDP; constraints (CM2, exposure, rank stability); engagement as soft viability.
- Implementation: Reward weights in `zoosim/core/config.py` and aggregation in `zoosim/dynamics/reward.py`.
- Labs: Compute reward under different weights; check CVR diagnostic; verify rank-stability penalty wiring.
- Acceptance: Reproduce example outputs in ch01; add assertion bound on δ/α in reward code.

2) Chapter 2 — Probability, Measure, and Click Models (PBM/DBN)
- Objectives: Model examination/click/purchase; define position bias and abandonment.
- Theory: PBM/DBN abstractions; stopping times; unbiased estimators via propensities.
- Implementation: Click/exam/buy paths in `zoosim/dynamics/behavior.py`; position-bias knobs in `zoosim/core/config.py`.
- Labs: Plot CTR vs rank by query type; simulate abandonment vs utility.
- Acceptance: CTR curves monotone; abandonment 10–15% on low utility sessions.

3) Chapter 3 — Stochastic Processes & Bellman Foundations
- Objectives: Connect single-step bandit to MDP/CMDP; introduce Bellman operators and contractions.
- Theory: Value functions; constraints via Lagrangian; preview of OPE as model-based DM.
- Implementation: No new code; formalism supports later chapters.
- Labs: Verify contraction on a toy MDP.
- Acceptance: Proof sketches compile and tie to later code (cross-referenced from labs).

## Part II — Simulator and Data

4) Chapter 4 — Catalog, Users, Queries (Generative Design)
- Objectives: Deterministic world generation with seeds; realistic priors.
- Implementation: `zoosim/world/{catalog,users,queries}.py`, config defaults in `zoosim/core/config.py`; demo script `scripts/ch04/ch04_demo.py`.
- Labs: Compute catalog price/margin stats; verify deterministic generation given seed.
- Acceptance: `uv run pytest -q` passes `tests/test_catalog_stats.py` and determinism checks.

5) Chapter 5 — Relevance, Features, Reward
- Objectives: Hybrid lexical/semantic relevance; engineered boost features; reward aggregation.
- Implementation: `zoosim/ranking/{relevance,features}.py`, `zoosim/dynamics/reward.py`; config-driven constants; demo script `scripts/ch05/ch05_demo.py`.
- Labs: Visualize feature interactions (cm2×cat, discount×price_sens); validate reward breakdown.
- Acceptance: Feature standardization applied; reward equals weighted sum expected from lab.

6) Chapter 6 — Discrete Template Bandits (LinUCB/TS)
- Objectives: Safe, interpretable baseline; honest failure modes; feature-driven improvement.
- Implementation: `policies/{templates,lin_ucb,thompson_sampling}.py`; `scripts/ch06/template_bandits_demo.py`.
- Labs: Simple vs rich context features; per-segment diagnostics; template selection analysis.
- Acceptance: Simple-features bandits underperform a strong static template; rich features measurably reduce the GMV gap vs static on average; per-segment heterogeneity and theory–practice gap clearly documented.

6A) Chapter 6A — Neural Bandits: From Neural Linear to NL-Bandit (optional bridge chapter)
- Objectives: Learn features with neural networks; bridge linear bandits (Ch6) → continuous Q(x,a) (Ch7).
- Theory: Neural Linear architecture (f_ψ + linear heads); discrete NL-Bandit (neural Q(x,a) on templates); sample complexity trade-offs.
- Implementation: `scripts/ch06a/{neural_linear_demo,neural_bandits_demo}.py`; reuse `policies/{lin_ucb,thompson_sampling,q_ensemble}.py`.
- Labs: Neural Linear vs rich features vs simple features; NL-Bandit ensemble calibration; overfit diagnosis (small data, large network); transfer learning across segments.
- Acceptance: Neural Linear beats rich features when nonlinearity exists + data is abundant (target: +5-10% GMV); degrades gracefully when data is scarce (honest failure modes documented); ensemble uncertainty calibrated (σ correlates with |r − μ(x,a)| on held-out data, ρ ≥ 0.5).

7) Chapter 7 — Continuous Actions via Q(x,a)
- Objectives: Regress E[R|x,a]; optimize bounded boosts with uncertainty and trust region.
- Implementation: `policies/q_ensemble.py`, `optimizers/cem.py`.
- Labs: UCB vs greedy selection; Δrank@k trust-region tuning.
- Acceptance: +10–15% GMV vs best static template and ≥ Chapter 6 rich-feature bandits, under guardrail compliance and stable uncertainty calibration.

8) Chapter 8 — Policy Gradient Methods (REINFORCE)
- Objectives: Derive the Policy Gradient Theorem; implement REINFORCE with Gaussian policies; compare on-policy policy gradients to Chapter 7’s value-based Q-learning and document the theory–practice gap.
- Implementation: `zoosim/policies/reinforce.py`; training and visualization scripts in `scripts/ch08/{reinforce_demo.py,neural_reinforce_demo.py,visualize_theory_practice_gap.py}`.
- Labs: Reproduce REINFORCE vs Q-learning learning curves; ablate entropy regularization and learning rate; analyze failure of deep end-to-end REINFORCE and explore RLHF-style extensions.
- Acceptance: REINFORCE reliably improves over random and discrete bandits but underperforms continuous Q-learning (target: ≈2× gap documented and explained); entropy collapse and variance issues are reproduced and diagnosed in labs.

## Part III — Evaluation, Robustness, Deployment

9) Chapter 9 — Off-Policy Evaluation (OPE)
- Objectives: IPS/SNIPS/DR; SWITCH/MAGIC; FQE sanity checks.
- Implementation: `evaluation/ope.py` with logging ε-mix protocol.
- Labs: Reproduce OPE vs on-policy consistency on simulator data.
- Acceptance: OPE estimates within 95% CI of online estimates on held-out seeds.

10) Chapter 10 — Robustness & Guardrails
- Objectives: Drift detection; safe fallback; monitoring (Δrank@k, CTR, CVR, GMV, CM2, latency).
- Implementation: `monitoring/metrics.py`; drift detectors (CUSUM/Page–Hinkley).
- Labs: Inject synthetic drifts; verify alarms and rollback.
- Acceptance: Automatic recovery to baseline within N episodes after drift.

11) Chapter 11 — Multi-Episode Inter-Session MDP (maps ch01 #EQ-1.2-prime)
- Objectives: Introduce retention/satisfaction state; show engagement implicit via long-term value.
- Theory: Hazard/survival modeling; episodic transitions s_{t+1}=f(s_t, clicks_t, buys_t, seasonality).
- Implementation: `zoosim/multi_episode/session_env.py`, `zoosim/multi_episode/retention.py` (hazard + satisfaction dynamics).
- Labs: Compare single-step proxy (δ·CLICKS) to multi-episode value (discounted GMV); retention curves by segment.
- Acceptance: Policy ordering agrees ≥80% short horizon; retention monotone in engagement; value estimates stable across seeds.

## Part IV — Frontier Methods

12) Chapter 12 — Slate RL & Differentiable Ranking
- Objectives: Optimize top-k lists; add diversity/novelty; enable gradient-based ranking under constraints.
- Implementation: `policies/slate_q.py`, `policies/diff_ranker.py` (SoftSort/NeuralSort/Gumbel–Sinkhorn wrappers), `evaluation/ope_slate.py`.
- Labs: Diversity@k vs GMV trade-off; top-k OPE correction; compare gradient vs discrete slate methods.
- Acceptance: +2–5% diversity@k at ≤1% GMV loss; OPE↔online agreement within 95% CI.

13) Chapter 13 — Offline RL for Boosts (Pessimistic)
- Objectives: Train from logs safely; select conservative policies.
- Implementation: `policies/offline/{cql.py,iql.py,td3bc.py}` for bounded continuous actions.
- Labs: Train on logged mixtures; evaluate with OPE; ablate pessimism strength.
- Acceptance: Offline policy ≥ NL-Bandit; no constraint regressions; stable under ε changes.

14) Chapter 14 — Multi-Objective RL & Fairness at Scale
- Objectives: CMDP + Pareto policies; exposure/utility parity across segments/brands.
- Implementation: `policies/mo_cmdp.py` with primal–dual; fairness metrics in `evaluation/fairness.py`.
- Labs: Plot Pareto fronts; fairness gap sweeps.
- Acceptance: <1% constraint violations; fair exposure within target bands with minimal GMV loss.

15) Chapter 15 — Non‑Stationarity & Meta‑Adaptation
- Objectives: Drift-aware bandits/RL; warm-start and adaptation playbooks.
- Implementation: `policies/adaptive_bandit.py`; change-point aware exploration; schedule updates.
- Labs: Regret under synthetic shifts; recovery speed.
- Acceptance: Bounded regret; quick recovery to baseline; no guardrail breaches.

## Part V — Optional Bayesian Appendix

Appendix A — Bayesian Preference Models (Hierarchical, Optional)
- Objectives: Infer user-level price and PL preferences (θ_price, θ_pl) from logged interactions via hierarchical Bayes; connect these estimates to the rich context features used in Chapter 6.
- Theory: Hierarchical priors over segments and users; partial pooling and shrinkage; simple hierarchical logistic/normal models; how posterior means and variances relate to the Bellman / value-function framing from Chapter 3.
- Implementation (planned): `bayes/price_sensitivity_model.py` (hierarchical model over segments and users, fit on simulator logs); hooks for exporting posterior means as φ_rich_est-style features.
- Labs: Fit the model on simulated logs; compare posterior means to true simulator θ_price, θ_pl; plug these estimates into the template bandit (in place of oracle latents) and compare performance across φ_simple, φ_rich (oracle), and φ_rich_est (estimated).
- Acceptance: Posterior estimates shrink sensibly toward segment means for low-data users; bandit performance with Bayesian θ̂ sits between φ_simple and oracle φ_rich, illustrating realistic gains from better preference modeling.

## Milestones and Estimates

- Part I (Ch 1–3): 1.5–2 weeks
- Part II (Ch 4–8): 2.5–3 weeks
- Part III (Ch 9–11): 2–2.5 weeks
- Part IV (Ch 12–15): 3–4 weeks
- Total (MVP + Frontier): ~9–11.5 weeks (parallelizable in Parts III–IV)

## Cross‑References and “Code ↔ X” Conventions

- Use equation anchors `{#EQ-X.Y}` immediately after tags and cite in prose (e.g., “By #EQ-1.2…”).
- Place “Code ↔ X” admonitions in chapters to link theory to implementation with file references (line numbers when stable). For planned modules above, reference the file path until lines exist.

## Quick Validation Commands

- Environment
```
python -m venv .venv && source .venv/bin/activate
pip install uv && uv pip install -r pyproject.toml && uv pip install -e .
```
- Smoke tests
```
uv run pytest -q
```
Output (representative):
```
2 passed in 3.2s
```

- Viz example: reuse the listing embedded in Chapter 5 (seed 42) to reproduce the scatter/histogram plots referenced in the text.
