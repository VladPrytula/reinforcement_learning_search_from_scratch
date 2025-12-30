# Chapter 14 Update Plan

## Scope Narrowing (no multi-brand scope creep)

- We drop "multi-brand fairness" as a core promise: the simulator has no `Product.brand` (`zoosim/world/catalog.py:15`), and "brand" currently exists only as a query-type label (`zoosim/core/config.py:140`).
- We redefine Chapter 14 fairness to operate over **provider groups** already modeled: `is_pl` (private-label vs national-brand), `category`, and optionally `strategic_flag`. We keep "brand queries" as a specificity regime (position bias / feature scaling), not a brand-ID mechanic.

---

## CRITICAL PREREQUISITE: "Multi-Objective" Title Justification

**The chapter title "Multi-Objective RL & Fairness" is defensible ONLY if Phase 1's bridge subsection is written.**

Without the "Multi-objective and Pareto fronts" bridge subsection, the title is misleading. Here's why:

| What readers might expect | What we actually deliver |
|---------------------------|--------------------------|
| Vector-reward MORL (Pareto Q-learning, multi-policy returns) | CMDP with scalar reward + constraints |
| Dominance-based policy selection | Lagrangian primal-dual optimization |
| Multiple non-dominated policies | One policy per constraint-threshold setting |

**Why CMDP is still legitimately "multi-objective":**

The CMDP + threshold-sweep approach is the **$\varepsilon$-constraint method** from multi-objective optimization ([@miettinen:multi_objective_optimization:1999], [@ehrgott:multicriteria_optimization:2005]). By varying constraint thresholds $(\tau_{\text{cm2}}, \tau_{\text{exp}}, \tau_{\text{stab}})$ and training a policy for each, we trace the Pareto frontier empirically. The Lagrange multipliers $\lambda^*$ at optimum encode marginal rates of substitution between objectives. For CMDP foundations, see [@altman:constrained_mdps:1999].

**Hard requirement:** The bridge subsection in Phase 1 MUST:

1. **Define Pareto dominance** for the outcome vector $\mathbf{v} = (\text{GMV}, \text{CM2}, -\Delta\text{rank}, -\text{exposure\_gap})$
   - **Sign convention**: GMV and CM2 are maximized; stability ($\Delta$-rank) and exposure gap are costs, so we negate them or use minimization dominance. Explicitly: $\mathbf{v} \succ \mathbf{v}'$ iff $\forall i: v_i \geq v'_i$ and $\exists j: v_j > v'_j$.
2. **Explain that CMDP + threshold sweeps IS the $\varepsilon$-constraint method**---cite [@miettinen:multi_objective_optimization:1999, Chapter 4] and [@ehrgott:multicriteria_optimization:2005, Chapter 4].
3. **Be honest that this differs from true vector-reward MORL**---forward-reference Appendix E for Pareto Q-learning, coverage sets, and when vector-reward MORL is necessary.
4. **Connect to Phase 6 labs** (`scripts/ch14/lab_01_pareto_fronts.py`, `scripts/ch14/lab_02_fairness_gap_sweeps.py`) which actually produce the Pareto plots.

**Gate condition:** Do not proceed to Phases 2--7 until the bridge subsection is:

1. Written following the four requirements above
2. Self-reviewed against Appendix C $\S$C.3--C.4 (policy space convexity, KKT conditions)
3. Cross-checked against at least one primary source ([@miettinen:multi_objective_optimization:1999] or [@ehrgott:multicriteria_optimization:2005])
4. Verified to use LaTeX notation ($\varepsilon$, $\tau$, $\lambda$) throughout---no Unicode Greek letters

The chapter's intellectual honesty depends on this prerequisite.

---

## Chapter 14 Implementation Plan (Code + Doc alignment, non-breaking)

### Phase 1 --- Syllabus + Chapter-14 contract (remove ambiguity, fix paths)

Goal: no false promises; paths match the repo.

- Update `docs/book/syllabus.md:110--113`
    - Change "exposure/utility parity across segments/brands" $\to$ "across segments and provider groups (private-label vs national-brand, category)".
    - Fix implementation paths to real packages:
        - `policies/mo_cmdp.py` $\to$ `zoosim/policies/mo_cmdp.py`
        - `evaluation/fairness.py` $\to$ `zoosim/evaluation/fairness.py`
- Mirror the same edits in `docs/book/syllabus_github.md` (it repeats the same promise).
- Update `docs/book/ch14/ch14_multiobjective_rl_and_fairness.md`
    - Add a short "Terminology contract" subsection:
        - "brand queries" $\neq$ product brand IDs in this simulator
        - fairness "brands" are modeled as `is_pl` and categories
    - Add a short "Multi-objective and Pareto fronts" bridge subsection (so the chapter title matches the method, and the labs are motivated by the text):
        - Define Pareto dominance and the Pareto front for a vector of outcomes (GMV, CM2, stability, exposure parity / fairness gaps).
        - Explain why a CMDP formulation (single scalar reward plus constraints) still encodes a genuinely multi-objective trade-off: changing thresholds and target bands changes which feasible policies exist and which ones are optimal under the constraint set.
        - State the two frontier constructions we will use in this repository:
            - **Construction A**: the $\varepsilon$-constraint method (primary). We sweep constraint targets (CM2 floor, stability $\tau$, exposure targets and bands), train with primal-dual updates, then evaluate and plot the resulting trade-off curve.
            - **Construction B**: scalarization sweeps (secondary). We sweep fixed weights / penalties in a scalar objective, train, and compare the induced curve to Construction A to show where scalarization hides parts of the frontier (supported vs unsupported points---see Appendix E).
        - Point explicitly to Phase 6 labs (`scripts/ch14/lab_01_pareto_fronts.py`, `scripts/ch14/lab_02_fairness_gap_sweeps.py`) as the operationalization, and tie back to the syllabus acceptance criterion ($< 1\%$ violations; fairness within target bands; GMV loss quantified vs an unconstrained baseline).
    - Make the roadmap consistent (list only `zoosim/policies/mo_cmdp.py`, `zoosim/evaluation/fairness.py`).
- Repo-wide audit step (doc trust): search for "segments/brands" phrasing and update to "provider groups" where it refers to product-side fairness.

### Phase 2 --- Define fairness/exposure metrics (new module)

Deliverable: `zoosim/evaluation/fairness.py` (pure functions; no env coupling).

- Exposure model (use simulator's own position bias):
    - `position_weights(cfg, query_type, k) -> np.ndarray` using `cfg.behavior.pos_bias` (`zoosim/core/config.py:180`).
- Groupings (provider groups)
    - `group_key(product, scheme) -> str` with built-in schemes:
        - `scheme="pl"`: groups are `pl` / `non_pl` via `Product.is_pl`
        - `scheme="category"`: groups are category names
        - `scheme="strategic"`: groups are `strategic` / `non_strategic`
- Metrics
    - `exposure_by_group(ranking_topk, catalog, weights, group_fn) -> dict[str,float]`
    - `exposure_share_by_group(...) -> dict[str,float]` (normalized)
    - `l1_gap(shares, targets)` and optional `kl_divergence(shares, targets)`
    - `within_band(shares, targets, band)` returning per-group booleans + max deviation
- Segment reporting (utility parity reporting)
    - Utilities already exist as per-episode reward/GMV/CM2; add helpers to aggregate per `user_segment` (reporting first; constraints optional).

### Phase 3 --- Make stability measurable (wire $\lambda_{\text{rank}}$ pathway without changing config line refs)

Goal: add additive info needed for $\Delta$-Rank constraints; do not break old tests.

- Update `zoosim/envs/search_env.py`
    - Compute a baseline ranking from the same (`base_scores`, `feature_matrix`) using zero action.
    - Add to `info`:
        - `baseline_ranking` (top-$k$)
        - `delta_rank_at_k_vs_baseline` using `compute_delta_rank_at_k` (`zoosim/monitoring/metrics.py:89`)
        - `user_segment`, `query_type` (so fairness reports don't need private env access)
- Update `zoosim/multi_episode/session_env.py` similarly (optional but keeps multi-episode compatible).
- Keep existing keys unchanged; only add keys (existing tests assert presence of old keys, not absence of new ones).

### Phase 4 --- Constraint specification layer (small, explicit)

Deliverable: implemented inside `zoosim/policies/mo_cmdp.py` (or `zoosim/policies/constraints.py` if you prefer separation).

- `ConstraintSpec` dataclass:
    - `name`
    - `sense`: `"ge"` | `"le"`
    - `threshold`
    - `metric_from_info(info, cfg) -> float`
    - `slack(metric) -> float` producing $g_i(\theta)$ in the same sign convention as Ch14.
- Built-in specs (provider-group version):
    - Stability: `delta_rank_at_k_vs_baseline` $\leq \tau_{\text{stability}}$ ($\tau$ comes from trainer config, not core config).
    - CM2 floor: $\mathbb{E}[\text{cm2}] \geq$ `cfg.action.cm2_floor` when set.
    - Exposure floors: generated from `cfg.action.exposure_floors` with a documented key scheme, e.g. keys like `pl`, `non_pl`, `category:dog_food`, `strategic`.

### Phase 5 --- Primal--dual CMDP reference implementation

Deliverable: `zoosim/policies/mo_cmdp.py`.

- `PrimalDualCMDPConfig`:
    - `gamma`, `batch_episodes`, `dual_lr`, `dual_update_every`
    - `tau_stability`, `k_stability`
    - optional `exposure_targets` + `band` (to satisfy "target bands" in syllabus without touching `zoosim/core/config.py`)
    - `seed`, `device`
- Policy backbone:
    - Start by reusing `REINFORCEBaselineAgent` (`zoosim/policies/reinforce_baseline.py:48`) via `GymZooplusEnv(rich_features=True)` (`zoosim/envs/gym_env.py:25`) so the trainer consumes flat numpy obs + continuous actions.
- Dual variables:
    - Maintain `lambdas: dict[str,float]` per constraint, projected to $\geq 0$.
    - Initialize stability $\lambda$ from `cfg.action.lambda_rank` (this is where the Ch01 "placeholder" becomes operational without changing config structure).
- Training loop:
    - Roll out episodes, compute `metric_i`, `slack_i`, and shaped reward $r' = r + \sum_i \lambda_i \cdot \text{slack}_i$.
    - Actor update on $r'$.
    - Dual update with batch-averaged slacks: $\lambda_i \leftarrow \max(0, \lambda_i - \eta_\lambda \cdot \overline{\text{slack}}_i)$.
- Outputs:
    - Training history: per-episode reward/GMV/CM2, slacks, violation indicators, $\lambda$ trajectories, fairness gaps (from `zoosim/evaluation/fairness.py`).

### Phase 6 --- Chapter 14 labs/scripts (Pareto + fairness sweeps)

Deliverables: new folder `scripts/ch14/`.

- `scripts/ch14/lab_01_pareto_fronts.py`
    - Sweep constraint targets (recommended) such as `cm2_floor`, exposure target/band, stability $\tau$.
    - For each setting: train $\to$ evaluate on held-out seeds $\to$ write CSV.
- `scripts/ch14/lab_02_fairness_gap_sweeps.py`
    - Fix a trained policy, sweep evaluation targets/bands and report fairness/GMV tradeoffs.
- Output location: `docs/book/ch14/data/` (match Chapter 11 pattern).

### Phase 7 --- Tests (minimal but decisive)

Deliverables: `tests/ch14/...` (fast unit + tiny smoke).

- Fairness module:
    - exposure shares sum to 1
    - group parsing works (`pl`, `category:...`, etc.)
- Env instrumentation:
    - `delta_rank_at_k_vs_baseline == 0` when action is zero
    - new info keys exist (additive)
- Primal--dual:
    - unit test dual projection + direction ($\lambda$ increases when violated under chosen sign convention)
    - tiny smoke run (few episodes) asserts finite metrics and no crashes.

### Phase 8 --- Acceptance + verification checklist (Springer-quality "trust contract")

- Verify doc promises match code paths and implemented modules:
    - `rg "policies/mo_cmdp.py|evaluation/fairness.py|segments/brands" docs/book`
- Run `pytest -q` (must remain green).
- Generate one Pareto CSV + plot in `scripts/ch14/...` and ensure:
    - reported constraint violation rate $< 1\%$ per constraint (as per syllabus)
    - fairness within band for configured provider group(s)
    - GMV loss is explicitly quantified vs unconstrained baseline.
