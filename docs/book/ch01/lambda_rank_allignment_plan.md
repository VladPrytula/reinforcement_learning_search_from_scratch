# Lambda-Rank Alignment Plan

## Goal

- Make the `lambda_rank` story 100% honest and consistent across Ch01, Ch10, and Appendix C, while adopting **Option A**: `lambda_rank` remains a config placeholder until Chapter 14 (primal-dual / soft constraints).
- Remove/repair any statements that imply Ch10 implements primal-dual or that `SafetyMonitor` enforces CM2 floors when it doesn't.

---

## Canonical Narrative (the "Truth Contract")

- **Appendix C** provides the mathematical foundations: Lagrangian duality, KKT conditions, primal-dual algorithms for constrained optimization.
- **Chapter 10** is production monitoring + hard guardrails (drift detection, fallback, stability monitoring). It does **not** implement primal-dual optimization. Hard constraint enforcement (e.g., CM2 floor rejection) is demonstrated as an **exercise pattern** (Exercise 10.3), not built into `SafetyMonitor`.
- **Chapter 14** implements **general CMDP with primal-dual optimization** (`policies/mo_cmdp.py` per syllabus). This framework handles **any** constraint as a soft penalty---CM2, exposure, rank stability, fairness---via learned Lagrange multipliers and Pareto policies.
- `lambda_rank` is one specific multiplier within this framework, associated with the rank-stability constraint (Delta-Rank@k). Other multipliers (e.g., for CM2, exposure) follow the same primal-dual pattern.

### Hard vs. Soft Constraints

The book presents two complementary approaches to constraint satisfaction:

| Constraint | Hard (Ch10 pattern) | Soft (Ch14 CMDP) |
|------------|---------------------|------------------|
| CM2 floor | Rejection-resampling wrapper (Ex 10.3) | Lagrangian penalty on margin violations |
| Rank stability | Action filter on Delta-Rank@k | `lambda_rank` penalty in reward |
| Exposure floors | Pre-filter rankings | Lagrangian penalty on exposure gaps |
| Fairness | Reject unfair slates | Pareto trade-off across segments |

### Design Rationale

**Why this framing?** We distinguish hard and soft constraints because they serve fundamentally different purposes in production systems:

1. **Hard constraints** (Chapter 10 pattern) provide **safety guarantees**---the system never executes a violating action. Appropriate when violations are categorically unacceptable: negative margin that loses money on every transaction, regulatory requirements, or user-facing commitments. Implementation: action feasibility filter that rejects before execution.

2. **Soft constraints** (Chapter 14 CMDP) enable **adaptive trade-offs**---the system learns how severely to penalize violations relative to the primary objective. Appropriate when the optimal balance is context-dependent: GMV vs. fairness Pareto frontier, short-term revenue vs. long-term retention. Implementation: Lagrangian penalty with learned multipliers via primal-dual optimization.

**Why not conflate them?** A common mistake is to treat all constraints as soft penalties. This fails when:
- Violations have discontinuous costs (e.g., legal liability at a threshold)
- The learned multiplier converges too slowly, allowing unacceptable violations during training
- Business stakeholders require hard guarantees, not probabilistic compliance

Conversely, treating all constraints as hard rejects can be overly conservative, leaving value on the table when slight constraint relaxation would significantly improve the primary objective.

**The book's architecture:** Chapter 10 teaches the hard-constraint pattern (monitoring, rejection, fallback). Chapter 14 teaches the soft-constraint pattern (CMDP, primal-dual, Pareto). Practitioners choose based on business requirements.

### Dependency Diagram

```
                              THEORY
                                |
                                v
                      +------------------+
                      |   Appendix C     |
                      | (Lagrangian      |
                      |  duality, KKT,   |
                      |  primal-dual)    |
                      +------------------+
                         /            \
                        /              \
            theory tie-in              theory foundation
                      /                  \
                     v                    v
          +------------------+    +------------------+
          |   Chapter 1      |    |   Chapter 14     |
          |   Section 1.9    |    | (general CMDP,   |
          | (Lagrangian      |    |  primal-dual,    |
          |  formulation)    |    |  Pareto, fairness)|
          +------------------+    +------------------+
                     |                    ^
                     |                    |
         forward ref |                    | soft constraints
           (honest)  |                    | are operational
                     v                    |
          +------------------+    complements
          |   Chapter 10     |------------+
          | (monitoring,     |
          |  hard guardrails,|
          |  Exercise 10.3)  |
          +------------------+

Legend:
  - Appendix C = mathematical foundations (Lagrangian duality, KKT, convergence guarantees)
  - Ch01 = problem formulation (constraints introduced, forward references)
  - Ch10 = production safety (monitoring, drift detection, hard constraint pattern via Ex 10.3)
  - Ch14 = soft constraint optimization (general CMDP, all multipliers learned, Pareto policies)
```

---

## Work Plan (Detailed, by Artifact)

### 1. Inventory + Classify All "Claims"

**Scope:** Repo-wide search for `lambda_rank`, "wiring introduced", "primal-dual in Ch10", and "SafetyMonitor CM2 floor".

**False claims identified:**

| File:Line | Claim | Category |
|-----------|-------|----------|
| `docs/book/outline.md:76` | Appendix C dependencies list only `Ch01 §1.9, Ch10` (missing Ch14 / can be read as “Ch10 implements constrained RL”) | (b) unclear |
| `docs/book/outline.md:81` | "Before implementing constrained RL (Chapter 10)" | (c) false |
| `docs/book/ch01/ch01_foundations_revised_math+pedagogy_v3.md:158-160` | "enforcement is introduced in Chapter 10" (`cm2_floor`, `exposure_floors`) | (c) false |
| `docs/book/ch01/ch01_foundations_revised_math+pedagogy_v3.md:1063` | "We enforce constraints via Lagrangian methods (Chapter 10...)" | (c) false |
| `docs/book/ch01/ch01_foundations_revised_math+pedagogy_v3.md:1095` | "Chapter 10 exploits this to implement primal-dual RL" | (c) false |
| `docs/book/ch01/ch01_foundations_revised_math+pedagogy_v3.md:1103` | "In Chapter 10 (Section 10.4.2), we implement constraint-aware RL" | (c) false |
| `docs/book/ch03/ch03_stochastic_processes_bellman_foundations.md:426` | "Appendix C develops ... primal--dual ...; Chapter 10 applies these ideas" (add Ch14 reference to avoid misreading) | (b) unclear |
| `docs/book/ch10/ch10_robustness_guardrails.md:1009` | "SafetyMonitor can check CM2 floors via enable_cm2_floor" | (c) false |
| `docs/book/ch10/ch10_robustness_guardrails.md:1520` | Production checklist implies CM2 floor is enforceable | (c) false |
| `docs/book/ch10/ch10_robustness_guardrails.md:1526` | Alerting implies SafetyMonitor emits CM2 breach alerts | (b) unclear |
| `docs/book/appendix_c_convex_optimization.md:437` | "Chapter 10: Implementing primal-dual algorithms" | (c) false |
| `docs/book/ch01/exercises_labs.md:481-484` | "placeholder for Chapter 10 stability guardrails" | (c) false |
| `docs/book/ch03/ch03_bellman_and_regret.md:557` | "Chapter 8 extends this..." | (a) true (OK) |

**Acceptance:** Checklist above with target replacement language for each (c) item.

---

### 2. Chapter 1 Updates (Remove False Forward References)

**Files:**
- `docs/book/ch01/exercises_labs.md` (Lab 1.8 note + task text)
- `docs/book/ch01/ch01_foundations_revised_math+pedagogy_v3.md`:
  - Lines 158-160 ("Code - Config (constraints)" box)
  - Line 1063 (Lagrangian methods forward ref)
  - Line 1095 (primal-dual implementation claim)
  - Line 1103 (Ch10 Section 10.4.2 implementation preview)

**Edits:**

**(a) Lab 1.8 (exercises_labs.md:481-484):**

Replace:
> "treat it as a placeholder for Chapter 10 stability guardrails"

With:
> "`lambda_rank` is currently **unused in code**. The roadmap:
> - **Appendix C**: theory for lambda (Lagrangian duality, why nonnegative)
> - **Ch10**: hard guardrails/monitoring (drift detection, fallback---no lambda usage)
> - **Ch14**: primal-dual soft constraint trade-off (lambda is learned/operational)"

**(b) Code-Config box (ch01_foundations:158-160):**

Replace:
> "enforcement is introduced in Chapter 10"

With:
> "config placeholder; enforcement pattern in Exercise 10.3; primal-dual optimization in Chapter 14"

**(c) Lines 1063, 1095, 1103:**

Replace all references to "Chapter 10 implements primal-dual" with:
> "Chapter 10 provides hard guardrails and monitoring; Chapter 14 implements primal-dual constrained RL (theory in Appendix C)."

**Acceptance:**
- No Ch01 text implies `lambda_rank` gets wired in Ch10.
- Ch01 explicitly says `lambda_rank` is currently unused in code.

---

### 2b. Book-Level Cross-References (Outline + Chapter 3)

**Files:**
- `docs/book/outline.md` (Appendix C dependencies + "When to read" bullets)
- `docs/book/ch03/ch03_stochastic_processes_bellman_foundations.md` (Remark 3.5.3 CMDP preview)
- (Verify only) `docs/book/ch03/ch03_bellman_and_regret.md` (primal-dual demo does not attribute implementation to Ch10)

**Edits:**
- Update `docs/book/outline.md` so Appendix C is positioned as:
  - The theory prerequisite for the Lagrangian formulation in Chapter 1 (§1.9),
  - The conceptual foundation for constraints discussed in Chapter 10,
  - The implementation foundation for primal-dual constrained RL in Chapter 14.
- Update Remark 3.5.3 to separate roles explicitly: Appendix C (theory), Chapter 10 (guardrails), Chapter 14 (primal-dual implementation).

**Acceptance:**
- No book-level or early-chapter text suggests primal-dual constrained RL is implemented in Chapter 10.

---

### 3. Chapter 10 Alignment (Match Actual Implementation)

**File:** `docs/book/ch10/ch10_robustness_guardrails.md`

**Architectural context:** `SafetyMonitor` currently only sees `(action, features, reward)` (`zoosim/monitoring/guardrails.py:115`) and has **no access to CM2**, so it cannot "check/enforce CM2 floors" as written.

**Exercise 10.3** already uses the correct pattern: a **constraint filter wrapper** around the policy (`docs/book/ch10/exercises_labs.md:349`), where CM2 can be estimated/checked before accepting an action.

**Three specific fixes:**

**(a) Fix false "already enforceable" claim (line 1009):**

Replace:
> "SafetyMonitor (Section 10.5) can check CM2 floors at line `zoosim/monitoring/guardrails.py:41-42` via config flag `enable_cm2_floor`."

With:
> "`GuardrailConfig.enable_cm2_floor` and `min_cm2` exist as configuration fields (`zoosim/monitoring/guardrails.py:41-42`), but CM2-floor enforcement is **not implemented** in the `SafetyMonitor` reference implementation presented in this chapter: `SafetyMonitor` receives `(action, features, reward)` and does not evaluate CM2 values. See **Exercise 10.3** for the hard-constraint implementation pattern (action feasibility filter / rejection-resampling wrapper)."

**(b) Fix production checklist (line 1520):**

Replace:
> "CM2 floor: Set `GuardrailConfig.min_cm2`... Enable via `enable_cm2_floor` flag"

With:
> "**CM2 floor (hard):** Implement as an action feasibility filter / rejection-resampling wrapper (see **Exercise 10.3**). In the Chapter 10 reference implementation, `SafetyMonitor` does not evaluate CM2, so CM2-floor enforcement must live at action selection time (outside the monitor). For soft constraint optimization, see **Chapter 14**."

**(c) Clarify alerting language (line 1526):**

Replace:
> "(4) CM2 floor breached"

With:
> "(4) CM2 floor breached (if you implement the constraint filter from Exercise 10.3 and log CM2 violations)"

**(d) Additional clarifications:**

- The "Lagrangian relaxation" bullet (line 992-998) is an **approach**, not implemented here. Add: "See Chapter 14 for primal-dual operationalization."
- Stability check is currently **action-switch-rate monitoring**, not Delta-Rank@k enforcement. Clarify this in Section 10.5's SafetyMonitor description.

**Acceptance:**
- No Ch10 sentence claims primal-dual is implemented.
- No Ch10 sentence claims CM2 floor enforcement exists in `SafetyMonitor`.
- Exercise 10.3 is the canonical reference for hard CM2 enforcement pattern.

---

### 4. Appendix C Cross-Reference Correction

**File:** `docs/book/appendix_c_convex_optimization.md`

**Fix "When to consult this appendix" bullets (lines 434-438):**

Replace:
> "- **Chapter 10**: Implementing primal-dual algorithms for CM2/exposure constraints"

With:
> "- **Chapter 1, Section 1.9**: Verifying that the Lagrangian reformulation is equivalent to the constrained problem
> - **Chapter 10**: Understanding constraints/guardrails context (monitoring, hard rejection---but **not** primal-dual implementation)
> - **Chapter 14**: Implementing primal-dual constrained RL (where lambda is operational)"

**Acceptance:** Appendix C contains zero claims that Ch10 implements primal-dual algorithms.

---

### 5. Draft Chapter 14 (Pay Off the Forward Reference)

**New files (proposed):**
- `docs/book/ch14/ch14_multiobjective_rl_and_fairness.md` (main draft)
- `docs/book/ch14/exercises_labs.md` (stub with 2-3 concrete tasks, optional)

**Scope per syllabus** (`syllabus.md:109-113`):
- **Objectives:** CMDP + Pareto policies; exposure/utility parity across segments/brands
- **Implementation:** `policies/mo_cmdp.py` with primal-dual; fairness metrics in `evaluation/fairness.py`
- **Labs:** Plot Pareto fronts; fairness gap sweeps
- **Acceptance:** <1% constraint violations; fair exposure within target bands with minimal GMV loss

**Minimum viable content for alignment fix:**

1. **Problem statement:** General CMDP framing---any constraint (CM2, exposure, rank stability, fairness) can be handled as a Lagrangian penalty with learned multipliers
2. **Primal-dual algorithm box:** Policy update on Lagrangian + dual ascent on violations; show how `lambda_rank` (rank stability), `lambda_cm2` (margin), `lambda_exposure` (fairness) all follow the same pattern
3. **Hard vs. soft bridge:**
   - Appendix C provides theory (duality, Slater's condition, convergence)
   - Ch10 provides hard guardrails (monitoring, fallback, Exercise 10.3 rejection pattern)
   - Ch14 provides soft/adaptive optimization (all multipliers learned, Pareto trade-offs)
4. **When to use which:** Decision framework for practitioners (see Design Rationale above)
5. **"Status" admonition:** What's implemented vs. planned in code (`policies/mo_cmdp.py` planned)

**Integration:**
- Add Ch14 to `mkdocs.yml` nav under "Part IV --- Frontier Methods" (per syllabus structure), so forward references are clickable.

**Acceptance:**
- Readers can navigate to Ch14 and see the general CMDP framework
- `lambda_rank` is shown as one instance of the pattern, not the entire chapter
- Clear guidance on when to use hard (Ch10) vs. soft (Ch14) constraints

---

### 6. Consistency + Build Checks

**Searches:**
```bash
rg "lambda_rank" docs/book  # Ensure each mention is consistent with roadmap
rg "primal[- ]dual" docs/book/ch10  # Ensure wording matches reality
rg "Chapter 10.*implement" docs/book  # Catch any remaining false claims
rg "enforce.*CM2.*SafetyMonitor" docs/book  # Verify no false enforcement claims
```

**Build:**
```bash
mkdocs build  # Confirm no broken paths from adding Ch14
```

**Optional:** Run knowledge-graph validation if we touch KG.

**Acceptance:**
- No stale "introduced in Chapter 10" statements about `lambda_rank`
- No claims that `SafetyMonitor` enforces CM2 floors
- `mkdocs build` succeeds

---

## Summary Table

| Step | Files | Key Fix |
|------|-------|---------|
| 1 | (inventory) | Classify all false claims |
| 2 | `docs/book/ch01/exercises_labs.md`, `docs/book/ch01/ch01_foundations_revised_math+pedagogy_v3.md` | Remove false Ch10 forward refs |
| 2b | `docs/book/outline.md`, `docs/book/ch03/ch03_stochastic_processes_bellman_foundations.md` | Align cross-chapter references (Ch10 vs Ch14 roles) |
| 3 | `docs/book/ch10/ch10_robustness_guardrails.md` | Fix CM2/SafetyMonitor claims, point to Ex 10.3 |
| 4 | `docs/book/appendix_c_convex_optimization.md` | Fix "When to consult" bullets |
| 5 | `docs/book/ch14/` (new) | Draft primal-dual chapter |
| 6 | (validation) | rg searches + mkdocs build |
