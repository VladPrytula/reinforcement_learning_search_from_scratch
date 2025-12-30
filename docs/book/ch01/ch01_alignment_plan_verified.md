# Ch01 Alignment Plan - Critical Verification & Execution Plan

## Executive Summary

This plan implements **40 fixes across 12 files**:
- **30 Ch01 fixes** (from `docs/book/ch01/ch1_update_plan.md`, originally 22, expanded and re-verified)
- **7 cross-chapter doc fixes** (constraint chapter references + STRAT exposure-vs-purchases notation)
- **3 global alignment fixes** (`docs/book/syllabus.md` + `zoosim/monitoring/metrics.py`)

**Verification Status:** All claims have been **critically verified** against the actual codebase.

**Review Additions (2024-12-22):**
- FIX-5.1a: Fix `strat_purchases` generator to derive from actual purchases (not Poisson proxy)
- FIX-3.1a: Standardize Lab 1.1 on self-contained script (canonical path)
- Expanded scope notes for Appendix C and Ch03 STRAT clarifications
- Verification command runs to completion (no `head -50` truncation)

**Review Additions (2025-12-27):**
- FIX-1.8: Align Ch01 Δrank@k definition to production (set-churn) and Chapter 10
- FIX-6.1–6.2: Update `docs/book/syllabus.md` to match Ch01 RPC naming and Ch10 constraints placement
- FIX-7.1: Fix `zoosim/monitoring/metrics.py` reference to the correct Delta-Rank definition anchor

---

## Critical Verification Results

### VERIFIED CORRECT

| Claim | Verification | Evidence |
|-------|--------------|----------|
| FIX-1.4 line numbers | **ALL 6 CORRECT** | seed:252, a_max:229, lambda_rank:230, standardize_features:231, delta/alpha:56, np.clip:50 |
| FIX-1.3 Appendix C false claim | **CONFIRMED FALSE** | THM-C.2.1 = Slater's Strong Duality, NOT delta/alpha bounds. Line 215 of ch01_foundations is incorrect. |
| Production STRAT = purchases | **CONFIRMED** | `reward.py:34-38` increments `strat` only inside `if buys[idx]:` block |
| strat_exposure naming misleading | **CONFIRMED** | Tests use `strat_exposure` but production counts purchases, not exposure |
| CVR/RPC distinction | **CONFIRMED** | `zoosim/monitoring/metrics.py:38` has `compute_cvr()` = purchases/clicks. Tests use `compute_conversion_quality()` for GMV/clicks = RPC semantics |
| Ch05 line 916 STRAT definition | **CONFIRMED** | DEF-5.7 at line 916, EQ-5.7 defines Strategic as purchases |
| Files to modify exist | **ALL EXIST** | All 12 files in the plan exist and contain the claimed issues |

### ISSUES FOUND (Plan Gaps) --- Updated Per Review

| Issue | Finding | Resolution |
|-------|---------|------------|
| **CRITICAL: strat_exposure generator mismatch** | `lab_solutions.py:263-266` generates `strat_exposure` via Poisson ("top-10 exposure" proxy), but production `reward.py:34-38` counts STRAT only inside `if buys[idx]:` (purchases). Renaming without fixing generator shifts inconsistency from "name vs meaning" to "Ch01 vs rest of book". | **Fix both name AND generator.** New FIX-5.1a added below. |
| **Lab 1.1 canonical path mismatch** | `exercises_labs.md:9` uses `ZooplusSearchEnv`, but `ch01_lab_solutions.md:30` uses self-contained script with "may not be fully implemented" caveat. Different outputs for same seed. | **Standardize Lab 1.1 on the self-contained script** for Ch01 labs; keep `ZooplusSearchEnv` as an optional end-to-end check in the main chapter. The full env integration narrative starts in Ch05. FIX-3.1a added. |
| Production uses `strat` not `strat_purchases` | Field is `RewardBreakdown.strat` (abbreviated) | Keep plan's `strat_purchases` for clarity in pedagogical code |
| **CRITICAL: Δrank@k definition mismatch** | Ch01 defines Δrank@k as position-wise mismatch; Chapter 10 and `zoosim/monitoring/metrics.py:89-118` implement set-churn. These disagree even on simple swaps (e.g., [A,B,…]→[B,A,…]). | **Align Ch01 to set-churn** (recommended) and cross-reference Chapter 10 DEF-10.4; optionally mention position-wise as a separate variant. New FIX-1.8 added below. |
| **CRITICAL: syllabus mismatch (constraints + RPC)** | `docs/book/syllabus.md` references constraints in Ch8 and “CVR diagnostic” in Ch01 acceptance, both inconsistent with the planned RPC rename and Chapter 10 guardrails placement. | Add FIX-6.1 and FIX-6.2 to update `docs/book/syllabus.md`. |
| **CRITICAL: Lab 1.1 tasks inconsistent after canonical path change** | After switching Lab 1.1 to the self-contained script, tasks in `exercises_labs.md` that instruct perturbing `cfg.reward` / triggering `zoosim/dynamics/reward.py` no longer match the canonical lab path. | Extend FIX-3.1a to update Lab 1.1 task text: use the self-contained validation, and treat production-assertion probing as optional. |
| Plan line 94: reward.py:34 claim | Incomplete - line 34 is conditional, increment is line 38 | Minor - the logic spans 34-38 |
| Minor: misleading `reward.py:1` line refs | `compute_reward()` starts at `zoosim/dynamics/reward.py:42` (assert at 56). Some docs/tests reference `reward.py:1`. | Fold into FIX-4.4 and FIX-3.2: update to `reward.py:42-66`. |
| Minor: metrics.py DEF anchor typo | `zoosim/monitoring/metrics.py:10` references `[DEF-10.3]`, but Delta-Rank is `{#DEF-10.4}` in Ch10. | Add FIX-7.1 (code change). |

### CROSS-CHAPTER FIXES (Now In Scope)

These will be included in this PR:

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `ch05/ch05_relevance_features_reward.md` | 1286 | "Chapter 8" | -> "Chapter 10" |
| `ch05/ch05_relevance_features_reward.md` | 1729 | "Chapter 8" | -> "Chapter 10" |
| `ch05/ch05_relevance_features_reward.md` | 1793 | "Chapter 8" | -> "Chapter 10" |
| `ch06/discrete_template_bandits.md` | 2549 | "Chapter 8" | -> "Chapter 10" |
| `ch06/appendices.md` | 344 | "Chapter 8" | -> "Chapter 10" |
| `ch03/ch03_bellman_and_regret.md` | 427 | `E[STRAT] >= tau_STRAT` | Add clarifying note or rename to `E[Exposure_strategic]` |
| `appendix_c_convex_optimization.md` | 46 | `E[STRAT]` in constraint | Add clarifying note: "STRAT here denotes exposure, distinct from STRAT in EQ-1.2 (purchases)"

---

## Execution Order

### Phase 1: Code Files (Tests + Scripts)

These must change FIRST so doc references to actual output are accurate.

#### 1.1 Update `tests/ch01/test_reward_examples.py`

**FIX-4.1:** Rename `strat_exposure` -> `strat_purchases`
- Line 19: `strat_exposure: int` -> `strat_purchases: int`
- Line 19 docstring: "Number of strategic products in top-10" -> "Number of strategic purchases"
- Line 43: `outcome.strat_exposure` -> `outcome.strat_purchases`
- All `SessionOutcome(...)` constructor calls

**FIX-4.2:** Rename `compute_conversion_quality` -> `compute_rpc`
- Line 47: Function name
- Update docstring: "GMV per click (revenue per click)" not "conversion quality"
- Update all call sites and print labels

**FIX-4.3:** Fix test name and labels
- "CVR Diagnostic" -> "RPC Diagnostic" (for GMV/click contexts)
- Update print statements

**FIX-4.4:** Update stale docstring references
- Line 37: "strategic exposure" -> "strategic purchases"
- Line 39: `config.py:193` -> `config.py:195`
- Line 37 reference: `reward.py:1` -> `reward.py:42-66` (production `compute_reward` location)

#### 1.2 Update `scripts/ch01/lab_solutions.py`

**FIX-5.1:** Mirror test renames
- Lines 54, 60, 266, 271: `strat_exposure` -> `strat_purchases`
- Line 172: `compute_conversion_quality` -> `compute_rpc`
- Update all docstrings and print labels

**FIX-5.1a:** Fix `strat_purchases` generator to align with production semantics
- Location: `simulate_session()` function, lines 263-266
- **Current (broken):** Poisson-based "exposure proxy" unrelated to purchases:
  ```python
  strat_base = rng.poisson(2.0)
  strat_modifier = 1.0 + 0.5 * w_quality
  strat_exposure = max(0, min(10, int(strat_base * strat_modifier)))
  ```
- **Fixed (aligned with production):** Derive from actual `n_purchases`:
  ```python
  # Strategic purchases: fraction of purchases that are strategic
  # Hardcoded 30% base + quality correlation (keeps Ch01 focused on reward concepts)
  strategic_prob = 0.3 + 0.2 * np.clip(w_quality, -1, 1)  # 10%-50% range
  strat_purchases = sum(rng.random() < strategic_prob for _ in range(n_purchases))
  ```
- This ensures: (1) 0 purchases -> 0 strategic, (2) semantics match production

**FIX-5.2:** Fix output labels
- Line 327: "strategic products in top-10" -> "strategic purchases"
- Lines 472-487, 624: "CVR" -> "RPC" where computing GMV/click

#### 1.3 Update `zoosim/monitoring/metrics.py`

**FIX-7.1:** Correct Delta-Rank definition anchor reference
- Line 10: `[DEF-10.3]` -> `[DEF-10.4]` (Delta-Rank@k definition)

#### 1.4 Verify Phase 1
```bash
pytest tests/ch01/test_reward_examples.py -v
python scripts/ch01/lab_solutions.py --all  # Run to completion
```

---

### Phase 2: Documentation Files

#### 2.1 Update `docs/book/ch01/ch01_foundations_revised_math+pedagogy_v3.md`

**FIX-1.1:** STRAT Semantics Alignment
- After EQ-1.2: Add "Two strategic quantities" clarification note
- Clarify STRAT(omega) = strategic **purchases** in reward
- Section 1.9: Ensure constraint uses `E[Exposure_strategic] >= tau_strat`
- Add note: pedagogical `strat_purchases` matches production `RewardBreakdown.strat` (abbreviated), i.e., same quantity

**FIX-1.2:** CVR/RPC Terminology
- Section 1.2.1: CVR := purchases/clicks
- REM-1.2.1: Change "CVR = GMV/click" to "RPC = GMV/click"
- Production Checklist (~line 1148): Clarify which metric

**FIX-1.3:** Remove Incorrect Appendix C Claim
- Location: ~line 215
- Remove reference to THM-C.2.1 deriving delta/alpha bounds
- Reframe as engineering heuristic

**FIX-1.4:** Update Line Number References
- Update all Code-Config boxes to correct line numbers:
  - seed: 252, a_max: 229, lambda_rank: 230, standardize_features: 231
  - delta/alpha assertion: reward.py:56
  - np.clip: search_env.py:50

**FIX-1.5:** OPE Continuous Actions Clarification
- Section 1.7.1-1.7.4: Add density ratio remark for continuous actions

**FIX-1.6:** End-of-Chapter Lab Linkage
- Add explicit links to companion files after Exercises section

**FIX-1.7:** CM2 Floor Wiring Status Note
- Update Code-Config note to state cm2_floor is placeholder
- Reference Chapter 10 for actual enforcement

**FIX-1.8:** Δrank@k Definition Alignment (Ch01 ↔ Ch10 ↔ code)
- Replace Ch01 position-wise Δrank@k definition with the set-churn definition used in Chapter 10 (DEF-10.4) and `compute_delta_rank_at_k` (`zoosim/monitoring/metrics.py:89-118`)
- Add a brief note: position-wise mismatch is a different metric; if kept, label it explicitly as a variant (and do not call it “Delta-Rank@k” without qualification)

#### 2.2 Update `docs/book/ch01/ch01_lab_solutions.md`

**FIX-2.1:** Remove "Not Implemented" Caveat
- Line 30: Remove "simulator may not be fully implemented"

**FIX-2.2:** Production Pattern Mismatch
- Lines 149-165: Replace with accurate production API description
- Show correct signature: `compute_reward(*, ranking, clicks, buys, catalog, config)`
- Show correct return type: `Tuple[float, RewardBreakdown]`
- Show correct validation: `assert 0.01 <= ratio <= 0.10`

**FIX-2.3:** Chapter Reference Error
- Line 231: "Chapter 8" -> "Chapter 10"

**FIX-2.4:** STRAT Output Label
- "strategic products in top-10" -> "strategic purchases in session"

**FIX-2.5:** CVR -> RPC Labels
- Replace "CVR (GMV/click)" with "RPC (GMV/click)" throughout

**FIX-2.6:** Doc Code Blocks
- Lines 89, 94, 163, 208, 211: `strat_exposure` -> `strat_purchases`
- Lines 192, 241, 245: `compute_conversion_quality` -> `compute_rpc`

**FIX-2.7:** Remove Strategic Exposure Violation Section
- Lines 184-231: Replace entire section with simplified reference to test file

#### 2.3 Update `docs/book/ch01/exercises_labs.md`

**FIX-3.1:** Lab 1.2 Pytest Invocation
- Update invocation: `pytest tests/ch01/test_reward_examples.py -v`
- Capture and paste REAL output after Phase 1 changes

**FIX-3.1a:** Standardize Lab 1.1 on self-contained script (canonical path fix)
- Lines 9-21: Replace `ZooplusSearchEnv` code block with import from `scripts/ch01/lab_solutions.py`
- Add note: "Ch01 labs use the self-contained script; the main chapter includes an optional env smoke test; Ch05 introduces the full `ZooplusSearchEnv` integration narrative."
- Rationale: Ch01 focuses on reward concepts; full simulator integration is Ch05 scope
- Regenerate "actual output" blocks from canonical script after FIX-5.1a applied
- Update Lab 1.1 tasks to match the canonical script path:
  - Task 1: Recompute reward using the self-contained `RewardConfig` / outcome printed by `lab_1_1_reward_aggregation` (not `cfg.reward` from `ZooplusSearchEnv`)
  - Task 2: Replace “perturb `cfg.reward.delta_clicks` until the assertion fires” with the self-contained `validate_delta_alpha_bound` workflow; add an optional extension: reproduce the same bound failure via the production assertion in `zoosim/dynamics/reward.py`
  - Task 3: Keep as-is (propagate findings back into Chapter 1 text)

**FIX-3.2:** Update Code Reference Line Numbers
- Line 105: `config.py:193` -> `config.py:195`
- Line 436: `search_env.py:47` -> `search_env.py:50`
- Line 103: `reward.py:1` -> `reward.py:42-66`

**FIX-3.3:** CVR -> RPC in Lab 1.5
- Rename to "RPC (Revenue per Click)"

**FIX-3.4:** Add Rank-Stability Lab Preview (Lab 1.8)
- Add honest preview showing lambda_rank exists but is not wired

**FIX-3.5:** Doc Code Blocks
- Lines 44, 48, 52, 87, 114-117, 154, 171-172, 215, 218, 233-234:
  - `strat_exposure` -> `strat_purchases`
  - `compute_conversion_quality` -> `compute_rpc`

**FIX-3.6:** Remove Strategic Exposure Violation Task
- Lines 65-67: Replace with aligned tasks

**FIX-3.7:** Add Exposure-Floors Status Note
- Add note that exposure_floors exists but is not wired

---

### Phase 3: Cross-Chapter + Book-Level Fixes

#### 3.1 Update `docs/book/ch05/ch05_relevance_features_reward.md`

- Line 1286: "In Chapter 8 (Constraints)" -> "In Chapter 10 (Robustness & Guardrails)"
- Line 1729: "(Chapter 8)" -> "(Chapter 10)"
- Line 1793: "**Chapter 8**: Constrained optimization" -> "**Chapter 10**: Robustness and Guardrails"

#### 3.2 Update `docs/book/ch06/discrete_template_bandits.md`

- Line 2549: "**Chapter 8**: Hard constraints" -> "**Chapter 10**: Hard constraints"

#### 3.3 Update `docs/book/ch06/appendices.md`

- Line 344: "Chapter 8 adds hard constraints" -> "Chapter 10 adds hard constraints"

#### 3.4 Update `docs/book/ch03/ch03_bellman_and_regret.md`

- Line 427: Add clarifying note after `E[STRAT] >= tau_STRAT`:
  ```
  > **Notation (applies throughout Section 3.6):** In constraint contexts, STRAT
  > denotes strategic *exposure* (items shown in ranking), which differs from
  > STRAT in reward functions (EQ-1.2) where it counts strategic *purchases*.
  ```

#### 3.5 Update `docs/book/appendix_c_convex_optimization.md`

- Line 46: Add clarifying note with explicit scope after strategic exposure floor definition:
  ```
  > **Notation (applies throughout this appendix):** Here STRAT denotes strategic
  > exposure (items shown in ranking), distinct from STRAT in reward function
  > EQ-1.2 which counts strategic purchases.
  ```

#### 3.6 Update `docs/book/syllabus.md`

**FIX-6.1:** Ch01 acceptance terminology
- Line 18: "check CVR diagnostic" -> "check RPC diagnostic"

**FIX-6.2:** Constraint chapter placement
- Line 142: "Ch8 (constraints)" -> "Ch10 (constraints)"

---

### Phase 4: Verification

Run all verification commands from the plan:

```bash
# 1. STRAT Terminology Check
grep -n "STRAT\|strategic" docs/book/ch01/*.md | grep -i "reward\|EQ-1.2" | head -10
grep -n "EQ-1.3b\|exposure.*floor\|floor.*exposure" docs/book/ch01/*.md | head -10

# 2. CVR/RPC Terminology Check (should be EMPTY)
grep -rn "CVR" docs/book/ch01/ tests/ch01/ scripts/ch01/ | grep -iE "gmv|revenue"

# 3. Line Number Verification
grep -n "seed" zoosim/core/config.py | grep "252"
grep -n "a_max" zoosim/core/config.py | grep "229"
grep -n "lambda_rank" zoosim/core/config.py | grep "230"
grep -n "assert.*ratio" zoosim/dynamics/reward.py | grep "56"
grep -n "np.clip" zoosim/envs/search_env.py | grep "50"

# 4. Test Execution
pytest tests/ch01/test_reward_examples.py -v
python scripts/ch01/lab_solutions.py --all  # Run to completion; capture excerpts separately for docs

# 5. No strat_exposure Remnants (should be EMPTY)
grep -rn "strat_exposure" docs/book/ch01/ tests/ch01/ scripts/ch01/

# 6. No compute_conversion_quality Remnants (should be EMPTY)
grep -rn "compute_conversion_quality" docs/book/ch01/ tests/ch01/ scripts/ch01/

# 7. No Strategic Exposure Violation Sections (should be EMPTY)
grep -rn "strategic.*exposure.*violation\|test_strategic_exposure" docs/book/ch01/

# 8. Cross-chapter "Chapter 8" references fixed (should be EMPTY)
grep -rn "Chapter 8" docs/book/ch01/ docs/book/ch03/ docs/book/ch05/ docs/book/ch06/ docs/book/syllabus.md docs/book/appendix_c*.md | grep -i "constraint\|hard\|floor"

# 9. Metrics docstring references correct DEF anchor
grep -n "DEF-10\\.[0-9]" zoosim/monitoring/metrics.py | grep "Delta-Rank"
```

---

## Files to Modify (Complete List)

### Ch01 Core Files (Original Scope + Review Additions)

| File | Fix Count | Type |
|------|-----------|------|
| `tests/ch01/test_reward_examples.py` | 4 | Code |
| `scripts/ch01/lab_solutions.py` | 3 (includes FIX-5.1a generator fix) | Code |
| `docs/book/ch01/ch01_foundations_revised_math+pedagogy_v3.md` | 8 | Doc |
| `docs/book/ch01/ch01_lab_solutions.md` | 7 | Doc |
| `docs/book/ch01/exercises_labs.md` | 8 (includes FIX-3.1a canonical path) | Doc |

### Cross-Chapter Files (Extended Scope)

| File | Fix Count | Type |
|------|-----------|------|
| `docs/book/ch05/ch05_relevance_features_reward.md` | 3 | Doc |
| `docs/book/ch06/discrete_template_bandits.md` | 1 | Doc |
| `docs/book/ch06/appendices.md` | 1 | Doc |
| `docs/book/ch03/ch03_bellman_and_regret.md` | 1 | Doc |
| `docs/book/appendix_c_convex_optimization.md` | 1 | Doc |

### Global Alignment Files (Book ↔ Code)

| File | Fix Count | Type |
|------|-----------|------|
| `zoosim/monitoring/metrics.py` | 1 | Code |
| `docs/book/syllabus.md` | 2 | Doc |

**Total: 40 fixes across 12 files** (34 original + FIX-5.1a + FIX-3.1a + FIX-1.8 + FIX-6.1–6.2 + FIX-7.1)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking tests | Run pytest after each file change |
| Stale output blocks in docs | Capture fresh output after code changes |
| Cross-chapter inconsistency | Phase 3 updates + grep verification (#8) for constraint-related chapter refs |
| Production code drift | Line numbers verified against current code |

---

## Success Criteria

1. All verification grep commands return expected results
2. `pytest tests/ch01/test_reward_examples.py -v` passes
3. `scripts/ch01/lab_solutions.py --all` runs without errors
4. No `strat_exposure` or `compute_conversion_quality` remnants in Ch01 files
5. No CVR used for GMV/click contexts (RPC used instead)
6. All line number references accurate (including `reward.py:42-66`, `config.py:195`, `search_env.py:50`)
7. No "Chapter 8" references for constraints/floors/guardrails in docs (including `docs/book/syllabus.md`)
8. STRAT notation clarified in Ch03 and Appendix C (with explicit scope statements)
9. Ch01 Δrank@k definition matches Chapter 10 + `compute_delta_rank_at_k`
10. `docs/book/syllabus.md` aligned with Ch01 and Chapter 10 constraint placement
11. `zoosim/monitoring/metrics.py` references the correct Delta-Rank definition anchor
12. `strat_purchases` in `lab_solutions.py` derived from `n_purchases` (0 purchases -> 0 strategic)
13. `exercises_labs.md` Lab 1.1 uses self-contained script pattern (not `ZooplusSearchEnv`) and tasks match that canonical path
