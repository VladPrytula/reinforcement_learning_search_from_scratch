# Ch01 Alignment Plan (Merged)

**Philosophy:** Align Chapter 1 with canonical definitions from the rest of the book, not vice versa. Ch01 is the outlier; Ch05, production code, and later chapters already agree on terminology.

**Origin:** Merged from `ch1_update_plan.md` (alignment approach) + `cheerful-weaving-fog.md` (specific fixes).

---

## Executive Summary

| Issue Category | Count | Key Changes |
|----------------|-------|-------------|
| Terminology alignment (STRAT, CVR/RPC) | 6 fixes | Match Ch05 + production semantics |
| Code reference updates (line numbers) | 2 fixes | Use verified line numbers |
| False claims / outdated content | 3 fixes | Remove Appendix C claim, caveats |
| Runnable artifact sync | 5 fixes | Tests + scripts + doc code blocks match |
| Conflicting content removal | 2 fixes | Remove strategic-exposure-violation sections |
| Missing content | 4 fixes | Lab linkage, rank-stability preview, exposure-floors status, cm2-floor status |

**Total: 22 atomic fixes across 5 files**

---

## Canonical Definitions (Source of Truth)

These definitions come from Ch05 and production code. Ch01 must align to them.

| Term | Canonical Definition | Source |
|------|---------------------|--------|
| **STRAT** | Strategic **purchases** (count of strategic products bought) | `zoosim/dynamics/reward.py:34` (increments only when `buys[idx]`), Ch05 #EQ-5.7 |
| **Exposure floors** | Separate constraint on strategic products **shown** in top-k | `zoosim/core/config.py:233` (`exposure_floors`), controllable via ranking |
| **CVR** | Conversion rate = purchases / clicks | Industry standard, `zoosim/monitoring/metrics.py:38` |
| **RPC** | Revenue per click = GMV / clicks | Clickbait diagnostic metric |

---

## Files to Modify

| File | Type | Purpose |
|------|------|---------|
| `docs/book/ch01/ch01_foundations_revised_math+pedagogy_v3.md` | Doc | Main chapter narrative |
| `docs/book/ch01/exercises_labs.md` | Doc | Student-facing labs |
| `docs/book/ch01/ch01_lab_solutions.md` | Doc | Worked solutions narrative |
| `tests/ch01/test_reward_examples.py` | Code | Regression tests |
| `scripts/ch01/lab_solutions.py` | Code | Runnable lab driver |

---

## Atomic Fixes

### File 1: ch01_foundations_revised_math+pedagogy_v3.md

**FIX-1.1: STRAT Semantics Alignment**
- **Issue:** Ch01 ambiguously uses "STRAT" for both reward and constraint, conflating strategic exposure with strategic purchases
- **Fix:** Align with canonical definition:
  - **EQ-1.2 (reward):** STRAT(omega) = strategic **purchases** (matches Ch05 #EQ-5.7, production code)
  - **EQ-1.3b (constraint):** Explicitly label as `Exposure_strategic(x,a)` = strategic items **shown** in top-k (separate controllable quantity)
- **Key insight:** These are different because:
  - Purchases depend on user latent preferences (stochastic, not directly controllable)
  - Exposure depends on ranking (deterministic given action, directly controllable)
- **Locations to update:**
  - After EQ-1.2: Add "Two strategic quantities" clarification note
  - Section 1.9 Lagrangian: Ensure constraint uses `E[Exposure_strategic] >= tau_strat`, not STRAT
  - Business tensions list: Clarify this distinction
- **DO NOT:** Change `STRAT` symbol in EQ-1.2 (just clarify it means purchases)

**FIX-1.2: CVR/RPC Terminology**
- **Issue:** CVR defined inconsistently (purchases/clicks vs GMV/clicks)
- **Fix:** Apply canonical naming:
  - `CVR := purchases/clicks` (conversion rate, industry standard)
  - `RPC := GMV/clicks` (revenue per click, clickbait diagnostic)
- **Locations:**
  - Section 1.2.1 (main CVR discussion)
  - REM-1.2.1 (clickbait diagnostic): Change "CVR = GMV/click" to "RPC = GMV/click"
  - Production Checklist (line ~1148): "Monitor CVR" context should clarify which metric
- **Verification:** `grep -rn "CVR" docs/book/ch01/ | grep -iE "gmv|revenue"` should return EMPTY after fix

**FIX-1.3: Remove Incorrect Appendix C Claim**
- **Issue:** Claims "Appendix C derives principled bounds on delta/alpha [THM-C.2.1]" -- FALSE
- **Actual:** Appendix C covers Lagrangian duality, NOT delta/alpha numerical bounds
- **Fix:** Reframe as engineering heuristic:
  ```
  The delta/alpha ratio in [0.01, 0.10] is an engineering guideline based on
  typical GMV/click magnitude ratios and empirical clickbait thresholds,
  not a theoretically derived bound.
  ```
- **Location:** Near REM-1.2.1, Code-Config box

**FIX-1.4: Update Line Number References**
- **Issue:** Stale file:line references throughout
- **Verified Current Locations:**
  | Reference | Old | Correct | Symbol |
  |-----------|-----|---------|--------|
  | seed | config.py:231 | config.py:252 | `SimulatorConfig.seed` |
  | a_max | config.py:208 | config.py:229 | `ActionConfig.a_max` |
  | standardize_features | config.py:210 | config.py:231 | `ActionConfig.standardize_features` |
  | delta/alpha assertion | reward.py:25 | reward.py:56 | `assert 0.01 <= ratio <= 0.10` |
  | np.clip | search_env.py:47 | search_env.py:50 | `np.clip(...)` |
  | lambda_rank | (new) | config.py:230 | `ActionConfig.lambda_rank` |
- **Fix:** Update all Code-Config boxes and Production Checklist
- **Format:** `module.py:LINE (symbol_name)` for robustness against future drift

**FIX-1.5: OPE Continuous Actions Clarification**
- **Issue:** Mentions propensity ratios but doesn't clarify density ratio requirement for continuous actions
- **Fix:** Add brief remark in OPE preview section:
  ```
  For continuous actions a in [-a_max, a_max]^K, importance ratios become
  density ratios pi_e(a|x)/pi_b(a|x), requiring stochastic policies with
  full support. Variance issues are more severe than discrete case (Ch9).
  ```
- **Location:** Section 1.7.1-1.7.4 (OPE preview)

**FIX-1.6: End-of-Chapter Lab Linkage**
- **Issue:** Chapter ends with `## Exercises` but doesn't explicitly link to companion files
- **Fix:** Add after Exercises section:
  ```markdown
  ## Exercises & Labs

  Hands-on implementations and worked solutions are in companion files:
  - **Labs:** `docs/book/ch01/exercises_labs.md` -- runnable code exercises
  - **Solutions:** `docs/book/ch01/ch01_lab_solutions.md` -- worked solutions with discussion
  - **Tests:** `tests/ch01/test_reward_examples.py` -- regression tests for code examples
  ```

**FIX-1.7: CM2 Floor Wiring Status Note** [NEW]
- **Issue:** Ch01 implies `ActionConfig.cm2_floor` implements the CM2-floor constraint #EQ-1.3a, but `cm2_floor` is not enforced by `ZooplusSearchEnv` or `compute_reward()` in Chapter 1.
- **Fix:** Update the Chapter 1 "Code ↔ Config (constraints)" note to explicitly state:
  - `ActionConfig.cm2_floor` is a configuration placeholder in Chapter 1 (the simulator does not enforce it).
  - **Chapter 10** treats CM2 floors as a **hard feasibility** pattern at action selection time (reject/resample; Exercise 10.3). The monitor layer in §10.5 is reward/stability oriented and does not evaluate CM2 in the reference implementation, even though CM2-related config fields exist.
  - **Chapter 14** (with Appendix C theory) is the place for **soft** constraint handling via learned Lagrange multipliers in a primal–dual CMDP formulation.

---

### File 2: ch01_lab_solutions.md

**FIX-2.1: Remove "Not Implemented" Caveat**
- **Issue:** Line 30 says "simulator may not be fully implemented"
- **Reality:** `zoosim/` is implemented (envs, dynamics, core, etc.)
- **Fix:** Remove caveat entirely

**FIX-2.2: Production Pattern Mismatch**
- **Issue:** Lab 1.1 Task 3 shows production pattern that doesn't match `zoosim/dynamics/reward.py`:
  - Wrong API: Shows `compute_reward(outcome: SessionOutcome, cfg: RewardConfig)` but production takes `(ranking, clicks, buys, catalog, config)`
  - Wrong validation: Shows `if delta_alpha > 0.10: raise ValueError(...)` but production uses `assert 0.01 <= ratio <= 0.10` (BOTH bounds)
  - Wrong return type: Shows `-> float` but production returns `Tuple[float, RewardBreakdown]`
- **Fix:** Replace snippet (around lines 149-165) with accurate representation:
  ```python
  # Production implementation pattern (zoosim/dynamics/reward.py:42-66)
  # Note: Production API differs from pedagogical SessionOutcome/BusinessWeights
  # for integration with simulator internals.
  #
  # Signature: compute_reward(*, ranking, clicks, buys, catalog, config)
  # Returns:   Tuple[float, RewardBreakdown]
  #
  # Key differences from pedagogical version:
  # 1. Takes raw arrays (ranking, clicks, buys) not SessionOutcome
  # 2. Computes components internally via _compute_components()
  # 3. Validates BOTH bounds: 0.01 <= delta/alpha <= 0.10
  # 4. Returns breakdown alongside scalar for diagnostics

  # The bound validation (reward.py:52-59):
  alpha = float(cfg.alpha_gmv)
  ratio = float("inf") if alpha == 0.0 else float(cfg.delta_clicks) / alpha
  assert 0.01 <= ratio <= 0.10, (
      f"Engagement weight outside safe range [0.01, 0.10]: "
      f"delta/alpha = {ratio:.3f}. Adjust RewardConfig to avoid clickbait optimization."
  )
  ```
- **Note:** Message must match production EXACTLY (includes "optimization" at end)

**FIX-2.3: Chapter Reference Error**
- **Issue:** References "Chapter 8" for constraint handling
- **Fix:** Change to "Chapter 10" (Robustness & Guardrails) per actual book structure
- **Note:** Ch08 = Policy Gradients, Ch10 = Guardrails/Constraints

**FIX-2.4: STRAT Output Label**
- **Issue:** Output shows "strategic products in top-10" (exposure semantics)
- **Fix:** Change to "strategic purchases in session" (matches canonical definition)

**FIX-2.5: CVR -> RPC Labels**
- **Issue:** Labels GMV/click as "CVR"
- **Fix:** Replace "CVR (GMV/click)" with "RPC (GMV/click)" throughout

**FIX-2.6: Doc Code Blocks Use strat_exposure** [NEW]
- **Issue:** Code blocks in doc still use `strat_exposure` and `compute_conversion_quality`
- **Locations:**
  - Line 89: `SessionOutcome(gmv=112.70, cm2=22.54, strat_exposure=3, clicks=4)`
  - Line 94: `cfg.gamma_strat * outcome.strat_exposure`
  - Line 163: `cfg.gamma_strat * outcome.strat_exposure`
  - Line 192: `compute_conversion_quality` import
  - Line 208: `strat_exposure=0` in SessionOutcome
  - Line 211: `strat_exposure=5` in SessionOutcome
  - Line 241: `compute_conversion_quality(baseline)`
  - Line 245: `compute_conversion_quality(clickbait)`
- **Fix:** Replace all `strat_exposure` → `strat_purchases` and `compute_conversion_quality` → `compute_rpc` in doc code blocks

**FIX-2.7: Remove Strategic Exposure Violation Section** [NEW]
- **Issue:** Lines 184-231 propose `test_strategic_exposure_violation()` that:
  1. Treats `strat_exposure` as STRAT (conflicts with STRAT=purchases alignment)
  2. References nonexistent test fixtures
  3. Says "In Chapter 8, we'll handle this via Lagrangian constraint optimization" (wrong chapter)
- **Fix:** Replace entire section with:
  ```markdown
  ### Solution

  The test file `tests/ch01/test_reward_examples.py` contains core validations
  for reward computation and RPC diagnostics. Constraint monitoring (exposure
  floors, rank stability) is treated operationally in Chapter 10 (guardrails, monitoring, hard feasibility filters) and optimized via primal–dual CMDPs in Chapter 14 (Appendix C).

  **Key tests to understand:**
  - `test_basic_reward_comparison`: Validates EQ-1.2 arithmetic
  - `test_profitability_weighting`: Shows how weights change optimal strategy
  - `test_rpc_diagnostic`: Implements clickbait detection from REM-1.2.1
  - `test_delta_alpha_bounds`: Validates engagement weight bounds
  ```
- **Rationale:** Don't propose tests that conflict with canonical definitions or don't exist

---

### File 3: exercises_labs.md

**FIX-3.1: Lab 1.2 Pytest Invocation**
- **Issue:** Snippet calls `pytest.main(["-k", "reward_section_examples"])` with fabricated output
- **Fix:**
  1. Update invocation: `pytest tests/ch01/test_reward_examples.py -v`
  2. Capture and paste REAL output (run actual tests)
  3. Update test function names to match actual file

**FIX-3.2: Update Code Reference Line Numbers**
- **Issue:** Stale line numbers in Code-Config boxes
- **Specific locations:**
  | File:Line | Reference | Actual |
  |-----------|-----------|--------|
  | `exercises_labs.md:105` | `config.py:193` | RewardConfig at line **195** |
  | `exercises_labs.md:436` | `search_env.py:47` | np.clip at line **50** |
- **Fix:** Update to correct line numbers + add symbol names for robustness

**FIX-3.3: CVR -> RPC in Lab 1.5**
- **Issue:** Lab 1.5 defines "CVR" as GMV/click
- **Fix:** Rename to "RPC (Revenue per Click)" and add note that CVR = conversions/click

**FIX-3.4: Add Rank-Stability Lab Preview (Lab 1.8)**
- **Issue:** Syllabus requires "verify rank-stability penalty wiring" but config knob exists without wiring
- **Fix:** Add honest preview lab:
  ```python
  # Lab 1.8: Rank-Stability Configuration Preview
  #
  # Goal: Verify that the rank-stability knob exists in ActionConfig
  # and understand its role. In Chapter 1 this knob is a configuration placeholder;
  # it becomes operational in the primal–dual CMDP pathway (Chapter 14, Appendix C).

  from zoosim.core.config import SimulatorConfig

  cfg = SimulatorConfig()

  # The rank-stability regularization knob exists in ActionConfig
  print(f"lambda_rank = {cfg.action.lambda_rank}")
  assert hasattr(cfg.action, 'lambda_rank'), "lambda_rank missing from ActionConfig"
  assert cfg.action.lambda_rank >= 0.0, "lambda_rank must be non-negative"

  # IMPORTANT: As of Chapter 1, lambda_rank is a configuration placeholder.
  # Chapter 10 introduces the production guardrail viewpoint (stability metrics,
  # monitoring, fallback). Chapter 14 introduces soft constraint optimization
  # (primal–dual learning of multipliers such as lambda_rank).
  #
  # Preview of the soft-constraint penalty idea (Chapter 14):
  #   r_t^(lambda) = r_t - lambda_rank * DeltaRankAtK_t

  print("\n[INFO] lambda_rank exists but is not used by the Chapter 1 simulator.")
  print("       See Chapter 10 for stability metrics + guardrails, and Chapter 14 for primal–dual CMDP optimization.")
  print("       Current config value: lambda_rank =", cfg.action.lambda_rank)
  ```
- **Rationale:** Honest about current state (config exists, wiring doesn't)

**FIX-3.5: Doc Code Blocks Use strat_exposure** [NEW]
- **Issue:** Code blocks in exercises_labs.md still use `strat_exposure` and `compute_conversion_quality`
- **Locations:**
  - Line 44: `compute_conversion_quality,` import
  - Line 48: `strat_exposure=1` in SessionOutcome
  - Line 52: `compute_conversion_quality(outcome_a)`
  - Line 87: `strat_exposure: int` in class definition
  - Line 114-117: `strat_exposure=1`, `strat_exposure=3` in examples
  - Line 154: `strat_exposure: int` in class definition
  - Line 171-172: `strat_exposure=1`, `strat_exposure=3` in examples
  - Line 215: `strat_exposure: int` in class definition
  - Line 218: `compute_conversion_quality` function definition
  - Line 233-234: `strat_exposure=1`, `strat_exposure=3` in examples
- **Fix:** Replace all `strat_exposure` → `strat_purchases` and `compute_conversion_quality` → `compute_rpc`

**FIX-3.6: Remove Strategic Exposure Violation Task** [NEW]
- **Issue:** Lines 65-67 task says "Extend `tests/ch01/test_reward_examples.py` with at least one new fixture representing a strategic exposure violation"
- **Problems:**
  1. Conflicts with STRAT=purchases alignment
  2. No such test exists or should exist in the aligned codebase
  3. Encourages students to create conflicting code
- **Fix:** Replace task with:
  ```markdown
  **Tasks**
  1. Run `pytest tests/ch01/test_reward_examples.py -v` and verify all tests pass.
  2. Read the test assertions and verify they match [EQ-1.2] and [REM-1.2.1].
  3. (Optional) Add a test for a new weight configuration of your choice.
  ```
- **Rationale:** Tasks should align with canonical definitions, not conflict with them

**FIX-3.7: Add Exposure-Floors Status Note** [NEW]
- **Issue:** Like `lambda_rank`, `exposure_floors` exists in config but is NOT wired anywhere
- **Verification:** `grep -rn "exposure_floors" zoosim/` returns ONLY `config.py:233`
- **Fix:** Add note to Lab 1.8 or create Lab 1.9:
  ```python
  # Similarly, exposure_floors exists but is not used by the Chapter 1 simulator:
  print(f"exposure_floors = {cfg.action.exposure_floors}")
  # Chapter 10 treats such constraints as hard feasibility filters; Chapter 14 treats them as soft constraints via primal–dual optimization.
  ```
- **Rationale:** Honest about all "knob exists, wiring deferred" situations

---

### File 4: tests/ch01/test_reward_examples.py

**FIX-4.1: Rename strat_exposure -> strat_purchases**
- **Locations:**
  - Line 19: `strat_exposure: int` -> `strat_purchases: int`
  - Line 19 docstring: "Number of strategic products in top-10" -> "Number of strategic purchases"
  - Line 43: `outcome.strat_exposure` -> `outcome.strat_purchases`
  - All `SessionOutcome(...)` constructor calls
- **Rationale:** Aligns with canonical STRAT = purchases semantics

**FIX-4.2: Rename compute_conversion_quality -> compute_rpc**
- **Location:** Line 47
- **Update docstring:** "GMV per click (revenue per click)" not "conversion quality"
- **Update all call sites and print labels**

**FIX-4.3: Fix Test 3 Name and Labels**
- **Current:** "CVR Diagnostic (Clickbait Detection)"
- **Fix:** "RPC Diagnostic (Clickbait Detection)"
- **Update all print statements that say "CVR" when computing GMV/click**

**FIX-4.4: Update Stale Production References in Docstring** [NEW]
- **Issue:** `compute_reward()` docstring references the wrong semantics and line numbers:
  - `tests/ch01/test_reward_examples.py:37` says "strategic exposure"
  - `tests/ch01/test_reward_examples.py:39` points to `zoosim/core/config.py:193` (stale; RewardConfig is at line 195)
- **Fix:** Update the docstring to say "strategic purchases" and point to `zoosim/core/config.py:195` (RewardConfig).

**FIX-4.5: (Optional) Add True CVR Helper**
- **Add:** Simple `compute_cvr(purchases, clicks)` function with one test
- **Purpose:** Pin the acronym meaning (conversions/click) without changing core reward example
- **Rationale:** If students see both CVR and RPC, they understand the distinction

---

### File 5: scripts/ch01/lab_solutions.py

**FIX-5.1: Mirror Test Renames**
- `strat_exposure` -> `strat_purchases` (lines 54, 60, 266, 271, and all print/compute lines)
- `compute_conversion_quality` -> `compute_rpc` (line 172)
- Update all docstrings and print labels

**FIX-5.2: Fix Output Labels**
- Line 327: "strategic products in top-10" -> "strategic purchases"
- Lines 472-487, 624: "CVR" -> "RPC" where computing GMV/click

---

## Verification Steps (Post-Implementation)

### 1. STRAT Terminology Check
```bash
# Reward context (EQ-1.2): should say "purchases" not "exposure"
grep -n "STRAT\|strategic" docs/book/ch01/*.md | grep -i "reward\|EQ-1.2" | head -10

# Constraint context (EQ-1.3b): should explicitly say "exposure"
grep -n "EQ-1.3b\|exposure.*floor\|floor.*exposure" docs/book/ch01/*.md | head -10
```

### 2. CVR/RPC Terminology Check
```bash
# CVR should NEVER be GMV/clicks
grep -rn "CVR" docs/book/ch01/ tests/ch01/ scripts/ch01/ | grep -iE "gmv|revenue"
# Expected: EMPTY

# RPC should be GMV/clicks
grep -rn "RPC" docs/book/ch01/ tests/ch01/ scripts/ch01/ | head -10
# Expected: All RPC = GMV/click contexts
```

### 3. Line Number Verification
```bash
grep -n "seed" zoosim/core/config.py | grep "252"
grep -n "a_max" zoosim/core/config.py | grep "229"
grep -n "lambda_rank" zoosim/core/config.py | grep "230"
grep -n "assert.*ratio" zoosim/dynamics/reward.py | grep "56"
grep -n "np.clip" zoosim/envs/search_env.py | grep "50"
```

### 4. Test Execution
```bash
# All tests should pass
pytest tests/ch01/test_reward_examples.py -v

# Script should run without errors
python scripts/ch01/lab_solutions.py --all 2>&1 | head -50
```

### 5. No strat_exposure Remnants
```bash
grep -rn "strat_exposure" docs/book/ch01/ tests/ch01/ scripts/ch01/
# Expected: EMPTY (all changed to strat_purchases)
```

### 6. No compute_conversion_quality Remnants
```bash
grep -rn "compute_conversion_quality" docs/book/ch01/ tests/ch01/ scripts/ch01/
# Expected: EMPTY (all changed to compute_rpc)
```

### 7. No Strategic Exposure Violation Sections
```bash
grep -rn "strategic.*exposure.*violation\|test_strategic_exposure" docs/book/ch01/
# Expected: EMPTY (conflicting sections removed)
```

### 8. Output Block Accuracy
```bash
# Run actual pytest and compare to documented output
pytest tests/ch01/test_reward_examples.py -v 2>&1 | head -30
# Compare to exercises_labs.md Lab 1.2 Output: block
```

---

## Execution Order

### Phase 1: Code Files (tests + scripts)
These must change first so doc references to actual output are accurate.

1. **FIX-4.1 to FIX-4.4:** Update `tests/ch01/test_reward_examples.py`
2. **FIX-5.1 to FIX-5.2:** Update `scripts/ch01/lab_solutions.py`
3. Run tests to verify: `pytest tests/ch01/ -v`

### Phase 2: Documentation Files
Now update docs to match the aligned code.

4. **FIX-1.1 to FIX-1.6:** Update main chapter `ch01_foundations_revised_math+pedagogy_v3.md`
5. **FIX-2.1 to FIX-2.7:** Update `ch01_lab_solutions.md` (includes code block renames + section removal)
6. **FIX-3.1 to FIX-3.7:** Update `exercises_labs.md` (includes code block renames + task removal + exposure-floors note)

### Phase 3: Verification
7. Run all verification steps
8. (Optional) PDF compilation test

---

## Cross-Chapter Impact Analysis

### These Fixes Are SAFE
All fixes bring Ch01 INTO alignment with existing canonical definitions:

| Fix | Risk | Analysis |
|-----|------|----------|
| STRAT = purchases | SAFE | Ch05 already defines this way (line 916), production code agrees |
| Exposure = separate | SAFE | Config already has `exposure_floors` as separate concept |
| CVR/RPC split | SAFE | Ch00 uses CVR correctly, this aligns Ch01 |
| Line numbers | SAFE | Ch01-specific references |
| Appendix C claim | SAFE | Localized correction |

### Follow-Up Tasks (Out of Scope)
These were exposed by the review but are NOT Ch01 fixes:

| File | Issue | Fix Needed |
|------|-------|------------|
| `docs/book/ch05/ch05_relevance_features_reward.md:1286` | "Chapter 8 (Constraints)" | -> "Chapter 10" |
| `docs/book/ch05/ch05_relevance_features_reward.md:1729` | "Chapter 8" | -> "Chapter 10" |
| `docs/book/ch06/discrete_template_bandits.md:2549` | "Chapter 8: Hard constraints" | -> "Chapter 10" |
| `docs/book/ch06/appendices.md:344` | "Chapter 8 adds hard constraints" | -> "Chapter 10" |
| `CLAUDE.md` outline | Ch08 described as "Constraints" | Should say "Policy Gradients" |
| `docs/book/ch03/ch03_bellman_and_regret.md:427` | Constraint notation uses `E[STRAT] >= τ_STRAT` (ambiguous after Ch01 disambiguates exposure) | Rename to `E[Exposure_strategic] >= τ_strat` or explicitly define STRAT-as-exposure in constraint context |
| `docs/book/appendix_c_convex_optimization.md:46` | Constraint notation uses `g_2(π) = E_π[STRAT] - τ_STRAT` (strategic exposure floor) | Same as above: rename to exposure notation or add an explicit definition to prevent symbol drift |

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Keep STRAT symbol in EQ-1.2? | YES - just clarify it means purchases (minimal disruption) |
| Introduce CVR explicitly? | YES - brief note that CVR = conversions/click, distinguish from RPC |
| Update tests/scripts? | YES - critical for "math compiles" contract |

---

## Summary

This plan applies the **alignment philosophy**: Chapter 1 adopts canonical definitions from the rest of the book rather than inventing Ch01-specific terminology. The result is:

1. **Single source of truth:** STRAT in reward (EQ-1.2) = strategic **purchases**; exposure is a **separate** constraint metric
2. **Clear separation:** Exposure floors (EQ-1.3b) are constraints on what's **shown**, distinct from what's **bought**
3. **Industry-standard naming:** CVR = conversions/click, RPC = revenue/click
4. **Runnable artifacts match prose:** Tests, scripts, AND doc code blocks use aligned terminology
5. **Honest about implementation state:** `lambda_rank`, `exposure_floors`, and `cm2_floor` knobs exist in config; Chapter 10 covers guardrails + hard feasibility patterns, while Chapter 14 is the planned primal–dual CMDP payoff for learned multipliers
6. **No conflicting content:** Removed "strategic exposure violation" sections that contradicted alignment

**Total: 22 fixes across 5 files, with verification steps and cross-chapter safety analysis.**
