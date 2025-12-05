# Chapter 6 Scripts: Cleanup Analysis

**Date:** 2025-11-26
**Purpose:** Document the relationship between ROOT and OPTIMIZATION scripts before cleanup

---

## Executive Summary

The `scripts/ch06/` directory contains **two versions** of the same scripts:

1. **ROOT level** (`scripts/ch06/*.py`) — **OUTDATED/BROKEN** versions
2. **OPTIMIZATION folder** (`scripts/ch06/optimization/*.py`) — **CANONICAL/WORKING** versions

**Critical Finding:** The main chapter narrative (`discrete_template_bandits.md`) references ROOT scripts in its prose, but the **actual experimental results** (winning parameters, +27% GMV figures) come from the OPTIMIZATION scripts. This is a documentation inconsistency that needs resolution.

---

## Directory Structure (Current State)

```
scripts/ch06/
├── __pycache__/
├── ch06_compute_arc.py          # ROOT - OLD (232 lines)
├── template_bandits_demo.py     # ROOT - OLD (860 lines)
├── plot_results.py              # ROOT - DUPLICATE of optimization/
├── cpu_gpu_impl_diff.md         # Documentation of CPU vs GPU differences
├── how_to_compare_gpu_vanilla.md # How to run parity comparisons
├── data/                        # CPU/GPU parity comparison artifacts
│   ├── cpu/
│   └── gpu/
├── lab_solutions/               # Lab solution code (ACTIVE)
│   ├── __init__.py
│   ├── __main__.py
│   ├── exercises.py
│   └── labs.py
├── optimization/                # CANONICAL CPU scripts
│   ├── __pycache__/
│   ├── ch06_compute_arc.py      # CURRENT (277 lines)
│   ├── template_bandits_demo.py # CURRENT (1476 lines)
│   ├── run_bandit_matrix.py     # Batch runner (422 lines)
│   ├── plot_results.py          # Plotting utilities
│   └── summary_of_matrix_run.md # Results summary
└── optimization_gpu/            # GPU-accelerated versions
    ├── ch06_compute_arc_gpu.py
    ├── run_bandit_matrix_gpu.py
    └── template_bandits_gpu.py
```

---

## Detailed Comparison: ROOT vs OPTIMIZATION Scripts

### 1. `template_bandits_demo.py`

| Aspect | ROOT (860 lines) | OPTIMIZATION (1476 lines) |
|--------|------------------|---------------------------|
| **Feature source** | Imports from `zoosim.ranking.features` | Inline implementation |
| **Simple feature dims** | 7 (no bias term) | **8 (with bias term)** |
| **Rich feature dims** | 17 (no bias term) | **18 (with bias term)** |
| **Prior seeding** | None | **`_seed_policy_with_static_prior()`** |
| **Hyperparameter tuning** | Fixed defaults only | **`resolve_prior_weight()`, `resolve_lin_alpha()`, `resolve_ts_sigma()`** |
| **Rich regularization** | None | **`RichRegularizationConfig` dataclass** |
| **CLI arguments** | Basic (`--features`, `--n-static`, `--n-bandit`) | Extended (`--prior-weight`, `--lin-alpha`, `--ts-sigma`, `--hparam-mode`, `--rich-regularization`, etc.) |

#### Why the Bias Term Matters

The OPTIMIZATION version appends a bias term to all features:

```python
# OPTIMIZATION version (template_bandits_demo.py:~180)
def _append_bias(features: np.ndarray) -> np.ndarray:
    """Append bias term (1.0) to feature vector."""
    return np.concatenate([features, [1.0]])
```

This is **critical** for:
1. **Prior seeding** — The bias term allows template priors to have a non-zero intercept
2. **Ridge regression** — Proper regularization of the constant term
3. **Numerical stability** — Better conditioning of the design matrix

Without the bias term, the bandit priors cannot be properly initialized from static template performance, leading to **worse exploration-exploitation tradeoffs**.

#### Prior Seeding (OPTIMIZATION only)

```python
# OPTIMIZATION version (template_bandits_demo.py:~540)
def _seed_policy_with_static_prior(
    policy: Union[LinUCB, ThompsonSampling],
    static_results: Dict[str, Any],
    prior_weight: int,
    feature_dim: int,
) -> None:
    """Warm-start bandit with pseudo-observations from static template performance."""
```

This function:
1. Takes the best static template's performance
2. Creates `prior_weight` pseudo-observations for each template
3. Updates the bandit's sufficient statistics (A, b matrices)

Without this, bandits start from uniform priors and waste episodes re-discovering what static evaluation already knows.

### 2. `ch06_compute_arc.py`

| Aspect | ROOT (232 lines) | OPTIMIZATION (277 lines) |
|--------|------------------|---------------------------|
| **Imports** | Only `run_template_bandits_experiment` | Also imports `resolve_lin_alpha`, `resolve_prior_weight`, `resolve_ts_sigma` |
| **CLI arguments** | Basic | Extended (`--prior-weight`, `--lin-alpha`, `--ts-sigma`) |
| **Hyperparameter handling** | Uses defaults | **Resolves per feature setting** |

### 3. `plot_results.py`

Both versions are **IDENTICAL** (verified via `diff`). One copy is redundant.

### 4. `run_bandit_matrix.py` (OPTIMIZATION only)

This script exists **only in optimization/** and provides batch execution of multiple scenarios:
- `simple_baseline`
- `rich_oracle_raw`
- `rich_oracle_blend`
- `rich_oracle_quantized`
- `rich_estimated`

It imports from the local `template_bandits_demo.py` (OPTIMIZATION version).

---

## Documentation References Analysis

### Main Chapter: `discrete_template_bandits.md`

References to ROOT scripts (incorrect for results reproduction):
```
Line 1470: scripts/ch06/template_bandits_demo.py
Line 1478: scripts/ch06/template_bandits_demo.py
Line 1910: scripts/ch06/template_bandits_demo.py
Line 1919: python scripts/ch06/template_bandits_demo.py \
Line 1997: scripts/ch06/template_bandits_demo.py
Line 2204: scripts/ch06/template_bandits_demo.py
Line 2211: scripts/ch06/template_bandits_demo.py
Line 2226: python scripts/ch06/template_bandits_demo.py \
Line 2333: scripts/ch06/template_bandits_demo.py
```

### Winning Parameters: `winning_params_linucb_ts.md`

References OPTIMIZATION scripts (correct):
```
Line 1: python scripts/ch06/optimization/template_bandits_demo.py \
```

### Lab Solutions: `ch06_lab_solutions.md`

References OPTIMIZATION scripts (correct):
```
Line 593: scripts/ch06/optimization/template_bandits_demo.py
Line 595: python scripts/ch06/optimization/template_bandits_demo.py \
Line 650: python scripts/ch06/optimization/template_bandits_demo.py \
Line 1053: python scripts/ch06/optimization/template_bandits_demo.py \
Line 1058: python scripts/ch06/optimization/template_bandits_demo.py \
```

### Exercises & Labs: `exercises_labs.md`

References ROOT scripts:
```
Line 413: python scripts/ch06/template_bandits_demo.py \
Line 437: python scripts/ch06/template_bandits_demo.py \
```

### Advanced GPU Lab: `ch06_advanced_gpu_lab.md`

References OPTIMIZATION scripts (correct):
```
Line 20: scripts/ch06/optimization/template_bandits_demo.py
Line 21: scripts/ch06/optimization/run_bandit_matrix.py
Line 60-77: Multiple references to scripts/ch06/optimization/
Line 96: python scripts/ch06/optimization/run_bandit_matrix.py \
```

### Reviews: `docs/book/reviews/ch06/`

The math review (`discrete_template_bandits_math_review.md`) lists:
```
Files Reviewed:
- scripts/ch06/template_bandits_demo.py    # ROOT (old)
- scripts/ch06/ch06_compute_arc.py         # ROOT (old)
```

But the actual results being reviewed came from OPTIMIZATION scripts!

---

## Evidence: OPTIMIZATION Scripts Produce the Chapter Results

### 1. Winning Parameters Command

From `winning_params_linucb_ts.md`:
```bash
source .venv/bin/activate && python scripts/ch06/optimization/template_bandits_demo.py \
    --features rich \
    --n-static 1000 \
    --n-bandit 20000 \
    --prior-weight 50 \      # OPTIMIZATION-only parameter
    --lin-alpha 0.2 \        # OPTIMIZATION-only parameter
    --ts-sigma 1.0 \         # OPTIMIZATION-only parameter
    --base-seed 2025_0601
```

These `--prior-weight`, `--lin-alpha`, `--ts-sigma` arguments **do not exist** in the ROOT version.

### 2. Lab Solutions Use OPTIMIZATION

From `ch06_lab_solutions.md`:
```bash
python scripts/ch06/optimization/template_bandits_demo.py \
    --features simple \
    --n-static 1000 \
    --n-bandit 5000 \
    --prior-weight 50 \
    --lin-alpha 0.5 \
    --ts-sigma 1.0
```

### 3. Batch Matrix Runner

From `summary_of_matrix_run.md`, the results artifact is:
```
docs/book/drafts/ch06/data/bandit_matrix_20251118T031855Z.json
```

This was generated by `scripts/ch06/optimization/run_bandit_matrix.py`, which imports from the OPTIMIZATION `template_bandits_demo.py`.

---

## Impact of Using ROOT Scripts (Hypothetical)

If someone runs the commands in `discrete_template_bandits.md` using the ROOT scripts:

1. **Missing CLI arguments** — Commands with `--prior-weight`, `--lin-alpha`, `--ts-sigma` will **fail with errors**
2. **Wrong feature dimensions** — 7/17 dims instead of 8/18 dims
3. **No prior seeding** — Bandits start cold, requiring more episodes to learn
4. **Different hyperparameters** — Fixed defaults instead of tuned values
5. **Results won't match** — The +27% GMV figures in the chapter are **not reproducible** with ROOT scripts

---

## Relationship to `zoosim.ranking.features`

The ROOT scripts import from `zoosim/ranking/features.py`:
```python
from zoosim.ranking.features import (
    compute_context_features_simple,
    compute_context_features_rich,
    compute_context_features_rich_estimated,
)
```

These functions return:
- `compute_context_features_simple`: 7-dim array (NO bias)
- `compute_context_features_rich`: 17-dim array (NO bias)
- `compute_context_features_rich_estimated`: 17-dim array (NO bias)

The OPTIMIZATION scripts have their own inline implementations:
- `context_features_simple`: 8-dim array (WITH bias via `_append_bias`)
- `context_features_rich`: 18-dim array (WITH bias via `_append_bias`)
- `context_features_rich_estimated`: 18-dim array (WITH bias via `_append_bias`)

**Question for future:** Should `zoosim.ranking.features` be updated to include bias terms? Or should scripts always use their own feature functions?

---

## `scripts/part4_global_comparison.py` Analysis

This script does **NOT** use any `scripts/ch06/` scripts. It imports directly from zoosim:
```python
from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
from zoosim.policies.q_ensemble import QEnsemblePolicy, QEnsembleConfig
from zoosim.policies.reinforce import REINFORCEAgent, REINFORCEConfig
from zoosim.policies.slate_q import SlateQAgent
from zoosim.policies.offline.cql import CQLAgent
from zoosim.policies.templates import BoostTemplate
```

The `LinUCBAdapter` in this file manually creates templates with weights and uses the `zoosim.policies.lin_ucb.LinUCB` class directly. It's a separate implementation path, not dependent on the ch06 scripts.

---

## Recommended Cleanup Actions

### Option A: Promote OPTIMIZATION to Root (Recommended)

1. **Archive ROOT scripts** to `scripts/ch06/archive/`:
   - `ch06_compute_arc.py` (old)
   - `template_bandits_demo.py` (old)
   - `plot_results.py` (duplicate)

2. **Move OPTIMIZATION scripts** to `scripts/ch06/` root:
   - `optimization/ch06_compute_arc.py` → `scripts/ch06/ch06_compute_arc.py`
   - `optimization/template_bandits_demo.py` → `scripts/ch06/template_bandits_demo.py`
   - `optimization/run_bandit_matrix.py` → `scripts/ch06/run_bandit_matrix.py`
   - `optimization/plot_results.py` → `scripts/ch06/plot_results.py`
   - `optimization/summary_of_matrix_run.md` → `scripts/ch06/summary_of_matrix_run.md`

3. **Move CPU/GPU comparison docs** to `optimization_gpu/`:
   - `cpu_gpu_impl_diff.md` → `scripts/ch06/optimization_gpu/`
   - `how_to_compare_gpu_vanilla.md` → `scripts/ch06/optimization_gpu/`

4. **Update documentation references**:
   - `discrete_template_bandits.md`: Change `scripts/ch06/template_bandits_demo.py` (no change needed after promotion)
   - `exercises_labs.md`: Change `scripts/ch06/template_bandits_demo.py` (no change needed after promotion)
   - `ch06_lab_solutions.md`: Change `scripts/ch06/optimization/` → `scripts/ch06/`
   - `ch06_advanced_gpu_lab.md`: Change `scripts/ch06/optimization/` → `scripts/ch06/`
   - `winning_params_linucb_ts.md`: Change `scripts/ch06/optimization/` → `scripts/ch06/`

5. **Remove empty `optimization/` folder** after moving files

### Option B: Update All References to Use optimization/ Path

Keep current structure but update all documentation to explicitly reference `scripts/ch06/optimization/`. This is more conservative but leaves the confusing duplicate scripts in place.

---

## Post-Cleanup Structure (Option A)

```
scripts/ch06/
├── archive/                     # OLD scripts (reference only)
│   ├── ch06_compute_arc.py
│   ├── template_bandits_demo.py
│   └── plot_results.py
│
├── ch06_compute_arc.py          # CANONICAL (promoted from optimization/)
├── template_bandits_demo.py     # CANONICAL (promoted from optimization/)
├── run_bandit_matrix.py         # Batch runner (promoted from optimization/)
├── plot_results.py              # Plotting (promoted from optimization/)
├── summary_of_matrix_run.md     # Results summary
│
├── lab_solutions/               # Lab code (unchanged)
│
├── optimization_gpu/            # GPU versions (unchanged)
│   ├── ch06_compute_arc_gpu.py
│   ├── run_bandit_matrix_gpu.py
│   ├── template_bandits_gpu.py
│   ├── cpu_gpu_impl_diff.md     # MOVED here
│   └── how_to_compare_gpu_vanilla.md  # MOVED here
│
└── data/                        # Parity artifacts (can archive or keep)
```

---

## Questions for Author Decision

1. **Should `zoosim.ranking.features` be updated** to include bias terms, making it consistent with the OPTIMIZATION scripts?

2. **Should the ROOT scripts be deleted entirely** or archived for historical reference?

3. **What should happen to `scripts/ch06/data/`?** It contains CPU/GPU parity comparison artifacts. Archive, keep, or delete?

4. **Should documentation references be updated** to match the new structure, or should we update the new scripts to match the old references (keeping `scripts/ch06/template_bandits_demo.py` as the canonical path)?

---

## Verification Checklist (Post-Cleanup)

After cleanup, verify:

- [ ] `python scripts/ch06/template_bandits_demo.py --help` shows all CLI arguments including `--prior-weight`, `--lin-alpha`, `--ts-sigma`
- [ ] Running the command from `winning_params_linucb_ts.md` works
- [ ] Lab solutions commands work
- [ ] `run_bandit_matrix.py` runs successfully
- [ ] All documentation references resolve to existing files
- [ ] GPU scripts in `optimization_gpu/` still work (they import from local directory)

---

## Cleanup Completed

**Date:** 2025-12-03
**Executor:** Claude (cleanup session)

### Actions Taken (Option A: Promote OPTIMIZATION to Root)

1. **Archived ROOT scripts** to `scripts/ch06/archive/`:
   - `ch06_compute_arc_OLD.py` (old 231-line version)
   - `template_bandits_demo_OLD.py` (old 859-line version)
   - `plot_results_DUPLICATE.py` (duplicate of optimization/)

2. **Promoted OPTIMIZATION scripts** to `scripts/ch06/` root:
   - `ch06_compute_arc.py` (277 lines, with --prior-weight, --lin-alpha, --ts-sigma)
   - `template_bandits_demo.py` (1475 lines, with bias terms, prior seeding, hyperparameter tuning)
   - `run_bandit_matrix.py` (422 lines, batch runner)
   - `plot_results.py` (plotting utilities)
   - `summary_of_matrix_run.md` (results summary)

3. **Moved CPU/GPU comparison docs** to `optimization_gpu/`:
   - `cpu_gpu_impl_diff.md`
   - `how_to_compare_gpu_vanilla.md`

4. **Updated documentation references** (changed `scripts/ch06/optimization/` → `scripts/ch06/`):
   - `ch06_lab_solutions.md` (5 occurrences)
   - `ch06_advanced_gpu_lab.md` (12 occurrences)
   - `winning_params_linucb_ts.md` (1 occurrence)
   - `exercises_labs.md` (1 occurrence - narrative text)

5. **Removed empty `optimization/` folder** (only had .DS_Store and __pycache__)

### Verification Results

- [x] `python scripts/ch06/template_bandits_demo.py --help` shows all CLI arguments including `--prior-weight`, `--lin-alpha`, `--ts-sigma`
- [x] `python scripts/ch06/ch06_compute_arc.py --help` shows extended CLI arguments
- [x] `python scripts/ch06/run_bandit_matrix.py --help` works correctly
- [x] All documentation references resolve to existing files
- [x] GPU scripts in `optimization_gpu/` still work (verified CLI help)
- [x] Functional test: `template_bandits_demo.py` runs with extended arguments

### Final Directory Structure

```
scripts/ch06/
├── archive/                     # OLD scripts (reference only)
│   ├── ch06_compute_arc_OLD.py
│   ├── template_bandits_demo_OLD.py
│   └── plot_results_DUPLICATE.py
│
├── ch06_compute_arc.py          # CANONICAL (promoted from optimization/)
├── template_bandits_demo.py     # CANONICAL (promoted from optimization/)
├── run_bandit_matrix.py         # Batch runner (promoted from optimization/)
├── plot_results.py              # Plotting (promoted from optimization/)
├── summary_of_matrix_run.md     # Results summary
│
├── lab_solutions/               # Lab code (unchanged)
│
├── optimization_gpu/            # GPU versions + CPU/GPU docs
│   ├── ch06_compute_arc_gpu.py
│   ├── run_bandit_matrix_gpu.py
│   ├── template_bandits_gpu.py
│   ├── cpu_gpu_impl_diff.md     # MOVED here
│   └── how_to_compare_gpu_vanilla.md  # MOVED here
│
├── data/                        # Parity artifacts (kept)
└── CLEANUP_ANALYSIS.md          # This document
```

### Notes for Future Maintenance

1. **Documentation references** now use `scripts/ch06/template_bandits_demo.py` (no optimization/ prefix)
2. **The canonical version** has the bias term (8/18 dims) and prior seeding functionality
3. **Archive folder** preserved for reference if anyone needs to compare old vs new behavior
4. **GPU scripts** are self-contained and do not import from the root scripts

---

**Document Author:** Claude (analysis session 2025-11-26)
**Cleanup completed by:** Claude (2025-12-03)
