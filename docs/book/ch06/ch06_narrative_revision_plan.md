# Chapter 6 Narrative Revision Plan: Algorithm Selection as Feature Quality Function

**Created:** 2025-12-04
**Status:** Approved for Implementation
**Data Sources:** `scripts/ch06/full_run_results.md`, `ch06_verification.log`

---

## 0. Prerequisites for Restart

### 0.1 Required Files (all paths relative to repo root)

| File | Purpose | Status |
|------|---------|--------|
| `scripts/ch06/ch06_compute_arc.py` | Three-stage compute arc | TO UPDATE |
| `scripts/ch06/template_bandits_demo.py` | Core bandit experiment runner | EXISTS (dependency) |
| `docs/book/drafts/ch06/discrete_template_bandits.md` | Main chapter narrative | TO UPDATE |
| `docs/book/drafts/ch06/exercises_labs.md` | Lab exercises | TO UPDATE |
| `docs/book/drafts/ch06/winning_params_linucb_ts.md` | Hyperparameter documentation | TO UPDATE |
| `scripts/ch06/full_run_results.md` | Verification data source | EXISTS (reference) |
| `ch06_verification.log` | Verification data source | EXISTS (reference) |

### 0.2 Environment Check

```bash
# Activate environment
cd /Volumes/Lexar2T/src/rl_search_from_scratch
source .venv/bin/activate  # or your env

# Verify dependencies
python -c "from zoosim.core.config import SimulatorConfig; print('OK')"

# Verify bandit demo works
python scripts/ch06/template_bandits_demo.py --help
```

### 0.3 Quick Context Reload

**The Problem:** Chapter claims "TS +27% with rich features" but verification shows:
- Oracle features (`--features rich`): LinUCB wins (+31%), TS loses (+5%)
- Estimated features (`--features rich_est`): TS wins (+31%), LinUCB loses (+6%)

**The Solution:** Three-stage compute arc that teaches "algorithm selection depends on feature quality"

---

## Executive Summary

Verification tests revealed that the Chapter 6 narrative claims ("Thompson Sampling achieves +27% GMV with rich features") are configuration-dependent. This document proposes elevating the underlying phenomenon—**algorithm performance depends on feature quality**—into the chapter's fifth and most sophisticated lesson.

Instead of hiding a discrepancy, we transform it into the chapter's crown jewel.

---

## 1. Empirical Data (from Verification Suite)

### 1.1 Consolidated Results Table

| Test | Configuration | Features | LinUCB Δ% | TS Δ% | Winner |
|------|--------------|----------|-----------|-------|--------|
| **Test 1** | Simple Baseline | segment + query (8-dim) | **-27.95%** | **-13.07%** | Static |
| **Test 2** | Rich + Blend | oracle latents + blend reg | **+31.31%** | **+5.10%** | LinUCB |
| **Test 3** | Rich + Quantized | oracle latents + quant reg | **+5.79%** | **+3.14%** | LinUCB |
| **Test 4** | Rich Estimated | estimated latents | **+5.79%** | **+30.95%** | **TS** |

**Source:** `scripts/ch06/full_run_results.md` lines 262-263, 551-552, 1535-1536, 1823-1824

### 1.2 Key Insight: The Algorithm-Feature Quality Interaction

```
Feature Quality Spectrum:

  NOISY/ESTIMATED ◄─────────────────────────► CLEAN/ORACLE
        │                                           │
        │   Thompson Sampling                       │   LinUCB
        │   dominates (+31% GMV)                    │   dominates (+31% GMV)
        │                                           │
        │   Robust exploration                      │   Precise exploitation
        │   handles uncertainty                     │   leverages accuracy
        │                                           │
        ▼                                           ▼
   PRODUCTION                                  IDEALIZED
   REALITY                                     BENCHMARK
```

### 1.3 Theoretical Explanation

**LinUCB (Scalpel - Precise):**
- UCB bonus: $\sqrt{\phi^\top A^{-1} \phi}$ shrinks deterministically
- Commits hard to empirical best arm
- Optimal when linear model is accurate (oracle features)
- Brittle to feature noise—can lock into suboptimal arms

**Thompson Sampling (Swiss Army Knife - Robust):**
- Posterior sampling maintains exploration via $\Sigma_a$ variance
- Never fully commits; acts like an ensemble
- Handles model misspecification gracefully
- Literature: Russo & Van Roy (2018), Osband et al. (2019)

---

## 2. Current Chapter Problems

### 2.1 Text-Code Inconsistency

**Chapter §6.7 claims** (lines 2260-2261):
```
- LinUCB with φ_rich: 7.10 GMV (+3% vs. static)
- Thompson Sampling with φ_rich: 8.73 GMV (+27% vs. static)
```

**Code `--features rich` produces** (Test 2):
```
- LinUCB: +31.31% (WINS)
- TS: +5.10%
```

**Only `--features rich_est` matches narrative** (Test 4):
```
- LinUCB: +5.79%
- TS: +30.95% (WINS)
```

### 2.2 Hidden Pedagogical Opportunity

The current narrative implies "use TS with rich features" as a fixed recipe.

The **real lesson** is: algorithm choice is a function of feature quality—a much more valuable insight for practitioners.

---

## 3. The Optimal Solution: Three-Stage Compute Arc

### 3.1 New Experimental Structure

```
Stage 1: SIMPLE FEATURES (Failure)
  └─► φ_simple = [segment, query_type] (8-dim)
  └─► Both algorithms fail vs. static baseline
  └─► Teaches: feature poverty breaks linear models

Stage 2: RICH FEATURES — ORACLE (LinUCB Wins)
  └─► φ_rich = [segment, query, TRUE user latents, aggregates] (18-dim)
  └─► LinUCB: +31%, TS: +5%
  └─► Teaches: clean features enable precise exploitation

Stage 3: RICH FEATURES — ESTIMATED (TS Wins)  ← NEW
  └─► φ_rich_est = [segment, query, ESTIMATED user latents, aggregates] (18-dim)
  └─► LinUCB: +6%, TS: +31%
  └─► Teaches: noisy features require robust exploration
```

### 3.2 Pedagogical Arc

```
"Beautiful Theory" → "Unexpected Failure" → "Diagnosis" → "Fix" → "DEEPER INSIGHT"
     §6.1-6.3            §6.5               §6.6         §6.7      §6.7 NEW
```

The chapter gains a **fifth act**: discovering that the "fix" itself reveals algorithm selection principles.

---

## 4. Detailed Implementation Plan

### 4.1 Update `scripts/ch06/ch06_compute_arc.py`

**Current structure:**
```
Experiment 1: Simple Features
Experiment 2: Rich Features (oracle)
```

**New structure:**
```python
# Experiment 1: Simple Features (failure)
simple_results = run_experiment(feature_mode="simple", ...)

# Experiment 2: Rich Features — Oracle (LinUCB wins)
rich_oracle_results = run_experiment(feature_mode="rich", ...)

# Experiment 3: Rich Features — Estimated (TS wins)  ← NEW
rich_estimated_results = run_experiment(feature_mode="rich_est", ...)

# Generate three-way comparison plots
```

**Output files:**
```
data/template_bandits_simple_summary.json      (existing)
data/template_bandits_rich_oracle_summary.json (renamed from rich)
data/template_bandits_rich_estimated_summary.json (new)
```

---

### 4.2 Update `docs/book/drafts/ch06/discrete_template_bandits.md`

#### 4.2.1 Restructure §6.7

**Current §6.7:**
```
§6.7.1 What Makes Features "Rich"
§6.7.2 Rich Feature Design
§6.7.3 Running the Retry
§6.7.4 Why Thompson Sampling Pulls Ahead
§6.7.5 Per-Segment Breakdown
```

**New §6.7:**
```
§6.7.1 What Makes Features "Rich"
  [Existing content - define φ_rich components]

§6.7.2 Oracle vs. Estimated: The Production Reality Gap  ← NEW (~300 words, KEEP SNAPPY)
  - In simulation, we have TRUE user latent preferences
  - In production, we ESTIMATE preferences from behavioral signals
  - This distinction has profound algorithmic implications
  - We will run TWO experiments to reveal this
```

---

**⚠️ IMPLEMENTATION NOTE: Avoiding Reader Fatigue at §6.6 → §6.7.2 Transition**

**Risk:** The reader just finished the dense §6.6 Diagnosis section. Adding "yet another setup" before seeing results could cause fatigue.

**Mitigation:** Keep §6.7.2 to ~300 words max. Frame it as a *revelation*, not more setup.

**Draft text for §6.7.2 (use as template):**

```markdown
### 6.7.2 Oracle vs. Estimated: The Production Reality Gap

We diagnosed feature poverty. Now we fix it. But in fixing it, we face
a choice that reveals something deeper about algorithm selection.

Our simulator knows each user's *true* latent preferences—their exact
price sensitivity ($\theta_{\text{price}}$) and private-label affinity
($\theta_{\text{pl}}$). In production, you never have this luxury.
Real systems estimate preferences from noisy behavioral signals: clicks,
dwell time, purchase history.

We will run **two experiments** with rich features:

1. **Oracle latents**: Give the bandit the *true* user preferences
   (idealized benchmark, like having perfect logged data)

2. **Estimated latents**: Give the bandit *noisy estimates* of preferences
   (production reality, like real-time inference from clicks)

The results will surprise you—and teach you how to choose algorithms.
```

**Key principle:** The reader should feel "we're about to see something interesting" not "here's more setup to wade through."

---

**New §6.7 (continued):**

```
§6.7.3 Experiment A: Rich Features with Oracle Latents  ← RENAMED
  Results:
    - Static (best): 7.11 GMV
    - LinUCB: 9.34 GMV (+31.31%)  ← UPDATED NUMBER
    - TS: 7.47 GMV (+5.10%)        ← UPDATED NUMBER

  LinUCB dominates. With perfect feature accuracy, its deterministic
  confidence bounds exploit the signal surgically.

§6.7.4 Experiment B: Rich Features with Estimated Latents  ← NEW
  Results:
    - Static (best): 7.11 GMV
    - LinUCB: 7.52 GMV (+5.79%)
    - TS: 9.31 GMV (+30.95%)

  Thompson Sampling dominates. With noisy features, its Bayesian
  exploration handles the ambiguity gracefully.

§6.7.5 The Algorithm Selection Principle  ← NEW (~800 words)

  | Feature Quality | Best Algorithm | Mechanism |
  |-----------------|----------------|-----------|
  | Clean/Oracle | LinUCB | Precise exploitation |
  | Noisy/Estimated | Thompson Sampling | Robust exploration |

  Production systems have noisy features. Default to Thompson Sampling
  unless you have strong evidence that your features are accurate.

  This is the chapter's deepest lesson: algorithm selection is not
  about "LinUCB vs. TS" in the abstract—it's about matching your
  algorithm to your feature quality regime.

§6.7.6 Per-Segment Breakdown  ← RENUMBERED
  [Existing content]
```

#### 4.2.2 Update §6.8 "Four Lessons" → "Five Lessons"

**Add Lesson 5:**

```markdown
**Lesson 5 — Algorithm selection depends on feature quality.**

LinUCB is a scalpel—it exploits clean signals precisely but is brittle to
noise. Thompson Sampling is a Swiss Army knife—it handles uncertainty
robustly but may under-exploit when signals are clear.

In our experiments:
- With oracle features (idealized benchmark): LinUCB +31%, TS +5%
- With estimated features (production reality): LinUCB +6%, TS +31%

In production, your features are always estimates. The safe default is
Thompson Sampling. Reach for LinUCB only when you can validate that your
feature pipeline produces low-noise representations.
```

---

### 4.3 Update `docs/book/drafts/ch06/exercises_labs.md`

#### Lab 6.2 → Split into Lab 6.2a, 6.2b, 6.2c

```markdown
### Lab 6.2a: Rich Features with Oracle Latents (15 min)

**Objective:** Observe LinUCB's advantage with perfect feature information.

**Procedure:**
```bash
python scripts/ch06/template_bandits_demo.py \
    --n-static 2000 \
    --n-bandit 20000 \
    --features rich
```

**Expected result:** LinUCB ≈ +31% GMV, TS ≈ +5% GMV. LinUCB wins.

---

### Lab 6.2b: Rich Features with Estimated Latents (15 min)

**Objective:** Observe Thompson Sampling's advantage with noisy features.

**Procedure:**
```bash
python scripts/ch06/template_bandits_demo.py \
    --n-static 2000 \
    --n-bandit 20000 \
    --features rich_est
```

**Expected result:** LinUCB ≈ +6% GMV, TS ≈ +31% GMV. TS wins.

---

### Lab 6.2c: Synthesis — Algorithm Selection Principle (10 min)

**Objective:** Articulate why the winner flipped between Labs 6.2a and 6.2b.

**Questions:**
1. Why does LinUCB win with oracle features but lose with estimated features?
2. What property of Thompson Sampling makes it robust to feature noise?
3. In a production system where user preferences are estimated from clicks,
   which algorithm would you deploy? Why?

**Expected insights:**
- LinUCB's UCB bonus shrinks deterministically → commits hard → brittle to noise
- TS maintains posterior variance → never fully commits → robust to misspecification
- Production = noisy features → default to TS
```

---

### 4.4 Update `docs/book/drafts/ch06/winning_params_linucb_ts.md`

**Current:** Documents a single "winning config"

**New structure:**
```markdown
# Winning Configurations for Chapter 6 Template Bandits

## Configuration A: Oracle Features (LinUCB-Optimal)

Use when features are clean (offline evaluation, logged data analysis):

```bash
python scripts/ch06/template_bandits_demo.py \
    --features rich \
    --rich-regularization blend \
    --prior-weight 50 \
    --lin-alpha 0.2 \
    --ts-sigma 0.5
```

Results (seed=20250601):
- LinUCB: +31.31% GMV vs static
- TS: +5.10% GMV vs static

---

## Configuration B: Estimated Features (TS-Optimal)

Use when features are noisy (online production, estimated user preferences):

```bash
python scripts/ch06/template_bandits_demo.py \
    --features rich_est \
    --prior-weight 50 \
    --lin-alpha 0.2 \
    --ts-sigma 0.2
```

Results (seed=20250601):
- LinUCB: +5.79% GMV vs static
- TS: +30.95% GMV vs static

---

## When to Use Which Configuration

| Deployment Context | Recommended Config | Reason |
|--------------------|-------------------|--------|
| Offline evaluation | Config A (Oracle) | Clean logged data |
| A/B test analysis | Config A (Oracle) | Controlled conditions |
| Online production | Config B (Estimated) | Noisy real-time signals |
| Cold-start users | Config B (Estimated) | Limited behavioral data |
```

---

## 5. Verification Matrix Update

### 5.1 Update `scripts/ch06/run_full_verification.sh`

Ensure the compute arc test runs all three stages:

```bash
# Test 5: Three-Stage Compute Arc
echo "Test 5: Three-Stage Compute Arc (Simple → Rich Oracle → Rich Estimated)"
python scripts/ch06/ch06_compute_arc.py \
    --n-static 2000 \
    --n-bandit 20000 \
    --out-dir "$DATA_DIR"
```

### 5.2 Expected Output After Changes

```
================================================================================
CHAPTER 6 — COMPUTE ARC: SIMPLE → RICH ORACLE → RICH ESTIMATED
================================================================================

EXPERIMENT 1: SIMPLE FEATURES (Segment + Query Type)
  Static (best):  GMV = 7.11
  LinUCB:         GMV = 5.12  (-27.95%)
  TS:             GMV = 6.18  (-13.07%)
  ⚠️  BANDITS UNDERPERFORM STATIC (expected for pedagogical arc!)

EXPERIMENT 2: RICH FEATURES — ORACLE LATENTS
  Static (best):  GMV = 7.11
  LinUCB:         GMV = 9.34  (+31.31%)  ← LINUCB WINS
  TS:             GMV = 7.47  (+5.10%)
  ✓ LINUCB EXPLOITS CLEAN SIGNAL PRECISELY

EXPERIMENT 3: RICH FEATURES — ESTIMATED LATENTS
  Static (best):  GMV = 7.11
  LinUCB:         GMV = 7.52  (+5.79%)
  TS:             GMV = 9.31  (+30.95%)  ← TS WINS
  ✓ TS HANDLES NOISY FEATURES ROBUSTLY

SUMMARY: Algorithm selection depends on feature quality!
  - Clean features → LinUCB (precision)
  - Noisy features → TS (robustness)
================================================================================
```

---

## 6. Implementation Order

| Step | File | Change | Effort |
|------|------|--------|--------|
| 1 | `scripts/ch06/ch06_compute_arc.py` | Add Experiment 3 (rich_est) | 30 min |
| 2 | `discrete_template_bandits.md` §6.7 | Restructure with new sections | 2 hours |
| 3 | `discrete_template_bandits.md` §6.8 | Add Lesson 5 | 30 min |
| 4 | `exercises_labs.md` | Split Lab 6.2 into 6.2a/b/c | 45 min |
| 5 | `winning_params_linucb_ts.md` | Document both configs | 30 min |
| 6 | Run verification suite | Confirm all tests pass | 30 min |
| **Total** | | | **~5 hours** |

---

## 7. Why This is the Optimal Path

| Criterion | Quick Fix (rich→rich_est) | This Plan |
|-----------|---------------------------|-----------|
| **Intellectual honesty** | Hides complexity | Embraces it |
| **Pedagogical depth** | Single recipe | Deep principle |
| **Text-code alignment** | Creates inconsistency | Full alignment |
| **Production relevance** | Implicit | Explicit |
| **Uniqueness** | Standard textbook | Rare insight |
| **Student takeaway** | "Use TS" | "Match algorithm to feature quality" |

---

## 8. Appendix: Raw Data References

### Test 1 (Simple Features)
- Source: `full_run_results.md` lines 262-263
- LinUCB: 5.62 reward, 5.12 GMV, -27.95%
- TS: 6.69 reward, 6.18 GMV, -13.07%

### Test 2 (Rich + Blend)
- Source: `full_run_results.md` lines 551-552
- LinUCB: 10.11 reward, 9.34 GMV, +31.31%
- TS: 8.15 reward, 7.47 GMV, +5.10%

### Test 3 (Rich + Quantized)
- Source: `full_run_results.md` lines 1535-1536
- LinUCB: 8.20 reward, 7.52 GMV, +5.79%
- TS: 7.99 reward, 7.34 GMV, +3.14%

### Test 4 (Rich Estimated)
- Source: `full_run_results.md` lines 1823-1824
- LinUCB: 8.20 reward, 7.52 GMV, +5.79%
- TS: 10.08 reward, 9.31 GMV, +30.95%

---

## 9. Sign-Off

- [ ] Plan reviewed and approved
- [ ] `ch06_compute_arc.py` updated
- [ ] `discrete_template_bandits.md` §6.7 restructured
- [ ] `discrete_template_bandits.md` §6.8 updated with Lesson 5
- [ ] `exercises_labs.md` Labs 6.2a/b/c created
- [ ] `winning_params_linucb_ts.md` updated
- [ ] Full verification suite passes
- [ ] Chapter narrative matches reproducible code output
