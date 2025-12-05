# Chapter 4 — Catalog, Users, Queries: Generative World Design

**Status:** First complete draft ✓
**Mode:** Application Mode (Vlad Prytula)
**Date:** 2025-01-14

## Overview

Chapter 4 establishes the **generative world model** that powers all RL experiments in the textbook. It seamlessly integrates mathematical formalism with production-quality code, following Application Mode standards.

## What's Included

### Main Chapter Content
**File:** `ch04_generative_world_design.md` (28,000 words)

**Structure:**
1. **Motivation** (Why we need a simulator)
   - Can't experiment on production search
   - Need counterfactual evaluation
   - Requirements: deterministic, realistic, configurable, fast

2. **Mathematical Framework**
   - [DEF-4.1]: Generative world model $\mathcal{W} = (\mathcal{C}, \mathcal{U}, \mathcal{Q}, \mu_{\mathcal{C}}, \mu_{\mathcal{U}}, \mu_{\mathcal{Q}}, \text{seed})$
   - [EQ-4.10]: Determinism property (same seed → same world)

3. **Catalog Generation** (weaved math + code)
   - [EQ-4.2]: Lognormal price distributions by category
   - [EQ-4.3]: Linear margin structure with category-specific slopes
   - [EQ-4.4]: Private label sampling (Bernoulli)
   - [EQ-4.5]: Zero-inflated discount distribution
   - [EQ-4.6]: Clustered embeddings (semantic similarity)
   - Implementation: `zoosim/world/catalog.py`
   - Numerical verification throughout

4. **User Generation** (weaved math + code)
   - [DEF-4.3], [EQ-4.7]: User segments with preference parameters
   - [EQ-4.8]: Utility function preview (detailed in Ch5)
   - Four segments: price_hunter, pl_lover, premium, litter_heavy
   - Implementation: `zoosim/world/users.py`
   - Segment validation with plots

5. **Query Generation** (weaved math + code)
   - [DEF-4.4], [EQ-4.9]: Query intent, type, embeddings, tokens
   - Query-user coupling via category affinities
   - Implementation: `zoosim/world/queries.py`
   - Intent rate verification by segment

6. **Determinism & Reproducibility**
   - Pseudo-random number generators (NumPy, PyTorch)
   - Seed management protocol
   - Cross-library consistency
   - Validation tests

7. **Statistical Validation**
   - Price/margin distributions match specifications
   - Segment preferences cluster correctly
   - Query intents align with user affinities
   - Plots demonstrating realism

8. **Theory-Practice Gap**
   - What's missing: temporal dynamics, cold-start, network effects, adversarial patterns
   - Sim-to-real transfer: domain randomization, offline RL fine-tuning
   - Modern context: MuZero, Dreamer, Decision Transformer (learned world models)
   - Hand-crafted vs. learned simulators trade-offs

9. **Production Checklist**
   - Configuration management
   - Reproducibility requirements
   - Validation tests
   - Code-theory mapping

### Exercises & Labs
**File:** `exercises_labs.md` (7,500 words)

**Contents:**
- Exercise 1: Catalog Statistics (30 min)
- Exercise 2: User Segment Analysis (30 min)
- Exercise 3: Query Intent Coupling (30 min)
- Exercise 4: Determinism Verification (15 min)
- Exercise 5: Domain Randomization (45 min, advanced)
- Exercise 6: Statistical Tests (20 min, optional)
- Lab: Complete World Generation Pipeline (30 min)
- Bonus: Catalog Embeddings Visualization (optional)

**Total time:** ~2.5 hours for core exercises

### Demonstration Script
**File:** `scripts/ch04_demo.py`

**Demonstrates:**
- Catalog generation with statistics by category
- User segment sampling and preference distributions
- Query intent coupling to user affinities
- Deterministic reproducibility (same seed → identical world)

**Run with:** `python scripts/ch04_demo.py`

## Validation

### Tests Pass ✓
```bash
pytest tests/test_catalog_stats.py -v
# 2 passed in 6.57s
```

### Demo Runs Successfully ✓
```bash
python scripts/ch04_demo.py
# All demonstrations complete with expected outputs
```

## Code-Theory Integration

Every mathematical concept is linked to implementation:

| Equation/Definition | Code Location | Validation |
|---------------------|---------------|------------|
| [DEF-4.1] Generative world | `zoosim/world/` modules | Determinism test |
| [EQ-4.2] Lognormal prices | `catalog.py:28-31` | Median matches $e^{\mu}$ |
| [EQ-4.3] Linear margins | `catalog.py:34-37` | Empirical slope = theory |
| [EQ-4.6] Clustered embeddings | `catalog.py:51-58` | Intra-category similarity |
| [DEF-4.3] User segments | `users.py:15-22` | Preference distributions |
| [DEF-4.4] Query model | `queries.py:16-23` | Intent coupling verified |
| [EQ-4.10] Determinism | All `world/` modules | `test_deterministic_catalog()` |

## Vlad Prytula Application Mode Standards

**Checklist:**
- [x] Seamless math-code integration (no separate theory/implementation sections)
- [x] RL-first motivation (simulator needed for counterfactual evaluation)
- [x] Production-quality code (NumPy, PyTorch, type hints, docstrings)
- [x] Numerical verification throughout (plots, statistics, tests)
- [x] Theory-practice gap addressed (sim-to-real transfer, learned vs. hand-crafted models)
- [x] Modern context (MuZero, Dreamer, Decision Transformer)
- [x] Code ↔ Config integration boxes (at least one per major section)
- [x] Production checklist (configuration, reproducibility, validation)
- [x] Exercises & labs (runnable code, ~2.5 hours)

## Alignment with Syllabus

**From `docs/book/drafts/syllabus.md` Ch04:**
- ✓ **Objectives**: Deterministic world generation with seeds; realistic priors
- ✓ **Implementation**: `zoosim/world/{catalog,users,queries}.py`, config in `core/config.py`
- ✓ **Labs**: Compute catalog stats; verify determinism
- ✓ **Acceptance**: Tests pass; determinism verified; distributions match specs

## Key Mathematical Results

**Definitions:**
- [DEF-4.1] Generative World Model
- [DEF-4.2] Price Sampling (lognormal)
- [DEF-4.3] User Segment
- [DEF-4.4] Query

**Equations:**
- [EQ-4.1] Product tuple
- [EQ-4.2] Lognormal price distribution
- [EQ-4.3] Linear margin structure
- [EQ-4.4] Private label Bernoulli
- [EQ-4.5] Zero-inflated discount
- [EQ-4.6] Clustered embeddings
- [EQ-4.7] User preference parameters
- [EQ-4.8] Utility function (preview for Ch5)
- [EQ-4.9] Query tuple
- [EQ-4.10] Determinism property
- [EQ-4.11] Domain randomization (sim-to-real)

## Cross-References

**To earlier chapters:**
- Chapter 1: Reward formalism (GMV, CM2, engagement) — implementation in Ch5
- Chapter 2: Click models (PBM, DBN) — position bias and abandonment in Ch5
- Chapter 3: MDP formalism — environment implementation in Ch5

**To later chapters:**
- Chapter 5: Relevance scoring, features, counterfactual ranking
- Chapter 6: LinUCB/Thompson Sampling bandits (use this simulator)
- Chapter 9: Off-policy evaluation (needs logged data from simulator)
- Chapter 10: Robustness and guardrails (drift detection on simulator)
- Chapter 13: Offline RL (train from logged simulator data)
- Chapter 15: Domain randomization implementation (preview in Exercise 5)

## What's Next

**Chapter 5 — Relevance, Features, and Counterfactual Ranking:**
- Relevance scoring: $\text{rel}(q, p) = w_{\text{sem}} \cdot \langle \boldsymbol{\phi}_{\text{emb}}, \mathbf{e}_p \rangle + w_{\text{lex}} \cdot \text{BM25}(\text{tokens}, p)$
- Feature extraction for RL state representation
- Click models (PBM, DBN) with position bias
- Reward computation (GMV, CM2, engagement)
- Complete search environment (MDP)

## Usage

**To read this chapter:**
1. Load persona: `@vlad_prytula.md @vlad_application_mode.md`
2. Read: `docs/book/drafts/ch04/ch04_generative_world_design.md`
3. Complete exercises: `docs/book/drafts/ch04/exercises_labs.md`
4. Run demo: `python scripts/ch04_demo.py`
5. Verify tests: `pytest tests/test_catalog_stats.py -v`

**To cite in later chapters:**
- Definitions: [DEF-4.1], [DEF-4.3], [DEF-4.4]
- Equations: [EQ-4.2] (prices), [EQ-4.3] (margins), [EQ-4.7] (users), [EQ-4.9] (queries)
- Code: `zoosim/world/catalog.py:61-90`, `zoosim/world/users.py:29-48`, `zoosim/world/queries.py:42-63`

## Acknowledgments

**Reviewers:** (To be added after review passes)
- Math review: Dr. Elena Sokolov (Springer GTM standards)
- Pedagogy review: Dr. Marcus Chen (graduate student learnability)
- RL bridge review: Dr. Benjamin Recht (UC Berkeley PhD standards)
- Implementation enhancement: Dr. Max Rubin (production RL quality)

**Current status:** First draft, pending review

---

**Author:** Vlad Prytula (Application Mode)
**Textbook:** RL Search From Scratch
**Chapter:** 4 of 15
**Part:** II — Simulator and Data
