# Full Codebase Validation Guide

This guide provides step-by-step instructions to validate all code across the textbook.
Run this after any significant changes to ensure the overall story remains intact.

## Quick Validation (~10 seconds)

```bash
# Activate environment
source .venv/bin/activate

# Run fast unit tests only (skips slow integration tests)
pytest tests/ch00 tests/ch01 tests/ch02 tests/ch03 tests/ch05 \
       tests/ch07 tests/ch08 tests/ch09 tests/ch11 \
       tests/ch06/test_linucb.py tests/ch06/test_templates.py tests/ch06/test_thompson_sampling.py \
       tests/test_catalog_stats.py tests/test_env_basic.py -q

# Expected: 63 tests pass in ~5 seconds
```

## Medium Validation (~1-2 minutes)

```bash
# Includes key chapter scripts
source .venv/bin/activate
pytest tests/ --ignore=tests/ch06/test_feature_modes_integration.py --ignore=tests/ch06/test_integration.py -q
python scripts/ch01/lab_solutions.py --lab 1.1
python scripts/ch03/lab_solutions.py --all
```

---

## Full Chapter-by-Chapter Validation

### Prerequisites

```bash
# Ensure environment is set up
cd /Volumes/Lexar2T/src/reinforcement_learning_search_from_scratch
source .venv/bin/activate

# Verify installation
python -c "import zoosim; print('zoosim OK')"
python -c "import numpy, torch; print('deps OK')"
```

---

## Part I: Foundations (Ch00-Ch03)

### Chapter 0: Motivation & First Experiment

```bash
# Tests (3 tests)
pytest tests/ch00/test_toy_example.py -v

# Lab solutions
python scripts/ch00/lab_solutions.py --all

# Toy Q-learning (may take ~30s)
python scripts/ch00/toy_problem_solution.py
```

**Expected**: Q-learning converges, learns context-adaptive boosts.

---

### Chapter 1: Search Ranking as Optimization

```bash
# Tests (5 tests)
pytest tests/ch01/test_reward_examples.py -v

# Lab solutions (all labs)
python scripts/ch01/lab_solutions.py --all

# Individual labs
python scripts/ch01/lab_solutions.py --lab 1.1
python scripts/ch01/lab_solutions.py --lab 1.2
python scripts/ch01/lab_solutions.py --exercise sensitivity
python scripts/ch01/lab_solutions.py --exercise contextual
```

**Expected**:
- All 5 tests pass
- Lab 1.1 shows reward = 134.09 for seed=11
- Contextual improvement ~21.8% over static

---

### Chapter 2: Probability, Measure, and Click Models

```bash
# Tests
pytest tests/ch02/test_behavior.py -v
pytest tests/ch02/test_segment_sampling.py -v

# Lab solutions
python scripts/ch02/lab_solutions.py --all
```

**Expected**: Click model probabilities are calibrated, position bias decreasing.

---

### Chapter 3: Bellman Operators & Contractions

```bash
# Tests (2 tests)
pytest tests/ch03/test_value_iteration.py -v

# Lab solutions
python scripts/ch03/lab_solutions.py --all
```

**Expected**: Value iteration converges, contraction property verified.

---

## Part II: Simulator (Ch04-Ch05)

### Chapter 4: Catalog, Users, Queries

```bash
# Tests
pytest tests/test_catalog_stats.py -v

# Demo script
python scripts/ch04/ch04_demo.py
```

**Expected**: Catalog stats within expected ranges, deterministic given seed.

---

### Chapter 5: Relevance, Features, Reward

```bash
# Tests
pytest tests/ch05/test_ch05_core.py -v
pytest tests/test_env_basic.py -v

# Demo script
python scripts/ch05/ch05_demo.py
```

**Expected**: Features standardized, reward breakdown matches EQ-1.2.

---

## Part III: Policies (Ch06-Ch08)

### Chapter 6: Discrete Template Bandits (COMPREHENSIVE)

Chapter 6 teaches a **pedagogical failure-then-fix arc**. Validation must check both.

#### Tier 1: Fast Unit Tests (~1 second)

```bash
pytest tests/ch06/test_linucb.py tests/ch06/test_templates.py \
       tests/ch06/test_thompson_sampling.py -v
```

**23 tests verify**: Template semantics, LinUCB UCB computation, TS posterior sampling, reproducibility.

#### Tier 2: Integration Tests (~5 seconds)

```bash
pytest tests/ch06/test_integration.py -v
```

**4 tests verify**: Training loops converge to optimal template, exploration decays.

#### Tier 3: Narrative Smoke Test (~60 seconds)

```bash
# Quick check that the demo runs and produces sensible output
python scripts/ch06/template_bandits_demo.py \
    --n-static 200 --n-bandit 1000 --features simple
```

**Check**: Script completes, prints static/LinUCB/TS comparison table.

#### Tier 4: Full Narrative Verification (~10-30 minutes, OPTIONAL)

Run only when validating chapter narrative claims for publishing:

```bash
# Claim 1: Simple features FAIL (~5 min)
python scripts/ch06/template_bandits_demo.py \
    --n-static 2000 --n-bandit 20000 --features simple
# Expected: LinUCB ~-30% GMV, TS ~-10% GMV vs static

# Claim 2: Rich oracle features BOTH WIN (~5 min)
python scripts/ch06/template_bandits_demo.py \
    --n-static 2000 --n-bandit 20000 --features rich \
    --rich-regularization blend --prior-weight 50 \
    --lin-alpha 0.2 --ts-sigma 0.5
# Expected: Both LinUCB and TS ~+32% GMV vs static

# Claim 3: Rich estimated TS WINS (~5 min)
python scripts/ch06/template_bandits_demo.py \
    --n-static 2000 --n-bandit 20000 --features rich_est \
    --prior-weight 50 --lin-alpha 0.2 --ts-sigma 0.5
# Expected: TS ~+31%, LinUCB ~+6% (Algorithm Selection Principle)
```

#### Tier 5: Statistical Multi-Seed (~15-30 minutes, RARE)

```bash
# Only for pre-publication validation
pytest tests/ch06/test_feature_modes_integration.py -v -s
```

**Expected**: Rich features statistically improve over simple across multiple seeds.

#### Template & Feature Sanity Check (~1 second)

```bash
python -c "
from zoosim.policies.templates import create_standard_templates, compute_catalog_stats
from zoosim.world.catalog import generate_catalog
from zoosim.core.config import SimulatorConfig
import numpy as np

cfg = SimulatorConfig(seed=42)
products = generate_catalog(cfg.catalog, np.random.default_rng(42))
templates = create_standard_templates(compute_catalog_stats(products))
print(f'Templates: M={len(templates)} (expected 8)')
print(f'Feature dims: simple=8, rich=18')
for t in templates: print(f'  t{t.id}: {t.name}')
"
```

| Tier | Time | When to Run |
|------|------|-------------|
| 1 (Unit) | ~1s | Every commit |
| 2 (Integration) | ~5s | Every commit |
| 3 (Smoke) | ~60s | After Ch6 changes |
| 4 (Narrative) | ~15min | Pre-publication |
| 5 (Statistical) | ~30min | Major releases only |

---

### Chapter 6A: Neural Bandits (Optional)

```bash
# Neural linear demo
python scripts/ch06a/neural_linear_demo.py

# Neural bandits demo
python scripts/ch06a/neural_bandits_demo.py

# Calibration check
python scripts/ch06a/calibration_check.py
```

**Expected**: Neural features improve over hand-crafted when data abundant.

---

### Chapter 7: Continuous Actions via Q(x,a)

```bash
# Tests
pytest tests/ch07/test_cem.py -v
pytest tests/ch07/test_q_ensemble.py -v

# Demo
python scripts/ch07/continuous_actions_demo.py

# Comparison
python scripts/ch07/compare_discrete_continuous.py
```

**Expected**: CEM optimizer finds good actions; continuous beats discrete templates.

---

### Chapter 8: Policy Gradient Methods

```bash
# Tests
pytest tests/ch08/test_reinforce.py -v

# REINFORCE demo
python scripts/ch08/reinforce_demo.py

# Neural REINFORCE
python scripts/ch08/neural_reinforce_demo.py

# Theory-practice gap visualization
python scripts/ch08/visualize_theory_practice_gap.py

# REINFORCE with baseline
python scripts/ch08/reinforce_baseline_demo.py
```

**Expected**: REINFORCE improves over random but underperforms Q-learning (~2x gap).

---

## Part IV: Evaluation & Robustness (Ch09-Ch11)

### Chapter 9: Off-Policy Evaluation

```bash
# Tests
pytest tests/ch09/test_ope.py -v

# OPE demo
python scripts/ch09/ch09_ope_demo.py
```

**Expected**: IPS/SNIPS/DR estimates within confidence intervals of online.

---

### Chapter 10: Robustness & Guardrails

```bash
# Drift detection demo
python scripts/ch10/ch10_drift_demo.py
```

**Expected**: CUSUM detects synthetic drift, recovery within N episodes.

---

### Chapter 11: Multi-Episode MDP

```bash
# Tests
pytest tests/ch11/ -v

# Lab 1: Single vs Multi-episode
python scripts/ch11/lab_01_single_vs_multi_episode.py

# Lab 2: Retention curves
python scripts/ch11/lab_02_retention_curves.py

# Lab 3: Value stability
python scripts/ch11/lab_03_value_stability.py
```

**Expected**: Retention monotone in engagement; value estimates stable across seeds.

---

## Part V: Frontier Methods (Ch12-Ch13)

### Chapter 12: Slate RL

```bash
# Demo
python scripts/ch12/slate_q_demo.py
```

**Expected**: Diversity@k improves with minimal GMV loss.

---

### Chapter 13: Offline RL

```bash
# Demo
python scripts/ch13/offline_rl_demo.py
```

**Expected**: Offline policy >= behavioral; no constraint regressions.

---

## Global Validation Commands

### All Tests (Fast)

```bash
pytest -q
# Expected: All tests pass (14+ tests across chapters)
```

### All Tests with Verbose Output

```bash
pytest -v --tb=short
```

### All Tests with Coverage

```bash
pytest --cov=zoosim --cov-report=term-missing
```

### Knowledge Graph Validation

```bash
make validate-kg
# or
python scripts/validate_knowledge_graph.py \
  --graph docs/knowledge_graph/graph.yaml \
  --schema docs/knowledge_graph/schema.yaml
```

---

## Chapter 1 Specific Verification

After alignment fixes, run these verification commands:

```bash
# 1. STRAT terminology (should show clarification note)
grep -n "Two strategic quantities" docs/book/ch01/ch01_foundations_revised_math+pedagogy_v3.md

# 2. No strat_exposure remnants (should be EMPTY)
grep -rn "strat_exposure" docs/book/ch01/ tests/ch01/ scripts/ch01/ | grep -v "plan\|update"

# 3. No compute_conversion_quality (should be EMPTY)
grep -rn "compute_conversion_quality" tests/ch01/ scripts/ch01/

# 4. CVR/RPC check (should be EMPTY - no CVR for GMV/click)
grep -rn "CVR" docs/book/ch01/*.md tests/ch01/ scripts/ch01/ | grep -v "plan\|update" | grep -iE "gmv|revenue"

# 5. Line number verification
grep -n "seed" zoosim/core/config.py | grep "252"
grep -n "a_max" zoosim/core/config.py | grep "229"
grep -n "assert.*ratio" zoosim/dynamics/reward.py | grep "56"
grep -n "np.clip" zoosim/envs/search_env.py | grep "50"

# 6. Cross-chapter "Chapter 8" constraint refs (should be EMPTY)
grep -rn "Chapter 8" docs/book/ch05/ docs/book/ch06/ | grep -i "constraint\|floor\|hard"
```

---

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `pip install -e .` was run
2. **Missing torch**: Some Ch06+ demos need PyTorch: `pip install torch`
3. **Slow tests**: Use `pytest -x` to stop on first failure
4. **Seed mismatches**: Check that seeds are deterministic (same output each run)

### Environment Reset

```bash
# Full reset
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

---

## Summary Checklist

| Chapter | Tests | Scripts | Status |
|---------|-------|---------|--------|
| Ch00 | `tests/ch00/` | `scripts/ch00/` | |
| Ch01 | `tests/ch01/` | `scripts/ch01/` | |
| Ch02 | `tests/ch02/` | `scripts/ch02/` | |
| Ch03 | `tests/ch03/` | `scripts/ch03/` | |
| Ch04 | `tests/test_catalog_stats.py` | `scripts/ch04/` | |
| Ch05 | `tests/ch05/`, `test_env_basic.py` | `scripts/ch05/` | |
| Ch06 | `tests/ch06/` | `scripts/ch06/` | |
| Ch06A | - | `scripts/ch06a/` | |
| Ch07 | `tests/ch07/` | `scripts/ch07/` | |
| Ch08 | `tests/ch08/` | `scripts/ch08/` | |
| Ch09 | `tests/ch09/` | `scripts/ch09/` | |
| Ch10 | - | `scripts/ch10/` | |
| Ch11 | `tests/ch11/` | `scripts/ch11/` | |
| Ch12 | - | `scripts/ch12/` | |
| Ch13 | - | `scripts/ch13/` | |

---

*Last updated: 2025-12-27*
