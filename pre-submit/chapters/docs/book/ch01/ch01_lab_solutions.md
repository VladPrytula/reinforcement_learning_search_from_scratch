# Chapter 1 — Lab Solutions

*Vlad Prytula*

These solutions demonstrate the seamless integration of mathematical formalism and executable code that defines our approach to RL textbook writing. Every solution weaves theory ([EQ-1.2], [REM-1.2.1]) with runnable implementations, following the principle: **if the math doesn't compile, it's not ready**.

All outputs shown are actual results from running the code with specified seeds.

---

## Lab 1.1 — Reward Aggregation in the Simulator

**Goal:** Inspect a real simulator step, record the GMV/CM2/STRAT/CLICKS decomposition, and verify that it matches the derivation of [EQ-1.2].

### Theoretical Foundation

Recall from Section 1.2 that the scalar reward aggregates multiple business objectives:

$$
R(\mathbf{w}, u, q, \omega) = \alpha \cdot \text{GMV} + \beta \cdot \text{CM2} + \gamma \cdot \text{STRAT} + \delta \cdot \text{CLICKS}
\tag{1.2}
$$

where $\omega$ represents stochastic user behavior conditioned on the ranking induced by boost weights $\mathbf{w}$. The parameters $(\alpha, \beta, \gamma, \delta)$ encode business priorities—a choice that shapes what the RL agent learns to optimize.

This lab verifies that our simulator implements [EQ-1.2] correctly and explores the sensitivity of rewards to these parameters.

### Solution

To keep the lab fully reproducible, we provide a self-contained reference implementation in `scripts/ch01/lab_solutions.py` that mirrors the production architecture. The code below runs Lab 1.1 end-to-end with a fixed seed and prints the reward decomposition.

```python
from scripts.ch01.lab_solutions import (
    lab_1_1_reward_aggregation,
    RewardConfig,
    SessionOutcome,
)

# Run Lab 1.1 with default configuration
results = lab_1_1_reward_aggregation(seed=11, verbose=True)
```

**Actual Output:**
```
======================================================================
Lab 1.1: Reward Aggregation in the Simulator
======================================================================

Session simulation (seed=11):
  User segment: price_hunter
  Query: "cat food"

Outcome breakdown:
  GMV:    €124.46 (gross merchandise value)
  CM2:    € 18.67 (contribution margin 2)
  STRAT:  0 purchases  (strategic purchases in session)
  CLICKS: 3        (total clicks)

Reward weights (from RewardConfig):
  alpha (alpha_gmv):     1.00
  beta (beta_cm2):       0.50
  gamma (gamma_strat):   0.20
  delta (delta_clicks):  0.10

Manual computation of R = alpha*GMV + beta*CM2 + gamma*STRAT + delta*CLICKS:
  = 1.00 x 124.46 + 0.50 x 18.67 + 0.20 x 0 + 0.10 x 3
  = 124.46 + 9.34 + 0.00 + 0.30
  = 134.09

Simulator-reported reward: 134.09

Verification: |computed - reported| = 0.00 < 0.01 [OK]

The simulator correctly implements [EQ-1.2].
```

### Task 1: Recompute and Confirm Agreement

The solution above demonstrates that the reward is computed exactly as [EQ-1.2] specifies. Let's verify with different configurations:

```python
# Different weight configurations
configs = [
    ("Balanced", RewardConfig(alpha_gmv=1.0, beta_cm2=0.5, gamma_strat=0.2, delta_clicks=0.1)),
    ("Profit-focused", RewardConfig(alpha_gmv=0.5, beta_cm2=1.0, gamma_strat=0.5, delta_clicks=0.1)),
    ("GMV-focused", RewardConfig(alpha_gmv=1.0, beta_cm2=0.3, gamma_strat=0.0, delta_clicks=0.05)),
]

outcome = SessionOutcome(gmv=112.70, cm2=22.54, strat_purchases=3, clicks=4)

for name, cfg in configs:
    R = (cfg.alpha_gmv * outcome.gmv +
         cfg.beta_cm2 * outcome.cm2 +
         cfg.gamma_strat * outcome.strat_purchases +
         cfg.delta_clicks * outcome.clicks)
    print(f"{name}: R = {R:.2f}")
```

**Output:**
```
Balanced: R = 124.97
Profit-focused: R = 80.79
GMV-focused: R = 119.66
```

**Analysis:** The same session outcome produces different rewards depending on business priorities. The profit-focused configuration amplifies the CM2 contribution but reduces the GMV weight, resulting in a lower total reward for this particular outcome. This illustrates why weight calibration is critical—the RL agent will learn to optimize whatever the weights incentivize.

### Task 2: Delta/Alpha Bound Violation

From [REM-1.2.1], we established that $\delta/\alpha \in [0.01, 0.10]$ to prevent clickbait strategies. Let's find the smallest violation that triggers a warning:

```python
from scripts.ch01.lab_solutions import lab_1_1_delta_alpha_violation

lab_1_1_delta_alpha_violation(verbose=True)
```

**Actual Output:**
```
======================================================================
Lab 1.1 Task 2: Delta/Alpha Bound Violation
======================================================================

Testing progressively higher delta values...
Bound from [REM-1.2.1]: delta/alpha in [0.01, 0.10]

delta/alpha = 0.08: [OK] VALID
delta/alpha = 0.10: [OK] VALID
delta/alpha = 0.11: [X] VIOLATION
delta/alpha = 0.12: [X] VIOLATION
delta/alpha = 0.15: [X] VIOLATION
delta/alpha = 0.20: [X] VIOLATION

Smallest violation: delta/alpha = 0.11 (1.10x the bound)
```

**Why this matters:** At $\delta/\alpha = 0.11$, the engagement term contributes 11% of the GMV weight per click. With typical sessions generating 3-5 clicks vs. EUR 100-200 GMV, this can shift 1-3% of total reward toward engagement—enough for gradient-based optimizers to find clickbait strategies that inflate CTR at the expense of conversion.

### Task 3: Connection to Remark 1.2.1

The bound enforcement connects directly to [REM-1.2.1] (The Role of Engagement in Reward Design). The key insights:

1. **Incomplete attribution**: Clicks proxy for future GMV that attribution systems miss
2. **Exploration value**: Clicks reveal preferences even without conversion
3. **Platform health**: Zero-CTR systems are brittle despite high GMV

The bound $\delta/\alpha \leq 0.10$ ensures engagement remains a **tiebreaker**, not the primary signal. The code enforces this mathematically:

```python
from typing import Sequence, Tuple

from zoosim.core.config import SimulatorConfig
from zoosim.dynamics.reward import RewardBreakdown, compute_reward
from zoosim.world.catalog import Product

# Production signature (see `zoosim/dynamics/reward.py:42-66`):
# compute_reward(
#     *,
#     ranking: Sequence[int],
#     clicks: Sequence[int],
#     buys: Sequence[int],
#     catalog: Sequence[Product],
#     config: SimulatorConfig,
# ) -> Tuple[float, RewardBreakdown]

# Engagement bound (see `zoosim/dynamics/reward.py:52-59`):
# alpha = float(cfg.alpha_gmv)
# ratio = float("inf") if alpha == 0.0 else float(cfg.delta_clicks) / alpha
# assert 0.01 <= ratio <= 0.10
```

---

## Lab 1.2 — Delta/Alpha Bound Regression Test

**Goal:** Keep the published examples executable via `pytest` so every edit to Chapter 1 remains tethered to code.

### Why Regression Tests Matter

The reward function [EQ-1.2] and its constraints [REM-1.2.1] are the **mathematical contract** between business stakeholders and the RL system. If code drifts from documentation, one of two bad things happens:

1. **Silent behavior change**: The agent optimizes something different than documented
2. **Broken examples**: Readers can't reproduce chapter results

Regression tests prevent both. They encode the mathematical relationships as executable assertions.

### Solution

The canonical regression tests for Chapter 1 live in `tests/ch01/test_reward_examples.py`. They validate the worked examples from §1.2 and the engagement guardrail from [REM-1.2.1]. Constraint enforcement (CM2 floors, exposure floors, Delta-Rank guardrails) is introduced as an implementation pattern in Chapter 10; in Chapter 1 we keep tests focused on the reward contract and its immediate failure modes.

Run:

```bash
pytest tests/ch01/test_reward_examples.py -v
```

**Output:**
```
============================= test session starts =============================
collecting ... collected 5 items

tests/ch01/test_reward_examples.py::test_basic_reward_comparison PASSED  [ 20%]
tests/ch01/test_reward_examples.py::test_profitability_weighting PASSED  [ 40%]
tests/ch01/test_reward_examples.py::test_rpc_diagnostic PASSED           [ 60%]
tests/ch01/test_reward_examples.py::test_delta_alpha_bounds PASSED       [ 80%]
tests/ch01/test_reward_examples.py::test_rpc_edge_cases PASSED           [100%]

============================== 5 passed in 0.15s ===============================
```

### Task 2: Explicit Ties to Chapter Text

Each test is explicitly tied to chapter equations and remarks:

| Test | Chapter Reference | What It Validates |
|------|-------------------|-------------------|
| `test_basic_reward_comparison` | §1.2 (worked example) | Correct arithmetic for the published Strategy A vs. B comparison |
| `test_profitability_weighting` | §1.2 (weight flip) | The profitability-weighted configuration flips the preference |
| `test_rpc_diagnostic` | [REM-1.2.1] | RPC (GMV/click) diagnostic for clickbait detection |
| `test_delta_alpha_bounds` | [REM-1.2.1] | Engagement bound $\delta/\alpha \in [0.01, 0.10]$ |
| `test_rpc_edge_cases` | [REM-1.2.1] | Edge cases for RPC computation (e.g., zero clicks) |

These connections ensure that:
1. **Documentation stays accurate**: If [EQ-1.2] changes, tests fail
2. **Examples remain executable**: Readers can run any code from the chapter
3. **Theory-practice gaps are caught**: Mathematical claims are empirically verified

---

## Extended Exercise: Weight Sensitivity Analysis

**Goal:** Understand how business weight changes affect optimal policy behavior.

This exercise bridges Lab 1.1 and Lab 1.2 by exploring the **policy implications** of weight choices.

### Solution

```python
from scripts.ch01.lab_solutions import weight_sensitivity_analysis

results = weight_sensitivity_analysis(n_sessions=500, seed=42)
```

**Actual Output:**
```
======================================================================
Weight Sensitivity Analysis
======================================================================

Simulating 500 sessions across 4 weight configurations...

Configuration: Balanced (alpha=1.0, beta=0.5, gamma=0.2, delta=0.1)
  Mean reward:     EUR 237.64 +/- 224.52
  Mean GMV:        EUR 213.65
  Mean CM2:        EUR  46.98
  Mean STRAT:        0.57
  Mean CLICKS:       3.86
  RPC (GMV/click): EUR55.35

Configuration: GMV-Focused (alpha=1.0, beta=0.2, gamma=0.1, delta=0.05)
  Mean reward:     EUR 223.30 +/- 211.42
  Mean GMV:        EUR 213.65
  Mean CM2:        EUR  46.98
  Mean STRAT:        0.57
  Mean CLICKS:       3.86
  RPC (GMV/click): EUR55.35

Configuration: Profit-Focused (alpha=0.5, beta=1.0, gamma=0.3, delta=0.05)
  Mean reward:     EUR 154.17 +/- 145.20
  Mean GMV:        EUR 213.65
  Mean CM2:        EUR  46.98
  Mean STRAT:        0.57
  Mean CLICKS:       3.86
  RPC (GMV/click): EUR55.35

Configuration: Engagement-Heavy (alpha=1.0, beta=0.3, gamma=0.2, delta=0.09)
  Mean reward:     EUR 228.21 +/- 215.83
  Mean GMV:        EUR 213.65
  Mean CM2:        EUR  46.98
  Mean STRAT:        0.57
  Mean CLICKS:       3.86
  RPC (GMV/click): EUR55.35

----------------------------------------------------------------------
Key Insight:
  Same outcomes, different rewards! The underlying user behavior
  (GMV, CM2, STRAT, CLICKS) is IDENTICAL across configurations.

  Only the WEIGHTING changes how we value those outcomes.

  This is why weight calibration is critical:
  - An RL agent will optimize whatever the weights incentivize
  - Poorly chosen weights -> agent learns wrong behavior
  - [REM-1.2.1] bounds prevent one failure mode (clickbait)
  - [EQ-1.3] constraints prevent others (margin collapse, etc.)
```

### Interpretation

**Why are the underlying metrics identical?** Because we're computing rewards for the **same sessions** with different weights. The weights don't change user behavior—they change **how we value** that behavior.

This is the core insight of [EQ-1.2]: the reward function is a **value judgment** encoded as mathematics. An RL agent will faithfully optimize whatever objective we specify. We must choose wisely.

**Practical implications:**
1. **Weight changes are policy changes**: Increasing $\beta$ (CM2 weight) will cause the agent to favor high-margin products
2. **Constraints are essential**: Without [EQ-1.3] constraints, weight optimization is unconstrained and can produce pathological policies
3. **Monitoring is mandatory**: Track RPC, constraint satisfaction, and reward decomposition during training

---

## Exercise: Contextual Reward Variation

**Goal:** Verify that optimal actions vary by context, motivating contextual bandits.

From [EQ-1.5] vs [EQ-1.6], static optimization finds a single $\mathbf{w}$ for all contexts, while contextual optimization finds $\pi(x)$ that adapts to each context. Let's see why this matters.

### Solution

```python
from scripts.ch01.lab_solutions import contextual_reward_variation

results = contextual_reward_variation(seed=42)
```

**Actual Output:**
```
======================================================================
Contextual Reward Variation
======================================================================

Simulating different user segments with same boost configuration...

Static boost weights: w_discount=0.5, w_quality=0.3

Results by user segment (static policy):
  price_hunter   : Mean R = EUR144.59 +/- 109.91 (n=100)
  premium        : Mean R = EUR335.25 +/- 238.51 (n=100)
  bulk_buyer     : Mean R = EUR374.17 +/- 279.94 (n=100)
  pl_lover       : Mean R = EUR212.25 +/- 141.55 (n=100)

Optimal boost per segment (grid search):
  price_hunter   : w_discount=+0.8, w_quality=+0.8 -> R = EUR182.49
  premium        : w_discount=+0.2, w_quality=+1.0 -> R = EUR414.54
  bulk_buyer     : w_discount=+0.5, w_quality=+0.8 -> R = EUR468.58
  pl_lover       : w_discount=+1.0, w_quality=+0.8 -> R = EUR233.01

Static vs Contextual Comparison:
  Static (best single w):     Mean R = EUR266.57 across all segments
  Contextual (w per segment): Mean R = EUR324.66 across all segments

  Improvement: +21.8% by adapting to context!

This validates [EQ-1.6]: contextual optimization > static optimization.
The gap would widen with more user heterogeneity.
```

### Analysis

The 21.8% improvement from contextual policies is **free value**—it comes purely from adaptation, not from more data or better features. This is the fundamental motivation for contextual bandits:

- **Static** [EQ-1.5]: $\max_{\mathbf{w}} \mathbb{E}[R]$ finds one compromise $\mathbf{w}$ for all users
- **Contextual** [EQ-1.6]: $\max_{\pi} \mathbb{E}[R(\pi(x), x, \omega)]$ learns $\pi(x)$ that adapts

In production search with millions of queries daily, a 21.8% reward improvement translates to substantial GMV gains. This is why we formulate search ranking as a contextual bandit, not a static optimization problem.

---

## Summary: Theory-Practice Insights

These labs validated the mathematical foundations of Chapter 1:

| Lab | Key Discovery | Chapter Reference |
|-----|--------------|-------------------|
| Lab 1.1 | Reward computed exactly per [EQ-1.2] | Section 1.2 |
| Lab 1.1 Task 2 | $\delta/\alpha > 0.10$ triggers violation | [REM-1.2.1] |
| Lab 1.2 | Regression tests catch documentation drift | [EQ-1.2], [EQ-1.3] |
| Weight Sensitivity | Same outcomes, different rewards | [EQ-1.2] weights |
| Contextual Variation | 21.8% gain from adaptation | [EQ-1.5] vs [EQ-1.6] |

**Key Lessons:**

1. **The reward function is a value judgment**: [EQ-1.2] encodes business priorities as mathematics. The agent optimizes whatever we specify—choose wisely.

2. **Bounds prevent pathologies**: The $\delta/\alpha \leq 0.10$ constraint from [REM-1.2.1] isn't arbitrary—it's motivated by the engagement-vs-conversion tradeoff and clickbait failure modes.

3. **Constraints are essential**: Without [EQ-1.3] constraints, reward maximization can produce degenerate policies (zero margin, no strategic exposure, etc.).

4. **Context matters**: The gap between static and contextual optimization justifies the complexity of RL. Adapting to user/query context captures substantial value.

5. **Code must match math**: Regression tests ensure that simulator behavior matches chapter documentation. When they drift, something is wrong.

---

## Running the Code

All solutions are in `scripts/ch01/lab_solutions.py`:

```bash
# Run all labs
python scripts/ch01/lab_solutions.py --all

# Run specific lab
python scripts/ch01/lab_solutions.py --lab 1.1
python scripts/ch01/lab_solutions.py --lab 1.2

# Run extended exercises
python scripts/ch01/lab_solutions.py --exercise sensitivity
python scripts/ch01/lab_solutions.py --exercise contextual

# Run tests
pytest tests/ch01/test_reward_examples.py -v
```

---

*End of Lab Solutions*
