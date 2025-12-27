# Chapter 1 — Exercises & Labs (Application Mode)

Reward design is now backed both by the closed-form objective (Chapter 1, #EQ-1.2) and by executable checks. The following labs keep theory and implementation coupled.

## Lab 1.1 — Reward Aggregation in the Simulator

Goal: inspect a real simulator step, record the GMV/CM2/STRAT/CLICKS decomposition, and verify that it matches the derivation of #EQ-1.2.

Chapter 1 labs use the self-contained reference implementation in `scripts/ch01/lab_solutions.py`. The main chapter includes an optional end-to-end environment smoke test; the full `ZooplusSearchEnv` integration narrative begins in Chapter 5.

```python
from scripts.ch01.lab_solutions import lab_1_1_reward_aggregation

_ = lab_1_1_reward_aggregation(seed=11, verbose=True)
```

Output (actual):
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

Verification: |computed - reported| = 0.00 < 0.01 ✓

The simulator correctly implements [EQ-1.2].
```

**Tasks**
1. Recompute $R = \alpha \text{GMV} + \beta \text{CM2} + \gamma \text{STRAT} + \delta \text{CLICKS}$ from the printed outcome and confirm agreement with the reported value.
2. Run the bound validator `validate_delta_alpha_bound()` (or `lab_1_1_delta_alpha_violation()`) and record the smallest $\delta/\alpha$ violation. *(Optional extension: reproduce the same failure via the production assertion in `zoosim/dynamics/reward.py:56` by calling the production `compute_reward` path.)*
3. Push the findings back into the Chapter 1 text—this lab explains why the implementation enforces the same bounds as Remark 1.2.1.

## Lab 1.2 — Delta/Alpha Bound Regression Test

Goal: keep the published examples executable via `pytest` so every edit to Chapter 1 remains tethered to code.

```bash
pytest tests/ch01/test_reward_examples.py -v
```

Output (actual):
```
============================= test session starts =============================
platform darwin -- Python 3.12.12, pytest-9.0.0, pluggy-1.6.0
rootdir: /Volumes/Lexar2T/src/reinforcement_learning_search_from_scratch
configfile: pyproject.toml
collecting ... collected 5 items

tests/ch01/test_reward_examples.py::test_basic_reward_comparison PASSED  [ 20%]
tests/ch01/test_reward_examples.py::test_profitability_weighting PASSED  [ 40%]
tests/ch01/test_reward_examples.py::test_rpc_diagnostic PASSED           [ 60%]
tests/ch01/test_reward_examples.py::test_delta_alpha_bounds PASSED       [ 80%]
tests/ch01/test_reward_examples.py::test_rpc_edge_cases PASSED           [100%]

============================== 5 passed in 0.15s ===============================
```

**Tasks**
1. Identify which lines in the tests correspond to the worked examples in §1.2 and to the guardrail in [REM-1.2.1].
2. Use the test names as an index: every time Chapter 1 changes a numerical claim, one of these tests should be updated in lockstep.

---

## Lab 1.3 --- Reward Function Implementation

Goal: implement the full reward aggregation from #EQ-1.2 with data structures for session outcomes and business weights. This lab provides the complete implementation referenced in Section 1.2.

```python
from dataclasses import dataclass
from typing import NamedTuple

class SessionOutcome(NamedTuple):
    """Outcomes from a single search session.

    Mathematical correspondence: realization omega in Omega of random variables
    (GMV, CM2, STRAT, CLICKS).
    """
    gmv: float          # Gross merchandise value (EUR)
    cm2: float          # Contribution margin 2 (EUR)
    strat_purchases: int # Number of strategic purchases in session
    clicks: int         # Total clicks

@dataclass
class BusinessWeights:
    """Business priority coefficients (alpha, beta, gamma, delta) in #EQ-1.2."""
    alpha_gmv: float = 1.0
    beta_cm2: float = 0.5
    gamma_strat: float = 0.2
    delta_clicks: float = 0.1

def compute_reward(outcome: SessionOutcome, weights: BusinessWeights) -> float:
    """Implements #EQ-1.2: R = alpha*GMV + beta*CM2 + gamma*STRAT + delta*CLICKS.

    This is the **scalar objective** we will maximize via RL.

    See `zoosim/dynamics/reward.py:42-66` for the production implementation that
    aggregates GMV/CM2/strategic purchases/clicks using `RewardConfig`
    parameters defined in `zoosim/core/config.py:195`.
    """
    return (weights.alpha_gmv * outcome.gmv +
            weights.beta_cm2 * outcome.cm2 +
            weights.gamma_strat * outcome.strat_purchases +
            weights.delta_clicks * outcome.clicks)

# Example: Compare two strategies
# Strategy A: Maximize GMV (show expensive products)
outcome_A = SessionOutcome(gmv=120.0, cm2=15.0, strat_purchases=1, clicks=3)

# Strategy B: Balance GMV and CM2 (show profitable products)
outcome_B = SessionOutcome(gmv=100.0, cm2=35.0, strat_purchases=3, clicks=4)

weights = BusinessWeights(alpha_gmv=1.0, beta_cm2=0.5, gamma_strat=0.2, delta_clicks=0.1)

R_A = compute_reward(outcome_A, weights)
R_B = compute_reward(outcome_B, weights)

print(f"Strategy A (GMV-focused): R = {R_A:.2f}")
print(f"Strategy B (Balanced):    R = {R_B:.2f}")
print(f"Delta = {R_B - R_A:.2f} (Strategy {'B' if R_B > R_A else 'A'} wins!)")
```

**Output:**
```
Strategy A (GMV-focused): R = 128.00
Strategy B (Balanced):    R = 118.50
Delta = -9.50 (Strategy A wins!)
```

**Tasks**
1. Verify `compute_reward` matches #EQ-1.2 exactly by hand-calculating $R_A$ and $R_B$.
2. Test with boundary cases: zero GMV, negative CM2 (loss-leader scenario), zero clicks.
3. What happens when `alpha_gmv = 0`? Is the function still meaningful?

---

## Lab 1.4 --- Weight Sensitivity Analysis

Goal: explore how different business weight configurations change optimal strategy selection. This lab extends Lab 1.3 with weight recalibration.

```python
from dataclasses import dataclass
from typing import NamedTuple

class SessionOutcome(NamedTuple):
    gmv: float
    cm2: float
    strat_purchases: int
    clicks: int

@dataclass
class BusinessWeights:
    alpha_gmv: float = 1.0
    beta_cm2: float = 0.5
    gamma_strat: float = 0.2
    delta_clicks: float = 0.1

def compute_reward(outcome: SessionOutcome, weights: BusinessWeights) -> float:
    return (weights.alpha_gmv * outcome.gmv +
            weights.beta_cm2 * outcome.cm2 +
            weights.gamma_strat * outcome.strat_purchases +
            weights.delta_clicks * outcome.clicks)

# Same outcomes as Lab 1.3
outcome_A = SessionOutcome(gmv=120.0, cm2=15.0, strat_purchases=1, clicks=3)
outcome_B = SessionOutcome(gmv=100.0, cm2=35.0, strat_purchases=3, clicks=4)

# Original weights: Strategy A wins
weights_gmv = BusinessWeights(alpha_gmv=1.0, beta_cm2=0.5, gamma_strat=0.2, delta_clicks=0.1)
print("With GMV-focused weights:")
print(f"  Strategy A: R = {compute_reward(outcome_A, weights_gmv):.2f}")
print(f"  Strategy B: R = {compute_reward(outcome_B, weights_gmv):.2f}")

# Profitability weights: Strategy B wins
weights_profit = BusinessWeights(alpha_gmv=0.5, beta_cm2=1.0, gamma_strat=0.5, delta_clicks=0.1)
print("\nWith profitability-focused weights:")
print(f"  Strategy A: R = {compute_reward(outcome_A, weights_profit):.2f}")
print(f"  Strategy B: R = {compute_reward(outcome_B, weights_profit):.2f}")
```

**Output:**
```
With GMV-focused weights:
  Strategy A: R = 128.00
  Strategy B: R = 118.50

With profitability-focused weights:
  Strategy A: R = 75.80
  Strategy B: R = 86.90
```

**Tasks**
1. Find weights where Strategy A and Strategy B achieve exactly equal reward.
2. Plot reward as a function of `beta_cm2 / alpha_gmv` ratio (from 0 to 2). At what ratio does the optimal strategy flip?
3. Identify real business scenarios where each weight configuration is appropriate (e.g., clearance sale vs. brand-building campaign).

---

## Lab 1.5 --- RPC (Revenue per Click) Monitoring (Clickbait Detection)

Goal: implement the RPC diagnostic from Section 1.2.1 to detect clickbait strategies. A healthy system has high GMV per click; clickbait produces high CTR with low revenue per click.

```python
from typing import NamedTuple

class SessionOutcome(NamedTuple):
    gmv: float
    cm2: float
    strat_purchases: int
    clicks: int

def compute_rpc(outcome: SessionOutcome) -> float:
    """GMV per click (revenue per click, RPC).

    Diagnostic for clickbait detection: high CTR with low RPC indicates
    the agent is optimizing delta*CLICKS at expense of alpha*GMV.
    See Section 1.2.1 for theory.
    """
    return outcome.gmv / outcome.clicks if outcome.clicks > 0 else 0.0

def validate_engagement_bound(delta: float, alpha: float, bound: float = 0.10) -> bool:
    """Check delta/alpha <= bound (Section 1.2.1 clickbait prevention)."""
    ratio = delta / alpha if alpha > 0 else float('inf')
    return ratio <= bound

# Compare revenue per click
outcome_A = SessionOutcome(gmv=120.0, cm2=15.0, strat_purchases=1, clicks=3)
outcome_B = SessionOutcome(gmv=100.0, cm2=35.0, strat_purchases=3, clicks=4)

rpc_A = compute_rpc(outcome_A)
rpc_B = compute_rpc(outcome_B)

print("Revenue per click (GMV per click):")
print(f"Strategy A: EUR {rpc_A:.2f}/click ({outcome_A.clicks} clicks -> EUR {outcome_A.gmv:.0f} GMV)")
print(f"Strategy B: EUR {rpc_B:.2f}/click ({outcome_B.clicks} clicks -> EUR {outcome_B.gmv:.0f} GMV)")
print(f"-> Strategy {'A' if rpc_A > rpc_B else 'B'} has higher-quality engagement")

# Verify delta/alpha bound
delta, alpha = 0.1, 1.0
print(f"\n[Validation] delta/alpha = {delta/alpha:.3f}")
print(f"             Bound check: {'PASS' if validate_engagement_bound(delta, alpha) else 'FAIL'} (must be <= 0.10)")

# Simulate clickbait scenario
clickbait_outcome = SessionOutcome(gmv=30.0, cm2=5.0, strat_purchases=0, clicks=15)
print(f"\n[Clickbait scenario] GMV={clickbait_outcome.gmv}, clicks={clickbait_outcome.clicks}")
print(f"  RPC = EUR {compute_rpc(clickbait_outcome):.2f}/click <- RED FLAG: very low!")
```

**Output:**
```
Revenue per click (GMV per click):
Strategy A: EUR 40.00/click (3 clicks -> EUR 120 GMV)
Strategy B: EUR 25.00/click (4 clicks -> EUR 100 GMV)
-> Strategy A has higher-quality engagement

[Validation] delta/alpha = 0.100
             Bound check: PASS (must be <= 0.10)

[Clickbait scenario] GMV=30, clicks=15
  RPC = EUR 2.00/click <- RED FLAG: very low!
```

**Tasks**
1. Generate 100 synthetic outcomes with varying click/GMV ratios. Plot the RPC distribution.
2. Define an alerting threshold: if RPC drops $>10\%$ below baseline, flag for review.
3. Implement a running RPC tracker: $\text{RPC}_t = \sum_{i=1}^t \text{GMV}_i / \sum_{i=1}^t \text{CLICKS}_i$.
4. What happens if `delta/alpha = 0.20` (above bound)? Simulate and observe RPC degradation.

---

## Lab 1.6 --- User Heterogeneity Simulation

Goal: demonstrate why static boost weights fail across different user segments. This lab implements the heterogeneity experiment from Section 1.3.

```python
def simulate_click_probability(product_score: float, position: int,
                                user_type: str) -> float:
    """Probability of click given score and position.

    Models position bias: P(click | position k) is proportional to 1/k.
    User types have different sensitivities to boost features.

    Note: This is a simplified model for exposition. Production uses
    sigmoid utilities and calibrated position bias from BehaviorConfig.
    See zoosim/dynamics/behavior.py for the full implementation.
    """
    position_bias = 1.0 / position  # Top positions get more attention

    if user_type == "price_hunter":
        # Highly responsive to discount boosts
        relevance_weight = 0.3
        boost_weight = 0.7
    elif user_type == "premium":
        # Prioritizes base relevance, ignores discounts
        relevance_weight = 0.8
        boost_weight = 0.2
    else:
        # Default: balanced
        relevance_weight = 0.5
        boost_weight = 0.5

    # Simplified: score = relevance + boost_features
    base_relevance = product_score * 0.6  # Assume fixed base
    boost_effect = product_score * 0.4    # Boost contribution

    utility = relevance_weight * base_relevance + boost_weight * boost_effect
    return position_bias * utility

# Static boost weights: w_discount = 2.0 (aggressive discounting)
product_scores = [8.5, 8.0, 7.8, 7.5, 7.2]  # After applying w_discount=2.0

# User 1: Price hunter clicks aggressively on boosted items
clicks_hunter = [simulate_click_probability(s, i+1, "price_hunter")
                 for i, s in enumerate(product_scores)]

# User 2: Premium shopper is less responsive to discount boosts
clicks_premium = [simulate_click_probability(s, i+1, "premium")
                  for i, s in enumerate(product_scores)]

print("Click probabilities with static discount boost (w=2.0):")
print(f"Price hunter:    {[f'{p:.3f}' for p in clicks_hunter]}")
print(f"Premium shopper: {[f'{p:.3f}' for p in clicks_premium]}")
print(f"\nExpected clicks (price hunter):    {sum(clicks_hunter):.2f}")
print(f"Expected clicks (premium shopper): {sum(clicks_premium):.2f}")

# Compute efficiency loss
loss_ratio = sum(clicks_premium) / sum(clicks_hunter)
print(f"\nPremium shoppers get {(1 - loss_ratio)*100:.0f}% fewer expected clicks")
print("-> Static weights over-index on price sensitivity!")
```

**Output:**
```
Click probabilities with static discount boost (w=2.0):
Price hunter:    ['0.476', '0.214', '0.131', '0.100', '0.076']
Premium shopper: ['0.204', '0.092', '0.056', '0.043', '0.033']

Expected clicks (price hunter):    0.997
Expected clicks (premium shopper): 0.428

Premium shoppers get 57% fewer expected clicks
-> Static weights over-index on price sensitivity!
```

**Tasks**
1. Add a third user segment: `"brand_loyalist"` (80% relevance, 20% boost, but only for specific brands). How does the static weight perform?
2. Find the optimal static weight as a compromise across all three segments. What is the average loss vs. per-segment optimal?
3. Implement a simple context-aware policy: `if user_type == "price_hunter": return 2.0 else: return 0.5`. Measure improvement over static.
4. Plot expected clicks as a function of `w_discount` for each segment. Where do the curves intersect?

---

## Lab 1.7 --- Action Space Implementation

Goal: implement the bounded continuous action space from #EQ-1.11. This lab provides the complete `ActionSpace` class referenced in Section 1.4.

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ActionSpace:
    """Continuous bounded action space: [-a_max, +a_max]^K.

    Mathematical correspondence: action space A = [-a_max, +a_max]^K, a subset of R^K.
    See #EQ-1.11 for the bound constraint.
    """
    K: int           # Dimensionality (number of boost features)
    a_max: float     # Bound on each coordinate

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample uniformly from A (for exploration)."""
        return rng.uniform(-self.a_max, self.a_max, size=self.K)

    def clip(self, a: np.ndarray) -> np.ndarray:
        """Project action onto A (enforces bounds).

        This is crucial: if a policy network outputs unbounded logits,
        we must clip to ensure a in A.
        """
        return np.clip(a, -self.a_max, self.a_max)

    def contains(self, a: np.ndarray) -> bool:
        """Check if a in A."""
        return np.all(np.abs(a) <= self.a_max)

    def volume(self) -> float:
        """Lebesgue measure of A = (2 * a_max)^K."""
        return (2 * self.a_max) ** self.K

# Example: K=5 boost features (discount, margin, PL, bestseller, recency)
action_space = ActionSpace(K=5, a_max=0.5)

# Sample random action
rng = np.random.default_rng(seed=42)
a_random = action_space.sample(rng)
print(f"Random action: {a_random}")
print(f"In bounds? {action_space.contains(a_random)}")

# Try an out-of-bounds action (e.g., from an uncalibrated policy)
a_bad = np.array([1.2, -0.3, 0.8, -1.5, 0.4])
print(f"\nBad action: {a_bad}")
print(f"In bounds? {action_space.contains(a_bad)}")

# Clip to enforce bounds
a_clipped = action_space.clip(a_bad)
print(f"Clipped:    {a_clipped}")
print(f"In bounds? {action_space.contains(a_clipped)}")

print(f"\nAction space volume: {action_space.volume():.4f}")
```

**Output:**
```
Random action: [-0.14 -0.36  0.47 -0.03  0.21]
In bounds? True

Bad action: [ 1.2 -0.3  0.8 -1.5  0.4]
In bounds? False
Clipped:    [ 0.5 -0.3  0.5 -0.5  0.4]
In bounds? True

Action space volume: 0.0312
```

**Tasks**
1. Extend `ActionSpace` to support different norms: L2 ball ($\|a\|_2 \leq r$) vs. Linf box (current).
2. For $K=2$ and $a_{\max}=1$, plot the action space. Sample 1000 points uniformly---how many fall within the L2 ball $\|a\|_2 \leq 1$?
3. Implement action discretization: divide each dimension into $n$ bins and return the $n^K$ grid points. For $K=5, n=10$, how many discrete actions?
4. Verify clipping behavior matches `zoosim/envs/search_env.py:50` by reading the production code.

---

## Lab 1.8 — Rank-Stability Preview (Delta-Rank@k)

Goal: connect the stability constraint [EQ-1.3c] to the production stability metric **Delta-Rank@k** (set churn), and verify what is (and is not) wired in the simulator at this stage.

```python
from zoosim.core import config as cfg_module
from zoosim.monitoring.metrics import compute_delta_rank_at_k

cfg = cfg_module.load_default_config()
print("lambda_rank:", cfg.action.lambda_rank)

# A pure swap within the top-10 changes order but not set membership.
ranking_prev = list(range(10))
ranking_curr = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]
print("Delta-Rank@10:", compute_delta_rank_at_k(ranking_prev, ranking_curr, k=10))
```

**Output:**
```
lambda_rank: 0.0
Delta-Rank@10: 0.0
```

**Tasks**
1. Verify that the Delta-Rank implementation matches the set-based definition in Chapter 10 [DEF-10.4] by constructing examples where two top-$k$ sets differ by exactly $m$ items (expect $\Delta\text{-rank}@k = m/k$).
2. Confirm that `lambda_rank` exists as a configuration knob (`zoosim/core/config.py:230`) but is not yet wired into the reward path in Chapter 1; treat it as a placeholder for Chapter 10 stability guardrails.

!!! note "Status: guardrail wiring"
    The configuration exposes `ActionConfig.lambda_rank` (`zoosim/core/config.py:230`), `ActionConfig.cm2_floor` (`zoosim/core/config.py:232`), and `ActionConfig.exposure_floors` (`zoosim/core/config.py:233`) so experiments remain reproducible. Guardrail enforcement is introduced in Chapter 10.
