# Chapter 11 Implementation Plan

**Created:** 2025-12-09
**Status:** Planning complete, ready for implementation
**Author Mode:** Application Mode (`vlad_prytula.md` + `vlad_application_mode.md`)

---

## Overview

Chapter 11 extends single-session contextual bandits (Ch01-06) to **multi-episode MDPs** with inter-session dynamics. The key insight: engagement (clicks) enters the optimization *implicitly* through retention dynamics, eliminating the need for the pragmatic $\delta \cdot \text{CLICKS}$ proxy from Ch01.

**Mathematical Foundation:** EQ-1.2-prime (defined in Ch01, line 197-200)
$$
V^\pi(s_0) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t \text{GMV}_t \mid s_0 \right]
$$

---

## Code Status Assessment

### Existing Code (Complete)

| Module | Status | Description |
|--------|--------|-------------|
| `zoosim/multi_episode/session_env.py` | ✅ 100% | `MultiSessionEnv` with reset(), step(), `SessionMDPState` |
| `zoosim/multi_episode/retention.py` | ✅ 100% | `return_probability()`, `sample_return()` with logistic hazard |
| `zoosim/core/config.py::RetentionConfig` | ✅ 100% | `base_rate=0.35`, `click_weight=0.20`, `satisfaction_weight=0.15` |

### Key Implementation Details

**SessionMDPState** (from `session_env.py`):
```python
@dataclass
class SessionMDPState:
    t: int                    # Episode counter
    user_segment: str         # Carries forward across sessions
    query_type: str           # Sampled from user preference each session
    phi_cat: Sequence[float]  # Query embedding
    last_satisfaction: float  # From previous session outcome
    last_clicks: int          # Count from previous session
```

**Retention Model** (from `retention.py`):
```python
def return_probability(clicks, satisfaction, config) -> float:
    """
    P(return) = sigmoid(logit_base + click_weight * clicks + satisfaction_weight * satisfaction)

    - logit_base = logit(base_rate) where base_rate = 0.35
    - click_weight = 0.20 (additive logit per click)
    - satisfaction_weight = 0.15 (additive logit per satisfaction point)
    """
```

**MultiSessionEnv.step()** behavior:
- Returns `(next_state, reward, done, info)`
- `done=True` when user churns (doesn't return)
- `done=False` when user returns (same user, new query sampled)
- Reward is immediate GMV (no explicit click term)

---

## Labs Required (per syllabus.md)

### Lab 11.1 --- Single-Step Proxy vs. Multi-Episode Value

**Objective:** Compare policy rankings under two reward formulations:
1. **Single-step proxy** (EQ-1.2): $R = \alpha \cdot \text{GMV} + \delta \cdot \text{CLICKS}$
2. **Multi-episode value** (EQ-1.2-prime): $V^\pi = \sum_t \gamma^t \text{GMV}_t$

**Acceptance Criterion:** Policy ordering agrees $\geq 80\%$ (Spearman $\rho \geq 0.80$)

**Implementation Approach:**
```python
# scripts/ch11/lab_01_single_vs_multi_episode.py

def collect_multi_episode_trajectory(env, policy, gamma=0.95, max_sessions=100):
    """
    Run policy until user churns.
    Return discounted sum of GMV rewards.
    """
    state = env.reset()
    total_discounted = 0.0
    discount = 1.0

    for _ in range(max_sessions):
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        total_discounted += discount * reward
        discount *= gamma
        if done:
            break
        state = next_state

    return total_discounted

def compare_policy_rankings(policies, env, n_users=200, gamma=0.95):
    """
    For each policy:
    1. Compute multi-episode value (average over n_users)
    2. Compute single-step value (sum of immediate rewards with delta*clicks)
    3. Rank policies by each metric
    4. Compute Spearman correlation
    """
```

**Key Insight:** Single-step proxy adds explicit click incentive. Multi-episode value captures clicks *implicitly* because clicks increase retention, which increases continuation value. If policies disagree, multi-episode identifies longer-term value that single-step misses.

---

### Lab 11.2 --- Retention Curves by Segment

**Objective:** Visualize and validate the retention model.

**Acceptance Criterion:** Retention monotone in engagement (both clicks and satisfaction)

**Implementation Approach:**
```python
# scripts/ch11/lab_02_retention_curves.py

def plot_retention_heatmap(config):
    """
    2D heatmap: P(return) as function of (clicks, satisfaction).
    X-axis: clicks (0-10)
    Y-axis: satisfaction (0-1)
    Color: P(return)
    """

def validate_monotonicity(config):
    """
    Verify that:
    - P(return | clicks+1, sat) >= P(return | clicks, sat) for all (clicks, sat)
    - P(return | clicks, sat+epsilon) >= P(return | clicks, sat) for all (clicks, sat)

    Return: Dict with validation results
    """

def plot_retention_vs_clicks(config, satisfaction_levels=[0.0, 0.3, 0.6, 1.0]):
    """
    Line plots: P(return) vs clicks at fixed satisfaction levels.
    Shows how engagement drives retention.
    """
```

**Visualization Outputs:**
1. Heatmap: P(return) surface over (clicks, satisfaction)
2. Line plots: P(return) vs clicks for different satisfaction levels
3. Comparison: theoretical model vs empirical simulator data

---

### Lab 11.3 --- Value Estimation Stability

**Objective:** Verify reproducibility of value estimates.

**Acceptance Criterion:** Coefficient of variation (CV) < 0.10 across seeds

**Implementation Approach:**
```python
# scripts/ch11/lab_03_value_stability.py

def estimate_policy_value(env_seed, policy, n_users=100, gamma=0.95):
    """Monte Carlo estimate of V^pi."""
    env = MultiSessionEnv(seed=env_seed)
    values = []
    for _ in range(n_users):
        v = collect_multi_episode_trajectory(env, policy, gamma)
        values.append(v)
    return np.mean(values)

def stability_analysis(policy, seeds, n_users=100):
    """
    Run value estimation across multiple seeds.
    Compute: mean, std, 95% CI, CV
    """
    estimates = [estimate_policy_value(seed, policy, n_users) for seed in seeds]
    return {
        'mean': np.mean(estimates),
        'std': np.std(estimates),
        'cv': np.std(estimates) / np.mean(estimates),
        'ci_95': (np.percentile(estimates, 2.5), np.percentile(estimates, 97.5))
    }
```

---

## Scripts Directory Structure

```
scripts/ch11/
├── __init__.py
├── lab_01_single_vs_multi_episode.py    # Lab 11.1: Policy ranking comparison
├── lab_02_retention_curves.py           # Lab 11.2: Retention visualization
├── lab_03_value_stability.py            # Lab 11.3: Stability analysis
└── utils/
    ├── __init__.py
    ├── trajectory.py                    # Multi-episode trajectory collection
    └── policies.py                      # Test policies (uniform, greedy, segment-aware)
```

### Utility: trajectory.py

```python
from dataclasses import dataclass
from typing import List, Dict, Callable
import numpy as np
from zoosim.multi_episode.session_env import MultiSessionEnv

@dataclass
class MultiEpisodeTrajectory:
    """Complete user journey from first session to churn."""
    user_segment: str
    n_sessions: int
    total_gmv: float
    total_clicks: int
    discounted_value: float
    session_rewards: List[float]
    session_clicks: List[int]

def collect_trajectory(
    env: MultiSessionEnv,
    policy: Callable[[SessionMDPState], np.ndarray],
    gamma: float = 0.95,
    max_sessions: int = 100
) -> MultiEpisodeTrajectory:
    """
    Run policy until user churns.

    Args:
        env: Multi-session environment
        policy: Maps state dict to action (boost vector)
        gamma: Discount factor
        max_sessions: Safety limit on episode length

    Returns:
        Complete trajectory with discounted value
    """

def collect_dataset(
    env: MultiSessionEnv,
    policy: Callable[[SessionMDPState], np.ndarray],
    n_users: int,
    gamma: float = 0.95,
    seed: int = 42
) -> List[MultiEpisodeTrajectory]:
    """Collect trajectories for n_users with deterministic seeding."""
```

### Utility: policies.py

```python
import numpy as np
from typing import Callable
from zoosim.multi_episode.session_env import SessionMDPState

def uniform_policy() -> Callable[[SessionMDPState], np.ndarray]:
    """Zero boost baseline (no personalization)."""
    return lambda state: np.zeros(10)

def random_policy(rng: np.random.Generator, a_max: float = 0.5) -> Callable[[SessionMDPState], np.ndarray]:
    """Random boost in [-a_max, a_max]."""
    return lambda state: rng.uniform(-a_max, a_max, size=10)

def cm2_heavy_policy() -> Callable[[SessionMDPState], np.ndarray]:
    """Boost CM2-related features (heuristic for GMV)."""
    boost = np.array([0.5, 0.0, 0.3, 0.0, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0])
    return lambda state: boost

def segment_aware_policy(segment_boosts: dict[str, np.ndarray]) -> Callable[[SessionMDPState], np.ndarray]:
    """Apply segment-specific boost templates."""
    def policy(state: SessionMDPState) -> np.ndarray:
        segment = state.user_segment
        return segment_boosts.get(segment, np.zeros(10))
    return policy
```

---

## Tests Directory Structure

```
tests/ch11/
├── __init__.py
├── test_session_env.py      # Validate MultiSessionEnv behavior
├── test_retention.py        # Validate retention model properties
└── test_multi_episode_value.py  # Validate value computation
```

### test_session_env.py

```python
import pytest
import numpy as np
from zoosim.multi_episode.session_env import MultiSessionEnv, SessionMDPState
from zoosim.core.config import load_default_config

class TestMultiSessionEnv:

    def test_reset_returns_valid_state(self):
        """Reset should return SessionMDPState with sane defaults."""
        env = MultiSessionEnv(seed=42)
        state = env.reset()
        assert isinstance(state, SessionMDPState)
        assert state.user_segment is not None
        assert state.query_type is not None
        assert state.last_clicks == 0  # Fresh user

    def test_step_returns_correct_types(self):
        """Step should return (SessionMDPState, float, bool, dict)."""
        env = MultiSessionEnv(seed=42)
        env.reset()
        action = np.zeros(10)
        next_state, reward, done, info = env.step(action)

        assert isinstance(next_state, SessionMDPState)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_same_user_until_churn(self):
        """User segment should persist until done=True."""
        env = MultiSessionEnv(seed=42)
        state = env.reset()
        initial_segment = state.user_segment

        action = np.zeros(10)
        for _ in range(50):
            next_state, _, done, _ = env.step(action)
            if done:
                break
            assert next_state.user_segment == initial_segment

    def test_deterministic_with_seed(self):
        """Same seed should produce identical trajectories."""
        def run_episode(seed):
            env = MultiSessionEnv(seed=seed)
            state = env.reset()
            rewards = []
            action = np.zeros(10)
            for _ in range(10):
                _, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break
            return rewards

        r1 = run_episode(42)
        r2 = run_episode(42)
        assert r1 == r2
```

### test_retention.py

```python
import pytest
import numpy as np
from zoosim.multi_episode.retention import return_probability, sample_return
from zoosim.core.config import load_default_config

class TestRetentionModel:

    def test_probability_bounds(self):
        """P(return) should always be in [0, 1]."""
        config = load_default_config()
        for clicks in range(20):
            for sat in np.linspace(0, 2, 21):  # Include out-of-range
                p = return_probability(clicks=clicks, satisfaction=sat, config=config)
                assert 0.0 <= p <= 1.0

    def test_monotone_in_clicks(self):
        """P(return) should increase with clicks."""
        config = load_default_config()
        for sat in [0.0, 0.5, 1.0]:
            prev_p = 0.0
            for clicks in range(15):
                p = return_probability(clicks=clicks, satisfaction=sat, config=config)
                assert p >= prev_p, f"Not monotone at clicks={clicks}, sat={sat}"
                prev_p = p

    def test_monotone_in_satisfaction(self):
        """P(return) should increase with satisfaction."""
        config = load_default_config()
        for clicks in [0, 3, 7]:
            prev_p = 0.0
            for sat in np.linspace(0, 1, 11):
                p = return_probability(clicks=clicks, satisfaction=sat, config=config)
                assert p >= prev_p, f"Not monotone at clicks={clicks}, sat={sat}"
                prev_p = p

    def test_sample_return_deterministic(self):
        """Same RNG state should produce same sample."""
        config = load_default_config()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        results1 = [sample_return(clicks=3, satisfaction=0.5, config=config, rng=rng1) for _ in range(10)]
        results2 = [sample_return(clicks=3, satisfaction=0.5, config=config, rng=rng2) for _ in range(10)]

        assert results1 == results2
```

---

## Chapter Structure Outline

```
## 11.1 Motivation: Beyond Single-Session Optimization
- Why single-step proxy (delta*CLICKS) is pragmatic but incomplete
- EQ-1.2 vs EQ-1.2-prime: theory vs. practice tradeoff
- What this chapter delivers

## 11.2 Multi-Episode MDP Formulation {#S11.2}
- State space: SessionMDPState with inter-session memory
- Action space: boost vector a in R^K
- Transition: same user returns with P(return|clicks, satisfaction)
- Reward: immediate GMV (engagement implicit through continuation)
- Discount: gamma > 0 (contrast with gamma=0 bandits)

**Definition 11.2.1** (Multi-Episode Search MDP)
**Theorem 11.2.1** (Implicit Engagement): Maximizing V^pi automatically incentivizes clicks

## 11.3 Retention Modeling
- Hazard/survival interpretation
- Logistic model: P(return) = sigma(logit_base + alpha*clicks + beta*satisfaction)
- Connection to zoosim/multi_episode/retention.py
- Why this model? (Simple, interpretable, monotone, calibratable)

## 11.4 Retention Experiments {#S11.4}
- Lab 11.2: Retention curves by segment
- Empirical validation against simulator ground truth
- Monotonicity verification

## 11.5 Implementation: MultiSessionEnv
- API walkthrough (reset, step, SessionMDPState)
- Trajectory collection patterns
- Code <-> Theory boxes

## 11.6 Single-Step vs. Multi-Episode Value
- Lab 11.1: Policy ranking comparison
- When do they agree? When do they disagree?
- Theory-practice gap: single-step captures 80% with 20% complexity

## 11.7 Retention-Aware Policy Design
- Policy gradient with retention in the Bellman backup
- Balancing immediate GMV vs. retention value
- Connection to Ch10 guardrails (churn is irreversible)

## 11.8 Value Estimation and Stability
- Lab 11.3: Monte Carlo value estimation
- Variance reduction via seeding
- Acceptance criteria validation

## 11.9 Theory-Practice Gap
- What we model vs. real retention
- Limitations of logistic hazard
- Open problems in long-horizon user modeling

## 11.10 Summary
- Key results and connections
- Forward references to offline RL (Ch13 preview)

## Exercises & Labs
```

---

## Promised Sections (from forward references)

These sections were promised in other chapters:

| Section | Referenced From | Content Required |
|---------|-----------------|------------------|
| **S11.2** | Ch10 line 1122 | Retention-aware policy design |
| **S11.4** | Ch10 line 964 | Retention experiments, user return rate measurement |

---

## Broken Forward References to Fix

These Ch01 references incorrectly point to Ch11 (should be Ch10):

| File | Line | Current Text | Fix |
|------|------|--------------|-----|
| `ch01_foundations_revised_math+pedagogy_v3.md` | 850 | "Chapter 11: A/B testing, monitoring" | Change to "Chapter 10" |
| `ch01_foundations_revised_math+pedagogy_v3.md` | 854 | "Chapter 11: Production deployment" | Change to "Chapter 10" |
| `ch01_foundations_revised_math+pedagogy_v3.md` | 1444 | "Chapters 10-11: A/B testing" | Change to "Chapter 10 only" |

**Note:** A/B testing, monitoring, latency are all in Chapter 10, NOT Chapter 11.

---

## Acceptance Criteria Summary

| Criterion | Lab | Threshold | Test Method |
|-----------|-----|-----------|-------------|
| Policy ordering correlation | Lab 11.1 | Spearman rho >= 0.80 | Compare 5+ policies |
| Retention monotone in clicks | Lab 11.2 | d P/d clicks >= 0 everywhere | Sweep clicks 0-15 |
| Retention monotone in satisfaction | Lab 11.2 | d P/d sat >= 0 everywhere | Sweep sat 0-1 |
| Value stability (CV) | Lab 11.3 | CV < 0.10 across seeds | 10+ seeds |

---

## Dependencies

**What Ch11 builds on:**
- Ch01: EQ-1.2-prime, reward formulation, delta*CLICKS proxy
- Ch02: Click models, satisfaction signals
- Ch03: Bellman equation, gamma > 0 discount
- Ch04-05: Catalog, users, queries, features
- Ch06-07: Agent policies that operate on states

**What builds on Ch11:**
- Ch09: OPE for multi-episode trajectories (importance weights compound)
- Ch10: Guardrails must account for retention (churn is irreversible)

---

## Implementation Order

**Recommended sequence:**

1. **Tests first** (`tests/ch11/`) --- Validate existing code, establish baselines
2. **Utility scripts** (`scripts/ch11/utils/`) --- Trajectory collection, test policies
3. **Lab 11.2** (retention curves) --- Simpler, validates retention model
4. **Lab 11.1** (single vs multi) --- Core experiment, needs trajectory utils
5. **Lab 11.3** (stability) --- Validation of reproducibility
6. **Chapter text** --- Write with labs as evidence
7. **exercises_labs.md** --- Package labs with exercises

---

## Knowledge Graph Entry (when complete)

```yaml
CH-11:
  kind: chapter
  title: "Multi-Episode Inter-Session MDP"
  status: complete  # Update from 'planned' when done
  file: docs/book/ch11/ch11_multi_episode_mdp.md
  summary: "Extends single-session bandits to multi-episode MDPs with retention/satisfaction state. Shows engagement enters implicitly through retention dynamics. Implements EQ-1.2-prime."
  forward_refs:
    - EQ-1.2-prime
    - S11.2
    - S11.4
  depends_on: [CH-1, CH-3, CH-10]
  code_modules:
    - zoosim/multi_episode/session_env.py
    - zoosim/multi_episode/retention.py
  labs:
    - Lab 11.1: Single-step vs multi-episode value
    - Lab 11.2: Retention curves by segment
    - Lab 11.3: Value estimation stability
```

---

## Resume Instructions

**When resuming tomorrow:**

1. Load context:
   - `@vlad_prytula.md` + `@vlad_application_mode.md` (Application Mode)
   - This plan file
   - `zoosim/multi_episode/session_env.py` and `retention.py`

2. Start with tests:
   ```bash
   mkdir -p tests/ch11
   # Create test_session_env.py, test_retention.py
   pytest tests/ch11/ -v
   ```

3. Create utility scripts:
   ```bash
   mkdir -p scripts/ch11/utils
   # Create trajectory.py, policies.py
   ```

4. Implement labs in order: 11.2 -> 11.1 -> 11.3

5. Write chapter text with labs as evidence

---

**Last updated:** 2025-12-09
**Next action:** Create tests/ch11/ directory and test files
