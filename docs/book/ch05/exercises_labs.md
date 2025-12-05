# Chapter 5 — Exercises & Labs

## Overview

This document contains exercises and labs for Chapter 5 (Relevance, Features, and Reward). Total estimated time: **90-120 minutes**.

**Exercise breakdown:**
- **Analytical (20%)**: Proofs and mathematical derivations
- **Implementation (50%)**: Code exercises building on existing modules
- **Experimental (25%)**: Run ablations, visualize results
- **Conceptual (5%)**: Explain design choices and trade-offs

---

## Analytical Exercises

### Exercise 5.1: Cosine Similarity Properties (15 min)

**Question:**

Given query embedding $\mathbf{q} \in \mathbb{R}^d$ and product embeddings $\mathbf{e}_1, \mathbf{e}_2 \in \mathbb{R}^d$, revisit Proposition 5.1 and prove its parts from first principles:

a) Using the Cauchy–Schwarz inequality, show that $s_{\text{sem}}(\mathbf{q}, \mathbf{e}_1) \in [-1,1]$ and explain when equality is attained.

b) Prove symmetry and scale invariance: $\cos(\mathbf{q}, \mathbf{e}_1) = \cos(\mathbf{e}_1, \mathbf{q})$ and, for $\alpha, \beta > 0$, $\cos(\alpha \mathbf{q}, \beta \mathbf{e}_1) = \cos(\mathbf{q}, \mathbf{e}_1)$.

c) Show that $\cos(\mathbf{q}, \mathbf{e}_1) = 0$ if and only if $\mathbf{q} \perp \mathbf{e}_1$, and interpret this in terms of semantic relevance for search.

d) (Concept check) Cosine similarity is not a metric, so it does not satisfy the triangle inequality in general. Construct a simple counterexample in $\mathbb{R}^2$ where $\cos(\mathbf{q}, \mathbf{e}_1) + \cos(\mathbf{e}_1, \mathbf{e}_2) < \cos(\mathbf{q}, \mathbf{e}_2)$.

**References:** [DEF-5.2], [EQ-5.1], [PROP-5.1]

---

### Exercise 5.2: Feature Standardization Invariance (20 min)

**Question:**

Suppose we standardize features via [EQ-5.6]: $\tilde{\phi}_i^{(j)} = \frac{\phi_i^{(j)} - \mu^{(j)}}{\sigma^{(j)}}$. Use Proposition 5.2 as a guide and fill in the details.

a) Starting from the definition of $\mu^{(j)}$ and $\sigma^{(j)}$, prove that the standardized coordinates have zero mean and unit variance, i.e.
$$
\frac{1}{N} \sum_{i=1}^N \tilde{\phi}_i^{(j)} = 0,
\qquad
\frac{1}{N} \sum_{i=1}^N (\tilde{\phi}_i^{(j)})^2 = 1
$$
whenever $\sigma^{(j)} > 0$.

b) Show that standardization preserves the relative ordering of feature values: if $\phi_i^{(j)} > \phi_k^{(j)}$, then $\tilde{\phi}_i^{(j)} > \tilde{\phi}_k^{(j)}$. Explain why the proof uses only the fact that standardization is an affine map with positive slope.

c) Consider an affine transformation $\phi_i^{(j)} \mapsto a \phi_i^{(j)} + b$ with $a \neq 0$. Derive the new mean and standard deviation and argue that, after standardization, the ordering of the original values is still preserved.

d) Discuss the constant-feature case: if all products have the same value for feature $j$, how does the definition in [DEF-5.6] ensure that the standardized values are well defined? Why is setting $\sigma^{(j)} = 1$ a natural convention here?

**References:** [DEF-5.6], [EQ-5.6], [PROP-5.2]

---

### Exercise 5.3: Reward Weight Constraints (15 min)

**Question:**

Consider the engagement weight guideline [CONSTRAINT-5.8]: $\frac{\delta}{\alpha} \in [0.01, 0.10]$.

a) Discuss the role of a strictly positive lower bound. What happens if $\delta = 0$ (no engagement term at all) in terms of the long-run modeling intent described in Remark 5.1? When might a nonzero but very small $\delta$ be preferable to exactly zero?

b) Using [EQ-5.7], construct a simple pair of scenarios (you may assume $\alpha = 1.0$, $\beta = 1.0$, $\gamma = 0$, and CM2 $= 0$ for simplicity) where a large value of $\delta / \alpha$ makes a policy with many low-value clicks but low GMV more attractive than a policy with fewer clicks but higher GMV. Compute the rewards explicitly and identify for which ratios $\delta / \alpha$ this inversion occurs.

c) Show by example that the guideline in [CONSTRAINT-5.8] is **not** a formal guarantee of “no clickbait”. Construct two policies with the same GMV but different click patterns and argue that, even with $\delta / \alpha \le 0.10$, the policy with slightly higher clicks could still be preferred. Explain how this illustrates the heuristic nature of the bound.

d) Propose an alternative or complementary constraint design that directly controls clickbait risk. Examples include explicit constraints on conversion rate (CVR), or a penalty term that downweights clicks that do not lead to purchases. Explain how your proposal would interact with [EQ-5.7] and with the multi-objective view in Section 5.6.

**References:** [EQ-5.7], [EQ-5.8], [CONSTRAINT-5.8], [REM-5.1]

---

## Implementation Exercises

### Exercise 5.4: Implement Lexical Component with Fuzzy Matching (30 min)

**Background:**

The current lexical component [EQ-5.2] uses exact token overlap: $s_{\text{lex}}(q, p) = \log(1 + |T_q \cap T_p|)$. This fails on misspellings (e.g., `"liter"` vs `"litter"`).

**Task:**

Implement a **fuzzy lexical component** that allows approximate matches using **edit distance** (Levenshtein distance).

**Requirements:**

a) Write a function `fuzzy_lexical_component(query_tokens, product_category, threshold=2)` that:
   - Computes edit distance between each query token and each product token
   - Counts a match if edit distance ≤ threshold
   - Returns $\log(1 + \text{fuzzy\_matches})$

b) Test on examples:
   - Query: `{"liter"}`, Product: `"litter"` → Should match (edit distance = 1)
   - Query: `{"dog"}`, Product: `"cat_food"` → Should not match
   - Query: `{"prmeium"}`, Product: `"premium_dog_food"` → Should match (edit distance = 2)

c) Compare fuzzy vs. exact lexical matching on 100 random query-product pairs. Report:
   - Number of matches gained by fuzzy matching
   - False positive rate (matches that shouldn't match)

**Starter code:**

```python
import numpy as np
from Levenshtein import distance as edit_distance  # pip install python-Levenshtein

def fuzzy_lexical_component(query_tokens: set[str], product_category: str, threshold: int = 2) -> float:
    """Fuzzy lexical relevance via edit distance.

    Args:
        query_tokens: Set of query tokens
        product_category: Product category string (e.g., "cat_food")
        threshold: Maximum edit distance for a match

    Returns:
        Log(1 + fuzzy_matches) where fuzzy_matches counts approximate token overlaps
    """
    product_tokens = set(product_category.split("_"))
    fuzzy_matches = 0

    for q_token in query_tokens:
        for p_token in product_tokens:
            if edit_distance(q_token, p_token) <= threshold:
                fuzzy_matches += 1
                break  # Count each query token at most once

    return np.log1p(fuzzy_matches)

# TODO: Implement and test
```

**Deliverable:** Working function + test results showing improvement on misspelled queries.

**Time estimate:** 30 minutes

---

### Exercise 5.5: Add Temporal Features (40 min)

**Background:**

Our feature vector [EQ-5.5] is static (no time-dependent features). In production, seasonality and trends matter:
- Winter: Dog coats, holiday treats
- Summer: Cooling mats, outdoor toys

**Task:**

Extend `compute_features()` to include **temporal features** that capture seasonality and trends.

**Requirements:**

a) Add a `timestamp` parameter to `compute_features()` (assume it's a Unix timestamp or datetime object).

b) Compute three temporal features:
   - **Day of week** (0-6): `datetime.weekday()` → one-hot encode or use sine/cosine encoding
   - **Month of year** (1-12): `datetime.month` → sine/cosine encoding (captures periodicity)
   - **Days since product launch**: `(timestamp - product.launch_date).days` (trend signal)

c) For sine/cosine encoding of periodic features:
   - Day of week: $\sin(2\pi \cdot \text{weekday} / 7)$, $\cos(2\pi \cdot \text{weekday} / 7)$
   - Month: $\sin(2\pi \cdot \text{month} / 12)$, $\cos(2\pi \cdot \text{month} / 12)$

d) Update `ActionConfig.feature_dim` from 10 to 15 (10 existing + 5 temporal).

e) Test on synthetic data:
   - Generate 100 episodes across different dates (e.g., one per week for 2 years)
   - Verify that winter months have different feature distributions than summer

**Starter code:**

```python
from datetime import datetime
import math

def compute_features_with_time(
    *,
    user: User,
    query: Query,
    product: Product,
    config: SimulatorConfig,
    timestamp: datetime
) -> List[float]:
    """Compute feature vector with temporal features.

    Extends [EQ-5.5] with 5 additional temporal features:
    - [10-11]: Day of week (sin, cos)
    - [12-13]: Month of year (sin, cos)
    - [14]: Days since product launch (log-scaled)

    Args:
        user, query, product, config: As in original compute_features
        timestamp: Current timestamp

    Returns:
        Feature vector, length 15
    """
    # Original 10 features
    base_features = compute_features(user=user, query=query, product=product, config=config)

    # Temporal features
    weekday = timestamp.weekday()
    month = timestamp.month
    days_since_launch = (timestamp - product.launch_date).days

    temporal_features = [
        math.sin(2 * math.pi * weekday / 7),
        math.cos(2 * math.pi * weekday / 7),
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
        math.log1p(days_since_launch),  # Log-scale to compress large values
    ]

    return base_features + temporal_features

# TODO: Implement Product.launch_date generation in catalog.py
# TODO: Test temporal feature distributions by season
```

**Deliverable:** Extended feature function + plot showing seasonal variation.

**Time estimate:** 40 minutes

---

### Exercise 5.6: Implement Reward Capping for Robustness (25 min)

**Background:**

The reward function [EQ-5.7] is unbounded: if a user purchases 10 expensive products, reward could be $1000+. This makes RL training unstable (rare high-reward episodes dominate gradients).

**Task:**

Implement **reward clipping** to bound the reward magnitude while preserving relative ordering.

**Requirements:**

a) Modify `compute_reward()` to add an optional `reward_cap` parameter.

b) Apply **soft clipping** via $\text{tanh}$ transformation:
   $$
   R_{\text{clipped}} = R_{\text{cap}} \cdot \tanh\left(\frac{R}{R_{\text{cap}}}\right)
   $$

   Properties:
   - $R \in (-\infty, +\infty)$ → $R_{\text{clipped}} \in (-R_{\text{cap}}, +R_{\text{cap}})$
   - Preserves sign: positive rewards stay positive
   - Smooth: differentiable (good for gradient-based RL)
   - Asymptotic: $\lim_{R \to \infty} R_{\text{clipped}} = R_{\text{cap}}$

c) Compare clipped vs. unclipped rewards on 1000 simulated episodes:
   - Compute reward variance (should decrease with clipping)
   - Check that relative ordering is mostly preserved (correlation > 0.95)

d) Visualize reward distributions (histogram) before and after clipping.

**Starter code:**

```python
import numpy as np

def compute_reward_with_clipping(
    *,
    ranking: Sequence[int],
    clicks: Sequence[int],
    buys: Sequence[int],
    catalog: Sequence[Product],
    config: SimulatorConfig,
    reward_cap: float = 100.0,
) -> Tuple[float, RewardBreakdown]:
    """Compute reward with soft clipping for stability.

    Extends [EQ-5.7] with tanh clipping to bound reward magnitude.

    Args:
        Same as compute_reward, plus:
        reward_cap: Clipping threshold (default: 100.0)

    Returns:
        Clipped reward, original breakdown
    """
    reward, breakdown = compute_reward(
        ranking=ranking, clicks=clicks, buys=buys, catalog=catalog, config=config
    )

    # Soft clipping via tanh
    reward_clipped = reward_cap * np.tanh(reward / reward_cap)

    return reward_clipped, breakdown

# TODO: Implement and test on simulated episodes
```

**Deliverable:** Working function + comparison plots showing variance reduction.

**Time estimate:** 25 minutes

---

## Experimental Exercises

### Lab 5.A: Feature Distribution Analysis by Segment (30 min)

**Objective:** Validate that features capture user segment heterogeneity.

**Setup:**

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import generate_catalog
from zoosim.world.users import sample_user
from zoosim.world.queries import sample_query
from zoosim.ranking.features import compute_features

config = SimulatorConfig(seed=42)
rng = np.random.default_rng(config.seed)
catalog = generate_catalog(config=config, rng=rng)
```

**Tasks:**

a) **Sample features for each segment:**
   - For each of the 4 user segments (price_hunter, pl_lover, premium, litter_heavy)
   - Sample 200 users from that segment
   - For each user, sample a query and compute features for 20 random products
   - Store features in DataFrame with columns: `['segment', 'cm2', 'discount', 'personalization', ...]`

b) **Visualize distributions:**
   - Create a 3×3 grid of subplots (9 features, excluding `cm2_litter` which is sparse)
   - For each feature, plot histograms for all 4 segments (overlaid, different colors)
   - Add legend and axis labels

c) **Statistical tests:**
   - For each feature, run ANOVA to test if means differ significantly across segments
   - Report p-values (expect personalization, discount_x_price_sens, pl_x_pl_aff to have p < 0.01)

d) **Interpretation:**
   - Which features show the largest segment differences?
   - Which features are segment-invariant (as expected)?
   - Are there any surprising patterns?

**Expected output:**
- 9-panel figure: `feature_distributions_by_segment.png`
- ANOVA p-values table printed to console

**Time estimate:** 30 minutes

---

### Lab 5.B: Pareto Frontier Sweep (35 min)

**Objective:** Trace the Pareto frontier by sweeping reward weights.

**Setup:**

```python
from zoosim.core.config import SimulatorConfig
from zoosim.envs.search_env import ZooplusSearchEnv
import numpy as np
import matplotlib.pyplot as plt

config = SimulatorConfig(seed=2025)
```

**Tasks:**

a) **Simulate policies with different weights:**
   - Create a grid of $(\alpha, \beta)$ values: $\alpha, \beta \in \{0.5, 0.75, 1.0, 1.25, 1.5\}$
   - For each $(\alpha, \beta)$ pair:
     - Update `config.reward.alpha_gmv` and `config.reward.beta_cm2`
     - Run 100 episodes using a random baseline policy (uniform random boosts in $[-0.1, +0.1]$)
     - Record mean GMV, mean CM2, mean strategic purchases, mean clicks

b) **Plot Pareto frontier:**
   - Scatter plot: x-axis = mean GMV, y-axis = mean CM2
   - Color points by $\alpha$ value (colorbar)
   - Mark the default policy ($\alpha = \beta = 1.0$) with a red star
   - Add grid lines for readability

c) **Identify Pareto-optimal policies:**
   - Compute the **Pareto frontier**: subset of policies where no other policy dominates on both GMV and CM2
   - Overlay Pareto frontier as a red line on the scatter plot

d) **Sensitivity analysis:**
   - Fix $\beta = 1.0$, sweep $\alpha \in [0.1, 2.0]$ (20 values)
   - Plot GMV vs. $\alpha$ and CM2 vs. $\alpha$ (two subplots)
   - Identify the $\alpha$ value that maximizes GMV (should be $\alpha \gg \beta$)
   - Identify the $\alpha$ value that maximizes CM2 (should be $\alpha \ll \beta$)

**Expected output:**
- Figure 1: `pareto_frontier_gmv_cm2.png` (scatter plot with Pareto frontier)
- Figure 2: `sensitivity_alpha.png` (GMV and CM2 vs. $\alpha$)

**Time estimate:** 35 minutes

---

### Lab 5.C: Relevance Model Ablation (30 min)

**Objective:** Compare semantic-only, lexical-only, and hybrid relevance models.

**Setup:**

```python
from zoosim.core.config import SimulatorConfig, RelevanceConfig
from zoosim.world.catalog import generate_catalog
from zoosim.world.users import sample_user
from zoosim.world.queries import sample_query
from zoosim.ranking.relevance import batch_base_scores
import numpy as np

config = SimulatorConfig(seed=42)
rng = np.random.default_rng(config.seed)
catalog = generate_catalog(config=config, rng=rng)
```

**Tasks:**

a) **Define three relevance configurations:**
   - **Semantic-only**: `w_sem=1.0, w_lex=0.0`
   - **Lexical-only**: `w_sem=0.0, w_lex=1.0`
   - **Hybrid (default)**: `w_sem=0.7, w_lex=0.3`

b) **Sample 100 queries** (varied query types: category, brand, generic) and compute base scores for all products under each configuration.

c) **Compute ranking quality metrics:**
   - **Category precision @10**: Fraction of top-10 products matching query's intended category
   - **Diversity @10**: Number of unique categories in top 10
   - **Mean Reciprocal Rank (MRR)**: $\frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank of first relevant product}}$

d) **Aggregate results by query type:**
   - For each query type (category, brand, generic), report metrics for all three configurations
   - Identify which configuration performs best for each query type

e) **Visualize:**
   - Bar plot: x-axis = query type, y-axis = precision @10, grouped by configuration (3 bars per query type)

**Expected findings:**
- **Semantic-only**: High precision for generic queries (embeddings capture intent)
- **Lexical-only**: High precision for category queries (exact token match)
- **Hybrid**: Best overall (balanced performance across query types)

**Expected output:**
- Table: Metrics by query type and configuration
- Figure: `relevance_ablation_precision.png`

**Time estimate:** 30 minutes

---

## Conceptual Exercises

### Exercise 5.7: Designing an Engagement Metric (10 min)

**Question:**

The current engagement proxy is **click count** $\delta \cdot \text{Clicks}$ [EQ-5.7]. This is imperfect (see Section 5.7.3).

a) **Alternative metrics**: Propose three alternative engagement metrics that could replace or complement click count. For each, discuss:
   - What signal does it capture?
   - How would you measure it in the simulator?
   - What are its limitations?

   Example alternatives: dwell time, scroll depth, add-to-cart rate, bounce rate

b) **Multi-metric approach**: Instead of a single engagement term, consider:
   $$
   R = \alpha \cdot \text{GMV} + \beta \cdot \text{CM2} + \delta_1 \cdot \text{Clicks} + \delta_2 \cdot \text{AddToCarts} + \delta_3 \cdot \text{DwellTime}
   $$

   What challenges arise when aggregating multiple engagement signals? How would you set $\delta_1, \delta_2, \delta_3$?

c) **Long-term engagement**: In Chapter 11, we'll model retention (users returning for future sessions). How could retention rate be incorporated into the reward? Sketch a formula.

**Deliverable:** Written responses (1-2 paragraphs per part)

**Time estimate:** 10 minutes

---

### Exercise 5.8: Scalarization and Non-Convex Pareto Frontiers (15 min)

Theorem 5.1 shows that if we maximize a positively weighted scalarization of several objectives, the resulting policy is weakly Pareto optimal. The converse is not true: in general, there are Pareto-optimal policies that cannot be obtained as maximizers of any linear scalarization with positive weights. In this exercise you will construct a simple example illustrating this gap and connect it to real-world trade-offs.

Consider a simplified two-objective setting with:
- $C_1$: Expected GMV (normalized to lie in $[0,1]$),
- $C_2$: A fairness or exposure-parity metric (also normalized to $[0,1]$).

Suppose we have three candidate policies with the following expected outcomes:
$$
C(\pi_A) = (0, 1), \quad
C(\pi_B) = (1, 0), \quad
C(\pi_C) = (0.4, 0.4).
$$

1. Show that all three policies are weakly Pareto optimal in the sense of Definition 5.8. In particular, check that no policy dominates $\pi_C$ on both objectives, even though both $\pi_A$ and $\pi_B$ are strictly better on one coordinate.

2. For an arbitrary weight vector $(\alpha, \beta) \in \mathbb{R}_+^2$ with $\alpha, \beta > 0$, consider the scalarized objective
   $$
   J(\pi) = \alpha \, C_1(\pi) + \beta \, C_2(\pi).
   $$
   Compute $J(\pi_A)$, $J(\pi_B)$, and $J(\pi_C)$ in terms of $\alpha$ and $\beta$. Show that, for every such weight vector, either $\pi_A$ or $\pi_B$ has strictly larger $J(\cdot)$ than $\pi_C$. Conclude that $\pi_C$ is never a maximizer of any positively weighted scalarization, even though it is weakly Pareto optimal.

3. Interpret this example in a real search-ranking setting. Think of $\pi_A$ as an “extreme fairness” policy (perfect parity but no GMV), $\pi_B$ as a “pure revenue” policy (maximal GMV but severe exposure imbalance), and $\pi_C$ as a “compromise” that sacrifices some GMV and some fairness to achieve a more balanced outcome. Explain why linear scalarization may systematically favor extreme policies like $\pi_A$ or $\pi_B$ and miss moderate compromises like $\pi_C$, especially when the feasible set of policies leads to a non-convex Pareto frontier.

4. Briefly connect your analysis to Theorem 5.1: what does the theorem guarantee, and what does this exercise show it does *not* guarantee? How does this motivate the use of alternative approaches (e.g., constrained optimization or explicit frontier exploration in later chapters) when balancing GMV against fairness, CM2, or long-term retention?

**References:** [EQ-5.7], [DEF-5.8], [THM-5.1]

---

## Summary

**Total time:** ~90-120 minutes

| Exercise | Type | Time | Topics |
|----------|------|------|--------|
| 5.1 | Analytical | 15 min | Cosine similarity properties |
| 5.2 | Analytical | 20 min | Feature standardization |
| 5.3 | Analytical | 15 min | Reward weight constraints |
| 5.4 | Implementation | 30 min | Fuzzy lexical matching |
| 5.5 | Implementation | 40 min | Temporal features |
| 5.6 | Implementation | 25 min | Reward clipping |
| Lab 5.A | Experimental | 30 min | Feature distributions |
| Lab 5.B | Experimental | 35 min | Pareto frontier |
| Lab 5.C | Experimental | 30 min | Relevance ablation |
| 5.7 | Conceptual | 10 min | Engagement metric design |

**Next steps:**
- Complete exercises for full understanding of relevance, features, and reward
- Prepare for Chapter 6: RL agents (LinUCB, Thompson Sampling) will use these features
- In Chapter 10, we'll implement production guardrails (CM2 floors, ΔRank@k) applying CMDP theory from §3.6

---

## Solutions

Solutions for analytical exercises (5.1-5.3) and starter code for implementation exercises (5.4-5.6) will be provided in the course repository under `solutions/ch05/`.

**Self-check:**
- After completing exercises, run the validation script:
  ```bash
  python scripts/validate_ch05.py
  ```
- This checks that your implementations match expected outputs on test cases.
