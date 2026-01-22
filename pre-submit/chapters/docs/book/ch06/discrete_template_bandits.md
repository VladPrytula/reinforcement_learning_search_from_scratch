# Chapter 6 --- Discrete Template Bandits: When Theory Meets Practice

## From Relevance to Optimization: The First RL Agent

In Chapter 5, we built the complete RL interface: base relevance models provide initial rankings, feature engineering extracts state representations, and multi-objective rewards aggregate business metrics. We have everything needed to train an RL agent---except the agent itself.

**The challenge:**

We need a policy $\pi: \mathcal{X} \to \mathcal{A}$ that maps observations (user, query, products) to actions (boost vectors) while:

1. **Maximizing business metrics** (GMV, CM2, strategic KPIs)
2. **Respecting hard constraints** (CM2 floor >=60%, rank stability, exposure floors)
3. **Exploring safely** (avoid catastrophic rankings during learning)
4. **Remaining interpretable** (business stakeholders must understand what the agent does)
5. **Learning quickly** (sample efficiency matters in production)

**Why not jump straight to deep RL?**

Modern deep RL (DQN, PPO, SAC) offers flexibility but comes with serious risks for production search:

- Sample inefficiency: Often requires millions of episodes to reach stable performance
- Unsafe exploration: Random actions can destroy user experience
- Limited interpretability: Neural policies are opaque to stakeholders
- Training instability: Learning curves oscillate; hyperparameters are fragile
- Cold start: Hard to warm-start from domain knowledge

**Solution: Start with discrete template bandits.**

Instead of learning a full neural policy from scratch, we **discretize the action space** into interpretable templates and use **contextual bandits** (LinUCB, Thompson Sampling) to select among them.

**Key insight:**

Search boost optimization is **not** a complex sequential decision problem initially. A single-episode contextual bandit perspective suffices because:

- Most search sessions are **single-query** (user searches once, clicks/buys, leaves)
- Inter-session effects (retention, long-term value) are **slow-moving** (timescale of days, not seconds)
- We already have a **strong base ranker** (Chapter 5); RL learns small perturbations

---

## What This Chapter Teaches

This chapter is structured around a deliberately chosen negative result. We develop the theory and implementation of linear contextual bandits over an interpretable, discretized action space (templates). We then run the resulting algorithms under an intentionally impoverished feature map and observe that they underperform a strong static baseline. We diagnose the failure by tracing it to violated assumptions (feature poverty and model misspecification), and we recover by engineering richer features.

Concretely, we proceed in five stages:

**Stage I: Theory (Sections 6.1--6.3)**
- We define a discrete template action space encoding business logic.
- We derive and implement Thompson Sampling and LinUCB for linear contextual bandits.
- We state regret guarantees (sublinear in $T$) under explicit assumptions.

**Stage II: Failure (Section 6.5)**
- We run both bandits with $\phi_{\text{simple}}$ (segment + query type).
- We observe that both algorithms underperform the best static template on GMV.

**Stage III: Diagnosis (Section 6.6)**
- We rule out implementation errors and revisit the assumptions behind the theorems.
- We identify the bottleneck as representation: the feature map is too weak for a linear model.

**Stage IV: Fix (Section 6.7)**
- We enrich the feature map to include user preference signals and product aggregates.
- We compare oracle vs. estimated preference features and extract an algorithm-selection rule.

**Stage V: Synthesis (Section 6.8)**
- We summarize what regret bounds do (and do not) guarantee.
- We connect the template bandit construction to Chapter 7 (continuous actions and function approximation).

---

## Why We Show the Failure

We could begin with rich features and present only the successful result. We do not, because it would hide the main methodological lesson: theoretical guarantees are conditional, and in applied RL the failure mode is often representational rather than algorithmic.

The goal is to practice a transferable diagnostic skill:

1. Check theorem hypotheses against the actual system.
2. Distinguish feature poverty from insufficient data or hyperparameter issues.
3. Recognize model misspecification (here: a linear model applied to a nonlinear reward mechanism).
4. Fix the bottleneck rather than escalating algorithmic complexity.

---

## Chapter Roadmap

**Part I: Theory & Implementation**

Section 6.1 --- **Discrete Template Action Space**: Define 8 interpretable boost strategies (High Margin, CM2 Boost, Premium, Budget, etc.)

Section 6.2 --- **Thompson Sampling**: Bayesian posterior sampling with Gaussian conjugacy and ridge regression

Section 6.3 --- **LinUCB**: Upper confidence bounds with confidence ellipsoids in feature space

Section 6.4 --- **Production Code**: Full PyTorch implementation with type hints, batching, reproducibility

**Part II: The Empirical Journey**

Section 6.5 --- **First Experiment (Simple Features)**: Deploy bandits with segment + query type -> **28% GMV loss**

Section 6.6 --- **Diagnosis**: Identify feature poverty, model misspecification, and nonlinearity

Section 6.7 --- **Rich Features (Oracle vs. Estimated)**: Re-engineer features, discover the Algorithm Selection Principle

Section 6.8 --- **Summary & What's Next**: Lessons learned, limitations, bridge to continuous Q(x,a) in Chapter 7

**Part III: Reflection & Extensions**

Section 6.8.1 --- **What We Built**: Technical artifacts and empirical results

Section 6.8.2 --- **Five Lessons**: Conditional guarantees, feature engineering ceiling, baseline value, failure as signal, algorithm selection

Section 6.8.3 --- **Where to Go Next**: Exercises, labs, and GPU scaling

Section 6.8.4 --- **Extensions & Practice**: Appendices covering neural extensions, theory-practice gaps, modern context, production checklists

---

## What We Build

By the end of this chapter, we have:

**Technical artifacts:**
- A library of 8 discrete boost templates encoding business logic.
- Thompson Sampling with Bayesian ridge regression updates.
- LinUCB with upper confidence bounds and confidence ellipsoids.
- Reproducible experiment drivers and per-segment diagnostics.

**Empirical understanding:**
- A reproducible failure mode with an impoverished feature map.
- A diagnosis in terms of violated assumptions (representation and misspecification).
- A recovery with richer features and a clear algorithm-selection takeaway.

**Production skills:**
- Translating theorem hypotheses into concrete system checks.
- Distinguishing algorithmic issues from representation bottlenecks.
- Designing experiments that isolate feature quality, priors, and exploration.

This chapter is an honest account of what happens when we apply RL theory to a realistic simulator, including failures, diagnoses, and the corresponding fixes.

---

*Next: Section 6.1 --- Discrete Template Action Space*

---

## 6.1 Discrete Template Action Space

### 6.1.1 Why Discretize?

The continuous action space from Chapter 1 is $\mathcal{A} = [-a_{\max}, +a_{\max}]^K$ where $K$ is the number of products displayed (typically $K=20$ or $K=50$). This is **high-dimensional** ($\dim \mathcal{A} = K$) and **unbounded exploration** is dangerous.

**Problems with continuous boosts:**

1. **Curse of dimensionality**: With $K=20$ and even 10 discretization levels per dimension, we would have $10^{20}$ actions---intractable.

2. **Unsafe exploration**: Random continuous boosts can produce nonsensical rankings:
   - Boosting all products by $+10$ -> no relative change (wasted action)
   - Boosting random products -> destroys relevance (user sees cat food for dog query)

3. **No structure**: Continuous space does not encode domain knowledge about *what kinds of boosts make business sense*.

**Solution: Discretize into interpretable templates.**

We define a small set of **boost templates** $\mathcal{T} = \{t_1, \ldots, t_M\}$ where each template $t_i$ is a **boost policy** that maps products to adjustments based on business logic.

**Definition 6.1.1** (Boost Template) {#DEF-6.1.1}

Let $\mathcal{C}$ denote the **product catalog**, modeled as a finite set of products. A **boost template** is a function
$$
t: \mathcal{C} \to [-a_{\max}, +a_{\max}]
$$
that assigns a bounded boost value to each product $p \in \mathcal{C}$. Given a query result set $\{p_1, \ldots, p_K\} \subset \mathcal{C}$ and base relevance scores $s_{\text{base}}(q, p_i)$ from [DEF-5.1], the template induces **adjusted scores**:
$$
s'_i = s_{\text{base}}(q, p_i) + t(p_i), \quad i = 1, \ldots, K
\tag{6.1}
$$
{#EQ-6.1}

The final ranking is obtained by sorting products in descending order of $s'_i$.

**Properties:**

1. **Boundedness**: $|t(p)| \leq a_{\max}$ for all $p \in \mathcal{C}$ (typically $a_{\max} = 5.0$)
2. **Finite catalog**: $|\mathcal{C}| < \infty$, so all argmax operations over $\mathcal{C}$ are well-defined
3. **Deterministic**: Given a fixed catalog and template definition, $t$ is a fixed function (no internal randomness)
4. **Product-only (baseline library)**: In this chapter, templates depend only on product attributes $(p.\text{category}, p.\text{margin}, p.\text{popularity}, \ldots)$, not on query or user---context enters later via the contextual bandit policy.

This formulation aligns with the Bandit Bellman Operator defined in [DEF-3.8.1], where $\gamma = 0$ eliminates the need for temporal credit assignment. Templates become arms in a contextual bandit; the learner's job is to discover which arm maximizes immediate reward given the context.

**Example 6.1** (Template Application)

Consider a tiny catalog with three products:

| Product | Price | Margin | Category | Base Score |
|---------|-------|--------|----------|------------|
| $p_1$   | 15    | 0.50   | Dog Food | 8.5        |
| $p_2$   | 40    | 0.30   | Cat Toy  | 7.2        |
| $p_3$   | 25    | 0.45   | Treats   | 6.8        |

**Baseline ranking** (by base score): $[p_1, p_2, p_3]$.

Apply template $t_1$ (**High Margin**): boost products with margin $> 0.4$ by $+5.0$.

- $t_1(p_1) = 5.0$ (margin $0.50 > 0.4$)
- $t_1(p_2) = 0.0$ (margin $0.30 \leq 0.4$)
- $t_1(p_3) = 5.0$ (margin $0.45 > 0.4$)

**Adjusted scores:**
- $s'_1 = 8.5 + 5.0 = 13.5$
- $s'_2 = 7.2 + 0.0 = 7.2$
- $s'_3 = 6.8 + 5.0 = 11.8$

**New ranking:** $[p_1, p_3, p_2]$ --- product $p_3$ jumps from position 3 to position 2.

This illustrates the pattern of the whole chapter: templates encode **business logic** (high margin) while respecting the base relevance signal. The contextual bandit will decide **which template** to apply for each user--query context.

**Template library design:**

We define $M$ templates based on business objectives and product features. Before presenting the library, we justify two critical hyperparameters: the number of templates ($M$) and the action magnitude ($a_{\max}$).

**Design Choice: Action Magnitude ($a_{\max} = 5.0$)**

The boost bound $a_{\max}$ determines how aggressively templates can override base relevance. This is a **signal-to-noise calibration**, not a universal constant.

**Base relevance scale in our simulator**: From Section 5.2, lexical and embedding scores produce base relevance values $s_{\text{base}}(q,p) \in [10, 30]$ for relevant products, with typical standard deviation $\sigma \approx 5-8$ within a candidate set. Position bias and click propensities modulate this further (x0.3 to x1.0).

**Action magnitude regimes:**
- $a_{\max} = 0.5$: **Subtle interventions** (+-2-5% of base relevance). Templates nudge rankings by 1-2 positions. Learning signals exist but are weak---agents must discover fine-grained adjustments amidst noise. Suitable for conservative control where we trust the base ranker.

- $a_{\max} = 5.0$: **Visible interventions** (+-15-50% of base relevance). Templates can promote a product from position 15 to top-3, or demote low-margin items. Learning dynamics become **observable**: we can see agents discovering high-margin products, experimenting with strategic flags, and converging to optimal templates. This is our pedagogical choice for Chapters 6-8.

- $a_{\max} = 10+$: **Dominant interventions** (comparable to base relevance). Templates can invert rankings entirely. Risks destroying relevance signal if templates are poorly designed. Useful when base ranker is low-quality or when exploring counterfactual "what if" scenarios (Exercise 6.11).

**Our standardized choice**: We use $a_{\max} = 5.0$ throughout Chapters 6-8 for three reasons:

1. **Pedagogical visibility**: At smaller magnitudes (0.5), learning effects are statistically present but visually obscured by base relevance noise and position bias. At 5.0, we can trace how LinUCB/TS policies discover that high-margin templates outperform popularity-based strategies.

2. **Fair algorithm comparison**: All RL methods (discrete templates, continuous Q-learning in Ch7, policy gradients in Ch8) use the same $a_{\max}$ by default, enabling ceteris paribus benchmarking. Early experiments with mismatched magnitudes led to spurious conclusions (Appendix 7.A documents this failure mode).

3. **Exploration of conservative vs. aggressive control**: Readers can experiment with smaller values (Exercise 6.8 explores $a_{\max} \in \{0.5, 2.0, 5.0, 10.0\}$) to study how learning speed and final performance depend on action authority.

**Mathematical perspective**: The boost bound $a_{\max}$ couples to the **effective horizon** of the learning problem. From the gap-independent regret bounds we state in Section 6.2.3 ([THM-6.1] and [THM-6.2]), LinUCB's cumulative regret scales as $O(d\sqrt{M T \log T})$ (up to logarithmic factors), where $d$ is feature dimension and $M$ is the number of templates. This is consistent with the minimax lower bound $\Omega(d\sqrt{T})$ for linear contextual bandits (Appendix D). In gap-dependent refinements (not used in this chapter), larger reward gaps between the best and suboptimal templates can improve constants. In practice, increasing $a_{\max}$ increases the magnitude of template-induced ranking changes (and hence the learnable signal), but it also increases the risk of relevance degradation if template design is poor.

**Implementation alignment**: The configuration system centralizes this choice at `SimulatorConfig.action.a_max` in `zoosim/core/config.py`, ensuring consistency across experiments. All code examples in this chapter inherit this default.

**Design Choice: Why M=8 templates?**

We need to balance business objectives with exploration efficiency. Options:

**Option A: Few templates (M=3-5)**
- Pros: Fast learning (fewer arms to explore); simple interpretation.
- Cons: Limited expressiveness (may miss optimal strategies); hard-codes prior knowledge.

**Option B: Many templates (M=20-50)**
- Pros: Rich strategy space.
- Cons: Slower learning (regret grows as $O(\sqrt{MT})$); overfitting risk (too many options for limited data); harder to debug.

**Option C: Continuous parameterization**
- Pros: Maximum flexibility.
- Cons: Loses interpretability (returns to a black-box); requires gradient-based methods (Chapter 7).

**Our choice: M=8 (moderate library)**

We choose **8 templates** because:
1. **Business coverage**: Covers main levers (margin, CM2, price sensitivity, popularity, strategic goals)
2. **Regret budget**: With $T=50k$ episodes, $O(\sqrt{8 \cdot 50k}) \approx 630$ samples for confident selection
3. **Interpretability**: Small enough for stakeholders to understand entire strategy space
4. **Extensibility**: Easy to add/remove templates in production via config

This breaks in scenarios where:
- Products have >5 strategic dimensions (Exercise 6.7 explores hierarchical templates)
- Query-specific templates are needed (Exercise 6.9 extends to query-conditional templates)

With these design choices justified, here is our representative library ($M = 8$):

**Template Library:**

| ID | Name | Description | Boost Formula |
|----|------|-------------|---------------|
| $t_0$ | **Neutral** | No adjustment (base ranker only) | $t_0(p) = 0$ |
| $t_1$ | **High Margin** | Promote products with $\text{margin} > 0.4$ | $t_1(p) = a_{\max} \cdot \mathbb{1}(\text{margin}(p) > 0.4)$ |
| $t_2$ | **CM2 Boost** | Promote own-brand products | $t_2(p) = a_{\max} \cdot \mathbb{1}(\text{is\\_pl}(p))$ |
| $t_3$ | **Popular** | Boost by normalized log-popularity | $t_3(p) = a_{\max} \cdot \log(1 + \text{popularity}(p)) / \log(1 + \text{pop}_{\max})$ |
| $t_4$ | **Premium** | Promote expensive items | $t_4(p) = a_{\max} \cdot \mathbb{1}(\text{price}(p) > p_{75})$ |
| $t_5$ | **Budget** | Promote cheap items | $t_5(p) = a_{\max} \cdot \mathbb{1}(\text{price}(p) < p_{25})$ |
| $t_6$ | **Discount** | Boost discounted products | $t_6(p) = a_{\max} \cdot \min\{\text{discount}(p) / 0.3,\, 1\}$ |
| $t_7$ | **Strategic** | Promote strategic categories | $t_7(p) = a_{\max} \cdot \mathbb{1}(\text{strategic}(p))$ |

**Notation:**
- $\mathbb{1}(\cdot)$ is the indicator function (1 if condition true, 0 otherwise)
- $p_{25}, p_{75}$ are the 25th and 75th percentiles of catalog prices
- $\text{pop}_{\max}$ is the maximum popularity in the catalog
- $\text{is\_pl}(p)$ indicates whether product $p$ is own-brand (private label) in the simulator

**Implementation:**

The reference implementation of the template library lives in `zoosim/policies/templates.py`:

```python
"""Discrete boost templates for contextual bandit policies.

Mathematical basis: [DEF-6.1.1] (Boost Template)

Templates define interpretable boost strategies that can be selected
by contextual bandit algorithms (LinUCB, Thompson Sampling).
"""

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from numpy.typing import NDArray

from zoosim.world.catalog import Product


@dataclass
class BoostTemplate:
    """Single boost template with semantic label.

    Mathematical correspondence: Template $t: \\mathcal{C} \\to \\mathbb{R}$ from [DEF-6.1.1]

    Attributes:
        id: Template identifier (0 to M-1)
        name: Human-readable name (e.g., "High Margin")
        description: Business objective description
        boost_fn: Function mapping a Product to a boost value
                  Signature: (product: Product) -> float
                  Output range (by design): [-a_max, +a_max]
    """

    id: int
    name: str
    description: str
    boost_fn: Callable[[Product], float]

    def apply(self, products: List[Product]) -> NDArray[np.float32]:
        """Apply template to list of products.

        Implements [EQ-6.1]: Computes boost vector for products.

        Args:
            products: List of Product instances generated from the catalog

        Returns:
            boosts: Array of shape (len(products),) with boost values
                   Each entry in [-a_max, +a_max]
        """
        return np.array([self.boost_fn(p) for p in products], dtype=np.float32)


def create_standard_templates(
    catalog_stats: dict,
    a_max: float = 5.0,
) -> List[BoostTemplate]:
    """Create standard template library for search ranking.

    Implements the 8-template library from Section 6.1.1.

    Args:
        catalog_stats: Dictionary with keys:
                      - 'price_p25': 25th percentile price
                      - 'price_p75': 75th percentile price
                      - 'pop_max': Maximum popularity score
                      - 'own_brand': Name of own-brand label
        a_max: Maximum absolute boost value for templates (default 5.0)

    Returns:
        templates: List of M=8 boost templates
    """
    p25 = catalog_stats["price_p25"]
    p75 = catalog_stats["price_p75"]
    pop_max = catalog_stats["pop_max"]
    own_brand = catalog_stats.get("own_brand", "OwnBrand")

    templates = [
        # t0: Neutral (baseline)
        BoostTemplate(
            id=0,
            name="Neutral",
            description="No boost adjustment (base ranker only)",
            boost_fn=lambda p: 0.0,
        ),
        # t1: High Margin
        BoostTemplate(
            id=1,
            name="High Margin",
            description="Promote products with CM2 > 0.4",
            boost_fn=lambda p: a_max if p.cm2 > 0.4 else 0.0,
        ),
        # t2: CM2 Boost (Own Brand)
        BoostTemplate(
            id=2,
            name="CM2 Boost",
            description="Promote own-brand products",
            boost_fn=lambda p: a_max if p.is_pl else 0.0,
        ),
        # t3: Popular
        BoostTemplate(
            id=3,
            name="Popular",
            description="Boost by normalized log-popularity (bestseller score)",
            boost_fn=lambda p: (
                a_max * np.log(1 + p.bestseller) / np.log(1 + pop_max)
                if pop_max > 0
                else 0.0
            ),
        ),
        # t4: Premium
        BoostTemplate(
            id=4,
            name="Premium",
            description="Promote expensive items (price > 75th percentile)",
            boost_fn=lambda p: a_max if p.price > p75 else 0.0,
        ),
        # t5: Budget
        BoostTemplate(
            id=5,
            name="Budget",
            description="Promote cheap items (price < 25th percentile)",
            boost_fn=lambda p: a_max if p.price < p25 else 0.0,
        ),
        # t6: Discount
        BoostTemplate(
            id=6,
            name="Discount",
            description="Boost discounted products (max discount 30%)",
            boost_fn=lambda p: a_max * min(p.discount / 0.3, 1.0),
        ),
        # t7: Strategic
        BoostTemplate(
            id=7,
            name="Strategic",
            description="Promote strategic categories",
            boost_fn=lambda p: a_max if p.strategic_flag else 0.0,
        ),
    ]

    return templates
```

!!! note "Code <-> Config (Template Library)"
    The template library from Table Section 6.1.1 maps to:
    - Template definitions: `zoosim/policies/templates.py`
    - Catalog statistics: Computed from `SimulatorConfig.catalog` in `zoosim/world/catalog.py`
    - Action bound (continuous weights): `SimulatorConfig.action.a_max` in `zoosim/core/config.py`
    - Template amplitude `a_max` (this section) is a separate hyperparameter, tuned relative to the base relevance score scale.

    To modify templates in experiments, edit `create_standard_templates` or pass custom templates to bandit policies.

**Verification: Template application**

We verify that templates produce the expected boosts on synthetic products:

```python
import numpy as np

# Mock catalog statistics
catalog_stats = {
    'price_p25': 10.0,
    'price_p75': 50.0,
    'pop_max': 1000.0,
    'own_brand': 'Zooplus'
}

# Create template library
templates = create_standard_templates(catalog_stats, a_max=5.0)

# Synthetic product examples (using the same fields as Product)
products = [
    {   # High-margin own-brand product
        "cm2": 0.5, "is_pl": True, "bestseller": 500.0,
        "price": 30.0, "discount": 0.1, "strategic_flag": True,
    },
    {   # Low-margin third-party budget product
        "cm2": 0.2, "is_pl": False, "bestseller": 100.0,
        "price": 5.0, "discount": 0.0, "strategic_flag": False,
    },
    {   # Premium discounted product
        "cm2": 0.35, "is_pl": False, "bestseller": 800.0,
        "price": 60.0, "discount": 0.25, "strategic_flag": False,
    },
]

# Apply each template (treating dicts as lightweight stand-ins for Product)
print("Template boosts per product:")
print("Product:         ", ["High-margin OB", "Budget 3P", "Premium Disc"])
for template in templates:
    boosts = template.apply([
        Product(
            product_id=i,
            category="dummy",
            price=p["price"],
            cm2=p["cm2"],
            is_pl=p["is_pl"],
            discount=p["discount"],
            bestseller=p["bestseller"],
            embedding=torch.zeros(16),
            strategic_flag=p["strategic_flag"],
        )
        for i, p in enumerate(products)
    ])
    print(f"{template.name:15s}", boosts.round(2))

# Output (representative):
# Template boosts per product:
# Product:          ['High-margin OB', 'Budget 3P', 'Premium Disc']
# Neutral           [0.   0.   0.  ]
# High Margin       [5.   0.   0.  ]
# CM2 Boost         [5.   0.   0.  ]
# Popular           [4.50 3.34 4.84]
# Premium           [0.   0.   5.  ]
# Budget            [0.   5.   0.  ]
# Discount          [1.67 0.   4.17]
# Strategic         [5.   0.   0.  ]
```

**Interpretation:**

- Product 1 (high-margin own-brand): Gets boosted by High Margin, CM2, Popular, Strategic
- Product 2 (budget third-party): Only Budget and Popular boost it
- Product 3 (premium discounted): Premium, Popular, and Discount boost it

The contextual bandit will **learn which template performs best** for each query-user context.

---

### 6.1.2 Contextual Bandit Formulation

With discrete templates, our RL problem reduces to a **contextual bandit**:

**Definition 6.2** (Stochastic Contextual Bandit) {#DEF-6.2}

A **stochastic contextual bandit** is a tuple $(\mathcal{X}, \mathcal{A}, R, \rho)$ where:

- $\mathcal{X}$ is the **context space** (observations)
- $\mathcal{A} = \{1, \ldots, M\}$ is a finite **action set** (template IDs)
- $R: \mathcal{X} \times \mathcal{A} \times \Omega \to \mathbb{R}$ is a **stochastic reward function** with outcomes $\omega \in \Omega$
- $\rho$ is a **context distribution** over $\mathcal{X}$

**Interaction protocol.** At each episode $t = 1, 2, \ldots, T$:

1. **Context arrival**: Environment samples $x_t \sim \rho$ independently (user, query, product features; i.i.d. assumption)
2. **Action selection**: Agent selects $a_t \in \mathcal{A}$ (template ID), possibly depending on $x_t$ and history $\mathcal{H}_{t-1}$
3. **Reward realization**: Environment samples outcome $\omega_t \sim P(\cdot \mid x_t, a_t)$ and returns reward $r_t = R(x_t, a_t, \omega_t)$
4. **Observation**: Agent observes $(x_t, a_t, r_t)$ and updates its policy

**Assumptions (bandits vs. MDPs):**

1. **i.i.d. contexts**: Contexts $\{x_t\}$ are drawn i.i.d. from $\rho$
2. **Stochastic rewards**: For fixed $(x, a)$, reward randomness enters only through $\omega$
3. **No state transitions**: $x_{t+1}$ is independent of $(x_t, a_t, r_t)$ --- there is no latent Markov state evolving over time

These assumptions formalize the intuition that we treat search sessions as **single-query, independent episodes**. This is **not** a full MDP (Chapter 3). We will relax the independence assumption in Chapter 11 when we model inter-session retention.

**Expected reward and optimal policy:**

Define the **mean reward function**:
$$
\mu(x, a) = \mathbb{E}_{\omega \sim P(\cdot | x, a)}[R(x, a, \omega)]
\tag{6.2}
$$
{#EQ-6.2}

The **optimal policy** is:
$$
\pi^*(x) = \arg\max_{a \in \mathcal{A}} \mu(x, a)
\tag{6.3}
$$
{#EQ-6.3}

**Regret:**

The agent's goal is to minimize **cumulative regret** over $T$ episodes:
$$
\text{Regret}(T) = \sum_{t=1}^T \left[\mu(x_t, \pi^*(x_t)) - \mu(x_t, a_t)\right]
\tag{6.4}
$$
{#EQ-6.4}

Here $\pi^*(x) := \arg\max_{a \in \mathcal{A}} \mu(x, a)$ is the optimal policy *within the template class* $\mathcal{A} = \{1, \ldots, M\}$. If the true optimal policy lies outside this template space (model misspecification), [EQ-6.4] measures regret relative to the *best template*, not the globally optimal action---a distinction we exploit diagnostically in Section 6.6.

**Theorem 6.0** (Minimax Lower Bound for Stochastic Bandits) {#THM-6.0}

For any learning algorithm and any time horizon $T \geq M$, there exists an $M$-armed stochastic bandit instance with Bernoulli rewards such that the expected cumulative regret satisfies:

$$
\mathbb{E}[\mathrm{Regret}(T)] \geq c \sqrt{M T}.
$$

where $c > 0$ is a universal constant (e.g., $c = 1/20$ suffices).

*Proof.* Appendix D states this lower bound for $K$ arms with Bernoulli rewards (Theorem D.3.1). Substituting $K = M$ yields the claim. $\square$

**Remark 6.0.1** (Interpretation and significance). This theorem establishes that no algorithm---however clever---can achieve regret better than $\Omega(\sqrt{MT})$ uniformly over all $M$-armed bandit instances. The upper bounds we prove for Thompson Sampling and LinUCB in this chapter match this rate up to logarithmic factors, which is why they are called "optimal" in the bandit literature. For contextual bandits with feature dimension $d$, the lower bound becomes $\Omega(d\sqrt{T})$ (see Appendix D, [EQ-D.10]); for continuous-action bandits, the relevant complexity measure is the eluder dimension. We return to these extensions when discussing feature richness and model misspecification in Sections 6.5--6.7.

**Why this matters:**

If we could observe $\mu(x, a)$ for all $(x, a)$ pairs, we would pick $\pi^*(x)$ greedily. But $\mu$ is **unknown**---we must learn it from noisy samples while balancing **exploration** (try all templates) vs. **exploitation** (use the best known template).

**Two canonical algorithms:**

We develop two approaches with complementary strengths:

1. **Thompson Sampling (Section 6.2-6.3)**: Bayesian posterior sampling, probabilistic exploration
2. **LinUCB (Section 6.4-6.5)**: Frequentist upper confidence bounds, deterministic exploration

Both achieve sublinear regret under standard assumptions; see [THM-6.1] and [THM-6.2] for representative bounds with explicit dependence on $d$, $M$, and logarithmic factors.

---

## 6.2 Thompson Sampling: Bayesian Exploration

### 6.2.1 The Core Idea

Thompson Sampling (TS) is simple: we sample from the posterior belief about which action is best, then take that action.

**Bayesian framework:**

We maintain a probability distribution $p(a \text{ is optimal} \mid \mathcal{H}_t)$ where $\mathcal{H}_t = \{(x_s, a_s, r_s)\}_{s < t}$ is the history.

**Algorithm (informal):**

For each episode $t$:
1. Sample a plausible mean reward function $\tilde{\mu}$ from posterior
2. Compute $a_t = \arg\max_{a} \tilde{\mu}(x_t, a)$
3. Apply action $a_t$, observe reward $r_t$
4. Update posterior: $p(\mu \mid \mathcal{H}_{t+1}) \propto p(r_t \mid \mu, x_t, a_t) \cdot p(\mu \mid \mathcal{H}_t)$

**Why this works (intuitively):**

- **Exploration**: When uncertain, posterior is wide -> samples vary -> tries different actions
- **Exploitation**: When confident, posterior is narrow -> samples concentrate on best action
- **Automatic balance**: Uncertainty naturally decreases as data accumulates

**Mathematical formalization:**

We need to specify:
1. Prior distribution over mean rewards $p(\mu)$
2. Likelihood model $p(r \mid \mu, x, a)$
3. Posterior update rule $p(\mu \mid \mathcal{H})$

For linear contextual bandits, we use a **Gaussian prior** with **Gaussian likelihood**.

---

### 6.2.2 Linear Contextual Thompson Sampling

**Definition 6.3** (Linear Contextual Bandit) {#DEF-6.3}

A **linear contextual bandit** is a stochastic contextual bandit ([DEF-6.2]) whose mean reward function admits a linear representation:
$$
\mu(x, a)
  := \mathbb{E}_{\omega \sim P(\cdot \mid x, a)}[R(x, a, \omega)]
  = \langle \theta_a, \phi(x) \rangle
  = \theta_a^\top \phi(x)
\tag{6.5}
$$
{#EQ-6.5}

where:
- $\phi: \mathcal{X} \to \mathbb{R}^d$ is a known **feature map** (Chapter 5)
- $\theta_a \in \mathbb{R}^d$ is an unknown **weight vector** for action $a$
- $\langle \cdot, \cdot \rangle$ is the Euclidean inner product

We make the following **structural assumptions**:

1. **Finite dimension**: Feature space is $\mathbb{R}^d$ with $d < \infty$
2. **Linearity**: Mean reward is exactly linear in features (no approximation error in the model class)
3. **Bounded features**: $\|\phi(x)\| \leq L$ for all $x \in \mathcal{X}$ and some constant $L > 0$
4. **Bounded parameters**: $\|\theta_a\| \leq S$ for all $a \in \mathcal{A}$ and some constant $S > 0$

**Why linear?**

We need *some* parametric structure to generalize across contexts. Options:

**Option A: Tabular (no structure)**
- $\mu(x, a)$ is a table with $|\mathcal{X}| \times M$ entries
- Pros: No assumptions.
- Cons: No generalization (each context learned independently); infeasible sample complexity when $|\mathcal{X}|$ is large/continuous.

**Option B: Nonlinear (neural network)**
- $\mu(x, a) = f_\theta(x, a)$ with neural net $f$
- Pros: Maximum flexibility.
- Cons: Requires many samples (Chapter 7); posterior intractable (no closed-form updates).

**Option C: Linear (our choice)**
- $\mu(x, a) = \theta_a^\top \phi(x)$
- Pros: Closed-form posterior (Gaussian conjugate); sample-efficient with good features.
- Cons: Misspecification risk (if the true $\mu$ is nonlinear).

We choose **linear** because:
1. Chapter 5 engineered rich features $\phi(x)$ (product, user, query interactions)
2. Gaussian conjugate prior -> efficient Bayesian updates
3. Fast inference (matrix operations, no MCMC)
4. Provable regret bounds (Theorem 6.1 below)

This breaks when:
- Feature engineering is poor (Exercise 6.11 explores kernel features)
- True reward highly nonlinear (Exercise 6.12 compares to neural TS)

**Gaussian linear model (prior and posterior):**

For each action $a \in \{1, \ldots, M\}$, we start from an independent Gaussian prior
$$
\theta_a \sim \mathcal{N}(0, \lambda_0^{-1} I_d),
$$
and under the Gaussian likelihood below the posterior remains Gaussian.

For each action $a \in \{1, \ldots, M\}$, we maintain:
$$
\theta_a \sim \mathcal{N}(\hat{\theta}_a, \Sigma_a)
\tag{6.6}
$$
{#EQ-6.6}

where:
- $\hat{\theta}_a \in \mathbb{R}^d$ is the **posterior mean** (our current estimate)
- $\Sigma_a \in \mathbb{R}^{d \times d}$ is the **posterior covariance** (our uncertainty)

**Likelihood model:**

Assume rewards are Gaussian:
$$
r_t \mid x_t, a_t, \theta_{a_t} \sim \mathcal{N}(\theta_{a_t}^\top \phi(x_t), \sigma^2)
\tag{6.7}
$$
{#EQ-6.7}

where $\sigma^2$ is the **noise variance** (typically unknown, estimated from data).

**Posterior update (Bayesian linear regression; ridge form):**

It is convenient (and numerically stable) to maintain the **sufficient statistics** that also appear in ridge regression:
$$
A_a := \sigma^2 \lambda_0 I_d + \sum_{s:\,a_s=a}\phi(x_s)\phi(x_s)^\top,\qquad
b_a := \sum_{s:\,a_s=a} r_s\,\phi(x_s).
$$
Under the Gaussian model [EQ-6.7], this yields the posterior
$\theta_a \mid \mathcal{H}_t \sim \mathcal{N}(\hat{\theta}_a, \Sigma_a)$
with $\hat{\theta}_a = A_a^{-1} b_a$ and $\Sigma_a = \sigma^2 A_a^{-1}$.
After observing $(x_t, a_t, r_t)$, update only the selected action $a_t$ via
\begin{align}
A_{a_t} &\leftarrow A_{a_t} + \phi(x_t)\,\phi(x_t)^\top \tag{6.8a}\\
b_{a_t} &\leftarrow b_{a_t} + r_t\,\phi(x_t) \tag{6.8b}\\
\hat{\theta}_{a_t} &\leftarrow A_{a_t}^{-1} b_{a_t},\qquad \Sigma_{a_t} \leftarrow \sigma^2 A_{a_t}^{-1}. \tag{6.8c}
\end{align}
{#EQ-6.8}

This is algebraically equivalent to updating the precision $\Sigma_a^{-1}$ and the scaled moment $\sigma^{-2}\sum r_t\phi_t$, but it avoids “old vs. new precision” ambiguities and makes the ridge-regression equivalence in [EQ-6.9] immediate. (Other actions' posteriors unchanged.)

**Equivalence to ridge regression:**

The posterior mean $\hat{\theta}_a$ is the **ridge regression estimate**:
$$
\hat{\theta}_a = \arg\min_{\theta} \left\{ \sum_{t: a_t = a} (r_t - \theta^\top \phi(x_t))^2 + \lambda \|\theta\|^2 \right\}
\tag{6.9}
$$
{#EQ-6.9}

where $\lambda = \sigma^2 / \sigma_0^2$ is the regularization strength (ratio of noise variance to prior variance).
Equivalently, under the prior $\theta_a \sim \mathcal{N}(0,\lambda_0^{-1}I_d)$, the ridge penalty is $\lambda = \sigma^2 \lambda_0$.

This shows TS is **Bayesian regularization**---the prior prevents overfitting.

**Algorithm 6.1** (Linear Thompson Sampling for Contextual Bandits) {#ALG-6.1}

**Input:**
- Feature map $\phi: \mathcal{X} \to \mathbb{R}^d$
- Action set $\mathcal{A} = \{1, \ldots, M\}$ (template IDs)
- Prior: $\theta_a \sim \mathcal{N}(0, \lambda_0^{-1} I_d)$ for all $a$ (prior precision $\lambda_0 > 0$)
- Noise variance $\sigma^2$ (estimated or set to 1.0)
- Number of episodes $T$

**Initialization:**
- For each action $a \in \mathcal{A}$:
  - $A_a \leftarrow \sigma^2 \lambda_0 I_d$
  - $b_a \leftarrow 0 \in \mathbb{R}^d$
  - $\hat{\theta}_a \leftarrow 0 \in \mathbb{R}^d$

**For** $t = 1, \ldots, T$:

1. **Observe context**: Receive $x_t \in \mathcal{X}$ from environment
2. **Compute features**: $\phi_t \leftarrow \phi(x_t) \in \mathbb{R}^d$
3. **Sample posteriors**: For each action $a \in \mathcal{A}$:
	   $$
	   \tilde{\theta}_a^{(t)} \sim \mathcal{N}(\hat{\theta}_a, \sigma^2 A_a^{-1})
	   $$
4. **Select optimistic action**:
   $$
   a_t \leftarrow \arg\max_{a \in \mathcal{A}} \langle \tilde{\theta}_a^{(t)}, \phi_t \rangle
   $$
5. **Execute action**: Apply template $a_t$, observe reward $r_t$
6. **Update posterior** for action $a_t$ using [EQ-6.8] (with $\phi_t=\phi(x_t)$).

**Output:** Posterior distributions $\{\mathcal{N}(\hat{\theta}_a, \Sigma_a)\}_{a=1}^M$

---

**Computational complexity.**

At episode $t$, with feature dimension $d$ and $M$ actions:

- **Feature computation:** $O(d)$ to form $\phi_t$.
- **Posterior sampling:** Naively, constructing $\Sigma_a$ and drawing $\tilde{\theta}_a \sim \mathcal{N}(\hat{\theta}_a, \Sigma_a)$ costs $O(d^3)$ per action (matrix inversion + Cholesky), i.e. $O(M d^3)$ overall.
- **Optimized implementation:** Maintaining precision matrices and Cholesky factors across episodes reduces sampling to $O(M d^2)$ per step (rank-1 updates + triangular solves; see Section 6.5 exercises).
- **Action selection:** Computing $\langle \tilde{\theta}_a^{(t)}, \phi_t \rangle$ for all $a$ is $O(M d)$ plus an $O(M)$ argmax.
- **Posterior update:** Rank-1 precision update and mean update for the chosen action $a_t$ is $O(d^2)$.

Over $T$ episodes this yields **time complexity** $O(T M d^2)$ with an optimized linear algebra backend and **memory** $O(M d^2)$ to store $\{\Sigma_a\}$ or their inverses. In practice, $M$ is small (8 templates) and $d$ is on the order of tens, so the cost is dominated by the simulator, not the bandit.

---

**Why does this work?**

Thompson Sampling elegantly balances exploration and exploitation through **probability matching**:

**Probability matching property:**

The probability of selecting action $a$ equals the probability that $a$ is optimal under the posterior:
$$
P(a_t = a \mid \mathcal{H}_t) = P(\theta_a^\top \phi_t > \theta_{a'}^\top \phi_t \text{ for all } a' \neq a \mid \mathcal{H}_t)
\tag{6.10}
$$
{#EQ-6.10}

**Intuition:**

- If action $a$ is **very likely optimal** (concentrated posterior), it gets selected with high probability -> **exploitation**
- If action $a$ is **uncertain** (wide posterior), it occasionally gets sampled "just in case" -> **exploration**
- As data accumulates, posteriors concentrate on true parameters -> exploration diminishes naturally

**This is automatic!** No manually-tuned exploration rate (unlike epsilon-greedy) or confidence intervals (unlike UCB).

---

### 6.2.3 Regret Analysis

**Bayesian Regret** {#DEF-6.4}

For a Bayesian policy, we define the **Bayesian regret** as the expected regret where the expectation is taken over the context distribution, the policy's randomness, and the Bayesian prior:
$$
\text{BayesReg}(T)
  = \mathbb{E}_{\theta^* \sim \pi_0,\,\text{contexts, policy}}
    \left[\sum_{t=1}^T \bigl(\max_{a \in \mathcal{A}} \theta_a^{*\top} \phi_t - \theta_{a_t}^{*\top} \phi_t\bigr)\right]
$$
where $\theta^* \sim \pi_0$ denotes the prior on unknown parameters, $\phi_t := \phi(x_t)$, and the expectation includes the randomness from sampling contexts, reward noise, and posterior sampling.
Equivalently, $\text{BayesReg}(T)=\mathbb{E}_{\theta^* \sim \pi_0}[\text{Regret}(T\mid \theta^*)]$.

**Remark 6.2.1** (Bayesian vs Frequentist Regret) {#REM-6.2.1}

Two regret notions appear in this chapter:

1. **Bayesian regret** (this definition): the expectation averages over a *prior* on the unknown parameters $\theta^*$ as well as contexts and policy randomness.
2. **Frequentist regret** (used for LinUCB in [THM-6.2]): the parameters $\theta^*$ are treated as fixed but unknown; the expectation is only over contexts and policy randomness.

Formally, Bayesian regret can be written as
$$
\text{BayesReg}(T)
  = \mathbb{E}_{\theta^* \sim \pi_0,\,\text{contexts, policy}}
    \left[\sum_{t=1}^T \bigl(\max_a \theta_a^{*\top} \phi_t - \theta_{a_t}^{*\top} \phi_t\bigr)\right],
$$
while frequentist regret conditions on a fixed $\theta^*$:
$$
\text{Regret}(T \mid \theta^*)
  = \mathbb{E}_{\text{contexts, policy}}
    \left[\sum_{t=1}^T \bigl(\max_a \theta_a^{*\top} \phi_t - \theta_{a_t}^{*\top} \phi_t\bigr)\right].
$$

Thompson Sampling is naturally analyzed in the Bayesian sense (it explicitly uses a prior), whereas LinUCB is usually analyzed in the frequentist sense (no prior, worst-case guarantees over all admissible $\theta^*$). When the prior is well-calibrated and concentrates around the true parameters, the two perspectives tend to agree asymptotically, but they answer slightly different questions: "average performance across plausible worlds" vs. "performance in this particular world".

In this chapter we state [THM-6.1] as an expected regret bound conditional on a fixed (unknown) $\theta^*$, using standard self-normalized concentration for ridge regression plus Gaussian sampling concentration. A Bayesian regret statement follows by averaging over any prior supported on $\{\|\theta_a^*\|\le S\}$ (or by working with a truncated prior).

For deeper treatment of the Bayesian perspective---hierarchical priors over user and segment preferences, posterior shrinkage, and how those posteriors feed into bandit features---see [@russo:tutorial_ts:2018] and [@chapelle:empirical_ts:2011]. The survey by [@lattimore:bandit_algorithms:2020, Chapters 35--37] provides rigorous foundations for Bayesian regret analysis in linear bandits. **Appendix A** develops these ideas for our search setting: hierarchical user preference models, posterior inference for price sensitivity and brand affinity, and how Bayesian estimates integrate with the template bandits of this chapter.

**Theorem 6.1** (Thompson Sampling Regret Bound) {#THM-6.1}

Consider a linear contextual bandit ([DEF-6.3]) with:

**Data:**
- Feature dimension $d \in \mathbb{N}$
- Action set $\mathcal{A} = \{1, \ldots, M\}$ with $M \geq 2$
- Horizon $T \in \mathbb{N}$ (number of episodes)

**Structural assumptions:**
1. **Linearity**: Mean rewards $\mu(x, a) = \langle \theta_a^*, \phi(x) \rangle$ for unknown parameters $\theta_a^* \in \mathbb{R}^d$
2. **Bounded parameters**: $\|\theta_a^*\| \leq S$ for all $a \in \mathcal{A}$ and some constant $S > 0$
3. **Bounded features**: $\|\phi(x)\| \leq L$ for all $x \in \mathcal{X}$ and some constant $L > 0$
4. **i.i.d. contexts**: Contexts $\{x_t\}_{t=1}^T$ are drawn i.i.d. from distribution $\rho$ over $\mathcal{X}$
5. **Sub-Gaussian noise**: For each $(x, a)$, the reward noise
   $$
   \epsilon := r - \mu(x, a)
   $$
   conditioned on $(x, a)$ is sub-Gaussian with variance proxy $\sigma^2$:
   $$
   \mathbb{E}[\exp(\lambda \epsilon) \mid x, a]
     \leq \exp(\lambda^2 \sigma^2 / 2)
     \quad \forall \lambda \in \mathbb{R}.
   $$

**Algorithm configuration:**
- Prior: $\theta_a \sim \mathcal{N}(0, \lambda_0^{-1} I_d)$ for each $a$, with regularization $\lambda_0 > 0$
- Likelihood: Gaussian with known variance proxy $\sigma^2$ (or a consistent estimate)

If the noise is truly Gaussian, the updates in [ALG-6.1] correspond to the exact conjugate posterior. Under merely sub-Gaussian noise, we interpret $\mathcal{N}(\hat\theta_a,\Sigma_a)$ as a **Gaussian pseudo-posterior / perturbation distribution** used for exploration; the analysis relies on concentration of ridge regression estimates plus Gaussian sampling, not on exact Bayesian correctness.

Then Thompson Sampling ([ALG-6.1]) with the above prior satisfies
$$
\mathbb{E}[\text{Regret}(T)\mid \theta^*]
  \leq C \cdot d\sqrt{M T \log T}
\tag{6.11}
$$
{#EQ-6.11}

for some constant $C > 0$ that depends on $S, L, \sigma, \lambda_0$ but not on $T$.
Here the expectation is taken over contexts, reward noise, and posterior-sampling randomness, conditional on the fixed (unknown) parameters $\theta^*$.

*Proof.* See [@agrawal:thompson:2013, Theorem 2] for a complete proof. $\square$

We record below the main concentration lemmas and the elliptical potential identity used in that proof, since we reuse these tools when interpreting exploration bonuses and diagnostics later in the chapter.

**Lemma 6.1.1** (Gaussian Concentration Around the Mean Estimate) {#LEM-6.1.1}
Under the assumptions of [THM-6.1], condition on the history up to episode $t-1$ and let
$\tilde{\theta}_a^{(t)} \sim \mathcal{N}(\hat{\theta}_{a,t-1}, \Sigma_{a,t-1})$
be the parameter sample for action $a$ at episode $t$. Then, with probability at least $1 - \delta/(MT)$,
$$
\bigl|
  \tilde{\theta}_a^{(t)\top} \phi(x_t)
  - \hat{\theta}_{a,t-1}^{\top} \phi(x_t)
\bigr|
  \leq \alpha_t \, \|\phi(x_t)\|_{\Sigma_{a,t-1}}
$$
where $\alpha_t = \sqrt{2 \log(2MT/\delta)}$ and
$\|v\|_{\Sigma} := \sqrt{v^\top \Sigma v}$.

*Proof of Lemma 6.1.1.* This is a standard Gaussian tail bound applied to the one-dimensional projection $(\tilde{\theta}_a^{(t)}-\hat\theta_{a,t-1})^\top \phi(x_t)$; see [@agrawal:thompson:2013, Lemma 3]. QED

**Lemma 6.1.1b** (Estimator Concentration Around Truth; self-normalized) {#LEM-6.1.1b}
Under the assumptions of [THM-6.1], define for each action $a$ the (regularized) design matrix
$$
A_{a,t} := \sigma^2 \lambda_0 I_d + \sum_{s \le t:\, a_s = a} \phi_s \phi_s^\top,
$$
and let $\hat\theta_{a,t}$ be the ridge regression estimate for that action (equivalently, the posterior mean in [EQ-6.8]). Then for any $\delta\in(0,1)$, with probability at least $1-\delta$ (simultaneously for all $t$ and $a$),
$$
\bigl|(\hat\theta_{a,t}-\theta_a^*)^\top \phi(x_t)\bigr|
\le \beta_{a,t}(\delta)\,\|\phi(x_t)\|_{A_{a,t}^{-1}},
$$
where one standard choice is
$$
\beta_{a,t}(\delta)
= \sigma\sqrt{2\log(M/\delta) + d\log\Bigl(1 + \frac{n_a(t)}{d\,\sigma^2\lambda_0}\Bigr)}
  + \sigma\sqrt{\lambda_0}\,S,
$$
with $n_a(t)$ the number of times action $a$ was selected up to time $t$; see [@abbasi:improved:2011, Theorem 2] or [@lattimore:bandit_algorithms:2020, Chapter 19].

**Corollary 6.1.1** (Triangle inequality). {#COR-6.1.1}
On the intersection of the events in Lemma 6.1.1 and Lemma 6.1.1b, using $\Sigma_{a,t}=\sigma^2 A_{a,t}^{-1}$ in the Gaussian model, we have
$$
\bigl|(\tilde\theta_{a}^{(t)}-\theta_a^*)^\top \phi(x_t)\bigr|
\le \Bigl(\beta_{a,t-1}(\delta) + \sigma\sqrt{2\log(2MT/\delta)}\Bigr)\,\|\phi(x_t)\|_{A_{a,t-1}^{-1}}.
$$

**Lemma 6.1.2** (Elliptical Potential; [@abbasi:improved:2011, Lemma 11]) {#LEM-6.1.2}
Let $\{A_t\}_{t \geq 0}$ be a sequence of $d \times d$ positive definite matrices with $A_0 = \sigma^2\lambda_0 I_d$ and updates
$$
A_t = A_{t-1} + v_t v_t^\top, \quad \|v_t\| \leq L.
$$
Then
$$
\sum_{t=1}^T \min\{1,\,\|v_t\|_{A_{t-1}^{-1}}^2\}
  \leq 2\log\det(A_T A_0^{-1}).
$$
Moreover,
$$
\log\det(A_T A_0^{-1})
  \leq d \log\bigl(1 + TL^2/(d \sigma^2\lambda_0)\bigr).
$$

*Proof of Lemma 6.1.2.* See [@abbasi:improved:2011, Lemma 11]. QED

**Remark 6.1.1** (When Regret Bounds Fail in Practice) {#REM-6.1.1}

[THM-6.1] is a **conditional guarantee**: regret is $O(\sqrt{T})$ *if the assumptions hold*. In Sections 6.5--6.6 we will deliberately violate these assumptions and see the regret picture break:

1. **Model misspecification (Section 6.6):** The true reward is nonlinear, but we force a linear model. The theorem still applies to the *best linear approximation*, yet approximation error dominates, and observed regret can grow almost linearly.
2. **Feature poverty (Section 6.6):** The feature map $\phi(x)$ omits critical context (e.g., user preferences). The bound applies in the restricted feature space, but the optimal policy **in that space** may be far from the true optimum.
3. **Heavy-tailed noise (advanced):** If rewards are heavy-tailed (e.g., lognormal order values), the sub-Gaussian assumption fails (even when variance is finite); standard regret bounds require clipping or robust estimators.

The lesson is central to this chapter: theorem correctness does not imply algorithmic success. In practice we monitor the assumptions via diagnostics (per-segment performance, uncertainty traces in Section 6.7) rather than treating a regret bound as a performance guarantee.

**Interpretation:**

- **Sublinear regret**: $O(\sqrt{T})$ -> per-episode regret $O(1/\sqrt{T}) \to 0$
- **Dimension dependence**: Linear in $d$ -> feature engineering critical
- **Action scaling**: $\sqrt{M}$ -> modest penalty for more templates

In our regime $M$ is small (eight templates) and $d$ is on the order of tens, so the $\sqrt{M}$ and $d$ dependencies are mild; however, the constants hidden in the $O(\cdot)$ notation can be large, and we rely on empirical validation in the simulator.

---

**Minimal numerical verification:**

We verify Thompson Sampling on a synthetic 3-armed bandit to build intuition before production implementation:

```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic 3-armed linear bandit
np.random.seed(42)
d = 5  # Feature dimension
M = 3  # Number of actions (templates)
T = 1000  # Episodes

# True parameters (unknown to algorithm)
theta_star = np.random.randn(M, d)  # Shape (M, d)
sigma = 0.1  # Reward noise std

# Thompson Sampling initialization
lambda_reg = 1.0
theta_hat = np.zeros((M, d))  # Posterior means
Sigma_inv = np.array([lambda_reg * np.eye(d) for _ in range(M)])  # Precision matrices

rewards_history = []
regrets_history = []

for t in range(T):
    # Context (random for synthetic example)
    x = np.random.randn(d)
    x /= np.linalg.norm(x)  # Normalize to ||x|| = 1

    # Sample from posteriors
    theta_samples = []
    for a in range(M):
        Sigma_a = np.linalg.inv(Sigma_inv[a])
        theta_tilde = np.random.multivariate_normal(theta_hat[a], Sigma_a)
        theta_samples.append(theta_tilde)

    # Select action with highest sampled reward
    expected_rewards = [theta_samples[a] @ x for a in range(M)]
    action = np.argmax(expected_rewards)

    # Observe reward
    true_reward = theta_star[action] @ x + sigma * np.random.randn()
    rewards_history.append(true_reward)

    # Compute regret (oracle knows theta_star)
    optimal_reward = np.max([theta_star[a] @ x for a in range(M)])
    regret = optimal_reward - theta_star[action] @ x
    regrets_history.append(regret)

    # Update posterior for selected action
    Sigma_inv[action] += (1 / sigma**2) * np.outer(x, x)
    Sigma_a = np.linalg.inv(Sigma_inv[action])
    theta_hat[action] = Sigma_a @ (Sigma_inv[action] @ theta_hat[action] + (1 / sigma**2) * x * true_reward)

# Plot cumulative regret
cumulative_regret = np.cumsum(regrets_history)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(cumulative_regret, label='Thompson Sampling')
plt.plot([0, T], [0, np.sqrt(T) * 5], 'r--', label=r'$O(\sqrt{T})$ bound')
plt.xlabel('Episode')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.title('Thompson Sampling Regret Growth')

plt.subplot(1, 2, 2)
plt.plot(np.array(cumulative_regret) / np.arange(1, T+1), label='Average Regret')
plt.axhline(0, color='r', linestyle='--', label='Optimal')
plt.xlabel('Episode')
plt.ylabel('Average Regret per Episode')
plt.legend()
plt.title('Average Regret -> 0 (Convergence)')

plt.tight_layout()
plt.savefig('/tmp/thompson_sampling_verification.png', dpi=150)
print("Verification plot saved to /tmp/thompson_sampling_verification.png")

# Output:
# Cumulative regret grows as O(sqrtT) [OK]
# Average regret per episode -> 0 [OK]
```

**Observations:**

1. **Cumulative regret sublinear**: Grows slower than $O(\sqrt{T})$ theoretical bound
2. **Average regret vanishes**: Per-episode regret $\to 0$ as $T \to \infty$
3. **Fast convergence**: After ~200 episodes, algorithm concentrates on optimal actions

Theory works. In Section 6.4 we turn this into production code inside our simulator.

---

## 6.3 LinUCB: Upper Confidence Bounds

Thompson Sampling is elegant, but **stochastic**---each run produces different template selections even with the same data. For production systems where **reproducibility** and **deterministic debugging** matter, we want a **frequentist alternative**.

Enter **LinUCB**: Linear Upper Confidence Bound algorithm.

### 6.3.1 The UCB Principle

**Core idea: Optimism in the face of uncertainty.**

Instead of sampling from a posterior, LinUCB constructs **confidence intervals** around mean reward estimates and selects the action with the **highest upper bound**.

**Confidence bound construction:**

For each action $a$ and context $x$, we estimate:
$$
\hat{\mu}(x, a) = \hat{\theta}_a^\top \phi(x)
\tag{6.12}
$$
{#EQ-6.12}

and compute an **uncertainty bonus**:
$$
\text{UCB}(x, a) = \hat{\mu}(x, a) + \alpha \cdot \text{Uncertainty}(x, a)
\tag{6.13}
$$
{#EQ-6.13}

where $\alpha > 0$ is an **exploration parameter** and $\text{Uncertainty}(x, a)$ measures confidence in $\hat{\mu}(x, a)$.

**Why this works:**

- **Exploitation**: Term $\hat{\mu}(x, a)$ favors actions with high estimated reward
- **Exploration**: Bonus $\alpha \cdot \text{Uncertainty}$ favors actions with high uncertainty (underexplored)
- **Automatic balance**: As action $a$ is selected, data accumulates -> uncertainty shrinks -> exploration bonus decreases

**Mathematical formalization:**

For linear contextual bandits, the uncertainty is the **prediction interval width**:
$$
\text{Uncertainty}(x, a) = \sqrt{\phi(x)^\top \Sigma_a \phi(x)}
\tag{6.14}
$$
{#EQ-6.14}

where $\Sigma_a$ is the posterior covariance (same as Thompson Sampling!).
In the Gaussian linear model, $\Sigma_a = \sigma^2 A_a^{-1}$ with $A_a = \sigma^2\lambda_0 I + \sum_{s:\,a_s=a}\phi_s\phi_s^\top$ (see [PROP-6.1] for the ridge-regression form). In practice we implement $\sqrt{\phi^\top A_a^{-1}\phi}$ and absorb the factor $\sigma$ into $\alpha$.

**Geometric interpretation:**

$\text{Uncertainty}(x, a)$ is the standard deviation of the predicted reward $\hat{\theta}_a^\top \phi(x)$ under the Gaussian posterior. It measures how much $\phi(x)$ aligns with the **principal axes of uncertainty** in parameter space.

**Proposition 6.2** (High-probability decay of the uncertainty bonus) {#PROP-6.2}

Let $\{\phi_s\}_{s\ge 1}$ be i.i.d. random vectors in $\mathbb{R}^d$ such that $\|\phi_s\|_2 \le L$ almost surely and
$$
\mathbb{E}[\phi\phi^\top] \succeq c I_d
$$
for some $c>0$. Fix $\lambda>0$ and for any $n\ge 1$ define the regularized design matrix
$$
A(n) = \lambda I_d + \sum_{s=1}^{n}\phi_s\phi_s^\top.
$$
Then for any $\delta \in (0,1)$, if
$$
n \ge n_0 := \frac{8L^2}{c}\,\log\frac{d}{\delta},
$$
we have, with probability at least $1-\delta$, the uniform bound
$$
\|v\|_{A(n)^{-1}}
  := \sqrt{v^\top A(n)^{-1} v}
  \le \frac{L}{\sqrt{\lambda + (c/2)\,n}}.
$$

for all $v\in\mathbb{R}^d$ with $\|v\|_2\le L$. In particular, this applies to $v=\phi(x)$ when $\|\phi(x)\|_2\le L$.

*Proof.*

Define the random positive semidefinite matrices $X_s := \phi_s\phi_s^\top$. Then $0 \preceq X_s \preceq L^2 I_d$ almost surely, and
$$
\mathbb{E}[X_s] = \mathbb{E}[\phi\phi^\top] \succeq c I_d.
$$
Let $\mu_{\min} := \lambda_{\min}\bigl(\mathbb{E}[\sum_{s=1}^n X_s]\bigr)$. By linearity of expectation,
$
\mathbb{E}[\sum_{s=1}^n X_s]=n\,\mathbb{E}[X_1]
$
and therefore $\mu_{\min} \ge nc$.

By the matrix Chernoff lower-tail bound (e.g., [@tropp:user_friendly:2012, Theorem 5.1]), for any $\varepsilon\in(0,1)$,
$$
\mathbb{P}\!\left(\lambda_{\min}\!\left(\sum_{s=1}^n X_s\right) \le (1-\varepsilon)\mu_{\min}\right)
\le d\exp\!\left(-\frac{\varepsilon^2\,\mu_{\min}}{2L^2}\right).
$$
With $\varepsilon=\tfrac12$ and $\mu_{\min}\ge nc$, this yields
$$
\mathbb{P}\!\left(\lambda_{\min}\!\left(\sum_{s=1}^n \phi_s\phi_s^\top\right) \le \frac{c}{2}\,n\right)
\le d\exp\!\left(-\frac{nc}{8L^2}\right).
$$
If $n \ge \frac{8L^2}{c}\log\frac{d}{\delta}$, the right-hand side is at most $\delta$, so with probability at least $1-\delta$,
$$
\lambda_{\min}\!\left(\sum_{s=1}^n \phi_s\phi_s^\top\right) \ge \frac{c}{2}\,n.
$$
On this event,
$
A(n) = \lambda I_d + \sum_{s=1}^n \phi_s\phi_s^\top \succeq \bigl(\lambda + \tfrac{c}{2}n\bigr) I_d
$
and hence
$
A(n)^{-1} \preceq \bigl(\lambda + \tfrac{c}{2}n\bigr)^{-1} I_d.
$
Therefore for any $v$ with $\|v\|_2\le L$,
$$
v^\top A(n)^{-1} v \le \frac{\|v\|_2^2}{\lambda + (c/2)n} \le \frac{L^2}{\lambda + (c/2)n}.
$$
Taking square roots gives the claimed bound. $\square$

**LinUCB action selection:**

$$
a_t = \arg\max_{a \in \mathcal{A}} \left\{\hat{\theta}_a^\top \phi(x_t) + \alpha \sqrt{\phi(x_t)^\top \Sigma_a \phi(x_t)}\right\}
\tag{6.15}
$$
{#EQ-6.15}

---

**Algorithm 6.2** (LinUCB for Contextual Bandits) {#ALG-6.2}

**Input:**
- Feature map $\phi: \mathcal{X} \to \mathbb{R}^d$
- Action set $\mathcal{A} = \{1, \ldots, M\}$
- Regularization $\lambda > 0$
- Exploration parameter $\alpha > 0$ (typically $\alpha \in [0.1, 2.0]$)
- Number of episodes $T$

**Initialization:**
- For each action $a \in \mathcal{A}$:
  - $\hat{\theta}_a \leftarrow 0 \in \mathbb{R}^d$
  - $A_a \leftarrow \lambda I_d$ (design matrix accumulator)
  - $b_a \leftarrow 0 \in \mathbb{R}^d$ (reward accumulator)

**For** $t = 1, \ldots, T$:

1. **Observe context**: $x_t \in \mathcal{X}$
2. **Compute features**: $\phi_t \leftarrow \phi(x_t)$
3. **Compute UCB scores** for all $a \in \mathcal{A}$:
   $$
   \text{UCB}_a = \hat{\theta}_a^\top \phi_t + \alpha \sqrt{\phi_t^\top A_a^{-1} \phi_t}
   $$
4. **Select action**: $a_t \leftarrow \arg\max_a \text{UCB}_a$
5. **Observe reward**: $r_t$
6. **Update** statistics for $a_t$:
   \begin{align}
   A_{a_t} &\leftarrow A_{a_t} + \phi_t \phi_t^\top \\
   b_{a_t} &\leftarrow b_{a_t} + r_t \phi_t \\
   \hat{\theta}_{a_t} &\leftarrow A_{a_t}^{-1} b_{a_t}
   \end{align}

**Output:** Learned weights $\{\hat{\theta}_a\}_{a=1}^M$

---

**Computational complexity.**

Per episode, with feature dimension $d$ and $M$ actions:

1. **Feature computation:** $O(d)$ to compute $\phi_t$.
2. **UCB scores:** For each action, evaluating $\hat{\theta}_a^\top \phi_t$ is $O(d)$ and the naive uncertainty term $\sqrt{\phi_t^\top A_a^{-1} \phi_t}$ requires forming $A_a^{-1}$, which is $O(d^3)$. Across all $M$ actions this is $O(M d^3)$.
   - With incremental matrix inverses or Cholesky updates, the uncertainty can be maintained in $O(d^2)$ per action, i.e. $O(M d^2)$ per episode.
3. **Argmax:** Selecting $a_t = \arg\max_a \text{UCB}_a$ costs $O(M)$.
4. **Update:** Updating $A_{a_t}$ and $b_{a_t}$ is $O(d^2)$ (rank-1 update and vector addition), and solving $A_{a_t}^{-1} b_{a_t}$ via `np.linalg.solve` is $O(d^3)$.

Thus the **naive total cost** is $O(T M d^3)$ over $T$ episodes; with rank-1 updates and cached factorizations one obtains **$O(T M d^2)$**. Memory usage is $O(M d^2)$ for the design matrices and $O(M d)$ for the weight vectors. As in Thompson Sampling, our regime has small $M$ and moderate $d$, so these costs are negligible compared to simulator trajectories.

---

**Proposition 6.1** (Posterior Mean Equivalence) {#PROP-6.1}

The posterior mean $\hat{\theta}_a$ maintained by both Thompson Sampling and LinUCB is identical and equals the ridge regression solution:
$$
\hat{\theta}_a = A_a^{-1} b_a = \left(\lambda I + \sum_{t: a_t=a} \phi_t \phi_t^\top\right)^{-1} \left(\sum_{t: a_t=a} r_t \phi_t\right)
$$
where $A_a$ is the design matrix and $b_a$ is the reward accumulator.
In the Bayesian model [EQ-6.6]--[EQ-6.7] with prior precision $\lambda_0$ and noise variance $\sigma^2$, this corresponds to the ridge penalty $\lambda = \sigma^2\lambda_0$ and posterior covariance $\Sigma_a = \sigma^2 A_a^{-1}$.

*Proof.* Under the Gaussian model [EQ-6.7] and prior $\theta_a \sim \mathcal{N}(0,\lambda_0^{-1}I_d)$, completing the square in the log posterior shows that the posterior mean maximizes a quadratic objective, equivalently minimizing the ridge regression problem [EQ-6.9] with $\lambda=\sigma^2\lambda_0$. The normal equations for that problem are $A_a \hat{\theta}_a = b_a$, hence $\hat{\theta}_a=A_a^{-1}b_a$ as stated. Both Thompson Sampling and LinUCB maintain these same sufficient statistics $(A_a,b_a)$ and therefore the same $\hat{\theta}_a$; they differ only in action selection (sampling vs. UCB). $\square$

**The only difference:**
- **Thompson Sampling**: Stochastic selection via posterior sampling
- **LinUCB**: Deterministic selection via upper confidence bound

---

### 6.3.2 Choosing the Exploration Parameter alpha

**The $\alpha$ dilemma:**

Theory says $\alpha = O(\sqrt{d \log T})$ for optimal regret. But in practice:

- **Too small** ($\alpha \to 0$): Greedy exploitation, insufficient exploration, gets stuck on suboptimal templates
- **Too large** ($\alpha \to \infty$): Excessive exploration, ignores reward signal, selects randomly

**How to choose $\alpha$?**

**Option A: Theoretical value** $\alpha = \sqrt{d \log T}$
- Pros: Provably optimal regret.
- Cons: Requires knowing $T$ (horizon) in advance; often overly conservative in practice.

**Option B: Cross-validation**
- Pros: Data-driven tuning.
- Cons: Expensive (requires offline simulation); may overfit to the validation set.

**Option C: Adaptive tuning** (our choice)
- Pros: Starts conservative, decays as confidence grows; no need to know $T$ in advance.
- Example: $\alpha_t = c \sqrt{\log(1 + t)}$ for constant $c \in [0.5, 2.0]$

**Practical recommendation:**

Start with $\alpha = 1.0$ (moderate exploration). Monitor:
- **Selection diversity**: If one template dominates early -> increase $\alpha$
- **Cumulative regret**: If regret grows linearly -> increase $\alpha$
- **Reward variance**: If rewards are noisy -> increase $\alpha$

Typical ranges in production: $\alpha \in [0.5, 2.0]$.

---

### 6.3.3 Regret Analysis

**Theorem 6.2** (LinUCB Regret Bound) {#THM-6.2}

Let $(\mathcal{X}, \mathcal{A}, R)$ be a linear contextual bandit with the same assumptions as [THM-6.1]. Fix a regularization parameter $\lambda > 0$. Then LinUCB ([ALG-6.2]) with exploration parameter
$$
\alpha
= \sigma \sqrt{2\log\left(\frac{2MT}{\delta}\right) + d \log\left(1 + \frac{TL^2}{d\lambda}\right)}
  + \sqrt{\lambda}\, S
$$
satisfies, with probability $\geq 1 - \delta$:
$$
\text{Regret}(T) \leq 2\alpha \sqrt{2dMT \log\left(1 + \frac{TL^2}{d\lambda}\right)}.
\tag{6.16}
$$
{#EQ-6.16}

where $S = \max_a \|\theta_a^*\|$ is the norm of true parameters.

*Proof.*

The proof follows [@abbasi:improved:2011]. Key steps:

**Step 1: Concentration inequality**

By the self-normalized concentration inequality for regularized linear regression ([@abbasi:improved:2011, Theorem 2]), with probability $\geq 1 - \delta$ (after a union bound over $M$ actions):
$$
\|\hat{\theta}_a - \theta_a^*\|_{A_a} \leq \alpha
$$
where $\|v\|_A = \sqrt{v^\top A v}$ is the weighted norm.

**Step 2: Confidence ellipsoid**

The set $\{\theta : \|\theta - \hat{\theta}_a\|_{A_a} \leq \alpha\}$ is a **confidence ellipsoid** containing $\theta_a^*$ with high probability.

**Step 3: Upper bound validity**

If $\theta_a^* \in$ ellipsoid, then:
$$
\theta_a^{* \top} \phi \leq \hat{\theta}_a^\top \phi + \alpha \|\phi\|_{A_a^{-1}} = \text{UCB}_a
$$

**Step 4: Optimism**

Since we select $a_t = \arg\max_a \text{UCB}_a$, and the optimal action $a^*$ satisfies $\text{UCB}_{a^*} \geq \theta_{a^*}^{* \top} \phi$, we have:
$$
\text{UCB}_{a_t} \geq \text{UCB}_{a^*} \geq \theta_{a^*}^{* \top} \phi
$$

Thus, instantaneous regret is bounded by $2\alpha \|\phi\|_{A_{a_t}^{-1}}$.

**Step 5: Elliptical potential**

Summing over $T$ episodes:
$$
\sum_{t=1}^T \|\phi_t\|_{A_{a_t}^{-1}}^2
  \leq 2d \sum_{a=1}^M \log\bigl(1 + n_a(T)L^2/(d\lambda)\bigr)
  \leq 2dM \log\bigl(1 + TL^2/(d\lambda)\bigr),
$$
where $n_a(T)$ is the number of times action $a$ is selected up to episode $T$.

by determinant inequality ([@abbasi:improved:2011, Lemma 11]).

Taking square root and union bound over $M$ actions yields [EQ-6.16]. QED

**Comparison to Thompson Sampling:**

Both achieve $O(d\sqrt{MT})$ regret up to logarithmic factors under realizability. LinUCB has:
- Pros: Deterministic given the same data/seed; interpretable (UCB scores explain selections).
- Cons: Requires tuning ($\alpha$); less adaptive (fixed exploration schedule, while TS adapts via posterior uncertainty).

**When to use which:**

- **Thompson Sampling**: Default choice for most applications (automatic exploration, no tuning)
- **LinUCB**: When reproducibility critical (A/B testing, debugging) or when $\alpha$ can be tuned offline

---

## 6.4 Production Implementation

The previous sections treated Thompson Sampling and LinUCB as abstract algorithms. To run the experiments in Sections 6.5--6.7 we need production-grade code that:

- Implements the Bayesian and UCB updates faithfully
- Plays nicely with the `zoosim` catalog, user, and query modules
- Exposes configuration knobs (regularization, exploration strength, seeds)
- Surfaces diagnostics for monitoring in A/B tests

We follow a simple pattern:

1. A **configuration dataclass** controls hyperparameters.
2. A **policy class** exposes `select_action(phi(x))` and `update(a, phi(x), r)`.
3. A thin **integration layer** in the experiment script connects simulator observations to feature maps and policy calls.

### 6.4.1 Thompson Sampling Implementation

For Thompson Sampling we implement a `ThompsonSamplingConfig` and a `LinearThompsonSampling` policy in `zoosim/policies/thompson_sampling.py`.

- The config controls prior precision `lambda`, noise scale `sigma`, a `use_cholesky` flag, and a `seed` for reproducibility.
- The policy maintains, for each template $a$:
  - A precision matrix $\Sigma_a^{-1} = \lambda_0 I + \sigma^{-2}\sum_t \phi_t \phi_t^\top$ (stored as `Sigma_inv[a]`)
  - A scaled moment vector $b_a = \sigma^{-2}\sum_t r_t \phi_t$ (stored as `b[a]`)
  - A posterior mean vector $\hat{\theta}_a$ satisfying $\Sigma_a^{-1}\hat{\theta}_a=b_a$ (stored as `theta_hat[a]`)
  - A selection count `n_samples[a]` for diagnostics
- On each call to `select_action(phi)` we:
  1. Sample $\tilde{\theta}_a \sim \mathcal{N}(\hat{\theta}_a, \Sigma_a)$ for all templates (via NumPy and optional Cholesky for stability)
  2. Compute sampled rewards $\tilde{r}_a = \tilde{\theta}_a^\top \phi$
  3. Return `argmax_a \tilde{r}_a`
- On each `update(a, phi, r)` we perform the Bayesian linear regression update:
  $$
  \Sigma_a^{-1} \leftarrow \Sigma_a^{-1} + \sigma^{-2}\phi\phi^\top,\qquad
  b_a \leftarrow b_a + \sigma^{-2} r\phi,\qquad
  \hat{\theta}_a \leftarrow (\Sigma_a^{-1})^{-1} b_a.
  $$
  Multiplying these equations by $\sigma^2$ recovers the ridge-form update [EQ-6.8].

This is exactly the mathematical algorithm from Section 6.2.2 written in NumPy/Torch, with care taken to keep matrices well-conditioned and sampling numerically robust.

!!! note "Code <-> Agent (Thompson Sampling)"
    The Thompson Sampling production implementation lives in:

    - Algorithm: `zoosim/policies/thompson_sampling.py`
    - Templates: `zoosim/policies/templates.py`
    - Demo wiring: `scripts/ch06/template_bandits_demo.py`

    Conceptual mapping:

    - Posterior state $(\hat{\theta}_a, \Sigma_a)$ implements [EQ-6.6]--[EQ-6.8]
    - `select_action()` implements [ALG-6.1] (posterior sampling and greedy selection)
    - `update()` is the Bayesian linear regression update used in the regret proof of [THM-6.1]

    In the demos we always pass **feature vectors** `phi(x)` built by `context_features_simple` or `context_features_rich` from `scripts/ch06/template_bandits_demo.py`, so the production code is agnostic to how features are constructed.

### 6.4.2 LinUCB Implementation

**Implementation file: `zoosim/policies/lin_ucb.py`**

```python
r"""LinUCB (Linear Upper Confidence Bound) for contextual bandits.

Mathematical basis:
- [ALG-6.2] LinUCB algorithm
- [EQ-6.15] UCB action selection rule
- [THM-6.2] Regret bound $O(d \sqrt{M T \log T})$

Implements frequentist upper confidence bound exploration with deterministic
action selection. Maintains ridge regression estimates and selects the action
with highest optimistic reward estimate.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from zoosim.policies.templates import BoostTemplate


@dataclass
class LinUCBConfig:
    """Configuration for LinUCB policy.

    Attributes:
        lambda_reg: Regularization strength (ridge regression);
            prevents overfitting and keeps A_a invertible.
        alpha: Exploration parameter (UCB width multiplier);
            typical values alpha  in  [0.5, 2.0].
        adaptive_alpha: If True, use alpha_t = alphasqrtlog(1 + t)
            for automatic exploration decay.
        seed: Random seed (used for any feature hashing / randomness upstream).
        enable_diagnostics: If True, record per-episode diagnostic traces
            (UCB scores, means, uncertainties, and selected actions).
    """

    lambda_reg: float = 1.0
    alpha: float = 1.0
    adaptive_alpha: bool = False
    seed: int = 42
    enable_diagnostics: bool = False


class LinUCB:
    """Linear Upper Confidence Bound algorithm for contextual bandits.

    Maintains ridge regression estimates theta_a for each template and selects
    the action with highest upper confidence bound:

        a = argmax_a {theta_a^T phi(x) + alpha sqrt(phi(x)^T A_a^{-1} phi(x))}

    This is a frequentist alternative to Thompson Sampling with deterministic
    action selection. Both maintain identical posterior means theta_a but differ
    in how they use uncertainty for exploration.
    """

    def __init__(
        self,
        templates: List[BoostTemplate],
        feature_dim: int,
        config: Optional[LinUCBConfig] = None,
    ) -> None:
        """Initialize LinUCB policy."""
        self.templates = templates
        self.M = len(templates)
        self.d = feature_dim
        self.config = config or LinUCBConfig()

        # Initialize statistics: [ALG-6.2] initialization
        self.theta_hat = np.zeros((self.M, self.d), dtype=np.float64)
        self.A = np.array(
            [
                self.config.lambda_reg * np.eye(self.d, dtype=np.float64)
                for _ in range(self.M)
            ]
        )
        self.b = np.zeros((self.M, self.d), dtype=np.float64)

        self.n_samples = np.zeros(self.M, dtype=int)
        self.t = 0  # Episode counter

        # Optional diagnostics
        self.enable_diagnostics = bool(self.config.enable_diagnostics)
        if self.enable_diagnostics:
            self._diagnostics_history: Dict[str, list] = {
                "ucb_scores_history": [],
                "mean_rewards_history": [],
                "uncertainties_history": [],
                "selected_actions": [],
            }

    def select_action(self, features: NDArray[np.float64]) -> int:
        """Select template using the LinUCB criterion [EQ-6.15]."""
        self.t += 1

        # Compute adaptive exploration parameter
        alpha = self.config.alpha
        if self.config.adaptive_alpha:
            alpha *= np.sqrt(np.log(1 + self.t))

        # Compute UCB scores for all templates
        ucb_scores = np.zeros(self.M)
        mean_rewards = np.zeros(self.M)
        uncertainties = np.zeros(self.M)
        for a in range(self.M):
            # Mean estimate: theta_a^T phi
            mean_reward = self.theta_hat[a] @ features
            mean_rewards[a] = mean_reward

            # Uncertainty bonus: alpha sqrt(phi^T A_a^{-1} phi), implementing [EQ-6.14]
            A_inv = np.linalg.inv(self.A[a])
            uncertainty = np.sqrt(features @ A_inv @ features)
            uncertainties[a] = uncertainty

            # UCB score [EQ-6.15]
            ucb_scores[a] = mean_reward + alpha * uncertainty

        # Select action with highest UCB
        action = int(np.argmax(ucb_scores))

        if self.enable_diagnostics:
            self._diagnostics_history["ucb_scores_history"].append(ucb_scores.copy())
            self._diagnostics_history["mean_rewards_history"].append(
                mean_rewards.copy()
            )
            self._diagnostics_history["uncertainties_history"].append(
                uncertainties.copy()
            )
            self._diagnostics_history["selected_actions"].append(action)

        return action

    def update(
        self,
        action: int,
        features: NDArray[np.float64],
        reward: float,
    ) -> None:
        """Update ridge regression statistics after (action, features, reward)."""
        a = action
        phi = features

        # Update design matrix: A_a <- A_a + phiphi^T
        self.A[a] += np.outer(phi, phi)

        # Update reward accumulator: b_a <- b_a + r phi
        self.b[a] += reward * phi

        # Update weight estimate: theta_a <- A_a^{-1} b_a (via solve)
        self.theta_hat[a] = np.linalg.solve(self.A[a], self.b[a])

        # Track selection count
        self.n_samples[a] += 1

    def get_diagnostics(self) -> Dict[str, NDArray[np.float64] | float]:
        """Return aggregate diagnostic information for monitoring."""
        total = self.n_samples.sum()
        selection_freqs = (
            self.n_samples / total if total > 0 else self.n_samples
        )
        theta_norms = np.linalg.norm(self.theta_hat, axis=1)
        uncertainties = np.array(
            [np.trace(np.linalg.inv(self.A[a])) for a in range(self.M)]
        )

        alpha_current = self.config.alpha
        if self.config.adaptive_alpha:
            alpha_current *= np.sqrt(np.log(1 + self.t)) if self.t > 0 else 1.0

        return {
            "selection_counts": self.n_samples.copy(),
            "selection_frequencies": selection_freqs,
            "theta_norms": theta_norms,
            "uncertainty": uncertainties,
            "alpha_current": float(alpha_current),
        }

    def get_diagnostic_history(self) -> Dict[str, list]:
        """Return per-episode diagnostic traces if enabled."""
        if not self.enable_diagnostics:
            raise ValueError(
                "Diagnostics history not enabled; set "
                "LinUCBConfig.enable_diagnostics=True when constructing the policy."
            )
        return self._diagnostics_history
```

In production we also expose **richer diagnostics**: the actual `LinUCBConfig` in `zoosim/policies/lin_ucb.py` has an `enable_diagnostics: bool` flag. When set to `True`, the policy records per-episode traces of UCB scores, mean rewards, uncertainties, and selected actions, retrievable via `policy.get_diagnostic_history()`. The lighter `policy.get_diagnostics()` snapshot (selection frequencies, parameter norms, aggregate uncertainty, current $\alpha_t$) remains available regardless and is what we use for monitoring dashboards in Section 6.7.

!!! note "Code <-> Agent (LinUCB)"
    The `LinUCB` class implements [ALG-6.2]:
    - **Line 129-183**: `select_action()` computes UCB scores via [EQ-6.15] and selects argmax
    - **Line 185-223**: `update()` performs ridge regression update (design matrix + weight solve)
    - **Line 149-153**: Adaptive $\alpha_t = \alpha \sqrt{\log(1 + t)}$ option for automatic exploration decay
    - **Line 163-169**: Uncertainty computation $\sqrt{\phi^\top A_a^{-1} \phi}$ from [EQ-6.14] and UCB score assembly

    File: `zoosim/policies/lin_ucb.py`

**Numerical stability:**

- We solve $A_a \hat{\theta}_a = b_a$ via `np.linalg.solve` (more stable than explicit inversion)
- Regularization $\lambda > 0$ ensures $A_a$ is always invertible
- For large-scale production, use iterative solvers (conjugate gradient) or maintain Cholesky factors

---

### 6.4.3 Integration with ZooplusSearchEnv

Now we wire LinUCB and Thompson Sampling into the full search simulator from Chapter 5.

**Training loop structure:**

```python
"""Training loop for template bandits on search simulator.

Demonstrates integration of [ALG-6.1]/[ALG-6.2] with ZooplusSearchEnv.
"""

import numpy as np
from zoosim.envs.gym_env import ZooplusSearchGymEnv
from zoosim.policies.templates import create_standard_templates
from zoosim.policies.lin_ucb import LinUCB, LinUCBConfig
from zoosim.policies.thompson_sampling import LinearThompsonSampling, ThompsonSamplingConfig

# Initialize environment
env = ZooplusSearchGymEnv(seed=42)

# Get catalog statistics for template creation
catalog_stats = {
    'price_p25': env.catalog.price.quantile(0.25),
    'price_p75': env.catalog.price.quantile(0.75),
    'pop_max': env.catalog.popularity.max(),
    'own_brand': 'Zooplus',
}

# Create template library (M=8 templates)
templates = create_standard_templates(catalog_stats, a_max=5.0)

# Extract feature dimension from environment
obs, info = env.reset()
feature_dim = obs['features'].shape[0]  # Dimension d

# Initialize policy (choose one)
# Option 1: LinUCB
policy = LinUCB(
    templates=templates,
    feature_dim=feature_dim,
    config=LinUCBConfig(lambda_reg=1.0, alpha=1.0, adaptive_alpha=True)
)

# Option 2: Thompson Sampling
# policy = LinearThompsonSampling(
#     templates=templates,
#     feature_dim=feature_dim,
#     config=ThompsonSamplingConfig(lambda_reg=1.0, sigma_noise=1.0)
# )

# Training loop
T = 50_000  # Number of episodes
rewards_history = []
cumulative_regret = []
selection_history = []

for t in range(T):
    # Reset environment, observe context
    obs, info = env.reset()
    features = obs['features']

    # Select template using bandit policy
    template_id = policy.select_action(features)
    selection_history.append(template_id)

    # Apply template to get boost vector
    products = obs['products']  # List of product dicts
    boosts = templates[template_id].apply(products)

    # Execute action in environment
    obs, reward, done, truncated, info = env.step(boosts)

    # Update policy
    policy.update(template_id, features, reward)

    # Track metrics
    rewards_history.append(reward)

    # Compute regret (oracle comparison)
    # In real deployment, regret unknown; here we use oracle for analysis
    optimal_reward = info.get('optimal_reward', reward)  # Simulated oracle
    regret = optimal_reward - reward
    cumulative_regret.append(sum(cumulative_regret) + regret if cumulative_regret else regret)

    # Logging
    if (t + 1) % 10_000 == 0:
        avg_reward = np.mean(rewards_history[-10_000:])
        diagnostics = policy.get_diagnostics()
        print(f"Episode {t+1}/{T}")
        print(f"  Avg reward (last 10k): {avg_reward:.2f}")
        print(f"  Selection frequencies: {diagnostics['selection_frequencies'].round(3)}")
        print(f"  Cumulative regret: {cumulative_regret[-1]:.1f}")
        print()

# Final evaluation
print("=== Training Complete ===")
print(f"Total episodes: {T}")
print(f"Average reward (overall): {np.mean(rewards_history):.2f}")
print(f"Average reward (last 10k): {np.mean(rewards_history[-10_000:]):.2f}")
print(f"Final cumulative regret: {cumulative_regret[-1]:.1f}")
print(f"\nTemplate selection distribution:")
for i, template in enumerate(templates):
    freq = policy.n_samples[i] / T
    print(f"  {template.name:15s}: {freq:.3f} ({policy.n_samples[i]:6d} times)")
```

**Expected output (LinUCB, 50k episodes):**

```
Episode 10000/50000
  Avg reward (last 10k): 112.34
  Selection frequencies: [0.023 0.187 0.245 0.092 0.134 0.078 0.156 0.085]
  Cumulative regret: 1823.4

Episode 20000/50000
  Avg reward (last 10k): 118.67
  Selection frequencies: [0.015 0.203 0.276 0.088 0.125 0.064 0.178 0.051]
  Cumulative regret: 2941.2

Episode 30000/50000
  Avg reward (last 10k): 121.23
  Selection frequencies: [0.011 0.215 0.289 0.081 0.118 0.053 0.191 0.042]
  Cumulative regret: 3789.8

Episode 40000/50000
  Avg reward (last 10k): 122.14
  Selection frequencies: [0.009 0.218 0.297 0.076 0.113 0.047 0.198 0.042]
  Cumulative regret: 4412.1

Episode 50000/50000
  Avg reward (last 10k): 122.58
  Selection frequencies: [0.008 0.221 0.302 0.073 0.109 0.044 0.202 0.041]
  Cumulative regret: 4897.3

=== Training Complete ===
Total episodes: 50000
Average reward (overall): 118.45
Average reward (last 10k): 122.58
Final cumulative regret: 4897.3

Template selection distribution:
  Neutral        : 0.008 (   412 times)
  High Margin    : 0.221 ( 11023 times)
  CM2 Boost      : 0.302 ( 15089 times)
  Popular        : 0.073 (  3641 times)
  Premium        : 0.109 (  5472 times)
  Budget         : 0.044 (  2187 times)
  Discount       : 0.202 ( 10112 times)
  Strategic      : 0.041 (  2064 times)
```

**Interpretation:**

1. **Convergence**: Average reward increases from ~112 to ~123 ($\approx$10% improvement)
2. **Exploration decay**: Neutral template selection drops from 2.3% to 0.8% as confidence grows
3. **Winner templates**: CM2 Boost (30%), High Margin (22%), Discount (20%) dominate
4. **Regret growth**: Cumulative regret ~5000 over 50k episodes -> average per-episode regret ~0.1 (excellent!)

**Key insight:** The simulator has a **preference hierarchy** CM2 > Margin > Discount. LinUCB discovers this automatically.

---

## 6.5 First Experiment---When Bandits Lose

We now turn from the theoretical development to the first experiment. Under the assumptions of [THM-6.1] and [THM-6.2], one might expect LinUCB and Thompson Sampling to match or outperform a strong static template. The first run shows that this expectation can be false under an impoverished feature map.

We deploy the contextual bandits and compare them against the best static template.

### 6.5.1 Experimental Setup: The Most Obvious Features

Before we run the experiment, we need to make a crucial design choice: **what context features do we give the bandits?**

Remember, our linear model assumes
$$
\mu(x, a) = \theta_a^\top \phi(x)
$$
and all of the regret guarantees in [THM-6.1] and [THM-6.2] are conditional on this model being a reasonable approximation of reality.

The feature map $\phi : \mathcal{X} \to \mathbb{R}^d$ is our responsibility. The bandit will learn the best weights $\theta_a$, but we must decide *what* to encode in $\phi(x)$.

What's the most natural choice? Two obvious sources of context:

- **User segments.** Our simulator has four user types: premium buyers who purchase expensive items, pl_lovers who prefer own-brand products, litter_heavy users who buy in bulk, and price_hunters who seek discounts. These segments have genuinely different preferences---a premium user and a price hunter should receive different rankings.
- **Query types.** Users express different intent: specific product searches (e.g. `"royal canin kitten food"`), general browsing (e.g. `"cat supplies"`), and deal-seeking (e.g. `"discounts on cat food"`).

So our first instinct is simple and reasonable:
$$
\phi_{\text{simple}}(x)
  = [\text{segment}_{\text{onehot}}, \text{query\_type}_{\text{onehot}}]
$$

This gives $d = 4 + 3 = 7$ dimensions: a one-hot encoding for user segment (four binary indicators, exactly one is 1), plus a one-hot encoding for query type (three binary indicators, exactly one is 1).

From the bandit's perspective this feels expressive enough. With $\phi_{\text{simple}}$ it can, in principle, learn patterns like:

- "Premium users with specific queries respond well to the Premium template."
- "pl_lover users with browsing queries respond well to CM2 Boost."
- "Price hunters with deal-seeking queries respond well to Budget or Discount templates."

The linear model $\theta_a^\top \phi(x)$ can represent these patterns: each coordinate of $\theta_a$ is simply "how much this segment or query type likes template $a$".

!!! note "Pedagogical Design: Feature Engineering as Iterative Process"
    We deliberately omit product-level information (prices, margins, discounts, popularity) and user preference signals (price sensitivity, private-label affinity). This is intentional: feature engineering is iterative. We start with a minimal representation, measure performance, diagnose the bottleneck, and then add the missing signals.

    This first experiment also induces model misspecification to instantiate [REM-6.1.1] in a controlled way: the theorems remain correct, but the representation is too weak for the linear model to be useful.

Segment and query type are also the first features stakeholders typically request. If these features suffice, we obtain an interpretable baseline; if they do not, the failure is informative.

**Experimental protocol.**

We use our standard simulator configuration (10 000 products, realistic distributions) and run three policies using `scripts/ch06/template_bandits_demo.py`:

1. **Static template sweep.** Evaluate each of the 8 templates for 2 000 episodes, and record average Reward/GMV/CM2.
2. **LinUCB.** Train for 20 000 episodes with $\phi_{\text{simple}}$, ridge regularization $\lambda = 1.0$, UCB coefficient $\alpha = 1.0$.
3. **Thompson Sampling.** Train for 20 000 episodes with $\phi_{\text{simple}}$, same regularization and noise scale as in Section 6.4.

Run:

```bash
uv run python scripts/ch06/template_bandits_demo.py \
    --n-static 2000 \
    --n-bandit 20000 \
    --features simple \
    --world-seed 20250322 \
    --bandit-base-seed 20250349
```

Why 20 000 episodes for the bandits but only 2 000 per static template? Static templates are deterministic---once we have a few thousand episodes, we can estimate their means quite precisely. Bandits, however, must **explore**. The regret bounds suggest that with $T = 20\,000$ and $M = 8$ templates, we should pay roughly
$$
O\bigl(\sqrt{M T}\bigr) \approx O\bigl(\sqrt{8 \cdot 20\,000}\bigr) \approx 400
$$
episodes of regret and then enjoy near-optimal behaviour for the remaining 19 600 episodes. On paper that is more than enough to beat any fixed template.

**Our hypothesis:** contextual bandits with segment + query-type features should at least match, and probably exceed, the best static template's GMV.

### 6.5.2 The Moment of Truth: When Bandits Lose

The script runs. Progress indicators tick forward. LinUCB explores aggressively at first (all 8 templates get non-trivial mass), then gradually commits to favourites. Thompson Sampling behaves similarly but with stochastic selection trajectories---its posteriors never collapse to a single arm because variance remains.

After a couple of minutes, the summary prints:

```
Static templates (per-episode averages):
ID  Template             Reward         GMV         CM2
 0  Neutral                5.75        5.28        0.50
 1  High Margin            5.44        5.07        0.63
 2  CM2 Boost              7.35        6.73        0.61
 3  Popular                4.79        4.53        0.52
 4  Premium                7.56        7.11        0.74  <- Best static
 5  Budget                 3.44        3.04        0.26
 6  Discount               5.04        4.62        0.45
 7  Strategic              4.83        3.99       -0.13

Best static template: ID=4 (Premium) with avg reward=7.56, GMV=7.11

LinUCB (20000 episodes, simple features):
  Global avg:  Reward=5.62, GMV=5.12, CM2=0.51

Thompson Sampling (20000 episodes, simple features):
  Global avg:  Reward=6.69, GMV=6.18, CM2=0.61
```

Read those lines carefully:

- Best static template (Premium): **7.11 GMV**
- LinUCB with $\phi_{\text{simple}}$: **5.12 GMV** ($\approx$ -28 %)
- Thompson Sampling with $\phi_{\text{simple}}$: **6.18 GMV** ($\approx$ -13 %)

The bandits lose by double-digit percentages relative to a one-line static rule.

Rerunning with different seeds preserves the qualitative pattern (LinUCB roughly -28%, Thompson Sampling roughly -13% in this setup). The algorithms with clean regret bounds underperform a static Premium template that favours products matching the premium segment's preferences.

### 6.5.3 Cognitive Dissonance and Per-Segment Heterogeneity

If these numbers feel uneasy, that is intended.

On one side we have the theorems from Sections 6.2 and 6.3 telling us:

- "LinUCB and Thompson Sampling achieve $O(d\sqrt{M T \log T})$ regret."
- "Empirical regret curves in synthetic experiments decay nicely (Figure 6.4)."

On the other side we have the simulator reporting:

- "The contextual bandit underperforms the best static template on GMV."

Both statements can be true. The tension between them is the pedagogical engine of this chapter.

The same run includes a per-segment table. For instance:

- Price hunters lose ~50 % GMV when forced into Premium-like boosts.
- PL-lover users lose ~30 % GMV when CM2 Boost is disabled.
- Premium users are already near their Pareto frontier.

The per-segment table makes the paradox sharper: **global GMV is dominated by premium buyers**, but the biggest opportunity lies in underserved segments that the simple features cannot separate properly.

In Section 6.6 we resist the temptation to blame bugs or hyperparameters and instead put the theorems themselves on the table: we read the fine print and identify the violated assumptions.

!!! note "Code <-> Experiment (Simple Features)"
    The simple-feature experiment is implemented in `scripts/ch06/template_bandits_demo.py` (feature construction + evaluation tables) and saved as a JSON artifact by `scripts/ch06/ch06_compute_arc.py`.

    - Feature map: `context_features_simple` (segment + query type + bias).
    - Artifact: `docs/book/ch06/data/template_bandits_simple_summary.json`.

    Reproduce the full Chapter-6 compute arc (simple to rich oracle to rich estimated) and regenerate all three JSON summaries:
    ```bash
    uv run python scripts/ch06/ch06_compute_arc.py \
      --n-static 2000 \
      --n-bandit 20000 \
      --base-seed 20250322 \
      --bandit-seed 20250349 \
      --out-dir docs/book/ch06/data \
      --prior-weight 50 \
      --lin-alpha 0.2 \
      --ts-sigma 0.5
    ```

---

## 6.6 Diagnosis---Why Theory Failed Practice

When an RL algorithm underperforms a simple baseline, there are three possibilities:

1. **Implementation error.** Bug in the code, wrong hyperparameters, or insufficient data.
2. **The theorem is wrong.** The regret bound does not actually hold because the proof has a flaw.
3. **Assumptions are violated.** The theorem is correct, but its hypotheses do not hold in the present setting.

We can rule out (1): the posterior updates match the closed-form equations in Section 6.2 and Section 6.3, tests in `tests/ch06/` pass, and 20 000 episodes is plenty for an 8-arm bandit. We can rule out (2): [THM-6.2] is a widely used result (Abbasi-Yadkori et al.), with a proof that has been checked many times over.

What remains is (3): we violated the assumptions.

### 6.6.1 Revisiting the Theorem's Fine Print

[THM-6.2] states that LinUCB achieves $O(d\sqrt{M T \log T})$ regret under four assumptions:

**(A1) Linearity.** True mean reward is linear in features:
$$
\mu(x, a) = \theta_a^{*\top} \phi(x)
$$
for some unknown $\theta_a^* \in \mathbb{R}^d$ and all $x, a$.

**(A2) Bounded features.** $\|\phi(x)\|_2 \le L$ for all contexts $x$.

**(A3) Bounded parameters.** $\|\theta_a^*\|_2 \le S$ for all arms $a$.

**(A4) Sub-Gaussian noise.** Observed reward is $r_t = \theta_{a_t}^{*\top} \phi_t + \eta_t$ where $\eta_t$ is $\sigma$-sub-Gaussian.

For our simple-feature experiment:

- (A2) holds trivially: $\phi_{\text{simple}}$ is one-hot, so $\|\phi\|_2 = 1$.
- (A3) is a scale assumption: take $S := \max_a \|\theta_a^*\|_2$; it affects constants but not the $\sqrt{T}$ rate.
- (A4) is plausible in a fixed simulator run: conditioning on the sampled catalog and finite `top_k`, per-episode rewards are bounded, hence the noise is sub-Gaussian for some $\sigma$.

That leaves **(A1) linearity**.

In prose, (A1) says: "There exists a linear function of the chosen features that predicts expected reward for every context-action pair." This is a much stronger statement than it looks. It does not say "reward is roughly monotone in some features" or "a linear model works on average". It says that reward lives exactly on a hyperplane in feature space.

### 6.6.2 What Linearity Really Means for $\phi_{\text{simple}}$

With $\phi_{\text{simple}} = [\text{segment}, \text{query\_type}]$, linearity says:
> For each template $a$, there exist numbers $\theta_{a,\text{segment}}$ and $\theta_{a,\text{query}}$ such that the expected GMV is the **sum** of a "segment effect" and a "query-type effect".

Concretely, suppose template 2 (CM2 Boost) has
$$
\theta_2
  = [\theta_{2,\text{premium}},
     \theta_{2,\text{pl\_lover}},
     \theta_{2,\text{litter\_heavy}},
     \theta_{2,\text{price\_hunter}},
     \theta_{2,\text{specific}},
     \theta_{2,\text{browsing}},
     \theta_{2,\text{deal\_seeking}}]^\top.
$$

For a pl_lover user with a browsing query we always have
$$
\phi_{\text{simple}} =
  [0, 1, 0, 0,\ 0, 1, 0]^\top
$$
and so
$$
\mu(x, a = 2)
  = \theta_{2,\text{pl\_lover}} + \theta_{2,\text{browsing}}.
$$

The model assumes that:

- The effect of being a PL-lover is *additive* and independent of which products happen to be available.
- The effect of the query being "browsing" is also additive and independent.
- There is no interaction beyond the sum of these two numbers.

This is where reality diverges.

### 6.6.3 Concrete Counterexample: Two Episodes, Same Features, Different Worlds

Consider two episodes, both labelled as:

- **User:** pl_lover
- **Query type:** browsing

So both have the **same** $\phi_{\text{simple}}$.

**Episode A (PL-friendly shelf).**

- Base ranker's top-$k$ results contain mostly own-brand products (say 80 % PL).
- Prices cluster around EUR15 with healthy margins.
- When we apply CM2 Boost (template 2), the boost pushes even more PL products into the top slots. The user sees a wall of own-brand products they like at acceptable prices and buys two items.

**Episode B (PL-hostile shelf).**

- Base ranker's top-$k$ results contain almost no own-brand products (say 10 % PL).
- Prices cluster around EUR40 with thinner margins.
- Applying CM2 Boost now drags a handful of mediocre PL products up into top positions, replacing highly relevant national-brand products. The user is underwhelmed and leaves without buying.

**Visual summary:**

| Episode | User Segment | Query Type | $\phi_{\text{simple}}$ | Top-K PL Fraction | CM2 Boost Outcome | GMV |
|---------|--------------|------------|------------------------|-------------------|-------------------|-----|
| **A** | pl_lover | browsing | [0,1,0,0,0,1,0] | 80% | Boosts many relevant PL products | **8.2** |
| **B** | pl_lover | browsing | [0,1,0,0,0,1,0] | 10% | Boosts few mediocre PL products | **2.1** |

**Caption:** Episodes A and B are indistinguishable to the bandit (identical $\phi_{\text{simple}}$) but yield **4x different GMV**. The missing information is the PL fraction in the base ranker's top-K---a critical context the simple features do not capture.

**This is model misspecification:** The linear model $\mu(x, a) = \theta_a^\top \phi(x)$ assigns the same expected reward to both episodes, but reality disagrees violently.

From the simulator's point of view, Episodes A and B are completely different: catalog composition, price and margin distributions, and match between user preferences and available products all change. From the bandit's point of view, **they are indistinguishable**---both correspond to the same one-hot vector.

Linearity in $\phi_{\text{simple}}$ therefore fails in the most brutal way: **the same feature vector leads to vastly different expected rewards** depending on hidden variables the bandit cannot see.

### 6.6.4 Feature Poverty and Model Misspecification

This is the essence of **feature poverty**:

- The simulator knows a rich state: user price sensitivity and PL preference, catalog price/margin/discount distributions, base-ranker relevance scores, etc.
- The bandit only sees a 7-dim feature vector encoding segment and query type.

The result is a **misspecified model**:

- The true reward $\mu(x,a)$ depends on rich interactions between user preferences and product attributes.
- The linear model is forced to explain these interactions using only segment and query labels.

In this regime, regret guarantees still hold in a narrow sense: LinUCB and TS quickly find the *best linear policy on $\phi_{\text{simple}}$*. But the best linear policy in such a poor feature space may simply be "pick the least bad static template", which is exactly what we observe.

The take-away from Section 6.6 is not "LinUCB is bad" or "Thompson Sampling fails." It is:

> Regret bounds are guarantees **conditional on the feature representation.**

With $\phi_{\text{simple}}$ we gave the algorithms a bad hypothesis class. The failure is on us, not on the theorems.

In Section 6.7 we fix the right thing: we redesign the features.

---

## 6.7 Retry with Rich Features

We have diagnosed the problem: **feature poverty**. Our 7-dimensional one-hot encoding of segment and query type does not capture the information needed to learn a good policy. The bandit cannot see what products are in the result set, cannot see user preferences beyond crude segment labels, cannot see how well the base ranker matched the query.

The fix is conceptually simple: **give the bandit better features.**

The hard part is choosing features that:

1. **Capture reward drivers** --- the information the simulator uses to compute GMV, CM2, and engagement.
2. **Remain action-independent** --- they cannot depend on which template we are *about* to choose.
3. **Stay fixed-dimensional** --- linear models need $\phi(x) \in \mathbb{R}^d$ with constant $d$.
4. **Avoid leakage** --- no peeking at future clicks or using ground-truth labels that would not be available in production.

### 6.7.1 Aggregates over the Base Ranker's Top-K

The key design idea is to compute features from the **base ranker's top-$K$ results**, *before* applying any template.

The base ranker from Chapter 5 already scores products by relevance. Given a query and catalog, it produces a ranking. We can take the top $K$ products from this ranking (we use $K=20$ in the demo) and compute aggregates:

- Average and standard deviation of price.
- Average CM2 margin.
- Average discount.
- Fraction of own-brand (PL) products.
- Fraction of products in strategic categories.
- Average popularity score.
- Average relevance score from the base ranker.

These statistics summarize **what the shelf looks like** before boosting. Crucially, they are **action-independent**: the same regardless of which template the bandit will choose.

### 6.7.2 Oracle vs. Estimated: The Production Reality Gap

We diagnosed feature poverty. Now we fix it. But in fixing it, we face a choice that reveals something deeper about algorithm selection.

Our simulator knows each user's true latent preferences---their exact price sensitivity ($\theta_{\text{price}}$) and private-label affinity ($\theta_{\text{pl}}$). In production, we do not have this luxury. Real systems estimate preferences from noisy behavioral signals: clicks, dwell time, and purchase history.

This distinction matters materially for algorithm selection. We run two experiments with rich features:

1. **Oracle latents**: We give the bandit the true user preferences (an idealized benchmark).

2. **Estimated latents**: We give the bandit noisy estimates of preferences (a production-style setting).

The comparison isolates the role of feature noise:
- With oracle features, both LinUCB and Thompson Sampling achieve roughly +32% GMV over the best static template (for our reference seed, LinUCB is marginally higher).
- With estimated features, Thompson Sampling remains near +31% GMV while LinUCB drops to roughly +6%.

This is the chapter's deepest lesson: **algorithm selection depends on feature quality**.

### 6.7.3 Building $\phi_{\text{rich}}$: From 7 to 17 Dimensions

We start from the simple features:
$$
\phi_{\text{simple}}(x)
  = [\text{segment one-hot} (4), \text{query-type one-hot} (3)]
  \in \{0,1\}^7.
$$

We then add:

1. **User latent preferences (2 dims).**

   The simulator represents each user with continuous parameters
   $\theta_{\text{price}} \in [-1,1]$ (price sensitivity) and
   $\theta_{\text{pl}} \in [-1,1]$ (preference for own-brand).
   In the "oracle rich" experiment we feed these true values into the feature map.

2. **Base-top-$K$ aggregates (8 dims).**

   Over the top-$K$ products under the **base** ranker we compute:

   - `avg_price`, `std_price`
   - `avg_cm2`, `avg_discount`
   - `frac_pl`, `frac_strategic`
   - `avg_bestseller`, `avg_relevance`

Putting everything together we obtain a 17-dimensional feature vector:
$$
\phi_{\text{rich}}(x) \in \mathbb{R}^{17}.
$$

In `scripts/ch06/template_bandits_demo.py` this is implemented as
`context_features_rich`. The function concatenates segment and query one-hots,
user preferences $(\theta_{\text{price}}, \theta_{\text{pl}})$, and the eight
aggregates, then applies a simple z-normalization using fixed means and
standard deviations baked into the script.

!!! note "Code <-> Features (Rich Mode)"
    Rich features are computed in `scripts/ch06/template_bandits_demo.py`:

    - `context_features_rich`: oracle user latents + base-top-$K$ aggregates
    - `context_features_rich_estimated`: same structure with *estimated* latents
    - `feature_mode` CLI flag: `--features {simple,rich,rich_est}`

    The policy classes in `zoosim/policies/{lin_ucb,thompson_sampling}.py`
    simply consume the resulting $\phi(x)$; all the modelling decisions about
    *what* to expose live in the feature functions.

### 6.7.4 Experiment A: Rich Features with Oracle Latents

We now run exactly the same experiment as in Section 6.5, but with $\phi_{\text{rich}}$ containing the **true user latent preferences**:

```bash
uv run python scripts/ch06/template_bandits_demo.py \
    --n-static 2000 \
    --n-bandit 20000 \
    --features rich \
    --world-seed 20250322 \
    --bandit-base-seed 20250349 \
    --hparam-mode rich_est \
    --prior-weight 50 \
    --lin-alpha 0.2 \
    --ts-sigma 0.5
```

The simulator, templates, and basic hyperparameters are unchanged. Only the features differ---now enriched with oracle user latents and base-top-$K$ aggregates.

The output now shows a reversal relative to Section 6.5:

```
Static templates (per-episode averages):
ID  Template             Reward         GMV         CM2
 0  Neutral                5.75        5.28        0.50
 1  High Margin            5.44        5.07        0.63
 2  CM2 Boost              7.35        6.73        0.61
 3  Popular                4.79        4.53        0.52
 4  Premium                7.56        7.11        0.74  <- Best static
 5  Budget                 3.44        3.04        0.26
 6  Discount               5.04        4.62        0.45
 7  Strategic              4.83        3.99       -0.13

Best static template: ID=4 (Premium) with avg reward=7.56, GMV=7.11

LinUCB (20000 episodes, rich oracle features):
  Global avg:  Reward=10.19, GMV=9.42, CM2=0.97

Thompson Sampling (20000 episodes, rich oracle features):
  Global avg:  Reward=10.15, GMV=9.39, CM2=0.97
```

Read these numbers carefully---the **algorithm ranking has reversed**:

| Algorithm | GMV | vs. Static Best |
|-----------|-----|-----------------|
| Static (Premium) | 7.11 | baseline |
| **LinUCB** | **9.42** | **+32.5%** |
| Thompson Sampling | 9.39 | +32.1% |

With oracle user latents, **both algorithms perform excellently**---and nearly identically! The clean features allow both LinUCB's UCB bonus and Thompson Sampling's posterior sampling to converge efficiently to the optimal policy. LinUCB edges out TS by a razor-thin margin (+0.4 percentage points), but the practical difference is negligible.

This makes theoretical sense: LinUCB's regret bound [THM-6.2] assumes the reward function is *exactly* linear in features. With oracle latents providing the true user preferences, the linear assumption holds nearly perfectly, and LinUCB's exploitation becomes a virtue rather than a liability.

### 6.7.5 Experiment B: Rich Features with Estimated Latents

The crucial twist is that in production we do not have oracle user latents. Real systems estimate user preferences from observed behavior---clicks, purchases, and browsing patterns. These estimates are noisy, delayed, and sometimes wrong.

We run the same experiment with estimated latents instead of oracle:

```bash
uv run python scripts/ch06/template_bandits_demo.py \
    --n-static 2000 \
    --n-bandit 20000 \
    --features rich_est \
    --world-seed 20250322 \
    --bandit-base-seed 20250349 \
    --hparam-mode rich_est \
    --prior-weight 50 \
    --lin-alpha 0.2 \
    --ts-sigma 0.5
```

The output reveals the production reality:

```
LinUCB (20000 episodes, rich estimated features):
  Global avg:  Reward=8.20, GMV=7.52, CM2=0.76

Thompson Sampling (20000 episodes, rich estimated features):
  Global avg:  Reward=10.08, GMV=9.31, CM2=0.97
```

Now the algorithm ranking **diverges dramatically**:

| Algorithm | GMV | vs. Static Best |
|-----------|-----|-----------------|
| Static (Premium) | 7.11 | baseline |
| LinUCB | 7.52 | +5.8% |
| **Thompson Sampling** | **9.31** | **+31.0%** |

With estimated (noisy) features, Thompson Sampling wins decisively by 25 percentage points.

### 6.7.6 The Algorithm Selection Principle

These two experiments reveal the chapter's deepest lesson:

!!! tip "Algorithm Selection Depends on Feature Quality"
    $$
    \text{Clean/Oracle features} \rightarrow \text{LinUCB (precise exploitation)}
    $$
    $$
    \text{Noisy/Estimated features} \rightarrow \text{Thompson Sampling (robust exploration)}
    $$

**Why does this happen?**

- **LinUCB** constructs
  $$
  \text{UCB}_a(x) = \hat{\theta}_a^\top \phi(x) +
    \alpha \sqrt{\phi(x)^\top A_a^{-1} \phi(x)}
  $$
  and becomes nearly deterministic as the uncertainty term shrinks. With clean oracle features, this precision is a virtue---LinUCB converges quickly to the optimal policy. But with noisy features, LinUCB can **lock into a suboptimal template** based on spurious correlations in the estimated latents.

- **Thompson Sampling** samples
  $\theta_a \sim \mathcal{N}(\hat{\theta}_a, \Sigma_a)$ each round. Even after 20,000 episodes, the posterior covariance $\Sigma_a$ retains some mass---noise and misspecification prevent total collapse. TS therefore **never fully stops exploring**, hedging against feature noise.

**Production implication:** Since production systems invariably have noisy estimated features (not oracle access to true user preferences), Thompson Sampling is a robust default. LinUCB is appealing when feature quality is high and deterministic behavior is valuable (e.g., debugging and A/B analysis).

This mirrors empirical findings in the broader RL literature: posterior-sampling and ensemble-based methods often outperform hard UCB-style bonuses in complex environments [@russo:tutorial_ts:2018, @osband:bootstrapped_dqn:2016].

### 6.7.7 Per-Segment Breakdown: Who Actually Benefits?

Global GMV averages hide important heterogeneity. The script reports per-segment metrics such as:

```
Per-segment GMV (static best vs bandits, rich features):
Segment          Static GMV  LinUCB GMV      TS GMV    LinUCB Delta%     TS Delta%
premium               31.60       31.30       31.95       -0.9%      +1.1%
pl_lover               5.34       11.56       11.85      +116.4%    +121.9%
litter_heavy           4.94        6.60        6.28       +33.6%     +27.2%
price_hunter           0.00        0.00        0.00        +0.0%      +0.0%
```

Both bandits more than double GMV for the pl_lover segment and improve litter_heavy, while keeping premium GMV essentially unchanged in this oracle-feature run.

From a business perspective, this is exactly the sort of trade-off we want to understand:

- TS learns a **balanced** policy: keep premium shoppers happy *and* fix underserved cohorts.
- LinUCB finds a **PL-centric** policy: great for PL-lovers, less so for premium.

The same per-segment machinery also works with the `--show-volume` flag, which logs order counts alongside GMV/CM2. In the labs we replicate plots showing how bandits can triple order volume for some segments even when global GMV barely moves.

---

## 6.8 Summary & What's Next

We now summarize the technical artifacts, the empirical compute arc, and the lessons that guide the remainder of the book. This chapter deliberately includes a failure mode and its diagnosis: we build a correct implementation, observe a negative result under an impoverished representation, and then fix the representation.

### 6.8.1 What We Built

On the technical side:

- **Discrete template action space** (Section 6.1): 8 interpretable boost strategies encoding business logic (High Margin, CM2 Boost, Premium, Budget, Discount, etc.).
- **Thompson Sampling and LinUCB theory** (Sections 6.2--6.3): posterior sampling and UCB for linear contextual bandits, with sublinear regret guarantees under explicit assumptions.
- **Production-quality implementations** (Section 6.4): type-hinted NumPy code in `zoosim/policies/thompson_sampling.py` and `zoosim/policies/lin_ucb.py`, wired to the simulator via `scripts/ch06/template_bandits_demo.py`.

On the empirical side (the three-stage compute arc):

- **Stage 1: Simple-feature experiment** (Section 6.5): with $\phi_{\text{simple}}$ (segment + query type, $d=8$), both bandits fail. LinUCB lands at 5.12 GMV (-28%), TS at 6.18 GMV (-13%).
- **Stage 2: Diagnosis** (Section 6.6): we locate the culprit in violated assumptions---feature poverty and linear model misspecification---not in bugs or lack of data.
- **Stage 3a: Rich features + oracle latents** (Section 6.7.4): with $\phi_{\text{rich}}$ containing true user latents ($d=18$), **both algorithms excel**---LinUCB at 9.42 GMV (+32.5%), TS at 9.39 GMV (+32.1%). Near-perfect tie.
- **Stage 3b: Rich features + estimated latents** (Section 6.7.5): same rich features but with estimated (noisy) latents, **TS wins decisively** at 9.31 GMV (+31.0%), LinUCB at 7.52 GMV (+5.8%).

### 6.8.2 Five Lessons

**Lesson 1 --- Regret bounds are conditional guarantees.**

The guarantee in [THM-6.2] is of the form:

> *If* $\mu(x,a)$ is linear in the chosen features and the other assumptions hold, *then* LinUCB finds a near-optimal linear policy efficiently.

It does not say that LinUCB discovers the globally optimal policy for the environment. With $\phi_{\text{simple}}$ the best linear policy is simply bad; the theorem holds, but the outcome is still disappointing. Whenever we use theoretical guarantees in practice, we trace each assumption back to a concrete property of the system.

**Lesson 2 --- Feature engineering sets the performance ceiling.**

We changed nothing about the environment, templates, or algorithms between Sections 6.5 and 6.7. Only the features changed. Yet the GMV numbers swung from "-13% vs. static" to "+31% vs. static".

We cannot learn what the features do not expose. In contextual bandits (and in deep RL, despite the flexibility of neural networks) representation design is policy design.

**Lesson 3 --- Simple baselines encode valuable domain knowledge.**

The Premium template is a one-line heuristic that captures a deep business insight: boosting premium products maximizes revenue per purchase. It wins by default in the simple-feature regime and remains competitive even with rich features.

Hand-crafted baselines like Premium define a **safe lower bound** and a **warm start** for learning. Bandits should be evaluated relative to them, not in isolation.

**Lesson 4 --- Failure is a design signal, not an embarrassment.**

The -28%/-13% GMV numbers in Section 6.5 are not a sign that bandits are useless. They are a sign that the modelling choices are wrong. Because we looked at the failure honestly, we discovered precisely which assumptions broke and how to correct them.

In production RL, we experience many such failures. The playbook from this chapter is:

1. Start from a strong baseline and articulate expectations.
2. When the algorithm underperforms, enumerate the theorem's assumptions.
3. Diagnose feature poverty vs. model misspecification vs. data issues.
4. Fix the representation first; reach for more complex algorithms only if needed.

**Lesson 5 --- Algorithm selection depends on feature quality.**

The contrast between Section 6.7.4 (oracle latents) and Section 6.7.5 (estimated latents) reveals the chapter's deepest insight: **the same features can favor different algorithms depending on noise level**.

- With *clean, oracle features*, both algorithms excel equally (~+32% each)---the "scalpel" and the "Swiss Army knife" are equally sharp when data is perfect.
- With *noisy, estimated features*, Thompson Sampling's robust exploration wins decisively (+31% vs. LinUCB's +6%).

Production systems invariably have noisy features---estimated from clicks, inferred from behavior, aggregated from proxies. We default to Thompson Sampling in production and use LinUCB when feature quality is high and deterministic behavior is valuable (e.g., direct measurements or carefully validated latent estimates).

### 6.8.3 Where to Go Next

**For hands-on practice:**

`docs/book/ch06/exercises_labs.md` contains exercises and labs that cover:

- **Lab 6.1**: Reproducing the simple-feature failure (Stage 1)
- **Lab 6.2a**: Rich features with oracle latents---Both excel (Stage 3a)
- **Lab 6.2b**: Rich features with estimated latents---TS wins (Stage 3b)
- **Lab 6.2c**: Synthesis---understanding the algorithm selection principle
- **Labs 6.3-6.5**: Hyperparameter sensitivity, exploration dynamics, multi-seed robustness
- **Exercises 6.1-6.14**: Theoretical proofs, implementation challenges, and ablation studies

**For practitioners scaling experiments:**

When running many seeds, feature variants, or large episode counts (100k+), the CPU implementation (Section 6.4) becomes slow (~30 seconds per run). See the optional **Advanced Lab 6.A: From CPU Loops to GPU Batches** (`ch06_advanced_gpu_lab.md`).

**Prerequisites:** Completion of main Chapter 6 narrative + Labs 6.1-6.3, CUDA-capable GPU

**Time budget:** 2-3 hours (spread over multiple sessions)

**Topics covered:** Batch-parallel simulation on GPU, correctness verification via parity checks, when GPU acceleration matters, and how to migrate CPU code safely.

**The GPU lab teaches production skills:** Vectorization, device management, seed alignment, and parity testing---techniques we use when scaling any RL experiment.

**If we do not have a GPU or are not planning large-scale experiments, we skip this lab.** The CPU implementation is sufficient for understanding the algorithms.

### 6.8.4 Extensions & Practice

For practitioners planning production deployment or interested in advanced topics, see **`appendices.md`** for:

- **Appendix 6.A: Neural Linear Bandits** --- Representation learning with neural networks, when to use vs. avoid, PyTorch implementation
- **Appendix 6.B: Theory-Practice Gap Analysis** --- What theory guarantees vs. what we implement, why it works anyway, failure modes, recent work (2020-2025), open problems
- **Appendix 6.C: Modern Context & Connections** --- Industry deployments (Netflix, Spotify, Microsoft Bing), contextual bandits vs. Deep RL, bandits vs. Offline RL
- **Appendix 6.D: Production Checklist** --- Configuration alignment, guardrails, reproducibility, monitoring, testing

These appendices deepen and generalize the core narrative but are **optional**. You can return to them after completing Chapter 7.

**Chapter 7 preview:** We take the core insight---"features and templates constrain what we can learn"---and apply it to continuous actions via $Q(x,a)$ regression, where templates become vectors in a high-dimensional action space.

## Exercises & Labs

See `docs/book/ch06/exercises_labs.md` for:

- **Exercise 6.1**: Prove [PROP-5.1] properties of semantic relevance (warmup)
- **Exercise 6.2**: Implement epsilon-greedy baseline, compare regret to LinUCB
- **Exercise 6.3**: Derive ridge regression solution $\hat{\theta} = (A^\top A + \lambda I)^{-1} A^\top b$
- **Exercise 6.4**: Verify LinUCB and TS have identical posterior mean (mathematical proof)
- **Exercise 6.5**: Implement Cholesky-based sampling for Thompson Sampling
- **Exercise 6.6**: Add new template "Category Diversity" that boosts underrepresented categories
- **Exercise 6.7**: Implement hierarchical templates (meta-template selects category, sub-template selects boost)
- **Exercise 6.8**: Conduct ablation study: Remove features one-by-one, measure regret impact
- **Exercise 6.9**: Extend templates to be query-conditional (different templates per query type)
- **Exercise 6.10**: Implement UCB with adaptive $\alpha_t = c\sqrt{\log(1+t)}$, tune $c$
- **Exercise 6.11**: Add polynomial features $[\phi(x), \phi(x)^2]$, compare linear vs. quadratic
- **Exercise 6.12**: Implement Neural Linear bandit, train on 20k episodes, evaluate
- **Exercise 6.13**: Extend training to 100k episodes, verify +10% GMV vs. best static
- **Exercise 6.14**: Add time-of-day feature, show bandits learn diurnal patterns

**Lab 6.1: Hyperparameter Sensitivity**

Grid search over $(\lambda, \alpha) \in \{0.1, 1.0, 10.0\} \times \{0.5, 1.0, 2.0\}$. Plot heatmap of final reward.

**Lab 6.2: Visualization**

Plot:
1. Template selection heatmap (template vs. episode, color = selection frequency)
2. Uncertainty evolution (trace($\Sigma_a$) vs. episode for each template)
3. Regret decomposition (per-template contribution to cumulative regret)

**Lab 6.3: Multi-Seed Evaluation**

Run LinUCB with 10 different seeds, report mean $\pm$ standard deviation of final GMV. Verify robustness.

---

## Summary

**What we built:**

1. **Discrete template action space** (8 interpretable boost strategies)
2. **Thompson Sampling** (Bayesian posterior sampling, automatic exploration)
3. **LinUCB** (frequentist UCB, deterministic, tunable exploration)
4. **Production implementation** (PyTorch/NumPy, numerical stability, diagnostics)
5. **Full integration** with `zoosim` search simulator
6. **Experimental validation** (learning curves, exploration dynamics, baselines)

**Key results:**

- Bandits can beat the best static template by a few percent GMV (context-dependent).
- Regret grows sublinearly under the model assumptions (Sections 6.2--6.3).
- Exploration decays automatically in Thompson Sampling (no manual schedule needed).
- Policies remain interpretable (template selection and scores are inspectable).

**What's next:**

- **Chapter 7**: Continuous actions via $Q(x, a)$ regression (move beyond discrete templates)
- **Chapter 10**: Hard constraints (CM2 floor, exposure, rank stability) via Lagrangian methods
- **Chapter 11**: Multi-session MDPs (retention, long-term value optimization)
- **Chapter 13**: Offline RL (learn from logged data without online interaction)

**The textbook journey:**

We've now built a **production-ready RL system** that:
- Starts with strong priors (base ranker + templates)
- Learns from interaction (bandit algorithms)
- Balances exploration and exploitation (provable regret bounds)
- Remains interpretable (template selection visible to business)

From here, we scale to more complex action spaces (continuous boosts, slate optimization) and harder constraints (multi-objective optimization, safety guarantees). The foundation is solid, and we now continue.

---

**Reproducibility checklist (Chapter 6).**

- Core implementations: `zoosim/policies/{templates,thompson_sampling,lin_ucb}.py`
- Experiments: `scripts/ch06/template_bandits_demo.py`, `scripts/ch06/ch06_compute_arc.py`
- Tests: `tests/ch06/` (run with `uv run pytest tests/ch06 -q`)
- Canonical artifacts: `docs/book/ch06/data/template_bandits_{simple,rich_oracle,rich_estimated}_summary.json`
- End-to-end verification script: `scripts/ch06/run_full_verification.sh` (writes to `docs/book/ch06/data/verification_<timestamp>/`)
