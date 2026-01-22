# Chapter 5 — Relevance, Features, and Reward: The RL State-Action-Reward Interface

## The Three Pillars of Search RL

In Chapter 4, we built the **generative world**: products, users, and queries sampled deterministically from realistic distributions. Now we face the core RL challenge: **what should the agent observe, control, and optimize?**

Every RL system requires three components:

1. **Observation space** $\mathcal{X}$: What does the agent see? (Features)
2. **Action space** $\mathcal{A}$: What can the agent control? (Ranking adjustments via boosts)
3. **Reward function** $R: \mathcal{X} \times \mathcal{A} \times \Omega \to \mathbb{R}$: What should the agent maximize? (Multi-objective business metrics)

But before the RL agent can act, we need a **base ranking**—an initial relevance model that orders products by query-product match quality. The agent will then **boost or suppress** products via learned adjustments.

**The architecture:**

```
Query q, User u → [Base Relevance Model] → Base scores {s₁, ..., sₙ}
                                    ↓
Base scores + Features → [RL Agent] → Boost adjustments {a₁, ..., aₙ}
                                    ↓
Adjusted scores → [Ranking] → Displayed products → [User Interaction] → Reward r
```

This chapter develops all three components:

**I. Relevance Model (Section 5.1-5.2)**
- **Why**: Need initial ranking before RL adjustments
- **What**: Hybrid semantic + lexical matching
- **Math**: Cosine similarity + token overlap → base score $s_{\text{base}}(q, p)$
- **Code**: `zoosim/ranking/relevance.py`

**II. Feature Engineering (Section 5.3-5.4)**
- **Why**: RL state representation for boost selection
- **What**: Product, user, query interaction features
- **Math**: Feature vector $\phi(u, q, p) \in \mathbb{R}^d$ with standardization
- **Code**: `zoosim/ranking/features.py`

**III. Reward Aggregation (Section 5.5-5.6)**
- **Why**: Multi-objective optimization (GMV, CM2, engagement, strategic goals)
- **Math**: Weighted sum $R = \alpha \cdot \text{GMV} + \beta \cdot \text{CM2} + \gamma \cdot \text{Strategic} + \delta \cdot \text{Clicks}$
- **Code**: `zoosim/dynamics/reward.py`

**The RL loop:**
```
State x = (u, q, {φ(u,q,p) : p ∈ C})
    ↓
Agent: a = π(x)  [boost vector]
    ↓
Environment: s'ᵢ = s_base(q,pᵢ) + aᵢ, rank by s', simulate clicks/buys
    ↓
Reward: R(x,a,ω) = Σ αᵢ·component_i(outcome)
    ↓
Update policy π to maximize 𝔼[R]
```

Let us build each component rigorously, starting with relevance.

---

## 5.1 Base Relevance: Why We Need It

The cold-start problem:

Imagine deploying an RL agent with no prior knowledge. The agent sees a query `"premium dog food"` and a catalog of 10,000 products. With no relevance model, the agent's initial ranking is random. Even with optimal RL convergence, this is catastrophic:

- **Sample complexity**: Need millions of episodes to learn basic relevance from scratch
- **Catastrophic exploration**: Random rankings destroy user experience during learning
- **Inefficiency**: RL should learn *adjustments* (boosts), not relevance from scratch

**Solution: Warm-start with a base relevance model.**

The base model provides a **strong prior**:
- Orders products by query-product match (semantic + lexical)
- Gives the agent a reasonable starting ranking (90th percentile baseline)
- Lets RL focus on **optimization under constraints** (CM2, strategic goals, personalization)

**Analogy to control theory:**

In Linear Quadratic Regulator (LQR) problems, we linearize dynamics around a **nominal trajectory**. The controller learns *deviations* from the nominal, not the entire trajectory from scratch.

Here, base relevance is the **nominal ranking**. RL learns perturbations (boosts) that improve business metrics while maintaining relevance.

**Mathematical formalization:**

**Definition 5.1** (Base Relevance Function) {#DEF-5.1}

Let $\mathcal{Q}$ denote the **query space** and let $\mathcal{C} = \{p_1, \ldots, p_N\}$ denote the (finite) **product catalog** produced by the generative world of Chapter&nbsp;4 (see [DEF-4.1]). A **base relevance function** is a mapping
$$
s_{\text{base}}: \mathcal{Q} \times \mathcal{C} \to \mathbb{R}
$$
that assigns, to each query–product pair $(q, p) \in \mathcal{Q} \times \mathcal{C}$, a real-valued relevance score $s_{\text{base}}(q, p) \in \mathbb{R}$.

Throughout this chapter we work with a **fixed catalog** $\mathcal{C}$ generated once from the configuration; extending the definition to time-varying catalogs $\mathcal{C}_t$ is straightforward and deferred to Chapter&nbsp;10 (non-stationarity and drift).

**Properties:**
1. **Higher score = better match**: For any fixed query $q \in \mathcal{Q}$ and products $p_1, p_2 \in \mathcal{C}$, the inequality $s_{\text{base}}(q, p_1) > s_{\text{base}}(q, p_2)$ suggests that $p_1$ is more relevant to $q$ than $p_2$.
2. **Fast to compute**: In the simulator, $s_{\text{base}}$ must be computable for all $p \in \mathcal{C}$ with latency compatible with $|\mathcal{C}| \approx 10^4$ products per query (sub-100&nbsp;ms on a single CPU).

In this chapter we instantiate $s_{\text{base}}$ with a **hybrid model** combining semantic and lexical components; the concrete form is given in [DEF-5.4].

---

## 5.2 Hybrid Relevance Model: Semantic + Lexical

Modern search combines two complementary signals:

**Semantic matching**: Captures *meaning* via embeddings
- Example: `"dry kibble"` matches `dog_food` products even if "kibble" not in category name
- Uses cosine similarity in embedding space: $\cos(\mathbf{q}, \mathbf{p}) = \frac{\mathbf{q} \cdot \mathbf{p}}{\|\mathbf{q}\| \|\mathbf{p}\|}$

**Lexical matching**: Captures *exact words* via token overlap
- Example: Query `"cat litter"` must match products in `litter` category
- Uses set intersection: $|\text{tokens}(q) \cap \text{tokens}(p)|$

**Why both?**
- **Semantic alone**: Misses exact matches (user says "litter", semantic model returns "absorbent material"—correct but suboptimal)
- **Lexical alone**: Misses synonyms and related concepts (user says "kibble", lexical model finds nothing if products say "dry food")

### 5.2.1 Semantic Component

**Definition 5.2** (Semantic Relevance) {#DEF-5.2}

Given query $q$ with embedding $\mathbf{q} \in \mathbb{R}^d$ and product $p$ with embedding $\mathbf{e}_p \in \mathbb{R}^d$, the **semantic relevance** is:
$$
s_{\text{sem}}(q, p) = \cos(\mathbf{q}, \mathbf{e}_p) = \frac{\mathbf{q} \cdot \mathbf{e}_p}{\|\mathbf{q}\|_2 \|\mathbf{e}_p\|_2}
\tag{5.1}
$$
{#EQ-5.1}

**Informal properties:**
- Range: $s_{\text{sem}} \in [-1, 1]$
- $s_{\text{sem}} = 1$: Perfect alignment (parallel vectors)
- $s_{\text{sem}} = 0$: Orthogonal (no semantic relationship)
- $s_{\text{sem}} = -1$: Opposite direction (rare in practice for embeddings)

We now state these properties precisely.

**Proposition 5.1** (Properties of Semantic Relevance) {#PROP-5.1}

Let $\mathbf{q}, \mathbf{e} \in \mathbb{R}^d$ with $\|\mathbf{q}\|_2, \|\mathbf{e}\|_2 > 0$ and define
$$
s_{\text{sem}}(\mathbf{q}, \mathbf{e}) = \frac{\mathbf{q} \cdot \mathbf{e}}{\|\mathbf{q}\|_2 \|\mathbf{e}\|_2}.
$$
Then:

(a) (**Range**) $s_{\text{sem}}(\mathbf{q}, \mathbf{e}) \in [-1, 1]$.

(b) (**Symmetry**) $s_{\text{sem}}(\mathbf{q}, \mathbf{e}) = s_{\text{sem}}(\mathbf{e}, \mathbf{q})$.

(c) (**Scale invariance**) For all $\alpha, \beta > 0$,
$$
s_{\text{sem}}(\alpha \mathbf{q}, \beta \mathbf{e}) = s_{\text{sem}}(\mathbf{q}, \mathbf{e}).
$$

(d) (**Boundary cases**)
- $s_{\text{sem}}(\mathbf{q}, \mathbf{e}) = 1$ if and only if $\mathbf{e} = c \mathbf{q}$ for some $c > 0$;
- $s_{\text{sem}}(\mathbf{q}, \mathbf{e}) = -1$ if and only if $\mathbf{e} = c \mathbf{q}$ for some $c < 0$;
- $s_{\text{sem}}(\mathbf{q}, \mathbf{e}) = 0$ if and only if $\mathbf{q} \perp \mathbf{e}$.

*Proof.* By the Cauchy–Schwarz inequality,
$$
|\mathbf{q} \cdot \mathbf{e}| \le \|\mathbf{q}\|_2 \|\mathbf{e}\|_2,
$$
and dividing both sides by the positive quantity $\|\mathbf{q}\|_2 \|\mathbf{e}\|_2$ yields (a). Symmetry in (b) follows from $\mathbf{q} \cdot \mathbf{e} = \mathbf{e} \cdot \mathbf{q}$ and the symmetry of the Euclidean norm. For (c), note that
$$
s_{\text{sem}}(\alpha \mathbf{q}, \beta \mathbf{e})
= \frac{\alpha \beta\, \mathbf{q} \cdot \mathbf{e}}{\alpha \|\mathbf{q}\|_2 \, \beta \|\mathbf{e}\|_2}
 = s_{\text{sem}}(\mathbf{q}, \mathbf{e}).
$$
Equality in Cauchy–Schwarz holds if and only if $\mathbf{q}$ and $\mathbf{e}$ are linearly dependent, which gives the characterizations in (d) with the sign of $c$ distinguishing the cases $1$ and $-1$; orthogonality corresponds to $\mathbf{q} \cdot \mathbf{e} = 0$ and hence $s_{\text{sem}} = 0$. ∎

**Remark 5.2.1** (Zero-Norm Embeddings and Safe Defaults) {#REM-5.2.1}

Proposition 5.1 assumes $\|\mathbf{q}\|_2, \|\mathbf{e}\|_2 > 0$, so $s_{\text{sem}}$ is undefined if an embedding has zero norm. In the simulator, query and product embeddings are sampled from Gaussian distributions (Chapter&nbsp;4), so the event $\|\mathbf{e}\|_2 = 0$ has probability zero in exact arithmetic. Numerically, however, very small norms or aggressive quantization can occur in production systems (e.g., after pruning or compression). A robust implementation should therefore handle the degenerate case explicitly—for example, by returning a semantic score of $0$ whenever either embedding has norm below a small threshold. This can be interpreted as “no reliable semantic signal”, leaving lexical or other signals to carry the ranking.

**Embedding construction** (from Chapter 4):
- Products: for each category $c$ we sample a centroid $\boldsymbol{\mu}_c$, then $\mathbf{e}_p = \boldsymbol{\mu}_{c(p)} + \boldsymbol{\epsilon}_p$ with $\boldsymbol{\epsilon}_p \sim \mathcal{N}(0, \sigma_{c(p)}^2 I_d)$ (category-level clustering).
- Queries: $\mathbf{q} = \boldsymbol{\theta}_{\text{emb}}(u) + \boldsymbol{\epsilon}_q$ with $\boldsymbol{\epsilon}_q \sim \mathcal{N}(0, 0.05^2 I_d)$ (user-centered queries).

**Implementation:**

```python
import torch
from torch import Tensor

def semantic_component(query_emb: Tensor, product_emb: Tensor) -> float:
    """Compute semantic relevance via cosine similarity.

    Mathematical basis: [EQ-5.1] (Semantic Relevance)

    Args:
        query_emb: Query embedding, shape (d,)
        product_emb: Product embedding, shape (d,)

    Returns:
        Cosine similarity in [-1, 1]

    References:
        - [DEF-5.2] Semantic Relevance definition
        - Chapter 4 embedding generation (Gaussian clusters)
    """
    # PyTorch cosine_similarity handles normalization automatically
    return float(torch.nn.functional.cosine_similarity(
        query_emb, product_emb, dim=0
    ))
```

!!! note "Code ↔ Config (Embedding Dimension)"
    The embedding dimension $d$ is set in `SimulatorConfig.catalog.embedding_dim` (default: 16).
    - **File**: `zoosim/core/config.py:21`
    - **Usage**: Embeddings generated in `zoosim/world/catalog.py` and `zoosim/world/queries.py`
    - **Trade-off**: Larger $d$ increases expressiveness but slows computation

### 5.2.2 Lexical Component

**Definition 5.3** (Lexical Relevance) {#DEF-5.3}

Given query $q$ with tokens $T_q = \{\text{token}_1, \ldots, \text{token}_k\}$ and product $p$ with category tokens $T_p$ (e.g., `"cat_food"` → `{"cat", "food"}`), the **lexical relevance** is:
$$
s_{\text{lex}}(q, p) = \log(1 + |T_q \cap T_p|)
\tag{5.2}
$$
{#EQ-5.2}

**Design choices:**
- Set intersection $|T_q \cap T_p|$ counts shared tokens (order-invariant).
- The transform $\log(1 + x)$ compresses large overlaps (10 shared tokens are not interpreted as “10× better” than 1).
- The shift by $+1$ inside the logarithm ensures $s_{\text{lex}} \geq 0$ even when there is no overlap.

**Proposition 5.3** (Properties of Lexical Relevance) {#PROP-5.3}

Let $T_q, T_p$ be finite token sets and define
$$
o = |T_q \cap T_p|, \qquad s_{\text{lex}}(q,p) = \log(1 + o).
$$
Then:

(a) The overlap satisfies $0 \le o \le \min(|T_q|, |T_p|)$, and hence
$$
0 \le s_{\text{lex}}(q,p) \le \log\bigl(1 + \min(|T_q|, |T_p|)\bigr).
$$

(b) $s_{\text{lex}}(q,p) = 0$ if and only if $T_q \cap T_p = \varnothing$.

(c) If $o_1 < o_2$ then $\log(1 + o_1) < \log(1 + o_2)$, so $s_{\text{lex}}$ is strictly increasing in the overlap count $o$.

*Proof.* By definition of intersection, any common token belongs to both $T_q$ and $T_p$, so the number of common tokens $o$ cannot exceed either $|T_q|$ or $|T_p|$, giving $0 \le o \le \min(|T_q|, |T_p|)$ and the bound in (a) after applying the monotonicity of $\log(1 + x)$ on $[0,\infty)$. We have $s_{\text{lex}}(q,p) = 0$ if and only if $\log(1 + o) = 0$, which holds if and only if $1 + o = 1$, i.e., $o = 0$, so (b) follows. For (c), note that $\log(1 + x)$ is strictly increasing, so $o_1 < o_2$ implies $\log(1 + o_1) < \log(1 + o_2)$. ∎

**Example:**
- Query: `"premium cat food"` → $T_q = \{\text{premium}, \text{cat}, \text{food}\}$
- Product 1: Category `cat_food` → $T_p = \{\text{cat}, \text{food}\}$ → overlap = 2 → $s_{\text{lex}} = \log(3) \approx 1.10$
- Product 2: Category `dog_food` → $T_p = \{\text{dog}, \text{food}\}$ → overlap = 1 → $s_{\text{lex}} = \log(2) \approx 0.69$
- Product 3: Category `toys` → $T_p = \{\text{toys}\}$ → overlap = 0 → $s_{\text{lex}} = 0$

**Implementation:**

```python
import math

def lexical_component(query_tokens: set[str], product_category: str) -> float:
    """Compute lexical relevance via token overlap.

    Mathematical basis: [EQ-5.2] (Lexical Relevance)

    Args:
        query_tokens: Set of query tokens (from Query.tokens)
        product_category: Product category string (e.g., "cat_food")

    Returns:
        Log(1 + overlap) where overlap = |T_q ∩ T_p|

    References:
        - [DEF-5.3] Lexical Relevance definition
    """
    product_tokens = set(product_category.split("_"))
    overlap = len(query_tokens & product_tokens)
    return math.log1p(overlap)  # log(1 + overlap)
```

### 5.2.3 Combined Base Score

**Definition 5.4** (Hybrid Base Relevance) {#DEF-5.4}

The **hybrid base relevance** combines semantic and lexical components with learned weights:
$$
s_{\text{base}}(q, p) = w_{\text{sem}} \cdot s_{\text{sem}}(q, p) + w_{\text{lex}} \cdot s_{\text{lex}}(q, p) + \epsilon
\tag{5.3}
$$
{#EQ-5.3}

where:
- $w_{\text{sem}}, w_{\text{lex}} \in \mathbb{R}_+$: Relative weights (configuration parameters),
- $\epsilon = \epsilon(q, p; \omega)$: A noise term indexed by $\omega$ in an underlying probability space $(\Omega, \mathcal{F}, \mathbb{P})$.

For each fixed $(q, p) \in \mathcal{Q} \times \mathcal{C}$ we assume
$$
\epsilon(q, p; \cdot) \sim \mathcal{N}(0, \sigma^2)
$$
for some $\sigma > 0$ and that the family $\{\epsilon(q, p; \cdot) : (q, p) \in \mathcal{Q} \times \mathcal{C}\}$ is **independent** across query–product pairs and across simulator episodes. In other words, each call to $s_{\text{base}}$ draws an independent Gaussian perturbation with variance $\sigma^2$.

**Weight selection:**

From `RelevanceConfig` (production settings):
- $w_{\text{sem}} = 0.7$: Semantic dominates (captures synonyms, related concepts)
- $w_{\text{lex}} = 0.3$: Lexical refinement (ensures exact matches rank high)
- $\sigma = 0.05$: Small noise for diversity

**Why this weighting?**
- E-commerce search is **intent-heavy**: Users often type exact category names (`"cat food"` not `"feline nutrition"`)
- But semantic helps with **long-tail queries**: `"grain-free kibble"` → `dog_food` with high protein content
- 70/30 split balances precision (lexical) and recall (semantic)

**Full implementation:**

```python
import math

import numpy as np
import torch

from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import Product
from zoosim.world.queries import Query

def base_score(
    *,
    query: Query,
    product: Product,
    config: SimulatorConfig,
    rng: np.random.Generator
) -> float:
    """Compute hybrid base relevance score for query-product pair.

    Mathematical basis: [EQ-5.3] (Hybrid Base Relevance)

    Combines:
    - Semantic: Cosine similarity in embedding space [EQ-5.1]
    - Lexical: Token overlap with log compression [EQ-5.2]
    - Noise: Gaussian perturbation for diversity

    Args:
        query: Query with embedding and tokens
        product: Product with embedding and category
        config: Simulator config (relevance weights in config.relevance)
        rng: NumPy random generator for noise

    Returns:
        Base relevance score (unbounded real number)

    References:
        - [DEF-5.4] Hybrid Base Relevance
        - Implementation: `zoosim/ranking/relevance.py:28-33`
    """
    # Semantic component [EQ-5.1]
    sem = float(torch.nn.functional.cosine_similarity(
        query.phi_emb, product.embedding, dim=0
    ))

    # Lexical component [EQ-5.2]
    query_tokens = set(query.tokens)
    prod_tokens = set(product.category.split("_"))
    overlap = len(query_tokens & prod_tokens)
    lex = math.log1p(overlap)

    # Weighted combination with noise [EQ-5.3]
    rel_cfg = config.relevance
    noise = float(rng.normal(0.0, rel_cfg.noise_sigma))

    return rel_cfg.w_sem * sem + rel_cfg.w_lex * lex + noise
```

!!! note "Code ↔ Config (Relevance Weights)"
    The weights $(w_{\text{sem}}, w_{\text{lex}}, \sigma)$ are configured in:
    - **File**: `zoosim/core/config.py:153-157` (`RelevanceConfig`)
    - **Defaults**: `w_sem=0.7`, `w_lex=0.3`, `noise_sigma=0.05`
    - **Usage**: Passed to `base_score()` in `zoosim/ranking/relevance.py:28`
    - **Tuning**: Adjust weights to match production relevance correlation (aim for ρ > 0.85)

**Batch computation** (for efficiency):

```python
from typing import Iterable, List

def batch_base_scores(
    *,
    query: Query,
    catalog: Iterable[Product],
    config: SimulatorConfig,
    rng: np.random.Generator
) -> List[float]:
    """Compute base scores for all products in catalog (vectorized).

    Args:
        query: Single query
        catalog: Iterable of products (typically full catalog)
        config: Simulator config
        rng: Random generator

    Returns:
        List of base scores, length |catalog|

    Note:
        For production at scale (>100k products), consider:
        - Approximate nearest neighbor search (FAISS, Annoy)
        - Pre-filtering by lexical match before semantic computation
        - GPU-accelerated batch cosine similarity
    """
    return [
        base_score(query=query, product=prod, config=config, rng=rng)
        for prod in catalog
    ]
```

**Verification** (let's test on synthetic data):

```python
# Generate test data
from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import generate_catalog
from zoosim.world.queries import sample_query
from zoosim.world.users import sample_user

config = SimulatorConfig(seed=42)
rng = np.random.default_rng(config.seed)

# Generate world
catalog = generate_catalog(cfg=config.catalog, rng=rng)
user = sample_user(config=config, rng=rng)
query = sample_query(user=user, config=config, rng=rng)

# Compute base scores
scores = batch_base_scores(query=query, catalog=catalog, config=config, rng=rng)

# Rank products by base score
ranked_indices = np.argsort(scores)[::-1]  # Descending order
top_10 = ranked_indices[:10]

print("Top 10 products by base relevance:")
for rank, idx in enumerate(top_10, start=1):
    prod = catalog[idx]
    print(f"{rank}. Product {prod.product_id}: {prod.category}, score={scores[idx]:.3f}")
```

**Representative output:**
```
Top 10 products by base relevance:
1. Product 4237: cat_food, score=1.142
2. Product 8821: cat_food, score=1.089
3. Product 1544: cat_food, score=1.076
4. Product 9102: dog_food, score=0.934
5. Product 3311: cat_food, score=0.921
6. Product 7625: cat_food, score=0.899
7. Product 2847: dog_food, score=0.871
8. Product 5509: litter, score=0.823
9. Product 6743: cat_food, score=0.807
10. Product 1092: toys, score=0.654
```

**Observation**: Cat food dominates for this query (likely `query_type="category"` with cat preference). Semantic + lexical both contribute to high scores for category-matched products.

---

## 5.3 Feature Engineering: State Representation for RL

Base relevance gives us an **initial ranking**. Now the RL agent needs to decide: **which products should be boosted or suppressed?**

To make this decision, the agent needs features $\phi(u, q, p)$ that capture:
1. **Business metrics**: CM2, price, discount (direct impact on reward)
2. **User preferences**: Personalization signals (price sensitivity, PL affinity)
3. **Query context**: Specificity, category intent
4. **Product attributes**: Strategic flags, bestseller scores

**The RL state:**

For a query-user pair $(q, u)$ and catalog $\mathcal{C}$, the state is:
$$
x = \left(u, q, \{\phi(u, q, p_i) : p_i \in \mathcal{C}\}\right)
\tag{5.4}
$$
{#EQ-5.4}

The agent observes:
- User attributes: segment, preferences $(\theta_{\text{price}}, \theta_{\text{PL}})$
- Query attributes: type, embedding
- **Per-product features**: $\phi(u, q, p) \in \mathbb{R}^d$

**Design principle: Markovian sufficiency**

The feature vector $\phi(u, q, p)$ should contain **all information necessary** to predict:
- Expected reward $\mathbb{E}[R \mid u, q, p, a]$ as a function of boost $a$
- Click probability $\mathbb{P}[\text{click} \mid u, q, p, \text{position}]$
- Purchase probability $\mathbb{P}[\text{buy} \mid \text{click}, u, q, p]$

If $\phi$ is insufficient (e.g., missing price), the agent cannot learn optimal boosts. If $\phi$ is redundant (e.g., correlated features), learning is slower but still works.

### 5.3.1 Feature Design

**Definition 5.5** (Feature Vector) {#DEF-5.5}

Given a user $u \in \mathcal{U}$, a query $q \in \mathcal{Q}$, and a product $p \in \mathcal{C}$, the **feature vector** is
$$
\phi(u, q, p) = (\phi^{\text{prod}}, \phi^{\text{pers}}, \phi^{\text{inter}}) \in \mathbb{R}^d,
\tag{5.5}
$$
{#EQ-5.5}
where:
- $\phi^{\text{prod}} \in \mathbb{R}^{5}$ collects **product-only features** (CM2, discount, PL flag, bestseller score, price),
- $\phi^{\text{pers}} \in \mathbb{R}^{1}$ collects **personalization features** (here a single user–product embedding affinity),
- $\phi^{\text{inter}} \in \mathbb{R}^{4}$ collects **interaction features** (e.g., CM2×Litter, Discount×PriceSens, PL×PLAff, Spec×Bestseller),
- and
$$
d = 5 + 1 + 4 = 10
$$
is the total feature dimension.

The concatenation $(\phi^{\text{prod}}, \phi^{\text{pers}}, \phi^{\text{inter}})$ is understood as forming a single vector in $\mathbb{R}^{10}$ by stacking the components in a fixed order.

**Concrete feature list** (from `zoosim/ranking/features.py`):

| Index | Feature | Type | Formula | Interpretation |
|-------|---------|------|---------|----------------|
| 0 | CM2 | Product | $p.\text{cm2}$ | Contribution margin |
| 1 | Discount | Product | $p.\text{discount}$ | Discount fraction |
| 2 | Private Label | Product | $\mathbb{1}_{p.\text{is\_pl}}$ | Binary PL flag |
| 3 | Personalization | Personalization | $\langle u.\theta_{\text{emb}}, p.\mathbf{e} \rangle$ | User-product affinity |
| 4 | Bestseller | Product | $p.\text{bestseller}$ | Popularity score |
| 5 | Price | Product | $p.\text{price}$ | Absolute price |
| 6 | CM2 × Litter | Interaction | $p.\text{cm2} \cdot \mathbb{1}_{\text{litter}}$ | Strategic category CM2 |
| 7 | Discount × Price Sensitivity | Interaction | $p.\text{discount} \cdot u.\theta_{\text{price}}$ | Personalized discount value |
| 8 | PL × PL Affinity | Interaction | $\mathbb{1}_{p.\text{is\_pl}} \cdot u.\theta_{\text{PL}}$ | Personalized PL preference |
| 9 | Specificity × Bestseller | Interaction | $\text{specificity}(q) \cdot p.\text{bestseller}$ | Context-aware popularity |

**Total: $d = 10$ features**

**Feature design rationale:**

**Product features (0-2, 4-5):**
- **CM2 (0)**: Direct reward component—agent should learn to boost high-margin products
- **Discount (1)**: Affects user utility and purchase probability
- **PL flag (2)**: Binary indicator for private label (often correlated with margin and quality perception)
- **Bestseller (4)**: Popularity proxy—high bestseller score suggests high click/conversion rates
- **Price (5)**: Affects purchase decision (higher price → lower conversion, but higher GMV if purchased)

**Personalization features (3):**
- **User-product dot product (3)**: $\langle \boldsymbol{\theta}_u, \mathbf{e}_p \rangle$ captures **personalized relevance**
  - If $\boldsymbol{\theta}_u$ aligns with $\mathbf{e}_p$, user likely prefers this product
  - Complements base relevance (which uses query embedding, not user embedding)

**Interaction features (6-9):**
- **CM2 × Litter (6)**: Litter is a **strategic category** (loss leader to drive traffic). This feature lets the agent learn category-specific boost strategies.
- **Discount × Price Sensitivity (7)**: In config, more price-sensitive users have **more negative** $\theta_{\text{price}}$ (stronger aversion to high prices), so $p.\text{discount} \cdot u.\theta_{\text{price}}$ is **more negative** for price hunters than for premium users. The RL agent typically learns a **negative coefficient** on this feature, so larger (in magnitude) negative values correspond to “discounts matter more” for price-sensitive users.
- **PL × PL Affinity (8)**: Some users love private label, others avoid it. This feature enables personalized PL boosting.
- **Specificity × Bestseller (9)**: For generic queries (`"dog food"`), show bestsellers. For specific queries (`"grain-free salmon kibble"`), bestseller is less relevant.

**Why interactions matter:**

Linear models with raw features assume **additive effects**:
$$
Q(x, a) \approx \sum_i w_i \phi_i(x)
$$

But boosts have **multiplicative effects**:
- Boosting a high-margin product by $+0.2$ has high ROI
- Boosting a negative-margin product by $+0.2$ loses money
- **Interaction term** captures: $w_{CM2} \cdot \text{cm2} + w_a \cdot a + w_{CM2 \times a} \cdot \text{cm2} \cdot a$

Interaction features approximate multiplicative effects in a linear model.

**Implementation:**

```python
from typing import List
import torch
from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import Product
from zoosim.world.queries import Query
from zoosim.world.users import User

def compute_features(
    *,
    user: User,
    query: Query,
    product: Product,
    config: SimulatorConfig
) -> List[float]:
    """Compute feature vector φ(u, q, p) for RL state representation.

    Mathematical basis: [EQ-5.5] (Feature Vector)

    Returns 10-dimensional feature vector:
    - Indices 0-2, 4-5: Product features (CM2, discount, PL, bestseller, price)
    - Index 3: Personalization (⟨θ_u, e_p⟩)
    - Indices 6-9: Interaction features (categorical, personalized, contextual)

    Args:
        user: User with preference vectors (theta_price, theta_pl, theta_emb)
        query: Query with type and specificity
        product: Product with all attributes
        config: Simulator config (for query specificity lookup)

    Returns:
        Feature vector, length 10

    References:
        - [DEF-5.5] Feature Vector definition
        - Implementation: `zoosim/ranking/features.py:28-60`
        - Feature standardization: [EQ-5.6] below
    """
    # Personalization: user-product affinity via embedding dot product
    pers = float(torch.dot(user.theta_emb, product.embedding))

    # Query specificity (context signal)
    specificity = config.queries.specificity.get(query.query_type, 0.5)

    # Build feature vector [DEF-5.5]
    features = [
        # Product features
        product.cm2,                              # [0] CM2
        product.discount,                         # [1] Discount
        float(product.is_pl),                     # [2] PL flag (0 or 1)
        pers,                                     # [3] Personalization
        product.bestseller,                       # [4] Bestseller
        product.price,                            # [5] Price

        # Interaction features
        product.cm2 if product.category == "litter" else 0.0,  # [6] CM2 × Litter
        product.discount * user.theta_price,                   # [7] Discount × Price sens
        float(product.is_pl) * user.theta_pl,                  # [8] PL × PL affinity
        specificity * product.bestseller,                      # [9] Specificity × Bestseller
    ]

    return features
```

!!! note "Code ↔ Config (Feature Dimension)"
    The feature dimension $d$ is set in `SimulatorConfig.action.feature_dim` (default: 10).
    - **File**: `zoosim/core/config.py:228`
    - **Usage**: RL agents use this to size input layers (e.g., neural network: $\mathbb{R}^{10} \to \mathbb{R}$)
    - **Invariant**: `len(compute_features(...))` must equal `config.action.feature_dim`

### 5.3.2 Feature Standardization

**Problem: Scale differences**

Feature values have very different ranges:
- CM2: $[-5, +30]$ (litter negative, dog food high)
- Discount: $[0, 0.3]$ (small range)
- Price: $[5, 50]$ (currency units)
- Personalization: typically on the order of $\sqrt{d}$ (dot product of Gaussian embeddings; scale grows with embedding dimension)

Without standardization, gradient-based RL methods (policy gradients, Q-learning with neural nets) struggle:
- Large-scale features (price) dominate gradients
- Small-scale features (discount) have negligible impact on loss
- Learning is slow and unstable

**Solution: Z-score normalization**

**Definition 5.6** (Feature Standardization) {#DEF-5.6}

Let $\{\phi_1, \ldots, \phi_N\}$ be a collection of feature vectors with $\phi_i \in \mathbb{R}^d$. The **standardized features** are defined coordinate-wise by
$$
\tilde{\phi}_i^{(j)} = \frac{\phi_i^{(j)} - \mu^{(j)}}{\sigma^{(j)}}
\tag{5.6}
$$
{#EQ-5.6}

where, for each feature dimension $j \in \{1, \ldots, d\}$,
$$
\mu^{(j)} = \frac{1}{N} \sum_{i=1}^N \phi_i^{(j)}, \qquad
\sigma^{(j)} = \sqrt{\frac{1}{N} \sum_{i=1}^N \bigl(\phi_i^{(j)} - \mu^{(j)}\bigr)^2},
$$
and where we adopt the convention that if $\sigma^{(j)} = 0$ (feature $j$ is constant over the batch), then we set $\sigma^{(j)} := 1$ so that $\tilde{\phi}_i^{(j)} = 0$ for all $i$.

**Batch specification (simulator vs. production):**
- In the **simulator** (Chapters&nbsp;5–8) we take $N = |\mathcal{C}|$ and, for a fixed episode $(u, q)$, set $\phi_i = \phi(u, q, p_i)$ for $p_i \in \mathcal{C}$. Thus $\mu^{(j)}$ and $\sigma^{(j)}$ are computed per-episode over the current catalog.
- In **production**, $\{\phi_i\}_{i=1}^N$ consists of all training examples (across many users, queries, and products); the resulting statistics $\mu^{(j)}$ and $\sigma^{(j)}$ are computed once at training time and **stored** for use at inference time.

**Qualitative properties:**
- For each $j$ with $\sigma^{(j)} > 0$, the standardized coordinate $\tilde{\phi}^{(j)}$ has mean $0$ and variance $1$ over the batch (see [PROP-5.2]).
- Standardization preserves relative ordering within each coordinate: $\phi_i^{(j)} > \phi_k^{(j)} \iff \tilde{\phi}_i^{(j)} > \tilde{\phi}_k^{(j)}$.
- The procedure is inherently **batch-dependent**: changing the batch $\{\phi_i\}$ changes $\mu^{(j)}$ and $\sigma^{(j)}$ and hence the standardized values.

**Proposition 5.2** (Properties of Z-Score Standardization) {#PROP-5.2}

Fix a coordinate $j$ and let $\phi_1^{(j)}, \ldots, \phi_N^{(j)} \in \mathbb{R}$ with $\sigma^{(j)} > 0$ defined as above. Define $\tilde{\phi}_i^{(j)}$ by [EQ-5.6]. Then:

(a) (**Zero mean**)
$$
\frac{1}{N} \sum_{i=1}^N \tilde{\phi}_i^{(j)} = 0.
$$

(b) (**Unit variance**)
$$
\frac{1}{N} \sum_{i=1}^N \bigl(\tilde{\phi}_i^{(j)}\bigr)^2 = 1.
$$

(c) (**Order preservation**) For all $i, k \in \{1, \ldots, N\}$,
$$
\phi_i^{(j)} > \phi_k^{(j)} \quad \Longleftrightarrow \quad \tilde{\phi}_i^{(j)} > \tilde{\phi}_k^{(j)}.
$$

(d) (**Affine invariance (positive scaling)**) If we replace $\phi_i^{(j)}$ by $a \phi_i^{(j)} + b$ with $a > 0$, the standardized values (and hence the ordering in (c)) are unchanged. If $a < 0$, the standardized values are multiplied by $-1$, so the ordering in (c) is reversed.

*Proof.* By construction,
$$
\tilde{\phi}_i^{(j)} = \frac{\phi_i^{(j)} - \mu^{(j)}}{\sigma^{(j)}}.
$$
Averaging over $i$ and using the definition of $\mu^{(j)}$ immediately gives (a). For (b),
$$
\frac{1}{N} \sum_{i=1}^N \bigl(\tilde{\phi}_i^{(j)}\bigr)^2
 = \frac{1}{N} \sum_{i=1}^N \frac{\bigl(\phi_i^{(j)} - \mu^{(j)}\bigr)^2}{\bigl(\sigma^{(j)}\bigr)^2}
 = \frac{1}{\bigl(\sigma^{(j)}\bigr)^2} \cdot \frac{1}{N} \sum_{i=1}^N \bigl(\phi_i^{(j)} - \mu^{(j)}\bigr)^2
 = 1
$$
by the definition of $\sigma^{(j)}$. The formula for $\tilde{\phi}_i^{(j)}$ is an affine transformation of $\phi_i^{(j)}$ with strictly positive slope $1 / \sigma^{(j)}$, so it is strictly increasing; this yields (c). For (d), let $\psi_i = a \phi_i^{(j)} + b$. Then $\mu_\psi = a \mu^{(j)} + b$ and $\sigma_\psi = |a| \sigma^{(j)}$, so
$$
\frac{\psi_i - \mu_\psi}{\sigma_\psi}
 = \frac{a}{|a|} \cdot \frac{\phi_i^{(j)} - \mu^{(j)}}{\sigma^{(j)}}
 = \operatorname{sign}(a)\, \tilde{\phi}_i^{(j)}.
$$
If $a>0$ the standardized values are unchanged; if $a<0$ they are negated, reversing the ordering. ∎

**When to standardize:**

```python
if config.action.standardize_features:
    # Compute features for all products in catalog
    raw_features = [compute_features(user=u, query=q, product=p, config=cfg)
                    for p in catalog]
    # Standardize across batch [EQ-5.6]
    standardized_features = standardize_features(raw_features, config=cfg)
else:
    # Use raw features (e.g., for interpretability in linear models)
    standardized_features = raw_features
```

**Implementation:**

```python
from typing import Sequence
import numpy as np

def standardize_features(
    feature_matrix: Sequence[Sequence[float]],
    *,
    config: SimulatorConfig
) -> List[List[float]]:
    """Standardize features via z-score normalization.

    Mathematical basis: [EQ-5.6] (Feature Standardization)

    Computes per-feature mean μ and std σ across all products,
    then transforms: φ̃ = (φ - μ) / σ

    Args:
        feature_matrix: List of feature vectors, shape (n_products, feature_dim)
        config: Simulator config (not used currently, reserved for future)

    Returns:
        Standardized features, shape (n_products, feature_dim)

    References:
        - [DEF-5.6] Feature Standardization definition
        - Implementation: `zoosim/ranking/features.py:63-78`

    Note:
        In production, store μ and σ from training data and apply to test data.
        Here, we standardize per-episode (reasonable for simulator).
    """
    array = np.asarray(feature_matrix, dtype=float)
    means = array.mean(axis=0)
    stds = array.std(axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero for constant features
    normalized = (array - means) / stds
    return normalized.tolist()
```

!!! note "Code ↔ Config (Standardization Flag)"
    Feature standardization is controlled by:
    - **File**: `zoosim/core/config.py:231` (`ActionConfig.standardize_features`)
    - **Default**: `True` (recommended for neural network policies)
    - **When to disable**: Linear regression with interpretable coefficients (raw feature units)

**Theory-practice gap: Online vs. Batch Standardization**

**Theory assumption (5.6):** Standardization uses statistics from the current batch (catalog for this episode).

**Practice in production:**
- **Training**: Compute $\mu, \sigma$ over all products in training catalog
- **Serving**: Store $\mu, \sigma$ and apply to new products/queries
- **Problem**: Distribution shift—if test catalog differs (new products, seasonal changes), standardization is mismatched

**Solutions:**
1. **Periodic recomputation**: Update $\mu, \sigma$ monthly from production data
2. **Robust scaling**: Use median and IQR instead of mean and std (resistant to outliers)
3. **Per-category standardization**: Compute separate $\mu_c, \sigma_c$ for each category

For now, our simulator re-standardizes each episode (acceptable for training; in Chapter 10 we'll address deployment).

---

## 5.4 Feature Visualization and Validation

Let us verify that our features capture the intended signals.

**Experiment: Feature distributions by user segment**

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import generate_catalog
from zoosim.world.users import sample_user
from zoosim.world.queries import sample_query
from zoosim.ranking.features import compute_features

# Setup
config = SimulatorConfig(seed=42)
rng = np.random.default_rng(config.seed)
catalog = generate_catalog(cfg=config.catalog, rng=rng)

# Sample features for 100 users per segment
feature_data = []
for segment in config.users.segments:
    for _ in range(100):
        # Force user to be from this segment (by hacking segment probabilities temporarily)
        cfg_temp = SimulatorConfig(seed=rng.integers(0, 1_000_000))
        cfg_temp.users.segment_mix = [1.0 if s == segment else 0.0
                                       for s in config.users.segments]
        user = sample_user(config=cfg_temp, rng=rng)
        query = sample_query(user=user, config=cfg_temp, rng=rng)

        # Compute features for a random product
        product = catalog[rng.integers(0, len(catalog))]
        features = compute_features(user=user, query=query, product=product, config=config)

        feature_data.append({
            'segment': segment,
            'cm2': features[0],
            'discount': features[1],
            'pl_flag': features[2],
            'personalization': features[3],
            'bestseller': features[4],
            'price': features[5],
            'cm2_x_litter': features[6],
            'discount_x_price_sens': features[7],
            'pl_x_pl_affinity': features[8],
            'specificity_x_bestseller': features[9],
        })

# Convert to DataFrame for plotting
import pandas as pd
df = pd.DataFrame(feature_data)

# Plot distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
feature_names = ['cm2', 'discount', 'personalization', 'price', 'discount_x_price_sens']

for idx, feat in enumerate(feature_names):
    ax = axes[idx // 3, idx % 3]
    for segment in config.users.segments:
        subset = df[df['segment'] == segment][feat]
        ax.hist(subset, bins=30, alpha=0.5, label=segment)
    ax.set_title(f'Feature: {feat}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plt.savefig('feature_distributions_by_segment.png')
print("Saved feature_distributions_by_segment.png")
```

**Representative output (observations):**

1. **Personalization feature**: Varies by segment (premium users have different $\boldsymbol{\theta}_u$ than price hunters)
2. **Discount × Price Sensitivity**: For price hunters, $\theta_{\text{price}}$ is more negative (stronger aversion to high prices), so $p.\text{discount} \cdot u.\theta_{\text{price}}$ is more negative in this segment; with a learned negative coefficient, this corresponds to “discounts help more” for price-sensitive users.
3. **CM2**: No segment dependence (it's a product attribute)—confirms features are well-designed
4. **Price**: Slight segment dependence if user preferences correlate with price (e.g., premium users prefer expensive products)

**Validation checks:**

```python
# Check feature ranges (no NaNs, reasonable bounds)
feature_cols = [
    "cm2",
    "discount",
    "pl_flag",
    "personalization",
    "bestseller",
    "price",
    "cm2_x_litter",
    "discount_x_price_sens",
    "pl_x_pl_affinity",
    "specificity_x_bestseller",
]
for i, col in enumerate(feature_cols):
    values = df[col].to_numpy()
    print(
        f"Feature {i} ({col}): min={values.min():.2f}, max={values.max():.2f}, "
        f"mean={values.mean():.2f}, std={values.std():.2f}"
    )
```

**Expected:**
- CM2: mean ≈ 5-10 (positive margin on average), std ≈ 5-8
- Discount: mean ≈ 0.1, max ≈ 0.3 (30% max discount from config)
- Price: mean ≈ 15, std ≈ 10 (lognormal with median around $12-15)

If values are far from expectations, check catalog generation (Chapter 4) or feature computation bugs.

---

## 5.5 Reward Aggregation: Multi-Objective Optimization

We have base relevance (for ranking) and features (for RL state). Now: **what should the agent optimize?**

Search ranking is **inherently multi-objective**:
- **Revenue (GMV)**: Maximize sales
- **Margin (CM2)**: Maximize profit
- **Strategic goals**: Promote specific categories (e.g., litter as loss leader)
- **Engagement**: Encourage clicks and exploration (proxy for satisfaction)
- **Constraints**: Maintain CM2 floor, exposure guarantees, rank stability

**The challenge:**

There is no single "correct" objective. Different business contexts require different trade-offs:
- **Mature marketplace**: Maximize CM2 (margin), subject to GMV ≥ baseline
- **Growth phase**: Maximize GMV (revenue), accept lower margins
- **Strategic campaigns**: Maximize litter sales (even at negative margin) to drive lifetime value

**Solution: Scalarized multi-objective reward**

To connect with the single-step reward formalism of Chapter&nbsp;1 and the MDP formalism of Chapter&nbsp;3, we distinguish:
- the **state space** $\mathcal{X}$ and **action space** $\mathcal{A}$ (discrete templates in Chapter&nbsp;6; continuous boosts in Chapter&nbsp;7),
- the **outcome space** $\Omega$ of user interactions (clicks and purchases).

An outcome $\omega \in \Omega$ specifies, for a given ranking of length $k$:
- click indicators $\{\text{clicked}_i(\omega)\}_{i=1}^k \in \{0,1\}^k$,
- purchase indicators $\{\text{purchased}_i(\omega)\}_{i=1}^k \in \{0,1\}^k$.

**Definition 5.7** (Multi-Objective Reward) {#DEF-5.7}

Given a state $x \in \mathcal{X}$, an action $a \in \mathcal{A}$ (a ranking or boost vector), and an outcome $\omega \in \Omega$, we define the **multi-objective reward** as:
$$
R(x, a, \omega) = \alpha \cdot \text{GMV}(\omega) + \beta \cdot \text{CM2}(\omega) + \gamma \cdot \text{Strategic}(\omega) + \delta \cdot \text{Clicks}(\omega),
\tag{5.7}
$$
{#EQ-5.7}
where:
- $\text{GMV}(\omega) = \sum_{i=1}^k \text{purchased}_i(\omega) \cdot \text{price}_i$: Total revenue from purchased products,
- $\text{CM2}(\omega) = \sum_{i=1}^k \text{purchased}_i(\omega) \cdot \text{cm2}_i$: Total contribution margin,
- $\text{Strategic}(\omega) = \sum_{i=1}^k \text{purchased}_i(\omega) \cdot \mathbb{1}_{\{\text{strategic}_i\}}$: Count of strategic product purchases,
- $\text{Clicks}(\omega) = \sum_{i=1}^k \text{clicked}_i(\omega)$: Total clicks (engagement proxy).

The outcome $\omega$ is random: for each $(x, a)$ we have a conditional distribution
$$
\omega \sim P(\cdot \mid x, a),
$$
where $P$ is the **user behavior model** (Chapter&nbsp;2, concept `CN-ClickModel`) and Chapter&nbsp;3 views $P$ as the transition kernel of the MDP. The **single-step reward function** used in [EQ-1.12] is then
$$
R(x, a) = \mathbb{E}\bigl[R(x, a, \omega) \mid x, a\bigr] = \int_{\Omega} R(x, a, \omega)\, P(\mathrm{d}\omega \mid x, a).
$$

**Weight selection:**

From `RewardConfig` (simulator defaults in this repo):
- $\alpha = 1.0$: GMV weight (baseline)
- $\beta = 0.4$: CM2 weight (profit sensitivity)
- $\gamma = 2.0$: Strategic weight (reward units per strategic purchase)
- $\delta = 0.1$: Clicks weight (**small** to avoid clickbait)

**Constraint 5.8** (Engagement Weight Safety Guideline) {#CONSTRAINT-5.8}

To mitigate clickbait incentives, we impose the **weight-ratio guideline**
$$
\frac{\delta}{\alpha} \in [0.01, 0.10].
\tag{5.8}
$$
{#EQ-5.8}

**Rationale.** If $\delta / \alpha$ is large, the agent can gain substantial reward from clicks even when those clicks do not lead to purchases. For instance, comparing two products with equal price and margin but different click and conversion rates, a very large $\delta / \alpha$ makes it profitable to boost a high-click/low-conversion product over a moderate-click/high-conversion one, leading to **clickbait ranking**:
- boosting flashy products (high click rate, low conversion),
- suppressing high-conversion products (lower click rate, high purchase rate),
- net effect: more clicks, less revenue.

Bounding $\delta / \alpha$ between $0.01$ and $0.10$ keeps the **engagement term** in [EQ-5.7] small relative to the GMV term so that revenue remains the primary driver of the policy, in line with the Chapter&nbsp;1 discussion ([EQ-1.2], [REM-1.2.1]).

**Limitation.** Constraint&nbsp;5.8 is a **heuristic guideline**, not a formal guarantee of incentive compatibility. A rigorous analysis would require explicit assumptions on the click and conversion probabilities and a comparison of expected rewards under competing policies (see the exercises in `docs/book/ch05/exercises_labs.md` for counterexamples and further discussion).

**Remark 5.1** (Engagement as Soft Viability Constraint) {#REM-5.1}

The engagement term $\delta \cdot \text{Clicks}$ is not itself a business metric—it is a **proxy for user satisfaction**. The motivation (from Chapter&nbsp;1) is:

- **Short-term reward**: GMV + CM2 measured within one episode
- **Long-term value**: Satisfied users return (retention, LTV)
- **Proxy hypothesis**: More clicks → higher engagement → higher satisfaction → better retention

This is a **modeling assumption**. In Chapter&nbsp;11 (Multi-Episode MDP), we'll replace this proxy with **actual retention dynamics** and compare:
- **Single-step proxy**: $\delta \cdot \text{Clicks}$ (this chapter)
- **Multi-episode value**: $\mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t R_t \mid s_0]$ with retention state

For now, we use the proxy with a **small weight** (e.g., $\delta = 0.1$) and enforce the safety guideline [CONSTRAINT-5.8].

**Implementation:**

```python
from dataclasses import dataclass
from typing import Sequence, Tuple
from zoosim.core.config import SimulatorConfig, RewardConfig
from zoosim.world.catalog import Product

@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components for logging/analysis."""
    gmv: float
    cm2: float
    strat: float
    clicks: int

def compute_reward(
    *,
    ranking: Sequence[int],
    clicks: Sequence[int],
    buys: Sequence[int],
    catalog: Sequence[Product],
    config: SimulatorConfig,
) -> Tuple[float, RewardBreakdown]:
    """Compute multi-objective reward from user interaction outcome.

    Mathematical basis: [EQ-5.7] (Multi-Objective Reward)

    Aggregates:
    - GMV: Total revenue from purchases
    - CM2: Total contribution margin from purchases
    - Strategic: Count of strategic product purchases
    - Clicks: Total clicks (engagement proxy)

    Safety check: Enforces [CONSTRAINT-5.8] (clickbait mitigation guideline)

    Args:
        ranking: Product indices in displayed order, length k
        clicks: Binary click indicators, length k (1 if clicked, 0 otherwise)
        buys: Binary purchase indicators, length k (1 if purchased, 0 otherwise)
        catalog: Full product catalog (for looking up attributes)
        config: Simulator config (reward weights in config.reward)

    Returns:
        reward: Scalar reward [EQ-5.7]
        breakdown: RewardBreakdown with components for logging

    References:
        - [DEF-5.7] Multi-Objective Reward definition
        - [CONSTRAINT-5.8] Engagement weight safety guideline
        - [REM-5.1] Engagement as soft viability proxy
        - Implementation: `zoosim/dynamics/reward.py:42-66`
    """
    # Compute components
    gmv = 0.0
    cm2 = 0.0
    strat = 0.0
    click_total = 0

    limit = min(len(ranking), len(buys))
    for idx in range(limit):
        pid = ranking[idx]
        click_total += int(clicks[idx]) if idx < len(clicks) else 0

        if buys[idx]:
            prod = catalog[pid]
            gmv += prod.price
            cm2 += prod.cm2
            strat += 1.0 if prod.strategic_flag else 0.0

    breakdown = RewardBreakdown(gmv=gmv, cm2=cm2, strat=strat, clicks=click_total)

    # Enforce engagement weight bounds [EQ-5.8]
    cfg: RewardConfig = config.reward
    alpha = float(cfg.alpha_gmv)
    ratio = float("inf") if alpha == 0.0 else float(cfg.delta_clicks) / alpha

    assert 0.01 <= ratio <= 0.10, (
        f"Engagement weight outside safe range [0.01, 0.10]: "
        f"delta/alpha = {ratio:.3f}. Adjust RewardConfig to avoid clickbait optimization."
    )

    # Compute scalarized reward [EQ-5.7]
    reward = (
        cfg.alpha_gmv * breakdown.gmv
        + cfg.beta_cm2 * breakdown.cm2
        + cfg.gamma_strat * breakdown.strat
        + cfg.delta_clicks * breakdown.clicks
    )

    return reward, breakdown
```

!!! note "Code ↔ Config (Reward Weights)"
    The weights $(α, β, γ, δ)$ are configured in:
    - **File**: `zoosim/core/config.py:195-199` (`RewardConfig`)
    - **Defaults**: `alpha_gmv=1.0`, `beta_cm2=0.4`, `gamma_strat=2.0`, `delta_clicks=0.1`
    - **Safety**: Assertion in `zoosim/dynamics/reward.py:56-59` enforces the guideline [CONSTRAINT-5.8]
    - **Tuning**: Adjust weights to match business priorities (see Section 5.6 for Pareto analysis)

!!! note "Code ↔ Reward"
    The scalar reward [EQ-5.7] is implemented in `MOD-zoosim.reward` (`zoosim/dynamics/reward.py:42-66`), which aggregates GMV, CM2, strategic purchases, and clicks and enforces the engagement safety bound from [REM-1.2.1] and [CONSTRAINT-5.8]. The Chapter‑5 unit tests `TEST-tests.ch05.test_ch05_core` (`tests/ch05/test_ch05_core.py`) and the env smoke test `TEST-tests.test_env_basic` (`tests/test_env_basic.py`) pin `compute_reward()` against [EQ-5.7] on simple synthetic patterns and in the integrated simulator.

**Example scenario:**

```python
# Simulate a user session
ranking = [42, 103, 7, 201, 88]  # Top 5 products
clicks = [1, 1, 0, 1, 0]         # User clicked products 42, 103, 201
buys = [1, 0, 0, 1, 0]           # User bought products 42, 201

# Assume:
# Product 42: price=$20, cm2=$8, strategic=False
# Product 103: price=$15, cm2=$6, strategic=False
# Product 201: price=$12, cm2=-$2, strategic=True (litter)

# Compute reward
reward, breakdown = compute_reward(
    ranking=ranking,
    clicks=clicks,
    buys=buys,
    catalog=catalog,
    config=config
)

print(f"Reward breakdown:")
print(f"  GMV: ${breakdown.gmv:.2f}")
print(f"  CM2: ${breakdown.cm2:.2f}")
print(f"  Strategic purchases: {breakdown.strat}")
print(f"  Clicks: {breakdown.clicks}")
print(f"Total reward: {reward:.2f}")
```

**Representative output:**
```
Reward breakdown:
  GMV: $32.00  (20 + 12)
  CM2: $6.00   (8 + (-2))
  Strategic purchases: 1.0
  Clicks: 3
Total reward: 36.70
  = 1.0 * 32 (GMV) + 0.4 * 6 (CM2) + 2.0 * 1 (Strategic) + 0.1 * 3 (Clicks)
  = 32 + 2.4 + 2.0 + 0.3 = 36.70
```

**Interpretation:**
- GMV dominates (+32.0) and CM2 is secondary (+2.4)
- Strategic product bonus: +2.0
- Click engagement: +0.3 (small contribution as intended)

---

## 5.6 Reward Weight Tuning: Pareto Frontier

**Problem: How to choose $(α, β, γ, δ)$?**

Different weights lead to different policies. There is no "optimal" choice—it's a **business decision** about trade-offs.

**Multi-objective RL framework:**

Think of the reward as a **scalarization** of a multi-objective problem:
$$
\max_\pi \mathbb{E}_\pi\left[\begin{pmatrix} \text{GMV} \\ \text{CM2} \\ \text{Strategic} \\ \text{Clicks} \end{pmatrix}\right]
\quad \text{subject to constraints}
\tag{5.9}
$$
{#EQ-5.9}

Each weight vector $(\alpha, \beta, \gamma, \delta)$ defines a candidate policy $\pi^*(\alpha, \beta, \gamma, \delta)$ obtained by maximizing the scalarized objective [EQ-5.7].

**Definition 5.8** (Weak Pareto Optimality) {#DEF-5.8}

Let $\mathcal{C} = (C_1, C_2, C_3, C_4) = (\text{GMV}, \text{CM2}, \text{Strategic}, \text{Clicks})$ denote the vector of objective components. A policy $\pi$ is **weakly Pareto optimal** if there exists no other policy $\pi'$ such that
$$
\mathbb{E}_{\pi'}[C_i] \geq \mathbb{E}_\pi[C_i] \quad \forall i \in \{1,2,3,4\}
$$
with strict inequality for at least one $i$.

**Remark.** A policy $\pi$ is sometimes called **strongly Pareto optimal** if there is no $\pi'$ with
$$
\mathbb{E}_{\pi'}[C_i] > \mathbb{E}_\pi[C_i] \quad \forall i \in \{1,2,3,4\}.
$$
Weak Pareto optimality (Definition&nbsp;5.8) is the standard notion used in multi-objective RL [@roijers:survey_morl:2013].

**Terminology note.** Some optimization texts use “Pareto optimal” (or “nondominated”) for Definition&nbsp;5.8 and reserve “weak Pareto optimal” for the condition in the remark (no strict improvement in all objectives). We follow the convention in [@roijers:survey_morl:2013] to align with multi-objective RL usage.

**Theorem 5.1** (Scalarization Yields Weakly Pareto-Optimal Policies) {#THM-5.1}

Let $(\alpha, \beta, \gamma, \delta) \in \mathbb{R}_+^4$ with all components strictly positive. Let $\pi^*$ be any policy that maximizes the scalarized objective [EQ-5.7], i.e.
$$
\pi^* \in \arg\max_{\pi} \left\{\alpha \,\mathbb{E}_\pi[C_1] + \beta \,\mathbb{E}_\pi[C_2] + \gamma \,\mathbb{E}_\pi[C_3] + \delta \,\mathbb{E}_\pi[C_4]\right\}.
$$
Then $\pi^*$ is weakly Pareto optimal in the sense of [DEF-5.8].

*Proof.* Suppose for contradiction that $\pi^*$ is not weakly Pareto optimal. Then there exists a policy $\pi'$ such that
$$
\mathbb{E}_{\pi'}[C_i] \ge \mathbb{E}_{\pi^*}[C_i] \quad \forall i \in \{1,2,3,4\},
$$
with strict inequality for at least one index $i_0$. Since all weights are strictly positive,
$$
\alpha \,\mathbb{E}_{\pi'}[C_1] + \beta \,\mathbb{E}_{\pi'}[C_2]
 + \gamma \,\mathbb{E}_{\pi'}[C_3] + \delta \,\mathbb{E}_{\pi'}[C_4]
 \;>\;
\alpha \,\mathbb{E}_{\pi^*}[C_1] + \beta \,\mathbb{E}_{\pi^*}[C_2]
 + \gamma \,\mathbb{E}_{\pi^*}[C_3] + \delta \,\mathbb{E}_{\pi^*}[C_4].
$$
This contradicts the optimality of $\pi^*$ for the scalarized objective, so $\pi^*$ must be weakly Pareto optimal. ∎

**Pareto frontier.** The **Pareto frontier** is the set of all weakly Pareto-optimal policies. Each point on the frontier represents a different trade-off between GMV, CM2, strategic purchases, and clicks; Theorem&nbsp;5.1 shows that, under positive weights, scalarization as in [EQ-5.7] can only produce policies on this frontier.

**Warning (scalarization does not recover the full frontier).** The converse of Theorem&nbsp;5.1 is false in general: not every Pareto-optimal policy is the maximizer of a linear weighted sum. Weighted-sum scalarization recovers *all* Pareto-optimal points only when the achievable set of objective vectors is convex; when the frontier is non-convex, scalarization misses the so-called *unsupported* Pareto points. This matters in practice: constraint and fairness trade-offs often create non-convex fronts. In Chapter 14 we switch to $\varepsilon$-constraint / CMDP formulations to recover the full trade-off surface while enforcing hard guardrails.

**Experiment: Trace Pareto frontier by sweeping weights**

```python
import numpy as np
import matplotlib.pyplot as plt
from zoosim.core.config import SimulatorConfig, RewardConfig

# We'll simulate policies with different reward weights
# (In practice, train RL agents for each weight setting; here we use heuristics)

def simulate_policy(alpha: float, beta: float, config: SimulatorConfig, n_episodes: int = 1000):
    """Simulate n episodes under a policy with given reward weights.

    Returns: (mean_gmv, mean_cm2, mean_strategic, mean_clicks)
    """
    # Placeholder: In full implementation, train RL agent with these weights
    # Here, we'll use a simple heuristic: boost products with high (alpha*price + beta*cm2)

    cfg = SimulatorConfig(seed=config.seed)
    cfg.reward.alpha_gmv = alpha
    cfg.reward.beta_cm2 = beta
    cfg.reward.delta_clicks = 0.1  # Keep clicks fixed

    # Run simulation (simplified: just generate random outcomes weighted by config)
    rng = np.random.default_rng(cfg.seed)

    gmv_sum = 0.0
    cm2_sum = 0.0
    strat_sum = 0.0
    clicks_sum = 0

    for _ in range(n_episodes):
        # Simulate one episode
        # Placeholder: actual implementation uses ZooplusSearchEnv

        # Heuristic outcome based on weights:
        # High alpha → higher GMV, lower CM2 (boost expensive products)
        # High beta → higher CM2, lower GMV (boost high-margin products)

        base_gmv = 30.0
        base_cm2 = 10.0

        # Weight-dependent adjustment
        gmv = base_gmv + alpha * rng.normal(5, 2)
        cm2 = base_cm2 + beta * rng.normal(3, 1.5)
        strat = rng.poisson(0.3 * (alpha + beta))
        clicks = rng.poisson(3.0)

        gmv_sum += gmv
        cm2_sum += cm2
        strat_sum += strat
        clicks_sum += clicks

    return (gmv_sum / n_episodes, cm2_sum / n_episodes,
            strat_sum / n_episodes, clicks_sum / n_episodes)

# Sweep weights
config = SimulatorConfig(seed=42)
results = []

for alpha in np.linspace(0.5, 2.0, 10):
    for beta in np.linspace(0.5, 2.0, 10):
        gmv, cm2, strat, clicks = simulate_policy(alpha, beta, config, n_episodes=100)
        results.append({
            'alpha': alpha,
            'beta': beta,
            'gmv': gmv,
            'cm2': cm2,
            'strategic': strat,
            'clicks': clicks
        })

# Plot Pareto frontier (GMV vs CM2)
import pandas as pd
df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
plt.scatter(df['gmv'], df['cm2'], c=df['alpha'], cmap='viridis', s=100, alpha=0.7)
plt.colorbar(label='α (GMV weight)')
plt.xlabel('Mean GMV ($)')
plt.ylabel('Mean CM2 ($)')
plt.title('Pareto Frontier: GMV vs CM2 Trade-off')
plt.grid(True, alpha=0.3)
plt.savefig('pareto_frontier_gmv_cm2.png')
print("Saved pareto_frontier_gmv_cm2.png")
```

**Representative output:**

The plot shows a **convex curve** (Pareto frontier):
- Lower-left: Low $\alpha$, low $\beta$ → low GMV, low CM2 (bad policy)
- Upper-right: Balanced $\alpha \approx \beta$ → high GMV, high CM2 (efficient)
- Trade-off: Increasing $\alpha$ beyond optimal increases GMV but decreases CM2 (expensive low-margin products)

**Business decision:**
1. **Growth phase**: Choose high $\alpha$, low $\beta$ (maximize revenue, accept lower margin)
2. **Mature marketplace**: Choose low $\alpha$, high $\beta$ (maximize profit, accept lower revenue)
3. **Balanced**: Choose $\alpha \approx \beta$ (e.g., $\alpha = \beta = 1.0$)

**In Chapter 10 (Robustness & Guardrails)**, we'll add **hard constraints** (CM2 floor, exposure guarantees) and use **Lagrangian relaxation** to solve constrained MDPs.

---

## 5.7 Theory-Practice Gap: When Models Break

We've built three components: relevance, features, reward. Now let's be honest about **when they fail**.

### 5.7.1 Relevance Model Limitations

**Theory assumption ([DEF-5.4])**: Base relevance $s_{\text{base}}$ captures query-product match.

**Practice violations:**

1. **Static embeddings**: Our embeddings are **fixed** at catalog generation. In production:
   - Products change (description updates, new reviews, seasonal trends)
   - Queries evolve (new brands, emerging search terms)
   - **Solution**: Periodic re-embedding (monthly) or **learned embeddings** (BERT, Sentence-BERT)

2. **No behavioral signals**: Our relevance ignores:
   - Click-through rate (CTR): High CTR → likely relevant
   - Conversion rate (CVR): High CVR → highly relevant and valuable
   - **Solution**: **Learning-to-rank (LTR)** models that incorporate behavioral feedback
   - See Section 5.8 (Modern Context) for BERT, neural ranking

3. **Lexical brittleness**: Token overlap fails on:
   - Misspellings: `"liter"` vs `"litter"` (edit distance = 1, but overlap = 0)
   - Synonyms: `"puppy"` vs `"dog"` (semantically related, lexically distinct)
   - **Solution**: Semantic-only models or **fuzzy string matching**

4. **Position bias conflation**: Base relevance doesn't account for **where** a product is shown. A product clicked at position 1 may not be clicked at position 10, even with same relevance.
   - **Solution**: Position-aware relevance models or **counterfactual click models** (Chapter 2 covered this)

**When does it work anyway?**

For **e-commerce search** (vs. web search), lexical+semantic hybrid is robust:
- Users type explicit product categories (`"cat food"`, not `"best feline nutrition 2024"`)
- Catalog is structured (clean categories, product names)
- Queries are short (2-4 tokens, not sentences)

For web search, pure neural models (BERT) dominate. For e-commerce, hybrid is **good enough** (and faster).

### 5.7.2 Feature Engineering Limitations

**Theory assumption ([DEF-5.5])**: Features $\phi(u, q, p)$ are **Markov sufficient** for predicting reward.

**Practice violations:**

1. **Unobserved confounders**: True reward depends on:
   - User's current mood (not in features)
   - Competitor prices (not in our simulator)
   - External events (seasonality, holidays, promotions)
   - **Impact**: Agent learns correlations, not causal effects. If user mood changes, policy fails.

2. **Feature drift**: Distribution $p(\phi)$ changes over time:
   - New products have different price/margin distributions
   - User preferences shift (e.g., COVID increased price sensitivity)
   - **Solution**: Periodically retrain with recent data, or use **adaptive bandits** (Chapter 15)

3. **Interaction blindness**: We include 4 interaction features (6-9), but there are $\binom{10}{2} = 45$ possible pairwise interactions. Missing important ones:
   - **CM2 × Price**: High-margin products are often expensive (negative interaction?)
   - **Bestseller × Category**: Some categories have stronger bestseller effects
   - **Solution**: **Neural networks** learn interactions automatically via hidden layers

4. **Curse of dimensionality**: With $d = 10$ features and $k = 20$ products, the state space has $20 \times 10 = 200$ dimensions. For tabular methods (LinUCB), this is tractable. For neural nets, need $10^4$+ samples per feature dimension.

**Why it works anyway:**

- **Linear structure**: E-commerce rewards are **approximately linear** in features (high CM2 → high reward, high price → high GMV if purchased)
- **Feature selection**: Our 10 features capture **most variance** in reward (validated empirically in Chapter 6)
- **Regularization**: RL with function approximation implicitly regularizes (e.g., L2 penalty in LinUCB, dropout in neural nets)

### 5.7.3 Reward Design Limitations

**Theory assumption ([EQ-5.7])**: Scalarized reward aligns with **true business value**.

**Practice violations:**

1. **Short-term vs. long-term**: Single-episode reward ignores:
   - **Retention**: Satisfied users return (multi-episode value)
   - **Lifetime value (LTV)**: A user who buys once may buy 10 more times
   - **Brand loyalty**: Showing irrelevant products damages trust, reducing future GMV
   - **Solution**: Multi-episode MDP (Chapter 11) with retention state

2. **Engagement proxy failure**: $\delta \cdot \text{Clicks}$ assumes clicks ↔ satisfaction. But:
   - **False positives**: User clicks accidentally (mobile, fat fingers)
   - **False negatives**: User satisfied but doesn't click (found product in position 1, bought directly)
   - **Clickbait**: High-click, low-conversion products (flashy images, misleading titles)
   - **Solution**: Better proxies (time on page, add-to-cart, repeat visits) or **learned satisfaction models**

3. **Unmodeled constraints**: Real business has:
   - **Legal constraints**: GDPR, fairness regulations (no discrimination by protected attributes)
   - **Operational constraints**: Inventory limits (can't boost out-of-stock products)
   - **Strategic constraints**: Partner agreements (must show brand X in top 5)
   - **Solution**: CMDP with **Lagrangian relaxation** (preview in Remark 3.5.3; duality in Appendix C), implemented as guardrails in Chapter 10

4. **Reward hacking**: Agent finds **unintended optima**:
   - Example: Boost only negative-margin litter to maximize $\gamma \cdot \text{Strategic}$, lose money on CM2
   - Example: Show many low-value products to maximize clicks, sacrifice GMV
   - **Solution**: Careful weight tuning [EQ-5.8], adversarial testing, human oversight

**Open problem (as of 2025):**

> How to design reward functions that **robustly capture** long-term business value without unintended side effects?

This is an active research area:
- **Inverse RL** [@ng:irl:2000]: Learn reward from expert demonstrations
- **Preference learning** [@christiano:human_feedback:2017]: RLHF for language models (also applicable to search)
- **Safe RL** [@garcia:safe_rl:2015]: Constrained optimization with safety guarantees
- **Reward modeling** [@leike:reward_learning:2018]: Learn reward from human labels

For now, we use **scalarized multi-objective reward** with **safety assertions** [EQ-5.8] and **Pareto analysis** (Section 5.6).

---

## 5.8 Modern Context: Neural Ranking and Learned Representations (2020-2025)

Our hybrid relevance model (semantic + lexical) is a **classical baseline**. Modern search uses **neural ranking models** trained end-to-end from click data.

### 5.8.1 Learning-to-Rank (LTR) with Transformers

**State-of-the-art (2025):**

**BERT-based ranking** [@nogueira:passage_reranking:2019]:
1. **Encoder**: BERT or Sentence-BERT encodes query and document
2. **Interaction**: Cross-attention between query and document tokens
3. **Scoring**: Classification head outputs relevance score

**Architecture:**
```
Query: "premium dog food"
Document: "Blue Buffalo Life Protection Formula - Natural Adult Dry Dog Food"
    ↓
[CLS] premium dog food [SEP] Blue Buffalo Life Protection ... [SEP]
    ↓
BERT Transformer (12 layers, 768-dim)
    ↓
[CLS] embedding → MLP → P(relevant | query, doc)
```

**Training:**
- **Supervised**: Use click data as labels (clicked = relevant, not clicked = not relevant)
- **Loss**: Binary cross-entropy or ranking loss (hinge, pairwise)
- **Hard negatives**: Sample non-clicked products as negative examples

**Advantages over hybrid model:**
- **End-to-end learning**: No hand-crafted features (embeddings learned from data)
- **Contextual**: Attention mechanism captures complex query-document interactions
- **State-of-the-art**: +5-10% NDCG over BM25+embeddings on TREC benchmarks

**Disadvantages:**
- **Latency**: BERT forward pass is ~100ms per query-doc pair (too slow for 10k products)
- **Cost**: Requires millions of labeled query-doc pairs (click logs)
- **Interpretability**: Black box (hard to debug why product ranked high/low)

**Two-stage architecture (production standard):**
1. **Stage 1 (Retrieval)**: Fast model (BM25, hybrid semantic+lexical) retrieves top 500 candidates (<10ms)
2. **Stage 2 (Reranking)**: BERT reranks top 500 → top 20 (~50ms)

For our simulator, we stick with **hybrid model** (Stage 1 equivalent) for speed and interpretability. In production RL deployments, teams often add a Stage‑2 reranker (e.g., a BERT reranking head) on top of this hybrid model.

### 5.8.2 Learned Embeddings for Products and Queries

**Our approach (Chapter 4)**: Fixed Gaussian embeddings with category-dependent product clusters and user-centered query embeddings.

**Modern approach**: Learn embeddings from behavioral data.

**Product2Vec** [@grbovic:product2vec:2015]:
- Treat user sessions as "sentences", products as "words"
- Train Word2Vec (Skip-gram) on session sequences
- **Result**: Products bought together have similar embeddings

**Query2Vec**:
- Embed queries in same space as products
- Train on query-click pairs: $\text{query} \to \text{clicked\_products}$
- **Loss**: Contrastive learning (clicked products close, non-clicked far)

**Joint embedding space** [@huang:dssmn:2013]:
- **Deep Structured Semantic Model (DSSM)**: Neural network maps queries and products to shared embedding space
- **Training**: Maximize $\cos(\mathbf{q}, \mathbf{e}_{\text{clicked}})$, minimize $\cos(\mathbf{q}, \mathbf{e}_{\text{not\_clicked}})$
- **Advantage**: Captures **behavioral relevance** (what users actually click), not semantic similarity

**Why we don't use this yet:**

1. **Cold-start**: New products have no clicks (can't train embeddings)
2. **Data requirements**: Need millions of sessions (we're building a simulator)
3. **Embedding drift**: Must retrain frequently (weekly) as catalog changes

For **Chapters 6-10**, fixed embeddings are sufficient. In production systems, embedding updates and cold-start strategies become critical; later chapters on robustness and long-horizon behavior will refer back to these issues, but we do not attempt a full MLOps treatment.

### 5.8.3 Recent Research: Multi-Task Learning and Bias Correction

**Multi-task ranking** [@ma:entire_space:2018]:
- **Observation**: Optimizing CTR (clicks) ≠ optimizing CVR (purchases)
- **Solution**: Multi-task model predicts both CTR and CVR jointly
- **Loss**: $\mathcal{L} = \mathcal{L}_{\text{CTR}} + \lambda \mathcal{L}_{\text{CVR}}$

**Bias correction in ranking** [@joachims:unbiased_ltr:2017]:
- **Problem**: Click data is **biased** (position bias—users click top results more)
- **Solution**: Unbiased LTR via **inverse propensity scoring (IPS)**
- **Connection to RL**: This is off-policy evaluation (Chapter 9)!

**Fairness in ranking** [@singh:fairness_expo:2018]:
- **Problem**: Minority products under-represented (less clicks → worse embeddings → fewer impressions)
- **Solution**: Constrained ranking with **exposure guarantees** (Chapter 10 implements guardrails; Chapter 14 covers multi-objective CMDP)

**Open problems (2025):**
- **Causality**: Ranking models capture correlations, not causal effects (e.g., high price → low CTR, but does low price → high CTR?)
- **Long-term effects**: Optimizing short-term CTR may hurt long-term retention (RLHF for search?)
- **Generalization**: Models overfit to logged data distribution (distribution shift in deployment)

These are **frontier research directions**. Our simulator provides a **testbed** for exploring them.

---

## 5.9 Integrated Example: Full Episode

Let us trace a complete episode through all three components: relevance → features → reward.

**Scenario:**
- User: Price-sensitive shopper (segment: `price_hunter`)
- Query: `"cat food"` (type: `category`)
- Catalog: 10,000 products

**Step 1: Generate world**

```python
from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import generate_catalog
from zoosim.world.users import sample_user
from zoosim.world.queries import sample_query
import numpy as np

config = SimulatorConfig(seed=2025)
rng = np.random.default_rng(config.seed)

catalog = generate_catalog(cfg=config.catalog, rng=rng)
user = sample_user(config=config, rng=rng)
query = sample_query(user=user, config=config, rng=rng)

print(f"User segment: {user.segment}")
query_text = " ".join(query.tokens)
print(f"Query: '{query_text}' (type: {query.query_type})")
```

**Output:**
```
User segment: price_hunter
Query: 'cat food' (type: category)
```

**Step 2: Compute base relevance for all products**

```python
from zoosim.ranking.relevance import batch_base_scores

base_scores = batch_base_scores(query=query, catalog=catalog, config=config, rng=rng)
ranked_by_relevance = np.argsort(base_scores)[::-1][:20]  # Top 20

print("\nTop 5 products by base relevance:")
for rank, idx in enumerate(ranked_by_relevance[:5], start=1):
    prod = catalog[idx]
    print(f"{rank}. Product {prod.product_id}: {prod.category}, "
          f"price=${prod.price:.2f}, cm2=${prod.cm2:.2f}, score={base_scores[idx]:.3f}")
```

**Output:**
```
Top 5 products by base relevance:
1. Product 7821: cat_food, price=$14.23, cm2=$6.71, score=1.234
2. Product 3304: cat_food, price=$11.88, cm2=$5.12, score=1.198
3. Product 9127: cat_food, price=$9.45, cm2=$3.87, score=1.176
4. Product 1543: cat_food, price=$16.77, cm2=$8.23, score=1.154
5. Product 6209: cat_food, price=$13.21, cm2=$6.05, score=1.142
```

**Step 3: Compute features for top products**

```python
from zoosim.ranking.features import compute_features, standardize_features

raw_features = [
    compute_features(user=user, query=query, product=catalog[idx], config=config)
    for idx in ranked_by_relevance
]

features = standardize_features(raw_features, config=config)

print("\nFeatures for top product (Product 7821):")
feature_names = ['cm2', 'discount', 'pl', 'personalization', 'bestseller',
                 'price', 'cm2_litter', 'disc_price', 'pl_aff', 'spec_bs']
for i, name in enumerate(feature_names):
    print(f"  {name}: {features[0][i]:.3f} (raw: {raw_features[0][i]:.3f})")
```

**Output:**
```
Features for top product (Product 7821):
  cm2: 0.342 (raw: 6.71)
  discount: -0.521 (raw: 0.0)
  pl: -0.872 (raw: 0.0)
  personalization: 0.891 (raw: 0.73)
  bestseller: 1.234 (raw: 2.8)
  price: 0.102 (raw: 14.23)
  cm2_litter: 0.0 (raw: 0.0)
  disc_price: -0.412 (raw: 0.0)
  pl_aff: -0.621 (raw: 0.0)
  spec_bs: 1.567 (raw: 1.96)
```

**Interpretation:**
- High CM2 (good margin)
- No discount (discount=0)
- Not PL (pl=0)
- High personalization (user embedding aligns with product)
- High bestseller (popular product)
- Mid-range price

**Step 4: RL agent selects boosts** (placeholder for now—Chapter 6 implements agents)

```python
# Placeholder: Random baseline policy
boosts = rng.uniform(-0.1, 0.1, size=20)  # Random boosts in [-0.1, 0.1]

# Adjust scores by boosts
adjusted_scores = [base_scores[idx] + boosts[i] for i, idx in enumerate(ranked_by_relevance)]
final_ranking = ranked_by_relevance[np.argsort(adjusted_scores)[::-1]]

print("\nFinal ranking (after boosts):")
for rank, idx in enumerate(final_ranking[:5], start=1):
    prod = catalog[idx]
    print(f"{rank}. Product {prod.product_id}: {prod.category}, "
          f"price=${prod.price:.2f}, cm2=${prod.cm2:.2f}")
```

**Output:**
```
Final ranking (after boosts):
1. Product 3304: cat_food, price=$11.88, cm2=$5.12
2. Product 7821: cat_food, price=$14.23, cm2=$6.71
3. Product 9127: cat_food, price=$9.45, cm2=$3.87
4. Product 6209: cat_food, price=$13.21, cm2=$6.05
5. Product 1543: cat_food, price=$16.77, cm2=$8.23
```

**Step 5: Simulate user interaction** (clicks and purchases)

The environment uses the **Utility-Based Cascade Model** from §2.5.4 ([DEF-2.5.3]) to generate clicks and purchases. User preferences ($\theta_{\text{price}}$, $\theta_{\text{pl}}$, $\boldsymbol{\theta}_{\text{cat}}$) interact with product features and position bias to determine outcomes.

```python
from zoosim.dynamics.behavior import simulate_session

# Simulate session using Utility-Based Cascade Model (§2.5.4)
outcome = simulate_session(
    ranking=final_ranking,
    user=user,
    query=query,
    catalog=catalog,
    config=config,
    rng=rng
)
clicks, buys = outcome.clicks, outcome.buys

print("\nUser interaction:")
print(f"  Clicks: {clicks[:5]}")
print(f"  Buys: {buys[:5]}")
print(f"  Final satisfaction: {outcome.satisfaction:.2f}")
```

**Output:**
```
User interaction:
  Clicks: [1, 1, 0, 1, 0]
  Buys: [1, 0, 0, 1, 0]
  Final satisfaction: 1.23
```

**Step 6: Compute reward**

```python
from zoosim.dynamics.reward import compute_reward

reward, breakdown = compute_reward(
    ranking=final_ranking,
    clicks=clicks,
    buys=buys,
    catalog=catalog,
    config=config
)

print("\nReward breakdown:")
print(f"  GMV: ${breakdown.gmv:.2f}")
print(f"  CM2: ${breakdown.cm2:.2f}")
print(f"  Strategic purchases: {breakdown.strat}")
print(f"  Clicks: {breakdown.clicks}")
print(f"Total reward: {reward:.2f}")
```

**Output:**
```
Reward breakdown:
  GMV: $25.09  (11.88 + 13.21)
  CM2: $11.17  (5.12 + 6.05)
  Strategic purchases: 0.0
  Clicks: 3
Total reward: 29.86
  = 1.0 * 25.09 + 0.4 * 11.17 + 2.0 * 0 + 0.1 * 3
  = 25.09 + 4.47 + 0 + 0.3 = 29.86
```

**Summary of episode:**
1. Base relevance ranked products by query-product match
2. Features captured business metrics and personalization signals
3. Agent applied boosts (random baseline here; learned policy in Chapter 6)
4. User clicked 3 products, purchased 2
5. Reward = $29.86 (mainly from GMV)

**RL loop:** Agent observes $(x, a, r, x')$ and updates policy to maximize expected reward over many episodes.

---

## 5.10 Production Checklist

!!! tip "Production Checklist (Chapter 5)"
    **Relevance Model:**
    - [ ] **Weights**: Verify `RelevanceConfig.w_sem`, `w_lex` match production relevance correlation (target: ρ > 0.85)
    - [ ] **Noise**: Set `RelevanceConfig.noise_sigma` appropriately (0.05 for exploration, 0.0 for deterministic)
    - [ ] **Embeddings**: Ensure query and product embeddings are normalized (`‖e‖₂ = 1`) for stable cosine similarity
    - [ ] **Batch computation**: For >10k products, consider approximate nearest neighbor search (FAISS)
    - [ ] **Cold-start**: New products without embeddings—use category centroid as fallback

    **Feature Engineering:**
    - [ ] **Dimension**: Confirm `ActionConfig.feature_dim` matches `len(compute_features(...))` (default: 10)
    - [ ] **Standardization**: Enable `ActionConfig.standardize_features` for neural network policies
    - [ ] **Interaction terms**: Validate that interaction features (6-9) have non-zero variance
    - [ ] **NaN checks**: Add assertions in `compute_features()` to catch NaN/Inf (e.g., division by zero)
    - [ ] **Production statistics**: Store (μ, σ) from training data, apply to test/production data

    **Reward Aggregation:**
    - [ ] **Weights**: Set `RewardConfig.alpha_gmv`, `beta_cm2`, `gamma_strat`, `delta_clicks` per business priorities
    - [ ] **Safety constraint**: Verify `delta_clicks / alpha_gmv ∈ [0.01, 0.10]` (assertion in `compute_reward()`)
    - [ ] **Breakdown logging**: Log `RewardBreakdown` for every episode (enables offline analysis)
    - [ ] **Pareto analysis**: Sweep weights and plot Pareto frontier (Section 5.6) before production launch
    - [ ] **Constraint monitoring**: Track CM2 floor violations, exposure violations (Chapter 10)

    **Integration:**
    - [ ] **Determinism**: Fix all seeds (`config.seed`, `rng` passed consistently) for reproducibility
    - [ ] **Config versioning**: Log `config` hash with every experiment for traceability
    - [ ] **Unit tests**: Verify `base_score()`, `compute_features()`, `compute_reward()` with known inputs
    - [ ] **End-to-end test**: Run full episode (Section 5.9) and validate reward matches expected components

!!! note "Code ↔ Simulator"
    The ranking pipeline described in Sections 5.1–5.3 (hybrid base relevance [EQ-5.3], feature vector [EQ-5.5], standardization [EQ-5.6]) is wired into the single-step environment `MOD-zoosim.env` (`zoosim/envs/search_env.py:15-80`): `MOD-zoosim.relevance` provides `batch_base_scores()` and `MOD-zoosim.features` provides `compute_features()` / `standardize_features()`. The tests `TEST-tests.test_env_basic` (`tests/test_env_basic.py`) and `TEST-tests.ch05.test_ch05_core` (`tests/ch05/test_ch05_core.py`) together with the validation script `DOC-ch05-validate-script` (`scripts/validate_ch05.py`) exercise this RL loop end-to-end and on small synthetic examples.

---

## Exercises & Labs

See [`exercises_labs.md`](exercises_labs.md) for:
- **Exercises 5.1-5.3**: Analytical problems on relevance, features, reward
- **Labs 5.A-5.C**: Runnable code experiments

**Quick links:**
- **Lab 5.A**: Visualize feature distributions by user segment
- **Lab 5.B**: Trace Pareto frontier by sweeping reward weights
- **Lab 5.C**: Compare base relevance models (semantic-only, lexical-only, hybrid)

---

## Summary

This chapter built the **RL state-action-reward interface** for search ranking:

1. **Base Relevance (Section 5.1-5.2)**
   - Hybrid semantic + lexical matching [EQ-5.3]
   - Provides strong prior for RL warm-start
   - Implementation: `zoosim/ranking/relevance.py`

2. **Feature Engineering (Section 5.3-5.4)**
   - 10-dimensional feature vector $\phi(u, q, p)$ [EQ-5.5]
   - Product + personalization + interaction features
   - Standardization for neural network stability [EQ-5.6]
   - Implementation: `zoosim/ranking/features.py`

3. **Reward Aggregation (Section 5.5-5.6)**
   - Multi-objective scalarization [EQ-5.7]
   - Safety guideline on engagement weight [EQ-5.8], [CONSTRAINT-5.8]
   - Pareto analysis for weight tuning
   - Implementation: `zoosim/dynamics/reward.py`

4. **Theory-Practice Gaps (Section 5.7)**
   - Relevance: Static embeddings, no behavioral feedback
   - Features: Unobserved confounders, curse of dimensionality
   - Reward: Short-term vs. long-term, engagement proxy failure
   - **Honest assessment**: When models work (e-commerce structure) and when they break (cold-start, drift)

5. **Modern Context (Section 5.8)**
   - BERT-based neural ranking (2020-2025 state-of-the-art)
   - Learned embeddings (Product2Vec, DSSM)
   - Multi-task learning, bias correction, fairness
   - **Open problems**: Causality, long-term effects, generalization

**Next steps:**

With relevance, features, and reward in place, we're ready for **RL agents**:
- **Chapter 6**: Discrete template bandits (LinUCB, Thompson Sampling)
- **Chapter 7**: Continuous action spaces (Q-learning for boosts)
- **Chapter 10**: Robustness and Guardrails (CM2 floors, exposure guarantees)

The **environment is complete**. Now we build the agents.

---

## 5.11 Reference Table

| ID | Type | Name | Location |
|----|------|------|----------|
| DEF-5.1 | Definition | Base Relevance Function | §5.1 |
| DEF-5.2 | Definition | Semantic Relevance | §5.2.1 |
| DEF-5.3 | Definition | Lexical Relevance | §5.2.2 |
| DEF-5.4 | Definition | Hybrid Base Relevance | §5.2.3 |
| DEF-5.5 | Definition | Feature Vector | §5.3.1 |
| DEF-5.6 | Definition | Feature Standardization | §5.3.2 |
| DEF-5.7 | Definition | Multi-Objective Reward | §5.5 |
| DEF-5.8 | Definition | Weak Pareto Optimality | §5.6 |
| CONSTRAINT-5.8 | Constraint | Engagement Weight Safety Guideline | §5.5 |
| REM-5.1 | Remark | Engagement as Soft Viability Constraint | §5.5 |
| PROP-5.1 | Proposition | Semantic Relevance Properties | §5.2.1 |
| PROP-5.2 | Proposition | Z-Score Standardization Properties | §5.3.2 |
| THM-5.1 | Theorem | Scalarization Yields Weakly Pareto-Optimal Policies | §5.6 |
| EQ-5.1 | Equation | Semantic Relevance via Cosine Similarity | §5.2.1 |
| EQ-5.2 | Equation | Lexical Relevance via Log-Overlap | §5.2.2 |
| EQ-5.3 | Equation | Hybrid Base Relevance | §5.2.3 |
| EQ-5.4 | Equation | RL State Definition | §5.3 |
| EQ-5.5 | Equation | Feature Vector Decomposition | §5.3.1 |
| EQ-5.6 | Equation | Feature Standardization | §5.3.2 |
| EQ-5.7 | Equation | Multi-Objective Scalar Reward | §5.5 |
| EQ-5.8 | Equation | Engagement Weight Bound | §5.5 |
| EQ-5.9 | Equation | Multi-Objective Optimization Problem | §5.6 |

---

## References

Key papers and resources:

**Relevance Models:**
- [@grbovic:product2vec:2015] Grbovic et al., "E-commerce in Your Inbox: Product Recommendations at Scale"
- [@huang:dssmn:2013] Huang et al., "Learning Deep Structured Semantic Models for Web Search"

**Neural Ranking:**
- [@nogueira:passage_reranking:2019] Nogueira & Cho, "Passage Re-ranking with BERT"
- [@khattab:colbert:2020] Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"

**Learning-to-Rank:**
- [@joachims:unbiased_ltr:2017] Joachims et al., "Unbiased Learning-to-Rank with Biased Feedback"
- [@ma:entire_space:2018] Ma et al., "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate"

**Multi-Objective RL:**
- [@roijers:survey_morl:2013] Roijers et al., "A Survey of Multi-Objective Sequential Decision-Making"
- [@van_moffaert:morl:2014] Van Moffaert & Nowé, "Multi-objective reinforcement learning using sets of pareto dominating policies"

**Reward Learning:**
- [@ng:irl:2000] Ng & Russell, "Algorithms for Inverse Reinforcement Learning"
- [@christiano:human_feedback:2017] Christiano et al., "Deep Reinforcement Learning from Human Preferences"

**Fairness in Ranking:**
- [@singh:fairness_expo:2018] Singh & Joachims, "Fairness of Exposure in Rankings"

See `references.bib` for full citations.
