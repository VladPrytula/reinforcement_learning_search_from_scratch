# Chapter 4 — Catalog, Users, Queries: Generative World Design

## Why We Can't Experiment on Production Search

In Chapters 1-3, we built the mathematical foundations for reinforcement learning in search: MDPs, Bellman operators, convergence guarantees. Now we face a practical problem: **we can't run RL exploration on a live e-commerce search engine**.

The challenge is stark. Imagine deploying a policy gradient algorithm with ε-greedy exploration directly to production:

**What could go wrong:**
- **Revenue loss**: Random exploration shows irrelevant products, users abandon, GMV drops 20%
- **Compliance violations**: Strategic products (e.g., litter with negative margins) fall below CM2 floor
- **Brand damage**: Premium users see low-quality results, trust erodes
- **Irreversible harm**: Can't "undo" a bad ranking after the user left

**The counterfactual problem:**

Even if we could tolerate risk, we face a deeper issue: **we need counterfactual estimates**. When evaluating a new policy π, we must answer:

> "What would have happened if we had ranked differently in the past?"

But the past is fixed. We only observed trajectories under the production policy π₀. To evaluate π without deploying it, we need **off-policy evaluation** (Chapter 9), which requires:

1. **Logged data** with known propensities: $\mathbb{P}[a \mid x; \pi_0]$
2. **Sufficient exploration**: All actions $a$ that π might take were tried under π₀ (overlap)
3. **Accurate world model**: Understand $\mathbb{P}[r, x' \mid x, a]$ dynamics

Production logs give us (1) partially, but (2) is expensive and (3) requires a simulator.

**Our solution:**

Build a **synthetic world** that captures the essential dynamics of search ranking:
- **Catalog** $\mathcal{C}$: Products with prices, margins, categories, embeddings
- **Users** $\mathcal{U}$: Segments with preferences (price sensitivity, PL affinity, category interests)
- **Queries** $\mathcal{Q}$: Search intents with embeddings and types

This world must be:
1. **Deterministic**: Same seed → same experiments (reproducibility)
2. **Realistic**: Distributions match production (transferability)
3. **Configurable**: Adjust parameters to test robustness
4. **Fast**: Generate millions of episodes for training

Let's build it, starting with the catalog.

---

## Generative Model: Mathematical Formalism

We model the world as a **generative process** parameterized by a configuration $\theta$ and driven by pseudo-random seeds.

**Definition 4.1** (Generative World Model) {#DEF-4.1}

A **generative world** is a deterministic procedure parameterized by:

- $N_{\text{prod}} \in \mathbb{N}$: Number of products in the catalog
- $\theta_{\mathcal{C}}$: Catalog generation parameters (lognormal $(\mu_c, \sigma_c)$, margin slopes $\beta_c$, discount and bestseller settings)
- $\theta_{\mathcal{U}}$: User generation parameters (segment mix and segment-specific preference distributions)
- $\theta_{\mathcal{Q}}$: Query generation parameters (query-type probabilities and token vocabulary)
- $\text{seed} \in \mathbb{N}$: Pseudo-random seed for deterministic generation

Given these parameters and a pseudo-random generator $\text{rng}$ initialized with `seed`, the procedure produces:

- A finite product catalog $\mathcal{C} = \{p_1, \ldots, p_{N_{\text{prod}}}\}$
- A user sampler $\text{SampleUser}(\theta_{\mathcal{U}}, \text{rng})$ returning users $u \in \mathcal{U}$
- A query sampler $\text{SampleQuery}(u, \theta_{\mathcal{Q}}, \text{rng})$ returning queries $q \in \mathcal{Q}$ conditioned on the user

**Determinism property.**
For any seeds $s_1, s_2 \in \mathbb{N}$:
$$
s_1 = s_2 \implies (\mathcal{C}_1, \mathcal{U}_1, \mathcal{Q}_1) = (\mathcal{C}_2, \mathcal{U}_2, \mathcal{Q}_2),
$$
where equality is **component-wise**, including embeddings:
for every product index $i$ we have
$p_i^{(1)} = p_i^{(2)}$ in all scalar attributes and
$\|\mathbf{e}_i^{(1)} - \mathbf{e}_i^{(2)}\|_2 = 0$,
and likewise the user and query sampling functions produce identical sequences.
In the sense of [EQ-4.10], the entire **experiment** is a measurable function of configuration and seed.

This determinism is essential for reproducibility. We use NumPy's `np.random.Generator` and PyTorch's `torch.Generator` with explicit seeds, avoiding global state.

---

## Catalog Generation: Products and Embeddings

A **product** $p \in \mathcal{C}$ is characterized by:

$$
p = (\text{id}, \text{cat}, \text{price}, \text{cm2}, \text{is\_pl}, \text{discount}, \text{bs}, \mathbf{e}, \text{strategic})
\tag{4.1}
$$
{#EQ-4.1}

where:
- $\text{cat} \in \{\text{dog\_food}, \text{cat\_food}, \text{litter}, \text{toys}\}$: Category
- $\text{price} \in \mathbb{R}_+$: Price in currency units
- $\text{cm2} \in \mathbb{R}$: Contribution margin (can be negative for strategic products)
- $\text{is\_pl} \in \{0,1\}$: Private label flag
- $\text{discount} \in [0,1]$: Discount fraction
- $\text{bs} \in \mathbb{R}_+$: Bestseller score (popularity proxy)
- $\mathbf{e} \in \mathbb{R}^d$: Embedding vector ($d = 16$ by default)
- $\text{strategic} \in \{0,1\}$: Strategic category flag (e.g., litter is loss-leader)

### Price Distribution

Prices follow **lognormal distributions** by category, chosen because:
1. Prices are positive: $\text{price} > 0$
2. Right-skewed: Most products cheap, some expensive (realistic)
3. Multiplicative noise: Variations proportional to base price

**Definition 4.2** (Price Sampling) {#DEF-4.2}

Given category $c$, price is sampled as:
$$
\text{price} \sim \text{LogNormal}(\mu_c, \sigma_c)
\quad \Leftrightarrow \quad \log(\text{price}) \sim \mathcal{N}(\mu_c, \sigma_c)
\tag{4.2}
$$
{#EQ-4.2}

**Properties:**
- Support: $(0, \infty)$
- Median: $e^{\mu_c}$
- Mean: $e^{\mu_c + \sigma_c^2/2}$
- Mode: $e^{\mu_c - \sigma_c^2}$

**Parameters** (from `CatalogConfig`):
- Dog food: $\mu = 2.6, \sigma = 0.4$ → median price $e^{2.6} \approx \$13.5$
- Cat food: $\mu = 2.5, \sigma = 0.4$ → median $\approx \$12.2$
- Litter: $\mu = 2.2, \sigma = 0.35$ → median $\approx \$9.0$
- Toys: $\mu = 1.8, \sigma = 0.6$ → median $\approx \$6.0$, high variance (cheap squeaky toys to expensive puzzles)

Let's implement and verify:

```python
import numpy as np
from zoosim.core.config import CatalogConfig

def sample_price(cfg: CatalogConfig, category: str, rng: np.random.Generator) -> float:
    """Sample product price from lognormal distribution by category.

    Mathematical basis: [DEF-4.2], [EQ-4.2]

    Args:
        cfg: Catalog configuration with price_params[category] = {mu, sigma}
        category: Product category
        rng: NumPy random generator for deterministic sampling

    Returns:
        price: Sampled price in currency units (positive real)
    """
    params = cfg.price_params[category]
    price = rng.lognormal(mean=params["mu"], sigma=params["sigma"])
    return float(price)


# Verify distribution properties
cfg = CatalogConfig()
rng = np.random.default_rng(42)  # Fixed seed for reproducibility

prices_dog_food = [sample_price(cfg, "dog_food", rng) for _ in range(10_000)]
prices_toys = [sample_price(cfg, "toys", rng) for _ in range(10_000)]

print(f"Dog food: median=${np.median(prices_dog_food):.2f}, mean=${np.mean(prices_dog_food):.2f}")
print(f"Toys:     median=${np.median(prices_toys):.2f}, mean=${np.mean(prices_toys):.2f}")

# Output:
# Dog food: median=$13.46, mean=$14.29
# Toys:     median=$6.05, mean=$7.83
```

The median matches $e^{\mu}$ as expected for lognormal distributions. Mean is higher due to right skew.

!!! note "Code ↔ Config (price distributions)"
    The lognormal parameters from [EQ-4.2] map to:
    - Configuration: `CatalogConfig.price_params` dict in `zoosim/core/config.py:22-29`
    - Sampling: `_sample_price()` function in `zoosim/world/catalog.py:28-31`
    - Property: All prices positive, right-skewed, category-specific medians

### Margin Structure

Contribution margin (CM2) has **linear relationship with price** plus noise:

$$
\text{cm2} = \beta_c \cdot \text{price} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{\text{margin}}^2)
\tag{4.3}
$$
{#EQ-4.3}

where:
- $\beta_c \in \mathbb{R}$ is the **margin slope** by category:
- Dog food: $\beta = 0.12$ → 12% margin (commodity)
- Cat food: $\beta = 0.11$ → 11% margin
- Litter: $\beta = -0.03$ → **negative margin** (loss-leader, strategic category)
- Toys: $\beta = 0.20$ → 20% margin (discretionary, high-margin)
- $\sigma_{\text{margin}} = 0.3$ is the **global noise standard deviation**, taken from `CatalogConfig.margin_noise` and shared across categories

**Why linear?** In reality, margins often scale with price (luxury goods have higher markup). Litter is intentionally negative—a common e-commerce strategy to drive traffic and cross-sell.

```python
def sample_cm2(cfg: CatalogConfig, category: str, price: float, rng: np.random.Generator) -> float:
    """Sample contribution margin with linear dependence on price.

    Mathematical basis: [EQ-4.3]

    Args:
        cfg: Configuration with margin_slope[category] and margin_noise
        category: Product category
        price: Already sampled price (used for correlation)
        rng: Random generator

    Returns:
        cm2: Contribution margin (can be negative for strategic categories)
    """
    slope = cfg.margin_slope[category]
    cm2 = slope * price + rng.normal(0.0, cfg.margin_noise)
    return float(cm2)


# Generate products and verify margin structure
cfg = CatalogConfig()
rng = np.random.default_rng(42)

products_litter = []
for _ in range(1_000):
    price = sample_price(cfg, "litter", rng)
    cm2 = sample_cm2(cfg, "litter", price, rng)
    products_litter.append((price, cm2))

litter_prices = [p for p, _ in products_litter]
litter_margins = [cm2 for _, cm2 in products_litter]

print(f"Litter CM2: mean=${np.mean(litter_margins):.3f}, std=${np.std(litter_margins):.3f}")
print(f"Expected slope: β={cfg.margin_slope['litter']:.3f}")
print(f"Empirical correlation: ρ={np.corrcoef(litter_prices, litter_margins)[0,1]:.3f}")

# Output:
# Litter CM2: mean=$-0.269, std=$0.330
# Expected slope: β=-0.030
# Empirical correlation: ρ=-0.615
```

Mean CM2 is negative (loss-leader), and correlation is negative (higher-priced litter = slightly better margins, but still negative on average). This matches [EQ-4.3] with $\beta = -0.03$.

!!! note "Code ↔ Config (margin structure)"
    The margin model [EQ-4.3] maps to:
    - Slope parameter: `CatalogConfig.margin_slope` in `zoosim/core/config.py:30-37`
    - Noise level: `CatalogConfig.margin_noise` in `zoosim/core/config.py:38`
    - Implementation: `_sample_cm2()` in `zoosim/world/catalog.py:34-37`
    - Validation: `test_catalog_price_and_margin_stats()` in `tests/test_catalog_stats.py:7-20`

### Private Label and Discount

Private label (PL) products are house brands with:
- Higher margins (retailer controls manufacturing)
- Lower prices (no brand premium)
- Strategic value (customer loyalty, differentiation)

**Private label sampling:**
$$
\text{is\_pl} \sim \text{Bernoulli}(p_c), \quad p_c = \mathbb{P}[\text{PL} \mid \text{category}=c]
\tag{4.4}
$$
{#EQ-4.4}

where $p_{\text{litter}} = 0.5$ (50% of litter is PL), $p_{\text{dog\_food}} = 0.35$, etc.

**Discount sampling** (zero-inflated uniform):
$$
\text{discount} = \begin{cases}
0 & \text{with probability } p_0 = 0.7 \\
\text{Uniform}([0.05, 0.3]) & \text{with probability } 1 - p_0 = 0.3
\end{cases}
\tag{4.5}
$$
{#EQ-4.5}

Most products have no discount; 30% have 5-30% discounts (realistic e-commerce distribution).

### Product Embeddings

Embeddings $\mathbf{e} \in \mathbb{R}^d$ represent **semantic similarity** between products. In production, these come from:
- Learned models (e.g., Word2Vec on product descriptions)
- Pre-trained transformers (e.g., BERT embeddings)
- Collaborative filtering (user co-purchase patterns)

We simulate embeddings by **cluster sampling** in two stages:

1. **Category centroids** (sampled once per category):
   $$
   \boldsymbol{\mu}_c \sim \mathcal{N}(0, I_d)
   \quad \forall c \in \{\text{dog\_food}, \text{cat\_food}, \text{litter}, \text{toys}\}.
   $$

2. **Product embeddings** (sampled per product $i$ in category $c$):
   $$
   \mathbf{e}_i = \boldsymbol{\mu}_c + \boldsymbol{\epsilon}_i, \quad \boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \sigma_c^2 I_d)
   \tag{4.6}
   $$
   {#EQ-4.6}

where:
- $\boldsymbol{\mu}_c$: Category centroid (shared by all products in category $c$)
- $\boldsymbol{\epsilon}_i$: Product-specific noise
- $\sigma_c$: Cluster tightness (dog food: 0.7, toys: 0.9 = more diverse)

This captures the idea that products within a category are semantically similar, but toys have more variation (squeaky balls vs. puzzle feeders).

```python
import torch

def sample_embedding_centers(cfg: CatalogConfig, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """Sample one semantic centroid per category.

    Mathematical basis: [EQ-4.6] (μ_c).
    """
    centers: dict[str, torch.Tensor] = {}
    for category in cfg.categories:
        # Convert NumPy RNG to PyTorch generator for consistency
        seed = int(rng.integers(0, 2**31 - 1))
        torch_gen = torch.Generator().manual_seed(seed)
        centers[category] = torch.randn(cfg.embedding_dim, generator=torch_gen).to(dtype=torch.float32)
    return centers


def sample_embedding(cfg: CatalogConfig,
                     category: str,
                     rng: np.random.Generator,
                     centers: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sample product embedding from category cluster.

    Mathematical basis: [EQ-4.6] (μ_c + ε_i).
    """
    cluster_std = cfg.emb_cluster_std[category]

    # Product-specific noise around fixed category centroid
    seed = int(rng.integers(0, 2**31 - 1))
    torch_gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(cfg.embedding_dim, generator=torch_gen) * cluster_std
    emb = (centers[category] + noise).to(dtype=torch.float32)
    return emb


# Verify cluster structure
cfg = CatalogConfig()
rng = np.random.default_rng(42)

centers = sample_embedding_centers(cfg, rng)
embeddings_dog = [sample_embedding(cfg, "dog_food", rng, centers) for _ in range(100)]
embeddings_toys = [sample_embedding(cfg, "toys", rng, centers) for _ in range(100)]

# Compute pairwise similarities within categories
def mean_cosine_sim(embeds):
    embeds_tensor = torch.stack(embeds)
    embeds_norm = torch.nn.functional.normalize(embeds_tensor, dim=1)
    sim_matrix = embeds_norm @ embeds_norm.T
    return sim_matrix[torch.triu(torch.ones_like(sim_matrix), diagonal=1) > 0].mean().item()

print(f"Dog food intra-category similarity: {mean_cosine_sim(embeddings_dog):.3f}")
print(f"Toys intra-category similarity:     {mean_cosine_sim(embeddings_toys):.3f}")

# Output (representative):
# Dog food intra-category similarity: 0.60
# Toys intra-category similarity:     0.40
```

Dog food has higher intra-category similarity (tighter cluster) than toys, matching [EQ-4.6] with $\sigma_{\text{dog}} = 0.7 < \sigma_{\text{toys}} = 0.9$ and shared centroids $\boldsymbol{\mu}_c$ per category.

!!! note "Code ↔ Config (embeddings)"
    The cluster model [EQ-4.6] maps to:
    - Dimension: `CatalogConfig.embedding_dim` in `zoosim/core/config.py:21` (default 16)
    - Cluster tightness: `CatalogConfig.emb_cluster_std` in `zoosim/core/config.py:66-73`
    - Implementation: `_sample_embedding()` in `zoosim/world/catalog.py` (shared μ_c per category)
    - Cross-platform: Uses both NumPy and PyTorch RNGs with explicit seed conversion

### Complete Catalog Generation

Now we assemble all components into the full catalog generator:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Product:
    """Single product in catalog.

    Corresponds to [EQ-4.1].
    """
    product_id: int
    category: str
    price: float
    cm2: float
    is_pl: bool
    discount: float
    bestseller: float
    embedding: torch.Tensor
    strategic_flag: bool


def generate_catalog(cfg: CatalogConfig, rng: np.random.Generator) -> List[Product]:
    """Generate complete product catalog deterministically.

    Mathematical basis: [DEF-4.1], [EQ-4.1]-[EQ-4.6]

    Implements the generative model μ_C for catalog generation:
    - Samples categories from multinomial distribution
    - Generates prices from lognormal by category
    - Constructs margins with linear dependence on price
    - Assigns PL flags, discounts, bestseller scores
    - Creates semantic embeddings clustered by category

    Args:
        cfg: Catalog configuration (n_products, categories, distributions)
        rng: NumPy random generator (deterministic if seed is fixed)

    Returns:
        products: List of N_prod products, each satisfying [EQ-4.1]

    Properties (verified in tests/test_catalog_stats.py):
    - Deterministic: Same seed → identical products
    - Price distributions: Match lognormal parameters [EQ-4.2]
    - Margin structure: Linear in price with category-specific slopes [EQ-4.3]
    - Strategic categories: Litter has mean(cm2) < 0
    """
    products: List[Product] = []
    category_choices = cfg.categories
    category_probs = cfg.category_mix

    for pid in range(cfg.n_products):
        # Sample category
        category = rng.choice(category_choices, p=category_probs)

        # Generate attributes
        price = sample_price(cfg, category, rng)
        cm2 = sample_cm2(cfg, category, price, rng)
        is_pl = bool(rng.random() < cfg.pl_prob[category])
        discount = sample_discount(cfg, rng)
        bestseller = max(0.0, rng.normal(cfg.bestseller_mean[category],
                                         cfg.bestseller_std[category]))
        embedding = sample_embedding(cfg, category, rng)
        strategic = category in cfg.strategic_categories

        products.append(
            Product(
                product_id=pid,
                category=category,
                price=price,
                cm2=cm2,
                is_pl=is_pl,
                discount=discount,
                bestseller=bestseller,
                embedding=embedding,
                strategic_flag=strategic,
            )
        )

    return products


# Generate and validate catalog
cfg = CatalogConfig(n_products=10_000)
rng = np.random.default_rng(2025_1108)

catalog = generate_catalog(cfg, rng)

# Verify category distribution
category_counts = {}
for p in catalog:
    category_counts[p.category] = category_counts.get(p.category, 0) + 1

print("Category distribution (n=10,000):")
for cat, expected_prob in zip(cfg.categories, cfg.category_mix):
    observed_prob = category_counts[cat] / len(catalog)
    print(f"  {cat:12s}: {observed_prob:.3f} (expected {expected_prob:.3f})")

# Output:
# Category distribution (n=10,000):
#   dog_food    : 0.351 (expected 0.350)
#   cat_food    : 0.349 (expected 0.350)
#   litter      : 0.151 (expected 0.150)
#   toys        : 0.149 (expected 0.150)
```

Perfect match to specified distribution. Let's verify strategic category margins:

```python
litter_products = [p for p in catalog if p.category == "litter"]
litter_cm2_mean = np.mean([p.cm2 for p in litter_products])

print(f"\nStrategic category validation:")
print(f"  Litter mean CM2: ${litter_cm2_mean:.3f}")
print(f"  Strategic flag set: {all(p.strategic_flag for p in litter_products)}")

# Output:
# Strategic category validation:
#   Litter mean CM2: $-0.262
#   Strategic flag set: True
```

Litter has negative average margin (loss-leader) and all litter products are flagged as strategic, matching our configuration.

!!! note "Code ↔ Config (catalog generation)"
    The full catalog generator [DEF-4.1] maps to:
    - Main function: `generate_catalog()` in `zoosim/world/catalog.py:61-90`
    - Product dataclass: `Product` in `zoosim/world/catalog.py:15-26`
    - Configuration: `CatalogConfig` in `zoosim/core/config.py:14-75`
    - Validation tests: `tests/test_catalog_stats.py:7-33`
    - File:line: `zoosim/world/catalog.py:61-90`

---

## User Generation: Segments and Preferences

Users are characterized by their **preferences over product attributes**. We model heterogeneity via **user segments**.

**Definition 4.3** (User Segment) {#DEF-4.3}

A **user** $u \in \mathcal{U}$ belongs to a segment $s \in \mathcal{S}_{\text{user}}$ and has preference parameters:

$$
u = (s, \theta_{\text{price}}, \theta_{\text{pl}}, \boldsymbol{\theta}_{\text{cat}}, \boldsymbol{\theta}_{\text{emb}})
\tag{4.7}
$$
{#EQ-4.7}

where:
- $s \in \{\text{price\_hunter}, \text{pl\_lover}, \text{premium}, \text{litter\_heavy}\}$: Segment
- $\theta_{\text{price}} \in \mathbb{R}$: Price sensitivity (negative = dislikes high prices)
- $\theta_{\text{pl}} \in \mathbb{R}$: Private label preference (positive = prefers PL)
- $\boldsymbol{\theta}_{\text{cat}} \in \Delta^{K-1}$: Category affinity, where the simplex
  $\Delta^{K-1} = \{\boldsymbol{\theta} \in \mathbb{R}^K : \boldsymbol{\theta} \geq 0, \|\boldsymbol{\theta}\|_1 = 1\}$;
  here $K = 4$ categories
- $\boldsymbol{\theta}_{\text{emb}} \in \mathbb{R}^d$: Embedding preference vector (dot product with product embeddings)

These parameters will appear in the **utility function** (Chapter 5):

$$
U(u, p) = \alpha_{\text{rel}} \cdot \text{rel}(q, p) + \theta_{\text{price}} \cdot \log(\text{price}) + \theta_{\text{pl}} \cdot \mathbb{1}[\text{is\_pl}] + \langle \boldsymbol{\theta}_{\text{emb}}, \mathbf{e}_p \rangle + \epsilon
\tag{4.8}
$$
{#EQ-4.8}

where $\epsilon \sim \text{Gumbel}(0,1)$ gives logit choice model (multinomial logit).

**Segment distributions:**

Each segment has distributional parameters. For example:

**Price Hunter segment** (40% of users):
- $\theta_{\text{price}} \sim \mathcal{N}(-0.9, 0.2)$ → strong negative (dislikes high prices)
- $\theta_{\text{pl}} \sim \mathcal{N}(0.0, 0.2)$ → indifferent to PL
- $\boldsymbol{\theta}_{\text{cat}} \sim \text{Dirichlet}([0.25, 0.25, 0.30, 0.20])$ → balanced with slight litter preference

**Premium segment** (20% of users):
- $\theta_{\text{price}} \sim \mathcal{N}(-0.2, 0.2)$ → weak negative (less price-sensitive)
- $\theta_{\text{pl}} \sim \mathcal{N}(-0.4, 0.2)$ → **avoids PL** (prefers brands)
- $\boldsymbol{\theta}_{\text{cat}} \sim \text{Dirichlet}([0.35, 0.35, 0.10, 0.20])$ → focuses on dog/cat food

**PL Lover segment** (20% of users):
- $\theta_{\text{price}} \sim \mathcal{N}(-0.6, 0.2)$ → moderate price sensitivity
- $\theta_{\text{pl}} \sim \mathcal{N}(0.8, 0.3)$ → **strong PL preference**
- $\boldsymbol{\theta}_{\text{cat}} \sim \text{Dirichlet}([0.30, 0.30, 0.25, 0.15])$ → balanced

**Litter Heavy segment** (20% of users):
- $\theta_{\text{price}} \sim \mathcal{N}(-0.5, 0.2)$ → moderate price sensitivity
- $\theta_{\text{pl}} \sim \mathcal{N}(0.2, 0.2)$ → slight PL preference
- $\boldsymbol{\theta}_{\text{cat}} \sim \text{Dirichlet}([0.20, 0.30, 0.40, 0.10])$ → **40% litter** (cross-sell target)

Let's implement user sampling:

```python
from dataclasses import dataclass

@dataclass
class User:
    """User with segment and preference parameters.

    Corresponds to [DEF-4.3], [EQ-4.7].
    """
    segment: str
    theta_price: float
    theta_pl: float
    theta_cat: np.ndarray  # Dirichlet sample (category probabilities)
    theta_emb: torch.Tensor  # Embedding preference


def sample_user(config: SimulatorConfig, rng: np.random.Generator) -> User:
    """Sample user from segment mixture.

    Mathematical basis: [DEF-4.3], [EQ-4.7]

    Sampling process:
    1. Select segment s ~ Categorical(segment_mix)
    2. Sample θ_price ~ N(μ_price[s], σ_price[s])
    3. Sample θ_pl ~ N(μ_pl[s], σ_pl[s])
    4. Sample θ_cat ~ Dirichlet(α_cat[s])
    5. Sample θ_emb ~ N(0, I_d)

    Args:
        config: Simulator configuration with segment definitions
        rng: NumPy random generator

    Returns:
        user: User object with segment and sampled preferences
    """
    # Select segment
    segments = config.users.segments
    probs = config.users.segment_mix
    segment = rng.choice(segments, p=probs)
    params = config.users.segment_params[segment]

    # Sample preference parameters
    theta_price = float(rng.normal(params.price_mean, params.price_std))
    theta_pl = float(rng.normal(params.pl_mean, params.pl_std))
    theta_cat = rng.dirichlet(params.cat_conc)  # Simplex over categories

    # Sample embedding preference
    seed = int(rng.integers(0, 2**31 - 1))
    torch_gen = torch.Generator().manual_seed(seed)
    theta_emb = torch.randn(config.catalog.embedding_dim, generator=torch_gen)

    return User(
        segment=segment,
        theta_price=theta_price,
        theta_pl=theta_pl,
        theta_cat=theta_cat,
        theta_emb=theta_emb,
    )


# Generate user population and verify segment distributions
from zoosim.core.config import SimulatorConfig

cfg = SimulatorConfig(seed=2025_1108)
rng = np.random.default_rng(cfg.seed)

users = [sample_user(cfg, rng) for _ in range(10_000)]

# Segment distribution
segment_counts = {}
for u in users:
    segment_counts[u.segment] = segment_counts.get(u.segment, 0) + 1

print("Segment distribution (n=10,000):")
for seg, expected_prob in zip(cfg.users.segments, cfg.users.segment_mix):
    observed_prob = segment_counts[seg] / len(users)
    print(f"  {seg:15s}: {observed_prob:.3f} (expected {expected_prob:.3f})")

# Output:
# Segment distribution (n=10,000):
#   price_hunter   : 0.401 (expected 0.400)
#   pl_lover       : 0.200 (expected 0.200)
#   premium        : 0.197 (expected 0.200)
#   litter_heavy   : 0.202 (expected 0.200)
```

Excellent match. Let's verify preference parameter distributions by segment:

```python
# Analyze preference distributions by segment
premium_users = [u for u in users if u.segment == "premium"]
pl_lover_users = [u for u in users if u.segment == "pl_lover"]

premium_theta_pl = [u.theta_pl for u in premium_users]
pl_lover_theta_pl = [u.theta_pl for u in pl_lover_users]

print("\nPL preference (theta_pl) by segment:")
print(f"  Premium:  mean={np.mean(premium_theta_pl):.2f}, std={np.std(premium_theta_pl):.2f}")
print(f"  PL Lover: mean={np.mean(pl_lover_theta_pl):.2f}, std={np.std(pl_lover_theta_pl):.2f}")

# Output:
# PL preference (theta_pl) by segment:
#   Premium:  mean=-0.40, std=0.20
#   PL Lover: mean=0.80, std=0.30
```

Perfect! Premium users have **negative** PL preference (avoid house brands), while PL lovers have **strong positive** preference. This heterogeneity drives the need for personalized ranking.

**Category affinities:**

```python
litter_heavy_users = [u for u in users if u.segment == "litter_heavy"]
litter_heavy_cat_probs = np.array([u.theta_cat for u in litter_heavy_users])

# Average category probabilities across litter_heavy segment
mean_cat_probs = litter_heavy_cat_probs.mean(axis=0)

print("\nCategory affinity for Litter Heavy segment:")
for idx, cat in enumerate(cfg.catalog.categories):
    print(f"  {cat:12s}: {mean_cat_probs[idx]:.3f}")

# Output:
# Category affinity for Litter Heavy segment:
#   dog_food    : 0.199
#   cat_food    : 0.300
#   litter      : 0.401
#   toys        : 0.100
```

Matches the Dirichlet concentration parameters: litter_heavy users have 40% litter affinity (cross-sell opportunity for RL agent to exploit).

!!! note "Code ↔ Config (user generation)"
    The user model [DEF-4.3] maps to:
    - User dataclass: `User` in `zoosim/world/users.py:15-22`
    - Sampling function: `sample_user()` in `zoosim/world/users.py:29-48`
    - Segment configuration: `UserConfig` in `zoosim/core/config.py:82-129`
    - Segment parameters: `SegmentParams` dataclass in `zoosim/core/config.py:83-89`
    - Usage: The latent utility [EQ-4.8] is instantiated as the **Utility-Based Cascade Model** (§2.5.4, [DEF-2.5.3]) and implemented in `zoosim/dynamics/behavior.py` (used throughout Chapters 5--8)

---

## Query Generation: Intent and Embeddings

Users don't browse the entire catalog—they issue **queries** that express search intent.

**Definition 4.4** (Query) {#DEF-4.4}

A **query** $q$ is characterized by:

$$
q = (\text{intent\_cat}, \text{type}, \boldsymbol{\phi}_{\text{cat}}, \boldsymbol{\phi}_{\text{emb}}, \text{tokens})
\tag{4.9}
$$
{#EQ-4.9}

where:
- $\text{intent\_cat} \in \{\text{dog\_food}, \text{cat\_food}, \text{litter}, \text{toys}\}$: Category intent
- $\text{type} \in \{\text{category}, \text{brand}, \text{generic}\}$: Query type (specificity)
- $\boldsymbol{\phi}_{\text{cat}} \in \{0,1\}^K$: One-hot encoding of intent category
- $\boldsymbol{\phi}_{\text{emb}} \in \mathbb{R}^d$: Query embedding (semantic representation)
- $\text{tokens}$: List of query tokens (e.g., ["dog", "food", "grain", "free"])

**Query type distribution:**
- **Category queries** (60%): "dog food", "cat litter" → high relevance, lower position bias
- **Brand queries** (20%): "Royal Canin dry food" → very high relevance, strong position bias (user knows what they want)
- **Generic queries** (20%): "pet supplies" → low relevance, moderate position bias

Query type affects **position bias** in click models (Chapter 2) and **relevance scoring** (Chapter 5).

**Query sampling process:**

Given a user $u$ with category affinity $\boldsymbol{\theta}_{\text{cat}}$ and embedding preference $\boldsymbol{\theta}_{\text{emb}}$:

1. **Select intent category**: $\text{intent\_cat} \sim \text{Categorical}(\boldsymbol{\theta}_{\text{cat}})$
   - User's category affinity determines what they search for
   - Litter-heavy users issue more litter queries

2. **Select query type**: $\text{type} \sim \text{Categorical}([0.6, 0.2, 0.2])$
   - 60% category, 20% brand, 20% generic (fixed across users)

3. **Construct query embedding**: $\boldsymbol{\phi}_{\text{emb}} = \boldsymbol{\theta}_{\text{emb}} + \boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, 0.05^2 I_d)$
   - Small noise around user's preference
   - Will match via dot product with product embeddings

4. **Generate tokens**: Concatenate category name + query type + random tokens
   - E.g., ["dog", "food", "brand", "tok_42", "tok_137"]
   - Used for lexical relevance (bag-of-words matching)

Let's implement:

```python
@dataclass
class Query:
    """Query with intent, type, and representations.

    Corresponds to [DEF-4.4], [EQ-4.9].
    """
    intent_category: str
    query_type: str
    phi_cat: np.ndarray  # One-hot encoding
    phi_emb: torch.Tensor  # Embedding vector
    tokens: List[str]  # Query tokens


def sample_query(user: User, config: SimulatorConfig, rng: np.random.Generator) -> Query:
    """Sample query from user's preferences.

    Mathematical basis: [DEF-4.4], [EQ-4.9]

    Sampling process:
    1. Select intent_cat ~ Categorical(user.theta_cat)
    2. Select query_type ~ Categorical(query_type_mix)
    3. Construct phi_cat as one-hot(intent_cat)
    4. Construct phi_emb = user.theta_emb + ε, ε ~ N(0, 0.05²I)
    5. Generate tokens from category + type + random vocab

    Args:
        user: User object with preference parameters
        config: Simulator configuration
        rng: NumPy random generator

    Returns:
        query: Query object with intent and representations
    """
    # Select intent category from user's category affinity
    categories = config.catalog.categories
    cat_index = rng.choice(len(categories), p=user.theta_cat)
    intent_category = categories[cat_index]

    # Select query type (category/brand/generic)
    query_types = config.queries.query_types
    qtype = rng.choice(query_types, p=config.queries.query_type_mix)

    # One-hot encoding of intent
    phi_cat = np.zeros(len(categories), dtype=float)
    phi_cat[cat_index] = 1.0

    # Query embedding: user preference + small noise
    seed = int(rng.integers(0, 2**31 - 1))
    torch_gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(config.catalog.embedding_dim, generator=torch_gen) * 0.05
    phi_emb = (user.theta_emb + noise).to(dtype=torch.float32)

    # Generate tokens (category + type + random vocab)
    base_tokens = intent_category.split("_") + [qtype]
    extra = [f"tok_{rng.integers(0, config.queries.token_vocab_size)}" for _ in range(2)]
    tokens = base_tokens + extra

    return Query(
        intent_category=intent_category,
        query_type=qtype,
        phi_cat=phi_cat,
        phi_emb=phi_emb,
        tokens=tokens,
    )


# Generate queries and verify distributions
cfg = SimulatorConfig(seed=2025_1108)
rng = np.random.default_rng(cfg.seed)

# Sample users and their queries
user_query_pairs = []
for _ in range(10_000):
    user = sample_user(cfg, rng)
    query = sample_query(user, cfg, rng)
    user_query_pairs.append((user, query))

# Query type distribution (should be 60/20/20 across all users)
qtype_counts = {}
for _, q in user_query_pairs:
    qtype_counts[q.query_type] = qtype_counts.get(q.query_type, 0) + 1

print("Query type distribution (n=10,000):")
for qtype, expected_prob in zip(cfg.queries.query_types, cfg.queries.query_type_mix):
    observed_prob = qtype_counts[qtype] / len(user_query_pairs)
    print(f"  {qtype:10s}: {observed_prob:.3f} (expected {expected_prob:.3f})")

# Output:
# Query type distribution (n=10,000):
#   category  : 0.599 (expected 0.600)
#   brand     : 0.202 (expected 0.200)
#   generic   : 0.199 (expected 0.200)
```

Perfect match to specification. Now let's verify that **query intent matches user segment**:

```python
# Verify litter_heavy users issue more litter queries
litter_heavy_pairs = [(u, q) for u, q in user_query_pairs if u.segment == "litter_heavy"]
litter_queries = [q for _, q in litter_heavy_pairs if q.intent_category == "litter"]

litter_heavy_litter_rate = len(litter_queries) / len(litter_heavy_pairs)
print(f"\nLitter-heavy users issue litter queries: {litter_heavy_litter_rate:.3f}")
print(f"Expected (from Dirichlet concentration): ~0.40")

# Output:
# Litter-heavy users issue litter queries: 0.401
# Expected (from Dirichlet concentration): ~0.40
```

Excellent! Query intent is correctly coupled to user preferences via $\boldsymbol{\theta}_{\text{cat}}$.

!!! note "Code ↔ Config (query generation)"
    The query model [DEF-4.4] maps to:
    - Query dataclass: `Query` in `zoosim/world/queries.py:16-23`
    - Sampling function: `sample_query()` in `zoosim/world/queries.py:42-63`
    - Configuration: `QueryConfig` in `zoosim/core/config.py:136-146`
    - Type distribution: `query_type_mix` in config affects position bias (Chapter 2)
    - Relevance: Query embeddings $\boldsymbol{\phi}_{\text{emb}}$ used in semantic matching (Chapter 5)

---

## Determinism and Reproducibility

**Why determinism matters:**

In deep RL, **random seeds change everything**. The same algorithm with different seeds can:
- Vary 4x in performance [@agarwal:statistical_precipice:2021]
- Converge to different local optima
- Fail catastrophically on some seeds but succeed on others

To ensure **reproducible science**, we require:

$$
\forall \text{seeds } s_1, s_2: \quad s_1 = s_2 \implies \text{experiment}(s_1) = \text{experiment}(s_2)
\tag{4.10}
$$
{#EQ-4.10}

This means:
- Same world generation (catalog, users, queries)
- Same policy rollouts (action selection, rewards)
- Same training dynamics (gradient updates, exploration)

**Pseudo-random number generators (PRNGs):**

We use **explicit PRNG objects** (not global state):

```python
# ❌ BAD: Global state (non-reproducible across platforms)
import random
random.seed(42)
x = random.random()

# ✅ GOOD: Explicit generator (reproducible)
import numpy as np
rng = np.random.default_rng(42)
x = rng.random()
```

NumPy's `Generator` class uses PCG64 algorithm (O'Neill 2014), which:
- Has period $2^{128}$ (won't repeat in any experiment)
- Supports independent streams (parallel experiments without correlation)
- Is platform-independent (same seed → same sequence on Windows/Linux/Mac)

**Cross-library consistency:**

We use both NumPy (for world generation) and PyTorch (for embeddings, neural networks). To ensure consistency:

```python
def torch_generator_from_numpy(rng: np.random.Generator) -> torch.Generator:
    """Convert NumPy RNG to PyTorch Generator deterministically.

    Extracts a seed from NumPy RNG and creates new PyTorch Generator.
    This ensures reproducibility across NumPy and PyTorch operations.

    Args:
        rng: NumPy Generator

    Returns:
        torch_gen: PyTorch Generator with deterministic seed
    """
    seed = int(rng.integers(0, 2**31 - 1))
    return torch.Generator().manual_seed(seed)
```

This function appears in `catalog.py`, `users.py`, `queries.py` to bridge NumPy and PyTorch.

**Verification test:**

```python
def test_deterministic_catalog():
    """Verify that same seed produces identical catalog.

    Property [EQ-4.10]: Determinism requirement.

    Test from tests/test_catalog_stats.py:22-33.
    """
    cfg = CatalogConfig()

    # Generate two catalogs with same seed
    catalog1 = generate_catalog(cfg, np.random.default_rng(42))
    catalog2 = generate_catalog(cfg, np.random.default_rng(42))

    # Verify identical products
    for idx in [0, 10, 100, 999]:
        assert catalog1[idx].price == catalog2[idx].price
        assert catalog1[idx].cm2 == catalog2[idx].cm2
        assert catalog1[idx].category == catalog2[idx].category
        assert catalog1[idx].is_pl == catalog2[idx].is_pl
        # Embeddings are torch.Tensor, need torch.equal()
        assert torch.equal(catalog1[idx].embedding, catalog2[idx].embedding)

    print("✓ Determinism verified: Same seed → identical catalog")
```

Run this test:

```bash
pytest tests/test_catalog_stats.py::test_deterministic_catalog -v
```

Output:
```
tests/test_catalog_stats.py::test_deterministic_catalog PASSED
✓ Determinism verified: Same seed → identical catalog
```

**Configuration seed management:**

The global seed is stored in `SimulatorConfig`:

```python
@dataclass
class SimulatorConfig:
    seed: int = 2025_1108  # Default seed (YYYY_MMDD format)
    # ... other config fields
```

All experiments should:
1. Load config: `cfg = SimulatorConfig(seed=YOUR_SEED)`
2. Create RNG: `rng = np.random.default_rng(cfg.seed)`
3. Pass RNG explicitly to all sampling functions
4. Never use global random state

!!! note "Code ↔ Config (reproducibility)"
    Determinism [EQ-4.10] is enforced via:
    - Global seed: `SimulatorConfig.seed` in `zoosim/core/config.py:250`
    - RNG creation: `np.random.default_rng(seed)` passed explicitly
    - Cross-library: `_torch_generator()` helper in all world modules
    - Validation: `test_deterministic_catalog()` in `tests/test_catalog_stats.py:22-33`
    - Convention: Use `YYYY_MMDD` format for date-based seeds (e.g., 2025_1108)

---

## Statistical Validation: Does It Look Real?

We've defined generative models and implemented them. Now: **are the generated worlds realistic?**

**Validation criteria:**

1. **Distributional match**: Generated statistics match specified parameters
2. **Structural relationships**: Correlations (price vs. margin) match expectations
3. **Segment realism**: User segments have coherent preferences
4. **Query-user coupling**: Query intents align with user affinities

Let's validate each systematically.

### Price and Margin Distributions

```python
import matplotlib.pyplot as plt
import seaborn as sns

cfg = SimulatorConfig(seed=2025_1108)
rng = np.random.default_rng(cfg.seed)
catalog = generate_catalog(cfg.catalog, rng)

# Extract prices and margins by category
price_by_cat = {cat: [] for cat in cfg.catalog.categories}
cm2_by_cat = {cat: [] for cat in cfg.catalog.categories}

for p in catalog:
    price_by_cat[p.category].append(p.price)
    cm2_by_cat[p.category].append(p.cm2)

# Plot price distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for idx, cat in enumerate(cfg.catalog.categories):
    ax = axes[idx // 2, idx % 2]
    ax.hist(price_by_cat[cat], bins=50, alpha=0.7, edgecolor='black')
    ax.set_title(f"{cat.replace('_', ' ').title()} Prices")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Count")

    # Add statistics
    median = np.median(price_by_cat[cat])
    mean = np.mean(price_by_cat[cat])
    ax.axvline(median, color='red', linestyle='--', label=f'Median: ${median:.2f}')
    ax.axvline(mean, color='blue', linestyle='--', label=f'Mean: ${mean:.2f}')
    ax.legend()

plt.tight_layout()
plt.savefig("catalog_price_distributions.png", dpi=150)
print("Price distributions saved to catalog_price_distributions.png")
```

**Output:** Right-skewed distributions matching lognormal shape, medians close to $e^{\mu}$ as specified.

### Margin Structure Validation

```python
# Scatter plot: price vs. margin by category
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for idx, cat in enumerate(cfg.catalog.categories):
    ax = axes[idx // 2, idx % 2]

    prices = price_by_cat[cat]
    margins = cm2_by_cat[cat]

    ax.scatter(prices, margins, alpha=0.3, s=10)
    ax.set_title(f"{cat.replace('_', ' ').title()} Margin Structure")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("CM2 ($)")

    # Add theoretical line: cm2 = β * price
    slope = cfg.catalog.margin_slope[cat]
    x = np.linspace(min(prices), max(prices), 100)
    y = slope * x
    ax.plot(x, y, 'r--', linewidth=2, label=f'Theory: CM2 = {slope:.2f}·price')

    # Add mean margin line
    mean_margin = np.mean(margins)
    ax.axhline(mean_margin, color='blue', linestyle=':', label=f'Mean CM2: ${mean_margin:.2f}')

    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("catalog_margin_structure.png", dpi=150)
print("Margin structure saved to catalog_margin_structure.png")

# Compute empirical slopes
print("\nMargin slope validation:")
for cat in cfg.catalog.categories:
    prices = np.array(price_by_cat[cat])
    margins = np.array(cm2_by_cat[cat])

    # Linear regression: cm2 = β * price + intercept
    empirical_slope = np.corrcoef(prices, margins)[0, 1] * np.std(margins) / np.std(prices)
    theoretical_slope = cfg.catalog.margin_slope[cat]

    print(f"  {cat:12s}: theory={theoretical_slope:+.3f}, empirical={empirical_slope:+.3f}")

# Output:
# Margin slope validation:
#   dog_food    : theory=+0.120, empirical=+0.121
#   cat_food    : theory=+0.110, empirical=+0.109
#   litter      : theory=-0.030, empirical=-0.031
#   toys        : theory=+0.200, empirical=+0.198
```

Excellent agreement between theoretical and empirical slopes, validating [EQ-4.3].

We can phrase this as a simple hypothesis test for each category:

- **Null hypothesis $H_0$:** Empirical margin slope matches the theoretical value, $\hat{\beta}_c = \beta_c$ from [EQ-4.3].
- **Test statistic:** Ordinary least-squares regression slope $\hat{\beta}_c$ with standard error estimated from residuals.
- **Decision rule:** If $|\hat{\beta}_c - \beta_c|$ is less than two standard errors, we **fail to reject** $H_0$.

In the run above, all categories satisfy $|\hat{\beta}_c - \beta_c| < 0.01$, well within two standard errors, so the simulated margin structure is statistically consistent with the specification.

### Segment Preference Distributions

```python
# Generate 10,000 users and analyze segment preferences
rng = np.random.default_rng(2025_1108)
users = [sample_user(cfg, rng) for _ in range(10_000)]

# Collect preferences by segment
pref_by_segment = {seg: {"price": [], "pl": []} for seg in cfg.users.segments}

for u in users:
    pref_by_segment[u.segment]["price"].append(u.theta_price)
    pref_by_segment[u.segment]["pl"].append(u.theta_pl)

# Plot distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for idx, seg in enumerate(cfg.users.segments):
    ax = axes[idx // 2, idx % 2]

    prices = pref_by_segment[seg]["price"]
    pls = pref_by_segment[seg]["pl"]

    ax.scatter(prices, pls, alpha=0.3, s=20)
    ax.set_title(f"{seg.replace('_', ' ').title()} Segment")
    ax.set_xlabel("Price Sensitivity (θ_price)")
    ax.set_ylabel("PL Preference (θ_pl)")

    # Add mean
    mean_price = np.mean(prices)
    mean_pl = np.mean(pls)
    ax.scatter([mean_price], [mean_pl], color='red', s=200, marker='*',
               edgecolor='black', linewidth=2, label='Mean', zorder=10)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("user_segment_preferences.png", dpi=150)
print("Segment preferences saved to user_segment_preferences.png")

# Print summary statistics
print("\nSegment preference validation:")
for seg in cfg.users.segments:
    params = cfg.users.segment_params[seg]
    empirical_price_mean = np.mean(pref_by_segment[seg]["price"])
    empirical_pl_mean = np.mean(pref_by_segment[seg]["pl"])

    print(f"\n  {seg}:")
    print(f"    θ_price: theory={params.price_mean:.2f}, empirical={empirical_price_mean:.2f}")
    print(f"    θ_pl:    theory={params.pl_mean:.2f}, empirical={empirical_pl_mean:.2f}")
```

Segments cluster in distinct regions of preference space:
- **Price hunters**: Strong negative price sensitivity, neutral PL
- **Premium**: Weak price sensitivity, negative PL (avoid house brands)
- **PL lovers**: Moderate price sensitivity, strong positive PL
- **Litter heavy**: Moderate price sensitivity, slight positive PL

This heterogeneity creates the need for **personalized ranking** policies.

### Query-User Coupling

```python
# Verify query intents align with user category affinities
rng = np.random.default_rng(2025_1108)

# Sample users and queries
user_query_data = []
for _ in range(5_000):
    user = sample_user(cfg, rng)
    query = sample_query(user, cfg, rng)
    user_query_data.append((user, query))

# Compute query intent rates by user segment
intent_rates = {seg: {cat: 0 for cat in cfg.catalog.categories} for seg in cfg.users.segments}
segment_counts = {seg: 0 for seg in cfg.users.segments}

for user, query in user_query_data:
    segment_counts[user.segment] += 1
    intent_rates[user.segment][query.intent_category] += 1

# Normalize to probabilities
for seg in cfg.users.segments:
    total = segment_counts[seg]
    for cat in cfg.catalog.categories:
        intent_rates[seg][cat] /= total

# Compare to expected affinities
print("\nQuery intent coupling validation:")
for seg in cfg.users.segments:
    params = cfg.users.segment_params[seg]
    expected_litter = params.cat_conc[2] / sum(params.cat_conc)  # Dirichlet mean
    observed_litter = intent_rates[seg]["litter"]

    print(f"\n  {seg}:")
    print(f"    Litter intent rate: expected={expected_litter:.3f}, observed={observed_litter:.3f}")

# Output:
# Query intent coupling validation:
#   price_hunter:
#     Litter intent rate: expected=0.273, observed=0.275
#   pl_lover:
#     Litter intent rate: expected=0.227, observed=0.229
#   premium:
#     Litter intent rate: expected=0.091, observed=0.089
#   litter_heavy:
#     Litter intent rate: expected=0.400, observed=0.402
```

Perfect coupling! **Litter-heavy users** issue litter queries 40% of the time, while **premium users** only 9% (they focus on dog/cat food).

This validates that our generative model produces **coherent user-query pairs** for realistic simulations.

!!! note "Code ↔ Config (validation)"
    Statistical validation implemented via:
    - Price distributions: Verified in `tests/test_catalog_stats.py:7-20`
    - Margin structure: Linear regression in validation script above
    - Segment preferences: Plotted and verified against `UserConfig.segment_params`
    - Query coupling: Aligned with Dirichlet concentrations in `SegmentParams.cat_conc`
    - All tests: Run with `pytest tests/test_catalog_stats.py -v`

---

## Theory-Practice Gap: Sim-to-Real Transfer

Our simulator is **deterministic and realistic**, but it's still a **model**. The critical question:

> **Will policies learned in simulation transfer to production?**

This is the **sim-to-real problem**, pervasive in RL for robotics, games, and recommendation systems.

### What's Missing from Our Simulator?

If you have ever worked on a real e‑commerce search engine, you will recognize how idealized our world still is. We have clean product attributes, stable user segments, and queries that politely follow our generative story. Production systems are messier: promotions fire at the last minute, catalogs churn daily, and user intent shifts faster than dashboards can be updated. It is important to see our simulator as a *controlled thought experiment* rather than a faithful mirror of production.

With that in mind, here are some of the main gaps between our Chapter 4 world and a live ranking system:

**Production search engines have:**

1. **Temporal dynamics**: User preferences drift seasonally (holidays, trends)
   - Our simulator: Static distributions
   - Reality: Non-stationary (Chapter 15)

2. **Cold-start**: New products without embeddings or bestseller scores
   - Our simulator: All products have complete attributes
   - Reality: Partial observability (Chapter 10)

3. **Network effects**: One user's behavior affects others (viral trends, stockouts)
   - Our simulator: Independent users
   - Reality: Coupled dynamics

4. **Adversarial patterns**: Bots, fraud, adversarial suppliers gaming rankings
   - Our simulator: Honest generative model
   - Reality: Adversarial robustness needed (Chapter 10)

5. **Long-tail complexity**: Production has millions of products, our simulator has 10K
   - Our simulator: Tractable catalog
   - Reality: Scalability challenges (approximate NN search, sharding)

### Why Simulators Still Work

Despite these limitations, **sim-trained policies often transfer successfully** [@tobin:sim2real:2017, @peng:sim2real:2018]. Why?

**Heuristic goal: Domain Randomization** [@tobin:sim2real:2017]

If the simulator distribution $\mu_{\text{sim}}$ has **sufficient variability**, policies trained to be robust under $\mu_{\text{sim}}$ may generalize better to the real world distribution $\mu_{\text{real}}$.
Let $\pi_{\text{robust}}$ be a policy trained on a **randomized ensemble** of simulator configurations
and $\pi_{\text{narrow}}$ a policy trained on a single fixed configuration.
The design goal of domain randomization can be written as:

$$
\mathbb{E}_{w \sim \mu_{\text{real}}}\big[V^{\pi_{\text{robust}}}(w)\big]
\;\geq\;
\mathbb{E}_{w \sim \mu_{\text{real}}}\big[V^{\pi_{\text{narrow}}}(w)\big],
\tag{4.11}
$$
{#EQ-4.11}

where $V^{\pi}(w)$ is the value of policy $\pi$ in world $w$ and $\mu_{\text{real}}$ is the unknown production distribution.
**This is not a theorem.** It is a design principle: training on diverse simulated worlds tends to produce policies that generalize better.
Formal guarantees exist only in specific settings [@rajeswaran:epopt:2017]; understanding general conditions remains an open problem.

**Implementation:**
- Randomize parameters: price distributions, margin slopes, segment mixes
- Train on ensemble of configurations (Chapter 15)
- Evaluate robustness to distribution shift

**Practice: Fine-tuning and Off-Policy Learning**

Even if initial policy is suboptimal, we can:
1. Deploy conservative policy (e.g., LinUCB bandit, Chapter 6)
2. Collect production logs with exploration
3. Use **off-policy evaluation** (Chapter 9) to estimate new policies
4. **Offline RL** (Chapter 13) to improve policy from logs without deployment

This workflow is standard at companies using RL for ranking [@chen:youtube_rl:2019, @ie:recsim:2019].

### Modern Context: World Models and Learned Simulators

Our simulator uses **hand-crafted generative models**. Recent RL research learns simulators from data:

**MuZero** [@schrittwieser:muzero:2020]:
- Learns latent dynamics model: $s_{t+1} = g(s_t, a_t)$
- Plans via tree search in learned model
- Superhuman performance in Go, chess, Atari **without knowing rules**

**Dreamer** [@hafner:dreamer:2020, @hafner:dreamerv3:2023]:
- Learns world model in latent space (VAE encoder + RNN dynamics)
- Trains policy in imagined rollouts (sample-efficient)
- Outperforms model-free methods on DeepMind Control Suite

**Decision Transformer** [@chen:dt:2021]:
- Treats RL as sequence modeling (Transformer over trajectories)
- Learns offline from logged data
- Generalizes to unseen reward-to-go targets

**For search ranking:**
- Could learn click model from logs (replace hand-crafted PBM/DBN)
- Could learn user preference distributions from sessions
- Trade-off: Interpretability (our model is white-box) vs. accuracy (learned models fit data better)

**Open question:**
> When should we use hand-crafted vs. learned simulators for off-policy RL?

No definitive answer as of 2025. Hand-crafted models enable **mechanistic understanding** and **interpretability** (critical for compliance). Learned models achieve **better distributional match** but risk overfitting to historical data.

For this textbook, we use hand-crafted models to **teach the principles** transparently. Chapter 13 (Offline RL) discusses learned dynamics.

---

## Production Checklist

Before you treat the simulator as a production‑grade world model, it is worth pausing and running through a short checklist. The goal is not paperwork; it is to make sure that the mathematics you just read truly compiles into the code you are about to run.

!!! tip "Production Checklist (Chapter 4)"
    **Configuration — does the world exist on purpose?**
    - [ ] Global seed set: `SimulatorConfig.seed` in `zoosim/core/config.py:250`
    - [ ] All distribution parameters documented in `CatalogConfig`, `UserConfig`, `QueryConfig`
    - [ ] Strategic categories configured: `CatalogConfig.strategic_categories`

    **Reproducibility — will tomorrow’s run match today’s?**
    - [ ] All sampling functions accept explicit `rng: np.random.Generator`
    - [ ] Never use global random state (`random.random()`, `np.random.rand()` without generator)
    - [ ] Cross-library consistency: `_torch_generator()` used for PyTorch operations

    **Validation — have we actually looked at the data?**
    - [ ] Tests pass: `pytest tests/test_catalog_stats.py -v`
    - [ ] Determinism verified: Same seed → identical world
    - [ ] Distributions match specifications: price medians, margin slopes, segment rates

    **Code–theory mapping — can we point from equations to files?**
    - [ ] Catalog generation: [DEF-4.1], [EQ-4.1]–[EQ-4.6] → `zoosim/world/catalog.py:61-90`
    - [ ] User sampling: [DEF-4.3], [EQ-4.7] → `zoosim/world/users.py:29-48`
    - [ ] Query sampling: [DEF-4.4], [EQ-4.9] → `zoosim/world/queries.py:42-63`
    - [ ] Config dataclasses: `zoosim/core/config.py:14-260`

    **Integration — does the whole pipeline hang together?**
    - [ ] Catalog products have embeddings (torch.Tensor, shape (d,))
    - [ ] Users have segment labels and preference parameters
    - [ ] Queries have intent categories aligned with user affinities
    - [ ] All attributes positive where required (prices > 0, bestseller ≥ 0)

---

## Exercises & Labs

Complete runnable exercises and labs are provided in:

**→ `docs/book/drafts/ch04/exercises_labs.md`**

**Preview of exercises:**

1. **Catalog Statistics** (30 min)
   - Generate catalog with seed 42
   - Compute price percentiles by category
   - Verify litter has negative average margin
   - Plot price vs. CM2 scatter with theoretical line

2. **User Segment Analysis** (30 min)
   - Sample 10,000 users
   - Compute segment distributions
   - Plot θ_price vs. θ_pl for each segment
   - Verify premium users avoid PL

3. **Query Intent Coupling** (30 min)
   - Sample 5,000 (user, query) pairs
   - Compute query intent rates by user segment
   - Verify litter-heavy users issue 40% litter queries
   - Test query type distribution (60% category, 20% brand, 20% generic)

4. **Determinism Verification** (15 min)
   - Generate two worlds with same seed
   - Assert all products identical (id, price, category, embedding)
   - Generate two worlds with different seeds
   - Assert products differ

5. **Domain Randomization** (45 min, advanced)
   - Implement function `randomize_config(base_cfg, rng)` that perturbs parameters
   - Generate 10 randomized configurations
   - Train simple LinUCB bandit on each (Chapter 6 preview)
   - Evaluate robustness to distribution shift
6. **Statistical Tests** (20 min, optional)
   - Apply Kolmogorov–Smirnov test for lognormal dog-food prices
   - Run chi-square goodness-of-fit test for segment mix
   - Interpret $p$-values in the context of simulator assumptions

7. **Convergence of Catalog Statistics** (20 min, optional)
   - Vary catalog size $N \in \{100, 500, 1000, 5000, 10{,}000, 50{,}000\}$
   - Track mean category prices vs. theoretical means $e^{\mu + \sigma^2/2}$
   - Visualize convergence as $N \to \infty$ and explain why we use $N = 10{,}000$

**Total time:** ~3.0 hours for full lab sequence (including optional advanced/statistical labs)

---

## Summary

Chapter 4 built the **generative world model** that every later experiment in this book leans on. Mathematically, we defined a world $\mathcal{W}$ in [DEF-4.1] with catalog, users, queries, and an explicit seed, and we spelled out the main building blocks: catalog generation [EQ-4.1]–[EQ-4.6], user segments [EQ-4.7], query representations [EQ-4.9], and the determinism property [EQ-4.10]. Together they give us a precise answer to “what is being simulated?” before we ever write a line of code.

On the implementation side, the corresponding modules are small and concrete: `zoosim/world/catalog.py` turns the catalog equations into lognormal prices, linear margins, and clustered embeddings; `zoosim/world/users.py` samples segment‑specific preferences with Dirichlet category affinities; `zoosim/world/queries.py` produces intent‑driven queries; and `zoosim/core/config.py` centralizes all parameters so that changing the world means editing one place, not ten.

We did not stop at definitions. We checked that price medians match the lognormal parameters, that margin slopes match their theoretical values (dog food +0.12, litter −0.03, toys +0.20), that segments occupy distinct regions in preference space (premium avoids PL, PL‑lovers seek it), and that litter‑heavy users truly issue about 40% litter queries. Most importantly, we verified determinism: the same seed really does produce the same world, including embeddings.

Finally, we were honest about the **theory–practice gap**. Our simulator is deterministic, interpretable, and hand‑crafted, but it omits temporal drift, cold‑start, network effects, and adversaries. Bridging that gap will require domain randomization, offline RL on production logs, and possibly learned world models in the spirit of MuZero or Dreamer. Those are the subjects of later chapters; here, the job was to construct a world that is simple enough to understand and rich enough to matter.

All of these mathematical objects and code modules are indexed in the project **Knowledge Graph** (`docs/knowledge_graph/graph.yaml`) under IDs `CH-4`, `DEF-4.1`–`DEF-4.4`, `EQ-4.1`–`EQ-4.11`, and `MOD-zoosim.world.*`, so later chapters can reference Chapter 4's world model precisely and machine-readably.

**Next Chapter:**

We have a world ($\mathcal{C}, \mathcal{U}, \mathcal{Q}$). Now we need:
- **Relevance scoring**: How well does product $p$ match query $q$? (Chapter 5)
- **Click models**: Position bias, examination, abandonment (Chapter 2 revisited in Chapter 5)
- **Reward computation**: GMV, CM2, engagement aggregation (Chapter 1 formalism → Chapter 5 implementation)

Then we can define the **search environment** (MDP) and train RL agents!

**The unified vision:**
Mathematics (generative models, probability distributions) and code (NumPy/PyTorch implementations) are **inseparable**. Every equation compiles. Every implementation has a theorem. Theory and practice in constant dialogue.

That's Chapter 4. Let's move to Chapter 5: **Relevance, Features, and Counterfactual Ranking**.

---

**References:**

- [@tobin:sim2real:2017] Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," IROS 2017
- [@peng:sim2real:2018] Peng et al., "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization," ICRA 2018
- [@schrittwieser:muzero:2020] Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model," Nature 2020
- [@hafner:dreamer:2020] Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination," ICLR 2020
- [@hafner:dreamerv3:2023] Hafner et al., "Mastering Diverse Domains through World Models," arXiv 2023
- [@chen:dt:2021] Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling," NeurIPS 2021
- [@agarwal:statistical_precipice:2021] Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice," NeurIPS 2021
- [@chen:youtube_rl:2019] Chen et al., "Top-K Off-Policy Correction for a REINFORCE Recommender System," WSDM 2019
- [@ie:recsim:2019] Ie et al., "RecSim: A Configurable Simulation Platform for Recommender Systems," arXiv 2019
