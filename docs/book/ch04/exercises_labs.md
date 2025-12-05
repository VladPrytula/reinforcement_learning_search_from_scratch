# Chapter 4 — Exercises & Labs

**Total estimated time:** 2.5 hours

These exercises provide hands-on practice with the generative world model from Chapter 4. All code should be runnable in a Jupyter notebook or Python script with the `zoosim` package installed.

---

## Exercise 1: Catalog Statistics (30 minutes)

**Objective:** Generate a synthetic catalog and verify distributional properties.

In a real retailer, the first thing analysts do with a new dataset is not train a model; it is to look at basic distributions. Are dog‑food prices where merchandising expects them to be? Is litter really being run as a loss‑leader, or did something drift? This exercise puts you in that role for our simulator: you sanity‑check that the synthetic catalog behaves like a plausible e‑commerce assortment before trusting any downstream RL experiments.

**Setup:**
```python
import numpy as np
import matplotlib.pyplot as plt
from zoosim.core.config import SimulatorConfig
from zoosim.world.catalog import generate_catalog

cfg = SimulatorConfig(seed=42)
rng = np.random.default_rng(cfg.seed)
catalog = generate_catalog(cfg.catalog, rng)
```

**Tasks:**

1. **Price percentiles** (10 min)
   - Compute 25th, 50th (median), 75th percentiles for each category
   - Compare median to theoretical value $e^{\mu}$ from config
   - Print results in a formatted table

   ```python
   # Your code here
   ```

   **Expected output:**
   ```
   Category      | P25    | Median | P75    | Theory (e^μ)
   ------------------------------------------------------------
   dog_food      | $9.34  | $13.46 | $19.27 | $13.46
   cat_food      | $8.82  | $12.18 | $17.51 | $12.18
   litter        | $6.87  | $9.03  | $11.92 | $9.03
   toys          | $3.71  | $6.05  | $9.87  | $6.05
   ```

2. **Margin verification** (10 min)
   - Compute mean CM2 for each category
   - Verify litter has **negative** average margin
   - Verify toys have **highest** average margin

   ```python
   # Your code here
   ```

   **Expected results:**
   - Litter: mean CM2 < 0
   - Toys: mean CM2 > all other categories

3. **Price vs. CM2 scatter plot** (10 min)
   - Create 2x2 subplot grid (one subplot per category)
   - Scatter plot of price (x-axis) vs. CM2 (y-axis)
   - Overlay theoretical line: CM2 = β·price (from config)
   - Add title, axis labels, legend

   ```python
   # Your code here
   # Hint: Use cfg.catalog.margin_slope[category] for slope β
   ```

   **Expected visualization:** See Figure 4.1 in chapter text

---

## Exercise 2: User Segment Analysis (30 minutes)

**Objective:** Sample users and analyze segment-specific preferences.

Personalization only makes sense if different users truly want different things. In production, teams maintain audience definitions (“value shoppers”, “premium”, “private‑label loyalists”) and routinely inspect how those segments behave. Here you will do the same with our simulated users: verify that the segment mix matches the configuration and that each segment occupies a distinct region in preference space, just as a marketing or CRM team would expect.

**Setup:**
```python
from zoosim.world.users import sample_user

cfg = SimulatorConfig(seed=2025_1108)
rng = np.random.default_rng(cfg.seed)

# Generate 10,000 users
users = [sample_user(config=cfg, rng=rng) for _ in range(10_000)]
```

**Tasks:**

1. **Segment distribution** (5 min)
   - Count users per segment
   - Compute empirical probabilities
   - Compare to `cfg.users.segment_mix`

   ```python
   # Your code here
   ```

   **Expected output:**
   ```
   Segment        | Count | Empirical | Expected
   -----------------------------------------------
   price_hunter   | 4,013 | 0.401     | 0.400
   pl_lover       | 2,001 | 0.200     | 0.200
   premium        | 1,974 | 0.197     | 0.200
   litter_heavy   | 2,012 | 0.201     | 0.200
   ```

2. **Preference scatter plots** (15 min)
   - Create 2x2 subplot grid (one subplot per segment)
   - Each subplot: scatter plot of θ_price (x-axis) vs. θ_pl (y-axis)
   - Mark segment mean with large red star
   - Add horizontal/vertical lines at zero
   - Add title with segment name

   ```python
   # Your code here
   ```

   **Expected pattern:**
   - Price hunters: Strong negative θ_price, neutral θ_pl
   - Premium: Weak negative θ_price, **negative** θ_pl (avoid PL)
   - PL lovers: Moderate negative θ_price, **strong positive** θ_pl
   - Litter heavy: Moderate negative θ_price, slight positive θ_pl

3. **Category affinity validation** (10 min)
   - For litter-heavy segment, compute mean category affinity vector
   - Print probabilities for each category
   - Verify litter affinity ≈ 0.40 (40%)

   ```python
   litter_heavy_users = [u for u in users if u.segment == "litter_heavy"]
   # Your code here
   ```

   **Expected output:**
   ```
   Litter-heavy segment category affinities:
     dog_food: 0.199
     cat_food: 0.300
     litter:   0.401  ← Should be ~40%
     toys:     0.100
   ```

---

## Exercise 3: Query Intent Coupling (30 minutes)

**Objective:** Verify query intents align with user category affinities.

Search logs are full of hints about what customers actually want: some queries scream “litter refill”, others quietly suggest “browse toys while I’m here”. In a healthy system, the distribution of query intents should line up with who is visiting the site. This exercise checks that our simulator respects that principle: litter‑heavy users should fire more litter queries, premium users should lean toward food queries, and the overall query‑type mix should look like a real e‑commerce search bar.

**Setup:**
```python
from zoosim.world.queries import sample_query

cfg = SimulatorConfig(seed=2025_1108)
rng = np.random.default_rng(cfg.seed)

# Generate 5,000 (user, query) pairs
user_query_pairs = []
for _ in range(5_000):
    user = sample_user(config=cfg, rng=rng)
    query = sample_query(user=user, config=cfg, rng=rng)
    user_query_pairs.append((user, query))
```

**Tasks:**

1. **Query type distribution** (10 min)
   - Count queries by type (category, brand, generic)
   - Compute empirical probabilities
   - Verify matches `cfg.queries.query_type_mix` (60% category, 20% brand, 20% generic)

   ```python
   # Your code here
   ```

   **Expected output:**
   ```
   Query Type | Count | Empirical | Expected
   ------------------------------------------
   category   | 2,998 | 0.600     | 0.600
   brand      | 1,012 | 0.202     | 0.200
   generic    | 990   | 0.198     | 0.200
   ```

2. **Intent rate by segment** (15 min)
   - For each segment, compute percentage of queries that have intent_category = "litter"
   - Compare to expected litter affinity from Dirichlet concentration parameters

   ```python
   # Your code here
   # Hint: Dirichlet mean for dimension k is α_k / Σ_j α_j
   ```

   **Expected output:**
   ```
   Segment        | Litter Query Rate | Expected Affinity
   -------------------------------------------------------
   price_hunter   | 0.273             | ~0.273
   pl_lover       | 0.229             | ~0.227
   premium        | 0.089             | ~0.091
   litter_heavy   | 0.402             | ~0.400
   ```

3. **Embedding similarity** (5 min)
   - For first 100 (user, query) pairs, compute cosine similarity between user.theta_emb and query.phi_emb
   - Compute mean similarity
   - Verify high similarity (> 0.8) since query embedding is user embedding + small noise

   ```python
   import torch.nn.functional as F

   # Your code here
   # Hint: Use F.cosine_similarity() or manual dot product / norms
   ```

   **Expected result:** Mean similarity ≈ 0.95-0.99 (query ≈ user preference)

---

## Exercise 4: Determinism Verification (15 minutes)

**Objective:** Verify same seed produces identical worlds.

In a production A/B test, you do not get to re‑run yesterday’s traffic; you only see it once. Reproducible simulators are the opposite: you want to be able to rewind and replay exactly the same synthetic experiment to debug a policy change or a subtle regression. This exercise formalizes that discipline: with the same seed you must recover the same catalog and users, bit‑for‑bit, so that any change in results can be traced back to code or configuration—not to random noise.

**Tasks:**

1. **Identical catalogs** (5 min)
   - Generate two catalogs with seed 42
   - Assert products are identical at indices [0, 10, 100, 999]
   - Check: price, cm2, category, is_pl, embedding

   ```python
   cfg = CatalogConfig()

   catalog1 = generate_catalog(cfg, np.random.default_rng(42))
   catalog2 = generate_catalog(cfg, np.random.default_rng(42))

   # Your assertions here
   # For embeddings: use torch.equal(emb1, emb2)
   ```

2. **Different catalogs with different seeds** (5 min)
   - Generate two catalogs with seeds 42 and 123
   - Assert products differ at index 0
   - Verify at least one attribute is different

   ```python
   # Your code here
   ```

3. **Full world determinism** (5 min)
   - Generate 100 users with seed 2025
   - Generate 100 users with seed 2025 again
   - Assert all users have identical segments and preferences

   ```python
   # Your code here
   ```

**Success criteria:** All assertions pass, demonstrating [EQ-4.10] from chapter.

---

## Exercise 5: Domain Randomization (45 minutes, Advanced)

**Objective:** Implement domain randomization for robust policy training.

**Background:** Policies robust to simulator variability often transfer better to production (sim-to-real transfer). We randomize parameters to create diverse training environments.

Think of launching the same ranking policy in ten different countries or seasons: prices, margins, and customer mixes all shift, sometimes dramatically. If you tuned your agent to a single “average” configuration, it will often break in at least one of those markets. Domain randomization is the simulator analogue of that reality check: by sampling slightly different but plausible worlds, you force the policy to learn behaviors that survive small changes in catalog economics and audience composition.

**Tasks:**

1. **Randomization function** (15 min)
   - Implement `randomize_config(base_cfg, rng, perturbation=0.1)`
   - Perturb price distribution parameters: `μ ± perturbation`, `σ ± perturbation`
   - Perturb margin slopes: `β ± perturbation`
   - Perturb segment mix probabilities (renormalize to sum to 1)
   - Return new `SimulatorConfig`

   ```python
   def randomize_config(base_cfg: SimulatorConfig, rng: np.random.Generator,
                        perturbation: float = 0.1) -> SimulatorConfig:
       """Create randomized configuration for domain randomization.

       Perturbs:
       - Catalog price parameters (μ, σ)
       - Margin slopes (β)
       - Segment mix probabilities

       Args:
           base_cfg: Base configuration
           rng: Random generator
           perturbation: Relative perturbation magnitude (default 0.1 = ±10%)

       Returns:
           Randomized configuration
       """
       # Your implementation here
       # Hint: Use copy.deepcopy(base_cfg) and modify in place
       pass
   ```

2. **Generate ensemble** (10 min)
   - Create 10 randomized configurations
   - For each, generate a 1,000-product catalog
   - Compute mean litter CM2 for each configuration
   - Print distribution of mean litter CM2 across configurations

   ```python
   # Your code here
   ```

   **Expected output:** Range of mean litter CM2 values (e.g., [-0.35, -0.15])

3. **Robustness experiment (optional, 20 min, requires Chapter 6)**
   - Implement simple LinUCB bandit (preview of Chapter 6)
   - Train on base configuration for 1,000 episodes
   - Evaluate on 10 randomized configurations
   - Measure GMV degradation: `(GMV_base - GMV_randomized) / GMV_base`
   - Plot histogram of degradation across configurations

   ```python
   # This requires Chapter 6 LinUCB implementation
   # Skip if not yet covered
   ```

**Conceptual question:**
- Why does training on randomized configurations improve robustness?
- What's the trade-off between realism and randomization?

**Answer (brief):** Randomization forces policy to learn features robust to distribution shift, avoiding overfitting to specific parameter values. Trade-off: Too much randomization creates unrealistic scenarios the policy will never see, wasting training data.

---

## Exercise 6: Statistical Tests (20 minutes, Optional)

**Objective:** Apply formal statistical tests to generated data.

Data scientists in large e‑commerce companies do not stop at eyeballing histograms; they routinely run goodness‑of‑fit tests to catch subtle drifts and modeling mistakes. If your simulated dog‑food prices stop looking lognormal, or your segment mix no longer matches the business definition, any conclusions drawn by RL agents become suspect. This exercise gives you a light‑weight version of that toolkit: formal tests that say “this looks consistent with our assumptions” rather than relying on visual judgment alone.

**Tasks:**

1. **Goodness-of-fit test for lognormal prices** (10 min)
   - Generate 1,000 dog food products
   - Extract prices
   - Apply Kolmogorov-Smirnov test: Is data consistent with LogNormal(2.6, 0.4)?

   ```python
   from scipy.stats import lognorm, kstest

   cfg = CatalogConfig()
   rng = np.random.default_rng(42)

   # Generate products and extract dog_food prices
   # ...

   # KS test
   # lognorm parameters: s=sigma, scale=exp(mu)
   result = kstest(prices, lambda x: lognorm.cdf(x, s=0.4, scale=np.exp(2.6)))
   print(f"KS statistic: {result.statistic:.4f}, p-value: {result.pvalue:.4f}")

   # If p-value > 0.05, fail to reject null hypothesis (data is lognormal)
   ```

2. **Chi-square test for segment distribution** (10 min)
   - Sample 10,000 users
   - Count users per segment
   - Apply chi-square goodness-of-fit test
   - Null hypothesis: Observed counts match expected probabilities from `segment_mix`

   ```python
   from scipy.stats import chisquare

   # Your code here
   # Hint: chisquare(observed_counts, expected_counts)
   ```

**Expected results:** Both tests should fail to reject null hypotheses (p > 0.05), confirming our generator matches specified distributions.

---

## Exercise 7: Convergence of Catalog Statistics (20 minutes, Optional)

**Objective:** Verify law of large numbers for lognormal price means.

In Chapter 1 we treated expectations as mathematical objects. Here you get to see one of those expectations—**the mean of a lognormal price distribution**—emerge empirically as you increase catalog size. This is exactly the kind of sanity check production teams run before trusting summary dashboards or offline simulations.

**Tasks:**

1. **Mean price vs. catalog size** (15 min)
   - Consider dog-food prices, which follow $\text{LogNormal}(\mu=2.6, \sigma=0.4)$
   - Recall from Chapter 1: $\mathbb{E}[\text{price}] = e^{\mu + \sigma^2/2}$
   - For $N \in \{100, 500, 1000, 5000, 10{,}000, 50{,}000\}$:
     - Create a `CatalogConfig` with `n_products = N`
     - Generate a catalog with fixed seed 42
     - Compute mean price for the dog_food category

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from zoosim.core.config import CatalogConfig
   from zoosim.world.catalog import generate_catalog

   Ns = [100, 500, 1000, 5000, 10_000, 50_000]
   mu, sigma = 2.6, 0.4
   true_mean = np.exp(mu + sigma**2 / 2)

   mean_prices = []
   for N in Ns:
       cfg = CatalogConfig(n_products=N)
       rng = np.random.default_rng(42)
       catalog = generate_catalog(cfg, rng)
       dog_prices = [p.price for p in catalog if p.category == "dog_food"]
       mean_prices.append(np.mean(dog_prices))

   plt.figure(figsize=(8, 4))
   plt.plot(Ns, mean_prices, marker="o", label="Empirical mean (dog_food)")
   plt.axhline(true_mean, color="red", linestyle="--",
               label=f"Theoretical mean = e^(μ+σ²/2) ≈ ${true_mean:.2f}")
   plt.xscale("log")
   plt.xlabel("Catalog size N (log scale)")
   plt.ylabel("Mean price ($)")
   plt.title("Convergence of Dog-Food Mean Price")
   plt.legend()
   plt.grid(alpha=0.3)
   plt.savefig("catalog_mean_price_convergence.png", dpi=150)
   print("Saved plot to catalog_mean_price_convergence.png")

   print("\nMean dog-food prices by N:")
   for N, m in zip(Ns, mean_prices):
       print(f"  N={N:6d}: mean=${m:6.2f}")
   ```

   **Output (representative):**
   ```
   Mean dog-food prices by N:
     N=   100: mean=$14.83
     N=   500: mean=$14.41
     N=  1000: mean=$14.35
     N=  5000: mean=$14.31
     N= 10000: mean=$14.30
     N= 50000: mean=$14.29

   Theoretical mean (e^{2.6 + 0.4^2/2}) ≈ $14.29
   ```

   As $N$ grows, the empirical mean converges to the theoretical value, illustrating the law of large numbers in the concrete setting of catalog statistics.

2. **Conceptual reflection** (5 min)
   - Why does the simulator default to $N = 10{,}000$ products?
   - What breaks if you only use $N = 100$ products for RL training?

   **Hint:** Think about variance of estimates, coverage of rare but important products, and the stability of downstream policy gradients.

---

## Lab: Complete World Generation Pipeline (30 minutes)

**Objective:** Integrate catalog, users, and queries into a complete world generation workflow.

In production settings, teams often maintain nightly “world snapshots”: rolled‑up statistics and JSON dumps that downstream dashboards, notebooks, and training jobs consume. The goal is not to store every click, but to have a coherent view of “what the world looked like” on a given day. This lab has the same flavor. You will wire together catalog, user, and query generation and materialize a small, self‑contained snapshot that other chapters—and future you—can reuse without regenerating everything from scratch.

**Task:**

Write a script that:
1. Loads configuration from file or default
2. Generates catalog (10,000 products)
3. Samples 1,000 users
4. For each user, samples 5 queries
5. Saves results to disk (CSV or JSON)
6. Prints summary statistics

**Starter code:**
```python
import json
from pathlib import Path

def generate_world(config: SimulatorConfig, output_dir: Path):
    """Generate complete world and save to disk.

    Args:
        config: Simulator configuration
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    # 1. Generate catalog
    print("Generating catalog...")
    catalog = generate_catalog(config.catalog, rng)

    # Save catalog statistics (not full catalog, too large)
    catalog_stats = {
        "n_products": len(catalog),
        "price_mean_by_category": {
            cat: float(np.mean([p.price for p in catalog if p.category == cat]))
            for cat in config.catalog.categories
        },
        # Add more statistics here
    }

    with open(output_dir / "catalog_stats.json", "w") as f:
        json.dump(catalog_stats, f, indent=2)

    # 2. Generate users
    print("Generating users...")
    # Your code here

    # 3. Generate queries
    print("Generating queries...")
    # Your code here

    # 4. Save and print summary
    print("\nSummary:")
    print(f"  Catalog: {len(catalog)} products")
    # Your summary here

if __name__ == "__main__":
    cfg = SimulatorConfig(seed=2025_1108)
    generate_world(cfg, Path("./world_output"))
```

**Deliverables:**
- `catalog_stats.json`: Price/margin statistics by category
- `users.json`: User segments and aggregate statistics
- `queries.json`: Query type distribution and intent coupling metrics
- Console output with summary statistics

**Success criteria:**
- All files generated
- Statistics match expectations from chapter
- Script runs in < 30 seconds on standard laptop

---

## Bonus Challenge: Catalog Embeddings Visualization (Optional)

**Objective:** Visualize product embeddings in 2D using dimensionality reduction.

Modern search teams routinely project high‑dimensional embeddings down to two or three dimensions to debug models and explain behavior to stakeholders. If “dog food” and “litter” products are hopelessly entangled in embedding space, no amount of clever ranking logic will fully fix relevance. This bonus challenge gives you the simulator version of that diagnostic: you look at the learned‑by‑construction clusters and convince yourself that categories are separated the way a human merchandiser would expect.

**Tasks:**

1. Generate catalog with 10,000 products
2. Extract embeddings (16D) for all products
3. Apply UMAP or t-SNE to reduce to 2D
4. Create scatter plot colored by category
5. Verify products cluster by category around shared centroids

**Starter code:**
```python
from umap import UMAP  # pip install umap-learn
# or: from sklearn.manifold import TSNE

import torch

# Generate catalog
cfg = CatalogConfig(n_products=10_000)
rng = np.random.default_rng(42)
catalog = generate_catalog(cfg, rng)

# Extract embeddings
embeddings = torch.stack([p.embedding for p in catalog]).numpy()  # (10000, 16)

# Reduce to 2D
reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
for cat in cfg.categories:
    mask = [p.category == cat for p in catalog]
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                label=cat, alpha=0.5, s=10)

plt.legend()
plt.title("Product Embeddings (UMAP projection)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig("embeddings_umap.png", dpi=150)
plt.show()
```

**Expected result:** Four distinct clusters (one per category), each concentrated around a distinct category centroid with varying tightness based on `emb_cluster_std` from config.

---

## Solution Hints

**Exercise 1:**
- Use `np.percentile(prices, [25, 50, 75])`
- Theoretical median: `np.exp(cfg.catalog.price_params[cat]["mu"])`
- Margin verification: `litter_cm2 = [p.cm2 for p in catalog if p.category == "litter"]`

**Exercise 2:**
- Segment counting: `collections.Counter([u.segment for u in users])`
- Category affinity: `np.mean([u.theta_cat for u in litter_heavy_users], axis=0)`

**Exercise 3:**
- Query type: `collections.Counter([q.query_type for _, q in user_query_pairs])`
- Intent coupling: Group by segment, compute fraction where `q.intent_category == "litter"`

**Exercise 4:**
- Embedding comparison: `torch.equal(catalog1[idx].embedding, catalog2[idx].embedding)`

**Exercise 5:**
- Deep copy config: `import copy; new_cfg = copy.deepcopy(base_cfg)`
- Perturb: `new_mu = mu * (1 + rng.uniform(-perturbation, perturbation))`
- Renormalize simplex: `new_probs / new_probs.sum()`

---

## Testing Your Solutions

Run all exercises in a Jupyter notebook or as Python scripts. Expected total runtime: ~20 minutes (excluding optional exercises).

**Validation:**
- All numerical results should match expected outputs within ±5% (stochastic variation)
- Plots should show expected patterns (clusters, correlations, distributions)
- Determinism tests should pass exactly (no variation allowed)

**Common issues:**
- **RNG state**: Always create new `rng = np.random.default_rng(seed)` before each exercise
- **Tensor comparisons**: Use `torch.equal()` for exact equality, not `==`
- **Floating-point precision**: Use `np.allclose(a, b, rtol=1e-5)` instead of `a == b`

---

## Discussion Questions

1. **Realism vs. Simplicity**: Our simulator uses lognormal prices and linear margins. What real-world phenomena do we miss? (seasonality, promotions, competitor pricing)

2. **Segment heterogeneity**: We have 4 segments. Production might have 100+. How would you:
   - Learn segments from data (clustering, mixture models)?
   - Handle continuous preference distributions instead of discrete segments?

3. **Sim-to-real gap**: If our production transfer fails (sim-trained policy performs poorly), what debugging steps would you take?
   - Compare distributions (price, CTR, query types)
   - Check feature coverage (are production features in simulator?)
   - Evaluate on randomized configurations (domain randomization)
   - Fine-tune with offline RL on production logs (Chapter 13)

4. **Embedding generation**: We use Gaussian clusters. In production, embeddings come from learned models (Word2Vec, transformers). What properties must these embeddings have for our simulator to be realistic?
   - Smooth: Similar products → similar embeddings
   - Separable: Different categories → distinguishable
   - Aligned with user preferences: User query embedding → high similarity with relevant products

5. **Scalability**: Our simulator generates 10K products, 10K users. Production has 100M+ products, billions of users. What computational bottlenecks arise?
   - Catalog generation: Vectorize with NumPy instead of Python loops
   - Embedding storage: Use approximate nearest neighbors (FAISS, Annoy)
   - User sampling: Pre-generate user population, sample from cache

---

**End of Chapter 4 Exercises & Labs**

These exercises reinforce the generative world model concepts from Chapter 4. By completing them, you'll have hands-on experience with:
- Catalog generation and statistical validation
- User segment modeling and preference distributions
- Query intent coupling and embedding similarity
- Deterministic reproducibility (critical for RL experiments)
- Domain randomization for robust policy learning

Next: **Chapter 5 — Relevance, Features, and Counterfactual Ranking**
