# Appendix A --- Bayesian Preference Models for Search Ranking

**Vlad Prytula**

---

## Motivation

Chapter 6 introduced Thompson Sampling as a Bayesian approach to exploration: maintain a posterior distribution over reward parameters, sample from it, and act greedily on the sample. The regret bounds ([THM-6.1]) depend crucially on prior choice---a well-calibrated prior that concentrates around true parameters yields tighter regret than a diffuse or misspecified prior.

But where does the prior come from? In e-commerce search, we have **structure**: users belong to segments, products belong to categories, preferences correlate across similar items. A customer who frequently buys premium dog food probably has high price tolerance across the pet category. A user who clicks on organic products in groceries likely prefers organic options in pet food too. Ignoring this structure wastes information; exploiting it accelerates learning.

This appendix develops **hierarchical Bayesian models** for user preferences in search ranking. The key ideas:

1. **Partial pooling**: User-level parameters are drawn from segment-level distributions, which are themselves drawn from population-level hyperpriors. This "borrows strength" across users---a new user's preferences are informed by their segment until we observe enough individual data.

2. **Conjugate inference**: For carefully chosen likelihood-prior pairs, posterior updates have closed forms. This enables **online Bayesian learning** without expensive MCMC.

3. **Integration with bandits**: Posterior means and variances become **features** for LinUCB and Thompson Sampling. Uncertainty-aware features let the bandit explore efficiently where preferences are unknown.

**Prerequisites.** This appendix assumes familiarity with:
- Bayesian inference basics (prior, likelihood, posterior, Bayes' theorem)
- Exponential family distributions (Gaussian, Beta, Gamma)
- Sufficient statistics and conjugate priors

For readers unfamiliar with Bayesian statistics, consult [@gelman:bayesian_data_analysis:2013, Chapters 1--5] before proceeding.

**Connection to Chapter 6.** The material here directly supports Section 6.4 (Thompson Sampling) and the "rich features" discussion in Section 6.6. When we wrote "the feature vector $\phi(x)$ encodes user and product context," this appendix specifies *how* to construct those features from Bayesian posteriors.

---

## A.1 Hierarchical Priors for User Preferences

### A.1.1 The Structure of Preferences

Consider a search ranking system with:
- $U$ users indexed by $u \in \{1, \ldots, U\}$
- $S$ segments (user clusters) indexed by $s \in \{1, \ldots, S\}$, with $s(u)$ denoting user $u$'s segment
- $P$ products indexed by $p \in \{1, \ldots, P\}$
- $C$ categories indexed by $c \in \{1, \ldots, C\}$, with $c(p)$ denoting product $p$'s category

Each user $u$ has latent preferences:
- **Price sensitivity** $\beta_u \in \mathbb{R}$: How much does price affect $u$'s purchase probability?
- **Brand affinity** $\alpha_{u,b} \in \mathbb{R}$: User $u$'s preference for brand $b$
- **Category preference** $\gamma_{u,c} \in \mathbb{R}$: User $u$'s baseline interest in category $c$

A **flat prior** would treat each $\beta_u$ as independent: $\beta_u \sim \mathcal{N}(0, \sigma_\beta^2)$. With millions of users and sparse observations per user, learning is slow---each user's data is "siloed."

A **hierarchical prior** shares information across the hierarchy:

$$
\begin{aligned}
\mu_0 &\sim \mathcal{N}(\mu_{\text{pop}}, \sigma_{\text{pop}}^2) && \text{(population mean)} \\
\mu_s &\sim \mathcal{N}(\mu_0, \sigma_{\text{seg}}^2) && \text{(segment mean, } s = 1, \ldots, S\text{)} \\
\beta_u &\sim \mathcal{N}(\mu_{s(u)}, \sigma_u^2) && \text{(user parameter, } u = 1, \ldots, U\text{)}
\end{aligned}
\tag{A.1}
$$
{#EQ-A.1}

This is a **three-level hierarchy**: population $\to$ segment $\to$ user. The key property is **partial pooling** (also called shrinkage): estimates for individual users are "shrunk" toward their segment mean, and segment estimates are shrunk toward the population mean.

### A.1.2 Shrinkage and Information Borrowing

**Why shrinkage helps.** Consider two users:
- User A: 500 purchases, estimated $\hat{\beta}_A = -0.8$ (price sensitive)
- User B: 3 purchases, estimated $\hat{\beta}_B = +0.5$ (price insensitive)

Under a flat prior, we trust both estimates equally (modulo confidence intervals). Under a hierarchical prior with segment mean $\mu_s = -0.3$:
- User A's posterior mean stays near $-0.8$ (lots of data overwhelms the prior)
- User B's posterior mean shrinks toward $-0.3$ (weak data, prior dominates)

This is **rational**: User B's apparent price insensitivity is likely noise from small sample size. Borrowing strength from the segment gives better predictions.

**Formal shrinkage formula.** For a Gaussian hierarchy with known variances, the posterior mean for $\beta_u$ given $n_u$ observations with sample mean $\bar{\beta}_u$ is:

$$
\mathbb{E}[\beta_u \mid \text{data}] = \frac{n_u / \sigma_\epsilon^2}{n_u / \sigma_\epsilon^2 + 1/\sigma_u^2} \bar{\beta}_u + \frac{1/\sigma_u^2}{n_u / \sigma_\epsilon^2 + 1/\sigma_u^2} \mu_{s(u)}
\tag{A.2}
$$
{#EQ-A.2}

where $\sigma_\epsilon^2$ is the observation noise variance. This is a **precision-weighted average** of the sample mean and the segment prior mean. As $n_u \to \infty$, the weight shifts entirely to the data; as $n_u \to 0$, we fall back to the segment mean.

> **Remark A.1.1** (Connection to regularization). Shrinkage in Bayesian inference is the probabilistic analog of **regularization** in frequentist estimation. The prior $\mathcal{N}(\mu_s, \sigma_u^2)$ corresponds to an $L^2$ penalty pulling $\beta_u$ toward $\mu_s$. Ridge regression is the MAP estimate under a Gaussian prior. The Bayesian view additionally provides **uncertainty quantification**: the posterior variance tells us how much to trust the estimate.

---

## A.2 Price Sensitivity Model

### A.2.1 Model Specification

We now develop a specific model for **price sensitivity** in search ranking. Let:
- $r_{u,p,t} \in \{0, 1\}$: Binary outcome (purchase) for user $u$, product $p$, at time $t$
- $\text{price}_{p,t}$: Price of product $p$ at time $t$ (normalized to $[0, 1]$)
- $\phi_{p}$: Product features (category, brand, quality embedding)

**Likelihood model.** We assume purchases follow a logistic model:

$$
\mathbb{P}(r_{u,p,t} = 1 \mid \beta_u, \theta) = \sigma\left(\theta_0 + \beta_u \cdot \text{price}_{p,t} + \theta^\top \phi_p\right)
\tag{A.3}
$$
{#EQ-A.3}

where $\sigma(z) = 1/(1 + e^{-z})$ is the logistic function, $\theta_0$ is the intercept, and $\theta$ are product feature weights shared across users.

**Interpretation.** A negative $\beta_u$ means user $u$ is price-sensitive (higher prices reduce purchase probability); a positive $\beta_u$ means the user is price-insensitive or even prefers premium products. The product features $\theta^\top \phi_p$ capture relevance independent of user preferences.

**Hierarchical prior.** Following [EQ-A.1]:

$$
\begin{aligned}
\mu_0 &\sim \mathcal{N}(-0.5, 0.5^2) && \text{(population: slight price sensitivity)} \\
\mu_s &\sim \mathcal{N}(\mu_0, 0.3^2) && \text{(segment deviations)} \\
\beta_u &\sim \mathcal{N}(\mu_{s(u)}, 0.2^2) && \text{(user deviations)}
\end{aligned}
\tag{A.4}
$$
{#EQ-A.4}

The hyperparameters (0.5, 0.3, 0.2) encode prior beliefs: most variation is at the population level, with smaller segment and user deviations. These can be learned from data via empirical Bayes or full Bayesian inference on the hyperpriors.

### A.2.2 Conjugate Approximation

The logistic likelihood [EQ-A.3] is **not conjugate** to the Gaussian prior---there is no closed-form posterior. Two approaches:

**Approach 1: Laplace approximation.** Approximate the posterior by a Gaussian centered at the MAP estimate:

$$
p(\beta_u \mid \text{data}) \approx \mathcal{N}\left(\hat{\beta}_u^{\text{MAP}}, \, H_u^{-1}\right)
\tag{A.5}
$$
{#EQ-A.5}

where $H_u = -\nabla^2 \log p(\beta_u \mid \text{data})\big|_{\beta_u = \hat{\beta}_u^{\text{MAP}}}$ is the Hessian (observed Fisher information). This is fast (single optimization + Hessian computation) and often accurate for moderate sample sizes.

**Approach 2: Polya-Gamma augmentation.** The logistic likelihood admits an exact data augmentation that renders inference conditionally conjugate. Introduce auxiliary variables $\omega_{u,p,t} \sim \text{PG}(1, 0)$ (Polya-Gamma distribution), and the conditional posterior for $\beta_u$ becomes Gaussian. See [@polson:polya_gamma:2013] for details. This enables Gibbs sampling without Metropolis-Hastings.

**Approach 3: Linear approximation.** For online bandit applications where speed is critical, approximate the logistic by a linear model in a trust region:

$$
r_{u,p,t} \approx \beta_u \cdot \text{price}_{p,t} + \theta^\top \phi_p + \epsilon
\tag{A.6}
$$
{#EQ-A.6}

with $\epsilon \sim \mathcal{N}(0, \sigma^2)$. This is exactly conjugate to Gaussian priors, enabling closed-form online updates. The approximation is reasonable when $\sigma(\cdot)$ operates in its near-linear regime (roughly $|z| < 2$).

> **Remark A.2.1** (Theory-practice gap). In production, the linear approximation [EQ-A.6] often outperforms more sophisticated methods because it enables **real-time updates** at query time. The logistic model is more accurate but requires batch inference. Chapter 6's Thompson Sampling uses the linear model for computational tractability. When accuracy matters (e.g., offline analysis), use Laplace or Polya-Gamma.

---

## A.3 Brand and Category Affinity

### A.3.1 Brand Affinity Model

Users have preferences over brands. Let $\alpha_{u,b} \in \mathbb{R}$ denote user $u$'s affinity for brand $b$:

$$
\begin{aligned}
\mu_b^{(0)} &\sim \mathcal{N}(0, \sigma_{\text{brand}}^2) && \text{(brand baseline popularity)} \\
\alpha_{u,b} &\sim \mathcal{N}(\mu_b^{(0)}, \sigma_{\text{user-brand}}^2) && \text{(user-brand affinity)}
\end{aligned}
\tag{A.7}
$$
{#EQ-A.7}

The brand baseline $\mu_b^{(0)}$ captures aggregate popularity (everyone likes brand X); the user deviation captures individual preferences (user $u$ particularly likes brand Y).

**Sparsity challenge.** With thousands of brands and millions of users, the matrix $\{\alpha_{u,b}\}$ is enormous and sparse---most users interact with a tiny fraction of brands. Two solutions:

1. **Factorization**: Assume $\alpha_{u,b} \approx \mathbf{u}_u^\top \mathbf{b}_b$ where $\mathbf{u}_u, \mathbf{b}_b \in \mathbb{R}^k$ are low-dimensional embeddings. This reduces parameters from $U \times B$ to $(U + B) \times k$.

2. **Category-level pooling**: Share strength across brands in the same category:
$$
\mu_b^{(0)} \sim \mathcal{N}(\mu_{c(b)}^{\text{cat}}, \sigma_{\text{brand-cat}}^2)
$$
where $c(b)$ is brand $b$'s primary category.

### A.3.2 Category Preference Model

Similarly, users have baseline interest levels in product categories:

$$
\begin{aligned}
\nu_c^{(0)} &\sim \mathcal{N}(0, \sigma_{\text{cat}}^2) && \text{(category baseline)} \\
\gamma_{u,c} &\sim \mathcal{N}(\nu_c^{(0)} + \delta_{s(u),c}, \sigma_{\text{user-cat}}^2) && \text{(user-category affinity)}
\end{aligned}
\tag{A.8}
$$
{#EQ-A.8}

where $\delta_{s,c}$ is a segment-category interaction term. For example, the "premium pet owners" segment has higher baseline interest in the "premium pet food" category.

### A.3.3 Combined Preference Model

The full preference model combines price sensitivity, brand affinity, and category preference:

$$
\mathbb{E}[\text{utility}_{u,p}] = \beta_u \cdot \text{price}_p + \alpha_{u, b(p)} + \gamma_{u, c(p)} + \theta^\top \phi_p
\tag{A.9}
$$
{#EQ-A.9}

where $b(p)$ and $c(p)$ are product $p$'s brand and category. This utility feeds into the bandit's reward model: higher utility implies higher click/purchase probability.

---

## A.4 Posterior Inference

### A.4.1 Conjugate Gaussian Updates

For the linear approximation [EQ-A.6] with Gaussian likelihood and Gaussian hierarchical prior, posterior updates are conjugate. We derive the formulas for the price sensitivity parameter $\beta_u$.

**Prior.** User $u$ in segment $s$ has prior $\beta_u \sim \mathcal{N}(\mu_s, \sigma_u^2)$.

**Likelihood.** Given $n_u$ observations $\{(\text{price}_{u,i}, r_{u,i})\}_{i=1}^{n_u}$ with design matrix $X_u \in \mathbb{R}^{n_u \times 1}$ (prices) and response $\mathbf{r}_u \in \mathbb{R}^{n_u}$:

$$
\mathbf{r}_u \mid \beta_u \sim \mathcal{N}(X_u \beta_u, \sigma_\epsilon^2 I)
\tag{A.10}
$$
{#EQ-A.10}

**Posterior.** By Bayes' theorem with Gaussian-Gaussian conjugacy:

$$
\beta_u \mid \mathbf{r}_u \sim \mathcal{N}\left(\mu_u^{\text{post}}, (\sigma_u^{\text{post}})^2\right)
\tag{A.11}
$$
{#EQ-A.11}

where:

$$
\begin{aligned}
(\sigma_u^{\text{post}})^{-2} &= \sigma_u^{-2} + \sigma_\epsilon^{-2} X_u^\top X_u \\
\mu_u^{\text{post}} &= (\sigma_u^{\text{post}})^2 \left(\sigma_u^{-2} \mu_s + \sigma_\epsilon^{-2} X_u^\top \mathbf{r}_u\right)
\end{aligned}
\tag{A.12}
$$
{#EQ-A.12}

This is the standard Bayesian linear regression update, specialized to scalar $\beta_u$.

### A.4.2 Online Updates

For real-time bandit applications, we update the posterior after each observation. Let subscript $t$ denote the posterior after $t$ observations.

**Recursive update.** Given new observation $(\text{price}_{t+1}, r_{t+1})$:

$$
\begin{aligned}
\lambda_{t+1} &= \lambda_t + \sigma_\epsilon^{-2} \cdot \text{price}_{t+1}^2 && \text{(precision update)} \\
\nu_{t+1} &= \nu_t + \sigma_\epsilon^{-2} \cdot \text{price}_{t+1} \cdot r_{t+1} && \text{(precision-weighted sum update)} \\
\mu_{t+1}^{\text{post}} &= \nu_{t+1} / \lambda_{t+1} && \text{(posterior mean)} \\
(\sigma_{t+1}^{\text{post}})^2 &= 1 / \lambda_{t+1} && \text{(posterior variance)}
\end{aligned}
\tag{A.13}
$$
{#EQ-A.13}

Initialize with $\lambda_0 = \sigma_u^{-2}$ and $\nu_0 = \sigma_u^{-2} \mu_s$. This is $O(1)$ per update---critical for real-time systems.

### A.4.3 Empirical Bayes for Hyperparameters

The segment-level means $\mu_s$ and variances $\sigma_u^2$ are **hyperparameters**. Two approaches:

**Type II Maximum Likelihood (Empirical Bayes).** Maximize the marginal likelihood:

$$
\hat{\mu}_s, \hat{\sigma}_u^2 = \arg\max \prod_{u \in \text{segment } s} p(\mathbf{r}_u \mid \mu_s, \sigma_u^2)
\tag{A.14}
$$
{#EQ-A.14}

For Gaussian models, this has closed-form solutions via EM or direct optimization.

**Full Bayes.** Place hyperpriors on $\mu_s$ and $\sigma_u^2$ and infer their posteriors. This propagates uncertainty about hyperparameters into user-level inference. More principled but computationally expensive; typically done offline.

> **Remark A.4.1** (When to use which). For production bandits, use empirical Bayes with periodic batch updates to hyperparameters (e.g., nightly). For research and model validation, use full Bayes to quantify uncertainty about the hierarchy itself.

---

## A.5 Integration with Bandits

### A.5.1 Posterior Features for LinUCB

Chapter 6's LinUCB ([ALG-6.2]) requires a feature vector $\phi(x) \in \mathbb{R}^d$ for each context $x = (u, q, p)$ (user, query, product). Bayesian preference models provide **uncertainty-aware features**:

**Feature construction:**

$$
\phi(x) = \begin{pmatrix}
\mu_u^{\text{post}} & \text{(posterior mean price sensitivity)} \\
\sigma_u^{\text{post}} & \text{(posterior std: exploration signal)} \\
\mu_{u,b(p)}^{\text{brand}} & \text{(posterior brand affinity)} \\
\sigma_{u,b(p)}^{\text{brand}} & \text{(brand uncertainty)} \\
\gamma_{u,c(p)} & \text{(category preference)} \\
\text{price}_p & \text{(raw price)} \\
\theta^\top \phi_p & \text{(product embedding score)} \\
\vdots &
\end{pmatrix}
\tag{A.15}
$$
{#EQ-A.15}

The posterior standard deviations $\sigma_u^{\text{post}}, \sigma_{u,b}^{\text{brand}}$ are particularly valuable: they tell LinUCB where uncertainty is high, encouraging exploration of users/products with unknown preferences.

### A.5.2 Posterior Sampling for Thompson Sampling

Thompson Sampling ([ALG-6.1]) samples reward parameters from the posterior and acts greedily. With hierarchical Bayesian preferences:

**Algorithm A.5.1** (Thompson Sampling with Hierarchical Posteriors) {#ALG-A.5.1}

**Input:** User $u$, query $q$, candidate products $\{p_1, \ldots, p_K\}$

1. **Sample user preferences:**
   $$
   \tilde{\beta}_u \sim \mathcal{N}(\mu_u^{\text{post}}, (\sigma_u^{\text{post}})^2)
   $$

2. **Sample brand affinities** (for brands in candidate set):
   $$
   \tilde{\alpha}_{u,b} \sim \mathcal{N}(\mu_{u,b}^{\text{brand}}, (\sigma_{u,b}^{\text{brand}})^2) \quad \forall b \in \{b(p_1), \ldots, b(p_K)\}
   $$

3. **Compute sampled utilities:**
   $$
   \tilde{U}(p_k) = \tilde{\beta}_u \cdot \text{price}_{p_k} + \tilde{\alpha}_{u, b(p_k)} + \gamma_{u, c(p_k)} + \theta^\top \phi_{p_k}
   $$

4. **Rank by sampled utility:**
   $$
   \text{ranking} = \text{argsort}_{k}(-\tilde{U}(p_k))
   $$

**Output:** Ranked product list

This directly implements the "sample-then-rank" strategy from Chapter 6, but with structured posteriors over user preferences rather than independent arm parameters.

### A.5.3 Cold-Start Handling

Hierarchical priors naturally handle **cold-start** problems:

**New user.** A user $u_{\text{new}}$ with no observations has posterior equal to the prior:
$$
\beta_{u_{\text{new}}} \sim \mathcal{N}(\mu_{s(u_{\text{new}})}, \sigma_u^2)
$$
The segment mean $\mu_s$ provides a reasonable starting point. Thompson Sampling will explore efficiently as observations accumulate.

**New product.** A product $p_{\text{new}}$ with no user interactions uses the category/brand priors:
$$
\mathbb{E}[\text{utility}_{u, p_{\text{new}}}] = \beta_u \cdot \text{price}_{p_{\text{new}}} + \mu_{b(p_{\text{new}})}^{(0)} + \nu_{c(p_{\text{new}})}^{(0)} + \theta^\top \phi_{p_{\text{new}}}
$$
Product embeddings $\phi_p$ from Chapter 4's content model provide additional signal.

> **Remark A.5.1** (Cold-start is bandit exploration). The cold-start problem is fundamentally an exploration problem: how to learn preferences with minimal regret. Hierarchical Bayesian models + Thompson Sampling provide a principled solution: use prior structure for initial estimates, explore where uncertainty is high, and converge to personalized preferences as data accumulates.

---

## A.6 Computational Considerations

### A.6.1 Online vs. Batch Inference

**Online inference** (per-query):
- Update posteriors after each observation using [EQ-A.13]
- Complexity: $O(1)$ per update for scalar parameters, $O(d^2)$ for $d$-dimensional
- Latency: sub-millisecond, suitable for real-time ranking

**Batch inference** (periodic):
- Re-estimate hyperparameters $\{\mu_s, \sigma_u^2, \sigma_{\text{seg}}^2\}$ from accumulated data
- Fit full hierarchical model using Stan, PyMC, or variational inference
- Complexity: $O(U + S)$ linear in users/segments for conjugate models
- Schedule: nightly or weekly, depending on data velocity

**Hybrid architecture:**
```
Real-time: Online posterior updates (fixed hyperparameters)
    ↓
Nightly: Batch hyperparameter re-estimation
    ↓
Weekly: Full hierarchical model refit (detect segment drift)
```

### A.6.2 Scalability

**Challenge.** Millions of users $\times$ thousands of brands = billions of affinity parameters. Even with factorization, this is substantial.

**Solutions:**

1. **Lazy evaluation**: Only maintain posteriors for recently active users. Inactive users revert to segment priors.

2. **Approximate posteriors**: Use diagonal Gaussian approximations (ignore correlations). For most bandit applications, marginal posteriors suffice.

3. **Streaming variational inference**: Update approximate posteriors via stochastic gradient descent on the ELBO. See [@hoffman:svi:2013] for the algorithm.

4. **Segment-level only**: For users with $< k$ observations, use segment-level posteriors directly without user-level personalization. Set $k = 10$ as a reasonable threshold.

### A.6.3 When Full Bayes Is Overkill

Hierarchical Bayesian models add complexity. When are they worth it?

**Use hierarchical models when:**
- User base is large but data per user is sparse
- Cold-start is frequent (new users, new products)
- Preferences have known structure (segments, categories)
- Exploration-exploitation trade-off matters (bandit setting)

**Use simpler models when:**
- Most users have abundant data (>100 observations)
- Preferences are approximately uniform across users
- Computational budget is tight (edge inference)
- Point estimates suffice (no uncertainty quantification needed)

> **Remark A.6.1** (Practical guidance). In e-commerce search, hierarchical models almost always help for price sensitivity and category preferences. For brand affinity, the benefit depends on brand diversity: if 90% of purchases come from 10 brands, simpler models may suffice. Profile your data before committing to hierarchy.

---

## A.7 Summary

**Key results established:**

1. **Hierarchical priors** [EQ-A.1]: User preferences are drawn from segment distributions, which are drawn from population distributions. This enables **partial pooling** (shrinkage) that borrows strength across users.

2. **Shrinkage formula** [EQ-A.2]: Posterior means are precision-weighted averages of sample means and prior means. Users with more data see less shrinkage.

3. **Conjugate online updates** [EQ-A.13]: For Gaussian models, posteriors update in $O(1)$ time per observation, enabling real-time bandit inference.

4. **Integration with bandits** [EQ-A.15], [ALG-A.5.1]: Posterior means and variances become features for LinUCB; posterior sampling enables Thompson Sampling with structured priors.

5. **Cold-start handling**: New users/products inherit segment/category priors, with uncertainty driving exploration via Thompson Sampling.

**Connection to Chapter 6:**

The "rich features" discussion in Section 6.6 is now concrete: construct $\phi(x)$ using posterior statistics from hierarchical preference models. The Thompson Sampling regret bound ([THM-6.1]) tightens when the prior concentrates around true parameters---hierarchical models, fit to historical data, provide better-calibrated priors than generic diffuse priors.

**When to consult this appendix:**

- **Chapter 6, Section 6.4**: When implementing Thompson Sampling, use [ALG-A.5.1] for posterior sampling with structured user preferences
- **Chapter 6, Section 6.6**: When constructing feature vectors for LinUCB, use [EQ-A.15] for uncertainty-aware features
- **Chapter 4, Section 4.3**: When modeling user behavior in the simulator, the hierarchical structure [EQ-A.1] provides a principled generative model

---

## A.8 References and Further Reading

**Bayesian inference foundations:**
- [@gelman:bayesian_data_analysis:2013]: Comprehensive treatment of Bayesian data analysis, including hierarchical models and computational methods
- [@murphy:machine_learning:2012, Chapters 5, 9]: Bayesian inference and hierarchical models from a machine learning perspective

**Hierarchical models for recommendation:**
- [@stern:matchbox:2009]: Matchbox---Bayesian matrix factorization with hierarchical priors for recommendation
- [@gopalan:content_poisson:2014]: Bayesian nonparametric Poisson factorization for implicit feedback

**Thompson Sampling theory:**
- [@russo:tutorial_ts:2018]: Tutorial on Thompson Sampling---the key reference for Bayesian regret analysis
- [@chapelle:empirical_ts:2011]: Empirical evaluation of Thompson Sampling in display advertising

**Bandits with structured priors:**
- [@lattimore:bandit_algorithms:2020, Chapters 35--37]: Rigorous foundations for Bayesian bandits, including hierarchical settings
- [@kveton:meta_ts:2021]: Meta-Thompson Sampling for learning shared structure across related bandit tasks

**Computational methods:**
- [@polson:polya_gamma:2013]: Polya-Gamma augmentation for exact Bayesian inference in logistic models
- [@hoffman:svi:2013]: Stochastic variational inference for scalable Bayesian learning

**Applications in e-commerce:**
- [@hill:yahoo_ts:2017]: Thompson Sampling at Yahoo---practical deployment with hierarchical priors
- [@mcmahan:ad_click:2013]: Ad click prediction with billions of features---scalable Bayesian methods at Google
