# Chapter 14: Multi-Objective RL and Fairness at Scale

**Vlad Prytula**

---

## 14.0 Why This Chapter Exists

In production ranking systems, reward maximization alone is not a faithful specification. We also impose constraints: profitability floors, exposure requirements, stability SLOs, and fairness commitments. The book treats these constraints in two complementary ways:

1. **Hard guardrails** (Chapter 10): monitoring, fallback, and hard feasibility filters that prevent catastrophic violations at serving time.
2. **Soft constrained optimization** (this chapter): we treat constraints as part of the learning objective via Lagrange multipliers, and we learn the trade-off by solving a **constrained Markov decision process (CMDP)**.

This separation resolves an engineering truth: even if a learning algorithm is mathematically correct, production systems still require independent safety layers. Conversely, hard guardrails alone do not answer the optimization question of *which* feasible policy is best.

---

## 14.1 Terminology Contract

Before proceeding, we establish precise definitions to avoid confusion between this chapter's terminology and other chapters.

### 14.1.1 Brand Queries vs. Product Brands

In this simulator, "brand queries" refers to a **query specificity regime**, not product-level brand identifiers:

| Term | Definition | Location |
|------|------------|----------|
| **Query type "brand"** | A query specificity level (high specificity, sharp position bias) | `QueryConfig.query_types` in `zoosim/core/config.py:139--145` |
| **Product brand ID** | Not implemented | `Product` dataclass has no `brand` field (`zoosim/world/catalog.py:15--26`) |

The query types `["category", "brand", "generic"]` control position bias curves and feature scaling (see Chapter 5), but they do not represent product-level brand ownership. We do not model product brands in the current simulator.

### 14.1.2 Provider Groups for Fairness

This chapter's fairness constraints operate over **provider groups** already modeled in the simulator:

| Provider Group | Attribute | Semantics | Config Reference |
|----------------|-----------|-----------|------------------|
| **Private-label vs. national-brand** | `Product.is_pl` | Binary: `True` = store brand, `False` = national brand | `CatalogConfig.pl_prob` at `zoosim/core/config.py:39--46` |
| **Category** | `Product.category` | One of `["dog_food", "cat_food", "litter", "toys"]` | `CatalogConfig.categories` at `zoosim/core/config.py:17--19` |
| **Strategic flag** | `Product.strategic_flag` | Derived from `strategic_categories` membership | `CatalogConfig.strategic_categories` at `zoosim/core/config.py:74` |

When we refer to "exposure parity across provider groups," we mean exposure balance across these attributes---not across product brand IDs (which the simulator does not model).

!!! note "Code $\leftrightarrow$ Config (provider groups)"
    The fairness module (`zoosim/evaluation/fairness.py`) implements group-based exposure metrics over:
    - `scheme="pl"`: groups are `pl` / `non_pl` via `Product.is_pl`
    - `scheme="category"`: groups are category names from `Product.category`
    - `scheme="strategic"`: groups are `strategic` / `non_strategic` via `Product.strategic_flag`

    These align with the `Product` dataclass at `zoosim/world/catalog.py:15--26`.

---

## 14.2 Multi-Objective Optimization and Pareto Fronts

The chapter title "Multi-Objective RL" requires justification. We are **not** implementing true vector-reward multi-objective RL (Pareto Q-learning, coverage sets); we use CMDP with scalar reward plus constraints. This section explains why the title is nonetheless accurate and what we actually deliver.

### 14.2.1 The Outcome Vector

In our setting, each policy $\pi$ produces an outcome vector:

$$
\mathbf{v}(\pi) = \begin{pmatrix} \text{GMV}(\pi) \\ \text{CM2}(\pi) \\ -\Delta\text{rank}@k(\pi) \\ -\text{exposure\_gap}(\pi) \end{pmatrix} \in \mathbb{R}^4
\tag{14.1}
$$
{#EQ-14.1}

The sign convention ensures all components are **maximized**: GMV and CM2 are intrinsically positive goods, while stability ($\Delta$-rank@$k$) and exposure gap are costs, so we negate them.

**Definition 14.2.1** (Pareto Dominance for Outcomes) {#DEF-14.2.1}

A policy $\pi$ **Pareto dominates** policy $\pi'$ (written $\mathbf{v}(\pi) \succ \mathbf{v}(\pi')$) if:

$$
\forall i \in \{1, 2, 3, 4\}: v_i(\pi) \geq v_i(\pi') \quad \text{and} \quad \exists j: v_j(\pi) > v_j(\pi')
\tag{14.2}
$$
{#EQ-14.2}

That is, $\pi$ is at least as good as $\pi'$ on all objectives and strictly better on at least one.

**Definition 14.2.2** (Pareto Front) {#DEF-14.2.2}

The **Pareto front** $\mathcal{P}$ is the set of outcome vectors achievable by non-dominated policies:

$$
\mathcal{P} = \{\mathbf{v}(\pi) : \nexists \pi' \text{ such that } \mathbf{v}(\pi') \succ \mathbf{v}(\pi)\}
\tag{14.3}
$$
{#EQ-14.3}

The Pareto front represents the boundary of achievable trade-offs. Any improvement in one objective requires sacrificing another.

### 14.2.2 CMDP as the $\varepsilon$-Constraint Method

The CMDP formulation in $\S$14.3 is not an ad hoc simplification---it is the **$\varepsilon$-constraint method** from multi-objective optimization ([@miettinen:multi_objective_optimization:1999, Chapter 4], [@ehrgott:multicriteria_optimization:2005, Chapter 4]).

**The $\varepsilon$-constraint method.** Given $m$ objectives, fix thresholds $\varepsilon_2, \ldots, \varepsilon_m$ for objectives $2$ through $m$, and optimize objective $1$ subject to constraints $v_i(\pi) \geq \varepsilon_i$ for $i = 2, \ldots, m$. Sweeping over threshold values traces the Pareto front.

In our formulation:
- **Primary objective**: GMV (or composite reward $R$)
- **Constraints**: CM2 floor $\tau_{\text{cm2}}$, stability threshold $\tau_{\text{stability}}$, exposure bands $[\tau_{\text{exp}}^{\text{lo}}, \tau_{\text{exp}}^{\text{hi}}]$

By varying these thresholds and training a policy for each setting, we empirically trace the Pareto frontier.

**Theorem 14.2.1** ($\varepsilon$-Constraint Yields Pareto Optimal Points) {#THM-14.2.1}

Let $\pi^*$ solve the CMDP:

$$
\max_{\pi} \; J(\pi) \quad \text{subject to} \quad g_i(\pi) \geq 0, \; i = 1, \ldots, m
$$

If $\pi^*$ is a unique optimal solution, then $\pi^*$ is Pareto optimal. More generally, every Pareto optimal point (including unsupported points) can be found by some choice of constraint thresholds.

*Proof.* See [@miettinen:multi_objective_optimization:1999, Theorem 4.1]. The key insight: if $\pi^*$ were dominated by some $\pi'$, then $\pi'$ would be feasible (since $\mathbf{v}(\pi') \geq \mathbf{v}(\pi^*)$ componentwise implies constraint satisfaction) and achieve higher objective $J(\pi') > J(\pi^*)$, contradicting optimality of $\pi^*$. $\square$

> **Remark 14.2.1** (Supported vs. unsupported points). Linear scalarization (weighted sums) can only find **supported** Pareto points---those on the convex hull of the Pareto front. The $\varepsilon$-constraint method finds **both** supported and unsupported points, including those in "concave" regions of the front. This is why we prefer constraint-based methods over pure scalarization. See Appendix E $\S$E.3.2--E.3.3 for the geometric distinction.

### 14.2.3 Pareto Front Construction Methods

We use two complementary constructions in this chapter:

**Construction A: $\varepsilon$-constraint sweeps (primary).** We sweep constraint targets (CM2 floor, stability $\tau$, exposure targets and bands), train a policy with primal-dual updates for each setting, then evaluate and plot the resulting trade-off curve.

**Construction B: Scalarization sweeps (secondary).** We sweep fixed weights/penalties in a scalar objective, train, and compare the induced curve to Construction A. This reveals where scalarization hides parts of the frontier (unsupported points).

The labs in $\S$14.8 operationalize both constructions:
- `scripts/ch14/lab_01_pareto_fronts.py`: Traces the Pareto front via Construction A
- `scripts/ch14/lab_02_fairness_gap_sweeps.py`: Compares $\varepsilon$-constraint to scalarization

### 14.2.4 Lagrange Multipliers as Marginal Rates of Substitution

The optimal Lagrange multipliers $\boldsymbol{\lambda}^*$ from primal-dual optimization have an economic interpretation. By the KKT conditions (Appendix C $\S$C.2.1, [THM-C.2.1]):

$$
\lambda_i^* = -\frac{\partial J^*}{\partial \tau_i}
\tag{14.4}
$$
{#EQ-14.4}

That is, $\lambda_i^*$ is the **marginal rate of substitution** between the primary objective and constraint $i$. This provides the same trade-off information as the Pareto front slope, but in a form directly usable for business decisions: "How much GMV would we gain if we relaxed the CM2 floor by $\Delta\tau$?"

### 14.2.5 How This Differs from True Vector-Reward MORL

We are honest about what this chapter does **not** deliver:

| What readers might expect | What we actually deliver |
|---------------------------|--------------------------|
| Vector-reward MORL (Pareto Q-learning, multi-policy returns) | CMDP with scalar reward + constraints |
| Dominance-based policy selection | Lagrangian primal-dual optimization |
| Multiple non-dominated policies (coverage set) | One policy per constraint-threshold setting |

**Why CMDP suffices for production ranking:**

1. **Objectives are comparable**: GMV, CM2, $\Delta$-rank, and exposure gaps all have natural units and business-defined acceptable ranges.
2. **Thresholds set by business, not per-query**: Constraint thresholds are stakeholder decisions, not user-specific preferences.
3. **Computational tractability**: CMDP adds only $m$ dual variables to standard policy gradient algorithms; Pareto Q-learning has complexity $O(|\mathcal{S}|^{m-1})$ per update.

For scenarios where true vector-reward MORL is necessary (incomparable objectives, user-specific preferences at query time), see Appendix E.

---

## 14.3 CMDP Formulation for Ranking Constraints

Let $\pi_\theta$ be a parameterized policy. We consider the constrained optimization problem:

$$
\begin{aligned}
\max_{\theta} \quad & J(\theta) := \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{\infty} \gamma^t r_t\right] \\
\text{subject to} \quad & g_i(\theta) := \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{\infty} \gamma^t c_{i,t}\right] - b_i \ge 0,\quad i=1,\ldots,m,
\end{aligned}
\tag{14.5}
$$
{#EQ-14.5}

where $c_{i,t}$ is the $i$-th cost signal and $b_i$ is the constraint threshold. This notation matches Appendix C [EQ-C.1]--[EQ-C.4]: constraints are written in "$\ge 0$ slack" form.

**Example 14.3.1** (Rank stability constraint). {#EX-14.3.1}

If we require $\mathbb{E}[\Delta\text{-rank}@k] \le \tau_{\text{stability}}$, we set:

$$
g_{\text{stability}}(\theta) := \tau_{\text{stability}} - \mathbb{E}_{\tau \sim \pi_\theta}[\Delta\text{-rank}@k] \ge 0
\tag{14.6}
$$
{#EQ-14.6}

When the constraint is satisfied (stability metric below threshold), $g_{\text{stability}} > 0$ (positive slack). When violated, $g_{\text{stability}} < 0$ (negative slack).

**Example 14.3.2** (CM2 floor constraint). {#EX-14.3.2}

For a CM2 floor of $\tau_{\text{cm2}}$:

$$
g_{\text{cm2}}(\theta) := \mathbb{E}_{\tau \sim \pi_\theta}[\text{CM2}] - \tau_{\text{cm2}} \ge 0
\tag{14.7}
$$
{#EQ-14.7}

**Example 14.3.3** (Exposure band constraint). {#EX-14.3.3}

For private-label exposure within band $[\tau_{\text{lo}}, \tau_{\text{hi}}]$:

$$
\begin{aligned}
g_{\text{exp-lo}}(\theta) &:= \mathbb{E}_{\tau \sim \pi_\theta}[\text{exposure}_{\text{pl}}] - \tau_{\text{lo}} \ge 0 \\
g_{\text{exp-hi}}(\theta) &:= \tau_{\text{hi}} - \mathbb{E}_{\tau \sim \pi_\theta}[\text{exposure}_{\text{pl}}] \ge 0
\end{aligned}
\tag{14.8}
$$
{#EQ-14.8}

This requires two constraints (lower and upper bound) to enforce a band.

---

## 14.4 Lagrangian Relaxation and Multipliers

We form the Lagrangian:

$$
\mathcal{L}(\theta, \boldsymbol{\lambda})
=
J(\theta) + \sum_{i=1}^{m} \lambda_i\, g_i(\theta),
\quad \boldsymbol{\lambda} \in \mathbb{R}_+^m
\tag{14.9}
$$
{#EQ-14.9}

where $\boldsymbol{\lambda}$ are nonnegative multipliers. Under Slater's condition (Appendix C [THM-C.2.1]), the saddle-point problem:

$$
\max_\theta \min_{\boldsymbol{\lambda} \ge 0} \mathcal{L}(\theta, \boldsymbol{\lambda})
\tag{14.10}
$$
{#EQ-14.10}

is equivalent to the constrained optimum, with zero duality gap.

**Interpretation.** The Lagrangian augments the reward with constraint terms:

$$
r'_t = r_t + \sum_{i=1}^{m} \lambda_i \cdot (\text{constraint slack})_i
\tag{14.11}
$$
{#EQ-14.11}

When constraint $i$ is violated, $\lambda_i$ penalizes the shaped reward, steering the policy toward feasibility.

---

## 14.5 Primal-Dual Learning

The canonical algorithm alternates:

**Primal step**: improve policy parameters by ascending the Lagrangian:

$$
\theta \leftarrow \theta + \eta_\theta \nabla_\theta \mathcal{L}(\theta, \boldsymbol{\lambda})
\tag{14.12}
$$
{#EQ-14.12}

**Dual step**: adjust multipliers by projected ascent on violations:

$$
\boldsymbol{\lambda} \leftarrow \Pi_{\mathbb{R}_+^m}\!\left(\boldsymbol{\lambda} - \eta_\lambda \nabla_{\boldsymbol{\lambda}} \mathcal{L}(\theta, \boldsymbol{\lambda})\right)
=
\max\!\bigl(0,\, \boldsymbol{\lambda} - \eta_\lambda\, \mathbf{g}(\theta)\bigr)
\tag{14.13}
$$
{#EQ-14.13}

The dual update increases multipliers when constraints are violated ($g_i(\theta) < 0$) and relaxes them when slack exists ($g_i(\theta) > 0$). Appendix C $\S$C.5 gives convergence guarantees under standard step-size conditions.

**Algorithm 14.5.1** (Primal-Dual CMDP Training) {#ALG-14.5.1}

**Input**: Initial policy parameters $\theta_0$, initial multipliers $\boldsymbol{\lambda}_0 = \boldsymbol{0}$, step sizes $\eta_\theta$, $\eta_\lambda$, constraint specs $\{(\text{sense}_i, \tau_i)\}_{i=1}^m$

**For** episode $t = 1, 2, \ldots, T$:

1. **Sample trajectory**: Roll out policy $\pi_{\theta_t}$, observe return $G_t$ and constraint metrics $\{c_i^{(t)}\}_{i=1}^m$

2. **Compute slacks**: For each constraint $i$:
   - If sense is "$\geq$": $\text{slack}_i = c_i^{(t)} - \tau_i$
   - If sense is "$\leq$": $\text{slack}_i = \tau_i - c_i^{(t)}$

3. **Shaped reward**: $r'_t = r_t + \sum_i \lambda_{i,t} \cdot \text{slack}_i$

4. **Primal step** (policy improvement): Update $\theta$ using policy gradient on $r'_t$

5. **Dual step** (multiplier adjustment):
   $$\lambda_{i, t+1} = \max\left(0, \lambda_{i,t} - \eta_\lambda \cdot \text{slack}_i\right)$$

**Output**: Policy $\pi_{\theta_T}$ and multipliers $\boldsymbol{\lambda}_T$

!!! note "Code $\leftrightarrow$ Agent (primal-dual trainer)"
    The reference implementation `zoosim/policies/mo_cmdp.py` implements [ALG-14.5.1]:
    - Constraint specification via `ConstraintSpec` dataclass
    - Dual variable initialization from `ActionConfig.lambda_rank` at `zoosim/core/config.py:230`
    - Policy backbone: REINFORCE with baseline from `zoosim/policies/reinforce_baseline.py`
    - Dual update with projection to $\mathbb{R}_+$

---

## 14.6 Where `lambda_rank` Fits

In the simulator configuration, `lambda_rank` is a named slot for the stability multiplier.

!!! note "Code $\leftrightarrow$ Config (stability multiplier)"
    The configuration exposes `ActionConfig.lambda_rank` at `zoosim/core/config.py:230`. In Chapter 1 this value is not used by the simulator; we reserve it for the CMDP/primal-dual optimization pathway introduced in this chapter.

    To connect it to the Lagrangian, we treat `lambda_rank` as the initial $\lambda_{\text{stability}}$ component in [EQ-14.9], which the dual step in $\S$14.5 then adapts during training.

---

## 14.7 Hard vs. Soft Constraints: Layered Defense

We use consistent language across the book:

| Chapter | Mechanism | Terminology |
|---------|-----------|-------------|
| **Chapter 10** (production safety) | Monitoring, fallback, hard feasibility filters | **Guardrails** |
| **Chapter 14** (optimization) | CMDP, Lagrange multipliers, primal-dual | **Soft constraints** |

The intended workflow is **layered**: even when soft constraints are optimized during training, hard guardrails remain the serving-time safety net. If the CMDP-trained policy somehow violates constraints at inference time (due to distribution shift, for instance), the guardrails catch the violation before it reaches users.

---

## 14.8 Status and Roadmap

This chapter operationalizes the `lambda_rank` placeholder introduced in Chapter 1 and the duality foundations in Appendix C.

### 14.8.1 Implemented Modules

| Module | Purpose | Reference |
|--------|---------|-----------|
| `zoosim/evaluation/fairness.py` | Exposure metrics by provider group | [DEF-14.2.1], $\S$14.1.2 |
| `zoosim/policies/mo_cmdp.py` | Primal-dual CMDP trainer | [ALG-14.5.1] |

### 14.8.2 Lab Scripts

| Script | Purpose |
|--------|---------|
| `scripts/ch14/lab_01_pareto_fronts.py` | Trace Pareto front via $\varepsilon$-constraint sweeps (Construction A) |
| `scripts/ch14/lab_02_fairness_gap_sweeps.py` | Compare $\varepsilon$-constraint to scalarization; fairness gap analysis |

### 14.8.3 Acceptance Criteria (from Syllabus)

The syllabus specifies:

1. **Constraint violation rate**: $< 1\%$ per constraint at convergence
2. **Fairness within bands**: Exposure shares within configured target bands for each provider group
3. **GMV loss quantified**: Explicit comparison to unconstrained baseline (the "price of fairness")

The labs produce CSVs in `docs/book/ch14/data/` documenting these metrics across the Pareto sweep.

---

## 14.9 Exercises and Labs

Exercises and lab solutions are in `docs/book/ch14/exercises_labs.md`.

**Exercise 14.1** (Verify Slater's Condition). For the search ranking CMDP with CM2 floor $\tau_{\text{cm2}} = 0.1$ and stability threshold $\tau_{\text{stability}} = 0.2$, construct a policy (e.g., $\varepsilon$-greedy on a balanced template) that satisfies both constraints with strict slack. This verifies Slater's condition holds.

**Exercise 14.2** (Pareto Front Plotting). Run `scripts/ch14/lab_01_pareto_fronts.py` with CM2 floors $\{0.05, 0.10, 0.15, 0.20\}$. Plot the resulting (GMV, CM2) pairs and identify the Pareto front. Which points are supported? Can you find an unsupported point by varying the stability threshold?

**Exercise 14.3** (Dual Variable Dynamics). Implement a training loop where you log $\lambda_{\text{stability}}$ over episodes. Initialize with $\lambda_0 = 0$ and use $\eta_\lambda = 0.01$. At what episode does $\lambda$ stabilize? Does the final value match the marginal rate of substitution implied by the Pareto front slope?

**Exercise 14.4** (Fairness Band Constraints). Configure exposure bands $[0.3, 0.4]$ for private-label products. Train with primal-dual updates. Does the final policy satisfy the band? What happens when the band is set to $[0.6, 0.7]$ (infeasible for the catalog composition)?

---

## Summary

This chapter established:

1. **CMDP formulation for ranking constraints** ([EQ-14.5]--[EQ-14.8]): CM2 floors, stability thresholds, and exposure bands as slack constraints.

2. **Primal-dual optimization** ([ALG-14.5.1]): Alternating policy improvement and multiplier adjustment to find the constrained optimum.

3. **Multi-objective interpretation** ($\S$14.2): CMDP + threshold sweeps is the $\varepsilon$-constraint method, which traces the Pareto front including unsupported points.

4. **Lagrange multipliers as marginal rates** ([EQ-14.4]): The optimal $\lambda_i^*$ encodes trade-off information directly usable for business decisions.

5. **Layered defense** ($\S$14.7): Soft constraints (CMDP) during training, hard guardrails (Chapter 10) at serving time.

The chapter delivers what the title promises---multi-objective RL for fairness and constraints---via the $\varepsilon$-constraint method rather than vector-reward MORL. For the latter, see Appendix E.

---

## References

**Multi-objective optimization:**
- [@miettinen:multi_objective_optimization:1999, Chapter 4] --- The $\varepsilon$-constraint method for tracing Pareto fronts
- [@ehrgott:multicriteria_optimization:2005, Chapter 4] --- Supported vs. unsupported Pareto points; geometric interpretation

**Constrained MDPs:**
- [@altman:constrained_mdps:1999] --- The foundational monograph on constrained Markov decision processes

**Convex optimization and duality:**
- [@boyd:convex_optimization:2004, Chapter 5] --- Lagrangian duality, Slater's condition, KKT conditions

**Cross-references:**
- Appendix C --- Convex optimization for constrained MDPs (Slater's condition, strong duality, primal-dual convergence)
- Appendix E --- Vector-reward multi-objective RL (Pareto Q-learning, coverage sets, when true MORL is necessary)
- Chapter 10 --- Robustness and guardrails (hard constraints at serving time)
