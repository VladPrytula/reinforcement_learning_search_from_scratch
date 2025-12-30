# Appendix C --- Convex Optimization for Constrained MDPs

**Vlad Prytula**

---

## Motivation

Chapter 1 introduced the constrained optimization problem [EQ-1.18]: maximize expected reward subject to margin floors (CM2), exposure requirements, and rank stability constraints. We claimed that under **Slater's condition**, the Lagrangian saddle-point problem [EQ-1.19] is equivalent to the original constrained problem---strong duality holds, with no gap between primal and dual optimal values.

This appendix provides the rigorous foundations for that claim. We develop **Lagrangian duality** for constrained optimization, state and prove **Slater's condition** as a sufficient condition for strong duality, and show how these results apply to constrained MDPs. The treatment follows the presentation in [@boyd:convex_optimization:2004, Chapter 5], adapted to the RL setting with randomized policies.

**Prerequisites.** This appendix assumes familiarity with convex sets and convex functions ([@boyd:convex_optimization:2004, Chapters 2--3]). Readers unfamiliar with convex analysis should review these foundations before proceeding.

**Why this matters for RL.** Constrained RL is ubiquitous in production systems: budget constraints in recommendation, safety constraints in robotics, fairness constraints in ranking. Without strong duality, there is no guarantee that Lagrangian methods find the constrained optimum. Slater's condition is the bridge between the tractable Lagrangian formulation and the intractable constrained problem.

---

## C.1 Lagrangian Duality: Setup and Definitions

### C.1.1 The Primal Problem

Consider a constrained optimization problem in standard form:

$$
\begin{aligned}
\text{maximize} \quad & J(\pi) \\
\text{subject to} \quad & g_i(\pi) \geq 0, \quad i = 1, \ldots, m
\end{aligned}
\tag{C.1}
$$
{#EQ-C.1}

where:
- $\Pi$ is the **policy space** (set of admissible policies)
- $J: \Pi \to \mathbb{R}$ is the **objective function** (expected return)
- $g_i: \Pi \to \mathbb{R}$ are **constraint functions** (e.g., $g_1(\pi) = \mathbb{E}_\pi[\text{CM2}] - C_{\min}$)

The constraint $g_i(\pi) \geq 0$ means the policy must satisfy the $i$-th requirement with nonnegative slack. We denote the **optimal primal value** as $p^*$.

**Example C.1.1** (Search ranking constraints from Chapter 1). {#EX-C.1.1}

In the search ranking problem, we have:
- $J(\pi) = \mathbb{E}_\pi[R]$ where $R$ is the composite reward [EQ-1.2]
- $g_1(\pi) = \mathbb{E}_\pi[\text{CM2}] - \tau_{\text{CM2}}$ (margin floor)
- $g_2(\pi) = \mathbb{E}_\pi[\text{STRAT}] - \tau_{\text{STRAT}}$ (strategic exposure floor)

The feasible set $\{\pi \in \Pi : g_i(\pi) \geq 0, \forall i\}$ consists of policies satisfying both business constraints.

> **Notation (scope: Appendix C):** Here $\text{STRAT}$ denotes strategic *exposure* (items shown in the ranking). This is distinct from $\text{STRAT}$ in the reward [EQ-1.2], where it counts strategic *purchases*.

### C.1.2 The Lagrangian

**Definition C.1.1** (Lagrangian) {#DEF-C.1.1}

The **Lagrangian** $\mathcal{L}: \Pi \times \mathbb{R}_+^m \to \mathbb{R}$ is defined as:

$$
\mathcal{L}(\pi, \boldsymbol{\lambda}) = J(\pi) + \sum_{i=1}^{m} \lambda_i g_i(\pi)
\tag{C.2}
$$
{#EQ-C.2}

where $\boldsymbol{\lambda} = (\lambda_1, \ldots, \lambda_m) \in \mathbb{R}_+^m$ are **Lagrange multipliers** (dual variables).

The Lagrangian penalizes constraint violations: if $g_i(\pi) < 0$ (constraint violated), the term $\lambda_i g_i(\pi)$ subtracts from the objective. Conversely, if $g_i(\pi) > 0$ (constraint satisfied with slack), the penalty is positive.

### C.1.3 The Dual Function and Dual Problem

**Definition C.1.2** (Dual Function) {#DEF-C.1.2}

The **Lagrangian dual function** $d: \mathbb{R}_+^m \to \mathbb{R} \cup \{+\infty\}$ is:

$$
d(\boldsymbol{\lambda}) = \sup_{\pi \in \Pi} \mathcal{L}(\pi, \boldsymbol{\lambda})
\tag{C.3}
$$
{#EQ-C.3}

The dual function gives the optimal value of the unconstrained Lagrangian for fixed multipliers. It is always convex in $\boldsymbol{\lambda}$ (even if the primal problem is non-convex), as the pointwise supremum of affine functions.

**Definition C.1.3** (Dual Problem) {#DEF-C.1.3}

The **Lagrangian dual problem** is:

$$
\text{minimize} \quad d(\boldsymbol{\lambda}) \quad \text{subject to} \quad \boldsymbol{\lambda} \geq 0
\tag{C.4}
$$
{#EQ-C.4}

We denote the **optimal dual value** as $d^*$.

### C.1.4 Weak Duality

**Lemma C.1.1** (Weak Duality) {#LEM-C.1.1}

For any feasible $\pi$ (satisfying $g_i(\pi) \geq 0$) and any $\boldsymbol{\lambda} \geq 0$:

$$
J(\pi) \leq \mathcal{L}(\pi, \boldsymbol{\lambda}) \leq d(\boldsymbol{\lambda})
$$

Consequently, $p^* \leq d^*$ (the optimal dual value is an upper bound on the optimal primal value).

*Proof.*

**Step 1** (First inequality). Since $\pi$ is feasible, $g_i(\pi) \geq 0$ for all $i$. Since $\lambda_i \geq 0$, we have $\lambda_i g_i(\pi) \geq 0$. Thus:
$$
\mathcal{L}(\pi, \boldsymbol{\lambda}) = J(\pi) + \sum_{i=1}^{m} \lambda_i g_i(\pi) \geq J(\pi)
$$

**Step 2** (Second inequality). By definition, $d(\boldsymbol{\lambda}) = \sup_{\pi' \in \Pi} \mathcal{L}(\pi', \boldsymbol{\lambda}) \geq \mathcal{L}(\pi, \boldsymbol{\lambda})$.

**Step 3** (Weak duality). Taking supremum over feasible $\pi$ on the left and infimum over $\boldsymbol{\lambda} \geq 0$ on the right:
$$
p^* = \sup_{\pi \text{ feasible}} J(\pi) \leq \inf_{\boldsymbol{\lambda} \geq 0} d(\boldsymbol{\lambda}) = d^*
$$
$\square$

> **Remark C.1.1** (The duality gap). The quantity $d^* - p^* \geq 0$ is called the **duality gap**. Weak duality says the gap is nonnegative. Strong duality (when it holds) says the gap is zero. The duality gap measures how much we "lose" by solving the easier dual problem instead of the harder primal problem.

---

## C.2 Slater's Condition and Strong Duality

### C.2.1 Statement of the Main Result

We now establish conditions under which weak duality becomes **strong duality**: $p^* = d^*$. The key is **Slater's condition**, which requires the existence of a strictly feasible point.

**Definition C.2.1** (Strict Feasibility / Slater Point) {#DEF-C.2.1}

A policy $\pi_0 \in \Pi$ is **strictly feasible** (or a **Slater point**) if:

$$
g_i(\pi_0) > 0 \quad \text{for all } i = 1, \ldots, m
$$

That is, $\pi_0$ satisfies all constraints with **strict inequality** (positive slack).

**Theorem C.2.1** (Slater's Condition for Strong Duality) {#THM-C.2.1}

Consider the constrained optimization problem [EQ-C.1]. Assume:

1. **(Convexity of policy space)** $\Pi$ is a **convex** set
2. **(Concavity of objective)** $J: \Pi \to \mathbb{R}$ is **concave**
3. **(Concavity of constraints)** Each $g_i: \Pi \to \mathbb{R}$ is **concave**
4. **(Slater's condition)** There exists a strictly feasible policy $\pi_0 \in \Pi$ with $g_i(\pi_0) > 0$ for all $i$

**Then:**

**(a)** Strong duality holds: $p^* = d^*$

**(b)** The dual optimum is attained: there exists $\boldsymbol{\lambda}^* \geq 0$ with $d(\boldsymbol{\lambda}^*) = d^* = p^*$

**(c)** Saddle-point equivalence: If $\pi^*$ is an optimal primal solution and $\boldsymbol{\lambda}^*$ is an optimal dual solution, then $(\pi^*, \boldsymbol{\lambda}^*)$ is a saddle point of the Lagrangian:

$$
\mathcal{L}(\pi^*, \boldsymbol{\lambda}) \leq \mathcal{L}(\pi^*, \boldsymbol{\lambda}^*) \leq \mathcal{L}(\pi, \boldsymbol{\lambda}^*)
\tag{C.5}
$$
{#EQ-C.5}

for all $\pi \in \Pi$ and $\boldsymbol{\lambda} \geq 0$.

**(d)** KKT conditions are necessary and sufficient: $(\pi^*, \boldsymbol{\lambda}^*)$ is a primal-dual optimal pair if and only if:

- **Primal feasibility:** $g_i(\pi^*) \geq 0$ for all $i$
- **Dual feasibility:** $\lambda_i^* \geq 0$ for all $i$
- **Complementary slackness:** $\lambda_i^* g_i(\pi^*) = 0$ for all $i$
- **Stationarity:** $\pi^*$ maximizes $\mathcal{L}(\cdot, \boldsymbol{\lambda}^*)$ over $\Pi$

### C.2.2 Proof of Theorem C.2.1

*Proof.*

We prove the theorem in several steps, following the geometric approach of [@boyd:convex_optimization:2004, Section 5.2.3].

**Step 1** (Define the value function). For $\boldsymbol{u} \in \mathbb{R}^m$, define the **perturbation function**:

$$
p(\boldsymbol{u}) = \sup\{J(\pi) : \pi \in \Pi, \; g_i(\pi) \geq u_i, \; \forall i\}
$$

This is the optimal value when constraints are perturbed by $\boldsymbol{u}$. Note that $p(\boldsymbol{0}) = p^*$ (the original primal value).

**Step 2** (Concavity of $p$). Under assumptions (1)--(3), the feasible set $\{\pi : g_i(\pi) \geq u_i\}$ is convex for any $\boldsymbol{u}$, and $J$ is concave. The supremum of a concave function over a convex set that varies convexly with parameters yields a concave perturbation function. Thus $p$ is concave on its effective domain.

**Step 3** (Dual function as conjugate). The dual function satisfies:

$$
d(\boldsymbol{\lambda}) = \sup_{\pi \in \Pi} \left[J(\pi) + \sum_i \lambda_i g_i(\pi)\right] = \sup_{\boldsymbol{u}} \left[p(\boldsymbol{u}) + \boldsymbol{\lambda}^\top \boldsymbol{u}\right]
$$

where the second equality uses the definition of $p$ and rearranges the optimization. This shows $d(\boldsymbol{\lambda})$ is the **conjugate** of $-p$ evaluated at $-\boldsymbol{\lambda}$.

**Step 4** (Apply Slater's condition). Slater's condition ensures that $\boldsymbol{0}$ lies in the **interior** of the effective domain of $p$: since $\pi_0$ satisfies $g_i(\pi_0) > 0$, we can perturb constraints slightly ($u_i < 0$) while maintaining feasibility. By convex analysis ([@rockafellar:convex_analysis:1970, Theorem 28.2]), if a concave function's domain has nonempty interior and the function is not identically $-\infty$, then:

$$
p(\boldsymbol{0}) = \inf_{\boldsymbol{\lambda} \geq 0} d(\boldsymbol{\lambda})
$$

That is, $p^* = d^*$. This establishes **(a)**.

**Step 5** (Dual attainment). The infimum in Step 4 is attained under Slater's condition: the sublevel sets of $d$ are bounded (finite optimal dual value) and closed (lower semicontinuity). By the extreme value theorem on compact sets, there exists $\boldsymbol{\lambda}^*$ achieving $d(\boldsymbol{\lambda}^*) = d^*$. This establishes **(b)**.

**Step 6** (Saddle-point characterization). If $\pi^*$ is primal optimal and $\boldsymbol{\lambda}^*$ is dual optimal:

- **Left inequality** [EQ-C.5]: For any $\boldsymbol{\lambda} \geq 0$,
$$
\mathcal{L}(\pi^*, \boldsymbol{\lambda}) = J(\pi^*) + \sum_i \lambda_i g_i(\pi^*) \leq J(\pi^*) + \sum_i \lambda_i^* g_i(\pi^*) = \mathcal{L}(\pi^*, \boldsymbol{\lambda}^*)
$$
The inequality follows from complementary slackness: $\lambda_i^* g_i(\pi^*) = 0$ and $g_i(\pi^*) \geq 0$, so $\sum_i \lambda_i g_i(\pi^*) \leq 0 = \sum_i \lambda_i^* g_i(\pi^*)$ when $\boldsymbol{\lambda} \neq \boldsymbol{\lambda}^*$ (we maximize over $\boldsymbol{\lambda}$... actually, this direction is subtle).

Let us be more careful. By strong duality, $J(\pi^*) = p^* = d^* = d(\boldsymbol{\lambda}^*)$. Since $\pi^*$ is feasible, $g_i(\pi^*) \geq 0$, so:

$$
\mathcal{L}(\pi^*, \boldsymbol{\lambda}) = J(\pi^*) + \sum_i \lambda_i g_i(\pi^*) \leq J(\pi^*) = \mathcal{L}(\pi^*, \boldsymbol{\lambda}^*)
$$

where the last equality holds because complementary slackness gives $\sum_i \lambda_i^* g_i(\pi^*) = 0$. This is the left inequality.

- **Right inequality** [EQ-C.5]: By definition of $d(\boldsymbol{\lambda}^*)$,
$$
\mathcal{L}(\pi, \boldsymbol{\lambda}^*) \leq \sup_{\pi' \in \Pi} \mathcal{L}(\pi', \boldsymbol{\lambda}^*) = d(\boldsymbol{\lambda}^*) = p^* = J(\pi^*) = \mathcal{L}(\pi^*, \boldsymbol{\lambda}^*)
$$

This establishes **(c)**.

**Step 7** (KKT conditions). The KKT conditions follow from the saddle-point characterization. If $(\pi^*, \boldsymbol{\lambda}^*)$ is a saddle point:

- **Primal feasibility** and **dual feasibility** are built into the problem formulation.
- **Complementary slackness**: From the left inequality with $\boldsymbol{\lambda} = \boldsymbol{0}$:
$$
J(\pi^*) = \mathcal{L}(\pi^*, \boldsymbol{0}) \leq \mathcal{L}(\pi^*, \boldsymbol{\lambda}^*) = J(\pi^*) + \sum_i \lambda_i^* g_i(\pi^*)
$$
Thus $\sum_i \lambda_i^* g_i(\pi^*) \geq 0$. But since $\lambda_i^* \geq 0$ and $g_i(\pi^*) \geq 0$, each term is nonnegative, and the only way equality holds (required by strong duality) is if each term is zero: $\lambda_i^* g_i(\pi^*) = 0$ for all $i$.
- **Stationarity**: The right inequality shows $\pi^*$ maximizes $\mathcal{L}(\cdot, \boldsymbol{\lambda}^*)$.

Conversely, if KKT holds, one can verify that $(\pi^*, \boldsymbol{\lambda}^*)$ is a saddle point, hence primal-dual optimal. This establishes **(d)**.

$\square$

> **Remark C.2.1** (The proof technique). This proof uses the **conjugate duality** framework: the dual function is the conjugate of the negative perturbation function. Slater's condition ensures the perturbation function is well-behaved (concave, finite, interior point), which guarantees zero duality gap via convex conjugate theory. This is a **duality argument**: we pass to the conjugate space where the optimization structure is more transparent.

> **Remark C.2.2** (Necessity of Slater's condition). Slater's condition is **sufficient** but not **necessary** for strong duality. There are convex problems where strong duality holds without a strictly feasible point (e.g., linear programs). However, Slater's condition is the **weakest** easily checkable sufficient condition for a broad class of problems.

---

## C.3 Randomized Policies as Convex Sets

A crucial question: when does the policy space $\Pi$ satisfy the convexity assumption (1) of Theorem C.2.1? In RL, deterministic policies are typically **not** convex (they form the vertices of a polytope). However, **randomized policies** are convex.

### C.3.1 The Policy Polytope

**Proposition C.3.1** (Randomized Policies Form a Convex Set) {#PROP-C.3.1}

Let $\Pi_{\det}$ be the set of deterministic policies for a finite MDP. Define the set of randomized policies:

$$
\Pi_{\text{rand}} = \left\{\pi : \pi(a|s) = \sum_{k=1}^{|\Pi_{\det}|} \alpha_k \pi_k(a|s), \; \alpha_k \geq 0, \; \sum_k \alpha_k = 1\right\}
$$

where each $\pi_k \in \Pi_{\det}$ is a deterministic policy. Then $\Pi_{\text{rand}}$ is **convex**.

*Proof.* By construction, $\Pi_{\text{rand}}$ is the convex hull of $\Pi_{\det}$. The convex hull of any set is convex (the smallest convex set containing it). $\square$

### C.3.2 Linearity of Objective and Constraints

**Proposition C.3.2** (Expected Values are Linear in Randomized Policies) {#PROP-C.3.2}

For any reward function $R$ and constraint function $c$:

- The expected return $J(\pi) = \mathbb{E}_\pi[R]$ is **linear** in $\pi$ over $\Pi_{\text{rand}}$
- The expected constraint value $g(\pi) = \mathbb{E}_\pi[c] - \tau$ is **linear** in $\pi$ over $\Pi_{\text{rand}}$

*Proof.* Linearity of expectation. If $\pi = \alpha \pi_1 + (1-\alpha)\pi_2$ (convex combination of policies), then:

$$
\mathbb{E}_\pi[R] = \sum_a \pi(a|s) R(s,a) = \alpha \sum_a \pi_1(a|s) R(s,a) + (1-\alpha) \sum_a \pi_2(a|s) R(s,a) = \alpha \mathbb{E}_{\pi_1}[R] + (1-\alpha) \mathbb{E}_{\pi_2}[R]
$$

This is precisely linearity. $\square$

> **Remark C.3.1** (Linear functions are both convex and concave). A linear function $f$ satisfies $f(\alpha x + (1-\alpha) y) = \alpha f(x) + (1-\alpha) f(y)$, which is the equality case of both the convex and concave definitions. Thus linear objectives and constraints automatically satisfy hypotheses (2) and (3) of Theorem C.2.1.

### C.3.3 Slater's Condition for MDPs

Combining the above:

**Corollary C.3.1** (Strong Duality for Constrained MDPs) {#COR-C.3.1}

Consider a constrained MDP with:
- Finite state and action spaces
- Linear reward $J(\pi) = \mathbb{E}_\pi[R]$
- Linear constraints $g_i(\pi) = \mathbb{E}_\pi[c_i] - \tau_i$ for $i = 1, \ldots, m$

If there exists a policy $\pi_0$ (deterministic or randomized) such that $\mathbb{E}_{\pi_0}[c_i] > \tau_i$ for all $i$ (strict feasibility), then:

1. Strong duality holds: the constrained MDP value equals the Lagrangian saddle-point value
2. The optimal policy may be randomized, even if the unconstrained optimal policy is deterministic
3. Primal-dual algorithms converge to the constrained optimum

*Proof.* Apply Theorem C.2.1 with $\Pi = \Pi_{\text{rand}}$ (convex by Proposition C.3.1), noting that $J$ and $g_i$ are linear (hence concave) by Proposition C.3.2. $\square$

> **Remark C.3.2** (Deterministic policies may be suboptimal under constraints). A classic result in constrained MDPs: the optimal policy may require randomization even when the unconstrained problem has a deterministic optimal policy. Constraints can force the agent to "hedge" between deterministic policies to satisfy all requirements simultaneously. See [@altman:constrained_mdps:1999, Chapter 3] for examples.

---

## C.4 Application to Search Ranking (Chapter 1)

We now connect the theory to the search ranking problem from Chapter 1.

### C.4.1 The Constrained Search Problem

Recall the constrained optimization from [EQ-1.18]:

$$
\begin{aligned}
\max_{\pi} \quad & \mathbb{E}_\pi[R] \\
\text{s.t.} \quad & \mathbb{E}_\pi[\text{CM2}] \geq \tau_{\text{CM2}} \\
& \mathbb{E}_\pi[\text{STRAT}] \geq \tau_{\text{STRAT}}
\end{aligned}
$$

Defining:
- $g_1(\pi) = \mathbb{E}_\pi[\text{CM2}] - \tau_{\text{CM2}}$
- $g_2(\pi) = \mathbb{E}_\pi[\text{STRAT}] - \tau_{\text{STRAT}}$

This is exactly the form [EQ-C.1].

### C.4.2 Verifying Slater's Condition

**How to check Slater's condition in practice:**

1. **Baseline policy test**: If the current production policy $\pi_{\text{prod}}$ satisfies:
$$
\mathbb{E}_{\pi_{\text{prod}}}[\text{CM2}] > \tau_{\text{CM2}} \quad \text{and} \quad \mathbb{E}_{\pi_{\text{prod}}}[\text{STRAT}] > \tau_{\text{STRAT}}
$$
then Slater's condition holds (the baseline is strictly feasible).

2. **$\varepsilon$-greedy exploration**: An $\varepsilon$-greedy policy with uniform exploration often satisfies constraints with slack, because it samples across the action space (including high-CM2 and high-STRAT actions).

3. **Constraint relaxation**: If Slater's condition fails, relax constraints slightly:
$$
\mathbb{E}_\pi[\text{CM2}] \geq \tau_{\text{CM2}} - \varepsilon
$$
This often restores strict feasibility for small $\varepsilon > 0$.

### C.4.3 When Slater Fails

If **no** strictly feasible policy exists, the constraints may be **infeasible**: you're asking for profitability floors that no ranking can achieve. Symptoms:

- Dual variables $\boldsymbol{\lambda}$ diverge during primal-dual training
- No policy in the training set satisfies all constraints
- The Pareto frontier (GMV vs. CM2 tradeoff) lies entirely below the constraint threshold

**Diagnosis**: Plot $\mathbb{E}_\pi[\text{CM2}]$ vs. $\mathbb{E}_\pi[R]$ across a range of policies. If the CM2 floor $\tau_{\text{CM2}}$ lies above the entire frontier, the constraint is infeasible.

---

## C.5 Primal-Dual Algorithms

### C.5.1 Primal-Dual Gradient Ascent-Descent

Under the strong duality guarantee of Theorem C.2.1, we can solve the constrained MDP via **primal-dual gradient methods**:

**Algorithm C.5.1** (Primal-Dual RL) {#ALG-C.5.1}

**Input**: Initial policy parameters $\theta_0$, initial multipliers $\boldsymbol{\lambda}_0 = \mathbf{0}$, step sizes $\eta_\theta$, $\eta_\lambda$

**For** episode $t = 1, 2, \ldots$:

1. **Sample trajectory**: Roll out policy $\pi_{\theta_t}$, observe return $G_t$ and constraint violations $v_i^{(t)} = \tau_i - c_i^{(t)}$ where $c_i^{(t)}$ is the empirical constraint value

2. **Primal step** (policy improvement):
$$
\theta_{t+1} = \theta_t + \eta_\theta \nabla_\theta \mathcal{L}(\pi_{\theta_t}, \boldsymbol{\lambda}_t)
$$
where $\nabla_\theta \mathcal{L} = \nabla_\theta J(\theta) + \sum_i \lambda_i \nabla_\theta g_i(\theta)$

3. **Dual step** (multiplier adjustment):
$$
\lambda_{i, t+1} = \max\left(0, \lambda_{i,t} + \eta_\lambda \cdot v_i^{(t)}\right)
$$
If constraint $i$ is violated ($v_i > 0$), increase $\lambda_i$; if satisfied with slack ($v_i < 0$), decrease $\lambda_i$ (but keep $\lambda_i \geq 0$).

**Output**: Policy $\pi_{\theta_T}$ and multipliers $\boldsymbol{\lambda}_T$

### C.5.2 Convergence Guarantee

**Theorem C.5.1** (Primal-Dual Convergence) {#THM-C.5.1}

Under the hypotheses of Theorem C.2.1 (convexity + Slater), if:

1. The policy class $\{\pi_\theta\}$ can represent the optimal constrained policy
2. Step sizes satisfy $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$ (Robbins-Monro conditions)
3. Gradient estimates are unbiased with bounded variance

Then Algorithm C.5.1 converges:

$$
(\theta_t, \boldsymbol{\lambda}_t) \to (\theta^*, \boldsymbol{\lambda}^*)
$$

where $(\theta^*, \boldsymbol{\lambda}^*)$ is a saddle point of the Lagrangian.

*Proof sketch.* Under convexity, the Lagrangian is concave in $\theta$ and convex in $\boldsymbol{\lambda}$. Primal-dual gradient dynamics converge to saddle points for convex-concave functions under the Robbins-Monro step size conditions. The proof uses stochastic approximation theory; see [@borkar:stochastic_approximation:2008, Chapter 6] for details. $\square$

### C.5.3 Practical Considerations

1. **Learning rate ratio**: The dual step size $\eta_\lambda$ should typically be smaller than $\eta_\theta$ (e.g., $\eta_\lambda = 0.1 \eta_\theta$). Dual variables adapt to aggregate constraint satisfaction; they should not overreact to single-episode noise.

2. **Constraint scaling**: Normalize constraints to comparable scales. If CM2 is in dollars and STRAT is a count, their violations have different magnitudes. Rescale to $[0, 1]$ or use adaptive normalization.

3. **Warm start**: Initialize $\boldsymbol{\lambda}_0$ near zero but positive (e.g., $\lambda_i = 0.01$). Zero initialization can cause slow constraint activation.

4. **Monitoring**: Track both primal objective $J(\theta_t)$ and constraint violations $v_i^{(t)}$. A good primal-dual run shows $J$ increasing while violations decrease.

---

## C.6 Summary

**Key results established:**

1. **Theorem C.2.1 (Slater's condition)**: Under convexity + strict feasibility, strong duality holds---the constrained MDP value equals the Lagrangian saddle-point value, with zero duality gap.

2. **Randomized policies form convex sets** (Proposition C.3.1): Even when deterministic policies are non-convex (finitely many vertices), their randomized extensions are convex.

3. **Linear objectives/constraints satisfy convexity** (Proposition C.3.2): Expected rewards and expected constraint values are linear in the policy distribution, hence both convex and concave.

4. **Primal-dual algorithms converge** (Theorem C.5.1): Under Slater's condition, alternating primal-dual gradient updates find the constrained optimum.

**When to consult this appendix:**

- **Chapter 1, Section 1.9**: Verifying that the Lagrangian reformulation [EQ-1.19] is equivalent to the constrained problem [EQ-1.18]
- **Chapter 10**: Constraints in production: guardrails (monitoring, fallback, and hard feasibility filters)
- **Chapter 14**: Multi-objective RL and fairness constraints (Pareto frontier computation)

For readers familiar with convex optimization, this appendix transfers existing knowledge to constrained MDPs. For those new to duality theory, the key takeaway is: **if a strictly feasible policy exists, you can safely use Lagrangian methods**---they will find the constrained optimum without duality gap.

---

## C.7 References and Further Reading

**Convex optimization foundations:**
- [@boyd:convex_optimization:2004, Chapter 5]: Comprehensive treatment of Lagrangian duality, Slater's condition, and KKT conditions
- [@rockafellar:convex_analysis:1970]: Classical reference for convex analysis, conjugate functions, and saddle-point theory

**Constrained MDPs:**
- [@altman:constrained_mdps:1999]: The foundational monograph on constrained Markov decision processes, including long-run average constraints and randomized optimal policies
- [@borkar:stochastic_approximation:2008, Chapter 6]: Stochastic approximation theory for primal-dual convergence

**Applications in RL:**
- [@achiam:constrained_policy:2017]: Constrained Policy Optimization (CPO) for safe RL with trust region methods
- [@tessler:reward_constrained:2019]: Reward-constrained policy optimization for production systems
- [@stooke:responsive_safety:2020]: Responsive safety layers for constraint satisfaction during training
