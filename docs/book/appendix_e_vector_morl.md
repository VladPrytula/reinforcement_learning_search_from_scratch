# Appendix E --- Vector-Reward Multi-Objective RL

**Vlad Prytula**

---

## Motivation

Chapter 14 uses CMDP + $\varepsilon$-constraint sweeps to trace Pareto frontiers for search ranking. This appendix covers the alternative: **vector-reward multi-objective RL (MORL)**, where the agent maintains a vector of Q-values and policies are compared by Pareto dominance rather than scalar reward.

We explain:
1. When this generalization is necessary
2. Why CMDP suffices for production ranking
3. The computational trade-offs between approaches

**Prerequisites.** This appendix assumes familiarity with MDP fundamentals (Chapter 3), Lagrangian duality (Appendix C), and the CMDP formulation (Chapter 14 $\S$14.1--14.3).

**Why this matters.** The Chapter 14 title "Multi-Objective RL" is accurate but requires context. True vector-reward MORL is a distinct and more general framework. This appendix provides that context, ensuring readers understand both what we deliver and what we omit.

---

## E.1 Vector-Reward MDPs

### E.1.1 Definition

**Definition E.1.1** (Vector-Reward MDP) {#DEF-E.1.1}

A **vector-reward MDP** is a tuple $(\mathcal{S}, \mathcal{A}, P, \mathbf{r}, \gamma)$ where:
- $\mathcal{S}$ is the state space
- $\mathcal{A}$ is the action space
- $P: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ is the transition kernel
- $\mathbf{r}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^m$ is a **vector-valued** reward function
- $\gamma \in [0,1)$ is the discount factor

The return under policy $\pi$ is also a vector:

$$
\mathbf{G}^\pi = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t \mathbf{r}(s_t, a_t)\right] \in \mathbb{R}^m
\tag{E.1}
$$
{#EQ-E.1}

**Example E.1.1** (Search ranking as vector MDP). In our setting, we could define:
$$
\mathbf{r}(s,a) = \begin{pmatrix} \text{GMV}(s,a) \\ \text{CM2}(s,a) \\ -\Delta\text{rank}(s,a) \\ -\text{exposure\_gap}(s,a) \end{pmatrix} \in \mathbb{R}^4
$$

Each component is a distinct objective. The sign convention ensures all components should be maximized (costs are negated).

### E.1.2 Pareto Dominance

**Definition E.1.2** (Pareto Dominance) {#DEF-E.1.2}

For vectors $\mathbf{v}, \mathbf{v}' \in \mathbb{R}^m$, we say $\mathbf{v}$ **Pareto dominates** $\mathbf{v}'$ (written $\mathbf{v} \succ \mathbf{v}'$) if:

$$
\forall i \in \{1, \ldots, m\}: v_i \geq v'_i \quad \text{and} \quad \exists j: v_j > v'_j
\tag{E.2}
$$
{#EQ-E.2}

That is, $\mathbf{v}$ is at least as good in all objectives and strictly better in at least one.

**Definition E.1.3** (Pareto Optimal Policy) {#DEF-E.1.3}

A policy $\pi^*$ is **Pareto optimal** if there exists no policy $\pi$ such that $\mathbf{G}^\pi \succ \mathbf{G}^{\pi^*}$.

**Definition E.1.4** (Pareto Front) {#DEF-E.1.4}

The **Pareto front** (or Pareto frontier) is the set of all return vectors achievable by Pareto optimal policies:

$$
\mathcal{P} = \{\mathbf{G}^\pi : \pi \text{ is Pareto optimal}\}
\tag{E.3}
$$
{#EQ-E.3}

> **Remark E.1.1** (Non-uniqueness). Unlike scalar MDPs, vector MDPs typically have **infinitely many** Pareto optimal policies. The goal of MORL is either to (a) find a single policy given user preferences, or (b) approximate the entire Pareto front.

---

## E.2 Pareto Q-Learning

### E.2.1 The Q-Set Approach

In scalar Q-learning, we maintain $Q(s,a) \in \mathbb{R}$. In Pareto Q-learning, we maintain a **set** of non-dominated Q-vectors for each state-action pair.

**Definition E.2.1** (Pareto Q-Set) {#DEF-E.2.1}

For each $(s,a)$, the **Pareto Q-set** $\mathcal{Q}(s,a) \subseteq \mathbb{R}^m$ contains all non-dominated vectors representing possible returns from taking action $a$ in state $s$ and following some Pareto optimal policy thereafter.

**Algorithm E.2.1** (Pareto Q-Learning Update) {#ALG-E.2.1}

Given transition $(s, a, \mathbf{r}, s')$:

1. Compute candidate vectors: $\mathcal{C} = \{\mathbf{r} + \gamma \mathbf{v} : \mathbf{v} \in \mathcal{Q}(s', a') \text{ for some } a'\}$
2. Compute Pareto filter: $\mathcal{Q}(s,a) \leftarrow \text{ND}(\mathcal{Q}(s,a) \cup \mathcal{C})$

where $\text{ND}(\cdot)$ returns the non-dominated subset.

> **Remark E.2.1** (Computational complexity). The Q-sets can grow exponentially in the number of objectives $m$ and the number of states $|\mathcal{S}|$. In the worst case, $|\mathcal{Q}(s,a)|$ can be $O(|\mathcal{S}|^{m-1})$. This makes Pareto Q-learning impractical for large state spaces or many objectives. See [@van_moffaert:morl:2014] for analysis.

### E.2.2 Practical Approximations

Several approaches mitigate the complexity:

1. **Pruning**: Keep only the $k$ most diverse non-dominated points
2. **Clustering**: Group similar Q-vectors and keep representatives
3. **Function approximation**: Learn a parameterized mapping $Q_\theta(s,a) \to \mathbb{R}^m$ (single vector per $(s,a)$, losing multi-policy coverage)

---

## E.3 Coverage Sets and Pareto Front Approximation

### E.3.1 Coverage Sets

**Definition E.3.1** (Coverage Set) {#DEF-E.3.1}

A **coverage set** (CS) for a vector-reward MDP is a finite set of policies $\Pi_{\text{CS}} = \{\pi_1, \ldots, \pi_k\}$ such that for any user preference vector $\mathbf{w} \in \mathbb{R}^m_+$, there exists $\pi_i \in \Pi_{\text{CS}}$ that is optimal or near-optimal under the scalarization $\mathbf{w}^\top \mathbf{G}^\pi$.

Coverage sets are the MORL analog of covering all user preferences with a finite policy library.

### E.3.2 Supported vs. Unsupported Pareto Points

A crucial distinction for understanding the relationship between CMDP/$\varepsilon$-constraint methods and true MORL:

**Definition E.3.2** (Supported Pareto Point) {#DEF-E.3.2}

A Pareto optimal return $\mathbf{v}^* \in \mathcal{P}$ is **supported** if there exists a weight vector $\mathbf{w} \in \mathbb{R}^m_{++}$ (all strictly positive) such that:

$$
\mathbf{v}^* \in \arg\max_{\mathbf{v} \in \mathcal{P}} \mathbf{w}^\top \mathbf{v}
\tag{E.4}
$$
{#EQ-E.4}

**Definition E.3.3** (Unsupported Pareto Point) {#DEF-E.3.3}

A Pareto optimal return $\mathbf{v}^* \in \mathcal{P}$ is **unsupported** if no such weight vector exists. Geometrically, unsupported points lie in "concave" regions of the Pareto front.

> **Remark E.3.1** (Scalarization limitation). Linear scalarization (weighted sums) can only find **supported** Pareto points. The $\varepsilon$-constraint method (Chapter 14) can find **both** supported and unsupported points, which is why it is preferred for tracing the full Pareto front. See [@miettinen:multi_objective_optimization:1999, Chapter 4] for the geometric intuition.

**Example E.3.1** (Unsupported point in ranking). Consider a trade-off between GMV and fairness where:
- Policy A: (GMV=100, fairness=0.5)
- Policy B: (GMV=90, fairness=0.9)
- Policy C: (GMV=80, fairness=0.95)

If the Pareto front is concave between A and B, then B might be unsupported: no linear combination $w_1 \cdot \text{GMV} + w_2 \cdot \text{fairness}$ yields B as optimal. The $\varepsilon$-constraint method finds B by setting $\text{fairness} \geq 0.9$ and maximizing GMV.

### E.3.3 Empirical Pareto Front Evaluation

**Definition E.3.4** (Hypervolume Indicator) {#DEF-E.3.4}

Given a reference point $\mathbf{r}_{\text{ref}} \in \mathbb{R}^m$ (typically a lower bound on all objectives), the **hypervolume indicator** of a Pareto front approximation $\mathcal{P}'$ is the $m$-dimensional volume dominated by $\mathcal{P}'$ and bounded by $\mathbf{r}_{\text{ref}}$:

$$
\text{HV}(\mathcal{P}', \mathbf{r}_{\text{ref}}) = \text{Vol}\left(\bigcup_{\mathbf{v} \in \mathcal{P}'} [\mathbf{r}_{\text{ref}}, \mathbf{v}]\right)
\tag{E.5}
$$
{#EQ-E.5}

Hypervolume is a standard metric for comparing Pareto front approximations ([@vamplew:morl:2011]).

---

## E.4 Multi-Policy vs. Single-Policy MORL

### E.4.1 Taxonomy

MORL algorithms fall into two categories:

| Approach | Goal | Output | Example |
|----------|------|--------|---------|
| **Multi-policy** | Approximate entire Pareto front | Set of policies $\Pi_{\text{CS}}$ | Pareto Q-learning |
| **Single-policy** | Find one policy given preferences | Single policy $\pi^*$ | Scalarized RL, CMDP |

### E.4.2 When to Use Each

**Multi-policy MORL** is appropriate when:
1. User preferences are unknown at training time
2. Different users have different (and potentially incomparable) preferences
3. The system must serve diverse user populations with a policy library

**Single-policy MORL** (including CMDP) is appropriate when:
1. Business constraints are fixed by domain experts
2. One policy serves all users
3. Objectives can be scalarized or constrained with known thresholds

---

## E.5 Why CMDP Suffices for Production Ranking

Chapter 14 uses CMDP with primal-dual optimization rather than true vector-reward MORL. Here's why this is the right choice for e-commerce search ranking:

### E.5.1 Objectives Are Comparable

In our setting:
- **GMV** and **CM2** are both measured in currency (dollars, euros)
- **Stability** ($\Delta$-rank) has a clear business threshold (e.g., $\leq 5$ position changes)
- **Fairness gaps** have regulatory or policy-defined bands

These objectives have natural units and business-defined acceptable ranges. There is no "incomparability" requiring user-specific preference elicitation.

### E.5.2 Thresholds Set by Business, Not Per-Query

A key observation: constraint thresholds $\tau_{\text{cm2}}$, $\tau_{\text{stability}}$, $\tau_{\text{fairness}}$ are set by business stakeholders, not by individual users at query time. This means:

- **One policy per threshold setting** is sufficient
- The $\varepsilon$-constraint sweep (varying thresholds, training one policy each) traces the Pareto front adequately
- No need for a coverage set or multi-policy library

### E.5.3 Computational Tractability

| Approach | Complexity | Practicality |
|----------|------------|--------------|
| Pareto Q-learning | $O(|\mathcal{S}|^{m-1})$ per update | Impractical for $m > 3$ |
| CMDP + primal-dual | Same as single-objective RL | Production-ready |

CMDP adds only $m$ dual variables (Lagrange multipliers) to a standard policy gradient algorithm. The overhead is negligible.

### E.5.4 Lagrange Multipliers Encode Trade-offs

The optimal multipliers $\lambda^*$ from CMDP have an economic interpretation: $\lambda_i^*$ is the **marginal rate of substitution** between the reward and constraint $i$. This provides the same trade-off information as the Pareto front slope, but in a form directly usable for business decisions.

---

## E.6 When You Actually Need True MORL

Despite the above, there are scenarios where CMDP is insufficient:

1. **Incomparable objectives**: When objectives have no natural common scale (e.g., "user satisfaction" vs. "environmental impact") and no stakeholder can set thresholds.

2. **User-specific preferences at query time**: If different users want different GMV-fairness trade-offs and these preferences are revealed at serving time, a coverage set is needed.

3. **Fairness as Pareto constraint**: Some fairness definitions require that no policy should dominate another with respect to group-specific outcomes. This is a Pareto-based fairness criterion that cannot be expressed as a scalar constraint.

4. **Exploratory analysis**: When the Pareto front shape is unknown and business stakeholders want to see all achievable trade-offs before setting thresholds.

For these cases, the references in $\S$E.7 provide starting points.

---

## Summary

This appendix established the relationship between CMDP (Chapter 14) and true vector-reward MORL:

| Aspect | CMDP + $\varepsilon$-constraint | Vector-reward MORL |
|--------|--------------------------------|-------------------|
| **Reward** | Scalar + constraints | Vector-valued |
| **Policy comparison** | Feasibility + scalar optimality | Pareto dominance |
| **Output** | One policy per threshold | Coverage set |
| **Pareto coverage** | Supported + unsupported | Full front |
| **Complexity** | Standard RL + $m$ multipliers | Exponential in $m$ |
| **Use case** | Business-defined thresholds | User-specific preferences |

**Key takeaway**: For production search ranking with business-defined constraints, CMDP is the practical choice. It traces the Pareto frontier via threshold sweeps without the computational burden of maintaining Q-sets. True vector-reward MORL is reserved for scenarios with incomparable objectives or user-specific preferences revealed at serving time.

---

## E.7 References and Further Reading

**Foundational surveys:**
- [@roijers:survey_morl:2013] --- Comprehensive survey on multi-objective sequential decision-making; defines taxonomy and coverage sets
- [@hayes:morl_survey:2022] --- Modern practical guide with algorithm implementations

**Pareto Q-learning:**
- [@van_moffaert:morl:2014] --- Pareto Q-learning with non-dominated policy sets; complexity analysis

**Multi-objective optimization (classical):**
- [@miettinen:multi_objective_optimization:1999] --- Foundational text on $\varepsilon$-constraint and scalarization methods
- [@ehrgott:multicriteria_optimization:2005] --- Comprehensive treatment of Pareto optimality; supported vs. unsupported points

**Empirical evaluation:**
- [@vamplew:morl:2011] --- Coverage sets, hypervolume indicator, and empirical Pareto front evaluation methods

**Constrained MDPs:**
- [@altman:constrained_mdps:1999] --- The foundational monograph on constrained Markov decision processes
