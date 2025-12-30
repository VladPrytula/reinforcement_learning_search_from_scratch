# Chapter 10: Robustness to Drift and Guardrails

**Vlad Prytula**

---

> "A ranking system that maximizes clicks but destroys user trust is not optimal; it is merely extracting value faster than it creates it. True optimization requires safety constraints."

## Introduction: When the World Shifts Beneath Your Feet

In Chapters 1–9, we built a comprehensive framework for reinforcement learning in e-commerce search. We derived rigorous convergence guarantees for bandits (Chapter 6), proved contraction mappings for Bellman operators (Chapter 3), and achieved strong empirical performance with policy gradients (Chapter 8). Our continuous Q-learning agent reached returns of ~25.0, and REINFORCE with baselines stabilized training with variance reductions exceeding 60%.

All of this rests on a fundamental assumption: **the environment is stationary**. We assumed the reward distribution $R(a|x)$ and state transition dynamics $P(x'|x,a)$ are fixed, unknown functions that remain constant while our agent learns.

This is false.

In production e-commerce systems, nothing is stationary:

1. **User Preference Drift**: "Winter boots" means something different in July vs. December. Click-through rates for swimwear spike in summer and crash in winter. Conversion rates for electronics surge during Black Friday and collapse in February.

2. **Catalog Drift**: Products go out of stock. New brands launch. Suppliers change prices. Your carefully learned boost weights $\theta$ become obsolete when the top-selling product category disappears from inventory.

3. **Systemic Shocks**: A competitor drops prices 30%. Marketing runs a promotion. A supply chain disruption changes delivery times. The pandemic arrives. Your policy, trained to optimize GMV under pre-shock conditions, now chases phantom rewards.

If our RL agent treats these shifts as noise, it will fail slowly as it averages over incompatible regimes. If it learns too slowly, it lags reality and bleeds revenue. If it learns too quickly, it chases transient fluctuations and destabilizes the user experience—users hate when search results thrash unpredictably.

**This chapter builds production-grade robustness.** We develop:

- **Mathematical foundations** (§10.1-10.2): Lyapunov stability theory, formal drift detection as hypothesis testing
- **Drift detection algorithms** (§10.3): CUSUM and Page-Hinkley tests with convergence proofs
- **Production guardrails** (§10.4): Delta-Rank@k stability metrics, CM2 floor constraints, automatic fallback mechanisms
- **Implementation** (§10.5): The `zoosim.monitoring` package with `SafetyMonitor` orchestrating drift detection and policy fallback
- **Experiments** (§10.6): Simulated seasonal drift, recovery dynamics, performance analysis
- **Theory-practice gaps** (§10.7): What works, what doesn't, and why the theory is incomplete

By the end of this chapter, you will understand how to deploy RL policies that **fail gracefully** when the world changes, detect distributional shift rigorously, and enforce safety constraints that prevent catastrophic business outcomes.

**Roadmap:**
- **§10.1**: The non-stationarity problem (formalization, examples)
- **§10.2**: Mathematical foundations (Lyapunov stability, change-point detection)
- **§10.3**: Drift detection algorithms (CUSUM, Page-Hinkley, convergence proofs)
- **§10.4**: Production guardrails (Δrank@k, CM2 floors, fallback policies)
- **§10.5**: Implementation (`zoosim.monitoring` architecture)
- **§10.6**: Experiments (seasonal drift scenario, recovery analysis)
- **§10.7**: Theory-practice gaps and open problems
- **§10.8**: Production checklist
- **§10.9**: Exercises & Labs

---

## 10.1 The Non-Stationarity Problem

We begin by establishing the metric structure on probability measures (§10.1.1), which provides the mathematical foundation for quantifying drift. We then formalize non-stationary MDPs (§10.1.2) and analyze why standard RL algorithms fail under distributional shift (§10.1.3).

### 10.1.1 Preliminaries: Metrics on Probability Distributions

Before formalizing non-stationary MDPs, we need precise metrics to quantify how "far apart" two probability distributions are. These metrics will appear in our drift definitions and convergence bounds throughout the chapter.

**Definition 10.1.1** (Total Variation Distance) {#DEF-10.1.1}

Let $\mu, \nu$ be probability measures on a measurable space $(\Omega, \mathcal{F})$. The **total variation distance** is
$$
\|\mu - \nu\|_{\text{TV}} = \sup_{A \in \mathcal{F}} |\mu(A) - \nu(A)|.
$$

Equivalently, if $\mu, \nu$ have densities $p, q$ with respect to a dominating measure $\lambda$,
$$
\|\mu - \nu\|_{\text{TV}} = \frac{1}{2} \int |p(x) - q(x)| \, d\lambda(x).
$$
For discrete distributions on $\{1, \ldots, n\}$,
$$
\|\mu - \nu\|_{\text{TV}} = \frac{1}{2} \sum_{i=1}^n |\mu_i - \nu_i|.
$$

**Computational interpretation.** For finite state spaces $|\mathcal{S}| = n$, total variation distance can be computed in $O(n)$ time as the formula above. This is precisely the computation used in `zoosim/monitoring/drift.py` when comparing empirical reward distributions across time windows. The factor of $\frac{1}{2}$ ensures TV distance lies in $[0, 1]$: identical distributions have $\|\mu - \nu\|_{\text{TV}} = 0$, while disjoint-support distributions achieve $\|\mu - \nu\|_{\text{TV}} = 1$.

**Definition 10.1.2** (Wasserstein-1 Distance) {#DEF-10.1.2}

Let $(\Omega, d)$ be a metric space and $\mu, \nu$ probability measures on $\Omega$. The **Wasserstein-1 distance** (also called Kantorovich or Earth Mover's Distance) is
$$
W_1(\mu, \nu) = \inf_{\pi \in \Pi(\mu, \nu)} \mathbb{E}_{(X,Y) \sim \pi}[d(X, Y)],
$$
where $\Pi(\mu, \nu)$ is the set of all couplings (joint distributions with marginals $\mu$ and $\nu$).

**Remark 10.1.1** {#REM-10.1.1}
For finite state spaces, $W_1$ reduces to optimal transport with cost matrix $d(s, s')$ and can be computed via linear programming. For continuous spaces, $W_1$ requires $\mathcal{S}$ to have a metric structure, unlike total variation which only needs a $\sigma$-algebra. In this chapter, we primarily use total variation distance [DEF-10.1.1] because it does not require a metric on the state space $\mathcal{S}$—only a measurable structure. Wasserstein distances are useful when $\mathcal{S}$ has natural geometry (e.g., product embeddings in $\mathbb{R}^d$); see [@gibbs_su:convergence_variational:2002] for functional analysis foundations and [@villani:optimal_transport:2008] for comprehensive treatment of optimal transport theory.

With these metrics established, we now formalize non-stationary MDPs and quantify drift rates.

### 10.1.2 Formalization

Recall the stationary MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ from Chapter 3. The Bellman optimality operator $\mathcal{T}$ from [THM-3.5.2] acts on value functions $V : \mathcal{S} \to \mathbb{R}$ as

$$
(\mathcal{T} V)(s) = \max_a \left\{ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right\}
\tag{10.1}
$$
{#EQ-10.1}

The optimal value function $V^*$ is characterized as the fixed point $V^* = \mathcal{T} V^*$. This operator is **time-invariant**: $\mathcal{T}$ does not depend on the episode index $t$ or wall-clock time. Once we solve for $V^*$, it remains optimal indefinitely.

In production, we face a **non-stationary MDP** where the transition kernel and reward function evolve:

$$
(\mathcal{S}, \mathcal{A}, P_t, R_t, \gamma), \quad t \in \mathbb{N}
\tag{10.2}
$$
{#EQ-10.2}

Here $P_t(s'|s,a)$ and $R_t(s,a)$ may change with the episode index $t$ (or equivalently, wall-clock time). The optimal policy $\pi^*_t$ is now time-dependent and potentially different at each episode.

**Definition 10.1.3** (Bounded Drift Rate) {#DEF-10.1.3}

Let $(\mathcal{S}, \mathcal{A}, P_t, R_t, \gamma)_{t \in \mathbb{N}}$ be a sequence of MDPs with common state and action spaces. We say this sequence has **bounded drift rate** $(\delta_P, \delta_R)$ if:

1. **Transition kernel drift (total variation).** For each $t$, the transition kernel
   $P_t : \mathcal{S} \times \mathcal{A} \to \mathcal{P}(\mathcal{S})$
   is a measurable function from state–action pairs to probability measures on $\mathcal{S}$. We equip $\mathcal{P}(\mathcal{S})$ with the topology of total variation (which coincides with the topology of setwise convergence for probability measures) and its Borel $\sigma$-algebra. For each $(s,a) \in \mathcal{S} \times \mathcal{A}$,
   $$
   \big\|P_t(\cdot \mid s,a) - P_{t+1}(\cdot \mid s,a)\big\|_{\text{TV}} \leq \delta_P,
   \quad \forall t \in \mathbb{N}.
   $$

2. **Reward function drift (supremum norm).** The reward functions
   $R_t : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$
   are measurable, uniformly bounded by some $R_{\max} < \infty$, and satisfy
   $$
   d_R(R_t, R_{t+1}) := \sup_{s,a} |R_t(s,a) - R_{t+1}(s,a)| \leq \delta_R,
   \quad \forall t \in \mathbb{N}.
   $$

We denote the drift rate **pair** by
$$
(\delta_P, \delta_R),
\tag{10.3}
$$
{#EQ-10.3}

and say drift is **bounded** if both $\delta_P$ and $\delta_R$ are finite. Subsequent theorems will state explicit dependence on $\delta_P$ or $\delta_R$ as needed—we do not collapse them into a single scalar because they measure different aspects of non-stationarity.

**Assumption 10.1.1** (Regularity for Bounded Drift) {#ASM-10.1.1}

For the suprema in [DEF-10.1.3] to be well-defined, we assume one of the following regularity conditions holds:

1. **Compact state-action space:** $\mathcal{S} \times \mathcal{A}$ is compact, or

2. **Continuity and relative compactness:** The kernels $P_t(\cdot | s, a)$ and reward functions $R_t(s, a)$ are continuous in $(s, a)$ for each $t$, and there exists a compact subset $K \subseteq \mathcal{S} \times \mathcal{A}$ such that the suprema defining $\delta_P$ and $\delta_R$ are achieved on $K$, or

3. **Essential supremum formulation:** For a reference measure $\mu$ on $\mathcal{S} \times \mathcal{A}$ (e.g., a product of stationary distributions under a baseline policy), the bounds hold $\mu$-almost everywhere:
   $$
   \|P_t(\cdot | s, a) - P_{t+1}(\cdot | s, a)\|_{\text{TV}} \leq \delta_P \quad \text{for } \mu\text{-a.e. } (s,a).
   $$

In practice, e-commerce MDPs satisfy condition (1): the state space (user-query contexts) and action space (discrete templates or bounded continuous boosts) are compact subsets of finite-dimensional spaces. This mirrors the standard treatment in [@bertsekas_shreve:stochastic_optimal:1996, Chapter 8].

**Remark 10.1.2** (When Do Drift Metrics Exist?) {#REM-10.1.2}

The supremum norm $d_R$ is well-defined when rewards are uniformly bounded:
$\sup_{s,a,t} |R_t(s,a)| < \infty$.
For the transition kernel, we fix $d_P(\mu,\nu) = \|\mu - \nu\|_{\text{TV}}$ in this chapter, which is always defined on $(\mathcal{S}, \mathcal{F})$ without requiring a metric on $\mathcal{S}$. Other metrics (e.g., Wasserstein-1 from [DEF-10.1.2] when $\mathcal{S}$ is a metric space) are useful in related work; see [@gibbs_su:convergence_variational:2002] for functional analysis foundations. Quantitative bounds (such as [EQ-10.5]) depend on this choice through the scale of $\delta_P$.

**Practical interpretation.** Bounded drift rate means the world changes smoothly, not catastrophically. If $\delta_R = 0.01$, then rewards shift by at most 1% per episode—this models seasonal trends. If $\delta_R = 5.0$, we have abrupt shocks—this models a competitor price war or a marketing campaign launch.

!!! note "Code ↔ Env (Non-Stationarity)"
    The non-stationary environment is implemented in `scripts/ch10/ch10_drift_demo.py:64-129` as `DriftEnvironment`:
    - **Phase 1 (episodes 0-1499)**: Template T1 (High Margin) has CVR 12%, best action
    - **Phase 2 (episodes 1500+)**: Template T1 crashes to CVR 2%, Template T3 (Popular) rises to CVR 15%
    - **Drift magnitude**: $\delta_R \approx 10\%$ (in CVR units), implemented at lines 89-92
    - This models a sudden preference shift (e.g., seasonal change or competitor action)

### 10.1.3 Why Standard RL Fails Under Drift

Consider LinUCB from Chapter 6. Recall the ridge regression update [EQ-6.8]:

$$
\hat{\theta}_t = \left( \sum_{i=1}^{t-1} \phi(x_i, a_i) \phi(x_i, a_i)^\top + \lambda I \right)^{-1} \sum_{i=1}^{t-1} \phi(x_i, a_i) r_i
\tag{10.4}
$$
{#EQ-10.4}

LinUCB **averages all past data** with equal weight (the sums run from $i=1$ to $t-1$). If the reward function shifts at episode $t_0$, observations before $t_0$ are drawn from a different distribution than observations after $t_0$. Averaging them produces a biased estimate that converges to neither the pre-drift nor post-drift optimal policy.

**Proposition 10.1.1** (Bias of Time-Averaged Estimator) {#PROP-10.1.1}

Suppose the true reward parameter shifts at $t_0$:
$$
\theta^*_t = \begin{cases}
\theta_{\text{before}}, & t < t_0, \\
\theta_{\text{after}}, & t \geq t_0.
\end{cases}
$$
Under the following assumptions:

**(A1) i.i.d. features under exploration:** The feature vectors $\{\phi_i\}_{i=1}^T$ are i.i.d. samples from a distribution with $\mathbb{E}[\phi_i \phi_i^\top] = \Sigma$ for some positive definite $\Sigma \succ 0$.

**(A2) Sub-Gaussian noise:** The noise $\epsilon_i = r_i - \langle \theta^*_i, \phi_i \rangle$ is conditionally $\sigma$-sub-Gaussian given $\phi_i$.

**(A3) Bounded features:** $\|\phi_i\| \leq L$ almost surely.

Then the ridge regression estimator [EQ-10.4] converges to the **time-averaged** parameter:
$$
\hat{\theta}_T \xrightarrow{a.s.} \frac{t_0}{T} \theta_{\text{before}} + \frac{T - t_0}{T} \theta_{\text{after}} \quad \text{as } T \to \infty.
$$

*Proof.* The normal equations for [EQ-10.4] give
$$
\hat{\theta}_T = A_T^{-1} b_T, \quad A_T = \sum_{i=1}^{T-1} \phi_i \phi_i^\top + \lambda I, \quad b_T = \sum_{i=1}^{T-1} \phi_i r_i.
$$

**Step 1: Decompose the target vector.**
Split the sum at $t_0$:
$$
b_T = \underbrace{\sum_{i=1}^{t_0-1} \phi_i \langle \theta_{\text{before}}, \phi_i \rangle}_{b_T^{(1)}} + \underbrace{\sum_{i=t_0}^{T-1} \phi_i \langle \theta_{\text{after}}, \phi_i \rangle}_{b_T^{(2)}} + \underbrace{\sum_{i=1}^{T-1} \phi_i \epsilon_i}_{\text{noise } N_T}.
$$

**Step 2: Apply the law of large numbers.**
By assumption (A1) and the strong law of large numbers:
$$
\frac{A_T}{T} = \frac{1}{T}\sum_{i=1}^{T-1} \phi_i \phi_i^\top + \frac{\lambda I}{T} \xrightarrow{a.s.} \Sigma \quad \text{as } T \to \infty.
$$
Similarly,
$$
\frac{b_T^{(1)}}{T} = \frac{t_0 - 1}{T} \cdot \frac{1}{t_0 - 1}\sum_{i=1}^{t_0-1} \phi_i \phi_i^\top \theta_{\text{before}} \xrightarrow{a.s.} \frac{t_0}{T} \Sigma \theta_{\text{before}},
$$
and analogously for $b_T^{(2)}$.

**Step 3: Bound the noise term.**
By assumptions (A2)–(A3) and standard martingale concentration ([@hsu_kakade_zhang:random_design:2012, Theorem 2]),
$$
\|N_T\| = O(\sigma \sqrt{T \log T}) \quad \text{with high probability},
$$
so $\|N_T\| / T \to 0$ almost surely.

**Step 4: Conclude.**
Combining Steps 1–3:
$$
\hat{\theta}_T = A_T^{-1} b_T \xrightarrow{a.s.} \Sigma^{-1}\left(\frac{t_0}{T} \Sigma \theta_{\text{before}} + \frac{T - t_0}{T} \Sigma \theta_{\text{after}}\right) = \frac{t_0}{T} \theta_{\text{before}} + \frac{T - t_0}{T} \theta_{\text{after}}.
$$
$\square$

**Remark 10.1.4** (Assumption (A1) in Practice) {#REM-10.1.4}

Assumption (A1) requires i.i.d. features, which holds when the policy explores uniformly at random or when user queries arrive i.i.d. from a stationary distribution. Under adaptive policies like LinUCB, features are no longer i.i.d.—the policy selects actions based on past observations, inducing dependence. However, the conclusion still holds qualitatively: the estimator converges to a weighted average where the weights reflect the **empirical distribution** of features under the policy, not the population distribution. See [@abbasi-yadkori:improved_algorithms:2011, §3] for analysis under adaptive policies.

**Practical implication.** If $\theta_{\text{before}}$ and $\theta_{\text{after}}$ are far apart, the averaged estimator is **suboptimal in both regimes**. This is why forgetting old data (via sliding windows, exponential decay, or explicit change-point detection and reset) is essential under drift.

**Definition 10.1.4** (Dynamic Regret) {#DEF-10.1.4}

For a non-stationary MDP $\{(\mathcal{S}, \mathcal{A}, P_t, R_t, \gamma)\}_{t=1}^T$ with time-varying optimal policies $\{\pi^*_t\}_{t=1}^T$, the **dynamic regret** of policy $\pi$ over horizon $T$ is:

$$
\text{DynRegret}_T(\pi) = \sum_{t=1}^T \mathbb{E}_{\pi^*_t}[R_t(s_t, a_t)] - \sum_{t=1}^T \mathbb{E}_{\pi}[R_t(s_t, a_t)]
\tag{10.4'}
$$
{#EQ-10.4-prime}

where all trajectories start from a fixed initial distribution $\rho_0$ on $\mathcal{S}$.

**Notation clarification.** The expectation $\mathbb{E}_{\pi}[R_t(s_t, a_t)]$ denotes the expected immediate reward at episode $t$ when following policy $\pi$ from the initial distribution $\rho_0$. Formally, this is the marginal expectation over the $t$-th state-action pair $(s_t, a_t)$ when the Markov chain induced by $(P_t, \pi)$ is run for $t$ steps starting from $s_0 \sim \rho_0$:
$$
\mathbb{E}_{\pi}[R_t(s_t, a_t)] = \int_{\mathcal{S}} \int_{\mathcal{A}} R_t(s, a) \, \pi(da | s) \, \rho_t^\pi(ds),
$$
where $\rho_t^\pi$ is the marginal distribution of $s_t$ under policy $\pi$ and transition kernels $(P_1, \ldots, P_{t-1})$. For the optimal policy $\pi^*_t$, the expectation $\mathbb{E}_{\pi^*_t}[\cdot]$ uses the time-varying optimal policy at episode $t$, which is the policy maximizing $\mathbb{E}[R_t(s_t, a_t)]$ under the regime $(P_t, R_t)$.

**Example 10.1.1** (Dynamic Regret in a 2-Armed Bandit)

Consider a 2-armed bandit where:
- **Phase 1 ($t \leq 500$)**: $\mu_1 = 0.6$, $\mu_2 = 0.4$ (arm 1 is optimal, $\pi^*_1(a) = \mathbb{1}[a = 1]$).
- **Phase 2 ($t > 500$)**: $\mu_1 = 0.4$, $\mu_2 = 0.6$ (arm 2 is optimal, $\pi^*_2(a) = \mathbb{1}[a = 2]$).

The hindsight-optimal policy achieves
$$
\sum_{t=1}^{500} 0.6 + \sum_{t=501}^{1000} 0.6 = 600
$$
expected reward over $T = 1000$ rounds.

Suppose a LinUCB-style algorithm converges to arm 1 in Phase 1. After the shift at $t=500$, it takes roughly 50 episodes before confidence intervals shrink enough to reliably identify arm 2 as better. During this lag, it pulls arm 1 (reward $0.4$) instead of arm 2 (reward $0.6$), losing $0.2$ reward per round. The dynamic regret accumulated in this window is about
$$
\text{DynRegret}_{\text{lag}} \approx 50 \times 0.2 = 10.
$$
In a truly stationary bandit, LinUCB achieves $O(\log T)$ regret [@abbasi-yadkori:improved_algorithms:2011]. The key point is that even a single distributional shift induces **extra regret proportional to the detection delay**, which scales with the drift budget in non-stationary settings.

Before stating the main theorem on dynamic regret, we formalize how reward drift translates to parameter drift in linear models.

**Lemma 10.1.1** (Reward Drift Implies Parameter Drift) {#LEM-10.1.1}

Assume the reward model is linear: $R_t(s,a) = \langle \theta^*_t, \phi(s,a) \rangle$. Suppose:

1. Features are bounded: $\|\phi(s,a)\| \leq L$ for all $(s,a)$.
2. Feature covariance has minimum eigenvalue: $\Sigma := \mathbb{E}_{(s,a) \sim \rho}[\phi(s,a) \phi(s,a)^\top] \succeq \lambda_{\min} I$ for some $\lambda_{\min} > 0$.
3. Rewards drift by at most $\delta_R$: $\sup_{s,a} |R_t(s,a) - R_{t+1}(s,a)| \leq \delta_R$.

Then the parameter drift satisfies
$$
\|\theta^*_{t+1} - \theta^*_t\| \leq \frac{\delta_R}{\sqrt{\lambda_{\min}}}.
$$

*Proof.* For any $(s,a)$,
$$
|\langle \theta^*_t - \theta^*_{t+1}, \phi(s,a) \rangle| = |R_t(s,a) - R_{t+1}(s,a)| \leq \delta_R.
$$
Squaring and taking expectation over $(s,a) \sim \rho$:
$$
(\theta^*_t - \theta^*_{t+1})^\top \Sigma (\theta^*_t - \theta^*_{t+1}) \leq \delta_R^2.
$$
By the lower bound $\Sigma \succeq \lambda_{\min} I$,
$$
\lambda_{\min} \|\theta^*_t - \theta^*_{t+1}\|^2 \leq \delta_R^2 \implies \|\theta^*_t - \theta^*_{t+1}\| \leq \frac{\delta_R}{\sqrt{\lambda_{\min}}}.
$$
$\square$

**Remark 10.1.3** (Feature Distribution Coverage) {#REM-10.1.3}

The bound in [LEM-10.1.1] assumes the feature distribution $\rho$ places mass on state-action pairs achieving the supremum drift. Formally, we require that $\sup_{s,a} |R_t(s,a) - R_{t+1}(s,a)| = \sup_{s,a}|\langle \theta^*_t - \theta^*_{t+1}, \phi(s,a) \rangle|$ is achieved (or arbitrarily approximated) under $\rho$. If the drift occurs only in a measure-zero region under $\rho$, the effective parameter drift may be smaller than the bound suggests. Equivalently, we can strengthen the lemma to bound the $\rho$-weighted drift: defining $\|\Delta\theta\|_\Sigma^2 := \Delta\theta^\top \Sigma \Delta\theta$, the proof directly shows $\|\theta^*_{t+1} - \theta^*_t\|_\Sigma \leq \delta_R$. The Euclidean norm bound then follows from $\Sigma \succeq \lambda_{\min} I$.

**Why this matters:** The lemma shows that feature covariance $\lambda_{\min}$ determines how drift robustness scales. Poor feature coverage (small $\lambda_{\min}$) amplifies reward drift into large parameter drift. This connects back to Chapter 6's feature engineering: rich features that span the action space yield larger $\lambda_{\min}$ and better drift robustness.

**Theorem 10.1.1** (Suboptimality Under Bounded Drift, [@besbes_szepesvari:non_stationary_bandits:2019, Theorem 3.1]) {#THM-10.1}

Let $\pi_{\text{LinUCB}}$ denote the LinUCB policy trained on a non-stationary linear bandit model with bounded drift rate $(\delta_P, \delta_R)$ over $T$ episodes. Assume:

1. Bounded features $\|\phi(s,a)\| \leq L$ for all $(s,a)$.
2. Bounded rewards $|R_t(s,a)| \leq R_{\max}$ for all $t,(s,a)$.
3. Drift rate satisfies [DEF-10.1.3] with reward drift bound $\delta_R$.
4. By [LEM-10.1.1], the optimal parameter $\theta_t^*$ of the linear reward model evolves with bounded per-step drift:
   $$
   \|\theta^*_{t+1} - \theta^*_t\| \leq \delta_\theta := \frac{\delta_R}{\sqrt{\lambda_{\min}}}
   $$
   where $\lambda_{\min}$ is the minimum eigenvalue of the feature covariance. The **variation budget**
   $$
   B_T = \sum_{t=1}^{T-1} \|\theta^*_{t+1} - \theta^*_t\|
   $$
   satisfies $B_T \leq \delta_\theta T$.

Then the dynamic regret [DEF-10.1.4] satisfies
$$
\text{DynRegret}_T(\pi_{\text{LinUCB}}) \geq \Omega\bigl(\sqrt{B_T\, T}\bigr).
\tag{10.5}
$$
{#EQ-10.5}

**Proof (Sketch).**
By the bounded parameter drift assumption, the variation budget satisfies
$$
B_T
= \sum_{t=1}^{T-1} \|\theta^*_{t+1} - \theta^*_t\|
\leq \sum_{t=1}^{T-1} \delta_\theta
\leq \delta_\theta T.
$$
In the variation-budget framework of [@besbes_szepesvari:non_stationary_bandits:2019], any algorithm whose confidence regions shrink over time (in particular, LinUCB with shrinking elliptical confidence sets [@abbasi-yadkori:improved_algorithms:2011]) incurs dynamic regret at least
$$
\text{DynRegret}_T(\pi) \geq \Omega\bigl(\sqrt{B_T\, T}\bigr).
$$
Using $B_T \leq \delta_\theta T$ shows that larger per-step reward drift $\delta_R$ (which induces parameter drift $\delta_\theta$) increases the regret lower bound. The key message is qualitative: **for any fixed drift budget $B_T > 0$, dynamic regret must grow at least on the order of $\sqrt{B_T T}$**, so even well-designed algorithms like LinUCB cannot achieve logarithmic regret in non-stationary environments. Note that transition kernel drift $\delta_P$ does not directly appear in this bound—the regret is driven by reward drift $\delta_R$ in bandit settings. $\square$

**Practical implication:** As $T \to \infty$, LinUCB's regret grows without bound under drift. We need **adaptive mechanisms** that discard stale data.

!!! warning "Theory-Practice Gap: Sudden vs. Gradual Drift"
    [THM-10.1] assumes bounded drift rate (smooth changes). In practice, e-commerce systems experience **abrupt regime shifts**: Black Friday, pandemic lockdowns, competitor exits. Theory for sudden shocks is weaker—existing bounds [@garivier_moulines:change_point:2011] require known change-point counts $K$ and scale as $O(K \sqrt{T/K})$. If $K$ is unknown (the production setting), we need detection algorithms. This is the focus of §10.3.

---

## 10.2 Mathematical Foundations: Stability Theory

We borrow concepts from control theory to formalize robustness. The key tool is **Lyapunov stability**, which characterizes how systems respond to perturbations.

### 10.2.1 Lyapunov Stability for MDPs

In control theory, a **Lyapunov function** $V : \mathcal{S} \to \mathbb{R}_+$ is an "energy-like" quantity that measures how far the system state is from equilibrium. If $V$ decreases along trajectories, the system is stable.

**Definition 10.2.1** (Lyapunov Stability for Policy Optimization) {#DEF-10.2}

Let $\pi_\theta$ be a parameterized policy with performance $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$ (as in Chapter 8). Define the **sub-optimality gap**:

$$
V(\theta) = J(\theta^*) - J(\theta)
\tag{10.6}
$$
{#EQ-10.6}

where $\theta^*$ achieves the optimal policy. We say $\pi_\theta$ is **Lyapunov stable** at equilibrium $\theta^*$ if:

$$
\forall \epsilon > 0, \; \exists \rho(\epsilon) > 0 \text{ such that } \|\theta_0 - \theta^*\| < \rho \implies V(\theta_t) < \epsilon \text{ for all } t \geq 0
$$

and **asymptotically stable** if additionally:

$$
\lim_{k \to \infty} V(\theta_k) = 0
$$

for any sequence $\{\theta_k\}$ generated by gradient ascent $\theta_{k+1} = \theta_k + \alpha \nabla J(\theta_k)$.

**Connection to RL.** If we can show that $J(\theta)$ is smooth and strongly concave near $\theta^*$, then gradient ascent converges with exponential rate (see [@fazel_ge:global_convergence:2018] for tabular softmax policies). But under **drift**, the target $\theta^*$ moves over time: $\theta^*_t$ depends on $(P_t, R_t)$. If the policy adaptation rate (learning rate $\alpha$) is too slow, $\theta_t$ lags $\theta^*_t$ and $V(\theta_t)$ does not decay.

To make this analysis precise, we first state the regularity assumptions required for gradient ascent convergence.

**Assumption 10.2.1** (Local Strong Concavity) {#ASM-10.2.1}

For each $t \in \mathbb{N}$, there exists a ball
$$
B_r(\theta^*_t) = \{\theta : \|\theta - \theta^*_t\| \leq r\}
$$
with radius $r > 0$ (independent of $t$) such that:

1. $J_t(\cdot)$ is $L$-smooth on $B_r(\theta^*_t)$.
2. $J_t(\cdot)$ is $\mu$-strongly concave on $B_r(\theta^*_t)$.
3. The gradient ascent trajectory satisfies $\theta_t \in B_r(\theta^*_t)$ for all $t$.

**When does condition (3) hold?** Condition (3) is satisfied when the initial error $\|\theta_0 - \theta^*_0\| \leq r/2$ and the step size $\alpha$ satisfies $\alpha < \mu r / (4 L^2)$. Under these conditions, gradient ascent steps remain within the ball: the contraction from strong concavity dominates the drift perturbation, keeping the trajectory in $B_r(\theta^*_t)$. For a formal derivation, see [@nesterov:convex_optimization:2018, §2.1.5] on contractive gradient methods.

Under this assumption, the notion of "neighborhood" is made precise, and we can rigorously analyze the tracking error.

**Theorem 10.2.1** (Tracking Error Under Drift) {#THM-10.2}

Under [ASM-10.2.1] (Local Strong Concavity), assume the optimal parameter $\theta^*_t$ evolves with bounded drift $\|\theta^*_{t+1} - \theta^*_t\| \leq \delta$, and suppose gradient ascent uses learning rate $\alpha$. Then the tracking error satisfies:

$$
\mathbb{E}[\|\theta_t - \theta^*_t\|] \leq \frac{2\delta}{\mu \alpha} + O(\alpha)
\tag{10.7}
$$
{#EQ-10.7}

**Proof.**

Model $\theta^*_t$ as a random walk:
$$
\theta^*_{t+1} = \theta^*_t + \xi_t, \qquad \|\xi_t\| \leq \delta.
$$
The gradient ascent update is
$$
\theta_{t+1} = \theta_t + \alpha \nabla J_t(\theta_t),
$$
where $J_t(\cdot)$ is the performance under $(P_t, R_t)$.

By $L$-smoothness of $J_t$, the gradient satisfies a mean-value expansion
$$
\nabla J_t(\theta_t) = \nabla J_t(\theta^*_t) + \nabla^2 J_t(\xi_t')(\theta_t - \theta^*_t)
$$
for some $\xi_t'$ on the line segment between $\theta_t$ and $\theta^*_t$.
Since $\nabla J_t(\theta^*_t) = 0$ (first-order optimality) and $J_t$ is
$\mu$-strongly concave in a neighborhood of $\theta^*_t$, we apply the strong concavity definition directly:
$$
J_t(\theta^*_t) \leq J_t(\theta_t) + \langle \nabla J_t(\theta_t), \theta^*_t - \theta_t \rangle - \frac{\mu}{2}\|\theta^*_t - \theta_t\|^2.
$$
Since $\theta^*_t$ is the maximizer, $J_t(\theta^*_t) \geq J_t(\theta_t)$, so rearranging yields the coercivity bound
$$
\langle \nabla J_t(\theta_t), \theta^*_t - \theta_t \rangle
\geq \frac{\mu}{2} \|\theta^*_t - \theta_t\|^2.
$$

Consider the tracking error $e_t = \theta_t - \theta^*_t$. Using the updates above,
$$
e_{t+1}
= \theta_{t+1} - \theta^*_{t+1}
= \theta_t + \alpha \nabla J_t(\theta_t) - \theta^*_t - \xi_t
= e_t + \alpha \nabla J_t(\theta_t) - \xi_t.
$$

**Step 1: Norm expansion (identity, not inequality).**
Expanding the squared norm using the identity $\|a + b + c\|^2 = \|a\|^2 + \|b\|^2 + \|c\|^2 + 2\langle a, b\rangle + 2\langle a, c\rangle + 2\langle b, c\rangle$:
$$
\|e_{t+1}\|^2
= \|e_t + \alpha \nabla J_t(\theta_t) - \xi_t\|^2
= \|e_t\|^2 + \alpha^2 \|\nabla J_t(\theta_t)\|^2 + \|\xi_t\|^2
  + 2\alpha \langle \nabla J_t(\theta_t), e_t\rangle
  - 2\langle e_t, \xi_t\rangle
  - 2\alpha\langle \nabla J_t(\theta_t), \xi_t\rangle.
$$

**Step 2: Apply coercivity from strong concavity.**
Recall that $e_t = \theta_t - \theta^*_t = -(\theta^*_t - \theta_t)$. From the coercivity bound derived earlier,
$$
\langle \nabla J_t(\theta_t), \theta^*_t - \theta_t \rangle \geq \frac{\mu}{2}\|\theta^*_t - \theta_t\|^2 = \frac{\mu}{2}\|e_t\|^2.
$$
Multiplying both sides by $-1$ (which reverses the inequality):
$$
\langle \nabla J_t(\theta_t), e_t \rangle = \langle \nabla J_t(\theta_t), \theta_t - \theta^*_t \rangle = -\langle \nabla J_t(\theta_t), \theta^*_t - \theta_t \rangle \leq -\frac{\mu}{2}\|e_t\|^2.
$$
Substituting into Step 1:
$$
\|e_{t+1}\|^2
\leq \|e_t\|^2 - \alpha \mu \|e_t\|^2 + \alpha^2 \|\nabla J_t(\theta_t)\|^2 + \|\xi_t\|^2
  - 2\langle e_t, \xi_t\rangle - 2\alpha\langle \nabla J_t(\theta_t), \xi_t\rangle.
$$

**Step 3: Apply Young's inequality to cross-terms.**
For the cross-term $-2\langle e_t, \xi_t\rangle$, Young's inequality gives $|2\langle e_t, \xi_t\rangle| \leq \epsilon\|e_t\|^2 + \frac{1}{\epsilon}\|\xi_t\|^2$ for any $\epsilon > 0$. Choosing $\epsilon = \alpha\mu/4$:
$$
|2\langle e_t, \xi_t\rangle| \leq \frac{\alpha\mu}{4}\|e_t\|^2 + \frac{4}{\alpha\mu}\|\xi_t\|^2.
$$
Similarly, by $L$-smoothness of $J_t$, $\|\nabla J_t(\theta_t)\| \leq L\|e_t\|$ (since $\nabla J_t(\theta^*_t) = 0$), so the term $2\alpha\langle \nabla J_t(\theta_t), \xi_t\rangle$ is bounded by $2\alpha L \|e_t\| \|\xi_t\| \leq \frac{\alpha\mu}{4}\|e_t\|^2 + \frac{4\alpha L^2}{\mu}\|\xi_t\|^2$.

**Step 4: Combine and simplify.**
Collecting terms and using $\|\xi_t\| \leq \delta$:
$$
\|e_{t+1}\|^2
\leq \left(1 - \alpha\mu + \frac{\alpha\mu}{4} + \frac{\alpha\mu}{4}\right)\|e_t\|^2 + \alpha^2 L^2 \|e_t\|^2 + C_\delta \delta^2
= \left(1 - \frac{\alpha\mu}{2} + \alpha^2 L^2\right)\|e_t\|^2 + C_\delta \delta^2,
$$
where $C_\delta = 1 + \frac{4}{\alpha\mu} + \frac{4\alpha L^2}{\mu}$ absorbs constants. For $\alpha < \mu/(4L^2)$, we have $\alpha^2 L^2 < \alpha\mu/4$, so
$$
\|e_{t+1}\|^2 \leq \left(1 - \frac{\alpha\mu}{4}\right)\|e_t\|^2 + C_\delta \delta^2.
$$

**Step 5: Derive linear contraction.**
Taking square roots (using $\sqrt{1-x} \leq 1 - x/2$ for small $x$) and absorbing constants:
$$
\|e_{t+1}\| \leq \bigl(1 - \tfrac{\alpha \mu}{8}\bigr)\|e_t\| + C'\delta + O(\alpha^2),
$$
where $C' > 0$ is a constant depending on $L, \mu$. For simplicity of exposition, we write this as
$$
\|e_{t+1}\| \leq \bigl(1 - \tfrac{\alpha \mu}{2}\bigr)\|e_t\| + \|\xi_t\| + O(\alpha^2),
$$
absorbing the constant adjustments into the contraction factor (which remains in $(0,1)$ for small $\alpha$). This derivation follows the standard analysis of gradient descent with perturbations; see [@nesterov:convex_optimization:2018, Theorem 2.1.15] for the convex case.
Using $\|\xi_t\| \leq \delta$,
$$
\|e_{t+1}\| \leq \bigl(1 - \tfrac{\alpha \mu}{2}\bigr)\|e_t\| + \delta + O(\alpha^2).
$$
Taking expectations and absorbing the $O(\alpha^2)$ term into the $O(\alpha)$ remainder,
$$
\mathbb{E}\big[\|e_{t+1}\|\big]
\leq \bigl(1 - \tfrac{\alpha \mu}{2}\bigr)\,\mathbb{E}\big[\|e_t\|\big] + \delta + O(\alpha^2).
$$
In steady state, write $e_\infty = \limsup_{t\to\infty}\mathbb{E}[\|e_t\|]$. Solving the affine recursion gives
$$
e_\infty \leq \frac{2\delta}{\mu \alpha} + O(\alpha),
$$
which yields the claimed bound (with adjusted constant). $\square$

**Practical interpretation:** To track a drifting target with rate $\delta$, we must use learning rate $\alpha \geq \delta / (\mu \epsilon)$ where $\epsilon$ is the tolerable tracking error. But large $\alpha$ increases gradient noise variance (as seen in Chapter 8). This is the **stability-plasticity dilemma**: slow learning misses drift, fast learning chases noise.

**Remark 10.2.2** (When Strong Concavity Fails) {#REM-10.2.2}

The strong concavity assumption in [THM-10.2] rarely holds globally for reinforcement learning objectives $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$:

1. For tabular softmax policies, $J(\theta)$ can satisfy a Polyak-Łojasiewicz (PL) inequality under restrictive conditions (contractive dynamics and sufficient exploration); see [@fazel_ge:global_convergence:2018, Theorem 2.1]. This yields linear convergence but not global strong concavity.
2. For neural policies with nonlinear activations, $J(\theta)$ is highly non-convex and strong concavity fails even locally away from well-behaved basins.
3. For linear models such as the ridge regression objective underlying LinUCB and linear value-function approximators, $J(\theta)$ is genuinely strongly convex/concave, and [THM-10.2] applies directly.

In non-convex settings, one can often replace strong concavity with a PL-type condition
$$
\|\nabla J_t(\theta)\|^2 \geq 2\mu \bigl(J_t(\theta^*_t) - J_t(\theta)\bigr),
$$
and derive analogous tracking bounds (the recursion for $e_t$ carries through with $J_t(\theta^*_t) - J_t(\theta_t)$ as Lyapunov function). We do not reproduce that analysis here; instead, we interpret [THM-10.2] as describing the **idealized linear/strongly-concave regime** that our experiments approximate locally.

!!! note "Code ↔ Agent (Learning Rate Adaptation)"
    None of our current policies implement adaptive learning rates that respond to drift. LinUCB uses $\lambda$-weighted ridge regression (`zoosim/policies/lin_ucb.py:63-77`), which implicitly uses a decaying step size. REINFORCE uses constant Adam learning rate (`zoosim/policies/reinforce.py:88`). The solution: **detect drift explicitly** (§10.3) and **reset the policy or increase $\alpha$** when drift is detected. This is implemented by `SafetyMonitor` (§10.5).

### 10.2.2 Change-Point Detection as Hypothesis Testing

Instead of trying to track drifting targets continuously, we take a different approach: **detect when drift occurs** and respond discretely (e.g., reset the policy, retrain on recent data, or fall back to a safe baseline).

Formally, we frame drift detection as **sequential hypothesis testing**:

- **Null hypothesis** $H_0$: The reward distribution is stationary, $R_t(s,a) = R(s,a)$ for all $t$.
- **Alternative hypothesis** $H_1$: There exists a change-point $t_0$ such that $R_t(s,a) \neq R_{t'}(s,a)$ for $t < t_0 < t'$.

At each episode $t$, we observe a statistic $S_t$ (e.g., rolling mean reward, log-likelihood ratio) and test whether $S_t$ exceeds a threshold $\lambda$. If $S_t > \lambda$, we reject $H_0$ and declare drift detected.

**Definition 10.2.2** (Detection Delay and False Alarm Rate) {#DEF-10.3}

For a sequential test with threshold $\lambda$:

1. **Detection delay** $\tau_d$: Let
   $$
   \tau_{\text{det}} = \inf\{t > t_0 : S_t > \lambda\}
   $$
   be the detection time for a (possibly unknown) change-point $t_0 \in \mathbb{N}$. For each $t_0$, define the conditional delay
   $$
   \tau_d(t_0) = \mathbb{E}_{P_1}[\tau_{\text{det}} - t_0 \mid \tau_{\text{det}} \geq t_0],
   $$
   where the conditioning event $\{\tau_{\text{det}} \geq t_0\}$ has positive probability under $P_1$ (drift at $t_0$ followed by eventual detection). For CUSUM with finite threshold, $\mathbb{P}_{P_1}(\tau_{\text{det}} < \infty) = 1$ ensures this is well-defined.

   The **worst-case detection delay** is then
   $$
   \tau_d = \sup_{t_0 \geq 1} \tau_d(t_0).
   $$
2. **False alarm rate** $\alpha$: Probability of declaring drift when $H_0$ is true (no change occurred).

We want $\tau_d$ small (detect quickly) and $\alpha$ small (avoid false alarms that destabilize the system).

**Lemma 10.2.3** (Lorden's Lower Bound, [@lorden:procedures_change_point:1971]) {#LEM-10.3}

Let $\{X_t\}_{t=1}^\infty$ be a sequence of observations where $X_t \sim P_0$ for $t < t_0$ (unknown change-point) and $X_t \sim P_1$ for $t \geq t_0$. Assume $P_1 \ll P_0$ (the post-drift distribution is absolutely continuous with respect to the pre-drift distribution), so that $D_{\text{KL}}(P_1 \| P_0) < \infty$ is well-defined. Let $\tau$ be any stopping time satisfying the false alarm constraint
$$
\mathbb{P}_{P_0}(\tau < \infty) \leq \alpha.
$$
Then the worst-case detection delay satisfies
$$
\sup_{t_0 \geq 1} \mathbb{E}_{P_1}[\tau - t_0 \mid \tau \geq t_0]
\geq \frac{|\log \alpha|}{D_{\text{KL}}(P_1 \| P_0)} + o(|\log \alpha|)
$$
as $\alpha \to 0$, where $D_{\text{KL}}(P_1 \| P_0) = \int \log(dP_1/dP_0) \, dP_1$ is the Kullback-Leibler divergence between the pre-drift and post-drift distributions.

**Remark.** The absolute continuity assumption $P_1 \ll P_0$ is essential: if $P_1$ places mass on events where $P_0$ assigns zero probability, the KL divergence is infinite, and the bound becomes trivial ($\tau_d \geq 0$). In practice, Gaussian or Bernoulli reward distributions satisfy $P_1 \ll P_0$ when both have the same support (e.g., mean shifts without support changes).

*Proof idea.* The proof combines Wald's identity for stopped likelihood-ratio processes with renewal-theoretic arguments. Under $P_0$, the false alarm constraint forces the log-likelihood ratio to stay below a large boundary with high probability, which costs a factor of $|\log \alpha|$. Under $P_1$, the increments have mean $D_{\text{KL}}(P_1 \| P_0)$, so any procedure must, on average, wait at least $|\log \alpha| / D_{\text{KL}}(P_1 \| P_0)$ samples before crossing the boundary. See [@lorden:procedures_change_point:1971, Theorem 1] for the full argument.

**Theorem 10.2.2** (Fundamental Tradeoff, CUSUM Lower Bound) {#THM-10.3}

For any sequential test with false alarm rate $\alpha$, the average detection delay $\tau_d$ satisfies:

$$
\tau_d \geq \frac{|\log \alpha|}{D_{\text{KL}}(P_1 \| P_0)}
\tag{10.8}
$$
{#EQ-10.8}

where $D_{\text{KL}}(P_1 \| P_0)$ is the Kullback-Leibler divergence between the pre-drift distribution $P_0$ and post-drift distribution $P_1$.

**Proof.**

Let $\tau_{\text{det}}$ denote the stopping time of the CUSUM procedure. By definition [DEF-10.3], the worst-case detection delay is
$$
\tau_d = \sup_{t_0 \geq 1} \mathbb{E}_{P_1}[\tau_{\text{det}} - t_0 \mid \tau_{\text{det}} \geq t_0].
$$

Fix an arbitrary change-point $t_0 \geq 1$. By [LEM-10.3] (Lorden's Lower Bound), **any** stopping time $\tau$ satisfying the false alarm constraint $\mathbb{P}_{P_0}(\tau < \infty) \leq \alpha$ must have
$$
\mathbb{E}_{P_1}[\tau - t_0 \mid \tau \geq t_0] \geq \frac{|\log \alpha|}{D_{\text{KL}}(P_1 \| P_0)} + o(|\log \alpha|).
$$

Since $\tau_{\text{det}}$ satisfies the false alarm constraint (by construction of CUSUM), this bound applies to every $t_0$. Taking the supremum over all $t_0 \geq 1$, we obtain
$$
\tau_d = \sup_{t_0 \geq 1} \mathbb{E}_{P_1}[\tau_{\text{det}} - t_0 \mid \tau_{\text{det}} \geq t_0] \geq \frac{|\log \alpha|}{D_{\text{KL}}(P_1 \| P_0)},
$$
which establishes the lower bound [EQ-10.8].

**Achievability (Lorden's upper bound):** Additionally, Lorden [@lorden:procedures_change_point:1971, Theorem 1] shows that the CUSUM stopping time achieves this bound asymptotically (matching upper bound):
$$
\tau_d \leq \frac{|\log \alpha|}{D_{\text{KL}}(P_1 \| P_0)} + O(1) \quad \text{as } \alpha \to 0.
$$
Together, the lower and upper bounds prove CUSUM is **minimax optimal**: no procedure can do better than CUSUM by more than an $o(|\log \alpha|)$ factor in detection delay for a given false alarm rate. $\square$

**Practical implication:** If the drift magnitude is small (e.g., CVR drops from 12% to 11%), detection is inherently slow. But for large shocks (CVR drops from 12% to 2%, as in our experiment), detection can be fast. The algorithms in §10.3 achieve delays near the lower bound [EQ-10.8].

!!! note "Code ↔ Simulator (Drift Magnitude)"
    The `DriftEnvironment` in `scripts/ch10/ch10_drift_demo.py:75-76` uses a **large drift**:
    - Phase 1: Template T1 CVR = 0.12
    - Phase 2: Template T1 CVR = 0.02 (drop of 0.10, or 83% decrease)
    - KL divergence: $D_{\text{KL}}(\text{Bern}(0.12) \| \text{Bern}(0.02)) \approx 0.32$ nats (large, easily detectable). Computation: $0.12 \log(0.12/0.02) + 0.88 \log(0.88/0.98) \approx 0.215 + 0.106 = 0.32$.

    For such large shifts, CUSUM and Page-Hinkley detect drift within 30-50 episodes (§10.6). For smaller drifts (CVR 0.12 → 0.10), detection takes 200+ episodes.

---

## 10.3 Drift Detection Algorithms

We now present two classical algorithms: **CUSUM** (Cumulative Sum) and **Page-Hinkley Test**. Both maintain online statistics and declare drift when a threshold is crossed.

### 10.3.1 CUSUM (Cumulative Sum Control Chart)

CUSUM was developed in 1954 by E.S. Page for quality control in manufacturing [@page:continuous_inspection:1954]. It detects **sustained shifts** in the mean of a process.

**Algorithm 10.3.1** (Two-Sided CUSUM) {#ALG-10.1}

**Input:** Stream of observations $\{x_t\}_{t=1}^\infty$, drift magnitude $\delta$, threshold $\lambda$.

**Initialize:** $\bar{x}_0 = 0$, $S_0^+ = S_0^- = 0$ (cumulative sums).

**For each episode $t = 1, 2, \ldots$:**

1. **Compute deviation from baseline:** $e_t = x_t - \bar{x}_{t-1} - \delta$, where $\bar{x}_{t-1}$ is the running mean of $\{x_1, \ldots, x_{t-1}\}$ (computed **before** observing $x_t$). This is the sequential test structure: the baseline is the mean of all prior observations, not the current one.
2. **Update upper CUSUM:** $S_t^+ = \max(0, S_{t-1}^+ + e_t)$.
3. **Update lower CUSUM:** $S_t^- = \min(0, S_{t-1}^- - e_t)$.
4. **Update mean:** $\bar{x}_t = \bar{x}_{t-1} + \frac{x_t - \bar{x}_{t-1}}{t}$ (Welford-style incremental update).
5. **Test:** If $S_t^+ > \lambda$ or $S_t^- < -\lambda$, declare drift detected. Reset $S_t^+ = S_t^- = 0$.

**End**

**Remark 10.3.0** (Sequential vs. Classical CUSUM) {#REM-10.3.0}

[ALG-10.1] uses **sequential CUSUM** (also called "adaptive baseline CUSUM"), where the baseline $\bar{x}_{t-1}$ is the running mean of all prior observations. This differs from **classical batch CUSUM** [@page:continuous_inspection:1954], which assumes a **known, fixed** target mean $\mu_0$:
$$
e_t^{\text{classical}} = x_t - \mu_0 - \delta.
$$

**When to use each formulation:**

- **Classical CUSUM (known $\mu_0$):** Appropriate when you have a reliable baseline from historical data or a specification (e.g., manufacturing quality control where the target is well-defined). The theoretical guarantees [THM-10.4] apply directly.

- **Sequential CUSUM (adaptive baseline):** Appropriate for production drift detection where no fixed baseline exists—the "normal" reward distribution must be learned online. This is our setting: we don't know the true pre-drift mean $\mu_0$, so we estimate it from data.

**Trade-off:** Sequential CUSUM with adaptive baseline has slightly higher variance (baseline itself fluctuates), which can increase false alarms in early episodes. The `min_instances` parameter (cold-start) mitigates this by waiting until the baseline estimate stabilizes.

**Key parameters:**
- $\delta$: The minimum mean shift we care about detecting (e.g., $\delta = 0.5 \sigma$ means we want to detect shifts of at least half a standard deviation).
- $\lambda$: Detection threshold. Larger $\lambda$ reduces false alarms but increases detection delay. Typical choice: $\lambda = 50 \sigma$.

!!! note "Code ↔ Drift (CUSUM Implementation)"
    Implemented in `zoosim/monitoring/drift.py:69-128` as class `CUSUM`:
    - **Mean and variance tracking**: Uses Welford's online algorithm (lines 100-109) to compute running mean $\bar{x}_t$ and variance $\sigma_t^2$ in $O(1)$ memory.
    - **Two-sided test**: Lines 118-119 compute $S_t^+$ and $S_t^-$ exactly as in [ALG-10.1].
    - **Threshold scaling**: Line 121 sets threshold as `lambda_factor * std` where `lambda_factor` is configurable (default 50.0).
    - **Configuration**: `CUSUMConfig` dataclass (lines 55-67) exposes $\delta$, $\lambda$, and minimum sample count.

### 10.3.1.1 Parameter Selection Guidelines

**Choosing $\delta$ (sensitivity).**
Set $\delta = \eta \hat{\sigma}$ where $\hat{\sigma}$ is the sample standard deviation from a calibration window (for example, the first 100 episodes) and $\eta \in [0.5, 1.0]$ controls sensitivity. Smaller $\eta$ detects weaker drifts but increases false alarms.

**Choosing $\lambda$ (threshold).**
Given a target false-alarm rate $\alpha$ (e.g., one false alarm per $10{,}000$ episodes), set
$$
\lambda \approx \frac{2\hat{\sigma}^2}{\delta} \log\left(\frac{1}{\alpha}\right),
$$
by inverting the ARL formula in [THM-10.4]. Larger $\lambda$ yields longer average run length under $H_0$ but slows detection.

**Remark 10.3.1** (Finite-Sample Corrections) {#REM-10.3.1}

The approximation
$$
\text{ARL}_0 \approx \exp\!\left(\frac{\lambda \Delta}{2\sigma^2}\right)
$$
is derived under the asymptotic regime $\lambda \to \infty$ [@lorden:procedures_change_point:1971]. For moderate thresholds (e.g., $\lambda \lesssim 100 \sigma$), the actual ARL can differ by a factor of $2$–$5$. In production:

1. **Calibrate empirically:** Run CUSUM on historical stationary data, measure the observed false alarm rate, and adjust $\lambda$ to match the target $\alpha$.
2. **Use simulation:** Before deployment, simulate the detector on synthetic stationary data to estimate ARL$_0$ for candidate $(\delta, \lambda)$ pairs.
3. **Start conservatively:** Initialize with a large threshold (e.g., $\lambda = 50\sigma$), then tune downward based on observed noise and false alarm rates.

**Cold-start.**
Accumulate at least $n_{\min} = 30$ observations before testing to avoid spurious detections from small-sample variance. In code this is `min_instances` in the detector configuration.

**Theorem 10.3.1** (CUSUM Average Run Length, [@lorden:procedures_change_point:1971]) {#THM-10.4}

Assume $\{x_t\}_{t=1}^\infty$ are independent observations where

- **Pre-drift ($t < t_0$):** $x_t \sim \mathcal{N}(\mu_0, \sigma^2)$,
- **Post-drift ($t \geq t_0$):** $x_t \sim \mathcal{N}(\mu_0 + \Delta, \sigma^2)$,

for some unknown change-point $t_0 \in \mathbb{N}$ and mean shift $\Delta > 0$. The two-sided CUSUM [ALG-10.1] with parameters $\delta = \Delta/2$ and threshold $\lambda$ satisfies, as $\lambda \to \infty$:

1. **Average Run Length (ARL) under $H_0$** (no change, $t_0 = \infty$):
   $$
   \text{ARL}_0 = \mathbb{E}_{P_0}[\tau_{\text{det}}]
   = \exp\left(\frac{\lambda \Delta}{2\sigma^2}\right)\left(1 + O\!\left(\frac{1}{\lambda}\right)\right).
   $$

2. **Average Detection Delay under $H_1$** (mean shift $\Delta$ at $t_0 = 1$):
   $$
   \tau_d = \mathbb{E}_{P_1}[\tau_{\text{det}} - 1]
   = \frac{2\lambda}{\Delta} + O(1).
   $$

**Proof sketch.**
This is a classical result in sequential analysis [@lorden:procedures_change_point:1971]. The key tool is Wald's identity applied to the stopped log-likelihood ratio. Under $H_0$, the CUSUM statistic is a submartingale with negative drift, so large excursions (crossings of $\lambda$) are exponentially rare, yielding ARL$_0$ of order $\exp(\lambda \Delta /(2\sigma^2))$. Under $H_1$, the statistic drifts upward at rate $\Delta/2$, so the stopping time scales as $\lambda / (\Delta/2) = 2\lambda / \Delta$ up to $O(1)$ corrections. $\square$

**Practical interpretation:** If we want ARL$_0 = 10{,}000$ episodes (false alarm every 10k episodes), set $\lambda = \log(10{,}000) \cdot 2\sigma^2 / \Delta \approx 18.4 \sigma^2 / \Delta$. For $\Delta = \sigma$ (one standard deviation shift), $\lambda \approx 18 \sigma$. This is why default configs often use $\lambda = 50\sigma$—it's conservative.

**Remark 10.3.2** (Known vs. Estimated Variance) {#REM-10.3.2}

[THM-10.4] assumes **known variance** $\sigma^2$. In practice, $\sigma$ is estimated online from the data stream (as in `zoosim/monitoring/drift.py:100-109`, which uses Welford's algorithm). When $\sigma$ is estimated online, the ARL bound holds **asymptotically** as $t \to \infty$ (when the variance estimate stabilizes), but finite-sample corrections apply:

1. **Early episodes ($t < 30$):** The variance estimate $\hat{\sigma}_t^2$ has high variance, so detection thresholds based on $\hat{\sigma}_t$ are unstable. This is why we enforce `min_instances` (default 30) before testing.
2. **Adaptive threshold scaling:** The implementation in `drift.py:121` sets threshold as $\lambda \cdot \hat{\sigma}_t$, not $\lambda \cdot \sigma$. As $\hat{\sigma}_t \to \sigma$, the theoretical guarantees apply.
3. **Practical calibration:** For production deployment, calibrate thresholds empirically on historical stationary data rather than relying solely on asymptotic formulas. See [REM-10.3.1] for calibration guidance.

### 10.3.2 Page-Hinkley Test

The **Page-Hinkley Test** [@page:test_change:1954] is a variant optimized for detecting **abrupt increases** (or decreases) in the mean. It's slightly simpler than CUSUM and works well for reward degradation detection (where we care about **reward drops**, not rises).

**Algorithm 10.3.2** (Page-Hinkley Test) {#ALG-10.2}

**Input:** Stream of observations $\{x_t\}_{t=1}^\infty$, tolerance $\delta$, threshold $\lambda$.

**Initialize:** $\bar{x}_0 = 0$, $m_0 = 0$, $M_0 = 0$ (cumulative sum and its minimum).

**For each episode $t = 1, 2, \ldots$:**

1. **Update mean:** $\bar{x}_t = \bar{x}_{t-1} + (x_t - \bar{x}_{t-1})/t$.
2. **Update cumulative deviation:** $m_t = m_{t-1} + (x_t - \bar{x}_{t-1} - \delta)$.
3. **Track minimum:** $M_t = \min(M_{t-1}, m_t)$.
4. **Test:** If $m_t - M_t > \lambda$, declare drift detected. Reset $m_t = M_t = 0$.

**End**

**Why this works.** The statistic $m_t - M_t$ measures the maximum upward excursion of the cumulative deviation $m_t$ from its historical minimum $M_t$. If the mean drops (reward degrades), $m_t$ trends downward, so $m_t - M_t$ stays small. If the mean increases (reward improves or error spikes), $m_t$ trends upward, eventually exceeding $\lambda$.

**For reward monitoring**, we feed $-r_t$ (negative reward) to Page-Hinkley, so reward drops become increases in $-r_t$ and trigger detection.

!!! note "Code ↔ Drift (Page-Hinkley Implementation)"
    Implemented in `zoosim/monitoring/drift.py:148-195` as class `PageHinkley`:
    - **Online mean**: Line 179 computes $\bar{x}_t$ incrementally (Welford's method).
    - **Cumulative deviation**: Line 182 computes $m_t = m_{t-1} + (x_t - \bar{x}_t - \delta)$.
    - **Minimum tracking**: Lines 184-185 maintain $M_t = \min_{s \leq t} m_s$.
    - **Drift signal**: Line 190 tests $m_t - M_t > \lambda$.
    - **Used in SafetyMonitor**: At `zoosim/monitoring/guardrails.py:138`, we feed `-reward` to detect reward degradation.

**Theorem 10.3.2** (Page-Hinkley Detection Guarantee) {#THM-10.5}

Under the same assumptions as [THM-10.4] (Gaussian rewards, mean shift $\Delta$), the Page-Hinkley test with $\delta = \Delta/2$ and threshold $\lambda$ has:

$$
\text{ARL}_0 \geq \exp(\lambda \Delta / \sigma^2), \quad \tau_d \leq \frac{\lambda}{\Delta} + O(\log \lambda)
$$

**Proof.**

Similar to CUSUM, relies on martingale stopping time analysis. The key difference: Page-Hinkley uses a one-sided test (detects increases, not decreases), so the bound is slightly tighter for detecting reward drops (by feeding $-r_t$). See [@page:test_change:1954] and [@basseville_nikiforov:detection_abrupt:1993] for full proofs. $\square$

**Comparison: CUSUM vs. Page-Hinkley.**

- **CUSUM**: Two-sided, detects both increases and decreases. More general but slightly higher variance.
- **Page-Hinkley**: One-sided, optimized for detecting shifts in a specific direction (e.g., reward drops). Slightly faster detection for the targeted shift direction.

In practice, both work well. We default to **Page-Hinkley** in `SafetyMonitor` because reward degradation (not improvement) is what triggers fallback.

---

## 10.4 Production Guardrails

Drift detection tells us **when** the world has changed. Guardrails tell us **what to do about it**. This section formalizes two key production constraints:

1. **Delta-Rank@k**: Stability metric that bounds how much rankings can change between episodes.
2. **CM2 Floor**: Business constraint that prevents negative-margin outcomes.

### 10.4.1 Delta-Rank@k: The Stability Metric

In e-commerce search, users **hate instability**. If a user searches for "dog food" and gets results A, B, C, then refreshes and gets C, D, E, they lose trust in the system. This is especially damaging for top results—users expect the first page to be stable unless their query or context changes significantly.

**Definition 10.4.1** (Delta-Rank@k Stability) {#DEF-10.4}

Let $\sigma_t : \{1, \ldots, K\} \to \{1, \ldots, K\}$ denote the ranking (permutation) of $K$ items at episode $t$. The **Delta-Rank@k** metric measures the fraction of items in the top-$k$ that changed compared to the previous episode:

$$
\Delta\text{-rank}@k(\sigma_t, \sigma_{t-1}) = 1 - \frac{|\text{Top}_k(\sigma_t) \cap \text{Top}_k(\sigma_{t-1})|}{k}
\tag{10.9}
$$
{#EQ-10.9}

where $\text{Top}_k(\sigma) = \{\sigma(1), \ldots, \sigma(k)\}$ is the set of items in positions 1 through $k$.

**Example 10.4.1** (Delta-Rank@10 Computation)

Consider two rankings of 12 items:

- **Episode $t-1$:** $\sigma_{t-1} = [3, 7, 1, 5, 9, 2, 8, 4, 10, 6, 11, 12]$
- **Episode $t$:** $\sigma_t = [3, 7, 12, 5, 9, 2, 8, 11, 10, 6, 1, 4]$

The top-10 sets are:
$$
\text{Top}_{10}(\sigma_{t-1}) = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}, \quad \text{Top}_{10}(\sigma_t) = \{2, 3, 5, 6, 7, 8, 9, 10, 11, 12\}.
$$

The intersection is $\{2, 3, 5, 6, 7, 8, 9, 10\}$ (8 items). Thus,
$$
\Delta\text{-rank}@10(\sigma_t, \sigma_{t-1}) = 1 - \frac{8}{10} = 0.2.
$$

**Interpretation:** 20% of the top-10 changed—items 1 and 4 dropped out (pushed to positions 11 and 12), while items 11 and 12 entered (promoted to positions 8 and 3). Note that internal reordering within the top-10 (e.g., item 12 jumped from position 12 to position 3) does **not** affect Delta-Rank@k—only set membership matters.

**Code verification:** This matches the output of `compute_delta_rank_at_k(ranking_prev, ranking_curr, k=10)` in `zoosim/monitoring/metrics.py:116-117`.

**Definition 10.4.2** (Deterministic Tie-Breaking Rule) {#DEF-10.4.2}

A **tie-breaking rule** is a function $\tau : 2^{\text{Items}} \to \text{Items}$ that selects a single item from any nonempty set of tied items (items with identical scores). The rule is **deterministic** if $\tau(S)$ produces the same output for the same input set $S$ across all episodes.

**Examples:**

1. **Lexicographic by ID:** $\tau(S) = \arg\min_{i \in S} i$ (choose the item with smallest ID).
2. **Stable sort:** If items $i, j$ were tied at episode $t-1$ with $i$ ranked above $j$, then $\tau(\{i,j\}) = i$ at episode $t$ (preserve previous order when possible).
3. **Random with fixed seed:** $\tau(S) = \text{random\_choice}(S, \text{seed}=42)$ (deterministic pseudo-random selection).

A deterministic tie-breaking rule [DEF-10.4.2] ensures that identical score configurations produce identical rankings, making $\Delta\text{-rank}@k$ a well-defined metric.

**Remark 10.4.1** (Tie-Breaking and Consistency) {#REM-10.4.1}

If multiple items have identical scores at the $k$-th position boundary, the ranking $\sigma$ must be determined by a deterministic tie-breaking rule [DEF-10.4.2] (for example, lexicographic order by item ID). The value of $\Delta\text{-rank}@k$ does depend on this rule: different tie-breaking conventions can produce different churn values for the same underlying scores. What matters for monitoring is **consistency over time**—using a fixed deterministic rule ensures that identical score configurations yield identical rankings at $t$ and $t-1$. In production, a stable sort (preserving previous order whenever possible, Example 2 in [DEF-10.4.2]) helps minimize churn caused purely by ties.

**Interpretation:**
- $\Delta\text{-rank}@k = 0$: The top-$k$ sets are identical (perfect stability).
- $\Delta\text{-rank}@k = 1$: The top-$k$ sets are completely disjoint (maximum churn).
- Typical production SLO: $\Delta\text{-rank}@10 \leq 0.2$ (at most 2 out of top 10 items change per refresh).

**Why not position-weighted metrics?** One might consider Kendall-$\tau$ or Spearman correlation, which penalize position swaps. But these are too strict for production: if items 3 and 4 swap (both are still in the top-5), users barely notice. Delta-Rank@k only cares about **set membership**, not order within the top-$k$, which matches user perception better.

!!! note "Code ↔ Metrics (Delta-Rank Implementation)"
    Implemented in `zoosim/monitoring/metrics.py:89-118` as `compute_delta_rank_at_k`:
    - **Input**: Two ranking lists `ranking_prev` and `ranking_curr` (item IDs in display order).
    - **Top-k extraction**: Lines 109-110 convert to sets `set_prev = set(ranking_prev[:k])`.
    - **Set intersection**: Line 116 computes overlap as `len(set_prev.intersection(set_curr))`.
    - **Churn formula**: Line 117 returns $1 - (\text{intersection} / k)$, exactly [EQ-10.9].
    - **Used in experiments**: `scripts/ch10/ch10_drift_demo.py:244` computes Delta-Rank@10 every episode and logs to `sim_data.stability`.

**Assumption 10.4.1** (Linear Dissatisfaction Model) {#ASM-10.4.1}

We assume user satisfaction $U_t$ decreases linearly with the number of unexpected changes in top-$k$ results:

$$
U_t = U_{\max} - \beta \cdot (\text{number of changes in top-}k)
$$

for some sensitivity parameter $\beta > 0$ and baseline satisfaction $U_{\max}$.

**Remark** (Empirical Validity of Linear Model)

The linear model [ASM-10.4.1] is a **simplification for analytical tractability**. Empirical validation in Chapter 11 (§11.4) reveals that user dissatisfaction is better modeled by a **quadratic function** of $\Delta\text{-rank}@k$, suggesting threshold effects (users tolerate small churn but disengage sharply after exceeding a threshold). Nevertheless, the linear approximation suffices for **setting production SLOs** (e.g., $\Delta\text{-rank}@10 \leq 0.3$), where we operate in a regime where linear and quadratic models agree qualitatively. See [REM-10.4.4] after [PROP-10.6] for full empirical analysis.

**Proposition 10.4.1** (Delta-Rank as Satisfaction Surrogate) {#PROP-10.6}

Under [ASM-10.4.1], expected user satisfaction satisfies

$$
\mathbb{E}[U_t \mid \sigma_t, \sigma_{t-1}] = U_{\max} - \beta k \cdot \Delta\text{-rank}@k(\sigma_t, \sigma_{t-1})
\tag{10.10}
$$
{#EQ-10.10}

**Proof.**

By definition [EQ-10.9], the number of changes is:

$$
k - |\text{Top}_k(\sigma_t) \cap \text{Top}_k(\sigma_{t-1})| = k \cdot \Delta\text{-rank}@k
$$

Substitute into $U_t = U_{\max} - \beta \cdot (\text{changes})$. $\square$

**Practical use.** Enforce $\Delta\text{-rank}@k \leq \delta$ as a hard constraint:

- **Online monitoring:** Compute Delta-Rank@k for every query. If $\Delta\text{-rank}@k > \delta$ for more than $p\%$ of queries in a rolling window, trigger an alert or revert to the previous policy.
- **Offline evaluation:** When comparing two policies $\pi_A$ and $\pi_B$, report Delta-Rank@k alongside GMV/CTR. A policy that increases GMV by 2% but has $\Delta\text{-rank}@10 = 0.8$ (unstable) is often rejected in practice.

**Counterexample 10.4.1** (Position-Sensitive Dissatisfaction)

Consider two ranking changes for top-10 results:

- **Change A:** Items in positions 9 and 10 swap. The top-10 set is unchanged, so $\Delta\text{-rank}@10 = 0$.
- **Change B:** The item in position 1 drops to position 11 and is replaced by a new item. The top-10 set changes by one item, so $\Delta\text{-rank}@10 = 1/10 = 0.1$.

Under the linear dissatisfaction model [ASM-10.4.1], Change A causes zero dissatisfaction, while Change B causes a small dissatisfaction proportional to $0.1$. In practice, users strongly notice Change B (the #1 result disappeared) and barely notice Change A (low positions swapped). A more faithful proxy is a **position-weighted churn metric**
$$
\Delta\text{-rank-weighted}@k
= \sum_{i=1}^k w_i \,\mathbb{1}\big[\sigma_t(i) \neq \sigma_{t-1}(i)\big],
\tag{10.10'}
$$
{#EQ-10.10-prime}
with weights $w_1 \geq w_2 \geq \cdots \geq w_k$ decaying with position (for example, $w_1 = 1.0, w_2 = 0.5, \ldots, w_{10} = 0.1$). In this metric, Change B incurs much larger dissatisfaction than Change A.

**Definition 10.4.3** (Position-Weighted Delta-Rank) {#DEF-10.4.3}

The **position-weighted Delta-Rank@k** metric accounts for the varying user sensitivity to changes at different positions:
$$
\Delta\text{-rank-weighted}@k(\sigma_t, \sigma_{t-1}) = \sum_{i=1}^k w_i \cdot \mathbb{1}[\sigma_t(i) \neq \sigma_{t-1}(i)]
$$
where $w_1 \geq w_2 \geq \cdots \geq w_k > 0$ are position weights. Common choices include:
- **Reciprocal rank:** $w_i = 1/i$ (position 1 has weight 1, position 10 has weight 0.1)
- **DCG weighting:** $w_i = 1/\log_2(i+1)$ (standard in information retrieval)
- **Exponential decay:** $w_i = \gamma^{i-1}$ for some $\gamma \in (0,1)$

**Remark.** Unlike [DEF-10.4], position-weighted Delta-Rank@k is sensitive to **where** changes occur, not just **how many**. In production, use position-weighted metrics when top-1 or top-3 stability is critical; use set-based Delta-Rank@k when only overall top-10 stability matters.

Conclusion: Delta-Rank@k is a **coarse stability proxy**. For production, it is best combined with position-weighted metrics or direct engagement signals (e.g., click-through rate on changed positions).

**Remark 10.4.4** (Empirical Validation of Linear Dissatisfaction) {#REM-10.4.4}

The linear model [ASM-10.4.1] is a simplification. Real user behavior may exhibit:
- **Threshold effects**: Users tolerate 1–2 changes but disengage sharply after 3+.
- **Position sensitivity**: Changes in top-3 are more disruptive than changes in positions 8–10.
- **Adaptation**: Frequent users may become tolerant of moderate churn.

**Empirical test.** In Chapter 11's retention experiments (§11.4), we measure user return rate as a function of historical $\Delta\text{-rank}@k$ and find that a **quadratic model** $U_t = U_{\max} - \beta_1 (\Delta\text{-rank}@k) - \beta_2 (\Delta\text{-rank}@k)^2$ fits better ($R^2 = 0.82$ vs. $0.67$ for linear). However, the linear approximation suffices for production SLO setting (§10.4).

### 10.4.2 CM2 Floor: Business Viability Constraint

**Contribution Margin 2 (CM2)** measures profitability after variable costs:

$$
\text{CM2} = \text{GMV} - \text{COGS} - \text{Logistics} - \text{Marketing}
\tag{10.11}
$$
{#EQ-10.11}

For e-commerce platforms, **negative CM2 is unsustainable**—you're losing money on every transaction. An RL policy that maximizes clicks but recommends low-margin products can be catastrophic.

**Definition 10.4.2** (CM2 Floor Constraint) {#DEF-10.5}

A policy $\pi$ satisfies the **CM2 floor** $C_{\min}$ if:

$$
\mathbb{E}_{\tau \sim \pi}[\text{CM2}(\tau)] \geq C_{\min}
\tag{10.12}
$$
{#EQ-10.12}

Typically $C_{\min} = 0$ (non-negative margin) or $C_{\min} = 0.15 \cdot \mathbb{E}[\text{GMV}]$ (15% margin floor).

**Implementation approaches:**

1. **Lagrangian relaxation:** Augment the reward with a penalty term
   $$
   r'_t = r_t + \mu \max\bigl(0, C_{\min} - \text{CM2}_t\bigr),
   $$
   where $\mu > 0$ is a Lagrange multiplier tuned (for example, via binary search) to enforce $\mathbb{E}[\text{CM2}_t] \geq C_{\min}$ on average. This is the standard **primal–dual method** for constrained MDPs (see **Appendix C** for Lagrangian duality theory and Slater's condition; [@altman:constrained_mdps:1999] for constrained MDP foundations). We return to this optimization viewpoint and implement `primal--dual` constrained RL in Chapter 14.

   **Regularity condition (Slater's condition).** Strong duality holds—meaning the Lagrangian approach finds the true constrained optimum—when Slater's condition is satisfied: there exists a **strictly feasible** policy $\pi$ with $\mathbb{E}_\pi[\text{CM2}] > C_{\min}$ (strict inequality). This is typically satisfied in e-commerce when at least one template has positive margin exceeding the floor. If all templates have margins exactly at $C_{\min}$, the constraint qualification fails and the Lagrangian method may not converge to the optimal constrained policy. See [@altman:constrained_mdps:1999, Chapter 4] for constrained MDP duality theory and [@boyd_vandenberghe:convex_optimization:2004, §5.2.3] for the general convex optimization perspective.

2. **Hard rejection sampling**: If an action $a$ leads to $\text{CM2} < C_{\min}$, reject it and resample. This requires predicting CM2 before executing the action (feasible if product margins are known).

3. **Post-hoc filtering**: After the policy proposes a ranking, remove items with negative individual margins. This is the simplest approach and is used in most production systems.

!!! note "Code ↔ Metrics (CM2 Computation)"
    CM2 computation is defined in `zoosim/monitoring/metrics.py:66-86`:
    - **Inputs**: GMV, COGS, logistics cost, marketing cost (all in currency units).
    - **Formula**: Line 86 returns `gmv - costs - logistics_cost - marketing_cost`, exactly [EQ-10.11].
    - **Usage in experiments**: `scripts/ch10/ch10_drift_demo.py:102-103` computes CM2 as 40% of GMV (simplified model where margin is fixed at 40%).
    - **Guardrails and CM2 floors**: `GuardrailConfig` includes CM2-related fields (`enable_cm2_floor`, `min_cm2`) at `zoosim/monitoring/guardrails.py:41-42`, but the `SafetyMonitor` reference implementation in §10.5 operates on scalar rewards and does not evaluate CM2 directly. A CM2 floor is therefore enforced as an action-feasibility filter at action selection time (Exercise 10.3).

!!! warning "Sim-to-Real Note: Variable vs. Fixed Margins"
    In our production definition [EQ-10.11], CM2 depends on variable costs (COGS, logistics). In the `ch10_drift_demo.py` experiment, we simplify this by assuming a fixed 40% margin (`cm2 = gmv * 0.4`). This simplification is acceptable for demonstrating drift detection, but real-world implementations must use the full cost model. See **Exercise 10.3** for an extension with per-template margins.

### 10.4.3 Fallback Policies: Safe Baselines

When drift is detected or guardrails are breached, we need a **fallback policy** that is safe (known to satisfy constraints) even if suboptimal.

**Standard fallback strategies:**

1. **Best Fixed Template** (used in our experiments): Revert to the single template that performed best during training. This is stationary, stable (Delta-Rank@k = 0), and usually satisfies CM2 floors if the baseline catalog is healthy.

2. **Last Known Good Policy**: Checkpoint the policy parameters every $N$ episodes. When drift is detected, revert to the checkpoint from before the drift.

3. **Uniform Random**: Sample actions uniformly. This is unbiased but high-variance. Rarely used in production except for cold-start or as an exploration layer.

4. **Human-Curated Heuristic**: Fall back to a business-logic ranking (e.g., sort by GMV-per-impression). Many production systems use this as the ultimate fallback.

**Theorem 10.4.2** (Performance Guarantee of Best Fixed Template) {#THM-10.7}

Let $\pi_{\text{BFT}}$ denote the best fixed template policy (the one with highest average reward over all training data). Assume:

1. Bounded drift [DEF-10.1.3] with rates $(\delta_P, \delta_R)$.
2. **Uniform ergodicity:** There exist constants $C > 0$ and $\rho \in (0,1)$, independent of $t$, such that for all initial distributions $\nu$ on $\mathcal{S}$ and all $n \geq 1$:
   $$
   \|P_t^n \nu - \rho_t\|_{\text{TV}} \leq C \rho^n,
   $$
   where $P_t^n$ denotes the $n$-step transition kernel and $\rho_t$ is the unique stationary distribution of $P_t$. The **mixing time** is then bounded by
   $$
   T_{\text{mix}} := \inf\{n : \sup_\nu \|P_t^n \nu - \rho_t\|_{\text{TV}} \leq 1/e\} \leq \frac{\log(Ce)}{|\log \rho|} = O\left(\frac{1}{1-\rho}\right),
   $$
   uniformly in $t$.

Let $\pi^*_t$ denote the optimal time-varying policy at time $t$. Then:

$$
\mathbb{E}_{\pi_{\text{BFT}}}[R] \geq \min_{t} \mathbb{E}_{\pi^*_t}[R] - \epsilon
\tag{10.13}
$$
{#EQ-10.13}

where $\epsilon = O((\delta_P \cdot T_{\text{mix}} + \delta_R) \cdot R_{\max})$ bounds the loss from suboptimality within each regime.

**Proof.**

Let $\mathcal{T} = \{T_1, \ldots, T_K\}$ denote the set of available templates. For each regime $t \in \{1, \ldots, T\}$, define
$$
V_t(T_i)
= \mathbb{E}_{s \sim \rho,\, a = T_i(s)}[R_t(s,a)]
$$
as the expected reward of template $T_i$ in regime $t$ under some stationary distribution $\rho$ (assuming the MDP mixes within each regime).

The best fixed template is, by definition,
$$
\pi_{\text{BFT}} = \arg\max_{T_i \in \mathcal{T}} \frac{1}{T} \sum_{t=1}^T V_t(T_i),
$$
the template that maximizes the time-averaged reward over all regimes.

Let $\pi^*_t$ denote the optimal time-varying policy in regime $t$, and write $V_t(\pi^*_t) = \max_{T_i} V_t(T_i)$ for the best achievable reward among the templates in regime $t$. Then
$$
\frac{1}{T} \sum_{t=1}^T V_t(\pi_{\text{BFT}})
\geq \frac{1}{T} \sum_{t=1}^T \min_{T_i \in \mathcal{T}} V_t(T_i)
\geq \min_t V_t(\pi^*_t) - \epsilon,
$$
where $\epsilon$ accounts for finite-horizon mixing effects and the fact that $\pi^*_t$ may not lie exactly in the template set $\mathcal{T}$.

**Bounding $\epsilon$ via mixing times.** Under bounded drift [DEF-10.1.3], the time-varying transition kernels $\{P_t\}_{t=1}^T$ satisfy
$$
\|P_t - P_{t+1}\|_{\text{TV}} \leq \delta_P \quad \text{for all } t.
$$
Assume each $P_t$ has uniform mixing time bound $T_{\text{mix}}$ (i.e., from any initial state, the distribution converges to the stationary distribution $\rho_t$ within $T_{\text{mix}}$ steps with TV distance $\leq 1/e$). By the ergodicity preservation lemma [@levin-peres-wilmer:markov_chains:2017, Theorem 12.4], if $\|P_t - P_{t+1}\|_{\text{TV}} \leq \delta_P$, then the stationary distributions satisfy
$$
\|\rho_t - \rho_{t+1}\|_{\text{TV}} \leq \delta_P \cdot T_{\text{mix}}.
$$
Consequently, the value function for template $T_i$ satisfies
$$
|V_t(T_i) - V_{t+1}(T_i)|
= \left|\mathbb{E}_{s \sim \rho_t}[R_t(s, T_i(s))] - \mathbb{E}_{s \sim \rho_{t+1}}[R_{t+1}(s, T_i(s))]\right|
\leq R_{\max} \cdot \|\rho_t - \rho_{t+1}\|_{\text{TV}} + \delta_R
\leq R_{\max} \cdot \delta_P \cdot T_{\text{mix}} + \delta_R,
$$
where the first inequality uses total variation coupling and the second uses $|R_t(s,a) - R_{t+1}(s,a)| \leq \delta_R$ from [DEF-10.1.3]. Thus,
$$
\epsilon = O\bigl((\delta_P \cdot T_{\text{mix}} + \delta_R) \cdot R_{\max}\bigr).
$$
For uniformly ergodic MDPs with bounded mixing times, this yields a computable bound. For non-ergodic or slowly-mixing MDPs ($T_{\text{mix}} \to \infty$), the bound may not be useful, and alternative fallback strategies (e.g., Last Known Good Checkpoint, §10.4.3) should be used. $\square$

**Practical note.** The best fixed template is a reasonable fallback for **bounded drift** (seasonal trends, gradual catalog changes). But for **catastrophic drift** (entire product category goes out of stock), even the best fixed template may fail. In such cases, fallback to human-curated heuristics is necessary.

**Remark 10.4.2** (Conservative Nature of Best Fixed Template Bound) {#REM-10.4.2}

The guarantee in [THM-10.7] is deliberately conservative: it compares $\pi_{\text{BFT}}$ to the **worst** regime $\min_t \mathbb{E}_{\pi^*_t}[R]$. In practice, when regime heterogeneity is bounded, the best fixed template often tracks the **time-averaged optimal** reward much more closely. Stronger results are available in the online optimization literature under additional assumptions on the drift sequence (see, e.g., tracking regret bounds in [@jadbabaie_rakhlin:online_optimization:2015]), but we keep the statement here simple and interpretable for production use.

**Remark 10.4.3** (Ergodicity Assumption in [THM-10.7]) {#REM-10.4.3}

The bound [EQ-10.13] requires **uniform ergodicity**: each $P_t$ must have mixing time $T_{\text{mix}}$ bounded independently of $t$. This assumption appears implicitly in the proof when we invoke the ergodicity preservation lemma [@levin-peres-wilmer:markov_chains:2017, Theorem 12.4] to bound $\|\rho_t - \rho_{t+1}\|_{\text{TV}}$. Without uniform ergodicity, two failure modes arise:

1. **Non-ergodic chains** (e.g., periodic or reducible MDPs): The stationary distribution $\rho_t$ may not exist or may not be unique, invalidating the value function definition $V_t(T_i) = \mathbb{E}_{s \sim \rho_t}[R_t(s, T_i(s))]$.

2. **Slowly-mixing chains** ($T_{\text{mix}} \to \infty$): The bound $\epsilon = O(\delta_P \cdot T_{\text{mix}} + \delta_R)$ becomes vacuous—even small drift $\delta_P$ accumulates into large value function changes when mixing is slow.

**Counterexample 10.4.2** (Non-Ergodic User Population)

Consider a user population with two **absorbing states**: "churned" (user never returns) and "loyal" (user returns every session). Suppose an aggressive RL policy—one that maximizes short-term GMV by showing low-relevance, high-margin products—increases the churn rate. Users who churn cannot be "recovered" back to the loyal state through any action: the MDP becomes **non-ergodic** because the churned state is absorbing.

Formally, let $\mathcal{S} = \{\text{loyal}, \text{at-risk}, \text{churned}\}$ with transitions:
- From "loyal": stay loyal (probability $0.9$) or become at-risk (probability $0.1$)
- From "at-risk": return to loyal (probability $p_\pi$) or churn (probability $1 - p_\pi$), where $p_\pi$ depends on policy quality
- From "churned": stay churned (probability $1$) — **absorbing state**

An aggressive policy reduces $p_\pi$ (e.g., from $0.7$ to $0.3$), accelerating flow to the absorbing "churned" state. The mixing time analysis in [THM-10.7] breaks down: there is no stationary distribution because churned users accumulate indefinitely. The fallback guarantee [EQ-10.13] becomes vacuous—even the best fixed template cannot recover churned users.

**Lesson:** When user retention dynamics include absorbing states (churn, account deletion), ergodicity assumptions fail. The fallback strategy must account for irreversible damage, not just temporary performance loss. See Chapter 11 (§11.2) for retention-aware policy design.

**When ergodicity holds in practice:** E-commerce search MDPs are typically ergodic when restricted to **active users** (users can reach any query state from any other via browsing), but mixing times can be long when user sessions span many interactions. When user churn is possible, the "Last Known Good Checkpoint" fallback strategy (§10.4.3) is preferable: rather than relying on time-averaged performance bounds, we explicitly store and restore known-good policy parameters before damage becomes irreversible.

!!! note "Code ↔ Guardrails (Fallback Policy)"
    Implemented in `scripts/ch10/ch10_drift_demo.py:52-61` as `StaticPolicy`:
    - **Action selection**: Line 58 always returns `self.fixed_action` (no learning, no randomness).
    - **Used as fallback**: `SafetyMonitor` at line `zoosim/monitoring/guardrails.py:72` accepts `safe_policy` as a constructor argument, defaults to a static policy.
    - **Trigger**: Line 166 switches `self.in_fallback_mode = True` when drift is detected, and line 101 delegates `select_action` to `self.safe_policy`.

---

## 10.5 Implementation: The `zoosim.monitoring` Package

We now integrate everything into production-grade infrastructure. The `zoosim.monitoring` package provides:

1. **`drift.py`**: Abstract `DriftDetector` interface, `CUSUM`, `PageHinkley` implementations.
2. **`guardrails.py`**: `SafetyMonitor` wrapper that orchestrates drift detection, fallback, and automatic recovery.
3. **`metrics.py`**: Business and stability metrics (CTR, CVR, GMV, CM2, Delta-Rank@k).

### 10.5.1 Architecture Overview

```
┌────────────────────────────────────────────────────┐
│              User Application Layer                │
│   (scripts/ch10/ch10_drift_demo.py)               │
└────────────────┬───────────────────────────────────┘
                 │ select_action(features)
                 │ update(action, features, reward)
                 ▼
┌────────────────────────────────────────────────────┐
│            SafetyMonitor Wrapper                   │
│   ┌──────────────────────────────────────────┐   │
│   │  Primary Policy (LinUCB / Q-learning)    │   │
│   │  Safe Policy (Best Fixed Template)       │   │
│   └──────────────────────────────────────────┘   │
│   ┌──────────────────────────────────────────┐   │
│   │  DriftDetector (CUSUM / Page-Hinkley)    │   │
│   │  GuardrailConfig (thresholds, SLOs)      │   │
│   └──────────────────────────────────────────┘   │
└────────────────┬───────────────────────────────────┘
                 │ monitor rewards, trigger fallback
                 ▼
┌────────────────────────────────────────────────────┐
│          Environment (Production / Simulator)      │
└────────────────────────────────────────────────────┘
```

**Key design principles:**

1. **Separation of concerns**: `SafetyMonitor` wraps any policy that implements `select_action(features)` and `update(...)` (duck typing via `PolicyProtocol` at `guardrails.py:22`).

2. **Stateful drift detection**: Drift detectors maintain internal state (cumulative sums, rolling statistics). `SafetyMonitor` owns the detector instance and calls `detector.update(reward)` after each episode.

3. **Automatic recovery**: When in fallback mode, `SafetyMonitor` periodically probes the primary policy (every `probe_frequency` episodes) to check if the world has stabilized. If drift is no longer detected, it switches back to the primary policy.

### 10.5.2 `SafetyMonitor` Implementation Walkthrough

Let's trace a typical episode through the code:

**1. Action Selection** (`zoosim/monitoring/guardrails.py:94-113`)

```python
def select_action(self, features: NDArray[np.float64]) -> int:
    if self.in_fallback_mode:
        return self.safe_policy.select_action(features)  # Use fallback

    action = self.primary_policy.select_action(features)  # Use primary

    if self.config.enable_stability_check:
        self._check_stability(action)  # Monitor action switching (proxy for churn)

    return action
```

- **Line 100**: If in fallback mode, delegate to `safe_policy` (e.g., static template).
- **Line 103**: Otherwise, use the primary policy (e.g., LinUCB).
- **Line 106-109**: Optionally check stability (in our implementation, we track action switching rate as a coarse proxy for ranking churn).

**2. Reward Update and Drift Monitoring** (`guardrails.py:115-145`)

```python
def update(self, action: int, features: NDArray[np.float64], reward: float) -> None:
    if self.in_fallback_mode:
        self.safe_policy.update(action, features, reward)
        self.fallback_counter += 1

        # Automatic Recovery Probing
        if self.fallback_counter >= self.config.probe_frequency:
            print("[SafetyMonitor] 🔄 Probing Primary Policy")
            self.reset_to_primary()
    else:
        self.primary_policy.update(action, features, reward)

    # Monitor Drift (feed -reward to detect reward degradation)
    if self.config.enable_drift_detection:
        drift = self.drift_detector.update(-reward)  # Note: negative reward
        if drift:
            self.drift_counter += 1
            if self.drift_counter >= self.config.drift_patience:
                self._trigger_fallback("Reward Drift Detected")
        else:
            self.drift_counter = max(0, self.drift_counter - 1)  # Decay counter
```

- **Line 120-128**: If in fallback mode, update the safe policy and count episodes. After `probe_frequency` episodes (default 100), attempt to reset to primary.
- **Line 130**: Update the primary policy with the new reward observation.
- **Line 133-145**: Feed $-r_t$ to the drift detector (Page-Hinkley detects increases, so negative reward → reward drop → drift signal). If drift is detected for `drift_patience` consecutive episodes (default 5), trigger fallback.
- **Line 144**: If no drift, decay the counter to avoid false alarms from transient noise.

**3. Fallback Trigger** (`guardrails.py:162-170`)

```python
def _trigger_fallback(self, reason: str) -> None:
    if not self.in_fallback_mode:
        print(f"[SafetyMonitor] 🚨 FALLBACK TRIGGERED: {reason}")
        self.in_fallback_mode = True
        self.violations["drift_events"] += 1
        self.drift_detector.reset()  # Clear detector state
        self.drift_counter = 0
```

- **Line 166**: Set `in_fallback_mode = True`. Subsequent `select_action` calls will use `safe_policy`.
- **Line 168**: Reset the drift detector to avoid "drift lock" (continuous false alarms after the first detection).

**4. Automatic Recovery** (`guardrails.py:172-178`)

```python
def reset_to_primary(self) -> None:
    self.in_fallback_mode = False
    self.drift_detector.reset()
    self.fallback_counter = 0
    print("[SafetyMonitor] ✅ Restored Primary Policy")
```

- Called after `probe_frequency` episodes in fallback mode (line 127).
- Resets flags and gives the primary policy another chance. If drift recurs, it will trigger fallback again.

!!! note "Code ↔ Guardrails (SafetyMonitor Parameters)"
    Configuration via `GuardrailConfig` dataclass (`guardrails.py:32-47`):
    - `enable_drift_detection`: Turn drift monitoring on/off (default True).
    - `drift_patience`: Number of consecutive drift signals before fallback (default 5). Lower values are more sensitive but increase false alarms.
    - `probe_frequency`: Episodes to wait in fallback before probing primary (default 100). This implements automatic recovery.
    - `enable_stability_check`: Monitor action switching rate (default True). Not currently used to trigger fallback, but logged for diagnostics.

    Instantiation in experiment at `scripts/ch10/ch10_drift_demo.py:193-206`:
    ```python
    guard_config = GuardrailConfig(
        enable_drift_detection=True,
        drift_patience=5,
        probe_frequency=100,
        enable_stability_check=True
    )
    monitor = SafetyMonitor(primary_policy, safe_policy, guard_config, drift_detector)
    ```

### 10.5.3 Design Pattern: Policy as Protocol

Notice that `SafetyMonitor` does not depend on specific policy classes (`LinUCB`, `REINFORCE`, etc.). Instead, it uses **structural subtyping** via `PolicyProtocol` (`guardrails.py:22-28`):

```python
class PolicyProtocol(Protocol):
    def select_action(self, features: NDArray[np.float64]) -> int: ...
    def update(self, action: int, features: NDArray[np.float64], reward: float) -> None: ...
```

Any class that implements these two methods is compatible with `SafetyMonitor`, even if it doesn't inherit from a common base class. This is Python's **Protocol** feature (similar to Go interfaces or Rust traits).

**Benefits:**
- **Composability**: Wrap any policy (bandit, Q-learning, policy gradient) with the same monitoring infrastructure.
- **Testability**: Easy to inject mock policies for unit tests.
- **Maintainability**: No tight coupling to specific algorithm implementations.

This is a **production design pattern**. In research code, we might just pass `policy` as an untyped object. But for systems deployed at scale, explicit interfaces prevent bugs and enable code reuse.

---

## 10.6 Experiments: Seasonal Drift Scenario

We now validate the theory with a controlled experiment simulating **abrupt preference drift**. This corresponds to a realistic production scenario: user preferences shift seasonally, and our RL policy must detect and adapt.

### 10.6.1 Experimental Setup

**Environment:** `DriftEnvironment` (`scripts/ch10/ch10_drift_demo.py:64-129`)

- **Phase 1 (episodes 0-1499)**: Template T1 (High Margin) has best CVR (12%). Optimal action: $a^* = 1$.
- **Phase 2 (episodes 1500-2999)**: Template T1 crashes to CVR 2%, Template T3 (Popular) becomes best (CVR 15%). Optimal action: $a^* = 3$.
- **Drift event**: Occurs at $t_0 = 1500$ (halfway through 3000-episode run).

**Policies:**

1. **Primary (Adaptive):** LinUCB with $\alpha = 1.5$, $\lambda = 1.0$ (ridge regression regularization). This is our learning agent.
2. **Safe (Fallback):** Static policy always selecting Template T0 (Neutral). This is intentionally mediocre but stable.

**Drift Detector:** Page-Hinkley with $\delta = 0.01$, $\lambda = 30.0$.

**Safety Monitor:** Configured with:
- `drift_patience = 5` (trigger fallback after 5 consecutive drift signals)
- `probe_frequency = 100` (attempt recovery every 100 episodes in fallback)
- `enable_stability_check = True` (monitor action switching rate; proxy for churn)

**Metrics Tracked:**

- **Reward (composite)**: CM2 + noise (primary optimization target)
- **GMV**: Gross Merchandise Value (business metric)
- **CVR**: Conversion rate (ground truth quality)
- **Delta-Rank@10**: Stability metric (set churn in top-10 results)
- **Latency**: Simulated system health metric (milliseconds per query)
- **Mode**: Binary indicator (0 = primary policy active, 1 = fallback active)

**Code Location:** `scripts/ch10/ch10_drift_demo.py:172-264` (function `simulate_drift_scenario`)

### 10.6.2 Results and Analysis

**Figure 10.1** shows the 4-panel visualization produced by the experiment:

![Drift Detection Demo](../../../ch10_drift_demo.png)

Let's analyze each panel:

**Panel 1: Reward and Fallback Mode**

- **Pre-drift (t < 1500)**: LinUCB learns that Template T1 is best. Reward converges to $\approx 1.2$ (50-episode moving average).
- **Drift event (t = 1500)**: Marked by red dashed line. Reward drops sharply as T1's CVR crashes from 12% to 2%.
- **Fallback trigger (t ≈ 1530-1540)**: Orange shaded region indicates fallback mode activated. The Page-Hinkley detector accumulates drift signals over ~30-40 episodes, then triggers fallback after 5 consecutive signals (patience threshold).
- **Recovery (t ≈ 1640)**: SafetyMonitor probes primary policy after 100 episodes in fallback. The drift signal clears (the world has stabilized in Phase 2), so the monitor switches back to primary.
- **Post-recovery**: LinUCB adapts to the new regime, learning that T3 is now optimal. Reward converges to $\approx 0.8$ (lower than Phase 1 because T3 has lower margin than T1 did).

**Key observations:**

1. **Detection delay**: $\tau_d = 35.2 \pm 4.1$ episodes (mean $\pm$ 1.96 SE, $n = 20$ seeds). The theoretical lower bound [EQ-10.8] gives $\tau_{\min} \approx 18.4$ episodes for the CVR drop from 12% to 2% (KL divergence $\approx 0.28$), so the observed delay is about $1.9\times$ the asymptotic optimum.
2. **Fallback protects reward**: During fallback (orange region), reward stabilizes at $\approx 0.5$ (the safe static policy). Without fallback, reward would continue to drop as LinUCB chases the now-suboptimal T1.
3. **Automatic recovery works**: The monitor successfully probes and restores the primary policy, allowing it to adapt to Phase 2.

**Statistical validation.** Treating the theoretical lower bound $\tau_{\min} = 18.4$ as a reference, a Welch's $t$-test comparing the sample mean $\bar{\tau}_d = 35.2$ (SE $\approx 4.1$) to $\tau_{\min}$ yields a highly significant difference ($t \approx 4.1$, $\text{df} \approx 19$, $p < 0.001$). The gap arises from:

1. **Discretization:** [EQ-10.8] assumes continuous monitoring; our experiment observes rewards once per episode.
2. **Finite-sample effects:** The CUSUM/Page-Hinkley statistics are random walks with variance, so detection is a noisy threshold crossing.
3. **Patience threshold:** We require 5 consecutive drift signals before triggering fallback, adding roughly 5 episodes of delay.

Adjusting for the patience threshold, the "raw" detection delay of the detector is closer to $\approx 30$ episodes, reducing the factor to about $1.6\times$ the theoretical lower bound—acceptable for a finite-sample, discretized implementation.

Running `simulate_drift_scenario` over multiple random seeds yields:

- **Detection delay:** $\tau_d = 35.2 \pm 4.1$ episodes (mean ± 1.96 SE over 20 seeds)
- **Recovery time:** $140 \pm 18$ episodes (mean ± 1.96 SE)

The variability arises from:
1. **Stochastic rewards**: CVR is Bernoulli, leading to noisy observations.
2. **CUSUM/Page-Hinkley accumulation**: The cumulative sum statistic is a random walk with drift, so detection time varies by ±10–20 episodes.
3. **LinUCB exploration**: Confidence-based exploration introduces randomness in action selection and thus in when drift becomes evident.

**Panel 2: Policy Selection (Template Actions)**

- **Pre-drift**: LinUCB quickly learns to select T1 (green dots at $y=1$). Exploration decreases over time as confidence intervals shrink.
- **Post-drift**: After recovery, LinUCB explores briefly (scatter across multiple templates), then converges to T3 (blue dots at $y=3$).
- **Fallback period**: Dense cluster of orange dots at $y=0$ (static policy always selects T0).

**Theory-practice gap:** LinUCB's confidence-based exploration (UCB) should eventually discover T3 is better. But this takes time—our experiment shows $\approx 300$ episodes to fully converge post-drift. Theory says regret should be $O(\log T)$ [@abbasi-yadkori:improved_algorithms:2011], but constants are large when the action space has many arms ($K=4$ here is actually small; production systems have $K > 100$ templates).

**Panel 3: Stability (Delta-Rank@10)**

- **Pre-drift**: Delta-Rank@10 oscillates around 0.2-0.3. This is expected—LinUCB explores occasionally, causing ranking changes. But the top-10 set is mostly stable.
- **Drift event**: Spike to 0.6-0.8 immediately after $t=1500$. The policy is confused as T1 suddenly performs poorly, so it thrashes between alternatives.
- **Fallback period**: Delta-Rank drops to $\approx 0.1$. The static policy is maximally stable (always returns the same ranking).
- **Post-recovery**: Delta-Rank rises back to 0.3 as LinUCB explores to learn the new regime.

**Production implication:** If we enforce a hard SLO of $\Delta\text{-rank}@10 \leq 0.4$, this experiment would not breach it (max observed is 0.6 for a brief transient). But if the drift were more gradual, the policy might thrash longer. In practice, **stability constraints and drift detection must be tuned together**.

**Panel 4: System Health (Latency)**

- **Stable at ~30-40ms** throughout the run. The spikes are synthetic noise (gamma distribution added at `ch10_drift_demo.py:237`). Note that this latency is **simulated** to demonstrate the monitoring infrastructure; it does not reflect the actual compute time of the LinUCB agent.
- **No degradation during drift or fallback**: This demonstrates that the SafetyMonitor overhead is negligible (O(1) per episode for drift detection).

**Real production systems** would also track:
- **P99 latency** (99th percentile, not just mean)
- **Cache hit rates** (policy changes invalidate cached rankings)
- **Database load** (retraining might trigger heavy queries)

Our simulator doesn't model these, but the monitoring framework (`metrics.py`) is extensible—add new metrics as dataclass fields.

### 10.6.3 Recovery Time Analysis

The experiment computes **recovery time** as the number of episodes needed to restore 90% of pre-drift baseline performance (function `calculate_recovery_time` at `ch10_drift_demo.py:148-169`).

**Result:** Recovery time = **140 episodes** (printed by the script).

**Breakdown:**
- **Detection delay**: 30-40 episodes (from drift event at $t=1500$ to fallback trigger at $t \approx 1540$).
- **Fallback duration**: 100 episodes (the `probe_frequency` setting).
- **Adaptation time**: 0-20 episodes after recovery (LinUCB needs to explore and learn that T3 is now best).

**Total**: 130-160 episodes, consistent with the measured 140.

**Theory-practice gap:** The baseline is Phase 1 reward ($\approx 1.2$), but Phase 2 optimal reward is only $\approx 0.8$ (because T3 has lower margin than T1). So we never fully "recover" to the baseline—**the world itself is worse**. This highlights a subtle issue: in non-stationary environments, **performance can drop permanently** if the new regime is fundamentally less favorable. Drift detection doesn't prevent this; it only helps the policy adapt to the new reality.

---

## 10.7 Theory-Practice Gaps and Open Problems

This chapter has developed rigorous drift detection algorithms, implemented production guardrails, and validated them empirically. But significant gaps remain between theory and practice.

### 10.7.1 What Theory Guarantees (and Doesn't)

**Theory tells us:**

1. **CUSUM and Page-Hinkley are asymptotically optimal** for detecting mean shifts in Gaussian processes, achieving detection delays near the information-theoretic lower bound [EQ-10.8].

2. **Lyapunov stability analysis** characterizes tracking error under bounded drift [THM-10.2], giving us a principled way to set learning rates.

3. **Delta-Rank@k** is a valid surrogate for user satisfaction [PROP-10.6], grounded in a linear dissatisfaction model.

**Theory does NOT tell us:**

1. **How to detect drift in high-dimensional, non-Gaussian reward distributions.** Our implementations assume univariate rewards. Production systems have multivariate metrics (GMV, CTR, CVR, margin) with complex correlations. Multivariate CUSUM exists [@crosier:multivariate_cusum:1988], but tuning the covariance matrix is hard. A prototype `MultivariateCUSUM` implementation appears in the Chapter 10 labs (Exercise 10.4 solution); promoting it to `zoosim.monitoring.drift` with dedicated tests is recommended for production.

2. **How to distinguish between drift and non-stationarity within a regime.** E-commerce has **structured non-stationarity**: weekday vs. weekend patterns, hourly trends, seasonal cycles. Drift detection should filter out predictable variation. Current algorithms treat all variation as potential drift.

3. **How to set thresholds $\lambda$ without hindsight.** In our experiment, we chose $\lambda = 30.0$ by trial and error. Theory says $\lambda$ controls the false alarm rate [THM-10.4], but we don't know the pre-drift and post-drift distributions in advance. This is the **calibration problem**—it's unsolved for general MDPs.

4. **How to recover from **catastrophic drift**** (entire state space changes). Our experiments simulate **parameter drift** (CVRs change, but the action space is the same). What if the catalog is replaced entirely? Theory has no guidance here.

!!! warning "Open Problem 10.7.1: Calibration Without Hindsight"

    **Problem.** Given a non-stationary MDP with unknown drift profile $\{\delta_t\}_{t \geq 1}$ and no prior on $(P_0, P_1)$, determine the threshold $\lambda$ for Page-Hinkley such that:

    1. False alarm rate $\mathbb{P}_{P_0}(\text{detect drift} \mid H_0 \text{ true}) \leq \alpha$ (e.g., $\alpha = 0.01$)
    2. Detection delay $\tau_d \leq \tau_{\max}$ (e.g., 50 episodes)

    **Why it's hard.** [THM-10.3] requires knowing $D_{\text{KL}}(P_1 \| P_0)$ to set $\lambda$, but $(P_0, P_1)$ are learned online from the same data stream used for detection. Existing approaches [@maillard:online_changepoint:2019] assume finite change-point counts or bounded KL divergence—neither holds in production.

    **Heuristic solution.** Tune $\lambda$ on held-out historical drift events (if available) or use adaptive thresholds [@ross:nonparametric_changepoint:2012]. Chapter 11 explores this in multi-session experiments.

!!! warning "Open Problem 10.7.2: Multivariate Drift Detection with Unknown Covariance"

    **Problem.** Extend CUSUM/Page-Hinkley to vector-valued metrics $(r_{\text{GMV}}, r_{\text{CTR}}, r_{\text{CVR}})$ when the covariance matrix is unknown and time-varying.

    **Difficulty.** Estimating a full covariance matrix $\boldsymbol{\Sigma}_t$ from streaming data requires $O(d^2)$ samples per update, but production systems need $O(1)$ online algorithms.

    **Partial solutions.** [@ross:nonparametric_changepoint:2012] uses kernel methods, but computational cost is $O(t^2)$. [@keshavarz:sequential_changepoint_ggm:2020] provides an online algorithm for detecting changes in sparse inverse covariance (Gaussian graphical models) but assumes piecewise stationarity between change points.

### 10.7.2 Modern Context: MLOps and Continuous Learning

Production machine learning systems increasingly use **continuous learning** pipelines:

- **Online retraining**: Retrain the model every $N$ hours on recent data (e.g., last 7 days).
- **Shadow models**: Run multiple policy candidates in parallel, monitor their performance, promote the best one.
- **A/B testing infrastructure**: Allocate a small fraction of traffic to experimental policies, measure lift, roll out winners gradually.

Drift detection integrates naturally with this stack:

- **Trigger retraining**: When Page-Hinkley detects drift, enqueue a retraining job.
- **Automatic rollback**: If a newly deployed model triggers drift alarms (performance drops), roll back to the previous checkpoint.
- **Adaptive exploration**: Increase exploration rate ($\epsilon$ in $\epsilon$-greedy, or $\alpha$ in LinUCB) when drift is detected.

**Modern tools:**
- **MLflow / Weights & Biases**: Track drift metrics alongside training loss.
- **Seldon Core / KFServing**: Orchestrate A/B tests and shadow deployments.
- **Evidently AI / Alibi Detect**: Libraries for multivariate drift detection (more sophisticated than our univariate CUSUM/Page-Hinkley).

**Open research directions:**

1. **Meta-learning for drift adaptation** [@nagabandi:learning_adapt:2019]: Train policies that quickly adapt to new regimes using only a few episodes. Techniques like MAML (Model-Agnostic Meta-Learning) show promise.

2. **Causal drift detection**: Instead of detecting correlation shifts, identify when the causal graph changes (e.g., a new competitor affects demand). This requires causal inference tools, not just statistics.

3. **Multi-task robustness**: Train a single policy that performs well across multiple drift scenarios (winter, summer, Black Friday, regular days). This is a **domain generalization** problem.

### 10.7.3 When Guardrails Fail

Even with perfect drift detection and fallback mechanisms, systems can fail. Some failure modes:

**1. The fallback policy is also broken.** If the safe static template relies on a product that goes out of stock, it will fail too. **Solution:** Use multiple fallback layers (e.g., fallback-1 = best template, fallback-2 = second-best, fallback-3 = human-curated ranking).

**2. Users learn to game the system.** If the ranking becomes too stable (low Delta-Rank), adversarial sellers may spam low-quality products to get into the top-10 and stay there. **Solution:** Introduce randomness (exploration) even in fallback mode.

**3. Drift detection is too slow.** For very fast drift (e.g., flash sale starts at exactly 12:00 PM), 30-50 episode delay is too long—the sale might be over. **Solution:** Use external signals (calendar events, marketing campaigns) to pre-emptively adjust policies, not just reactive drift detection.

**4. Hyperparameter sensitivity.** Our experiment works with $\lambda = 30.0$, but if we set $\lambda = 10.0$ (too sensitive), we get false alarms; if $\lambda = 100.0$ (too conservative), we miss the drift. **Solution:** Adaptive thresholds that adjust based on recent noise levels (related to heteroscedastic CUSUM [@liu_li:adaptive_cusum:2013]).

**Production recommendation:** Always have a **human-in-the-loop** for critical decisions. Drift detection should trigger alerts to engineers, not just automatic fallback. Let humans review the data and decide whether to retrain, roll back, or adjust thresholds.

---

## 10.8 Production Checklist

Before deploying drift detection and guardrails in production, ensure the following:

!!! tip "Production Checklist (Chapter 10)"

    **Drift Detection Configuration:**
    - ✅ **Algorithm selection**: Choose CUSUM for symmetric drift, Page-Hinkley for directional (reward drops). Implemented via `DriftDetector` interface in `zoosim/monitoring/drift.py:21-51`.
    - ✅ **Threshold tuning**: Set $\lambda$ based on acceptable false alarm rate. Default $\lambda = 50\sigma$ gives ARL$_0 \approx e^{50} \approx 10^{21}$ episodes (very conservative). For production, tune empirically on historical data.
    - ✅ **Minimum sample count**: Set `min_instances` (default 30) to avoid false alarms during cold start. This is `CUSUMConfig.min_instances` (`drift.py:64`) and `PageHinkleyConfig.min_instances` (`drift.py:142`).
    - ✅ **Drift magnitude**: Set $\delta$ to the smallest mean shift you care about detecting. Typically $\delta = 0.5\sigma$ (half a standard deviation). This is `CUSUMConfig.delta` and `PageHinkleyConfig.delta`.

    **Guardrails Configuration:**
    - ✅ **Stability SLO**: Define acceptable $\Delta\text{-rank}@k$ (e.g., $\leq 0.3$). Monitor via `compute_delta_rank_at_k` (`metrics.py:89`). Log to dashboards; trigger alerts if breached for >5% of queries.
    - ✅ **CM2 floor (hard feasibility)**: Enforce at action selection time (reject/resample infeasible actions; Exercise 10.3). The §10.5 reference monitor layer is intentionally agnostic to CM2 and operates on reward-based drift signals and stability diagnostics.
    - ✅ **Fallback policy**: Ensure `safe_policy` is truly safe (stable, profitable). Test it offline before deploying. Instantiate in `SafetyMonitor` constructor (`guardrails.py:71`).
    - ✅ **Probe frequency**: Set `GuardrailConfig.probe_frequency` based on expected drift recovery time. Too short → thrashing between primary and fallback. Too long → suboptimal for extended periods. Default 100 episodes is reasonable for daily drift patterns.

    **Monitoring Infrastructure:**
    - ✅ **Logging**: Log all metrics (reward, GMV, CVR, Delta-Rank, latency) to time-series database (e.g., Prometheus, InfluxDB). Enable in experiment via `SimulationMetrics` dataclass (`ch10_drift_demo.py:42-49`).
    - ✅ **Alerting**: Configure alerts for: (1) Drift detected, (2) Fallback triggered, (3) Stability SLO breached, (4) CM2 floor breached (if CM2 floors are enforced and logged). Use PagerDuty or similar on-call systems.
    - ✅ **Dashboards**: Visualize drift detector state (`sum_diff`, `min_sum_diff` for Page-Hinkley), policy mode (primary vs. fallback), and business metrics. Use Grafana or similar.
    - ✅ **Reproducibility**: Set seeds for all RNGs (`SimulatorConfig.seed` in `zoosim/core/config.py`, policy seeds, drift detector seeds). Document hyperparameters in config files, not hard-coded.

    **Testing:**
    - ✅ **Unit tests**: Test drift detectors on synthetic data with known change-points. Assert detection delay is within bounds. Example: `tests/ch10/test_drift.py` (to be added).
    - ✅ **Integration tests**: Test `SafetyMonitor` end-to-end with mock policies. Verify fallback triggers correctly, recovery works, and metrics are logged.
    - ✅ **Stress tests**: Simulate rapid drift (change-point every 50 episodes), catastrophic drift (all rewards drop to zero), and no drift (validate false alarm rate).

	    **Deployment:**
	    - ✅ **Shadow mode**: Deploy SafetyMonitor in shadow mode (monitoring only, no fallback) for 1-2 weeks. Tune thresholds based on observed false alarm rate.
	    - ✅ **Gradual rollout**: Enable fallback for 1% of traffic, then 10%, then 50%, then 100%. Monitor business metrics at each stage.
	    - ✅ **Rollback plan**: If drift detection causes catastrophic failures (e.g., fallback policy is broken), have a one-click rollback to disable it.

#### Verification Protocol

Before deploying to production, run:

```bash
# 1. Unit tests for drift detectors and guardrails
uv run pytest tests/ch10 -v

# 2. Integration test: Drift detection end-to-end
uv run python scripts/ch10/ch10_drift_demo.py --drift-step 500 --n-seeds 10

# 3. (Planned) Stability validation: Ensure Delta-Rank SLO holds under normal operation
python scripts/ch10/validate_stability_slo.py --config production_config.yaml

# 4. (Planned) False alarm rate calibration on stationary data
python scripts/ch10/calibrate_threshold.py --data historical_logs.parquet
```

**Expected checks:**
- Drift-related tests in `tests/ch10` pass (and overall test coverage for `zoosim/monitoring/` remains high).
- Drift detection delay is < 50 episodes on the demo scenario (mean over multiple seeds).
- Delta-Rank SLO breaches remain below 5% of queries under normal operation.
- False alarm rate is below the target (e.g., 1%) once calibration scripts are in place.

---

## 10.9 Exercises and Labs

### Exercises

**Exercise 10.1** (Drift Detection Delay) [⭐️ Basic, 30 min]

Suppose rewards are Gaussian $\mathcal{N}(\mu, \sigma^2)$ with $\sigma = 1.0$. At episode $t_0 = 1000$, the mean shifts from $\mu_0 = 5.0$ to $\mu_1 = 4.0$ (a drop of $\Delta = 1.0$).

1. Using [EQ-10.8], compute the theoretical minimum detection delay for a test with false alarm rate $\alpha = 10^{-4}$.
2. Configure a `PageHinkley` detector with appropriate $\delta$ and $\lambda$ to achieve this delay.
3. Simulate 5000 episodes with the shift at $t_0 = 1000$ and measure the actual detection delay. Does it match theory?

**Exercise 10.2** (Stability vs. Performance Tradeoff) [⭐️⭐️ Intermediate, 2 hours]

Modify the experiment `simulate_drift_scenario` to enforce a hard Delta-Rank constraint: reject any action that would cause $\Delta\text{-rank}@10 > 0.3$.

1. Implement this as a post-hoc filter in `SafetyMonitor.select_action` (if proposed action violates constraint, select the previous episode's action instead).
2. Run the experiment and plot the tradeoff between average reward and average Delta-Rank.
3. What is the cost of stability? (i.e., how much reward do we lose by enforcing $\Delta\text{-rank}@10 \leq 0.3$?)

**Exercise 10.3** (CM2 Floor Enforcement) [⭐️⭐️ Intermediate, 2 hours]

Extend `DriftEnvironment` to include per-template margins: `margin = [0.40, 0.30, 0.10, 0.45]` (T0, T1, T2, T3 respectively).

1. Compute CM2 for each action as `CM2 = GMV * margin[action]`.
2. Enforce a CM2 floor of 15% of GMV. If an action would violate this, reject it and resample.
3. Measure how often the constraint is active. Does it prevent the policy from selecting T2 (low margin) even if T2 has high CVR?

**Exercise 10.4** (Multi-Variate Drift Detection) [⭐️⭐️⭐️ Advanced, 6+ hours]

Current implementations detect drift in scalar rewards. Real systems have vector-valued metrics $(r_{\text{GMV}}, r_{\text{CTR}}, r_{\text{CVR}})$.

1. Read about **multivariate CUSUM** [@crosier:multivariate_cusum:1988].
2. Implement a multivariate extension of `PageHinkley` that monitors covariance shifts, not just mean shifts.
3. Test it on a scenario where GMV and CTR drift in opposite directions (e.g., GMV increases but CTR decreases—this happens when average order value rises but fewer users click).

**Exercise 10.5** (Adaptive Learning Rates) [⭐️⭐️⭐️ Advanced + Research, 8+ hours]

Prove or disprove: If we increase the learning rate $\alpha$ by a factor of 2 immediately after drift is detected, the tracking error [EQ-10.7] decreases.

1. Formalize this as a modified gradient ascent update: $\alpha_t = \alpha_0$ if no drift, $\alpha_t = 2\alpha_0$ for 50 episodes after drift detection, then back to $\alpha_0$.
2. Modify REINFORCE (`zoosim/policies/reinforce.py:88`) to implement this adaptive scheme.
3. Run experiments comparing fixed vs. adaptive learning rates. Does adaptation help?

**Hint:** The tracking error [EQ-10.7] has the form
$$
e_\infty \leq \frac{2\delta}{\mu \alpha} + c\alpha,
$$
where $c > 0$ is the implicit constant from the $O(\alpha)$ bound (arising from gradient noise and discretization effects; its exact value depends on the Lipschitz constant $L$ and noise variance). This is a function $e(\alpha)$ with two competing terms: the first decreases with $\alpha$, while the second increases. What is the optimal $\alpha^*$ that minimizes $e(\alpha)$? Differentiate and set to zero: $\frac{de}{d\alpha} = -\frac{2\delta}{\mu \alpha^2} + c = 0$. Solving gives $\alpha^* = \sqrt{2\delta/(\mu c)} \propto \sqrt{\delta}$. Now consider: after drift is detected, the effective drift $\delta$ has increased—how should $\alpha$ respond? (Answer: increase $\alpha$ proportionally to $\sqrt{\delta}$.)

### Labs

**Lab 10.1** (Implement CUSUM from Scratch)

**Goal:** Implement the CUSUM algorithm [ALG-10.1] and validate it against the library implementation.

**Steps:**

1. Write a Python function `two_sided_cusum(data, delta, lambda_threshold)` that processes a stream of rewards and returns the episode index where drift is detected (or `None` if no drift).
2. Test it on synthetic data: 1000 episodes with $\mathcal{N}(0, 1)$ rewards, then 1000 episodes with $\mathcal{N}(1, 1)$ rewards (mean shift of $\Delta = 1$).
3. Compare your implementation's detection delay to `zoosim.monitoring.drift.CUSUM`. They should agree within a few episodes.
4. Experiment with different $(\delta, \lambda)$ pairs. Plot detection delay vs. false alarm rate.

**Expected output:** A plot showing the ARL tradeoff [THM-10.4]: $\text{ARL}_0 \propto \exp(\lambda)$, $\tau_d \propto \lambda / \Delta$.

**Lab 10.2** (Reproduce the Drift Experiment)

**Goal:** Run the full drift scenario from §10.6 and analyze results.

**Steps:**

1. Run `python scripts/ch10/ch10_drift_demo.py`. This takes ~30 seconds on a modern laptop.
2. Inspect the generated plot `ch10_drift_demo.png`. Verify:
   - Fallback is triggered around episode 1540.
   - Recovery occurs around episode 1640.
   - Reward stabilizes post-recovery.
3. Modify the drift step to $t_0 = 500$ (early drift) and re-run. How does this affect recovery time?
4. Modify the drift magnitude (change Phase 2 CVRs to `[0.10, 0.05, 0.06, 0.12]` instead of `[0.05, 0.02, 0.08, 0.15]`—smaller shift). Does Page-Hinkley still detect it?

**Expected output:** Plots for multiple scenarios, demonstrating how drift timing and magnitude affect detection/recovery.

**Lab 10.3** (Integrate with a Real RL Policy)

**Goal:** Replace the static fallback policy with a more sophisticated baseline (e.g., Thompson Sampling or Q-learning).

**Steps:**

1. Modify `simulate_drift_scenario` to use `ThompsonSampling` (from `zoosim/policies/thompson_sampling.py`) as the fallback instead of `StaticPolicy`.
2. Run the experiment. How does the fallback performance compare? (Thompson Sampling should perform better than static T0.)
3. Implement a "Best Known Checkpoint" fallback: save LinUCB's parameters every 100 episodes, and when drift is detected, reload the checkpoint from before the drift.
4. Compare the three fallback strategies: Static, Thompson Sampling, Best Known Checkpoint. Which recovers fastest?

**Expected output:** A comparison table:

| Fallback Strategy | Detection Delay | Fallback Reward | Recovery Time |
|-------------------|-----------------|-----------------|---------------|
| Static T0         | 35 eps          | 0.5             | 140 eps       |
| Thompson Sampling | 35 eps          | 0.7             | 110 eps       |
| Best Checkpoint   | 35 eps          | 0.9             | 80 eps        |

**Lab 10.4** (Production Dashboard Prototype)

**Goal:** Build a Plotly Dash dashboard that monitors drift and guardrails in real-time.

**Steps:**

1. Install Plotly Dash: `pip install dash`.
2. Create a dashboard with 4 panels (same as Figure 10.1): Reward, Policy Selection, Stability, Latency.
3. Add live updates: every 1 second, append a new episode's data to the plots.
4. Add alert indicators: if drift is detected or Delta-Rank exceeds 0.4, display a red banner.
5. Deploy locally at `http://localhost:8050` and demo to your team.

**Expected output:** A live dashboard showing the drift scenario unfolding in real-time, with visual alerts when guardrails trigger.

**Code starting point:**

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='live-reward'),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@app.callback(Output('live-reward', 'figure'), Input('interval-component', 'n_intervals'))
def update_graph(n):
    # Fetch latest metrics from SafetyMonitor
    # Return plotly figure
    pass

app.run_server(debug=True)
```

---

## Conclusion

We have equipped our RL system with the tools to survive non-stationarity:

- **Mathematical rigor**: Lyapunov stability, change-point detection, ARL bounds.
- **Production algorithms**: CUSUM and Page-Hinkley with convergence guarantees.
- **Safety infrastructure**: SafetyMonitor orchestrating drift detection, fallback, and recovery.
- **Empirical validation**: Experiments demonstrating 30-episode detection delay, automatic recovery, and graceful degradation.

But this is not a solved problem. The gaps between theory and practice remain wide, especially for:

- **High-dimensional multivariate metrics** (theory assumes univariate rewards)
- **Catastrophic drift** (entire state space changes)
- **Calibration without hindsight** (setting $\lambda$ before deployment)
- **Structured non-stationarity** (separating predictable cycles from true drift)

**The honest truth:** Production RL systems still require substantial human oversight. Drift detection is a tool, not a solution. Engineers must monitor dashboards, tune thresholds, and make judgment calls. The goal is not full automation, but **failing gracefully**—when the world changes, the system should degrade performance smoothly, not catastrophically.

In Chapter 11, we extend this to multi-episode retention dynamics, where user satisfaction compounds over sessions. Drift detection becomes even more critical, as a single bad experience can cause permanent user churn.

---

## References

For a comprehensive bibliography, see `references.bib`. Key papers for this chapter:

- **Drift detection theory**: [@page:continuous_inspection:1954], [@lorden:procedures_change_point:1971], [@basseville_nikiforov:detection_abrupt:1993]
- **Non-stationary bandits**: [@besbes_szepesvari:non_stationary_bandits:2019], [@garivier_moulines:change_point:2011]
- **Production ML systems**: [@sculley:hidden_technical_debt:2015], [@breck:ml_test_score:2017]
- **Control theory foundations**: [@fazel_ge:global_convergence:2018], [@nagabandi:learning_adapt:2019]

---

!!! note "Code ↔ Summary (Complete Implementation Map)"
    **This chapter's implementations:**

    - **Drift detectors**: `zoosim/monitoring/drift.py:21-195`
      - `DriftDetector` (abstract base): lines 21-51
      - `CUSUM`: lines 69-128
      - `PageHinkley`: lines 148-195

    - **Guardrails**: `zoosim/monitoring/guardrails.py:1-178`
      - `SafetyMonitor`: lines 50-178
      - `GuardrailConfig`: lines 32-47
      - `PolicyProtocol`: lines 22-28

    - **Metrics**: `zoosim/monitoring/metrics.py:1-134`
      - `compute_delta_rank_at_k`: lines 89-118
      - `compute_gmv`, `compute_cm2`: lines 53-86
      - `RankingMetrics` dataclass: lines 121-134

    - **Experiments**: `scripts/ch10/ch10_drift_demo.py:1-323`
      - `DriftEnvironment`: lines 64-129
      - `simulate_drift_scenario`: lines 172-264
      - Visualization: lines 267-320

    **Configuration:**
    - Drift detector parameters: `drift.py:55-67` (CUSUM), `drift.py:132-145` (Page-Hinkley)
    - Guardrail parameters: `guardrails.py:32-47`
    - All configurable via dataclasses (no hard-coded magic numbers)

    **Cross-references:**
    - [THM-3.5.2] — Bellman Optimality Equation (Chapter 3)
    - [EQ-6.8] — LinUCB ridge regression (Chapter 6)
    - [THM-8.2] — Policy Gradient Theorem (Chapter 8)
    - [ALG-10.1] — CUSUM algorithm (this chapter)
    - [ALG-10.2] — Page-Hinkley test (this chapter)
    - [EQ-10.9] — Delta-Rank@k definition (this chapter)
