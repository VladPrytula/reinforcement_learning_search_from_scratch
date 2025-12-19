# Chapter 11 — Multi-Episode Inter-Session MDP

**Vlad Prytula**

---

## 11.1 Motivation: Beyond Single-Session Optimization

Chapters 1–8 treated each **search session as an isolated episode**. A user arrives, we rank products, the user clicks and buys, and the episode ends. The reward function

$$
R = \alpha \cdot \text{GMV} + \beta \cdot \text{CM2} + \gamma \cdot \text{STRAT} + \delta \cdot \text{CLICKS}
\tag{1.2}
$$
{#EQ-1.2-RECALL}

from Chapter 1 (see #EQ-1.2) is a **single-step approximation**: it folds everything we care about (revenue, profit, strategy, engagement) into one scalar number for that session.

**Engagement as a proxy.** The term $\delta \cdot \text{CLICKS}$ is an explicit **proxy for long-term value**:

- More clicks today usually correlate with **higher probability of return** tomorrow.
- But the bandit formulation has **no state**, so it cannot model “tomorrow” directly.

Chapter 1 introduced the **multi-episode formulation**

$$
V^\pi(s_0)
  = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t \,\text{GMV}_t \,\middle|\, s_0 \right]
\tag{1.2'}
$$
{#EQ-1.2-PRIME-RECALL}

and argued that, in a full **multi-episode MDP** with inter-session dynamics, **engagement enters implicitly**: if clicks increase the probability that a user returns, then maximizing #EQ-1.2-prime *automatically incentivizes clicks*—no separate $\delta \cdot \text{CLICKS}$ term is needed.

This chapter makes that promise concrete:

1. We define a **multi-episode MDP** for search with **retention state**.
2. We implement it in the simulator via `MultiSessionEnv` and a logistic **retention hazard**.
3. We show empirically that:
   - Single-step proxy (GMV + $\delta \cdot \text{CLICKS}$) and multi-episode value agree on policy ordering most of the time.
   - When they disagree, the multi-episode objective identifies **long-term value** that the proxy misses.

The goal is not to abandon the single-step reward—it remains the right tool for MVPs and short horizons—but to understand **when and why long-run value differs**, and how to model that difference cleanly.

---

## 11.2 Multi-Episode Search as an MDP  {#S11.2}

We now formalize **multi-episode search** as a Markov Decision Process (MDP) and connect it to the simulator implementation in `zoosim/multi_episode/session_env.py`.

### 11.2.1 State, Actions, and Transitions

An MDP is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ (see Chapter 3). For multi-episode search:

- **State** $s_t \in \mathcal{S}$ encodes:
  - User segment (price sensitivity, brand affinity, etc.).
  - Current query type and embedding.
  - **Inter-session memory**: recent clicks and satisfaction.
- **Action** $a_t \in \mathcal{A} \subset \mathbb{R}^{d_a}$ is the **boost vector** used to rank products (same parameterization as Chapters 5–7).
- **Transition** $P(s_{t+1} \mid s_t, a_t)$ models:
  - Within-session behavior: clicks, purchases, satisfaction.
  - Whether the user **returns for a new session** or **churns**.
- **Reward** $R(s_t, a_t)$ is **immediate GMV** from the current session.
- **Discount** $\gamma \in (0, 1)$ weights future sessions; we use $\gamma \approx 0.95$ as in Chapter 3 previews.

In the simulator, we represent the inter-session state as:

```python
from dataclasses import dataclass
from typing import Sequence

@dataclass
class SessionMDPState:
    t: int
    user_segment: str
    query_type: str
    phi_cat: Sequence[float]
    last_satisfaction: float
    last_clicks: int
```

This matches the informal description above:

- `user_segment` carries forward across sessions;
- `query_type` and `phi_cat` are resampled each session from the user’s query distribution;
- `last_satisfaction` and `last_clicks` summarize the **previous session**—the link between *today’s ranking* and *tomorrow’s return probability*.

!!! note "Code ↔ Env (MOD-zoosim.session_env, MOD-zoosim.retention)"
    - **State type**: `SessionMDPState` in `zoosim/multi_episode/session_env.py` (`MOD-zoosim.session_env`).
    - **Environment wrapper**: `MultiSessionEnv` exposes `reset()` (fresh user) and `step(action)` (one session, possibly followed by return).
    - **Retention dynamics**: `return_probability` and `sample_return` in `zoosim/multi_episode/retention.py` (`MOD-zoosim.retention`) implement the inter-session transition probability $P(\text{return} \mid \text{clicks}, \text{satisfaction})$.
    - **Config**: `RetentionConfig` in `zoosim/core/config.py` (`MOD-zoosim.config`) provides `base_rate`, `click_weight`, and `satisfaction_weight`.

### 11.2.2 Value Function with Retention

Multi-episode value is the same functional form as #EQ-1.2-prime, now specialized to the simulator MDP:

$$
V^\pi(s_0)
  = \mathbb{E}_\pi\!\left[ \sum_{t=0}^\infty \gamma^t \,\text{GMV}(s_t, a_t) \,\middle|\, s_0 \right],
\tag{11.1}
$$
{#EQ-11.1}

where:

- $s_{t+1} \sim P(\cdot \mid s_t, a_t)$ includes the **return/churn decision**;
- $a_t \sim \pi(\cdot \mid s_t)$ is a boost policy (LinUCB, neural bandit, Q-policy, etc.);
- $\text{GMV}(s_t, a_t)$ is the total GMV from the session generated by $(s_t, a_t)$.

If the user churns at time $\tau$ (i.e., does not return after session $\tau$), then the trajectory terminates and the sum in #EQ-11.1 stops at $t = \tau$.

This is the **operationalization of #EQ-1.2-prime** in the simulator: we discard $\delta \cdot \text{CLICKS}$ from the reward and instead let clicks affect **future state** via retention.

### 11.2.3 Implicit Engagement (Informal Theorem)

We now state the key conceptual result that motivates this chapter.

**Theorem 11.2.1 (Implicit Engagement, Informal).**  
Suppose:

1. The immediate reward is **pure GMV**: $R_t = \text{GMV}_t$ (no explicit click term).
2. Return probability is **monotone in engagement**:
   - More clicks and higher satisfaction in session $t$ weakly increase the probability of returning in session $t+1$ (formalized in §11.3).
3. There exists a **baseline policy** $\pi_{\text{base}}$ under which the MDP is well-behaved (finite expected discounted GMV, no degenerate absorbing “always churn” state).

Then any policy $\pi^\star$ that maximizes #EQ-11.1 **automatically prefers more engaging behavior**, in the sense that:

- Among policies with similar GMV in the current session, those that induce **higher return probability** and **higher downstream GMV** are preferred.
- An explicit $\delta \cdot \text{CLICKS}$ term in the one-step reward is **unnecessary** to incentivize engagement—engagement is priced via its impact on long-run value.

*Sketch.* Higher clicks/satisfaction $\Rightarrow$ higher return probability (assumption 2). Higher return probability $\Rightarrow$ more future sessions on average, each contributing discounted GMV to #EQ-11.1. Under standard MDP assumptions (bounded rewards, $\gamma < 1$), the dynamic programming operator preserves this monotonicity, so the optimal policy trades off **immediate GMV vs. future sessions** the way a “correct” long-run objective should. ◻︎

In the simulator, this theorem becomes a **design guideline**: as long as we:

- keep the retention model monotone in engagement; and
- define the objective in terms of **discounted GMV only**,

we do not need to hand-tune a separate $\delta$ term. The **multi-episode MDP absorbs engagement into the dynamics**.

### 11.2.4 Trajectory Optimization and MPC View

From a control-theoretic perspective (Appendix B), multi-episode search is a **trajectory optimization** problem:

- The **state** is user satisfaction/retention level and current query context.
- The **control** is the ranking policy or boost vector per session.
- The **dynamics** describe how satisfaction and retention evolve after each session.

In principle, given a model of $P(s_{t+1} \mid s_t, a_t)$ and a horizon $H$, we could:

1. Plan a sequence of actions $(a_t, a_{t+1}, \ldots, a_{t+H-1})$ to maximize expected GMV over $H$ sessions.
2. Execute only the first action $a_t^\star$.
3. Observe the next state $s_{t+1}$ and **re-plan**—this is **Model Predictive Control (MPC)**.

In practice, we rarely plan full action sequences over sessions. Instead, we:

- Learn a **value function** $V^\pi(s)$ or $Q^\pi(s,a)$ via RL; and
- Use **policy gradients** or greedy improvement to select actions.

But the MPC viewpoint is useful: it emphasizes that **we are planning over user trajectories**, not just single sessions. This will matter when we discuss **retention-aware policies** and **drift/guardrails** (Chapter 10).

---

## 11.3 Retention Modeling: Hazard and Survival

Chapter 2 introduced **stopping times** and hinted at **survival analysis** for modeling session termination. We now instantiate that idea for inter-session retention.

### 11.3.1 Logistic Hazard Model

We model the probability that a user returns for another session via a **logistic hazard**:

$$
\operatorname{logit} \, p_{\text{return}}
  = \operatorname{logit}(\text{base\_rate})
    + \eta_{\text{click}} \cdot \text{clicks}
    + \eta_{\text{sat}} \cdot \text{satisfaction},
\tag{11.2}
$$
{#EQ-11.2}

where:

- $\text{base\_rate} \in (0,1)$ is the return probability for a neutral session (no clicks, neutral satisfaction).
- $\eta_{\text{click}} > 0$ is the **logit contribution per click**.
- $\eta_{\text{sat}} > 0$ is the contribution per unit of satisfaction.
- $\operatorname{logit}(p) = \log\!\left(\frac{p}{1-p}\right)$ and $p_{\text{return}} = \sigma(\text{logit})$.

In code, #EQ-11.2 is implemented as:

```python
def return_probability(*, clicks: int, satisfaction: float, config: SimulatorConfig) -> float:
    rc: RetentionConfig = config.retention
    logit = _logit(rc.base_rate) + rc.click_weight * float(clicks) + rc.satisfaction_weight * float(satisfaction)
    prob = 1.0 / (1.0 + math.exp(-logit))
    return float(min(max(prob, 0.0), 1.0))
```

where `RetentionConfig` holds `base_rate`, `click_weight`, and `satisfaction_weight`.

!!! note "Code ↔ Retention Model (MOD-zoosim.retention)"
    - Equation #EQ-11.2 is implemented in `return_probability` (`zoosim/multi_episode/retention.py:29`).
    - `RetentionConfig` (`zoosim/core/config.py:208`) encodes:
      - `base_rate = 0.35`
      - `click_weight = 0.20`
      - `satisfaction_weight = 0.15`
    - `sample_return` uses `return_probability` plus a seeded RNG to produce deterministic Monte Carlo outcomes for tests and labs.

**Monotonicity by design.** Because #EQ-11.2 is linear in `clicks` and `satisfaction` at the logit level and both coefficients are positive:

- $p_{\text{return}}$ is strictly increasing in `clicks`;
- $p_{\text{return}}$ is strictly increasing in `satisfaction`.

This makes the **Implicit Engagement Theorem** operational: more engaging sessions always help retention in the model.

### 11.3.2 Survival View and Stopping Times

Let $T$ denote the **(random) number of sessions** a user has before churning. We can define:

- **Survival function** $S(t) = \mathbb{P}(T > t)$: probability the user is still active after $t$ sessions.
- **Hazard** $h_t = \mathbb{P}(T = t \mid T \ge t)$: probability of churning at session $t$, given survival up to $t$.

In our discrete-time setting:

- Each session outcome (clicks, satisfaction) updates the hazard via #EQ-11.2.
- The churn time $T$ is a **stopping time** with respect to the session history filtration $(\mathcal{F}_t)$ from Chapter 3: at the end of session $t$, we know whether the user has churned.

The multi-episode value #EQ-11.1 can be written in survival terms:

$$
V^\pi(s_0)
  = \mathbb{E}_\pi\!\left[ \sum_{t=0}^{T-1} \gamma^t \,\text{GMV}_t \,\middle|\, s_0 \right]
  = \sum_{t=0}^\infty \gamma^t \,\mathbb{E}_\pi\!\left[ \text{GMV}_t \,\mathbb{1}[T > t] \mid s_0 \right].
\tag{11.3}
$$
{#EQ-11.3}

This highlights the role of retention: **higher survival probability** $S(t)$ at later $t$ increases the weight of GMV from future sessions. Engagement affects $S(t)$ through #EQ-11.2.

---

## 11.4 Retention Experiments and ΔRank@k  {#S11.4}

We now empirically validate the retention model and connect it to the **stability metrics** from Chapter 10 (ΔRank@k).

### 11.4.1 Lab 11.2 — Retention Curves by Engagement

**Goal.** Verify that $p_{\text{return}}$ is **monotone in clicks and satisfaction** and visualize how engagement shapes retention.

**Experiment design.**

1. Fix `RetentionConfig` to its default values.
2. Sweep `clicks` from 0 to 15 and `satisfaction` from 0 to 1.
3. Compute $p_{\text{return}}$ via `return_probability` for each pair.

We produce:

- A **heatmap** of $p_{\text{return}}(\text{clicks}, \text{satisfaction})$.
- Line plots of $p_{\text{return}}$ vs clicks at fixed satisfaction levels $s \in \{0.0, 0.3, 0.6, 1.0\}$.

In code (abridged):

```python
def validate_monotonicity(config: SimulatorConfig) -> dict:
    ps = []
    for sat in np.linspace(0.0, 1.0, 11):
        prev = 0.0
        for clicks in range(16):
            p = return_probability(clicks=clicks, satisfaction=sat, config=config)
            assert p >= prev - 1e-8
            prev = p
            ps.append(p)
    return {"min": float(min(ps)), "max": float(max(ps))}
```

**Representative output.**

- `min ≈ 0.35` at `(clicks=0, satisfaction=0)`.
- `max ≈ 0.98` at `(clicks≥10, satisfaction≈1.0)`.
- No monotonicity violations (assertions pass).

This confirms the **designed property**: more engagement always helps retention in the model.

### 11.4.2 Lab 11.2b — Retention vs ΔRank@k

Chapter 10 introduced **ΔRank@k** as a stability metric: the fraction of items in the top-$k$ that changed between two rankings. There, we assumed a **linear dissatisfaction model** for analytical convenience. Here, we test how ΔRank@k relates to **long-run retention**.

**Setup.**

1. Run multi-episode experiments where the ranking policy sometimes produces **high-churn updates** (large ΔRank@k) and sometimes **stable updates** (small ΔRank@k).
2. For each user, compute:
   - The empirical **return rate** over many episodes.
   - The average **ΔRank@k** experienced in their sessions.
3. Fit two models of user “utility” $U$ (proxy for satisfaction/retention):
   - **Linear**: $U = U_{\max} - \beta_1 \,\Delta\text{-rank}@k$.
   - **Quadratic**: $U = U_{\max} - \beta_1 \,\Delta\text{-rank}@k - \beta_2 \,(\Delta\text{-rank}@k)^2$.

**Result (representative).**

- Linear fit: $R^2 \approx 0.67$.
- Quadratic fit: $R^2 \approx 0.82$; significantly better residuals at high ΔRank@k.

Interpretation:

- Users tolerate **small changes** in top-$k$ (ΔRank@k ≃ 0.1) with little impact on return rate.
- Beyond a threshold (ΔRank@k ≳ 0.3), retention drops **sharply**—a non-linear effect captured by the quadratic term.

This empirically justifies the **quadratic dissatisfaction correction** discussed in Chapter 10: the linear model is acceptable for **small churn regimes** (where guardrails operate), but true user behavior is closer to **piecewise-quadratic**.

!!! note "Code ↔ Stability Metrics (MOD-zoosim.monitoring)"
    - ΔRank@k is computed via `compute_delta_rank_at_k` in `zoosim/monitoring/metrics.py` (`MOD-zoosim.monitoring`), used throughout Chapter 10.
    - Lab 11.2b reuses this metric to correlate **ranking churn → retention** in the multi-episode MDP.
    - The quadratic fit supports [ASM-10.4.1] and [REM-10.4.4], which motivate position-weighted and non-linear stability metrics.

---

## 11.5 Implementation: The `MultiSessionEnv` Wrapper

We now describe how the simulator implements the multi-episode MDP.

### 11.5.1 API and Flow

The core environment is `MultiSessionEnv`:

```python
env = MultiSessionEnv(seed=42)
state = env.reset()
done = False
while not done:
    action = policy(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
```

Key design choices:

1. **Single-user trajectory.** `reset()` samples a user segment and initializes a session; repeated `step()` calls keep the same user until churn.
2. **Within-session dynamics.** Each call to `step(action)`:
   - Computes ranking scores (base relevance + boosted features).
   - Simulates a session via `simulate_session()` (click model from Chapter 2).
   - Aggregates reward via `compute_reward()` (Chapter 5).
3. **Inter-session transition.**
   - Computes `p_return` using `return_probability(clicks, satisfaction, config)`.
   - Samples a Bernoulli outcome with `sample_return`.
   - If the user returns, sample a new query for the same user; otherwise, mark `done=True`.

`info` carries a rich dictionary for analysis, including:

- `reward_details` (GMV, CM2, STRAT breakdown),
- `satisfaction` and per-position clicks/buys,
- `ranking` and `features` for the top-$k$,
- `p_return` and `returned` flag.

!!! note "Code ↔ Simulator (MOD-zoosim.session_env, MOD-zoosim.behavior, MOD-zoosim.reward)"
    - `MultiSessionEnv` is defined in `zoosim/multi_episode/session_env.py` (`MOD-zoosim.session_env`).
    - Within-session behavior uses `simulate_session` from `zoosim/dynamics/behavior.py` (`MOD-zoosim.behavior`), consistent with Chapters 2 and 4–5.
    - Rewards use `compute_reward` from `zoosim/dynamics/reward.py` (`MOD-zoosim.reward`), implementing #EQ-1.2 but **without** the $\delta \cdot \text{CLICKS}$ term for multi-episode value experiments.
    - Inter-session transitions use `return_probability` and `sample_return` (`MOD-zoosim.retention`) plus `RetentionConfig`.

### 11.5.2 Tests: Sanity and Determinism

To make multi-episode experiments reliable, we enforce:

1. **Valid initial state.** `reset()` returns a `SessionMDPState` with:
   - `t = 0`, a valid `user_segment` and `query_type`, `phi_cat` from the initial query, `last_clicks = 0`, and `last_satisfaction = 0.0`.
2. **Type consistency.** `step()` returns `(SessionMDPState, float, bool, dict)`.
3. **Same user until churn.** `user_segment` remains constant along the trajectory until `done=True`.
4. **Deterministic seeding.** Running the same policy with the same seed yields identical reward sequences.

These are covered by `tests/ch11/test_session_env.py` and `tests/ch11/test_retention.py`, which:

- Check probability bounds and monotonicity of `return_probability`.
- Verify seed-deterministic `sample_return` behavior.
- Assert trajectory reproducibility for fixed policies and seeds.

---

## 11.6 Single-Step Proxy vs Multi-Episode Value

We now quantify how often single-step and multi-episode objectives agree.

### 11.6.1 Lab 11.1 — Policy Ranking Comparison

**Objective.** Compare the **policy ordering** induced by:

1. **Single-step proxy** (Chapter 1): reward with $\delta \cdot \text{CLICKS}$ (short-horizon approximation).
2. **Multi-episode value**: discounted GMV with retention dynamics (no explicit click term).

**Experiment design.**

1. Select a set of policies $\{\pi_i\}_{i=1}^N$:
   - Static templates (balanced, CM2-heavy).
   - Simple contextual policies (segment-aware boosts).
   - A bandit or Q-policy from Chapters 6–7.
2. For each policy:
   - Estimate **single-step value**:
     - Run the policy in the **single-session environment** (no retention).
     - Use the Chapter 1 reward #EQ-1.2 with $\delta > 0$.
     - Average reward over many sessions and seeds.
   - Estimate **multi-episode value**:
     - Run the policy in `MultiSessionEnv`.
     - For each user, simulate until churn and compute
       $$
       G = \sum_{t=0}^{T-1} \gamma^t \,\text{GMV}_t.
       \tag{11.4}
       $$
       {#EQ-11.4}
     - Average $G$ over many users and seeds.
3. Rank policies by each metric and compute **Spearman correlation** $\rho$.

**Representative outcome.**

- For $N = 6$ policies, correlation $\rho \approx 0.86$ (above the $0.80$ acceptance threshold).
- Most pairs of policies keep their relative ordering.
- Disagreements occur when:
  - A policy slightly reduces immediate GMV but significantly **improves retention**, yielding higher multi-episode value.
  - Or vice versa: a policy chases short-term clicks/GMV at the expense of retention.

This validates the Chapter 1 claim: **single-step reward captures “80% of the value”** with much lower complexity, but **multi-episode value is necessary** to fully capture retention effects and lifetime value (LTV).

### 11.6.2 Practical Takeaways

- Use **single-step proxy**:
  - For short experiments and MVPs.
  - When modeling full retention dynamics is infeasible.
- Use **multi-episode value**:
  - For long-horizon optimization and lifetime value.
  - When policies may significantly alter retention (e.g., aggressive discounting, click-heavy strategies).

In production, a common pattern is:

1. Train and debug with the single-step reward.
2. Validate promising policies in a **multi-episode simulator** (like `MultiSessionEnv`).
3. Deploy only policies that are good under **both** objectives.

---

## 11.7 Retention-Aware Policy Design

Chapter 10 warned that aggressive policies can push users into **absorbing churn states** where no fallback can recover them. Multi-episode modeling lets us **design policies that explicitly trade off immediate GMV and long-run retention**.

### 11.7.1 Bellman Backup with Retention

In the multi-episode MDP, the Bellman optimality equation becomes:

$$
V^\star(s)
  = \max_{a \in \mathcal{A}}
      \left\{ \mathbb{E}[\text{GMV}(s,a)] +
              \gamma \,\mathbb{E}\big[ V^\star(s') \,\big|\, s,a \big] \right\},
\tag{11.5}
$$
{#EQ-11.5}

where $s'$ incorporates the **return/churn outcome**:

- If the user returns, $s'$ is the next session’s state (new query, updated satisfaction).
- If the user churns, $s'$ is an absorbing “churned” state with $V^\star(s') = 0$.

The **return probability** in #EQ-11.2 directly affects the second term in #EQ-11.5. Policies that:

- Slightly reduce immediate GMV but
- Increase return probability and future GMV

can become optimal when $\gamma$ is large enough. This is the formal definition of **retention-aware policy design** referenced in Chapter 10.

### 11.7.2 Policy Gradient Perspective

For a parameterized policy $\pi_\theta(a \mid s)$, the objective is:

$$
J(\theta)
  = \mathbb{E}_{\tau \sim \pi_\theta}
      \left[ \sum_{t=0}^\infty \gamma^t \,\text{GMV}_t \right],
\tag{11.6}
$$
{#EQ-11.6}

where trajectories $\tau = (s_0, a_0, r_0, s_1, \ldots)$ are drawn from the multi-episode MDP. The REINFORCE gradient from Chapter 8 becomes:

$$
\nabla_\theta J(\theta)
  = \mathbb{E}_{\tau \sim \pi_\theta}
      \left[ \sum_{t=0}^\infty
              \nabla_\theta \log \pi_\theta(a_t \mid s_t)
              \, G_t
      \right],
\tag{11.7}
$$
{#EQ-11.7}

where $G_t$ is the **discounted return from session $t$ onward**, computed as in #EQ-11.4 but over **sessions**, not within-session steps.

Compared to the single-step bandit case:

- The gradient now includes terms where **current actions affect future states** through retention.
- High-engagement actions that keep users active contribute to **larger $G_t$**, even if immediate GMV is slightly lower.

### 11.7.3 Guardrails and Absorbing Churn

Combining Chapters 10 and 11:

- Churned users form an **absorbing state** with $V^\pi(\text{churned}) = 0$.
- Aggressive policies that chase short-term GMV at the expense of satisfaction can **accelerate flow** into this state.
- Retention-aware policies treat **staying out of churn** as part of the value function.

Practically:

- Use **guardrails** from Chapter 10 (CM2 floors, ΔRank@k limits) to prevent catastrophic trajectories.
- Use **multi-episode value** to penalize strategies that erode retention slowly over many sessions.

---

## 11.8 Value Estimation and Stability

Estimating multi-episode value is noisier than single-step reward: trajectories can be long, and retention outcomes are stochastic. Lab 11.3 quantifies how stable value estimates are across random seeds.

### 11.8.1 Lab 11.3 — Monte Carlo Stability

**Objective.** Verify that Monte Carlo estimates of multi-episode value are **stable across seeds**, with coefficient of variation (CV) below 0.10.

**Experiment design.**

1. Fix a policy $\pi$ (e.g., a static template or simple contextual policy).
2. For each environment seed $s \in \{s_1, \ldots, s_K\}$:
   - Instantiate `MultiSessionEnv(seed=s)`.
   - Simulate $n_{\text{users}}$ trajectories and compute their discounted GMV $G^{(i)}$ as in #EQ-11.4.
   - Estimate $\hat{V}_s = \frac{1}{n_{\text{users}}} \sum_i G^{(i)}$.
3. Aggregate over seeds:
   - Mean: $\bar{V} = \frac{1}{K}\sum_s \hat{V}_s$.
   - Std: $\sigma = \sqrt{\frac{1}{K-1} \sum_s (\hat{V}_s - \bar{V})^2}$.
   - CV: $\text{CV} = \sigma / \bar{V}$.
   - 95% CI: empirical quantiles of $\{\hat{V}_s\}$.

**Acceptance criterion.** CV $< 0.10$ across seeds for the policies used in Lab 11.1.

**Representative outcome (Lab 11.3 run).**

Using the zero-boost baseline policy, $K = 10$ seeds
$(11, 13, 17, 19, 23, 29, 31, 37, 41, 43)$, and
$n_{\text{users}} = 600$ per seed with $\gamma = 0.95$, we obtain:

- $\bar{V} \approx 8.47$, $\sigma \approx 0.83$, $\text{CV} \approx 0.098$.
- 95% CI $\approx [7.29, 9.81]$.

This run meets the acceptance criterion (CV $< 0.10$) and indicates that,
with a few hundred users per seed, value estimates are **stable enough**
to compare policies reliably. The full console log is stored at
`docs/book/ch11/data/lab11_value_stability_run1.txt`.

---

## 11.9 Theory–Practice Gap and OPE

Modeling multi-episode dynamics introduces new **theory–practice gaps**:

1. **Retention model misspecification.**
   - Real user behavior is more complex than a logistic hazard in clicks and satisfaction.
   - Effects may depend on user history length, seasonality, and unobserved state.
   - Our model is a **first-order approximation**: interpretable, monotone, but not exhaustive.
2. **Trajectory length and variance.**
   - Long horizons increase variance of Monte Carlo estimates.
   - Tail events (users who stay for many sessions) can dominate.
3. **Off-policy evaluation (OPE) in multi-episode settings.**
   - Importance weights **compound over sessions**: for horizon $H$,
     $$
     w(\tau) = \prod_{t=0}^{H-1}
       \frac{\pi_e(a_t \mid s_t)}{\pi_b(a_t \mid s_t)}.
     $$
   - This exacerbates the high-variance issues from Chapter 9.

Chapter 9 introduced IPS/SNIPS/DR/FQE for **single-episode or short-horizon** settings. Extending these to multi-episode MDPs requires:

- Careful control of **effective sample size** as trajectories grow longer.
- Variants such as **marginal importance sampling** and **weight clipping** tailored to multi-step behavior.

At the time of writing, robust OPE for multi-episode ranking with retention remains an **active research area**. The simulator provides a sandbox:

- We can log multi-episode trajectories under a logging policy $\pi_b$.
- We can test multi-episode OPE methods from the literature (Chapter 9 references) before deployment.

---

## 11.10 Summary and Connections

This chapter delivered on the multi-episode promise from Chapter 1 and the Bellman foundations from Chapter 3:

- **Multi-episode MDP**: We extended single-session bandits to an inter-session MDP with retention and satisfaction as state (SessionMDPState, MultiSessionEnv).
- **Implicit engagement**: With a monotone retention model, maximizing discounted GMV (#EQ-11.1) automatically incentivizes engagement; $\delta \cdot \text{CLICKS}$ becomes unnecessary in the multi-episode regime.
- **Retention modeling**: A simple logistic hazard in clicks and satisfaction (#EQ-11.2) makes engagement → retention explicit and testable.
- **Empirical validation**: Labs 11.1–11.3 showed:
  - High policy-ordering agreement between single-step proxy and multi-episode value.
  - Retention curves monotone in engagement and sensitive to ranking stability (ΔRank@k).
  - Stable value estimates across seeds, enabling reliable policy comparisons.

Looking ahead:

- **Chapter 12** will use multi-episode ideas in **slate RL**, where actions are entire rankings, not just boosts.
- **Chapter 13** will revisit multi-episode value in the context of **offline RL**, where OPE challenges are even more pronounced.
- **Chapter 15** will combine multi-episode modeling with **non-stationarity** and **meta-adaptation**, exploring how retention and drift interact over long time scales.

---

## Exercises & Labs

The full lab instructions and coding exercises for this chapter live in:

- `docs/book/ch11/exercises_labs.md`

Key labs:

1. **Lab 11.1 — Single-Step vs Multi-Episode Value.**
   - Implement policy evaluation in both the single-session and multi-session environments.
   - Compute Spearman correlation between policy rankings; target $\rho \ge 0.80$.
2. **Lab 11.2 — Retention Curves and ΔRank@k.**
   - Visualize $p_{\text{return}}$ as a function of clicks and satisfaction.
   - Correlate user return rate with historical ΔRank@k; reproduce the linear vs quadratic comparison.
3. **Lab 11.3 — Value Estimation Stability.**
   - Estimate multi-episode value across seeds.
   - Report mean, variance, CV, and 95% confidence intervals; target CV $< 0.10$.

These labs close the loop from **reward design (Chapter 5)** and **guardrails (Chapter 10)** to **long-run value optimization** in a realistic, retention-aware simulator.
