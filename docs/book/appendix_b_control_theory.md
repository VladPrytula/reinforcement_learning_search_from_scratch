# Appendix B --- Control-Theoretic Background (LQR, HJB, and Deep RL)

**Vlad Prytula**

---

## Motivation

Reinforcement learning did not emerge *ex nihilo*. Its mathematical core---value functions, policy optimization, contraction mappings---descends from optimal control theory, a discipline with roots in the calculus of variations (Euler, Lagrange) and mid-20th century engineering (Bellman, Pontryagin, Kalman). Readers with control-theoretic background will find that RL is a natural generalization: classical control assumes *known dynamics* and *quadratic costs*; RL extends to *unknown dynamics* and *arbitrary rewards*.

This appendix bridges these worlds. We develop the Linear Quadratic Regulator (LQR) as the canonical solved case, derive the Hamilton-Jacobi-Bellman (HJB) PDE that governs continuous-time optimal control, and show how both connect to the discrete-time Bellman equation central to RL. The goal is not a comprehensive treatment of control theory---excellent textbooks exist ([@kirk:optimal_control:2004], [@bertsekas:dynamic_programming:2012])---but rather to equip readers to recognize control-theoretic structure in RL algorithms and transfer intuitions between fields.

**If you are unfamiliar with control theory, you may skip this appendix initially.** Return when control-theoretic tools appear in later chapters: Lyapunov analysis for convergence (Chapter 10), robust control for guardrails (Chapter 10), trajectory optimization for multi-episode MDPs (Chapter 11). The key insight to carry forward: **RL generalizes classical control from known dynamics and quadratic costs to unknown dynamics and arbitrary rewards.**

---

## B.1 Linear Quadratic Regulator (LQR) Analogy

### The LQR Problem

Consider a discrete-time linear system with state $x_t \in \mathbb{R}^n$ and control $u_t \in \mathbb{R}^m$:

$$
x_{t+1} = Ax_t + Bu_t + w_t
\tag{B.1}
$$
{#EQ-B.1}

where $A \in \mathbb{R}^{n \times n}$ is the state transition matrix, $B \in \mathbb{R}^{n \times m}$ is the control input matrix, and $w_t$ is zero-mean process noise. We seek a control policy $u_t = \pi(x_t)$ that minimizes the infinite-horizon quadratic cost:

$$
J(\pi) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t (x_t^\top Q x_t + u_t^\top R u_t)\right]
\tag{B.2}
$$
{#EQ-B.2}

where $Q \succeq 0$ penalizes state deviation, $R \succ 0$ penalizes control effort, and $\gamma \in (0,1)$ is the discount factor.

**Theorem B.1.1** (Optimal LQR Solution) {#THM-B.1.1}

Under the hypotheses:
1. $(A, B)$ is controllable (i.e., $\text{rank}[B \; AB \; \cdots \; A^{n-1}B] = n$)
2. $Q \succeq 0$, $R \succ 0$, $\gamma \in (0,1)$

The optimal policy is **linear** in the state:

$$
u^*(x) = -K^* x
\tag{B.3}
$$
{#EQ-B.3}

where the gain matrix $K^* = (R + \gamma B^\top P B)^{-1} \gamma B^\top P A$ and $P \succeq 0$ is the unique positive semidefinite solution to the **Discrete Algebraic Riccati Equation (DARE)**:

$$
P = Q + \gamma A^\top P A - \gamma^2 A^\top P B (R + \gamma B^\top P B)^{-1} B^\top P A
\tag{B.4}
$$
{#EQ-B.4}

*Proof sketch.* The Bellman equation for the value function $V(x) = x^\top P x$ (quadratic Ansatz) yields the DARE upon minimizing over $u$ and matching coefficients. Controllability ensures the DARE has a unique stabilizing solution. The full proof appears in [@bertsekas:dynamic_programming:2012, Chapter 4]. $\square$

### Connection to Search Ranking

We now draw the analogy to our search ranking problem from Chapter 1:

| Control Theory | Search Ranking (Chapter 1) |
|----------------|---------------------------|
| State $x_t$ | Context $(u, q)$ (user, query) |
| Control $u_t$ | Boost action $a \in \mathcal{A}$ |
| Cost $c(x, u)$ | Negative reward $-R(x, a)$ |
| Dynamics $x_{t+1} = Ax_t + Bu_t$ | **None** (single-step bandit) |
| Linear policy $u = -Kx$ | Linear policy $a = Wx$ |

The key difference: LQR has *known dynamics* [EQ-B.1] and *quadratic costs* [EQ-B.2], yielding a closed-form linear solution. Our search problem has *unknown dynamics* (or none, in the bandit case) and *non-quadratic rewards* (clicks are nonlinear, purchases are discrete)---hence we require **nonlinear function approximation** (neural networks) and **learning from data**.

> **Remark B.1.1** (When LQR-like structure appears). In some RL settings, approximate linearity holds locally. Near an equilibrium, many systems behave as $x_{t+1} \approx Ax_t + Bu_t$ (first-order Taylor expansion). This motivates **iLQR** (iterative LQR), which linearizes around a nominal trajectory and solves a sequence of LQR problems. We encounter this in trajectory optimization (Chapter 11) when planning multi-session user engagement.

> **Remark B.1.2** (From Riccati to Policy Gradient). The LQR solution [EQ-B.3] can be recovered by **policy gradient descent** on the space of linear policies $u = -Kx$. The gradient $\nabla_K J(K)$ has a closed form involving the state covariance $\Sigma_K$ under policy $K$ (see [@fazel:global_convergence:2018] for the convergence analysis). This observation---that policy gradient recovers the optimal gain---is the foundation of **DDPG** ([@lillicrap:ddpg:2016]) and **TD3** ([@fujimoto:td3:2018]): replace linear $Kx$ with neural $\pi_\theta(x)$, estimate $\nabla_\theta J$ via a learned critic, and descend.

---

## B.2 Hamilton-Jacobi-Bellman (HJB) Connection

### Continuous-Time Optimal Control

Consider continuous-time dynamics $\dot{x} = f(x, u)$ with running reward $r(x, u)$ and discount rate $\rho > 0$. The infinite-horizon value function $V(x)$ satisfies the **Hamilton-Jacobi-Bellman PDE**:

$$
\rho V(x) = \max_u \left\{r(x, u) + \nabla V(x)^\top f(x, u)\right\}
\tag{B.5}
$$
{#EQ-B.5}

where $\nabla V(x) \in \mathbb{R}^n$ is the gradient of $V$ and the maximum is over admissible controls $u \in \mathcal{U}$. For finite-horizon problems with terminal time $T$, the time-dependent form is:

$$
-\frac{\partial V}{\partial t}(x,t) = \max_u \left\{r(x,u) + \nabla_x V(x,t)^\top f(x,u)\right\}
$$

with boundary condition $V(x, T) = g(x)$ (terminal cost).

### From HJB to Bellman

The discrete-time Bellman equation we use throughout this book:

$$
V(x) = \max_a \left\{R(x, a) + \gamma \mathbb{E}_{x'}[V(x')]\right\}
\tag{B.6}
$$
{#EQ-B.6}

arises as a **discretization** of HJB. With time step $\Delta t$:
- The discount factor relates to the continuous rate: $\gamma = e^{-\rho \Delta t}$
- The transition approximates the dynamics: $x' \approx x + f(x, u)\Delta t + \sqrt{\Delta t}\,\sigma(x, u)\,\xi$ where $\xi \sim \mathcal{N}(0, I)$
- The reward is integrated: $R(x, a) \approx r(x, a) \Delta t$

For our **single-step bandit problem** (no dynamics, $\gamma = 0$), the Bellman equation collapses to:

$$
V(x) = \max_a Q(x, a)
\tag{B.7}
$$
{#EQ-B.7}

which is exactly [EQ-1.9] from Chapter 1. The bandit is the degenerate case of control with zero dynamics and no continuation value.

> **Remark B.2.1** (Viscosity solutions). For general nonlinear dynamics $f(x, u)$, the HJB PDE [EQ-B.5] may not admit classical $C^2$ solutions. The gradient $\nabla V$ may fail to exist at points where the optimal control switches. The correct mathematical framework is **viscosity solutions** ([@crandall:viscosity:1992]), which extend the notion of solution to non-smooth value functions. The discrete Bellman equation sidesteps these regularity issues entirely: the expectation over stochastic transitions provides natural smoothing, and contraction mapping arguments (Chapter 3) give existence and uniqueness without PDE machinery.

> **Remark B.2.2** (HJB in RL). We rarely solve HJB directly in RL---the PDE is intractable in high dimensions (the "curse of dimensionality"). Instead, we learn value functions from sampled transitions. However, HJB provides:
> - **Intuition**: The value function is "almost everywhere" smooth, with kinks at control switching boundaries
> - **Verification**: If we have a candidate $V$, checking the HJB PDE verifies optimality
> - **Continuous-time RL**: Recent work on neural ODEs and continuous-time actor-critic uses HJB structure ([@yildiz:continuous_rl:2022])

---

## B.3 From Control Theory to RL Algorithms

The connections above are not merely academic parallels---they inspire concrete algorithms that power modern deep RL.

### B.3.1 From LQR to Deterministic Policy Gradients

The LQR optimal gain $K^*$ solves $\nabla_K J(K) = 0$ where:

$$
\nabla_K J(K) = 2(RK - \gamma B^\top P A) \Sigma_K
\tag{B.8}
$$
{#EQ-B.8}

and $\Sigma_K = \mathbb{E}[x_t x_t^\top]$ is the state covariance under policy $K$. This observation has profound implications:

1. **LQR is a special case of policy gradient**: Even though we have a closed-form Riccati solution, gradient descent on $K$ converges to the same optimum
2. **Nonlinear extension**: Replace $u = -Kx$ with $u = \pi_\theta(x)$ (neural network), approximate $\nabla_\theta J$ via a learned critic $Q_\phi(x, u)$, and descend

This yields **DDPG** (Deep Deterministic Policy Gradient, [@lillicrap:ddpg:2016]):

$$
\nabla_\theta J \approx \mathbb{E}_{x \sim \mathcal{D}}\left[\nabla_u Q_\phi(x, u)\big|_{u=\pi_\theta(x)} \nabla_\theta \pi_\theta(x)\right]
\tag{B.9}
$$
{#EQ-B.9}

where $\mathcal{D}$ is a replay buffer of past transitions. DDPG is LQR for the neural age.

**TD3** ([@fujimoto:td3:2018]) refines DDPG with three tricks: twin Q-networks (minimum of two critics reduces overestimation), delayed policy updates (critic trains faster than actor), and target policy smoothing (regularization).

### B.3.2 From HJB to Fitted Value Iteration

The HJB fixed-point $V^* = \mathcal{T}V^*$ (where $\mathcal{T}$ is the Bellman operator) motivates **fitted value iteration**:

1. Collect transitions $(x, a, r, x')$ using current or exploratory policy
2. Fit $V_\theta$ to minimize $\|V_\theta(x) - (r + \gamma V_{\theta'}(x'))\|^2$ over dataset
3. Update target $\theta' \leftarrow \theta$ periodically
4. Repeat

This is the continuous-state analog of tabular value iteration (Chapter 3). Convergence requires:
- **Approximate completeness**: Function class $\{V_\theta\}$ can represent $V^*$ (or close approximation)
- **Sufficient exploration**: Dataset covers state-action space
- **Stable targets**: Target network $V_{\theta'}$ updates slowly to prevent oscillation

The **DQN** algorithm ([@mnih:dqn:2015]) implements this for Q-functions with neural networks, adding experience replay and target networks for stability. Despite incomplete theory---we lack guarantees that neural fitted iteration converges---DQN achieved superhuman Atari performance and launched modern deep RL.

---

## B.4 Why This Matters: Control Theory Tools in Later Chapters

We now preview how control theory tools will reappear throughout the book:

### Lyapunov Analysis (Chapter 10)

A **Lyapunov function** $L: \mathcal{X} \to \mathbb{R}_{\geq 0}$ satisfies:
- $L(x^*) = 0$ at the equilibrium $x^*$
- $L(x) > 0$ for $x \neq x^*$
- $L$ decreases along system trajectories: $L(x_{t+1}) \leq L(x_t)$

If such $L$ exists, the system is stable---trajectories converge to $x^*$. We use Lyapunov-like constructions in Chapter 10 to:
- Prove convergence of primal-dual algorithms for constrained optimization
- Bound sub-optimality under distribution drift
- Design drift detectors that monitor a Lyapunov-like "energy" measure

### Robust Control (Chapter 10)

When the simulator differs from reality (model mismatch), standard RL fails. Robust control asks: find a policy that performs well under the **worst-case** dynamics within an uncertainty set $\mathcal{P}$:

$$
\max_\pi \min_{P \in \mathcal{P}} J(\pi, P)
$$

This **minimax** formulation inspires:
- **Domain randomization**: Train on a distribution of simulators
- **Robust MDPs**: Optimize against adversarial transition perturbations
- **Sim-to-real transfer**: Chapter 10's guardrails apply robust control insights to handle distribution shift between simulator and production

### Trajectory Optimization (Chapter 11)

Multi-episode MDPs (Chapter 11) model user engagement across sessions. Planning optimal engagement trajectories resembles trajectory optimization in robotics:
- **State**: User satisfaction/retention level
- **Control**: Ranking policy per session
- **Dynamics**: Satisfaction evolves based on session outcomes

We borrow from **model predictive control (MPC)**: plan a trajectory over a finite horizon, execute the first action, re-plan with updated state. The HJB viewpoint illuminates the infinite-horizon structure.

---

## B.5 Timeline of Deep RL Milestones

The interplay between control theory and deep learning has produced remarkable progress. We trace the major algorithms, noting their control-theoretic ancestry:

| Year | Algorithm | Key Contribution | Control Connection |
|------|-----------|------------------|--------------------|
| 2013 | **DQN** (Mnih et al.) | Neural fitted Q-iteration for Atari | Bellman equation + replay |
| 2015 | **DDPG** (Lillicrap et al.) | Continuous control via deterministic policy gradients | LQR policy gradient |
| 2016 | **A3C** (Mnih et al.) | Asynchronous actor-critic | Parallel trajectory sampling |
| 2017 | **PPO** (Schulman et al.) | Stable policy updates via clipped objectives | Trust region (TRPO) |
| 2018 | **SAC** (Haarnoja et al.) | Maximum entropy RL for robust exploration | Stochastic optimal control |
| 2018 | **TD3** (Fujimoto et al.) | Twin critics, delayed updates | Stabilized DDPG |
| 2020 | **MuZero** (Schrittwieser et al.) | Model-based planning without known dynamics | Learned dynamics model |
| 2021 | **Decision Transformer** (Chen et al.) | RL as sequence modeling | Trajectory optimization |
| 2022 | **RLHF** (Ouyang et al.) | Policy gradients for LLM alignment | Reward shaping |

Each advance addresses specific failure modes that control theory predicted: DQN's target networks fix Q-learning instability (cf. unstable Riccati iteration), PPO's clipping prevents policy collapse (cf. trust region methods), SAC's entropy regularization maintains exploration (cf. stochastic control). The progression reflects a dialogue between classical insights and deep learning engineering.

> **Remark B.5.1** (The theory-practice gap). A striking feature of deep RL is how often algorithms work despite lacking theoretical guarantees. DQN has no convergence proof for neural function approximation. PPO's clipping heuristic lacks rigorous justification. SAC's entropy temperature is tuned empirically. This gap between theory and practice---algorithms that "shouldn't work" but do---is both humbling and intellectually exciting. We address it honestly throughout this book: when theory guarantees something, we prove it; when practice outpaces theory, we say so.

---

## B.6 Summary

**Key connections established:**

1. **LQR $\to$ Linear Policies**: Quadratic costs yield linear optimal policies; RL extends to nonlinear rewards via neural function approximation
2. **HJB $\to$ Bellman**: The discrete Bellman equation is a discretization of the continuous-time HJB PDE
3. **Policy Gradient $\leftrightarrow$ Riccati**: LQR's closed-form Riccati solution can be recovered by policy gradient descent on linear policies
4. **Lyapunov $\to$ Stability**: Control-theoretic stability analysis (Lyapunov functions) informs RL convergence proofs and drift detection

**When to consult this appendix:**

- **Chapter 3**: Bellman operators as discrete-time HJB, contraction mappings as control stability
- **Chapter 8**: Policy gradients as continuous relaxation of LQR (Section 8.6)
- **Chapter 10**: Lyapunov analysis for drift detection and algorithm convergence (Section 10.3)
- **Chapter 11**: Multi-episode dynamics via trajectory optimization (Section 11.2)

For readers with control background, these connections provide intuition and transfer existing knowledge to RL. For those learning RL first, return to this appendix after completing the main text---the parallels will illuminate both fields.

---

## B.7 References and Further Reading

**Classical optimal control:**
- [@kirk:optimal_control:2004]: Comprehensive treatment of optimal control theory, HJB derivation, Pontryagin maximum principle
- [@bertsekas:dynamic_programming:2012, Chapters 1--4]: Discrete-time optimal control, Bellman equations, LQR with full proofs

**Deep RL algorithms:**
- [@mnih:dqn:2015]: Deep Q-Networks (the Atari breakthrough)
- [@lillicrap:ddpg:2016]: Deep Deterministic Policy Gradient (LQR for neural policies)
- [@schulman:proximal_policy:2017]: Proximal Policy Optimization (stable policy gradients)
- [@haarnoja:soft_actor_critic:2018]: Soft Actor-Critic (maximum entropy RL)

**Control-RL connections:**
- [@recht:tour:2019]: "A Tour of Reinforcement Learning: The View from Continuous Control" (highly recommended bridge paper)
- [@levine:reinforcement:2020]: "Reinforcement Learning and Control as Probabilistic Inference"
- [@fazel:global_convergence:2018]: "Global Convergence of Policy Gradient Methods for the Linear Quadratic Regulator" (policy gradient recovers LQR)
