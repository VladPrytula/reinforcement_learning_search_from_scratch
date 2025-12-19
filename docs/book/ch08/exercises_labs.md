# Chapter 8: Policy Gradients --- Exercises & Labs

**Companion to Chapter 8: Policy Gradient Methods**

This document provides detailed solutions, hints, and experimental protocols for the exercises and labs in Chapter 8.

---

## Part A: Mathematical Exercises

### Exercise 8.1: Baseline Variance Reduction {#EX-8.1}

**Problem:** Derive the state-dependent baseline $b^*(s)$ that minimizes the variance of the policy gradient estimator at state $s$, and explain how it relates to the commonly used value-function baseline $V^\pi(s)$.

**Solution:**

Consider the gradient estimator at a single state $s$ (we drop time subscripts for clarity):

$$
\hat{g} = \nabla_\theta \log \pi_\theta(a|s) \, (Q^\pi(s,a) - b(s))
$$

where $a \sim \pi_\theta(\cdot|s)$. We want to minimize the variance:

$$
\text{Var}[\hat{g}] = \mathbb{E}_{a \sim \pi}[\hat{g}^2] - (\mathbb{E}_{a \sim \pi}[\hat{g}])^2
$$

From [THM-8.2.2], we know $\mathbb{E}_{a \sim \pi}[\hat{g}] = \nabla_\theta J(\theta)$ regardless of $b(s)$ (constant in expectation). Thus minimizing variance is equivalent to minimizing:

$$
\mathbb{E}_{a \sim \pi}[\hat{g}^2]
= \mathbb{E}_{a \sim \pi}\left[(\nabla_\theta \log \pi(a|s))^2 (Q^\pi(s,a) - b(s))^2\right]
$$

Define $g(a) = \nabla_\theta \log \pi(a|s)$ and $Q(a) = Q^\pi(s,a)$ for brevity. We minimize:

$$
f(b) = \mathbb{E}_a[g(a)^2 (Q(a) - b)^2]
$$

Take the derivative w.r.t. $b$:

$$
\frac{df}{db} = \mathbb{E}_a[g(a)^2 \cdot 2(Q(a) - b) \cdot (-1)]
= -2 \mathbb{E}_a[g(a)^2 (Q(a) - b)]
$$

Set to zero:

$$
\mathbb{E}_a[g(a)^2 Q(a)] - b \mathbb{E}_a[g(a)^2] = 0
$$

Solve for $b$:

$$
b^*(s) = \frac{\mathbb{E}_{a \sim \pi(\cdot|s)}[g(a)^2 Q(a)]}{\mathbb{E}_{a \sim \pi(\cdot|s)}[g(a)^2]}
$$

This is the unique minimizer of $\text{Var}[\hat{g}]$; it is a **weighted** average of $Q(a)$ with weights proportional to the squared score function. The familiar value-function baseline $V^\pi(s) = \mathbb{E}_{a}[Q^\pi(s,a)]$ drops the weights. It is optimal under the additional assumption that $\|g(a)\|$ is approximately constant in $a$ (true for small action perturbations or isotropic Gaussians near the mean), and it remains a powerful heuristic because $V^\pi$ is much easier to estimate than the weighted ratio above.

**Optional check.** Show that if $(\nabla_\theta \log \pi(a|s))^2 = c(s)$ is constant in $a$, then $b^*(s) = V^\pi(s)$.

**Numerical Verification (Lab Extension):**

Implement multiple baselines and measure gradient variance empirically:

```python
# In zoosim/policies/reinforce.py, modify update() to test:
# 1. b = 0 (no baseline)
# 2. b = mean(returns) (constant)
# 3. b = V_network(s_t) (learned value function)

# Track: gradient_norms = []
# Plot variance across episodes
```

---

### Exercise 8.2: Entropy and Boltzmann Policy {#EX-8.2}

**Problem:** Show that maximizing entropy $H(\pi)$ subject to expected reward $\mathbb{E}_{a \sim \pi}[Q(a)] \geq \bar{R}$ yields Boltzmann policy $\pi(a) \propto \exp(\beta Q(a))$.

**Solution:**

We solve the constrained optimization problem:

$$
\max_\pi \, H(\pi) = -\sum_a \pi(a) \log \pi(a)
\quad \text{s.t.} \quad
\sum_a \pi(a) Q(a) = \bar{R}, \quad \sum_a \pi(a) = 1
$$

Form the Lagrangian:

$$
\mathcal{L}(\pi, \lambda, \mu) = -\sum_a \pi(a) \log \pi(a)
+ \lambda \left(\sum_a \pi(a) Q(a) - \bar{R}\right)
+ \mu \left(\sum_a \pi(a) - 1\right)
$$

Take the derivative w.r.t. $\pi(a)$:

$$
\frac{\partial \mathcal{L}}{\partial \pi(a)}
= -\log \pi(a) - 1 + \lambda Q(a) + \mu
$$

Set to zero (first-order condition):

$$
-\log \pi(a) = 1 - \lambda Q(a) - \mu
$$

Exponentiate:

$$
\pi(a) = \exp(\lambda Q(a) + \mu - 1)
= C \exp(\lambda Q(a))
$$

where $C = \exp(\mu - 1)$ is determined by the normalization constraint $\sum_a \pi(a) = 1$:

$$
\pi(a) = \frac{\exp(\lambda Q(a))}{\sum_{a'} \exp(\lambda Q(a'))}
\quad \text{(Boltzmann/softmax policy)}
$$

The Lagrange multiplier $\lambda$ is the **inverse temperature** (often denoted $\beta$). High $\lambda$ (low temperature) -> deterministic policy (argmax). Low $\lambda$ (high temperature) -> uniform policy (maximum entropy).

**Connection to Maximum Entropy RL:** SAC [haarnoja:soft_actor_critic:2018] maximizes $J(\pi) = \mathbb{E}[R] + \alpha H(\pi)$, which is the unconstrained version of this problem. The optimal policy is softmax w.r.t. the "soft" Q-function $Q_{\text{soft}}(s,a)$.

---

### Exercise 8.3: REINFORCE Convergence {#EX-8.3}

**Problem:** Show that [ALG-8.1] converges to a local optimum under standard stochastic approximation conditions.

**Solution (Sketch):**

We apply the **Robbins-Monro theorem** [robbins:stochastic_approximation:1951] for stochastic gradient ascent. The update is:

$$
\theta_{k+1} = \theta_k + \alpha_k \hat{g}_k
$$

where $\hat{g}_k = \sum_t \nabla_\theta \log \pi_{\theta_k}(a_t|s_t) (G_t - b)$ is the gradient estimate at episode $k$.

**Conditions (Robbins-Monro):**
1. $\mathbb{E}[\hat{g}_k \mid \theta_k] = \nabla_\theta J(\theta_k)$ (unbiased) --- Proven in [THM-8.2]
2. $\text{Var}[\hat{g}_k \mid \theta_k] < \infty$ --- Requires bounded rewards and policy smoothness
3. Learning rate schedule: $\sum_k \alpha_k = \infty$ and $\sum_k \alpha_k^2 < \infty$ (e.g., $\alpha_k = 1/k$)

Under these conditions, $\theta_k$ converges to a **stationary point** $\theta^*$ where $\nabla_\theta J(\theta^*) = 0$ with probability 1.

**Local vs. Global Optimum:** Since $J(\theta)$ is generally nonconvex (neural network policies have many local optima), we only guarantee convergence to a local optimum. However, for certain policy classes (e.g., LQR with linear policy), all local optima are global [fazel:global_convergence:2018].

**Practical Note:** Modern implementations use **constant learning rate** $\alpha$ (e.g., $3 \times 10^{-4}$) with early stopping, not diminishing $\alpha_k$. This trades asymptotic convergence for faster practical learning.

**Reference:** [sutton:policy_gradient:2000, Theorem 2] provides a complete proof for the softmax policy case.

---

### Exercise 8.4: Policy Gradient for LQR {#EX-8.4}

**Problem:** Implement policy gradient for the 1D discounted LQR system and verify convergence toward the analytic gain obtained from the discounted algebraic Riccati equation (DARE).

**Analytical reference.** For dynamics $s_{t+1} = s_t + a_t$, stage cost $c_t = s_t^2 + a_t^2$, and discount $\gamma = 0.99$, the DARE

$$
P = Q + \gamma A^\top P A - \gamma^2 A^\top P B (R + \gamma B^\top P B)^{-1} B^\top P A
$$

with $A=B=Q=R=1$ yields the positive root $P \approx 1.62$ and feedback gain

$$
K^* = -\frac{\gamma P}{R + \gamma P} \approx -0.615.
$$

Script (requires `scipy`) to compute the reference:

```python
import numpy as np
from scipy.linalg import solve_discrete_are

def discounted_lqr_gain(gamma: float = 0.99) -> float:
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[1.0]])

    A_bar = np.sqrt(gamma) * A
    B_bar = np.sqrt(gamma) * B
    P = solve_discrete_are(A_bar, B_bar, Q, R)

    gain = -(gamma * B.T @ P @ A) / (R + gamma * B.T @ P @ B)
    return gain.item()

print(f"K* = {discounted_lqr_gain():.4f}")
```

**Policy gradient experiment.**

```python
"""1D discounted LQR via REINFORCE.

System:  s_{t+1} = s_t + a_t
Reward:  r_t = -(s_t^2 + a_t^2)
Policy:  pi_theta(a|s) = N(theta * s, sigma^2)
Target:  theta* ~= -0.615 (DARE)
"""

import numpy as np
import matplotlib.pyplot as plt

def lqr_policy_gradient(
    num_episodes: int = 500,
    horizon: int = 20,
    learning_rate: float = 0.05,
    sigma: float = 0.5,
    gamma: float = 0.99,
    seed: int = 42,
    target_gain: float = -0.615,
):
    rng = np.random.default_rng(seed)
    theta = 0.0
    theta_history = [theta]

    for episode in range(num_episodes):
        s = 1.0
        states, actions, rewards = [], [], []

        for _ in range(horizon):
            mu = theta * s
            a = mu + sigma * rng.normal()
            r = -(s**2 + a**2)
            s_next = s + a

            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s_next

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        grad = 0.0
        for s_t, a_t, G_t in zip(states, actions, returns):
            score = (a_t - theta * s_t) * s_t / (sigma**2)
            grad += score * G_t

        theta += learning_rate * grad / horizon
        theta_history.append(theta)

        if episode % 50 == 0:
            avg_return = np.mean(returns)
            err = abs(theta - target_gain)
            print(f"Ep {episode:3d} | theta={theta:6.3f} | Return={avg_return:7.2f} | |theta-K*|={err:6.3f}")

    return theta, theta_history

theta_star = -0.615  # from discounted_lqr_gain()
theta_final, history = lqr_policy_gradient(target_gain=theta_star)

print(f"\nFinal theta = {theta_final:.4f}")
print(f"Analytical theta* = {theta_star:.4f}")
print(f"|theta - theta*| = {abs(theta_final - theta_star):.4f}")

plt.figure(figsize=(10, 4))
plt.plot(history, label='theta (policy gradient)')
plt.axhline(theta_star, color='red', linestyle='--', label='theta* (DARE)')
plt.xlabel('Episode')
plt.ylabel('Policy Parameter theta')
plt.title('Policy Gradient vs. DARE Solution')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lqr_policy_gradient.png', dpi=150)
print("Plot saved: lqr_policy_gradient.png")
```

**Expected Output:**
```
Ep   0 | theta=-0.022 | Return=-27.91 | |theta-K*|= 0.593
Ep  50 | theta=-0.392 | Return=-22.83 | |theta-K*|= 0.223
Ep 100 | theta=-0.541 | Return=-21.10 | |theta-K*|= 0.074
Ep 150 | theta=-0.594 | Return=-20.58 | |theta-K*|= 0.021
Ep 200 | theta=-0.611 | Return=-20.45 | |theta-K*|= 0.004
...

Final theta = -0.6148
Analytical theta* = -0.6150
|theta - theta*| = 0.0002
```

**Analysis.**
- REINFORCE converges to the discounted Riccati gain within numerical tolerance.
- The DARE baseline provides an exact target for debugging policy gradient implementations.
- Because discounted LQR is convex in the policy parameter, every stationary point is global [@fazel:global_convergence:2018].

**Extension:** Extend to $s \in \mathbb{R}^2$ with matrix gains $K \in \mathbb{R}^{2 \times 2}$ and verify convergence to the DARE solution on each axis.

!!! note "Status --- Advanced Labs"
    Lab 8.1 (REINFORCE with rich features) and Lab 8.2 (Deep end-to-end REINFORCE) are fully wired to existing scripts under `scripts/ch08/`. Lab 8.3 (Gaussian policy ablations) and Lab 8.4 (RLHF simulation) now ship with concrete demo scripts (`scripts/ch08/policy_ablation.py` and `scripts/ch08/rlhf_demo.py`), but remain advanced: readers are encouraged to treat them as starting points for their own ablation pipelines and RLHF variants rather than fixed production code.

---

## Part B: Computational Labs

### Lab 8.1: REINFORCE with Rich Features

**Objective:** Reproduce Chapter 8 results and analyze hyperparameter sensitivity.

#### Protocol

**Step 1: Baseline Run**

```bash
cd /path/to/rl_search_from_scratch

# Ensure environment is set up
source .venv/bin/activate
pip install -e .

# Run REINFORCE with rich features
python scripts/ch08/reinforce_demo.py \
  --episodes 3000 \
  --lr 3e-4 \
  --gamma 0.99 \
  --entropy 0.01 \
  --a_max 5.0 \
  --rich-features \
  --seed 2025
```

**Expected Output:**
```
Chapter 8: REINFORCE on Zooplus Search
Episodes: 3000 | Seed: 2025
Features: Rich (Dim 17)
Action Limit: 5.0 | LR: 0.0003

Observation Dim: 17
Action Dim: 10 (Continuous)

Ep   100 | Return:   9.12 | Entropy:  2.47 | Loss: -31.52
Ep   200 | Return:   9.87 | Entropy:  2.31 | Loss: -34.21
...
Ep  2900 | Return:  11.43 | Entropy:  1.98 | Loss: -42.67
Ep  3000 | Return:  11.56 | Entropy:  1.95 | Loss: -43.12

Final 100-ep Average Return: 11.56
Improvement: 9.02 -> 11.56
Agent learned to improve policy.
```

**Step 2: Ablation --- No Entropy Regularization**

```bash
python scripts/ch08/reinforce_demo.py \
  --episodes 3000 \
  --entropy 0.0 \
  --seed 2025
```

**Observation:** Policy entropy collapses to near-zero by episode 500, final return ~9.0 (worse than baseline). This demonstrates necessity of entropy regularization.

**Step 3: Ablation --- Learning Rate Sweep**

Run with $\alpha \in \{1e-5, 1e-4, 3e-4, 1e-3, 3e-3\}$:

```bash
for lr in 1e-5 1e-4 3e-4 1e-3 3e-3; do
  python scripts/ch08/reinforce_demo.py \
    --episodes 3000 \
    --lr $lr \
    --seed 2025 \
    > results_lr_${lr}.txt
done
```

**Expected Findings:**
- $\alpha = 1e-5$: Too slow, doesn't converge
- $\alpha = 1e-4$: Converges to ~10.8
- $\alpha = 3e-4$: **Optimal**, converges to ~11.6
- $\alpha = 1e-3$: Slightly unstable, ~11.2
- $\alpha = 3e-3$: Diverges (gradient steps too large)

**Step 4: Comparison to Q-Learning**

Run Chapter 7 Q-learning for comparison:

```bash
python scripts/ch07/continuous_actions_demo.py \
  --n-episodes 3000 \
  --seed 2025
```

**Expected:** Q-learning achieves ~25.0 (2.2x better than REINFORCE).

**Analysis Questions:**
1. Why does Q-learning outperform REINFORCE by 2x?
2. Plot variance of returns: REINFORCE vs. Q-learning. Which is noisier?
3. Compute sample efficiency: episodes to reach return 10.0. Q-learning ~300, REINFORCE ~1500.

#### Deliverables

**Plots:**
1. Learning curves: return vs. episode (3 seeds, mean +/- std)
2. Entropy trajectory: $H(\pi_\theta)$ vs. episode
3. Gradient norm: $\|\nabla J\|$ vs. episode

**Table:**

| Hyperparameter | Value | Final Return | Convergence Episode |
|:---------------|:------|:-------------|:--------------------|
| $\alpha = 3e-4$ | 0.01 | 11.56 | ~2500 |
| $\alpha = 3e-4$, no entropy | 0 | 9.02 | ~1200 (collapsed) |
| $\alpha = 1e-3$ | 0.01 | 11.18 | ~2200 |

---

### Lab 8.1b: The Variance Reduction Challenge

**Objective:** Empirically verify the benefit of a learned baseline in a stochastic environment.

This lab directly implements the "Failure -> Diagnosis -> Fix" narrative from Section 8.5: we run Vanilla REINFORCE (Act I), observe the variance problem, then run REINFORCE + Baseline (Act III) to see the fix in action.

#### Protocol

**Step 1: Run Vanilla REINFORCE**

```bash
python scripts/ch08/reinforce_demo.py \
  --episodes 5000 \
  --seed 2025 \
  --lr 3e-4 \
  --entropy 0.01 \
  --a_max 5.0 \
  --rich-features
```

**What to observe:**
- Jagged loss curves with large spikes
- "Forgetting" --- performance spikes up then suddenly drops
- Oscillating returns between 3-17, never stabilizing
- Final 100-episode average: ~7.99 (worse than random baseline 8.7!)

**Step 2: Run REINFORCE with Learned Baseline**

```bash
python scripts/ch08/reinforce_baseline_demo.py \
  --episodes 5000 \
  --seed 2025 \
  --lr 3e-4 \
  --value-lr 1e-3 \
  --entropy 0.01 \
  --a_max 5.0 \
  --rich-features
```

**What to observe:**
- Smoother entropy decay
- Monotonic improvement in returns
- Value loss decreases from 100s to near-zero (critic learning $V^\pi(s)$)
- Final 100-episode average: **13.19** (65% improvement!)

#### Expected Results

| Method | Architecture | Final Return | Convergence | Stability |
|:-------|:-------------|:------------|:-----------|:----------|
| **Vanilla REINFORCE** | Single policy net | ~7.99 | Never (tested to 50k episodes) | High variance, oscillates |
| **REINFORCE + Baseline** | Policy + Value nets | **~13.19** | ~5,000 episodes | Stable, monotonic improvement |
| **Improvement** | - | **+65%** | **10x faster** | - |

#### Analysis Tasks

**Task 1: Overlay Learning Curves**

Plot return vs. episode for both methods on the same axes. Annotate:
- Episode 500: Baseline stabilizes (~12), Vanilla still oscillating (3-15)
- Episode 5000: Baseline converges (13.19), Vanilla unstable (7.99)

**Task 2: Entropy Trajectory**

Verify both methods maintain exploration (entropy ~14-15 throughout). This confirms the improvement is **pure variance reduction**, not an exploration/exploitation artifact.

**Task 3: Advantage Distribution**

Create histograms comparing:
- **Vanilla**: $(G_t - b)$ where $b$ is scalar EMA baseline
- **Baseline**: $(G_t - V_\phi(s_t))$ where $V_\phi$ is learned value network

You should see:
- Vanilla distribution: Bimodal (mode at -10 for no-purchase episodes, scattered +30 to +50 for purchases)
- Baseline distribution: Symmetric around 0, tighter std (~7 vs ~15 for vanilla)

**Task 4: Variance Quantification**

Compute gradient variance empirically:
```python
# During training, collect:
gradients = []  # List of gradient norms per episode
# Then compute:
gradient_variance = np.var(gradients)
```

Expected: Baseline reduces gradient variance by ~50% compared to vanilla.

#### Key Insight

The learned value baseline $V_\phi(s)$ removes **state-dependent variance** that a scalar baseline cannot. This is [THM-8.5.2] in action:
- States with high expected return (e.g., `price_sensitive` user on `budget` query): $V(s) \approx 15$
- States with low expected return (e.g., `random` user on `generic` query): $V(s) \approx 5$
- Scalar baseline: $b = 10$ (global average, wrong for both states!)

By adapting the baseline to each state's intrinsic value, we eliminate the noise from "environmental luck" and allow the policy to learn from genuine action quality.

#### Deliverables

**Plots:**
1. Learning curves (overlay): return vs. episode for vanilla and baseline
2. Entropy trajectories: verify exploration maintained in both
3. Value loss curve (baseline only): should decrease from $10^2$ to $<5$
4. Advantage distribution histograms: $(G_t - b)$ vs $(G_t - V(s))$

**Table:**

| Metric | Vanilla REINFORCE | REINFORCE + Baseline | Improvement |
|:-------|:-----------------|:--------------------|:------------|
| Final Return | 7.99 | **13.19** | **+65%** |
| Convergence Episodes | Never (50k+) | ~5,000 | **10x faster** |
| Gradient Std Dev | ~15 | ~7 | **50% reduction** |
| Stability | Oscillates | Monotonic | Qualitative |

---

### Lab 8.2: Deep End-to-End REINFORCE (Failure Analysis)

**Objective:** Understand why end-to-end learning from raw features fails.

#### Protocol

**Step 1: Reproduce Failure**

```bash
python scripts/ch08/neural_reinforce_demo.py \
  --episodes 5000 \
  --lr 3e-4 \
  --hidden-dim 128 \
  --device auto \
  --seed 2025
```

**Expected Output:**
```
Deep REINFORCE (End-to-End Learning)
Device: CPU (or CUDA/MPS if available)
Episodes: 5000 | Seed: 2025
Network: Raw Input -> [128, 128] -> Action

Observation Dim: 8 (Raw Context)
Action Dim: 10

Ep   200 | Return:   7.23 | Loss: -28.91 | 45.2s
Ep   400 | Return:   6.87 | Loss: -26.34 | 91.8s
...
Ep  4800 | Return:   5.91 | Loss: -22.15 | 1103.2s
Ep  5000 | Return:   5.93 | Loss: -22.08 | 1150.1s

Final Average Return: 5.93
---
Contextual Baselines (Approximate):
  Random Agent:      ~8.7
  Rich Features (Ch6/8): ~10.5 - 11.5
  Continuous Q (Ch7):    ~25.0
---
NOTE: Training unstable or insufficient. Deep RL from scratch is hard!
```

**Observation:** Final return 5.93 is **worse than random** (8.7). The network failed to learn useful features.

**Step 2: Increase Training Budget**

```bash
python scripts/ch08/neural_reinforce_demo.py \
  --episodes 10000 \
  --seed 2025
```

**Result:** Performance improves to ~7.5 by episode 10k, but still below random. Simply training longer doesn't solve the problem.

**Step 3: Add Supervised Pretraining**

Modify `neural_reinforce_demo.py` to pretrain feature layers on supervised task:

```python
# Pretrain feature extractor on category prediction
# Input: user segment + query type -> Output: clicked category (classification)

# Load logged data from Chapter 6
logged_data = load_logged_trajectories()  # (s, a, r, clicked_cat)

# Train feature net
feature_net = GaussianPolicy.net  # Extract feature layers
optimizer = Adam(feature_net.parameters(), lr=1e-3)

for epoch in range(10):
    for batch in logged_data:
        s, _, _, cat = batch
        pred_cat = feature_net(s)  # Add classification head
        loss = cross_entropy(pred_cat, cat)
        loss.backward()
        optimizer.step()

# Then initialize REINFORCE policy with pretrained features
```

**Expected:** Pretraining improves final return to ~9.5 (better, but still below rich features).

**Step 4: Comparison to Neural Linear Bandit (Chapter 6A)**

Why did neural features work in Chapter 6A but fail here?

```bash
# Run Ch6A neural linear bandit
python scripts/ch06a/neural_linear_demo.py --episodes 3000 --seed 2025
# Expected: ~10.0 return (success!)
```

**Analysis:** Neural linear learns a **value function** (dense reward signal every step). REINFORCE learns from **episodic returns** (sparse signal). Value-based learning is more sample-efficient for feature learning.

#### Deliverables

**Report:**
1. Plot: Return vs. episode for raw REINFORCE, pretrained REINFORCE, rich-feature REINFORCE
2. Gradient flow analysis: Compute $\|\nabla \theta_{\text{features}}\|$ (feature gradient norm) vs. $\|\nabla \theta_{\text{policy head}}\|$. Show that feature gradients vanish.
3. Recommendation: Use Actor-Critic (Chapter 12) or pretrained representations for end-to-end learning.

---

### Lab 8.3: Gaussian Policy Ablations

**Objective:** Compare action distribution parameterizations.

Turnkey script: `scripts/ch08/policy_ablation.py` implements the ablation harness used in this lab; you can run it directly with `--policy {gaussian,beta,squashed,fixed_std}` as shown below.

#### Implementations

**Modify `zoosim/policies/reinforce.py`:**

**1. Beta Distribution Policy**

```python
from torch.distributions import Beta

class BetaPolicy(nn.Module):
    """Beta policy with support [0, 1], scaled to [-a_max, +a_max]."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes)
        self.alpha_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.beta_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> Distribution:
        x = self.net(obs)
        alpha = torch.exp(self.alpha_head(x)) + 1.0  # alpha > 1 for unimodal
        beta = torch.exp(self.beta_head(x)) + 1.0    # beta > 1 for unimodal
        return Beta(alpha, beta)

    def sample_action(self, obs: torch.Tensor, a_max: float) -> torch.Tensor:
        dist = self.forward(obs)
        # Beta samples in [0, 1], scale to [-a_max, +a_max]
        a_01 = dist.sample()
        return 2 * a_max * (a_01 - 0.5)
```

**2. Squashed Gaussian (Tanh)**

```python
class SquashedGaussianPolicy(nn.Module):
    """Gaussian with tanh squashing (SAC-style)."""

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_head(self.net(obs))
        std = torch.exp(torch.clamp(self.log_std, -2, 1))

        # Sample from unbounded Gaussian
        z = mu + std * torch.randn_like(mu)

        # Squash to [-1, 1] via tanh
        action = torch.tanh(z)

        # Compute log-prob with change-of-variables
        # log pi(a) = log pi(z) - log |da/dz|
        # |da/dz| = 1 - tanh^2(z)
        log_prob = Normal(mu, std).log_prob(z).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob
```

**3. Fixed Std Gaussian**

```python
class FixedStdGaussianPolicy(nn.Module):
    """Learn mean only, fix std=1."""

    def __init__(self, obs_dim, action_dim, hidden_sizes, fixed_std=1.0):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes)
        self.mu_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.fixed_std = fixed_std

    def forward(self, obs):
        mu = self.mu_head(self.net(obs))
        std = torch.full_like(mu, self.fixed_std)
        return Normal(mu, std)
```

#### Experiments

Run each variant for 3000 episodes, 3 seeds:

```bash
for policy in gaussian beta squashed fixed_std; do
  for seed in 2025 2026 2027; do
    python scripts/ch08/policy_ablation.py \
      --policy $policy \
      --episodes 3000 \
      --seed $seed \
      > results_${policy}_seed${seed}.txt
  done
done
```

**Compare:**

| Policy | Mean Return | Std Dev | Entropy (final) | Training Stability |
|:-------|:------------|:--------|:----------------|:-------------------|
| Gaussian | 11.56 | 0.42 | 1.95 | Stable |
| Beta | 10.87 | 0.68 | 1.72 | Mode collapse |
| Squashed | 11.34 | 0.51 | 1.88 | Stable |
| Fixed Std | 10.21 | 0.73 | 2.30 (const) | Underexplores late |

**Findings:**
- **Gaussian**: Best performance, standard choice
- **Beta**: Natural bounds but harder to optimize (mode collapse to boundary)
- **Squashed**: Nearly as good as Gaussian, better if strict bounds needed (SAC)
- **Fixed Std**: Can't adapt exploration, underperforms

---

### Lab 8.4: RLHF Simulation (Advanced)

**Objective:** Simulate the InstructGPT/RLHF pipeline for search ranking.

We do not introduce new core theory here. Instead, we **compose three ingredients you already know**—(i) a supervised Gaussian policy (Chapter 8 + SFT), (ii) a learned scalar reward model from pairwise preferences (Bradley–Terry), and (iii) a REINFORCE-style policy gradient with a KL regularizer—to build an InstructGPT-style pipeline in the Zoosim search environment.

!!! note "Big Picture — What's Reused vs New?"
    **Reused from earlier chapters and labs**
    - Gaussian policy $\pi_\theta(a|s)$ with entropy regularization (Chapter 8, Lab 8.1).
    - Policy gradient / REINFORCE update with log-prob weights (Chapter 8, [ALG-8.1]).
    - KL divergence as a regularizer (previewed in PPO Section 8.7.2; used here to keep the RL policy close to the SFT policy).
    
    **New in this lab (structural additions)**
    - Trajectory-level reward model $r_\phi(\tau)$ trained from pairwise preferences via a Bradley–Terry objective.
    - Three-phase workflow: SFT -> reward modeling -> RL fine-tuning with KL penalty, mirroring the InstructGPT RLHF pipeline.
    - Simple "expert" template policy in Zoosim used to generate demonstration data and preference pairs (standing in for human raters).

Turnkey demo: `scripts/ch08/rlhf_demo.py` provides a compact, runnable version of the three-phase RLHF-style pipeline described here (SFT -> reward modeling -> KL-regularized REINFORCE on the learned reward).

#### Three-Phase Protocol

**Phase 1: Supervised Fine-Tuning (SFT)**

Train policy via imitation learning on expert demonstrations:

```python
# Load expert trajectories (discrete templates from Ch6)
expert_data = load_expert_trajectories()  # (s, a_expert, r)

# Train policy to match expert actions
policy = GaussianPolicy(obs_dim, action_dim, hidden_sizes)
optimizer = Adam(policy.parameters(), lr=1e-3)

for epoch in range(50):
    for batch in expert_data:
        s, a_expert = batch['state'], batch['action']

        # Supervised loss: negative log-likelihood
        dist = policy(s)
        loss = -dist.log_prob(a_expert).mean()

        loss.backward()
        optimizer.step()

# This policy pi_SFT mimics expert behavior
```

**Phase 2: Reward Modeling (Bradley-Terry)**

Learn reward function from pairwise comparisons:

```python
# Minimal setup
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Generate trajectory pairs and get human preferences
# (In practice, this would be A/B test data or crowdsourced labels)

trajectories = generate_trajectory_pairs(policy_sft, env, num_pairs=1000)

# For each pair (tau_1, tau_2), ask: "Which ranking is better?"
# Simulate with ground-truth GMV: prefer higher GMV
preferences = []
for tau1, tau2 in trajectories:
    if GMV(tau1) > GMV(tau2):
        preferences.append((tau1, tau2, torch.tensor(1.0)))  # tau1 preferred -> label 1
    else:
        preferences.append((tau1, tau2, torch.tensor(0.0)))  # tau2 preferred -> label 0

# Train reward model: r_phi(tau) such that P(tau1 > tau2) = sigmoid(r_phi(tau1) - r_phi(tau2))
reward_model = RewardMLP(state_dim, hidden=128)
optimizer = Adam(reward_model.parameters(), lr=1e-3)

for epoch in range(20):
    for tau1, tau2, label in preferences:
        r1 = reward_model(tau1)  # Aggregate trajectory features
        r2 = reward_model(tau2)

        # Bradley-Terry loss: logit = r1 - r2, target = 1 if tau1 preferred
        logit = r1 - r2
        loss = F.binary_cross_entropy_with_logits(logit, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

The target `label` equals 1 when $\tau_1 \succ \tau_2$ so that $\sigma(r_\phi(\tau_1) - r_\phi(\tau_2))$ approximates $\mathbb{P}(\tau_1 \text{ preferred})`, matching the classical Bradley-Terry model.

**Phase 3: RL Fine-Tuning (PPO with KL Penalty)**

Optimize learned reward + KL penalty from SFT policy:

```python
# Objective: max E[r_phi(tau)] - beta * KL(pi_theta || pi_SFT)
# This prevents the policy from deviating too far from safe SFT behavior

policy_rl = copy.deepcopy(policy_sft)  # Initialize from SFT
policy_sft.eval()  # Freeze SFT policy as reference

for episode in range(2000):
    # Generate trajectory under pi_theta
    tau = rollout(policy_rl, env)

    # Compute rewards from learned model
    rewards_learned = [reward_model(s, a) for s, a in tau]

    # Compute KL penalty
    kl_penalty = 0
    for s, a in tau:
        dist_rl = policy_rl(s)
        dist_sft = policy_sft(s)
        kl = torch.distributions.kl_divergence(dist_rl, dist_sft).mean()
        kl_penalty += kl

    # Total objective
    total_reward = sum(rewards_learned) - beta * kl_penalty

    # REINFORCE update (or use PPO for better stability)
    loss = compute_policy_gradient_loss(policy_rl, tau, total_reward)
    loss.backward()
    optimizer.step()
```

#### Evaluation

**Metrics:**
1. **Reward Model Accuracy**: Test set AUC for pairwise preferences
2. **KL Divergence**: $D_{\text{KL}}(\pi_{\text{RL}} \| \pi_{\text{SFT}})$ (should stay below threshold)
3. **Ground Truth GMV**: Final policy's actual GMV (not learned reward)

**Expected Results:**
- Phase 1 (SFT): GMV ~10.4 (matches expert templates)
- Phase 2: Reward model achieves 78% pairwise accuracy
- Phase 3 (RL): GMV improves to ~12.1 (beats SFT, stays safe via KL penalty)

**Comparison to Standard RL:**
- Standard REINFORCE (Ch8): GMV ~11.6 but high variance
- RLHF pipeline: GMV ~12.1, more stable (SFT initialization + KL constraint)

#### Connection to LLM RLHF

This is structurally identical to InstructGPT [ouyang:training_language:2022]:
- **SFT**: Train on human demonstrations -> mimics expert search rankers
- **RM**: Learn from pairwise preferences -> "which result set is better?"
- **PPO**: Optimize learned reward + KL penalty -> improves while staying safe

The key insight: **RL on learned rewards** is more sample-efficient than RL on sparse ground-truth rewards (GMV), especially when human feedback is available.

---

## Part C: Additional Investigations

### Investigation 8.A: Variance Reduction Techniques

**Compare variance reduction methods:**

1. **Baseline**: None, constant, value network
2. **Return normalization**: Raw, normalized, standardized
3. **GAE (Generalized Advantage Estimation)**: $\lambda \in \{0, 0.9, 0.95, 0.99\}$

**Protocol:**
- Implement GAE in `zoosim/policies/reinforce.py` (requires value network)
- Measure gradient variance: $\text{Var}[\nabla J]$ across episodes
- Plot bias-variance tradeoff: GAE $\lambda$ vs. final return

**Expected:** GAE with $\lambda=0.95$ reduces variance by 5x while introducing <10% bias.

---

### Investigation 8.B: Trust Region Analysis

**Question:** Why does REINFORCE sometimes collapse catastrophically?

**Approach:**
- Track policy change per update: $D_{\text{KL}}(\pi_{\theta_k} \| \pi_{\theta_{k-1}})$
- Identify episodes where KL divergence spikes (large policy change)
- Correlate with performance drops

**Finding:** Large KL steps (>0.5 nats) often precede performance collapse. This motivates PPO's trust region constraint.

**Implement Simple Trust Region:**

```python
# In update(), after computing loss:
kl_div = compute_kl_divergence(policy_new, policy_old)
if kl_div > kl_threshold:
    # Reject update, halve learning rate
    learning_rate *= 0.5
```

---

### Investigation 8.C: Sample Complexity Lower Bounds

**Theoretical Question:** Is REINFORCE's sample inefficiency fundamental or fixable?

**Analysis:**
- Compare sample complexity: REINFORCE $O(1/\varepsilon^2)$, Q-learning $O(1/\varepsilon)$
- Derive information-theoretic lower bound for policy gradient (see [agarwal:theory_of_reinforcement_learning:2021])
- Show that Monte Carlo estimation inherently has higher variance than TD

**Conclusion:** On-policy methods fundamentally need more samples than off-policy (no replay buffer). PPO + parallelism (A3C-style) mitigates this in practice.

---

## Summary

These exercises and labs provide hands-on experience with policy gradient methods, from rigorous mathematical proofs (optimal baseline, entropy regularization) to production implementation challenges (sample efficiency, feature learning, RLHF pipelines).

**Key Skills Developed:**
- Variance reduction techniques (baselines, normalization, GAE)
- Action distribution design (Gaussian, Beta, squashed)
- Failure mode diagnosis (end-to-end learning, entropy collapse)
- Modern RL pipelines (RLHF simulation)

**Next Steps:**
- Chapter 9: Off-Policy Evaluation (learn from logged data without live interaction)
- Chapter 12: Actor-Critic methods (reduce variance via value function bootstrapping)
- Chapter 13: PPO and trust regions (prevent catastrophic updates)

---

## References

Code files referenced:
- `zoosim/policies/reinforce.py` — REINFORCEAgent implementation
- `scripts/ch08/reinforce_demo.py` — Main training script
- `scripts/ch08/neural_reinforce_demo.py` — Deep end-to-end experiment

External references:
- [@robbins:stochastic_approximation:1951] — Robbins-Monro theorem
- [@fazel:global_convergence:2018] — LQR policy gradient convergence
- [@ouyang:training_language:2022] — RLHF for InstructGPT
- [@agarwal:theory_of_reinforcement_learning:2021] — Sample complexity lower bounds
