# Chapter 6A — Neural Bandits: Complete Planning Document

**Status:** Planning complete, ready for drafting
**Author:** Vlad Prytula (with comprehensive exercise design)
**Date:** 2025-11-19

This document is the complete blueprint for **Chapter 6A — Neural Bandits: From Neural Linear to NL-Bandit**, an optional bridge chapter in Part II. It combines structural planning, exercise design, and integration specifications.

---

## Table of Contents

1. [Position in the Book](#1-position-in-the-book)
2. [Learning Objectives](#2-learning-objectives)
3. [Scope and Non-Goals](#3-scope-and-non-goals)
4. [Proposed Chapter Structure](#4-proposed-chapter-structure)
5. [Exercise Design (Complete)](#5-exercise-design-complete)
6. [Code & File Plan](#6-code--file-plan)
7. [Knowledge Graph & Cross-References](#7-knowledge-graph--cross-references)
8. [Syllabus Integration](#8-syllabus-integration)
9. [Next Steps for Drafting](#9-next-steps-for-drafting)

---

## 1. Position in the Book

**Location:**
- Part II (Simulator and Policies), immediately after **Chapter 6 — Discrete Template Bandits** and before **Chapter 7 — Continuous Actions via Q(x,a)**
- Treated as an **optional** appendix-style chapter: "Chapter 6A — Neural Bandits"

**Dependencies:**
- **Depends on:**
  - CH-5: Feature engineering and reward
  - CH-6: Discrete template bandits (LinUCB, Thompson Sampling, regret, feature poverty story)
- **Provides:**
  - Complete "Neural Linear" and "NL-Bandit" story for Ch13 (Offline RL) baselines
  - Bridge from hand-crafted features (Ch6) to learned representations (Ch6A) to continuous actions (Ch7)

**Relationship to adjacent chapters:**

**Ch06 → Ch06A:**
- Ch06 shows **linear bandits on hand-crafted features** with failure→diagnosis→fix arc (feature poverty)
- Ch06A answers: "What if we **learn the features** or the entire Q(x,a) with a neural network?"

**Ch06A → Ch07:**
- Ch06A keeps action space **discrete templates** (contextual bandit; γ=0)
- Ch07 generalizes same Q(x,a) idea to **continuous actions** (boost vectors) with CEM and trust regions

**Knowledge Graph node (to be added):**
```yaml
- id: CH-6A
  kind: chapter
  title: Chapter 6A — Neural Bandits: From Neural Linear to NL-Bandit
  status: planned
  file: docs/book/drafts/ch06a/ch06a_neural_bandits.md
  summary: Neural feature learning and neural Q(x,a) for discrete template bandits; bridge from linear (Ch6) to continuous (Ch7).
  depends_on: [CH-5, CH-6]
  forward_refs: [CH-7, CH-13]
  defines: [EQ-6A.1, EQ-6A.2, ALG-6A.1, ALG-6A.2]
```

---

## 2. Learning Objectives

By the end of Chapter 6A, the reader should be able to:

### Conceptual Understanding

1. **Explain why pure linear bandits can fail** in high-dimensional or non-linear settings even with rich features
2. **Distinguish between:**
   - **Neural Linear bandits** (learned representation f_ψ(x) with linear bandit heads)
   - **NL-Bandit** as a neural **Q(x,a)** bandit on discrete action set
3. **Articulate trade-offs:**
   - Statistical efficiency vs expressivity
   - Interpretability vs raw performance
   - Stability and reproducibility concerns
4. **Recognize failure modes:**
   - Overfitting when data is scarce
   - Miscalibrated uncertainty
   - Representation collapse in joint training

### Mathematical Formalization

1. **Formalize Neural Linear architecture:**
   - $f_\psi: \mathcal{X} \to \mathbb{R}^d$ (non-linear feature map)
   - For each action $a$, $\theta_a \in \mathbb{R}^d$ with LinUCB/TS updates over $\phi(x) = f_\psi(x)$
   - Connect to EQ-6.17 and clarify split between representation and head
2. **Define neural Q(x,a) model for discrete actions:**
   - $Q_\theta(x,a) \approx \mathbb{E}[R | X=x, A=a]$ where $a \in \{0,\ldots,M-1\}$ indexes templates
   - Parameterizations: multi-head network vs full Q(x,a) with action features
3. **Reason qualitatively about:**
   - Function approximation error and representation error effects on regret
   - Sample complexity trade-offs: linear vs Neural Linear vs full neural Q(x,a)

### Implementation Skills

1. **Implement Neural Linear feature extractor in PyTorch:**
   - Train $f_\psi(x)$ on logged data (supervised regression)
   - Freeze and plug into existing `LinUCB`/`LinearThompsonSampling`
2. **Implement discrete-action NL-Bandit using Q-ensemble:**
   - Option A: One-hot action encoding with concat input
   - Option B: Multi-head architecture with shared body
3. **Understand code reuse across chapters:**
   - `LinUCB`/`ThompsonSampling` (Ch6) + neural features (Ch6A)
   - `QEnsembleRegressor` (Ch7) in discrete-action mode (Ch6A)

### Diagnostic & Experimental Skills

1. **Run comparative experiments:**
   - Hand-crafted features + LinUCB (Ch6 baseline)
   - Neural Linear (f_ψ + LinUCB)
   - NL-Bandit (discrete Q(x,a) with neural network)
2. **Diagnose failures:**
   - Detect overfitting via validation loss
   - Identify miscalibrated uncertainty (σ vs actual regret)
   - Recognize representation collapse
3. **Interpret results:**
   - When Neural Linear helps (nonlinearity + data)
   - When it hurts (scarce data, overparameterization)

---

## 3. Scope and Non-Goals

### In Scope

- **Contextual bandits** (γ = 0) with discrete template actions
- **Neural architectures at two levels:**
  - Representation learning (Neural Linear)
  - Full Q(x,a) for discrete actions (NL-Bandit)
- **Tight, code-linked narrative:**
  - PyTorch implementation for $f_\psi(x)$
  - Lightweight use of Q-ensemble machinery for discrete actions
- **Honest failure modes:**
  - Overfitting, miscalibration, when to avoid neural methods

### Out of Scope (Deferred)

- Full **continuous action** Q(x,a) with CEM optimization → **Chapter 7**
- **Offline RL** and pessimistic objectives → **Chapter 13**
- Slate-level neural ranking → **Chapter 12**
- Joint training of $f_\psi$ during bandit exploration (mentioned as advanced exercise only)

---

## 4. Proposed Chapter Structure

Working table of contents for `ch06a_neural_bandits.md`:

### 6A.0 Orientation — Why Neural Bandits?

**Content:**
- Short recap of Chapter 6:
  - Linear contextual bandits, discrete templates, feature poverty failure
- **Motivating example:**
  - High-dimensional representation: text embeddings, image features, rich user embeddings
  - Linear $\phi(x)$ cannot capture necessary structure
- **Statement of purpose:**
  - We want **more expressivity** than linear bandits, but retain:
    - Reasonable sample efficiency
    - Some interpretability/diagnostics
    - Clear bridge to continuous Q(x,a) (Chapter 7)

**Pedagogical strategy:**
- Use concrete example: "What if user segments are not one-hot but 100-dim embeddings from BERT?"
- Show simple 2D nonlinear reward where linear model fails (visual)

---

### 6A.1 Neural Linear Bandits (Representation Learning)

**Definition:**
- $f_\psi: \mathcal{X} \to \mathbb{R}^d$ learned by neural network
- $\phi(x) = f_\psi(x)$ used as features for LinUCB/Thompson Sampling

**Connection to EQ-6.17:**
```markdown
$$
\mu(x, a) = \theta_a^\top f_\psi(x)
\tag{6A.1}
$$
{#EQ-6A.1}
```

Clarify split:
- **Representation** $f_\psi$ learns nonlinear structure
- **Bandit head** $\theta_a$ retains linear model + confidence intervals

**Training $f_\psi$:**

**Supervised learning on logged data:**
- Input: $(x, a, r)$ tuples from baseline policy
- Objective: Minimize $(g_\psi(x,a) - r)^2$ or similar
- Derive $f_\psi(x)$ as intermediate representation

**Practical simplification:**
- Keep $f_\psi$ small (2 layers, 64 units) to make point
- Freeze after pretraining (joint training is advanced exercise)

**Implementation hooks:**
```python
# scripts/ch06a/neural_linear_demo.py
feature_extractor = NeuralFeatureExtractor(
    input_dim=100,  # Raw context dimension
    hidden_dim=64,
    output_dim=20   # Learned feature dimension
)

# Train on logged data
optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=1e-3)
for epoch in range(100):
    # ... supervised training loop ...

# Freeze and plug into LinUCB
feature_extractor.eval()
policy = NeuralLinearUCB(
    templates=templates,
    feature_extractor=feature_extractor,
    alpha=1.0,
    lambda_reg=1.0
)
```

**Code ↔ Agent boxes:**
- Torch feature extractor: `scripts/ch06a/neural_linear_demo.py`
- LinUCB/TS: `zoosim/policies/{lin_ucb,thompson_sampling}.py`
- Feature config: `zoosim/ranking/features.py`, `zoosim/core/config.py`

---

### 6A.2 Neural Linear in Zoosim (Integration)

**Pipeline:**
1. Generate logs with simple baseline policy (e.g., uniform random or CM2 Boost)
2. Train $f_\psi(x)$ on these logs (off-policy supervised)
3. Freeze $f_\psi$ and run new bandit experiment using neural features
4. Compare to Ch6 rich features

**Diagnostics:**
- **GMV, CM2, STRAT** metrics
- **Stability over seeds** (10 seeds)
- **Pretraining loss convergence** (validation split)
- **Feature visualization** (t-SNE of $f_\psi(x)$ colored by segment)

**Expected results:**
- **Success case** (nonlinearity + data): Neural Linear beats rich features by 5-10% GMV
- **Failure case** (scarce data): Neural Linear ≈ rich features or worse
- **Catastrophic case** (overparameterized + scarce): Neural Linear < simple features

---

### 6A.3 NL-Bandit as Neural Q(x,a) on Discrete Templates

**Definition:**
- Action set: discrete templates $a \in \{0,\ldots,M-1\}$ (same as Ch6)
- Neural Q(x,a) model:
$$
Q_\theta(x,a) \approx \mathbb{E}[R | X=x, A=a]
\tag{6A.2}
$$
{#EQ-6A.2}

**Implementation options:**

**Option A: Multi-head architecture**
```python
class MultiHeadQ(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_actions)
        ])

    def forward(self, x):
        h = self.body(x)
        return torch.stack([head(h).squeeze() for head in self.heads])
```

**Option B: Concatenated action encoding**
```python
class ConcatQ(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + n_actions, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, a_onehot):
        return self.net(torch.cat([x, a_onehot], dim=-1)).squeeze()
```

**Relation to Q-ensemble (Ch7 code):**
- Use `QEnsembleRegressor` for discrete actions via:
  - Encode actions as one-hot $a_{\text{onehot}} \in \mathbb{R}^M$
  - Input: concat(x, $a_{\text{onehot}}$)
- **For action selection:**
  - Loop over candidate actions $a$
  - Compute $Q(x,a)$ and $\sigma(x,a)$ (ensemble variance)
  - Choose $\arg\max_a [Q(x,a) + \beta \sigma(x,a)]$ (UCB)

**Algorithm 6A.1: NL-Bandit with Ensemble** {#ALG-6A.1}

```
Initialize: Ensemble of K networks {Q_θ^(k)}_{k=1}^K
For episode t = 1, ..., T:
    Observe context x_t
    For each template a in {0, ..., M-1}:
        Compute μ_a = mean_k Q_θ^(k)(x_t, a)
        Compute σ_a = std_k Q_θ^(k)(x_t, a)
        Compute UCB_a = μ_a + β·σ_a
    Select a_t = argmax_a UCB_a
    Apply template a_t, observe reward r_t
    Add (x_t, a_t, r_t) to replay buffer
    Update ensemble via gradient descent on MSE loss
```

---

### 6A.4 Neural Bandit Failure Modes and Calibration

**Discuss:**

1. **Overfitting in $f_\psi(x)$ when data is scarce:**
   - Symptom: Low training loss, high validation loss
   - Result: Poor generalization to bandit's exploration distribution
   - Fix: Regularization (dropout, weight decay), early stopping

2. **Miscalibrated uncertainty:**
   - **Single network with heuristic variance:**
     - σ ∝ inverse density (fails when network overconfident)
   - **Ensembles without regularization:**
     - Models collapse to similar predictions (low variance)
   - Fix: Use ensembles + proper training (different seeds, dropout)

3. **Interaction with exploration:**
   - **Overconfident predictions** → kill exploration → get stuck
   - **Underconfident predictions** → over-explore → slow convergence
   - Calibration check: Plot predicted σ vs actual regret

**Labs:**
- Lab 6A.3: Overfit diagnosis (small data, large network)
- Calibration plot: σ(x,a) vs |Q(x,a) - r| on held-out data

---

### 6A.5 Bridge to Continuous Q(x,a) (Chapter 7)

**Conceptual bridge:**
- **Ch06A:** Actions are discrete templates $a \in \{0,\ldots,7\}$
- **Ch07:** Actions are continuous boost vectors $a \in \mathbb{R}^K$, same Q(x,a) idea:
  - Q(x,a) modeled by neural ensemble
  - Actions optimized with CEM + trust regions

**Implementation bridge:**
- Discrete NL-Bandit (Ch06A) and continuous Q-ensemble (Ch07) **share code:**
  - `zoosim/policies/q_ensemble.py` (ensemble, uncertainty)
  - `zoosim/optimizers/cem.py` (only used in Ch07)

**Narrative bridge:**
- Ch06A is **optional but recommended** if reader cares about:
  - Neural architectures for bandits
  - How representation learning interacts with exploration
  - Understanding NL-Bandit baseline used in Ch13 (Offline RL)

---

### 6A.6 Summary & What's Next

**What we built:**
1. Neural Linear bandits (learned features + linear heads)
2. NL-Bandit (full neural Q(x,a) for discrete templates)
3. Diagnostic tools (overfit detection, calibration checks)

**Key lessons:**
1. Neural methods help when: nonlinearity + abundant data
2. Neural methods hurt when: scarce data, overparameterization, poor pretraining
3. Uncertainty calibration is critical for exploration
4. Bridge complete: Linear (Ch6) → Neural Linear (Ch6A) → Continuous Q (Ch7)

**Where to go next:**
- **Chapter 7:** Continuous actions, CEM, trust regions
- **Chapter 13:** Offline RL with NL-Bandit as baseline
- **Advanced:** Joint training, transfer learning, meta-learning (exercises)

---

## 5. Exercise Design (Complete)

### Design Principles (Vlad Prytula Style)

1. **Show success AND failure** — Neural methods help when there's nonlinearity + data, hurt when data is scarce
2. **Bridge three worlds** — Linear bandits (Ch6) → Neural Linear → NL-Bandit → Continuous Q (Ch7)
3. **Diagnostic thinking** — Teach how to detect overfitting, miscalibration, representation collapse
4. **Production-ready code** — Type hints, seeds, reproducibility from the start

---

### Theory Exercises (30 min total)

#### Exercise 6A.1: When Linear Features Fail (10 min)

**Problem:**

Consider a contextual bandit with context $x = (x_1, x_2) \in [0,1]^2$ and 2 actions $\{a_0, a_1\}$.

True reward functions:
$$
\mu(x, a_0) = x_1 \cdot x_2 \quad \text{(interaction term)}
$$
$$
\mu(x, a_1) = 0.3 + 0.1(x_1 + x_2)
$$

**(a)** Show that a **linear model** $\hat{\mu}(x,a) = \theta_a^\top [1, x_1, x_2]$ cannot perfectly represent $\mu(x, a_0)$.

**(b)** Propose a **hand-crafted feature** $\phi(x)$ that allows perfect linear representation.

**(c)** Explain why this motivates **learned representations** $f_\psi(x)$ for high-dimensional settings where hand-crafting $x_1 \cdot x_2$ is infeasible.

**Solution hints:**
- (a) Expand $x_1 \cdot x_2$ and show no linear combination of $[1, x_1, x_2]$ equals it
- (b) $\phi(x) = [1, x_1, x_2, x_1 \cdot x_2]$ includes interaction
- (c) In 100-dim space, $O(d^2)$ interactions are intractable; neural nets learn them implicitly

---

#### Exercise 6A.2: Neural Linear Posterior (15 min)

**Problem:**

In **Neural Linear bandits**, we have:
- Feature extractor $f_\psi: \mathcal{X} \to \mathbb{R}^d$ (neural network, frozen after pretraining)
- Linear heads $\theta_a \in \mathbb{R}^d$ updated via ridge regression

**(a)** Write the posterior update equations for $\theta_a$ given features $\phi_t = f_\psi(x_t)$, reward $r_t$, and regularization $\lambda$.

**(b)** Compare to **pure linear bandits** with hand-crafted $\phi(x)$:
- What assumption does Neural Linear make about $f_\psi$?
- When does freezing $f_\psi$ hurt performance?

**(c)** Sketch a **joint training** variant where $f_\psi$ is updated periodically. What are the risks?

**Solution:**
- (a) Same as [EQ-6.6]–[EQ-6.8] from Ch6 with $\phi(x) = f_\psi(x)$
- (b) Assumes $f_\psi$ captures relevant structure; hurts if pretraining is poor or task shifts
- (c) Risks: distribution shift (bandit explores different $x$ than pretraining), representation collapse, instability

---

#### Exercise 6A.3: Sample Complexity Trade-offs (5 min)

**Problem:**

Compare sample complexity for learning a good policy:
- **Linear bandit** with $d=17$ hand-crafted features (Ch6 rich features)
- **Neural Linear** with $d=20$ learned features from 2-layer MLP
- **NL-Bandit** with full neural Q(x,a)

Rank these by sample efficiency (data needed for convergence). Justify your ranking.

**Answer:**
1. **Linear bandit** (most efficient): $O(\sqrt{dT})$ regret, closed-form updates, stable
2. **Neural Linear** (middle): Pretraining cost + linear bandit efficiency after freezing
3. **NL-Bandit** (least efficient): Full neural training, no closed-form uncertainty, ensemble overhead

**But:** NL-Bandit may achieve **lower final regret** if reward is highly nonlinear and data is abundant.

---

### Implementation Exercises (40 min total)

#### Exercise 6A.4: Implement Neural Feature Extractor (20 min)

**Problem:**

Implement a simple 2-layer MLP feature extractor in PyTorch:

```python
import torch
import torch.nn as nn

class NeuralFeatureExtractor(nn.Module):
    """Neural network for representation learning.

    Maps raw context x to learned features f_ψ(x).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # TODO: Implement 2-layer MLP
        # input_dim → hidden_dim (ReLU) → hidden_dim (ReLU) → output_dim
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute learned features f_ψ(x).

        Args:
            x: Raw context, shape (batch, input_dim)

        Returns:
            features: Learned representation, shape (batch, output_dim)
        """
        # TODO: Implement forward pass
        pass
```

**(a)** Complete the implementation.

**(b)** Train it on synthetic data where true reward is $r = \sin(2\pi x_1) + 0.5 x_2 + \epsilon$. Use MSE loss.

**(c)** Verify that learned features $f_\psi(x)$ capture the nonlinearity better than raw $x$.

**Test:**
```python
# Generate synthetic data
n_samples = 1000
x = torch.rand(n_samples, 5)  # 5-dim input
r = torch.sin(2 * torch.pi * x[:, 0]) + 0.5 * x[:, 1] + 0.1 * torch.randn(n_samples)

# Train feature extractor
model = NeuralFeatureExtractor(input_dim=5, hidden_dim=32, output_dim=10)
# ... training loop ...

# Compare: Linear model on raw x vs learned f_ψ(x)
```

---

#### Exercise 6A.5: Integrate Neural Features with LinUCB (20 min)

**Problem:**

Extend the Ch6 `LinUCB` to accept a **frozen feature extractor**:

```python
from zoosim.policies.lin_ucb import LinUCB
from typing import Optional, Callable
import numpy as np

class NeuralLinearUCB(LinUCB):
    """LinUCB with learned neural features.

    Uses f_ψ(x) instead of hand-crafted φ(x).
    """
    def __init__(
        self,
        templates: List[BoostTemplate],
        feature_extractor: Optional[Callable] = None,  # f_ψ(x)
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        seed: int = 42
    ):
        self.feature_extractor = feature_extractor
        # Determine feature_dim from extractor output
        if feature_extractor is not None:
            dummy_input = ...  # TODO: How to infer dim?
            feature_dim = len(feature_extractor(dummy_input))
        else:
            feature_dim = ???  # Use hand-crafted dim

        super().__init__(templates, feature_dim, alpha, lambda_reg, seed)

    def _extract_features(self, context: dict) -> np.ndarray:
        """Extract features using neural network or fallback."""
        if self.feature_extractor is not None:
            # TODO: Convert context → tensor → f_ψ(x) → numpy
            pass
        else:
            # Fallback to hand-crafted features
            return super()._extract_features(context)
```

**(a)** Complete the implementation.

**(b)** Test on `ZooplusSearchEnv` with:
- Pretrained $f_\psi$ (from Exercise 6A.4)
- Ch6 rich features (baseline)

**(c)** Report GMV comparison over 10k episodes.

**Expected result:** Neural features should match or slightly beat rich features if nonlinearity exists.

---

### Labs (80 min total)

#### Lab 6A.1: Neural Linear vs Rich Features vs Simple Features (30 min)

**Objective:** Reproduce Ch6 failure-diagnosis-fix arc with **Neural Linear** as third option.

**Procedure:**

1. **Baseline:** Ch6 simple features ($d=7$, segment + query type)
2. **Rich features:** Ch6 rich features ($d=17$, + user latents + product aggregates)
3. **Neural Linear:** Pretrain $f_\psi$ on 5k logged episodes, freeze, use with LinUCB ($d=20$)

**Run:**
```bash
python scripts/ch06a/neural_linear_demo.py \
    --n-pretrain 5000 \
    --n-bandit 20000 \
    --feature-dim 20 \
    --hidden-dim 64
```

**Analysis:**
- Plot GMV curves for all 3 methods
- Identify segments where Neural Linear helps most
- Check: Does Neural Linear overfit when pretrain data is scarce (try `--n-pretrain 500`)?

**Expected result:**
- Simple features: ~4.6 GMV (Ch6 baseline)
- Rich features: ~7.1 GMV (Ch6 success)
- Neural Linear: ~7.3–7.8 GMV (slight improvement if nonlinearity captured)
- **Failure mode:** With `--n-pretrain 500`, Neural Linear may degrade to ~5.5 GMV (overfitting)

**Deliverable:**
- GMV curves plot (3 lines)
- Per-segment breakdown table
- 2-paragraph analysis of when Neural Linear helps/hurts

---

#### Lab 6A.2: NL-Bandit with Discrete Q(x,a) (30 min)

**Objective:** Implement full neural Q(x,a) for discrete templates using ensemble.

**Procedure:**

1. Use `QEnsembleRegressor` from `zoosim/policies/q_ensemble.py`
2. For each template $a \in \{0, \ldots, 7\}$:
   - Encode as one-hot $a_{\text{onehot}} \in \mathbb{R}^8$
   - Input to ensemble: $[x, a_{\text{onehot}}]$
   - Train on $(x, a, r)$ tuples from bandit exploration
3. Action selection: $\arg\max_a \text{UCB}(x, a)$ where $\text{UCB} = \mu(x,a) + \beta \sigma(x,a)$

**Run:**
```bash
python scripts/ch06a/neural_bandits_demo.py \
    --model nl-bandit \
    --ensemble-size 5 \
    --ucb-beta 2.0
```

**Analysis:**
- Compare to Neural Linear and rich features
- Check ensemble variance: Does it shrink over time?
- Calibration plot: Predicted $\sigma(x,a)$ vs actual regret

**Expected result:**
- NL-Bandit: ~7.5–8.5 GMV (competitive with TS from Ch6)
- Ensemble variance should decrease as $O(1/\sqrt{T})$
- **Failure mode:** Single network (no ensemble) may be overconfident → poor exploration

**Deliverable:**
- GMV comparison table
- Ensemble variance over time plot
- Calibration scatter plot (σ vs regret)

---

#### Lab 6A.3: Overfit Diagnosis — Small Data, Large Network (20 min)

**Objective:** Show when Neural Bandits fail.

**Procedure:**

Run Lab 6A.1 with three scenarios:
- **Scenario A (Normal):** $N_{\text{pretrain}} = 5000$, hidden=64
- **Scenario B (Data-scarce):** $N_{\text{pretrain}} = 500$, hidden=64
- **Scenario C (Overparameterized):** $N_{\text{pretrain}} = 500$, hidden=256

**Metrics:**
- GMV (online performance)
- Pretraining MSE (representation quality)
- Per-segment regret

**Expected findings:**
- **Scenario A:** Neural Linear wins (~7.5 GMV)
- **Scenario B:** Neural Linear ≈ Rich features (~7.0 GMV, not enough data to beat hand-crafted)
- **Scenario C:** Neural Linear < Simple features (~4.0 GMV, catastrophic overfitting)

**Deliverable:**
- Results table (3 scenarios × 3 metrics)
- 1-paragraph diagnosis: "When to avoid neural features"

---

### Advanced Exercises (Optional, 110+ min)

#### Exercise 6A.6: Joint Training — Representation + Bandit Head (30 min)

**Problem:**

Instead of **freezing** $f_\psi$ after pretraining, update it **jointly** with bandit exploration.

**Algorithm sketch:**
1. Maintain replay buffer $\mathcal{D} = \{(x_t, a_t, r_t)\}_{t=1}^T$
2. Every $K$ episodes:
   - Sample minibatch from $\mathcal{D}$
   - Update $f_\psi$ via gradient descent on $\mathcal{L}(\psi) = \sum (r - \theta_a^\top f_\psi(x))^2$
   - Recompute bandit statistics $A_a, b_a$ with new features

**Implementation:**
- Extend `NeuralLinearUCB` with `.update_representation()` method
- Add replay buffer

**Challenge:**
- How to handle **distribution shift**? (Bandit explores different $x$ than pretraining)
- **Representation collapse risk:** $f_\psi$ might collapse to constant features

**Test:** Run 50k episodes, updating $f_\psi$ every 1000 episodes. Does GMV improve or degrade?

---

#### Exercise 6A.7: Multi-Head vs Concatenated Architectures (20 min)

**Problem:**

Compare two architectures for discrete NL-Bandit:

**Architecture A (Multi-Head):**
```python
class MultiHeadQ(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_actions)
        ])

    def forward(self, x):
        h = self.body(x)
        return torch.stack([head(h) for head in self.heads])  # (n_actions,)
```

**Architecture B (Concatenated):**
```python
class ConcatQ(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + n_actions, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, a_onehot):
        return self.net(torch.cat([x, a_onehot], dim=-1))
```

**(a)** Implement both architectures.

**(b)** Train on same data, compare:
- Parameter count
- Training time
- Final GMV

**(c)** Which architecture is better for $M=8$ templates? For $M=100$?

---

#### Exercise 6A.8: Transfer Learning Across Segments (30 min)

**Problem:**

Can we pretrain $f_\psi$ on one user segment and transfer to another?

**Experiment:**
1. Pretrain on `premium` segment only (5k episodes)
2. Test Neural Linear on:
   - `premium` (in-distribution)
   - `pl_lover` (out-of-distribution)
   - `price_hunter` (OOD)

**Metrics:**
- GMV by segment
- Feature similarity: $\|f_\psi(x_{\text{premium}}) - f_\psi(x_{\text{pl\_lover}})\|_2$

**Expected result:**
- Transfer works if segments share structure (e.g., both care about relevance)
- Fails if preferences are orthogonal (e.g., premium wants high-price, price_hunter wants discounts)

---

#### Exercise 6A.9: Ensemble Size Ablation (15 min)

**Problem:**

Study effect of ensemble size on NL-Bandit performance.

**Procedure:**
Run Lab 6A.2 with ensemble sizes: 1, 3, 5, 10, 20

**Metrics:**
- Final GMV
- Calibration (σ vs regret correlation)
- Training time

**Expected findings:**
- Single network: Overconfident, poor exploration, lowest GMV
- Small ensemble (3-5): Good trade-off
- Large ensemble (20): Marginal gains, expensive

---

#### Exercise 6A.10: Open-Ended — Design Your Own Neural Bandit (30+ min)

**Problem:**

Design and implement a novel neural bandit architecture:

**Ideas:**
- **Attention mechanism** over action embeddings
- **Graph neural network** over product similarity graph
- **Recurrent features** that track user history within session
- **Hierarchical bandit** with meta-features for action groups

**Requirements:**
- Implement in PyTorch
- Test on ZooplusSearchEnv
- Compare to baselines (Linear, Neural Linear, NL-Bandit)
- Document when your architecture helps/hurts

---

### Time Allocation Summary

| Category | Time (min) | Exercises |
|----------|------------|-----------|
| **Theory** | 30 | 6A.1 (10), 6A.2 (15), 6A.3 (5) |
| **Implementation** | 40 | 6A.4 (20), 6A.5 (20) |
| **Labs** | 80 | Lab 6A.1 (30), Lab 6A.2 (30), Lab 6A.3 (20) |
| **Advanced (Optional)** | 135+ | 6A.6 (30), 6A.7 (20), 6A.8 (30), 6A.9 (15), 6A.10 (30+) |
| **Total (Core)** | 150 | |
| **Total (with Advanced)** | 285+ | |

**Recommended paths:**
- **Minimal (90 min):** Theory 6A.1-6A.3, Impl 6A.4, Lab 6A.1
- **Standard (150 min):** All core exercises + Labs 6A.1-6A.3
- **Deep dive (285 min):** Core + Advanced 6A.6-6A.10

---

## 6. Code & File Plan

### Code Modules (Reuse existing)

**From Ch6:**
- `zoosim/policies/lin_ucb.py` — Bandit head for Neural Linear
- `zoosim/policies/thompson_sampling.py` — Bayesian alternative with neural features

**From Ch7:**
- `zoosim/policies/q_ensemble.py` — Core Q-ensemble implementation used for:
  - Discrete-action NL-Bandit (Ch6A)
  - Continuous Q(x,a) policy (Ch7)

**From simulator:**
- `zoosim/ranking/features.py` — Feature extraction
- `zoosim/core/config.py` — Configuration
- `zoosim/envs/search_env.py` — Environment

### New Scripts (To be created)

**`scripts/ch06a/neural_linear_demo.py`**
- Train small $f_\psi(x)$ on logs
- Plug $f_\psi(x)$ into LinUCB/TS
- Compare to Ch6 rich features

**`scripts/ch06a/neural_bandits_demo.py`**
- Use `QEnsembleRegressor` on discrete actions (NL-Bandit)
- Compare to:
  - Purely linear bandits (Ch6)
  - Neural Linear bandits (this chapter)

**`scripts/ch06a/calibration_check.py`**
- Plot predicted σ(x,a) vs actual regret
- Detect miscalibration

### Tests (Planned, optional)

**`tests/ch06a/test_neural_linear.py`**
- Basic check: Neural Linear improves synthetic nonlinear bandit
- Regression test: Ensure feature extractor produces correct dims

**`tests/ch06a/test_discrete_q_ensemble.py`**
- Check: Enumerating discrete actions via `QEnsembleRegressor` yields sensible behavior
- Verify: Ensemble variance shrinks over time

---

## 7. Knowledge Graph & Cross-References

### Nodes to Add to `graph.yaml`

**Chapter node:**
```yaml
- id: CH-6A
  kind: chapter
  title: Chapter 6A — Neural Bandits: From Neural Linear to NL-Bandit
  status: planned
  file: docs/book/drafts/ch06a/ch06a_neural_bandits.md
  summary: Neural feature learning and neural Q(x,a) for discrete template bandits; bridge from linear (Ch6) to continuous (Ch7).
  depends_on: [CH-5, CH-6]
  forward_refs: [CH-7, CH-13]
  defines: [EQ-6A.1, EQ-6A.2, ALG-6A.1]
```

**Equation nodes:**
```yaml
- id: EQ-6A.1
  kind: equation
  title: Neural Linear bandit architecture
  status: planned
  file: docs/book/drafts/ch06a/ch06a_neural_bandits.md
  anchor: "{#EQ-6A.1}"
  summary: μ(x,a) = θ_a^T f_ψ(x) with learned representation f_ψ and linear heads θ_a
  depends_on: [CH-6, EQ-6.17]

- id: EQ-6A.2
  kind: equation
  title: Neural Q(x,a) for discrete actions (NL-Bandit)
  status: planned
  file: docs/book/drafts/ch06a/ch06a_neural_bandits.md
  anchor: "{#EQ-6A.2}"
  summary: Q_θ(x,a) ≈ E[R|X=x,A=a] with neural network Q_θ
  depends_on: [CH-6, CH-6A, EQ-6A.1]
```

**Algorithm node:**
```yaml
- id: ALG-6A.1
  kind: algorithm
  title: NL-Bandit with Ensemble
  status: planned
  file: docs/book/drafts/ch06a/ch06a_neural_bandits.md
  anchor: "{#ALG-6A.1}"
  summary: Neural Q(x,a) bandit with ensemble uncertainty for discrete template selection
  depends_on: [ALG-6.1, ALG-6.2, CH-6A]
  implements: [EQ-6A.2]
```

### Cross-Reference Updates

**Update EQ-6.17:**
```yaml
- id: EQ-6.17
  kind: equation
  title: Neural Linear bandit architecture
  status: complete
  file: docs/book/drafts/ch06/appendices.md
  anchor: "{#EQ-6.17}"
  summary: μ(x,a) = θ_a^T f_ψ(x) with learned representation f_ψ and linear heads θ_a
  depends_on: [CH-6, CH-7]
  see_also: [CH-6A, EQ-6A.1]  # Add cross-reference
```

**Update MOD-zoosim.policies.q_ensemble:**
```yaml
- id: MOD-zoosim.policies.q_ensemble
  kind: module
  title: Q-Ensemble for continuous and discrete actions
  status: complete
  file: zoosim/policies/q_ensemble.py
  summary: Ensemble of neural networks for Q(x,a) regression with uncertainty quantification
  implements: [EQ-6A.2, EQ-7.1]  # Add Ch6A equation
  used_by: [CH-6A, CH-7, CH-13]  # Add Ch6A reference
  see_also: [EQ-6.17, ALG-6A.1]
```

---

## 8. Syllabus Integration

**Updated entry (already applied to `syllabus.md`):**

```markdown
6A) Chapter 6A — Neural Bandits: From Neural Linear to NL-Bandit (optional bridge chapter)
- Objectives: Learn features with neural networks; bridge linear bandits (Ch6) → continuous Q(x,a) (Ch7).
- Theory: Neural Linear architecture (f_ψ + linear heads); discrete NL-Bandit (neural Q(x,a) on templates); sample complexity trade-offs.
- Implementation: `scripts/ch06a/{neural_linear_demo,neural_bandits_demo}.py`; reuse `policies/{lin_ucb,thompson_sampling,q_ensemble}.py`.
- Labs: Neural Linear vs rich features vs simple features; NL-Bandit ensemble calibration; overfit diagnosis (small data, large network); transfer learning across segments.
- Acceptance: Neural Linear beats rich features when nonlinearity exists + data is abundant (target: +5-10% GMV); degrades gracefully when data is scarce (honest failure modes documented); ensemble uncertainty calibrated (σ correlates with regret).
```

---

## 9. Next Steps for Drafting

### Phase 1: Draft Core Narrative (Priority 1)

1. **Start scaffolding** `ch06a_neural_bandits.md`:
   - Use section structure from §4 (6A.0 - 6A.6)
   - Write in Vlad Application Mode voice
   - Tight math-code integration throughout

2. **Migrate Neural Linear content** from Ch6 Appendix 6.A:
   - Move `docs/book/drafts/ch06/appendices.md` § Appendix 6.A → Ch6A §6A.1
   - Expand with full theory, algorithm, implementation
   - Add cross-reference in Appendix 6.A: "See Chapter 6A for complete treatment"

3. **Write §6A.0-6A.3** (orientation + Neural Linear + NL-Bandit theory)
   - Target: 1500-2000 lines
   - Include all definitions, equations, algorithms
   - Add "Code ↔ X" boxes linking to implementations

### Phase 2: Implement Labs (Priority 2)

4. **Create `scripts/ch06a/neural_linear_demo.py`:**
   - Pretrain feature extractor
   - Integrate with LinUCB
   - Run Lab 6A.1 experiments

5. **Create `scripts/ch06a/neural_bandits_demo.py`:**
   - Discrete NL-Bandit using q_ensemble
   - Run Lab 6A.2 experiments

6. **Verify experiments reproduce expected results:**
   - Neural Linear: +5-10% GMV in success case
   - NL-Bandit: Competitive with TS from Ch6
   - Failure modes: Documented and reproducible

### Phase 3: Write Exercises & Labs Section (Priority 3)

7. **Create `docs/book/drafts/ch06a/exercises_labs.md`:**
   - Copy exercise design from §5 of this plan
   - Add solution sketches
   - Link to lab scripts

8. **Add time allocation table** (same format as Ch6)

### Phase 4: Integration & Polish (Priority 4)

9. **Update cross-references:**
   - Ch6 → Ch6A forward reference in §6.8.4
   - Ch6A → Ch7 bridge in §6A.5
   - Ch13 → Ch6A (NL-Bandit baseline)

10. **Update Knowledge Graph:**
    - Add CH-6A, EQ-6A.1, EQ-6A.2, ALG-6A.1 nodes
    - Update EQ-6.17, MOD-zoosim.policies.q_ensemble

11. **Run validation:**
    ```bash
    source .venv/bin/activate
    python docs/knowledge_graph/validate_kg.py
    ```

12. **Production checklist** (add to §6A.6):
    - Neural feature extractor guidelines
    - NL-Bandit ensemble best practices
    - When to avoid neural methods

---

## Production Checklist for Ch6A

```markdown
!!! tip "Production Checklist (Chapter 6A)"
    **Neural Feature Extractor:**
    - ✅ Pretrain on **logged data** (not online exploration)
    - ✅ Use **separate validation set** to detect overfitting
    - ✅ **Freeze** after pretraining (don't update during bandit unless advanced)
    - ✅ Sanity check: Validation MSE < threshold before deployment

    **NL-Bandit Q(x,a):**
    - ✅ Use **ensemble** (≥3 networks) for uncertainty quantification
    - ✅ Calibration check: $\sigma(x,a)$ correlates with true regret on held-out data
    - ✅ UCB coefficient $\beta$: Tune on held-out seeds (typical range: 1.0-3.0)
    - ✅ Monitor ensemble variance: Should decrease over time

    **When to avoid neural methods:**
    - ❌ Data < 1000 episodes → Stick to linear bandits (Ch6)
    - ❌ Feature dim < 50 → Hand-crafted features likely better
    - ❌ Production constraints (latency < 10ms, interpretability critical) → Use Ch6 templates
    - ❌ Reward is nearly linear → Neural complexity not justified

    **Monitoring in production:**
    - Track pretraining loss convergence (early stopping criterion)
    - Monitor ensemble variance over time (should decrease as $O(1/\sqrt{T})$)
    - Alert if GMV < Ch6 baseline for 1000+ consecutive episodes
    - Compare to Ch6 rich features as sanity check

    **Reproducibility:**
    - Set seeds for: PyTorch, NumPy, environment, data shuffling
    - Log hyperparameters: learning rate, hidden dims, ensemble size, β
    - Version feature extractor weights (git-lfs or artifact store)
```

---

## Summary: Ready to Draft

**This plan is comprehensive and ready for implementation.**

**Key decisions made:**
1. ✅ Ch6A is optional bridge chapter between Ch6 and Ch7
2. ✅ Covers Neural Linear + NL-Bandit (discrete templates only)
3. ✅ Reuses existing code (`lin_ucb`, `thompson_sampling`, `q_ensemble`)
4. ✅ Complete exercise design (150 min core + 135 min advanced)
5. ✅ Syllabus updated with acceptance criteria
6. ✅ Knowledge Graph nodes specified
7. ✅ Production checklist included

**Next action:** Begin drafting `ch06a_neural_bandits.md` starting with §6A.0-6A.1.

**Vlad Prytula here:** This chapter will show both the promise and peril of neural bandits—honest empiricism demands we show the failures alongside the successes. The bridge from linear (Ch6) → neural (Ch6A) → continuous (Ch7) is now pedagogically complete.

---

**End of Planning Document**
