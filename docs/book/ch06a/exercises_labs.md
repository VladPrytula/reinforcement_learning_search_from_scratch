# Chapter 6A — Exercises & Labs

**Context:** Neural Bandits (Neural Linear & NL-Bandit)
**Prerequisites:** Chapter 6 (Discrete Template Bandits)
**Code:** `scripts/ch06a/`

---

## Table of Contents

- [Theory Exercises](#theory-exercises)
- [Implementation Exercises](#implementation-exercises)
- [Labs](#labs)
- [Advanced Exercises](#advanced-exercises)

---

## Theory Exercises

### Exercise 6A.1: When Linear Features Fail (10 min)

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

### Exercise 6A.2: Neural Linear Posterior (15 min)

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

### Exercise 6A.3: Sample Complexity Trade-offs (5 min)

**Problem:**

Compare sample complexity for learning a good policy:
- **Linear bandit** with $d=17$ hand-crafted features (Ch6 rich features)
- **Neural Linear** with $d=20$ learned features from 2-layer MLP
- **NL-Bandit** with full neural Q(x,a)

Rank these by sample efficiency (data needed for convergence). Justify your ranking.

**Answer:**
1. **Linear bandit** (most efficient): $O(d\sqrt{T})$ regret by Theorem 6.1 (Chapter 6), closed-form updates, stable
2. **Neural Linear** (middle): Pretraining cost + linear bandit efficiency after freezing
3. **NL-Bandit** (least efficient): Full neural training, no closed-form uncertainty, ensemble overhead

**But:** NL-Bandit may achieve **lower final regret** if reward is highly nonlinear and data is abundant.

---

## Implementation Exercises

### Exercise 6A.4: Implement Neural Feature Extractor (20 min)

**Problem:**

Implement a simple 2-layer MLP feature extractor in PyTorch. (See `scripts/ch06a/neural_linear_demo.py` for reference solution).

```python
import torch
import torch.nn as nn

class NeuralFeatureExtractor(nn.Module):
    """Neural network for representation learning."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # TODO: Implement 2-layer MLP
        # input_dim → hidden_dim (ReLU) → hidden_dim (ReLU) → output_dim
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute learned features f_ψ(x)."""
        # TODO: Implement forward pass
        pass
```

**(a)** Complete the implementation.

**(b)** Train it on synthetic data where true reward is $r = \sin(2\pi x_1) + 0.5 x_2 + \epsilon$. Use MSE loss.

**(c)** Verify that learned features $f_\psi(x)$ capture the nonlinearity better than raw $x$.

---

### Exercise 6A.5: Integrate Neural Features with LinUCB (20 min)

**Problem:**

Extend the Ch6 `LinUCB` to accept a **frozen feature extractor**. (See `scripts/ch06a/neural_linear_demo.py` for reference solution).

```python
from zoosim.policies.lin_ucb import LinUCB
from typing import Optional, Callable
import numpy as np

class NeuralLinearUCB(LinUCB):
    """LinUCB with learned neural features."""
    def __init__(
        self,
        templates: list,
        feature_extractor: Optional[Callable] = None,  # f_ψ(x)
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        seed: int = 42
    ):
        # TODO: Initialize parent LinUCB with neural feature dimension
        pass

    def _extract_features(self, context: dict) -> np.ndarray:
        """Extract features using neural network or fallback."""
        # TODO: Implement extraction
        pass
```

**(a)** Complete the implementation.

**(b)** Test on `ZooplusSearchEnv` with pretrained $f_\psi$.

**(c)** Report GMV comparison over 10k episodes.

---

## Labs

### Lab 6A.1: Neural Linear vs Rich Features vs Simple Features (30 min)

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
    --hidden-dim 64
```

**Analysis:**
- Plot GMV curves for all 3 methods
- Identify segments where Neural Linear helps most
- Check: Does Neural Linear overfit when pretrain data is scarce (try `--n-pretrain 500`)?

---

### Lab 6A.2: NL-Bandit with Discrete Q(x,a) (30 min)

**Objective:** Implement full neural Q(x,a) for discrete templates using ensemble.

**Procedure:**

1. Use `QEnsembleRegressor` from `zoosim/policies/q_ensemble.py`
2. For each template $a 
in \{0, \ldots, 7\}$:
   - Encode as one-hot $a_{\text{onehot}} \in \mathbb{R}^8$
   - Input to ensemble: $[x, a_{\text{onehot}}]$
   - Train on $(x, a, r)$ tuples from bandit exploration
3. Action selection: $\arg\max_a \text{UCB}(x, a)$ where $\text{UCB} = \mu(x,a) + \beta \sigma(x,a)$

**Run:**
```bash
python scripts/ch06a/neural_bandits_demo.py \
    --ensemble-size 5 \
    --ucb-beta 2.0
```

**Analysis:**
- Compare to Neural Linear and rich features
- Check ensemble variance: Does it shrink over time?
- Calibration plot: Predicted $\sigma(x,a)$ vs prediction error $|r - \mu(x,a)|$

---

### Lab 6A.3: Overfit Diagnosis — Small Data, Large Network (20 min)

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

**Deliverable:**
- Results table (3 scenarios × 3 metrics)
- 1-paragraph diagnosis: "When to avoid neural features"

---

## Advanced Exercises

### Exercise 6A.6: Joint Training — Representation + Bandit Head (30 min)

**Problem:** Instead of **freezing** $f_\psi$ after pretraining, update it **jointly** with bandit exploration. (See Ch6A plan for algorithm sketch).

### Exercise 6A.7: Multi-Head vs Concatenated Architectures (20 min)

**Problem:** Compare two architectures for discrete NL-Bandit (Multi-Head vs Concatenated).

### Exercise 6A.8: Transfer Learning Across Segments (30 min)

**Problem:** Can we pretrain $f_\psi$ on one user segment and transfer to another?

### Exercise 6A.9: Ensemble Size Ablation (15 min)

**Problem:** Study effect of ensemble size on NL-Bandit performance.

### Exercise 6A.10: Open-Ended — Design Your Own Neural Bandit (30+ min)

**Problem:** Design and implement a novel neural bandit architecture.

```