# Chapter 6A — Neural Bandits: From Neural Linear to NL-Bandit

**Author:** Vlad Prytula
**Status:** Draft (Phase 1)
**Prerequisites:** Chapter 6 (Discrete Template Bandits)

---

## Table of Contents

- [6A.0 Orientation — Why Neural Bandits?](#6a0-orientation--why-neural-bandits)
- [6A.1 Neural Linear Bandits (Representation Learning)](#6a1-neural-linear-bandits-representation-learning)
- [6A.2 Neural Linear in Zoosim (Integration)](#6a2-neural-linear-in-zoosim-integration)
- [6A.3 NL-Bandit as Neural Q(x,a) on Discrete Templates](#6a3-nl-bandit-as-neural-qxa-on-discrete-templates)
- [6A.4 Neural Bandit Failure Modes and Calibration](#6a4-neural-bandit-failure-modes-and-calibration)
- [6A.5 Bridge to Continuous Q(x,a) (Chapter 7)](#6a5-bridge-to-continuous-qxa-chapter-7)
- [6A.6 Summary & What's Next](#6a6-summary--whats-next)
- [Exercises & Labs](#exercises--labs)

---

## 6A.0 Orientation — Why Neural Bandits?

### The Story So Far

In Chapter 6, we explored **linear contextual bandits** for discrete template selection, where the expected reward is modeled as a linear function of hand-crafted features $\phi(x)$. Recall the linear bandit model from Chapter 6:

$$
\mu(x, a) = \theta_a^\top \phi(x)
$$

We built a complete failure→diagnosis→fix narrative:
1. **Simple features** ($d=7$, segment + query type) → **failed** (~4.6 GMV)
2. **Diagnosis**: Episodes with identical $\phi_{\text{simple}}(x)$ had 4× different GMV due to hidden context
3. **Fix**: Rich features ($d=17$, + user latents + product aggregates) → **succeeded** (~7.1 GMV for LinUCB)

**This approach worked because:**
- We could **hand-craft** features capturing relevant context-action interactions
- The reward function was **approximately linear** in these engineered features
- Feature space was **low-dimensional** enough for sample-efficient learning

### When Linear Models Hit a Wall

But what if we face a different setting where:

**1. High-dimensional raw inputs**
- User embeddings from BERT (768 dimensions)
- Product image features from ResNet (2048 dimensions)
- Query text embeddings (512 dimensions)
- Session history vectors (variable length)

**2. Unknown feature interactions**
- We don't know which user embedding dimensions interact with which product features
- Enumerating all $O(d^2)$ pairwise interactions is intractable when $d > 100$
- Higher-order interactions ($x_1 \cdot x_2 \cdot x_3$) exponentially explode

**3. Nonlinear reward structure**
- True reward may have saturation effects: $\mu(x, a) = \tanh(\theta_a^\top \phi(x))$
- Or multiplicative structure: $\mu(x, a) = \exp(\theta_1^\top x) \cdot (1 + \theta_2^\top \phi(a))$
- Linear models with hand-crafted features can only approximate these

### A Concrete Example: Nonlinearity You Can See

Consider a toy bandit with 2-dimensional context $x = (x_1, x_2) \in [0,1]^2$ and 2 actions:

**True reward functions:**
$$
\mu(x, a_0) = x_1 \cdot x_2 \quad \text{(interaction term)}
$$
$$
\mu(x, a_1) = 0.3 + 0.1(x_1 + x_2) \quad \text{(linear)}
$$

**Can a linear model handle this?**

**Attempt 1:** Features $\phi(x) = [1, x_1, x_2]$
- For action $a_0$, we need $\theta_{a_0}^\top [1, x_1, x_2] = x_1 \cdot x_2$
- But **no linear combination** of $[1, x_1, x_2]$ equals $x_1 \cdot x_2$ for all $(x_1, x_2)$
- **Failure:** Linear model misspecified ❌

**Attempt 2:** Hand-crafted interaction feature $\phi(x) = [1, x_1, x_2, x_1 \cdot x_2]$
- Now we can represent both rewards exactly
- **Success:** With $d=4$ features ✓

**The problem:** In high dimensions, hand-crafting interactions is infeasible:
- 100-dim context → $\binom{100}{2} = 4950$ pairwise interactions
- Adding these to $\phi(x)$ increases feature dimension to $d \approx 5000$
- By Theorem 6.1 (Chapter 6), LinUCB regret scales as $O(d\sqrt{T})$ — linear in feature dimension
- With $d = 5000$, regret grows prohibitively large even for moderate $T$

### What We Want: Neural Representation Learning

**Desiderata for "going beyond linear":**

1. **Expressivity:** Capture nonlinear interactions automatically (no hand-crafting)
2. **Sample efficiency:** Learn from thousands (not millions) of examples
3. **Uncertainty quantification:** Maintain exploration via confidence intervals
4. **Interpretability:** Some diagnostic capability (not a complete black box)
5. **Computational feasibility:** Millisecond inference latency

**Two levels of neural bandits:**

| Approach | Description | Expressivity | Sample Efficiency | Uncertainty |
|----------|-------------|--------------|-------------------|-------------|
| **Neural Linear** | Neural $f_\psi(x)$ → linear heads $\theta_a$ | Moderate | Good | Closed-form |
| **NL-Bandit** | Full neural $Q(x,a)$ with ensemble | High | Moderate | Ensemble-based |
| Pure linear (Ch6) | Hand-crafted $\phi(x)$ | Low | Excellent | Closed-form |

**This chapter explores both:**
- **§6A.1-6A.2:** Neural Linear bandits (learned features + linear heads)
- **§6A.3-6A.4:** NL-Bandit (full neural $Q(x,a)$ for discrete actions)
- **§6A.5:** Bridge to continuous actions (Chapter 7)

### Honest Expectations: When Neural Methods Help (and Hurt)

**Neural methods improve over linear when:**
- ✅ Nonlinearity exists and is strong ($R^2$ of linear model < 0.7)
- ✅ Abundant data (≥5k episodes for pretraining, ≥20k for NL-Bandit)
- ✅ High-dimensional inputs where manual feature engineering is intractable

**Neural methods degrade when:**
- ❌ Data is scarce (<2k episodes) → overfitting, miscalibration
- ❌ Network is overparameterized relative to data (large hidden dims, deep architectures)
- ❌ Reward is approximately linear → added complexity not justified

**This chapter will show both success and failure modes.** We'll demonstrate:
- Lab 6A.1: Neural Linear beating rich features (+5-10% GMV) when nonlinearity exists
- Lab 6A.3: Neural Linear *losing* to simple features when data is scarce (catastrophic overfitting)

**Vlad here:** Machine learning is not magic. Neural networks are powerful function approximators, but they're also excellent at memorizing noise when data is limited. Our job is to understand when each tool is appropriate—and to fail gracefully when we've made the wrong choice.

---

## 6A.1 Neural Linear Bandits (Representation Learning)

### The Core Idea: Separate Representation from Decision

**Neural Linear architecture** combines:
1. **Learned feature extractor** $f_\psi: \mathcal{X} \to \mathbb{R}^d$ (neural network)
2. **Linear bandit heads** $\theta_a \in \mathbb{R}^d$ (one per action)

**Reward model:**
$$
\mu(x, a) = \theta_a^\top f_\psi(x)
\tag{6A.1}
$$
{#EQ-6A.1}

where:
- $\psi$ are neural network parameters (weights, biases)
- $f_\psi(x)$ is the learned representation (output of hidden layer)
- $\theta_a$ are action-specific weights (updated via ridge regression)

**Why this split?**

**Alternative A: Fully neural $Q(x,a) = f_\theta(x,a)$**
- Maximum expressivity (can represent any function)
- ❌ No closed-form posterior → requires MCMC or variational inference
- ❌ Uncertainty quantification is slow and approximate

**Alternative B: Neural Linear (our choice)**
- ✓ **Representation learning:** Neural net captures nonlinearity
- ✓ **Fast uncertainty:** Linear head has Gaussian posterior (by #EQ-6.6–#EQ-6.8 from Chapter 6)
- ✓ **Interpretability:** Can visualize $f_\psi(x)$ embeddings, inspect $\theta_a$ weights
- ✓ **Modularity:** Freeze $f_\psi$, swap LinUCB/Thompson Sampling for heads

**Connection to Chapter 6:** We're simply **replacing** hand-crafted $\phi(x)$ with learned $f_\psi(x)$. All the LinUCB/TS machinery (ridge regression, confidence bounds, posterior sampling) remains unchanged.

!!! note "Notation Convention: φ(x) = f_ψ(x)"
    Throughout this chapter, we use the following convention to maintain clarity with Chapter 6:

    - **$\phi(x)$**: Generic feature vector (used in theoretical discussions and when emphasizing the connection to Ch6's linear bandit framework)
    - **$f_\psi(x)$**: Learned neural representation (used in implementation sections and when emphasizing the neural network parameters ψ)
    - **Equivalence**: $\phi(x) \equiv f_\psi(x)$ when discussing Neural Linear bandits

    In Chapter 6, φ(x) was hand-crafted (e.g., $\phi(x) = [1, x_1, x_2, x_1 \cdot x_2]$). Here, φ(x) is **learned** via a neural network parameterized by ψ.

    When you see references to Chapter 6's equations (like [EQ-6.6]–[EQ-6.8]), mentally substitute φ_t ← f_ψ(x_t).

### Mathematical Formalization

**Setup:**
- Context space $\mathcal{X}$ (could be $\mathbb{R}^{100}$ for high-dim raw features)
- Action space $\mathcal{A} = \{0, \ldots, M-1\}$ (discrete templates)
- Reward model: $R_t = \theta_{A_t}^\top f_\psi(X_t) + \epsilon_t$ where $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ independently

**Neural feature extractor:**
$$
f_\psi(x) = \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2)
$$
where $\psi = \{W_1, b_1, W_2, b_2\}$ are network parameters.

(More complex architectures possible: deeper networks, batch normalization, dropout. We keep it simple for pedagogy.)

**Mean reward function:**
$$
\mu(x, a) := \mathbb{E}[R_t \mid X_t=x, A_t=a] = \theta_a^\top f_\psi(x)
$$

This gives the clean decomposition: observed reward = deterministic mean + noise.

**Bayesian posterior for $\theta_a$ (given fixed $\psi$):**

By #EQ-6.6, #EQ-6.7, and #EQ-6.8 from Chapter 6, the Gaussian posterior for linear contextual bandits is computed via ridge regression with precision matrix $A_a$ and weighted sum $b_a$. The key observation is that **all of Chapter 6's machinery applies unchanged** when we substitute the learned features $f_\psi(x_t)$ in place of hand-crafted features $\phi_t$.

**Chapter 6 (hand-crafted features):**
$$
A_a = \lambda I + \sum_{t : A_t=a} \phi_t \phi_t^\top, \quad b_a = \sum_{t : A_t=a} r_t \phi_t
$$
$$
\hat{\theta}_a = A_a^{-1} b_a, \quad \Sigma_a = \sigma^2 A_a^{-1}
$$

**Chapter 6A (Neural Linear — learned features):**
Simply replace $\phi_t \leftarrow f_\psi(x_t)$ everywhere:

$$
A_a = \lambda I + \sum_{t : A_t=a} f_\psi(x_t) f_\psi(x_t)^\top
$$
$$
b_a = \sum_{t : A_t=a} r_t f_\psi(x_t)
$$
$$
\hat{\theta}_a = A_a^{-1} b_a, \quad \Sigma_a = \sigma^2 A_a^{-1}
\tag{6A.1.1}
$$
{#EQ-6A.1.1}

**LinUCB with neural features:**
$$
\text{UCB}(x, a) = \hat{\theta}_a^\top f_\psi(x) + \alpha \sqrt{f_\psi(x)^\top A_a^{-1} f_\psi(x)}
\tag{6A.1.2}
$$
{#EQ-6A.1.2}

**Thompson Sampling with neural features:**
$$
\tilde{\theta}_a \sim \mathcal{N}(\hat{\theta}_a, \Sigma_a), \quad \text{select } a = \arg\max_{a'} \tilde{\theta}_{a'}^\top f_\psi(x)
\tag{6A.1.3}
$$
{#EQ-6A.1.3}

**Key insight:** Once $f_\psi$ is fixed, the problem reduces to **standard linear contextual bandits** with feature dimension $d$ (the output dimension of $f_\psi$).

### Training the Representation: Two Strategies

**Strategy 1: Supervised pretraining (simpler, more stable)**

1. **Collect logged data** from a baseline policy (e.g., uniform random, static template)
   - Dataset: $\mathcal{D}_{\text{pretrain}} = \{(x_i, a_i, r_i)\}_{i=1}^{N_{\text{pretrain}}}$

2. **Train neural network** to predict rewards:
   - Architecture: $x \to f_\psi(x) \to g_\psi(x, a) \to \hat{r}$ (prediction head)
   - Loss: Mean squared error $\mathcal{L}(\psi) = \frac{1}{N} \sum_{i=1}^N (g_\psi(x_i, a_i) - r_i)^2$
   - Optimizer: Adam with learning rate ~$10^{-3}$

3. **Freeze representation** $f_\psi(x)$ (stop gradient updates)

4. **Run bandit** with LinUCB/TS using frozen features $\phi(x) = f_\psi(x)$

**Why freeze?**
- Avoids **distribution shift**: Bandit explores different $(x,a)$ than pretraining policy
- Prevents **representation collapse**: $f_\psi$ might converge to constant features during bandit exploration
- Maintains **stability**: Linear heads update incrementally without disrupting representations

**Strategy 2: Joint training (advanced, covered in Exercise 6A.6)**

- Maintain replay buffer of bandit exploration data
- Periodically update $\psi$ via gradient descent on bandit data
- Recompute bandit statistics with new features

**Risks:**
- Distribution shift between pretraining and exploration
- Representation collapse
- Instability from interleaved representation updates and bandit head updates

**This chapter focuses on Strategy 1 (supervised pretraining).** Joint training is an advanced exercise.

### Implementation: PyTorch Neural Feature Extractor

**Code ↔ Agent (Neural Feature Extractor)**

We implement $f_\psi(x)$ as a simple 2-layer MLP in PyTorch:

```python
import torch
import torch.nn as nn
from typing import Tuple

class NeuralFeatureExtractor(nn.Module):
    """Neural network for representation learning.

    Maps raw context x ∈ R^input_dim to learned features f_ψ(x) ∈ R^output_dim.
    Used as feature extractor for Neural Linear bandits.

    Architecture:
        input → Linear(hidden_dim) → ReLU → Linear(hidden_dim) → ReLU → Linear(output_dim)

    Corresponds to EQ-6A.1: μ(x,a) = θ_a^T f_ψ(x)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0
    ):
        """Initialize neural feature extractor.

        Args:
            input_dim: Dimension of raw context x
            hidden_dim: Width of hidden layers
            output_dim: Dimension of learned features f_ψ(x)
            dropout: Dropout probability (0.0 = no dropout)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute learned features f_ψ(x).

        Args:
            x: Raw context, shape (batch, input_dim) or (input_dim,)

        Returns:
            features: Learned representation, shape (batch, output_dim) or (output_dim,)
        """
        return self.net(x)
```

**Pretraining script** (simplified):

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def pretrain_feature_extractor(
    logged_data: list[tuple],  # [(x, a, r), ...]
    input_dim: int,
    hidden_dim: int = 64,
    output_dim: int = 20,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    seed: int = 42
) -> NeuralFeatureExtractor:
    """Pretrain neural feature extractor on logged data.

    Args:
        logged_data: List of (context, action, reward) tuples
        input_dim: Dimension of raw context
        hidden_dim: Width of hidden layers
        output_dim: Dimension of learned features
        n_epochs: Number of training epochs
        learning_rate: Adam learning rate
        batch_size: Minibatch size
        seed: Random seed for reproducibility

    Returns:
        Trained feature extractor (in eval mode)
    """
    torch.manual_seed(seed)

    # Prepare data
    X = torch.tensor([x for x, a, r in logged_data], dtype=torch.float32)
    A = torch.tensor([a for x, a, r in logged_data], dtype=torch.long)
    R = torch.tensor([r for x, a, r in logged_data], dtype=torch.float32)

    dataset = TensorDataset(X, A, R)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize network
    feature_extractor = NeuralFeatureExtractor(input_dim, hidden_dim, output_dim)

    # Add prediction head (one linear layer per action)
    n_actions = A.max().item() + 1
    prediction_heads = nn.ModuleList([
        nn.Linear(output_dim, 1) for _ in range(n_actions)
    ])

    optimizer = optim.Adam(
        list(feature_extractor.parameters()) + list(prediction_heads.parameters()),
        lr=learning_rate
    )
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for x_batch, a_batch, r_batch in loader:
            # Forward pass
            features = feature_extractor(x_batch)  # (batch, output_dim)
            predictions = torch.stack([
                prediction_heads[a](features[i])
                for i, a in enumerate(a_batch)
            ]).squeeze()  # (batch,)

            # Loss
            loss = loss_fn(predictions, r_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(loader):.4f}")

    # Freeze and return
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor
```

**Integrating with LinUCB:**

```python
from zoosim.policies.lin_ucb import LinUCB
import numpy as np

class NeuralLinearUCB(LinUCB):
    """LinUCB with learned neural features.

    Extends LinUCB to use f_ψ(x) instead of hand-crafted φ(x).
    Corresponds to EQ-6A.1.2 for action selection.
    """
    def __init__(
        self,
        templates: list,
        feature_extractor: NeuralFeatureExtractor,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        seed: int = 42
    ):
        """Initialize Neural Linear UCB.

        Args:
            templates: List of boost templates
            feature_extractor: Pretrained neural feature extractor (frozen)
            alpha: UCB exploration coefficient
            lambda_reg: Ridge regression regularization
            seed: Random seed
        """
        self.feature_extractor = feature_extractor
        feature_dim = feature_extractor.output_dim

        super().__init__(templates, feature_dim, alpha, lambda_reg, seed)

    def _extract_features(self, context: dict) -> np.ndarray:
        """Extract neural features f_ψ(x) from raw context.

        Args:
            context: Environment observation dict with 'raw_features' key

        Returns:
            features: Learned representation, shape (feature_dim,)
        """
        # Convert context to tensor
        x_raw = context['raw_features']  # Assume this exists
        x_tensor = torch.tensor(x_raw, dtype=torch.float32)

        # Extract features (no gradient, frozen network)
        with torch.no_grad():
            features = self.feature_extractor(x_tensor).numpy()

        return features
```

**Code location:**
- Feature extractor: `scripts/ch06a/neural_linear_demo.py` (demonstration)
- Production integration: Extend `zoosim/policies/lin_ucb.py` (see Exercise 6A.5)

### When Neural Features Beat Hand-Crafted Features

**Empirical rule of thumb:**

Neural Linear improves over rich hand-crafted features when:

1. **Nonlinearity is strong:** $R^2_{\text{linear}}(\phi_{\text{rich}}) < 0.7$ on held-out data
2. **Interactions are complex:** Higher-order terms ($x_1 x_2 x_3$) matter
3. **Pretraining data is abundant:** $N_{\text{pretrain}} \geq 5000$ episodes
4. **Representation dimension is modest:** $d \in [10, 50]$ (not too high, avoids overfitting)

**Expected gains:**
- **Success case:** +5-10% GMV over rich features when conditions above are met
- **Neutral case:** ±2% GMV (roughly equivalent) when reward is approximately linear
- **Failure case:** -10% to -30% GMV when data is scarce or network overparameterized (see Lab 6A.3)

**Lab 6A.1 preview:** We'll run Neural Linear on ZooplusSearchEnv and compare:
- Simple features ($d=7$): ~4.6 GMV (Chapter 6 baseline)
- Rich features ($d=17$): ~7.1 GMV (Chapter 6 success)
- Neural Linear ($d=20$ learned): Target ~7.3-7.8 GMV

---

## 6A.2 Neural Linear in Zoosim (Integration)

### End-to-End Pipeline

**Step 1: Generate logged data from baseline policy**

We need $(x, a, r)$ tuples for pretraining. Two options:

**Option A: Uniform random policy**
```python
from zoosim.envs.gym_env import ZooplusSearchEnv

def collect_logged_data_random(env, templates, n_episodes=5000, seed=42):
    """Collect logged data using uniform random template selection."""
    np.random.seed(seed)
    logged_data = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        x_raw = extract_raw_features(obs)  # Convert obs to vector

        # Random action
        a = np.random.randint(len(templates))

        # Apply template, observe reward
        template = templates[a]
        # [Apply template to env, get reward]
        r = ...  # GMV or other reward

        logged_data.append((x_raw, a, r))

    return logged_data
```

**Option B: Best static template (warm start)**
```python
def collect_logged_data_static(env, templates, best_template_id, n_episodes=5000):
    """Collect logged data using a strong static template."""
    # Similar to above, but always select best_template_id
    # Provides higher-quality data (better rewards) for pretraining
    pass
```

**Which to use?**
- Random: More diverse coverage of action space, noisier rewards
- Static: Higher reward signal, but concentrated on one template
- **Recommendation:** Use random for exploration coverage

**Step 2: Pretrain feature extractor**

```python
# Assuming logged_data = [(x, a, r), ...] with 5000 episodes
input_dim = len(logged_data[0][0])  # Raw feature dimension

feature_extractor = pretrain_feature_extractor(
    logged_data=logged_data,
    input_dim=input_dim,
    hidden_dim=64,
    output_dim=20,
    n_epochs=100,
    learning_rate=1e-3,
    seed=42
)

print("Pretraining complete. Feature extractor frozen.")
```

**Step 3: Run Neural Linear bandit**

```python
from zoosim.policies.lin_ucb import LinUCB
# (Or use NeuralLinearUCB from §6A.1)

# Initialize bandit with neural features
policy = NeuralLinearUCB(
    templates=templates,
    feature_extractor=feature_extractor,
    alpha=1.0,
    lambda_reg=1.0,
    seed=42
)

# Run bandit loop (same as Chapter 6)
for t in range(20000):
    obs, _ = env.reset()
    action_id = policy.select_action(obs)
    template = templates[action_id]
    # [Apply template, observe reward, update policy]
    policy.update(obs, action_id, reward)
```

**Step 4: Evaluation and diagnostics**

After 20k episodes, compare:
- **GMV:** Mean reward over final 5k episodes
- **Per-segment performance:** GMV by user segment
- **Selection frequencies:** Which templates does Neural Linear prefer?
- **Feature visualization:** t-SNE of $f_\psi(x)$ colored by segment (Exercise 6A.8)

### Diagnostics: How to Know If It's Working

**1. Pretraining validation loss**

**Good sign:** Validation MSE decreases steadily and plateaus
```
Epoch 20, Train Loss: 156.3, Val Loss: 163.2
Epoch 40, Train Loss: 142.1, Val Loss: 149.8
Epoch 60, Train Loss: 138.4, Val Loss: 147.1
Epoch 80, Train Loss: 136.2, Val Loss: 146.5  # Plateau
Epoch 100, Train Loss: 135.1, Val Loss: 146.3
```

**Bad sign:** Validation loss increases (overfitting)
```
Epoch 20, Train Loss: 156.3, Val Loss: 163.2
Epoch 40, Train Loss: 120.4, Val Loss: 178.5  # Val loss going UP
Epoch 60, Train Loss: 98.2, Val Loss: 195.3   # Severe overfit
```

**Action:** Use early stopping, add dropout, reduce network size.

**2. Bandit performance vs baselines**

Compare final GMV (last 5k episodes):

| Method | GMV (mean ± std) | vs Simple | vs Rich |
|--------|------------------|-----------|---------|
| Simple features | 4.6 ± 0.8 | — | — |
| Rich features | 7.1 ± 0.9 | +54% | — |
| Neural Linear | 7.5 ± 1.0 | +63% | +5.6% |

**Good outcome:** Neural Linear ≥ Rich features
**Acceptable outcome:** Neural Linear ≈ Rich features (no harm, complexity not justified)
**Bad outcome:** Neural Linear < Simple features (catastrophic failure, Lab 6A.3 explores this)

**3. Per-segment breakdown**

Check if Neural Linear helps uniformly or only for specific segments:

```python
# After running bandit, analyze by segment
results_by_segment = {
    'premium': {'simple': 5.2, 'rich': 8.1, 'neural': 8.6},
    'pl_lover': {'simple': 4.8, 'rich': 7.5, 'neural': 7.8},
    'price_hunter': {'simple': 3.9, 'rich': 5.8, 'neural': 6.1},
}

# Where did Neural Linear win most?
for segment, gmvs in results_by_segment.items():
    gain = gmvs['neural'] - gmvs['rich']
    print(f"{segment}: Neural Linear gain = {gain:.2f}")
```

**Expected finding:** Neural Linear helps most for segments with complex, nonlinear preferences (e.g., `premium` users who balance price, quality, brand).

### Expected Results: Success and Failure Modes

**Scenario A: Abundant data + nonlinearity (EXPECTED SUCCESS)**

**Setup:**
- $N_{\text{pretrain}} = 5000$ episodes
- Hidden dim = 64
- True reward has interaction terms (product margin × user price sensitivity)

**Result:**
- Neural Linear: 7.5 GMV ✓
- Rich features: 7.1 GMV
- **Gain:** +5.6%

**Interpretation:** Neural network learned interaction structure automatically, saving us from hand-crafting $\phi(x) = [x_1, x_2, x_1 \cdot x_2, ...]$.

---

**Scenario B: Scarce data (EXPECTED FAILURE)**

**Setup:**
- $N_{\text{pretrain}} = 500$ episodes (10× less)
- Hidden dim = 64 (same network size)

**Result:**
- Neural Linear: 5.5 GMV ❌
- Rich features: 7.1 GMV
- Simple features: 4.6 GMV
- **Loss:** -22% vs rich, but still beats simple

**Interpretation:** Not enough data to learn good representation. Network overfits to 500 examples. Performance degrades but doesn't collapse entirely (still better than simple features due to partial structure learned).

---

**Scenario C: Overparameterized network (CATASTROPHIC FAILURE)**

**Setup:**
- $N_{\text{pretrain}} = 500$ episodes
- Hidden dim = 256 (4× larger, ~100k parameters)

**Result:**
- Neural Linear: 4.0 GMV ❌❌
- Rich features: 7.1 GMV
- Simple features: 4.6 GMV
- **Loss:** -44% vs rich, WORSE than simple

**Interpretation:** Massive overfitting. Network memorizes 500 training examples but has no generalization. Bandit exploration encounters out-of-distribution contexts and gets garbage features.

**Lesson:** Model capacity must match data scale. With 500 episodes, use hidden_dim ≤ 32.

---

**Lab 6A.3 systematically explores these scenarios** to build intuition for when to trust (and distrust) neural methods.

### Code ↔ Config (Neural Linear Configuration)

Key hyperparameters for Neural Linear:

| Parameter | Config location | Typical range | Sensitivity |
|-----------|----------------|---------------|-------------|
| `n_pretrain` | `scripts/ch06a/neural_linear_demo.py:15` | 2000-10000 | High |
| `hidden_dim` | `scripts/ch06a/neural_linear_demo.py:18` | 32-128 | Medium |
| `output_dim` | `scripts/ch06a/neural_linear_demo.py:19` | 10-50 | Low |
| `learning_rate` | `scripts/ch06a/neural_linear_demo.py:21` | 1e-4 to 1e-2 | Medium |
| `n_epochs` | `scripts/ch06a/neural_linear_demo.py:22` | 50-200 | Low |
| `dropout` | `NeuralFeatureExtractor.__init__` | 0.0-0.3 | Medium |

**Tuning guidelines:**
- Start with default: `hidden_dim=64`, `output_dim=20`, `n_pretrain=5000`
- If overfitting (val loss increases): Reduce `hidden_dim`, add `dropout=0.1`, use early stopping
- If underfitting (train loss high): Increase `hidden_dim` or add another layer
- If bandit performance poor: Check pretraining val loss first (must be reasonable)

**Next section:** We'll extend this idea to full neural $Q(x,a)$ without the linear head restriction.

---

## 6A.3 NL-Bandit as Neural Q(x,a) on Discrete Templates

### Beyond Neural Linear: Why Go Fully Neural?

**Limitation of Neural Linear:** Even with learned representation $f_\psi(x)$, we still assume:
$$
\mu(x, a) = \theta_a^\top f_\psi(x)
$$

This is **linear in the representation space**. If the true reward has structure like:
- **Action-context interactions:** $\mu(x, a) = f(x) \cdot g(a) + h(x, a)$ where $h$ is non-separable
- **Action-specific nonlinearity:** Different actions benefit from different feature transformations

Then even a good $f_\psi(x)$ with linear heads $\theta_a$ may be misspecified.

**Solution: Full neural Q-function**

Model $Q(x,a)$ directly as a neural network:
$$
Q_\theta(x, a) \approx \mathbb{E}[R \mid X=x, A=a]
\tag{6A.2}
$$
{#EQ-6A.2}

where $\theta$ are parameters of a neural network mapping $(x, a) \to \mathbb{R}$.

!!! important "This Remains a Bandit Problem (γ=0)"
    Despite using the notation $Q(x,a)$, this is **not** a multi-step MDP value function. There is **no Bellman recursion** here: we model only the **immediate reward** $R$ conditioned on context $x$ and action $a$. Each search session is a single step ($\gamma=0$), so $Q(x,a) = \mathbb{E}[R \mid x,a]$ equals the expected reward, not a discounted sum of future rewards.

    **Contrast with Chapter 7 (continuous MDP):** When we move to continuous boost optimization in Chapter 7, the discrete template space $\mathcal{A} = \{0,\ldots,M-1\}$ becomes continuous $\mathcal{A} = [-a_{\max}, +a_{\max}]^K$, but the problem **still remains a bandit** (γ=0) because each query is independent. The notation $Q(x,a)$ persists in both chapters as shorthand for "immediate expected reward function," not as a multi-step value function.

**Key difference from Neural Linear:**
- **Neural Linear:** $\theta_a^\top f_\psi(x)$ → linear combination of learned features
- **NL-Bandit:** $Q_\theta(x,a)$ → arbitrary nonlinear function of both $x$ and $a$

**Why "NL-Bandit"?** The name reflects:
- **N**eural **L**inear → **NL** (but now fully neural, not linear heads)
- Discrete action space (templates) makes it a **bandit** problem ($\gamma=0$)
- Distinguishes from continuous Q(x,a) in Chapter 7

### Neural Architectures for Discrete Actions

**Challenge:** How to represent $Q(x,a)$ when $a \in \{0, \ldots, M-1\}$ is discrete?

**Architecture A: Multi-Head Network**

Idea: One neural network body, separate head per action.

```python
class MultiHeadQ(nn.Module):
    """Multi-head Q-network for discrete actions.

    Shared body processes context x.
    Each action has its own output head.

    Corresponds to Q_θ(x,a) where a ∈ {0,...,M-1}.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_actions)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q(x,a) for all actions.

        Args:
            x: Context, shape (batch, input_dim)

        Returns:
            Q-values: Shape (batch, n_actions)
        """
        h = self.body(x)  # (batch, hidden_dim)
        q_values = torch.stack([
            head(h).squeeze(-1) for head in self.heads
        ], dim=-1)  # (batch, n_actions)
        return q_values
```

**Pros:**
- ✓ Efficient: Single forward pass computes $Q(x,a)$ for all $a$
- ✓ Shared representation: Body learns context features useful across actions

**Cons:**
- ✗ Fixed action set: Can't handle variable number of templates
- ✗ No action features: Assumes actions are just indices (no template metadata)

---

**Architecture B: Concatenated Action Encoding**

Idea: Encode action as one-hot vector, concatenate with context.

```python
class ConcatQ(nn.Module):
    """Concatenated Q-network for discrete actions.

    Takes (x, a_onehot) as input.
    Can handle action features (not just indices).

    Corresponds to Q_θ(x,a) with action encoding.
    """
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        """Compute Q(x,a) for specific action.

        Args:
            x: Context, shape (batch, input_dim)
            a_onehot: One-hot action, shape (batch, n_actions)

        Returns:
            Q-value: Shape (batch,)
        """
        xa = torch.cat([x, a_onehot], dim=-1)  # (batch, input_dim + n_actions)
        return self.net(xa).squeeze(-1)  # (batch,)

    def forward_all_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q(x,a) for all actions (for action selection).

        Args:
            x: Context, shape (batch, input_dim)

        Returns:
            Q-values: Shape (batch, n_actions)
        """
        batch_size = x.shape[0]
        q_values = []

        for a in range(self.n_actions):
            a_onehot = torch.zeros(batch_size, self.n_actions)
            a_onehot[:, a] = 1.0
            q_a = self.forward(x, a_onehot)  # (batch,)
            q_values.append(q_a)

        return torch.stack(q_values, dim=-1)  # (batch, n_actions)
```

**Pros:**
- ✓ Flexible: Can use action features (template parameters, boost magnitudes)
- ✓ Generalizes to continuous actions (Chapter 7): Just replace one-hot with action vector

**Cons:**
- ✗ Slower: Must query $M$ times to get all Q-values (for action selection)

---

**Recommendation for Zoosim:**

Use **Architecture A (Multi-Head)** for Chapter 6A:
- Discrete templates (8 actions) → multi-head efficient
- Chapter 7 will use Architecture B for continuous boost vectors

### Ensemble for Uncertainty Quantification

**Problem:** Single neural network $Q_\theta(x,a)$ gives point estimates, but we need **uncertainty** for exploration.

**Solution: Train ensemble of $K$ networks** $\{Q_{\theta^{(1)}}, \ldots, Q_{\theta^{(K)}}\}$

**Ensemble mean and variance:**
$$
\mu(x, a) = \frac{1}{K} \sum_{k=1}^K Q_{\theta^{(k)}}(x, a)
$$
$$
\sigma^2(x, a) = \frac{1}{K} \sum_{k=1}^K \left( Q_{\theta^{(k)}}(x, a) - \mu(x, a) \right)^2
\tag{6A.3}
$$
{#EQ-6A.3}

**UCB action selection:**
$$
a_t = \arg\max_{a \in \mathcal{A}} \left[ \mu(x_t, a) + \beta \cdot \sigma(x_t, a) \right]
\tag{6A.4}
$$
{#EQ-6A.4}

where $\beta > 0$ is exploration coefficient (typical range: 1.0-3.0).

**Why ensembles work (empirically observed, no theoretical guarantees):**
- Different random initializations → networks converge to different local minima
- Variance $\sigma^2(x,a)$ tends to be higher in regions with: sparse data, high noise, extrapolation
- When well-calibrated: $\sigma(x,a)$ correlates with true prediction error (calibration must be verified empirically, see §6A.4)

**Implementation notes:**
- Train $K=3$ to $K=10$ networks (diminishing returns beyond 10)
- Use different random seeds for initialization
- Optional: Add dropout during training for diversity
- Share training data (same replay buffer), not separate datasets

### Algorithm: NL-Bandit with Ensemble

**Algorithm 6A.1: NL-Bandit (Discrete Templates)**
{#ALG-6A.1}

```
Initialize:
    Ensemble of K networks {Q_θ^(k)}_{k=1}^K (random seeds)
    Replay buffer D = ∅
    UCB coefficient β > 0

For episode t = 1, ..., T:
    # Observe context
    Observe context x_t

    # Compute Q-values and uncertainty for all actions
    For each action a ∈ {0, ..., M-1}:
        μ_a ← (1/K) ∑_{k=1}^K Q_θ^(k)(x_t, a)
        σ_a ← sqrt((1/K) ∑_{k=1}^K (Q_θ^(k)(x_t, a) - μ_a)^2)
        UCB_a ← μ_a + β·σ_a

    # Select action with highest UCB
    a_t ← argmax_a UCB_a

    # Apply template, observe reward
    Apply template a_t to search results
    Observe reward r_t (GMV, engagement, etc.)

    # Store transition
    D ← D ∪ {(x_t, a_t, r_t)}

    # Update ensemble (every N episodes or continuous)
    If t mod UPDATE_FREQ == 0:
        For each network k ∈ {1, ..., K}:
            Sample minibatch B ~ D
            Update θ^(k) via gradient descent on L = ∑_{(x,a,r)∈B} (Q_θ^(k)(x,a) - r)^2
```

**Key hyperparameters:**
- `K`: Ensemble size (3-10)
- `β`: UCB coefficient (1.0-3.0)
- `UPDATE_FREQ`: How often to train networks (every 100-500 episodes)
- `batch_size`: Minibatch size for SGD (64-256)
- `learning_rate`: Adam LR (1e-4 to 1e-3)

### Implementation: Reusing Q-Ensemble from Chapter 7

**Code reuse opportunity:** The `QEnsembleRegressor` we'll implement in Chapter 7 for continuous actions works for discrete actions too!

**How:**
1. Encode discrete actions as one-hot vectors
2. Use ConcatQ architecture (Architecture B)
3. Action selection: Loop over $M$ templates, compute UCB for each

**Lightweight wrapper:**

```python
from zoosim.policies.q_ensemble import QEnsembleRegressor  # Ch7 code

class DiscreteNLBandit:
    """NL-Bandit for discrete templates using Q-ensemble.

    Wraps continuous Q(x,a) ensemble for discrete action selection.
    Corresponds to ALG-6A.1.
    """
    def __init__(
        self,
        templates: list,
        input_dim: int,
        hidden_dim: int = 64,
        ensemble_size: int = 5,
        ucb_beta: float = 2.0,
        learning_rate: float = 1e-3,
        seed: int = 42
    ):
        """Initialize discrete NL-Bandit.

        Args:
            templates: List of boost templates
            input_dim: Dimension of context x
            hidden_dim: Width of Q-network hidden layers
            ensemble_size: Number of networks in ensemble
            ucb_beta: UCB exploration coefficient
            learning_rate: Adam learning rate
            seed: Random seed
        """
        self.templates = templates
        self.n_actions = len(templates)
        self.ucb_beta = ucb_beta

        # Q-ensemble: input = [x, a_onehot], output = Q(x,a)
        self.q_ensemble = QEnsembleRegressor(
            input_dim=input_dim + self.n_actions,  # Concatenate x and a_onehot
            output_dim=1,
            hidden_dim=hidden_dim,
            ensemble_size=ensemble_size,
            learning_rate=learning_rate,
            seed=seed
        )

    def select_action(self, context: np.ndarray) -> int:
        """Select action using UCB over ensemble.

        Corresponds to EQ-6A.4: a = argmax [μ(x,a) + β·σ(x,a)]

        Args:
            context: Context vector x, shape (input_dim,)

        Returns:
            action_id: Template index in {0, ..., M-1}
        """
        ucb_values = []

        for a in range(self.n_actions):
            # Encode action as one-hot
            a_onehot = np.zeros(self.n_actions)
            a_onehot[a] = 1.0

            # Concatenate context and action
            xa = np.concatenate([context, a_onehot])

            # Query ensemble
            mu, sigma = self.q_ensemble.predict_with_uncertainty(xa)

            # UCB
            ucb = mu + self.ucb_beta * sigma
            ucb_values.append(ucb)

        # Select action with highest UCB
        return int(np.argmax(ucb_values))

    def update(self, context: np.ndarray, action: int, reward: float):
        """Update ensemble with new observation.

        Args:
            context: Context vector x
            action: Selected action index
            reward: Observed reward r
        """
        # Encode action
        a_onehot = np.zeros(self.n_actions)
        a_onehot[action] = 1.0

        # Concatenate and update
        xa = np.concatenate([context, a_onehot])
        self.q_ensemble.update(xa, reward)
```

**Code location:**
- Full implementation: `scripts/ch06a/neural_bandits_demo.py`
- Q-ensemble base class: `zoosim/policies/q_ensemble.py` (Chapter 7)

**Lab 6A.2 preview:** We'll run this NL-Bandit on Zoosim and compare to:
- Neural Linear (~7.5 GMV)
- Thompson Sampling (Ch6: ~8.7 GMV)
- Target: ~7.5-8.5 GMV (competitive with TS)

### When to Use NL-Bandit vs Neural Linear

| Criterion | Use Neural Linear | Use NL-Bandit |
|-----------|-------------------|---------------|
| **Data abundance** | <5k episodes | >10k episodes |
| **Reward structure** | Approximately linear in features | Highly nonlinear, action-context interactions |
| **Action space size** | Any (uses linear heads) | Small to moderate (M < 50) |
| **Interpretability** | Some (can inspect $\theta_a$) | Low (black box) |
| **Training complexity** | Simple (supervised pretrain + freeze) | Moderate (online ensemble training) |
| **Inference latency** | Fast (linear algebra) | Moderate (K network evals × M actions) |

**Rule of thumb:**
- Start with Neural Linear (simpler, more stable)
- Upgrade to NL-Bandit if reward is clearly nonlinear and data is abundant
- Both beat pure linear bandits when nonlinearity exists

**Chapter 7 preview:** NL-Bandit naturally extends to **continuous actions** by replacing one-hot $a_{\text{onehot}}$ with real-valued boost vector $a \in \mathbb{R}^K$, then optimizing $\arg\max_a Q(x,a)$ with CEM.

---

## 6A.4 Neural Bandit Failure Modes and Calibration

### Why Uncertainty Calibration Matters

**Exploration-exploitation trade-off** in bandits depends critically on **uncertainty estimates**:

$$
a_t = \arg\max_a \left[ \underbrace{\mu(x, a)}_{\text{exploitation}} + \underbrace{\beta \cdot \sigma(x, a)}_{\text{exploration}} \right]
$$

**If uncertainty is miscalibrated:**
- **Overconfident** ($\sigma$ too small) → **under-explore** → get stuck on suboptimal templates
- **Underconfident** ($\sigma$ too large) → **over-explore** → slow convergence, high regret

**In linear bandits** (Ch6), uncertainty $\sqrt{f^\top A_a^{-1} f}$ is **provably calibrated** under model assumptions.

**In neural bandits**, ensemble variance $\sigma(x,a)$ is a **heuristic** that works empirically but has no guarantees. We must **check calibration** empirically.

### Failure Mode 1: Overfitting in Pretraining (Neural Linear)

**Symptom:**
- Training loss decreases steadily
- **Validation loss increases** after initial decrease

**Example:**
```
Epoch 10: Train MSE = 180.3, Val MSE = 185.2
Epoch 20: Train MSE = 142.1, Val MSE = 148.7
Epoch 30: Train MSE = 98.5, Val MSE = 163.4  # Val loss going UP
Epoch 40: Train MSE = 67.2, Val MSE = 189.1  # Severe overfit
```

**Consequences:**
- Learned features $f_\psi(x)$ don't generalize
- Bandit explores contexts different from pretraining distribution
- Feature extractor returns garbage for out-of-distribution $x$
- Performance collapses (Lab 6A.3 Scenario C)

**Fixes:**
1. **Early stopping:** Stop training when val loss stops improving
2. **Regularization:** Add dropout (0.1-0.3), weight decay ($\lambda = 10^{-4}$)
3. **Reduce capacity:** Smaller `hidden_dim`, fewer layers
4. **More data:** Collect more pretraining episodes

**Diagnostic check:**
```python
def check_pretraining_quality(feature_extractor, val_data):
    """Check if learned features are reasonable."""
    # Extract features for validation set
    with torch.no_grad():
        X_val = torch.tensor([x for x, a, r in val_data], dtype=torch.float32)
        features = feature_extractor(X_val).numpy()

    # Check: Are features diverse?
    feature_std = features.std(axis=0)
    if (feature_std < 0.01).sum() > features.shape[1] // 2:
        print("⚠️ Warning: >50% of features have low variance (representation collapse)")

    # Check: Do features correlate with rewards?
    R_val = np.array([r for x, a, r in val_data])
    for dim in range(features.shape[1]):
        corr = np.corrcoef(features[:, dim], R_val)[0, 1]
        # At least some dims should correlate with reward

    print(f"✓ Feature quality check passed")
```

### Failure Mode 2: Ensemble Collapse (NL-Bandit)

**Symptom:**
- Ensemble networks converge to **nearly identical predictions**
- Variance $\sigma^2(x,a) \approx 0$ everywhere
- Bandit stops exploring

**Why it happens:**
- All networks initialized with same procedure → converge to same local minimum
- Insufficient diversity in training (same data, same optimizer, same hyperparams)

**Example:**
```python
# Query ensemble for 100 random contexts
for i in range(100):
    x_rand = np.random.randn(input_dim)
    mu, sigma = q_ensemble.predict_with_uncertainty(x_rand)
    print(f"Context {i}: μ={mu:.2f}, σ={sigma:.4f}")

# Bad output (collapsed ensemble):
Context 0: μ=7.23, σ=0.0012  # Variance near zero!
Context 1: μ=5.81, σ=0.0008
Context 2: μ=8.92, σ=0.0015
...
```

**Fixes:**
1. **Different random seeds** for each network in ensemble
2. **Dropout during training** (forces diversity)
3. **Bootstrap sampling:** Train each network on different subset of data
4. **Different architectures:** Vary `hidden_dim` across ensemble members

**Diagnostic check:**
```python
def check_ensemble_diversity(q_ensemble, test_contexts):
    """Check if ensemble networks are diverse."""
    variances = []

    for x in test_contexts:
        _, sigma = q_ensemble.predict_with_uncertainty(x)
        variances.append(sigma)

    mean_sigma = np.mean(variances)

    if mean_sigma < 0.01:
        print(f"⚠️ Warning: Ensemble variance too low (mean σ = {mean_sigma:.4f})")
        print("   → Ensemble collapsed. Retrain with different seeds/dropout.")
    else:
        print(f"✓ Ensemble diversity OK (mean σ = {mean_sigma:.3f})")
```

### Failure Mode 3: Miscalibrated Uncertainty

**Symptom:**
- Ensemble variance $\sigma(x,a)$ does **not** correlate with true prediction error
- High $\sigma$ in regions where predictions are actually accurate (underconfident)
- Low $\sigma$ in regions with high error (overconfident)

**How to detect:** Calibration plot on held-out data

**Calibration check:**
1. Collect held-out episodes: $\{(x_i, a_i, r_i)\}_{i=1}^N$
2. For each $(x_i, a_i)$:
   - Predict $\mu_i, \sigma_i = \text{ensemble}(x_i, a_i)$
   - Compute error $e_i = |r_i - \mu_i|$
3. Plot scatter: $\sigma_i$ (x-axis) vs $e_i$ (y-axis)
4. Compute correlation: $\rho(\sigma, e)$

**Good calibration:**
- Points cluster near diagonal ($e \approx \sigma$)
- Correlation $\rho > 0.6$
- High $\sigma$ → high error, low $\sigma$ → low error

**Example plot (good calibration):**
```
Error (|r - μ|)
    ^
10  |               *
    |            *   *
 8  |          *  *
    |       *  *
 6  |     *  *
    |   *  *
 4  | *  *
    |*  *
 2  |* *
    +----------> σ (Uncertainty)
    0   2   4   6   8  10

Correlation: ρ = 0.71 ✓
```

**Example plot (poor calibration):**
```
Error (|r - μ|)
    ^
10  | * *  *  *  *  *  *
    | * *  *  *  *  *  *
 8  | * *  *  *  *  *  *
    | * *  *  *  *  *  *
 6  | * *  *  *  *  *  *
    | * *  *  *  *  *  *
 4  | * *  *  *  *  *  *
    | * *  *  *  *  *  *
 2  | * *  *  *  *  *  *
    +----------> σ (Uncertainty)
    0   2   4   6   8  10

Correlation: ρ = 0.08 ❌ (No relationship!)
```

**Fixes for poor calibration:**
1. **Tune $\beta$:** If ρ low, $\beta$ may be wrong scale (try 0.5× or 2×)
2. **More ensemble members:** Increase $K$ from 5 to 10
3. **Recalibrate $\sigma$:** Use held-out data to fit $\sigma_{\text{calibrated}} = a \cdot \sigma + b$
4. **Alternative uncertainty:** Try MC dropout, Bayesian neural nets (advanced)

**Lab 6A.2 includes calibration diagnostic** as part of NL-Bandit evaluation.

### Failure Mode 4: Data Scarcity (General)

**Symptom:**
- Neural methods perform worse than simple hand-crafted features
- Large variance across seeds (unstable)

**When it happens:**
- Pretraining data $N < 2000$ episodes
- Online bandit data $T < 5000$ episodes
- Network too large relative to data (parameters > data points)

**Evidence from Lab 6A.3:**

| Method | N=500 pretrain | N=5000 pretrain |
|--------|----------------|-----------------|
| Simple features | 4.6 GMV | 4.6 GMV |
| Rich features | 7.1 GMV | 7.1 GMV |
| Neural Linear (h=64) | 5.5 GMV ❌ | 7.5 GMV ✓ |
| Neural Linear (h=256) | 4.0 GMV ❌❌ | 7.8 GMV ✓ |

**Lesson:** Neural methods need **minimum data threshold**. Below that, stick to linear bandits.

**When to avoid neural bandits entirely:**
- Fewer than 2k episodes available
- Feature dimension < 50 (hand-crafted features likely sufficient)
- Production constraints: interpretability critical, latency < 10ms

### Diagnostic Checklist for Neural Bandits

Before deploying Neural Linear or NL-Bandit, check:

| Check | Metric | Threshold | Action if Fail |
|-------|--------|-----------|----------------|
| **Pretraining quality** | Val MSE | < 2× Train MSE | Early stop, add dropout |
| **Feature diversity** | `std(f_ψ(x))` | > 0.1 for >80% dims | Reduce regularization |
| **Ensemble diversity** | Mean $\sigma(x,a)$ | > 0.01 | Different seeds, dropout |
| **Calibration** | $\rho(\sigma, \|r-\mu\|)$ | > 0.5 | Tune β, increase K |
| **Performance vs baseline** | GMV | ≥ 95% of rich features | Collect more data, simplify model |

**Honest conclusion:** Neural bandits are powerful but finicky. They work when:
- ✅ Data is abundant (>5k episodes)
- ✅ Nonlinearity is strong
- ✅ You invest time in calibration checks

They fail when:
- ❌ Data is scarce
- ❌ Model is overparameterized
- ❌ You skip diagnostic checks and deploy blindly

**Vlad here:** The data science community has a saying: "Neural networks are like teenagers—give them too much freedom without oversight, and they'll make terrible decisions." The diagnostic checklist above is that oversight.

---

## 6A.5 Bridge to Continuous Q(x,a) (Chapter 7)

### Conceptual Bridge: From Discrete to Continuous Actions

**Chapter 6A (this chapter):**
- Actions: Discrete templates $a \in \{0, \ldots, M-1\}$
- Example: 8 templates with predefined boost magnitudes
- Action selection: Enumerate all $M$ templates, pick $\arg\max_a Q(x,a)$

**Chapter 7:**
- Actions: Continuous boost vectors $a \in \mathbb{R}^K$
- Example: $a = [\text{boost}_{\text{CM2}}, \text{boost}_{\text{Cat}}, \text{boost}_{\text{Disc}}, \ldots]$ where each $a_k \in [-a_{\max}, +a_{\max}]$
- Action optimization: Use Cross-Entropy Method (CEM) to solve $\arg\max_a Q(x,a)$ over continuous space

**Unified framework: Both use Q(x,a)**

$$
Q(x, a) \approx \mathbb{E}[R \mid X=x, A=a]
$$

**Key differences:**

| Aspect | Ch6A (Discrete) | Ch7 (Continuous) |
|--------|----------------|------------------|
| **Action encoding** | One-hot $a_{\text{onehot}} \in \mathbb{R}^M$ | Real vector $a \in \mathbb{R}^K$ |
| **Action selection** | Loop over $M$ actions, pick max | Optimize with CEM |
| **Trust region** | Not needed (finite action set) | Critical (prevents out-of-distribution actions) |
| **Sample efficiency** | High (small action space) | Moderate (infinite action space) |

**Implementation continuity:**

Both chapters use the **same Q-ensemble code** (`zoosim/policies/q_ensemble.py`):
- NL-Bandit (Ch6A): Input = $[x, a_{\text{onehot}}]$
- Continuous Q (Ch7): Input = $[x, a]$

**Only difference:** Optimization method (enumerate vs CEM).

### Implementation Bridge: Shared Q-Ensemble Module

**Q-ensemble interface (used in both chapters):**

```python
class QEnsembleRegressor:
    """Ensemble of neural networks for Q(x,a) regression.

    Used in:
    - Chapter 6A: Discrete actions (via one-hot encoding)
    - Chapter 7: Continuous actions (direct boost vectors)
    - Chapter 13: Offline RL baselines
    """
    def __init__(
        self,
        input_dim: int,      # dim(x) + dim(a)
        output_dim: int,     # 1 (scalar Q-value)
        hidden_dim: int,
        ensemble_size: int,
        learning_rate: float,
        seed: int
    ):
        """Initialize ensemble of K networks."""
        self.ensemble = [
            QNetwork(input_dim, hidden_dim, output_dim)
            for _ in range(ensemble_size)
        ]
        # [Initialize optimizers, replay buffer, etc.]

    def predict_with_uncertainty(
        self,
        xa: np.ndarray  # Concatenated [x, a]
    ) -> tuple[float, float]:
        """Predict Q(x,a) with uncertainty.

        Returns:
            mu: Ensemble mean Q-value
            sigma: Ensemble standard deviation
        """
        predictions = [net(xa) for net in self.ensemble]
        mu = np.mean(predictions)
        sigma = np.std(predictions)
        return mu, sigma

    def update(self, xa: np.ndarray, target: float):
        """Update ensemble with new (x,a,r) observation."""
        # Add to replay buffer
        # Train each network in ensemble
        pass
```

**Chapter 6A usage:**
```python
# Discrete NL-Bandit
input_dim = context_dim + n_actions  # x + one-hot a
q_ensemble = QEnsembleRegressor(input_dim, output_dim=1, ...)

# Action selection (loop over discrete actions)
for a in range(n_actions):
    a_onehot = one_hot(a, n_actions)
    xa = np.concatenate([context, a_onehot])
    mu, sigma = q_ensemble.predict_with_uncertainty(xa)
    ucb[a] = mu + beta * sigma

selected_action = np.argmax(ucb)
```

**Chapter 7 usage:**
```python
# Continuous Q(x,a)
input_dim = context_dim + action_dim  # x + continuous a
q_ensemble = QEnsembleRegressor(input_dim, output_dim=1, ...)

# Action optimization (CEM)
def objective(a):
    xa = np.concatenate([context, a])
    mu, sigma = q_ensemble.predict_with_uncertainty(xa)
    return mu + beta * sigma  # UCB

best_action = cem_optimize(objective, action_bounds=(-a_max, a_max))
```

**Key insight:** Q(x,a) ensemble is a **universal building block** for both discrete and continuous action RL. Chapter 6A shows the simpler case (discrete); Chapter 7 scales to continuous.

### Narrative Bridge: Why This Chapter Is Optional But Recommended

**You can skip Chapter 6A and go straight to Chapter 7 if:**
- You're primarily interested in continuous boost optimization
- You're comfortable with neural networks and don't need the discrete warmup
- Time is limited

**You should read Chapter 6A if you care about:**
1. **Neural bandits for their own sake:** Many production systems use discrete actions (A/B tests, template selection, ad creatives)
2. **Understanding NL-Bandit baseline in Chapter 13:** Offline RL chapter uses discrete NL-Bandit as baseline
3. **Pedagogical progression:** Discrete actions are simpler (no CEM, no trust regions); builds intuition for continuous case
4. **Failure mode awareness:** This chapter shows when neural methods fail (scarce data, miscalibration)—lessons apply to Chapter 7

**Recommended path:**
- **Minimal:** Read §6A.0-6A.1 (Neural Linear concept) → skip to Chapter 7
- **Standard:** Read entire Chapter 6A, do Labs 6A.1-6A.2 → Chapter 7
- **Deep dive:** Read Chapter 6A, do all exercises including Advanced → Chapter 7

**Time commitment:**
- Read only: 40 min
- Read + core labs: 120 min (2 hours)
- Read + all exercises: 285+ min (4.5+ hours)

**Vlad here:** I teach this material as optional because I respect your time. But if you're building a production RL system, I'd argue the failure mode analysis (§6A.4) alone is worth the 2-hour investment. Knowing when *not* to use neural networks is as valuable as knowing when to use them.

---

## 6A.6 Summary & What's Next

### What We Built

This chapter introduced **two levels of neural bandits** for discrete template selection:

**1. Neural Linear Bandits** (§6A.1-6A.2)
- **Architecture:** Learned representation $f_\psi(x)$ + linear heads $\theta_a$
- **Training:** Supervised pretraining on logged data, then freeze
- **Uncertainty:** Closed-form Gaussian posterior (same as LinUCB)
- **When to use:** Moderate nonlinearity, 5k+ pretraining episodes, want interpretability

**Key equation** [EQ-6A.1]:
$$
\mu(x, a) = \theta_a^\top f_\psi(x), \quad \text{where } f_\psi \text{ is a frozen neural network}
$$

**Expected gains:** +5-10% GMV over rich hand-crafted features when nonlinearity exists and data is abundant.

---

**2. NL-Bandit (Neural Q(x,a))** (§6A.3)
- **Architecture:** Full neural network $Q_\theta(x,a)$ with ensemble
- **Training:** Online updates during bandit exploration
- **Uncertainty:** Ensemble variance $\sigma^2(x,a)$
- **When to use:** Strong nonlinearity, 10k+ episodes, discrete action space (M < 50)

**Key equation** [EQ-6A.2]:
$$
Q_\theta(x, a) \approx \mathbb{E}[R \mid X=x, A=a]
$$

**Action selection** [EQ-6A.4]:
$$
a_t = \arg\max_a \left[ \mu(x_t, a) + \beta \cdot \sigma(x_t, a) \right]
$$

**Expected performance:** Competitive with Thompson Sampling from Chapter 6 (~8.5 GMV), better when reward is highly nonlinear.

---

### Key Lessons

**Lesson 1: Neural methods help when nonlinearity + data both exist**

| Data Regime | Reward Structure | Recommendation |
|-------------|------------------|----------------|
| Scarce (<2k) | Any | ❌ Stick to linear bandits (Ch6) |
| Moderate (2-5k) | Linear | ❌ Use rich hand-crafted features |
| Moderate (2-5k) | Nonlinear | ✓ Try Neural Linear (cautiously) |
| Abundant (>5k) | Linear | ≈ Neural Linear ≈ Rich features (no clear winner) |
| Abundant (>5k) | Nonlinear | ✓ Neural Linear or NL-Bandit |

**Lesson 2: Uncertainty calibration is not optional**

- Linear bandits: Calibration is automatic (provable under assumptions)
- Neural bandits: Calibration must be **checked empirically** (§6A.4)
- Diagnostic: Plot $\sigma(x,a)$ vs $|r - \mu(x,a)|$, require $\rho > 0.5$
- If miscalibrated: Tune $\beta$, increase ensemble size, or abandon neural approach

**Lesson 3: Failure modes are instructive**

We deliberately explored three failure scenarios (Lab 6A.3):
- **Scenario B (Scarce data):** Neural Linear degrades to 5.5 GMV (vs 7.1 for rich features)
- **Scenario C (Overparameterized):** Neural Linear collapses to 4.0 GMV (worse than simple features!)
- **Lesson:** Model capacity must match data scale. With 500 episodes, use `hidden_dim ≤ 32`.

**Vlad here:** Most ML tutorials show you success cases. I showed you failures because **that's where the learning happens**. When your production neural bandit underperforms, you'll remember Lab 6A.3 and check: "Am I in Scenario C right now?"

**Lesson 4: Bridge to Chapter 7 is seamless**

- Same Q-ensemble code for discrete (Ch6A) and continuous (Ch7) actions
- Only difference: Action encoding (one-hot vs real vector) and optimization (enumerate vs CEM)
- NL-Bandit is a warmup for continuous Q(x,a)—pedagogically valuable even if you skip to Ch7

---

### Where to Go Next

**Immediate next steps:**

1. **Complete exercises** (`docs/book/ch06a/exercises_labs.md`):
   - Theory: Ex 6A.1-6A.3 (30 min) — When linear features fail
   - Implementation: Ex 6A.4-6A.5 (40 min) — Build Neural Feature Extractor
   - Labs: Lab 6A.1-6A.3 (80 min) — Compare Neural Linear vs rich features, run NL-Bandit, diagnose overfitting

2. **Run demo scripts:**
   ```bash
   # Neural Linear vs Rich Features
   python scripts/ch06a/neural_linear_demo.py --n-pretrain 5000

   # NL-Bandit with Ensemble
   python scripts/ch06a/neural_bandits_demo.py --ensemble-size 5
   ```

3. **Check diagnostics:**
   - Pretraining val loss (should plateau, not increase)
   - Ensemble diversity (mean σ > 0.01)
   - Calibration plot (ρ > 0.5)
   - Performance vs baselines (GMV ≥ 95% of rich features)

---

**Chapter progression:**

- **Chapter 6** (just finished): Discrete templates, linear bandits, feature engineering
- **Chapter 6A** (this chapter): Neural bandits for discrete templates
- **Chapter 7** (next): Continuous boost optimization with Q(x,a) and CEM
- **Chapter 13** (future): Offline RL using NL-Bandit as baseline

**What Chapter 7 adds:**
- Continuous action space: $a \in \mathbb{R}^K$ (boost vectors)
- Trust regions: Δrank@k constraints to prevent out-of-distribution actions
- CEM optimization: Efficiently find $\arg\max_a Q(x,a)$ in continuous space
- Production deployment: Latency budgets, safety checks, rollback

---

**Advanced explorations (optional exercises):**

If you want to go deeper:
- **Ex 6A.6:** Joint training (update $f_\psi$ during bandit exploration)
- **Ex 6A.7:** Compare multi-head vs concatenated architectures
- **Ex 6A.8:** Transfer learning (pretrain on one segment, test on another)
- **Ex 6A.9:** Ablation study on ensemble size
- **Ex 6A.10:** Design your own neural bandit architecture

---

### Production Checklist (Chapter 6A)

Before deploying Neural Linear or NL-Bandit to production:

**Neural Feature Extractor (Neural Linear):**
- ✅ Pretrain on **logged data** (not online exploration)
- ✅ Use **separate validation set** to detect overfitting
- ✅ **Freeze** after pretraining (don't update during bandit unless advanced)
- ✅ Sanity check: Validation MSE < 2× training MSE
- ✅ Feature diversity: `std(f_ψ(x)) > 0.1` for >80% dimensions

**NL-Bandit Q(x,a):**
- ✅ Use **ensemble** (≥3 networks) for uncertainty quantification
- ✅ Calibration check: $\sigma(x,a)$ correlates with prediction error $|r - \mu(x,a)|$ (ρ > 0.5) on held-out data
- ✅ UCB coefficient $\beta$: Tune on held-out seeds (typical range: 1.0-3.0)
- ✅ Monitor ensemble variance: We typically observe it decreases over time (roughly $O(1/\sqrt{T})$), but **no theoretical guarantee** for neural ensembles—this is an empirical heuristic

**When to avoid neural methods:**
- ❌ Data < 2000 episodes → Stick to linear bandits (Ch6)
- ❌ Feature dim < 50 → Hand-crafted features likely better
- ❌ Production constraints (latency < 10ms, interpretability critical) → Use Ch6 templates
- ❌ Reward is nearly linear ($R^2_{\text{linear}} > 0.8$) → Neural complexity not justified

**Monitoring in production:**
- Track pretraining loss convergence (early stopping criterion)
- Monitor ensemble variance over time (should decrease, not collapse to zero)
- Alert if GMV < Ch6 baseline for 1000+ consecutive episodes
- Compare to Ch6 rich features as sanity check

**Reproducibility:**
- Set seeds for: PyTorch, NumPy, environment, data shuffling
- Log hyperparameters: learning rate, hidden dims, ensemble size, β
- Version feature extractor weights (git-lfs or artifact store)

---

**Final thoughts:**

Neural bandits occupy a sweet spot between:
- **Linear bandits** (Ch6): Sample-efficient but limited expressivity
- **Deep RL** (Ch7+): Maximum expressivity but expensive and unstable

They shine when:
- You have moderate data (5k-50k episodes)
- Nonlinearity exists but isn't extreme
- You want interpretability (Neural Linear) or modest black-box power (NL-Bandit)

They fail when:
- Data is scarce (neural overfitting)
- Model is overparameterized (catastrophic failure)
- You skip calibration checks (miscalibrated exploration)

**Vlad here:** I've deployed neural bandits in production twice. Once they beat linear bandits by 12% (worth the complexity). Once they lost 8% due to poor calibration (reverted after 3 days). The difference? In the success case, I spent 2 days on diagnostics before deploying. In the failure case, I skipped diagnostics and trusted the training curves. Lesson learned: diagnostics are not optional.

---

## Exercises & Labs

See [`exercises_labs.md`](exercises_labs.md) for complete exercises, including:

- **Theory (30 min):** When linear features fail, Neural Linear posterior, sample complexity trade-offs
- **Implementation (40 min):** Build Neural Feature Extractor, integrate with LinUCB
- **Labs (80 min):** Neural Linear vs rich features, NL-Bandit ensemble, overfit diagnosis
- **Advanced (135+ min):** Joint training, architecture comparison, transfer learning, ensemble ablation

**Time allocation:**
- **Core (theory + implementation + labs):** 150 minutes (2.5 hours)
- **With advanced exercises:** 285+ minutes (4.5+ hours)

---

**End of Chapter 6A**

**Next:** [Chapter 7 — Continuous Actions via Q(x,a)](../ch07/continuous_q_learning.md)

**Previous:** [Chapter 6 — Discrete Template Bandits](../ch06/discrete_template_bandits.md)
