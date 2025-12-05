# Scripts Reference Documentation

This directory contains the executable scripts for the **RL for Search** project (`zoosim`). The scripts are organized by chapter and function, ranging from fundamental bandit demonstrations to advanced deep reinforcement learning benchmarks.

## 📂 Global & Benchmarking

### `scripts/part4_global_comparison.py`
**The "Battle Royale" Benchmark.**
Runs a head-to-head comparison of all major algorithms implemented in the book against a common environment. It serves as the final validation for Part IV.

-   **Algorithms Compared**:
    1.  **Static (Neutral)**: Baseline floor.
    2.  **Random**: Exploration baseline.
    3.  **LinUCB (Ch6)**: Discrete template bandit.
    4.  **Continuous Q (Ch7)**: Ensemble Q-learning + CEM.
    5.  **REINFORCE (Ch8)**: Policy Gradient.
    6.  **SlateQ (Ch12)**: Deep List-wise Ranking.
    7.  **CQL (Ch13)**: Offline Conservative Q-Learning.
-   **Key Metrics**: Average Reward, Final 50-episode average (convergence), Wall-clock time.
-   **Usage**:
    ```bash
    python scripts/part4_global_comparison.py --episodes 1000
    ```

---

## 📂 Chapter 0: Foundations

### `scripts/ch00/toy_problem_solution.py`
**Toy Bandit Problem.**
Solves the motivational example from Chapter 1 using Tabular Q-Learning.
-   **Purpose**: Demonstrates the exploration-exploitation trade-off and value estimation in a simplified 5x5 discrete action grid.
-   **Key Features**: Compares Q-Learning against Random, Static, and Oracle baselines. Generates `toy_problem_learning_curves.png`.

---

## 📂 Chapter 4: World Generation

### `scripts/ch04_demo.py`
**Generative World Demonstration.**
Validates the procedural generation of the e-commerce environment.
-   **Verifies**:
    -   **Catalog**: Lognormal price distributions, linear margins.
    -   **Users**: Segment-specific preferences ($\theta_{price}$, $\theta_{pl}$).
    -   **Queries**: Intent coupling (users query categories they like).
    -   **Determinism**: Ensures strict reproducibility via seeding.

---

## 📂 Chapter 5: Relevance & Reward

### `scripts/ch05_demo.py`
**Relevance & Reward Pipeline Walkthrough.**
Demonstrates the end-to-end simulation flow for a single step.
1.  **Relevance**: Hybrid BM25 + Embedding similarity.
2.  **Features**: 10-dim feature vector construction (standardized).
3.  **Reward**: Multi-objective scalarization (GMV + CM2 + Clicks).

### `scripts/validate_ch05.py`
**Unit Validation.**
Runs deterministic checks to ensure the math in Chapter 5 matches the code (e.g., reward = sum of weighted components).

---

## 📂 Chapter 6: Discrete Template Bandits

This chapter focuses on selecting the best "Boost Template" (e.g., *High Margin*, *Popular*) using Contextual Bandits.

### `scripts/ch06/template_bandits_demo.py`
**Core CPU Demonstration.**
Runs the primary comparison: **Static vs. LinUCB vs. Thompson Sampling**.
-   **Modes**:
    -   `--features simple`: Fails to beat static (demonstrates need for context).
    -   `--features rich`: Beats static (uses user latents + aggregates).
-   **Usage**:
    ```bash
    python scripts/ch06/template_bandits_demo.py --features rich --n-episodes 5000
    ```

### `scripts/ch06/ch06_compute_arc.py`
**Experimental Arc Runner.**
Runs the full narrative sequence (Failure $\to$ Success) and saves JSON summaries for book plots.

### `scripts/ch06/run_bandit_matrix.py`
**Batch Experiment Runner.**
Executes multiple scenarios (seeds, feature modes) in parallel threads to generate robust statistics.

### `scripts/ch06/optimization_gpu/`
**GPU-Accelerated variants.**
Re-implements the environment and bandit loop in PyTorch for massive throughput.
-   `template_bandits_gpu.py`: PyTorch implementation of the environment.
-   `run_bandit_matrix_gpu.py`: GPU batch runner (orders of magnitude faster).

---

## 📂 Chapter 6A: Neural Bandits (Bridge)

### `scripts/ch06a/neural_linear_demo.py`
**Neural Linear UCB.**
Demonstrates representation learning:
1.  **Pretrain**: Learns a neural feature extractor $f_\psi(x)$ from logged data.
2.  **Fine-tune**: Uses LinUCB on top of the learned features.

### `scripts/ch06a/neural_bandits_demo.py`
**End-to-End Neural Bandit.**
Uses a **Deep Q-Ensemble** to estimate $Q(x, a)$ for discrete templates directly, skipping the linear approximation.

### `scripts/ch06a/calibration_check.py`
**Uncertainty Calibration.**
Validates if the ensemble's predicted standard deviation $\sigma(x,a)$ actually correlates with the empirical prediction error $|y - \hat{y}|$.

---

## 📂 Chapter 7: Continuous Control

Moving from discrete templates to continuous ranking weights $w \in \mathbb{R}^d$.

### `scripts/ch07/continuous_actions_demo.py`
**Continuous Optimization Demo.**
-   **Method**: **Q-Ensemble + CEM** (Cross-Entropy Method).
-   **Process**: Learns a value function $Q(x, w)$ and optimizes weights $w$ at runtime using derivative-free optimization.

### `scripts/ch07/compare_discrete_continuous.py`
**Head-to-Head: Ch6 vs Ch7.**
Directly compares **Discrete LinUCB** (limited flexibility) vs **Continuous Q-Learning** (infinite flexibility).
-   **Result**: Continuous usually wins if tuned well, as it can find weight combinations that templates miss.

---

## 📂 Chapter 8: Policy Gradients

### `scripts/ch08/reinforce_demo.py`
**REINFORCE (Policy Gradient).**
Trains a Gaussian Policy $\pi_\theta(a|s)$ to directly optimize expected return.
-   **Observation**: Often high variance and sample inefficient compared to value-based methods in this domain.

### `scripts/ch08/neural_reinforce_demo.py`
**Deep REINFORCE.**
Attempts end-to-end learning from raw features without manual engineering.

---

## 📂 Chapter 12: Deep Ranking (Frontier)

### `scripts/ch12/slate_q_demo.py`
**SlateQ / Neural Ranker.**
A "Learning to Rank" approach where a neural network scores $(Context, Item)$ pairs to construct a slate.
-   **Innovation**: Handles the list-wise nature of the problem better than item-wise bandits.

---

## 📂 Chapter 13: Offline RL (Frontier)

### `scripts/ch13/offline_rl_demo.py`
**Conservative Q-Learning (CQL).**
Demonstrates safe learning from static logs (historical data).
1.  **Collect**: Generates a dataset using a Random policy.
2.  **Train**: Trains a CQL agent on the fixed dataset.
3.  **Eval**: Tests the agent online.
-   **Result**: Beats naive Q-learning, which typically crashes due to overestimating out-of-distribution actions.

---

## 📂 Utilities

### `scripts/validate_knowledge_graph.py`
Validates the project's Knowledge Graph (`docs/knowledge_graph/graph.yaml`) against its schema to ensure structural integrity of concepts and dependencies.

```