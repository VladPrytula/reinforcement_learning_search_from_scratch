# Chapter 3 --- Exercises & Labs

We use these labs to keep the operator-theoretic proofs in sync with runnable Bellman code. Each snippet is self-contained so we can execute it directly (e.g., `.venv/bin/python - <<'PY' ... PY`) while cross-referencing Sections 3.7--3.9.

## Lab 3.1 --- Contraction Ratio Tracker

Objective: log $\| \mathcal{T}V_1 - \mathcal{T}V_2 \|_\infty / \|V_1 - V_2\|_\infty$ and compare it to $\gamma$.

```python
import numpy as np

gamma = 0.9
P = np.array(
    [
        [[0.7, 0.3, 0.0], [0.4, 0.6, 0.0]],
        [[0.0, 0.6, 0.4], [0.0, 0.3, 0.7]],
        [[0.2, 0.0, 0.8], [0.1, 0.0, 0.9]],
    ]
)
R = np.array(
    [
        [1.0, 0.5],
        [0.8, 1.2],
        [0.0, 0.4],
    ]
)

def bellman_operator(V, P, R, gamma):
    Q = R + gamma * np.einsum("ijk,k->ij", P, V)
    return Q.max(axis=1)

rng = np.random.default_rng(0)
V1 = rng.normal(size=3)
V2 = rng.normal(size=3)
ratio = np.linalg.norm(
    bellman_operator(V1, P, R, gamma) - bellman_operator(V2, P, R, gamma),
    ord=np.inf,
) / np.linalg.norm(V1 - V2, ord=np.inf)
print(f"Contraction ratio: {ratio:.3f} (theory bound = {gamma:.3f})")
```

Output:
```
Contraction ratio: 0.872 (theory bound = 0.900)
```

**Tasks**
1. Explain the slack between the bound and the observation (hint: the max operator is 1-Lipschitz, so the true ratio is often strictly less than $\gamma$).
2. Log the ratio across multiple seeds and include the extrema in Chapter 3 to make [EQ-3.16] concrete.

## Lab 3.2 --- Value Iteration Wall-Clock Profiling

Objective: verify the $O\!\left(\frac{1}{1-\gamma}\right)$ convergence rate numerically by reusing the same toy kernel.

```python
import numpy as np

P = np.array(
    [
        [[0.7, 0.3, 0.0], [0.4, 0.6, 0.0]],
        [[0.0, 0.6, 0.4], [0.0, 0.3, 0.7]],
        [[0.2, 0.0, 0.8], [0.1, 0.0, 0.9]],
    ]
)
R = np.array(
    [
        [1.0, 0.5],
        [0.8, 1.2],
        [0.0, 0.4],
    ]
)

def value_iteration(P, R, gamma, tol=1e-6, max_iters=500):
    V = np.zeros(P.shape[0])
    for k in range(max_iters):
        Q = R + gamma * np.einsum("ijk,k->ij", P, V)
        V_new = Q.max(axis=1)
        if np.linalg.norm(V_new - V, ord=np.inf) < tol:
            return V_new, k + 1
        V = V_new
    raise RuntimeError("Value iteration did not converge")

stats = {}
for gamma in [0.5, 0.7, 0.9]:
    _, iters = value_iteration(P, R, gamma=gamma)
    stats[gamma] = iters
print(stats)
```

Output:
```
{0.5: 21, 0.7: 39, 0.9: 128}
```

**Tasks**
1. Plot the iteration counts against $\frac{1}{1-\gamma}$ and reference the figure when explaining [COR-3.7.3] (value iteration convergence rate).
2. Re-run the sweep after perturbing `R` with zero-mean noise to visualize the reward-perturbation sensitivity bound [PROP-3.7.4].
