# Chapter 2 — Exercises & Labs (Application Mode)

Measure theory meets sampling: every probabilistic definition in Chapter 2 has a concrete simulator counterpart. Use these labs to validate the σ-algebra intuition numerically.

## Lab 2.1 — Segment Mix Sanity Check

Objective: verify that the empirical segment frequencies from `zoosim/world/users.py::sample_user` converge to the probability vector $\rho$ used in §2.3.

```python
import numpy as np
from zoosim.core import config
from zoosim.world import users

cfg = config.load_default_config()
rng = np.random.default_rng(21)
segments = [users.sample_user(config=cfg, rng=rng).segment for _ in range(10_000)]
unique, counts = np.unique(segments, return_counts=True)
empirical = counts / counts.sum()
theoretical = dict(zip(cfg.users.segments, cfg.users.segment_mix))
print("Empirical mix:", dict(zip(unique, empirical.round(3))))
print("Theoretical :", {seg: round(prob, 3) for seg, prob in theoretical.items()})
```

Output:
```
Empirical mix: {'price_hunter': 0.397, 'premium': 0.304, 'bulk_buyer': 0.299}
Theoretical : {'price_hunter': 0.40, 'premium': 0.30, 'bulk_buyer': 0.30}
```

**Tasks**
1. Repeat the experiment with different seeds and report the $\ell_\infty$ deviation $\|\hat{\rho} - \rho\|_\infty$; relate the result to the law of large numbers discussed in Chapter 2.
2. Modify `cfg.users.segment_mix` to create a degenerate distribution and document how this lab exposes the violation.

## Lab 2.2 — Query Measure and Base Score Integration

Objective: link the click-model measure $\mathbb{P}$ defined in §2.6 to simulator code paths.

```python
import numpy as np
from zoosim.core import config
from zoosim.world import catalog, queries, users
from zoosim.ranking import relevance

cfg = config.load_default_config()
rng = np.random.default_rng(3)
cat = catalog.generate_catalog(cfg.catalog, rng)
user = users.sample_user(config=cfg, rng=rng)
query = queries.sample_query(user=user, config=cfg, rng=rng)
scores = relevance.batch_base_scores(query=query, catalog=cat, config=cfg, rng=rng)
print(f"Base score mean: {np.mean(scores):.3f}")
print(f"Base score std : {np.std(scores):.3f}")
```

Output:
```
Base score mean: 0.512
Base score std : 0.207
```

**Tasks**
1. Replace the placeholder `user` with an actual draw from `users.sample_user` and confirm the statistics remain bounded as predicted by Proposition 2.8 on score integrability.
2. Push the histogram of `scores` into the chapter to make the Radon–Nikodym argument tangible (same figure can later fuel Chapter 5 when features are added).
