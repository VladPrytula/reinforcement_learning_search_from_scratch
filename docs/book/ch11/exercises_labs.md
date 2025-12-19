# Chapter 11 — Exercises & Labs

**Companion to Chapter 11: Multi-Episode Inter-Session MDP**

This document provides detailed lab protocols for the experiments described in Chapter 11. Each lab connects the multi-episode theory to concrete simulator code.

---

## Lab 11.1 — Single-Step vs Multi-Episode Value {#lab-111-single-vs-multi}

**Objective.** Compare policy rankings under:

- Single-session proxy reward (GMV + δ·CLICKS) in the standard `ZooplusSearchEnv`.
- Multi-episode discounted GMV in `MultiSessionEnv` using the retention model.

**Scripts.**

- `scripts/ch11/lab_01_single_vs_multi_episode.py`

**Protocol.**

1. Define a small family of policies (e.g., zero-boost baseline and a CM2-heavy heuristic) using `scripts/ch11/utils/policies.py`.
2. For each policy:
   - Estimate single-session value by running `ZooplusSearchEnv` for many users with the fixed boost vector.
   - Estimate multi-episode value by running `MultiSessionEnv` and computing discounted GMV using `scripts/ch11/utils/trajectory.py`.
3. Rank policies under each metric and compute Spearman correlation.

**Acceptance criterion.** Spearman ρ between the two rankings is at least 0.80 for the policies considered.

---

## Lab 11.2 — Retention Curves and Monotonicity {#lab-112-retention-curves}

**Objective.** Validate that the retention model is monotone in engagement and visualize how clicks and satisfaction shape return probability.

**Scripts.**

- `scripts/ch11/lab_02_retention_curves.py`

**Protocol.**

1. Fix `RetentionConfig` at its default values (`base_rate`, `click_weight`, `satisfaction_weight`).
2. Sweep `clicks ∈ {0, …, 15}` and `satisfaction ∈ [0, 1]` on a grid.
3. Compute `P(return)` for each pair using `zoosim/multi_episode/retention.py:return_probability`.
4. Generate:
   - A heatmap of `P(return)` over the (clicks, satisfaction) grid.
   - Line plots of `P(return)` vs clicks at several satisfaction levels.
   - For the canonical run in this book, we save:
     - `docs/book/ch11/figures/lab11_retention_heatmap.png`
     - `docs/book/ch11/figures/lab11_retention_lines.png`
5. Run the monotonicity check to ensure `P(return)` does not decrease when clicks or satisfaction increase.

**Acceptance criteria.**

- `P(return)` is monotone non-decreasing in both clicks and satisfaction (no assertion failures).
- The heatmap and line plots qualitatively match the behavior described in §11.3–11.4 (higher engagement → higher retention).

---

## Lab 11.3 — Value Estimation Stability {#lab-113-value-stability}

**Objective.** Assess the stability of multi-episode value estimates across environment seeds.

**Scripts.**

- `scripts/ch11/lab_03_value_stability.py`

**Protocol.**

1. Fix a simple policy (e.g., the zero-boost baseline from `scripts/ch11/utils/policies.py`).
2. Choose a set of environment seeds (e.g., 10 distinct seeds).
3. For each seed:
   - Instantiate `MultiSessionEnv` with that seed.
   - Simulate a fixed number of users and compute their discounted GMV using `collect_trajectory`.
   - Record the mean value estimate for that seed.
4. Aggregate across seeds to compute:
   - Mean, standard deviation, coefficient of variation (CV).
   - Empirical 95% confidence interval from the seed-wise estimates.

**Acceptance criterion.** Coefficient of variation (CV) across seeds is below 0.10 for the chosen policy and sample size.

For the representative run used in the chapter text, see:
- `docs/book/ch11/data/lab11_value_stability_run1.txt`

---

## Additional Exercises

1. **Policy family extension.** Extend Lab 11.1 with additional policies (e.g., segment-aware boosts) and study when single-step and multi-episode rankings diverge.
2. **Retention parameter sensitivity.** Modify `RetentionConfig` (e.g., stronger click weight) and repeat Lab 11.2 to see how the heatmap deforms.
3. **Sample size vs stability.** In Lab 11.3, vary the number of users per seed and plot CV as a function of sample size to quantify the variance–cost trade-off.
