# Chapter 7 Labs: Continuous Control for Ranking

These labs explore the transition from discrete template selection to continuous action optimization. You will use the Q-Ensemble architecture and the Cross-Entropy Method (CEM) optimizer to learn fine-grained ranking policies.

## Lab 7.1: The Q-Ensemble and Uncertainty

**Objective:** Understand how the ensemble provides uncertainty estimates that enable exploration.

**Files:**
- `zoosim/policies/q_ensemble.py`: The implementation of the regressor.
- `tests/ch07/test_q_ensemble.py`: Validation tests.

**Instructions:**
1.  Run the unit tests to verify the ensemble logic:
    ```bash
    uv run pytest tests/ch07/test_q_ensemble.py
    ```
2.  Open `zoosim/policies/q_ensemble.py`. Look at `predict`. Notice how it returns `std` across the ensemble members.
3.  **Experiment:** In a Python shell (or new script), create a `QEnsembleRegressor`. Train it on a small dataset (e.g., points on a sine wave). Query it at points *far* from the training data.
    - Does `std` increase?
    - How does `n_ensembles` affect the quality of the uncertainty estimate?

## Lab 7.2: Optimizing with CEM

**Objective:** Visualize how CEM finds the maximum of a function without gradients.

**Files:**
- `zoosim/optimizers/cem.py`: The optimizer.
- `tests/ch07/test_cem.py`: Validation tests.

**Instructions:**
1.  Run the CEM tests:
    ```bash
    uv run pytest tests/ch07/test_cem.py
    ```
2.  **Constraint Check:** The test `test_cem_trust_region_projection` verifies that the optimizer respects trust regions. Why is this important for a live production system? (Hint: Safety).

## Lab 7.3: Continuous Actions Demo (The "Optimizer in the Loop")

**Objective:** Train a continuous RL agent to beat the best static template.

**Files:**
- `scripts/ch07/continuous_actions_demo.py`: The main training loop.

**Instructions:**
1.  Run the demo with default settings (this takes ~1-2 minutes):
    ```bash
    python scripts/ch07/continuous_actions_demo.py --n-episodes 3000
    ```
2.  **Observe the learning curve:**
    - Early episodes: The agent (CEM) performs worse than random/static because it is exploring (high `beta`).
    - Middle episodes: As `beta` decays and the Q-model improves, performance climbs.
    - Final episodes: Does it beat the "Static" baseline? (It should, by 10-30%).

**Extensions (Try these!):**
1.  **Greedy vs. UCB:** Modify the script to set `beta = 0.0` from the start. Does the agent get stuck in a local optimum (suboptimal policy)?
2.  **Trust Regions:** Modify `CEMAgent.select_action` in the script to pass `trust_region_center=prev_action` (you'll need to track the previous action). Set `trust_region_radius=0.1`.
    - *Hypothesis:* This should make learning smoother but potentially slower to react to context changes.

## Acceptance Criteria

- [ ] `pytest tests/ch07` passes (verifies core math components).
- [ ] `continuous_actions_demo.py` runs to completion.
- [ ] The CEM agent achieves > 1.1x the GMV of the Static baseline in the final evaluation blocks of the demo.
- [ ] Compared on the same evaluation protocol, the CEM agent matches or exceeds the best rich-feature bandit from Chapter 6 (LinUCB/TS) on GMV, within error bars.
- [ ] Trust-region tuning experiments keep $\Delta\text{rank}@k$ within an acceptable band (no catastrophic rank flips) while still improving GMV over the static and rich-feature baselines.
- [ ] An uncertainty calibration check (following §7.5.3) has been run on logged $(\mu, \sigma, r)$ triples from `continuous_actions_demo.py`, and the resulting standardized errors show reasonable coverage (e.g., empirical 95% interval close to the nominal target).
