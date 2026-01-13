# Chapter 2 — Exercises & Labs (Application Mode)

Measure theory meets sampling: every probabilistic definition in Chapter 2 has a concrete simulator counterpart. Use these labs to validate the $\sigma$-algebra intuition numerically.

## Lab 2.1 — Segment Mix Sanity Check {#lab-21--segment-mix-sanity-check}

Objective: verify that empirical segment frequencies converge to the segment distribution $\mathbf{p}_{\text{seg}}$ from [DEF-2.2.6].

This lab is implemented in `scripts/ch02/lab_solutions.py` (see `ch02_lab_solutions.md` for a full transcript).

```python
from scripts.ch02.lab_solutions import lab_2_1_segment_mix_sanity_check

_ = lab_2_1_segment_mix_sanity_check(seed=21, n_samples=10_000, verbose=True)
```

Output (actual):
```
======================================================================
Lab 2.1: Segment Mix Sanity Check
======================================================================

Sampling 10,000 users from segment distribution (seed=21)...

Theoretical segment mix (from config):
  price_hunter   : p_seg = 0.350
  pl_lover       : p_seg = 0.250
  premium        : p_seg = 0.150
  litter_heavy   : p_seg = 0.250

Empirical segment frequencies (n=10,000):
  price_hunter   : p_hat_seg = 0.335  (Δ = -0.015)
  pl_lover       : p_hat_seg = 0.254  (Δ = +0.004)
  premium        : p_hat_seg = 0.153  (Δ = +0.003)
  litter_heavy   : p_hat_seg = 0.258  (Δ = +0.008)

Deviation metrics:
  L∞ (max deviation): 0.015
  L1 (total variation): 0.030
  L2 (Euclidean):       0.018

[!] L∞ deviation (0.015) exceeds 3$\sigma$ (0.014)
```

**Tasks**
1. Repeat the experiment with different seeds and report the $\ell_\infty$ deviation $\|\hat{\mathbf{p}}_{\text{seg}} - \mathbf{p}_{\text{seg}}\|_\infty$; relate the result to the law of large numbers discussed in Chapter 2.
2. Run `scripts/ch02/lab_solutions.py::lab_2_1_degenerate_distribution` and interpret each test case in terms of positivity/overlap from §2.6 (support coverage for Radon–Nikodym derivatives).

## Lab 2.2 — Query Measure and Base Score Integration {#lab-22--query-measure-and-base-score-integration}

Objective: link the click-model measure $\mathbb{P}$ defined in §2.6 to simulator code paths, and verify square-integrability predicted by [PROP-2.8.1].

```python
from scripts.ch02.lab_solutions import lab_2_2_base_score_integration

_ = lab_2_2_base_score_integration(seed=3, verbose=True)
```

Output (actual):
```
======================================================================
Lab 2.2: Query Measure and Base Score Integration
======================================================================

Generating catalog and sampling users/queries (seed=3)...

Catalog statistics:
  Products: 10,000 (simulated)
  Categories: ['dog_food', 'cat_food', 'litter', 'toys']
  Embedding dimension: 16

User/Query samples (n=100):

Sample 1:
  User segment: litter_heavy
  Query type: brand
  Query intent: litter

Sample 2:
  User segment: price_hunter
  Query type: category
  Query intent: litter

...

Base score statistics across 100 queries × 100 products each:

  Score mean:  0.098
  Score std:   0.221
  Score min:   -0.558
  Score max:   0.933

Score percentiles:
  5th: -0.258
  25th: -0.057
  50th: 0.095
  75th: 0.248
  95th: 0.466

[OK] Scores are square-integrable (finite variance) as required by Proposition 2.8.1
[OK] Score std $\approx 0.22$ (finite second moment)
[!] Scores NOT bounded to [0,1]---Gaussian noise makes them unbounded
```

**Tasks**
1. Examine the score distribution: compute mean, std, min, max, and selected quantiles (5%, 95%). Note that scores are *not* bounded to $[0,1]$ but are square-integrable with finite variance, as predicted by [PROP-2.8.1]. What empirical distribution do we observe? Do any scores fall outside $[-1, 2]$?
2. Push the histogram of `scores` into the chapter to make the Radon-Nikodym argument tangible (same figure can later fuel Chapter 5 when features are added).

## Lab 2.3 — Textbook Click Model Verification {#lab-23--textbook-click-model-verification}

Objective: verify that toy implementations of PBM ([DEF-2.5.1], [EQ-2.1]) and DBN ([DEF-2.5.2], [EQ-2.3]) match their theoretical predictions exactly.

```python
from scripts.ch02.lab_solutions import lab_2_3_textbook_click_models

_ = lab_2_3_textbook_click_models(seed=42, verbose=True)
```

Output (actual):
```
======================================================================
Lab 2.3: Textbook Click Model Verification
======================================================================

Verifying PBM [DEF-2.5.1] and DBN [DEF-2.5.2] match theory exactly.

--- Part A: Position Bias Model (PBM) ---

Configuration:
  Positions: 10
  theta_k (examination): exponential decay with lambda=0.3
  rel(p_k) (relevance): linear decay from 0.70 to 0.25

Theoretical prediction [EQ-2.1]:
  P(C_k = 1) = rel(p_k) * theta_k

Simulating 50,000 sessions...

Position |  theta_k | rel(p_k) | CTR theory | CTR empirical |    Error
----------------------------------------------------------------------
       1 |    0.900 |     0.70 |     0.6300 |        0.6305 |   0.0005
       2 |    0.667 |     0.65 |     0.4334 |        0.4300 |   0.0034
       3 |    0.494 |     0.60 |     0.2964 |        0.2957 |   0.0007
       4 |    0.366 |     0.55 |     0.2013 |        0.2015 |   0.0002
       5 |    0.271 |     0.50 |     0.1355 |        0.1376 |   0.0020
       6 |    0.201 |     0.45 |     0.0904 |        0.0888 |   0.0015
       7 |    0.149 |     0.40 |     0.0595 |        0.0587 |   0.0008
       8 |    0.110 |     0.35 |     0.0386 |        0.0387 |   0.0001
       9 |    0.082 |     0.30 |     0.0245 |        0.0250 |   0.0005
      10 |    0.060 |     0.25 |     0.0151 |        0.0148 |   0.0003

Max absolute error: 0.0034
checkmark PBM: Empirical CTRs match [EQ-2.1] within 1% tolerance

--- Part B: Dynamic Bayesian Network (DBN) ---

Configuration:
  rel(p_k) * s(p_k) (relevance * satisfaction):
    [0.14, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

Theoretical prediction [EQ-2.3]:
  P(E_k = 1) = prod_{j<k} [1 - rel(p_j) * s(p_j)]

Simulating 50,000 sessions...

Position | P(E_k) theory | P(E_k) empirical |    Error
-------------------------------------------------------
       1 |        1.0000 |           1.0000 |   0.0000
       2 |        0.8600 |           0.8580 |   0.0020
       3 |        0.7538 |           0.7536 |   0.0002
       4 |        0.6724 |           0.6714 |   0.0010
       5 |        0.6095 |           0.6081 |   0.0014
       6 |        0.5608 |           0.5595 |   0.0012
       7 |        0.5229 |           0.5238 |   0.0009
       8 |        0.4936 |           0.4953 |   0.0017
       9 |        0.4712 |           0.4728 |   0.0017
      10 |        0.4542 |           0.4565 |   0.0023

Max absolute error: 0.0023
checkmark DBN: Examination probabilities match [EQ-2.3] within 1% tolerance

--- Part C: PBM vs DBN Comparison ---

Examination probability at position 5:
  PBM: P(E_5) = theta_5 = 0.271 (fixed by position)
  DBN: P(E_5) = 0.610 (depends on cascade)

Key insight:
  DBN predicts HIGHER examination at later positions because users
  who reach position 5 are 'unsatisfied browsers' who continue scanning.
  PBM's fixed theta_k is simpler but ignores this selection effect.
```

**Tasks**
1. Verify that the DBN simulation in `scripts/ch02/lab_solutions.py::simulate_dbn` implements [EQ-2.3]: $P(E_k = 1) = \prod_{j < k} [1 - \text{rel}(p_j) \cdot s(p_j)]$, then vary satisfaction probabilities and re-run.
2. Compare PBM and DBN examination probabilities at position 5. Explain why DBN predicts higher examination for users who reach later positions.

## Lab 2.4 — Nesting Verification ([PROP-2.5.4]) {#lab-24--nesting-verification}

Objective: demonstrate that the Utility-Based Cascade Model (Section 2.5.4) reduces to PBM when utility weights are zeroed, verifying the **nesting property** from [PROP-2.5.4].

```python
from scripts.ch02.lab_solutions import lab_2_4_nesting_verification

_ = lab_2_4_nesting_verification(seed=42, verbose=True)
```

Output (actual):
```
======================================================================
Lab 2.4: Nesting Verification ([PROP-2.5.4])
======================================================================

Goal: Show that Utility-Based Cascade reduces to PBM when utility
weights are zeroed, verifying the nesting property from [PROP-2.5.4].

--- Configuration ---

Full Utility-Based Cascade:
  alpha_price = 0.8
  alpha_pl = 1.2
  sigma_u = 0.8
  satisfaction_gain = 0.5
  abandonment_threshold = -2.0

PBM-like Configuration:
  alpha_price = 0.0
  alpha_pl = 0.0
  sigma_u = 0.0
  satisfaction_gain = 0.0
  abandonment_threshold = -100.0

Simulating 5,000 sessions for each configuration...

--- Results ---

Position |   Full CTR | PBM-like CTR | Difference
--------------------------------------------------
       1 |     0.4168 |       0.5096 |    -0.0928
       2 |     0.2394 |       0.3620 |    -0.1226
       3 |     0.1376 |       0.2342 |    -0.0966
       4 |     0.0726 |       0.1502 |    -0.0776
       5 |     0.0448 |       0.0872 |    -0.0424
       6 |     0.0272 |       0.0444 |    -0.0172
       7 |     0.0078 |       0.0246 |    -0.0168
       8 |     0.0068 |       0.0128 |    -0.0060
       9 |     0.0024 |       0.0064 |    -0.0040
      10 |     0.0004 |       0.0034 |    -0.0030

--- Stop Reason Distribution ---

Reason          |  Full Config |   PBM-like
---------------------------------------------
exam_fail       |        94.6% |      99.3%
abandonment     |         5.1% |       0.0%
purchase_limit  |         0.2% |       0.0%
end             |         0.2% |       0.7%

--- Interpretation ---

Key observations:
  1. PBM-like config has NO abandonment (threshold = -100)
  2. PBM-like config has NO purchase limit stopping
  3. PBM-like CTR depends only on position (via pos_bias)
  4. Full config CTR varies with utility (price, PL, noise)

This verifies [PROP-2.5.4]: Utility-Based Cascade nests PBM
as a special case when utility dependence is disabled.
```

**Tasks**
1. Re-run the lab with different seeds and verify that the PBM-like configuration produces history-independent click patterns (no abandonment, no purchase-limit stopping), while the full model exhibits cascade effects.
2. Progressively re-enable utility terms ($\alpha_{\text{price}}$, then $\alpha_{\text{pl}}$, then $\alpha_{\text{cat}}$) in `scripts/ch02/lab_solutions.py::lab_2_4_nesting_verification` and record how CTR by position changes.

## Lab 2.5 — Utility-Based Cascade Dynamics ([DEF-2.5.3]) {#lab-25--utility-based-cascade-dynamics}

Objective: verify the three key mechanisms of the production click model from Section 2.5.4: position decay, satisfaction dynamics, and stopping conditions.

```python
from scripts.ch02.lab_solutions import lab_2_5_utility_cascade_dynamics

_ = lab_2_5_utility_cascade_dynamics(seed=42, verbose=True)
```

Output (actual):
```
======================================================================
Lab 2.5: Utility-Based Cascade Dynamics ([DEF-2.5.3])
======================================================================

Verifying three key mechanisms:
  1. Position decay (pos_bias)
  2. Satisfaction dynamics (gain/decay)
  3. Stopping conditions

Configuration:
  Positions: 20
  satisfaction_gain: 0.5
  satisfaction_decay: 0.2
  abandonment_threshold: -2.0
  pos_bias (category, first 5): [1.2, 0.9, 0.7, 0.5, 0.3]

Simulating 2,000 sessions...

--- Part 1: Position Decay ---

Position |  Exam Rate |   CTR|Exam |   pos_bias
--------------------------------------------------
       1 |      0.767 |      0.387 |       1.20
       2 |      0.520 |      0.563 |       0.90
       3 |      0.349 |      0.401 |       0.70
       4 |      0.197 |      0.353 |       0.50
       5 |      0.100 |      0.485 |       0.30
       6 |      0.052 |      0.533 |       0.20
       7 |      0.025 |      0.353 |       0.20
       8 |      0.015 |      0.600 |       0.20
       9 |      0.005 |      0.400 |       0.20
      10 |      0.002 |      1.000 |       0.20

Observation: Examination rate decays with position, matching pos_bias pattern.

--- Part 2: Satisfaction Dynamics ---

Sample satisfaction trajectories (first 5 sessions):
  Session 1: 0.00 -> -0.20 (exam_fail)
  Session 2: 0.00 -> -0.20 -> 0.22 -> 0.02 -> -1.75 (exam_fail)
  Session 3: 0.00 -> -0.20 -> 0.18 -> -0.29 (exam_fail)
  Session 4: 0.00 -> -0.20 -> 0.23 -> 0.03 -> -0.44 -> -0.64 -> -0.33 -> -0.53 ... (exam_fail)
  Session 5: 0.00 -> -0.20 (exam_fail)

Final satisfaction statistics:
  Mean: -0.49
  Std:  0.71
  Min:  -3.47
  Max:  1.79

--- Part 3: Stopping Conditions ---

Stop Reason        |    Count | Percentage
---------------------------------------------
exam_fail          |     1900 |      95.0%
abandonment        |       98 |       4.9%
purchase_limit     |        2 |       0.1%
end                |        0 |       0.0%

Session length statistics:
  Mean: 2.0 positions
  Std:  1.9
  Median: 2

Clicks per session:
  Mean: 0.90
  Max:  7

--- Verification Summary ---

checkmark Position decay: Examination rate follows pos_bias pattern
checkmark Satisfaction dynamics: Trajectories show gain on click, decay on no-click
checkmark Stopping conditions: All three mechanisms observed (exam, abandon, limit)
```

**Tasks**
1. Plot the satisfaction trajectory $S_k$ for 10 representative sessions. Identify sessions that ended due to: (a) examination failure, (b) satisfaction threshold crossing, (c) purchase limit.
2. Verify that the mean examination decay matches the position bias vector `pos_bias` used in the model.
3. Modify `satisfaction_gain` and `satisfaction_decay` parameters. Document how this affects: session length distribution, abandonment rate, and total clicks per session.
