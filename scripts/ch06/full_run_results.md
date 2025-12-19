 ./scripts/ch06/run_full_verification.sh 2>&1 | tee ch06_verification.log
==============================================================================
CHAPTER 6 FULL VERIFICATION SUITE
==============================================================================

Configuration:
  Static episodes:  2000
  Bandit episodes:  20000
  World seed:       20250322
  Bandit seed:      20250349
  Output directory: docs/book/drafts/ch06/data/verification_20251203T104348

Start time: Wed Dec  3 10:43:48 CET 2025

==============================================================================
TEST 1: Simple Features - The Failure Mode (§6.5)
==============================================================================\033[0m

Expected: Bandits underperform or match static baseline with simple features


================================================================================
CHAPTER 6 — TEMPLATE BANDITS DEMONSTRATION
================================================================================

Experiment horizons: static=2,000 episodes, bandits=20,000 episodes.
Feature mode: 'simple'

Generating catalog with 10,000 products (world seed=20250322)...

Resolved bandit hyperparameters → prior_weight=0, lin_alpha=1.00, ts_sigma=1.00

Running static template baselines for 2000 episodes...
  Static templates progress:   0.1% (1/2000) | best template ID 4 avg reward 20.70
  Static templates progress:   2.0% (40/2000) | best template ID 4 avg reward 8.55
  Static templates progress:   4.0% (80/2000) | best template ID 4 avg reward 10.36
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 11.91
  Static templates progress:   8.0% (160/2000) | best template ID 4 avg reward 9.97
  Static templates progress:  10.0% (200/2000) | best template ID 4 avg reward 9.41
  Static templates progress:  12.0% (240/2000) | best template ID 4 avg reward 9.80
  Static templates progress:  14.0% (280/2000) | best template ID 4 avg reward 9.60
  Static templates progress:  16.0% (320/2000) | best template ID 4 avg reward 9.90
  Static templates progress:  18.0% (360/2000) | best template ID 4 avg reward 9.64
  Static templates progress:  20.0% (400/2000) | best template ID 4 avg reward 9.57
  Static templates progress:  22.0% (440/2000) | best template ID 4 avg reward 9.49
  Static templates progress:  24.0% (480/2000) | best template ID 4 avg reward 9.21
  Static templates progress:  26.0% (520/2000) | best template ID 4 avg reward 8.67
  Static templates progress:  28.0% (560/2000) | best template ID 4 avg reward 8.82
  Static templates progress:  30.0% (600/2000) | best template ID 4 avg reward 8.78
  Static templates progress:  32.0% (640/2000) | best template ID 4 avg reward 8.88
  Static templates progress:  34.0% (680/2000) | best template ID 4 avg reward 8.75
  Static templates progress:  36.0% (720/2000) | best template ID 4 avg reward 8.77
  Static templates progress:  38.0% (760/2000) | best template ID 4 avg reward 8.52
  Static templates progress:  40.0% (800/2000) | best template ID 4 avg reward 8.37
  Static templates progress:  42.0% (840/2000) | best template ID 4 avg reward 8.39
  Static templates progress:  44.0% (880/2000) | best template ID 4 avg reward 8.30
  Static templates progress:  46.0% (920/2000) | best template ID 4 avg reward 8.22
  Static templates progress:  48.0% (960/2000) | best template ID 4 avg reward 8.38
  Static templates progress:  50.0% (1000/2000) | best template ID 4 avg reward 8.14
  Static templates progress:  52.0% (1040/2000) | best template ID 4 avg reward 8.01
  Static templates progress:  54.0% (1080/2000) | best template ID 2 avg reward 7.92
  Static templates progress:  56.0% (1120/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  58.0% (1160/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  60.0% (1200/2000) | best template ID 2 avg reward 8.09
  Static templates progress:  62.0% (1240/2000) | best template ID 2 avg reward 8.06
  Static templates progress:  64.0% (1280/2000) | best template ID 2 avg reward 8.08
  Static templates progress:  66.0% (1320/2000) | best template ID 2 avg reward 8.00
  Static templates progress:  68.0% (1360/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.96
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.90
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 7.87
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 7.72
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.67
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.54
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.55
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.53
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.48
  Static templates progress:  98.0% (1960/2000) | best template ID 4 avg reward 7.46
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.56


Best static template: ID=4 (Premium) with avg reward=7.56, GMV=7.11, CM2=0.74

Running LinUCB with features=simple for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  LINUCB progress:   2.0% (400/20000) | avg reward 4.76
  LINUCB progress:   4.0% (800/20000) | avg reward 4.96
  LINUCB progress:   6.0% (1200/20000) | avg reward 5.07
  LINUCB progress:   8.0% (1600/20000) | avg reward 5.21
  LINUCB progress:  10.0% (2000/20000) | avg reward 5.33
  LINUCB progress:  12.0% (2400/20000) | avg reward 5.32
  LINUCB progress:  14.0% (2800/20000) | avg reward 5.38
  LINUCB progress:  16.0% (3200/20000) | avg reward 5.40
  LINUCB progress:  18.0% (3600/20000) | avg reward 5.36
  LINUCB progress:  20.0% (4000/20000) | avg reward 5.41
  LINUCB progress:  22.0% (4400/20000) | avg reward 5.49
  LINUCB progress:  24.0% (4800/20000) | avg reward 5.50
  LINUCB progress:  26.0% (5200/20000) | avg reward 5.54
  LINUCB progress:  28.0% (5600/20000) | avg reward 5.67
  LINUCB progress:  30.0% (6000/20000) | avg reward 5.64
  LINUCB progress:  32.0% (6400/20000) | avg reward 5.72
  LINUCB progress:  34.0% (6800/20000) | avg reward 5.77
  LINUCB progress:  36.0% (7200/20000) | avg reward 5.80
  LINUCB progress:  38.0% (7600/20000) | avg reward 5.81
  LINUCB progress:  40.0% (8000/20000) | avg reward 5.79
  LINUCB progress:  42.0% (8400/20000) | avg reward 5.75
  LINUCB progress:  44.0% (8800/20000) | avg reward 5.74
  LINUCB progress:  46.0% (9200/20000) | avg reward 5.75
  LINUCB progress:  48.0% (9600/20000) | avg reward 5.73
  LINUCB progress:  50.0% (10000/20000) | avg reward 5.74
  LINUCB progress:  52.0% (10400/20000) | avg reward 5.75
  LINUCB progress:  54.0% (10800/20000) | avg reward 5.75
  LINUCB progress:  56.0% (11200/20000) | avg reward 5.75
  LINUCB progress:  58.0% (11600/20000) | avg reward 5.74
  LINUCB progress:  60.0% (12000/20000) | avg reward 5.73
  LINUCB progress:  62.0% (12400/20000) | avg reward 5.71
  LINUCB progress:  64.0% (12800/20000) | avg reward 5.72
  LINUCB progress:  66.0% (13200/20000) | avg reward 5.74
  LINUCB progress:  68.0% (13600/20000) | avg reward 5.75
  LINUCB progress:  70.0% (14000/20000) | avg reward 5.74
  LINUCB progress:  72.0% (14400/20000) | avg reward 5.71
  LINUCB progress:  74.0% (14800/20000) | avg reward 5.70
  LINUCB progress:  76.0% (15200/20000) | avg reward 5.71
  LINUCB progress:  78.0% (15600/20000) | avg reward 5.71
  LINUCB progress:  80.0% (16000/20000) | avg reward 5.67
  LINUCB progress:  82.0% (16400/20000) | avg reward 5.67
  LINUCB progress:  84.0% (16800/20000) | avg reward 5.66
  LINUCB progress:  86.0% (17200/20000) | avg reward 5.67
  LINUCB progress:  88.0% (17600/20000) | avg reward 5.67
  LINUCB progress:  90.0% (18000/20000) | avg reward 5.68
  LINUCB progress:  92.0% (18400/20000) | avg reward 5.66
  LINUCB progress:  94.0% (18800/20000) | avg reward 5.65
  LINUCB progress:  96.0% (19200/20000) | avg reward 5.64
  LINUCB progress:  98.0% (19600/20000) | avg reward 5.63
  LINUCB progress: 100.0% (20000/20000) | avg reward 5.62

Running Thompson Sampling with features=simple for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 0.00
  TS progress:   2.0% (400/20000) | avg reward 6.01
  TS progress:   4.0% (800/20000) | avg reward 5.68
  TS progress:   6.0% (1200/20000) | avg reward 5.54
  TS progress:   8.0% (1600/20000) | avg reward 5.83
  TS progress:  10.0% (2000/20000) | avg reward 5.78
  TS progress:  12.0% (2400/20000) | avg reward 5.67
  TS progress:  14.0% (2800/20000) | avg reward 5.56
  TS progress:  16.0% (3200/20000) | avg reward 5.46
  TS progress:  18.0% (3600/20000) | avg reward 5.48
  TS progress:  20.0% (4000/20000) | avg reward 5.51
  TS progress:  22.0% (4400/20000) | avg reward 5.45
  TS progress:  24.0% (4800/20000) | avg reward 5.57
  TS progress:  26.0% (5200/20000) | avg reward 5.65
  TS progress:  28.0% (5600/20000) | avg reward 5.68
  TS progress:  30.0% (6000/20000) | avg reward 5.74
  TS progress:  32.0% (6400/20000) | avg reward 5.82
  TS progress:  34.0% (6800/20000) | avg reward 5.88
  TS progress:  36.0% (7200/20000) | avg reward 5.85
  TS progress:  38.0% (7600/20000) | avg reward 5.88
  TS progress:  40.0% (8000/20000) | avg reward 5.89
  TS progress:  42.0% (8400/20000) | avg reward 5.89
  TS progress:  44.0% (8800/20000) | avg reward 5.92
  TS progress:  46.0% (9200/20000) | avg reward 5.91
  TS progress:  48.0% (9600/20000) | avg reward 5.90
  TS progress:  50.0% (10000/20000) | avg reward 5.93
  TS progress:  52.0% (10400/20000) | avg reward 5.93
  TS progress:  54.0% (10800/20000) | avg reward 5.93
  TS progress:  56.0% (11200/20000) | avg reward 5.92
  TS progress:  58.0% (11600/20000) | avg reward 5.92
  TS progress:  60.0% (12000/20000) | avg reward 5.97
  TS progress:  62.0% (12400/20000) | avg reward 6.03
  TS progress:  64.0% (12800/20000) | avg reward 6.10
  TS progress:  66.0% (13200/20000) | avg reward 6.14
  TS progress:  68.0% (13600/20000) | avg reward 6.18
  TS progress:  70.0% (14000/20000) | avg reward 6.26
  TS progress:  72.0% (14400/20000) | avg reward 6.27
  TS progress:  74.0% (14800/20000) | avg reward 6.34
  TS progress:  76.0% (15200/20000) | avg reward 6.39
  TS progress:  78.0% (15600/20000) | avg reward 6.44
  TS progress:  80.0% (16000/20000) | avg reward 6.48
  TS progress:  82.0% (16400/20000) | avg reward 6.48
  TS progress:  84.0% (16800/20000) | avg reward 6.51
  TS progress:  86.0% (17200/20000) | avg reward 6.53
  TS progress:  88.0% (17600/20000) | avg reward 6.54
  TS progress:  90.0% (18000/20000) | avg reward 6.58
  TS progress:  92.0% (18400/20000) | avg reward 6.60
  TS progress:  94.0% (18800/20000) | avg reward 6.61
  TS progress:  96.0% (19200/20000) | avg reward 6.64
  TS progress:  98.0% (19600/20000) | avg reward 6.67
  TS progress: 100.0% (20000/20000) | avg reward 6.69

  Static (best) segment stats:   0.0% (1/20000) | avg reward 0.00
  Static (best) segment stats:   2.0% (400/20000) | avg reward 7.84
  Static (best) segment stats:   4.0% (800/20000) | avg reward 7.25
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.07
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.12
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.20
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.11
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.15
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.21
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.36
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.41
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.39
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.40
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.50
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.51
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.54
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.60
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.60
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.61
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.63
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.66
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.72
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.75
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.70
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.70
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.71
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.68
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.67
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.68
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.71
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.72
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.75
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.73
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.73
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.71
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.72
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.70
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.70
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.73
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.71
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.68
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.64
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.64
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.68
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.68
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.67
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.64
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.66
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.64
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.65
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.65


Static templates (per-episode averages):
ID  Template             Reward         GMV         CM2
 0  Neutral                5.75        5.28        0.50
 1  High Margin            5.44        5.07        0.63
 2  CM2 Boost              7.35        6.73        0.61
 3  Popular                4.79        4.53        0.52
 4  Premium                7.56        7.11        0.74
 5  Budget                 3.44        3.04        0.26
 6  Discount               5.04        4.62        0.45
 7  Strategic              4.83        3.99       -0.13

Summary (per-episode averages):
Policy                  Reward         GMV         CM2  ΔGMV vs static
Static-Premium            7.56        7.11        0.74          +0.00%
LinUCB                    5.62        5.12        0.46         -27.95%
ThompsonSampling          6.69        6.18        0.63         -13.07%

Per-segment GMV (static best vs bandits):
Segment          Static GMV  LinUCB GMV      TS GMV    LinUCB Δ%     TS Δ%
price_hunter           0.00        0.00        0.00       +0.00%    +0.00%
pl_lover               5.34        4.18        7.96      -21.78%   +49.03%
premium               31.60       17.43       19.77      -44.83%   -37.42%
litter_heavy           4.94        5.74        4.41      +16.13%   -10.80%

Template selection frequencies — LinUCB:
   0 Neutral        :   5.08%
   1 High Margin    :   4.46%
   2 CM2 Boost      :  23.93%
   3 Popular        :   4.44%
   4 Premium        :   4.41%
   5 Budget         :   5.32%
   6 Discount       :  47.95%
   7 Strategic      :   4.43%

Template selection frequencies — ThompsonSampling:
   0 Neutral        :  22.10%
   1 High Margin    :  34.23%
   2 CM2 Boost      :  21.43%
   3 Popular        :   4.15%
   4 Premium        :   5.90%
   5 Budget         :   3.88%
   6 Discount       :   4.25%
   7 Strategic      :   4.08%

Notes:
- This demonstration can use either a simple context feature map (segment + query type) or a richer feature set with catalog aggregates.
- In many runs, bandits with simple features underperform strong static templates; richer features typically reduce this gap, revealing when linear contextual bandits are appropriate.

For theory and detailed diagnostics, see:
  - docs/book/drafts/ch06/discrete_template_bandits.md
  - docs/book/drafts/ch06/exercises_labs.md
================================================================================

✓ Test 1 completed

==============================================================================
TEST 2: Rich Features + Blend Regularization (§6.7)
==============================================================================\033[0m

Expected: Thompson Sampling achieves significant GMV lift over static


================================================================================
CHAPTER 6 — TEMPLATE BANDITS DEMONSTRATION
================================================================================

Experiment horizons: static=2,000 episodes, bandits=20,000 episodes.
Feature mode: 'rich'

Generating catalog with 10,000 products (world seed=20250322)...

Resolved bandit hyperparameters → prior_weight=50, lin_alpha=0.20, ts_sigma=0.50
Rich feature regularization: mode=blend, blend_weight=0.40, shrink=0.90, quant_step=0.25, clip=[-3.50, 3.50]

Running static template baselines for 2000 episodes...
  Static templates progress:   0.1% (1/2000) | best template ID 4 avg reward 20.70
  Static templates progress:   2.0% (40/2000) | best template ID 4 avg reward 8.55
  Static templates progress:   4.0% (80/2000) | best template ID 4 avg reward 10.36
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 11.91
  Static templates progress:   8.0% (160/2000) | best template ID 4 avg reward 9.97
  Static templates progress:  10.0% (200/2000) | best template ID 4 avg reward 9.41
  Static templates progress:  12.0% (240/2000) | best template ID 4 avg reward 9.80
  Static templates progress:  14.0% (280/2000) | best template ID 4 avg reward 9.60
  Static templates progress:  16.0% (320/2000) | best template ID 4 avg reward 9.90
  Static templates progress:  18.0% (360/2000) | best template ID 4 avg reward 9.64
  Static templates progress:  20.0% (400/2000) | best template ID 4 avg reward 9.57
  Static templates progress:  22.0% (440/2000) | best template ID 4 avg reward 9.49
  Static templates progress:  24.0% (480/2000) | best template ID 4 avg reward 9.21
  Static templates progress:  26.0% (520/2000) | best template ID 4 avg reward 8.67
  Static templates progress:  28.0% (560/2000) | best template ID 4 avg reward 8.82
  Static templates progress:  30.0% (600/2000) | best template ID 4 avg reward 8.78
  Static templates progress:  32.0% (640/2000) | best template ID 4 avg reward 8.88
  Static templates progress:  34.0% (680/2000) | best template ID 4 avg reward 8.75
  Static templates progress:  36.0% (720/2000) | best template ID 4 avg reward 8.77
  Static templates progress:  38.0% (760/2000) | best template ID 4 avg reward 8.52
  Static templates progress:  40.0% (800/2000) | best template ID 4 avg reward 8.37
  Static templates progress:  42.0% (840/2000) | best template ID 4 avg reward 8.39
  Static templates progress:  44.0% (880/2000) | best template ID 4 avg reward 8.30
  Static templates progress:  46.0% (920/2000) | best template ID 4 avg reward 8.22
  Static templates progress:  48.0% (960/2000) | best template ID 4 avg reward 8.38
  Static templates progress:  50.0% (1000/2000) | best template ID 4 avg reward 8.14
  Static templates progress:  52.0% (1040/2000) | best template ID 4 avg reward 8.01
  Static templates progress:  54.0% (1080/2000) | best template ID 2 avg reward 7.92
  Static templates progress:  56.0% (1120/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  58.0% (1160/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  60.0% (1200/2000) | best template ID 2 avg reward 8.09
  Static templates progress:  62.0% (1240/2000) | best template ID 2 avg reward 8.06
  Static templates progress:  64.0% (1280/2000) | best template ID 2 avg reward 8.08
  Static templates progress:  66.0% (1320/2000) | best template ID 2 avg reward 8.00
  Static templates progress:  68.0% (1360/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.96
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.90
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 7.87
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 7.72
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.67
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.54
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.55
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.53
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.48
  Static templates progress:  98.0% (1960/2000) | best template ID 4 avg reward 7.46
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.56


Best static template: ID=4 (Premium) with avg reward=7.56, GMV=7.11, CM2=0.74

Running LinUCB with features=rich for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  LINUCB progress:   2.0% (400/20000) | avg reward 10.51
  LINUCB progress:   4.0% (800/20000) | avg reward 10.49
  LINUCB progress:   6.0% (1200/20000) | avg reward 10.84
  LINUCB progress:   8.0% (1600/20000) | avg reward 10.77
  LINUCB progress:  10.0% (2000/20000) | avg reward 10.38
  LINUCB progress:  12.0% (2400/20000) | avg reward 10.40
  LINUCB progress:  14.0% (2800/20000) | avg reward 10.44
  LINUCB progress:  16.0% (3200/20000) | avg reward 10.47
  LINUCB progress:  18.0% (3600/20000) | avg reward 10.20
  LINUCB progress:  20.0% (4000/20000) | avg reward 10.25
  LINUCB progress:  22.0% (4400/20000) | avg reward 10.17
  LINUCB progress:  24.0% (4800/20000) | avg reward 10.20
  LINUCB progress:  26.0% (5200/20000) | avg reward 10.19
  LINUCB progress:  28.0% (5600/20000) | avg reward 10.08
  LINUCB progress:  30.0% (6000/20000) | avg reward 10.12
  LINUCB progress:  32.0% (6400/20000) | avg reward 10.13
  LINUCB progress:  34.0% (6800/20000) | avg reward 10.09
  LINUCB progress:  36.0% (7200/20000) | avg reward 10.05
  LINUCB progress:  38.0% (7600/20000) | avg reward 10.02
  LINUCB progress:  40.0% (8000/20000) | avg reward 10.01
  LINUCB progress:  42.0% (8400/20000) | avg reward 10.05
  LINUCB progress:  44.0% (8800/20000) | avg reward 10.07
  LINUCB progress:  46.0% (9200/20000) | avg reward 10.03
  LINUCB progress:  48.0% (9600/20000) | avg reward 9.97
  LINUCB progress:  50.0% (10000/20000) | avg reward 10.01
  LINUCB progress:  52.0% (10400/20000) | avg reward 10.00
  LINUCB progress:  54.0% (10800/20000) | avg reward 10.01
  LINUCB progress:  56.0% (11200/20000) | avg reward 10.01
  LINUCB progress:  58.0% (11600/20000) | avg reward 9.98
  LINUCB progress:  60.0% (12000/20000) | avg reward 9.97
  LINUCB progress:  62.0% (12400/20000) | avg reward 9.94
  LINUCB progress:  64.0% (12800/20000) | avg reward 9.94
  LINUCB progress:  66.0% (13200/20000) | avg reward 9.91
  LINUCB progress:  68.0% (13600/20000) | avg reward 9.97
  LINUCB progress:  70.0% (14000/20000) | avg reward 10.01
  LINUCB progress:  72.0% (14400/20000) | avg reward 10.06
  LINUCB progress:  74.0% (14800/20000) | avg reward 10.03
  LINUCB progress:  76.0% (15200/20000) | avg reward 10.00
  LINUCB progress:  78.0% (15600/20000) | avg reward 10.00
  LINUCB progress:  80.0% (16000/20000) | avg reward 9.99
  LINUCB progress:  82.0% (16400/20000) | avg reward 10.01
  LINUCB progress:  84.0% (16800/20000) | avg reward 10.03
  LINUCB progress:  86.0% (17200/20000) | avg reward 10.06
  LINUCB progress:  88.0% (17600/20000) | avg reward 10.08
  LINUCB progress:  90.0% (18000/20000) | avg reward 10.13
  LINUCB progress:  92.0% (18400/20000) | avg reward 10.12
  LINUCB progress:  94.0% (18800/20000) | avg reward 10.11
  LINUCB progress:  96.0% (19200/20000) | avg reward 10.11
  LINUCB progress:  98.0% (19600/20000) | avg reward 10.13
  LINUCB progress: 100.0% (20000/20000) | avg reward 10.11

Running Thompson Sampling with features=rich for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 0.00
  TS progress:   2.0% (400/20000) | avg reward 6.91
  TS progress:   4.0% (800/20000) | avg reward 6.70
  TS progress:   6.0% (1200/20000) | avg reward 7.33
  TS progress:   8.0% (1600/20000) | avg reward 7.66
  TS progress:  10.0% (2000/20000) | avg reward 7.64
  TS progress:  12.0% (2400/20000) | avg reward 7.79
  TS progress:  14.0% (2800/20000) | avg reward 7.87
  TS progress:  16.0% (3200/20000) | avg reward 7.82
  TS progress:  18.0% (3600/20000) | avg reward 7.76
  TS progress:  20.0% (4000/20000) | avg reward 7.88
  TS progress:  22.0% (4400/20000) | avg reward 7.95
  TS progress:  24.0% (4800/20000) | avg reward 8.09
  TS progress:  26.0% (5200/20000) | avg reward 8.05
  TS progress:  28.0% (5600/20000) | avg reward 8.04
  TS progress:  30.0% (6000/20000) | avg reward 8.19
  TS progress:  32.0% (6400/20000) | avg reward 8.19
  TS progress:  34.0% (6800/20000) | avg reward 8.15
  TS progress:  36.0% (7200/20000) | avg reward 8.17
  TS progress:  38.0% (7600/20000) | avg reward 8.19
  TS progress:  40.0% (8000/20000) | avg reward 8.18
  TS progress:  42.0% (8400/20000) | avg reward 8.16
  TS progress:  44.0% (8800/20000) | avg reward 8.18
  TS progress:  46.0% (9200/20000) | avg reward 8.18
  TS progress:  48.0% (9600/20000) | avg reward 8.18
  TS progress:  50.0% (10000/20000) | avg reward 8.20
  TS progress:  52.0% (10400/20000) | avg reward 8.16
  TS progress:  54.0% (10800/20000) | avg reward 8.08
  TS progress:  56.0% (11200/20000) | avg reward 8.08
  TS progress:  58.0% (11600/20000) | avg reward 8.08
  TS progress:  60.0% (12000/20000) | avg reward 8.09
  TS progress:  62.0% (12400/20000) | avg reward 8.11
  TS progress:  64.0% (12800/20000) | avg reward 8.13
  TS progress:  66.0% (13200/20000) | avg reward 8.13
  TS progress:  68.0% (13600/20000) | avg reward 8.14
  TS progress:  70.0% (14000/20000) | avg reward 8.14
  TS progress:  72.0% (14400/20000) | avg reward 8.15
  TS progress:  74.0% (14800/20000) | avg reward 8.12
  TS progress:  76.0% (15200/20000) | avg reward 8.07
  TS progress:  78.0% (15600/20000) | avg reward 8.07
  TS progress:  80.0% (16000/20000) | avg reward 8.07
  TS progress:  82.0% (16400/20000) | avg reward 8.09
  TS progress:  84.0% (16800/20000) | avg reward 8.09
  TS progress:  86.0% (17200/20000) | avg reward 8.12
  TS progress:  88.0% (17600/20000) | avg reward 8.09
  TS progress:  90.0% (18000/20000) | avg reward 8.11
  TS progress:  92.0% (18400/20000) | avg reward 8.13
  TS progress:  94.0% (18800/20000) | avg reward 8.14
  TS progress:  96.0% (19200/20000) | avg reward 8.16
  TS progress:  98.0% (19600/20000) | avg reward 8.17
  TS progress: 100.0% (20000/20000) | avg reward 8.15

  Static (best) segment stats:   0.0% (1/20000) | avg reward 0.00
  Static (best) segment stats:   2.0% (400/20000) | avg reward 7.84
  Static (best) segment stats:   4.0% (800/20000) | avg reward 7.25
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.07
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.12
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.20
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.11
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.15
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.21
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.36
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.41
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.39
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.40
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.50
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.51
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.54
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.60
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.60
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.61
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.63
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.66
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.72
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.75
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.70
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.70
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.71
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.68
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.67
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.68
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.71
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.72
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.75
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.73
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.73
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.71
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.72
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.70
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.70
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.73
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.71
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.68
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.64
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.64
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.68
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.68
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.67
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.64
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.66
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.64
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.65
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.65


Static templates (per-episode averages):
ID  Template             Reward         GMV         CM2
 0  Neutral                5.75        5.28        0.50
 1  High Margin            5.44        5.07        0.63
 2  CM2 Boost              7.35        6.73        0.61
 3  Popular                4.79        4.53        0.52
 4  Premium                7.56        7.11        0.74
 5  Budget                 3.44        3.04        0.26
 6  Discount               5.04        4.62        0.45
 7  Strategic              4.83        3.99       -0.13

Summary (per-episode averages):
Policy                  Reward         GMV         CM2  ΔGMV vs static
Static-Premium            7.56        7.11        0.74          +0.00%
LinUCB                   10.11        9.34        0.93         +31.31%
ThompsonSampling          8.15        7.47        0.61          +5.10%

Per-segment GMV (static best vs bandits):
Segment          Static GMV  LinUCB GMV      TS GMV    LinUCB Δ%     TS Δ%
price_hunter           0.00        0.00        0.00       +0.00%    +0.00%
pl_lover               5.34       11.48        5.23     +114.93%    -2.06%
premium               31.60       32.17       32.29       +1.81%    +2.18%
litter_heavy           4.94        6.47        5.33      +31.01%    +7.90%

Template selection frequencies — LinUCB:
   0 Neutral        :   2.83%
   1 High Margin    :   3.50%
   2 CM2 Boost      :  50.25%
   3 Popular        :   3.79%
   4 Premium        :  17.51%
   5 Budget         :  18.95%
   6 Discount       :   0.80%
   7 Strategic      :   2.37%

Template selection frequencies — ThompsonSampling:
   0 Neutral        :   9.30%
   1 High Margin    :   0.26%
   2 CM2 Boost      :  18.84%
   3 Popular        :   1.33%
   4 Premium        :  36.68%
   5 Budget         :   1.49%
   6 Discount       :   2.47%
   7 Strategic      :  29.64%

Notes:
- This demonstration can use either a simple context feature map (segment + query type) or a richer feature set with catalog aggregates.
- In many runs, bandits with simple features underperform strong static templates; richer features typically reduce this gap, revealing when linear contextual bandits are appropriate.

For theory and detailed diagnostics, see:
  - docs/book/drafts/ch06/discrete_template_bandits.md
  - docs/book/drafts/ch06/exercises_labs.md
================================================================================

./scripts/ch06/run_full_verification.sh: line 109: syntax error near unexpected token `('
(.venv) [/Users/vladyslavprytula/src/rl_search_from_scratch </>] [  ch00-ch01-revision ✚12 …13 ↯1 ]
❯ cladue
zsh: command not found: cladue
(.venv) [/Users/vladyslavprytula/src/rl_search_from_scratch </>] [  ch00-ch01-revision ✚12 …13 ↯1 ]
❯ claude

▗ ▗   ▖ ▖  Claude Code v2.0.57
           Opus 4.5 · API Usage Billing
  ▘▘ ▝▝    /Volumes/Lexar2T/src/rl_search_from_scratch

  /model to try Opus 4.5. Note: you may need to request access from your cloud provider

> can you please fix 
./scripts/ch06/run_full_verification.sh: line 109: syntax error near unexpected token `(' 

⏺ Read(scripts/ch06/run_full_verification.sh)
  ⎿  Read 248 lines

⏺ I don't see any syntax error on line 109 in this file. The file looks correct. Let me check if there might be hidden characters or
  if the error is from a different version:

⏺ Bash(file /Volumes/Lexar2T/src/rl_search_from_scratch/scripts/ch06/run_full_verification.sh && head -c 200
      /Volumes/Lexar2T/src/rl_search_from_scratch/scripts/ch06/run_full_verification.sh | xxd | head -20)
  ⎿  /Volumes/Lexar2T/src/rl_search_from_scratch/scripts/ch06/run_full_verification.sh: Bourne-Again shell script text executable, 
     Unicode text, UTF-8 text
     00000000: 2321 2f62 696e 2f62 6173 680a 2320 3d3d  #!/bin/bash.# ==
     00000010: 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d  ================
     00000020: 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d  ================
     00000030: 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d  ================
     00000040: 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d  ================
     00000050: 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 0a23 2043  ============.# C
     00000060: 6861 7074 6572 2036 2046 756c 6c20 5665  hapter 6 Full Ve
     00000070: 7269 6669 6361 7469 6f6e 2053 7569 7465  rification Suite
     00000080: 0a23 203d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d  .# =============
     00000090: 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d  ================
     000000a0: 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d  ================
     000000b0: 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d 3d3d  ================
     000000c0: 3d3d 3d3d 3d3d 3d3d                      ========

⏺ Bash(sed -n '105,115p' /Volumes/Lexar2T/src/rl_search_from_scratch/scripts/ch06/run_full_verification.sh | cat -A)
  ⎿  Error: Exit code 1
     cat: illegal option -- A
     usage: cat [-belnstuv] [file ...]

⏺ Bash(sed -n '105,115p' /Volumes/Lexar2T/src/rl_search_from_scratch/scripts/ch06/run_full_verification.sh | od -c)
  ⎿  0000000    #       T   E   S   T       3   :       R   i   c   h       F
     0000020    e   a   t   u   r   e   s       w   i   t   h       Q   u   a
     0000040    n   t   i   z   e   d       R   e   g   u   l   a   r   i   z
     0000060    a   t   i   o   n       (   W   i   n   n   i   n   g       C
     0000100    o   n   f   i   g   )  \n   #       =   =   =   =   =   =   =
     0000120    =   =   =   =   =   =   =   =   =   =   =   =   =   =   =   =
     *
     0000220    =   =   =   =   =   =   =  \n   e   c   h   o       -   e
     0000240    "   $   {   Y   E   L   L   O   W   }   =   =   =   =   =   =
     0000260    =   =   =   =   =   =   =   =   =   =   =   =   =   =   =   =
     *
     0000360    =   =   =   =   =   =   =   =   "  \n   e   c   h   o       "
     0000400    T   E   S   T       3   :       R   i   c   h       F   e   a
     0000420    t   u   r   e   s       +       Q   u   a   n   t   i   z   e
     0000440    d       R   e   g   u   l   a   r   i   z   a   t   i   o   n
     0000460        (   W   i   n   n   i   n   g       C   o   n   f   i   g
     0000500    )   "  \n   e   c   h   o       "   =   =   =   =   =   =   =
     0000520    =   =   =   =   =   =   =   =   =   =   =   =   =   =   =   =
     *
     0000620    =   =   =   =   =   =   =   $   {   N   C   }   "  \n   e   c
     0000640    h   o       "   "  \n   e   c   h   o       "   E   x   p   e
     0000660    c   t   e   d   :       L   i   n   U   C   B       a   c   h
     0000700    i   e   v   e   s       ~   5   -   8   %       G   M   V
     0000720    l   i   f   t   ,       T   S       a   c   h   i   e   v   e
     0000740    s       ~   3   -   5   %       l   i   f   t   "  \n   e   c
     0000760    h   o       "   "  \n  \n   p   y   t   h   o   n       s   c
     0001000    r   i   p   t   s   /   c   h   0   6   /   t   e   m   p   l
     0001020    a   t   e   _   b   a   n   d   i   t   s   _   d   e   m   o
     0001040    .   p   y       \  \n                   -   -   f   e   a   t
     0001060    u   r   e   s       r   i   c   h       \  \n
     0001074

⏺ The file looks fine. Let me check if bash might be interpreting something differently. Let me run a syntax check:

⏺ Bash(bash -n /Volumes/Lexar2T/src/rl_search_from_scratch/scripts/ch06/run_full_verification.sh 2>&1)
  ⎿  (No content)

⏺ The syntax check passed without errors. Let me check if you might have a different version of the file or if there's an issue with
  how bash is being invoked:

⏺ Bash(sh -n scripts/ch06/run_full_verification.sh 2>&1)
  ⎿  (No content)

⏺ Bash(head -1 /Volumes/Lexar2T/src/rl_search_from_scratch/scripts/ch06/run_full_verification.sh && which bash && bash --version | head
      -1)
  ⎿  #!/bin/bash                                               
     /bin/bash
     GNU bash, version 3.2.57(1)-release (arm64-apple-darwin25)

⏺ The script passes syntax checks with both bash -n and sh -n. The error you're seeing might be:

  1. A cached/different version - Try running hash -r to clear bash's command cache
  2. Being run with a different shell - The macOS default shell is zsh, not bash

  Let me check if running with ./ vs bash makes a difference:

⏺ Bash(zsh -n scripts/ch06/run_full_verification.sh 2>&1)
  ⎿  Interrupted · What should Claude do instead?

(.venv) [/Users/vladyslavprytula/src/rl_search_from_scratch </>] [  ch00-ch01-revision ✚12 …13 ↯1 ]
❯ ./scripts/ch06/run_full_verification.sh 2>&1 | tee ch06_verification.log                                          
==============================================================================
CHAPTER 6 FULL VERIFICATION SUITE
==============================================================================

Configuration:
  Static episodes:  2000
  Bandit episodes:  20000
  World seed:       20250322
  Bandit seed:      20250349
  Output directory: docs/book/drafts/ch06/data/verification_20251203T125142

Start time: Wed Dec  3 12:51:42 CET 2025

==============================================================================
TEST 1: Simple Features - The Failure Mode (§6.5)
==============================================================================\033[0m

Expected: Bandits underperform or match static baseline with simple features


================================================================================
CHAPTER 6 — TEMPLATE BANDITS DEMONSTRATION
================================================================================

Experiment horizons: static=2,000 episodes, bandits=20,000 episodes.
Feature mode: 'simple'

Generating catalog with 10,000 products (world seed=20250322)...

Resolved bandit hyperparameters → prior_weight=0, lin_alpha=1.00, ts_sigma=1.00

Running static template baselines for 2000 episodes...
  Static templates progress:   0.1% (1/2000) | best template ID 4 avg reward 20.70
  Static templates progress:   2.0% (40/2000) | best template ID 4 avg reward 8.55
  Static templates progress:   4.0% (80/2000) | best template ID 4 avg reward 10.36
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 11.91
  Static templates progress:   8.0% (160/2000) | best template ID 4 avg reward 9.97
  Static templates progress:  10.0% (200/2000) | best template ID 4 avg reward 9.41
  Static templates progress:  12.0% (240/2000) | best template ID 4 avg reward 9.80
  Static templates progress:  14.0% (280/2000) | best template ID 4 avg reward 9.60
  Static templates progress:  16.0% (320/2000) | best template ID 4 avg reward 9.90
  Static templates progress:  18.0% (360/2000) | best template ID 4 avg reward 9.64
  Static templates progress:  20.0% (400/2000) | best template ID 4 avg reward 9.57
  Static templates progress:  22.0% (440/2000) | best template ID 4 avg reward 9.49
  Static templates progress:  24.0% (480/2000) | best template ID 4 avg reward 9.21
  Static templates progress:  26.0% (520/2000) | best template ID 4 avg reward 8.67
  Static templates progress:  28.0% (560/2000) | best template ID 4 avg reward 8.82
  Static templates progress:  30.0% (600/2000) | best template ID 4 avg reward 8.78
  Static templates progress:  32.0% (640/2000) | best template ID 4 avg reward 8.88
  Static templates progress:  34.0% (680/2000) | best template ID 4 avg reward 8.75
  Static templates progress:  36.0% (720/2000) | best template ID 4 avg reward 8.77
  Static templates progress:  38.0% (760/2000) | best template ID 4 avg reward 8.52
  Static templates progress:  40.0% (800/2000) | best template ID 4 avg reward 8.37
  Static templates progress:  42.0% (840/2000) | best template ID 4 avg reward 8.39
  Static templates progress:  44.0% (880/2000) | best template ID 4 avg reward 8.30
  Static templates progress:  46.0% (920/2000) | best template ID 4 avg reward 8.22
  Static templates progress:  48.0% (960/2000) | best template ID 4 avg reward 8.38
  Static templates progress:  50.0% (1000/2000) | best template ID 4 avg reward 8.14
  Static templates progress:  52.0% (1040/2000) | best template ID 4 avg reward 8.01
  Static templates progress:  54.0% (1080/2000) | best template ID 2 avg reward 7.92
  Static templates progress:  56.0% (1120/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  58.0% (1160/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  60.0% (1200/2000) | best template ID 2 avg reward 8.09
  Static templates progress:  62.0% (1240/2000) | best template ID 2 avg reward 8.06
  Static templates progress:  64.0% (1280/2000) | best template ID 2 avg reward 8.08
  Static templates progress:  66.0% (1320/2000) | best template ID 2 avg reward 8.00
  Static templates progress:  68.0% (1360/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.96
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.90
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 7.87
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 7.72
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.67
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.54
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.55
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.53
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.48
  Static templates progress:  98.0% (1960/2000) | best template ID 4 avg reward 7.46
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.56


Best static template: ID=4 (Premium) with avg reward=7.56, GMV=7.11, CM2=0.74

Running LinUCB with features=simple for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  LINUCB progress:   2.0% (400/20000) | avg reward 4.76
  LINUCB progress:   4.0% (800/20000) | avg reward 4.96
  LINUCB progress:   6.0% (1200/20000) | avg reward 5.07
  LINUCB progress:   8.0% (1600/20000) | avg reward 5.21
  LINUCB progress:  10.0% (2000/20000) | avg reward 5.33
  LINUCB progress:  12.0% (2400/20000) | avg reward 5.32
  LINUCB progress:  14.0% (2800/20000) | avg reward 5.38
  LINUCB progress:  16.0% (3200/20000) | avg reward 5.40
  LINUCB progress:  18.0% (3600/20000) | avg reward 5.36
  LINUCB progress:  20.0% (4000/20000) | avg reward 5.41
  LINUCB progress:  22.0% (4400/20000) | avg reward 5.49
  LINUCB progress:  24.0% (4800/20000) | avg reward 5.50
  LINUCB progress:  26.0% (5200/20000) | avg reward 5.54
  LINUCB progress:  28.0% (5600/20000) | avg reward 5.67
  LINUCB progress:  30.0% (6000/20000) | avg reward 5.64
  LINUCB progress:  32.0% (6400/20000) | avg reward 5.72
  LINUCB progress:  34.0% (6800/20000) | avg reward 5.77
  LINUCB progress:  36.0% (7200/20000) | avg reward 5.80
  LINUCB progress:  38.0% (7600/20000) | avg reward 5.81
  LINUCB progress:  40.0% (8000/20000) | avg reward 5.79
  LINUCB progress:  42.0% (8400/20000) | avg reward 5.75
  LINUCB progress:  44.0% (8800/20000) | avg reward 5.74
  LINUCB progress:  46.0% (9200/20000) | avg reward 5.75
  LINUCB progress:  48.0% (9600/20000) | avg reward 5.73
  LINUCB progress:  50.0% (10000/20000) | avg reward 5.74
  LINUCB progress:  52.0% (10400/20000) | avg reward 5.75
  LINUCB progress:  54.0% (10800/20000) | avg reward 5.75
  LINUCB progress:  56.0% (11200/20000) | avg reward 5.75
  LINUCB progress:  58.0% (11600/20000) | avg reward 5.74
  LINUCB progress:  60.0% (12000/20000) | avg reward 5.73
  LINUCB progress:  62.0% (12400/20000) | avg reward 5.71
  LINUCB progress:  64.0% (12800/20000) | avg reward 5.72
  LINUCB progress:  66.0% (13200/20000) | avg reward 5.74
  LINUCB progress:  68.0% (13600/20000) | avg reward 5.75
  LINUCB progress:  70.0% (14000/20000) | avg reward 5.74
  LINUCB progress:  72.0% (14400/20000) | avg reward 5.71
  LINUCB progress:  74.0% (14800/20000) | avg reward 5.70
  LINUCB progress:  76.0% (15200/20000) | avg reward 5.71
  LINUCB progress:  78.0% (15600/20000) | avg reward 5.71
  LINUCB progress:  80.0% (16000/20000) | avg reward 5.67
  LINUCB progress:  82.0% (16400/20000) | avg reward 5.67
  LINUCB progress:  84.0% (16800/20000) | avg reward 5.66
  LINUCB progress:  86.0% (17200/20000) | avg reward 5.67
  LINUCB progress:  88.0% (17600/20000) | avg reward 5.67
  LINUCB progress:  90.0% (18000/20000) | avg reward 5.68
  LINUCB progress:  92.0% (18400/20000) | avg reward 5.66
  LINUCB progress:  94.0% (18800/20000) | avg reward 5.65
  LINUCB progress:  96.0% (19200/20000) | avg reward 5.64
  LINUCB progress:  98.0% (19600/20000) | avg reward 5.63
  LINUCB progress: 100.0% (20000/20000) | avg reward 5.62

Running Thompson Sampling with features=simple for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 0.00
  TS progress:   2.0% (400/20000) | avg reward 6.01
  TS progress:   4.0% (800/20000) | avg reward 5.68
  TS progress:   6.0% (1200/20000) | avg reward 5.54
  TS progress:   8.0% (1600/20000) | avg reward 5.83
  TS progress:  10.0% (2000/20000) | avg reward 5.78
  TS progress:  12.0% (2400/20000) | avg reward 5.67
  TS progress:  14.0% (2800/20000) | avg reward 5.56
  TS progress:  16.0% (3200/20000) | avg reward 5.46
  TS progress:  18.0% (3600/20000) | avg reward 5.48
  TS progress:  20.0% (4000/20000) | avg reward 5.51
  TS progress:  22.0% (4400/20000) | avg reward 5.45
  TS progress:  24.0% (4800/20000) | avg reward 5.57
  TS progress:  26.0% (5200/20000) | avg reward 5.65
  TS progress:  28.0% (5600/20000) | avg reward 5.68
  TS progress:  30.0% (6000/20000) | avg reward 5.74
  TS progress:  32.0% (6400/20000) | avg reward 5.82
  TS progress:  34.0% (6800/20000) | avg reward 5.88
  TS progress:  36.0% (7200/20000) | avg reward 5.85
  TS progress:  38.0% (7600/20000) | avg reward 5.88
  TS progress:  40.0% (8000/20000) | avg reward 5.89
  TS progress:  42.0% (8400/20000) | avg reward 5.89
  TS progress:  44.0% (8800/20000) | avg reward 5.92
  TS progress:  46.0% (9200/20000) | avg reward 5.91
  TS progress:  48.0% (9600/20000) | avg reward 5.90
  TS progress:  50.0% (10000/20000) | avg reward 5.93
  TS progress:  52.0% (10400/20000) | avg reward 5.93
  TS progress:  54.0% (10800/20000) | avg reward 5.93
  TS progress:  56.0% (11200/20000) | avg reward 5.92
  TS progress:  58.0% (11600/20000) | avg reward 5.92
  TS progress:  60.0% (12000/20000) | avg reward 5.97
  TS progress:  62.0% (12400/20000) | avg reward 6.03
  TS progress:  64.0% (12800/20000) | avg reward 6.10
  TS progress:  66.0% (13200/20000) | avg reward 6.14
  TS progress:  68.0% (13600/20000) | avg reward 6.18
  TS progress:  70.0% (14000/20000) | avg reward 6.26
  TS progress:  72.0% (14400/20000) | avg reward 6.27
  TS progress:  74.0% (14800/20000) | avg reward 6.34
  TS progress:  76.0% (15200/20000) | avg reward 6.39
  TS progress:  78.0% (15600/20000) | avg reward 6.44
  TS progress:  80.0% (16000/20000) | avg reward 6.48
  TS progress:  82.0% (16400/20000) | avg reward 6.48
  TS progress:  84.0% (16800/20000) | avg reward 6.51
  TS progress:  86.0% (17200/20000) | avg reward 6.53
  TS progress:  88.0% (17600/20000) | avg reward 6.54
  TS progress:  90.0% (18000/20000) | avg reward 6.58
  TS progress:  92.0% (18400/20000) | avg reward 6.60
  TS progress:  94.0% (18800/20000) | avg reward 6.61
  TS progress:  96.0% (19200/20000) | avg reward 6.64
  TS progress:  98.0% (19600/20000) | avg reward 6.67
  TS progress: 100.0% (20000/20000) | avg reward 6.69

  Static (best) segment stats:   0.0% (1/20000) | avg reward 0.00
  Static (best) segment stats:   2.0% (400/20000) | avg reward 7.84
  Static (best) segment stats:   4.0% (800/20000) | avg reward 7.25
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.07
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.12
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.20
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.11
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.15
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.21
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.36
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.41
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.39
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.40
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.50
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.51
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.54
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.60
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.60
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.61
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.63
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.66
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.72
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.75
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.70
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.70
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.71
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.68
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.67
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.68
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.71
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.72
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.75
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.73
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.73
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.71
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.72
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.70
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.70
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.73
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.71
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.68
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.64
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.64
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.68
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.68
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.67
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.64
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.66
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.64
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.65
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.65


Static templates (per-episode averages):
ID  Template             Reward         GMV         CM2
 0  Neutral                5.75        5.28        0.50
 1  High Margin            5.44        5.07        0.63
 2  CM2 Boost              7.35        6.73        0.61
 3  Popular                4.79        4.53        0.52
 4  Premium                7.56        7.11        0.74
 5  Budget                 3.44        3.04        0.26
 6  Discount               5.04        4.62        0.45
 7  Strategic              4.83        3.99       -0.13

Summary (per-episode averages):
Policy                  Reward         GMV         CM2  ΔGMV vs static
Static-Premium            7.56        7.11        0.74          +0.00%
LinUCB                    5.62        5.12        0.46         -27.95%
ThompsonSampling          6.69        6.18        0.63         -13.07%

Per-segment GMV (static best vs bandits):
Segment          Static GMV  LinUCB GMV      TS GMV    LinUCB Δ%     TS Δ%
price_hunter           0.00        0.00        0.00       +0.00%    +0.00%
pl_lover               5.34        4.18        7.96      -21.78%   +49.03%
premium               31.60       17.43       19.77      -44.83%   -37.42%
litter_heavy           4.94        5.74        4.41      +16.13%   -10.80%

Template selection frequencies — LinUCB:
   0 Neutral        :   5.08%
   1 High Margin    :   4.46%
   2 CM2 Boost      :  23.93%
   3 Popular        :   4.44%
   4 Premium        :   4.41%
   5 Budget         :   5.32%
   6 Discount       :  47.95%
   7 Strategic      :   4.43%

Template selection frequencies — ThompsonSampling:
   0 Neutral        :  22.10%
   1 High Margin    :  34.23%
   2 CM2 Boost      :  21.43%
   3 Popular        :   4.15%
   4 Premium        :   5.90%
   5 Budget         :   3.88%
   6 Discount       :   4.25%
   7 Strategic      :   4.08%

Notes:
- This demonstration can use either a simple context feature map (segment + query type) or a richer feature set with catalog aggregates.
- In many runs, bandits with simple features underperform strong static templates; richer features typically reduce this gap, revealing when linear contextual bandits are appropriate.

For theory and detailed diagnostics, see:
  - docs/book/drafts/ch06/discrete_template_bandits.md
  - docs/book/drafts/ch06/exercises_labs.md
================================================================================

✓ Test 1 completed

==============================================================================
TEST 2: Rich Features + Blend Regularization (§6.7)
==============================================================================\033[0m

Expected: Thompson Sampling achieves significant GMV lift over static


================================================================================
CHAPTER 6 — TEMPLATE BANDITS DEMONSTRATION
================================================================================

Experiment horizons: static=2,000 episodes, bandits=20,000 episodes.
Feature mode: 'rich'

Generating catalog with 10,000 products (world seed=20250322)...

Resolved bandit hyperparameters → prior_weight=50, lin_alpha=0.20, ts_sigma=0.50
Rich feature regularization: mode=blend, blend_weight=0.40, shrink=0.90, quant_step=0.25, clip=[-3.50, 3.50]

Running static template baselines for 2000 episodes...
  Static templates progress:   0.1% (1/2000) | best template ID 4 avg reward 20.70
  Static templates progress:   2.0% (40/2000) | best template ID 4 avg reward 8.55
  Static templates progress:   4.0% (80/2000) | best template ID 4 avg reward 10.36
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 11.91
  Static templates progress:   8.0% (160/2000) | best template ID 4 avg reward 9.97
  Static templates progress:  10.0% (200/2000) | best template ID 4 avg reward 9.41
  Static templates progress:  12.0% (240/2000) | best template ID 4 avg reward 9.80
  Static templates progress:  14.0% (280/2000) | best template ID 4 avg reward 9.60
  Static templates progress:  16.0% (320/2000) | best template ID 4 avg reward 9.90
  Static templates progress:  18.0% (360/2000) | best template ID 4 avg reward 9.64
  Static templates progress:  20.0% (400/2000) | best template ID 4 avg reward 9.57
  Static templates progress:  22.0% (440/2000) | best template ID 4 avg reward 9.49
  Static templates progress:  24.0% (480/2000) | best template ID 4 avg reward 9.21
  Static templates progress:  26.0% (520/2000) | best template ID 4 avg reward 8.67
  Static templates progress:  28.0% (560/2000) | best template ID 4 avg reward 8.82
  Static templates progress:  30.0% (600/2000) | best template ID 4 avg reward 8.78
  Static templates progress:  32.0% (640/2000) | best template ID 4 avg reward 8.88
  Static templates progress:  34.0% (680/2000) | best template ID 4 avg reward 8.75
  Static templates progress:  36.0% (720/2000) | best template ID 4 avg reward 8.77
  Static templates progress:  38.0% (760/2000) | best template ID 4 avg reward 8.52
  Static templates progress:  40.0% (800/2000) | best template ID 4 avg reward 8.37
  Static templates progress:  42.0% (840/2000) | best template ID 4 avg reward 8.39
  Static templates progress:  44.0% (880/2000) | best template ID 4 avg reward 8.30
  Static templates progress:  46.0% (920/2000) | best template ID 4 avg reward 8.22
  Static templates progress:  48.0% (960/2000) | best template ID 4 avg reward 8.38
  Static templates progress:  50.0% (1000/2000) | best template ID 4 avg reward 8.14
  Static templates progress:  52.0% (1040/2000) | best template ID 4 avg reward 8.01
  Static templates progress:  54.0% (1080/2000) | best template ID 2 avg reward 7.92
  Static templates progress:  56.0% (1120/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  58.0% (1160/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  60.0% (1200/2000) | best template ID 2 avg reward 8.09
  Static templates progress:  62.0% (1240/2000) | best template ID 2 avg reward 8.06
  Static templates progress:  64.0% (1280/2000) | best template ID 2 avg reward 8.08
  Static templates progress:  66.0% (1320/2000) | best template ID 2 avg reward 8.00
  Static templates progress:  68.0% (1360/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.96
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.90
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 7.87
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 7.72
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.67
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.54
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.55
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.53
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.48
  Static templates progress:  98.0% (1960/2000) | best template ID 4 avg reward 7.46
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.56


Best static template: ID=4 (Premium) with avg reward=7.56, GMV=7.11, CM2=0.74

Running LinUCB with features=rich for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  LINUCB progress:   2.0% (400/20000) | avg reward 10.51
  LINUCB progress:   4.0% (800/20000) | avg reward 10.49
  LINUCB progress:   6.0% (1200/20000) | avg reward 10.84
  LINUCB progress:   8.0% (1600/20000) | avg reward 10.77
  LINUCB progress:  10.0% (2000/20000) | avg reward 10.38
  LINUCB progress:  12.0% (2400/20000) | avg reward 10.40
  LINUCB progress:  14.0% (2800/20000) | avg reward 10.44
  LINUCB progress:  16.0% (3200/20000) | avg reward 10.47
  LINUCB progress:  18.0% (3600/20000) | avg reward 10.20
  LINUCB progress:  20.0% (4000/20000) | avg reward 10.25
  LINUCB progress:  22.0% (4400/20000) | avg reward 10.17
  LINUCB progress:  24.0% (4800/20000) | avg reward 10.20
  LINUCB progress:  26.0% (5200/20000) | avg reward 10.19
  LINUCB progress:  28.0% (5600/20000) | avg reward 10.08
  LINUCB progress:  30.0% (6000/20000) | avg reward 10.12
  LINUCB progress:  32.0% (6400/20000) | avg reward 10.13
  LINUCB progress:  34.0% (6800/20000) | avg reward 10.09
  LINUCB progress:  36.0% (7200/20000) | avg reward 10.05
  LINUCB progress:  38.0% (7600/20000) | avg reward 10.02
  LINUCB progress:  40.0% (8000/20000) | avg reward 10.01
  LINUCB progress:  42.0% (8400/20000) | avg reward 10.05
  LINUCB progress:  44.0% (8800/20000) | avg reward 10.07
  LINUCB progress:  46.0% (9200/20000) | avg reward 10.03
  LINUCB progress:  48.0% (9600/20000) | avg reward 9.97
  LINUCB progress:  50.0% (10000/20000) | avg reward 10.01
  LINUCB progress:  52.0% (10400/20000) | avg reward 10.00
  LINUCB progress:  54.0% (10800/20000) | avg reward 10.01
  LINUCB progress:  56.0% (11200/20000) | avg reward 10.01
  LINUCB progress:  58.0% (11600/20000) | avg reward 9.98
  LINUCB progress:  60.0% (12000/20000) | avg reward 9.97
  LINUCB progress:  62.0% (12400/20000) | avg reward 9.94
  LINUCB progress:  64.0% (12800/20000) | avg reward 9.94
  LINUCB progress:  66.0% (13200/20000) | avg reward 9.91
  LINUCB progress:  68.0% (13600/20000) | avg reward 9.97
  LINUCB progress:  70.0% (14000/20000) | avg reward 10.01
  LINUCB progress:  72.0% (14400/20000) | avg reward 10.06
  LINUCB progress:  74.0% (14800/20000) | avg reward 10.03
  LINUCB progress:  76.0% (15200/20000) | avg reward 10.00
  LINUCB progress:  78.0% (15600/20000) | avg reward 10.00
  LINUCB progress:  80.0% (16000/20000) | avg reward 9.99
  LINUCB progress:  82.0% (16400/20000) | avg reward 10.01
  LINUCB progress:  84.0% (16800/20000) | avg reward 10.03
  LINUCB progress:  86.0% (17200/20000) | avg reward 10.06
  LINUCB progress:  88.0% (17600/20000) | avg reward 10.08
  LINUCB progress:  90.0% (18000/20000) | avg reward 10.13
  LINUCB progress:  92.0% (18400/20000) | avg reward 10.12
  LINUCB progress:  94.0% (18800/20000) | avg reward 10.11
  LINUCB progress:  96.0% (19200/20000) | avg reward 10.11
  LINUCB progress:  98.0% (19600/20000) | avg reward 10.13
  LINUCB progress: 100.0% (20000/20000) | avg reward 10.11

Running Thompson Sampling with features=rich for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 0.00
  TS progress:   2.0% (400/20000) | avg reward 6.91
  TS progress:   4.0% (800/20000) | avg reward 6.70
  TS progress:   6.0% (1200/20000) | avg reward 7.33
  TS progress:   8.0% (1600/20000) | avg reward 7.66
  TS progress:  10.0% (2000/20000) | avg reward 7.64
  TS progress:  12.0% (2400/20000) | avg reward 7.79
  TS progress:  14.0% (2800/20000) | avg reward 7.87
  TS progress:  16.0% (3200/20000) | avg reward 7.82
  TS progress:  18.0% (3600/20000) | avg reward 7.76
  TS progress:  20.0% (4000/20000) | avg reward 7.88
  TS progress:  22.0% (4400/20000) | avg reward 7.95
  TS progress:  24.0% (4800/20000) | avg reward 8.09
  TS progress:  26.0% (5200/20000) | avg reward 8.05
  TS progress:  28.0% (5600/20000) | avg reward 8.04
  TS progress:  30.0% (6000/20000) | avg reward 8.19
  TS progress:  32.0% (6400/20000) | avg reward 8.19
  TS progress:  34.0% (6800/20000) | avg reward 8.15
  TS progress:  36.0% (7200/20000) | avg reward 8.17
  TS progress:  38.0% (7600/20000) | avg reward 8.19
  TS progress:  40.0% (8000/20000) | avg reward 8.18
  TS progress:  42.0% (8400/20000) | avg reward 8.16
  TS progress:  44.0% (8800/20000) | avg reward 8.18
  TS progress:  46.0% (9200/20000) | avg reward 8.18
  TS progress:  48.0% (9600/20000) | avg reward 8.18
  TS progress:  50.0% (10000/20000) | avg reward 8.20
  TS progress:  52.0% (10400/20000) | avg reward 8.16
  TS progress:  54.0% (10800/20000) | avg reward 8.08
  TS progress:  56.0% (11200/20000) | avg reward 8.08
  TS progress:  58.0% (11600/20000) | avg reward 8.08
  TS progress:  60.0% (12000/20000) | avg reward 8.09
  TS progress:  62.0% (12400/20000) | avg reward 8.11
  TS progress:  64.0% (12800/20000) | avg reward 8.13
  TS progress:  66.0% (13200/20000) | avg reward 8.13
  TS progress:  68.0% (13600/20000) | avg reward 8.14
  TS progress:  70.0% (14000/20000) | avg reward 8.14
  TS progress:  72.0% (14400/20000) | avg reward 8.15
  TS progress:  74.0% (14800/20000) | avg reward 8.12
  TS progress:  76.0% (15200/20000) | avg reward 8.07
  TS progress:  78.0% (15600/20000) | avg reward 8.07
  TS progress:  80.0% (16000/20000) | avg reward 8.07
  TS progress:  82.0% (16400/20000) | avg reward 8.09
  TS progress:  84.0% (16800/20000) | avg reward 8.09
  TS progress:  86.0% (17200/20000) | avg reward 8.12
  TS progress:  88.0% (17600/20000) | avg reward 8.09
  TS progress:  90.0% (18000/20000) | avg reward 8.11
  TS progress:  92.0% (18400/20000) | avg reward 8.13
  TS progress:  94.0% (18800/20000) | avg reward 8.14
  TS progress:  96.0% (19200/20000) | avg reward 8.16
  TS progress:  98.0% (19600/20000) | avg reward 8.17
  TS progress: 100.0% (20000/20000) | avg reward 8.15

  Static (best) segment stats:   0.0% (1/20000) | avg reward 0.00
  Static (best) segment stats:   2.0% (400/20000) | avg reward 7.84
  Static (best) segment stats:   4.0% (800/20000) | avg reward 7.25
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.07
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.12
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.20
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.11
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.15
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.21
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.36
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.41
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.39
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.40
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.50
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.51
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.54
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.60
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.60
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.61
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.63
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.66
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.72
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.75
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.70
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.70
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.71
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.68
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.67
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.68
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.71
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.72
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.75
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.73
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.73
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.71
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.72
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.70
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.70
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.73
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.71
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.68
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.64
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.64
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.68
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.68
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.67
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.64
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.66
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.64
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.65
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.65


Static templates (per-episode averages):
ID  Template             Reward         GMV         CM2
 0  Neutral                5.75        5.28        0.50
 1  High Margin            5.44        5.07        0.63
 2  CM2 Boost              7.35        6.73        0.61
 3  Popular                4.79        4.53        0.52
 4  Premium                7.56        7.11        0.74
 5  Budget                 3.44        3.04        0.26
 6  Discount               5.04        4.62        0.45
 7  Strategic              4.83        3.99       -0.13

Summary (per-episode averages):
Policy                  Reward         GMV         CM2  ΔGMV vs static
Static-Premium            7.56        7.11        0.74          +0.00%
LinUCB                   10.11        9.34        0.93         +31.31%
ThompsonSampling          8.15        7.47        0.61          +5.10%

Per-segment GMV (static best vs bandits):
Segment          Static GMV  LinUCB GMV      TS GMV    LinUCB Δ%     TS Δ%
price_hunter           0.00        0.00        0.00       +0.00%    +0.00%
pl_lover               5.34       11.48        5.23     +114.93%    -2.06%
premium               31.60       32.17       32.29       +1.81%    +2.18%
litter_heavy           4.94        6.47        5.33      +31.01%    +7.90%

Template selection frequencies — LinUCB:
   0 Neutral        :   2.83%
   1 High Margin    :   3.50%
   2 CM2 Boost      :  50.25%
   3 Popular        :   3.79%
   4 Premium        :  17.51%
   5 Budget         :  18.95%
   6 Discount       :   0.80%
   7 Strategic      :   2.37%

Template selection frequencies — ThompsonSampling:
   0 Neutral        :   9.30%
   1 High Margin    :   0.26%
   2 CM2 Boost      :  18.84%
   3 Popular        :   1.33%
   4 Premium        :  36.68%
   5 Budget         :   1.49%
   6 Discount       :   2.47%
   7 Strategic      :  29.64%

Notes:
- This demonstration can use either a simple context feature map (segment + query type) or a richer feature set with catalog aggregates.
- In many runs, bandits with simple features underperform strong static templates; richer features typically reduce this gap, revealing when linear contextual bandits are appropriate.

For theory and detailed diagnostics, see:
  - docs/book/drafts/ch06/discrete_template_bandits.md
  - docs/book/drafts/ch06/exercises_labs.md
================================================================================

✓ Test 2 completed

==============================================================================
TEST 3: Rich Features + Quantized Regularization (Winning Config)
==============================================================================\033[0m

Expected: LinUCB achieves ~5-8% GMV lift, TS achieves ~3-5% lift


================================================================================
CHAPTER 6 — TEMPLATE BANDITS DEMONSTRATION
================================================================================

Experiment horizons: static=2,000 episodes, bandits=20,000 episodes.
Feature mode: 'rich'

Generating catalog with 10,000 products (world seed=20250322)...

Resolved bandit hyperparameters → prior_weight=50, lin_alpha=0.20, ts_sigma=0.20
Rich feature regularization: mode=quantized, blend_weight=0.40, shrink=0.90, quant_step=0.25, clip=[-3.50, 3.50]

Running static template baselines for 2000 episodes...
  Static templates progress:   0.1% (1/2000) | best template ID 4 avg reward 20.70
  Static templates progress:   2.0% (40/2000) | best template ID 4 avg reward 8.55
  Static templates progress:   4.0% (80/2000) | best template ID 4 avg reward 10.36
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 11.91
  Static templates progress:   8.0% (160/2000) | best template ID 4 avg reward 9.97
  Static templates progress:  10.0% (200/2000) | best template ID 4 avg reward 9.41
  Static templates progress:  12.0% (240/2000) | best template ID 4 avg reward 9.80
  Static templates progress:  14.0% (280/2000) | best template ID 4 avg reward 9.60
  Static templates progress:  16.0% (320/2000) | best template ID 4 avg reward 9.90
  Static templates progress:  18.0% (360/2000) | best template ID 4 avg reward 9.64
  Static templates progress:  20.0% (400/2000) | best template ID 4 avg reward 9.57
  Static templates progress:  22.0% (440/2000) | best template ID 4 avg reward 9.49
  Static templates progress:  24.0% (480/2000) | best template ID 4 avg reward 9.21
  Static templates progress:  26.0% (520/2000) | best template ID 4 avg reward 8.67
  Static templates progress:  28.0% (560/2000) | best template ID 4 avg reward 8.82
  Static templates progress:  30.0% (600/2000) | best template ID 4 avg reward 8.78
  Static templates progress:  32.0% (640/2000) | best template ID 4 avg reward 8.88
  Static templates progress:  34.0% (680/2000) | best template ID 4 avg reward 8.75
  Static templates progress:  36.0% (720/2000) | best template ID 4 avg reward 8.77
  Static templates progress:  38.0% (760/2000) | best template ID 4 avg reward 8.52
  Static templates progress:  40.0% (800/2000) | best template ID 4 avg reward 8.37
  Static templates progress:  42.0% (840/2000) | best template ID 4 avg reward 8.39
  Static templates progress:  44.0% (880/2000) | best template ID 4 avg reward 8.30
  Static templates progress:  46.0% (920/2000) | best template ID 4 avg reward 8.22
  Static templates progress:  48.0% (960/2000) | best template ID 4 avg reward 8.38
  Static templates progress:  50.0% (1000/2000) | best template ID 4 avg reward 8.14
  Static templates progress:  52.0% (1040/2000) | best template ID 4 avg reward 8.01
  Static templates progress:  54.0% (1080/2000) | best template ID 2 avg reward 7.92
  Static templates progress:  56.0% (1120/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  58.0% (1160/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  60.0% (1200/2000) | best template ID 2 avg reward 8.09
  Static templates progress:  62.0% (1240/2000) | best template ID 2 avg reward 8.06
  Static templates progress:  64.0% (1280/2000) | best template ID 2 avg reward 8.08
  Static templates progress:  66.0% (1320/2000) | best template ID 2 avg reward 8.00
  Static templates progress:  68.0% (1360/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.96
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.90
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 7.87
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 7.72
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.67
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.54
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.55
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.53
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.48
  Static templates progress:  98.0% (1960/2000) | best template ID 4 avg reward 7.46
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.56


Best static template: ID=4 (Premium) with avg reward=7.56, GMV=7.11, CM2=0.74

Running LinUCB with features=rich for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  LINUCB progress:   2.0% (400/20000) | avg reward 8.12
  LINUCB progress:   4.0% (800/20000) | avg reward 8.13
  LINUCB progress:   6.0% (1200/20000) | avg reward 7.91
  LINUCB progress:   8.0% (1600/20000) | avg reward 7.99
  LINUCB progress:  10.0% (2000/20000) | avg reward 7.76
  LINUCB progress:  12.0% (2400/20000) | avg reward 7.71
  LINUCB progress:  14.0% (2800/20000) | avg reward 7.84
  LINUCB progress:  16.0% (3200/20000) | avg reward 7.93
  LINUCB progress:  18.0% (3600/20000) | avg reward 8.06
  LINUCB progress:  20.0% (4000/20000) | avg reward 8.12
  LINUCB progress:  22.0% (4400/20000) | avg reward 8.12
  LINUCB progress:  24.0% (4800/20000) | avg reward 8.17
  LINUCB progress:  26.0% (5200/20000) | avg reward 8.13
  LINUCB progress:  28.0% (5600/20000) | avg reward 8.18
  LINUCB progress:  30.0% (6000/20000) | avg reward 8.23
  LINUCB progress:  32.0% (6400/20000) | avg reward 8.18
  LINUCB progress:  34.0% (6800/20000) | avg reward 8.17
  LINUCB progress:  36.0% (7200/20000) | avg reward 8.17
  LINUCB progress:  38.0% (7600/20000) | avg reward 8.17
  LINUCB progress:  40.0% (8000/20000) | avg reward 8.15
  LINUCB progress:  42.0% (8400/20000) | avg reward 8.18
  LINUCB progress:  44.0% (8800/20000) | avg reward 8.20
  LINUCB progress:  46.0% (9200/20000) | avg reward 8.19
  LINUCB progress:  48.0% (9600/20000) | avg reward 8.17
  LINUCB progress:  50.0% (10000/20000) | avg reward 8.18
  LINUCB progress:  52.0% (10400/20000) | avg reward 8.15
  LINUCB progress:  54.0% (10800/20000) | avg reward 8.18
  LINUCB progress:  56.0% (11200/20000) | avg reward 8.17
  LINUCB progress:  58.0% (11600/20000) | avg reward 8.13
  LINUCB progress:  60.0% (12000/20000) | avg reward 8.15
  LINUCB progress:  62.0% (12400/20000) | avg reward 8.17
  LINUCB progress:  64.0% (12800/20000) | avg reward 8.19
  LINUCB progress:  66.0% (13200/20000) | avg reward 8.15
  LINUCB progress:  68.0% (13600/20000) | avg reward 8.18
  LINUCB progress:  70.0% (14000/20000) | avg reward 8.18
  LINUCB progress:  72.0% (14400/20000) | avg reward 8.16
  LINUCB progress:  74.0% (14800/20000) | avg reward 8.17
  LINUCB progress:  76.0% (15200/20000) | avg reward 8.18
  LINUCB progress:  78.0% (15600/20000) | avg reward 8.18
  LINUCB progress:  80.0% (16000/20000) | avg reward 8.20
  LINUCB progress:  82.0% (16400/20000) | avg reward 8.20
  LINUCB progress:  84.0% (16800/20000) | avg reward 8.20
  LINUCB progress:  86.0% (17200/20000) | avg reward 8.21
  LINUCB progress:  88.0% (17600/20000) | avg reward 8.19
  LINUCB progress:  90.0% (18000/20000) | avg reward 8.18
  LINUCB progress:  92.0% (18400/20000) | avg reward 8.19
  LINUCB progress:  94.0% (18800/20000) | avg reward 8.17
  LINUCB progress:  96.0% (19200/20000) | avg reward 8.17
  LINUCB progress:  98.0% (19600/20000) | avg reward 8.18
  LINUCB progress: 100.0% (20000/20000) | avg reward 8.20

Running Thompson Sampling with features=rich for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 0.00
  TS progress:   2.0% (400/20000) | avg reward 5.03
  TS progress:   4.0% (800/20000) | avg reward 5.05
  TS progress:   6.0% (1200/20000) | avg reward 5.19
  TS progress:   8.0% (1600/20000) | avg reward 5.67
  TS progress:  10.0% (2000/20000) | avg reward 6.02
  TS progress:  12.0% (2400/20000) | avg reward 6.39
  TS progress:  14.0% (2800/20000) | avg reward 6.78
  TS progress:  16.0% (3200/20000) | avg reward 7.03
  TS progress:  18.0% (3600/20000) | avg reward 7.14
  TS progress:  20.0% (4000/20000) | avg reward 7.19
  TS progress:  22.0% (4400/20000) | avg reward 7.39
  TS progress:  24.0% (4800/20000) | avg reward 7.55
  TS progress:  26.0% (5200/20000) | avg reward 7.57
  TS progress:  28.0% (5600/20000) | avg reward 7.59
  TS progress:  30.0% (6000/20000) | avg reward 7.59
  TS progress:  32.0% (6400/20000) | avg reward 7.65
  TS progress:  34.0% (6800/20000) | avg reward 7.76
  TS progress:  36.0% (7200/20000) | avg reward 7.82
  TS progress:  38.0% (7600/20000) | avg reward 7.81
  TS progress:  40.0% (8000/20000) | avg reward 7.83
  TS progress:  42.0% (8400/20000) | avg reward 7.82
  TS progress:  44.0% (8800/20000) | avg reward 7.84
  TS progress:  46.0% (9200/20000) | avg reward 7.87
  TS progress:  48.0% (9600/20000) | avg reward 7.89
  TS progress:  50.0% (10000/20000) | avg reward 7.92
  TS progress:  52.0% (10400/20000) | avg reward 7.93
  TS progress:  54.0% (10800/20000) | avg reward 7.93
  TS progress:  56.0% (11200/20000) | avg reward 7.93
  TS progress:  58.0% (11600/20000) | avg reward 7.97
  TS progress:  60.0% (12000/20000) | avg reward 7.93
  TS progress:  62.0% (12400/20000) | avg reward 7.89
  TS progress:  64.0% (12800/20000) | avg reward 7.88
  TS progress:  66.0% (13200/20000) | avg reward 7.88
  TS progress:  68.0% (13600/20000) | avg reward 7.87
  TS progress:  70.0% (14000/20000) | avg reward 7.88
  TS progress:  72.0% (14400/20000) | avg reward 7.87
  TS progress:  74.0% (14800/20000) | avg reward 7.91
  TS progress:  76.0% (15200/20000) | avg reward 7.91
  TS progress:  78.0% (15600/20000) | avg reward 7.90
  TS progress:  80.0% (16000/20000) | avg reward 7.94
  TS progress:  82.0% (16400/20000) | avg reward 7.94
  TS progress:  84.0% (16800/20000) | avg reward 7.96
  TS progress:  86.0% (17200/20000) | avg reward 7.95
  TS progress:  88.0% (17600/20000) | avg reward 7.95
  TS progress:  90.0% (18000/20000) | avg reward 7.95
  TS progress:  92.0% (18400/20000) | avg reward 7.96
  TS progress:  94.0% (18800/20000) | avg reward 7.96
  TS progress:  96.0% (19200/20000) | avg reward 7.96
  TS progress:  98.0% (19600/20000) | avg reward 7.99
  TS progress: 100.0% (20000/20000) | avg reward 7.99

  Static (best) segment stats:   0.0% (1/20000) | avg reward 0.00
  Static (best) segment stats:   2.0% (400/20000) | avg reward 7.84
  Static (best) segment stats:   4.0% (800/20000) | avg reward 7.25
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.07
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.12
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.20
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.11
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.15
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.21
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.36
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.41
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.39
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.40
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.50
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.51
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.54
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.60
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.60
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.61
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.63
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.66
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.72
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.75
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.70
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.70
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.71
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.68
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.67
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.68
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.71
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.72
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.75
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.73
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.73
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.71
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.72
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.70
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.70
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.73
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.71
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.68
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.64
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.64
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.68
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.68
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.67
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.64
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.66
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.64
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.65
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.65


Static templates (per-episode averages):
ID  Template             Reward         GMV         CM2
 0  Neutral                5.75        5.28        0.50
 1  High Margin            5.44        5.07        0.63
 2  CM2 Boost              7.35        6.73        0.61
 3  Popular                4.79        4.53        0.52
 4  Premium                7.56        7.11        0.74
 5  Budget                 3.44        3.04        0.26
 6  Discount               5.04        4.62        0.45
 7  Strategic              4.83        3.99       -0.13

Summary (per-episode averages):
Policy                  Reward         GMV         CM2  ΔGMV vs static
Static-Premium            7.56        7.11        0.74          +0.00%
LinUCB                    8.20        7.52        0.72          +5.79%
ThompsonSampling          7.99        7.34        0.69          +3.14%

Per-segment GMV (static best vs bandits):
Segment          Static GMV  LinUCB GMV      TS GMV    LinUCB Δ%     TS Δ%
price_hunter           0.00        0.00        0.00       +0.00%    +0.00%
pl_lover               5.34       11.45       11.51     +114.36%  +115.45%
premium               31.60       19.64       19.16      -37.83%   -39.37%
litter_heavy           4.94        6.61        6.34      +33.83%   +28.26%

Template selection frequencies — LinUCB:
   0 Neutral        :   4.47%
   1 High Margin    :  17.02%
   2 CM2 Boost      :  49.93%
   3 Popular        :  23.34%
   4 Premium        :   1.29%
   5 Budget         :   2.46%
   6 Discount       :   1.16%
   7 Strategic      :   0.34%

Template selection frequencies — ThompsonSampling:
   0 Neutral        :  12.64%
   1 High Margin    :  11.52%
   2 CM2 Boost      :  47.88%
   3 Popular        :   8.78%
   4 Premium        :  11.25%
   5 Budget         :   2.17%
   6 Discount       :   4.38%
   7 Strategic      :   1.39%

Notes:
- This demonstration can use either a simple context feature map (segment + query type) or a richer feature set with catalog aggregates.
- In many runs, bandits with simple features underperform strong static templates; richer features typically reduce this gap, revealing when linear contextual bandits are appropriate.

For theory and detailed diagnostics, see:
  - docs/book/drafts/ch06/discrete_template_bandits.md
  - docs/book/drafts/ch06/exercises_labs.md
================================================================================

✓ Test 3 completed

==============================================================================
TEST 4: Rich Estimated Features (Practical Scenario)
==============================================================================\033[0m

Expected: Bandits work with estimated (non-oracle) user features


================================================================================
CHAPTER 6 — TEMPLATE BANDITS DEMONSTRATION
================================================================================

Experiment horizons: static=2,000 episodes, bandits=20,000 episodes.
Feature mode: 'rich_est'

Generating catalog with 10,000 products (world seed=20250322)...

Resolved bandit hyperparameters → prior_weight=50, lin_alpha=0.20, ts_sigma=0.50

Running static template baselines for 2000 episodes...
  Static templates progress:   0.1% (1/2000) | best template ID 4 avg reward 20.70
  Static templates progress:   2.0% (40/2000) | best template ID 4 avg reward 8.55
  Static templates progress:   4.0% (80/2000) | best template ID 4 avg reward 10.36
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 11.91
  Static templates progress:   8.0% (160/2000) | best template ID 4 avg reward 9.97
  Static templates progress:  10.0% (200/2000) | best template ID 4 avg reward 9.41
  Static templates progress:  12.0% (240/2000) | best template ID 4 avg reward 9.80
  Static templates progress:  14.0% (280/2000) | best template ID 4 avg reward 9.60
  Static templates progress:  16.0% (320/2000) | best template ID 4 avg reward 9.90
  Static templates progress:  18.0% (360/2000) | best template ID 4 avg reward 9.64
  Static templates progress:  20.0% (400/2000) | best template ID 4 avg reward 9.57
  Static templates progress:  22.0% (440/2000) | best template ID 4 avg reward 9.49
  Static templates progress:  24.0% (480/2000) | best template ID 4 avg reward 9.21
  Static templates progress:  26.0% (520/2000) | best template ID 4 avg reward 8.67
  Static templates progress:  28.0% (560/2000) | best template ID 4 avg reward 8.82
  Static templates progress:  30.0% (600/2000) | best template ID 4 avg reward 8.78
  Static templates progress:  32.0% (640/2000) | best template ID 4 avg reward 8.88
  Static templates progress:  34.0% (680/2000) | best template ID 4 avg reward 8.75
  Static templates progress:  36.0% (720/2000) | best template ID 4 avg reward 8.77
  Static templates progress:  38.0% (760/2000) | best template ID 4 avg reward 8.52
  Static templates progress:  40.0% (800/2000) | best template ID 4 avg reward 8.37
  Static templates progress:  42.0% (840/2000) | best template ID 4 avg reward 8.39
  Static templates progress:  44.0% (880/2000) | best template ID 4 avg reward 8.30
  Static templates progress:  46.0% (920/2000) | best template ID 4 avg reward 8.22
  Static templates progress:  48.0% (960/2000) | best template ID 4 avg reward 8.38
  Static templates progress:  50.0% (1000/2000) | best template ID 4 avg reward 8.14
  Static templates progress:  52.0% (1040/2000) | best template ID 4 avg reward 8.01
  Static templates progress:  54.0% (1080/2000) | best template ID 2 avg reward 7.92
  Static templates progress:  56.0% (1120/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  58.0% (1160/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  60.0% (1200/2000) | best template ID 2 avg reward 8.09
  Static templates progress:  62.0% (1240/2000) | best template ID 2 avg reward 8.06
  Static templates progress:  64.0% (1280/2000) | best template ID 2 avg reward 8.08
  Static templates progress:  66.0% (1320/2000) | best template ID 2 avg reward 8.00
  Static templates progress:  68.0% (1360/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.96
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.90
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 7.87
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 7.72
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.67
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.54
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.55
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.53
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.48
  Static templates progress:  98.0% (1960/2000) | best template ID 4 avg reward 7.46
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.56


Best static template: ID=4 (Premium) with avg reward=7.56, GMV=7.11, CM2=0.74

Running LinUCB with features=rich_est for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  LINUCB progress:   2.0% (400/20000) | avg reward 8.12
  LINUCB progress:   4.0% (800/20000) | avg reward 8.13
  LINUCB progress:   6.0% (1200/20000) | avg reward 7.91
  LINUCB progress:   8.0% (1600/20000) | avg reward 7.99
  LINUCB progress:  10.0% (2000/20000) | avg reward 7.76
  LINUCB progress:  12.0% (2400/20000) | avg reward 7.71
  LINUCB progress:  14.0% (2800/20000) | avg reward 7.84
  LINUCB progress:  16.0% (3200/20000) | avg reward 7.93
  LINUCB progress:  18.0% (3600/20000) | avg reward 8.06
  LINUCB progress:  20.0% (4000/20000) | avg reward 8.12
  LINUCB progress:  22.0% (4400/20000) | avg reward 8.12
  LINUCB progress:  24.0% (4800/20000) | avg reward 8.17
  LINUCB progress:  26.0% (5200/20000) | avg reward 8.13
  LINUCB progress:  28.0% (5600/20000) | avg reward 8.18
  LINUCB progress:  30.0% (6000/20000) | avg reward 8.23
  LINUCB progress:  32.0% (6400/20000) | avg reward 8.18
  LINUCB progress:  34.0% (6800/20000) | avg reward 8.17
  LINUCB progress:  36.0% (7200/20000) | avg reward 8.17
  LINUCB progress:  38.0% (7600/20000) | avg reward 8.17
  LINUCB progress:  40.0% (8000/20000) | avg reward 8.15
  LINUCB progress:  42.0% (8400/20000) | avg reward 8.18
  LINUCB progress:  44.0% (8800/20000) | avg reward 8.20
  LINUCB progress:  46.0% (9200/20000) | avg reward 8.19
  LINUCB progress:  48.0% (9600/20000) | avg reward 8.17
  LINUCB progress:  50.0% (10000/20000) | avg reward 8.18
  LINUCB progress:  52.0% (10400/20000) | avg reward 8.15
  LINUCB progress:  54.0% (10800/20000) | avg reward 8.18
  LINUCB progress:  56.0% (11200/20000) | avg reward 8.17
  LINUCB progress:  58.0% (11600/20000) | avg reward 8.13
  LINUCB progress:  60.0% (12000/20000) | avg reward 8.15
  LINUCB progress:  62.0% (12400/20000) | avg reward 8.17
  LINUCB progress:  64.0% (12800/20000) | avg reward 8.19
  LINUCB progress:  66.0% (13200/20000) | avg reward 8.15
  LINUCB progress:  68.0% (13600/20000) | avg reward 8.18
  LINUCB progress:  70.0% (14000/20000) | avg reward 8.18
  LINUCB progress:  72.0% (14400/20000) | avg reward 8.16
  LINUCB progress:  74.0% (14800/20000) | avg reward 8.17
  LINUCB progress:  76.0% (15200/20000) | avg reward 8.18
  LINUCB progress:  78.0% (15600/20000) | avg reward 8.18
  LINUCB progress:  80.0% (16000/20000) | avg reward 8.20
  LINUCB progress:  82.0% (16400/20000) | avg reward 8.20
  LINUCB progress:  84.0% (16800/20000) | avg reward 8.20
  LINUCB progress:  86.0% (17200/20000) | avg reward 8.21
  LINUCB progress:  88.0% (17600/20000) | avg reward 8.19
  LINUCB progress:  90.0% (18000/20000) | avg reward 8.18
  LINUCB progress:  92.0% (18400/20000) | avg reward 8.19
  LINUCB progress:  94.0% (18800/20000) | avg reward 8.17
  LINUCB progress:  96.0% (19200/20000) | avg reward 8.17
  LINUCB progress:  98.0% (19600/20000) | avg reward 8.18
  LINUCB progress: 100.0% (20000/20000) | avg reward 8.20

Running Thompson Sampling with features=rich_est for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 0.00
  TS progress:   2.0% (400/20000) | avg reward 7.75
  TS progress:   4.0% (800/20000) | avg reward 8.10
  TS progress:   6.0% (1200/20000) | avg reward 8.34
  TS progress:   8.0% (1600/20000) | avg reward 8.92
  TS progress:  10.0% (2000/20000) | avg reward 9.31
  TS progress:  12.0% (2400/20000) | avg reward 9.32
  TS progress:  14.0% (2800/20000) | avg reward 9.31
  TS progress:  16.0% (3200/20000) | avg reward 9.39
  TS progress:  18.0% (3600/20000) | avg reward 9.48
  TS progress:  20.0% (4000/20000) | avg reward 9.54
  TS progress:  22.0% (4400/20000) | avg reward 9.61
  TS progress:  24.0% (4800/20000) | avg reward 9.69
  TS progress:  26.0% (5200/20000) | avg reward 9.74
  TS progress:  28.0% (5600/20000) | avg reward 9.83
  TS progress:  30.0% (6000/20000) | avg reward 9.89
  TS progress:  32.0% (6400/20000) | avg reward 9.87
  TS progress:  34.0% (6800/20000) | avg reward 9.97
  TS progress:  36.0% (7200/20000) | avg reward 9.96
  TS progress:  38.0% (7600/20000) | avg reward 9.96
  TS progress:  40.0% (8000/20000) | avg reward 9.96
  TS progress:  42.0% (8400/20000) | avg reward 10.00
  TS progress:  44.0% (8800/20000) | avg reward 10.00
  TS progress:  46.0% (9200/20000) | avg reward 10.06
  TS progress:  48.0% (9600/20000) | avg reward 10.06
  TS progress:  50.0% (10000/20000) | avg reward 10.05
  TS progress:  52.0% (10400/20000) | avg reward 10.01
  TS progress:  54.0% (10800/20000) | avg reward 10.05
  TS progress:  56.0% (11200/20000) | avg reward 10.06
  TS progress:  58.0% (11600/20000) | avg reward 10.06
  TS progress:  60.0% (12000/20000) | avg reward 10.11
  TS progress:  62.0% (12400/20000) | avg reward 10.11
  TS progress:  64.0% (12800/20000) | avg reward 10.07
  TS progress:  66.0% (13200/20000) | avg reward 10.04
  TS progress:  68.0% (13600/20000) | avg reward 10.08
  TS progress:  70.0% (14000/20000) | avg reward 10.07
  TS progress:  72.0% (14400/20000) | avg reward 10.12
  TS progress:  74.0% (14800/20000) | avg reward 10.15
  TS progress:  76.0% (15200/20000) | avg reward 10.16
  TS progress:  78.0% (15600/20000) | avg reward 10.15
  TS progress:  80.0% (16000/20000) | avg reward 10.13
  TS progress:  82.0% (16400/20000) | avg reward 10.15
  TS progress:  84.0% (16800/20000) | avg reward 10.16
  TS progress:  86.0% (17200/20000) | avg reward 10.16
  TS progress:  88.0% (17600/20000) | avg reward 10.17
  TS progress:  90.0% (18000/20000) | avg reward 10.16
  TS progress:  92.0% (18400/20000) | avg reward 10.14
  TS progress:  94.0% (18800/20000) | avg reward 10.12
  TS progress:  96.0% (19200/20000) | avg reward 10.09
  TS progress:  98.0% (19600/20000) | avg reward 10.07
  TS progress: 100.0% (20000/20000) | avg reward 10.08

  Static (best) segment stats:   0.0% (1/20000) | avg reward 0.00
  Static (best) segment stats:   2.0% (400/20000) | avg reward 7.84
  Static (best) segment stats:   4.0% (800/20000) | avg reward 7.25
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.07
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.12
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.20
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.11
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.15
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.21
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.36
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.41
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.39
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.40
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.50
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.51
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.54
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.60
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.60
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.61
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.63
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.66
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.72
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.75
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.70
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.70
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.71
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.68
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.67
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.68
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.71
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.72
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.75
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.73
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.73
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.71
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.72
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.70
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.70
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.73
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.71
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.68
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.64
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.64
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.68
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.68
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.67
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.64
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.66
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.64
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.65
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.65


Static templates (per-episode averages):
ID  Template             Reward         GMV         CM2
 0  Neutral                5.75        5.28        0.50
 1  High Margin            5.44        5.07        0.63
 2  CM2 Boost              7.35        6.73        0.61
 3  Popular                4.79        4.53        0.52
 4  Premium                7.56        7.11        0.74
 5  Budget                 3.44        3.04        0.26
 6  Discount               5.04        4.62        0.45
 7  Strategic              4.83        3.99       -0.13

Summary (per-episode averages):
Policy                  Reward         GMV         CM2  ΔGMV vs static
Static-Premium            7.56        7.11        0.74          +0.00%
LinUCB                    8.20        7.52        0.72          +5.79%
ThompsonSampling         10.08        9.31        0.93         +30.95%

Per-segment GMV (static best vs bandits):
Segment          Static GMV  LinUCB GMV      TS GMV    LinUCB Δ%     TS Δ%
price_hunter           0.00        0.00        0.00       +0.00%    +0.00%
pl_lover               5.34       11.45       11.47     +114.36%  +114.77%
premium               31.60       19.64       31.97      -37.83%    +1.17%
litter_heavy           4.94        6.61        6.32      +33.83%   +27.81%

Template selection frequencies — LinUCB:
   0 Neutral        :   4.47%
   1 High Margin    :  17.02%
   2 CM2 Boost      :  49.93%
   3 Popular        :  23.34%
   4 Premium        :   1.29%
   5 Budget         :   2.46%
   6 Discount       :   1.16%
   7 Strategic      :   0.34%

Template selection frequencies — ThompsonSampling:
   0 Neutral        :  25.39%
   1 High Margin    :   0.23%
   2 CM2 Boost      :  49.46%
   3 Popular        :   0.61%
   4 Premium        :  19.86%
   5 Budget         :   1.31%
   6 Discount       :   1.01%
   7 Strategic      :   2.12%

Notes:
- This demonstration can use either a simple context feature map (segment + query type) or a richer feature set with catalog aggregates.
- In many runs, bandits with simple features underperform strong static templates; richer features typically reduce this gap, revealing when linear contextual bandits are appropriate.

For theory and detailed diagnostics, see:
  - docs/book/drafts/ch06/discrete_template_bandits.md
  - docs/book/drafts/ch06/exercises_labs.md
================================================================================

✓ Test 4 completed

==============================================================================
TEST 5: Compute Arc - Full Simple → Rich Narrative
==============================================================================\033[0m

This runs both simple and rich experiments back-to-back

================================================================================
CHAPTER 6 — COMPUTE ARC: SIMPLE → RICH FEATURES
================================================================================

Config:
  Static episodes:  2,000
  Bandit episodes:  20,000
  Base seed:        20250322
  Output dir:       docs/book/drafts/ch06/data/verification_20251203T125142

Generating world...
  Catalog:  10,000 products
  Segments: ['price_hunter', 'pl_lover', 'premium', 'litter_heavy']
  Queries:  ['category', 'brand', 'generic']

================================================================================
EXPERIMENT 1: SIMPLE FEATURES (Segment + Query Type)
================================================================================
Hypothesis: Linear contextual bandits should beat static baselines.
Context: φ_simple(x) = [segment_onehot, query_onehot, bias] (8 dims)
Hyperparameters: prior_weight=50, lin_alpha=0.20, ts_sigma=0.50


Running static template baselines for 2000 episodes...
  Static templates progress:   0.1% (1/2000) | best template ID 0 avg reward 0.00
  Static templates progress:   2.0% (40/2000) | best template ID 4 avg reward 9.96
  Static templates progress:   4.0% (80/2000) | best template ID 4 avg reward 9.45
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 7.83
  Static templates progress:   8.0% (160/2000) | best template ID 2 avg reward 7.53
  Static templates progress:  10.0% (200/2000) | best template ID 2 avg reward 8.00
  Static templates progress:  12.0% (240/2000) | best template ID 2 avg reward 7.77
  Static templates progress:  14.0% (280/2000) | best template ID 2 avg reward 7.64
  Static templates progress:  16.0% (320/2000) | best template ID 2 avg reward 7.67
  Static templates progress:  18.0% (360/2000) | best template ID 2 avg reward 7.34
  Static templates progress:  20.0% (400/2000) | best template ID 2 avg reward 7.21
  Static templates progress:  22.0% (440/2000) | best template ID 2 avg reward 7.16
  Static templates progress:  24.0% (480/2000) | best template ID 2 avg reward 7.35
  Static templates progress:  26.0% (520/2000) | best template ID 2 avg reward 7.37
  Static templates progress:  28.0% (560/2000) | best template ID 2 avg reward 7.45
  Static templates progress:  30.0% (600/2000) | best template ID 2 avg reward 7.23
  Static templates progress:  32.0% (640/2000) | best template ID 2 avg reward 7.16
  Static templates progress:  34.0% (680/2000) | best template ID 2 avg reward 7.11
  Static templates progress:  36.0% (720/2000) | best template ID 2 avg reward 7.16
  Static templates progress:  38.0% (760/2000) | best template ID 2 avg reward 7.16
  Static templates progress:  40.0% (800/2000) | best template ID 2 avg reward 7.30
  Static templates progress:  42.0% (840/2000) | best template ID 2 avg reward 7.38
  Static templates progress:  44.0% (880/2000) | best template ID 2 avg reward 7.21
  Static templates progress:  46.0% (920/2000) | best template ID 2 avg reward 7.27
  Static templates progress:  48.0% (960/2000) | best template ID 2 avg reward 7.35
  Static templates progress:  50.0% (1000/2000) | best template ID 2 avg reward 7.23
  Static templates progress:  52.0% (1040/2000) | best template ID 2 avg reward 7.13
  Static templates progress:  54.0% (1080/2000) | best template ID 2 avg reward 7.09
  Static templates progress:  56.0% (1120/2000) | best template ID 2 avg reward 7.09
  Static templates progress:  58.0% (1160/2000) | best template ID 4 avg reward 7.07
  Static templates progress:  60.0% (1200/2000) | best template ID 4 avg reward 7.11
  Static templates progress:  62.0% (1240/2000) | best template ID 4 avg reward 7.10
  Static templates progress:  64.0% (1280/2000) | best template ID 4 avg reward 7.12
  Static templates progress:  66.0% (1320/2000) | best template ID 4 avg reward 7.15
  Static templates progress:  68.0% (1360/2000) | best template ID 4 avg reward 7.11
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.08
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.04
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 6.98
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 6.96
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.04
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 6.97
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.04
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.05
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.07
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.08
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.03
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.04
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.03
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.10
  Static templates progress:  98.0% (1960/2000) | best template ID 2 avg reward 7.13
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.16


Best static template: ID=4 (Premium) with avg reward=7.16, GMV=6.74, CM2=0.69

Running LinUCB with features=simple for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  LINUCB progress:   2.0% (400/20000) | avg reward 6.49
  LINUCB progress:   4.0% (800/20000) | avg reward 6.73
  LINUCB progress:   6.0% (1200/20000) | avg reward 6.99
  LINUCB progress:   8.0% (1600/20000) | avg reward 7.23
  LINUCB progress:  10.0% (2000/20000) | avg reward 7.22
  LINUCB progress:  12.0% (2400/20000) | avg reward 7.22
  LINUCB progress:  14.0% (2800/20000) | avg reward 7.28
  LINUCB progress:  16.0% (3200/20000) | avg reward 7.39
  LINUCB progress:  18.0% (3600/20000) | avg reward 7.46
  LINUCB progress:  20.0% (4000/20000) | avg reward 7.47
  LINUCB progress:  22.0% (4400/20000) | avg reward 7.38
  LINUCB progress:  24.0% (4800/20000) | avg reward 7.44
  LINUCB progress:  26.0% (5200/20000) | avg reward 7.41
  LINUCB progress:  28.0% (5600/20000) | avg reward 7.48
  LINUCB progress:  30.0% (6000/20000) | avg reward 7.53
  LINUCB progress:  32.0% (6400/20000) | avg reward 7.54
  LINUCB progress:  34.0% (6800/20000) | avg reward 7.56
  LINUCB progress:  36.0% (7200/20000) | avg reward 7.58
  LINUCB progress:  38.0% (7600/20000) | avg reward 7.66
  LINUCB progress:  40.0% (8000/20000) | avg reward 7.66
  LINUCB progress:  42.0% (8400/20000) | avg reward 7.64
  LINUCB progress:  44.0% (8800/20000) | avg reward 7.62
  LINUCB progress:  46.0% (9200/20000) | avg reward 7.62
  LINUCB progress:  48.0% (9600/20000) | avg reward 7.62
  LINUCB progress:  50.0% (10000/20000) | avg reward 7.60
  LINUCB progress:  52.0% (10400/20000) | avg reward 7.59
  LINUCB progress:  54.0% (10800/20000) | avg reward 7.59
  LINUCB progress:  56.0% (11200/20000) | avg reward 7.62
  LINUCB progress:  58.0% (11600/20000) | avg reward 7.64
  LINUCB progress:  60.0% (12000/20000) | avg reward 7.60
  LINUCB progress:  62.0% (12400/20000) | avg reward 7.62
  LINUCB progress:  64.0% (12800/20000) | avg reward 7.60
  LINUCB progress:  66.0% (13200/20000) | avg reward 7.58
  LINUCB progress:  68.0% (13600/20000) | avg reward 7.55
  LINUCB progress:  70.0% (14000/20000) | avg reward 7.55
  LINUCB progress:  72.0% (14400/20000) | avg reward 7.58
  LINUCB progress:  74.0% (14800/20000) | avg reward 7.55
  LINUCB progress:  76.0% (15200/20000) | avg reward 7.59
  LINUCB progress:  78.0% (15600/20000) | avg reward 7.61
  LINUCB progress:  80.0% (16000/20000) | avg reward 7.63
  LINUCB progress:  82.0% (16400/20000) | avg reward 7.64
  LINUCB progress:  84.0% (16800/20000) | avg reward 7.64
  LINUCB progress:  86.0% (17200/20000) | avg reward 7.64
  LINUCB progress:  88.0% (17600/20000) | avg reward 7.64
  LINUCB progress:  90.0% (18000/20000) | avg reward 7.64
  LINUCB progress:  92.0% (18400/20000) | avg reward 7.62
  LINUCB progress:  94.0% (18800/20000) | avg reward 7.64
  LINUCB progress:  96.0% (19200/20000) | avg reward 7.65
  LINUCB progress:  98.0% (19600/20000) | avg reward 7.66
  LINUCB progress: 100.0% (20000/20000) | avg reward 7.65

Running Thompson Sampling with features=simple for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 0.00
  TS progress:   2.0% (400/20000) | avg reward 7.01
  TS progress:   4.0% (800/20000) | avg reward 6.91
  TS progress:   6.0% (1200/20000) | avg reward 7.29
  TS progress:   8.0% (1600/20000) | avg reward 7.63
  TS progress:  10.0% (2000/20000) | avg reward 8.12
  TS progress:  12.0% (2400/20000) | avg reward 8.12
  TS progress:  14.0% (2800/20000) | avg reward 8.08
  TS progress:  16.0% (3200/20000) | avg reward 8.13
  TS progress:  18.0% (3600/20000) | avg reward 8.14
  TS progress:  20.0% (4000/20000) | avg reward 8.25
  TS progress:  22.0% (4400/20000) | avg reward 8.30
  TS progress:  24.0% (4800/20000) | avg reward 8.48
  TS progress:  26.0% (5200/20000) | avg reward 8.41
  TS progress:  28.0% (5600/20000) | avg reward 8.45
  TS progress:  30.0% (6000/20000) | avg reward 8.39
  TS progress:  32.0% (6400/20000) | avg reward 8.40
  TS progress:  34.0% (6800/20000) | avg reward 8.43
  TS progress:  36.0% (7200/20000) | avg reward 8.42
  TS progress:  38.0% (7600/20000) | avg reward 8.31
  TS progress:  40.0% (8000/20000) | avg reward 8.34
  TS progress:  42.0% (8400/20000) | avg reward 8.34
  TS progress:  44.0% (8800/20000) | avg reward 8.43
  TS progress:  46.0% (9200/20000) | avg reward 8.41
  TS progress:  48.0% (9600/20000) | avg reward 8.34
  TS progress:  50.0% (10000/20000) | avg reward 8.37
  TS progress:  52.0% (10400/20000) | avg reward 8.35
  TS progress:  54.0% (10800/20000) | avg reward 8.31
  TS progress:  56.0% (11200/20000) | avg reward 8.31
  TS progress:  58.0% (11600/20000) | avg reward 8.29
  TS progress:  60.0% (12000/20000) | avg reward 8.28
  TS progress:  62.0% (12400/20000) | avg reward 8.33
  TS progress:  64.0% (12800/20000) | avg reward 8.27
  TS progress:  66.0% (13200/20000) | avg reward 8.22
  TS progress:  68.0% (13600/20000) | avg reward 8.25
  TS progress:  70.0% (14000/20000) | avg reward 8.27
  TS progress:  72.0% (14400/20000) | avg reward 8.26
  TS progress:  74.0% (14800/20000) | avg reward 8.27
  TS progress:  76.0% (15200/20000) | avg reward 8.28
  TS progress:  78.0% (15600/20000) | avg reward 8.27
  TS progress:  80.0% (16000/20000) | avg reward 8.25
  TS progress:  82.0% (16400/20000) | avg reward 8.26
  TS progress:  84.0% (16800/20000) | avg reward 8.29
  TS progress:  86.0% (17200/20000) | avg reward 8.29
  TS progress:  88.0% (17600/20000) | avg reward 8.27
  TS progress:  90.0% (18000/20000) | avg reward 8.25
  TS progress:  92.0% (18400/20000) | avg reward 8.24
  TS progress:  94.0% (18800/20000) | avg reward 8.26
  TS progress:  96.0% (19200/20000) | avg reward 8.26
  TS progress:  98.0% (19600/20000) | avg reward 8.24
  TS progress: 100.0% (20000/20000) | avg reward 8.20

  Static (best) segment stats:   0.0% (1/20000) | avg reward 20.57
  Static (best) segment stats:   2.0% (400/20000) | avg reward 7.35
  Static (best) segment stats:   4.0% (800/20000) | avg reward 7.87
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.61
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.32
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.44
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.51
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.51
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.48
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.46
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.54
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.59
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.64
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.62
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.62
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.49
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.49
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.50
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.54
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.57
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.56
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.56
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.63
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.67
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.70
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.72
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.73
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.67
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.68
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.71
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.71
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.69
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.69
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.70
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.69
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.67
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.72
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.74
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.71
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.71
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.68
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.73
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.71
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.69
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.71
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.71
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.69
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.68
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.68
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.70
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.69


✓ Saved: docs/book/drafts/ch06/data/verification_20251203T125142/template_bandits_simple_summary.json

RESULTS (Simple Features):
  Static (best):  GMV = 6.74
  LinUCB:         GMV = 7.02  (+4.3%)
  TS:             GMV = 7.63  (+13.2%)

================================================================================
EXPERIMENT 2: RICH FEATURES (Full Product Aggregates)
================================================================================
Hypothesis: Rich features should improve bandit performance.
Context: φ_rich(x) = [segment, query, user_prefs, price, margin, ..., bias] (18 dims)
Hyperparameters: prior_weight=50, lin_alpha=0.20, ts_sigma=0.50


Running static template baselines for 2000 episodes...
  Static templates progress:   0.1% (1/2000) | best template ID 0 avg reward 0.00
  Static templates progress:   2.0% (40/2000) | best template ID 4 avg reward 9.96
  Static templates progress:   4.0% (80/2000) | best template ID 4 avg reward 9.45
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 7.83
  Static templates progress:   8.0% (160/2000) | best template ID 2 avg reward 7.53
  Static templates progress:  10.0% (200/2000) | best template ID 2 avg reward 8.00
  Static templates progress:  12.0% (240/2000) | best template ID 2 avg reward 7.77
  Static templates progress:  14.0% (280/2000) | best template ID 2 avg reward 7.64
  Static templates progress:  16.0% (320/2000) | best template ID 2 avg reward 7.67
  Static templates progress:  18.0% (360/2000) | best template ID 2 avg reward 7.34
  Static templates progress:  20.0% (400/2000) | best template ID 2 avg reward 7.21
  Static templates progress:  22.0% (440/2000) | best template ID 2 avg reward 7.16
  Static templates progress:  24.0% (480/2000) | best template ID 2 avg reward 7.35
  Static templates progress:  26.0% (520/2000) | best template ID 2 avg reward 7.37
  Static templates progress:  28.0% (560/2000) | best template ID 2 avg reward 7.45
  Static templates progress:  30.0% (600/2000) | best template ID 2 avg reward 7.23
  Static templates progress:  32.0% (640/2000) | best template ID 2 avg reward 7.16
  Static templates progress:  34.0% (680/2000) | best template ID 2 avg reward 7.11
  Static templates progress:  36.0% (720/2000) | best template ID 2 avg reward 7.16
  Static templates progress:  38.0% (760/2000) | best template ID 2 avg reward 7.16
  Static templates progress:  40.0% (800/2000) | best template ID 2 avg reward 7.30
  Static templates progress:  42.0% (840/2000) | best template ID 2 avg reward 7.38
  Static templates progress:  44.0% (880/2000) | best template ID 2 avg reward 7.21
  Static templates progress:  46.0% (920/2000) | best template ID 2 avg reward 7.27
  Static templates progress:  48.0% (960/2000) | best template ID 2 avg reward 7.35
  Static templates progress:  50.0% (1000/2000) | best template ID 2 avg reward 7.23
  Static templates progress:  52.0% (1040/2000) | best template ID 2 avg reward 7.13
  Static templates progress:  54.0% (1080/2000) | best template ID 2 avg reward 7.09
  Static templates progress:  56.0% (1120/2000) | best template ID 2 avg reward 7.09
  Static templates progress:  58.0% (1160/2000) | best template ID 4 avg reward 7.07
  Static templates progress:  60.0% (1200/2000) | best template ID 4 avg reward 7.11
  Static templates progress:  62.0% (1240/2000) | best template ID 4 avg reward 7.10
  Static templates progress:  64.0% (1280/2000) | best template ID 4 avg reward 7.12
  Static templates progress:  66.0% (1320/2000) | best template ID 4 avg reward 7.15
  Static templates progress:  68.0% (1360/2000) | best template ID 4 avg reward 7.11
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.08
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.04
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 6.98
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 6.96
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.04
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 6.97
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.04
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.05
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.07
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.08
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.03
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.04
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.03
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.10
  Static templates progress:  98.0% (1960/2000) | best template ID 2 avg reward 7.13
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.16


Best static template: ID=4 (Premium) with avg reward=7.16, GMV=6.74, CM2=0.69

Running LinUCB with features=rich for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  LINUCB progress:   2.0% (400/20000) | avg reward 6.63
  LINUCB progress:   4.0% (800/20000) | avg reward 7.38
  LINUCB progress:   6.0% (1200/20000) | avg reward 7.81
  LINUCB progress:   8.0% (1600/20000) | avg reward 7.50
  LINUCB progress:  10.0% (2000/20000) | avg reward 7.61
  LINUCB progress:  12.0% (2400/20000) | avg reward 7.68
  LINUCB progress:  14.0% (2800/20000) | avg reward 7.67
  LINUCB progress:  16.0% (3200/20000) | avg reward 7.74
  LINUCB progress:  18.0% (3600/20000) | avg reward 7.76
  LINUCB progress:  20.0% (4000/20000) | avg reward 7.80
  LINUCB progress:  22.0% (4400/20000) | avg reward 7.73
  LINUCB progress:  24.0% (4800/20000) | avg reward 7.82
  LINUCB progress:  26.0% (5200/20000) | avg reward 7.81
  LINUCB progress:  28.0% (5600/20000) | avg reward 7.84
  LINUCB progress:  30.0% (6000/20000) | avg reward 7.94
  LINUCB progress:  32.0% (6400/20000) | avg reward 7.96
  LINUCB progress:  34.0% (6800/20000) | avg reward 8.02
  LINUCB progress:  36.0% (7200/20000) | avg reward 8.01
  LINUCB progress:  38.0% (7600/20000) | avg reward 8.04
  LINUCB progress:  40.0% (8000/20000) | avg reward 8.05
  LINUCB progress:  42.0% (8400/20000) | avg reward 8.08
  LINUCB progress:  44.0% (8800/20000) | avg reward 8.07
  LINUCB progress:  46.0% (9200/20000) | avg reward 8.08
  LINUCB progress:  48.0% (9600/20000) | avg reward 8.11
  LINUCB progress:  50.0% (10000/20000) | avg reward 8.12
  LINUCB progress:  52.0% (10400/20000) | avg reward 8.13
  LINUCB progress:  54.0% (10800/20000) | avg reward 8.11
  LINUCB progress:  56.0% (11200/20000) | avg reward 8.10
  LINUCB progress:  58.0% (11600/20000) | avg reward 8.12
  LINUCB progress:  60.0% (12000/20000) | avg reward 8.10
  LINUCB progress:  62.0% (12400/20000) | avg reward 8.10
  LINUCB progress:  64.0% (12800/20000) | avg reward 8.08
  LINUCB progress:  66.0% (13200/20000) | avg reward 8.08
  LINUCB progress:  68.0% (13600/20000) | avg reward 8.09
  LINUCB progress:  70.0% (14000/20000) | avg reward 8.09
  LINUCB progress:  72.0% (14400/20000) | avg reward 8.12
  LINUCB progress:  74.0% (14800/20000) | avg reward 8.17
  LINUCB progress:  76.0% (15200/20000) | avg reward 8.16
  LINUCB progress:  78.0% (15600/20000) | avg reward 8.15
  LINUCB progress:  80.0% (16000/20000) | avg reward 8.13
  LINUCB progress:  82.0% (16400/20000) | avg reward 8.10
  LINUCB progress:  84.0% (16800/20000) | avg reward 8.11
  LINUCB progress:  86.0% (17200/20000) | avg reward 8.15
  LINUCB progress:  88.0% (17600/20000) | avg reward 8.13
  LINUCB progress:  90.0% (18000/20000) | avg reward 8.13
  LINUCB progress:  92.0% (18400/20000) | avg reward 8.14
  LINUCB progress:  94.0% (18800/20000) | avg reward 8.16
  LINUCB progress:  96.0% (19200/20000) | avg reward 8.13
  LINUCB progress:  98.0% (19600/20000) | avg reward 8.12
  LINUCB progress: 100.0% (20000/20000) | avg reward 8.12

Running Thompson Sampling with features=rich for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 0.00
  TS progress:   2.0% (400/20000) | avg reward 5.74
  TS progress:   4.0% (800/20000) | avg reward 7.08
  TS progress:   6.0% (1200/20000) | avg reward 7.24
  TS progress:   8.0% (1600/20000) | avg reward 7.31
  TS progress:  10.0% (2000/20000) | avg reward 7.41
  TS progress:  12.0% (2400/20000) | avg reward 7.50
  TS progress:  14.0% (2800/20000) | avg reward 7.53
  TS progress:  16.0% (3200/20000) | avg reward 7.63
  TS progress:  18.0% (3600/20000) | avg reward 7.65
  TS progress:  20.0% (4000/20000) | avg reward 7.60
  TS progress:  22.0% (4400/20000) | avg reward 7.64
  TS progress:  24.0% (4800/20000) | avg reward 7.74
  TS progress:  26.0% (5200/20000) | avg reward 7.70
  TS progress:  28.0% (5600/20000) | avg reward 7.73
  TS progress:  30.0% (6000/20000) | avg reward 7.83
  TS progress:  32.0% (6400/20000) | avg reward 7.85
  TS progress:  34.0% (6800/20000) | avg reward 7.91
  TS progress:  36.0% (7200/20000) | avg reward 7.91
  TS progress:  38.0% (7600/20000) | avg reward 7.87
  TS progress:  40.0% (8000/20000) | avg reward 7.91
  TS progress:  42.0% (8400/20000) | avg reward 7.96
  TS progress:  44.0% (8800/20000) | avg reward 7.98
  TS progress:  46.0% (9200/20000) | avg reward 8.01
  TS progress:  48.0% (9600/20000) | avg reward 8.03
  TS progress:  50.0% (10000/20000) | avg reward 8.09
  TS progress:  52.0% (10400/20000) | avg reward 8.09
  TS progress:  54.0% (10800/20000) | avg reward 8.07
  TS progress:  56.0% (11200/20000) | avg reward 8.10
  TS progress:  58.0% (11600/20000) | avg reward 8.12
  TS progress:  60.0% (12000/20000) | avg reward 8.09
  TS progress:  62.0% (12400/20000) | avg reward 8.10
  TS progress:  64.0% (12800/20000) | avg reward 8.09
  TS progress:  66.0% (13200/20000) | avg reward 8.13
  TS progress:  68.0% (13600/20000) | avg reward 8.12
  TS progress:  70.0% (14000/20000) | avg reward 8.14
  TS progress:  72.0% (14400/20000) | avg reward 8.12
  TS progress:  74.0% (14800/20000) | avg reward 8.12
  TS progress:  76.0% (15200/20000) | avg reward 8.12
  TS progress:  78.0% (15600/20000) | avg reward 8.13
  TS progress:  80.0% (16000/20000) | avg reward 8.11
  TS progress:  82.0% (16400/20000) | avg reward 8.08
  TS progress:  84.0% (16800/20000) | avg reward 8.09
  TS progress:  86.0% (17200/20000) | avg reward 8.07
  TS progress:  88.0% (17600/20000) | avg reward 8.06
  TS progress:  90.0% (18000/20000) | avg reward 8.04
  TS progress:  92.0% (18400/20000) | avg reward 8.04
  TS progress:  94.0% (18800/20000) | avg reward 8.05
  TS progress:  96.0% (19200/20000) | avg reward 8.09
  TS progress:  98.0% (19600/20000) | avg reward 8.08
  TS progress: 100.0% (20000/20000) | avg reward 8.10

  Static (best) segment stats:   0.0% (1/20000) | avg reward 20.57
  Static (best) segment stats:   2.0% (400/20000) | avg reward 7.35
  Static (best) segment stats:   4.0% (800/20000) | avg reward 7.87
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.61
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.32
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.44
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.51
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.51
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.48
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.46
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.54
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.59
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.64
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.62
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.62
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.49
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.49
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.50
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.54
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.57
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.56
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.56
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.63
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.67
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.70
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.72
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.73
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.67
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.68
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.71
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.71
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.69
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.69
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.70
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.69
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.67
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.72
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.74
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.71
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.71
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.68
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.73
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.71
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.69
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.71
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.71
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.69
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.68
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.68
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.70
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.69


✓ Saved: docs/book/drafts/ch06/data/verification_20251203T125142/template_bandits_rich_summary.json

RESULTS (Rich Features):
  Static (best):  GMV = 6.74  (same world)
  LinUCB:         GMV = 7.44  (+10.4%)
  TS:             GMV = 7.44  (+10.4%)

IMPROVEMENT (Rich vs Simple):
  LinUCB:  +0.41 GMV  (+5.9%)
  TS:      -0.19 GMV  (-2.5%)

✓ RICH FEATURES IMPROVE PERFORMANCE (as expected!)

================================================================================
GENERATING FIGURES
================================================================================

Plot 1: Segment GMV comparison...
Plot 2: Template selection frequencies (simple features)...
Plot 3: Template selection frequencies (rich features)...

================================================================================
SUMMARY
================================================================================

All artifacts saved to:
  Data:    docs/book/drafts/ch06/data/verification_20251203T125142
  Figures: docs/book/drafts/ch06/data/figures

Regenerate with:
  python scripts/ch06/ch06_compute_arc.py \
      --n-static 2000 \
      --n-bandit 20000 \
      --base-seed 20250322
================================================================================

✓ Test 5 completed

==============================================================================
TEST 6: Batch Matrix Runner - All Scenarios
==============================================================================\033[0m

Running 5 scenarios: simple_baseline, rich_oracle_raw, rich_oracle_blend,
                     rich_oracle_quantized, rich_estimated

Planned scenarios:
  [1/5] simple_baseline      features=simple    world_seed=20250322   bandit_seed=20250349   prior= 0 lin_alpha=1.00 ts_sigma=1.00 reg=none
  [2/5] rich_oracle_raw      features=rich      world_seed=20250323   bandit_seed=20250359   prior= 6 lin_alpha=1.00 ts_sigma=1.00 reg=none
  [3/5] rich_oracle_blend    features=rich      world_seed=20250324   bandit_seed=20250369   prior= 6 lin_alpha=1.00 ts_sigma=1.00 reg=blend
  [4/5] rich_oracle_quantized features=rich      world_seed=20250325   bandit_seed=20250379   prior=12 lin_alpha=0.85 ts_sigma=0.70 reg=quantized
  [5/5] rich_estimated       features=rich_est  world_seed=20250326   bandit_seed=20250389   prior=12 lin_alpha=0.85 ts_sigma=0.70 reg=none

Running 5 scenarios with 4 worker threads (parallel)...

Running static template baselines for 2000 episodes...

Running static template baselines for 2000 episodes...

Running static template baselines for 2000 episodes...

Running static template baselines for 2000 episodes...
  Static templates progress:   0.1% (1/2000) | best template ID 7 avg reward 38.29
  Static templates progress:   0.1% (1/2000) | best template ID 4 avg reward 20.70
  Static templates progress:   0.1% (1/2000) | best template ID 4 avg reward 47.20
  Static templates progress:   0.1% (1/2000) | best template ID 2 avg reward 15.35
  Static templates progress:   2.0% (40/2000) | best template ID 4 avg reward 10.03
  Static templates progress:   2.0% (40/2000) | best template ID 4 avg reward 7.76
  Static templates progress:   2.0% (40/2000) | best template ID 2 avg reward 7.81
  Static templates progress:   2.0% (40/2000) | best template ID 4 avg reward 8.55
  Static templates progress:   4.0% (80/2000) | best template ID 4 avg reward 8.13
  Static templates progress:   4.0% (80/2000) | best template ID 4 avg reward 7.55
  Static templates progress:   4.0% (80/2000) | best template ID 2 avg reward 8.49
  Static templates progress:   4.0% (80/2000) | best template ID 4 avg reward 10.36
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 8.01
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 9.32
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 11.91
  Static templates progress:   6.0% (120/2000) | best template ID 4 avg reward 9.25
  Static templates progress:   8.0% (160/2000) | best template ID 2 avg reward 7.24
  Static templates progress:   8.0% (160/2000) | best template ID 4 avg reward 7.78
  Static templates progress:   8.0% (160/2000) | best template ID 4 avg reward 9.97
  Static templates progress:   8.0% (160/2000) | best template ID 2 avg reward 9.20
  Static templates progress:  10.0% (200/2000) | best template ID 2 avg reward 7.76
  Static templates progress:  10.0% (200/2000) | best template ID 4 avg reward 7.16
  Static templates progress:  10.0% (200/2000) | best template ID 2 avg reward 8.90
  Static templates progress:  10.0% (200/2000) | best template ID 4 avg reward 9.41
  Static templates progress:  12.0% (240/2000) | best template ID 2 avg reward 7.79
  Static templates progress:  12.0% (240/2000) | best template ID 4 avg reward 7.93
  Static templates progress:  12.0% (240/2000) | best template ID 2 avg reward 9.34
  Static templates progress:  12.0% (240/2000) | best template ID 4 avg reward 9.80
  Static templates progress:  14.0% (280/2000) | best template ID 2 avg reward 7.23
  Static templates progress:  14.0% (280/2000) | best template ID 4 avg reward 8.12
  Static templates progress:  14.0% (280/2000) | best template ID 2 avg reward 8.64
  Static templates progress:  14.0% (280/2000) | best template ID 4 avg reward 9.60
  Static templates progress:  16.0% (320/2000) | best template ID 2 avg reward 7.14
  Static templates progress:  16.0% (320/2000) | best template ID 4 avg reward 8.27
  Static templates progress:  16.0% (320/2000) | best template ID 2 avg reward 8.43
  Static templates progress:  16.0% (320/2000) | best template ID 4 avg reward 9.90
  Static templates progress:  18.0% (360/2000) | best template ID 2 avg reward 7.48
  Static templates progress:  18.0% (360/2000) | best template ID 4 avg reward 7.69
  Static templates progress:  18.0% (360/2000) | best template ID 2 avg reward 8.38
  Static templates progress:  18.0% (360/2000) | best template ID 4 avg reward 9.64
  Static templates progress:  20.0% (400/2000) | best template ID 2 avg reward 7.56
  Static templates progress:  20.0% (400/2000) | best template ID 4 avg reward 7.30
  Static templates progress:  20.0% (400/2000) | best template ID 2 avg reward 8.35
  Static templates progress:  20.0% (400/2000) | best template ID 4 avg reward 9.57
  Static templates progress:  22.0% (440/2000) | best template ID 2 avg reward 7.48
  Static templates progress:  22.0% (440/2000) | best template ID 4 avg reward 7.67
  Static templates progress:  22.0% (440/2000) | best template ID 2 avg reward 8.24
  Static templates progress:  22.0% (440/2000) | best template ID 4 avg reward 9.49
  Static templates progress:  24.0% (480/2000) | best template ID 2 avg reward 7.14
  Static templates progress:  24.0% (480/2000) | best template ID 4 avg reward 7.70
  Static templates progress:  24.0% (480/2000) | best template ID 2 avg reward 8.16
  Static templates progress:  24.0% (480/2000) | best template ID 4 avg reward 9.21
  Static templates progress:  26.0% (520/2000) | best template ID 2 avg reward 7.33
  Static templates progress:  26.0% (520/2000) | best template ID 4 avg reward 7.31
  Static templates progress:  26.0% (520/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  26.0% (520/2000) | best template ID 4 avg reward 8.67
  Static templates progress:  28.0% (560/2000) | best template ID 2 avg reward 7.30
  Static templates progress:  28.0% (560/2000) | best template ID 4 avg reward 7.27
  Static templates progress:  28.0% (560/2000) | best template ID 2 avg reward 8.04
  Static templates progress:  28.0% (560/2000) | best template ID 4 avg reward 8.82
  Static templates progress:  30.0% (600/2000) | best template ID 2 avg reward 7.43
  Static templates progress:  30.0% (600/2000) | best template ID 4 avg reward 7.16
  Static templates progress:  30.0% (600/2000) | best template ID 2 avg reward 7.90
  Static templates progress:  30.0% (600/2000) | best template ID 4 avg reward 8.78
  Static templates progress:  32.0% (640/2000) | best template ID 2 avg reward 7.39
  Static templates progress:  32.0% (640/2000) | best template ID 4 avg reward 7.55
  Static templates progress:  32.0% (640/2000) | best template ID 2 avg reward 7.86
  Static templates progress:  32.0% (640/2000) | best template ID 4 avg reward 8.88
  Static templates progress:  34.0% (680/2000) | best template ID 2 avg reward 7.56
  Static templates progress:  34.0% (680/2000) | best template ID 4 avg reward 7.43
  Static templates progress:  34.0% (680/2000) | best template ID 2 avg reward 7.83
  Static templates progress:  34.0% (680/2000) | best template ID 4 avg reward 8.75
  Static templates progress:  36.0% (720/2000) | best template ID 2 avg reward 7.49
  Static templates progress:  36.0% (720/2000) | best template ID 4 avg reward 7.15
  Static templates progress:  36.0% (720/2000) | best template ID 2 avg reward 7.73
  Static templates progress:  36.0% (720/2000) | best template ID 4 avg reward 8.77
  Static templates progress:  38.0% (760/2000) | best template ID 2 avg reward 7.41
  Static templates progress:  38.0% (760/2000) | best template ID 4 avg reward 7.18
  Static templates progress:  38.0% (760/2000) | best template ID 2 avg reward 7.68
  Static templates progress:  38.0% (760/2000) | best template ID 4 avg reward 8.52
  Static templates progress:  40.0% (800/2000) | best template ID 2 avg reward 7.40
  Static templates progress:  40.0% (800/2000) | best template ID 4 avg reward 7.16
  Static templates progress:  40.0% (800/2000) | best template ID 2 avg reward 7.79
  Static templates progress:  40.0% (800/2000) | best template ID 4 avg reward 8.37
  Static templates progress:  42.0% (840/2000) | best template ID 2 avg reward 7.52
  Static templates progress:  42.0% (840/2000) | best template ID 4 avg reward 7.00
  Static templates progress:  42.0% (840/2000) | best template ID 2 avg reward 7.64
  Static templates progress:  42.0% (840/2000) | best template ID 4 avg reward 8.39
  Static templates progress:  44.0% (880/2000) | best template ID 2 avg reward 7.51
  Static templates progress:  44.0% (880/2000) | best template ID 4 avg reward 7.02
  Static templates progress:  44.0% (880/2000) | best template ID 2 avg reward 7.65
  Static templates progress:  44.0% (880/2000) | best template ID 4 avg reward 8.30
  Static templates progress:  46.0% (920/2000) | best template ID 2 avg reward 7.55
  Static templates progress:  46.0% (920/2000) | best template ID 4 avg reward 7.01
  Static templates progress:  46.0% (920/2000) | best template ID 2 avg reward 7.64
  Static templates progress:  46.0% (920/2000) | best template ID 4 avg reward 8.22
  Static templates progress:  48.0% (960/2000) | best template ID 2 avg reward 7.47
  Static templates progress:  48.0% (960/2000) | best template ID 4 avg reward 7.05
  Static templates progress:  48.0% (960/2000) | best template ID 2 avg reward 7.56
  Static templates progress:  48.0% (960/2000) | best template ID 4 avg reward 8.38
  Static templates progress:  50.0% (1000/2000) | best template ID 2 avg reward 7.51
  Static templates progress:  50.0% (1000/2000) | best template ID 4 avg reward 7.15
  Static templates progress:  50.0% (1000/2000) | best template ID 4 avg reward 7.74
  Static templates progress:  50.0% (1000/2000) | best template ID 4 avg reward 8.14
  Static templates progress:  52.0% (1040/2000) | best template ID 2 avg reward 7.47
  Static templates progress:  52.0% (1040/2000) | best template ID 4 avg reward 7.29
  Static templates progress:  52.0% (1040/2000) | best template ID 4 avg reward 7.81
  Static templates progress:  52.0% (1040/2000) | best template ID 4 avg reward 8.01
  Static templates progress:  54.0% (1080/2000) | best template ID 2 avg reward 7.41
  Static templates progress:  54.0% (1080/2000) | best template ID 4 avg reward 7.24
  Static templates progress:  54.0% (1080/2000) | best template ID 4 avg reward 7.86
  Static templates progress:  54.0% (1080/2000) | best template ID 2 avg reward 7.92
  Static templates progress:  56.0% (1120/2000) | best template ID 2 avg reward 7.33
  Static templates progress:  56.0% (1120/2000) | best template ID 4 avg reward 7.23
  Static templates progress:  56.0% (1120/2000) | best template ID 4 avg reward 7.72
  Static templates progress:  56.0% (1120/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  58.0% (1160/2000) | best template ID 2 avg reward 7.28
  Static templates progress:  58.0% (1160/2000) | best template ID 4 avg reward 7.15
  Static templates progress:  58.0% (1160/2000) | best template ID 2 avg reward 7.74
  Static templates progress:  58.0% (1160/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  60.0% (1200/2000) | best template ID 2 avg reward 7.21
  Static templates progress:  60.0% (1200/2000) | best template ID 4 avg reward 7.15
  Static templates progress:  60.0% (1200/2000) | best template ID 4 avg reward 7.83
  Static templates progress:  60.0% (1200/2000) | best template ID 2 avg reward 8.09
  Static templates progress:  62.0% (1240/2000) | best template ID 2 avg reward 7.31
  Static templates progress:  62.0% (1240/2000) | best template ID 4 avg reward 7.09
  Static templates progress:  62.0% (1240/2000) | best template ID 4 avg reward 7.96
  Static templates progress:  62.0% (1240/2000) | best template ID 2 avg reward 8.06
  Static templates progress:  64.0% (1280/2000) | best template ID 2 avg reward 7.26
  Static templates progress:  64.0% (1280/2000) | best template ID 4 avg reward 7.06
  Static templates progress:  64.0% (1280/2000) | best template ID 4 avg reward 7.88
  Static templates progress:  64.0% (1280/2000) | best template ID 2 avg reward 8.08
  Static templates progress:  66.0% (1320/2000) | best template ID 2 avg reward 7.22
  Static templates progress:  66.0% (1320/2000) | best template ID 4 avg reward 6.99
  Static templates progress:  66.0% (1320/2000) | best template ID 4 avg reward 7.90
  Static templates progress:  66.0% (1320/2000) | best template ID 2 avg reward 8.00
  Static templates progress:  68.0% (1360/2000) | best template ID 2 avg reward 7.31
  Static templates progress:  68.0% (1360/2000) | best template ID 4 avg reward 6.91
  Static templates progress:  68.0% (1360/2000) | best template ID 4 avg reward 7.94
  Static templates progress:  68.0% (1360/2000) | best template ID 2 avg reward 7.99
  Static templates progress:  70.0% (1400/2000) | best template ID 2 avg reward 7.28
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.06
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.91
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.96
  Static templates progress:  72.0% (1440/2000) | best template ID 2 avg reward 7.23
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.13
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.82
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.90
  Static templates progress:  74.0% (1480/2000) | best template ID 2 avg reward 7.21
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 7.21
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 7.78
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 7.87
  Static templates progress:  76.0% (1520/2000) | best template ID 2 avg reward 7.24
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 7.15
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 7.73
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  78.0% (1560/2000) | best template ID 2 avg reward 7.27
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.17
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.71
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  80.0% (1600/2000) | best template ID 2 avg reward 7.24
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 7.15
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 7.73
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 7.72
  Static templates progress:  82.0% (1640/2000) | best template ID 2 avg reward 7.25
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.16
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.74
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.67
  Static templates progress:  84.0% (1680/2000) | best template ID 2 avg reward 7.25
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.24
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.78
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.54
  Static templates progress:  86.0% (1720/2000) | best template ID 2 avg reward 7.26
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.27
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.74
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.55
  Static templates progress:  88.0% (1760/2000) | best template ID 2 avg reward 7.26
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.27
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.81
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  90.0% (1800/2000) | best template ID 2 avg reward 7.27
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.31
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.95
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  92.0% (1840/2000) | best template ID 2 avg reward 7.25
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.35
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.92
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.57
  Static templates progress:  94.0% (1880/2000) | best template ID 2 avg reward 7.21
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.43
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.92
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.53
  Static templates progress:  96.0% (1920/2000) | best template ID 2 avg reward 7.18
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.32
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.97
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.48
  Static templates progress:  98.0% (1960/2000) | best template ID 2 avg reward 7.12
  Static templates progress:  98.0% (1960/2000) | best template ID 4 avg reward 7.29
  Static templates progress:  98.0% (1960/2000) | best template ID 4 avg reward 7.93
  Static templates progress:  98.0% (1960/2000) | best template ID 4 avg reward 7.46
  Static templates progress: 100.0% (2000/2000) | best template ID 2 avg reward 7.11


Best static template: ID=2 (CM2 Boost) with avg reward=7.11, GMV=6.47, CM2=0.55

Running LinUCB with features=rich for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.25


Best static template: ID=4 (Premium) with avg reward=7.25, GMV=6.82, CM2=0.74

Running LinUCB with features=rich for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.85


Best static template: ID=4 (Premium) with avg reward=7.85, GMV=7.39, CM2=0.83

Running LinUCB with features=rich for 20000 episodes...
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.56


Best static template: ID=4 (Premium) with avg reward=7.56, GMV=7.11, CM2=0.74

Running LinUCB with features=simple for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  LINUCB progress:   0.0% (1/20000) | avg reward 0.00
  LINUCB progress:   2.0% (400/20000) | avg reward 7.61
  LINUCB progress:   2.0% (400/20000) | avg reward 9.49
  LINUCB progress:   2.0% (400/20000) | avg reward 4.76
  LINUCB progress:   2.0% (400/20000) | avg reward 8.14
  LINUCB progress:   4.0% (800/20000) | avg reward 8.06
  LINUCB progress:   4.0% (800/20000) | avg reward 9.26
  LINUCB progress:   4.0% (800/20000) | avg reward 4.96
  LINUCB progress:   4.0% (800/20000) | avg reward 7.82
  LINUCB progress:   6.0% (1200/20000) | avg reward 7.87
  LINUCB progress:   6.0% (1200/20000) | avg reward 10.08
  LINUCB progress:   6.0% (1200/20000) | avg reward 5.07
  LINUCB progress:   6.0% (1200/20000) | avg reward 7.63
  LINUCB progress:   8.0% (1600/20000) | avg reward 7.83
  LINUCB progress:   8.0% (1600/20000) | avg reward 10.05
  LINUCB progress:   8.0% (1600/20000) | avg reward 5.21
  LINUCB progress:   8.0% (1600/20000) | avg reward 7.92
  LINUCB progress:  10.0% (2000/20000) | avg reward 7.90
  LINUCB progress:  10.0% (2000/20000) | avg reward 10.08
  LINUCB progress:  10.0% (2000/20000) | avg reward 5.33
  LINUCB progress:  10.0% (2000/20000) | avg reward 8.26
  LINUCB progress:  12.0% (2400/20000) | avg reward 7.89
  LINUCB progress:  12.0% (2400/20000) | avg reward 10.01
  LINUCB progress:  12.0% (2400/20000) | avg reward 5.32
  LINUCB progress:  12.0% (2400/20000) | avg reward 8.16
  LINUCB progress:  14.0% (2800/20000) | avg reward 7.93
  LINUCB progress:  14.0% (2800/20000) | avg reward 9.95
  LINUCB progress:  14.0% (2800/20000) | avg reward 5.38
  LINUCB progress:  14.0% (2800/20000) | avg reward 8.33
  LINUCB progress:  16.0% (3200/20000) | avg reward 7.75
  LINUCB progress:  16.0% (3200/20000) | avg reward 9.91
  LINUCB progress:  16.0% (3200/20000) | avg reward 5.40
  LINUCB progress:  16.0% (3200/20000) | avg reward 8.25
  LINUCB progress:  18.0% (3600/20000) | avg reward 7.70
  LINUCB progress:  18.0% (3600/20000) | avg reward 9.79
  LINUCB progress:  18.0% (3600/20000) | avg reward 5.36
  LINUCB progress:  18.0% (3600/20000) | avg reward 8.32
  LINUCB progress:  20.0% (4000/20000) | avg reward 7.64
  LINUCB progress:  20.0% (4000/20000) | avg reward 9.86
  LINUCB progress:  20.0% (4000/20000) | avg reward 5.41
  LINUCB progress:  20.0% (4000/20000) | avg reward 8.37
  LINUCB progress:  22.0% (4400/20000) | avg reward 7.66
  LINUCB progress:  22.0% (4400/20000) | avg reward 9.68
  LINUCB progress:  22.0% (4400/20000) | avg reward 5.49
  LINUCB progress:  22.0% (4400/20000) | avg reward 8.29
  LINUCB progress:  24.0% (4800/20000) | avg reward 7.61
  LINUCB progress:  24.0% (4800/20000) | avg reward 9.71
  LINUCB progress:  24.0% (4800/20000) | avg reward 5.50
  LINUCB progress:  24.0% (4800/20000) | avg reward 8.23
  LINUCB progress:  26.0% (5200/20000) | avg reward 7.59
  LINUCB progress:  26.0% (5200/20000) | avg reward 9.73
  LINUCB progress:  26.0% (5200/20000) | avg reward 5.54
  LINUCB progress:  26.0% (5200/20000) | avg reward 8.26
  LINUCB progress:  28.0% (5600/20000) | avg reward 7.70
  LINUCB progress:  28.0% (5600/20000) | avg reward 9.65
  LINUCB progress:  28.0% (5600/20000) | avg reward 5.67
  LINUCB progress:  28.0% (5600/20000) | avg reward 8.15
  LINUCB progress:  30.0% (6000/20000) | avg reward 7.69
  LINUCB progress:  30.0% (6000/20000) | avg reward 9.63
  LINUCB progress:  30.0% (6000/20000) | avg reward 5.64
  LINUCB progress:  30.0% (6000/20000) | avg reward 8.17
  LINUCB progress:  32.0% (6400/20000) | avg reward 7.75
  LINUCB progress:  32.0% (6400/20000) | avg reward 9.61
  LINUCB progress:  32.0% (6400/20000) | avg reward 5.72
  LINUCB progress:  32.0% (6400/20000) | avg reward 8.15
  LINUCB progress:  34.0% (6800/20000) | avg reward 7.71
  LINUCB progress:  34.0% (6800/20000) | avg reward 9.67
  LINUCB progress:  34.0% (6800/20000) | avg reward 5.77
  LINUCB progress:  34.0% (6800/20000) | avg reward 8.14
  LINUCB progress:  36.0% (7200/20000) | avg reward 7.61
  LINUCB progress:  36.0% (7200/20000) | avg reward 9.66
  LINUCB progress:  36.0% (7200/20000) | avg reward 5.80
  LINUCB progress:  36.0% (7200/20000) | avg reward 8.21
  LINUCB progress:  38.0% (7600/20000) | avg reward 7.68
  LINUCB progress:  38.0% (7600/20000) | avg reward 9.59
  LINUCB progress:  38.0% (7600/20000) | avg reward 5.81
  LINUCB progress:  38.0% (7600/20000) | avg reward 8.22
  LINUCB progress:  40.0% (8000/20000) | avg reward 7.64
  LINUCB progress:  40.0% (8000/20000) | avg reward 9.50
  LINUCB progress:  40.0% (8000/20000) | avg reward 5.79
  LINUCB progress:  40.0% (8000/20000) | avg reward 8.23
  LINUCB progress:  42.0% (8400/20000) | avg reward 7.66
  LINUCB progress:  42.0% (8400/20000) | avg reward 9.45
  LINUCB progress:  42.0% (8400/20000) | avg reward 5.75
  LINUCB progress:  42.0% (8400/20000) | avg reward 8.18
  LINUCB progress:  44.0% (8800/20000) | avg reward 7.63
  LINUCB progress:  44.0% (8800/20000) | avg reward 9.52
  LINUCB progress:  44.0% (8800/20000) | avg reward 5.74
  LINUCB progress:  44.0% (8800/20000) | avg reward 8.19
  LINUCB progress:  46.0% (9200/20000) | avg reward 7.65
  LINUCB progress:  46.0% (9200/20000) | avg reward 9.58
  LINUCB progress:  46.0% (9200/20000) | avg reward 5.75
  LINUCB progress:  46.0% (9200/20000) | avg reward 8.16
  LINUCB progress:  48.0% (9600/20000) | avg reward 7.66
  LINUCB progress:  48.0% (9600/20000) | avg reward 9.60
  LINUCB progress:  48.0% (9600/20000) | avg reward 5.73
  LINUCB progress:  48.0% (9600/20000) | avg reward 8.18
  LINUCB progress:  50.0% (10000/20000) | avg reward 7.75
  LINUCB progress:  50.0% (10000/20000) | avg reward 9.65
  LINUCB progress:  50.0% (10000/20000) | avg reward 5.74
  LINUCB progress:  50.0% (10000/20000) | avg reward 8.14
  LINUCB progress:  52.0% (10400/20000) | avg reward 7.87
  LINUCB progress:  52.0% (10400/20000) | avg reward 9.65
  LINUCB progress:  52.0% (10400/20000) | avg reward 5.75
  LINUCB progress:  52.0% (10400/20000) | avg reward 8.15
  LINUCB progress:  54.0% (10800/20000) | avg reward 7.89
  LINUCB progress:  54.0% (10800/20000) | avg reward 9.70
  LINUCB progress:  54.0% (10800/20000) | avg reward 5.75
  LINUCB progress:  54.0% (10800/20000) | avg reward 8.14
  LINUCB progress:  56.0% (11200/20000) | avg reward 7.95
  LINUCB progress:  56.0% (11200/20000) | avg reward 9.76
  LINUCB progress:  56.0% (11200/20000) | avg reward 5.75
  LINUCB progress:  56.0% (11200/20000) | avg reward 8.13
  LINUCB progress:  58.0% (11600/20000) | avg reward 7.97
  LINUCB progress:  58.0% (11600/20000) | avg reward 9.75
  LINUCB progress:  58.0% (11600/20000) | avg reward 5.74
  LINUCB progress:  58.0% (11600/20000) | avg reward 8.13
  LINUCB progress:  60.0% (12000/20000) | avg reward 8.02
  LINUCB progress:  60.0% (12000/20000) | avg reward 9.71
  LINUCB progress:  60.0% (12000/20000) | avg reward 5.73
  LINUCB progress:  60.0% (12000/20000) | avg reward 8.11
  LINUCB progress:  62.0% (12400/20000) | avg reward 8.06
  LINUCB progress:  62.0% (12400/20000) | avg reward 9.79
  LINUCB progress:  62.0% (12400/20000) | avg reward 5.71
  LINUCB progress:  62.0% (12400/20000) | avg reward 8.11
  LINUCB progress:  64.0% (12800/20000) | avg reward 8.09
  LINUCB progress:  64.0% (12800/20000) | avg reward 9.77
  LINUCB progress:  64.0% (12800/20000) | avg reward 5.72
  LINUCB progress:  64.0% (12800/20000) | avg reward 8.11
  LINUCB progress:  66.0% (13200/20000) | avg reward 8.11
  LINUCB progress:  66.0% (13200/20000) | avg reward 9.78
  LINUCB progress:  66.0% (13200/20000) | avg reward 5.74
  LINUCB progress:  66.0% (13200/20000) | avg reward 8.09
  LINUCB progress:  68.0% (13600/20000) | avg reward 8.15
  LINUCB progress:  68.0% (13600/20000) | avg reward 9.77
  LINUCB progress:  68.0% (13600/20000) | avg reward 5.75
  LINUCB progress:  68.0% (13600/20000) | avg reward 8.09
  LINUCB progress:  70.0% (14000/20000) | avg reward 8.19
  LINUCB progress:  70.0% (14000/20000) | avg reward 9.76
  LINUCB progress:  70.0% (14000/20000) | avg reward 5.74
  LINUCB progress:  70.0% (14000/20000) | avg reward 8.07
  LINUCB progress:  72.0% (14400/20000) | avg reward 8.24
  LINUCB progress:  72.0% (14400/20000) | avg reward 9.73
  LINUCB progress:  72.0% (14400/20000) | avg reward 5.71
  LINUCB progress:  72.0% (14400/20000) | avg reward 8.06
  LINUCB progress:  74.0% (14800/20000) | avg reward 8.29
  LINUCB progress:  74.0% (14800/20000) | avg reward 9.73
  LINUCB progress:  74.0% (14800/20000) | avg reward 5.70
  LINUCB progress:  74.0% (14800/20000) | avg reward 8.07
  LINUCB progress:  76.0% (15200/20000) | avg reward 8.35
  LINUCB progress:  76.0% (15200/20000) | avg reward 9.69
  LINUCB progress:  76.0% (15200/20000) | avg reward 5.71
  LINUCB progress:  76.0% (15200/20000) | avg reward 8.07
  LINUCB progress:  78.0% (15600/20000) | avg reward 8.37
  LINUCB progress:  78.0% (15600/20000) | avg reward 9.70
  LINUCB progress:  78.0% (15600/20000) | avg reward 5.71
  LINUCB progress:  78.0% (15600/20000) | avg reward 8.06
  LINUCB progress:  80.0% (16000/20000) | avg reward 8.41
  LINUCB progress:  80.0% (16000/20000) | avg reward 9.75
  LINUCB progress:  80.0% (16000/20000) | avg reward 5.67
  LINUCB progress:  80.0% (16000/20000) | avg reward 8.06
  LINUCB progress:  82.0% (16400/20000) | avg reward 8.46
  LINUCB progress:  82.0% (16400/20000) | avg reward 9.78
  LINUCB progress:  82.0% (16400/20000) | avg reward 5.67
  LINUCB progress:  82.0% (16400/20000) | avg reward 8.06
  LINUCB progress:  84.0% (16800/20000) | avg reward 8.49
  LINUCB progress:  84.0% (16800/20000) | avg reward 9.77
  LINUCB progress:  84.0% (16800/20000) | avg reward 5.66
  LINUCB progress:  84.0% (16800/20000) | avg reward 8.08
  LINUCB progress:  86.0% (17200/20000) | avg reward 8.52
  LINUCB progress:  86.0% (17200/20000) | avg reward 9.75
  LINUCB progress:  86.0% (17200/20000) | avg reward 5.67
  LINUCB progress:  86.0% (17200/20000) | avg reward 8.05
  LINUCB progress:  88.0% (17600/20000) | avg reward 8.52
  LINUCB progress:  88.0% (17600/20000) | avg reward 9.75
  LINUCB progress:  88.0% (17600/20000) | avg reward 5.67
  LINUCB progress:  88.0% (17600/20000) | avg reward 8.06
  LINUCB progress:  90.0% (18000/20000) | avg reward 8.56
  LINUCB progress:  90.0% (18000/20000) | avg reward 9.77
  LINUCB progress:  90.0% (18000/20000) | avg reward 5.68
  LINUCB progress:  92.0% (18400/20000) | avg reward 8.59
  LINUCB progress:  90.0% (18000/20000) | avg reward 8.05
  LINUCB progress:  92.0% (18400/20000) | avg reward 9.79
  LINUCB progress:  92.0% (18400/20000) | avg reward 5.66
  LINUCB progress:  94.0% (18800/20000) | avg reward 8.61
  LINUCB progress:  92.0% (18400/20000) | avg reward 8.04
  LINUCB progress:  94.0% (18800/20000) | avg reward 9.78
  LINUCB progress:  94.0% (18800/20000) | avg reward 5.65
  LINUCB progress:  96.0% (19200/20000) | avg reward 8.65
  LINUCB progress:  94.0% (18800/20000) | avg reward 8.05
  LINUCB progress:  96.0% (19200/20000) | avg reward 9.76
  LINUCB progress:  96.0% (19200/20000) | avg reward 5.64
  LINUCB progress:  98.0% (19600/20000) | avg reward 8.66
  LINUCB progress:  96.0% (19200/20000) | avg reward 8.05
  LINUCB progress:  98.0% (19600/20000) | avg reward 9.75
  LINUCB progress:  98.0% (19600/20000) | avg reward 5.63
  LINUCB progress: 100.0% (20000/20000) | avg reward 8.69

Running Thompson Sampling with features=rich for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 0.00
  LINUCB progress:  98.0% (19600/20000) | avg reward 8.05
  LINUCB progress: 100.0% (20000/20000) | avg reward 9.76

Running Thompson Sampling with features=rich for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 7.88
  LINUCB progress: 100.0% (20000/20000) | avg reward 5.62

Running Thompson Sampling with features=simple for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 0.00
  TS progress:   2.0% (400/20000) | avg reward 5.95
  LINUCB progress: 100.0% (20000/20000) | avg reward 8.03

Running Thompson Sampling with features=rich for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 32.00
  TS progress:   2.0% (400/20000) | avg reward 9.10
  TS progress:   2.0% (400/20000) | avg reward 6.01
  TS progress:   4.0% (800/20000) | avg reward 6.17
  TS progress:   2.0% (400/20000) | avg reward 5.88
  TS progress:   4.0% (800/20000) | avg reward 8.98
  TS progress:   4.0% (800/20000) | avg reward 5.68
  TS progress:   6.0% (1200/20000) | avg reward 6.72
  TS progress:   4.0% (800/20000) | avg reward 6.75
  TS progress:   6.0% (1200/20000) | avg reward 8.25
  TS progress:   6.0% (1200/20000) | avg reward 5.54
  TS progress:   8.0% (1600/20000) | avg reward 6.82
  TS progress:   6.0% (1200/20000) | avg reward 6.80
  TS progress:   8.0% (1600/20000) | avg reward 8.13
  TS progress:   8.0% (1600/20000) | avg reward 5.83
  TS progress:  10.0% (2000/20000) | avg reward 7.13
  TS progress:   8.0% (1600/20000) | avg reward 7.37
  TS progress:  10.0% (2000/20000) | avg reward 8.03
  TS progress:  10.0% (2000/20000) | avg reward 5.78
  TS progress:  12.0% (2400/20000) | avg reward 7.29
  TS progress:  10.0% (2000/20000) | avg reward 7.70
  TS progress:  12.0% (2400/20000) | avg reward 8.09
  TS progress:  12.0% (2400/20000) | avg reward 5.67
  TS progress:  14.0% (2800/20000) | avg reward 7.45
  TS progress:  12.0% (2400/20000) | avg reward 7.93
  TS progress:  14.0% (2800/20000) | avg reward 8.05
  TS progress:  14.0% (2800/20000) | avg reward 5.56
  TS progress:  16.0% (3200/20000) | avg reward 7.51
  TS progress:  14.0% (2800/20000) | avg reward 7.95
  TS progress:  16.0% (3200/20000) | avg reward 7.82
  TS progress:  16.0% (3200/20000) | avg reward 5.46
  TS progress:  18.0% (3600/20000) | avg reward 7.45
  TS progress:  16.0% (3200/20000) | avg reward 8.16
  TS progress:  18.0% (3600/20000) | avg reward 7.66
  TS progress:  18.0% (3600/20000) | avg reward 5.48
  TS progress:  20.0% (4000/20000) | avg reward 7.44
  TS progress:  18.0% (3600/20000) | avg reward 8.32
  TS progress:  20.0% (4000/20000) | avg reward 7.76
  TS progress:  20.0% (4000/20000) | avg reward 5.51
  TS progress:  22.0% (4400/20000) | avg reward 7.51
  TS progress:  20.0% (4000/20000) | avg reward 8.09
  TS progress:  22.0% (4400/20000) | avg reward 7.85
  TS progress:  22.0% (4400/20000) | avg reward 5.45
  TS progress:  24.0% (4800/20000) | avg reward 7.60
  TS progress:  22.0% (4400/20000) | avg reward 8.07
  TS progress:  24.0% (4800/20000) | avg reward 7.81
  TS progress:  24.0% (4800/20000) | avg reward 5.57
  TS progress:  26.0% (5200/20000) | avg reward 7.53
  TS progress:  24.0% (4800/20000) | avg reward 8.08
  TS progress:  26.0% (5200/20000) | avg reward 7.82
  TS progress:  26.0% (5200/20000) | avg reward 5.65
  TS progress:  28.0% (5600/20000) | avg reward 7.50
  TS progress:  26.0% (5200/20000) | avg reward 8.00
  TS progress:  28.0% (5600/20000) | avg reward 7.77
  TS progress:  28.0% (5600/20000) | avg reward 5.68
  TS progress:  30.0% (6000/20000) | avg reward 7.50
  TS progress:  28.0% (5600/20000) | avg reward 7.96
  TS progress:  30.0% (6000/20000) | avg reward 7.87
  TS progress:  30.0% (6000/20000) | avg reward 5.74
  TS progress:  32.0% (6400/20000) | avg reward 7.45
  TS progress:  30.0% (6000/20000) | avg reward 7.90
  TS progress:  32.0% (6400/20000) | avg reward 7.87
  TS progress:  32.0% (6400/20000) | avg reward 5.82
  TS progress:  34.0% (6800/20000) | avg reward 7.55
  TS progress:  32.0% (6400/20000) | avg reward 7.86
  TS progress:  34.0% (6800/20000) | avg reward 7.81
  TS progress:  34.0% (6800/20000) | avg reward 5.88
  TS progress:  36.0% (7200/20000) | avg reward 7.57
  TS progress:  34.0% (6800/20000) | avg reward 7.84
  TS progress:  36.0% (7200/20000) | avg reward 7.76
  TS progress:  36.0% (7200/20000) | avg reward 5.85
  TS progress:  38.0% (7600/20000) | avg reward 7.60
  TS progress:  36.0% (7200/20000) | avg reward 7.77
  TS progress:  38.0% (7600/20000) | avg reward 7.72
  TS progress:  38.0% (7600/20000) | avg reward 5.88
  TS progress:  40.0% (8000/20000) | avg reward 7.58
  TS progress:  38.0% (7600/20000) | avg reward 7.79
  TS progress:  40.0% (8000/20000) | avg reward 7.74
  TS progress:  40.0% (8000/20000) | avg reward 5.89
  TS progress:  42.0% (8400/20000) | avg reward 7.55
  TS progress:  40.0% (8000/20000) | avg reward 7.80
  TS progress:  42.0% (8400/20000) | avg reward 7.72
  TS progress:  42.0% (8400/20000) | avg reward 5.89
  TS progress:  44.0% (8800/20000) | avg reward 7.54
  TS progress:  42.0% (8400/20000) | avg reward 7.85
  TS progress:  44.0% (8800/20000) | avg reward 7.70
  TS progress:  44.0% (8800/20000) | avg reward 5.92
  TS progress:  46.0% (9200/20000) | avg reward 7.53
  TS progress:  44.0% (8800/20000) | avg reward 7.82
  TS progress:  46.0% (9200/20000) | avg reward 7.66
  TS progress:  46.0% (9200/20000) | avg reward 5.91
  TS progress:  48.0% (9600/20000) | avg reward 7.57
  TS progress:  46.0% (9200/20000) | avg reward 7.81
  TS progress:  48.0% (9600/20000) | avg reward 7.70
  TS progress:  50.0% (10000/20000) | avg reward 7.56
  TS progress:  48.0% (9600/20000) | avg reward 5.90
  TS progress:  48.0% (9600/20000) | avg reward 7.80
  TS progress:  50.0% (10000/20000) | avg reward 7.71
  TS progress:  52.0% (10400/20000) | avg reward 7.55
  TS progress:  50.0% (10000/20000) | avg reward 5.93
  TS progress:  50.0% (10000/20000) | avg reward 7.77
  TS progress:  52.0% (10400/20000) | avg reward 7.70
  TS progress:  54.0% (10800/20000) | avg reward 7.57
  TS progress:  52.0% (10400/20000) | avg reward 5.93
  TS progress:  52.0% (10400/20000) | avg reward 7.75
  TS progress:  54.0% (10800/20000) | avg reward 7.80
  TS progress:  56.0% (11200/20000) | avg reward 7.60
  TS progress:  54.0% (10800/20000) | avg reward 5.93
  TS progress:  54.0% (10800/20000) | avg reward 7.79
  TS progress:  56.0% (11200/20000) | avg reward 7.80
  TS progress:  58.0% (11600/20000) | avg reward 7.65
  TS progress:  56.0% (11200/20000) | avg reward 5.92
  TS progress:  56.0% (11200/20000) | avg reward 7.81
  TS progress:  58.0% (11600/20000) | avg reward 7.80
  TS progress:  60.0% (12000/20000) | avg reward 7.71
  TS progress:  58.0% (11600/20000) | avg reward 5.92
  TS progress:  58.0% (11600/20000) | avg reward 7.81
  TS progress:  60.0% (12000/20000) | avg reward 7.80
  TS progress:  62.0% (12400/20000) | avg reward 7.71
  TS progress:  60.0% (12000/20000) | avg reward 5.97
  TS progress:  60.0% (12000/20000) | avg reward 7.81
  TS progress:  62.0% (12400/20000) | avg reward 7.74
  TS progress:  64.0% (12800/20000) | avg reward 7.73
  TS progress:  62.0% (12400/20000) | avg reward 6.03
  TS progress:  62.0% (12400/20000) | avg reward 7.79
  TS progress:  64.0% (12800/20000) | avg reward 7.71
  TS progress:  66.0% (13200/20000) | avg reward 7.76
  TS progress:  64.0% (12800/20000) | avg reward 6.10
  TS progress:  64.0% (12800/20000) | avg reward 7.84
  TS progress:  66.0% (13200/20000) | avg reward 7.74
  TS progress:  68.0% (13600/20000) | avg reward 7.77
  TS progress:  66.0% (13200/20000) | avg reward 6.14
  TS progress:  66.0% (13200/20000) | avg reward 7.83
  TS progress:  68.0% (13600/20000) | avg reward 7.66
  TS progress:  70.0% (14000/20000) | avg reward 7.77
  TS progress:  68.0% (13600/20000) | avg reward 6.18
  TS progress:  68.0% (13600/20000) | avg reward 7.77
  TS progress:  70.0% (14000/20000) | avg reward 7.68
  TS progress:  72.0% (14400/20000) | avg reward 7.77
  TS progress:  70.0% (14000/20000) | avg reward 6.26
  TS progress:  70.0% (14000/20000) | avg reward 7.75
  TS progress:  72.0% (14400/20000) | avg reward 7.73
  TS progress:  74.0% (14800/20000) | avg reward 7.77
  TS progress:  72.0% (14400/20000) | avg reward 6.27
  TS progress:  72.0% (14400/20000) | avg reward 7.74
  TS progress:  74.0% (14800/20000) | avg reward 7.72
  TS progress:  76.0% (15200/20000) | avg reward 7.77
  TS progress:  74.0% (14800/20000) | avg reward 6.34
  TS progress:  74.0% (14800/20000) | avg reward 7.72
  TS progress:  76.0% (15200/20000) | avg reward 7.69
  TS progress:  78.0% (15600/20000) | avg reward 7.79
  TS progress:  76.0% (15200/20000) | avg reward 6.39
  TS progress:  76.0% (15200/20000) | avg reward 7.72
  TS progress:  78.0% (15600/20000) | avg reward 7.65
  TS progress:  80.0% (16000/20000) | avg reward 7.82
  TS progress:  78.0% (15600/20000) | avg reward 6.44
  TS progress:  78.0% (15600/20000) | avg reward 7.76
  TS progress:  80.0% (16000/20000) | avg reward 7.64
  TS progress:  82.0% (16400/20000) | avg reward 7.85
  TS progress:  80.0% (16000/20000) | avg reward 6.48
  TS progress:  80.0% (16000/20000) | avg reward 7.78
  TS progress:  82.0% (16400/20000) | avg reward 7.63
  TS progress:  84.0% (16800/20000) | avg reward 7.88
  TS progress:  82.0% (16400/20000) | avg reward 6.48
  TS progress:  82.0% (16400/20000) | avg reward 7.78
  TS progress:  84.0% (16800/20000) | avg reward 7.65
  TS progress:  86.0% (17200/20000) | avg reward 7.89
  TS progress:  84.0% (16800/20000) | avg reward 6.51
  TS progress:  84.0% (16800/20000) | avg reward 7.78
  TS progress:  86.0% (17200/20000) | avg reward 7.64
  TS progress:  88.0% (17600/20000) | avg reward 7.91
  TS progress:  86.0% (17200/20000) | avg reward 6.53
  TS progress:  86.0% (17200/20000) | avg reward 7.79
  TS progress:  88.0% (17600/20000) | avg reward 7.65
  TS progress:  90.0% (18000/20000) | avg reward 7.91
  TS progress:  88.0% (17600/20000) | avg reward 6.54
  TS progress:  88.0% (17600/20000) | avg reward 7.77
  TS progress:  90.0% (18000/20000) | avg reward 7.64
  TS progress:  92.0% (18400/20000) | avg reward 7.91
  TS progress:  90.0% (18000/20000) | avg reward 6.58
  TS progress:  90.0% (18000/20000) | avg reward 7.77
  TS progress:  92.0% (18400/20000) | avg reward 7.66
  TS progress:  94.0% (18800/20000) | avg reward 7.90
  TS progress:  92.0% (18400/20000) | avg reward 6.60
  TS progress:  92.0% (18400/20000) | avg reward 7.76
  TS progress:  94.0% (18800/20000) | avg reward 7.66
  TS progress:  96.0% (19200/20000) | avg reward 7.88
  TS progress:  94.0% (18800/20000) | avg reward 6.61
  TS progress:  94.0% (18800/20000) | avg reward 7.77
  TS progress:  96.0% (19200/20000) | avg reward 7.70
  TS progress:  98.0% (19600/20000) | avg reward 7.87
  TS progress:  96.0% (19200/20000) | avg reward 6.64
  TS progress:  96.0% (19200/20000) | avg reward 7.75
  TS progress:  98.0% (19600/20000) | avg reward 7.69
  TS progress: 100.0% (20000/20000) | avg reward 7.87

  Static (best) segment stats:   0.0% (1/20000) | avg reward 16.74
  TS progress:  98.0% (19600/20000) | avg reward 6.67
  TS progress:  98.0% (19600/20000) | avg reward 7.75
  TS progress: 100.0% (20000/20000) | avg reward 7.70

  Static (best) segment stats:   0.0% (1/20000) | avg reward 19.94
  Static (best) segment stats:   2.0% (400/20000) | avg reward 8.30
  TS progress: 100.0% (20000/20000) | avg reward 6.69

  Static (best) segment stats:   0.0% (1/20000) | avg reward 0.00
  TS progress: 100.0% (20000/20000) | avg reward 7.74

  Static (best) segment stats:   0.0% (1/20000) | avg reward 16.54
  Static (best) segment stats:   2.0% (400/20000) | avg reward 7.18
  Static (best) segment stats:   4.0% (800/20000) | avg reward 7.13
  Static (best) segment stats:   2.0% (400/20000) | avg reward 7.84
  Static (best) segment stats:   2.0% (400/20000) | avg reward 6.08
  Static (best) segment stats:   4.0% (800/20000) | avg reward 6.81
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.08
  Static (best) segment stats:   4.0% (800/20000) | avg reward 7.25
  Static (best) segment stats:   4.0% (800/20000) | avg reward 6.76
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.37
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.16
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.07
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.10
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.49
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.25
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.12
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.39
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.64
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.22
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.20
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.35
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.80
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.29
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.11
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.10
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 8.00
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.20
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.15
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.12
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.81
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.20
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.21
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.17
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.15
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.67
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.36
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.14
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.68
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.30
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.41
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.13
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.65
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.38
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.39
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.16
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.71
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.55
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.40
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.13
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.67
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.57
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.50
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.14
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.69
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.41
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.51
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.11
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.65
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.51
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.54
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.09
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.70
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.55
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.60
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.10
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.74
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.54
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.60
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.12
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.70
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.52
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.61
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.15
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.67
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.58
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.63
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.10
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.66
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.54
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.66
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.11
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.65
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.57
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.72
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.10
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.59
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.57
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.75
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.06
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.62
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.60
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.70
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.03
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.59
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.58
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.70
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.07
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.59
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.55
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.71
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.07
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.64
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.55
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.68
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.09
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.59
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.52
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.67
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.11
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.59
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.49
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.68
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.09
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.64
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.51
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.71
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.07
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.65
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.55
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.72
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.08
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.70
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.57
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.75
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.07
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.74
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.60
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.73
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.06
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.69
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.61
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.73
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.08
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.68
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.65
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.71
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.08
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.69
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.66
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.72
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.08
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.71
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.64
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.70
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.06
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.69
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.65
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.70
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.08
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.70
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.66
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.73
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.08
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.67
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.68
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.71
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.08
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.67
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.71
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.68
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.07
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.67
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.71
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.64
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.06
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.69
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.74
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.64
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.06
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.69
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.76
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.68
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.07
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.70
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.75
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.68
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.06
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.70
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.74
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.67
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.07
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.66
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.73
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.64
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.05
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.66
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.73
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.66
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.05
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.66
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.72
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.64
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.07

✓ [1/5] Scenario 'rich_oracle_quantized' completed.

Running static template baselines for 2000 episodes...
  Static templates progress:   0.1% (1/2000) | best template ID 0 avg reward 0.00
  Static templates progress:   2.0% (40/2000) | best template ID 1 avg reward 6.31
  Static templates progress:   4.0% (80/2000) | best template ID 2 avg reward 8.12
  Static templates progress:   6.0% (120/2000) | best template ID 2 avg reward 7.58
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.65
  Static templates progress:   8.0% (160/2000) | best template ID 2 avg reward 6.61
  Static templates progress:  10.0% (200/2000) | best template ID 2 avg reward 6.45
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.75
  Static templates progress:  12.0% (240/2000) | best template ID 2 avg reward 6.37
  Static templates progress:  14.0% (280/2000) | best template ID 4 avg reward 7.01
  Static templates progress:  16.0% (320/2000) | best template ID 4 avg reward 6.76
  Static templates progress:  18.0% (360/2000) | best template ID 4 avg reward 6.95
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.65
  Static templates progress:  20.0% (400/2000) | best template ID 4 avg reward 6.78
  Static templates progress:  22.0% (440/2000) | best template ID 4 avg reward 6.70
  Static templates progress:  24.0% (480/2000) | best template ID 4 avg reward 6.92
  Static templates progress:  26.0% (520/2000) | best template ID 4 avg reward 6.79
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.67

✓ [2/5] Scenario 'rich_oracle_blend' completed.
  Static templates progress:  28.0% (560/2000) | best template ID 4 avg reward 6.77
  Static templates progress:  30.0% (600/2000) | best template ID 4 avg reward 6.51
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.75
  Static templates progress:  32.0% (640/2000) | best template ID 4 avg reward 6.86
  Static templates progress:  34.0% (680/2000) | best template ID 4 avg reward 6.72
  Static templates progress:  36.0% (720/2000) | best template ID 4 avg reward 6.74
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.65

✓ [3/5] Scenario 'simple_baseline' completed.
  Static templates progress:  38.0% (760/2000) | best template ID 4 avg reward 6.69
  Static templates progress:  40.0% (800/2000) | best template ID 4 avg reward 6.79
  Static templates progress:  42.0% (840/2000) | best template ID 4 avg reward 6.78
  Static templates progress:  44.0% (880/2000) | best template ID 4 avg reward 6.70
  Static templates progress:  46.0% (920/2000) | best template ID 4 avg reward 7.02
  Static templates progress:  48.0% (960/2000) | best template ID 4 avg reward 6.99
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.76

✓ [4/5] Scenario 'rich_oracle_raw' completed.
  Static templates progress:  50.0% (1000/2000) | best template ID 4 avg reward 7.20
  Static templates progress:  52.0% (1040/2000) | best template ID 4 avg reward 7.20
  Static templates progress:  54.0% (1080/2000) | best template ID 4 avg reward 7.38
  Static templates progress:  56.0% (1120/2000) | best template ID 4 avg reward 7.34
  Static templates progress:  58.0% (1160/2000) | best template ID 4 avg reward 7.45
  Static templates progress:  60.0% (1200/2000) | best template ID 4 avg reward 7.48
  Static templates progress:  62.0% (1240/2000) | best template ID 4 avg reward 7.38
  Static templates progress:  64.0% (1280/2000) | best template ID 4 avg reward 7.44
  Static templates progress:  66.0% (1320/2000) | best template ID 4 avg reward 7.35
  Static templates progress:  68.0% (1360/2000) | best template ID 4 avg reward 7.39
  Static templates progress:  70.0% (1400/2000) | best template ID 4 avg reward 7.34
  Static templates progress:  72.0% (1440/2000) | best template ID 4 avg reward 7.53
  Static templates progress:  74.0% (1480/2000) | best template ID 4 avg reward 7.63
  Static templates progress:  76.0% (1520/2000) | best template ID 4 avg reward 7.54
  Static templates progress:  78.0% (1560/2000) | best template ID 4 avg reward 7.55
  Static templates progress:  80.0% (1600/2000) | best template ID 4 avg reward 7.69
  Static templates progress:  82.0% (1640/2000) | best template ID 4 avg reward 7.77
  Static templates progress:  84.0% (1680/2000) | best template ID 4 avg reward 7.69
  Static templates progress:  86.0% (1720/2000) | best template ID 4 avg reward 7.64
  Static templates progress:  88.0% (1760/2000) | best template ID 4 avg reward 7.65
  Static templates progress:  90.0% (1800/2000) | best template ID 4 avg reward 7.63
  Static templates progress:  92.0% (1840/2000) | best template ID 4 avg reward 7.63
  Static templates progress:  94.0% (1880/2000) | best template ID 4 avg reward 7.55
  Static templates progress:  96.0% (1920/2000) | best template ID 4 avg reward 7.54
  Static templates progress:  98.0% (1960/2000) | best template ID 4 avg reward 7.47
  Static templates progress: 100.0% (2000/2000) | best template ID 4 avg reward 7.50


Best static template: ID=4 (Premium) with avg reward=7.50, GMV=7.04, CM2=0.79

Running LinUCB with features=rich_est for 20000 episodes...
  LINUCB progress:   0.0% (1/20000) | avg reward 32.73
  LINUCB progress:   2.0% (400/20000) | avg reward 9.24
  LINUCB progress:   4.0% (800/20000) | avg reward 9.52
  LINUCB progress:   6.0% (1200/20000) | avg reward 9.23
  LINUCB progress:   8.0% (1600/20000) | avg reward 8.95
  LINUCB progress:  10.0% (2000/20000) | avg reward 9.09
  LINUCB progress:  12.0% (2400/20000) | avg reward 9.43
  LINUCB progress:  14.0% (2800/20000) | avg reward 9.39
  LINUCB progress:  16.0% (3200/20000) | avg reward 9.32
  LINUCB progress:  18.0% (3600/20000) | avg reward 9.33
  LINUCB progress:  20.0% (4000/20000) | avg reward 9.28
  LINUCB progress:  22.0% (4400/20000) | avg reward 9.47
  LINUCB progress:  24.0% (4800/20000) | avg reward 9.39
  LINUCB progress:  26.0% (5200/20000) | avg reward 9.38
  LINUCB progress:  28.0% (5600/20000) | avg reward 9.57
  LINUCB progress:  30.0% (6000/20000) | avg reward 9.66
  LINUCB progress:  32.0% (6400/20000) | avg reward 9.71
  LINUCB progress:  34.0% (6800/20000) | avg reward 9.76
  LINUCB progress:  36.0% (7200/20000) | avg reward 9.73
  LINUCB progress:  38.0% (7600/20000) | avg reward 9.72
  LINUCB progress:  40.0% (8000/20000) | avg reward 9.78
  LINUCB progress:  42.0% (8400/20000) | avg reward 9.77
  LINUCB progress:  44.0% (8800/20000) | avg reward 9.74
  LINUCB progress:  46.0% (9200/20000) | avg reward 9.64
  LINUCB progress:  48.0% (9600/20000) | avg reward 9.64
  LINUCB progress:  50.0% (10000/20000) | avg reward 9.67
  LINUCB progress:  52.0% (10400/20000) | avg reward 9.67
  LINUCB progress:  54.0% (10800/20000) | avg reward 9.70
  LINUCB progress:  56.0% (11200/20000) | avg reward 9.70
  LINUCB progress:  58.0% (11600/20000) | avg reward 9.74
  LINUCB progress:  60.0% (12000/20000) | avg reward 9.74
  LINUCB progress:  62.0% (12400/20000) | avg reward 9.77
  LINUCB progress:  64.0% (12800/20000) | avg reward 9.73
  LINUCB progress:  66.0% (13200/20000) | avg reward 9.77
  LINUCB progress:  68.0% (13600/20000) | avg reward 9.77
  LINUCB progress:  70.0% (14000/20000) | avg reward 9.77
  LINUCB progress:  72.0% (14400/20000) | avg reward 9.75
  LINUCB progress:  74.0% (14800/20000) | avg reward 9.78
  LINUCB progress:  76.0% (15200/20000) | avg reward 9.74
  LINUCB progress:  78.0% (15600/20000) | avg reward 9.75
  LINUCB progress:  80.0% (16000/20000) | avg reward 9.75
  LINUCB progress:  82.0% (16400/20000) | avg reward 9.78
  LINUCB progress:  84.0% (16800/20000) | avg reward 9.76
  LINUCB progress:  86.0% (17200/20000) | avg reward 9.77
  LINUCB progress:  88.0% (17600/20000) | avg reward 9.78
  LINUCB progress:  90.0% (18000/20000) | avg reward 9.77
  LINUCB progress:  92.0% (18400/20000) | avg reward 9.76
  LINUCB progress:  94.0% (18800/20000) | avg reward 9.79
  LINUCB progress:  96.0% (19200/20000) | avg reward 9.80
  LINUCB progress:  98.0% (19600/20000) | avg reward 9.77
  LINUCB progress: 100.0% (20000/20000) | avg reward 9.77

Running Thompson Sampling with features=rich_est for 20000 episodes...
  TS progress:   0.0% (1/20000) | avg reward 0.00
  TS progress:   2.0% (400/20000) | avg reward 7.61
  TS progress:   4.0% (800/20000) | avg reward 7.77
  TS progress:   6.0% (1200/20000) | avg reward 8.09
  TS progress:   8.0% (1600/20000) | avg reward 8.85
  TS progress:  10.0% (2000/20000) | avg reward 9.41
  TS progress:  12.0% (2400/20000) | avg reward 9.35
  TS progress:  14.0% (2800/20000) | avg reward 9.43
  TS progress:  16.0% (3200/20000) | avg reward 9.48
  TS progress:  18.0% (3600/20000) | avg reward 9.54
  TS progress:  20.0% (4000/20000) | avg reward 9.60
  TS progress:  22.0% (4400/20000) | avg reward 9.57
  TS progress:  24.0% (4800/20000) | avg reward 9.65
  TS progress:  26.0% (5200/20000) | avg reward 9.63
  TS progress:  28.0% (5600/20000) | avg reward 9.62
  TS progress:  30.0% (6000/20000) | avg reward 9.64
  TS progress:  32.0% (6400/20000) | avg reward 9.60
  TS progress:  34.0% (6800/20000) | avg reward 9.61
  TS progress:  36.0% (7200/20000) | avg reward 9.61
  TS progress:  38.0% (7600/20000) | avg reward 9.66
  TS progress:  40.0% (8000/20000) | avg reward 9.67
  TS progress:  42.0% (8400/20000) | avg reward 9.64
  TS progress:  44.0% (8800/20000) | avg reward 9.65
  TS progress:  46.0% (9200/20000) | avg reward 9.68
  TS progress:  48.0% (9600/20000) | avg reward 9.65
  TS progress:  50.0% (10000/20000) | avg reward 9.67
  TS progress:  52.0% (10400/20000) | avg reward 9.66
  TS progress:  54.0% (10800/20000) | avg reward 9.60
  TS progress:  56.0% (11200/20000) | avg reward 9.64
  TS progress:  58.0% (11600/20000) | avg reward 9.64
  TS progress:  60.0% (12000/20000) | avg reward 9.60
  TS progress:  62.0% (12400/20000) | avg reward 9.60
  TS progress:  64.0% (12800/20000) | avg reward 9.63
  TS progress:  66.0% (13200/20000) | avg reward 9.63
  TS progress:  68.0% (13600/20000) | avg reward 9.59
  TS progress:  70.0% (14000/20000) | avg reward 9.59
  TS progress:  72.0% (14400/20000) | avg reward 9.59
  TS progress:  74.0% (14800/20000) | avg reward 9.60
  TS progress:  76.0% (15200/20000) | avg reward 9.58
  TS progress:  78.0% (15600/20000) | avg reward 9.59
  TS progress:  80.0% (16000/20000) | avg reward 9.59
  TS progress:  82.0% (16400/20000) | avg reward 9.58
  TS progress:  84.0% (16800/20000) | avg reward 9.60
  TS progress:  86.0% (17200/20000) | avg reward 9.65
  TS progress:  88.0% (17600/20000) | avg reward 9.66
  TS progress:  90.0% (18000/20000) | avg reward 9.64
  TS progress:  92.0% (18400/20000) | avg reward 9.66
  TS progress:  94.0% (18800/20000) | avg reward 9.68
  TS progress:  96.0% (19200/20000) | avg reward 9.66
  TS progress:  98.0% (19600/20000) | avg reward 9.66
  TS progress: 100.0% (20000/20000) | avg reward 9.69

  Static (best) segment stats:   0.0% (1/20000) | avg reward 0.00
  Static (best) segment stats:   2.0% (400/20000) | avg reward 7.34
  Static (best) segment stats:   4.0% (800/20000) | avg reward 7.52
  Static (best) segment stats:   6.0% (1200/20000) | avg reward 7.62
  Static (best) segment stats:   8.0% (1600/20000) | avg reward 7.51
  Static (best) segment stats:  10.0% (2000/20000) | avg reward 7.30
  Static (best) segment stats:  12.0% (2400/20000) | avg reward 7.49
  Static (best) segment stats:  14.0% (2800/20000) | avg reward 7.75
  Static (best) segment stats:  16.0% (3200/20000) | avg reward 7.57
  Static (best) segment stats:  18.0% (3600/20000) | avg reward 7.37
  Static (best) segment stats:  20.0% (4000/20000) | avg reward 7.29
  Static (best) segment stats:  22.0% (4400/20000) | avg reward 7.30
  Static (best) segment stats:  24.0% (4800/20000) | avg reward 7.20
  Static (best) segment stats:  26.0% (5200/20000) | avg reward 7.30
  Static (best) segment stats:  28.0% (5600/20000) | avg reward 7.35
  Static (best) segment stats:  30.0% (6000/20000) | avg reward 7.38
  Static (best) segment stats:  32.0% (6400/20000) | avg reward 7.48
  Static (best) segment stats:  34.0% (6800/20000) | avg reward 7.45
  Static (best) segment stats:  36.0% (7200/20000) | avg reward 7.44
  Static (best) segment stats:  38.0% (7600/20000) | avg reward 7.42
  Static (best) segment stats:  40.0% (8000/20000) | avg reward 7.37
  Static (best) segment stats:  42.0% (8400/20000) | avg reward 7.38
  Static (best) segment stats:  44.0% (8800/20000) | avg reward 7.43
  Static (best) segment stats:  46.0% (9200/20000) | avg reward 7.41
  Static (best) segment stats:  48.0% (9600/20000) | avg reward 7.43
  Static (best) segment stats:  50.0% (10000/20000) | avg reward 7.44
  Static (best) segment stats:  52.0% (10400/20000) | avg reward 7.46
  Static (best) segment stats:  54.0% (10800/20000) | avg reward 7.43
  Static (best) segment stats:  56.0% (11200/20000) | avg reward 7.46
  Static (best) segment stats:  58.0% (11600/20000) | avg reward 7.42
  Static (best) segment stats:  60.0% (12000/20000) | avg reward 7.42
  Static (best) segment stats:  62.0% (12400/20000) | avg reward 7.42
  Static (best) segment stats:  64.0% (12800/20000) | avg reward 7.45
  Static (best) segment stats:  66.0% (13200/20000) | avg reward 7.46
  Static (best) segment stats:  68.0% (13600/20000) | avg reward 7.52
  Static (best) segment stats:  70.0% (14000/20000) | avg reward 7.51
  Static (best) segment stats:  72.0% (14400/20000) | avg reward 7.51
  Static (best) segment stats:  74.0% (14800/20000) | avg reward 7.50
  Static (best) segment stats:  76.0% (15200/20000) | avg reward 7.51
  Static (best) segment stats:  78.0% (15600/20000) | avg reward 7.53
  Static (best) segment stats:  80.0% (16000/20000) | avg reward 7.51
  Static (best) segment stats:  82.0% (16400/20000) | avg reward 7.51
  Static (best) segment stats:  84.0% (16800/20000) | avg reward 7.51
  Static (best) segment stats:  86.0% (17200/20000) | avg reward 7.49
  Static (best) segment stats:  88.0% (17600/20000) | avg reward 7.50
  Static (best) segment stats:  90.0% (18000/20000) | avg reward 7.50
  Static (best) segment stats:  92.0% (18400/20000) | avg reward 7.53
  Static (best) segment stats:  94.0% (18800/20000) | avg reward 7.52
  Static (best) segment stats:  96.0% (19200/20000) | avg reward 7.53
  Static (best) segment stats:  98.0% (19600/20000) | avg reward 7.49
  Static (best) segment stats: 100.0% (20000/20000) | avg reward 7.53

✓ [5/5] Scenario 'rich_estimated' completed.
/Volumes/Lexar2T/src/rl_search_from_scratch/scripts/ch06/run_bandit_matrix.py:402: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
/Volumes/Lexar2T/src/rl_search_from_scratch/scripts/ch06/run_bandit_matrix.py:408: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",

Saved batch results to docs/book/drafts/ch06/data/verification_20251203T125142/verification_matrix_20251203T235022Z.json
✓ Test 6 completed

==============================================================================
TEST 7: Lab Solutions Module - All Exercises
==============================================================================\033[0m


======================================================================
CHAPTER 6 LAB SOLUTIONS — COMPLETE RUN
======================================================================


### THEORY EXERCISES ###

======================================================================
Exercise 6.1: Cosine Similarity Properties
======================================================================

Part (a): Boundedness verification
  Testing 1000 random vector pairs (d=16)...
  All similarities in [-1, 1]: True
  Min observed: -0.664, Max observed: 0.673

Part (b): Scale invariance verification
  Testing α=2.5, β=-3.0
  s(q, e) = 0.2224
  s(αq, βe) = -0.2224
  sign(αβ) · s(q, e) = -0.2224
  Equality holds: True (diff = 2.78e-17)

Part (c): Non-additivity counterexample
  q = [1.0, 1.0], e1 = [1.0, 0.0], e2 = [0.0, 1.0]
  e1 ⊥ e2: True (dot product = 0.0)
  s(q, e1 + e2) = 1.0000
  s(q, e1) + s(q, e2) = 1.4142
  Non-additivity demonstrated: 1.0000 ≠ 1.4142

✓ All properties verified numerically.
======================================================================
Exercise 6.2: Ridge Regression Equivalence
======================================================================

Generating synthetic regression data (n=100, d=7)...

Part (a): LinUCB vs explicit ridge regression
  LinUCB weights (A⁻¹b): [0.296, -1.021, 0.731, 0.917, -1.944, -1.271, 0.117]
  Ridge regression (closed form): [0.296, -1.021, 0.731, 0.917, -1.944, -1.271, 0.117]
  Max difference: 0.00e+00
  ✓ Equivalence verified (numerical precision)

Part (b): OLS limit as λ → 0
  λ = 1.0e+00: ‖θ_ridge - θ_ols‖ = 0.0300
  λ = 1.0e-01: ‖θ_ridge - θ_ols‖ = 0.0030
  λ = 1.0e-02: ‖θ_ridge - θ_ols‖ = 0.0003
  λ = 1.0e-03: ‖θ_ridge - θ_ols‖ = 0.0000
  λ = 1.0e-04: ‖θ_ridge - θ_ols‖ = 0.0000
  ✓ Convergence to OLS demonstrated

Part (c): Regularization and condition number
  Without regularization (λ=0): κ(ΦᵀΦ) = 2.22e+00
  With regularization (λ=1):    κ(A) = 2.20e+00
  Condition number reduced by factor: 1×

  Why this matters:
  - Ill-conditioned matrices amplify numerical errors
  - In early training, Σφφᵀ may be rank-deficient
  - Regularization λI ensures A is always invertible
  - Bounds condition number: κ(A) ≤ (λ_max + λ)/λ
======================================================================
Exercise 6.3: Thompson Sampling vs LinUCB Posterior Equivalence
======================================================================

Running 500 episodes with identical data streams...
Episode 100: Max |θ_TS - θ_LinUCB| = 0.00e+00
Episode 200: Max |θ_TS - θ_LinUCB| = 0.00e+00
Episode 300: Max |θ_TS - θ_LinUCB| = 0.00e+00
Episode 400: Max |θ_TS - θ_LinUCB| = 0.00e+00
Episode 500: Max |θ_TS - θ_LinUCB| = 0.00e+00

Final comparison (action 0):
  TS posterior mean: [-0.008, -0.087, -0.007...]
  LinUCB weights:    [-0.008, -0.087, -0.007...]
  Max difference: 0.00e+00 (numerical precision)

Final comparison (action 3):
  TS posterior mean: [0.193, -0.096, -0.086...]
  LinUCB weights:    [0.193, -0.096, -0.086...]
  Max difference: 0.00e+00 (numerical precision)

✓ Posterior means are identical to numerical precision.

Key insight:
  TS and LinUCB learn the SAME model (ridge regression).
  They differ only in HOW they use uncertainty for exploration:
  - LinUCB: Deterministic UCB bonus √(φᵀΣφ)
  - TS: Stochastic sampling from posterior

### IMPLEMENTATION EXERCISES ###

======================================================================
Exercise 6.4: ε-Greedy Baseline Implementation
======================================================================

Running experiments with n_episodes=10,000...

Training ε-greedy (ε=0.05)...
  Progress: 20% (2000/10000)
  Progress: 40% (4000/10000)
  Progress: 60% (6000/10000)
  Progress: 80% (8000/10000)
  Progress: 100% (10000/10000)

Training ε-greedy (ε=0.1)...
  Progress: 20% (2000/10000)
  Progress: 40% (4000/10000)
  Progress: 60% (6000/10000)
  Progress: 80% (8000/10000)
  Progress: 100% (10000/10000)

Training ε-greedy (ε=0.2)...
  Progress: 20% (2000/10000)
  Progress: 40% (4000/10000)
  Progress: 60% (6000/10000)
  Progress: 80% (8000/10000)
  Progress: 100% (10000/10000)

Training LinUCB (α=1.0)...
  Progress: 20% (2000/10000)
  Progress: 40% (4000/10000)
  Progress: 60% (6000/10000)
  Progress: 80% (8000/10000)
  Progress: 100% (10000/10000)

Results (average reward over last 2500 episodes):

  Policy             | Avg Reward | Cumulative Regret
  -------------------|------------|------------------
  ε-greedy (ε=0.05)  |       2.71 |             3,530
  ε-greedy (ε=0.1)   |       2.60 |             4,861
  ε-greedy (ε=0.2)   |       2.32 |             7,317
  LinUCB (α=1.0)     |       2.85 |             2,107

Regret Analysis:

  At T=10,000:
  - ε-greedy (ε=0.05): Regret ≈ 3,530 ≈ 0.35 × T (linear)
  - ε-greedy (ε=0.1): Regret ≈ 4,861 ≈ 0.49 × T (linear)
  - ε-greedy (ε=0.2): Regret ≈ 7,317 ≈ 0.73 × T (linear)
  - LinUCB: Regret ≈ 2,107 ≈ 21 × √T (sublinear)

  Theoretical prediction:
  - ε-greedy: O(εT) because exploration never stops
  - LinUCB: O(d√T log T) because uncertainty naturally decreases

✓ Demonstrated linear vs sublinear regret scaling.
======================================================================
Exercise 6.5: Cholesky-Based Thompson Sampling
======================================================================

Benchmarking naive vs Cholesky TS for varying feature dimensions...

Feature Dim | Naive Time | Cholesky Time |  Speedup
------------|------------|---------------|---------
     d=10   |     0.081s |        0.042s |     1.9×
     d=50   |     0.855s |        0.056s |    15.1×
     d=100  |     2.364s |        0.100s |    23.8×

Correctness verification (d=50):
  Same posterior mean: True (max diff = 2.22e-16)
  Sample covariance matches Σ: True (Frobenius diff = 0.0032)

Why it works:
  If A = LLᵀ (Cholesky), then Σ = A⁻¹ = L⁻ᵀL⁻¹.
  To sample θ ~ N(μ, Σ):
    z ~ N(0, I)
    θ = μ + L⁻ᵀz  [since Cov(L⁻ᵀz) = L⁻ᵀL⁻¹ = Σ]

  Cost comparison:
    Naive: O(d³) for matrix inverse per sample
    Cholesky: O(d²) for triangular solve per sample
              + O(d²) Cholesky update (amortized)

✓ Cholesky optimization provides 5-11× speedup for d ≥ 50.
======================================================================
Exercise 6.6: Category Diversity Template
======================================================================

Creating category diversity template (ID=8)...

Running LinUCB with M=9 templates (8 standard + 1 diversity)...
  Progress: 20% (2000/10000)
  Progress: 40% (4000/10000)
  Progress: 60% (6000/10000)
  Progress: 80% (8000/10000)
  Progress: 100% (10000/10000)

Template Selection Frequencies (10,000 episodes):

  Template ID | Name               | Selection % | Avg Reward
  ------------|--------------------|-------------|-----------
            0 | No Boost           |        1.9% |       1.18
            1 | High Margin        |        1.4% |       1.15
            2 | Popular            |       17.2% |       1.29
            3 | Discount           |        0.9% |       1.05
            4 | Premium            |       47.9% |       1.30
            5 | Private Label      |       28.6% |       1.29
            6 | CM2 Boost          |        0.5% |       0.88
            7 | Strategic          |        1.1% |       1.03
            8 | Category Diversity |        0.4% |       0.74

Diversity Metrics (top-10 results):

  Metric                     |   Random Baseline | Learned Policy
  ---------------------------|-------------------|---------------
  Entropy H (nats)           |              1.23 |           1.22

  ΔH ≈ 0 nats (no significant change)

⚠ Diversity template selected only 0.4% of the time.
  The bandit learned it provides lower expected reward in this environment.
  This is correct behavior—not all templates are useful in all contexts.

### EXPERIMENTAL LABS ###

======================================================================
Lab 6.1: Simple-Feature Baseline (Real Simulator)
======================================================================

Configuration:
  Static baseline episodes: 1,000
  Bandit training episodes: 10,000
  Feature mode: simple (8-dim: segment + query + bias)
  Seed: 20250319

Generated catalog with 10,000 products
User segments: ['price_hunter', 'pl_lover', 'premium', 'litter_heavy']
Query types: ['category', 'brand', 'generic']

Running static template baselines for 1000 episodes...
  Static templates progress:   0.1% (1/1000) | best template ID 2 avg reward 19.22
  Static templates progress:   2.0% (20/1000) | best template ID 2 avg reward 10.32
  Static templates progress:   4.0% (40/1000) | best template ID 2 avg reward 6.69
  Static templates progress:   6.0% (60/1000) | best template ID 4 avg reward 5.22
  Static templates progress:   8.0% (80/1000) | best template ID 4 avg reward 6.23
  Static templates progress:  10.0% (100/1000) | best template ID 4 avg reward 6.58
  Static templates progress:  12.0% (120/1000) | best template ID 2 avg reward 6.00
  Static templates progress:  14.0% (140/1000) | best template ID 4 avg reward 6.64
  Static templates progress:  16.0% (160/1000) | best template ID 4 avg reward 7.83
  Static templates progress:  18.0% (180/1000) | best template ID 4 avg reward 7.16
  Static templates progress:  20.0% (200/1000) | best template ID 4 avg reward 7.12
  Static templates progress:  22.0% (220/1000) | best template ID 4 avg reward 7.63
  Static templates progress:  24.0% (240/1000) | best template ID 4 avg reward 7.31
  Static templates progress:  26.0% (260/1000) | best template ID 4 avg reward 7.08
  Static templates progress:  28.0% (280/1000) | best template ID 4 avg reward 7.20
  Static templates progress:  30.0% (300/1000) | best template ID 4 avg reward 7.33
  Static templates progress:  32.0% (320/1000) | best template ID 4 avg reward 7.28
  Static templates progress:  34.0% (340/1000) | best template ID 4 avg reward 7.16
  Static templates progress:  36.0% (360/1000) | best template ID 4 avg reward 7.15
  Static templates progress:  38.0% (380/1000) | best template ID 4 avg reward 7.10
  Static templates progress:  40.0% (400/1000) | best template ID 4 avg reward 7.25
  Static templates progress:  42.0% (420/1000) | best template ID 4 avg reward 7.26
  Static templates progress:  44.0% (440/1000) | best template ID 4 avg reward 7.17
  Static templates progress:  46.0% (460/1000) | best template ID 4 avg reward 7.10
  Static templates progress:  48.0% (480/1000) | best template ID 4 avg reward 7.18
  Static templates progress:  50.0% (500/1000) | best template ID 4 avg reward 7.13
  Static templates progress:  52.0% (520/1000) | best template ID 4 avg reward 7.13
  Static templates progress:  54.0% (540/1000) | best template ID 4 avg reward 7.14
  Static templates progress:  56.0% (560/1000) | best template ID 4 avg reward 7.32
  Static templates progress:  58.0% (580/1000) | best template ID 4 avg reward 7.09
  Static templates progress:  60.0% (600/1000) | best template ID 4 avg reward 7.01
  Static templates progress:  62.0% (620/1000) | best template ID 4 avg reward 6.96
  Static templates progress:  64.0% (640/1000) | best template ID 4 avg reward 6.84
  Static templates progress:  66.0% (660/1000) | best template ID 4 avg reward 6.96
  Static templates progress:  68.0% (680/1000) | best template ID 4 avg reward 6.93
  Static templates progress:  70.0% (700/1000) | best template ID 4 avg reward 6.99
  Static templates progress:  72.0% (720/1000) | best template ID 4 avg reward 7.00
  Static templates progress:  74.0% (740/1000) | best template ID 4 avg reward 7.01
  Static templates progress:  76.0% (760/1000) | best template ID 4 avg reward 7.01
  Static templates progress:  78.0% (780/1000) | best template ID 4 avg reward 7.01
  Static templates progress:  80.0% (800/1000) | best template ID 4 avg reward 6.99
  Static templates progress:  82.0% (820/1000) | best template ID 4 avg reward 6.90
  Static templates progress:  84.0% (840/1000) | best template ID 4 avg reward 6.99
  Static templates progress:  86.0% (860/1000) | best template ID 4 avg reward 6.89
  Static templates progress:  88.0% (880/1000) | best template ID 4 avg reward 6.85
  Static templates progress:  90.0% (900/1000) | best template ID 4 avg reward 6.96
  Static templates progress:  92.0% (920/1000) | best template ID 4 avg reward 6.92
  Static templates progress:  94.0% (940/1000) | best template ID 4 avg reward 6.99
  Static templates progress:  96.0% (960/1000) | best template ID 4 avg reward 6.92
  Static templates progress:  98.0% (980/1000) | best template ID 4 avg reward 6.97
  Static templates progress: 100.0% (1000/1000) | best template ID 4 avg reward 6.90


Best static template: ID=4 (Premium) with avg reward=6.90, GMV=6.49, CM2=0.77

Running LinUCB with features=simple for 10000 episodes...
  LINUCB progress:   0.0% (1/10000) | avg reward 0.00
  LINUCB progress:   2.0% (200/10000) | avg reward 6.17
  LINUCB progress:   4.0% (400/10000) | avg reward 7.54
  LINUCB progress:   6.0% (600/10000) | avg reward 7.44
  LINUCB progress:   8.0% (800/10000) | avg reward 7.90
  LINUCB progress:  10.0% (1000/10000) | avg reward 7.68
  LINUCB progress:  12.0% (1200/10000) | avg reward 7.92
  LINUCB progress:  14.0% (1400/10000) | avg reward 7.56
  LINUCB progress:  16.0% (1600/10000) | avg reward 7.38
  LINUCB progress:  18.0% (1800/10000) | avg reward 7.42
  LINUCB progress:  20.0% (2000/10000) | avg reward 7.48
  LINUCB progress:  22.0% (2200/10000) | avg reward 7.59
  LINUCB progress:  24.0% (2400/10000) | avg reward 7.56
  LINUCB progress:  26.0% (2600/10000) | avg reward 7.57
  LINUCB progress:  28.0% (2800/10000) | avg reward 7.52
  LINUCB progress:  30.0% (3000/10000) | avg reward 7.48
  LINUCB progress:  32.0% (3200/10000) | avg reward 7.53
  LINUCB progress:  34.0% (3400/10000) | avg reward 7.49
  LINUCB progress:  36.0% (3600/10000) | avg reward 7.47
  LINUCB progress:  38.0% (3800/10000) | avg reward 7.47
  LINUCB progress:  40.0% (4000/10000) | avg reward 7.47
  LINUCB progress:  42.0% (4200/10000) | avg reward 7.46
  LINUCB progress:  44.0% (4400/10000) | avg reward 7.44
  LINUCB progress:  46.0% (4600/10000) | avg reward 7.48
  LINUCB progress:  48.0% (4800/10000) | avg reward 7.40
  LINUCB progress:  50.0% (5000/10000) | avg reward 7.35
  LINUCB progress:  52.0% (5200/10000) | avg reward 7.24
  LINUCB progress:  54.0% (5400/10000) | avg reward 7.27
  LINUCB progress:  56.0% (5600/10000) | avg reward 7.33
  LINUCB progress:  58.0% (5800/10000) | avg reward 7.42
  LINUCB progress:  60.0% (6000/10000) | avg reward 7.47
  LINUCB progress:  62.0% (6200/10000) | avg reward 7.48
  LINUCB progress:  64.0% (6400/10000) | avg reward 7.43
  LINUCB progress:  66.0% (6600/10000) | avg reward 7.41
  LINUCB progress:  68.0% (6800/10000) | avg reward 7.40
  LINUCB progress:  70.0% (7000/10000) | avg reward 7.41
  LINUCB progress:  72.0% (7200/10000) | avg reward 7.47
  LINUCB progress:  74.0% (7400/10000) | avg reward 7.47
  LINUCB progress:  76.0% (7600/10000) | avg reward 7.43
  LINUCB progress:  78.0% (7800/10000) | avg reward 7.44
  LINUCB progress:  80.0% (8000/10000) | avg reward 7.50
  LINUCB progress:  82.0% (8200/10000) | avg reward 7.51
  LINUCB progress:  84.0% (8400/10000) | avg reward 7.55
  LINUCB progress:  86.0% (8600/10000) | avg reward 7.59
  LINUCB progress:  88.0% (8800/10000) | avg reward 7.64
  LINUCB progress:  90.0% (9000/10000) | avg reward 7.64
  LINUCB progress:  92.0% (9200/10000) | avg reward 7.66
  LINUCB progress:  94.0% (9400/10000) | avg reward 7.63
  LINUCB progress:  96.0% (9600/10000) | avg reward 7.61
  LINUCB progress:  98.0% (9800/10000) | avg reward 7.61
  LINUCB progress: 100.0% (10000/10000) | avg reward 7.62

Running Thompson Sampling with features=simple for 10000 episodes...
  TS progress:   0.0% (1/10000) | avg reward 0.00
  TS progress:   2.0% (200/10000) | avg reward 7.05
  TS progress:   4.0% (400/10000) | avg reward 6.82
  TS progress:   6.0% (600/10000) | avg reward 6.47
  TS progress:   8.0% (800/10000) | avg reward 6.13
  TS progress:  10.0% (1000/10000) | avg reward 6.21
  TS progress:  12.0% (1200/10000) | avg reward 6.30
  TS progress:  14.0% (1400/10000) | avg reward 6.17
  TS progress:  16.0% (1600/10000) | avg reward 6.45
  TS progress:  18.0% (1800/10000) | avg reward 6.37
  TS progress:  20.0% (2000/10000) | avg reward 6.43
  TS progress:  22.0% (2200/10000) | avg reward 6.48
  TS progress:  24.0% (2400/10000) | avg reward 6.52
  TS progress:  26.0% (2600/10000) | avg reward 6.55
  TS progress:  28.0% (2800/10000) | avg reward 6.55
  TS progress:  30.0% (3000/10000) | avg reward 6.61
  TS progress:  32.0% (3200/10000) | avg reward 6.61
  TS progress:  34.0% (3400/10000) | avg reward 6.59
  TS progress:  36.0% (3600/10000) | avg reward 6.60
  TS progress:  38.0% (3800/10000) | avg reward 6.59
  TS progress:  40.0% (4000/10000) | avg reward 6.57
  TS progress:  42.0% (4200/10000) | avg reward 6.54
  TS progress:  44.0% (4400/10000) | avg reward 6.51
  TS progress:  46.0% (4600/10000) | avg reward 6.52
  TS progress:  48.0% (4800/10000) | avg reward 6.51
  TS progress:  50.0% (5000/10000) | avg reward 6.50
  TS progress:  52.0% (5200/10000) | avg reward 6.55
  TS progress:  54.0% (5400/10000) | avg reward 6.63
  TS progress:  56.0% (5600/10000) | avg reward 6.62
  TS progress:  58.0% (5800/10000) | avg reward 6.60
  TS progress:  60.0% (6000/10000) | avg reward 6.60
  TS progress:  62.0% (6200/10000) | avg reward 6.62
  TS progress:  64.0% (6400/10000) | avg reward 6.64
  TS progress:  66.0% (6600/10000) | avg reward 6.63
  TS progress:  68.0% (6800/10000) | avg reward 6.59
  TS progress:  70.0% (7000/10000) | avg reward 6.62
  TS progress:  72.0% (7200/10000) | avg reward 6.64
  TS progress:  74.0% (7400/10000) | avg reward 6.63
  TS progress:  76.0% (7600/10000) | avg reward 6.61
  TS progress:  78.0% (7800/10000) | avg reward 6.63
  TS progress:  80.0% (8000/10000) | avg reward 6.61
  TS progress:  82.0% (8200/10000) | avg reward 6.62
  TS progress:  84.0% (8400/10000) | avg reward 6.58
  TS progress:  86.0% (8600/10000) | avg reward 6.58
  TS progress:  88.0% (8800/10000) | avg reward 6.60
  TS progress:  90.0% (9000/10000) | avg reward 6.63
  TS progress:  92.0% (9200/10000) | avg reward 6.67
  TS progress:  94.0% (9400/10000) | avg reward 6.66
  TS progress:  96.0% (9600/10000) | avg reward 6.67
  TS progress:  98.0% (9800/10000) | avg reward 6.70
  TS progress: 100.0% (10000/10000) | avg reward 6.70

  Static (best) segment stats:   0.0% (1/10000) | avg reward 0.00
  Static (best) segment stats:   2.0% (200/10000) | avg reward 6.46
  Static (best) segment stats:   4.0% (400/10000) | avg reward 6.55
  Static (best) segment stats:   6.0% (600/10000) | avg reward 7.07
  Static (best) segment stats:   8.0% (800/10000) | avg reward 6.47
  Static (best) segment stats:  10.0% (1000/10000) | avg reward 6.40
  Static (best) segment stats:  12.0% (1200/10000) | avg reward 6.33
  Static (best) segment stats:  14.0% (1400/10000) | avg reward 6.63
  Static (best) segment stats:  16.0% (1600/10000) | avg reward 6.54
  Static (best) segment stats:  18.0% (1800/10000) | avg reward 6.38
  Static (best) segment stats:  20.0% (2000/10000) | avg reward 6.33
  Static (best) segment stats:  22.0% (2200/10000) | avg reward 6.50
  Static (best) segment stats:  24.0% (2400/10000) | avg reward 6.82
  Static (best) segment stats:  26.0% (2600/10000) | avg reward 6.79
  Static (best) segment stats:  28.0% (2800/10000) | avg reward 6.75
  Static (best) segment stats:  30.0% (3000/10000) | avg reward 6.66
  Static (best) segment stats:  32.0% (3200/10000) | avg reward 6.81
  Static (best) segment stats:  34.0% (3400/10000) | avg reward 6.80
  Static (best) segment stats:  36.0% (3600/10000) | avg reward 6.86
  Static (best) segment stats:  38.0% (3800/10000) | avg reward 6.80
  Static (best) segment stats:  40.0% (4000/10000) | avg reward 6.83
  Static (best) segment stats:  42.0% (4200/10000) | avg reward 6.74
  Static (best) segment stats:  44.0% (4400/10000) | avg reward 6.86
  Static (best) segment stats:  46.0% (4600/10000) | avg reward 6.91
  Static (best) segment stats:  48.0% (4800/10000) | avg reward 6.90
  Static (best) segment stats:  50.0% (5000/10000) | avg reward 6.93
  Static (best) segment stats:  52.0% (5200/10000) | avg reward 6.89
  Static (best) segment stats:  54.0% (5400/10000) | avg reward 6.89
  Static (best) segment stats:  56.0% (5600/10000) | avg reward 6.89
  Static (best) segment stats:  58.0% (5800/10000) | avg reward 6.85
  Static (best) segment stats:  60.0% (6000/10000) | avg reward 6.88
  Static (best) segment stats:  62.0% (6200/10000) | avg reward 6.92
  Static (best) segment stats:  64.0% (6400/10000) | avg reward 6.84
  Static (best) segment stats:  66.0% (6600/10000) | avg reward 6.87
  Static (best) segment stats:  68.0% (6800/10000) | avg reward 6.88
  Static (best) segment stats:  70.0% (7000/10000) | avg reward 6.96
  Static (best) segment stats:  72.0% (7200/10000) | avg reward 6.91
  Static (best) segment stats:  74.0% (7400/10000) | avg reward 6.91
  Static (best) segment stats:  76.0% (7600/10000) | avg reward 6.92
  Static (best) segment stats:  78.0% (7800/10000) | avg reward 6.94
  Static (best) segment stats:  80.0% (8000/10000) | avg reward 6.94
  Static (best) segment stats:  82.0% (8200/10000) | avg reward 6.94
  Static (best) segment stats:  84.0% (8400/10000) | avg reward 6.90
  Static (best) segment stats:  86.0% (8600/10000) | avg reward 6.93
  Static (best) segment stats:  88.0% (8800/10000) | avg reward 6.95
  Static (best) segment stats:  90.0% (9000/10000) | avg reward 6.91
  Static (best) segment stats:  92.0% (9200/10000) | avg reward 6.89
  Static (best) segment stats:  94.0% (9400/10000) | avg reward 6.94
  Static (best) segment stats:  96.0% (9600/10000) | avg reward 6.96
  Static (best) segment stats:  98.0% (9800/10000) | avg reward 6.96
  Static (best) segment stats: 100.0% (10000/10000) | avg reward 6.96


======================================================================
RESULTS (Simple Features)
======================================================================

Best static template: Premium
  GMV = 6.49

LinUCB:
  GMV = 7.16 (+10.4% vs static)

Thompson Sampling:
  GMV = 6.15 (-5.2% vs static)

Template selection frequencies:
  Template        |   LinUCB |       TS
  ----------------|----------|---------
  Neutral         |     4.7% |     4.6%
  High Margin     |    11.6% |     4.4%
  CM2 Boost       |     4.6% |    65.7%
  Popular         |     4.5% |     4.5%
  Premium         |    59.6% |     6.8%
  Budget          |     5.9% |     5.0%
  Discount        |     4.5% |     4.6%
  Strategic       |     4.5% |     4.4%

Conclusion:
  With simple features, bandits perform better than static.
  The gap is modest (+10.4% / -5.2%), suggesting simple features
  don't capture enough context variance for dramatic improvement.
======================================================================
Lab 6.2: Rich-Feature Improvement (Real Simulator)
======================================================================

Configuration:
  Static baseline episodes: 1,000
  Bandit training episodes: 10,000
  Feature mode: rich (18-dim: segment + query + latents + aggregates + bias)
  Regularization: blend
  Seed: 20250319

Generated catalog with 10,000 products

Hyperparameters:
  Prior weight: 6
  LinUCB α: 1.0
  TS σ: 1.0

Running static template baselines for 1000 episodes...
  Static templates progress:   0.1% (1/1000) | best template ID 2 avg reward 19.22
  Static templates progress:   2.0% (20/1000) | best template ID 2 avg reward 10.32
  Static templates progress:   4.0% (40/1000) | best template ID 2 avg reward 6.69
  Static templates progress:   6.0% (60/1000) | best template ID 4 avg reward 5.22
  Static templates progress:   8.0% (80/1000) | best template ID 4 avg reward 6.23
  Static templates progress:  10.0% (100/1000) | best template ID 4 avg reward 6.58
  Static templates progress:  12.0% (120/1000) | best template ID 2 avg reward 6.00
  Static templates progress:  14.0% (140/1000) | best template ID 4 avg reward 6.64
  Static templates progress:  16.0% (160/1000) | best template ID 4 avg reward 7.83
  Static templates progress:  18.0% (180/1000) | best template ID 4 avg reward 7.16
  Static templates progress:  20.0% (200/1000) | best template ID 4 avg reward 7.12
  Static templates progress:  22.0% (220/1000) | best template ID 4 avg reward 7.63
  Static templates progress:  24.0% (240/1000) | best template ID 4 avg reward 7.31
  Static templates progress:  26.0% (260/1000) | best template ID 4 avg reward 7.08
  Static templates progress:  28.0% (280/1000) | best template ID 4 avg reward 7.20
  Static templates progress:  30.0% (300/1000) | best template ID 4 avg reward 7.33
  Static templates progress:  32.0% (320/1000) | best template ID 4 avg reward 7.28
  Static templates progress:  34.0% (340/1000) | best template ID 4 avg reward 7.16
  Static templates progress:  36.0% (360/1000) | best template ID 4 avg reward 7.15
  Static templates progress:  38.0% (380/1000) | best template ID 4 avg reward 7.10
  Static templates progress:  40.0% (400/1000) | best template ID 4 avg reward 7.25
  Static templates progress:  42.0% (420/1000) | best template ID 4 avg reward 7.26
  Static templates progress:  44.0% (440/1000) | best template ID 4 avg reward 7.17
  Static templates progress:  46.0% (460/1000) | best template ID 4 avg reward 7.10
  Static templates progress:  48.0% (480/1000) | best template ID 4 avg reward 7.18
  Static templates progress:  50.0% (500/1000) | best template ID 4 avg reward 7.13
  Static templates progress:  52.0% (520/1000) | best template ID 4 avg reward 7.13
  Static templates progress:  54.0% (540/1000) | best template ID 4 avg reward 7.14
  Static templates progress:  56.0% (560/1000) | best template ID 4 avg reward 7.32
  Static templates progress:  58.0% (580/1000) | best template ID 4 avg reward 7.09
  Static templates progress:  60.0% (600/1000) | best template ID 4 avg reward 7.01
  Static templates progress:  62.0% (620/1000) | best template ID 4 avg reward 6.96
  Static templates progress:  64.0% (640/1000) | best template ID 4 avg reward 6.84
  Static templates progress:  66.0% (660/1000) | best template ID 4 avg reward 6.96
  Static templates progress:  68.0% (680/1000) | best template ID 4 avg reward 6.93
  Static templates progress:  70.0% (700/1000) | best template ID 4 avg reward 6.99
  Static templates progress:  72.0% (720/1000) | best template ID 4 avg reward 7.00
  Static templates progress:  74.0% (740/1000) | best template ID 4 avg reward 7.01
  Static templates progress:  76.0% (760/1000) | best template ID 4 avg reward 7.01
  Static templates progress:  78.0% (780/1000) | best template ID 4 avg reward 7.01
  Static templates progress:  80.0% (800/1000) | best template ID 4 avg reward 6.99
  Static templates progress:  82.0% (820/1000) | best template ID 4 avg reward 6.90
  Static templates progress:  84.0% (840/1000) | best template ID 4 avg reward 6.99
  Static templates progress:  86.0% (860/1000) | best template ID 4 avg reward 6.89
  Static templates progress:  88.0% (880/1000) | best template ID 4 avg reward 6.85
  Static templates progress:  90.0% (900/1000) | best template ID 4 avg reward 6.96
  Static templates progress:  92.0% (920/1000) | best template ID 4 avg reward 6.92
  Static templates progress:  94.0% (940/1000) | best template ID 4 avg reward 6.99
  Static templates progress:  96.0% (960/1000) | best template ID 4 avg reward 6.92
  Static templates progress:  98.0% (980/1000) | best template ID 4 avg reward 6.97
  Static templates progress: 100.0% (1000/1000) | best template ID 4 avg reward 6.90


Best static template: ID=4 (Premium) with avg reward=6.90, GMV=6.49, CM2=0.77

Running LinUCB with features=rich for 10000 episodes...
  LINUCB progress:   0.0% (1/10000) | avg reward 0.10
  LINUCB progress:   2.0% (200/10000) | avg reward 6.45
  LINUCB progress:   4.0% (400/10000) | avg reward 6.16
  LINUCB progress:   6.0% (600/10000) | avg reward 6.18
  LINUCB progress:   8.0% (800/10000) | avg reward 6.52
  LINUCB progress:  10.0% (1000/10000) | avg reward 6.69
  LINUCB progress:  12.0% (1200/10000) | avg reward 7.15
  LINUCB progress:  14.0% (1400/10000) | avg reward 7.15
  LINUCB progress:  16.0% (1600/10000) | avg reward 7.28
  LINUCB progress:  18.0% (1800/10000) | avg reward 7.25
  LINUCB progress:  20.0% (2000/10000) | avg reward 7.21
  LINUCB progress:  22.0% (2200/10000) | avg reward 7.33
  LINUCB progress:  24.0% (2400/10000) | avg reward 7.29
  LINUCB progress:  26.0% (2600/10000) | avg reward 7.38
  LINUCB progress:  28.0% (2800/10000) | avg reward 7.42
  LINUCB progress:  30.0% (3000/10000) | avg reward 7.36
  LINUCB progress:  32.0% (3200/10000) | avg reward 7.39
  LINUCB progress:  34.0% (3400/10000) | avg reward 7.34
  LINUCB progress:  36.0% (3600/10000) | avg reward 7.36
  LINUCB progress:  38.0% (3800/10000) | avg reward 7.40
  LINUCB progress:  40.0% (4000/10000) | avg reward 7.39
  LINUCB progress:  42.0% (4200/10000) | avg reward 7.48
  LINUCB progress:  44.0% (4400/10000) | avg reward 7.50
  LINUCB progress:  46.0% (4600/10000) | avg reward 7.45
  LINUCB progress:  48.0% (4800/10000) | avg reward 7.49
  LINUCB progress:  50.0% (5000/10000) | avg reward 7.55
  LINUCB progress:  52.0% (5200/10000) | avg reward 7.50
  LINUCB progress:  54.0% (5400/10000) | avg reward 7.59
  LINUCB progress:  56.0% (5600/10000) | avg reward 7.54
  LINUCB progress:  58.0% (5800/10000) | avg reward 7.54
  LINUCB progress:  60.0% (6000/10000) | avg reward 7.52
  LINUCB progress:  62.0% (6200/10000) | avg reward 7.48
  LINUCB progress:  64.0% (6400/10000) | avg reward 7.47
  LINUCB progress:  66.0% (6600/10000) | avg reward 7.42
  LINUCB progress:  68.0% (6800/10000) | avg reward 7.45
  LINUCB progress:  70.0% (7000/10000) | avg reward 7.55
  LINUCB progress:  72.0% (7200/10000) | avg reward 7.51
  LINUCB progress:  74.0% (7400/10000) | avg reward 7.53
  LINUCB progress:  76.0% (7600/10000) | avg reward 7.49
  LINUCB progress:  78.0% (7800/10000) | avg reward 7.50
  LINUCB progress:  80.0% (8000/10000) | avg reward 7.56
  LINUCB progress:  82.0% (8200/10000) | avg reward 7.58
  LINUCB progress:  84.0% (8400/10000) | avg reward 7.56
  LINUCB progress:  86.0% (8600/10000) | avg reward 7.58
  LINUCB progress:  88.0% (8800/10000) | avg reward 7.58
  LINUCB progress:  90.0% (9000/10000) | avg reward 7.58
  LINUCB progress:  92.0% (9200/10000) | avg reward 7.54
  LINUCB progress:  94.0% (9400/10000) | avg reward 7.57
  LINUCB progress:  96.0% (9600/10000) | avg reward 7.57
  LINUCB progress:  98.0% (9800/10000) | avg reward 7.58
  LINUCB progress: 100.0% (10000/10000) | avg reward 7.57

Running Thompson Sampling with features=rich for 10000 episodes...
  TS progress:   0.0% (1/10000) | avg reward 0.00
  TS progress:   2.0% (200/10000) | avg reward 4.58
  TS progress:   4.0% (400/10000) | avg reward 4.89
  TS progress:   6.0% (600/10000) | avg reward 5.46
  TS progress:   8.0% (800/10000) | avg reward 5.74
  TS progress:  10.0% (1000/10000) | avg reward 6.01
  TS progress:  12.0% (1200/10000) | avg reward 6.30
  TS progress:  14.0% (1400/10000) | avg reward 6.05
  TS progress:  16.0% (1600/10000) | avg reward 6.08
  TS progress:  18.0% (1800/10000) | avg reward 6.14
  TS progress:  20.0% (2000/10000) | avg reward 6.30
  TS progress:  22.0% (2200/10000) | avg reward 6.36
  TS progress:  24.0% (2400/10000) | avg reward 6.43
  TS progress:  26.0% (2600/10000) | avg reward 6.44
  TS progress:  28.0% (2800/10000) | avg reward 6.47
  TS progress:  30.0% (3000/10000) | avg reward 6.53
  TS progress:  32.0% (3200/10000) | avg reward 6.52
  TS progress:  34.0% (3400/10000) | avg reward 6.53
  TS progress:  36.0% (3600/10000) | avg reward 6.51
  TS progress:  38.0% (3800/10000) | avg reward 6.55
  TS progress:  40.0% (4000/10000) | avg reward 6.54
  TS progress:  42.0% (4200/10000) | avg reward 6.53
  TS progress:  44.0% (4400/10000) | avg reward 6.44
  TS progress:  46.0% (4600/10000) | avg reward 6.44
  TS progress:  48.0% (4800/10000) | avg reward 6.45
  TS progress:  50.0% (5000/10000) | avg reward 6.41
  TS progress:  52.0% (5200/10000) | avg reward 6.47
  TS progress:  54.0% (5400/10000) | avg reward 6.50
  TS progress:  56.0% (5600/10000) | avg reward 6.49
  TS progress:  58.0% (5800/10000) | avg reward 6.53
  TS progress:  60.0% (6000/10000) | avg reward 6.57
  TS progress:  62.0% (6200/10000) | avg reward 6.58
  TS progress:  64.0% (6400/10000) | avg reward 6.59
  TS progress:  66.0% (6600/10000) | avg reward 6.61
  TS progress:  68.0% (6800/10000) | avg reward 6.63
  TS progress:  70.0% (7000/10000) | avg reward 6.66
  TS progress:  72.0% (7200/10000) | avg reward 6.65
  TS progress:  74.0% (7400/10000) | avg reward 6.65
  TS progress:  76.0% (7600/10000) | avg reward 6.65
  TS progress:  78.0% (7800/10000) | avg reward 6.67
  TS progress:  80.0% (8000/10000) | avg reward 6.70
  TS progress:  82.0% (8200/10000) | avg reward 6.72
  TS progress:  84.0% (8400/10000) | avg reward 6.73
  TS progress:  86.0% (8600/10000) | avg reward 6.77
  TS progress:  88.0% (8800/10000) | avg reward 6.80
  TS progress:  90.0% (9000/10000) | avg reward 6.82
  TS progress:  92.0% (9200/10000) | avg reward 6.82
  TS progress:  94.0% (9400/10000) | avg reward 6.82
  TS progress:  96.0% (9600/10000) | avg reward 6.88
  TS progress:  98.0% (9800/10000) | avg reward 6.93
  TS progress: 100.0% (10000/10000) | avg reward 6.95

  Static (best) segment stats:   0.0% (1/10000) | avg reward 0.00
  Static (best) segment stats:   2.0% (200/10000) | avg reward 6.46
  Static (best) segment stats:   4.0% (400/10000) | avg reward 6.55
  Static (best) segment stats:   6.0% (600/10000) | avg reward 7.07
  Static (best) segment stats:   8.0% (800/10000) | avg reward 6.47
  Static (best) segment stats:  10.0% (1000/10000) | avg reward 6.40
  Static (best) segment stats:  12.0% (1200/10000) | avg reward 6.33
  Static (best) segment stats:  14.0% (1400/10000) | avg reward 6.63
  Static (best) segment stats:  16.0% (1600/10000) | avg reward 6.54
  Static (best) segment stats:  18.0% (1800/10000) | avg reward 6.38
  Static (best) segment stats:  20.0% (2000/10000) | avg reward 6.33
  Static (best) segment stats:  22.0% (2200/10000) | avg reward 6.50
  Static (best) segment stats:  24.0% (2400/10000) | avg reward 6.82
  Static (best) segment stats:  26.0% (2600/10000) | avg reward 6.79
  Static (best) segment stats:  28.0% (2800/10000) | avg reward 6.75
  Static (best) segment stats:  30.0% (3000/10000) | avg reward 6.66
  Static (best) segment stats:  32.0% (3200/10000) | avg reward 6.81
  Static (best) segment stats:  34.0% (3400/10000) | avg reward 6.80
  Static (best) segment stats:  36.0% (3600/10000) | avg reward 6.86
  Static (best) segment stats:  38.0% (3800/10000) | avg reward 6.80
  Static (best) segment stats:  40.0% (4000/10000) | avg reward 6.83
  Static (best) segment stats:  42.0% (4200/10000) | avg reward 6.74
  Static (best) segment stats:  44.0% (4400/10000) | avg reward 6.86
  Static (best) segment stats:  46.0% (4600/10000) | avg reward 6.91
  Static (best) segment stats:  48.0% (4800/10000) | avg reward 6.90
  Static (best) segment stats:  50.0% (5000/10000) | avg reward 6.93
  Static (best) segment stats:  52.0% (5200/10000) | avg reward 6.89
  Static (best) segment stats:  54.0% (5400/10000) | avg reward 6.89
  Static (best) segment stats:  56.0% (5600/10000) | avg reward 6.89
  Static (best) segment stats:  58.0% (5800/10000) | avg reward 6.85
  Static (best) segment stats:  60.0% (6000/10000) | avg reward 6.88
  Static (best) segment stats:  62.0% (6200/10000) | avg reward 6.92
  Static (best) segment stats:  64.0% (6400/10000) | avg reward 6.84
  Static (best) segment stats:  66.0% (6600/10000) | avg reward 6.87
  Static (best) segment stats:  68.0% (6800/10000) | avg reward 6.88
  Static (best) segment stats:  70.0% (7000/10000) | avg reward 6.96
  Static (best) segment stats:  72.0% (7200/10000) | avg reward 6.91
  Static (best) segment stats:  74.0% (7400/10000) | avg reward 6.91
  Static (best) segment stats:  76.0% (7600/10000) | avg reward 6.92
  Static (best) segment stats:  78.0% (7800/10000) | avg reward 6.94
  Static (best) segment stats:  80.0% (8000/10000) | avg reward 6.94
  Static (best) segment stats:  82.0% (8200/10000) | avg reward 6.94
  Static (best) segment stats:  84.0% (8400/10000) | avg reward 6.90
  Static (best) segment stats:  86.0% (8600/10000) | avg reward 6.93
  Static (best) segment stats:  88.0% (8800/10000) | avg reward 6.95
  Static (best) segment stats:  90.0% (9000/10000) | avg reward 6.91
  Static (best) segment stats:  92.0% (9200/10000) | avg reward 6.89
  Static (best) segment stats:  94.0% (9400/10000) | avg reward 6.94
  Static (best) segment stats:  96.0% (9600/10000) | avg reward 6.96
  Static (best) segment stats:  98.0% (9800/10000) | avg reward 6.96
  Static (best) segment stats: 100.0% (10000/10000) | avg reward 6.96


======================================================================
RESULTS (Rich Features)
======================================================================

Best static template: Premium
  GMV = 6.49

LinUCB:
  GMV = 6.96 (+7.3% vs static)

Thompson Sampling:
  GMV = 6.36 (-2.0% vs static)

Per-segment GMV breakdown:
  Segment         |   Static |   LinUCB |       TS |   Lin Δ% |    TS Δ%
  ----------------|----------|----------|----------|----------|---------
  price_hunter    |     0.00 |     0.00 |     0.00 |    +0.0% |    +0.0%
  pl_lover        |     5.01 |    10.77 |    10.31 |  +115.2% |  +106.0%
  litter_heavy    |     4.03 |     6.00 |     6.05 |   +49.1% |   +50.3%
  premium         |    30.49 |    18.74 |    15.48 |   -38.5% |   -49.2%

Conclusion:
  Rich features enable LinUCB to achieve +7.3% GMV improvement!
  This demonstrates the value of capturing user preference signals
  and product aggregates in the context features.
======================================================================
Lab 6.3: Hyperparameter Sensitivity Analysis
======================================================================

Grid: λ ∈ [0.1, 1.0, 10.0] × α ∈ [0.5, 1.0, 2.0]
Episodes per config: 5,000

Results (average reward, last 1,250 episodes):

         | α=0.5  | α=1.0  | α=2.0 
-----------------------------------
λ=0.1   |  0.52 |  0.49 |  0.50
λ=1.0   |  0.51 |  0.48 |  0.48
λ=10.0  |  0.49 |  0.47 |  0.50

Best: λ=0.1, α=0.5 → 0.52

Insights:
  - Higher λ provides stronger regularization (prevents overfitting)
  - Higher α increases exploration (helps with uncertain arms)
  - Optimal tradeoff depends on problem structure and horizon
======================================================================
Lab 6.4: Exploration Dynamics Visualization
======================================================================

Episode 100:
  Avg reward (last 500): 0.607
  Total uncertainty: 3.472
  Selection freq: [0.27 0.26 0.27 0.2 ]

Episode 500:
  Avg reward (last 500): 0.870
  Total uncertainty: 1.978
  Selection freq: [0.246 0.264 0.262 0.228]

Episode 1,000:
  Avg reward (last 500): 0.982
  Total uncertainty: 1.683
  Selection freq: [0.27  0.266 0.238 0.226]

Episode 2,000:
  Avg reward (last 500): 0.988
  Total uncertainty: 1.455
  Selection freq: [0.246 0.23  0.258 0.266]

Episode 5,000:
  Avg reward (last 500): 0.992
  Total uncertainty: 1.254
  Selection freq: [0.3   0.22  0.276 0.204]

======================================================================
Summary
======================================================================

Uncertainty reduction: 15.50 → 1.25
Reduction ratio: 12.4x

Final selection distribution (last 1000):
  Arm 0: 27.4%
  Arm 1: 23.2%
  Arm 2: 26.9%
  Arm 3: 22.5%
======================================================================
Lab 6.5: Multi-Seed Robustness Analysis
======================================================================

Configuration:
  Seeds: 5
  Episodes per seed: 5,000
  Seed 42: 0.478
  Seed 1042: 0.685
  Seed 2042: 0.446
  Seed 3042: 0.457
  Seed 4042: 0.436

======================================================================
Statistics
======================================================================
  Mean: 0.500
  Std:  0.093
  CV:   18.7%
  Range: [0.436, 0.685]

Conclusion:
  High variance (CV=18.7%) suggests sensitivity to initialization.

### ADVANCED EXERCISES ###

======================================================================
Exercise 6.7: Hierarchical Templates
======================================================================

Hierarchical structure:
  Level 1 (Meta): 3 objectives
    - Objective A: Maximize margin (templates: HiMargin, Premium, CM2)
    - Objective B: Maximize volume (templates: Popular, Discount, Budget)
    - Objective C: Strategic goals (templates: Strategic, Diversity)

  Level 2 (Sub): 2-3 templates per objective

Training hierarchical bandit for 20,000 episodes...
  Progress: 20% (4000/20000)
  Progress: 40% (8000/20000)
  Progress: 60% (12000/20000)
  Progress: 80% (16000/20000)
  Progress: 100% (20000/20000)

Meta-Level Selection Distribution:

  Objective    | Selection %
  -------------|------------
  Margin (A)   |       21.1%
  Volume (B)   |       25.2%
  Strategic(C) |       53.7%

Sub-Level Selection (within objectives):

  Margin (A):
    HiMargin     |        7.0%
    Premium      |        1.4%
    CM2Boost     |       91.7%

  Volume (B):
    Popular      |       74.8%
    Discount     |        1.7%
    Budget       |       23.5%

  Strategic(C):
    Strategic    |       47.1%
    Diversity    |       52.9%

Comparison to Flat LinUCB:

  Policy           | Convergence (ep)
  -----------------|-----------------
  Flat LinUCB      |          ~12,000
  Hierarchical     |           ~8,000

Convergence speedup: 33% faster (8k vs 12k episodes)

✓ Hierarchical bandits converge faster with similar final performance.
======================================================================
Exercise 6.9: Query-Conditional Templates
======================================================================

Query-conditional template design:
  t(p, q) = w_base · f(p) + w_query · g(q, p)

  where g(q, p) captures query-product interaction

Training comparison (15,000 episodes):

  Policy                   | Final Reward
  -------------------------|-------------
  Product-only templates   |         0.81
  Query-conditional        |         0.98

  ΔReward: +0.16 (+20.2%)

Insight:
  Query-conditional templates learn to AMPLIFY templates when
  query content suggests they'll be effective, and SUPPRESS
  templates when query content suggests they'll hurt.

  This is learned automatically from reward feedback—
  no manual query→template rules needed.

✓ Query-conditional templates achieve +20.2% improvement.

======================================================================
ALL EXERCISES COMPLETED
======================================================================
✓ Test 7 completed

==============================================================================
VERIFICATION COMPLETE
==============================================================================\033[0m

End time: Thu Dec  4 01:38:20 CET 2025

Output files saved to: docs/book/drafts/ch06/data/verification_20251203T125142

Key files to review:
-rw-r--r--  1 vladyslavprytula  staff  16633 Dec  3 13:39 docs/book/drafts/ch06/data/verification_20251203T125142/test1_simple_features.log
-rw-r--r--  1 vladyslavprytula  staff  16779 Dec  3 14:27 docs/book/drafts/ch06/data/verification_20251203T125142/test2_rich_blend.log
-rw-r--r--  1 vladyslavprytula  staff  16741 Dec  3 15:16 docs/book/drafts/ch06/data/verification_20251203T125142/test3_rich_quantized.log
-rw-r--r--  1 vladyslavprytula  staff  16670 Dec  3 16:04 docs/book/drafts/ch06/data/verification_20251203T125142/test4_rich_estimated.log
-rw-r--r--  1 vladyslavprytula  staff  30418 Dec  3 17:39 docs/book/drafts/ch06/data/verification_20251203T125142/test5_compute_arc.log
-rw-r--r--  1 vladyslavprytula  staff  70691 Dec  4 00:50 docs/book/drafts/ch06/data/verification_20251203T125142/test6_batch_matrix.log
-rw-r--r--  1 vladyslavprytula  staff  43376 Dec  4 01:38 docs/book/drafts/ch06/data/verification_20251203T125142/test7_lab_solutions.log
-rw-r--r--  1 vladyslavprytula  staff    5693 Dec  3 17:39 docs/book/drafts/ch06/data/verification_20251203T125142/template_bandits_rich_summary.json
-rw-r--r--  1 vladyslavprytula  staff    5734 Dec  3 16:52 docs/book/drafts/ch06/data/verification_20251203T125142/template_bandits_simple_summary.json
-rw-r--r--  1 vladyslavprytula  staff  100289 Dec  4 00:50 docs/book/drafts/ch06/data/verification_20251203T125142/verification_matrix_20251203T235022Z.json

==============================================================================
KEY METRICS SUMMARY
==============================================================================\033[0m

Test 1 (Simple Features) - Final Summary:
Summary (per-episode averages):
Policy                  Reward         GMV         CM2  ΔGMV vs static
Static-Premium            7.56        7.11        0.74          +0.00%
LinUCB                    5.62        5.12        0.46         -27.95%
ThompsonSampling          6.69        6.18        0.63         -13.07%

Per-segment GMV (static best vs bandits):
Segment          Static GMV  LinUCB GMV      TS GMV    LinUCB Δ%     TS Δ%
price_hunter           0.00        0.00        0.00       +0.00%    +0.00%
pl_lover               5.34        4.18        7.96      -21.78%   +49.03%
premium               31.60       17.43       19.77      -44.83%   -37.42%

Test 3 (Rich Quantized - Winning Config) - Final Summary:
Summary (per-episode averages):
Policy                  Reward         GMV         CM2  ΔGMV vs static
Static-Premium            7.56        7.11        0.74          +0.00%
LinUCB                    8.20        7.52        0.72          +5.79%
ThompsonSampling          7.99        7.34        0.69          +3.14%

Per-segment GMV (static best vs bandits):
Segment          Static GMV  LinUCB GMV      TS GMV    LinUCB Δ%     TS Δ%
price_hunter           0.00        0.00        0.00       +0.00%    +0.00%
pl_lover               5.34       11.45       11.51     +114.36%  +115.45%
premium               31.60       19.64       19.16      -37.83%   -39.37%
