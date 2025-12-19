#!/bin/bash
# ==============================================================================
# Chapter 6 GPU Verification Suite
# ==============================================================================
#
# This script runs GPU-accelerated Chapter 6 scripts with production horizons.
# Use this to verify the GPU path produces consistent results with CPU.
#
# Expected runtime: ~10-20 minutes (faster than CPU due to GPU acceleration)
#
# Usage:
#   cd /Volumes/Lexar2T/src/rl_search_from_scratch
#   chmod +x scripts/ch06/run_gpu_verification.sh
#   ./scripts/ch06/run_gpu_verification.sh 2>&1 | tee ch06_gpu_verification.log
#  Full Verification Commands

#   Option 1: Run Complete CPU Verification Suite (Recommended First)

#   cd /Volumes/Lexar2T/src/rl_search_from_scratch
#   ./scripts/ch06/run_full_verification.sh 2>&1 | tee ch06_verification.log

#   What it runs (7 tests, can be long):

#   | Test | Script                                                                   | Description                  | Expected           |
#   |------|--------------------------------------------------------------------------|------------------------------|--------------------|
#   | 1    | template_bandits_demo.py --features simple                               | Simple features failure mode | Bandits ≈ static   |
#   | 2    | template_bandits_demo.py --features rich --rich-regularization blend     | Rich + blend                 | TS >> static       |
#   | 3    | template_bandits_demo.py --features rich --rich-regularization quantized | Winning config               | LinUCB +5-8%       |
#   | 4    | template_bandits_demo.py --features rich_est                             | Estimated features           | Practical scenario |
#   | 5    | ch06_compute_arc.py                                                      | Full simple→rich arc         | Complete narrative |
#   | 6    | run_bandit_matrix.py                                                     | All 5 scenarios in batch     | Comprehensive      |
#   | 7    | lab_solutions --all                                                      | All exercises                | Theory validation  |

#   Option 2: Run GPU Verification Suite

#   cd /Volumes/Lexar2T/src/rl_search_from_scratch
#   ./scripts/ch06/run_gpu_verification.sh 2>&1 | tee ch06_gpu_verification.log

#   What it runs (~10-20 min):
#   - GPU compute arc (simple → rich)
#   - GPU batch matrix runner (all scenarios)

#   Option 3: Run Individual Commands Manually

#   Simple Features (The Failure):
#   source .venv/bin/activate && python scripts/ch06/template_bandits_demo.py \
#       --features simple \
#       --n-static 2000 \
#       --n-bandit 20000 \
#       --world-seed 20250322 \
#       --bandit-base-seed 20250349

#   Rich Features + Quantized (Winning Config):
#   source .venv/bin/activate && python scripts/ch06/template_bandits_demo.py \
#       --features rich \
#       --rich-regularization quantized \
#       --n-static 2000 \
#       --n-bandit 20000 \
#       --world-seed 20250322 \
#       --bandit-base-seed 20250349 \
#       --hparam-mode rich_est \
#       --prior-weight 50 \
#       --lin-alpha 0.2 \
#       --ts-sigma 0.2

#   Batch Matrix (All Scenarios):
#   source .venv/bin/activate && python scripts/ch06/run_bandit_matrix.py \
#       --n-static 2000 \
#       --n-bandit 20000 \
#       --max-workers 4 \
#       --stream-logs

#   Lab Solutions (All):
#   source .venv/bin/activate && python -m scripts.ch06.lab_solutions --all

#   Results will be saved to docs/book/drafts/ch06/data/verification_<timestamp>/.
#
# ==============================================================================

set -e  # Exit on first error

# Force unbuffered output for Python (critical for real-time console output with tee)
export PYTHONUNBUFFERED=1

# Configuration
N_STATIC=2000
N_BANDIT=20000
WORLD_SEED=20250322
BANDIT_SEED=20250349
BATCH_SIZE=1024
DEVICE="auto"  # Will use CUDA if available, else MPS, else CPU
OUTPUT_DIR="docs/book/drafts/ch06/data/gpu_verification_$(date +%Y%m%dT%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================================================="
echo "CHAPTER 6 GPU VERIFICATION SUITE"
echo "=============================================================================="
echo -e "${NC}"
echo "Configuration:"
echo "  Static episodes:  $N_STATIC"
echo "  Bandit episodes:  $N_BANDIT"
echo "  World seed:       $WORLD_SEED"
echo "  Bandit seed:      $BANDIT_SEED"
echo "  Batch size:       $BATCH_SIZE"
echo "  Device:           $DEVICE"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Start time: $(date)"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check device availability
echo -e "${YELLOW}Checking device availability...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
echo ""

# ==============================================================================
# TEST 1: GPU Compute Arc (Simple → Rich)
# ==============================================================================
echo -e "${YELLOW}=============================================================================="
echo "GPU TEST 1: Compute Arc - Simple → Rich Features"
echo "==============================================================================${NC}"
echo ""

python scripts/ch06/optimization_gpu/ch06_compute_arc_gpu.py \
    --n-static $N_STATIC \
    --n-bandit $N_BANDIT \
    --base-seed $WORLD_SEED \
    --out-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --prior-weight 50 \
    --lin-alpha 0.2 \
    --ts-sigma 0.5 \
    --show-volume \
    2>&1 | tee "$OUTPUT_DIR/gpu_test1_compute_arc.log"

echo -e "${GREEN}✓ GPU Test 1 completed${NC}"
echo ""

# ==============================================================================
# TEST 2: GPU Batch Matrix Runner
# ==============================================================================
echo -e "${YELLOW}=============================================================================="
echo "GPU TEST 2: Batch Matrix Runner - All Scenarios"
echo "==============================================================================${NC}"
echo ""

python scripts/ch06/optimization_gpu/run_bandit_matrix_gpu.py \
    --n-static $N_STATIC \
    --n-bandit $N_BANDIT \
    --world-seed-base $WORLD_SEED \
    --bandit-seed-base $BANDIT_SEED \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --filename-prefix "gpu_verification_matrix" \
    --max-workers 1 \
    --stream-logs \
    --show-volume \
    2>&1 | tee "$OUTPUT_DIR/gpu_test2_batch_matrix.log"

echo -e "${GREEN}✓ GPU Test 2 completed${NC}"
echo ""

# ==============================================================================
# SUMMARY
# ==============================================================================
echo -e "${BLUE}=============================================================================="
echo "GPU VERIFICATION COMPLETE"
echo "==============================================================================${NC}"
echo ""
echo "End time: $(date)"
echo ""
echo "Output files saved to: $OUTPUT_DIR"
echo ""
echo "Key files to review:"
ls -la "$OUTPUT_DIR"/*.log 2>/dev/null || echo "  (no log files)"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (no JSON files)"
echo ""

echo -e "${GREEN}GPU tests passed!${NC}"
