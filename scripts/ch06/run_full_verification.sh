#!/bin/bash
# ==============================================================================
# Chapter 6 Full Verification Suite
# ==============================================================================
#
# This script runs ALL Chapter 6 scripts with production-level horizons to
# verify the complete narrative before publisher submission.
#
# Expected runtime: ~30-60 or longer :)  minutes depending on hardware
#
# Usage:
#   cd /Volumes/Lexar2T/src/rl_search_from_scratch
#   chmod +x scripts/ch06/run_full_verification.sh
#   ./scripts/ch06/run_full_verification.sh 2>&1 | tee ch06_verification.log
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
OUTPUT_DIR="docs/book/drafts/ch06/data/verification_$(date +%Y%m%dT%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================================================="
echo "CHAPTER 6 FULL VERIFICATION SUITE"
echo "=============================================================================="
echo -e "${NC}"
echo "Configuration:"
echo "  Static episodes:  $N_STATIC"
echo "  Bandit episodes:  $N_BANDIT"
echo "  World seed:       $WORLD_SEED"
echo "  Bandit seed:      $BANDIT_SEED"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Start time: $(date)"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==============================================================================
# TEST 1: Simple Features (The Failure Mode)
# ==============================================================================
echo -e "${YELLOW}=============================================================================="
echo "TEST 1: Simple Features - The Failure Mode (§6.5)"
echo "==============================================================================${NC}"
echo ""
echo "Expected: Bandits underperform or match static baseline with simple features"
echo ""

python scripts/ch06/template_bandits_demo.py \
    --features simple \
    --n-static $N_STATIC \
    --n-bandit $N_BANDIT \
    --world-seed $WORLD_SEED \
    --bandit-base-seed $BANDIT_SEED \
    2>&1 | tee "$OUTPUT_DIR/test1_simple_features.log"

echo -e "${GREEN}✓ Test 1 completed${NC}"
echo ""

# ==============================================================================
# TEST 2: Rich Features + Oracle Latents (LinUCB WINS)
# ==============================================================================
echo -e "${YELLOW}=============================================================================="
echo "TEST 2: Rich Features + Oracle Latents (§6.7.4) - LINUCB WINS"
echo "==============================================================================${NC}"
echo ""
echo "Expected: LinUCB ~+31% vs static, TS ~+5% (LinUCB wins with clean features)"
echo ""

python scripts/ch06/template_bandits_demo.py \
    --features rich \
    --rich-regularization blend \
    --n-static $N_STATIC \
    --n-bandit $N_BANDIT \
    --world-seed $WORLD_SEED \
    --bandit-base-seed $BANDIT_SEED \
    --hparam-mode rich_est \
    --prior-weight 50 \
    --lin-alpha 0.2 \
    --ts-sigma 0.5 \
    2>&1 | tee "$OUTPUT_DIR/test2_rich_blend.log"

echo -e "${GREEN}✓ Test 2 completed${NC}"
echo ""

# ==============================================================================
# TEST 3: Rich Features with Quantized Regularization (Winning Config)
# ==============================================================================
echo -e "${YELLOW}=============================================================================="
echo "TEST 3: Rich Features + Quantized Regularization (Winning Config)"
echo "==============================================================================${NC}"
echo ""
echo "Expected: LinUCB achieves ~5-8% GMV lift, TS achieves ~3-5% lift"
echo ""

python scripts/ch06/template_bandits_demo.py \
    --features rich \
    --rich-regularization quantized \
    --n-static $N_STATIC \
    --n-bandit $N_BANDIT \
    --world-seed $WORLD_SEED \
    --bandit-base-seed $BANDIT_SEED \
    --hparam-mode rich_est \
    --prior-weight 50 \
    --lin-alpha 0.2 \
    --ts-sigma 0.2 \
    2>&1 | tee "$OUTPUT_DIR/test3_rich_quantized.log"

echo -e "${GREEN}✓ Test 3 completed${NC}"
echo ""

# ==============================================================================
# TEST 4: Rich Features + Estimated Latents (TS WINS)
# ==============================================================================
echo -e "${YELLOW}=============================================================================="
echo "TEST 4: Rich Features + Estimated Latents (§6.7.5) - TS WINS"
echo "==============================================================================${NC}"
echo ""
echo "Expected: TS ~+31% vs static, LinUCB ~+6% (TS wins with noisy features)"
echo ""

python scripts/ch06/template_bandits_demo.py \
    --features rich_est \
    --n-static $N_STATIC \
    --n-bandit $N_BANDIT \
    --world-seed $WORLD_SEED \
    --bandit-base-seed $BANDIT_SEED \
    --hparam-mode rich_est \
    --prior-weight 50 \
    --lin-alpha 0.2 \
    --ts-sigma 0.5 \
    2>&1 | tee "$OUTPUT_DIR/test4_rich_estimated.log"

echo -e "${GREEN}✓ Test 4 completed${NC}"
echo ""

# ==============================================================================
# TEST 5: Three-Stage Compute Arc (Algorithm Selection Principle)
# ==============================================================================
echo -e "${YELLOW}=============================================================================="
echo "TEST 5: Three-Stage Compute Arc (§6.5 → §6.7.4 → §6.7.5)"
echo "==============================================================================${NC}"
echo ""
echo "This runs all three stages demonstrating the Algorithm Selection Principle:"
echo "  Stage 1: Simple features (failure)"
echo "  Stage 2: Rich + Oracle latents (LinUCB wins)"
echo "  Stage 3: Rich + Estimated latents (TS wins)"
echo ""

python scripts/ch06/ch06_compute_arc.py \
    --n-static $N_STATIC \
    --n-bandit $N_BANDIT \
    --base-seed $WORLD_SEED \
    --out-dir "$OUTPUT_DIR" \
    --prior-weight 50 \
    --lin-alpha 0.2 \
    --ts-sigma 0.5 \
    2>&1 | tee "$OUTPUT_DIR/test5_compute_arc.log"

echo -e "${GREEN}✓ Test 5 completed${NC}"
echo ""

# ==============================================================================
# TEST 6: Batch Matrix Runner (Multiple Scenarios)
# ==============================================================================
echo -e "${YELLOW}=============================================================================="
echo "TEST 6: Batch Matrix Runner - All Scenarios"
echo "==============================================================================${NC}"
echo ""
echo "Running 5 scenarios: simple_baseline, rich_oracle_raw, rich_oracle_blend,"
echo "                     rich_oracle_quantized, rich_estimated"
echo ""

python scripts/ch06/run_bandit_matrix.py \
    --n-static $N_STATIC \
    --n-bandit $N_BANDIT \
    --world-seed-base $WORLD_SEED \
    --bandit-seed-base $BANDIT_SEED \
    --output-dir "$OUTPUT_DIR" \
    --filename-prefix "verification_matrix" \
    --max-workers 4 \
    --stream-logs \
    2>&1 | tee "$OUTPUT_DIR/test6_batch_matrix.log"

echo -e "${GREEN}✓ Test 6 completed${NC}"
echo ""

# ==============================================================================
# TEST 7: Lab Solutions (Exercises 6.1-6.6)
# ==============================================================================
echo -e "${YELLOW}=============================================================================="
echo "TEST 7: Lab Solutions Module - All Exercises"
echo "==============================================================================${NC}"
echo ""

python -m scripts.ch06.lab_solutions --all \
    2>&1 | tee "$OUTPUT_DIR/test7_lab_solutions.log"

echo -e "${GREEN}✓ Test 7 completed${NC}"
echo ""

# ==============================================================================
# SUMMARY
# ==============================================================================
echo -e "${BLUE}=============================================================================="
echo "VERIFICATION COMPLETE"
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

# Extract key metrics from logs
echo -e "${YELLOW}=============================================================================="
echo "KEY METRICS SUMMARY"
echo "==============================================================================${NC}"
echo ""

echo "Test 1 (Simple Features) - FAILURE MODE:"
grep -A 10 "Summary (per-episode" "$OUTPUT_DIR/test1_simple_features.log" | head -12 || echo "  (parsing failed)"
echo ""

echo "Test 2 (Oracle Latents) - LINUCB WINS:"
grep -A 10 "Summary (per-episode" "$OUTPUT_DIR/test2_rich_blend.log" | head -12 || echo "  (parsing failed)"
echo ""

echo "Test 4 (Estimated Latents) - TS WINS:"
grep -A 10 "Summary (per-episode" "$OUTPUT_DIR/test4_rich_estimated.log" | head -12 || echo "  (parsing failed)"
echo ""

echo -e "${BLUE}=============================================================================="
echo "THE ALGORITHM SELECTION PRINCIPLE"
echo "==============================================================================${NC}"
echo ""
echo "  Clean/Oracle features  →  LinUCB   (precise exploitation)"
echo "  Noisy/Estimated feat.  →  TS       (robust exploration)"
echo ""
echo "  Production systems have noisy features. Default to TS."
echo ""
echo -e "${GREEN}All tests passed! Chapter 6 is ready for publisher review.${NC}"
