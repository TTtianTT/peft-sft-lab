#!/bin/bash
set -euo pipefail

# ============================================================================
# Master Orchestrator for Rebuttal V2 Experiments
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$REPO_ROOT/src"

export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"

echo "============================================"
echo "Rebuttal V2: Explanatory Experiments"
echo "============================================"
echo "REPO_ROOT: $REPO_ROOT"
echo "Start time: $(date)"
echo ""

# ============================================================
# Phase 1: Analysis-only experiments (no GPU needed for eval)
# ============================================================

echo ""
echo "============================================"
echo "P0-2: Proxy Faithfulness Analysis"
echo "============================================"
python "$SCRIPT_DIR/p0_2_proxy_faithfulness.py" 2>&1 | tee "$REPO_ROOT/outputs/rebuttal_v2/p0_2_proxy_faithfulness/log.txt"

echo ""
echo "============================================"
echo "P0-4: Applicability Criterion"
echo "============================================"
python "$SCRIPT_DIR/p0_4_applicability.py" 2>&1 | tee "$REPO_ROOT/outputs/rebuttal_v2/p0_4_applicability/log.txt"

echo ""
echo "============================================"
echo "P1-5: Sigma-Only SGD Analysis"
echo "============================================"
python "$SCRIPT_DIR/p1_5_sigma_sgd.py" 2>&1 | tee "$REPO_ROOT/outputs/rebuttal_v2/p1_5_sigma_sgd/log.txt"

echo ""
echo "============================================"
echo "P1-6: IFEval Error Taxonomy"
echo "============================================"
python "$SCRIPT_DIR/p1_6_ifeval_taxonomy.py" 2>&1 | tee "$REPO_ROOT/outputs/rebuttal_v2/p1_6_ifeval_taxonomy/log.txt"

# ============================================================
# Phase 2: GPU experiments (P0-3 needs single-GPU calibration)
# ============================================================

echo ""
echo "============================================"
echo "P0-3: Fixed-Subspace Justification"
echo "============================================"
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/p0_3_fixed_subspace.py" --calib_samples 128 2>&1 | tee "$REPO_ROOT/outputs/rebuttal_v2/p0_3_fixed_subspace/log.txt"

# ============================================================
# Phase 3: Heavy GPU experiments (P0-1 needs multi-GPU parallel)
# ============================================================

echo ""
echo "============================================"
echo "P0-1: Random Mechanism Decomposition (Edit Phase)"
echo "============================================"
python "$SCRIPT_DIR/p0_1_random_mechanism.py" --n_random_seeds 50 --n_gpus 8 --edit_only 2>&1 | tee "$REPO_ROOT/outputs/rebuttal_v2/p0_1_random_mechanism/edit_log.txt"

echo ""
echo "============================================"
echo "P0-1: Random Mechanism Decomposition (Eval Phase)"
echo "============================================"
python "$SCRIPT_DIR/p0_1_random_mechanism.py" --n_random_seeds 50 --n_gpus 8 --tp 1 --eval_only 2>&1 | tee "$REPO_ROOT/outputs/rebuttal_v2/p0_1_random_mechanism/eval_log.txt"

echo ""
echo "============================================"
echo "P0-1: Analysis"
echo "============================================"
python "$SCRIPT_DIR/p0_1_analyze.py" 2>&1 | tee "$REPO_ROOT/outputs/rebuttal_v2/p0_1_random_mechanism/analysis_log.txt"

# ============================================================
# Phase 4: Final compilation
# ============================================================

echo ""
echo "============================================"
echo "Final: Compile Summaries"
echo "============================================"
python "$SCRIPT_DIR/compile_final.py" 2>&1 | tee "$REPO_ROOT/outputs/rebuttal_v2/final/compile_log.txt"

echo ""
echo "============================================"
echo "ALL DONE"
echo "End time: $(date)"
echo "============================================"
