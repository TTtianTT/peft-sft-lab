#!/bin/bash
# Master orchestrator for all rebuttal experiments.
#
# Usage:
#   bash scripts/rebuttal/run_all_rebuttal.sh [--dry-run] [--skip-eval] [--tp N]
#
# Environment:
#   RUNS_ROOT: Path to adapter checkpoints (default: runs_refactor_data_20260121)
#   OUT_ROOT:  Output root (default: outputs/rebuttal_exp)
#   TP:        Tensor parallel size for lm_eval (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Defaults
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs_refactor_data_20260121}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/rebuttal_exp}"
TP="${TP:-1}"
DRY_RUN=""
SKIP_EVAL=""

# Parse args
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry_run" ;;
        --skip-eval) SKIP_EVAL="--skip_eval" ;;
        --tp=*) TP="${arg#*=}" ;;
        --tp) shift; TP="$1" ;;
    esac
done

echo "============================================"
echo "Rebuttal Experiments Orchestrator"
echo "============================================"
echo "RUNS_ROOT:  ${RUNS_ROOT}"
echo "OUT_ROOT:   ${OUT_ROOT}"
echo "TP:         ${TP}"
echo "DRY_RUN:    ${DRY_RUN:-no}"
echo "SKIP_EVAL:  ${SKIP_EVAL:-no}"
echo "============================================"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ============================================================================
# Step 0: Create output directories
# ============================================================================
echo ""
echo ">>> Step 0: Creating output directories"
mkdir -p "${OUT_ROOT}"/{raw,summarized,plots,tables,notes,scripts}

# ============================================================================
# Step 1: P0-C Multi-Seed Stability (includes baseline, spectral edits,
#         continued FT, sigma-only)
# ============================================================================
echo ""
echo ">>> Step 1: Multi-Seed Stability (P0-C)"
echo "    Methods: baseline, random_index, smooth_abs, abs_select, grad_direction, continued_ft, sigma_only"
echo "    Seeds: 5 repeats (42-46)"
python scripts/rebuttal/run_multiseed_stability.py \
    --runs_root "${RUNS_ROOT}" \
    --out_root "${OUT_ROOT}/raw/multiseed" \
    --tasks math code alpaca csqa \
    --methods baseline random_index smooth_abs abs_select grad_direction continued_ft sigma_only \
    --n_repeats 5 \
    --base_seed 42 \
    --calib_samples 128 \
    --ft_steps 50 --ft_lr 5e-5 \
    --sigma_steps 50 --sigma_lr 5e-3 \
    --tensor_parallel_size "${TP}" \
    ${DRY_RUN} ${SKIP_EVAL}

# ============================================================================
# Step 2: P1-D Guided vs Random Analysis (no GPU needed, uses multiseed data)
# ============================================================================
echo ""
echo ">>> Step 2: Guided vs Random Analysis (P1-D)"
python scripts/rebuttal/analyze_guided_vs_random.py \
    --results_jsonl "${OUT_ROOT}/raw/multiseed/results.jsonl" \
    --out_dir "${OUT_ROOT}/summarized/guided_vs_random"

# ============================================================================
# Step 3: P1-E Proxy Faithfulness
# ============================================================================
echo ""
echo ">>> Step 3: Proxy Faithfulness (P1-E)"
python scripts/rebuttal/run_proxy_faithfulness.py \
    --runs_root "${RUNS_ROOT}" \
    --out_root "${OUT_ROOT}/raw/proxy_faithfulness" \
    --multiseed_root "${OUT_ROOT}/raw/multiseed" \
    --tasks math code alpaca csqa \
    --calib_samples 128 \
    --proxy_samples 128

# ============================================================================
# Step 4: P1-F Calibration-Size Sweep
# ============================================================================
echo ""
echo ">>> Step 4: Calibration-Size Sweep (P1-F)"
python scripts/rebuttal/run_calib_sweep.py \
    --runs_root "${RUNS_ROOT}" \
    --out_root "${OUT_ROOT}/raw/calib_sweep" \
    --tasks math code alpaca csqa \
    --ncals 32 64 128 256 \
    --n_repeats 3 \
    --ft_steps 50 --ft_lr 5e-5 \
    --sigma_steps 50 --sigma_lr 5e-3 \
    --tensor_parallel_size "${TP}" \
    ${DRY_RUN} ${SKIP_EVAL}

# ============================================================================
# Step 5: Plotting and reporting
# ============================================================================
echo ""
echo ">>> Step 5: Generating plots and report"

# Stability plots
python scripts/rebuttal/plot_stability.py \
    --results_jsonl "${OUT_ROOT}/raw/multiseed/results.jsonl" \
    --out_dir "${OUT_ROOT}/plots/stability"

# Calib sweep plots
if [ -f "${OUT_ROOT}/raw/calib_sweep/calib_sweep_summary.json" ]; then
    python scripts/rebuttal/plot_calib_sweep.py \
        --summary_json "${OUT_ROOT}/raw/calib_sweep/calib_sweep_summary.json" \
        --out_dir "${OUT_ROOT}/plots/calib_sweep"
fi

# Proxy faithfulness plots
if [ -f "${OUT_ROOT}/raw/proxy_faithfulness/all_proxy_results.json" ]; then
    python scripts/rebuttal/plot_proxy_faithfulness.py \
        --proxy_results "${OUT_ROOT}/raw/proxy_faithfulness/all_proxy_results.json" \
        --eval_results "${OUT_ROOT}/raw/multiseed/results.jsonl" \
        --out_dir "${OUT_ROOT}/plots/proxy_faithfulness"
fi

# Aggregate all results
python scripts/rebuttal/aggregate_all_results.py \
    --exp_root "${OUT_ROOT}" \
    --out_dir "${OUT_ROOT}/summarized"

# Generate report
python scripts/rebuttal/generate_rebuttal_report.py \
    --exp_root "${OUT_ROOT}" \
    --out_file "${OUT_ROOT}/notes/rebuttal_report.md"

echo ""
echo "============================================"
echo "All rebuttal experiments complete!"
echo "============================================"
echo "Results:  ${OUT_ROOT}/raw/"
echo "Summary:  ${OUT_ROOT}/summarized/"
echo "Plots:    ${OUT_ROOT}/plots/"
echo "Report:   ${OUT_ROOT}/notes/rebuttal_report.md"
echo "============================================"
