#!/bin/bash
# Parallel multi-seed editing: one GPU per (model, task) combination.
# Uses resume support in ResultsWriter to skip already-completed jobs.
#
# Usage: bash scripts/rebuttal/run_multiseed_parallel.sh [--skip-eval]

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUNS_ROOT="${REPO_ROOT}/runs_refactor_data_20260121"
OUT_ROOT="${REPO_ROOT}/outputs/rebuttal_exp/raw/multiseed"
SKIP_EVAL="${1:---skip_eval}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

METHODS="baseline random_index smooth_abs abs_select grad_direction continued_ft sigma_only"

# (model_dir, task, gpu_id) assignments — 8 combos, 6 GPUs, stagger as 2 waves
# Wave 1: 6 combos on GPU 0-5
# Wave 2: 2 combos reuse GPU 0-1 after wave 1

run_one() {
    local GPU=$1
    local MODEL=$2
    local TASK=$3
    local BATCH_SIZE=$4

    echo "[GPU ${GPU}] Starting ${MODEL}/${TASK} (bs=${BATCH_SIZE})"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/rebuttal/run_multiseed_stability.py \
        --runs_root "$RUNS_ROOT" \
        --out_root "$OUT_ROOT" \
        --tasks "$TASK" \
        --methods $METHODS \
        --n_repeats 5 \
        --base_seed 42 \
        --calib_samples 128 \
        --calib_batch_size "$BATCH_SIZE" \
        --ft_steps 50 --ft_lr 5e-5 \
        --sigma_steps 50 --sigma_lr 5e-3 \
        --tensor_parallel_size 1 \
        $SKIP_EVAL 2>&1 | tee "${OUT_ROOT}/log_${MODEL}_${TASK}.txt"
    echo "[GPU ${GPU}] Done ${MODEL}/${TASK}"
}

echo "============================================"
echo "Parallel Multi-Seed Editing"
echo "============================================"
echo "Will use GPUs 0-5 (max 6)"
echo "Resume: skips already-completed (model,task,method,seed) combos"
echo "============================================"

# Wave 1: 6 jobs on 6 GPUs
run_one 0 meta-llama-Llama-3.1-8B alpaca 2 &
PID_0=$!
run_one 1 meta-llama-Llama-3.1-8B math 2 &
PID_1=$!
run_one 2 meta-llama-Llama-3.1-8B code 1 &   # batch_size=1 for code (long seqs)
PID_2=$!
run_one 3 meta-llama-Llama-3.1-8B csqa 2 &
PID_3=$!
run_one 4 Qwen-Qwen3-8B alpaca 2 &
PID_4=$!
run_one 5 Qwen-Qwen3-8B math 2 &
PID_5=$!

echo "Wave 1 launched: PIDs $PID_0 $PID_1 $PID_2 $PID_3 $PID_4 $PID_5"
echo "Waiting for wave 1 to finish..."

# Wait for any 2 GPUs to free up for wave 2
wait $PID_0 $PID_3  # alpaca and csqa likely fastest
echo "Wave 1 partial done, launching wave 2..."

# Wave 2: remaining 2 combos
run_one 0 Qwen-Qwen3-8B code 1 &   # batch_size=1 for code
PID_6=$!
run_one 3 Qwen-Qwen3-8B csqa 2 &
PID_7=$!

echo "Wave 2 launched: PIDs $PID_6 $PID_7"

# Wait for all
wait
echo ""
echo "============================================"
echo "All parallel editing jobs complete!"
echo "============================================"
wc -l "${OUT_ROOT}/results.jsonl"
