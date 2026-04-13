#!/bin/bash
# Run all P0-1 control edits (no GPU needed for most, lightweight)
set -euo pipefail

PYTHON=/home/zailongtian/miniconda3/envs/magicoder_svd/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

OUT_BASE="$REPO_ROOT/outputs/rebuttal_v2/p0_1_random_mechanism/edited"

# Settings: model_dir / task / base_model
SETTINGS=(
    "Qwen-Qwen3-8B math Qwen/Qwen3-8B"
    "Qwen-Qwen3-8B alpaca Qwen/Qwen3-8B"
    "meta-llama-Llama-3.1-8B math meta-llama/Llama-3.1-8B"
    "meta-llama-Llama-3.1-8B csqa meta-llama/Llama-3.1-8B"
)

# Profile mapping
declare -A PROFILES
PROFILES[math]="paper_math_ift_3ep"
PROFILES[alpaca]="paper_alpaca_3ep"
PROFILES[csqa]="paper_csqa_3ep"

CONTROLS="flatten_spectrum shrink_top_only suppress_tail_only permute_sigma uniform_rescale"
SEEDS="0 1 2 3 4"

echo "Running P0-1 control edits..."
for setting in "${SETTINGS[@]}"; do
    read -r model_dir task base_model <<< "$setting"
    profile="${PROFILES[$task]}"
    lora_path="$REPO_ROOT/runs_refactor_data_20260121/$model_dir/$task/lora/profile-$profile/rank-16/seed42"

    for control in $CONTROLS; do
        for seed in $SEEDS; do
            out_dir="$OUT_BASE/$model_dir/$task/$control/seed$seed"
            if [ -f "$out_dir/spectral_edit_meta.json" ]; then
                echo "  [SKIP] $model_dir/$task/$control/seed$seed (already done)"
                continue
            fi
            echo "  [RUN] $model_dir/$task/$control/seed$seed"
            $PYTHON "$REPO_ROOT/scripts/rebuttal_v2/p0_1_control_edit.py" \
                --base_model "$base_model" \
                --lora_path "$lora_path" \
                --out_dir "$out_dir" \
                --control_method "$control" \
                --seed "$seed" 2>&1 | tail -1
        done
    done
done

echo "All control edits done!"
