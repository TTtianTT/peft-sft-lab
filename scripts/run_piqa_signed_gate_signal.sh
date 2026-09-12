#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE=/dataset1/zailong/models/Qwen3-8B
LORA=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-CommonSense170K-LoRA
HNS=/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Qwen3-8B/commonsense/hns-allmodules-8plus2
PIQA=/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-cross-task-20260908/commonsense/eval/lora/commonsense/piqa/predictions.jsonl
SFT_META=/dataset1/zailong/runs/peft-sft-lab/hns-module-utility-20260909/gradient/qwen_commonsense/spectral_edit_meta.json
TRAIN=/dataset1/zailong/data/peft-sft-lab/commonsense170k-train.parquet
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-signed-gates-20260910/qwen_commonsense_piqa

export PYTHONPATH="$PWD/src:$PWD/scripts"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/signals" "$RUN_ROOT/selection"

if [[ ! -f "$RUN_ROOT/signals/direction_signals.tsv" ]]; then
  "$PYTHON" scripts/measure_piqa_spectral_gate_signals.py \
    --base_model "$BASE" --lora_path "$LORA" --hns_path "$HNS" \
    --piqa_predictions "$PIQA" --sft_gradient_meta "$SFT_META" \
    --commonsense_train "$TRAIN" --output_dir "$RUN_ROOT/signals" \
    --signal_samples 128 --dose_samples 128 --validation_samples 512 \
    --batch_size 2 --seed 20260910 --dtype bf16 --max_seq_len 1024 --bootstrap 5000
fi

if [[ ! -f "$RUN_ROOT/selection/selection.json" ]]; then
  "$PYTHON" scripts/select_piqa_spectral_directions.py \
    --signals "$RUN_ROOT/signals/direction_signals.tsv" \
    --output_dir "$RUN_ROOT/selection" --pairs 12 --max_direction 8 --layer_bins 4
fi

echo "[Done] PIQA signed-gate signal stage: $RUN_ROOT"
