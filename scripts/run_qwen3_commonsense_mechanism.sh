#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE=/dataset1/zailong/models/Qwen3-8B
MODEL_ROOT=/dataset1/zailong/models/spectral-surgery
LORA="$MODEL_ROOT/Qwen3-8B-CommonSense170K-LoRA"
HNS="$MODEL_ROOT/Qwen3-8B-CommonSense170K-Spectral-Surgery"
TRAIN=/dataset1/zailong/data/peft-sft-lab/commonsense170k-train.parquet
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-cross-task-20260908/commonsense
ADAPTERS="$RUN_ROOT/adapters"

export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
mkdir -p "$RUN_ROOT/eval"

if [[ ! -f "$RUN_ROOT/spectra/spectrum_summary.json" ]]; then
  "$PYTHON" scripts/analyze_spectral_pair.py \
    --lora_path "$LORA" --hns_path "$HNS" --output_dir "$RUN_ROOT/spectra" --device cuda
fi
if [[ ! -f "$ADAPTERS/manifest.json" ]]; then
  "$PYTHON" scripts/build_hns_causal_controls.py \
    --lora_path "$LORA" --hns_path "$HNS" --out_root "$ADAPTERS" --device cuda
fi
"$PYTHON" - "$RUN_ROOT/experiment_paths.json" "$LORA" "$HNS" "$ADAPTERS" <<'PY'
import json, sys
from pathlib import Path
out, lora, hns, controls = sys.argv[1:]
rows = [
    {"label": "lora", "adapter_path": lora},
    {"label": "scalar_shrink", "adapter_path": controls + "/scalar-shrink"},
    {"label": "shape_only", "adapter_path": controls + "/shape-only"},
    {"label": "head_only", "adapter_path": controls + "/head-only"},
    {"label": "tail_only", "adapter_path": controls + "/tail-only"},
    {"label": "hns", "adapter_path": hns},
]
Path(out).write_text(json.dumps(rows, indent=2) + "\n")
PY

if [[ ! -f "$RUN_ROOT/activation/activation_analysis.json" ]]; then
  "$PYTHON" scripts/measure_lora_modification.py \
    --base_model "$BASE" --lora_path "$LORA" --hns_path "$HNS" \
    --control "scalar_shrink=$ADAPTERS/scalar-shrink" \
    --control "shape_only=$ADAPTERS/shape-only" \
    --control "head_only=$ADAPTERS/head-only" \
    --control "tail_only=$ADAPTERS/tail-only" \
    --dataset_path "$TRAIN" --dataset_kind commonsense --samples 256 \
    --batch_size 2 --max_seq_len 512 --seed 42 --dtype bf16 \
    --output_dir "$RUN_ROOT/activation"
fi
cp "$RUN_ROOT/spectra/module_spectra.csv" "$RUN_ROOT/module_spectra.csv"
cp "$RUN_ROOT/activation/direction_response.csv" "$RUN_ROOT/direction_response.csv"
cp "$RUN_ROOT/activation/layer_response.csv" "$RUN_ROOT/layer_response.csv"
cp "$RUN_ROOT/activation/module_response.csv" "$RUN_ROOT/module_response.csv"

for label in lora scalar_shrink shape_only head_only tail_only hns; do
  case "$label" in
    lora) adapter="$LORA" ;;
    hns) adapter="$HNS" ;;
    scalar_shrink) adapter="$ADAPTERS/scalar-shrink" ;;
    shape_only) adapter="$ADAPTERS/shape-only" ;;
    head_only) adapter="$ADAPTERS/head-only" ;;
    tail_only) adapter="$ADAPTERS/tail-only" ;;
  esac
  output="$RUN_ROOT/eval/$label/commonsense"
  if [[ ! -f "$output/summary.json" ]]; then
    "$PYTHON" scripts/eval_commonsense_8tasks.py \
      --base_model "$BASE" --adapter_dir "$adapter" --output_dir "$output" \
      --tasks all --backend vllm --chat_template_mode non_thinking --max_new_tokens 8 \
      --request_batch_size 256 --tensor_parallel_size 1 --vllm_max_model_len 2048 \
      --vllm_gpu_memory_utilization 0.90 --vllm_attention_backend FLASH_ATTN \
      --dtype bf16 --seed 42
  fi
  "$PYTHON" scripts/summarize_cross_task_task.py "$RUN_ROOT" --task commonsense
done

echo "[Done] Qwen3 commonsense mechanism: $RUN_ROOT"
