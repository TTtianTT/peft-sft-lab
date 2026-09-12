#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
MODEL_ROOT=/dataset1/zailong/models/spectral-surgery
DATA_ROOT=/dataset1/zailong/data/peft-sft-lab
OLD_QWEN=/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-cross-task-20260908/commonsense
OLD_LLAMA=/dataset1/zailong/runs/peft-sft-lab/Llama-3.1-8B-Instruct/hns-cross-task-20260908/tulu
OLD_AGG=/dataset1/zailong/runs/peft-sft-lab/hns-2x4-20260908
RUN_BASE=/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909
QWEN_ROOT="$RUN_BASE/Qwen3-8B/commonsense"
LLAMA_ROOT="$RUN_BASE/Llama-3.1-8B-Instruct/tulu"
OUTPUT_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-2x4-allmodules-20260909

export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export PATH="$(dirname "$PYTHON"):$PATH"

build_allmodule_hns() {
  local lora="$1" hns="$2"
  if [[ ! -f "$hns/spectral_edit_meta.json" ]]; then
    "$PYTHON" -m finetune.spectral_edit.cli hns \
      --lora_path "$lora" --out_dir "$hns" --target_modules all_modules \
      --output_rank 16 --fast_steps 8 --stable_steps 2 --cache_dir "$HF_HUB_CACHE"
  fi
}

prepare_mechanism() {
  local root="$1" base="$2" lora="$3" hns="$4" data="$5" kind="$6"
  local adapters="$root/adapters"
  mkdir -p "$root/eval"
  build_allmodule_hns "$lora" "$hns"
  if [[ ! -f "$root/spectra/spectrum_summary.json" ]]; then
    "$PYTHON" scripts/analyze_spectral_pair.py \
      --lora_path "$lora" --hns_path "$hns" --output_dir "$root/spectra" --device cuda
  fi
  if [[ ! -f "$adapters/manifest.json" ]]; then
    "$PYTHON" scripts/build_hns_causal_controls.py \
      --lora_path "$lora" --hns_path "$hns" --out_root "$adapters" --device cuda
  fi
  "$PYTHON" - "$root/experiment_paths.json" "$lora" "$hns" "$adapters" <<'PY'
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
  if [[ ! -f "$root/activation/activation_analysis.json" ]]; then
    "$PYTHON" scripts/measure_lora_modification.py \
      --base_model "$base" --lora_path "$lora" --hns_path "$hns" \
      --control "scalar_shrink=$adapters/scalar-shrink" \
      --control "shape_only=$adapters/shape-only" \
      --control "head_only=$adapters/head-only" \
      --control "tail_only=$adapters/tail-only" \
      --dataset_path "$data" --dataset_kind "$kind" --samples 256 \
      --batch_size 2 --max_seq_len 512 --seed 42 --dtype bf16 \
      --output_dir "$root/activation"
  fi
  cp "$root/spectra/module_spectra.csv" "$root/module_spectra.csv"
  cp "$root/activation/direction_response.csv" "$root/direction_response.csv"
  cp "$root/activation/layer_response.csv" "$root/layer_response.csv"
  cp "$root/activation/module_response.csv" "$root/module_response.csv"
}

adapter_for() {
  local root="$1" lora="$2" hns="$3" label="$4"
  case "$label" in
    lora) echo "$lora" ;;
    hns) echo "$hns" ;;
    scalar_shrink) echo "$root/adapters/scalar-shrink" ;;
    shape_only) echo "$root/adapters/shape-only" ;;
    head_only) echo "$root/adapters/head-only" ;;
    tail_only) echo "$root/adapters/tail-only" ;;
  esac
}

QWEN_BASE=/dataset1/zailong/models/Qwen3-8B
QWEN_LORA="$MODEL_ROOT/Qwen3-8B-CommonSense170K-LoRA"
QWEN_HNS="$QWEN_ROOT/hns-allmodules-8plus2"
prepare_mechanism "$QWEN_ROOT" "$QWEN_BASE" "$QWEN_LORA" "$QWEN_HNS" \
  "$DATA_ROOT/commonsense170k-train.parquet" commonsense
if [[ ! -f "$QWEN_ROOT/eval/lora/commonsense/summary.json" ]]; then
  mkdir -p "$QWEN_ROOT/eval/lora"
  cp -a "$OLD_QWEN/eval/lora/commonsense" "$QWEN_ROOT/eval/lora/commonsense"
fi
for label in scalar_shrink shape_only head_only tail_only hns; do
  adapter="$(adapter_for "$QWEN_ROOT" "$QWEN_LORA" "$QWEN_HNS" "$label")"
  output="$QWEN_ROOT/eval/$label/commonsense"
  if [[ ! -f "$output/summary.json" ]]; then
    "$PYTHON" scripts/eval_commonsense_8tasks.py \
      --base_model "$QWEN_BASE" --adapter_dir "$adapter" --output_dir "$output" \
      --tasks all --backend vllm --chat_template_mode non_thinking --max_new_tokens 8 \
      --request_batch_size 256 --tensor_parallel_size 1 --vllm_max_model_len 2048 \
      --vllm_gpu_memory_utilization 0.90 --vllm_attention_backend FLASH_ATTN \
      --dtype bf16 --seed 42
  fi
  "$PYTHON" scripts/summarize_cross_task_task.py "$QWEN_ROOT" --task commonsense
done

LLAMA_BASE=/dataset1/zailong/models/Llama-3.1-8B-Instruct
LLAMA_LORA="$MODEL_ROOT/Llama-3.1-8B-Instruct-InstructionFollowing-LoRA"
LLAMA_HNS="$LLAMA_ROOT/hns-allmodules-8plus2"
prepare_mechanism "$LLAMA_ROOT" "$LLAMA_BASE" "$LLAMA_LORA" "$LLAMA_HNS" \
  "$DATA_ROOT/cross-task-mechanism/tulu-3-sft-personas-instruction-following-train.parquet" tulu
if [[ ! -f "$LLAMA_ROOT/eval/lora/ifeval/metrics.json" ]]; then
  mkdir -p "$LLAMA_ROOT/eval/lora"
  cp -a "$OLD_LLAMA/eval/lora/ifeval" "$LLAMA_ROOT/eval/lora/ifeval"
fi
for label in scalar_shrink shape_only head_only tail_only hns; do
  adapter="$(adapter_for "$LLAMA_ROOT" "$LLAMA_LORA" "$LLAMA_HNS" "$label")"
  output="$LLAMA_ROOT/eval/$label/ifeval"
  if [[ ! -f "$output/metrics.json" ]]; then
    "$PYTHON" -m finetune.eval.eval_ifeval \
      --base_model "$LLAMA_BASE" --adapter_dir "$adapter" --output_dir "$output" \
      --dataset_path "$DATA_ROOT/cross-task-mechanism/ifeval-train.jsonl" --split train \
      --max_new_tokens 2048 --dtype bf16 --seed 42 --chat_template_mode auto \
      --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
      --vllm_gpu_memory_utilization 0.85 --vllm_attention_backend FLASH_ATTN \
      --vllm_disable_flashinfer_sampler --vllm_request_batch_size 128 --per_category_metrics
  fi
  "$PYTHON" scripts/summarize_cross_task_task.py "$LLAMA_ROOT" --task tulu
done

"$PYTHON" scripts/verify_hns_scope_extension.py \
  --lora_path "$QWEN_LORA" \
  --partial_hns_path "$MODEL_ROOT/Qwen3-8B-CommonSense170K-Spectral-Surgery" \
  --allmodule_hns_path "$QWEN_HNS" \
  --output "$QWEN_ROOT/scope_verification.json"
"$PYTHON" scripts/verify_hns_scope_extension.py \
  --lora_path "$LLAMA_LORA" \
  --partial_hns_path "$MODEL_ROOT/Llama-3.1-8B-Instruct-InstructionFollowing-SpectralSurgery-HNS8p2" \
  --allmodule_hns_path "$LLAMA_HNS" \
  --output "$LLAMA_ROOT/scope_verification.json"

"$PYTHON" scripts/summarize_2x4_hns_mechanism.py \
  --qwen_commonsense_root "$QWEN_ROOT" \
  --llama_tulu_root "$LLAMA_ROOT" \
  --previous_output_root "$OLD_AGG" \
  --output_root "$OUTPUT_ROOT"

echo "[Done] all-module scope completion: $OUTPUT_ROOT"
