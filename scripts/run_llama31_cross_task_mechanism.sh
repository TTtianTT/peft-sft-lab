#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE=/dataset1/zailong/models/Llama-3.1-8B-Instruct
MODEL_ROOT=/dataset1/zailong/models/spectral-surgery
DATA_ROOT=/dataset1/zailong/data/peft-sft-lab
FIXED_DATA="$DATA_ROOT/cross-task-mechanism"
RUN_BASE=/dataset1/zailong/runs/peft-sft-lab/Llama-3.1-8B-Instruct/hns-cross-task-20260908

export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export PATH="$(dirname "$PYTHON"):$PATH"
mkdir -p "$RUN_BASE"

declare -A LORA HNS TRAIN KIND
LORA[magicoder]="$MODEL_ROOT/Llama-3.1-8B-Instruct-Magicoder-50K-LoRA-E1"
HNS[magicoder]="$MODEL_ROOT/Llama-3.1-8B-Instruct-Magicoder-50K-SpectralSurgery-HNS4p1"
TRAIN[magicoder]="$DATA_ROOT/magicoder-train.parquet"
KIND[magicoder]=magicoder

LORA[metamath]="$MODEL_ROOT/Llama-3.1-8B-Instruct-MetaMathQA-50K-LoRA"
HNS[metamath]="$MODEL_ROOT/Llama-3.1-8B-Instruct-MetaMathQA-50K-SpectralSurgery-HNS4p1-AllMods"
TRAIN[metamath]="$DATA_ROOT/metamathqa-train.parquet"
KIND[metamath]=metamath

LORA[tulu]="$MODEL_ROOT/Llama-3.1-8B-Instruct-InstructionFollowing-LoRA"
HNS[tulu]="$MODEL_ROOT/Llama-3.1-8B-Instruct-InstructionFollowing-SpectralSurgery-HNS8p2"
TRAIN[tulu]="$FIXED_DATA/tulu-3-sft-personas-instruction-following-train.parquet"
KIND[tulu]=tulu

LORA[commonsense]="$MODEL_ROOT/Llama-3.1-8B-Instruct-CommonSense170K-LoRA-GBS64-Final"
HNS[commonsense]="$MODEL_ROOT/Llama-3.1-8B-Instruct-CommonSense170K-Spectral-Surgery-AllModules-8Plus2"
TRAIN[commonsense]="$DATA_ROOT/commonsense170k-train.parquet"
KIND[commonsense]=commonsense

prepare_task() {
  local task="$1" root="$RUN_BASE/$1" adapters="$RUN_BASE/$1/adapters"
  mkdir -p "$root/eval"
  if [[ ! -f "$root/spectra/spectrum_summary.json" ]]; then
    "$PYTHON" scripts/analyze_spectral_pair.py \
      --lora_path "${LORA[$task]}" --hns_path "${HNS[$task]}" \
      --output_dir "$root/spectra" --device cuda
  fi
  if [[ ! -f "$adapters/manifest.json" ]]; then
    "$PYTHON" scripts/build_hns_causal_controls.py \
      --lora_path "${LORA[$task]}" --hns_path "${HNS[$task]}" \
      --out_root "$adapters" --device cuda
  fi
  "$PYTHON" - "$root/experiment_paths.json" "${LORA[$task]}" "${HNS[$task]}" "$adapters" <<'PY'
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
      --base_model "$BASE" --lora_path "${LORA[$task]}" --hns_path "${HNS[$task]}" \
      --control "scalar_shrink=$adapters/scalar-shrink" \
      --control "shape_only=$adapters/shape-only" \
      --control "head_only=$adapters/head-only" \
      --control "tail_only=$adapters/tail-only" \
      --dataset_path "${TRAIN[$task]}" --dataset_kind "${KIND[$task]}" \
      --samples 256 --batch_size 2 --max_seq_len 512 --seed 42 --dtype bf16 \
      --output_dir "$root/activation"
  fi
  cp "$root/spectra/module_spectra.csv" "$root/module_spectra.csv"
  cp "$root/activation/direction_response.csv" "$root/direction_response.csv"
  cp "$root/activation/layer_response.csv" "$root/layer_response.csv"
  cp "$root/activation/module_response.csv" "$root/module_response.csv"
}

adapter_for() {
  local task="$1" label="$2"
  case "$label" in
    lora) echo "${LORA[$task]}" ;;
    hns) echo "${HNS[$task]}" ;;
    scalar_shrink) echo "$RUN_BASE/$task/adapters/scalar-shrink" ;;
    shape_only) echo "$RUN_BASE/$task/adapters/shape-only" ;;
    head_only) echo "$RUN_BASE/$task/adapters/head-only" ;;
    tail_only) echo "$RUN_BASE/$task/adapters/tail-only" ;;
  esac
}

eval_magicoder() {
  local task=magicoder root="$RUN_BASE/magicoder"
  # Evaluate all six variants with the same current tokenizer/template path.
  # Historical LoRA/HNS metrics used an older prompt fallback and are retained
  # in the source adapters, but are not mixed with these causal controls.
  for label in lora scalar_shrink shape_only head_only tail_only hns; do
    local adapter output
    adapter="$(adapter_for "$task" "$label")"
    output="$root/eval/$label/humaneval"
    if [[ ! -f "$output/metrics.json" ]]; then
      "$PYTHON" -m finetune.eval.eval_humaneval \
        --base_model "$BASE" --adapter_dir "$adapter" --config_src "${LORA[$task]}" \
        --dataset_path "$DATA_ROOT/humaneval-test.parquet" --split test --output_dir "$output" \
        --prompt_style chat --chat_user_prompt_style opencompass --chat_template_mode auto \
        --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
        --vllm_attention_backend FLASH_ATTN --vllm_disable_flashinfer_sampler \
        --vllm_request_batch_size 32 --max_new_tokens 512 --timeout_s 3.0 \
        --eval_n_workers 16 --seed 42
    fi
    output="$root/eval/$label/mbpp"
    if [[ ! -f "$output/metrics.json" ]]; then
      "$PYTHON" -m finetune.eval.eval_mbpp \
        --base_model "$BASE" --adapter_dir "$adapter" --config_src "${LORA[$task]}" \
        --dataset_path "$DATA_ROOT/mbpp-sanitized-test.parquet" --split test --output_dir "$output" \
        --prompt_style chat --chat_template_mode auto \
        --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
        --vllm_attention_backend FLASH_ATTN --vllm_disable_flashinfer_sampler \
        --vllm_request_batch_size 32 --max_new_tokens 512 --timeout_s 3.0 \
        --eval_n_workers 16 --seed 42
    fi
    "$PYTHON" scripts/summarize_spectral_ablation_eval.py "$root"
  done
}

eval_metamath() {
  local task=metamath root="$RUN_BASE/metamath"
  for label in lora scalar_shrink shape_only head_only tail_only hns; do
    local adapter output
    adapter="$(adapter_for "$task" "$label")"
    output="$root/eval/$label/gsm8k"
    if [[ ! -f "$output/metrics.json" ]]; then
      "$PYTHON" -m finetune.eval.eval_gsm8k \
        --base_model "$BASE" --adapter_dir "$adapter" --output_dir "$output" \
        --dataset_path "$FIXED_DATA/gsm8k-test.jsonl" --dataset_config main --split test \
        --max_new_tokens 2048 --dtype bf16 --seed 42 --chat_template_mode auto \
        --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
        --vllm_gpu_memory_utilization 0.85 --vllm_attention_backend FLASH_ATTN \
        --vllm_disable_flashinfer_sampler
    fi
    "$PYTHON" scripts/summarize_cross_task_task.py "$root" --task metamath
  done
}

eval_tulu() {
  local task=tulu root="$RUN_BASE/tulu"
  for label in lora scalar_shrink shape_only head_only tail_only hns; do
    local adapter output
    adapter="$(adapter_for "$task" "$label")"
    output="$root/eval/$label/ifeval"
    if [[ ! -f "$output/metrics.json" ]]; then
      "$PYTHON" -m finetune.eval.eval_ifeval \
        --base_model "$BASE" --adapter_dir "$adapter" --output_dir "$output" \
        --dataset_path "$FIXED_DATA/ifeval-train.jsonl" --split train \
        --max_new_tokens 2048 --dtype bf16 --seed 42 --chat_template_mode auto \
        --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
        --vllm_gpu_memory_utilization 0.85 --vllm_attention_backend FLASH_ATTN \
        --vllm_disable_flashinfer_sampler --vllm_request_batch_size 128 --per_category_metrics
    fi
    "$PYTHON" scripts/summarize_cross_task_task.py "$root" --task tulu
  done
}

eval_commonsense() {
  local task=commonsense root="$RUN_BASE/commonsense"
  mkdir -p "$root/eval/lora" "$root/eval/hns"
  for label in lora hns; do
    local source
    source="$(adapter_for "$task" "$label")/eval-commonsense8"
    [[ -f "$root/eval/$label/commonsense/summary.json" ]] || cp -a "$source" "$root/eval/$label/commonsense"
  done
  # The benchmark datasets are cached after the first evaluation. The base model
  # is an absolute local path, while Hub access remains available for a cold
  # benchmark-data cache.
  for label in scalar_shrink shape_only head_only tail_only; do
    local adapter output
    adapter="$(adapter_for "$task" "$label")"
    output="$root/eval/$label/commonsense"
    if [[ ! -f "$output/summary.json" ]]; then
      "$PYTHON" scripts/eval_commonsense_8tasks.py \
        --base_model "$BASE" --adapter_dir "$adapter" --output_dir "$output" \
        --tasks all --backend vllm --chat_template_mode auto --max_new_tokens 8 \
        --request_batch_size 256 --tensor_parallel_size 1 --vllm_max_model_len 2048 \
        --vllm_gpu_memory_utilization 0.90 --vllm_attention_backend FLASH_ATTN \
        --dtype bf16 --seed 42
    fi
    "$PYTHON" scripts/summarize_cross_task_task.py "$root" --task commonsense
  done
}

for task in magicoder metamath tulu commonsense; do
  echo "[Prepare] $task"
  prepare_task "$task"
done

eval_magicoder
eval_metamath
eval_tulu
eval_commonsense

echo "[Done] Llama 3.1 8B cross-task mechanism: $RUN_BASE"
