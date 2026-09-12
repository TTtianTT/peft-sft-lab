#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE_QWEN=/dataset1/zailong/models/Qwen3-8B
BASE_LLAMA=/dataset1/zailong/models/Llama-3.1-8B-Instruct
MODEL_ROOT=/dataset1/zailong/models/spectral-surgery
DATA_ROOT=/dataset1/zailong/data/peft-sft-lab
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-mechanism-first-batch-20260909

export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export PATH="$(dirname "$PYTHON"):$PATH"

build_control() {
  local name="$1" lora="$2" hns="$3"
  local root="$RUN_ROOT/headonly_fro_restore/$name"
  local adapter="$root/adapter"
  mkdir -p "$root"
  if [[ ! -f "$adapter/spectral_control_meta.json" ]]; then
    "$PYTHON" scripts/build_headonly_fro_restore.py \
      --lora_path "$lora" --hns_path "$hns" --out_dir "$adapter" --device cuda
  fi
  if [[ ! -f "$root/spectra/spectrum_summary.json" ]]; then
    "$PYTHON" scripts/analyze_spectral_pair.py \
      --lora_path "$lora" --hns_path "$adapter" --output_dir "$root/spectra" --device cuda
  fi
}

eval_ifeval() {
  local name="$1" base="$2" lora="$3" mode="$4"
  local root="$RUN_ROOT/headonly_fro_restore/$name"
  if [[ ! -f "$root/eval/ifeval/metrics.json" ]]; then
    "$PYTHON" -m finetune.eval.eval_ifeval \
      --base_model "$base" --adapter_dir "$root/adapter" --output_dir "$root/eval/ifeval" \
      --dataset_path "$DATA_ROOT/cross-task-mechanism/ifeval-train.jsonl" --split train \
      --max_new_tokens 2048 --dtype bf16 --seed 42 --chat_template_mode "$mode" \
      --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
      --vllm_gpu_memory_utilization 0.85 --vllm_attention_backend FLASH_ATTN \
      --vllm_disable_flashinfer_sampler --vllm_request_batch_size 128 --per_category_metrics
  fi
}

eval_commonsense() {
  local name="$1" base="$2" mode="$3"
  local root="$RUN_ROOT/headonly_fro_restore/$name"
  if [[ ! -f "$root/eval/commonsense/summary.json" ]]; then
    "$PYTHON" scripts/eval_commonsense_8tasks.py \
      --base_model "$base" --adapter_dir "$root/adapter" --output_dir "$root/eval/commonsense" \
      --tasks all --backend vllm --chat_template_mode "$mode" --max_new_tokens 8 \
      --request_batch_size 256 --tensor_parallel_size 1 --vllm_max_model_len 2048 \
      --vllm_gpu_memory_utilization 0.90 --vllm_attention_backend FLASH_ATTN \
      --dtype bf16 --seed 42
  fi
}

eval_magicoder() {
  local name="$1" base="$2" lora="$3" mode="$4"
  local root="$RUN_ROOT/headonly_fro_restore/$name"
  if [[ ! -f "$root/eval/humaneval/metrics.json" ]]; then
    "$PYTHON" -m finetune.eval.eval_humaneval \
      --base_model "$base" --adapter_dir "$root/adapter" --config_src "$lora" \
      --dataset_path "$DATA_ROOT/humaneval-test.parquet" --split test \
      --output_dir "$root/eval/humaneval" --prompt_style chat \
      --chat_user_prompt_style opencompass --chat_template_mode "$mode" \
      --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
      --vllm_attention_backend FLASH_ATTN --vllm_disable_flashinfer_sampler \
      --vllm_request_batch_size 32 --max_new_tokens 512 --timeout_s 3.0 \
      --eval_n_workers 16 --seed 42
  fi
  if [[ ! -f "$root/eval/mbpp/metrics.json" ]]; then
    "$PYTHON" -m finetune.eval.eval_mbpp \
      --base_model "$base" --adapter_dir "$root/adapter" --config_src "$lora" \
      --dataset_path "$DATA_ROOT/mbpp-sanitized-test.parquet" --split test \
      --output_dir "$root/eval/mbpp" --prompt_style chat --chat_template_mode "$mode" \
      --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
      --vllm_attention_backend FLASH_ATTN --vllm_disable_flashinfer_sampler \
      --vllm_request_batch_size 32 --max_new_tokens 512 --timeout_s 3.0 \
      --eval_n_workers 16 --seed 42
  fi
}

eval_metamath() {
  local name="$1" base="$2" mode="$3"
  local root="$RUN_ROOT/headonly_fro_restore/$name"
  if [[ ! -f "$root/eval/gsm8k/metrics.json" ]]; then
    "$PYTHON" -m finetune.eval.eval_gsm8k \
      --base_model "$base" --adapter_dir "$root/adapter" --output_dir "$root/eval/gsm8k" \
      --dataset_path "$DATA_ROOT/cross-task-mechanism/gsm8k-test.jsonl" \
      --dataset_config main --split test --max_new_tokens 2048 --dtype bf16 --seed 42 \
      --chat_template_mode "$mode" --use_vllm --tensor_parallel_size 1 \
      --vllm_max_model_len 4096 --vllm_gpu_memory_utilization 0.85 \
      --vllm_attention_backend FLASH_ATTN --vllm_disable_flashinfer_sampler
  fi
}

QWEN_TULU_LORA="$MODEL_ROOT/Qwen3-8B-InstructionFollowing-LoRA"
QWEN_TULU_HNS="$MODEL_ROOT/Qwen3-8B-InstructionFollowing-SpectralSurgery-HNS4p1"
build_control qwen_tulu "$QWEN_TULU_LORA" "$QWEN_TULU_HNS"
eval_ifeval qwen_tulu "$BASE_QWEN" "$QWEN_TULU_LORA" non_thinking

QWEN_COMMON_LORA="$MODEL_ROOT/Qwen3-8B-CommonSense170K-LoRA"
QWEN_COMMON_HNS=/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Qwen3-8B/commonsense/hns-allmodules-8plus2
build_control qwen_commonsense "$QWEN_COMMON_LORA" "$QWEN_COMMON_HNS"
eval_commonsense qwen_commonsense "$BASE_QWEN" non_thinking

LLAMA_MAGIC_LORA="$MODEL_ROOT/Llama-3.1-8B-Instruct-Magicoder-50K-LoRA-E1"
LLAMA_MAGIC_HNS="$MODEL_ROOT/Llama-3.1-8B-Instruct-Magicoder-50K-SpectralSurgery-HNS4p1"
build_control llama_magicoder "$LLAMA_MAGIC_LORA" "$LLAMA_MAGIC_HNS"
eval_magicoder llama_magicoder "$BASE_LLAMA" "$LLAMA_MAGIC_LORA" auto

LLAMA_MATH_LORA="$MODEL_ROOT/Llama-3.1-8B-Instruct-MetaMathQA-50K-LoRA"
LLAMA_MATH_HNS="$MODEL_ROOT/Llama-3.1-8B-Instruct-MetaMathQA-50K-SpectralSurgery-HNS4p1-AllMods"
build_control llama_metamath "$LLAMA_MATH_LORA" "$LLAMA_MATH_HNS"
eval_metamath llama_metamath "$BASE_LLAMA" auto

LLAMA_COMMON_LORA="$MODEL_ROOT/Llama-3.1-8B-Instruct-CommonSense170K-LoRA-GBS64-Final"
LLAMA_COMMON_HNS="$MODEL_ROOT/Llama-3.1-8B-Instruct-CommonSense170K-Spectral-Surgery-AllModules-8Plus2"
build_control llama_commonsense "$LLAMA_COMMON_LORA" "$LLAMA_COMMON_HNS"
eval_commonsense llama_commonsense "$BASE_LLAMA" auto

echo "[Done] missing five HeadOnly+FrobeniusRestore controls"
