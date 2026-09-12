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
mkdir -p "$RUN_ROOT"

build_and_measure() {
  local name="$1" base="$2" lora="$3" hns="$4" data="$5" kind="$6"
  local root="$RUN_ROOT/headonly_fro_restore/$name" adapter="$RUN_ROOT/headonly_fro_restore/$name/adapter"
  mkdir -p "$root"
  if [[ ! -f "$adapter/spectral_control_meta.json" ]]; then
    "$PYTHON" scripts/build_headonly_fro_restore.py \
      --lora_path "$lora" --hns_path "$hns" --out_dir "$adapter" --device cuda
  fi
  if [[ ! -f "$root/spectra/spectrum_summary.json" ]]; then
    "$PYTHON" scripts/analyze_spectral_pair.py \
      --lora_path "$lora" --hns_path "$adapter" --output_dir "$root/spectra" --device cuda
  fi
  if [[ ! -f "$root/activation/activation_analysis.json" ]]; then
    "$PYTHON" scripts/measure_lora_modification.py \
      --base_model "$base" --lora_path "$lora" --hns_path "$hns" \
      --control "head_fro_restore=$adapter" \
      --dataset_path "$data" --dataset_kind "$kind" --samples 256 \
      --batch_size 2 --max_seq_len 512 --seed 42 --dtype bf16 \
      --output_dir "$root/activation"
  fi
}

QWEN_MAGIC_LORA="$MODEL_ROOT/Qwen3-8B-Magicoder-50K-LoRA-E1"
QWEN_MAGIC_HNS="$MODEL_ROOT/Qwen3-8B-Magicoder-50K-SpectralSurgery-HNS4p1-AllMods"
build_and_measure qwen_magicoder "$BASE_QWEN" "$QWEN_MAGIC_LORA" "$QWEN_MAGIC_HNS" \
  "$DATA_ROOT/magicoder-train.parquet" magicoder
MAGIC_ROOT="$RUN_ROOT/headonly_fro_restore/qwen_magicoder"
if [[ ! -f "$MAGIC_ROOT/eval/humaneval/metrics.json" ]]; then
  "$PYTHON" -m finetune.eval.eval_humaneval \
    --base_model "$BASE_QWEN" --adapter_dir "$MAGIC_ROOT/adapter" --config_src "$QWEN_MAGIC_LORA" \
    --dataset_path "$DATA_ROOT/humaneval-test.parquet" --split test --output_dir "$MAGIC_ROOT/eval/humaneval" \
    --prompt_style chat --chat_user_prompt_style opencompass --chat_template_mode non_thinking \
    --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size 32 --max_new_tokens 512 --timeout_s 3.0 \
    --eval_n_workers 16 --seed 42
fi
if [[ ! -f "$MAGIC_ROOT/eval/mbpp/metrics.json" ]]; then
  "$PYTHON" -m finetune.eval.eval_mbpp \
    --base_model "$BASE_QWEN" --adapter_dir "$MAGIC_ROOT/adapter" --config_src "$QWEN_MAGIC_LORA" \
    --dataset_path "$DATA_ROOT/mbpp-sanitized-test.parquet" --split test --output_dir "$MAGIC_ROOT/eval/mbpp" \
    --prompt_style chat --chat_template_mode non_thinking \
    --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size 32 --max_new_tokens 512 --timeout_s 3.0 \
    --eval_n_workers 16 --seed 42
fi

QWEN_MATH_LORA="$MODEL_ROOT/Qwen3-8B-MetaMathQA-50K-LoRA"
QWEN_MATH_HNS="$MODEL_ROOT/Qwen3-8B-MetaMathQA-50K-Spectral-Surgery-AllModules-4Plus1"
build_and_measure qwen_metamath "$BASE_QWEN" "$QWEN_MATH_LORA" "$QWEN_MATH_HNS" \
  "$DATA_ROOT/metamathqa-train.parquet" metamath
MATH_ROOT="$RUN_ROOT/headonly_fro_restore/qwen_metamath"
if [[ ! -f "$MATH_ROOT/eval/gsm8k/metrics.json" ]]; then
  "$PYTHON" -m finetune.eval.eval_gsm8k \
    --base_model "$BASE_QWEN" --adapter_dir "$MATH_ROOT/adapter" --output_dir "$MATH_ROOT/eval/gsm8k" \
    --dataset_path "$DATA_ROOT/cross-task-mechanism/gsm8k-test.jsonl" --dataset_config main --split test \
    --max_new_tokens 2048 --dtype bf16 --seed 42 --chat_template_mode non_thinking \
    --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.85 --vllm_attention_backend FLASH_ATTN \
    --vllm_disable_flashinfer_sampler
fi

LLAMA_TULU_LORA="$MODEL_ROOT/Llama-3.1-8B-Instruct-InstructionFollowing-LoRA"
LLAMA_TULU_HNS=/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Llama-3.1-8B-Instruct/tulu/hns-allmodules-8plus2
build_and_measure llama_tulu "$BASE_LLAMA" "$LLAMA_TULU_LORA" "$LLAMA_TULU_HNS" \
  "$DATA_ROOT/cross-task-mechanism/tulu-3-sft-personas-instruction-following-train.parquet" tulu
TULU_ROOT="$RUN_ROOT/headonly_fro_restore/llama_tulu"
if [[ ! -f "$TULU_ROOT/eval/ifeval/metrics.json" ]]; then
  "$PYTHON" -m finetune.eval.eval_ifeval \
    --base_model "$BASE_LLAMA" --adapter_dir "$TULU_ROOT/adapter" --output_dir "$TULU_ROOT/eval/ifeval" \
    --dataset_path "$DATA_ROOT/cross-task-mechanism/ifeval-train.jsonl" --split train \
    --max_new_tokens 2048 --dtype bf16 --seed 42 --chat_template_mode auto \
    --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
    --vllm_gpu_memory_utilization 0.85 --vllm_attention_backend FLASH_ATTN \
    --vllm_disable_flashinfer_sampler --vllm_request_batch_size 128 --per_category_metrics
fi

# The historical Llama Commonsense LoRA/HNS directories contained only suite
# summaries. Re-evaluate those two exact adapters to make paired inference possible.
LLAMA_COMMON_LORA="$MODEL_ROOT/Llama-3.1-8B-Instruct-CommonSense170K-LoRA-GBS64-Final"
LLAMA_COMMON_HNS="$MODEL_ROOT/Llama-3.1-8B-Instruct-CommonSense170K-Spectral-Surgery-AllModules-8Plus2"
for label in lora hns; do
  adapter="$LLAMA_COMMON_LORA"
  [[ "$label" == hns ]] && adapter="$LLAMA_COMMON_HNS"
  output="$RUN_ROOT/paired_assets/llama_commonsense/$label/commonsense"
  if [[ ! -f "$output/summary.json" ]]; then
    "$PYTHON" scripts/eval_commonsense_8tasks.py \
      --base_model "$BASE_LLAMA" --adapter_dir "$adapter" --output_dir "$output" \
      --tasks all --backend vllm --chat_template_mode auto --max_new_tokens 8 \
      --request_batch_size 256 --tensor_parallel_size 1 --vllm_max_model_len 2048 \
      --vllm_gpu_memory_utilization 0.90 --vllm_attention_backend FLASH_ATTN \
      --dtype bf16 --seed 42
  fi
done

"$PYTHON" scripts/analyze_hns_paired_inference.py \
  --aggregate_root /dataset1/zailong/runs/peft-sft-lab/hns-2x4-allmodules-20260909 \
  --first_batch_root "$RUN_ROOT" \
  --output_dir "$RUN_ROOT/statistics" \
  --bootstrap_samples 10000 --permutation_samples 100000 --seed 42

echo "[Done] HNS first batch: $RUN_ROOT"
