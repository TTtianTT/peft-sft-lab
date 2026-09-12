#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE=/dataset1/zailong/models/Qwen3-8B
LORA=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-LoRA
HNS=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-Spectral-Surgery-AllModules-4Plus1
GSM8K=/dataset1/zailong/data/peft-sft-lab/cross-task-mechanism/gsm8k-test.jsonl
TRAIN=/dataset1/zailong/data/peft-sft-lab/metamathqa-train.parquet
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-signed-gates-20260910/qwen_metamath_reward_paths

export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$RUN_ROOT/rollouts" "$RUN_ROOT/gradients"

"$PYTHON" scripts/generate_metamath_reward_rollouts.py \
  --base_model "$BASE" \
  --lora_path "$LORA" \
  --dataset_path "$GSM8K" \
  --metamath_train "$TRAIN" \
  --output_dir "$RUN_ROOT/rollouts" \
  --questions 128 \
  --rollouts 4 \
  --seed 20260910 \
  --temperature 0.7 \
  --top_p 1.0 \
  --max_new_tokens 2048 \
  --max_model_len 4096 \
  --gpu_memory_utilization 0.85

"$PYTHON" scripts/measure_metamath_reward_path_gradients.py \
  --base_model "$BASE" \
  --lora_path "$LORA" \
  --hns_path "$HNS" \
  --rollouts "$RUN_ROOT/rollouts/rollouts.jsonl" \
  --output_dir "$RUN_ROOT/gradients" \
  --dtype fp32 \
  --temperature 0.7 \
  --expected_rollouts 4 \
  --smoke_samples 8 \
  --finite_difference_steps 0.01 0.05 \
  --finite_difference_rtol 0.35 \
  --bootstrap 10000 \
  --seed 20260910

echo "[Done] Qwen MetaMath reward-path pilot: $RUN_ROOT"
