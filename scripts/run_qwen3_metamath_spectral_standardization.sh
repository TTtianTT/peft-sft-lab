#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE=/dataset1/zailong/models/Qwen3-8B
LORA=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-LoRA
HNS=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-Spectral-Surgery-AllModules-4Plus1
SPLIT=/dataset1/zailong/runs/peft-sft-lab/hns-metamath-direct-dose-20260910/splits/validation.jsonl
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-spectral-standardization-20260911/qwen_metamath
ADAPTERS="$RUN_ROOT/adapters"
SMOKE="$RUN_ROOT/smoke"
EVAL="$RUN_ROOT/eval"
ANALYSIS="$RUN_ROOT/analysis"

export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_BATCH_INVARIANT=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$RUN_ROOT" "$SMOKE" "$EVAL" "$ANALYSIS"

"$PYTHON" scripts/build_standardized_spectral_adapters.py \
  --lora_path "$LORA" --hns_path "$HNS" --output_dir "$ADAPTERS" --device cuda

VARIANTS=(
  --variant "original_lora=$LORA"
  --variant "zero_rebuild=$ADAPTERS/zero-rebuild"
  --variant "global_0p40=$ADAPTERS/global-0p40"
  --variant "common_hns=$ADAPTERS/common-hns"
  --variant "zscore_variance=$ADAPTERS/zscore-variance"
  --variant "zscore_std=$ADAPTERS/zscore-std"
)

"$PYTHON" scripts/eval_gsm8k_deterministic_adapter_set.py \
  --base_model "$BASE" --dataset_path "$SPLIT" --output_dir "$SMOKE" \
  "${VARIANTS[@]}" --max_samples 32 \
  --max_new_tokens 2048 --max_model_len 4096 --max_num_seqs 64 --seed 42

"$PYTHON" scripts/eval_gsm8k_deterministic_adapter_set.py \
  --base_model "$BASE" --dataset_path "$SPLIT" --output_dir "$EVAL" \
  "${VARIANTS[@]}" \
  --max_new_tokens 2048 --max_model_len 4096 --max_num_seqs 64 --seed 42

"$PYTHON" scripts/analyze_standardized_spectral_eval.py \
  --eval_dir "$EVAL" --adapter_manifest "$ADAPTERS/manifest.json" \
  --output_dir "$ANALYSIS" --bootstrap 20000 --seed 20260911

echo "[Done] Spectral standardization evaluation: $RUN_ROOT"
