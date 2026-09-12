#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE=/dataset1/zailong/models/Qwen3-8B
LORA=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-LoRA
HNS=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-Spectral-Surgery-AllModules-4Plus1
SOURCE_SPLITS=/dataset1/zailong/runs/peft-sft-lab/hns-metamath-direct-dose-20260910/splits
CPU_AUDIT="$PWD/reports/hns_posthoc_lora_normalization_20260910"
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-posthoc-scaling-pilot-20260910/qwen_metamath
ADAPTERS="$RUN_ROOT/adapters"
EVAL_ROOT="$RUN_ROOT/eval"
ANALYSIS="$RUN_ROOT/analysis"

export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1

mkdir -p "$RUN_ROOT" "$EVAL_ROOT/calibration" "$EVAL_ROOT/validation" \
  "$ANALYSIS/calibration" "$ANALYSIS/validation"

"$PYTHON" scripts/build_posthoc_spectral_scaling_adapters.py \
  --lora_path "$LORA" \
  --module_scaling "$CPU_AUDIT/module_scaling.csv" \
  --shuffle_plan "$CPU_AUDIT/shuffle_scaling_plan.csv" \
  --base_model_label Qwen3-8B \
  --task metamath \
  --output_dir "$ADAPTERS"

GLOBAL="$ADAPTERS/global-norm-matched"
PER_MODULE="$ADAPTERS/per-module-spectral-scale"
SHUFFLE_1="$ADAPTERS/shuffled-scale-1"
SHUFFLE_2="$ADAPTERS/shuffled-scale-2"
SHUFFLE_3="$ADAPTERS/shuffled-scale-3"

for split in calibration validation; do
  "$PYTHON" scripts/eval_gsm8k_adapter_set.py \
    --base_model "$BASE" \
    --dataset_path "$SOURCE_SPLITS/$split.jsonl" \
    --output_dir "$EVAL_ROOT/$split" \
    --variant "lora=$LORA" \
    --variant "global_norm_matched=$GLOBAL" \
    --variant "per_module_spectral_scale=$PER_MODULE" \
    --variant "shuffled_scale_1=$SHUFFLE_1" \
    --variant "shuffled_scale_2=$SHUFFLE_2" \
    --variant "shuffled_scale_3=$SHUFFLE_3" \
    --variant "full_hns=$HNS" \
    --max_new_tokens 2048 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.85 \
    --seed 42

  "$PYTHON" scripts/analyze_posthoc_spectral_scaling_pilot.py \
    --stage "$split" \
    --eval_dir "$EVAL_ROOT/$split" \
    --adapter_manifest "$ADAPTERS/manifest.json" \
    --output_dir "$ANALYSIS/$split" \
    --bootstrap 20000 \
    --permutation 50000 \
    --seed 20260911
done

cp "$CPU_AUDIT/pilot_protocol.md" "$RUN_ROOT/protocol.md"
cp "$CPU_AUDIT/shuffle_scaling_plan.csv" "$RUN_ROOT/shuffle_scaling_plan.csv"
echo "[Done] Post-hoc spectral-scaling pilot: $RUN_ROOT"
