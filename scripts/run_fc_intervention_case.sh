#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {qwen_magicoder|qwen_commonsense|llama_tulu}" >&2
  exit 2
fi

CASE="$1"
PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
MODEL_ROOT=/dataset1/zailong/models/spectral-surgery
DATA_ROOT=/dataset1/zailong/data/peft-sft-lab
AGGREGATE_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-2x4-allmodules-20260909
GRADIENT_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-module-utility-20260909/gradient
FUNCTIONAL_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-functional-localization-20260909
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-fc-interventions-20260910

export PYTHONPATH="$PWD/src:$PWD/scripts"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export PATH="$(dirname "$PYTHON"):$PATH"

case "$CASE" in
  qwen_magicoder)
    BASE=/dataset1/zailong/models/Qwen3-8B
    BASE_LABEL=Qwen3-8B
    TASK=magicoder
    LORA="$MODEL_ROOT/Qwen3-8B-Magicoder-50K-LoRA-E1"
    HNS="$MODEL_ROOT/Qwen3-8B-Magicoder-50K-SpectralSurgery-HNS4p1-AllMods"
    DATA="$DATA_ROOT/magicoder-train.parquet"
    DATASET_NAME=ise-uiuc/Magicoder-Evol-Instruct-110K
    CHAT_MODE=non_thinking
    ;;
  qwen_commonsense)
    BASE=/dataset1/zailong/models/Qwen3-8B
    BASE_LABEL=Qwen3-8B
    TASK=commonsense
    LORA="$MODEL_ROOT/Qwen3-8B-CommonSense170K-LoRA"
    HNS=/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Qwen3-8B/commonsense/hns-allmodules-8plus2
    DATA="$DATA_ROOT/commonsense170k-train.parquet"
    DATASET_NAME=commonsense170k
    CHAT_MODE=non_thinking
    ;;
  llama_tulu)
    BASE=/dataset1/zailong/models/Llama-3.1-8B-Instruct
    BASE_LABEL=Llama-3.1-8B-Instruct
    TASK=tulu
    LORA="$MODEL_ROOT/Llama-3.1-8B-Instruct-InstructionFollowing-LoRA"
    HNS=/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Llama-3.1-8B-Instruct/tulu/hns-allmodules-8plus2
    DATA="$DATA_ROOT/cross-task-mechanism/tulu-3-sft-personas-instruction-following-train.parquet"
    DATASET_NAME=tulu_if
    CHAT_MODE=auto
    ;;
  *)
    echo "unknown case: $CASE" >&2
    exit 2
    ;;
esac

CASE_ROOT="$RUN_ROOT/$CASE"
ADAPTER_ROOT="$CASE_ROOT/adapters"
mkdir -p "$CASE_ROOT/eval"
if [[ ! -f "$ADAPTER_ROOT/manifest.json" ]]; then
  "$PYTHON" scripts/build_functional_compatibility_adapters.py \
    --lora_path "$LORA" --hns_path "$HNS" \
    --direction_response "$AGGREGATE_ROOT/direction_response.csv" \
    --gradient_meta "$GRADIENT_ROOT/$CASE/spectral_edit_meta.json" \
    --base_model_label "$BASE_LABEL" --task "$TASK" --out_root "$ADAPTER_ROOT" \
    --scopes 0.25 0.50 --num_layer_bins 4
else
  echo "[Resume] $ADAPTER_ROOT/manifest.json"
fi

mapfile -t LABELS < <("$PYTHON" - "$ADAPTER_ROOT/manifest.json" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["variants"]:
    print(row["label"])
PY
)

for label in "${LABELS[@]}"; do
  adapter="$ADAPTER_ROOT/$label"
  echo "[Eval] $CASE/$label"
  if [[ "$CASE" == qwen_magicoder ]]; then
    if [[ ! -f "$CASE_ROOT/eval/$label/humaneval/metrics.json" ]]; then
      "$PYTHON" -m finetune.eval.eval_humaneval \
        --base_model "$BASE" --adapter_dir "$adapter" --config_src "$LORA" \
        --dataset_path "$DATA_ROOT/humaneval-test.parquet" --split test \
        --output_dir "$CASE_ROOT/eval/$label/humaneval" \
        --prompt_style chat --chat_user_prompt_style opencompass \
        --chat_template_mode non_thinking --use_vllm --tensor_parallel_size 1 \
        --vllm_max_model_len 4096 --vllm_attention_backend FLASH_ATTN \
        --vllm_disable_flashinfer_sampler --vllm_request_batch_size 32 \
        --max_new_tokens 512 --timeout_s 3.0 --eval_n_workers 16 --seed 42
    fi
    if [[ ! -f "$CASE_ROOT/eval/$label/mbpp/metrics.json" ]]; then
      "$PYTHON" -m finetune.eval.eval_mbpp \
        --base_model "$BASE" --adapter_dir "$adapter" --config_src "$LORA" \
        --dataset_path "$DATA_ROOT/mbpp-sanitized-test.parquet" --split test \
        --output_dir "$CASE_ROOT/eval/$label/mbpp" --prompt_style chat \
        --chat_template_mode non_thinking --use_vllm --tensor_parallel_size 1 \
        --vllm_max_model_len 4096 --vllm_attention_backend FLASH_ATTN \
        --vllm_disable_flashinfer_sampler --vllm_request_batch_size 32 \
        --max_new_tokens 512 --timeout_s 3.0 --eval_n_workers 16 --seed 42
    fi
  elif [[ "$CASE" == qwen_commonsense ]]; then
    if [[ ! -f "$CASE_ROOT/eval/$label/commonsense/summary.json" ]]; then
      "$PYTHON" scripts/eval_commonsense_8tasks.py \
        --base_model "$BASE" --adapter_dir "$adapter" \
        --output_dir "$CASE_ROOT/eval/$label/commonsense" --tasks all \
        --backend vllm --chat_template_mode non_thinking --max_new_tokens 8 \
        --request_batch_size 256 --tensor_parallel_size 1 --vllm_max_model_len 2048 \
        --vllm_gpu_memory_utilization 0.90 --vllm_attention_backend FLASH_ATTN \
        --dtype bf16 --seed 42
    fi
  else
    if [[ ! -f "$CASE_ROOT/eval/$label/ifeval/metrics.json" ]]; then
      "$PYTHON" -m finetune.eval.eval_ifeval \
        --base_model "$BASE" --adapter_dir "$adapter" \
        --output_dir "$CASE_ROOT/eval/$label/ifeval" \
        --dataset_path "$DATA_ROOT/cross-task-mechanism/ifeval-train.jsonl" --split train \
        --max_new_tokens 2048 --dtype bf16 --seed 42 --chat_template_mode auto \
        --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
        --vllm_gpu_memory_utilization 0.85 --vllm_attention_backend FLASH_ATTN \
        --vllm_disable_flashinfer_sampler --vllm_request_batch_size 128 \
        --per_category_metrics
    fi
  fi
done

"$PYTHON" scripts/summarize_functional_localization.py --run_root "$RUN_ROOT" --case "$CASE"

if [[ ! -f "$CASE_ROOT/nll/adapter_nll.tsv" ]]; then
  "$PYTHON" scripts/measure_adapter_set_nll.py \
    --base_model "$BASE" --lora_path "$LORA" \
    --dataset_path "$DATA" --dataset_name "$DATASET_NAME" \
    --output_dir "$CASE_ROOT/nll" --samples 64 --sample_start 64 \
    --batch_size 4 --max_seq_len 512 --seed 42 --dtype bf16 \
    --chat_template_mode "$CHAT_MODE" \
    --adapter "full_hns=$HNS" \
    --adapter "functional_top50=$FUNCTIONAL_ROOT/$CASE/adapters/functional_top50" \
    --adapter "compatibility_top50=$ADAPTER_ROOT/compatibility_top50" \
    --adapter "high_f_high_c_top50=$ADAPTER_ROOT/high_f_high_c_top50"
fi

echo "[Done] F x C $CASE"
