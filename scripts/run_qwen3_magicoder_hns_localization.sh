#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/dataset1/zailong/workspace/peft-sft-lab}"
PYTHON="${PYTHON:-/dataset1/zailong/envs/peft-sft-lab/bin/python}"
BASE_MODEL="${BASE_MODEL:-/dataset1/zailong/models/Qwen3-8B}"
SOURCE_LORA="${SOURCE_LORA:-/dataset1/zailong/models/spectral-surgery/Qwen3-8B-Magicoder-50K-LoRA-E1}"
ALL_HNS="${ALL_HNS:-/dataset1/zailong/models/spectral-surgery/Qwen3-8B-Magicoder-50K-SpectralSurgery-HNS4p1-AllMods}"
MAGICODER_PATH="${MAGICODER_PATH:-/dataset1/zailong/data/peft-sft-lab/magicoder-train.parquet}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-/dataset1/zailong/data/peft-sft-lab/humaneval-test.parquet}"
MBPP_PATH="${MBPP_PATH:-/dataset1/zailong/data/peft-sft-lab/mbpp-sanitized-test.parquet}"
REFERENCE_RUN="${REFERENCE_RUN:-/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/magicoder-spectral-ablations-20260908}"
RUN_ROOT="${RUN_ROOT:-/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/magicoder-hns-localization-20260908}"
CALIB_SAMPLES="${CALIB_SAMPLES:-64}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-2}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-32}"

export HF_HOME="${HF_HOME:-/dataset1/zailong/cache/peft-sft-lab/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dataset1/zailong/cache/peft-sft-lab/xdg}"
export TORCH_HOME="${TORCH_HOME:-/dataset1/zailong/cache/peft-sft-lab/torch}"
export TMPDIR="${TMPDIR:-/dataset1/zailong/tmp/peft-sft-lab}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$PYTHON"):$PATH"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/adapters" "$RUN_ROOT/eval" "$TMPDIR"

LOCAL_ROOT="$RUN_ROOT/adapters/localized-hns4p1"
if [[ ! -f "$LOCAL_ROOT/manifest.json" ]]; then
  "$PYTHON" scripts/build_hns_localization_adapters.py \
    --lora_path "$SOURCE_LORA" \
    --out_root "$LOCAL_ROOT" \
    --device cuda \
    --fast_steps 4 \
    --stable_steps 1 \
    --num_layers 36
else
  echo "[Resume] Reusing $LOCAL_ROOT/manifest.json"
fi

GRADIENT_HNS="$RUN_ROOT/adapters/gradient-selected-hns"
if [[ ! -f "$GRADIENT_HNS/spectral_edit_meta.json" ]]; then
  "$PYTHON" -m finetune.spectral_edit.cli sensitivity-hns \
    --base_model "$BASE_MODEL" \
    --lora_path "$SOURCE_LORA" \
    --out_dir "$GRADIENT_HNS" \
    --target_modules all_modules \
    --module_budget 72 \
    --selection_rule importance_compatible \
    --calib_dataset ise-uiuc/Magicoder-Evol-Instruct-110K \
    --calib_dataset_path "$MAGICODER_PATH" \
    --calib_text_fields instruction response \
    --calib_samples "$CALIB_SAMPLES" \
    --calib_batch_size "$CALIB_BATCH_SIZE" \
    --calib_shuffle \
    --sft_format chat \
    --chat_template_mode non_thinking \
    --max_seq_len 2048 \
    --dtype bf16 \
    --fast_steps 4 \
    --stable_steps 1 \
    --seed 42
else
  echo "[Resume] Reusing $GRADIENT_HNS/spectral_edit_meta.json"
fi

labels=(
  control-lora
  control-hns4p1-allmods
  hns-attention
  hns-mlp
  hns-qkv
  hns-o-proj
  hns-gate-up
  hns-down-proj
  hns-layers-early
  hns-layers-middle
  hns-layers-late
  gradient-selected-hns
)
adapters=(
  "$SOURCE_LORA"
  "$ALL_HNS"
  "$LOCAL_ROOT/hns-attention"
  "$LOCAL_ROOT/hns-mlp"
  "$LOCAL_ROOT/hns-qkv"
  "$LOCAL_ROOT/hns-o-proj"
  "$LOCAL_ROOT/hns-gate-up"
  "$LOCAL_ROOT/hns-down-proj"
  "$LOCAL_ROOT/hns-layers-early"
  "$LOCAL_ROOT/hns-layers-middle"
  "$LOCAL_ROOT/hns-layers-late"
  "$GRADIENT_HNS"
)

"$PYTHON" - "$RUN_ROOT/experiment_paths.json" "${labels[@]}" -- "${adapters[@]}" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
sep = sys.argv.index("--")
labels, adapters = sys.argv[2:sep], sys.argv[sep + 1:]
out.write_text(json.dumps([{"label": x, "adapter_path": y} for x, y in zip(labels, adapters)], indent=2) + "\n")
PY

for label in control-lora control-hns4p1-allmods; do
  if [[ ! -d "$RUN_ROOT/eval/$label" && -d "$REFERENCE_RUN/eval/$label" ]]; then
    cp -a "$REFERENCE_RUN/eval/$label" "$RUN_ROOT/eval/$label"
    echo "[Reuse] Copied reference metrics for $label"
  fi
done

run_humaneval() {
  local label="$1" adapter="$2" output="$RUN_ROOT/eval/$1/humaneval"
  [[ -f "$output/metrics.json" ]] && { echo "[Resume] HumanEval $label"; return; }
  echo "[Eval] HumanEval $label"
  "$PYTHON" -m finetune.eval.eval_humaneval \
    --base_model "$BASE_MODEL" --adapter_dir "$adapter" --config_src "$SOURCE_LORA" \
    --dataset_path "$HUMANEVAL_PATH" --split test --output_dir "$output" \
    --prompt_style chat --chat_user_prompt_style opencompass --chat_template_mode non_thinking \
    --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size "$REQUEST_BATCH_SIZE" --max_new_tokens 512 \
    --timeout_s 3.0 --eval_n_workers 16 --seed 42
}

run_mbpp() {
  local label="$1" adapter="$2" output="$RUN_ROOT/eval/$1/mbpp"
  [[ -f "$output/metrics.json" ]] && { echo "[Resume] MBPP $label"; return; }
  echo "[Eval] MBPP $label"
  "$PYTHON" -m finetune.eval.eval_mbpp \
    --base_model "$BASE_MODEL" --adapter_dir "$adapter" --config_src "$SOURCE_LORA" \
    --dataset_path "$MBPP_PATH" --split test --output_dir "$output" \
    --prompt_style chat --chat_template_mode non_thinking \
    --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size "$REQUEST_BATCH_SIZE" --max_new_tokens 512 \
    --timeout_s 3.0 --eval_n_workers 16 --seed 42
}

for index in "${!labels[@]}"; do
  run_humaneval "${labels[$index]}" "${adapters[$index]}"
  run_mbpp "${labels[$index]}" "${adapters[$index]}"
  "$PYTHON" scripts/summarize_spectral_ablation_eval.py "$RUN_ROOT"
done

"$PYTHON" scripts/analyze_hns_localization.py "$RUN_ROOT"
echo "[Done] $RUN_ROOT/hns_localization_analysis.json"
