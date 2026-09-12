#!/usr/bin/env bash
set -euo pipefail

# Run this script inside one allocated GPU process, for example:
#   srun --partition=B300q --gres=gpu:1 --cpus-per-task=16 --mem=128G \
#     --time=08:00:00 bash scripts/run_qwen3_magicoder_spectral_ablations.sh

REPO_ROOT="${REPO_ROOT:-/dataset1/zailong/workspace/peft-sft-lab}"
PYTHON="${PYTHON:-/dataset1/zailong/envs/peft-sft-lab/bin/python}"
BASE_MODEL="${BASE_MODEL:-/dataset1/zailong/models/Qwen3-8B}"
SOURCE_LORA="${SOURCE_LORA:-/dataset1/zailong/models/spectral-surgery/Qwen3-8B-Magicoder-50K-LoRA-E1}"
HNS_ADAPTER="${HNS_ADAPTER:-/dataset1/zailong/models/spectral-surgery/Qwen3-8B-Magicoder-50K-SpectralSurgery-HNS4p1-AllMods}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-/dataset1/zailong/data/peft-sft-lab/humaneval-test.parquet}"
MBPP_PATH="${MBPP_PATH:-/dataset1/zailong/data/peft-sft-lab/mbpp-sanitized-test.parquet}"
RUN_ROOT="${RUN_ROOT:-/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/magicoder-spectral-ablations-20260908}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

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
mkdir -p "$RUN_ROOT" "$RUN_ROOT/eval" "$TMPDIR"

for required in \
  "$BASE_MODEL/config.json" \
  "$SOURCE_LORA/adapter_config.json" \
  "$SOURCE_LORA/adapter_model.safetensors" \
  "$HNS_ADAPTER/adapter_model.safetensors" \
  "$HUMANEVAL_PATH" \
  "$MBPP_PATH"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

echo "[Environment] node=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" -c 'import torch, vllm; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} vllm={vllm.__version__}")'

ADAPTER_ROOT="$RUN_ROOT/adapters"
if [[ ! -f "$ADAPTER_ROOT/manifest.json" ]]; then
  "$PYTHON" scripts/build_spectral_ablation_adapters.py \
    --lora_path "$SOURCE_LORA" \
    --out_root "$ADAPTER_ROOT" \
    --device cuda \
    --target_modules all \
    --temperatures 1.0 0.75 0.5 0.25 0.0
else
  echo "[Resume] Reusing $ADAPTER_ROOT/manifest.json"
fi

labels=(
  control-lora
  control-hns4p1-allmods
  scalar-shrink-fro-match-flat-nuclear
  exact-flat-nuclear
  top-shrink-to-mean
  tail-lift-to-mean
  temperature-tau-1p00
  temperature-tau-0p75
  temperature-tau-0p50
  temperature-tau-0p25
  temperature-tau-0p00
)
adapters=(
  "$SOURCE_LORA"
  "$HNS_ADAPTER"
  "$ADAPTER_ROOT/scalar-shrink-fro-match-flat-nuclear"
  "$ADAPTER_ROOT/exact-flat-nuclear"
  "$ADAPTER_ROOT/top-shrink-to-mean"
  "$ADAPTER_ROOT/tail-lift-to-mean"
  "$ADAPTER_ROOT/temperature-tau-1p00"
  "$ADAPTER_ROOT/temperature-tau-0p75"
  "$ADAPTER_ROOT/temperature-tau-0p50"
  "$ADAPTER_ROOT/temperature-tau-0p25"
  "$ADAPTER_ROOT/temperature-tau-0p00"
)

"$PYTHON" - "$RUN_ROOT/experiment_paths.json" "${labels[@]}" -- "${adapters[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
separator = sys.argv.index("--")
labels = sys.argv[2:separator]
adapters = sys.argv[separator + 1:]
if len(labels) != len(adapters):
    raise SystemExit("label/adapter count mismatch")
output.write_text(json.dumps([
    {"label": label, "adapter_path": adapter}
    for label, adapter in zip(labels, adapters)
], indent=2) + "\n", encoding="utf-8")
PY

sample_args=()
if [[ -n "$MAX_SAMPLES" ]]; then
  sample_args=(--max_samples "$MAX_SAMPLES")
fi

run_humaneval() {
  local label="$1"
  local adapter="$2"
  local output_dir="$RUN_ROOT/eval/$label/humaneval"
  if [[ -f "$output_dir/metrics.json" ]]; then
    echo "[Resume] HumanEval $label"
    return
  fi
  echo "[Eval] HumanEval $label"
  "$PYTHON" -m finetune.eval.eval_humaneval \
    --base_model "$BASE_MODEL" \
    --adapter_dir "$adapter" \
    --config_src "$SOURCE_LORA" \
    --dataset_path "$HUMANEVAL_PATH" \
    --split test \
    --output_dir "$output_dir" \
    --prompt_style chat \
    --chat_user_prompt_style opencompass \
    --chat_template_mode non_thinking \
    --use_vllm \
    --tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN \
    --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size "$REQUEST_BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --timeout_s 3.0 \
    --eval_n_workers 16 \
    --seed 42 \
    "${sample_args[@]}"
}

run_mbpp() {
  local label="$1"
  local adapter="$2"
  local output_dir="$RUN_ROOT/eval/$label/mbpp"
  if [[ -f "$output_dir/metrics.json" ]]; then
    echo "[Resume] MBPP $label"
    return
  fi
  echo "[Eval] MBPP $label"
  "$PYTHON" -m finetune.eval.eval_mbpp \
    --base_model "$BASE_MODEL" \
    --adapter_dir "$adapter" \
    --config_src "$SOURCE_LORA" \
    --dataset_path "$MBPP_PATH" \
    --split test \
    --output_dir "$output_dir" \
    --prompt_style chat \
    --chat_template_mode non_thinking \
    --use_vllm \
    --tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN \
    --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size "$REQUEST_BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --timeout_s 3.0 \
    --eval_n_workers 16 \
    --seed 42 \
    "${sample_args[@]}"
}

for index in "${!labels[@]}"; do
  run_humaneval "${labels[$index]}" "${adapters[$index]}"
  run_mbpp "${labels[$index]}" "${adapters[$index]}"
  "$PYTHON" scripts/summarize_spectral_ablation_eval.py "$RUN_ROOT"
done

echo "[Done] $RUN_ROOT/summary.tsv"
