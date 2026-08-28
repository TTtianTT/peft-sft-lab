#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-/root/autodl-tmp/humaneval-test.parquet}"

RUN_ROOT="${RUN_ROOT:-runs/Qwen-Qwen3-8B/code/lora/magicoder-chat-50k-3ep-seq4096-gbs32-lr2e-5/seed42}"
EPOCH3="${EPOCH3:-$RUN_ROOT/checkpoint-4683}"
HNS_EPOCH3_4P1="${HNS_EPOCH3_4P1:-runs/edited/Qwen-Qwen3-8B/code/magicoder-chat-50k-3ep/seed42/epoch3/hns-all-r16-4plus1}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VLLM_REQUEST_BATCH_SIZE="${VLLM_REQUEST_BATCH_SIZE:-8}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/eval_humaneval_opencompass_style_maxnew${MAX_NEW_TOKENS}_batch${VLLM_REQUEST_BATCH_SIZE}}"

if [[ ! -f "$HUMANEVAL_PATH" ]]; then
  echo "Missing HumanEval dataset: $HUMANEVAL_PATH" >&2
  exit 1
fi

for adapter in "$EPOCH3" "$HNS_EPOCH3_4P1"; do
  if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
    echo "Missing adapter weights: $adapter/adapter_model.safetensors" >&2
    exit 1
  fi
done

run_humaneval() {
  local label="$1"
  local adapter="${2:-}"
  local config_src="${3:-}"
  local adapter_args=()

  if [[ -n "$adapter" ]]; then
    adapter_args=(
      --adapter_dir "$adapter"
      --config_src "$config_src"
    )
  fi

  PYTHONPATH=src python -m finetune.eval.eval_humaneval \
    --base_model "$BASE_MODEL" \
    "${adapter_args[@]}" \
    --dataset_path "$HUMANEVAL_PATH" \
    --split test \
    --output_dir "$EVAL_ROOT/$label" \
    --prompt_style chat \
    --chat_user_prompt_style opencompass \
    --chat_template_mode non_thinking \
    --use_vllm \
    --tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN \
    --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size "$VLLM_REQUEST_BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --timeout_s 3.0 \
    --eval_n_workers 4 \
    --seed 42
}

run_humaneval base
run_humaneval epoch3-lora "$EPOCH3" "$EPOCH3"
run_humaneval epoch3-hns-all-4plus1 "$HNS_EPOCH3_4P1" "$EPOCH3"

python - "$EVAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print(f"{'experiment':32} {'pass@1':>10} {'correct':>9} {'total':>7}")
print("-" * 64)
for path in sorted(root.glob("*/metrics.json")):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"{path.parent.name:32} "
        f"{metrics.get('pass@1', 0.0):>10.6f} "
        f"{str(metrics.get('correct')):>9} "
        f"{str(metrics.get('total')):>7}"
    )
PY
