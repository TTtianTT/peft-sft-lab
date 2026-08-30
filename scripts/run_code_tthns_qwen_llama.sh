#!/usr/bin/env bash
set -euo pipefail

# Build and evaluate TT-HNS adapters for Qwen3-8B and/or
# Llama-3.1-8B-Instruct. Adapter paths are intentionally required because run
# directories differ across machines and should never be guessed.

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/peft-sft-lab}"
MAGICODER_PATH="${MAGICODER_PATH:-/root/autodl-tmp/magicoder-train.jsonl}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-/root/autodl-tmp/humaneval-test.parquet}"
MODEL_FAMILIES="${MODEL_FAMILIES:-qwen llama}"
SEED="${SEED:-42}"

QWEN_BASE_MODEL="${QWEN_BASE_MODEL:-Qwen/Qwen3-8B}"
QWEN_LORA_PATH="${QWEN_LORA_PATH:-}"
QWEN_OUT_ROOT="${QWEN_OUT_ROOT:-runs/edited/Qwen-Qwen3-8B/code/tthns/seed${SEED}}"

LLAMA_BASE_MODEL="${LLAMA_BASE_MODEL:-/root/autodl-tmp/Llama-3.1-8B-Instruct}"
LLAMA_LORA_PATH="${LLAMA_LORA_PATH:-}"
LLAMA_OUT_ROOT="${LLAMA_OUT_ROOT:-runs/edited/meta-llama-Llama-3.1-8B-Instruct/code/tthns/seed${SEED}}"

CALIB_SAMPLES="${CALIB_SAMPLES:-256}"
SELECTION_SAMPLES="${SELECTION_SAMPLES:-64}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"
SELECTION_BATCH_SIZE="${SELECTION_BATCH_SIZE:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_REQUEST_BATCH_SIZE="${EVAL_REQUEST_BATCH_SIZE:-8}"

cd "$REPO_DIR"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "$MAGICODER_PATH"
require_file "$HUMANEVAL_PATH"

run_family() {
  local family="$1"
  local base_model="$2"
  local lora_path="$3"
  local out_root="$4"
  local chat_template_mode="$5"

  if [[ -z "$lora_path" ]]; then
    echo "Missing ${family^^}_LORA_PATH. Set it to the trained code LoRA adapter directory." >&2
    exit 1
  fi
  require_file "$lora_path/adapter_config.json"
  if [[ ! -f "$lora_path/adapter_model.safetensors" && ! -f "$lora_path/adapter_model.bin" ]]; then
    echo "Missing adapter weights under: $lora_path" >&2
    exit 1
  fi

  local calibration_hns_path="$out_root/calibration-sensitivity-hns-8plus2"
  local tthns_path="$out_root/humaneval"
  local eval_root="$out_root/eval"

  echo "[$family] Stage 1/3: supervised Magicoder localization"
  PYTHONPATH=src python -m finetune.spectral_edit.cli sensitivity-hns \
    --base_model "$base_model" \
    --lora_path "$lora_path" \
    --out_dir "$calibration_hns_path" \
    --target_modules all_modules \
    --calib_dataset ise-uiuc/Magicoder-Evol-Instruct-110K \
    --calib_dataset_path "$MAGICODER_PATH" \
    --calib_text_fields instruction response \
    --calib_samples "$CALIB_SAMPLES" \
    --calib_batch_size "$CALIB_BATCH_SIZE" \
    --calib_shuffle \
    --sft_format chat \
    --chat_template_mode "$chat_template_mode" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --fast_steps 8 \
    --stable_steps 2 \
    --dtype bf16 \
    --cpu_activation_offload \
    --seed "$SEED"

  echo "[$family] Stage 2/3: label-free HumanEval TT-HNS routing"
  PYTHONPATH=src python scripts/build_code_tthns_adapter.py \
    --base_model "$base_model" \
    --lora_path "$lora_path" \
    --calibration_hns_path "$calibration_hns_path" \
    --out_dir "$tthns_path" \
    --dataset_path "$HUMANEVAL_PATH" \
    --split test \
    --selection_samples "$SELECTION_SAMPLES" \
    --selection_batch_size "$SELECTION_BATCH_SIZE" \
    --prompt_style chat \
    --chat_user_prompt_styles strict_continuation opencompass \
    --chat_template_mode "$chat_template_mode" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --dtype bf16 \
    --cpu_activation_offload \
    --seed "$SEED" \
    --overwrite

  echo "[$family] Stage 3/3: LoRA and TT-HNS HumanEval evaluation"
  for variant in lora tthns; do
    local adapter_dir="$lora_path"
    if [[ "$variant" == "tthns" ]]; then
      adapter_dir="$tthns_path"
    fi
    PYTHONPATH=src python -m finetune.eval.eval_humaneval \
      --base_model "$base_model" \
      --adapter_dir "$adapter_dir" \
      --config_src "$lora_path" \
      --dataset_path "$HUMANEVAL_PATH" \
      --split test \
      --output_dir "$eval_root/$variant" \
      --prompt_style chat \
      --chat_user_prompt_style opencompass \
      --chat_template_mode "$chat_template_mode" \
      --use_vllm \
      --tensor_parallel_size 1 \
      --vllm_max_model_len 4096 \
      --vllm_attention_backend FLASH_ATTN \
      --vllm_disable_flashinfer_sampler \
      --vllm_request_batch_size "$EVAL_REQUEST_BATCH_SIZE" \
      --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
      --timeout_s 3.0 \
      --eval_n_workers 4 \
      --seed "$SEED"
  done

  python - "$eval_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print(f"{'variant':12} {'pass@1':>10} {'correct':>9} {'total':>7}")
print("-" * 42)
for variant in ("lora", "tthns"):
    metrics = json.loads((root / variant / "metrics.json").read_text(encoding="utf-8"))
    print(
        f"{variant:12} {metrics.get('pass@1', 0.0):>10.6f} "
        f"{str(metrics.get('correct')):>9} {str(metrics.get('total')):>7}"
    )
PY
}

for family in $MODEL_FAMILIES; do
  case "$family" in
    qwen)
      run_family qwen "$QWEN_BASE_MODEL" "$QWEN_LORA_PATH" "$QWEN_OUT_ROOT" non_thinking
      ;;
    llama)
      run_family llama "$LLAMA_BASE_MODEL" "$LLAMA_LORA_PATH" "$LLAMA_OUT_ROOT" auto
      ;;
    *)
      echo "Unknown model family: $family (expected qwen and/or llama)" >&2
      exit 1
      ;;
  esac
done
