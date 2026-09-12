#!/usr/bin/env bash
set -euo pipefail

# Intended for one RTX 6000 allocation:
# srun --partition=RTXq --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=04:00:00 \
#   bash scripts/run_hns_mechanism_analysis.sh

REPO_ROOT="${REPO_ROOT:-/dataset1/zailong/workspace/peft-sft-lab}"
PYTHON="${PYTHON:-/dataset1/zailong/envs/peft-sft-lab/bin/python}"
BASE_MODEL="${BASE_MODEL:-/dataset1/zailong/models/Qwen3-8B}"
MODEL_ROOT="${MODEL_ROOT:-/dataset1/zailong/models/spectral-surgery}"
SOURCE_LORA="${SOURCE_LORA:-$MODEL_ROOT/Qwen3-8B-Magicoder-50K-LoRA-E1}"
HNS_ADAPTER="${HNS_ADAPTER:-$MODEL_ROOT/Qwen3-8B-Magicoder-50K-SpectralSurgery-HNS4p1-AllMods}"
MAGICODER_PATH="${MAGICODER_PATH:-/dataset1/zailong/data/peft-sft-lab/magicoder-train.parquet}"
HUMANEVAL_PATH="${HUMANEVAL_PATH:-/dataset1/zailong/data/peft-sft-lab/humaneval-test.parquet}"
MBPP_PATH="${MBPP_PATH:-/dataset1/zailong/data/peft-sft-lab/mbpp-sanitized-test.parquet}"
RUN_ROOT="${RUN_ROOT:-/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-mechanism-20260908}"
REFERENCE_RUN="${REFERENCE_RUN:-/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/magicoder-spectral-ablations-20260908}"

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

echo "[Environment] node=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'

if [[ ! -f "$RUN_ROOT/spectra/spectrum_analysis.json" ]]; then
  "$PYTHON" scripts/analyze_hns_spectra.py \
    --config configs/hns_mechanism_cases.json \
    --out_dir "$RUN_ROOT/spectra" \
    --device cuda
else
  echo "[Resume] spectra"
fi

CONTROL_ROOT="$RUN_ROOT/adapters"
if [[ ! -f "$CONTROL_ROOT/manifest.json" ]]; then
  "$PYTHON" scripts/build_hns_causal_controls.py \
    --lora_path "$SOURCE_LORA" \
    --hns_path "$HNS_ADAPTER" \
    --out_root "$CONTROL_ROOT" \
    --device cuda
else
  echo "[Resume] controls"
fi

if [[ ! -f "$RUN_ROOT/activation/activation_analysis.json" ]]; then
  "$PYTHON" scripts/measure_lora_modification.py \
    --base_model "$BASE_MODEL" \
    --lora_path "$SOURCE_LORA" \
    --hns_path "$HNS_ADAPTER" \
    --dataset_path "$MAGICODER_PATH" \
    --dataset_kind magicoder \
    --output_dir "$RUN_ROOT/activation" \
    --samples 256 \
    --batch_size 2 \
    --max_seq_len 512 \
    --seed 42 \
    --dtype bf16
else
  echo "[Resume] activation"
fi

labels=(LoRA ScalarShrink ShapeOnly HeadOnly TailOnly HNS)
adapters=(
  "$SOURCE_LORA"
  "$CONTROL_ROOT/scalar-shrink"
  "$CONTROL_ROOT/shape-only"
  "$CONTROL_ROOT/head-only"
  "$CONTROL_ROOT/tail-only"
  "$HNS_ADAPTER"
)

"$PYTHON" - "$RUN_ROOT/experiment_paths.json" "${labels[@]}" -- "${adapters[@]}" <<'PY'
import json, sys
from pathlib import Path
sep = sys.argv.index("--")
labels, adapters = sys.argv[2:sep], sys.argv[sep + 1:]
Path(sys.argv[1]).write_text(json.dumps([
    {"label": label, "adapter_path": adapter}
    for label, adapter in zip(labels, adapters)
], indent=2) + "\n")
PY

for label in LoRA HNS; do
  reference_label="control-lora"
  [[ "$label" == HNS ]] && reference_label="control-hns4p1-allmods"
  if [[ ! -d "$RUN_ROOT/eval/$label" && -d "$REFERENCE_RUN/eval/$reference_label" ]]; then
    cp -a "$REFERENCE_RUN/eval/$reference_label" "$RUN_ROOT/eval/$label"
    echo "[Reuse] $label metrics from identical prior evaluation"
  fi
done

run_humaneval() {
  local label="$1" adapter="$2" output="$RUN_ROOT/eval/$1/humaneval"
  [[ -f "$output/metrics.json" ]] && { echo "[Resume] HumanEval $label"; return; }
  "$PYTHON" -m finetune.eval.eval_humaneval \
    --base_model "$BASE_MODEL" --adapter_dir "$adapter" --config_src "$SOURCE_LORA" \
    --dataset_path "$HUMANEVAL_PATH" --split test --output_dir "$output" \
    --prompt_style chat --chat_user_prompt_style opencompass --chat_template_mode non_thinking \
    --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size 32 --max_new_tokens 512 --timeout_s 3.0 \
    --eval_n_workers 16 --seed 42
}

run_mbpp() {
  local label="$1" adapter="$2" output="$RUN_ROOT/eval/$1/mbpp"
  [[ -f "$output/metrics.json" ]] && { echo "[Resume] MBPP $label"; return; }
  "$PYTHON" -m finetune.eval.eval_mbpp \
    --base_model "$BASE_MODEL" --adapter_dir "$adapter" --config_src "$SOURCE_LORA" \
    --dataset_path "$MBPP_PATH" --split test --output_dir "$output" \
    --prompt_style chat --chat_template_mode non_thinking \
    --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
    --vllm_attention_backend FLASH_ATTN --vllm_disable_flashinfer_sampler \
    --vllm_request_batch_size 32 --max_new_tokens 512 --timeout_s 3.0 \
    --eval_n_workers 16 --seed 42
}

for index in "${!labels[@]}"; do
  run_humaneval "${labels[$index]}" "${adapters[$index]}"
  run_mbpp "${labels[$index]}" "${adapters[$index]}"
  "$PYTHON" scripts/summarize_spectral_ablation_eval.py "$RUN_ROOT"
done

"$PYTHON" scripts/summarize_hns_mechanism.py "$RUN_ROOT"

echo "[Done] $RUN_ROOT"
