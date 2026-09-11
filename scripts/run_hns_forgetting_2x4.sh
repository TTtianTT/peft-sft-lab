#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
CONFIG="$PWD/configs/hns_forgetting_2x4_20260911.json"
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-forgetting-2x4-20260911
ADAPTER_ROOT="$RUN_ROOT/adapters"
MANIFEST_ROOT="$RUN_ROOT/variant_manifests"
DIAG_ROOT="$RUN_ROOT/diagnostic"
FORMAL_ROOT="$RUN_ROOT/formal"
REPEAT_ROOT="$RUN_ROOT/formal_repeat"

export PYTHONPATH="$PWD/src:$PWD/scripts"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_BATCH_INVARIANT=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$RUN_ROOT" "$ADAPTER_ROOT" "$MANIFEST_ROOT" "$DIAG_ROOT" "$FORMAL_ROOT" "$REPEAT_ROOT"
cp "$CONFIG" "$RUN_ROOT/experiment_config.json"
cp reports/hns_forgetting_20260911/protocol.md "$RUN_ROOT/protocol.md"

build_controls() {
  local base_name="$1"
  local task="$2"
  local lora_path="$3"
  local hns_path="$4"
  "$PYTHON" scripts/build_forgetting_common_basis_controls.py \
    --lora_path "$lora_path" \
    --hns_path "$hns_path" \
    --output_dir "$ADAPTER_ROOT/$base_name/$task" \
    --gammas 0.25 0.40 0.55 0.70 0.85 1.0 \
    --device cuda
}

build_controls Qwen3-8B magicoder \
  /dataset1/zailong/models/spectral-surgery/Qwen3-8B-Magicoder-50K-LoRA-E1 \
  /dataset1/zailong/models/spectral-surgery/Qwen3-8B-Magicoder-50K-SpectralSurgery-HNS4p1-AllMods
build_controls Qwen3-8B metamath \
  /dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-LoRA \
  /dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-Spectral-Surgery-AllModules-4Plus1
build_controls Qwen3-8B tulu \
  /dataset1/zailong/models/spectral-surgery/Qwen3-8B-InstructionFollowing-LoRA \
  /dataset1/zailong/models/spectral-surgery/Qwen3-8B-InstructionFollowing-SpectralSurgery-HNS4p1
build_controls Qwen3-8B commonsense \
  /dataset1/zailong/models/spectral-surgery/Qwen3-8B-CommonSense170K-LoRA \
  /dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Qwen3-8B/commonsense/hns-allmodules-8plus2

build_controls Llama-3.1-8B-Instruct magicoder \
  /dataset1/zailong/models/spectral-surgery/Llama-3.1-8B-Instruct-Magicoder-50K-LoRA-E1 \
  /dataset1/zailong/models/spectral-surgery/Llama-3.1-8B-Instruct-Magicoder-50K-SpectralSurgery-HNS4p1
build_controls Llama-3.1-8B-Instruct metamath \
  /dataset1/zailong/models/spectral-surgery/Llama-3.1-8B-Instruct-MetaMathQA-50K-LoRA \
  /dataset1/zailong/models/spectral-surgery/Llama-3.1-8B-Instruct-MetaMathQA-50K-SpectralSurgery-HNS4p1-AllMods
build_controls Llama-3.1-8B-Instruct tulu \
  /dataset1/zailong/models/spectral-surgery/Llama-3.1-8B-Instruct-InstructionFollowing-LoRA \
  /dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Llama-3.1-8B-Instruct/tulu/hns-allmodules-8plus2
build_controls Llama-3.1-8B-Instruct commonsense \
  /dataset1/zailong/models/spectral-surgery/Llama-3.1-8B-Instruct-CommonSense170K-LoRA-GBS64-Final \
  /dataset1/zailong/models/spectral-surgery/Llama-3.1-8B-Instruct-CommonSense170K-Spectral-Surgery-AllModules-8Plus2

"$PYTHON" scripts/prepare_forgetting_variant_manifests.py \
  --config "$CONFIG" --adapter_root "$ADAPTER_ROOT" --output_dir "$MANIFEST_ROOT"

evaluate_matrix() {
  local base_path="$1"
  local manifest="$2"
  local output="$3"
  shift 3
  "$PYTHON" scripts/eval_forgetting_matrix_vllm.py \
    --base_model "$base_path" \
    --variant_manifest "$manifest" \
    --config "$CONFIG" \
    --output_dir "$output" \
    --tasks magicoder commonsense metamath tulu \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.92 \
    --max_num_seqs 256 \
    --max_num_batched_tokens 65536 \
    --adapter_block_size 8 \
    --prompt_chunk_short 1024 \
    --prompt_chunk_long 64 \
    --seed 42 "$@"
}

QWEN_MANIFEST="$MANIFEST_ROOT/Qwen3-8B.json"
LLAMA_MANIFEST="$MANIFEST_ROOT/Llama-3p1-8B-Instruct.json"
LLAMA_CORE_MANIFEST="$MANIFEST_ROOT/Llama-3p1-8B-Instruct-core.json"

"$PYTHON" scripts/filter_forgetting_variant_manifest.py \
  --input "$LLAMA_MANIFEST" --output "$LLAMA_CORE_MANIFEST"

if [[ ! -f "$DIAG_ROOT/Qwen3-8B/run1/generation_manifest.json" ]]; then
  evaluate_matrix /dataset1/zailong/models/Qwen3-8B "$QWEN_MANIFEST" "$DIAG_ROOT/Qwen3-8B/run1" \
    --diagnostic_max_samples 16
fi
if [[ ! -f "$DIAG_ROOT/Qwen3-8B/run2/generation_manifest.json" ]]; then
  evaluate_matrix /dataset1/zailong/models/Qwen3-8B "$QWEN_MANIFEST" "$DIAG_ROOT/Qwen3-8B/run2" \
    --diagnostic_max_samples 16 --reverse_variants
fi
"$PYTHON" scripts/check_forgetting_numerical_gate.py \
  --run1 "$DIAG_ROOT/Qwen3-8B/run1" \
  --run2 "$DIAG_ROOT/Qwen3-8B/run2" \
  --variant_manifest "$QWEN_MANIFEST" \
  --output "$DIAG_ROOT/Qwen3-8B/gate.json"

if [[ ! -f "$DIAG_ROOT/Llama-3.1-8B-Instruct/run1/generation_manifest.json" ]]; then
  evaluate_matrix /dataset1/zailong/models/Llama-3.1-8B-Instruct "$LLAMA_MANIFEST" \
    "$DIAG_ROOT/Llama-3.1-8B-Instruct/run1" --diagnostic_max_samples 16
fi
if [[ ! -f "$DIAG_ROOT/Llama-3.1-8B-Instruct/run3-stable-ids/generation_manifest.json" ]]; then
  evaluate_matrix /dataset1/zailong/models/Llama-3.1-8B-Instruct "$LLAMA_MANIFEST" \
    "$DIAG_ROOT/Llama-3.1-8B-Instruct/run3-stable-ids" --diagnostic_max_samples 16 --reverse_variants
fi
"$PYTHON" scripts/check_forgetting_numerical_gate.py \
  --run1 "$DIAG_ROOT/Llama-3.1-8B-Instruct/run1" \
  --run2 "$DIAG_ROOT/Llama-3.1-8B-Instruct/run3-stable-ids" \
  --variant_manifest "$LLAMA_MANIFEST" \
  --output "$DIAG_ROOT/Llama-3.1-8B-Instruct/gate.json" \
  --warn_only

evaluate_matrix /dataset1/zailong/models/Qwen3-8B "$QWEN_MANIFEST" "$FORMAL_ROOT/Qwen3-8B"
evaluate_matrix /dataset1/zailong/models/Llama-3.1-8B-Instruct "$LLAMA_MANIFEST" \
  "$FORMAL_ROOT/Llama-3.1-8B-Instruct"
evaluate_matrix /dataset1/zailong/models/Llama-3.1-8B-Instruct "$LLAMA_CORE_MANIFEST" \
  "$REPEAT_ROOT/Llama-3.1-8B-Instruct" --reverse_variants

"$PYTHON" scripts/score_forgetting_matrix.py --matrix_dir "$FORMAL_ROOT/Qwen3-8B" --workers 16
"$PYTHON" scripts/score_forgetting_matrix.py --matrix_dir "$FORMAL_ROOT/Llama-3.1-8B-Instruct" --workers 16
"$PYTHON" scripts/score_forgetting_matrix.py --matrix_dir "$REPEAT_ROOT/Llama-3.1-8B-Instruct" --workers 16

echo "[Done] HNS forgetting generation and scoring: $RUN_ROOT"
