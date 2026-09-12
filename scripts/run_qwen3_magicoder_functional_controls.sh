#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE=/dataset1/zailong/models/Qwen3-8B
LORA=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-Magicoder-50K-LoRA-E1
HNS=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-Magicoder-50K-SpectralSurgery-HNS4p1-AllMods
TRAIN=/dataset1/zailong/data/peft-sft-lab/magicoder-train.parquet
SOURCE_ROOT=/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-mechanism-20260908
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-cross-task-20260908/magicoder
ADAPTERS="$SOURCE_ROOT/adapters"

export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_OFFLINE=1
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export PATH="$(dirname "$PYTHON"):$PATH"
mkdir -p "$RUN_ROOT"

if [[ ! -f "$RUN_ROOT/spectra/spectrum_summary.json" ]]; then
  "$PYTHON" scripts/analyze_spectral_pair.py \
    --lora_path "$LORA" --hns_path "$HNS" --output_dir "$RUN_ROOT/spectra" --device cuda
fi
if [[ ! -f "$ADAPTERS/manifest.json" ]]; then
  "$PYTHON" scripts/build_hns_causal_controls.py \
    --lora_path "$LORA" --hns_path "$HNS" --out_root "$ADAPTERS" --device cuda
fi
if [[ ! -f "$RUN_ROOT/activation/activation_analysis.json" ]]; then
  "$PYTHON" scripts/measure_lora_modification.py \
    --base_model "$BASE" --lora_path "$LORA" --hns_path "$HNS" \
    --control "scalar_shrink=$ADAPTERS/scalar-shrink" \
    --control "shape_only=$ADAPTERS/shape-only" \
    --control "head_only=$ADAPTERS/head-only" \
    --control "tail_only=$ADAPTERS/tail-only" \
    --dataset_path "$TRAIN" --dataset_kind magicoder --samples 256 \
    --batch_size 2 --max_seq_len 512 --seed 42 --dtype bf16 \
    --output_dir "$RUN_ROOT/activation"
fi
cp "$RUN_ROOT/spectra/module_spectra.csv" "$RUN_ROOT/module_spectra.csv"
cp "$RUN_ROOT/activation/direction_response.csv" "$RUN_ROOT/direction_response.csv"
cp "$RUN_ROOT/activation/layer_response.csv" "$RUN_ROOT/layer_response.csv"
cp "$RUN_ROOT/activation/module_response.csv" "$RUN_ROOT/module_response.csv"
cp "$SOURCE_ROOT/summary.json" "$RUN_ROOT/summary.json"
cp "$SOURCE_ROOT/summary.tsv" "$RUN_ROOT/summary.tsv"

echo "[Done] Qwen3 Magicoder six-way functional controls: $RUN_ROOT"
