#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE=/dataset1/zailong/models/Qwen3-8B
LORA=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-LoRA
HNS=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-Spectral-Surgery-AllModules-4Plus1
DATA=/dataset1/zailong/data/peft-sft-lab/cross-task-mechanism
TRAIN=/dataset1/zailong/data/peft-sft-lab/metamathqa-train.parquet
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-cross-task-20260908/metamath
ADAPTERS="$RUN_ROOT/adapters"

export PYTHONPATH="$PWD/src"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
mkdir -p "$RUN_ROOT" "$RUN_ROOT/eval"

if [[ ! -f "$RUN_ROOT/spectra/spectrum_summary.json" ]]; then
  "$PYTHON" scripts/analyze_spectral_pair.py --lora_path "$LORA" --hns_path "$HNS" --output_dir "$RUN_ROOT/spectra" --device cuda
fi
if [[ ! -f "$ADAPTERS/manifest.json" ]]; then
  "$PYTHON" scripts/build_hns_causal_controls.py --lora_path "$LORA" --hns_path "$HNS" --out_root "$ADAPTERS" --device cuda
fi

"$PYTHON" - "$RUN_ROOT/experiment_paths.json" "$LORA" "$HNS" "$ADAPTERS" <<'PY'
import json, sys
out, lora, hns, root = sys.argv[1:]
rows = [
    {"label": "lora", "path": lora},
    {"label": "scalar_shrink", "path": root + "/scalar-shrink"},
    {"label": "shape_only", "path": root + "/shape-only"},
    {"label": "head_only", "path": root + "/head-only"},
    {"label": "tail_only", "path": root + "/tail-only"},
    {"label": "hns", "path": hns},
]
open(out, "w").write(json.dumps(rows, indent=2) + "\n")
PY

if [[ ! -f "$RUN_ROOT/activation/activation_analysis.json" ]]; then
  "$PYTHON" scripts/measure_lora_modification.py \
    --base_model "$BASE" --lora_path "$LORA" --hns_path "$HNS" \
    --control "scalar_shrink=$ADAPTERS/scalar-shrink" \
    --control "shape_only=$ADAPTERS/shape-only" \
    --control "head_only=$ADAPTERS/head-only" \
    --control "tail_only=$ADAPTERS/tail-only" \
    --dataset_path "$TRAIN" --dataset_kind metamath --samples 256 \
    --batch_size 2 --max_seq_len 512 --seed 42 --dtype bf16 \
    --output_dir "$RUN_ROOT/activation"
fi
cp "$RUN_ROOT/spectra/module_spectra.csv" "$RUN_ROOT/module_spectra.csv"
cp "$RUN_ROOT/activation/direction_response.csv" "$RUN_ROOT/direction_response.csv"
cp "$RUN_ROOT/activation/layer_response.csv" "$RUN_ROOT/layer_response.csv"

mkdir -p "$RUN_ROOT/eval/lora" "$RUN_ROOT/eval/hns"
if [[ ! -f "$RUN_ROOT/eval/lora/gsm8k/metrics.json" ]]; then
  cp -a "$LORA/eval-gsm8k-maxnew2048" "$RUN_ROOT/eval/lora/gsm8k"
fi
if [[ ! -f "$RUN_ROOT/eval/hns/gsm8k/metrics.json" ]]; then
  cp -a "$HNS/eval-gsm8k-maxnew2048" "$RUN_ROOT/eval/hns/gsm8k"
fi

for label in scalar_shrink shape_only head_only tail_only; do
  adapter_label=${label//_/-}
  output="$RUN_ROOT/eval/$label/gsm8k"
  if [[ ! -f "$output/metrics.json" ]]; then
    "$PYTHON" -m finetune.eval.eval_gsm8k \
      --base_model "$BASE" --adapter_dir "$ADAPTERS/$adapter_label" --output_dir "$output" \
      --dataset_path "$DATA/gsm8k-test.jsonl" --dataset_config main --split test \
      --max_new_tokens 2048 --dtype bf16 --seed 42 --chat_template_mode non_thinking \
      --use_vllm --tensor_parallel_size 1 --vllm_max_model_len 4096 \
      --vllm_gpu_memory_utilization 0.85 --vllm_attention_backend FLASH_ATTN \
      --vllm_disable_flashinfer_sampler
  fi
  "$PYTHON" scripts/summarize_cross_task_task.py "$RUN_ROOT" --task metamath
done

"$PYTHON" scripts/summarize_cross_task_task.py "$RUN_ROOT" --task metamath
echo "[Done] MetaMath mechanism experiment: $RUN_ROOT"
