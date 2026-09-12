#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
BASE=/dataset1/zailong/models/Qwen3-8B
LORA=/dataset1/zailong/models/spectral-surgery/Qwen3-8B-CommonSense170K-LoRA
HNS=/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Qwen3-8B/commonsense/hns-allmodules-8plus2
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-signed-gates-20260910/qwen_commonsense_piqa
SIGNALS="$RUN_ROOT/signals/split_manifest.json"
SELECTION="$RUN_ROOT/selection/selected_directions.tsv"
ADAPTERS="$RUN_ROOT/direction_adapters"
BLOCKS="$RUN_ROOT/block_adapters"

export PYTHONPATH="$PWD/src:$PWD/scripts"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export TOKENIZERS_PARALLELISM=false
export PATH="$(dirname "$PYTHON"):$PATH"

mkdir -p "$RUN_ROOT/eval/dose" "$RUN_ROOT/eval/validation" "$RUN_ROOT/analysis/dose" "$RUN_ROOT/analysis/validation"

if [[ ! -f "$ADAPTERS/manifest.json" ]]; then
  "$PYTHON" scripts/build_direction_dose_adapters.py \
    --lora_path "$LORA" --hns_path "$HNS" --selection "$SELECTION" \
    --output_dir "$ADAPTERS" --strengths 0.1 1.0
fi

eval_one() {
  local split=$1
  local label=$2
  local adapter=$3
  local output="$RUN_ROOT/eval/$split/$label/commonsense"
  if [[ -f "$output/summary.json" ]]; then
    echo "[Resume] $split/$label"
    return
  fi
  "$PYTHON" scripts/eval_commonsense_8tasks.py \
    --base_model "$BASE" --adapter_dir "$adapter" --output_dir "$output" \
    --tasks piqa --sample_manifest "$SIGNALS" --sample_split "$split" \
    --backend vllm --chat_template_mode non_thinking --max_new_tokens 8 \
    --request_batch_size 256 --tensor_parallel_size 1 --vllm_max_model_len 2048 \
    --vllm_gpu_memory_utilization 0.90 --vllm_attention_backend FLASH_ATTN \
    --dtype bf16 --seed 42
}

eval_one dose lora "$LORA"
while IFS= read -r label; do
  eval_one dose "$label" "$ADAPTERS/$label"
done < <("$PYTHON" -c 'import json,sys; [print(x["label"]) for x in json.load(open(sys.argv[1]))["variants"]]' "$ADAPTERS/manifest.json")

"$PYTHON" scripts/analyze_piqa_direction_interventions.py \
  --manifest "$ADAPTERS/manifest.json" --selection "$SELECTION" \
  --eval_root "$RUN_ROOT/eval/dose" --output_dir "$RUN_ROOT/analysis/dose" \
  --split dose --bootstrap 10000 --seed 20260910

if [[ ! -f "$BLOCKS/manifest.json" ]]; then
  "$PYTHON" scripts/build_piqa_direction_blocks.py \
    --lora_path "$LORA" --hns_path "$HNS" --selection "$SELECTION" \
    --dose_decision "$RUN_ROOT/analysis/dose/dose_decision.json" --output_dir "$BLOCKS"
fi

eval_one validation lora "$LORA"
while IFS= read -r label; do
  eval_one validation "$label" "$ADAPTERS/$label"
done < <("$PYTHON" -c 'import json,sys; [print(x["label"]) for x in json.load(open(sys.argv[1]))["variants"]]' "$ADAPTERS/manifest.json")

while IFS= read -r label; do
  eval_one validation "$label" "$BLOCKS/$label"
done < <("$PYTHON" -c 'import json,sys; [print(x["label"]) for x in json.load(open(sys.argv[1]))["variants"]]' "$BLOCKS/manifest.json")
eval_one validation full_hns "$HNS"

"$PYTHON" scripts/analyze_piqa_direction_interventions.py \
  --manifest "$ADAPTERS/manifest.json" --selection "$SELECTION" \
  --eval_root "$RUN_ROOT/eval/validation" --output_dir "$RUN_ROOT/analysis/validation" \
  --split validation --bootstrap 10000 --seed 20260910
"$PYTHON" scripts/analyze_piqa_block_interventions.py \
  --eval_root "$RUN_ROOT/eval/validation" --output_dir "$RUN_ROOT/analysis/validation" \
  --bootstrap 10000 --seed 20260910

"$PYTHON" -c 'import json,sys; from datetime import datetime,timezone; json.dump({"status":"complete","completed_at_utc":datetime.now(timezone.utc).isoformat(),"gpu_policy":"one serial GPU allocation"},open(sys.argv[1],"w"),indent=2)' "$RUN_ROOT/completion.json"
echo "[Done] PIQA signed-gate finite interventions: $RUN_ROOT"
