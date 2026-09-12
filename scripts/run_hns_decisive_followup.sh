#!/usr/bin/env bash
set -euo pipefail

PYTHON=/dataset1/zailong/envs/peft-sft-lab/bin/python
RUN_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-fc-interventions-20260910
FIRST_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-mechanism-first-batch-20260909
UTILITY_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-module-utility-20260909
FUNCTIONAL_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-functional-localization-20260909
AGGREGATE_ROOT=/dataset1/zailong/runs/peft-sft-lab/hns-2x4-allmodules-20260909

export PYTHONPATH="$PWD/src:$PWD/scripts"
export HF_HOME=/dataset1/zailong/cache/peft-sft-lab/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export PATH="$(dirname "$PYTHON"):$PATH"
mkdir -p "$RUN_ROOT/inference" "$RUN_ROOT/analysis"

# All work below is deliberately serial so the complete pipeline uses one GPU.
bash scripts/run_headfro_completion.sh

"$PYTHON" scripts/analyze_hns_paired_inference.py \
  --aggregate_root "$AGGREGATE_ROOT" \
  --first_batch_root "$FIRST_ROOT" \
  --output_dir "$FIRST_ROOT/statistics" \
  --bootstrap_samples 10000 --permutation_samples 100000 --seed 42

for case in qwen_magicoder qwen_commonsense llama_tulu; do
  bash scripts/run_fc_intervention_case.sh "$case"
done

"$PYTHON" scripts/analyze_fc_intervention_inference.py \
  --run_root "$RUN_ROOT" --output_dir "$RUN_ROOT/inference" \
  --bootstrap_samples 10000 --permutation_samples 100000 --seed 42

"$PYTHON" scripts/analyze_hns_nonadditivity.py \
  --run_root "$RUN_ROOT" \
  --module_utility_root "$UTILITY_ROOT" \
  --functional_localization_root "$FUNCTIONAL_ROOT" \
  --output "$RUN_ROOT/analysis/nonadditivity.tsv" \
  --bootstrap_samples 10000 --seed 42

"$PYTHON" - "$RUN_ROOT/completion.json" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "complete",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "gpu_policy": "one serial GPU allocation",
    "head_fro_checkpoints": 8,
    "fc_cases": ["qwen_magicoder", "qwen_commonsense", "llama_tulu"],
    "fc_variants_per_case": 10,
    "nonadditivity_adapter_sets_per_case": 4,
}, indent=2) + "\n")
PY

echo "[Done] decisive HNS follow-up: $RUN_ROOT"
