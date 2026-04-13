set -euo pipefail

# 需要从 peft-sft-lab 仓库根目录执行（确保 scripts/ 和 src/ 路径都对）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUNS_ROOTS=(
  "/home/zailongtian/workspace/peft-sft-lab/runs_refactor_data_20260121/meta-llama-Llama-3.1-8B"
  "/home/zailongtian/workspace/peft-sft-lab/runs_refactor_data_20260121/Qwen-Qwen3-8B"
)

BASE_OUT_ROOT="/home/zailongtian/workspace/peft-sft-lab/lm_eval_outputs_pe_none_sweep"
PRESERVE_ENERGY="none"

POLICIES=(abs_select smooth_abs random_index grad_direction)
CALIB_SAMPLES_LIST=(32 64 128 256)
ADAPTER_TYPES=(lora)

for adapter_type in "${ADAPTER_TYPES[@]}"; do
  echo "============================================================"
  echo "[PHASE] adapter_type=${adapter_type}"
  echo "============================================================"

  for cs in "${CALIB_SAMPLES_LIST[@]}"; do
    OUT_ROOT="${BASE_OUT_ROOT}_calib${cs}"
    echo "============================================================"
    echo "[RUN] adapter_type=${adapter_type} calib_samples=${cs} preserve_energy=${PRESERVE_ENERGY} -> out_root=${OUT_ROOT}"
    echo "============================================================"

    python scripts/run_lm_eval_harness_spectral_edits.py \
      --runs_roots "${RUNS_ROOTS[@]}" \
      --out_root "${OUT_ROOT}" \
      --policies "${POLICIES[@]}" \
      --calib_samples "${cs}" \
      --preserve_energy "${PRESERVE_ENERGY}" \
      --no-keep_edited_adapter \
      --adapter_types "${adapter_type}"
  done
done

echo "[DONE] All runs finished."
