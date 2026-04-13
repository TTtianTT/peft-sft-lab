set -euo pipefail

# 需要从 peft-sft-lab 仓库根目录执行（确保 scripts/ 和 src/ 路径都对）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUNS_ROOTS=(
  "/home/zailongtian/workspace/peft-sft-lab/runs_refactor_data_20260121/meta-llama-Llama-3.1-8B"
    "/home/zailongtian/workspace/peft-sft-lab/runs_refactor_data_20260121/Qwen-Qwen3-8B"
)

# 单独的输出路径（用于补 CSQA 的 l1 结果）
OUT_ROOT="/home/zailongtian/workspace/peft-sft-lab/lm_eval_outputs_pe_l1_csqa_calib32"

PRESERVE_ENERGY="l1"
CALIB_SAMPLES="32"
POLICIES=(abs_select smooth_abs random_index grad_direction)
ADAPTER_TYPES=(lora)

echo "============================================================"
echo "[RUN] adapter_types=${ADAPTER_TYPES[*]} calib_samples=${CALIB_SAMPLES} preserve_energy=${PRESERVE_ENERGY}"
echo "      out_root=${OUT_ROOT}"
echo "============================================================"

python scripts/run_lm_eval_harness_spectral_edits.py \
	  --runs_roots "${RUNS_ROOTS[@]}" \
	    --out_root "${OUT_ROOT}" \
	      --policies "${POLICIES[@]}" \
	        --calib_samples "${CALIB_SAMPLES}" \
		  --preserve_energy "${PRESERVE_ENERGY}" \
		    --no-keep_edited_adapter \
		      --adapter_types "${ADAPTER_TYPES[@]}"

echo "[DONE] calib32 + preserve_energy=l1 finished."
