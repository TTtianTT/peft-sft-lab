set -euo pipefail

# 需要从 peft-sft-lab 仓库根目录执行
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 输入路径 (保持你原有的设置)
RUNS_ROOTS=(
  "/home/zailongtian/workspace/peft-sft-lab/runs_refactor_data_20260121/meta-llama-Llama-3.1-8B"
  "/home/zailongtian/workspace/peft-sft-lab/runs_refactor_data_20260121/Qwen-Qwen3-8B"
)

# 基础输出路径 (会自动追加 setting 后缀)
BASE_OUT_ROOT="/home/zailongtian/workspace/peft-sft-lab/lm_eval_outputs_ablation_modules"

# === 核心控制变量 (根据之前的讨论固定) ===
PRESERVE_ENERGY="l1"    # 排除能量漂移干扰
CALIB_SAMPLES="128"     # 降低采样噪声
POLICIES=(abs_select smooth_abs random_index grad_direction)
ADAPTER_TYPE="lora"
TASKS=("alpaca")
CALIB_DATASET="tatsu-lab/alpaca"
CALIB_SPLIT="train"

# === 定义消融实验的 Setting 组 (Keys) ===
# 建议运行顺序：residual (A) -> input (B) -> full (D) -> internal (C)
# 如果时间不够，优先跑 residual 和 input
SETTINGS=("residual" "input" "full" "internal")

for setting in "${SETTINGS[@]}"; do
  echo "############################################################"
  echo "[PHASE] Starting Ablation Setting: ${setting}"
  echo "############################################################"

  # 根据 Setting 定义 Target Modules
  case $setting in
    "residual")
      # Setting A (Ours): Residual-Writing Modules
      # 预期效果：Best Performance (利用了几何骨架)
      TARGET_MODULES=("down_proj" "o_proj")
      ;;
    "input")
      # Setting B: Attention Inputs
      # 预期效果：No Gain / Degradation (破坏了局部特征变换)
      TARGET_MODULES=("q_proj" "k_proj" "v_proj")
      ;;
    "internal")
      # Setting C: MLP Internal
      # 预期效果：No Gain / Degradation
      TARGET_MODULES=("up_proj" "gate_proj")
      ;;
    "full")
      # Setting D: All Modules
      # 预期效果：Suboptimal (引入了 input/internal 的噪声)
      TARGET_MODULES=("down_proj" "o_proj" "q_proj" "k_proj" "v_proj" "up_proj" "gate_proj")
      ;;
    *)
      echo "Unknown setting: $setting"
      exit 1
      ;;
  esac

  # 定义该 Setting 的输出目录
  OUT_ROOT="${BASE_OUT_ROOT}_${setting}_calib${CALIB_SAMPLES}_${PRESERVE_ENERGY}"

  echo "------------------------------------------------------------"
  echo "[CONFIG] Setting: ${setting}"
  echo "[CONFIG] Modules: ${TARGET_MODULES[*]}"
  echo "[CONFIG] Output:  ${OUT_ROOT}"
  echo "------------------------------------------------------------"

  # 调用 Python 脚本
  # 注意：这里假设你的 run_lm_eval_harness_spectral_edits.py 已经透传了 --target_modules 参数
  python scripts/run_lm_eval_harness_spectral_edits.py \
    --runs_roots "${RUNS_ROOTS[@]}" \
    --out_root "${OUT_ROOT}" \
    --policies "${POLICIES[@]}" \
    --tasks "${TASKS[@]}" \
    --calib_samples "${CALIB_SAMPLES}" \
    --calib_dataset "${CALIB_DATASET}" \
    --calib_split "${CALIB_SPLIT}" \
    --preserve_energy "${PRESERVE_ENERGY}" \
    --no-keep_edited_adapter \
    --adapter_types "${ADAPTER_TYPE}" \
    --target_modules "${TARGET_MODULES[@]}"

done
