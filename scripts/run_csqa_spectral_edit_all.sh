#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/zailongtian/workspace/peft-sft-lab/runs/mistralai-Mistral-7B-v0.3/csqa}"
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-v0.3}"

# 输出目录（你也可以改到 _runs/ 下面）
OUT_ROOT="${OUT_ROOT:-${RUN_ROOT}/_lora_spectral_edit_csqa}"

# ===== CSQA eval 设置 =====
SPLIT="${SPLIT:-validation}"
MAX_SAMPLES="${MAX_SAMPLES:-}"          # 例如 100；留空表示全量
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"

USE_VLLM="${USE_VLLM:-1}"               # 1=用vLLM；0=transformers
TP="${TP:-1}"                           # vLLM tensor parallel

# ===== Spectral-edit 设置（按你 repo 的实现改）=====
EDIT_MODE="${EDIT_MODE:-robust_z}"      # 例如 abs_select / smooth_abs / robust_z ...
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_N="${CALIB_N:-256}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,down_proj}"

mkdir -p "${OUT_ROOT}"

pick_latest_adapter_dir () {
python - "$1" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

# 兼容 safetensors / bin
cands = []
for pat in ["adapter_model.safetensors", "adapter_model.bin"]:
    for p in root.rglob(pat):
        d = p.parent
        if (d / "adapter_config.json").exists():
            cands.append(d)

if not cands:
    raise SystemExit(f"[ERR] No adapter_model.* under: {root}")

cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(str(cands[0]))
PY
}

run_eval () {
  local adapter_dir="$1"
  local out_dir="$2"

  mkdir -p "$out_dir"

  local max_samples_args=()
  if [[ -n "${MAX_SAMPLES}" ]]; then
    max_samples_args+=( --max_samples "${MAX_SAMPLES}" )
  fi

  if [[ "${USE_VLLM}" == "1" ]]; then
    python -m finetune.eval.eval_csqa \
      --base_model "${BASE_MODEL}" \
      --adapter_dir "${adapter_dir}" \
      --output_dir "${out_dir}" \
      --split "${SPLIT}" \
      --seed "${SEED}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --use_vllm \
      --tensor_parallel_size "${TP}" \
      "${max_samples_args[@]}"
  else
    python -m finetune.eval.eval_csqa \
      --base_model "${BASE_MODEL}" \
      --adapter_dir "${adapter_dir}" \
      --output_dir "${out_dir}" \
      --split "${SPLIT}" \
      --seed "${SEED}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      "${max_samples_args[@]}"
  fi
}

run_edit () {
  local adapter_dir="$1"
  local out_dir="$2"

  mkdir -p "$out_dir"

  # ====== 关键：这里给你两种常见入口写法，你用“你仓库里存在的那条” ======
  # 1) 如果你的 lora-spectral-edit 是统一 CLI（常见形态：python -m lora_spectral_edit edit ...）
  if python -m lora_spectral_edit edit --help >/dev/null 2>&1; then
    python -m lora_spectral_edit edit \
      --base_model_id "${BASE_MODEL}" \
      --lora_dir "${adapter_dir}" \
      --out_dir "${out_dir}" \
      --edit_mode "${EDIT_MODE}" \
      --target_modules "${TARGET_MODULES}" \
      --calib_dataset "tau/commonsense_qa" \
      --calib_split "${CALIB_SPLIT}" \
      --calib_num_samples "${CALIB_N}" \
      --seed "${SEED}" \
      --device "cuda"
    return 0
  fi

  # 2) 如果你的 edit 是独立模块（形态：python -m lora_spectral_edit.edit_adapter ...）
  if python -m lora_spectral_edit.edit_adapter --help >/dev/null 2>&1; then
    python -m lora_spectral_edit.edit_adapter \
      --base_model_id "${BASE_MODEL}" \
      --lora_dir "${adapter_dir}" \
      --out_dir "${out_dir}" \
      --edit_mode "${EDIT_MODE}" \
      --target_modules "${TARGET_MODULES}" \
      --calib_dataset "tau/commonsense_qa" \
      --calib_split "${CALIB_SPLIT}" \
      --calib_num_samples "${CALIB_N}" \
      --seed "${SEED}" \
      --device "cuda"
    return 0
  fi

  echo "[ERR] Cannot find a runnable edit entrypoint."
  echo "      Tried: python -m lora_spectral_edit edit --help"
  echo "             python -m lora_spectral_edit.edit_adapter --help"
  exit 2
}

for variant in adalora lora loraplus pissa; do
  echo "=============================="
  echo "[VARIANT] ${variant}"
  echo "=============================="

  variant_root="${RUN_ROOT}/${variant}"
  adapter_dir="$(pick_latest_adapter_dir "${variant_root}")"
  echo "[INFO] picked adapter_dir: ${adapter_dir}"

  base_out="${OUT_ROOT}/${variant}"
  baseline_out="${base_out}/baseline_eval"
  edited_out="${base_out}/edited/${EDIT_MODE}"
  edited_eval_out="${base_out}/edited_eval/${EDIT_MODE}"

  echo "[1/3] baseline eval -> ${baseline_out}"
  run_eval "${adapter_dir}" "${baseline_out}"

  echo "[2/3] spectral edit -> ${edited_out}"
  run_edit "${adapter_dir}" "${edited_out}"

  # 约定：edit 输出目录本身就是可 load 的 adapter（含 adapter_config.json / adapter_model.*）
  # 如果你 repo 输出在子目录（比如 edited_out/adapter/），把下面这一行改一下即可。
  edited_adapter_dir="${edited_out}"

  echo "[3/3] edited eval -> ${edited_eval_out}"
  run_eval "${edited_adapter_dir}" "${edited_eval_out}"

  echo "[DONE] ${variant}"
done

echo "ALL DONE. Results under: ${OUT_ROOT}"
