#!/usr/bin/env bash
#SBATCH --job-name=hns-zscore-score
#SBATCH --output=/dataset1/zailong/runs/peft-sft-lab/hns-spectral-standardization-2x4-20260911/score-%A_%a.out
#SBATCH --error=/dataset1/zailong/runs/peft-sft-lab/hns-spectral-standardization-2x4-20260911/score-%A_%a.err
#SBATCH --partition=defq
#SBATCH --array=0-1%2
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

set -euo pipefail
cd /dataset1/zailong/workspace/peft-sft-lab
export HUMANEVAL_FORCE_SUBPROCESS=1
BASES=(Qwen3-8B Llama-3.1-8B-Instruct)
BASE_NAME="${BASES[$SLURM_ARRAY_TASK_ID]}"
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/score_forgetting_matrix.py \
  --matrix_dir "/dataset1/zailong/runs/peft-sft-lab/hns-spectral-standardization-2x4-20260911/eval/$BASE_NAME" \
  --workers 16
