#!/usr/bin/env bash
#SBATCH --job-name=hns-seeds-3task
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=B300q
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%2
#SBATCH --time=2-00:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=32

set -euo pipefail

REPO_ROOT=/dataset1/zailong/workspace/peft-sft-lab
cd "$REPO_ROOT"

BASES=(Qwen3-8B Llama-3.1-8B-Instruct Qwen3-8B Llama-3.1-8B-Instruct)
SEEDS=(43 43 44 44)

BASE_NAME="${BASES[$SLURM_ARRAY_TASK_ID]}"
TRAIN_SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "[Start] job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID base=$BASE_NAME seed=$TRAIN_SEED host=$(hostname)"
bash scripts/run_hns_training_seed_3task.sh "$BASE_NAME" "$TRAIN_SEED"
