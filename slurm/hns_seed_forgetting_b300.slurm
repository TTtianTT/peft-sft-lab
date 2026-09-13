#!/usr/bin/env bash
#SBATCH --job-name=hns-seed-forgetting
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=B300q
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --time=12:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=32
set -euo pipefail
cd /dataset1/zailong/workspace/peft-sft-lab
BASES=(Qwen3-8B Llama-3.1-8B-Instruct)
bash scripts/run_hns_seed_forgetting_b300.sh "${BASES[$SLURM_ARRAY_TASK_ID]}"
