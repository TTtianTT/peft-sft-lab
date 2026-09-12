#!/usr/bin/env bash
#SBATCH --job-name=hns-zrestore-2x4
#SBATCH --output=/dataset1/zailong/runs/peft-sft-lab/hns-restored-standardization-2x4-20260911/slurm-%A_%a.out
#SBATCH --error=/dataset1/zailong/runs/peft-sft-lab/hns-restored-standardization-2x4-20260911/slurm-%A_%a.err
#SBATCH --partition=B300q
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --time=04:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16

set -euo pipefail
cd /dataset1/zailong/workspace/peft-sft-lab
BASES=(Qwen3-8B Llama-3.1-8B-Instruct)
bash scripts/run_2x4_restored_standardization_base.sh "${BASES[$SLURM_ARRAY_TASK_ID]}"
