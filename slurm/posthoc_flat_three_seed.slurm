#!/usr/bin/env bash
#SBATCH --job-name=posthoc-flat
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=B300q
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --time=06:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=32
set -euo pipefail
cd /dataset1/zailong/workspace/peft-sft-lab
BASES=(Qwen3-8B Llama-3.1-8B-Instruct)
export PYTHONPATH="$PWD/src:$PWD/scripts"
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/run_posthoc_three_seed_eval.py --base "${BASES[$SLURM_ARRAY_TASK_ID]}" --phase flat
