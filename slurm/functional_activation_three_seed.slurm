#!/usr/bin/env bash
#SBATCH --job-name=functional-activation
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=B300q
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --time=04:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16
set -euo pipefail
cd /dataset1/zailong/workspace/peft-sft-lab
BASES=(Qwen3-8B Llama-3.1-8B-Instruct)
export PYTHONPATH="$PWD/src:$PWD/scripts"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDNN_SDPA_DEPRIORITIZED=1
export TMPDIR=/tmp
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/collect_functional_activation_three_seed.py --base "${BASES[$SLURM_ARRAY_TASK_ID]}"
