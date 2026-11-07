#!/usr/bin/env bash
#SBATCH --job-name=rebuttal27-r2-factor-audit
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=B300q
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
set -euo pipefail
cd /dataset1/zailong/workspace/peft-sft-lab
export PYTHONPATH="$PWD/src:$PWD/scripts"
export CUDA_VISIBLE_DEVICES=""
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/rebuttal_iclr27_r2_factor_audit.py
