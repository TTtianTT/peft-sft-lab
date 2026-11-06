#!/usr/bin/env bash
#SBATCH --job-name=rebuttal27-e1-e0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=B300q
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=384G
#SBATCH --cpus-per-task=64
set -euo pipefail
cd /dataset1/zailong/workspace/peft-sft-lab
export PYTHONPATH="$PWD/src:$PWD/scripts"
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/rebuttal_iclr27.py --pipeline
