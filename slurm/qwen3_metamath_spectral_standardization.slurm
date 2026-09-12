#!/usr/bin/env bash
#SBATCH --job-name=hns-zscore
#SBATCH --output=/dataset1/zailong/runs/peft-sft-lab/hns-spectral-standardization-20260911/slurm-%j.out
#SBATCH --error=/dataset1/zailong/runs/peft-sft-lab/hns-spectral-standardization-20260911/slurm-%j.err
#SBATCH --partition=B300q
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

set -euo pipefail
cd /dataset1/zailong/workspace/peft-sft-lab
bash scripts/run_qwen3_metamath_spectral_standardization.sh
