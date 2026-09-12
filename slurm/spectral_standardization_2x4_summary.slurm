#!/usr/bin/env bash
#SBATCH --job-name=hns-zscore-summary
#SBATCH --output=/dataset1/zailong/runs/peft-sft-lab/hns-spectral-standardization-2x4-20260911/summary-%j.out
#SBATCH --error=/dataset1/zailong/runs/peft-sft-lab/hns-spectral-standardization-2x4-20260911/summary-%j.err
#SBATCH --partition=defq
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16

set -euo pipefail
cd /dataset1/zailong/workspace/peft-sft-lab
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/summarize_2x4_spectral_standardization.py \
  --run_root /dataset1/zailong/runs/peft-sft-lab/hns-spectral-standardization-2x4-20260911
