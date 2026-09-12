#!/usr/bin/env bash
#SBATCH --job-name=hns-zrestore-summary
#SBATCH --output=/dataset1/zailong/runs/peft-sft-lab/hns-restored-standardization-2x4-20260911/summary-%j.out
#SBATCH --error=/dataset1/zailong/runs/peft-sft-lab/hns-restored-standardization-2x4-20260911/summary-%j.err
#SBATCH --partition=defq
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /dataset1/zailong/workspace/peft-sft-lab
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/summarize_2x4_restored_standardization.py \
  --run_root /dataset1/zailong/runs/peft-sft-lab/hns-restored-standardization-2x4-20260911
