# HNS all-module step-grid data

This directory contains the Git-sized outputs of the 2026-09-12 full
in-domain evaluation over Qwen3-8B and Llama-3.1-8B-Instruct, each with
Magicoder, MetaMathQA, Tulu/Instruction-Following and Commonsense170K LoRA
checkpoints.

The intervention grid is `fast_steps in {2,4,8}` by
`stable_steps in {0,1,2}`, plus a `0+0` SVD-reconstruction control. All
nonzero variants edit every LoRA module, preserve each module's nuclear norm,
use strength 1.0 and retain rank 16.

## Summary files

- `summary/main_table.tsv`: one wide row per base/task checkpoint.
- `summary/summary.tsv`: tidy primary scores and gains relative to LoRA.
- `summary/step_aggregate.tsv`: fixed-configuration mean/median gains and
  win/tie/loss counts across the eight checkpoints.
- `summary/paired_stats.tsv`: 20,000-draw paired bootstrap intervals,
  transition counts and exact McNemar-Holm tests.
- `summary/spectral_summary.tsv`: effective-rank and Frobenius-norm changes.

## Manifests

- `manifests/*_score_manifest.json`: complete benchmark metrics, including
  IFEval category and Commonsense subtask breakdowns.
- `manifests/*_generation_manifest.json`: inference settings and sample counts.
- `manifests/*_variant_manifest.json`: adapter paths and dispatch labels.
- `manifests/*_adapter_manifest.json`: per-checkpoint spectrum summaries and
  intervention construction metadata.

The human-readable interpretation is
[`../../rebuttal/hns_step_grid_report.md`](../../rebuttal/hns_step_grid_report.md).
Full generations, item-level scored rows and adapter weights remain under
`/dataset1/zailong/runs/peft-sft-lab/hns-step-grid-2x4-20260912/` and are not
committed because of their size.

## Regeneration

```bash
# Build and evaluate one base per GPU.
bash scripts/run_hns_step_grid_base.sh Qwen3-8B
bash scripts/run_hns_step_grid_base.sh Llama-3.1-8B-Instruct

# Recreate compact tables and paired statistics.
python scripts/summarize_hns_step_grid.py \
  --run_root /dataset1/zailong/runs/peft-sft-lab/hns-step-grid-2x4-20260912
python scripts/analyze_hns_step_grid_paired.py \
  --run_root /dataset1/zailong/runs/peft-sft-lab/hns-step-grid-2x4-20260912 \
  --bootstrap_draws 20000
```

The final stable B300 runs used `max_num_batched_tokens=65536`; 131,072-token
profiling/generation was not stable with the installed vLLM/Triton stack.
