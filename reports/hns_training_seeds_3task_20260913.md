# HNS three-task training-seed replication summary

Date: 2026-09-13. New training seeds: 43 and 44. Commonsense excluded.

Publication audit correction: the CLI warmup ratios did not take effect. All 12 saved training_args.json files have warmup_steps=0 and no warmup_ratio; Transformers 5.16.1 removed the warmup_ratio argument and the training code filtered it out. These runs used zero warmup. Magicoder selected 50K before truncation filtering; actual training examples were 49,936 for Qwen and 49,941 for Llama. This is an additional reason not to pool the original checkpoint with the new seeds as identical-recipe replicates.

## Completion and protocol

All 12 LoRA training runs completed. All four base/seed evaluation groups completed generation and scoring: 36 cells each, 144 cells total, including benchmark Base, original LoRA, SVD reconstruction control, and nine nonzero HNS variants per task. Slurm 930 finished successfully at 15:23:36 Singapore time; Llama seed43 had already completed in Slurm 928.

Metrics: HumanEval chat pass@1 (164 items), GSM8K strict accuracy (1,319 items), IFEval prompt-level strict accuracy (541 items). Training seed and Trainer data_seed are 43/44; dataset subset seed remains 42. Inference is greedy, seed42, batch-invariant. HNS uses all modules, preserves nuclear norm, strength1, rank16.

## Complete grid

Entries are percentage mean ± sample standard deviation across the TWO new training seeds. This is **not a confidence interval**. Rounded zero SD means identical aggregate benchmark scores, not identical models or per-item predictions. `0+0` is an SVD-factor reconstruction control, not an HNS edit.

| Base | Task | LoRA | 0+0 | 2+0 | 2+1 | 2+2 | 4+0 | 4+1 | 4+2 | 8+0 | 8+1 | 8+2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | HumanEval | 63.72±1.29 | 64.33±0.43 | 73.48±1.29 | 73.78±1.72 | 73.78±0.86 | 73.17±0.86 | 74.39±0.00 | 74.70±1.29 | 75.00±1.72 | 74.70±1.29 | 75.30±0.43 |
| Qwen3-8B | GSM8K | 84.00±0.00 | 84.00±0.11 | 87.45±0.05 | 87.87±0.21 | 87.26±0.43 | 87.34±0.21 | 87.11±0.43 | 87.23±0.05 | 87.41±0.43 | 87.53±0.48 | 87.34±0.32 |
| Qwen3-8B | IFEval | 66.54±0.52 | 68.21±0.52 | 71.90±1.05 | 72.27±1.57 | 72.09±0.52 | 72.46±0.00 | 71.72±1.05 | 71.63±0.13 | 70.89±0.65 | 72.64±0.78 | 71.44±0.65 |
| Llama-3.1-8B-Instruct | HumanEval | 54.57±1.29 | 55.18±0.43 | 56.40±2.16 | 55.79±3.02 | 56.10±0.86 | 54.88±0.86 | 54.57±1.29 | 54.88±0.00 | 55.79±1.29 | 54.88±1.72 | 54.88±0.86 |
| Llama-3.1-8B-Instruct | GSM8K | 74.98±0.54 | 75.21±0.00 | 79.15±0.00 | 78.66±0.27 | 77.79±0.21 | 79.15±0.00 | 78.77±0.54 | 78.28±0.48 | 79.30±0.00 | 78.58±0.38 | 78.58±0.38 |
| Llama-3.1-8B-Instruct | IFEval | 62.94±0.13 | 64.51±0.78 | 64.60±0.91 | 64.14±0.78 | 63.49±0.65 | 63.59±0.78 | 65.06±0.26 | 63.03±1.31 | 64.23±0.65 | 64.05±0.65 | 64.42±1.18 |

## Fixed-setting aggregate

Mean paired gain over original LoRA, percentage points, equally averaged across six base/task groups and two seeds. Win/tie/loss counts use the 12 base/task/seed checkpoints, which are NOT 12 independent training seeds.

| Setting | Mean gain (pp) | Win / tie / loss |
|---|---:|---:|
| 0+0 control | +0.78 | 8 / 2 / 2 |
| 2+0 | +4.37 | 12 / 0 / 0 |
| 2+1 | +4.29 | 11 / 1 / 0 |
| 2+2 | +3.96 | 12 / 0 / 0 |
| 4+0 | +3.97 | 11 / 1 / 0 |
| 4+1 | +4.15 | 10 / 2 / 0 |
| 4+2 | +3.83 | 10 / 0 / 2 |
| 8+0 | +4.31 | 11 / 0 / 1 |
| 8+1 | +4.27 | 11 / 1 / 0 |
| 8+2 | +4.20 | 11 / 0 / 1 |

`2+0` is the descriptive aggregate maximum, not an independently validated hyperparameter choice or evidence of statistically significant superiority. This aggregate excludes Commonsense and is not comparable to the old eight-checkpoint +2.60 pp aggregate.

## Interpretation and limitations

- Qwen HumanEval shows large positive gains; GSM8K gains are positive on both bases for every nonzero HNS setting and both new seeds. These are replicated positive point estimates, not newly computed significance claims.
- Llama HumanEval is weak and setting-dependent: 4+1 ties LoRA on both seeds, while 8+2 is negative on seed43 and positive on seed44. Do not claim universal improvement.
- IFEval reconstruction is a material confound: 0+0 versus original LoRA gains range from +0.92 to +2.40 pp across four checkpoints. Full HNS gains over original LoRA cannot all be attributed to spectral editing without comparing against 0+0 and auditing representation sensitivity.
- Base scores match between new seeds on HumanEval/GSM8K and Qwen IFEval. Llama IFEval Base differs by one correct prompt (62.11% versus 61.92%) between the two evaluation configurations, despite batch-invariant being enabled. Exact inference repeatability is therefore not established for all items.
- The new runs use micro-batch16 for seq4096 and micro-batch64 for seq1024, padding multiple8; old Qwen source run_args used micro-batch1. Global batches were retained, but these are not fully identical execution recipes to seed42.
- Original Llama MetaMath/Tulu model cards did not contain a complete recipe. New MetaMath used the repository's documented Llama-matching math profile values; Tulu used seq1024/GBS128 from the model card and remaining profile-derived settings. Exact agreement with the old checkpoint is not verified.
- Do not directly pool seed42/43/44 as a strict same-configuration three-training-seed result. Report seeds43/44 as a B300 replication under the documented recipe until original-recipe comparability is resolved. Two seeds do not eliminate randomness and give limited information about training variance.

## Artifacts

Raw run root: `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/`.

For each `<base>/seed<43|44>/`: training recipes and weights are in `lora/<task>/`; task config in `config/task_config.json`; HNS manifests in `hns_step_grid/adapters/<base>/`; generated/item-scored rows and metrics in `hns_step_grid/eval/<task>/<variant>/`; complete summaries in `hns_step_grid/eval/score_manifest.json`.
