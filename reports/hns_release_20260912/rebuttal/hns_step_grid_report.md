# HNS step-grid: 2 base models x 4 in-domain tasks

Date: 2026-09-12

## Protocol

- Base models: Qwen3-8B and Llama-3.1-8B-Instruct.
- One LoRA checkpoint per base/task pair: Magicoder, MetaMathQA, Tulu/Instruction-Following, and Commonsense170K.
- HNS scope: all LoRA modules.
- All variants preserve the original module nuclear norm, use HNS strength 1.0, keep the original output rank, and differ only in `(fast_steps, stable_steps)`.
- Grid: `{2,4,8} x {0,1,2}`. The existing `4+1` and `8+2` settings were rebuilt and rerun through the same implementation as anchors.
- `0+0` is an SVD-factor reconstruction control, not an HNS edit.
- Evaluation is diagonal/in-domain only and uses the full available test sets: HumanEval (164), GSM8K (1319), IFEval (541), and Commonsense-8 (22,419 total; equal-task macro accuracy).
- Greedy vLLM inference uses batch-invariant kernels, seed 42, and one persistent engine per base. The final stable run used 65,536 batched tokens; 131,072 triggered illegal-memory-access failures on this vLLM/B300 stack and its incomplete outputs were discarded.

## Full primary-metric results (%)

| Base | Task | Base | LoRA | 2+0 | 2+1 | 2+2 | 4+0 | 4+1 | 4+2 | 8+0 | 8+1 | 8+2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | HumanEval | 66.46 | 66.46 | 75.61 | 74.39 | **76.22** | 74.39 | 75.00 | 75.61 | 75.00 | 75.00 | 75.61 |
| Qwen3-8B | GSM8K | 85.67 | 84.15 | 88.25 | 88.25 | 88.25 | 88.25 | 88.02 | 88.17 | 88.40 | **88.55** | 88.48 |
| Qwen3-8B | IFEval | 70.61 | 67.65 | 70.06 | 70.79 | 70.06 | **71.35** | 70.24 | 70.43 | 70.61 | 70.43 | 68.95 |
| Qwen3-8B | Commonsense-8 | 82.70 | **90.79** | 90.23 | 90.30 | 90.30 | 90.31 | 90.32 | 90.27 | 90.30 | 90.26 | 90.27 |
| Llama-3.1-8B | HumanEval | 52.44 | 53.66 | 53.66 | 54.27 | 53.66 | 53.66 | **54.88** | 53.66 | 54.27 | **54.88** | 53.66 |
| Llama-3.1-8B | GSM8K | 62.40 | 77.10 | 80.89 | 80.82 | 80.29 | 80.89 | 80.67 | **81.05** | 80.67 | 80.14 | 80.82 |
| Llama-3.1-8B | IFEval | 62.11 | 63.22 | 65.06 | 65.06 | 64.51 | **65.43** | 64.88 | 65.06 | 64.70 | 64.33 | 65.25 |
| Llama-3.1-8B | Commonsense-8 | 71.39 | **87.96** | 87.41 | 87.56 | 87.58 | 87.52 | 87.52 | 87.53 | 87.46 | 87.66 | 87.58 |

Bold denotes the largest observed score in each row among LoRA and nonzero HNS variants. These are test-set descriptive maxima, not independently validated configuration selections.

## Fixed-configuration aggregate

All gains are percentage points relative to the original LoRA, equally averaged over the eight base/task checkpoints.

| Configuration | Mean gain | Median gain | Win / tie / loss |
|---|---:|---:|---:|
| 2+0 | +2.52 | +2.13 | 5 / 1 / 2 |
| 2+1 | +2.56 | +2.50 | 6 / 0 / 2 |
| 2+2 | +2.48 | +1.85 | 5 / 1 / 2 |
| **4+0** | **+2.60** | **+2.96** | 5 / 1 / 2 |
| 4+1 | +2.57 | +2.13 | 6 / 0 / 2 |
| 4+2 | +2.60 | +2.31 | 5 / 1 / 2 |
| 8+0 | +2.55 | +2.22 | 6 / 0 / 2 |
| 8+1 | +2.53 | +2.00 | 6 / 0 / 2 |
| 8+2 | +2.45 | +1.66 | 5 / 1 / 2 |

The two losses for every nonzero configuration are the two Commonsense checkpoints. `4+0` has the largest observed mean and median, but its mean advantage is only about 0.04 pp over `4+1` and 0.15 pp over `8+2`; the experiment does not establish that these HNS configurations differ significantly from one another.

## Paired statistical findings

- Qwen GSM8K: every nonzero configuration improves over LoRA by about +3.87 to +4.40 pp. All paired 95% bootstrap intervals exclude zero and all exact McNemar tests remain significant after within-checkpoint Holm correction.
- Llama GSM8K: every nonzero configuration improves by +3.03 to +3.94 pp, with all paired intervals excluding zero and all Holm-adjusted tests significant.
- Qwen HumanEval: every nonzero configuration improves by +7.93 to +9.76 pp; ordinary paired intervals exclude zero, but exact tests do not survive correction across the ten simultaneous comparisons because HumanEval has only 164 items.
- Qwen IFEval: `2+1` (+3.14 pp) and `4+0` (+3.70 pp) have ordinary paired intervals above zero; no configuration survives ten-way Holm correction.
- Llama HumanEval and IFEval: point estimates are nonnegative, but all paired intervals include zero.
- Qwen Commonsense: every nonzero edit decreases equal-task macro accuracy by -0.47 to -0.56 pp; all paired macro bootstrap intervals exclude zero.
- Llama Commonsense: every nonzero edit has a negative point estimate (-0.30 to -0.55 pp). The intervals exclude zero for several configurations, while `2+2`, `8+1`, and `8+2` remain statistically unresolved.
- The `0+0` reconstruction control stays within roughly +/-0.38 pp on six large-sample task/model pairs. It changes one or two HumanEval outcomes, confirming that factor representation can matter at the one-example level, but it is far too small to explain the roughly +4 pp GSM8K and +8 to +10 pp Qwen HumanEval gains.

## Conclusions

1. There is no universal best step count. The task sign matters much more than the distinction among mature HNS configurations.
2. `4+0` is the strongest fixed candidate in this grid: it has the best observed eight-checkpoint mean/median and is simpler than `4+1` or `8+2`. This is a descriptive recommendation, not evidence that it is statistically superior to the anchors.
3. If a hybrid fast+stable configuration is required, `2+1` is the most economical candidate and matches the anchors on average.
4. Additional stable steps rapidly saturate the parameter spectrum. Across checkpoints, one stable step generally drives mean effective rank close to 16 and makes the Frobenius ratio nearly identical to deeper settings. Downstream scores do not improve monotonically with this extra flattening.
5. For Commonsense-trained adapters, the correct policy in this grid is no spectral edit: original LoRA beats every nonzero HNS configuration on both bases. HNS nevertheless remains far above Base because the task LoRA gain is very large.
6. The cleanest positive cross-base regime is MetaMath/GSM8K; the cleanest negative cross-base regime is Commonsense. Code and instruction-following remain base-dependent in statistical certainty.

## Artifacts

- Git-tracked summary tables: `reports/hns_release_20260912/data/step_grid/summary/`
- Git-tracked generation, score, variant and adapter-build manifests:
  `reports/hns_release_20260912/data/step_grid/manifests/`
- Raw generations and scored rows: `/dataset1/zailong/runs/peft-sft-lab/hns-step-grid-2x4-20260912/eval/`
- Built adapters and per-module spectral metadata: `/dataset1/zailong/runs/peft-sft-lab/hns-step-grid-2x4-20260912/adapters/`
- Full score table: `/dataset1/zailong/runs/peft-sft-lab/hns-step-grid-2x4-20260912/summary/summary.tsv`
- Wide main table: `/dataset1/zailong/runs/peft-sft-lab/hns-step-grid-2x4-20260912/summary/main_table.tsv`
- Step aggregate: `/dataset1/zailong/runs/peft-sft-lab/hns-step-grid-2x4-20260912/summary/step_aggregate.tsv`
- Paired bootstrap and McNemar-Holm statistics: `/dataset1/zailong/runs/peft-sft-lab/hns-step-grid-2x4-20260912/summary/paired_stats.tsv`
- Spectral summaries: `/dataset1/zailong/runs/peft-sft-lab/hns-step-grid-2x4-20260912/summary/spectral_summary.tsv`
