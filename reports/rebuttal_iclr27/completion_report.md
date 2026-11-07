# ICLR 2027 rebuttal completion report

Completed: 2026-11-07 15:56 Singapore time. All E0-E6 packages and their saved output hashes were independently rechecked after completion. Slurm queue is empty; active GPU count0.

696 new main-protocol evaluation cells and7 new trainings completed. The fixed18-checkpoint E0-E3 cohort was not retrained. E4 uses Qwen seed42 and validation-only selection. Optional GPU extensions were omitted. E5 is the requested scope-comparison exception, with no LoDA GPU run; E6 is writing only.

| Experiment | New cells | Result package |
| --- | ---: | --- |
| E0 | 144 | [E0/report.md](E0/report.md) |
| E1 | 72 | [E1/report.md](E1/report.md) |
| E2 | 216 | [E2/report.md](E2/report.md) |
| E3 | 216 | [E3/report.md](E3/report.md) |
| E4 | 48 | [E4/report.md](E4/report.md) |
| E5 | 0 | [E5/report.md](E5/report.md) |
| E6 | 0 | [E6/report.md](E6/report.md) |

All packages include report.md, summary.json, checkpoint_results.tsv and paired_ci.tsv, plus coverage/artifact audits. The writing-only E5/E6 TSVs deliberately have headers only. See completion_summary.json for output SHA256s.

## Main paired comparisons

Differences are percentage points with95% descriptive paired-checkpoint intervals (20,000 draws). Test items are shared; these are not IID training-run confidence intervals. FG lower is favorable.

| Comparison | Target | Off | FG |
| --- | ---: | ---: | ---: |
| flat_nuclear − hns_f4_s1 | -0.070 [-0.431, +0.295] | -0.127 [-0.336, +0.079] | +0.038 [-0.113, +0.182] |
| flat_nuclear − flat_fro | +0.283 [-0.372, +0.901] | +0.454 [+0.123, +0.809] | -0.195 [-0.416, -0.005] |
| recon_0 − original_lora | +0.479 [+0.028, +0.930] | +0.215 [-0.010, +0.435] | -0.132 [-0.324, +0.065] |
| hns_f4_s1 − recon_0 | +3.607 [+2.078, +5.346] | +3.026 [+1.761, +4.363] | -2.232 [-3.259, -1.376] |
| hns_od − ss_smooth_abs | +1.432 [+0.715, +2.383] | +1.481 [+0.984, +1.974] | -1.285 [-1.762, -0.832] |
| hns_f4_s1 − ss_smooth_abs_all | +3.804 [+2.544, +5.063] | +3.989 [+2.623, +5.423] | -2.973 [-4.150, -1.952] |

Flat-Nuclear versus HNS intervals include0 on all three metrics; this is inconclusive about superiority, not proof of equivalence. Nuclear-preserving flattening improves Off/FG relative to Flat-Fro in this cohort. HNS gains survive the Recon-0 baseline. The matched-scope Spectral Surgery comparison uses the predeclared original smooth_abs defaults; no test-based setting was selected.

## Data-free factorial experiment

| Effect | Metric | Difference [95% CI] pp |
| --- | --- | ---: |
| shape_main | target | +2.470 [+1.309, +3.703] |
| shape_main | off_score | +1.901 [+1.057, +2.786] |
| shape_main | forgetting_gap | -1.402 [-2.095, -0.821] |
| magnitude_main | target | +1.137 [+0.565, +1.761] |
| magnitude_main | off_score | +1.125 [+0.651, +1.606] |
| magnitude_main | forgetting_gap | -0.830 [-1.196, -0.525] |
| interaction | target | -1.903 [-3.333, -0.712] |
| interaction | off_score | -1.091 [-1.873, -0.339] |
| interaction | forgetting_gap | +1.251 [+0.637, +1.965] |

Both shape and magnitude main effects favor the edit in this fixed design. Interaction is nonzero, so the effects must be interpreted at the specified settings rather than as universally independent contributions. Module scaling/participation-ratio plots and Scalar-E correlations are in E3/.

## Validation-selected sources and rank64

E4 completed all6 rank16 learning-rate runs and the one rank64 run. Magicoder selected LR1e-5 at100% training, MBPP151/200; MetaMath selected LR2e-4 at100%, held-out numeric-answer EM432/500. Warmup_steps157 for Magicoder and442 for MetaMath/rank64 were verified in saved TrainingArguments and actual Trainer schedules. All conditions are single seed; E4 intervals are paired question-level intervals and McNemar tests, not training-run intervals.

| Condition | Method | Target | Off | FG |
| --- | --- | ---: | ---: | ---: |
| magicoder_val_best_r16 | original_lora | 64.024 | 77.563 | 2.126 |
| magicoder_val_best_r16 | recon_0 | 64.024 | 77.979 | 1.728 |
| magicoder_val_best_r16 | hns_f4_s1 | 73.171 | 80.557 | 0.000 |
| magicoder_val_best_r16 | flat_nuclear | 73.171 | 80.497 | 0.000 |
| metamath_val_best_r16 | original_lora | 84.761 | 67.076 | 5.925 |
| metamath_val_best_r16 | recon_0 | 84.534 | 68.042 | 4.964 |
| metamath_val_best_r16 | hns_f4_s1 | 89.158 | 74.277 | 0.000 |
| metamath_val_best_r16 | flat_nuclear | 88.324 | 73.827 | 0.000 |
| metamath_r64 | original_lora | 85.292 | 63.360 | 9.581 |
| metamath_r64 | recon_0 | 85.444 | 63.075 | 9.866 |
| metamath_r64 | hns_f4_s1 | 89.234 | 74.837 | 0.000 |
| metamath_r64 | flat_nuclear | 89.083 | 74.577 | 0.185 |

Rank64 has 45/16128 source singular directions below0.005 of the module Frobenius norm (0.279%). HNS versus Flat-Nuclear intervals include0 for Target, Off and FG in this rank64 condition; the result does not establish HNS superiority. Full paired item comparisons, repair/damage counts and tail histograms are in E4/.

## Execution and provenance

| Slurm job | State | GPUs | Elapsed |
| --- | --- | ---: | --- |
| 14888 | Failed before generation | 2 | 00:03:03 |
| 14889 | Completed: E1/E0 | 2 | 01:50:08 |
| 14892 | Completed: E2/E3 | 2 | 03:43:36 |
| 14893 | Completed: E4/E5/E6 | 2 | 12:46:10 |

Jobs ran serially with maximum2 GPUs globally. Total allocation time including the failed preflight was 36.765 GPU-hours; this includes CPU audits/IO and is not kernel-active GPU time. The14888 failure came from a builder retaining an exclusive CUDA context; construction was isolated into an exited subprocess for the successful retry. Existing adapters were reused.

Recon-0 follows the actual main HNS balanced reconstruction function, correcting the unbalanced description in the proposed plan. Evaluation prompts, templates, parsers and scorers remained fixed. Cached cells, adapter/config hashes, module coverage and reconstruction errors are audited. Raw weights/predictions stay in ignored runtime/ paths. Original user changes were preserved.

Core experiment reports were already committed and pushed through result commit4b0e62f; this completion index updates the final queue/coverage records. Submission details are in submission.json; code/test verification is in validation.json.

<!-- R2-O1 -->

## Round 2: R2-O1

Status: complete; new cells: 176. [Full report](R2-O1/report.md), [summary](R2-O1/summary.json). Protocol, coverage and artifact audits are in the same directory.

<!-- R2-O2 -->

## Round 2: R2-O2

Status: complete; new cells: 144. [Full report](R2-O2/report.md), [summary](R2-O2/summary.json). Protocol, coverage and artifact audits are in the same directory.

<!-- R2-O3 -->

## Round 2: R2-O3

Status: complete; new cells: 72. [Full report](R2-O3/report.md), [summary](R2-O3/summary.json). Protocol, coverage and artifact audits are in the same directory.

<!-- R2-O4 -->

## Round 2: R2-O4

Status: complete; new cells: 216. [Full report](R2-O4/report.md), [summary](R2-O4/summary.json). Protocol, coverage and artifact audits are in the same directory.
