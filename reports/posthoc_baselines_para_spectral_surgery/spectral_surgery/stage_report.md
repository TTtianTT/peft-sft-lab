# Spectral Surgery stage report

Status: complete. All target-task cells were generated and scored before any off-task cell was launched.

| Method | n | Target % | Off % | FG pp |
| --- | ---: | ---: | ---: | ---: |
| base | 18 | 66.4730 | 69.9931 | 0.0000 |
| original_lora | 18 | 67.9980 | 69.5512 | 2.7082 |
| hns_f4_s1 | 18 | 72.0834 | 72.7915 | 0.3441 |
| spectral_surgery_grad_direction | 18 | 68.7637 | 68.2820 | 3.7225 |

## Method, configuration and timing audit

This run uses the repository's early [Spectral Surgery](https://arxiv.org/abs/2603.03995) implementation and the fixed representative gradient-guided configuration; it is not relabeled as an all-module HNS variant.

The fixed configuration was chosen before these evaluations from the paper's principal guided-vs-random experiment and the repository's publication figure/run configuration (`grad_direction_residual_l1_calib128`), not from current test scores: grad_direction, residual-writing o_proj/down_proj only, 128 shuffled training examples from each checkpoint's own training dataset/split, calibration seed 42, answer-token teacher forcing, mean-absolute gradient normalization, asymmetric multiplicative update (eta_suppress=2, eta_enhance=0.2), and L1/nuclear preservation.

Across 18 checkpoints, gradient calculation took 487.5s total (mean 27.1s/checkpoint), while end-to-end editing took 1061.2s total (mean 59.0s/checkpoint). Per-checkpoint dataset paths, hashes, sample counts, module counts and timings are in `checkpoint_edits.tsv`.

## Protocol and reproducibility

Cohort: 2 base models x 3 training tasks x seeds 42/43/44, retaining seed42 and reusing all 18 immutable source checkpoints. Current-wave evaluation uses HumanEval 164, GSM8K 1,319, IFEval 541 and the eight-task commonsense aggregate 22,419; greedy decoding seed 42; max model length 4096; and the existing scorers and FG definition. See the variant manifests, `checkpoint_results.tsv`, `grouped_results.tsv`, `paired_ci.tsv`, command histories and logs under this method directory.
