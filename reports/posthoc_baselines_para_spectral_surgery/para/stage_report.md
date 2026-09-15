# PARA stage report

Status: complete. All target-task cells were generated and scored before any off-task cell was launched.

| Method | n | Target % | Off % | FG pp |
| --- | ---: | ---: | ---: | ---: |
| base | 18 | 66.4730 | 69.9931 | 0.0000 |
| original_lora | 18 | 67.9980 | 69.5512 | 2.7082 |
| hns_f4_s1 | 18 | 72.0834 | 72.7915 | 0.3441 |
| para_e90 | 18 | 68.7778 | 70.0126 | 2.3706 |
| para_e95 | 18 | 68.5823 | 69.6357 | 2.6954 |
| para_e99 | 18 | 68.4421 | 69.7776 | 2.5637 |

## Method and implementation audit

This is epsilon-PARA from [arXiv:2604.27796](https://arxiv.org/abs/2604.27796), not DG-Hard or per-module top-k: each LoRA B@A update is decomposed with compact QR/SVD, then a single checkpoint-global threshold is applied to the pooled squared effective singular values. Retained singular values are unchanged and no nuclear/Frobenius restoration is applied. The paper says code will be published upon acceptance; no official implementation was available at execution time.

## Compression audit

All three predeclared epsilon values are reported; none was selected by test performance. The global threshold uses effective singular values. Since every source module has scaling 2, this is identical to pooling raw BA singular values.

| Method | checkpoints | Actual energy mean [min, max] | Parameter retention mean | Zero-rank modules mean [min, max] |
| --- | ---: | ---: | ---: | ---: |
| para_e90 | 18 | 0.90007335 [0.90000709, 0.90012725] | 0.381589 | 9.72 [0, 44] |
| para_e95 | 18 | 0.95003801 [0.95000838, 0.95007250] | 0.542753 | 0.56 [0, 6] |
| para_e99 | 18 | 0.99001105 [0.99000217, 0.99001756] | 0.801579 | 0.00 [0, 0] |

The maximum unpruned reconstruction relative error was 1.202e-05; the maximum saved-factor relative error was 2.704e-08. Exact per-checkpoint thresholds and ratios are in `checkpoint_compression.tsv`; all 4,284 module ranks per epsilon/checkpoint are in `module_ranks.tsv`.

## Protocol and reproducibility

Cohort: 2 base models x 3 training tasks x seeds 42/43/44, retaining seed42 and reusing all 18 immutable source checkpoints. Current-wave evaluation uses HumanEval 164, GSM8K 1,319, IFEval 541 and the eight-task commonsense aggregate 22,419; greedy decoding seed 42; max model length 4096; and the existing scorers and FG definition. See the variant manifests, `checkpoint_results.tsv`, `grouped_results.tsv`, `paired_ci.tsv`, command histories and logs under this method directory.
