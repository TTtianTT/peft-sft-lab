# Norm-restored singular-spectrum standardization: 2 bases × 4 tasks

Date: 2026-09-11

## Bottom line

Restoring each module's scale fixes the catastrophic failure of direct spectral standardization,
but it does **not** produce a general replacement for HNS. Of the four controls, standardization
followed by restoration of the **HNS Frobenius norm** is clearly strongest: it beats the original
LoRA in 5/8 cells and gains 0.38 percentage points on average. It nevertheless loses to HNS in all
8 cells, by 2.16 points on average.

The transform is applied independently to every LoRA module. For original singular values
\(\sigma\), it first forms signed standardized coefficients

\[
z_i = \frac{\sigma_i-\operatorname{mean}(\sigma)}
           {\operatorname{Std}_{\mathrm{population}}(\sigma)},
\]

then multiplies \(z\) by one scalar so that its L2 norm (Frobenius control) or L1 norm (nuclear
control) matches the corresponding original LoRA or HNS module. The update is reconstructed in the
original singular basis. Because norm restoration removes any positive common scale, dividing by
population variance instead of population standard deviation gives the same restored direction.

## Main-task results

All scores are percentages. HumanEval uses pass@1; GSM8K uses strict accuracy; IFEval uses
prompt-level strict accuracy; Commonsense uses the equal-task macro accuracy over eight tasks.

| Base | Task | LoRA | HNS | Restore LoRA Fro | Restore LoRA nuclear | Restore HNS Fro | Restore HNS nuclear |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | HumanEval | 66.46 | **75.00** | 65.24 | 67.07 | 73.17 | 67.07 |
| Qwen3-8B | GSM8K | 84.15 | **88.17** | 82.49 | 82.34 | 84.61 | 82.34 |
| Qwen3-8B | IFEval | 67.65 | **71.16** | 69.50 | 69.13 | 70.24 | 69.13 |
| Qwen3-8B | Commonsense-8 | **90.79** | 90.38 | 88.65 | 88.40 | 88.43 | 88.38 |
| Llama-3.1-8B-Instruct | HumanEval | 53.66 | 54.27 | **56.71** | 54.27 | 54.27 | 53.05 |
| Llama-3.1-8B-Instruct | GSM8K | 77.10 | **80.44** | 74.83 | 73.84 | 74.91 | 74.53 |
| Llama-3.1-8B-Instruct | IFEval | 63.22 | **64.33** | 63.40 | 61.92 | 63.77 | 62.29 |
| Llama-3.1-8B-Instruct | Commonsense-8 | **87.96** | 87.59 | 84.94 | 84.60 | 84.67 | 84.67 |

## Aggregate comparison

| Restored method | Mean delta vs LoRA | Cells above LoRA | Mean delta vs HNS | Cells above HNS |
|---|---:|---:|---:|---:|
| Restore LoRA Frobenius | -0.65 pp | 3/8 | -3.20 pp | 1/8 |
| Restore LoRA nuclear | -1.18 pp | 3/8 | -3.72 pp | 0/8 |
| Restore HNS Frobenius | **+0.38 pp** | **5/8** | **-2.16 pp** | 0/8 |
| Restore HNS nuclear | -1.19 pp | 2/8 | -3.73 pp | 0/8 |

The one restored result that beats HNS is Llama HumanEval with LoRA-Frobenius restoration:
56.71 versus 54.27. This does not generalize to the other seven cells. HNS-Frobenius restoration is
the most consistent control, especially on Qwen, but HNS itself remains better on every task.

## Construction audit and interpretation

- All 252 Qwen or 224 Llama LoRA modules per checkpoint were transformed at rank 16.
- Across all eight checkpoints, maximum target-norm relative error is below \(2.5\times10^{-7}\),
  maximum variance-vs-standard-deviation direction error is below \(2.7\times10^{-7}\), and the
  maximum saved-update reconstruction error is 0.0.
- Centering makes about 73.2% of the coefficients negative across the 1,904 audited modules. Thus
  this is a signed spectral-direction transform, not a nonnegative singular-value redistribution.
- HNS preserves the per-module nuclear norm of the original LoRA up to numerical precision
  (maximum relative discrepancy \(3.5\times10^{-6}\)). The two nuclear-restoration controls are
  therefore mathematically near-duplicates. Small floating-point differences can still change a
  few greedy outputs.
- Both GPU array tasks ran serially (`0-1%1`) after job 809, each requested one GPU, and completed
  successfully with exit code 0. The final CPU summary also completed with exit code 0.

## Decision

Scale mismatch explains much of the collapse seen under direct standardization, because restoring
a module norm recovers broadly usable models. But scale alone does not explain HNS's advantage:
even the best scale-matched standardized shape, HNS-Frobenius restoration, remains below HNS in
8/8 cells. Reject centered spectral standardization as an HNS replacement in its present form.

Raw results and generated adapters are under
`/dataset1/zailong/runs/peft-sft-lab/hns-restored-standardization-2x4-20260911`.

Key files:

- `summary.tsv`: final 48-row score/delta table.
- `summary.json`: machine-readable aggregate.
- `adapters/<base>/<task>/manifest.json`: module-level statistics and construction audit.
- `eval/<base>/score_manifest.json`: benchmark metrics for all generated cells.
