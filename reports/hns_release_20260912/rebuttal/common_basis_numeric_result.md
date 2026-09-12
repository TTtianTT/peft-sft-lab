# Qwen MetaMath common-basis numerical recheck

Date: 2026-09-10

## Bottom line

The numerical gate passed, but the controlled result does **not** establish that HNS is better than a
reasonably calibrated global LoRA scalar. Under one shared singular-vector representation, HNS retained a
small positive point estimate over the per-module norm-matched control, but its interval crossed zero. HNS
and the calibration-selected global scalar were effectively tied.

This places the result in the prespecified outcome branch: retain spectral redistribution as a possible
mechanistic contribution, narrow the method claim, and stop extending the normalization/direction-selection
branch on these same GSM8K questions.

## Run integrity

- Slurm job `773` (`hns-commonuv`) completed on one B300 in `00:21:28`, exit code `0:0`.
- Only one project GPU was used. Job `752` was not touched.
- All 252 LoRA modules were included.
- One fp32 SVD of the original LoRA update was used per module. Every common-path adapter used
  \(A=V^T\), \(B=U\operatorname{diag}(d)\); only \(d\) changed.
- Maximum saved-update reconstruction error was `0.0`; maximum HNS basis-alignment error was
  `8.40e-4`.
- The deterministic evaluation used batch invariance, synchronous scheduling, disabled prefix caching,
  fixed question order, greedy decoding, and fixed seeds. Complete output token IDs were saved.

## Numerical gate

The 32-question diagnostic deliberately included the 22 questions that had changed across earlier runs.
Every identical adapter produced exactly the same token sequence in two same-order and one reverse-order
fresh process: **32/32 for all adapter/run comparisons**. The formal evaluation therefore proceeded.

Determinism did not eliminate factor-representation sensitivity:

| Same intended update comparison | Exact tokens | Strict-status changes |
|---|---:|---:|
| Untouched LoRA vs zero rebuild | 22/32 | 3 |
| Original-factor PerModule vs common-basis PerModule | 19/32 | 3 |
| Existing HNS vs common-basis HNS | 20/32 | 5 |

The enriched 32-question set is not an effect-size estimate, but it confirms why the common-path comparison
was necessary: changing the low-rank factor representation can change generated sequences even when the
implied update is numerically the same or extremely close.

## Global-scalar calibration

The scalar was selected once on the fixed 256-question calibration split by strict accuracy; ties would favor
the value closest to one.

| \(\gamma\) | Strict | Numeric |
|---:|---:|---:|
| 1.00 | 84.77% | 85.16% |
| 0.85 | 85.55% | 86.72% |
| 0.70 | 85.94% | 87.50% |
| 0.60 | 85.16% | 86.72% |
| 0.50 | 86.72% | 88.67% |
| **0.40** | **88.67%** | **91.02%** |

The locked choice was \(\gamma=0.40\). It is the lower boundary of the predefined grid, so this experiment
does not identify the scalar optimum. The grid was not extended after seeing the result.

## Fixed 512-question validation

| Adapter | Strict | Numeric |
|---|---:|---:|
| Original LoRA | 83.40% | 84.57% |
| Zero rebuild | 82.81% | 83.98% |
| Selected global scalar, \(\gamma=0.40\) | 85.74% | **88.87%** |
| Common-basis PerModule | 84.77% | 86.72% |
| Common-basis HNS | **86.13%** | 88.67% |
| Existing HNS representation | 86.72% | 89.45% |

Primary paired comparisons:

| Comparison | Endpoint | Delta | Paired bootstrap 95% CI | Repair / break | McNemar \(p\) |
|---|---|---:|---:|---:|---:|
| HNS − PerModule | Strict | +1.37 pp | [−0.78, +3.52] | 20 / 13 | 0.296 |
| HNS − PerModule | Numeric | +1.95 pp | [0.00, +4.10] | 20 / 10 | 0.099 |
| HNS − selected global | Strict | +0.39 pp | [−1.56, +2.34] | 13 / 11 | 0.839 |
| HNS − selected global | Numeric | −0.20 pp | [−1.76, +1.56] | 9 / 10 | 1.000 |

Relative to the same-factorization zero-rebuild baseline, both HNS and the selected scalar improved:

- HNS: strict `+3.32 pp` (95% CI `[+0.59,+6.05]`), numeric `+4.69 pp`
  (`[+2.15,+7.42]`).
- Selected scalar: strict `+2.93 pp` (`[0.00,+5.86]`), numeric `+4.88 pp`
  (`[+2.15,+7.62]`).

## Interpretation

1. **The earlier HNS-versus-PerModule advantage was partly implementation-sensitive.** On the prior mixed
   factor paths it was roughly `+3.71/+4.49 pp`; on the common path it fell to `+1.37/+1.95 pp` and is not
   statistically resolved.
2. **There remains a shape-specific trend, not a confirmed gain.** HNS and PerModule have the same
   per-module Frobenius norms and shared \(U,V\), so their point difference isolates singular-value
   redistribution. The interval and McNemar test do not support a stable superiority claim.
3. **Simple adapter shrinkage is a serious alternative explanation.** A calibrated global scalar matched
   HNS on validation despite using a much smaller \(\gamma=0.40\) than HNS's total-norm-matched
   \(\gamma\approx0.609\). HNS therefore did not show added practical value over scalar calibration here.
4. **This is not independent confirmation.** Earlier project runs evaluated all 1,319 GSM8K test examples.
   The current 256/512 splits can validate numerical implementation and paired behavior, but not provide new
   downstream evidence.

## Project decision

Do not claim that HNS is generally superior to a tuned LoRA scaling coefficient from the current MetaMath
evidence. Preserve the dominant-mode/functional-concentration results as mechanism observations and describe
the common-basis HNS-versus-PerModule result as an unresolved positive trend. Per the stopping rule, do not
add doses, direction selection, or more same-question sampling. A future method comparison should use a
genuinely independent math benchmark or checkpoint and lock the scalar grid in advance.

Raw artifacts are under
`/dataset1/zailong/runs/peft-sft-lab/hns-common-basis-numeric-20260910/qwen_metamath`.
