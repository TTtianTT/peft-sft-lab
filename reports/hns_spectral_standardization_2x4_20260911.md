# Singular-spectrum standardization as an HNS alternative: 2 bases × 4 tasks

Date: 2026-09-11

## Bottom line

Direct per-module standardization of the LoRA spectrum is **not a viable HNS replacement** in this
2-base × 4-task evaluation. The literal requested transform

\[
c_i = \frac{\sigma_i-\operatorname{mean}(\sigma)}{\operatorname{Var}(\sigma)}
\]

lost almost all task capability. The conventional z-score sensitivity transform, which divides by
the population standard deviation instead, also underperformed the original LoRA in all 8/8 cells.
Neither transform beat HNS in any cell.

Because centering creates negative values, the transformed values are no longer singular values in
the strict mathematical sense. They were implemented as signed coefficients in the original LoRA
singular basis, with `A=Vh` and `B=U*coefficients`. No absolute value, clipping, interpolation, or
post-standardization rescaling was applied.

## Main-task results

All scores are percentages. HumanEval uses pass@1; GSM8K uses strict accuracy; IFEval uses
prompt-level strict accuracy; Commonsense uses the equal-task macro accuracy over eight tasks.

| Base | Task | LoRA | HNS | (σ−μ)/variance | (σ−μ)/std |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | HumanEval | 66.46 | **75.00** | 0.00 | 0.00 |
| Qwen3-8B | GSM8K | 84.15 | **88.17** | 0.00 | 0.68 |
| Qwen3-8B | IFEval | 67.65 | **71.16** | 19.78 | 46.77 |
| Qwen3-8B | Commonsense-8 | **90.79** | 90.38 | 0.00 | 89.08 |
| Llama-3.1-8B-Instruct | HumanEval | 53.66 | **54.27** | 0.00 | 0.00 |
| Llama-3.1-8B-Instruct | GSM8K | 77.10 | **80.44** | 0.00 | 2.43 |
| Llama-3.1-8B-Instruct | IFEval | 63.22 | **64.33** | 14.23 | 9.24 |
| Llama-3.1-8B-Instruct | Commonsense-8 | **87.96** | 87.59 | 0.00 | 0.03 |

The standard-deviation variant's closest result was Qwen Commonsense, but it was still 1.71 pp
below LoRA and 1.30 pp below HNS. Its remaining seven cells degraded by 20.89–87.92 pp relative to
LoRA. The literal variance variant degraded every cell by 47.87–90.79 pp.

## Why direct standardization fails as a controlled HNS alternative

The transform does not preserve the adapter scale. Its total update Frobenius norm relative to LoRA
varied drastically by checkpoint:

| Base | Task | /variance norm ratio | /std norm ratio |
|---|---|---:|---:|
| Qwen3-8B | Magicoder | 818.44× | 15.17× |
| Qwen3-8B | MetaMath | 33.97× | 3.43× |
| Qwen3-8B | Tulu | 17.75× | 2.62× |
| Qwen3-8B | Commonsense | 3.30× | 0.80× |
| Llama-3.1-8B-Instruct | Magicoder | 1251.73× | 17.37× |
| Llama-3.1-8B-Instruct | MetaMath | 59.38× | 4.13× |
| Llama-3.1-8B-Instruct | Tulu | 123.61× | 5.33× |
| Llama-3.1-8B-Instruct | Commonsense | 11.68× | 1.85× |

The literal variance denominator is especially unstable because it has squared units and becomes
small for low-magnitude spectra. Centering also reverses every below-mean singular direction. The
experiment therefore tests the exact direct-standardization proposal, but it is not a scale-matched
test of whether standardized spectral *shape* can replace HNS.

## Integrity and artifacts

- Bases/checkpoints: the same Qwen3-8B and Llama-3.1-8B-Instruct 2×4 LoRA/HNS bank used by the
  existing mechanism work.
- Modules: all 252 Qwen or 224 Llama LoRA modules per checkpoint; rank 16.
- Statistics: per-module population mean and population variance (`correction=0`).
- Reconstruction: maximum saved-update relative error was 0.0 for all eight checkpoints.
- Inference: greedy, batch-invariant vLLM, fixed seed 42, one persistent engine per base, with LoRA,
  HNS, and both standardization variants evaluated in the same run.
- Evaluation sizes: HumanEval 164; GSM8K 1,319; IFEval 541; Commonsense-8 22,419.
- HumanEval's installed package lacked a console entry point and its in-process API conflicted with
  Python 3.12/filelock. Scoring was recovered in a clean module subprocess without regenerating any
  model outputs. A punctuation-only fast rejection was added for degenerate 2,048-line generations.

Raw results and generated adapters are under
`/dataset1/zailong/runs/peft-sft-lab/hns-spectral-standardization-2x4-20260911`.

Key files:

- `summary.tsv`: final 32-row score/delta table.
- `summary.json`: machine-readable aggregate.
- `adapters/<base>/<task>/manifest.json`: per-module statistics, norm ratios, and reconstruction audit.
- `eval/<base>/score_manifest.json`: benchmark-level metrics for all generated cells.

## Decision

Reject direct z-score-style singular-spectrum standardization as an HNS replacement. If this branch
is revisited, the meaningful next control is a **scale-preserving signed shape transform** (for
example, standardize and then restore each module's LoRA or HNS Frobenius/nuclear norm). That would
answer a different question from the exact transform tested here and should be named separately.
