# HNS experiment release (2026-09-12)

This directory is the paper/rebuttal-facing index for the Qwen3-8B and
Llama-3.1-8B-Instruct HNS mechanism project. Scores are copied from immutable
run directories under `/dataset1/zailong/runs/peft-sft-lab`; model checkpoints
and generated adapters are intentionally not committed.

## Quick lookup

From the repository root:

```bash
# Read the paper-facing overview and complete method comparison.
less reports/hns_release_20260912/main/all_methods_main_table.md

# Open the two principal mechanism/forgetting reports.
less reports/hns_release_20260912/main/mechanism_main_report.md
less reports/hns_release_20260912/main/forgetting_main_report.md

# Inspect the compact numerical tables.
column -ts $'\t' reports/hns_release_20260912/main/main_results.tsv | less -S
column -ts $'\t' reports/hns_release_20260912/main/retention_scores.tsv | less -S

# Inspect the later all-module HNS iteration sweep.
less reports/hns_release_20260912/rebuttal/hns_step_grid_report.md
column -ts $'\t' reports/hns_release_20260912/data/step_grid/summary/main_table.tsv | less -S

# Verify the copied/compressed release artifacts.
sha256sum -c reports/hns_release_20260912/MANIFEST.sha256
```

| Question | Start here |
|---|---|
| What belongs in the main paper? | `main/` |
| What may answer reviewer questions? | `rebuttal/` |
| Where are spectra and hidden-state modification values? | `data/mechanism/` |
| Where are FunctionalTop-K and matched-random results? | `data/localization/` |
| Where are module-level utility/predictor results? | `data/utility/` |
| Where are Base -> LoRA -> HNS per-item transitions? | `data/forgetting/` |
| Where are scalar and standardization alternatives? | `data/standardization/` |
| Which HNS iteration count works best? | `rebuttal/hns_step_grid_report.md` and `data/step_grid/` |
| Where are the original full run outputs? | `/dataset1/zailong/runs/peft-sft-lab/` |

## Where to start

### Paper-facing material

- `main/all_methods_main_table.md`: every full-adapter method tried, formulas,
  code entry points, and the directly comparable 2x4 main table.
- `main/main_results.tsv`: compact machine-readable mechanism table.
- `main/mechanism_main_report.md`: complete 2-base x 4-task mechanism result.
- `main/mechanism_geometry_for_paper.md`: raw/functional spectrum and geometry.
- `main/forgetting_main_report.md`: cross-task retention/forgetting result.
- `main/retention_scores.tsv`, `forgetting_contrasts.tsv`, `pareto_summary.tsv`:
  tables suitable for manuscript generation.
- `main/claim_evidence_matrix.tsv`: claim-to-evidence guardrail.
- `main/paper_outline.md`: proposed narrative and table placement.

### Rebuttal/supporting material

- `rebuttal/mechanism_audit.json`: coverage and consistency audit.
- `rebuttal/failed_predictors.tsv`: predictors that did not generalize.
- `rebuttal/signed_gate_review.*`: PIQA signed-gate negative result and audit.
- `rebuttal/reward_gradient_*`: MetaMath reward-gradient pilot and stopping rule.
- `rebuttal/direct_dose_*`: HeadOnly dose and matched-scalar control.
- `rebuttal/common_basis_numeric_*`: factorization-controlled HNS/scalar check.
- `rebuttal/*standardization_report.md`: direct and norm-restored alternatives.
- `rebuttal/hns_step_grid_report.md`: full 2x4 all-module sweep over `2/4/8`
  fast steps and `0/1/2` stable steps, including the `0+0` reconstruction
  control and paired uncertainty analysis.
- `rebuttal/llama_metric_stability.json`: run-stability boundary.
- `rebuttal/above_base_rollout_report.md`: why HNS can exceed Base off-task.
- `rebuttal/*transition_summary.tsv`: Base -> LoRA -> HNS item transitions.

### Machine-readable data

- `data/mechanism/`: parameter spectra, functional energy, modification,
  paired inference and robustness tables. `direction_response.csv.gz` is gzip
  compressed and expands to the canonical all-module direction table.
- `data/localization/`: FunctionalTop-K/RawTop-K/random and F x C quadrant
  intervention summaries and paired inference.
- `data/utility/`: module-level utility, features, regression and per-example
  outputs (large per-example tables are gzip compressed).
- `data/forgetting/`: per-item transition table and rollout-stratified checks.
- `data/standardization/`: direct/restored standardization and scaling summaries.
- `data/step_grid/`: full primary-metric tables, 20,000-draw paired bootstrap
  intervals, exact McNemar-Holm tests, spectral summaries, generation/score
  manifests, and per-checkpoint adapter-build manifests for the HNS step sweep.

`MANIFEST.sha256` records checksums for every release artifact except itself.

## Evaluation-family warning

Do not subtract scores across evaluation families. The 2x4 mechanism table,
the later full-set standardization run, the 512-example MetaMath common-basis
check, and the forgetting evaluation use different run contexts. Comparisons
within a panel/run are paired and valid; cross-panel differences are descriptive.

## What is not in Git

The release excludes base-model weights, LoRA/HNS adapter weights, vLLM caches,
full generation dumps and scheduler logs. They are large, may duplicate public
datasets, and are not needed to recover the reported statistics. Their source
paths and run metadata remain recorded in the reports/configuration manifests.
The committed scripts reconstruct interventions from the original LoRA/HNS
paths and reproduce every aggregation in this release.
