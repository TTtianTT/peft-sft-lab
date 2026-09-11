# HNS capability-retention / forgetting protocol

Date: 2026-09-11. Status: designed and frozen; no GPU job submitted.

## Research question

The primary question is not merely whether an HNS adapter scores higher than its LoRA source on an
off-target benchmark. It is:

> At comparable target-task performance, does HNS preserve more pretrained capability than the original
> LoRA update and reasonable scale-only controls?

This distinction is essential because HNS usually reduces adapter Frobenius norm. A model can move closer to
the base model simply because its entire task adaptation has been weakened.

## Hypotheses

For checkpoint \(c\), let \(t(c)\) be its training task, \(b\) an evaluation task, and
\(S_{c,v,b}\) the score of variant \(v\). Let \(S_{0,b}\) be the corresponding pretrained-base score under
the same prompt and decoding protocol.

Define signed capability change relative to base:

\[
D_{c,v,b}=S_{c,v,b}-S_{0,b}.
\]

Negative \(D\) is forgetting; positive \(D\) is transfer. Define HNS recovery over LoRA:

\[
P_{c,b}=D_{c,\mathrm{HNS},b}-D_{c,\mathrm{LoRA},b}
=S_{c,\mathrm{HNS},b}-S_{c,\mathrm{LoRA},b}.
\]

The equal-task macro recovery over the three non-target task families is

\[
\bar P_c=\frac{1}{3}\sum_{b\ne t(c)}P_{c,b}.
\]

- **H1, raw retention:** \(\bar P_c>0\), while HNS target performance is not meaningfully worse than
  LoRA.
- **H2, shape-specific retention:** common-basis HNS is better than the per-module HNS-norm-matched scalar
  at comparable target performance.
- **H3, scale-adjusted retention:** HNS lies above the global-LoRA-scalar target/retention Pareto envelope.

H1 alone says that the edited checkpoint forgets less. H2 or H3 is required to attribute that benefit to
spectral redistribution rather than weaker adaptation.

## Checkpoints and task matrix

Use all eight all-module pairs: Qwen3-8B and Llama-3.1-8B-Instruct, each with Magicoder, MetaMath, Tulu and
Commonsense adapters. Paths are frozen in
`configs/hns_forgetting_2x4_20260911.json` and have been checked for adapter weights.

Every checkpoint is evaluated on all four benchmark families:

| Training adapter | Target | Retention anchors |
|---|---|---|
| Magicoder | HumanEval | GSM8K, IFEval, commonsense-8 |
| MetaMath | GSM8K | HumanEval, IFEval, commonsense-8 |
| Tulu | IFEval | HumanEval, GSM8K, commonsense-8 |
| Commonsense | commonsense-8 | HumanEval, GSM8K, IFEval |

Use full local evaluation sets: HumanEval 164, GSM8K 1,319, IFEval 541, and the eight full commonsense test
sets (22,419 questions). Commonsense primary score is the equal-weight macro average of its eight tasks, not
a micro average dominated by HellaSwag. GSM8K strict accuracy is primary and numeric equivalence secondary;
IFEval prompt-level strict accuracy and HumanEval pass@1 are primary.

The frozen commonsense inputs are reconstructed from the prior complete per-item evaluation artifacts, which
store immutable IDs, questions, choices, labels and rendered user instructions. This avoids a hidden network
dependency in the PIQA/SIQA loaders and guarantees the same 22,419-item data version used by the mechanism
study; prior model responses are ignored.

Within each base/benchmark cell, render every variant from one canonical tokenizer and chat template before
generation and verify that prompt token IDs are identical. The base condition means the same rendered prompt
with no LoRA request; it must not silently switch to a different template.

These are capability-retention anchors, not perfect mutually exclusive domains. Tulu in particular is broad
and may contain math, code or knowledge-like instructions. Therefore a positive off-target change is called
transfer, and a negative change forgetting; it is not automatically interpreted as preservation of an
unseen domain.

## Variants and causal controls

### Observed-checkpoint comparison

Evaluate pretrained base, untouched LoRA and the existing all-module HNS adapter. This answers whether the
actual saved HNS checkpoint shows less forgetting, but it does not isolate shape because the LoRA/HNS factor
representations differ.

### Common-basis comparison

For every module compute one fp32 SVD of the original LoRA update and store every formal control as

\[
A=V^T,\qquad B=U\operatorname{diag}(d).
\]

Only \(d\) changes. Construct:

1. common-basis LoRA, \(\gamma=1\);
2. common-basis HNS;
3. per-module scalar with
   \(\gamma_m=\|\Delta W_{\mathrm{HNS},m}\|_F/\|\Delta W_{\mathrm{LoRA},m}\|_F\);
4. global scalar curve \(\gamma\in\{0,0.25,0.40,0.55,0.70,0.85,1\}\), where \(\gamma=0\) is the
   true base model.

The HNS-versus-per-module comparison fixes every module's Frobenius norm and singular vectors. The global
curve asks whether any retention advantage is obtainable by uniformly withdrawing LoRA strength.

Do not select one scalar using the retention results. For target-matched analysis define the feasible set

\[
G_c(\epsilon)=\{\gamma:S_{c,\gamma,t(c)}\ge
S_{c,\mathrm{HNS},t(c)}-\epsilon\},
\]

with primary \(\epsilon=1\) pp and fixed sensitivity values 0.5 and 2 pp. Compare HNS with the **best-retaining
feasible scalar**. Giving the scalar the best feasible point is conservative for the HNS claim. If no nonzero
scalar is feasible, report the entire frontier rather than extrapolating between accuracy points.

## Numerical gate

Before formal evaluation:

1. Verify tensor names, ranks and module coverage for all eight pairs.
2. Record fp32 LoRA/HNS basis projection residuals, saved-dtype update errors, per-module norms and tokenizer
   template hashes. Do not conflate saved-update error with runtime function equality.
3. Use a fixed 64-prompt diagnostic per base, covering all four task formats. Run two fresh processes, with
   adapter order reversed in the second.
4. Require 100% token-ID repeatability for each identical adapter before proceeding. Representation
   differences are recorded but do not count as repeatability failure.

The formal common-basis comparisons remain primary even if original and reconstructed representations have
similar aggregate accuracy.

## Statistical analysis

Save one row per evaluation item and align all comparisons by immutable item ID.

- Paired 20,000-draw bootstrap confidence intervals for score differences.
- Exact McNemar tests for binary per-item outcomes; HumanEval is paired by problem and IFEval by prompt.
- Commonsense bootstrap is stratified by its eight subtasks and aggregates equal-weight subtask scores.
- Report repair and break counts, not only net accuracy.
- The primary family contains 16 tests: eight HNS-minus-LoRA non-target macro recoveries and eight
  HNS-minus-best-feasible-scalar gaps. Apply Holm correction.
- HNS-minus-per-module shape controls and individual off-target/category results are secondary and use
  BH-FDR. Report unadjusted intervals as well.
- Bootstrap the scalar-envelope construction itself: on every resample, recompute the feasible set and its
  best-retaining point. Do not treat the selected scalar as fixed after inspecting the same sample.

Use a 1 pp target-performance tolerance for the primary Pareto comparison, with 0.5 and 2 pp as preregistered
sensitivity analyses. A result crossing zero remains unresolved; it is not evidence of equivalence.

## Outcome taxonomy

| Observation | Interpretation |
|---|---|
| LoRA below base; HNS closer to base without target loss | forgetting recovery |
| LoRA below base; HNS reaches/exceeds base | forgetting prevention on that anchor |
| Both above base and HNS higher | additional transfer, not forgetting reduction |
| HNS closer to base but target performance falls | rollback of adaptation; insufficient evidence |
| HNS beats common PerModule and scalar frontier | evidence for shape-specific retention efficiency |
| HNS matches the scalar frontier | retention explained by adapter-strength calibration at current power |
| HNS is below the scalar frontier | HNS is an inefficient retention tradeoff in that setting |

## One-B300 execution plan

Use exactly one `srun --partition=B300q --gres=gpu:1` allocation. Run the two bases sequentially inside the
same allocation; never start a second GPU job.

To use the B300 efficiently, implement a multi-adapter evaluator rather than starting one engine per adapter:

- Load each base once, with all variants for that base available as vLLM LoRA requests.
- Interleave prompts across adapter IDs so continuous batching fills the device.
- Use `gpu_memory_utilization=0.92`, `max_num_batched_tokens=65536`, up to 256 short-output sequences or 128
  long-output sequences concurrently; lower only if the numerical smoke test shows an OOM.
- Keep synchronous scheduling, disable prefix caching, set `VLLM_BATCH_INVARIANT=1`, and save complete token
  IDs. On SM100 this may trade some speed for deterministic attention scheduling.
- Run the commonsense short-output pass separately from GSM8K/HumanEval/IFEval long-output passes so the
  scheduler does not strand KV cache behind very long generations.
- Generate HumanEval outputs first and score them with the allocated CPU workers while subsequent GPU
  generation continues. Save incremental manifests so the single allocation can resume a failed task without
  repeating completed outputs.

There are roughly forty non-base adapter variants per base. Rank-16 adapter weights are small relative to a
B300; residency and continuous batching, rather than tensor parallelism, should be used to exploit memory.
The base outputs are generated only once per base/task and reused in all paired contrasts.

### Numerical-gate amendment (2026-09-11)

Qwen passed exact token repeatability in all 164 task/condition cells. Llama did not: long autoregressive
outputs amplified small numerical differences across otherwise identical processes. The failed exact-token
gate remains recorded and is not relabeled as a pass. The Llama formal matrix is accompanied by a second,
reverse-order run of the 20 prespecified core variants (`original_lora`, `original_hns`, `common_lora`,
`common_hns`, and `common_per_module`). Llama claims require agreement across the two runs and must exceed
the observed outcome-level repeatability variation.

## Required outputs

- `experiment_manifest.json`
- `numerical_gate.json`
- `retention_scores.tsv`
- `forgetting_contrasts.tsv`
- `pareto_points.tsv`
- `pareto_summary.tsv`
- `category_retention.tsv`
- `per_item_predictions/`
- `forgetting_report.md`

The report must answer separately: whether the saved HNS checkpoints forget less, whether the conclusion
survives a common factor basis, and whether it survives target-performance-matched scalar controls.

## Decision rule

Call “HNS reduces forgetting” a cross-checkpoint finding only if the direction is replicated across both base
models, the Holm-adjusted primary recovery is supported, and target performance remains within the fixed
tolerance. Call it a shape-specific mechanism only if common HNS also beats both the per-module norm-matched
control and the best feasible global-scalar frontier. Otherwise the conclusion is restricted to individual
checkpoint/task observations.
