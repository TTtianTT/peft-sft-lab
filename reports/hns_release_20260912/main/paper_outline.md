# Revisiting Post-hoc Spectral Editing of LoRA: Scale, Shape, and Numerical Representation

## One-sentence thesis

Post-hoc spectral editing reliably changes how strongly LoRA directions are used, but neither functional
dominance nor reduced extreme modification determines task utility; credible evaluation must separately
control adapter scale, spectral shape, downstream objective, and low-rank numerical representation.

## Claim hierarchy

The paper should lead with a measurement and evaluation contribution, not a new normalization algorithm.

1. **Primary:** parameter-space concentration and data-conditioned functional concentration are distinct,
   and the latter exposes hidden-state amplification of dominant LoRA modes.
2. **Primary:** intervention magnitude is not signed utility. The same suppression pattern helps math/code,
   is mixed for instruction following, and is neutral or harmful for commonsense.
3. **Primary methodological:** spectral methods require calibrated scale controls and common-factor numerical
   controls; otherwise behavioral differences can be misattributed to spectral shape.
4. **Secondary:** dominant-head suppression remains a plausible task-dependent mechanism, with Qwen
   MetaMath the cleanest positive case, but no universal direction-selection rule is established.

## Proposed abstract

Low-rank adapters often exhibit concentrated singular spectra, motivating post-hoc edits that suppress
dominant modes and redistribute mass toward the tail. We study what such edits change functionally and when
those changes improve downstream behavior. Across two 8B base models and four task families, hidden-state
alignment increases top-mode concentration relative to the parameter spectrum, while HNS consistently
reduces extreme LoRA-induced modifications. These measurements, however, do not predict the sign of task
utility: code and math provide positive regimes, commonsense supplies counterexamples, and instruction
following exhibits strong non-additivity. Matched-scale, localization, and gradient-based controls further
show that functional use intensity, task sensitivity, and set interactions cannot be collapsed into a single
concentration score. Finally, a deterministic common-basis study on Qwen MetaMath finds an unresolved
shape-specific trend and no established advantage over a calibration-selected global LoRA scalar; changing
near-equivalent low-rank factor representations can itself alter greedy generations. Our results recast
post-hoc LoRA spectral editing as a mechanism and evaluation problem involving scale, shape, objective, and
numerical representation rather than a generally validated normalization rule.

## Paper structure

### 1. Introduction

- Motivate post-hoc adapter editing: inexpensive intervention after training, but unclear causal source.
- State the central ambiguity: a spectral edit simultaneously changes parameter shape, effective scale,
  hidden-state-conditioned behavior, and sometimes numerical factorization.
- List the three primary contributions from the claim hierarchy.
- Avoid promising a generally superior method.

### 2. Setup and definitions

- Define rank-16 LoRA update and HNS singular-value edit with fixed singular vectors.
- Separate raw spectral energy from
  \(E_i=\sigma_i^2\mathbb{E}[(v_i^T h)^2]\).
- Define raw/functional top shares, effective ranks, and
  \(\|\Delta Wh\|/\|Wh\|\).
- Define ScalarShrink, ShapeOnly, HeadOnly, TailOnly, Full HNS, and HeadOnly+FroRestore.
- State that the principal 2x4 activation analysis uses the same pretrained-base hidden-state trajectory.

### 3. Experimental design and inferential protocol

- Two bases: Qwen3-8B and Llama-3.1-8B-Instruct.
- Four task families: Magicoder, MetaMath, Tulu/instruction following, commonsense.
- All adapted modules, fixed 256-example activation samples, unchanged \(U,V\) in the six-way study.
- Paired bootstrap, permutation/McNemar, category intervals, multiple-testing correction, and leave-out
  checkpoint sensitivity.
- Explain that repeated generations are diagnostic repeats, not additional statistical samples.

### 4. Parameter dominance becomes functional dominance

- Present the 2x4 raw versus functional spectrum table.
- Show 8/8 alignment amplification and 8/8 p99 suppression.
- Emphasize that this is a descriptive mechanism chain, not yet a utility rule.

### 5. Suppression has task-dependent utility

- Present the six-way intervention table.
- Positive regime: MetaMath across bases; Magicoder with scale caveats.
- Mixed/non-additive regime: Qwen Tulu versus Llama Tulu.
- Negative regime: commonsense, with HellaSwag as a replicated category-specific failure.
- Use HeadOnly+FroRestore to narrow the pure-shape statement to Qwen MetaMath.

### 6. Why concentration does not localize useful edits

- Show checkpoint correlations and leave-out fragility.
- Compare FunctionalTop-K, RawTop-K, uniform random, and matched layer/type random.
- Report F-by-compatibility quadrants; highlight Low-F/High-C rather than High-F/High-C in Llama Tulu.
- Present single-module utility and adapter-set non-additivity.
- Separate teacher-forced SFT-NLL utility from generation/accuracy utility.

### 7. Scale, shape, and numerical representation

- Explain nuclear-norm-preserving flattening's mathematical coupling to Frobenius shrinkage.
- Present MetaMath direct-dose HeadOnly versus matched scalar as unresolved.
- Present the deterministic common-basis experiment:
  - repeatability gate;
  - representation sensitivity diagnostic;
  - HNS versus common PerModule;
  - HNS versus calibrated global scalar.
- State explicitly that no superiority or equivalence result was obtained.

### 8. Discussion and implications

- Functional energy answers “how much is this direction used?”, not “should it be suppressed?”.
- Useful and excessive can coexist at different doses.
- A future task-aware method needs signed downstream sensitivity plus finite-intervention validation.
- A future calibration-free HNS study is a separate research question requiring frozen rules and unseen
  checkpoints, not an extension of the present claim.

### 9. Limitations

- Eight checkpoints but only two base families and four task families.
- Fixed pretrained-base trajectories for the main functional measurements.
- Reuse of GSM8K questions in the common-basis numerical recheck.
- Small direction-level pilots and sparse accuracy flips.
- Greedy generation sensitivity to factor representation; the enriched 32-question diagnostic does not
  estimate population incidence.
- No equivalence/non-inferiority design for HNS versus scalar.

## Recommended main figures and tables

1. **Figure 1:** raw spectrum → functional spectrum → modification magnitude, followed by a split arrow to
   positive and negative downstream regimes. The split is the conceptual point of the paper.
2. **Figure 2:** raw versus functional top-1 for all eight checkpoints, paired within checkpoint.
3. **Figure 3:** modification p99 suppression versus HNS gain, labeled by base/task to expose task-dependent
   sign rather than emphasize a regression line.
4. **Table 1:** `main_results.tsv`, containing the 2x4 spectra, modification and six intervention gains.
5. **Table 2:** HeadOnly+FroRestore and its paired contrast with HeadOnly.
6. **Table 3:** common-basis MetaMath scores and paired HNS-minus-control intervals.
7. **Figure 4 or appendix:** localization and F-by-compatibility results, including matched random.
8. **Appendix table:** failed-predictor ledger, numerical settings, and all category-level results.

## Wording discipline

Use:

> Functional dominance identifies where a spectral edit acts strongly, whereas downstream task utility
> determines whether that action is beneficial.

> HNS consistently suppresses extreme LoRA modifications, but the benefit of doing so is task- and
> context-dependent.

> In a same-question Qwen MetaMath recheck, HNS did not establish superiority over a calibrated global
> scalar; the confidence interval also does not establish equivalence.

Avoid:

- “Functional concentration predicts HNS gain.”
- “Dominant modes are noise.”
- “HeadOnly generally explains HNS.”
- “HNS is a better LoRA normalization method.”
- “Numerically identical adapters behave differently” without distinguishing saved-update checks from
  runtime low-precision execution and near-equivalent factor representations.

## Artifact map

- Unified 2x4 table: `reports/hns_paper_package_20260910/main_results.tsv`
- Claim ledger: `reports/hns_paper_package_20260910/claim_evidence_matrix.tsv`
- Failed predictors: `reports/hns_paper_package_20260910/failed_predictors.tsv`
- Closeout decision: `reports/hns_project_closeout_20260910.md`
- Full 2x4 report: `reports/hns_2x4_mechanism_final_20260910.md`
- Common-basis result: `reports/hns_posthoc_lora_normalization_20260910/common_basis_numeric_result.md`
