# Post-hoc spectral scaling pilot protocol

Date: 2026-09-10. This protocol is defined from the CPU-only audit before any new downstream
predictions are generated. It tests whether HNS supplies a useful per-module amplitude allocation,
not whether an individual singular direction is excessive or task-essential.

## Primary question

Does the checkpoint-derived rule

\[
\gamma_m = \frac{\|\Delta W_{m,\mathrm{HNS}}\|_F}
                  {\|\Delta W_{m,\mathrm{LoRA}}\|_F}
\]

outperform a single global LoRA coefficient and structure-matched shuffled allocations when every
scalar variant has the same whole-adapter Frobenius norm?

## Stage 1 checkpoint and split

- Qwen3-8B MetaMathQA LoRA and its all-module Full-HNS adapter.
- Reuse the already locked direct-dose question split: 256 calibration questions and 512 independent
  validation questions. Do not use the previous 128 reward-gradient questions.
- Greedy GSM8K decoding, with strict extracted-answer equality as the primary endpoint and decimal
  numeric equivalence as the secondary endpoint.
- No rule or coefficient is selected from the validation set.

## Fixed variants

1. **Untouched LoRA.** Load the original adapter without SVD reconstruction.
2. **GlobalNormMatched.** Apply one coefficient
   \[
   \gamma_{global}=\sqrt{\frac{\sum_m\|\Delta W_{m,\mathrm{HNS}}\|_F^2}
                                   {\sum_m\|\Delta W_{m,\mathrm{LoRA}}\|_F^2}}.
   \]
3. **PerModuleSpectralScale.** Apply the original module-specific `gamma_m` values.
4. **ShuffledScale-1/2/3.** Permute `gamma_m` among layers within the same module type using fixed
   seeds 20260911, 20260912, and 20260913. Apply one checkpoint-wide correction after permutation so
   that total adapter Frobenius norm exactly matches Full HNS. The frozen mapping is in
   `shuffle_scaling_plan.csv`.
5. **Full HNS.** Use the existing edited adapter.

The CPU audit gives `gamma_global=0.6088123` for Qwen MetaMath. Its module-specific gamma distribution
has p05/median/p95 `0.450/0.673/0.840`; therefore the global and per-module interventions are materially
different despite having the same total norm.

## Numerical implementation

- Construct scalar variants by multiplying the original stored LoRA `B` factor for each module by
  its coefficient. Do not reconstruct `U diag(sigma) V^T` for scalar variants.
- The zero coefficient change must reuse the untouched LoRA path. Verify every constructed module by
  comparing its implied `B @ A` update against `gamma_m * (B_original @ A_original)` in fp32.
- Record maximum per-module relative error, achieved whole-adapter Frobenius norm, dtype, adapter
  scaling metadata, tokenizer, prompt, and decoding parameters.
- All GPU work runs serially in one `srun`; at most one project GPU may be allocated.

## Statistical analysis

- Report accuracy, paired question bootstrap 95% confidence intervals, repair/break counts, and exact
  McNemar tests against untouched LoRA.
- Primary mechanism contrasts:
  - `PerModuleSpectralScale - GlobalNormMatched`;
  - `PerModuleSpectralScale - mean(three ShuffledScale variants)` using a question-cluster bootstrap;
  - `Full HNS - PerModuleSpectralScale`.
- Also report each shuffled seed separately. Do not select the best or worst shuffled seed after seeing
  validation performance.
- Calibration results are diagnostic only because these fixed variants have no searched dose. The
  512-question split remains the locked inferential set.

## Predeclared interpretation

- **Specific allocation evidence:** per-module scaling beats the global scalar and the shuffled-family
  mean on the locked split, with intervals excluding zero or a consistent replicated effect.
- **Heterogeneity without spectral assignment:** per-module and shuffled scaling beat global, but the
  true assignment does not beat shuffled assignments.
- **Only global shrink is needed:** global, per-module, and shuffled variants are statistically and
  practically indistinguishable.
- **Within-module spectral value:** Full HNS additionally beats per-module scaling.
- **Stop condition:** if per-module scaling does not beat both global and shuffled controls, do not start
  transfer studies or data-conditioned normalization. If it does, replicate on Llama Magicoder as the
  strong positive-scalar regime and Qwen Tulu as the known failure regime before claiming a method.

This pilot does not test cross-checkpoint transfer, forgetting mitigation, or a tuned global-scalar
baseline. Those become justified only after the module-allocation comparison succeeds.
