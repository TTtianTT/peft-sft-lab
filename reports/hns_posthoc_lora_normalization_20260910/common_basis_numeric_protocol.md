# Common-basis HNS numerical confirmation protocol

Date: 2026-09-10. This is a same-question numerical recheck, not an independent downstream
confirmation: all GSM8K test questions were used by earlier project evaluations.

## Gate A: numerical diagnostic

- Fixed 32-question set: the 22 questions whose LoRA or HNS extracted answer changed across the two
  previous runs, plus 10 stable questions sampled once with seed 20260912.
- Three fresh vLLM processes: two with identical adapter order and one with reversed order.
- Explicit settings: `VLLM_BATCH_INVARIANT=1`, synchronous scheduling, prefix caching disabled,
  custom all-reduce disabled, fixed prompt order, `max_num_seqs=64`, greedy decoding and seed 42.
- Save and compare complete output token IDs, text, extracted answer and correctness.
- Pass only if every identical adapter has 100% token-ID agreement across all three processes and
  common-basis saved-update relative error is at most 5e-4. Otherwise stop before formal evaluation.

Diagnostic adapters are untouched LoRA, zero-rebuild, common-basis PerModule, common-basis HNS,
original-factor PerModule and the existing HNS adapter. Representation comparisons are diagnostic and
do not determine the repeatability gate.

## Common numerical representation

For each module, compute one fp32 SVD of the original LoRA update and use the same factors for every
common-path variant:

\[
A_m=V_m^T,\qquad B_m=U_m\operatorname{diag}(d_m).
\]

Only `d_m` changes. Zero-rebuild uses the original spectrum, PerModule uses `gamma_m sigma_m`, HNS uses
the aligned existing HNS spectrum, and global scalar candidates use `gamma sigma_m`. PEFT scaling is
unchanged. Store factors in the source adapter dtype and audit their implied fp32 update.

## Gate B: global-scalar selection and same-question recheck

If Gate A passes, select a global scalar on the existing 256-question calibration split from

\[
\gamma\in\{1.00,0.85,0.70,0.60,0.50,0.40\}.
\]

Maximize strict greedy accuracy; an exact tie selects the value closest to 1. Lock the result, then
evaluate on the existing 512-question split:

- common-basis HNS vs common-basis PerModule;
- common-basis HNS vs selected common-basis global scalar.

Untouched LoRA, zero-rebuild and existing HNS are retained as representation diagnostics. Strict answer
equality is primary; decimal numeric equivalence is secondary. Report paired question bootstrap intervals,
repair/break counts and exact McNemar tests. Repeated outputs are never treated as additional samples.

No direction selection, cross-checkpoint transfer or additional GPU job follows automatically from this run.
