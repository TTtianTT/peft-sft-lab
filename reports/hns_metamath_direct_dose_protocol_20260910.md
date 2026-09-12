# Qwen MetaMath direct HeadOnly-dose calibration protocol

Date locked: 2026-09-10. This is a new experiment, separate from the closed
reward-gradient pilot. No reward-gradient estimate is used to select a dose,
module, or singular direction.

## Question and intervention

The experiment asks whether a small greedy-evaluation budget can decide
whether and how strongly to suppress HNS head directions, and whether any
benefit exceeds a matched reduction of overall LoRA magnitude.

For every LoRA module, let

`sigma_head = min(sigma_LoRA, sigma_HNS)`

in the aligned original LoRA singular basis. At dose `alpha`:

`sigma_head(alpha) = sigma_LoRA + alpha * (sigma_head - sigma_LoRA)`.

Test `alpha in {0, 0.25, 0.5, 1}`. For every nonzero dose, construct a
per-module ScalarShrink control

`sigma_scalar(alpha) = gamma_m(alpha) * sigma_LoRA`,

where `gamma_m(alpha)` exactly matches the Frobenius norm of that module's
HeadOnly-dose spectrum. All variants preserve the original LoRA U/V basis.
Full HNS is retained as a separate complete-edit reference.

The zero option used for selection is the original LoRA adapter. A separately
rebuilt alpha=0 adapter is evaluated on calibration data only as a numerical
reconstruction/output audit; it is not a second statistical candidate.

## Data lock

- Source: the same local GSM8K test snapshot and prompt/parser used by the
  existing Qwen MetaMath evaluation.
- Exclude all 128 questions used in the reward-gradient pilot.
- From the remaining questions, seed 20260911 fixes 256 calibration questions
  and 512 locked-validation questions. The two sets are disjoint.
- Exact normalized question overlap with the available MetaMathQA training
  parquet must be zero.
- These questions are independent of this dose selection procedure, but the
  GSM8K benchmark has been evaluated previously; they are not described as a
  never-observed benchmark.

## Evaluation and selection

- Actual deployment endpoint: greedy decoding, non-thinking chat template,
  maximum 2048 new tokens, and the existing strict GSM8K answer parser.
- Numeric-decimal equivalence is reported as a secondary audit; it does not
  replace the locked strict primary endpoint.
- Calibration evaluates LoRA, alpha=0 rebuild, all six nonzero HeadOnly and
  matched-scalar variants, and Full HNS.
- Select the HeadOnly alpha with highest strict calibration accuracy among
  `{0, 0.25, 0.5, 1}`. Exact ties select the smaller alpha. Zero is always a
  valid selection. Scalar performance does not select the HeadOnly dose.
- Validation evaluates only LoRA, Full HNS, the selected HeadOnly dose, and
  its corresponding matched scalar control. If alpha=0 is selected, both
  selected labels point to the untouched LoRA baseline and are reported as
  such.

Report paired question bootstrap confidence intervals, exact McNemar tests,
wrong-to-correct and correct-to-wrong counts, strict and numeric-equivalent
scores, HeadOnly-minus-matched-scalar contrasts, and the zero-rebuild audit.
Calibration estimates are selection data; only the locked validation split is
used for the final mechanism claim.

## Stop rule

This experiment addresses whether/how strongly only. It does not proceed to
module blocks or singular-direction selection regardless of the result. A
nonzero selected dose supports further work only if its locked HeadOnly gain
is positive and the locked HeadOnly-minus-scalar comparison is directionally
positive; confidence intervals determine whether the language is conclusive
or suggestive.
