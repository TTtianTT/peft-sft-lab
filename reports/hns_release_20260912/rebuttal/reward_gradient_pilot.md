# Qwen MetaMath reward-gradient path pilot

Date: 2026-09-10. This pilot followed the locked decision in
`hns_signed_gate_review_20260910.md`: first validate the measurement, then ask
whether a task-reward gradient is identifiable for three predeclared global
spectral paths. No module or singular-direction selection was performed.

## Protocol and completion

- Model: Qwen3-8B + MetaMathQA-50K LoRA.
- Data: 128 fixed GSM8K test questions, sampled with seed 20260910 and split in
  advance into halves A/B of 64 questions. Exact normalized overlap with the
  available MetaMathQA `query` and `original_question` fields was zero.
- Sampling: four LoRA rollouts per question (512 total), temperature 0.7,
  top-p 1, no top-k/min-p/repetition adjustment, maximum 2048 new tokens.
- Reward: existing strict numeric-answer extractor/verifier.
- Estimator: same-question leave-one-out binary-reward baseline and the sum of
  log probabilities over the complete sampled response. Contributions and
  uncertainty are aggregated by question.
- Paths: Full HNS, HeadOnly, and per-module HNS-Frobenius-matched
  ScalarShrink. For each path,
  `sigma(lambda) = sigma_LoRA + lambda * (sigma_target - sigma_LoRA)`.
- Slurm 762 generated the rollouts. Slurm 763 completed the fp32 numerical
  audit and gradient measurement in 00:02:24. Both used one B300 serially.

All 512 rollouts were valid. Mean reward was 0.7910. There were 26 mixed,
87 all-correct, and 15 all-wrong questions. No response was empty or
truncated; response length had median 120, p95 214.45, and maximum 385 tokens.

## Measurement audit

The explicit gate leaves the original LoRA forward path untouched at
`lambda=0`. The zero-gate maximum logit and sequence-log-probability errors
were both exactly zero in the eight-sequence smoke test.

An initial bf16 audit was rejected before computing the full gradients:
finite differences were unstable under small layerwise spectral changes, with
minimum relative errors of 0.66--0.67 for HeadOnly and ScalarShrink. Repeating
teacher forcing in fp32 passed for all paths:

| Path | Best autograd/finite-difference relative error | Sign |
|---|---:|---|
| Full HNS | 0.000309 | match |
| HeadOnly | 0.000064 | match |
| ScalarShrink | 0.000872 | match |

The rollouts themselves were sampled with the bf16 vLLM policy. **Review
correction, 2026-09-10:** the original cross-engine likelihood comparison mixed
probability conventions. The generation script did not set `logprobs_mode`;
the installed vLLM V1 v0.25.1 defaults to `raw_logprobs`, computed before
temperature. Teacher forcing used `log_softmax(logits / 0.7)`. The reported
Pearson correlation 0.9899, median absolute difference 0.969 per sequence and
0.00830 per response token (p95 0.02119) therefore compare **raw versus
temperature-adjusted** probabilities in addition to different numerical
implementations. They do not measure a pure bf16/fp32 precision gap or a valid
sampling-policy likelihood ratio.

The fp32 gate gradient remains a numerically validated shadow-policy
measurement, not an established unbiased score-function estimate for the bf16
sampling kernel. The size of the actual policy discrepancy is unmeasured.
Existing rollout logprobs must not be used directly as importance-weight
denominators. The generation script now explicitly requests processed
logprobs and records their sampling convention; the replay audit excludes
legacy or incompatible scores from cross-policy comparisons. Existing run
artifacts have not been overwritten and no replacement GPU run was started.

## Reward-gradient result

Positive gradient means that an infinitesimal move from LoRA toward the named
path is predicted to increase expected reward under the stated stochastic
policy.

| Path | Mean gradient | Question-bootstrap 95% CI | Half A | Half B | Top-5 absolute share |
|---|---:|---:|---:|---:|---:|
| Full HNS | +0.0372 | [-0.0586, +0.1377] | -0.0496 | +0.1240 | 44.7% |
| HeadOnly | +0.0401 | [-0.0635, +0.1520] | -0.0531 | +0.1332 | 43.6% |
| ScalarShrink | +0.0114 | [-0.0542, +0.0773] | -0.0459 | +0.0686 | 41.3% |

All three confidence intervals cross zero and all three paths reverse sign
between the two predeclared halves. Only the 26 mixed-outcome questions have
nonzero RLOO contributions. Within those 26 questions, every path has 12
positive and 14 negative contributions; the medians are negative even though
the means are positive. The largest single question accounts for 9.5--11.0%
of absolute contribution and the largest five account for 41.3--44.7%.

The path signals are also nearly collinear: their per-question Pearson
correlations range from 0.958 to 0.992. Correlation alone does not establish
that their means are indistinguishable, because common noise may cancel in a
paired difference. A post-hoc review directly bootstrapped question-level
paired contrasts (20,000 resamples, seed 20260910):

| Local path-gradient contrast | Mean | Paired 95% CI |
|---|---:|---:|
| Full HNS minus HeadOnly | -0.002854 | [-0.018750, +0.012151] |
| Full HNS minus ScalarShrink | +0.025857 | [-0.012427, +0.068110] |
| HeadOnly minus ScalarShrink | +0.028711 | [-0.013660, +0.075536] |

All contrasts cross zero and each reverses sign between the predeclared
halves. Thus there is also no resolved differential signal on these particular
path parameterizations. These are derivatives in the path coordinate, not
finite downstream percentage-point gains or proof that the mechanisms are
equivalent.

## Supplementary reward audit

A conservative decimal-equivalence check found 12 strict-string false
negatives across three questions: `2.00` versus `2`, `14.00` versus `14`, and
`1.00` versus `1`. All four rollouts of each question had the same mismatch.
Counting these numerically equivalent answers as correct would change the
descriptive reward from 405/512 (79.1016%) to 417/512 (81.4453%). This is a
supplementary diagnostic; the established strict benchmark reward is unchanged.

For each of those three questions, the four RLOO advantages are zero both
before and after this relabeling. Therefore the 26 mixed-question count and
every gradient estimate above are unchanged. This reward-format issue does
not explain the observed signal failure.

## Decision

The implementation-level gradient check passed in fp32, but the task signal
did not pass the predeclared identifiability checks. The correct conclusion is
**insufficient reward-gradient signal at 128 questions x four rollouts**, not
that any of the three paths has established positive or negative utility.

Accordingly:

1. Do not perform the proposed finite-intervention validation from these
   estimates.
2. Do not select modules or individual singular directions.
3. Do not interpret the small positive full-sample means as explaining the
   previously observed greedy MetaMath HNS gain.
4. If this route is revisited, first align and document the likelihood
   convention. Additional rollouts cannot fix an unknown policy mismatch.
   A continuous verifier changes the objective and requires its own downstream
   validation; it is not an automatic remedy for noisy binary rewards.

Review recommendation: close this pilot under its original stopping rule.
Under the current low-cost constraint, do not expand reward-gradient sampling
or select directions from these means. Any next experiment should be a
separately specified direct finite-intervention calibration study, with actual
deployment decoding, a small predeclared dose grid, scalar controls matched to
the norm of each head-suppression dose, a zero-edit option, and an independent
validation split. Such a study would test practical dose selection rather than
claim that this failed gradient pilot passed its gate.

## Reproducible artifacts

- Run root:
  `/dataset1/zailong/runs/peft-sft-lab/hns-signed-gates-20260910/qwen_metamath_reward_paths`
- Rollout manifest and fixed trajectories: `rollouts/manifest.json`,
  `rollouts/rollouts.jsonl`.
- Numerical audit: `gradients/numerical_check.json`.
- Main results: `gradients/path_signal_summary.tsv`,
  `gradients/question_path_gradients.tsv`, and `gradients/rollout_scores.tsv`.
- Per-module path construction audit: `gradients/path_module_deltas.tsv`.
