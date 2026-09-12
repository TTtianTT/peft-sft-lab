# HNS signed spectral-gate follow-up: locked PIQA protocol

Protocol frozen: 2026-09-10, before finite-intervention outcomes were inspected.

## Question

Functional energy measures how much a spectral direction is used. A signed task
sensitivity measures whether its current amplitude should locally be retained.
Can the signed signal predict the real benefit of a finite HNS-path suppression
after matching use intensity and edit magnitude?

For direction `j=(module,direction)`:

\[
E_j=(\mathrm{scale}\,\sigma_j)^2\,\mathbb E[(v_j^\top h)^2],
\qquad
N_j=\left.\frac{\partial J}{\partial s_j}\right|_{s=1}.
\]

For PIQA, the screening objective is the correct-letter minus incorrect-letter
next-token logit margin. The first-order prediction along the observed HNS path
is

\[
\widehat{\Delta J}_j
=
\frac{\partial J}{\partial\sigma_j}
(\sigma_j^{HNS}-\sigma_j).
\]

This margin signal is not assumed to equal greedy accuracy utility; that claim
is the object of the finite-intervention test.

## Fixed data split

- Task: PIQA only; it was chosen in advance, not because it had the largest HNS drop.
- Source: the same official PIQA validation split and prompt/parser protocol as the existing eight-task evaluation.
- Seed: 20260910.
- Signal: 128 questions.
- Dose selection: 128 disjoint questions.
- Locked validation: 512 disjoint questions.
- IDs must be unique across splits.
- Exact normalized PIQA question text must not match a Commonsense170K train instruction.

## Screening and matching

- Compute the signed margin gradient and LoRA-trajectory functional energy on the signal split.
- Candidate directions are limited to singular directions 1--8 that the observed all-module HNS actually suppresses.
- `predicted_suppress`: the 95% batch-bootstrap CI of the predicted HNS-path margin gain is entirely above zero.
- `predicted_retain`: the CI is entirely below zero.
- If fewer than 12 directions exist in either class, stop rather than force a median split.
- Select 12 matched pairs, exact on module type and layer quartile, nearest on LoRA functional energy, HNS edit energy, and suppression fraction.
- The pair membership and the six/six A/B assignment are frozen before any finite-intervention outcome is read.

## Finite interventions

For every selected direction, interpolate only its singular value toward its
observed HNS value at strengths 0.1 and 1.0. All other singular values and all
other modules remain at LoRA. Preserve the original singular vectors.

Both doses are evaluated on the dose split. The single block dose used in the
secondary set-level experiment is chosen by a fixed rule: maximize the mean
dose-set accuracy gain over the 12 preselected `predicted_suppress` directions;
ties choose the smaller dose. Direction membership is never tuned on the dose
split.

Both predeclared doses for all 24 directions are then evaluated on the locked
512 questions.

## Primary locked analyses

1. At each dose, compare the mean paired accuracy change of the 12
   `predicted_suppress` directions with their 12 matched `predicted_retain`
   directions. Bootstrap both direction pairs and questions.
2. Report Spearman correlation between the signed first-order prediction and
   finite accuracy change.
3. Separately report wrong-to-correct and correct-to-wrong fractions, each over
   all 512 questions, plus format-invalid changes.
4. Compare the 0.1 and 1.0 results to detect local-approximation failure.

An individual direction is called practically excessive only if its accuracy
gain CI lies entirely above `+0.5 pp`; it is called suppression-sensitive only
if its CI lies entirely below `-0.5 pp`. Everything else remains uncertain.
This threshold is intentionally stricter than one PIQA answer flip
(`1/512 = 0.195 pp`). Individual-direction classification is secondary; the
matched group contrast is the powered primary analysis.

## Secondary set-level analyses

At the chosen dose evaluate LoRA, Full HNS, suppress block A, suppress block B,
A+B, the matched predicted-retain block, and a global effective-LoRA
Frobenius-matched scalar control. Report:

\[
I_{AB}=J(A+B)-J(A)-J(B)+J(0).
\]

These tests ask whether a useful signed single-direction signal survives
composition; they do not assume additivity.

## Interpretation boundaries

- A margin improvement without answer flips supports margin sensitivity only.
- Small-dose success and full-dose failure indicates local approximation failure.
- Single-direction success with A+B failure indicates composition failure.
- The experiment tests a decision-margin gradient, not a verifier-weighted rollout policy gradient.
- SFT compatibility is retained only as a baseline and is not treated as downstream utility.
