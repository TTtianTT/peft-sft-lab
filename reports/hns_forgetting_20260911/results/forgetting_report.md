# Does HNS reduce cross-task forgetting?

## Verdict

Yes, relative to the saved unedited LoRA adapters, HNS usually reduces measured cross-task forgetting in this
2-base x 4-task experiment. It is not, however, consistently better than a target-feasible calibrated global
scalar, so the current evidence supports an adapter-strength/regularization effect more strongly than a
universal shape-specific advantage.

## Completion and numerical audit

- Formal Qwen matrix: 164/164 task-condition cells.
- Formal Llama matrix: 164/164 task-condition cells.
- Reverse-order Llama core repeat: 84/84 cells.
- Qwen passed the exact-token numerical gate. Llama did not, especially for long generations, but all 84
  aggregate benchmark scores in the full-size core repeat exactly matched the first formal run.
- Confidence intervals below use 20,000 paired bootstrap draws, stratified by subtask for the commonsense macro.
  Holm correction is within each prespecified contrast family across the eight checkpoints.

## Saved HNS versus saved LoRA

All changes are percentage points. `Off-task` is the equal-weight mean of the other three benchmark families.

| Base | Training task | Target change | Off-task change | 95% paired CI | Holm p |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | Magicoder | +8.54 | +3.20 | [+1.98, +4.42] | <.0001 |
| Qwen3-8B | MetaMath | +4.02 | +3.44 | [+1.10, +5.76] | .0076 |
| Qwen3-8B | Tulu | +3.51 | -1.13 | [-3.15, +0.88] | .2762 |
| Qwen3-8B | Commonsense | -0.41 | +3.15 | [+1.17, +5.09] | .0076 |
| Llama-3.1-8B | Magicoder | +0.61 | +3.83 | [+2.45, +5.23] | <.0001 |
| Llama-3.1-8B | MetaMath | +3.34 | +5.00 | [+2.91, +7.14] | <.0001 |
| Llama-3.1-8B | Tulu | +1.11 | +2.56 | [+0.94, +4.20] | .0076 |
| Llama-3.1-8B | Commonsense | -0.37 | +14.55 | [+11.91, +17.32] | <.0001 |

HNS improves the off-task macro in 7/8 checkpoints, with a checkpoint-average gain of +4.33 pp. Among the 17
individual off-task evaluations where LoRA is below the pretrained base, HNS improves 15. The equal-weight
clipped forgetting gap falls from 12.63 pp to 8.91 pp, a 29.5% reduction. Qwen-Tulu is the checkpoint-level
counterexample. At the individual benchmark level, Qwen Commonsense -> HumanEval and Llama Tulu ->
Commonsense also worsen.

The large Commonsense-adapter recovery should not be described as complete preservation: those LoRAs cause
severe off-task damage, and HNS often recovers only part of the lost base performance.

## Representation and scale controls

Rebuilding LoRA and HNS in a shared singular-vector factorization gives the same high-level result: HNS beats
common-basis LoRA in 7/8 checkpoints, with mean off-task gain +4.58 pp. Thus the main HNS-versus-LoRA result is
not explained by the original adapters having different A/B factor representations.

Against the per-module HNS-norm-matched scalar, HNS has a positive point estimate in 7/8 checkpoints, but only
three comparisons survive Holm correction (Llama Magicoder, MetaMath, and Commonsense). The Qwen Commonsense
adapter strongly favors the per-module scalar: HNS is 17.36 pp worse on off-task macro. Therefore a universal
within-module spectral-shape advantage is not established.

## Target-feasible calibrated global scalar

For each checkpoint, the scalar is selected from gamma in {0.25, 0.40, 0.55, 0.70, 0.85, 1.00}, requiring
target score no more than 1 pp below HNS, and then maximizing off-task performance.

| Base | Training task | Selected gamma | HNS - scalar target | HNS - scalar off-task |
|---|---|---:|---:|---:|
| Qwen3-8B | Magicoder | 0.40 | +0.00 | +0.36 |
| Qwen3-8B | MetaMath | 0.40 | +0.68 | +1.37 |
| Qwen3-8B | Tulu | 0.85 | +0.92 | -1.25 |
| Qwen3-8B | Commonsense | 0.40 | +0.16 | -25.10 |
| Llama-3.1-8B | Magicoder | 0.25 | -1.83 | -0.39 |
| Llama-3.1-8B | MetaMath | 0.40 | -0.15 | +0.44 |
| Llama-3.1-8B | Tulu | 0.70 | -0.18 | -0.25 |
| Llama-3.1-8B | Commonsense | 0.55 | -0.12 | -9.86 |

HNS has better point-estimate retention in only 3/8 comparisons. The scalar is selected and reported on the
same data, so this table is an exploratory upper bound rather than an independently validated scalar method.
It nevertheless shows that much of the apparent forgetting reduction can be reproduced, and sometimes greatly
exceeded, by reducing overall adapter strength.

## Interpretation

The defensible claim is:

> HNS is an effective post-hoc regularizer for many over-strong LoRA adapters and usually improves their
> cross-task retention. The effect survives common-basis reconstruction, but is not consistently specific to
> spectral reshaping once target-feasible scalar controls are considered.

The strongest shape-specific evidence is currently localized to several Llama checkpoints, not universal.
The principal failure condition is a task-compatible dominant update: suppressing it can discard positive
transfer or preserve less capability than a simple scalar. Qwen-Tulu and Qwen-CommonSense provide the clearest
boundaries.
