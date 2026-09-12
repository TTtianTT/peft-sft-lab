# Qwen MetaMath direct HeadOnly-dose result

This experiment was specified separately from the closed reward-gradient pilot. Slurm 766 completed
successfully with one B300. The locked primary endpoint is strict greedy GSM8K accuracy; decimal-numeric
equivalence is secondary.

## Selection and locked result

Calibration selected the full HeadOnly dose (alpha=1): strict accuracies for alpha 0/0.25/0.5/1 were
86.72%, 86.33%, 86.72%, and 88.28%. Alpha=1 remains selected in all 256 leave-one-question-out
datasets, but only 62.0% of question bootstraps; dose selection is
directionally stable to single examples but still sampling-uncertain.

| Locked contrast | Strict delta [95% CI] | Numeric delta [95% CI] |
|---|---:|---:|
| HeadOnly(alpha=1) - LoRA | +2.34 pp [-0.78 pp, +5.47 pp] | +4.30 pp [+1.37 pp, +7.23 pp] |
| Matched Scalar - LoRA | +1.76 pp [-1.17 pp, +4.88 pp] | +3.12 pp [+0.39 pp, +5.86 pp] |
| HeadOnly - Matched Scalar | +0.59 pp [-1.37 pp, +2.54 pp] | +1.17 pp [-0.59 pp, +2.93 pp] |
| Full HNS - HeadOnly | +0.20 pp [-1.56 pp, +1.95 pp] | -0.20 pp [-1.76 pp, +1.56 pp] |

The full-dose HeadOnly point estimate is positive on the independent validation split, and Full HNS is
essentially tied with it. However, the matched scalar captures most of the point-estimate gain (75% strict,
73% numeric), while the direct HeadOnly-minus-scalar intervals cross zero. This supports suppression/scale
as a useful finite-dose regime but does not resolve a spectral-shape benefit beyond overall magnitude reduction.

## Reconstruction warning

The alpha=0 rebuilt adapter has maximum relative module reconstruction error 8.32e-06,
yet changes six calibration decisions and has strict delta -1.56 pp
[-3.52 pp, +0.00 pp] versus
the untouched LoRA. Autoregressive outputs are therefore sensitive to tiny factor-reconstruction changes.
Comparisons against untouched LoRA carry this implementation perturbation; HeadOnly versus its matched scalar
is cleaner because both adapters use the same SVD reconstruction path.

## Mechanism decision

The experiment improves whether/how-strongly calibration: the predeclared procedure chose alpha=1 and its
locked point estimate was positive. It does not establish HeadOnly > ScalarShrink, so it is premature to infer
that dominant-mode shape suppression rather than LoRA magnitude reduction causes the gain. Per protocol, stop
here and do not proceed to module/direction localization from this result alone.
