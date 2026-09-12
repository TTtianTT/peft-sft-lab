# Qwen MetaMath post-hoc spectral-scaling pilot

Slurm job 768 completed successfully on one B300 in 00:06:08. Seven fixed variants were
evaluated serially on 256 diagnostic calibration questions and 512 locked validation questions.
The primary endpoint is strict greedy GSM8K accuracy; numeric equivalence is secondary.

## Locked validation scores

| Variant | Strict | Numeric |
|---|---:|---:|
| Untouched LoRA | 83.20% | 84.38% |
| GlobalNormMatched | 84.18% | 86.13% |
| PerModuleSpectralScale | 83.20% | 85.35% |
| ShuffledScale-1 | 84.18% | 86.33% |
| ShuffledScale-2 | 84.18% | 86.13% |
| ShuffledScale-3 | 83.79% | 85.94% |
| Full HNS | 86.91% | 89.84% |

## Predeclared mechanism contrasts

| Contrast | Strict delta [95% CI] | p | Numeric delta [95% CI] | p |
|---|---:|---:|---:|---:|
| PerModule - Global | -0.98 pp [-2.54 pp, +0.39 pp] | 0.302 | -0.78 pp [-2.15 pp, +0.59 pp] | 0.424 |
| PerModule - shuffle mean | -0.85 pp [-2.28 pp, +0.52 pp] | 0.275 | -0.78 pp [-2.15 pp, +0.59 pp] | 0.307 |
| Full HNS - PerModule | +3.71 pp [+1.37 pp, +6.05 pp] | 0.0034 | +4.49 pp [+2.34 pp, +6.84 pp] | 0.0001168 |

The true spectrum-to-module assignment is not favored: PerModule is 0.98 pp below the global
norm-matched scalar and 0.85 pp below the mean shuffled allocation on the primary endpoint.
Neither difference is resolved, but both point in the direction opposite to the proposed method.
Full HNS exceeds PerModule by 3.71 pp with a paired interval excluding zero. Thus matching HNS's
per-module Frobenius allocation is insufficient; the remaining within-module spectral change or an
associated implementation effect matters for this checkpoint.

## Numerical repeatability boundary

The identical untouched LoRA changes by -0.39 pp
between this run and the prior direct-dose run; Full HNS changes by
+0.78 pp. Exact extracted predictions match on
97.9% (LoRA)
and 97.5%
(HNS) of questions. Greedy low-precision inference therefore has non-negligible cross-run branching.
The small Global/PerModule/Shuffled gaps must be treated as noise-level. The larger HNS-PerModule gap
is reproduced directionally by the earlier complete evaluation, but its exact significance is
run-dependent and should not be overinterpreted as a clean decomposition.

## Decision

The predeclared success condition is not met. Stop the per-module normalization branch: do not launch
Llama Magicoder/Qwen Tulu replication, transfer, or data-conditioned normalization on the basis of this
rule. The useful surviving result is narrower: global shrink has a small positive point estimate, while
Full HNS retains a substantially larger gain, so HNS cannot be reduced to its induced module-norm
allocation in this MetaMath pilot.
