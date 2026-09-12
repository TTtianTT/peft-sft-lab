# Post-hoc LoRA normalization: CPU evidence audit

This audit reinterprets the existing Full-HNS Frobenius-matched `ScalarShrink` as a candidate
post-hoc method. It does not add downstream evaluations and does not claim that a global scalar
or shuffled module allocation has already been tested.

## What the existing scalar actually is

For every module, the 2x4 control keeps the original LoRA singular-value shape and applies
`gamma_m = ||Delta W_HNS,m||_F / ||Delta W_LoRA,m||_F`. Thus it is a checkpoint-derived
per-module allocation rule, not one global LoRA scaling coefficient. The newer MetaMath
direct-dose control is separate: it matches each HeadOnly dose rather than Full HNS.

## Checkpoint-level scaling and observed performance

| Base | Task | global norm-matched gamma | module gamma p05 / median / p95 | scalar gain (pp) | HNS gain (pp) |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | commonsense | 0.765 | 0.494 / 0.752 / 0.879 | +0.39 | -0.07 |
| Llama-3.1-8B-Instruct | magicoder | 0.642 | 0.520 / 0.777 / 0.911 | +4.22 | +4.02 |
| Llama-3.1-8B-Instruct | metamath | 0.873 | 0.767 / 0.873 / 0.945 | +0.30 | +3.26 |
| Llama-3.1-8B-Instruct | tulu | 0.860 | 0.743 / 0.909 / 0.964 | +1.80 | +2.07 |
| Qwen3-8B | commonsense | 0.605 | 0.463 / 0.640 / 0.808 | +0.12 | -0.44 |
| Qwen3-8B | magicoder | 0.522 | 0.429 / 0.611 / 0.798 | +0.42 | +3.72 |
| Qwen3-8B | metamath | 0.609 | 0.450 / 0.673 / 0.840 | +0.99 | +4.40 |
| Qwen3-8B | tulu | 0.757 | 0.590 / 0.752 / 0.871 | -5.36 | +1.81 |

The per-module scalar point estimate is positive on 7/8 checkpoints. Across only eight
checkpoints, its gain has Spearman rho=0.667 with Full-HNS gain;
this is descriptive and is not a causal decomposition.

## Is the allocation actually spectrum-derived?

Across 1904 modules, gamma has Spearman rho=-0.953 with
top-1 spectral energy, rho=0.995 with effective rank, and
rho=-1.000 with spectral HHI. These strong relationships are largely
structural: nuclear-norm-preserving flattening necessarily couples concentration to Frobenius shrinkage.
They show that the rule is spectrum-conditioned; they do not show that the module-to-gamma assignment
is task-useful.

The module gamma distribution is broad within every checkpoint (see `checkpoint_summary.tsv` and
`module_type_summary.tsv`). Therefore a single global coefficient is not algebraically equivalent to
the existing scalar control. Whether this heterogeneity improves behavior remains untested.

For the newer Qwen MetaMath HeadOnly-matched control, the full-dose module gamma p05/median/p95 is
0.250/0.478/0.669, with adapter-global gamma 0.424. This is materially stronger shrinkage than the
Full-HNS-matched rule because HeadOnly lowers dominant values without the compensating HNS tail lift.
The two observed scalar results must therefore remain separate pieces of evidence.

## Evidence boundary and next decision

Existing results justify treating per-module HNS-matched scaling as a candidate method, but not yet as
a contribution. The decisive next comparison is untouched LoRA vs one global norm-matched scalar vs
the HNS-derived per-module scalar vs within-module-type shuffled allocations vs Full HNS. All scalar
variants must match the same whole-adapter Frobenius norm, and scalar adapters should be constructed by
directly scaling the original LoRA factors so that the zero-edit path is bit-identical to the original.

Three fixed shuffled allocation plans are materialized in `shuffle_scaling_plan.csv`; each is permuted
within module type and then corrected by one checkpoint-wide factor to match the Full-HNS total norm.
Do not proceed to transfer or data-conditioned normalization unless the per-module rule beats both the
global and shuffled controls on an independent split.
