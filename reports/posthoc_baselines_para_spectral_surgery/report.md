# PARA and Spectral Surgery post-hoc baselines

Status: complete.

## Protocol and ordering audit

The fixed cohort is Qwen3-8B and Llama-3.1-8B-Instruct × Magicoder/MetaMath/Tulu × seeds 42/43/44 (18 existing LoRA checkpoints; no retraining). PARA completed target evaluation, off-task evaluation, scoring, aggregation, and its stage report before the dependent Spectral Surgery job was allowed to start. Each method likewise completed all target cells before off-task generation.

The generation/scoring protocol is the current unified functional_hns_three_seed_20260914 protocol: greedy seed 42; max model length 4096; long-task max_num_seqs 2048; adapter block 5; max LoRA rank 16; exact existing prompt, parser and scoring code. Base, LoRA and HNS 4+1 values are hash-checked current-wave caches, not old-paper results.

## Method audit

PARA follows [epsilon-PARA](https://arxiv.org/abs/2604.27796): compact QR/SVD per LoRA BA update, one threshold over all modules in each checkpoint, squared-singular-value energy budgets epsilon={0.90,0.95,0.99}, untouched retained singular values, and no nuclear/Frobenius restoration. All source scalings are 2, and reconstructed rank/alpha patterns preserve that scaling. Zero-rank modules are excluded in the PEFT config. The paper states that official code will be released upon acceptance; no official implementation was available to copy.

[Spectral Surgery](https://arxiv.org/abs/2603.03995) follows the repository's original implementation. Its fixed configuration was chosen before evaluation from the paper's principal guided-vs-random result and the repository's matching publication configuration (`grad_direction_residual_l1_calib128`), not from current test scores: grad_direction; calibration size 128; training split only; deterministic shuffle seed 42; answer-only teacher-forced loss; mean-absolute normalization; asymmetric multiplicative step sizes 2.0/0.2; L1/nuclear preservation; and edits only o_proj/down_proj in every layer. This residual-writing scope differs from HNS 4+1, which edits all seven LoRA module families.

## Overall results

| Method | n | Target % | Off % | FG pp |
| --- | ---: | ---: | ---: | ---: |
| base | 18 | 66.4730 | 69.9931 | 0.0000 |
| original_lora | 18 | 67.9980 | 69.5512 | 2.7082 |
| hns_f4_s1 | 18 | 72.0834 | 72.7915 | 0.3441 |
| para_e90 | 18 | 68.7778 | 70.0126 | 2.3706 |
| para_e95 | 18 | 68.5823 | 69.6357 | 2.6954 |
| para_e99 | 18 | 68.4421 | 69.7776 | 2.5637 |
| spectral_surgery_grad_direction | 18 | 68.7637 | 68.2820 | 3.7225 |

All fixed PARA epsilon settings are shown; no test-set selection was performed. Paired source-checkpoint bootstrap tables and complete per-checkpoint/model-task tables are in the method subdirectories.

## Result interpretation

Relative to original LoRA, PARA e90 changes Target by +0.7798 pp, Off by +0.4614 pp, and FG by -0.3375 pp. The corresponding paired source-checkpoint bootstrap intervals exclude zero, but they remain descriptive intervals on a fixed shared test set. PARA e95/e99 also improve mean Target (+0.5844/+0.4441 pp); their Off and FG intervals overlap zero. These are comparisons of all predeclared epsilon settings, not a post-hoc choice of e90.

Spectral Surgery changes Target by +0.7657 pp versus original LoRA, but Off by -1.2692 pp and FG by +1.0144 pp; all three paired intervals exclude zero in this fixed cohort. HNS 4+1 remains higher in mean Target and Off and lower in FG than every evaluated PARA setting and the fixed Spectral Surgery configuration. This supports a tradeoff conclusion for these checkpoints and settings, not a universal ranking of the methods.

## Conclusion boundaries

The 18 runs share benchmark items, and seed42 has previously documented recipe/provenance differences from seeds43/44; source-checkpoint intervals describe paired run heterogeneity on a fixed test set and are not strict IID training-run confidence intervals. FG is the existing clipped metric, so Target and Off must be read alongside it. Spectral Surgery calibration gradients optimize answer-token language-model loss and can conflict with strict instruction-following behavior; results should not be generalized beyond the fixed configuration and cohort.

## Reproducibility

Successful scheduler jobs: PARA 1077; Spectral Surgery 1078, submitted with dependency afterok:1077. The initial numerical preflight 1075 stopped before evaluation because one fp32 reconstruction error (1.202e-5) narrowly exceeded the original 1e-5 gate; after confirming roundoff, the gate was set to 5e-5 while the actual maximum remained recorded. Its dependent job 1076 never started. See `scheduler_jobs.tsv`, `audit.json`, `lora_scaling_audit.tsv`, `commands_*.json`, both stage reports, build summaries, `module_ranks.tsv`, `checkpoint_results.tsv`, `grouped_results.tsv` and `paired_ci.tsv`. Timestamps are recorded in completion markers and command histories; large adapters, predictions and logs remain under this report directory's ignored runtime paths.
