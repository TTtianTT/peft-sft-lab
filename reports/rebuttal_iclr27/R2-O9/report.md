# R2-O9: completed results

Status: complete. All prespecified conditions are reported.

| Condition | Method | Target | Off | FG |
| --- | --- | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct_magicoder_val_best | original_lora | 54.2683 | 63.2499 | 3.4504 |
| Llama-3.1-8B-Instruct_magicoder_val_best | recon_0 | 54.8780 | 63.3927 | 3.5229 |
| Llama-3.1-8B-Instruct_magicoder_val_best | hns_f4_s1 | 53.0488 | 67.1092 | 0.3081 |
| Llama-3.1-8B-Instruct_magicoder_val_best | flat_nuclear | 52.4390 | 66.8462 | 0.6778 |
| Llama-3.1-8B-Instruct_metamath_val_best | original_lora | 74.6778 | 57.3996 | 5.7992 |
| Llama-3.1-8B-Instruct_metamath_val_best | recon_0 | 74.6020 | 57.9372 | 5.4649 |
| Llama-3.1-8B-Instruct_metamath_val_best | hns_f4_s1 | 79.6816 | 62.1018 | 1.5035 |
| Llama-3.1-8B-Instruct_metamath_val_best | flat_nuclear | 79.7574 | 62.4730 | 1.5389 |

Target/Off are percentages; FG/differences are pp. Off averages3 non-target families equally; FG clips each family before averaging. Commonsense is an equal8-subtask macro.

All20,000-draw paired intervals and wins/ties/losses or exact McNemar repair/damage counts are in paired_ci.tsv. Single-seed, paired question-level intervals; no training-run inference.

Source/config/weight SHA256, module coverage, reconstruction errors and ordered per-sample input hashes passed. All CUDA/model work ran through Slurm; at most2 allocated GPUs. Builders exited before vLLM began. Commands, timings, job IDs and logs are in commands_*.json. See protocol.md for fixed settings.

Single seed42; paired question-level intervals, not uncertainty over training runs. Explicit warmup verified against actual schedule; all candidate/training artifacts hashed.

All12 candidate validation scores are in validation_candidates.tsv; exactly one checkpoint/task was selected by the predeclared validation-only rule.
- magicoder: LR1e-05, fraction1, validation0.6350.
- metamath: LR5e-05, fraction1, validation0.8320.
