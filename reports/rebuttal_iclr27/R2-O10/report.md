# R2-O10: completed results

Status: complete. All prespecified conditions are reported.

| Condition | Method | Target | Off | FG |
| --- | --- | ---: | ---: | ---: |
| Qwen3-14B_metamath_lora_r16_seed42 | original_lora | 86.1259 | 71.7319 | 8.9756 |
| Qwen3-14B_metamath_lora_r16_seed42 | recon_0 | 86.2017 | 71.8419 | 8.8656 |
| Qwen3-14B_metamath_lora_r16_seed42 | hns_f4_s1 | 85.8226 | 76.7464 | 4.1459 |
| Qwen3-14B_metamath_lora_r16_seed42 | flat_nuclear | 86.5807 | 76.2913 | 4.5394 |

Target/Off are percentages; FG/differences are pp. Off averages3 non-target families equally; FG clips each family before averaging. Commonsense is an equal8-subtask macro.

All20,000-draw paired intervals and wins/ties/losses or exact McNemar repair/damage counts are in paired_ci.tsv. Single-seed, paired question-level intervals; no training-run inference.

Source/config/weight SHA256, module coverage, reconstruction errors and ordered per-sample input hashes passed. All CUDA/model work ran through Slurm; at most2 allocated GPUs. Builders exited before vLLM began. Commands, timings, job IDs and logs are in commands_*.json. See protocol.md for fixed settings.

Single seed42; paired question-level intervals, not uncertainty over training runs. Explicit warmup verified against actual schedule; all candidate/training artifacts hashed.

Qwen3-14B LR2e-4 was transferred from8B without model-specific tuning. A one-B300 vLLM fit check passed before the single fixed-recipe training. Capacity details are in capacity_preflight.json; total allocated GPUs remained2.

Protocol amendment (author-approved before any 14B training/inference): the first fit check with max_num_batched_tokens65536 failed with an Inductor int32 index overflow (cudaErrorIllegalAddress in the fused SiLU*mul kernel; 65536x2x17408>2^31), not GPU memory. O10 Base and all four adapters therefore use max_num_batched_tokens32768; every other inference setting is unchanged. This is a documented deviation from the main 65536 setting, so O10 is compared within itself, not as a literal same-config row of the main table.
