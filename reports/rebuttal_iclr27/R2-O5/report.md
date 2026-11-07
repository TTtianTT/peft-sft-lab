# R2-O5: completed results

Status: complete. All prespecified conditions are reported.

| Condition | Method | Target | Off | FG |
| --- | --- | ---: | ---: | ---: |
| Qwen3-8B_magicoder_lora_r16_seed43 | original_lora | 64.6341 | 77.5338 | 2.1404 |
| Qwen3-8B_magicoder_lora_r16_seed43 | recon_0 | 65.2439 | 78.1363 | 1.5543 |
| Qwen3-8B_magicoder_lora_r16_seed43 | hns_f4_s1 | 73.7805 | 80.5536 | 0.0000 |
| Qwen3-8B_magicoder_lora_r16_seed43 | flat_nuclear | 75.0000 | 80.9453 | 0.0000 |
| Qwen3-8B_metamath_lora_r16_seed43 | original_lora | 85.3677 | 71.1364 | 1.8050 |
| Qwen3-8B_metamath_lora_r16_seed43 | recon_0 | 85.0644 | 70.6558 | 2.2856 |
| Qwen3-8B_metamath_lora_r16_seed43 | hns_f4_s1 | 86.9598 | 73.6840 | 0.0000 |
| Qwen3-8B_metamath_lora_r16_seed43 | flat_nuclear | 87.7180 | 74.6609 | 0.0000 |
| Qwen3-8B_magicoder_lora_r16_seed44 | original_lora | 64.6341 | 77.6642 | 1.9998 |
| Qwen3-8B_magicoder_lora_r16_seed44 | recon_0 | 65.8537 | 77.5848 | 2.0978 |
| Qwen3-8B_magicoder_lora_r16_seed44 | hns_f4_s1 | 73.1707 | 80.5598 | 0.0000 |
| Qwen3-8B_magicoder_lora_r16_seed44 | flat_nuclear | 73.7805 | 80.9063 | 0.0000 |
| Qwen3-8B_metamath_lora_r16_seed44 | original_lora | 84.4579 | 67.2818 | 5.8143 |
| Qwen3-8B_metamath_lora_r16_seed44 | recon_0 | 84.5337 | 67.8167 | 5.2661 |
| Qwen3-8B_metamath_lora_r16_seed44 | hns_f4_s1 | 87.9454 | 73.8699 | 0.6098 |
| Qwen3-8B_metamath_lora_r16_seed44 | flat_nuclear | 87.6422 | 73.1060 | 0.6098 |
| Qwen3-8B_magicoder_lora_r16_seed42 | original_lora | 64.0244 | 77.5627 | 2.1262 |
| Qwen3-8B_magicoder_lora_r16_seed42 | recon_0 | 64.0244 | 77.9787 | 1.7281 |
| Qwen3-8B_magicoder_lora_r16_seed42 | hns_f4_s1 | 73.1707 | 80.5574 | 0.0000 |
| Qwen3-8B_magicoder_lora_r16_seed42 | flat_nuclear | 73.1707 | 80.4974 | 0.0000 |
| Qwen3-8B_metamath_lora_r16_seed42 | original_lora | 84.7612 | 67.0764 | 5.9255 |
| Qwen3-8B_metamath_lora_r16_seed42 | recon_0 | 84.5337 | 68.0421 | 4.9645 |
| Qwen3-8B_metamath_lora_r16_seed42 | hns_f4_s1 | 89.1585 | 74.2772 | 0.0000 |
| Qwen3-8B_metamath_lora_r16_seed42 | flat_nuclear | 88.3245 | 73.8269 | 0.0000 |

Target/Off are percentages; FG/differences are pp. Off averages3 non-target families equally; FG clips each family before averaging. Commonsense is an equal8-subtask macro.

All20,000-draw paired intervals and wins/ties/losses or exact McNemar repair/damage counts are in paired_ci.tsv. Single-seed, paired question-level intervals; no training-run inference.

Source/config/weight SHA256, module coverage, reconstruction errors and ordered per-sample input hashes passed. All CUDA/model work ran through Slurm; at most2 allocated GPUs. Builders exited before vLLM began. Commands, timings, job IDs and logs are in commands_*.json. See protocol.md for fixed settings.

All new sources use E4 validation-selected fixed learning rates; no new LR tuning. Warmup is explicit10% steps, verified in saved arguments and actual Trainer schedule. New trainings use a predeclared final checkpoint. See training_audit.json and training_commands_*.json for hashes, actual steps, GPU time/logs.

## Three training seeds

| Task | Method | Target mean ± SD | Off mean ± SD | FG mean ± SD |
| --- | --- | ---: | ---: | ---: |
| magicoder | base | 66.4634 ± 0.0000 | 79.3692 ± 0.0000 | 0.0000 ± 0.0000 |
| magicoder | original_lora | 64.4309 ± 0.3520 | 77.5869 ± 0.0685 | 2.0888 ± 0.0774 |
| magicoder | recon_0 | 65.0407 ± 0.9314 | 77.9000 ± 0.2841 | 1.7934 ± 0.2776 |
| magicoder | hns_f4_s1 | 73.3740 ± 0.3520 | 80.5569 ± 0.0031 | 0.0000 ± 0.0000 |
| magicoder | flat_nuclear | 73.9837 ± 0.9314 | 80.7830 ± 0.2481 | 0.0000 ± 0.0000 |
| metamath | base | 85.7468 ± 0.0000 | 72.9414 ± 0.0000 | 0.0000 ± 0.0000 |
| metamath | original_lora | 84.8623 ± 0.4632 | 68.4982 ± 2.2870 | 4.5149 ± 2.3475 |
| metamath | recon_0 | 84.7106 ± 0.3064 | 68.8382 ± 1.5781 | 4.1721 ± 1.6407 |
| metamath | hns_f4_s1 | 88.0212 ± 1.1013 | 73.9437 ± 0.3034 | 0.2033 ± 0.3520 |
| metamath | flat_nuclear | 87.8949 ± 0.3740 | 73.8646 ± 0.7781 | 0.2033 ± 0.3520 |

SD is across3 training seeds, with a fixed data subset and shared test sets; it does not measure independent test-set uncertainty. Per-seed paired item intervals and checkpoint descriptive intervals are both included with their statistical units.
