# R2-O8: completed results

Status: complete. All prespecified conditions are reported.

| Condition | Method | Target | Off | FG |
| --- | --- | ---: | ---: | ---: |
| Qwen3-8B_magicoder_loraplus_r16_seed42 | original_lora | 70.1220 | 76.8535 | 2.8136 |
| Qwen3-8B_magicoder_loraplus_r16_seed42 | recon_0 | 71.3415 | 77.2225 | 2.4612 |
| Qwen3-8B_magicoder_loraplus_r16_seed42 | hns_f4_s1 | 78.0488 | 81.0021 | 0.0000 |
| Qwen3-8B_magicoder_loraplus_r16_seed42 | flat_nuclear | 78.0488 | 80.5655 | 0.0000 |
| Qwen3-8B_metamath_loraplus_r16_seed42 | original_lora | 80.4397 | 38.2496 | 34.6918 |
| Qwen3-8B_metamath_loraplus_r16_seed42 | recon_0 | 80.8946 | 38.7887 | 34.1527 |
| Qwen3-8B_metamath_loraplus_r16_seed42 | hns_f4_s1 | 79.7574 | 40.5601 | 32.3813 |
| Qwen3-8B_metamath_loraplus_r16_seed42 | flat_nuclear | 79.2267 | 40.6608 | 32.2806 |

Target/Off are percentages; FG/differences are pp. Off averages3 non-target families equally; FG clips each family before averaging. Commonsense is an equal8-subtask macro.

All20,000-draw paired intervals and wins/ties/losses or exact McNemar repair/damage counts are in paired_ci.tsv. Single-seed, paired question-level intervals; no training-run inference.

Source/config/weight SHA256, module coverage, reconstruction errors and ordered per-sample input hashes passed. All CUDA/model work ran through Slurm; at most2 allocated GPUs. Builders exited before vLLM began. Commands, timings, job IDs and logs are in commands_*.json. See protocol.md for fixed settings.

All new sources use E4 validation-selected fixed learning rates; no new LR tuning. Warmup is explicit10% steps, verified in saved arguments and actual Trainer schedule. New trainings use a predeclared final checkpoint. See training_audit.json and training_commands_*.json for hashes, actual steps, GPU time/logs.

LoRA+ uses the repository default LR ratio20.0, explicitly supplied and checked in run_args.json. DoRA carries an independent magnitude vector and PiSSA rewrites the base residual, so both differ from editing BA alone and are excluded as prespecified.
