# E1: predeclared protocol

Frozen before new evaluation: 2026-11-06T12:57:52.028941+00:00.

Fixed cohort: Qwen3-8B and Llama-3.1-8B-Instruct x magicoder/metamath/tulu x seeds 42/43/44 (18 checkpoints). No training.
Reference: functional_hns_three_seed_20260914, including final refreshed Llama caches.
HumanEval pass@1 (164), GSM8K strict (1319), IFEval prompt strict (541), Commonsense-8 equal-subtask macro (22419 items).
Unchanged prompts, chat template, parsers, scorers and datasets. Greedy seed42; batch invariant; max_model_len4096; adapter block5; max_lora_rank16; max_num_batched_tokens65536; FLASH_ATTN; async scheduling off; prefix caching off; gpu_memory_utilization0.94.
Long max_num_seqs2048, commonsense4096; prompt chunks short4096/long1024. No adaptive test-set selection.
Reuse Base, original_lora, hns_f4_s1 only after SHA256 and ordered per-sample input-hash checks. Reuse Scalar-E, Flat-E and all three predeclared PARA settings and Spectral Surgery grad_direction under the same checks. Run only new cells.
Target is the training task's benchmark family; Off is the equal mean of the other three families. FG = mean over those three families of max(Base-family score - edited-family score, 0), clipping each family before averaging. Commonsense is one macro family.
Report percentage points. Paired source-checkpoint bootstrap: 20000 resamples, seed20261106, percentile95% interval. Shared draws within group, wins/ties/losses (FG lower is better), tolerance1e-10 pp. Overall and model/task/model-task subgroups. Descriptive fixed-shared-test-set intervals, not IID training-run confidence intervals; unadjusted comparisons.
Keep all configurations. HNS4+1 strength1 nuclear-preserving all seven module types. Source rank/alpha/scaling and storage dtype unchanged.
Audit weights/config SHA256, selected/total module coverage, finite spectra, relative reconstruction error. Predeclared fp32 relative-update tolerance5e-5; report maxima without rounding away anomalies. No threshold change after seeing results.
Global maximum2 GPUs across projects; all construction using CUDA and all model evaluation through Slurm.

Recon-0: call the existing build_hns_step_grid_2x4.build_checkpoint, apply_hns_to_svd and rebuild_ba_from_uv_sigma functions. fast_steps0, stable_steps0, hns_strength0 forces the original singular values exactly; preserve_nuclear_norm true. Identical fp32 compact QR/SVD and cast timing to the main HNS builder. All source scalings are2, so decomposing BA and retaining scaling is equivalent to editing DeltaW=sBA without changing scaling.
Implementation clarification: actual main HNS uses balanced B=U sqrt(sigma), A=sqrt(sigma) V^T. The proposal's unbalanced description does not match svd.py. Recon-0 follows actual main code. Optional alternative-factorization GPU runs are deferred; no claim of robustness from unrun controls.
72 new cells. Compare HNS, Flat-Nuclear, Scalar-E, Flat-E, PARA e90/e95/e99 and SS grad_direction against Recon-0, plus Recon-0 versus LoRA. Flat-Nuclear comparison depends on E0; interim E1 report remains awaiting_E0 until its main-wave cells exist. E1/E0 overlap is explicitly allowed in the request. No E2 until both have complete audited reports.
