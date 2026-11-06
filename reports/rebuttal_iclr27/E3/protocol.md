# E3: predeclared protocol

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


216 new cells; no datasets used to build any variant. Same source SVD basis, same actual balanced factor writeback as E1/HNS. For each module let h be the main HNS4+1 nuclear-restored spectrum, a=||h||2/||sigma||2. Mag-only=sigma*a. Shape-only=h/a. Global-gamma=sigma*mean_module(a), one arithmetic mean per adapter. No post-cast energy correction, no nuclear restoration for these three controls. Hold rank, alpha and dtype fixed. All18 checkpoints/4 families; no test-set selection. All three fixed settings reported.
Factorial cells: original shape/original magnitude=Recon-0; original shape/HNS magnitude=Mag-only; HNS shape/original magnitude=Shape-only; HNS shape/HNS magnitude=HNS. Shape main effect=0.5*((HNS-Mag)+(Shape-Recon)); magnitude main effect=0.5*((HNS-Shape)+(Mag-Recon)); interaction=HNS-Mag-Shape+Recon. For Target,Off,FG compute each contrast per checkpoint then paired bootstrap20000, seed20261106; shared draws, descriptive fixed-test intervals; FG lower favorable.
Plot per-module alpha vs PR1=(||sigma||1/||sigma||2)^2 and exact-flat sqrt(PR1/r), by module family. Report true HNS a and deviations from strict-flat identity. Join Scalar-E coefficients from frozen energy_match metadata/module audit; pooled and within-checkpoint Spearman/Pearson (descriptive; modules correlated). Compare Mag-only vs Global-gamma with paired checkpoint CIs. Prior Scalar-F is mathematically related but is not silently substituted: this experiment uses these exact factors/cast timing without its storage energy correction.
