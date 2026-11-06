# E2: predeclared protocol

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


Three mandatory fixed variants; 216 new cells. HNS-od is HNS4+1 strength1 nuclear restoration, only o_proj/down_proj; all other source factors remain byte-identical. SS-smooth_abs uses o_proj/down_proj. SS-smooth_abs-all uses q/k/v/o/gate/up/down_proj. Optional grad_direction-all omitted, no optional results claimed.
Paper: https://arxiv.org/html/2603.03995v1 section3.4,4.3. Original author repository code: https://github.com/TTtianTT/peft-sft-lab/blob/139bfd7/src/finetune/spectral_edit/edit_strategies.py ; independently fetched through GitHub connector and SHA256-matched before evaluation (bffb4aaa044f27a9e38718a832c30b93e29247ec71f267561c98ceb5c3acf2b9).
Copy original defaults exactly: smooth_abs; core_frac0.2; noise_frac0.2; amp_factor1.25; sup_factor0.8; mid_factor1; min_core_k1; smooth_temperature0.35; smooth_center_q0.5; smooth_align_mid true; grad_norm mean_abs; preserve_energy l1; sigma_clip_min0. calib128; batch2; train split; shuffle seed42; start0; existing formatter, answer-only teacher forcing, float16 base. No calibration/test-based tuning.
Fair scope tables: {o,down}: HNS-od and SS-smooth_abs plus cached SS-grad_direction; all7: HNS and SS-smooth_abs-all. Also compare original LoRA and Recon-0. SS has 128 calibration examples and64 forward/backward batches per checkpoint/variant; record actual metadata, wall seconds, forward/back counts and GPU seconds. HNS has0 samples/forward/backward; synchronized measured compact SVD/edit/rebuild time, and load/save/audit overhead separately. Budgets are reported, not asserted equal.
For SS, audit source and edited compact reconstructions, nuclear conservation, module coverage and unchanged unselected factor bytes. Direction preservation checked via a joint QR residual against the source U/V subspaces. Predeclared tolerance5e-5 relative update/nuclear/direction error. No hyperparameter search.
