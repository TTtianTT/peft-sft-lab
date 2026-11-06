# E0: predeclared protocol

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

144 new cells: Flat-Nuclear and Flat-Fro, 18 checkpoints, four benchmark families. Reuse the exact36 previously built adapters referenced by the Slurm992 forgetting manifest; verify weight/config SHA256 before use. Never rebuild them.
Unified table: LoRA, Recon-0, HNS, Flat-Nuclear, Flat-Fro, Scalar-E, Flat-E. Pair every method with LoRA/HNS; include Flat-Nuclear minus Flat-Fro. No expectation-based conclusion: intervals overlapping zero do not prove equivalence.
Audit both waves per sample: ordered input SHA256, changed generated token sequence, changed prediction text, changed binary strict correctness; totals by checkpoint/family/method and subtask. Historical outputs are comparisons only, not current scores.
