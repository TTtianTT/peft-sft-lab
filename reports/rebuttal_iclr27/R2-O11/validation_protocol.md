# R2-O11: validation-selected global gamma (locked before any O11 inference)

Author approval: 2026-11-09 (paper author, in Claude Code session). Frozen UTC 2026-11-09T03:56:57.968614+00:00.

Question (reviewer 8Euc, "appropriate scaling"): can one global scale of the update, chosen on independent validation data,
match the data-free nuclear-preserving flattening? This differs from E3 Global-gamma (data-free mean Frobenius ratio).

Cohort: the fixed 18 main checkpoints (Qwen3-8B / Llama-3.1-8B-Instruct x magicoder/metamath/tulu x seed42/43/44). No retraining.
Candidates: gamma in [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]. Candidate = gamma * Recon-0: shared fp32 compact SVD of BA, singular values multiplied by
gamma, balanced writeback B=U sqrt(gamma sigma), A=sqrt(gamma sigma) Vh, cast to source dtype, native rank/alpha/config preserved.
gamma=1.0 is the audited E1 Recon-0 adapter itself (weights SHA256 checked). Audit: relative reconstruction error, error to
gamma*source update and nuclear ratio to gamma*||sigma||_1 all below the fixed 5e-5, every module.

Validation (target task only; one score per checkpoint x gamma, 126 runs):
- magicoder: E4 MBPP200 (sanitized MBPP test, seed42 subset), unchanged eval_mbpp.py pass@1, max_new_tokens512.
- metamath: SVAMP500 (ChilleD/SVAMP@5e0bf1e5e7c0, train+test pool, seed42 subset), formatted as
  GSM8K-style '#### answer'; unchanged eval_gsm8k.py strict accuracy, max_new_tokens512. The E4 MetaMath held-out set is NOT used:
  it was drawn from the same 50K MetaMathQA subset the old sources trained on.
- tulu: 500 prompts from argilla/ifeval-like-data (config filtered@56505d1771e3), only instruction ids present in the main IFEval set and implemented by the
  unchanged main scorer, exact prompts of IFEval541 and of the local Tulu personas IF training table removed; unchanged eval_ifeval.py
  prompt-level strict accuracy (--fail_on_unknown), max_new_tokens2048.
All: non_thinking chat template as E4/O9 validation, vLLM TP1, max_model_len4096, FLASH_ATTN, flashinfer sampler off, seed42,
batch-invariant environment, inside Slurm, at most 2 GPUs (one worker per base model).

Selection rule (fixed now): highest validation score; ties broken by larger gamma (smaller intervention, closer to Recon-0).
Never use HumanEval/GSM8K/IFEval/Commonsense test scores. All 126 validation scores are reported.

Test: selected adapter = method gamma_val on the four unchanged main families. If gamma=1.0 is selected, gamma_val is the
E1 Recon-0 adapter and its audited cached cells are reused (no new inference). Compare with original_lora, recon_0 (E1 cache)
and hns_f4_s1 (main cache). Statistics: paired checkpoint bootstrap 20000, seed20261106, descriptive on shared tests,
wins/ties/losses; Target/Off/FG definitions unchanged. All outcomes are reported, including any unfavourable to HNS.
Validation budget is recorded separately (126 validation runs; HNS uses none).
