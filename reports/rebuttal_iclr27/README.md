# ICLR 2027 rebuttal execution

Branch: `refactor/sft-chat-template`; original source commit: `139bfd7`.

All main tests use the final `functional_hns_three_seed_20260914` protocol. The preflight verified 584 cached cells sample by sample, all 144 reference adapters, 36 unchanged historical Flat adapters, and exact library-version agreement. See `cache_audit.json`, `cache_input_audit.tsv`, `base_sample_input_hashes.tsv.gz`, `reference_artifact_audit.json`, and `environment_audit.json`.

Execution is serial across stages, with at most two allocated GPUs:

1. E1 Recon-0, then E0 main-wave Flat evaluations; E1's final Flat comparison waits for E0. Slurm 14889 retries failed initialization job 14888. Both share the two explicitly compatible P0 stages.
2. E2 module-matched HNS/Spectral Surgery, then E3 data-free factorial controls. Start only after audited E1/E0 packages exist. Each experiment completes its four requested files and integrity/coverage audits before the next starts.
3. E4a six Qwen trainings with validation-only LR/checkpoint selection, then E4b one rank64 training; 48 new main cells. Explicit warmup_steps and actual Trainer schedules are checked. Optional trainings/evaluations omitted.
4. E5 scope assessment and E6 code-exact algorithm box, after E4. LoDA public code is vision continual learning, so its direct Qwen port is omitted under the requested setting-mismatch exception.

Stage scripts refuse GPU construction/evaluation without a Slurm allocation. E1 uses the actual main HNS balanced factorization, rather than the unbalanced description in the proposed plan. The source update reconstruction errors are recorded; numerical identity of weights or token outputs is never presumed.

`submission.json` records job IDs, GPU counts and log paths. `protocol.md` and `protocol.lock.json` in each experiment freeze settings before its new work. Each finished package contains `report.md`, `summary.json`, `checkpoint_results.tsv`, `paired_ci.tsv`, and numerical/coverage audits. Expected outputs are validated before reporting completion. A result containing zero in its interval is not called equivalent; shared-test intervals are not IID training-run confidence intervals.

Completed reports are committed and pushed on the existing branch by the stage finalizer. It refuses to include pre-existing staged work or files outside this package. Large weights, raw predictions and logs stay under ignored `runtime/`. Publication failures are recorded separately from experiment completion.
