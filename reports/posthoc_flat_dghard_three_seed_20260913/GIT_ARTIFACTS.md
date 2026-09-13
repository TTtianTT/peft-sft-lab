# Git artifact package

The complete result report is [../posthoc_flat_dghard_three_seed_20260913.md](../posthoc_flat_dghard_three_seed_20260913.md). The primary five-method comparison is `final_summary.json`; `flat_summary.md/json` and `dg_initial_summary.json` retain the earlier evaluations.

Git includes source/build/experiment/variant manifests, adapter configs, per-module `posthoc_meta.json` audits, generation and score manifests, commands, completion markers, and losslessly compressed per-sample JSONL files and logs. `git_artifact_manifest.json` records original and compressed paths, sizes, and SHA256 hashes. Slurm logs are under `../../logs/posthoc-*.out.gz` and `*.err.gz`.

Adapter weights and copied `tokenizer.json` files remain at the local paths recorded in the source and variant manifests. They are excluded from this Git package. Compiler and multiprocessing scratch files are also excluded. This package contains evaluation evidence and reconstruction code; it is not a standalone model distribution.

To restore a JSONL or log file for an existing analysis command, decompress its sibling `.gz` file with `gzip -dk path/to/file.jsonl.gz`. Original uncompressed files remain available in the working experiment directory. The existing analysis scripts read uncompressed JSONL files.

Numerical tests: `PYTHONPATH=src OMP_NUM_THREADS=4 /dataset1/zailong/envs/peft-sft-lab/bin/python -m pytest -q tests/test_posthoc_flat.py tests/test_posthoc_dg_hard.py`.
