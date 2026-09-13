#!/usr/bin/env python3
"""Prepare and explicitly publish 12 audited B300 training-seed LoRA checkpoints.

Preparation is local. --publish creates new public repositories, uploads the
prepared artifacts, and adds them to the existing parent/task collections.
Existing unrelated repositories are never overwritten.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912')
OWNER = 'tianzl66'
PARENT = 'tianzl66/spectral-surgery-6a8fde803843c3f48b8fdcb1'
COLLECTIONS = {
    'magicoder': 'tianzl66/spectral-surgery-code-6a8fe1e39d935d63103c39aa',
    'metamath': 'tianzl66/spectral-surgery-math-6a8fed91e85118adb8360bb6',
    'tulu': 'tianzl66/spectral-surgery-instruction-following-6a8fef585e98ebd2e00c2c60',
}
BASES = {'Qwen3-8B': 'Qwen/Qwen3-8B', 'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct'}
TASKS = {
    'magicoder': ('Magicoder-50K-LoRA-E1', 'ise-uiuc/Magicoder-Evol-Instruct-110K', 'HumanEval', 'code'),
    'metamath': ('MetaMathQA-50K-LoRA', 'meta-math/MetaMathQA', 'GSM8K', 'math'),
    'tulu': ('InstructionFollowing-LoRA', 'allenai/tulu-3-sft-personas-instruction-following', 'IFEval', 'instruction-following'),
}
SCRIPTS = ['build_hns_step_grid_2x4.py', 'eval_forgetting_matrix_vllm.py', 'score_forgetting_matrix.py', 'eval_commonsense_8tasks.py']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def payload_files(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256'
                  and not any(part in ('__pycache__', '.cache') for part in p.parts) and p.suffix != '.pyc')


def clean(value):
    if isinstance(value, dict):
        return {k: ('<REDACTED>' if re.search(r'(^|_)(token|password|secret|credential|api_key)($|_)', k, re.I) and not isinstance(v, (dict, list, bool, int, float)) else clean(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        if value.startswith(('/dataset1/', '/export/', '/root/', '/tmp/')):
            return '<LOCAL_PATH>/' + Path(value).name
        return re.sub(r'hf_[A-Za-z0-9]{20,}', '<REDACTED>', value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def base_provenance(base):
    model = Path('/dataset1/zailong/models') / base
    revisions = {}
    for path in model.glob('.cache/huggingface/download/*.metadata'):
        lines = path.read_text().splitlines()
        if lines and re.fullmatch('[0-9a-f]{40}', lines[0]):
            revisions[path.name.removesuffix('.metadata')] = lines[0]
    tracked = {p.name: digest(p) for p in model.iterdir() if p.is_file() and (p.suffix == '.safetensors' or p.name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'model.safetensors.index.json'))}
    unique = set(revisions.values())
    return {'model_id': BASES[base], 'download_metadata_revisions': revisions,
            'revision': next(iter(unique)) if len(unique) == 1 else None,
            'files_sha256': tracked, 'revision_note': 'Revision is taken from local download metadata, not current Hub main.'}


def model_card(meta, results, stage):
    a, t, cfg = meta['requested_run_args'], meta['effective_training_args'], meta['run_config']
    base, task, seed = meta['base'], meta['train_task'], meta['training_seed']
    _, dataset_id, benchmark, tag = TASKS[task]
    metric = results[0]['primary_metric']
    original = next(r for r in results if r['variant'] == f'{task}__original_lora')
    baseline = original[metric]
    names = {'base': 'Base (not a training replicate)', f'{task}__original_lora': 'LoRA (weights in this repository)'}
    for fast in (0, 2, 4, 8):
        for stable in ((0,) if fast == 0 else (0, 1, 2)):
            names[f'{task}__hns_f{fast}_s{stable}'] = '0+0 SVD reconstruction control' if fast == 0 else f'HNS {fast}+{stable}, all modules'
    ordered = sorted(results, key=lambda r: (0 if r['variant'] == 'base' else 1 if r['variant'].endswith('original_lora') else 2, r['variant']))
    table = ['| Method | Score (%) | Correct / samples | Change vs LoRA (pp) |', '|---|---:|---|---:|']
    for r in ordered:
        count = r.get('correct', r.get('prompt_level_strict_correct'))
        if count is None:
            count = round(r[r['primary_metric']] * r['samples'])
        delta = '—' if r['variant'] == 'base' else f'{(r[r["primary_metric"]] - baseline) * 100:+.2f}'
        table.append(f'| {names[r["variant"]]} | {r[r["primary_metric"]] * 100:.2f} | {count}/{r["samples"]} | {delta} |')
    target_modules = ', '.join(meta['adapter_config']['target_modules'])
    inference = meta['inference']
    cache_note = meta.get('compilation_cache_note', 'Disabled, per-job cache root, short IPC temp path')
    base_revision = meta['base_provenance']['revision'] or 'not uniquely identified; see per-file revisions/hashes'
    return f'''---
base_model: {BASES[base]}
library_name: peft
pipeline_tag: text-generation
datasets:
- {dataset_id}
tags:
- lora
- spectral-surgery
- training-seed-replication
- b300
- {tag}
- seed-{seed}
---

# {base} + {task} — LoRA, training seed {seed}

This repository contains the **unedited final LoRA adapter**, not a full 8B base model and not an HNS-edited adapter. It is one of the 12 B300 replication runs (two bases × three training tasks × seeds43/44), completed September 13, 2026. HNS is a post-hoc transformation of this saved LoRA, not additional training. All HNS scores below use this source checkpoint; derived HNS weights are not uploaded here. Reconstruction code and metadata are included.

## Base Model

- Model: [{BASES[base]}](https://huggingface.co/{BASES[base]}).
- Download revision: `{base_revision}`. Per-file local download revisions and weight/tokenizer SHA256 hashes are in `publication.json`.
- Base weights are not redistributed. Users must obtain access to the base and comply with its license, acceptable-use policy and dataset terms. No independent license grant is implied by this adapter release.

## Training — requested versus effective configuration

| Field | Actual saved run / provenance |
|---|---|
| Dataset | `{dataset_id}`, local train parquet snapshot |
| Source rows | {meta['dataset']['source_rows']} |
| Selected rows before truncation filtering | {cfg['dataset_size']} |
| Actual response-supervised training rows | {meta['actual_training_examples']} |
| Training epochs | {a['num_train_epochs']} |
| Actual final optimizer updates | {meta['trainer_state']['global_step']} |
| Maximum sequence length | {a['max_seq_len']} |
| Per-device micro-batch / accumulation | {a['per_device_train_batch_size']} / {a['gradient_accumulation_steps']} |
| GPU count / effective global batch | 1 NVIDIA B300 / {cfg['effective_global_batch_size']} |
| LR / scheduler | {a['lr']} / `{t['lr_scheduler_type']}` |
| **Actual warmup** | **{t['warmup_steps']} steps (zero warmup)** |
| CLI-requested warmup ratio | {a['warmup_ratio']} — did NOT take effect |
| Scheduler kwargs | `{json.dumps(t.get('lr_scheduler_kwargs'))}` |
| Optimizer | `{t['optim']}`, Adam betas ({t['adam_beta1']}, {t['adam_beta2']}), epsilon {t['adam_epsilon']} |
| Weight decay / max gradient norm | {t['weight_decay']} / {t['max_grad_norm']} |
| LoRA | r={a['r']}, alpha={a['lora_alpha']}, dropout={a['lora_dropout']}, bias=none |
| Target modules | {target_modules} |
| Precision / checkpointing | bf16 / gradient checkpointing enabled; not QLoRA |
| SFT | Chat template `{a['chat_template_mode']}`; response-only loss, prompt labels masked -100 |
| Truncation / padding | Right truncation; drop examples with no supervised completion tokens; dynamic right padding to multiple8; no packing |
| Training seed / Trainer data_seed | {seed} / {t['data_seed']} |
| Dataset subset seed | 42, unchanged between seeds43/44 |
| Determinism | full_determinism={t['full_determinism']}; no claim of bitwise reproducibility |

**Warmup audit correction:** this code passes CLI `warmup_ratio`, then filters TrainingArguments kwargs against the installed signature. Transformers5.16.1 does not expose that argument, so the requested ratio was dropped and `warmup_steps=0` remained. `training_args.json` and the archived source are authoritative for the effective run, not the requested CLI alone. The reproduction command deliberately requests ratio0. Do not describe these runs as having 5% or 10% warmup.

Data selection: valid-format filtering, then `datasets.Dataset.shuffle(seed=42).select(range(50000))` for Magicoder/MetaMath; Tulu uses the full valid split without downsampling. The row-index list in `data/selected_source_indices.json.gz` references the exact local parquet row ordering **before** tokenization/truncation filtering. Source file SHA256: `{meta['dataset']['file_sha256']}`. Upstream dataset revision was not recorded by the original asset export; do not claim that downloading current main recreates the exact bytes/order. Verify the file hash or resolve snapshot provenance before claiming exact data reproduction. Training text is not redistributed here.

Saved evidence: `run_args.json` (requested), `run_config.json` (pre-tokenization estimates), `training_args.json` (effective), `trainer_state.json` (actual final steps and logged training metrics), `requirements-freeze.txt`, and `publication.json`. Local paths and credential fields are sanitized. Pickled optimizer states / training_args.bin are intentionally omitted.

## Evaluation

Benchmark: **{benchmark}**, primary metric `{metric}`, {original['samples']} items. Each trained checkpoint was evaluated once on the complete available in-domain split. This is not three repetitions of inference on one checkpoint. Seeds43/44 are separate training runs. No training-seed CI or significance claim is made from a single row.

| Setting | Value |
|---|---|
| Backend / attention | vLLM / FLASH_ATTN, tensor parallel1 |
| Sampling | Greedy: temperature0, top_p1, inference seed42 |
| Chat | non_thinking render mode; Qwen enable_thinking=False |
| Maximum model length / new tokens | 4096 / {meta['evaluation_task']['max_new_tokens']} |
| GPU memory / max concurrent sequences | {inference['gpu_memory_utilization']} / {inference['max_num_seqs']} |
| Token budget / adapter block / prompt chunk | {inference['max_num_batched_tokens']} / {inference['adapter_block_size']} / 512 |
| Scheduling / prefix cache | async_scheduling=False / enable_prefix_caching=False |
| Numerics | VLLM_BATCH_INVARIANT=1, CUBLAS_WORKSPACE_CONFIG=:4096:8 |
| Compilation cache | {cache_note} |

HumanEval uses chat strict-continuation prompts, max_new_tokens512, pass@1, code-execution timeout3s and 32 CPU workers. GSM8K uses max_new_tokens512 and the strict answer extractor in the archived scorer (not a 2048-token model-card evaluation). IFEval uses its 541-item `train`-named evaluation split, max_new_tokens2048, prompt-level strict accuracy. The GSM8K/IFEval local benchmark inputs were reconstructed from earlier scored outputs (gold and instruction metadata), not newly sampled; the exact input file and hash are included under `evaluation/benchmark_input/`. HumanEval input is the local test parquet. Do not mix these results with earlier model-card scores from other prompts/token budgets.

{os.linesep.join(table)}

HNS grid: all seven LoRA module types, output rank16, strength1, preserve original module nuclear norm, fast steps2/4/8 × stable steps0/1/2. `0+0` is an SVD-factorization reconstruction control, not spectral editing. Maxima on this test set are descriptive, not validated parameter selection. In IFEval, reconstruction itself can change scores materially; all gains over LoRA cannot automatically be attributed to spectral editing.

Machine-readable full metrics are in `evaluation/results.json`; paired per-item evidence and generated token IDs/text are in `evaluation/items/<variant>/scored.jsonl.gz` and `predictions.jsonl.gz`. Off-task forgetting evaluation was still incomplete at publication preparation; no incomplete forgetting scores are included or implied.

## Reproduction and loading

Use an isolated environment matching the recorded package versions. `requirements-freeze.txt` is the full training environment inventory, not a guarantee that all platform-specific packages install on arbitrary systems. The source archive is a publication-time snapshot, with per-file hashes; a clean training-time Git commit was not saved. Python version and evaluation-time package versions are recorded in `publication.json` (current evaluation environment observation is distinguished from training inventory).

```bash
# In the downloaded repository directory:
tar -xzf code/source_snapshot.tar.gz
pip install --no-deps -e source_snapshot
# Obtain the exact training parquet under /path/to/data/; its SHA256 is checked.
python code/reproduce_hns_seed_checkpoint.py train --data-dir /path/to/data --output-dir /path/to/new-run
# Rebuild all nine HNS variants plus the 0+0 control from the root LoRA:
python code/reproduce_hns_seed_checkpoint.py build-hns --output-dir /path/to/rebuilt-grid
# In-domain benchmark inputs are included; record any inference-budget override:
python code/reproduce_hns_seed_checkpoint.py evaluate --grid-dir /path/to/rebuilt-grid --output-dir /path/to/new-eval
```

The archived training entrypoint and the explicit command implement the actual zero-warmup run. With other Transformers versions, defaults/Trainer behavior may differ; recorded data hashes, model revisions, tokenizer files, preprocessing and effective settings are necessary checks, not a promise of bitwise identical training. For the successful Llama43 evaluation the recorded batched-token budget was131072, whereas the other new groups used65536. This stack has also failed at131072 in other initializations; `evaluate --token-budget 65536` is a safer alternative but a changed evaluation configuration and must be reported.

`build-hns` also fetches the other two source LoRAs for this same base/training seed, verifies their published weight hashes, and reconstructs the original three-task 33-adapter manifest ordering. This preserves adapter registration IDs for the target-task evaluation rather than renumbering an isolated 11-adapter grid. `--peer-root` can point to an offline directory containing those named repository folders. For stability the portable helper disables the compilation cache; this differs from the earlier successful Llama43 run, whose scheduler log shows default compile-cache use. No original compiled cache is redistributed, and exact-token numerical identity is not promised.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

repo_id = "{meta['repo_id']}"
base_id = "{BASES[base]}"
base_revision = {repr(meta['base_provenance']['revision'])}
tokenizer = AutoTokenizer.from_pretrained(repo_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id, revision=base_revision, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, repo_id)
model.eval()
```

`adapter_config.json` uses the public base-model ID rather than a private filesystem path; this metadata normalization does not change `adapter_model.safetensors`. Original adapter-config hash is recorded separately. Root tokenizer and chat-template files are the saved training artifacts. Loading example is not itself a benchmark reproduction protocol.

## Comparison with the historical seed42 checkpoint

The older checkpoint is separately listed in `comparison/three_run_scores.json` and `comparison/configuration_audit.md`. It is a historical reference, **not verified to be an identical-recipe third seed**. New runs use larger micro-batches, padding8 and actual zero warmup; original dataset identity and some Llama settings are not fully verified. Both new seeds share the same saved non-seed recipe. Old Llama Magicoder/MetaMath training seed labels lack complete original Trainer evidence in this audit. Do not pool 42/43/44 into a strict identical-configuration three-seed mean±SD. Standard deviation is not a confidence interval.

## Files and integrity

- `adapter_model.safetensors`, `adapter_config.json`, saved tokenizer/chat template: loadable PEFT source LoRA.
- Training JSON evidence, final `trainer_state.json`, requirements inventory.
- `publication.json`, `data/*`, `hns/*`, `evaluation/*`, `comparison/*`: provenance, HNS metadata, input/output evidence and historical comparison.
- `code/*`: source archive and portable train/build/evaluate helper.
- `MANIFEST.sha256`: hashes of all prepared payload files except itself; remote commit ID is tracked in the publisher's upload receipt.

The pre-normalization trained weight SHA256 is `{meta['adapter_weights_sha256']}`. No intermediate checkpoint, optimizer state, full base weight or scheduler log is uploaded. This release documents reproducibility boundaries rather than claiming random variation was eliminated.
'''


def prepare(stage_root):
    import numpy as np
    import pyarrow.parquet as pq
    from summarize_hns_three_seed_diagonal import collect
    stage_root.mkdir(parents=True, exist_ok=True)
    base_info = {}
    for base in BASES:
        print(f'[Hash base] {base}', flush=True)
        base_info[base] = base_provenance(base)
    datasets = {}
    for task in TASKS:
        a = json.loads((RUN / 'Qwen3-8B/seed43/lora' / task / 'run_args.json').read_text())
        path = Path(a['dataset_path'])
        n = pq.ParquetFile(path).metadata.num_rows
        selected = np.random.default_rng(42).permutation(n)[:50000].tolist() if task != 'tulu' else list(range(n))
        datasets[task] = ({'dataset_id': TASKS[task][1], 'filename': path.name, 'file_sha256': digest(path), 'source_rows': n,
                           'upstream_revision': None, 'selected_index_semantics': 'zero-based original parquet row index, before truncation filtering'}, selected)
    all_scores = collect()
    plan = []
    for base in BASES:
        for seed in (43, 44):
            seed_root = RUN / base / f'seed{seed}'
            generation = json.loads((seed_root / 'hns_step_grid/eval/generation_manifest.json').read_text())
            scores = json.loads((seed_root / 'hns_step_grid/eval/score_manifest.json').read_text())['records']
            task_config = json.loads((seed_root / 'config/task_config.json').read_text())
            for task in TASKS:
                name = f'{base}-{TASKS[task][0]}-Seed{seed}'
                repo = f'{OWNER}/{name}'
                output = stage_root / name
                output.mkdir(parents=True, exist_ok=True)
                source = seed_root / 'lora' / task
                requested = json.loads((source / 'run_args.json').read_text())
                effective = json.loads((source / 'training_args.json').read_text())
                assert effective['seed'] == effective['data_seed'] == seed
                assert effective['warmup_steps'] == 0 and 'warmup_ratio' not in effective
                states = sorted(source.glob('checkpoint-*/trainer_state.json'), key=lambda p: int(p.parent.name.split('-')[-1]))
                state = json.loads(states[-1].read_text())
                assert state['global_step'] == state['max_steps']
                config = json.loads((source / 'run_config.json').read_text())
                matches = re.findall(r'Prepared (\d+) tokenized SFT examples', (source / 'train.log').read_text())
                assert matches
                adapter_config = json.loads((source / 'adapter_config.json').read_text())
                weights_sha = digest(source / 'adapter_model.safetensors')
                meta = {'release': 'hns-b300-training-seeds-20260913', 'repo_id': repo, 'base': base, 'train_task': task,
                        'training_seed': seed, 'requested_run_args': clean(requested), 'effective_training_args': clean(effective),
                        'run_config': clean(config), 'actual_training_examples': int(matches[-1]), 'trainer_state': clean(state),
                        'adapter_config': clean(adapter_config), 'adapter_config_original_sha256': digest(source / 'adapter_config.json'),
                        'adapter_weights_sha256': weights_sha, 'base_provenance': base_info[base], 'dataset': datasets[task][0],
                        'inference': generation['configuration'], 'evaluation_task': clean(task_config['evaluation_tasks'][task]),
                        'historical_recipe_equivalence_verified': False, 'warmup_requested_but_effective_zero': True,
                        'training_python_version': 'not separately recorded', 'publication_python_version': os.sys.version,
                        'source_provenance': 'publication-time snapshot; no clean training-time Git commit recorded',
                        'prepared_at_utc': datetime.now(timezone.utc).isoformat()}
                import importlib.metadata
                meta['publication_eval_environment_observation'] = {k: importlib.metadata.version(k) for k in ('vllm', 'torch', 'transformers', 'peft', 'datasets', 'numpy', 'human-eval')}
                write_json(output / 'publication.json', meta)
                for filename in ('adapter_model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'special_tokens_map.json', 'requirements-freeze.txt'):
                    path = source / filename
                    if path.is_file():
                        shutil.copy2(path, output / filename)
                public_adapter = dict(adapter_config, base_model_name_or_path=BASES[base])
                if not public_adapter.get('revision') and base_info[base]['revision']:
                    public_adapter['revision'] = base_info[base]['revision']
                write_json(output / 'adapter_config.json', public_adapter)
                for filename, value in [('run_args.json', requested), ('run_config.json', config), ('training_args.json', effective), ('trainer_state.json', state)]:
                    write_json(output / filename, clean(value))
                data_dir = output / 'data'
                data_dir.mkdir(exist_ok=True)
                with gzip.GzipFile(filename=str(data_dir / 'selected_source_indices.json.gz'), mode='wb', mtime=0) as f:
                    f.write(json.dumps(datasets[task][1]).encode())
                write_json(data_dir / 'dataset_provenance.json', datasets[task][0])
                selected_scores = [r for r in scores if r['task'] == task and (r['variant'] == 'base' or r['variant'].startswith(task + '__'))]
                assert len(selected_scores) == 12
                write_json(output / 'evaluation/results.json', clean(selected_scores))
                write_json(output / 'evaluation/generation_manifest.json', clean(generation))
                benchmark = Path(task_config['evaluation_tasks'][task]['dataset'])
                target = output / 'evaluation/benchmark_input' / benchmark.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(benchmark, target)
                write_json(output / 'evaluation/benchmark_provenance.json', {'filename': benchmark.name, 'sha256': digest(benchmark), 'task': task, 'evaluation_spec': clean(task_config['evaluation_tasks'][task]), 'provenance_note': 'GSM8K/IFEval local inputs were reconstructed from historical scored outputs; HumanEval is local test parquet.'})
                for row in selected_scores:
                    variant_dir = seed_root / 'hns_step_grid/eval' / task / row['variant']
                    for filename in ('predictions.jsonl', 'scored.jsonl'):
                        destination = output / 'evaluation/items' / row['variant'] / (filename + '.gz')
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with (variant_dir / filename).open('rb') as src, gzip.GzipFile(filename=str(destination), mode='wb', mtime=0) as dst:
                            for line in src:
                                dst.write((json.dumps(clean(json.loads(line)), ensure_ascii=False) + '\n').encode())
                manifest = seed_root / 'hns_step_grid/adapters' / base / task / 'manifest.json'
                write_json(output / 'hns/build_manifest.json', clean(json.loads(manifest.read_text())))
                grid = json.loads((ROOT / 'configs/hns_step_grid_2x4_20260912.json').read_text())
                write_json(output / 'hns/grid_config.json', clean(grid))
                task_adapters = seed_root / 'hns_step_grid/adapters' / base / task
                for path in task_adapters.glob('*/**/*.json'):
                    if path.name not in ('adapter_config.json', 'run_args.json', 'run_config.json', 'training_args.json', 'tokenizer_config.json', 'tokenizer.json'):
                        write_json(output / 'hns/build_metadata' / path.relative_to(task_adapters), clean(json.loads(path.read_text())))
                write_json(output / 'comparison/three_run_scores.json', [r for r in all_scores if r['model'] == base and r['task'] == task])
                shutil.copy2(ROOT / 'reports/hns_three_seed_diagonal_20260913.md', output / 'comparison/configuration_audit.md')
                code = output / 'code'
                code.mkdir(exist_ok=True)
                shutil.copy2(ROOT / 'scripts/reproduce_hns_seed_checkpoint.py', code / 'reproduce_hns_seed_checkpoint.py')
                snapshot_files = sorted((ROOT / 'src').rglob('*.py')) + [ROOT / 'scripts' / s for s in SCRIPTS] + [ROOT / 'pyproject.toml', ROOT / 'README.md']
                if (ROOT / 'LICENSE').is_file():
                    snapshot_files.append(ROOT / 'LICENSE')
                with tarfile.open(code / 'source_snapshot.tar.gz', 'w:gz') as archive:
                    for path in snapshot_files:
                        archive.add(path, arcname='source_snapshot/' + str(path.relative_to(ROOT)), recursive=False)
                write_json(code / 'source_files_sha256.json', {str(p.relative_to(ROOT)): digest(p) for p in snapshot_files})
                (output / 'README.md').write_text(model_card(meta, selected_scores, output))
                assert digest(output / 'adapter_model.safetensors') == weights_sha
                files = payload_files(output)
                for path in files:
                    if path.suffix in ('.json', '.md', '.txt', '.py', '.jinja'):
                        assert not re.search(r'hf_[A-Za-z0-9]{20,}', path.read_text()), path
                (output / 'MANIFEST.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(output)}\n' for p in files))
                plan.append({'repo_id': repo, 'folder': str(output), 'task_collection': COLLECTIONS[task], 'parent_collection': PARENT,
                             'weights_sha256': weights_sha, 'files': len(files) + 1, 'bytes': sum(p.stat().st_size for p in files),
                             'base': base, 'task': task, 'seed': seed})
                print(f'[Prepared] {repo}: {len(files) + 1} files', flush=True)
    write_json(stage_root / 'upload_plan.json', plan)
    return plan


def finalize(stage_root):
    """Quarantine redundant generated tokenizer copies, then refresh manifests."""
    plan = json.loads((stage_root / 'upload_plan.json').read_text())
    for row in plan:
        folder = Path(row['folder'])
        assert folder.parent == stage_root
        meta = json.loads((folder / 'publication.json').read_text())
        meta['peer_source_adapters'] = [{k: p[k] for k in ('repo_id', 'task', 'weights_sha256')} for p in plan if p['base'] == row['base'] and p['seed'] == row['seed']]
        meta['compilation_cache_note'] = ('Default torch.compile cache used in the earlier successful Slurm928 run; cache-disable fix applied later' if row['base'] == 'Llama-3.1-8B-Instruct' and row['seed'] == 43 else 'Disabled, per-job cache root, short IPC temp path')
        write_json(folder / 'publication.json', meta)
        results = json.loads((folder / 'evaluation/results.json').read_text())
        (folder / 'README.md').write_text(model_card(meta, results, folder))
        for path in folder.glob('hns/build_metadata/*/tokenizer.json'):
            destination = stage_root / '_redundant_metadata' / folder.name / path.relative_to(folder)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert digest(destination) == digest(path)
                raise RuntimeError('Quarantine collision; do not overwrite')
            shutil.move(str(path), str(destination))
        # Always ship the current portable helper; it is not part of the frozen training entrypoint.
        shutil.copy2(ROOT / 'scripts/reproduce_hns_seed_checkpoint.py', folder / 'code/reproduce_hns_seed_checkpoint.py')
        files = payload_files(folder)
        (folder / 'MANIFEST.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(folder)}\n' for p in files))
        row['files'] = len(files) + 1
        row['bytes'] = sum(p.stat().st_size for p in files)
    write_json(stage_root / 'upload_plan.json', plan)
    print(f'[Finalized] {len(plan)} payloads; {sum(r["bytes"] for r in plan)/1e9:.3f} GB total', flush=True)


def publish(stage_root):
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    assert api.whoami()['name'] == OWNER
    parent = api.get_collection(PARENT)
    assert parent.title == 'Spectral Surgery'
    for slug in COLLECTIONS.values():
        api.get_collection(slug)
    plan = json.loads((stage_root / 'upload_plan.json').read_text())
    receipt_path = ROOT / 'reports/hns_hf_upload_receipt_20260913.json'
    receipts = json.loads(receipt_path.read_text()) if receipt_path.is_file() else []
    completed = {r['repo_id']: r for r in receipts}
    # Complete all collision checks before the first external write.
    for row in plan:
        if api.repo_exists(row['repo_id'], repo_type='model'):
            info = api.model_info(row['repo_id'])
            if info.siblings:
                names = {s.rfilename for s in info.siblings}
                if 'publication.json' not in names:
                    raise RuntimeError(f'Refusing to overwrite unrelated repository: {row["repo_id"]}')
                existing = json.loads(Path(hf_hub_download(row['repo_id'], 'publication.json')).read_text())
                if existing.get('release') != 'hns-b300-training-seeds-20260913' or existing.get('adapter_weights_sha256') != row['weights_sha256']:
                    raise RuntimeError(f'Existing publication does not match: {row["repo_id"]}')
    for row in plan:
        repo = row['repo_id']
        if repo in completed:
            print(f'[Skip published] {repo}', flush=True)
            continue
        api.create_repo(repo, repo_type='model', private=False, exist_ok=True)
        print(f'[Upload] {repo}', flush=True)
        commit = api.upload_folder(repo_id=repo, repo_type='model', folder_path=row['folder'],
                                   commit_message=f'Release audited B300 LoRA training seed {row["seed"]}; actual zero warmup and complete in-domain HNS results')
        info = api.model_info(repo, revision=commit.oid, files_metadata=True)
        remote_files = {s.rfilename: s for s in info.siblings}
        expected = {'README.md', 'adapter_model.safetensors', 'adapter_config.json', 'publication.json', 'training_args.json', 'evaluation/results.json', 'MANIFEST.sha256'}
        assert expected <= remote_files.keys()
        remote_sha = remote_files['adapter_model.safetensors'].lfs.sha256
        assert remote_sha == row['weights_sha256'], (repo, remote_sha)
        note = f'B300 training seed{row["seed"]}; source LoRA; actual warmup0. Full configuration, in-domain HNS grid, hashes and reconstruction code included.'
        api.add_collection_item(row['task_collection'], repo, 'model', note=note, exists_ok=True)
        api.add_collection_item(row['parent_collection'], repo, 'model', note=note, exists_ok=True)
        receipt = {**{k: v for k, v in row.items() if k != 'folder'}, 'commit_oid': commit.oid, 'url': f'https://huggingface.co/{repo}',
                   'verified_remote_weights_sha256': remote_sha, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
        receipts.append(receipt)
        write_json(receipt_path, receipts)
        print(f'[Published and verified] {repo} commit={commit.oid}', flush=True)
    expected_repos = {r['repo_id'] for r in plan}
    actual = {i.item_id for i in api.get_collection(PARENT).items if i.item_type == 'model'}
    assert expected_repos <= actual
    for task, slug in COLLECTIONS.items():
        actual = {i.item_id for i in api.get_collection(slug).items if i.item_type == 'model'}
        assert {r['repo_id'] for r in plan if r['task'] == task} <= actual
    print(f'[Done] {len(plan)} repositories uploaded, weight hashes verified, all collection memberships verified.', flush=True)


def repair_cache(stage_root):
    """Remove only this release's generated code bytecode and repair its manifest."""
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    api = HfApi()
    assert api.whoami()['name'] == OWNER
    receipt_path = ROOT / 'reports/hns_hf_upload_receipt_20260913.json'
    receipts = json.loads(receipt_path.read_text())
    plan_path = stage_root / 'upload_plan.json'
    plan = json.loads(plan_path.read_text())
    for row in plan:
        folder = Path(row['folder'])
        entries = [line.split('  ', 1) for line in (folder / 'MANIFEST.sha256').read_text().splitlines()]
        caches = [name for _, name in entries if '__pycache__' in Path(name).parts or Path(name).suffix == '.pyc']
        if not caches:
            continue
        assert all(name.startswith('code/__pycache__/') and name.endswith('.pyc') for name in caches)
        receipt = next(r for r in receipts if r['repo_id'] == row['repo_id'])
        info = api.model_info(row['repo_id'])
        assert info.sha == receipt['commit_oid'], 'Remote main changed; do not overwrite'
        for name in caches:
            path = folder / name
            destination = stage_root / '_runtime_cache_quarantine' / folder.name / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert not destination.exists()
            shutil.move(str(path), str(destination))
        (folder / 'MANIFEST.sha256').write_text(''.join(f'{sha}  {name}\n' for sha, name in entries if name not in caches))
        operations = [CommitOperationDelete(path_in_repo=name) for name in caches]
        operations.append(CommitOperationAdd(path_in_repo='MANIFEST.sha256', path_or_fileobj=str(folder / 'MANIFEST.sha256')))
        commit = api.create_commit(repo_id=row['repo_id'], repo_type='model', operations=operations,
                                   parent_commit=receipt['commit_oid'], commit_message='Exclude generated reproduction-test bytecode; fix payload integrity manifest')
        receipt['artifact_cleanup_previous_commit_oid'] = receipt['commit_oid']
        receipt['commit_oid'] = commit.oid
        receipt['generated_cache_files_removed'] = caches
        receipt['artifact_cleanup_at_utc'] = datetime.now(timezone.utc).isoformat()
        row['files'] = len(entries) - len(caches) + 1
        row['bytes'] = sum((folder / name).stat().st_size for _, name in entries if name not in caches) + (folder / 'MANIFEST.sha256').stat().st_size
        receipt['files'], receipt['bytes'] = row['files'], row['bytes']
        write_json(receipt_path, receipts)
        write_json(plan_path, plan)
        print(f'[Repaired generated cache only] {row["repo_id"]}: {caches}', flush=True)


def verify(stage_root):
    """Read-only post-upload verification of every prepared payload file."""
    from huggingface_hub import HfApi
    api = HfApi()
    receipts = json.loads((ROOT / 'reports/hns_hf_upload_receipt_20260913.json').read_text())
    by_repo = {r['repo_id']: r for r in receipts}
    plan = json.loads((stage_root / 'upload_plan.json').read_text())
    assert len(by_repo) == len(plan) == 12
    checked = 0
    for row in plan:
        folder = Path(row['folder'])
        receipt = by_repo[row['repo_id']]
        info = api.model_info(row['repo_id'], revision=receipt['commit_oid'], files_metadata=True)
        remote = {s.rfilename: s for s in info.siblings}
        entries = [line.split('  ', 1) for line in (folder / 'MANIFEST.sha256').read_text().splitlines()]
        entries.append((digest(folder / 'MANIFEST.sha256'), 'MANIFEST.sha256'))
        for expected, name in entries:
            assert name in remote, (row['repo_id'], name)
            if remote[name].lfs:
                assert remote[name].lfs.sha256 == expected, (row['repo_id'], name)
            else:
                data = (folder / name).read_bytes()
                assert hashlib.sha256(data).hexdigest() == expected, (row['repo_id'], name)
                blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                assert remote[name].blob_id == blob, (row['repo_id'], name)
            checked += 1
        print(f'[Verified all files] {row["repo_id"]}', flush=True)
    for slug, expected in [(PARENT, {r['repo_id'] for r in plan})] + [(slug, {r['repo_id'] for r in plan if r['task'] == task}) for task, slug in COLLECTIONS.items()]:
        actual = {i.item_id for i in api.get_collection(slug).items if i.item_type == 'model'}
        assert expected <= actual
    write_json(ROOT / 'reports/hns_hf_upload_verification_20260913.json', {
        'status': 'complete', 'repositories': 12, 'payload_files_verified': checked,
        'collection_memberships_verified': True, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'Every LFS SHA256 and every non-LFS Git blob SHA1 matched the prepared manifest at the uploaded commit.'})
    print(f'[Done verification] {checked} payload files and all collection memberships match.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-root', required=True, type=Path)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--repair-cache', action='store_true')
    args = parser.parse_args()
    if args.repair_cache:
        repair_cache(args.stage_root)
    elif args.verify:
        verify(args.stage_root)
    elif args.publish:
        publish(args.stage_root)
    elif args.finalize:
        finalize(args.stage_root)
    else:
        prepare(args.stage_root)


if __name__ == '__main__':
    main()
