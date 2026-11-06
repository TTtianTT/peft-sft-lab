#!/usr/bin/env python3
"""Locked, cache-first ICLR27 rebuttal experiments. GPU work requires Slurm."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src', ROOT / 'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
OUT = ROOT / 'reports/rebuttal_iclr27'
REF = ROOT / 'reports/functional_hns_three_seed_20260914'
FLAT = ROOT / 'reports/posthoc_flat_dghard_forgetting_three_seed_20260913'
ENERGY = ROOT / 'reports/hns_energy_matched_20260914'
BASELINES = ROOT / 'reports/posthoc_baselines_para_spectral_surgery'
BASES = ('Qwen3-8B', 'Llama-3.1-8B-Instruct')
TASKS = ('magicoder', 'metamath', 'tulu', 'commonsense')
COUNTS = dict(zip(TASKS, (164, 1319, 541, 22419)))
METRICS = ('target', 'off_score', 'forgetting_gap')
METHODS = {'E1': ('recon_0',), 'E0': ('flat_nuclear', 'flat_fro')}
NBOOT, BOOT_SEED = 20000, 20261106
OUTPUT_FIELDS = {'prediction_text', 'token_ids', 'finish_reason'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def tsv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_tsv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def source_rows():
    return read(ROOT / 'reports/analysis_section5/source_checkpoints.json')


def reference_cells():
    return [r for r in load_tsv(REF / 'matrix.tsv')
            if r['method'] in ('base', 'original_lora', 'hns_f4_s1')]


def check_lock(experiment):
    folder = OUT / experiment
    lock = read(folder / 'protocol.lock.json')
    for path, digest in lock['files'].items():
        assert sha(ROOT / path) == digest, ('Locked file changed', path)
    assert sha(folder / 'protocol.md') == lock['protocol_sha256']
    return lock


def prepare():
    """CPU provenance and protocol lock, performed before any new evaluation."""
    OUT.mkdir(parents=True, exist_ok=True)
    source = source_rows()
    assert len(source) == 18
    assert {(c['base'], c['task'], c['seed']) for c in source} == {
        (b, t, s) for b in BASES for t in TASKS[:3] for s in (42, 43, 44)}
    assert read(REF / 'artifact_integrity_audit.json')['status'] == 'pass'
    code = read(REF / 'manifest.json')['code_sha256']
    code_actual = {p: sha(ROOT / p) for p in code}
    for p in code:
        if p != 'scripts/eval_forgetting_matrix_vllm.py':
            assert code[p] == code_actual[p], ('Main protocol code changed', p)
    # 139bfd7 added off_diagonal_only, an adapter scheduling selector. It did
    # not change prepare_task, prompts, template, greedy decoding or scoring.
    for c in source:
        assert sha(Path(c['source']) / 'adapter_model.safetensors') == c['source_sha256']
        assert sha(Path(c['source']) / 'adapter_config.json') == c['config_sha256']
    flat_adapters = [r for r in read(FLAT / 'manifest.json')['adapters']
                     if r['method'] in METHODS['E0']]
    assert len(flat_adapters) == 36
    for r in flat_adapters:
        assert sha(Path(r['path']) / 'adapter_model.safetensors') == r['weight_sha256']
        assert sha(Path(r['path']) / 'adapter_config.json') == r['config_sha256']
    files = dict(code_actual)
    for p in ['scripts/build_hns_step_grid_2x4.py',
              'src/finetune/spectral_edit/svd.py', 'src/finetune/spectral_edit/posthoc_hns.py',
              'src/finetune/spectral_edit/io.py', 'src/finetune/spectral_edit/mechanism.py',
              'reports/analysis_section5/source_checkpoints.json']:
        files[p] = sha(ROOT / p)
    for b in BASES:
        p = REF / f'{b}_task_config.json'
        files[str(p.relative_to(ROOT))] = sha(p)
    common = f'''Frozen before new evaluation: {now()}.

Fixed cohort: Qwen3-8B and Llama-3.1-8B-Instruct x magicoder/metamath/tulu x seeds 42/43/44 (18 checkpoints). No training.
Reference: functional_hns_three_seed_20260914, including final refreshed Llama caches.
HumanEval pass@1 (164), GSM8K strict (1319), IFEval prompt strict (541), Commonsense-8 equal-subtask macro (22419 items).
Unchanged prompts, chat template, parsers, scorers and datasets. Greedy seed42; batch invariant; max_model_len4096; adapter block5; max_lora_rank16; max_num_batched_tokens65536; FLASH_ATTN; async scheduling off; prefix caching off; gpu_memory_utilization0.94.
Long max_num_seqs2048, commonsense4096; prompt chunks short4096/long1024. No adaptive test-set selection.
Reuse Base, original_lora, hns_f4_s1 only after SHA256 and ordered per-sample input-hash checks. Reuse Scalar-E, Flat-E and all three predeclared PARA settings and Spectral Surgery grad_direction under the same checks. Run only new cells.
Target is the training task's benchmark family; Off is the equal mean of the other three families. FG = mean over those three families of max(Base-family score - edited-family score, 0), clipping each family before averaging. Commonsense is one macro family.
Report percentage points. Paired source-checkpoint bootstrap: {NBOOT} resamples, seed{BOOT_SEED}, percentile95% interval. Shared draws within group, wins/ties/losses (FG lower is better), tolerance1e-10 pp. Overall and model/task/model-task subgroups. Descriptive fixed-shared-test-set intervals, not IID training-run confidence intervals; unadjusted comparisons.
Keep all configurations. HNS4+1 strength1 nuclear-preserving all seven module types. Source rank/alpha/scaling and storage dtype unchanged.
Audit weights/config SHA256, selected/total module coverage, finite spectra, relative reconstruction error. Predeclared fp32 relative-update tolerance5e-5; report maxima without rounding away anomalies. No threshold change after seeing results.
Global maximum2 GPUs across projects; all construction using CUDA and all model evaluation through Slurm.
'''
    details = {
        'E1': '''Recon-0: call the existing build_hns_step_grid_2x4.build_checkpoint, apply_hns_to_svd and rebuild_ba_from_uv_sigma functions. fast_steps0, stable_steps0, hns_strength0 forces the original singular values exactly; preserve_nuclear_norm true. Identical fp32 compact QR/SVD and cast timing to the main HNS builder. All source scalings are2, so decomposing BA and retaining scaling is equivalent to editing DeltaW=sBA without changing scaling.
Implementation clarification: actual main HNS uses balanced B=U sqrt(sigma), A=sqrt(sigma) V^T. The proposal's unbalanced description does not match svd.py. Recon-0 follows actual main code. Optional alternative-factorization GPU runs are deferred; no claim of robustness from unrun controls.
72 new cells. Compare HNS, Flat-Nuclear, Scalar-E, Flat-E, PARA e90/e95/e99 and SS grad_direction against Recon-0, plus Recon-0 versus LoRA. Flat-Nuclear comparison depends on E0; interim E1 report remains awaiting_E0 until its main-wave cells exist. E1/E0 overlap is explicitly allowed in the request. No E2 until both have complete audited reports.
''',
        'E0': '''144 new cells: Flat-Nuclear and Flat-Fro, 18 checkpoints, four benchmark families. Reuse the exact36 previously built adapters referenced by the Slurm992 forgetting manifest; verify weight/config SHA256 before use. Never rebuild them.
Unified table: LoRA, Recon-0, HNS, Flat-Nuclear, Flat-Fro, Scalar-E, Flat-E. Pair every method with LoRA/HNS; include Flat-Nuclear minus Flat-Fro. No expectation-based conclusion: intervals overlapping zero do not prove equivalence.
Audit both waves per sample: ordered input SHA256, changed generated token sequence, changed prediction text, changed binary strict correctness; totals by checkpoint/family/method and subtask. Historical outputs are comparisons only, not current scores.
'''}
    for exp in ('E1', 'E0'):
        folder = OUT / exp
        folder.mkdir(exist_ok=True)
        if (folder / 'protocol.lock.json').exists():
            check_lock(exp)
            continue
        protocol = folder / 'protocol.md'
        protocol.write_text(f'# {exp}: predeclared protocol\n\n' + common + '\n' + details[exp])
        write(folder / 'protocol.lock.json', dict(created_utc=now(), protocol_sha256=sha(protocol),
            files=files, historical_evaluation_sha256=code, actual_evaluation_sha256=code_actual,
            evaluator_difference='139bfd7 scheduling-only off_diagonal_only selector',
            source_checkpoints=source, flat_adapters=flat_adapters, methods=METHODS[exp],
            expected_new_cells=72 * len(METHODS[exp]), bootstrap_resamples=NBOOT, bootstrap_seed=BOOT_SEED))
    print('Prepared E1/E0: 18 source hashes, 36 unchanged Flat adapters, frozen protocols.', flush=True)


def input_hashes(folder):
    """Hash every original input independently; preserve ordering and IDs."""
    result = []
    with (Path(folder) / 'predictions.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            content = {k: v for k, v in row.items() if k not in OUTPUT_FIELDS}
            result.append((str(row['id']), hashlib.sha256(json.dumps(content,
                sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()))
    assert len({r[0] for r in result}) == len(result), (folder, 'duplicate sample ID')
    return result


def correctness(row, task):
    key = {'magicoder': 'correct', 'metamath': 'correct_strict',
           'tulu': 'prompt_strict_passed', 'commonsense': 'correct'}[task]
    assert isinstance(row[key], (bool, int)) and row[key] in (0, 1)
    return int(row[key])


def audit_cell(row, bases, expected_sha=None):
    folder = Path(row['metrics_path']).parent
    assert (folder / 'COMPLETE').exists(), folder
    metric = read(folder / 'metrics.json')
    task = row['eval_task']
    assert metric['samples'] == COUNTS[task], (folder, metric['samples'])
    digests = {file.split('.')[0] + '_sha256': sha(folder / file)
               for file in ('metrics.json', 'predictions.jsonl', 'scored.jsonl')}
    if row.get('metrics_sha256'):
        assert digests['metrics_sha256'] == row['metrics_sha256'], folder
    for key, value in (expected_sha or {}).items():
        if key in digests:
            assert digests[key] == value, (folder, key)
    inputs = input_hashes(folder)
    assert len(inputs) == COUNTS[task]
    ref = bases[row['base'], task]
    mismatch = sum(a != b for a, b in zip(inputs, ref)) + abs(len(inputs) - len(ref))
    assert mismatch == 0, (folder, 'sample inputs differ', mismatch)
    ids, bits, sub = [], [], {}
    with (folder / 'scored.jsonl').open() as handle:
        for line in handle:
            r = json.loads(line)
            ids.append(str(r['id']))
            bit = correctness(r, task)
            bits.append(bit)
            if task == 'commonsense':
                sub.setdefault(r['subtask'], []).append(bit)
    assert ids == [x[0] for x in inputs]
    score = 100 * (np.mean([np.mean(v) for v in sub.values()]) if sub else np.mean(bits))
    assert not sub or len(sub) == 8
    assert abs(score - 100 * metric[metric['primary_metric']]) < 1e-8, (folder, 'score mismatch')
    return {**row, **digests, 'samples': len(inputs), 'score': float(score),
        'sample_input_mismatches': mismatch, 'sample_input_sha256': hashlib.sha256(
            json.dumps(inputs, separators=(',', ':')).encode()).hexdigest()}


def cache_cells(include_baselines=True):
    result = reference_cells()
    if include_baselines:
        for r in load_tsv(ENERGY / 'evaluation_matrix.tsv'):
            if r['method'] in ('scalar_e', 'flat_e'):
                result.append(r)
        for method in ('para', 'spectral_surgery'):
            for base in BASES:
                manifest = read(BASELINES / method / f'{base}_variant_manifest.json')
                for v in manifest['variants']:
                    for task in TASKS:
                        result.append(dict(base=base, train_task=v['train_task'], seed=v['seed'],
                            method=v['method'], eval_task=task, metrics_path=str(BASELINES / method /
                                'eval' / base / task / v['label'] / 'metrics.json')))
    return result


def audit_caches():
    """Run before GPU work; no reference cache is modified or regenerated."""
    bases = {(r['base'], r['eval_task']): input_hashes(Path(r['metrics_path']).parent)
             for r in reference_cells() if r['method'] == 'base'}
    cells = []
    wanted = cache_cells()
    for i, row in enumerate(wanted, 1):
        cells.append(audit_cell(row, bases))
        if i % 32 == 0 or i == len(wanted):
            print(f'[Cache audit] {i}/{len(wanted)} cells', flush=True)
    tsv(OUT / 'cache_input_audit.tsv', cells)
    with gzip.open(OUT / 'base_sample_input_hashes.tsv.gz', 'wt') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['base', 'eval_task', 'id', 'input_sha256'])
        for (base, task), items in bases.items():
            writer.writerows((base, task, ident, digest) for ident, digest in items)
    write(OUT / 'cache_audit.json', dict(status='pass', completed_utc=now(), cells=len(cells),
        sample_input_mismatches=0, cells_tsv_sha256=sha(OUT / 'cache_input_audit.tsv'),
        sample_hashes_sha256=sha(OUT / 'base_sample_input_hashes.tsv.gz')))
    print('Cache audit passed:', len(cells), 'cells; all per-sample inputs match final Base.', flush=True)


def require_slurm():
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('GPU construction/evaluation must be run inside a Slurm allocation')


def audit_reference_artifacts():
    rows = []
    for cp in source_rows():
        for method, path, weight, config in [
            ('original_lora', cp['source'], cp['source_sha256'], cp['config_sha256']),
            ('hns_f4_s1', cp['hns_path'], cp['hns_weights_sha256'], cp['config_sha256']),
            *[(m, cp[m]['path'], cp[m]['weights_sha256'], cp['config_sha256']) for m in ('scalar_e', 'flat_e')],
        ]:
            path = Path(path)
            assert sha(path / 'adapter_model.safetensors') == weight, (path, 'weights changed')
            assert sha(path / 'adapter_config.json') == config, (path, 'config changed')
            rows.append(dict(base=cp['base'], task=cp['task'], seed=cp['seed'], method=method,
                path=str(path), weights_sha256=weight, config_sha256=config))
    for base in BASES:
        for method in ('para', 'spectral_surgery'):
            for variant in read(BASELINES / method / f'{base}_variant_manifest.json')['variants']:
                path = Path(variant['path'])
                if method == 'para':
                    meta = read(path / 'para_meta.json')
                    weight, config = meta['weights_sha256'], meta['adapter_config_sha256']
                else:
                    meta = next(r for r in read(BASELINES / method / f'{base}_build_summary.json')
                                if r['task'] == variant['train_task'] and r['seed'] == variant['seed'])
                    weight = meta['weights_sha256']
                    config = next(c['config_sha256'] for c in source_rows() if c['base'] == base
                                  and c['task'] == variant['train_task'] and c['seed'] == variant['seed'])
                assert sha(path / 'adapter_model.safetensors') == weight
                assert sha(path / 'adapter_config.json') == config
                rows.append(dict(base=base, **variant, weights_sha256=weight, config_sha256=config))
    write(OUT / 'reference_artifact_audit.json', dict(status='pass', completed_utc=now(), artifacts=rows))
    print('[Reference artifact audit]', len(rows), 'adapters verified', flush=True)


def build_recon(base):
    require_slurm()
    import torch
    from build_hns_step_grid_2x4 import build_checkpoint, collect_pairs, factor_error
    from finetune.spectral_edit.io import load_lora_state_dict
    from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = OUT / 'E1'
    summaries = []
    for cp in check_lock('E1')['source_checkpoints']:
        if cp['base'] != base:
            continue
        root = folder / 'runtime/adapters' / base / f'seed{cp["seed"]}'
        start = time.monotonic()
        grid = dict(preserve_nuclear_norm=True, hns_strength=0.0)
        manifest = build_checkpoint(dict(base=base, train_task=cp['task'], lora=cp['source']),
                                    root, grid, [(0, 0)], 'cuda')
        path = Path(manifest['variants'][0]['path'])
        # Audit actual serialized factors against both original BA and the
        # independently reconstructed identity target. No dense d_out*d_in BA.
        original, _ = load_lora_state_dict(cp['source'])
        saved, _ = load_lora_state_dict(str(path))
        pairs, saved_pairs = collect_pairs(original), collect_pairs(saved)
        assert pairs.keys() == saved_pairs.keys() and len(pairs) == cp['module_count']
        modules = []
        with torch.inference_mode():
            for name, p in pairs.items():
                a, b = p['A'][1].cuda(), p['B'][1].cuda()
                an, bn = saved_pairs[name]['A'][1].cuda(), saved_pairs[name]['B'][1].cuda()
                u, s, vh, _ = lowrank_svd_from_ba(b, a)
                bt, at = rebuild_ba_from_uv_sigma(u, vh, s)
                source_error = factor_error(bn.double(), an.double(), b.double(), a.double())
                target_error = factor_error(bn.double(), an.double(), bt.double(), at.double())
                assert max(source_error, target_error) < 5e-5, (name, source_error, target_error)
                modules.append(dict(module=name, rank=len(s), dtype_A=str(an.dtype), dtype_B=str(bn.dtype),
                    source_relative_error=source_error, target_relative_error=target_error))
        assert sha(path / 'adapter_config.json') == cp['config_sha256']
        summary = dict(base=base, task=cp['task'], seed=cp['seed'], method='recon_0', path=str(path),
            source_sha256=cp['source_sha256'], config_sha256=sha(path / 'adapter_config.json'),
            weights_sha256=sha(path / 'adapter_model.safetensors'), modules=len(modules),
            max_source_relative_error=max(m['source_relative_error'] for m in modules),
            max_target_relative_error=max(m['target_relative_error'] for m in modules),
            build_and_audit_seconds=time.monotonic() - start, job_id=os.getenv('SLURM_JOB_ID'))
        write(path / 'rebuttal_integrity.json', dict(summary=summary, modules=modules))
        summaries.append(summary)
    write(folder / f'{base}_build_summary.json', summaries)
    return [dict(label=f'{r["task"]}__seed{r["seed"]}__recon_0', train_task=r['task'],
                 seed=r['seed'], method='recon_0', path=r['path']) for r in summaries]


def audit_flat_adapters(base):
    """Recompute original CPU builder targets; never modify saved Flat weights."""
    require_slurm()
    import torch
    from build_hns_step_grid_2x4 import collect_pairs, factor_error
    from finetune.spectral_edit.flat import flat_spectrum
    from finetune.spectral_edit.io import load_lora_state_dict
    from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma
    torch.set_num_threads(4)
    lock = check_lock('E0')
    rows = []
    for cp in lock['source_checkpoints']:
        if cp['base'] != base:
            continue
        state, _ = load_lora_state_dict(cp['source'])
        pairs = collect_pairs(state)
        decompositions = {n: lowrank_svd_from_ba(p['B'][1], p['A'][1]) for n, p in pairs.items()}
        for method in METHODS['E0']:
            variant = next(v for v in lock['flat_adapters'] if v['base'] == base
                           and v['train_task'] == cp['task'] and v['seed'] == cp['seed'] and v['method'] == method)
            path = Path(variant['path'])
            assert sha(path / 'adapter_model.safetensors') == variant['weight_sha256']
            assert sha(path / 'adapter_config.json') == variant['config_sha256']
            saved, _ = load_lora_state_dict(str(path))
            edited = collect_pairs(saved)
            assert edited.keys() == pairs.keys() and len(edited) == cp['module_count']
            for n, pair in edited.items():
                u, s, vh, _ = decompositions[n]
                target = flat_spectrum(s, method)
                bt, at = rebuild_ba_from_uv_sigma(u, vh, target)
                bn, an = pair['B'][1], pair['A'][1]
                relative = factor_error(bn.double(), an.double(), bt.double(), at.double())
                _, actual, _, _ = lowrank_svd_from_ba(bn, an)
                norm = actual.sum() if method == 'flat_nuclear' else actual.norm()
                expected = s.sum() if method == 'flat_nuclear' else s.norm()
                budget = abs(float(norm / expected) - 1)
                assert max(relative, budget) < 5e-5, (path, n, relative, budget)
                rows.append(dict(base=base, task=cp['task'], seed=cp['seed'], method=method, module=n,
                    reconstruction_relative_error=relative, budget_relative_error=budget))
            # Hash a second time to assert the audit did not rewrite adapters.
            assert sha(path / 'adapter_model.safetensors') == variant['weight_sha256']
    tsv(OUT / 'E0' / f'{base}_flat_module_audit.tsv', rows)
    write(OUT / 'E0' / f'{base}_flat_integrity_audit.json', dict(status='pass', modules=len(rows),
        max_reconstruction_relative_error=max(m['reconstruction_relative_error'] for m in rows),
        max_budget_relative_error=max(m['budget_relative_error'] for m in rows),
        builder='same CPU compact SVD/reconstruction as original Flat construction; saved weights unchanged'))


def run_command(argv, log, history):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    entries = read(history) if Path(history).exists() else []
    if log.exists():
        log = log.with_name(f'{log.stem}.job{os.getenv("SLURM_JOB_ID", "cpu")}.{len(entries)+1}{log.suffix}')
    entry = dict(argv=argv, log=str(log), started_utc=now(), job_id=os.getenv('SLURM_JOB_ID'))
    entries.append(entry)
    write(history, entries)
    print('[Run]', ' '.join(argv), 'log=', log, flush=True)
    start = time.monotonic()
    with log.open('w') as handle:
        result = subprocess.run(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    entry.update(returncode=result.returncode, finished_utc=now(), elapsed_seconds=time.monotonic() - start)
    write(history, entries)
    if result.returncode:
        raise RuntimeError(f'Command failed: {log}; exit={result.returncode}')


def worker(experiment, base, variant_builder=None):
    require_slurm()
    check_lock(experiment)
    audit = read(OUT / 'cache_audit.json')
    assert audit['status'] == 'pass' and sha(OUT / 'cache_input_audit.tsv') == audit['cells_tsv_sha256']
    os.environ.update(PYTHONPATH=f'{ROOT}/src:{ROOT}/scripts', HF_HOME='/dataset1/zailong/cache/peft-sft-lab/huggingface',
        HF_HUB_CACHE='/dataset1/zailong/cache/peft-sft-lab/huggingface/hub',
        HF_DATASETS_CACHE='/dataset1/zailong/cache/peft-sft-lab/datasets', HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_BATCH_INVARIANT='1', PYTHONHASHSEED='42',
        VLLM_DISABLE_COMPILE_CACHE='1', TORCH_CUDNN_SDPA_DEPRIORITIZED='1')
    temp = Path(tempfile.mkdtemp(prefix=f'r27-{os.getenv("SLURM_JOB_ID")}-', dir='/tmp'))
    os.environ.update(TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp), VLLM_CACHE_ROOT=str(temp / 'vllm'),
                      TORCHINDUCTOR_CACHE_DIR=str(temp / 'inductor'), TRITON_CACHE_DIR=str(temp / 'triton'))
    folder = OUT / experiment
    status = folder / f'{base}_worker_status.json'
    write(status, dict(phase='build', job_id=os.getenv('SLURM_JOB_ID'), gpu_count=1, updated_utc=now()))
    if variant_builder is not None:
        variants = variant_builder(base)
    elif experiment == 'E1':
        # GPU nodes use exclusive-process mode. A builder CUDA context in
        # this worker would prevent the vLLM EngineCore process from opening
        # the device. Isolate construction, then let that process exit.
        run_command([sys.executable, __file__, '--build-recon', '--base', base],
                    folder / f'runtime/logs/{base}_build.log', folder / f'commands_{base}.json')
        variants = [dict(label=f'{x["task"]}__seed{x["seed"]}__recon_0', train_task=x['task'],
                         seed=x['seed'], method='recon_0', path=x['path'])
                    for x in read(folder / f'{base}_build_summary.json')]
    else:
        run_command([sys.executable, __file__, '--audit-flat', '--base', base],
                    folder / f'runtime/logs/{base}_flat_audit.log', folder / f'commands_{base}.json')
        variants = []
        for r in check_lock('E0')['flat_adapters']:
            if r['base'] != base:
                continue
            assert sha(Path(r['path']) / 'adapter_model.safetensors') == r['weight_sha256']
            assert sha(Path(r['path']) / 'adapter_config.json') == r['config_sha256']
            variants.append({k: r[k] for k in ('label', 'train_task', 'seed', 'method', 'path')})
    manifest_path = folder / f'{base}_variant_manifest.json'
    old = read(REF / f'{base}_variant_manifest.json')
    write(manifest_path, dict(base=base, base_model=old['base_model'],
                             task_config=str(REF / f'{base}_task_config.json'), variants=variants))
    destination = folder / 'runtime/eval' / base
    for r in reference_cells():
        if r['base'] == base and r['method'] == 'base':
            link = destination / r['eval_task'] / 'base'
            link.parent.mkdir(parents=True, exist_ok=True)
            target = Path(r['metrics_path']).parent.resolve()
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
            assert link.resolve() == target
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for tasks, seqs in ((TASKS[:3], 2048), (TASKS[3:], 4096)):
        write(status, dict(phase='evaluate', tasks=tasks, job_id=os.getenv('SLURM_JOB_ID'),
                           gpu_count=1, updated_utc=now()))
        if not all((destination / t / v['label'] / 'COMPLETE').exists() for t in tasks for v in variants):
            argv = [sys.executable, str(ROOT / 'scripts/eval_forgetting_matrix_vllm.py'),
                '--base_model', old['base_model'], '--variant_manifest', str(manifest_path),
                '--config', str(REF / f'{base}_task_config.json'), '--output_dir', str(destination),
                '--tasks', *tasks, '--max_model_len', '4096', '--gpu_memory_utilization', '0.94',
                '--max_num_seqs', str(seqs), '--max_num_batched_tokens', '65536', '--adapter_block_size', '5',
                '--max_lora_rank', '16', '--prompt_chunk_short', '4096', '--prompt_chunk_long', '1024', '--seed', '42']
            run_command(argv, folder / f'runtime/logs/{base}_{tasks[0]}_generate.log', folder / f'commands_{base}.json')
        run_command([sys.executable, str(ROOT / 'scripts/score_forgetting_matrix.py'),
            '--matrix_dir', str(destination), '--workers', '32'],
            folder / f'runtime/logs/{base}_{tasks[0]}_score.log', folder / f'commands_{base}.json')
    write(status, dict(phase='complete', job_id=os.getenv('SLURM_JOB_ID'), gpu_count=1,
                      updated_utc=now(), new_cells=4 * len(variants)))


def new_cells(experiment):
    result = []
    for base in BASES:
        manifest = OUT / experiment / f'{base}_variant_manifest.json'
        if not manifest.exists():
            continue
        for v in read(manifest)['variants']:
            for task in TASKS:
                result.append(dict(base=base, train_task=v['train_task'], seed=v['seed'], method=v['method'],
                    eval_task=task, metrics_path=str(OUT / experiment / 'runtime/eval' / base / task /
                                                   v['label'] / 'metrics.json')))
    return result


def checkpoint_results(cells):
    base_scores = {(r['base'], r['eval_task']): r['score'] for r in cells if r['method'] == 'base'}
    index = {}
    for r in cells:
        if r['method'] != 'base':
            index.setdefault((r['base'], r['train_task'], int(r['seed']), r['method']), {})[r['eval_task']] = r['score']
    result = []
    for (base, task, seed, method), scores in sorted(index.items()):
        assert set(scores) == set(TASKS)
        off = [t for t in TASKS if t != task]
        result.append(dict(base=base, task=task, seed=seed, method=method,
            checkpoint=f'{base}/{task}/seed{seed}', **scores, target=scores[task],
            off_score=float(np.mean([scores[t] for t in off])),
            forgetting_gap=float(np.mean([max(base_scores[base, t] - scores[t], 0) for t in off]))))
    return result


def paired(rows):
    methods = sorted({r['method'] for r in rows})
    index = {(r['checkpoint'], r['method']): r for r in rows}
    groups = {'overall': lambda r: True}
    groups.update({f'model/{b}': lambda r, b=b: r['base'] == b for b in BASES})
    groups.update({f'task/{t}': lambda r, t=t: r['task'] == t for t in TASKS[:3]})
    groups.update({f'model_task/{b}/{t}': lambda r, b=b, t=t: r['base'] == b and r['task'] == t
                   for b in BASES for t in TASKS[:3]})
    result = []
    for group, select in groups.items():
        checkpoints = sorted({r['checkpoint'] for r in rows if select(r)})
        draws = np.random.default_rng(BOOT_SEED).integers(0, len(checkpoints), (NBOOT, len(checkpoints)))
        for left, right in itertools.permutations(methods, 2):
            for metric in METRICS:
                diff = np.array([index[cp, left][metric] - index[cp, right][metric] for cp in checkpoints])
                lo, hi = np.quantile(diff[draws].mean(1), [.025, .975])
                favorable = -diff if metric == 'forgetting_gap' else diff
                result.append(dict(group=group, left=left, right=right, metric=metric,
                    n_checkpoints=len(diff), difference_pp=float(diff.mean()), ci_low_pp=float(lo), ci_high_pp=float(hi),
                    wins=int((favorable > 1e-10).sum()), ties=int((abs(favorable) <= 1e-10).sum()),
                    losses=int((favorable < -1e-10).sum()), bootstrap_resamples=NBOOT, bootstrap_seed=BOOT_SEED,
                    uncertainty='descriptive paired checkpoint bootstrap; fixed shared test set; not IID training runs'))
    return result


def wave_audit():
    rows = []
    for cell in new_cells('E0'):
        folder = Path(cell['metrics_path']).parent
        old = FLAT / 'eval' / cell['base'] / cell['eval_task'] / folder.name
        a = input_hashes(old)
        b = input_hashes(folder)
        assert a == b
        totals = {}
        with (old / 'scored.jsonl').open() as ah, (folder / 'scored.jsonl').open() as bh:
            for aline, bline in itertools.zip_longest(ah, bh):
                assert aline is not None and bline is not None
                x, y = json.loads(aline), json.loads(bline)
                assert x['id'] == y['id']
                subtask = x.get('subtask', cell['eval_task'])
                for key in dict.fromkeys(('all', subtask)):
                    count = totals.setdefault(key, dict(samples=0, token_changes=0, text_changes=0, correctness_changes=0))
                    count['samples'] += 1
                    count['token_changes'] += x['token_ids'] != y['token_ids']
                    count['text_changes'] += x['prediction_text'] != y['prediction_text']
                    count['correctness_changes'] += correctness(x, cell['eval_task']) != correctness(y, cell['eval_task'])
        for subtask, counts in totals.items():
            rows.append(dict(base=cell['base'], task=cell['train_task'], seed=cell['seed'], method=cell['method'],
                eval_task=cell['eval_task'], subtask=subtask, input_mismatches=0, **counts,
                historical_predictions_sha256=sha(old / 'predictions.jsonl'),
                current_predictions_sha256=sha(folder / 'predictions.jsonl')))
    tsv(OUT / 'E0/wave_consistency_audit.tsv', rows)
    return {k: sum(r[k] for r in rows if r['subtask'] == 'all')
            for k in ('samples', 'token_changes', 'text_changes', 'correctness_changes')}


def summarize(experiment, allow_pending=False, completion_status='complete'):
    check_lock(experiment)
    audit = read(OUT / 'cache_audit.json')
    assert audit['status'] == 'pass' and sha(OUT / 'cache_input_audit.tsv') == audit['cells_tsv_sha256']
    cached = load_tsv(OUT / 'cache_input_audit.tsv')
    for r in cached:
        r['score'] = float(r['score'])
    bases = {(r['base'], r['eval_task']): input_hashes(Path(r['metrics_path']).parent)
             for r in cached if r['method'] == 'base'}
    additions = []
    experiments = (('E1',) if experiment == 'E1' and allow_pending else
                   ('E1', 'E0') if experiment in ('E1', 'E0') else ('E1', 'E0', experiment))
    for exp in experiments:
        cells = new_cells(exp)
        assert len(cells) == 72 * len(METHODS[exp]), (exp, len(cells))
        additions.extend(audit_cell(r, bases) for r in cells)
    folder = OUT / experiment
    tsv(folder / 'evaluation_cells.tsv', additions)
    rows = checkpoint_results(cached + additions)
    methods = ({'original_lora', 'hns_f4_s1', 'scalar_e', 'flat_e', 'recon_0', 'flat_nuclear', 'flat_fro'}
               if experiment == 'E0' else {r['method'] for r in rows})
    rows = [r for r in rows if r['method'] in methods]
    assert all(sum(r['method'] == m for r in rows) == 18 for m in methods if any(r['method'] == m for r in rows))
    stats = paired(rows)
    tsv(folder / 'checkpoint_results.tsv', rows)
    tsv(folder / 'paired_ci.tsv', stats)
    means = {m: {metric: float(np.mean([r[metric] for r in rows if r['method'] == m]))
                  for metric in METRICS} for m in sorted({r['method'] for r in rows})}
    pending = experiment == 'E1' and allow_pending
    summary = dict(status='awaiting_E0' if pending else completion_status, completed_utc=now(),
        expected_new_cells=72 * len(METHODS[experiment]), n_checkpoints=18, methods=means,
        bootstrap_resamples=NBOOT, bootstrap_unit='source checkpoint',
        uncertainty='descriptive intervals on a fixed shared test set; not IID training-run CIs',
        protocol_sha256=sha(folder / 'protocol.md'), sample_input_mismatches=0,
        job_ids=sorted({str(read(OUT / exp / f'{b}_worker_status.json')['job_id']) for exp in experiments for b in BASES}),
        gpu_count=2, pending=['main-wave Flat-Nuclear comparison from E0'] if pending else [])
    if experiment == 'E0':
        summary['wave_consistency'] = wave_audit()
    artifact_rows = []
    for exp in experiments:
        for base in BASES:
            for v in read(OUT / exp / f'{base}_variant_manifest.json')['variants']:
                path = Path(v['path'])
                artifact_rows.append(dict(experiment=exp, base=base, **v,
                    weights_sha256=sha(path / 'adapter_model.safetensors'),
                    config_sha256=sha(path / 'adapter_config.json')))
    build = list(itertools.chain.from_iterable(read(OUT / 'E1' / f'{b}_build_summary.json') for b in BASES))
    write(folder / 'coverage_audit.json', dict(status='pass', checkpoints=18, expected_methods=sorted(means),
        actual_rows=len(rows), expected_rows=18 * len(means),
        new_cells=sum(c['method'] in METHODS[experiment] for c in additions),
        audited_new_and_dependency_cells=len(additions), sample_input_mismatches=0))
    write(folder / 'artifact_integrity_audit.json', dict(status='pass', artifacts=artifact_rows,
        recon_modules=sum(r['modules'] for r in build),
        max_recon_source_relative_error=max(r['max_source_relative_error'] for r in build),
        max_recon_target_relative_error=max(r['max_target_relative_error'] for r in build)))
    lines = [f'# {experiment}: unified main-protocol results', '', f'Status: {summary["status"]}.', '',
        '18 fixed source checkpoints, no retraining. Scores are %; differences and FG are percentage points. '
        'Off averages the three non-target benchmark families equally; FG clips each family before averaging. '
        'Commonsense-8 uses equal subtask weights.', '',
        '| Method | Target | Off | FG |', '| --- | ---: | ---: | ---: |']
    lines += [f'| {m} | {v["target"]:.4f} | {v["off_score"]:.4f} | {v["forgetting_gap"]:.4f} |' for m, v in means.items()]
    lines += ['', '| Comparison | Metric | Difference [95% CI] pp | Win/tie/loss |',
              '| --- | --- | ---: | ---: |']
    baselines = ('original_lora', 'hns_f4_s1') if experiment == 'E0' else ('original_lora', 'recon_0')
    for r in stats:
        if r['group'] == 'overall' and r['right'] in baselines:
            lines.append(f'| {r["left"]} − {r["right"]} | {r["metric"]} | '
                f'{r["difference_pp"]:+.4f} [{r["ci_low_pp"]:+.4f}, {r["ci_high_pp"]:+.4f}] | '
                f'{r["wins"]}/{r["ties"]}/{r["losses"]} |')
    lines += ['', '20,000 paired checkpoint bootstrap draws; all comparisons and model/task subgroups in paired_ci.tsv. '
        'These are descriptive intervals on shared test sets, not IID training-run confidence intervals. '
        'A CI containing zero is inconclusive, not evidence of equivalence. No configuration was selected by test scores.', '',
        'Recon-0 calls the actual HNS builder with an identity spectrum (strength0), preserving its balanced factors, '
        'fp32 SVD and cast timing. The proposal described unbalanced factors; this report follows actual main-table code. '
        'All source scalings remain2. No optional alternate-factorization runs are claimed.', '',
        f'Maximum Recon-0 relative update error to source: {max(r["max_source_relative_error"] for r in build):.9g}; '
        f'to identity reconstruction: {max(r["max_target_relative_error"] for r in build):.9g}. '
        'All new and reused cells pass ordered per-sample input hash checks. Weight/config hashes and coverage are audited.', '',
        f'Slurm jobs: {", ".join(summary["job_ids"])}; two allocated GPUs total, one worker per model. '
        'Actual commands, times and output paths are in commands_*.json; source/config/code locks in protocol.lock.json.']
    if experiment == 'E1':
        for metric in METRICS:
            r = next(r for r in stats if r['group'] == 'overall' and r['left'] == 'hns_f4_s1'
                     and r['right'] == 'recon_0' and r['metric'] == metric)
            conclusion = ('higher' if r['ci_low_pp'] > 0 else 'lower' if r['ci_high_pp'] < 0 else 'inconclusive')
            lines += ['', f'HNS versus Recon-0, {metric}: {conclusion}; '
                f'{r["difference_pp"]:+.4f} [{r["ci_low_pp"]:+.4f}, {r["ci_high_pp"]:+.4f}] pp.']
    if pending:
        lines += ['', 'E1 evaluation and audit are done; Flat-Nuclear comparison awaits E0. This is an interim report, not completed E1.']
    if experiment == 'E0':
        lines += ['', 'Historical Slurm992 adapters were reused byte-for-byte. Wave-change counts: ' +
                  json.dumps(summary['wave_consistency']) + '. See wave_consistency_audit.tsv for checkpoint/subtask details.']
    (folder / 'report.md').write_text('\n'.join(lines) + '\n')
    write(folder / 'summary.json', summary)
    print(experiment, summary['status'], means, flush=True)


def pipeline():
    require_slurm()
    devices = os.getenv('CUDA_VISIBLE_DEVICES', '').split(',')
    assert len(devices) == 2, ('Exactly two Slurm-allocated GPUs required', devices)
    audit_reference_artifacts()
    for exp in ('E1', 'E0'):
        states = []
        for base, device in zip(BASES, devices):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = device
            env.pop('CUBLAS_WORKSPACE_CONFIG', None)
            log = OUT / exp / f'runtime/{base}_worker.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open('a')
            process = subprocess.Popen([sys.executable, __file__, '--worker', exp, '--base', base],
                                       cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            states.append((base, process, handle))
        failed = []
        for base, process, handle in states:
            code = process.wait()
            handle.close()
            if code:
                failed.append((base, code))
        if failed:
            raise RuntimeError(f'{exp} workers failed: {failed}; inspect runtime/*_worker.log')
        summarize(exp, allow_pending=(exp == 'E1'))
    summarize('E1')
    write(OUT / 'p0_e1_e0_complete.json', dict(status='complete', completed_utc=now(), job_id=os.getenv('SLURM_JOB_ID')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--audit-caches', action='store_true')
    p.add_argument('--build-recon', action='store_true')
    p.add_argument('--audit-flat', action='store_true')
    p.add_argument('--pipeline', action='store_true')
    p.add_argument('--worker', choices=METHODS)
    p.add_argument('--base', choices=BASES)
    p.add_argument('--summarize', choices=METHODS)
    args = p.parse_args()
    if args.prepare:
        prepare()
    if args.audit_caches:
        audit_caches()
    if args.build_recon:
        build_recon(args.base)
    if args.audit_flat:
        audit_flat_adapters(args.base)
    if args.pipeline:
        pipeline()
    if args.worker:
        worker(args.worker, args.base)
    if args.summarize:
        summarize(args.summarize)


if __name__ == '__main__':
    main()
