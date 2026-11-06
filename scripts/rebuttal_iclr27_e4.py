#!/usr/bin/env python3
"""Seven fixed Qwen trainings, validation-only selection, and 48 main cells."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

import rebuttal_iclr27 as r
from rebuttal_iclr27_e2_e3 import require_complete, save_adapter

OUT = r.OUT / 'E4'
BASE = 'Qwen3-8B'
MODEL = '/dataset1/zailong/models/Qwen3-8B'
LRS = dict(magicoder=2e-5, metamath=1e-4)
EPOCHS = dict(magicoder=1, metamath=3)
MULTIPLIERS = (.5, 1., 2.)
METHODS = ('base', 'original_lora', 'recon_0', 'hns_f4_s1', 'flat_nuclear')
CONDITIONS = ('magicoder_val_best_r16', 'metamath_val_best_r16', 'metamath_r64')


def norm(text):
    return ' '.join(str(text).split()).strip().casefold()


def first(row, fields):
    for key in fields:
        value = row.get(key)
        if value is not None and str(value).strip() and str(value) != 'nan':
            return str(value)
    return ''


def prepare():
    import pandas as pd
    from finetune.eval.eval_gsm8k import _extract_answer, _norm
    for exp in ('E1', 'E0', 'E2', 'E3'):
        require_complete(exp)
    OUT.mkdir(exist_ok=True)
    if (OUT / 'protocol.lock.json').exists():
        return r.check_lock('E4')
    data = OUT / 'runtime/data'
    data.mkdir(parents=True, exist_ok=True)
    manifests = {}
    # Use the existing deterministic 50K subset, then hold out entire question
    # groups so alternate MetaMath derivations cannot cross the split.
    for task in ('magicoder', 'metamath'):
        source = Path('/dataset1/zailong/data/peft-sft-lab') / (
            'magicoder-train.parquet' if task == 'magicoder' else 'metamathqa-train.parquet')
        df = pd.read_parquet(source)
        order = np.random.default_rng(42).permutation(len(df))[:50000]
        subset = df.iloc[order].copy()
        selected_ids = list(map(int, order))
        if task == 'metamath':
            records = subset.to_dict('records')
            groups = [norm(first(x, ('original_question', 'query', 'question'))) for x in records]
            held, held_groups, golds = [], set(), []
            for i in np.random.default_rng(42).permutation(len(records)):
                record = records[int(i)]
                response = first(record, ('response', 'answer', 'output', 'solution'))
                gold = _norm(_extract_answer(response))
                # The validation scorer is exact final numeric answer. This
                # eligibility rule uses training answers only, before training.
                if groups[int(i)] not in held_groups and re.fullmatch(r'-?\d+(?:\.\d+)?', gold):
                    held.append(int(i)); held_groups.add(groups[int(i)]); golds.append(gold)
                if len(held) == 500:
                    break
            assert len(held) == 500
            keep = [i for i, group in enumerate(groups) if group not in held_groups]
            train = subset.iloc[keep]
            val = pd.DataFrame([dict(question=first(records[i], ('query', 'question', 'original_question')),
                                     answer='#### ' + gold, source_row=selected_ids[i])
                                for i, gold in zip(held, golds)])
            assert not set(groups[i] for i in keep) & held_groups
            train_ids = [selected_ids[i] for i in keep]
        else:
            mbpp_source = Path('/dataset1/zailong/data/peft-sft-lab/mbpp-sanitized-test.parquet')
            mbpp = pd.read_parquet(mbpp_source)
            val_ids = np.random.default_rng(42).permutation(len(mbpp))[:200]
            assert len(val_ids) == 200
            val = mbpp.iloc[val_ids]
            val_prompts = {norm(first(x, ('text', 'prompt', 'instruction'))) for x in val.to_dict('records')}
            keep = [i for i, x in enumerate(subset.to_dict('records'))
                    if norm(first(x, ('instruction', 'prompt', 'question', 'problem'))) not in val_prompts]
            train = subset.iloc[keep]
            train_ids = [selected_ids[i] for i in keep]
        train_path, val_path = data / f'{task}_train.parquet', data / f'{task}_validation.parquet'
        train.to_parquet(train_path, index=False)
        val.to_parquet(val_path, index=False)
        manifests[task] = dict(source=str(source), source_sha256=r.sha(source), source_subset_seed=42,
            original_subset_rows=50000, train_rows=len(train), validation_rows=len(val),
            train_path=str(train_path), train_sha256=r.sha(train_path), validation_path=str(val_path),
            validation_sha256=r.sha(val_path), train_source_ids_sha256=__import__('hashlib').sha256(
                json.dumps(train_ids).encode()).hexdigest(), removed_rows=50000-len(train),
            validation_source='held-out MetaMathQA training groups' if task == 'metamath' else str(mbpp_source))
    protocol = OUT / 'protocol.md'
    protocol.write_text('''# E4: predeclared training and validation selection

Seven mandatory trainings only: Qwen3-8B, seed42, all7 LoRA modules, dropout0.05, scaling2. E4a magicoder/metaMath x LR multipliers0.5,1,2; r16 alpha32. Baseline LRs2e-5/1e-4, epochs1/3, as existing Qwen source recipes. Sequence4096, microbatch16, global batch32, bf16, gradient checkpointing, cosine, Adam betas0.9/0.999, weight decay0, grad clip1. Data subset seed42; model/Trainer seed42. Preserve epoch budgets. No test-set selection.
Explicit warmup_steps=ceil(0.10*total optimizer steps) after actual tokenization/filtering. First run preprocess_only to establish counts without loading model weights. Pass explicit steps to training; verify training_args.json and on-train-begin warmup_schedule_audit.json against the actual Trainer total. Save at ceil(total/2) and final (two checkpoints). All training/inference via Slurm; max2 GPUs across projects.
Validation:500 numeric-answer MetaMathQA training examples, distinct original-question groups, exclude every held-out group from training within the fixed original50K subset. Extract final numeric gold with the existing answer extractor before training; use existing GSM8K strict final-answer scorer on held-out questions. This is a validation dataset, never the GSM8K test set. MBPP sanitized test random200 seed42, scored by unchanged eval_mbpp.py pass@1 with non-thinking chat; exact matching prompts excluded from Magicoder subset. Datasets/IDs hashed before training.
Six candidates per task: LR0.5/1/2 times50%/100%. Select highest validation score. Ties: closest LR to1x in abs(log2 multiplier), then lower multiplier (0.5 before2), then earlier checkpoint. Every candidate score reported; never choose with HumanEval/GSM8K/IFEval/Commonsense test scores.
E4b: one MetaMath r64 alpha128 training with E4a-selected LR, same dataset/epoch schedule/warmup. Predeclare final checkpoint for r64; no extra search. Optional2x-final main evaluations and4B run omitted.
Build Recon-0, HNS4+1 strength1 nuclear-preserving and Flat-Nuclear with shared original fp32 compact SVD and balanced reconstruction; preserve alpha/r and dtype. HNS and Flat are data-free, no tuning. Audit SHA256, all-module coverage, finite spectra, saved relative-update/nuclear errors threshold5e-5; actual maxima reported. Output all source directions sigma_i/||sigma||2, per-module minimum, and fraction below0.005, especially r64.
Main evaluation uses exactly functional_hns_three_seed_20260914 prompts/templates/parsers/scorers: HumanEval164 pass@1, GSM8K1319 strict, IFEval541 prompt strict, Commonsense8 equal-subtask macro22419. Greedy seed42, batch invariant, maxmodel4096, block5, long seqs2048, CS4096, tokens65536, chunks4096/1024, FLASH_ATTN, no async/prefix cache. max_lora_rank16 for E4a; only E4b max_lora_rank64. Reuse final Qwen Base caches after sample-input SHA256 verification. Three conditions x4 adapters x4 families=48 new cells.
Target/Off/FG definitions unchanged: FG clips each off-family before averaging. Single seed; paired question-level bootstrap20000, seed20261106, within each atomic benchmark; Commonsense resampled separately per subtask and macro averaged. Multinomial resampling of joint correctness patterns is exactly the paired item bootstrap for binary correctness. Preserve pairing across Base/LoRA/all edits. Report McNemar exact binomial p-values, repaired and damaged counts for each binary benchmark/subtask; macro Off/FG have no binary McNemar test. Family and Off/FG intervals propagate the paired item resamples. No checkpoint bootstrap; do not claim training-run uncertainty. Unadjusted, descriptive single-seed item-level intervals. Report all conditions separately.
''')
    files = dict(r.check_lock('E1')['files'])
    for name in ('scripts/rebuttal_iclr27.py', 'scripts/rebuttal_iclr27_e2_e3.py', 'scripts/rebuttal_iclr27_e4.py',
                 'src/finetune/train_sft_peft.py', 'src/finetune/eval/eval_mbpp.py',
                 'src/finetune/data/registry.py', 'src/finetune/data/chat_sft.py', 'src/finetune/spectral_edit/flat.py'):
        files[name] = r.sha(r.ROOT / name)
    lock = dict(created_utc=r.now(), files=files, protocol_sha256=r.sha(protocol), datasets=manifests,
                expected_trainings=7, expected_new_cells=48, optional_runs=False)
    r.write(OUT / 'protocol.lock.json', lock)
    return lock


def env_setup():
    from run_para_spectral_surgery_baselines import env_setup as original
    original()
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    # Cache preprocessing in the writable experiment directory.
    os.environ['HF_DATASETS_CACHE'] = str(OUT / 'runtime/datasets_cache')


def training_argv(task, path, rank, lr, warmup, save_steps):
    data = r.check_lock('E4')['datasets'][task]
    assert r.sha(data['train_path']) == data['train_sha256']
    return [sys.executable, '-m', 'finetune.train_sft_peft', '--base_model', MODEL,
        '--task', task, '--dataset_path', data['train_path'], '--peft_method', 'lora', '--train_profile', 'none',
        '--output_dir', str(path), '--seed', '42', '--dataset_seed', '42', '--sft_format', 'chat',
        '--chat_template_mode', 'non_thinking', '--target_modules', 'all', '--r', str(rank),
        '--lora_alpha', str(2*rank), '--lora_dropout', '0.05', '--num_train_epochs', str(EPOCHS[task]),
        '--max_seq_len', '4096', '--per_device_train_batch_size', '16', '--global_train_batch_size', '32',
        '--pad_to_multiple_of', '8', '--lr', str(lr), '--warmup_steps', str(warmup),
        '--lr_scheduler_type', 'cosine', '--adam_beta1', '0.9', '--adam_beta2', '0.999',
        '--weight_decay', '0', '--grad_clip', '1', '--bf16', '--gradient_checkpointing',
        '--save_strategy', 'steps', '--save_steps', str(save_steps), '--save_total_limit', '2', '--logging_steps', '25']


def run_training(task, multiplier, rank=16):
    name = f'{task}_r{rank}_lr{multiplier:g}'
    path = OUT / 'runtime/training' / name
    preparation = OUT / 'runtime/preprocessing' / task
    audit_path = preparation / 'preprocessing_audit.json'
    if not audit_path.exists():
        cmd = training_argv(task, preparation, rank, LRS[task]*multiplier, 0, 1) + ['--preprocess_only']
        r.run_command(cmd, OUT / f'runtime/logs/{task}_preprocess.log', OUT / f'commands_{task}.json')
    total = int(r.read(audit_path)['total_optimizer_steps'])
    warmup = math.ceil(.1 * total)
    if not (path / 'adapter_model.safetensors').exists():
        r.run_command(training_argv(task, path, rank, LRS[task]*multiplier, warmup, math.ceil(total/2)),
                      OUT / f'runtime/logs/{name}_train.log', OUT / f'commands_{task}.json')
    args = r.read(path / 'training_args.json')
    actual = r.read(path / 'warmup_schedule_audit.json')
    assert args['warmup_steps'] == actual['warmup_steps'] == warmup
    assert actual['total_optimizer_steps'] == total
    checkpoints = sorted(path.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    assert len(checkpoints) == 2, (path, checkpoints)
    assert r.read(checkpoints[-1] / 'trainer_state.json')['global_step'] == total
    config = r.read(path / 'adapter_config.json')
    assert config['r'] == rank and config['lora_alpha'] == rank*2
    return path, checkpoints[0], dict(task=task, multiplier=multiplier, lr=LRS[task]*multiplier,
        rank=rank, total_steps=total, warmup_steps=warmup, final=str(path), half=str(checkpoints[0]),
        weights_sha256=r.sha(path / 'adapter_model.safetensors'), config_sha256=r.sha(path / 'adapter_config.json'),
        warmup_audit_sha256=r.sha(path / 'warmup_schedule_audit.json'), job_id=os.getenv('SLURM_JOB_ID'))


def validate(task, adapter, tag):
    data = r.check_lock('E4')['datasets'][task]
    assert r.sha(data['validation_path']) == data['validation_sha256']
    destination = OUT / 'runtime/validation' / task / tag
    metric_path = destination / 'metrics.json'
    if not metric_path.exists():
        cmd = [sys.executable, '-m', 'finetune.eval.eval_mbpp' if task == 'magicoder' else 'finetune.eval.eval_gsm8k',
               '--base_model', MODEL, '--adapter_dir', str(adapter), '--output_dir', str(destination),
               '--dataset_path', data['validation_path'], '--chat_template_mode', 'non_thinking',
               '--use_vllm', '--tensor_parallel_size', '1', '--vllm_max_model_len', '4096',
               '--vllm_attention_backend', 'FLASH_ATTN', '--vllm_disable_flashinfer_sampler', '--max_new_tokens', '512']
        if task == 'magicoder':
            cmd += ['--seed', '42', '--eval_n_workers', '16']
        r.run_command(cmd, OUT / f'runtime/logs/{task}_{tag}_validation.log', OUT / f'commands_{task}.json')
    metric = r.read(metric_path)
    assert metric['total'] == (200 if task == 'magicoder' else 500)
    return metric['pass@1' if task == 'magicoder' else 'accuracy_strict'], r.sha(metric_path)


def train_task(task):
    r.require_slurm(); env_setup()
    candidates, runs = [], []
    for multiplier in MULTIPLIERS:
        final, half, audit = run_training(task, multiplier)
        runs.append(audit)
        for fraction, path in ((.5, half), (1., final)):
            tag = f'lr{multiplier:g}_fraction{fraction:g}'
            score, metric_sha = validate(task, path, tag)
            candidates.append(dict(task=task, multiplier=multiplier, lr=LRS[task]*multiplier,
                fraction=fraction, path=str(path), validation_score=score, metric_sha256=metric_sha,
                weights_sha256=r.sha(path / 'adapter_model.safetensors'), config_sha256=r.sha(path / 'adapter_config.json')))
    best = choose_candidate(candidates)
    r.tsv(OUT / f'{task}_validation_candidates.tsv', candidates)
    r.write(OUT / f'{task}_training_audit.json', runs)
    r.write(OUT / f'{task}_selection.json', dict(selected=best, candidates=candidates, selection='validation only; predeclared tie rules'))


def choose_candidate(candidates):
    return min(candidates, key=lambda x: (-x['validation_score'], abs(math.log2(x['multiplier'])), x['multiplier'], x['fraction']))


def source_for(condition):
    task = condition.split('_')[0]
    if condition.endswith('r64'):
        return task, Path(r.read(OUT / 'rank64_training_audit.json')['final']), 64
    return task, Path(r.read(OUT / f'{task}_selection.json')['selected']['path']), 16


def build_condition(condition):
    r.require_slurm()
    from rebuttal_iclr27_e2_e3 import operations
    torch, collect, error, load, svd, rebuild, hns, Config = operations()
    from finetune.spectral_edit.flat import flat_spectrum
    task, source, rank = source_for(condition)
    state, fmt = load(str(source))
    pairs = collect(state)
    outputs = {m: dict(state) for m in METHODS[2:]}
    modules = []
    with torch.inference_mode():
        for name, pair in sorted(pairs.items()):
            ka, a = pair['A']; kb, b = pair['B']
            u, s, vh, _ = svd(b.cuda(), a.cuda())
            _, _, hs, _ = hns(u, vh, s, Config(fast_steps=4, stable_steps=1))
            # Strength0 follows the exact HNS spectrum and writeback functions.
            _, _, rs, _ = hns(u, vh, s, Config(fast_steps=0, stable_steps=0, hns_strength=0))
            spectra = dict(recon_0=rs, hns_f4_s1=hs, flat_nuclear=flat_spectrum(s, 'flat_nuclear'))
            for method, target in spectra.items():
                bt, at = rebuild(u, vh, target)
                bn, an = bt.to(b.dtype), at.to(a.dtype)
                outputs[method][ka], outputs[method][kb] = an.cpu(), bn.cpu()
                relative = error(bn.double(), an.double(), bt.double(), at.double())
                source_relative = error(bn.double(), an.double(), b.cuda().double(), a.cuda().double())
                _, actual, _, _ = svd(bn, an)
                nuclear = abs(float(actual.double().sum()/s.double().sum())-1)
                assert max(relative, nuclear) < 5e-5
                if method == 'recon_0':
                    assert source_relative < 5e-5
                modules.append(dict(condition=condition, method=method, module=name, rank=rank,
                    reconstruction_relative_error=relative, source_relative_error=source_relative,
                    nuclear_relative_error=nuclear, min_sigma_over_fro=float((s.min()/s.norm()).item()),
                    below_005=int((s/s.norm() < .005).sum()), total_directions=len(s),
                    sigma_over_fro=json.dumps((s/s.norm()).tolist())))
    variants = [dict(label=f'{task}__{condition}__original_lora', train_task=task,
                     method='original_lora', seed=42, path=str(source), condition=condition)]
    for method, output in outputs.items():
        path = OUT / 'runtime/adapters' / condition / method
        marker = path / 'rebuttal_integrity.json'
        if not marker.exists():
            save_adapter(str(source), path, output, fmt)
            # Verify serialized bytes rather than only tensors before save.
            loaded, _ = load(str(path))
            assert loaded.keys() == output.keys() and all(torch.equal(loaded[k], v) for k, v in output.items())
            assert r.sha(path / 'adapter_config.json') == r.sha(source / 'adapter_config.json')
            r.write(marker, dict(weights_sha256=r.sha(path / 'adapter_model.safetensors'),
                config_sha256=r.sha(path / 'adapter_config.json'), selected_modules=len(pairs),
                source_weights_sha256=r.sha(source / 'adapter_model.safetensors')))
        else:
            assert r.sha(path / 'adapter_model.safetensors') == r.read(marker)['weights_sha256']
        variants.append(dict(label=f'{task}__{condition}__{method}', train_task=task, method=method,
                             seed=42, path=str(path), condition=condition))
    r.tsv(OUT / f'{condition}_module_audit.tsv', modules)
    r.write(OUT / f'{condition}_variant_manifest.json', dict(base=BASE, base_model=MODEL,
        task_config=str(r.REF / f'{BASE}_task_config.json'), variants=variants))


def evaluate_condition(condition):
    r.require_slurm(); env_setup(); r.check_lock('E4')
    task, source, rank = source_for(condition)
    # CUDA construction exits before vLLM acquires an exclusive-process GPU.
    r.run_command([sys.executable, __file__, '--build', condition], OUT / f'runtime/logs/{condition}_build.log',
                  OUT / f'commands_{condition}.json')
    destination = OUT / 'runtime/eval' / condition
    for row in r.reference_cells():
        if row['base'] == BASE and row['method'] == 'base':
            link = destination / row['eval_task'] / 'base'
            link.parent.mkdir(parents=True, exist_ok=True)
            target = Path(row['metrics_path']).parent.resolve()
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
            assert link.resolve() == target
    for tasks, seqs in ((r.TASKS[:3], 2048), (r.TASKS[3:], 4096)):
        cmd = [sys.executable, str(r.ROOT / 'scripts/eval_forgetting_matrix_vllm.py'), '--base_model', MODEL,
            '--variant_manifest', str(OUT / f'{condition}_variant_manifest.json'),
            '--config', str(r.REF / f'{BASE}_task_config.json'), '--output_dir', str(destination), '--tasks', *tasks,
            '--max_model_len', '4096', '--gpu_memory_utilization', '0.94', '--max_num_seqs', str(seqs),
            '--max_num_batched_tokens', '65536', '--adapter_block_size', '5', '--max_lora_rank', str(rank),
            '--prompt_chunk_short', '4096', '--prompt_chunk_long', '1024', '--seed', '42']
        r.run_command(cmd, OUT / f'runtime/logs/{condition}_{tasks[0]}_generate.log', OUT / f'commands_{condition}.json')
        r.run_command([sys.executable, str(r.ROOT / 'scripts/score_forgetting_matrix.py'), '--matrix_dir', str(destination),
                       '--workers', '32'], OUT / f'runtime/logs/{condition}_{tasks[0]}_score.log', OUT / f'commands_{condition}.json')


def item_statistics(condition):
    task, _, _ = source_for(condition)
    variants = r.read(OUT / f'{condition}_variant_manifest.json')['variants']
    paths = {'base': 'base', **{v['method']: v['label'] for v in variants}}
    patterns = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(float)
    draws, points, bits_by_task = {}, {}, {}
    rng = np.random.default_rng(r.BOOT_SEED)
    for family in r.TASKS:
        records = []
        for method in METHODS:
            path = OUT / 'runtime/eval' / condition / family / paths[method] / 'scored.jsonl'
            records.append([json.loads(x) for x in path.read_text().splitlines()])
        ids = [[x['id'] for x in rows] for rows in records]
        assert all(x == ids[0] for x in ids)
        bits = np.array([[r.correctness(x, family) for x in rows] for rows in records]).T
        subgroups = sorted({x['subtask'] for x in records[0]}) if family == 'commonsense' else [family]
        family_draws, family_points = [], []
        for sub in subgroups:
            mask = np.array([x.get('subtask', family) == sub for x in records[0]])
            subbits = bits[mask]
            code = subbits @ (2 ** np.arange(5))
            counts = np.bincount(code, minlength=32)
            samples = rng.multinomial(len(code), counts/len(code), size=r.NBOOT)
            distribution = samples @ patterns / len(code) * 100
            point = subbits.mean(0) * 100
            draws[sub], points[sub], bits_by_task[sub] = distribution, point, subbits
            family_draws.append(distribution); family_points.append(point)
        draws[family], points[family] = np.mean(family_draws, axis=0), np.mean(family_points, axis=0)
    off = [f for f in r.TASKS if f != task]
    draws['Target'], points['Target'] = draws[task], points[task]
    draws['Off'], points['Off'] = np.mean([draws[f] for f in off], axis=0), np.mean([points[f] for f in off], axis=0)
    draws['FG'] = np.mean([np.maximum(draws[f][:, :1] - draws[f], 0) for f in off], axis=0)
    points['FG'] = np.mean([np.maximum(points[f][0] - points[f], 0) for f in off], axis=0)
    rows, paired = [], []
    for i, method in enumerate(METHODS):
        rows.append(dict(condition=condition, base=BASE, task=task, seed=42, method=method,
            **{f: float(points[f][i]) for f in r.TASKS}, target=float(points['Target'][i]),
            off_score=float(points['Off'][i]), forgetting_gap=float(points['FG'][i])))
    for i in range(1, 5):
        for j in range(i):
            for benchmark in draws:
                delta = draws[benchmark][:, i] - draws[benchmark][:, j]
                low, high = np.quantile(delta, [.025, .975])
                binary = bits_by_task.get(benchmark, bits_by_task.get(task) if benchmark == 'Target' else None)
                extra = {}
                if binary is not None:
                    repaired = int(((binary[:, i] == 1) & (binary[:, j] == 0)).sum())
                    damaged = int(((binary[:, i] == 0) & (binary[:, j] == 1)).sum())
                    extra = dict(repaired=repaired, damaged=damaged, discordant=repaired+damaged,
                        mcnemar_exact_p=mcnemar_exact(repaired, damaged))
                paired.append(dict(condition=condition, left=METHODS[i], right=METHODS[j], benchmark=benchmark,
                    difference_pp=float(points[benchmark][i]-points[benchmark][j]), ci_low_pp=float(low), ci_high_pp=float(high),
                    bootstrap_resamples=r.NBOOT, uncertainty='single seed; paired question-level interval; no training-run inference', **extra))
    return rows, paired


def mcnemar_exact(repaired, damaged):
    """Two-sided exact binomial McNemar test, integer CDF before rounding."""
    n = repaired + damaged
    k = min(repaired, damaged)
    coefficient = total = 1
    for i in range(1, k + 1):
        coefficient = coefficient * (n - i + 1) // i
        total += coefficient
    return min(1., (2 * total) / (1 << n))


def finalize():
    cells, artifacts, modules = [], [], []
    bases = {(x['base'], x['eval_task']): r.input_hashes(Path(x['metrics_path']).parent)
             for x in r.reference_cells() if x['method'] == 'base' and x['base'] == BASE}
    for condition in CONDITIONS:
        task, _, _ = source_for(condition)
        variants = r.read(OUT / f'{condition}_variant_manifest.json')['variants']
        for v in variants:
            path = Path(v['path'])
            artifacts.append({**v, 'condition': condition, 'weights_sha256': r.sha(path / 'adapter_model.safetensors'),
                              'config_sha256': r.sha(path / 'adapter_config.json')})
            for family in r.TASKS:
                cells.append(r.audit_cell(dict(base=BASE, train_task=task, seed=42, method=v['method'], condition=condition,
                    eval_task=family, metrics_path=str(OUT / 'runtime/eval' / condition / family / v['label'] / 'metrics.json')), bases))
        modules.extend(r.load_tsv(OUT / f'{condition}_module_audit.tsv'))
    assert len(cells) == 48
    rows, stats = [], []
    for condition in CONDITIONS:
        a, b = item_statistics(condition); rows.extend(a); stats.extend(b)
    r.tsv(OUT / 'evaluation_cells.tsv', cells)
    r.tsv(OUT / 'checkpoint_results.tsv', rows)
    r.tsv(OUT / 'paired_ci.tsv', stats)
    r.tsv(OUT / 'module_tail_audit.tsv', modules)
    tail = [m for m in modules if m['condition'] == 'metamath_r64' and m['method'] == 'recon_0']
    tail_summary = dict(modules=len(tail), directions=sum(int(m['total_directions']) for m in tail),
        below_005=sum(int(m['below_005']) for m in tail),
        min_sigma_over_fro_quantiles=np.quantile([float(m['min_sigma_over_fro']) for m in tail], [0,.25,.5,.75,1]).tolist())
    tail_summary['fraction_below_005'] = tail_summary['below_005']/tail_summary['directions']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    axes[0].hist([float(m['min_sigma_over_fro']) for m in tail], bins=40)
    axes[0].set(xlabel='Module min(sigma) / Frobenius norm', ylabel='Modules')
    ratios = list(itertools.chain.from_iterable(json.loads(m['sigma_over_fro']) for m in tail))
    axes[1].hist(ratios, bins=60)
    axes[1].axvline(.005, color='red', lw=1)
    axes[1].set(xlabel='sigma_i / Frobenius norm (rank64)', ylabel='Directions')
    fig.savefig(OUT / 'rank64_spectral_tail.png', dpi=200)
    fig.savefig(OUT / 'rank64_spectral_tail.pdf')
    plt.close(fig)
    r.write(OUT / 'coverage_audit.json', dict(status='pass', expected_new_cells=48, actual_new_cells=len(cells),
        conditions=CONDITIONS, expected_training_runs=7, sample_input_mismatches=0))
    r.write(OUT / 'artifact_integrity_audit.json', dict(status='pass', adapters=artifacts,
        max_reconstruction_relative_error=max(float(m['reconstruction_relative_error']) for m in modules),
        max_nuclear_relative_error=max(float(m['nuclear_relative_error']) for m in modules)))
    lines = ['# E4: validation-selected sources and rank64', '', 'Status: complete.', '',
        'Single training seed42. All intervals are paired question-level intervals, not checkpoint or training-run intervals. '
        'Six predeclared validation candidates per task; both full candidate tables are published. '
        'The r64 source uses the selected MetaMath LR and a predeclared final checkpoint. Optional runs omitted.', '',
        '| Condition | Method | Target | Off | FG |', '| --- | --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["condition"]} | {row["method"]} | {row["target"]:.4f} | {row["off_score"]:.4f} | {row["forgetting_gap"]:.4f} |')
    lines += ['', '| Condition | Comparison | Metric | Difference [95% CI] pp | Repairs/damage |',
              '| --- | --- | --- | ---: | ---: |']
    for row in stats:
        if row['left'] in ('hns_f4_s1', 'recon_0', 'flat_nuclear') and row['right'] in ('original_lora', 'recon_0') and row['benchmark'] in ('Target', 'Off', 'FG'):
            lines.append(f'| {row["condition"]} | {row["left"]} − {row["right"]} | {row["benchmark"]} | '
                f'{row["difference_pp"]:+.4f} [{row["ci_low_pp"]:+.4f}, {row["ci_high_pp"]:+.4f}] | '
                f'{row.get("repaired", "NA")}/{row.get("damaged", "NA")} |')
    lines += ['', '20,000 paired item bootstrap draws; CS subtask macro preserved. McNemar exact tests and '
        'repair/damage counts for every atomic binary benchmark/subtask are in paired_ci.tsv. Off/FG are macro statistics '
        'and have no binary McNemar test. All main-protocol inputs hash-match final Base. '
        'Warmup settings and actual Trainer totals are verified for all seven trainings; see training audits.', '',
        'Module spectra, minima and counts below0.005 are in module_tail_audit.tsv. '
        'Code/config/weights/dataset hashes and commands are archived. No test-set choice or IID training-run claim.', '',
        f'Slurm job {os.getenv("SLURM_JOB_ID")}; two GPUs allocated, at most two1-GPU workers at any time.']
    (OUT / 'report.md').write_text('\n'.join(lines)+'\n')
    summary = dict(status='complete', completed_utc=r.now(), job_id=os.getenv('SLURM_JOB_ID'), allocated_gpus=2,
        trainings=7, new_cells=48, seed=42, interval_unit='paired questions; single seed', conditions=CONDITIONS,
        rank64_tail=tail_summary,
        selection={task: r.read(OUT / f'{task}_selection.json')['selected'] for task in LRS},
        outputs_sha256={n: r.sha(OUT / n) for n in ('report.md', 'checkpoint_results.tsv', 'paired_ci.tsv')})
    r.write(OUT / 'summary.json', summary)


def parallel(commands, devices):
    states = []
    for argv, device in zip(commands, devices):
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = device
        log = OUT / 'runtime/logs' / f'{argv[-2]}_{argv[-1]}_worker.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open('a')
        p = subprocess.Popen(argv, cwd=r.ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        states.append((p, handle))
    codes = []
    for p, handle in states:
        codes.append(p.wait()); handle.close()
    assert not any(codes), ('E4 worker failure', codes)


def pipeline():
    r.require_slurm(); env_setup(); prepare()
    devices = os.getenv('CUDA_VISIBLE_DEVICES', '').split(',')
    assert len(devices) == 2
    parallel([[sys.executable, __file__, '--train-task', t] for t in LRS], devices)
    parallel([[sys.executable, __file__, '--evaluate', c] for c in CONDITIONS[:2]], devices)
    selected = r.read(OUT / 'metamath_selection.json')['selected']['multiplier']
    # Keep rank64 training in a child process; parent never acquires CUDA.
    parallel([[sys.executable, __file__, '--train-rank64', str(selected)]], devices[:1])
    parallel([[sys.executable, __file__, '--evaluate', CONDITIONS[2]]], devices[:1])
    finalize()
    from rebuttal_iclr27_finish import publish, writing_stages
    publish(['E4'])
    writing_stages()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline', action='store_true')
    p.add_argument('--train-task', choices=LRS)
    p.add_argument('--train-rank64', type=float)
    p.add_argument('--build', choices=CONDITIONS)
    p.add_argument('--evaluate', choices=CONDITIONS)
    args = p.parse_args()
    if args.pipeline:
        pipeline()
    elif args.train_task:
        train_task(args.train_task)
    elif args.train_rank64 is not None:
        r.require_slurm(); env_setup()
        _, _, audit = run_training('metamath', args.train_rank64, rank=64)
        r.write(OUT / 'rank64_training_audit.json', audit)
    elif args.build:
        build_condition(args.build)
    elif args.evaluate:
        evaluate_condition(args.evaluate)
