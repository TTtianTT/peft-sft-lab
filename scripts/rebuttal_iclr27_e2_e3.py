#!/usr/bin/env python3
"""Serial E2 then E3, gated on completed E1/E0 reports; unchanged evaluator."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

import rebuttal_iclr27 as r

r.METHODS.update(E2=('hns_od', 'ss_smooth_abs', 'ss_smooth_abs_all'),
                 E3=('mag_only', 'shape_only', 'global_gamma'))
ALL_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
CALIBRATION = {
    'magicoder': ('ise-uiuc/Magicoder-Evol-Instruct-110K', '/dataset1/zailong/data/peft-sft-lab/magicoder-train.parquet'),
    'metamath': ('meta-math/MetaMathQA', '/dataset1/zailong/data/peft-sft-lab/metamathqa-train.parquet'),
    'tulu': ('tulu_if', '/dataset1/zailong/data/peft-sft-lab/cross-task-mechanism/tulu-3-sft-personas-instruction-following-train.parquet'),
}


def require_complete(experiment):
    folder = r.OUT / experiment
    assert r.read(folder / 'summary.json')['status'] == 'complete', experiment
    for name in ('report.md', 'checkpoint_results.tsv', 'paired_ci.tsv'):
        assert (folder / name).stat().st_size > 0
    assert r.read(folder / 'coverage_audit.json')['status'] == 'pass'
    assert r.read(folder / 'artifact_integrity_audit.json')['status'] == 'pass'


def prepare(experiment):
    for prior in (('E1', 'E0') if experiment == 'E2' else ('E1', 'E0', 'E2')):
        require_complete(prior)
    folder = r.OUT / experiment
    folder.mkdir(exist_ok=True)
    if (folder / 'protocol.lock.json').exists():
        return r.check_lock(experiment)
    inherited = r.check_lock('E1')
    common = (r.OUT / 'E1/protocol.md').read_text().split('Recon-0:')[0]
    specific = {
        'E2': '''Three mandatory fixed variants; 216 new cells. HNS-od is HNS4+1 strength1 nuclear restoration, only o_proj/down_proj; all other source factors remain byte-identical. SS-smooth_abs uses o_proj/down_proj. SS-smooth_abs-all uses q/k/v/o/gate/up/down_proj. Optional grad_direction-all omitted, no optional results claimed.
Paper: https://arxiv.org/html/2603.03995v1 section3.4,4.3. Original author repository code: https://github.com/TTtianTT/peft-sft-lab/blob/139bfd7/src/finetune/spectral_edit/edit_strategies.py ; independently fetched through GitHub connector and SHA256-matched before evaluation (bffb4aaa044f27a9e38718a832c30b93e29247ec71f267561c98ceb5c3acf2b9).
Copy original defaults exactly: smooth_abs; core_frac0.2; noise_frac0.2; amp_factor1.25; sup_factor0.8; mid_factor1; min_core_k1; smooth_temperature0.35; smooth_center_q0.5; smooth_align_mid true; grad_norm mean_abs; preserve_energy l1; sigma_clip_min0. calib128; batch2; train split; shuffle seed42; start0; existing formatter, answer-only teacher forcing, float16 base. No calibration/test-based tuning.
Fair scope tables: {o,down}: HNS-od and SS-smooth_abs plus cached SS-grad_direction; all7: HNS and SS-smooth_abs-all. Also compare original LoRA and Recon-0. SS has 128 calibration examples and64 forward/backward batches per checkpoint/variant; record actual metadata, wall seconds, forward/back counts and GPU seconds. HNS has0 samples/forward/backward; synchronized measured compact SVD/edit/rebuild time, and load/save/audit overhead separately. Budgets are reported, not asserted equal.
For SS, audit source and edited compact reconstructions, nuclear conservation, module coverage and unchanged unselected factor bytes. Direction preservation checked via a joint QR residual against the source U/V subspaces. Predeclared tolerance5e-5 relative update/nuclear/direction error. No hyperparameter search.
''',
        'E3': '''216 new cells; no datasets used to build any variant. Same source SVD basis, same actual balanced factor writeback as E1/HNS. For each module let h be the main HNS4+1 nuclear-restored spectrum, a=||h||2/||sigma||2. Mag-only=sigma*a. Shape-only=h/a. Global-gamma=sigma*mean_module(a), one arithmetic mean per adapter. No post-cast energy correction, no nuclear restoration for these three controls. Hold rank, alpha and dtype fixed. All18 checkpoints/4 families; no test-set selection. All three fixed settings reported.
Factorial cells: original shape/original magnitude=Recon-0; original shape/HNS magnitude=Mag-only; HNS shape/original magnitude=Shape-only; HNS shape/HNS magnitude=HNS. Shape main effect=0.5*((HNS-Mag)+(Shape-Recon)); magnitude main effect=0.5*((HNS-Shape)+(Mag-Recon)); interaction=HNS-Mag-Shape+Recon. For Target,Off,FG compute each contrast per checkpoint then paired bootstrap20000, seed20261106; shared draws, descriptive fixed-test intervals; FG lower favorable.
Plot per-module alpha vs PR1=(||sigma||1/||sigma||2)^2 and exact-flat sqrt(PR1/r), by module family. Report true HNS a and deviations from strict-flat identity. Join Scalar-E coefficients from frozen energy_match metadata/module audit; pooled and within-checkpoint Spearman/Pearson (descriptive; modules correlated). Compare Mag-only vs Global-gamma with paired checkpoint CIs. Prior Scalar-F is mathematically related but is not silently substituted: this experiment uses these exact factors/cast timing without its storage energy correction.
'''}[experiment]
    protocol = folder / 'protocol.md'
    protocol.write_text(common.replace('# E1:', f'# {experiment}:') + '\n' + specific)
    files = dict(inherited['files'])
    for name in ('scripts/rebuttal_iclr27.py', 'scripts/rebuttal_iclr27_e2_e3.py',
                 'src/finetune/spectral_edit/cli.py', 'src/finetune/spectral_edit/edit_strategies.py',
                 'src/finetune/spectral_edit/hooks.py', 'src/finetune/spectral_edit/calib.py',
                 'src/finetune/spectral_edit/runtime.py', 'src/finetune/spectral_edit/module_selection.py'):
        if (r.ROOT / name).exists():
            files[name] = r.sha(r.ROOT / name)
    for cp in inherited['source_checkpoints']:
        metadata = Path(cp['hns_path']) / 'spectral_edit_meta.json'
        assert r.sha(metadata) == cp['hns_metadata_sha256']
        files[str(metadata)] = cp['hns_metadata_sha256']
    if experiment == 'E3':
        name = 'reports/hns_energy_matched_20260914/old_scalar_energy_audit.tsv'
        files[name] = r.sha(r.ROOT / name)
    lock = dict(created_utc=r.now(), protocol_sha256=r.sha(protocol), files=files,
                source_checkpoints=inherited['source_checkpoints'], methods=r.METHODS[experiment],
                expected_new_cells=216, prior_experiments_complete=True)
    if experiment == 'E2':
        assert r.sha(r.ROOT / 'src/finetune/spectral_edit/edit_strategies.py') == 'bffb4aaa044f27a9e38718a832c30b93e29247ec71f267561c98ceb5c3acf2b9'
        lock['calibration'] = {t: dict(name=name, path=path, sha256=r.sha(path))
                               for t, (name, path) in CALIBRATION.items()}
    r.write(folder / 'protocol.lock.json', lock)
    return lock


def save_adapter(source, destination, state, fmt):
    from finetune.spectral_edit.io import save_lora_state_dict
    destination.mkdir(parents=True, exist_ok=False)
    for item in Path(source).iterdir():
        if item.is_file() and (item.name in ('adapter_config.json', 'README.md') or
                              item.name.startswith(('tokenizer', 'special_tokens', 'vocab', 'merges', 'added_tokens'))):
            shutil.copy2(item, destination / item.name)
    save_lora_state_dict(str(destination), state, fmt)


def operations():
    import torch
    from build_hns_step_grid_2x4 import collect_pairs, factor_error
    from finetune.spectral_edit.io import load_lora_state_dict
    from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma
    from finetune.spectral_edit.posthoc_hns import apply_hns_to_svd, HNSEditConfig
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch, collect_pairs, factor_error, load_lora_state_dict, lowrank_svd_from_ba, rebuild_ba_from_uv_sigma, apply_hns_to_svd, HNSEditConfig


def spectral_edit_in_builder(argv, log, history):
    """Use the original CLI handler in this CUDA process (exclusive GPUs)."""
    from finetune.spectral_edit.cli import build_arg_parser, run_edit
    log.parent.mkdir(parents=True, exist_ok=True)
    entries = r.read(history) if history.exists() else []
    if log.exists():
        log = log.with_name(f'{log.stem}.job{os.getenv("SLURM_JOB_ID")}.{len(entries)+1}{log.suffix}')
    entry = dict(argv=argv, execution='same original CLI handler inside isolated builder process',
                 started_utc=r.now(), log=str(log), job_id=os.getenv('SLURM_JOB_ID'))
    entries.append(entry); r.write(history, entries)
    start = time.monotonic()
    try:
        with log.open('w') as handle, redirect_stdout(handle), redirect_stderr(handle):
            run_edit(build_arg_parser().parse_args(argv[3:]))
        entry['returncode'] = 0
    except BaseException as exc:
        entry.update(returncode=1, error=str(exc))
        raise
    finally:
        entry.update(finished_utc=r.now(), elapsed_seconds=time.monotonic()-start)
        r.write(history, entries)


def build(base, experiment):
    r.require_slurm()
    os.environ['HF_DATASETS_CACHE'] = str(r.OUT / experiment / 'runtime/datasets_cache')
    torch, collect, error, load, svd, rebuild, hns, Config = operations()
    folder = r.OUT / experiment
    variants, summaries, all_modules = [], [], []
    for cp in r.check_lock(experiment)['source_checkpoints']:
        if cp['base'] != base:
            continue
        start = time.monotonic()
        source, fmt = load(cp['source'])
        pairs = collect(source)
        assert len(pairs) == cp['module_count']
        hmeta = r.read(Path(cp['hns_path']) / 'spectral_edit_meta.json')['module_stats']
        decompositions = {}
        torch.cuda.synchronize()
        svd_start = time.monotonic()
        with torch.inference_mode():
            for name, p in sorted(pairs.items()):
                u, s, vh, _ = svd(p['B'][1].cuda(), p['A'][1].cuda())
                _, _, hs, _ = hns(u, vh, s, Config(fast_steps=4, stable_steps=1))
                expected = torch.tensor(hmeta[name]['sigma_after'], device='cuda', dtype=s.dtype)
                assert torch.allclose(hs, expected, rtol=5e-5, atol=1e-7), (name, 'main HNS spectrum mismatch')
                # Explicitly use the exact cached main HNS spectrum.
                hs = expected
                scale = float((hs.norm() / s.norm()).item())
                assert scale > 0
                decompositions[name] = (u, s, vh, hs, scale)
                # Time includes reconstruction of the all-module HNS target.
                rebuild(u, vh, hs)
        torch.cuda.synchronize()
        hns_all_seconds = time.monotonic() - svd_start
        hns_od_seconds = None
        if experiment == 'E2':
            torch.cuda.synchronize()
            od_start = time.monotonic()
            with torch.inference_mode():
                for name, p in sorted(pairs.items()):
                    if name.split('.')[-1] not in ('o_proj', 'down_proj'):
                        continue
                    u_od, s_od, vh_od, _ = svd(p['B'][1].cuda(), p['A'][1].cuda())
                    _, _, h_od, _ = hns(u_od, vh_od, s_od, Config(fast_steps=4, stable_steps=1))
                    rebuild(u_od, vh_od, h_od)
            torch.cuda.synchronize()
            hns_od_seconds = time.monotonic() - od_start
        gamma = float(np.mean([d[4] for d in decompositions.values()]))
        for method in r.METHODS[experiment]:
            path = folder / 'runtime/adapters' / base / cp['task'] / f'seed{cp["seed"]}' / method
            marker = path / 'rebuttal_integrity.json'
            if marker.exists():
                meta = r.read(marker)
                assert r.sha(path / 'adapter_model.safetensors') == meta['summary']['weights_sha256']
                summaries.append(meta['summary'])
                all_modules.extend(meta['modules'])
            else:
                method_start = time.monotonic()
                output = dict(source)
                modules = []
                if method.startswith('ss_'):
                    targets = ALL_MODULES if method.endswith('_all') else ('o_proj', 'down_proj')
                    name, data = CALIBRATION[cp['task']]
                    argv = [sys.executable, '-m', 'finetune.spectral_edit.cli', 'edit',
                        '--base_model', str(Path(cp['source']).parent), '--lora_path', cp['source'], '--out_dir', str(path),
                        '--mode', 'smooth_abs', '--target_modules', *targets, '--calib_samples', '128',
                        '--calib_batch_size', '2', '--calib_dataset', name, '--calib_dataset_path', data,
                        '--calib_split', 'train', '--calib_shuffle', '--calib_seed', '42', '--seed', '42',
                        '--core_frac', '0.2', '--noise_frac', '0.2', '--amp_factor', '1.25', '--sup_factor', '0.8',
                        '--mid_factor', '1', '--min_core_k', '1', '--smooth_temperature', '0.35', '--smooth_center_q', '0.5',
                        '--grad_norm', 'mean_abs', '--preserve_energy', 'l1', '--sigma_clip_min', '0']
                    argv[argv.index('--base_model') + 1] = r.read(r.REF / f'{base}_variant_manifest.json')['base_model']
                    if not (path / 'spectral_edit_meta.json').exists():
                        spectral_edit_in_builder(argv, folder / f'runtime/logs/{base}_{cp["task"]}_{cp["seed"]}_{method}_edit.log',
                                                 folder / f'build_commands_{base}.json')
                    output, _ = load(str(path))
                    selected = {n for n in pairs if n.split('.')[-1] in targets}
                    metadata = r.read(path / 'spectral_edit_meta.json')
                    assert set(metadata['sigma_stats']) == selected
                    assert metadata['meta']['calib_samples_used'] == 128
                    # Saved metadata records sums/top1; independently audit the
                    # actual update subspace and restored nuclear mass below.
                else:
                    selected = {n for n in pairs if experiment == 'E3' or n.split('.')[-1] in ('o_proj', 'down_proj')}
                    with torch.inference_mode():
                        for name in selected:
                            u, s, vh, hs, scale = decompositions[name]
                            target = {'mag_only': lambda: s * scale, 'shape_only': lambda: hs / scale,
                                      'global_gamma': lambda: s * gamma, 'hns_od': lambda: hs}[method]()
                            bn, an = rebuild(u, vh, target)
                            ka, a = pairs[name]['A']; kb, b = pairs[name]['B']
                            output[ka], output[kb] = an.to(a.dtype).cpu(), bn.to(b.dtype).cpu()
                    save_adapter(cp['source'], path, output, fmt)
                saved_pairs = collect(output)
                assert saved_pairs.keys() == pairs.keys()
                with torch.inference_mode():
                    for name, pair in pairs.items():
                        ka, a = pair['A']; kb, b = pair['B']
                        if name not in selected:
                            assert torch.equal(output[ka], a) and torch.equal(output[kb], b)
                            continue
                        u, s, vh, hs, scale = decompositions[name]
                        bn, an = output[kb].cuda(), output[ka].cuda()
                        # Project using finite-basis duals; the two residuals
                        # test learned directions without a degenerate new SVD.
                        ud, vd = u.double(), vh.double()
                        bd, ad = bn.double(), an.double()
                        left = ud @ torch.linalg.solve(ud.T @ ud, ud.T @ bd)
                        right = (ad @ vd.T) @ torch.linalg.solve(vd @ vd.T, vd)
                        direction_error = max(error(bd, ad, left, ad), error(bd, ad, bd, right))
                        _, actual_s, _, _ = svd(bn, an)
                        nuclear_error = abs(float(actual_s.double().sum() / s.double().sum()) - 1)
                        if method.startswith('ss_'):
                            # Joint QR reconstruction checks serialization and
                            # rank16 arithmetic separately from the shape edit.
                            un, sn, vn, _ = svd(bn, an)
                            bt, at = rebuild(un, vn, sn)
                        else:
                            target = {'mag_only': lambda: s * scale, 'shape_only': lambda: hs / scale,
                                      'global_gamma': lambda: s * gamma, 'hns_od': lambda: hs}[method]()
                            bt, at = rebuild(u, vh, target)
                        reconstruction_error = error(bd, ad, bt.double(), at.double())
                        assert max(direction_error, reconstruction_error) < 5e-5
                        if method.startswith('ss_') or method == 'hns_od':
                            assert nuclear_error < 5e-5
                        pr1 = float((s.double().sum() / s.double().norm()).square())
                        modules.append(dict(base=base, task=cp['task'], seed=cp['seed'], method=method, module=name,
                            family=name.split('.')[-1], rank=len(s), fro_ratio_hns=scale, global_gamma=gamma,
                            participation_ratio_l1=pr1, exact_flat_ratio=float(np.sqrt(pr1 / len(s))),
                            reconstruction_relative_error=reconstruction_error, direction_relative_error=direction_error,
                            nuclear_relative_error=nuclear_error,
                            min_sigma_over_fro=float((s.min() / s.norm()).item())))
                assert r.sha(path / 'adapter_config.json') == cp['config_sha256']
                summary = dict(base=base, task=cp['task'], seed=cp['seed'], method=method, path=str(path),
                    modules_selected=len(selected), modules_total=len(pairs),
                    weights_sha256=r.sha(path / 'adapter_model.safetensors'), config_sha256=cp['config_sha256'],
                    max_reconstruction_relative_error=max(m['reconstruction_relative_error'] for m in modules),
                    max_direction_relative_error=max(m['direction_relative_error'] for m in modules),
                    hns_all_svd_edit_rebuild_seconds=hns_all_seconds,
                    hns_od_svd_edit_rebuild_seconds=hns_od_seconds,
                    build_audit_save_seconds=time.monotonic() - method_start, source_load_seconds=svd_start-start,
                    calibration_samples=128 if method.startswith('ss_') else 0,
                    forward_batches=64 if method.startswith('ss_') else 0,
                    backward_batches=64 if method.startswith('ss_') else 0, job_id=os.getenv('SLURM_JOB_ID'))
                if method.startswith('ss_'):
                    summary.update(gradient_seconds=metadata['meta']['gradient_seconds'],
                                   edit_seconds=metadata['meta']['total_edit_seconds'],
                                   editor_one_gpu_wall_seconds=metadata['meta']['total_edit_seconds'],
                                   editor_timing_boundary='original CLI total_edit_seconds; includes model load/calibration/SVD/save')
                elif method == 'hns_od':
                    summary.update(editor_one_gpu_wall_seconds=hns_od_seconds,
                                   editor_timing_boundary='synchronized selected-module SVD/HNS/rebuild; excludes disk IO and audits')
                r.write(marker, dict(summary=summary, modules=modules))
                summaries.append(summary)
                all_modules.extend(modules)
            variants.append(dict(label=f'{cp["task"]}__seed{cp["seed"]}__{method}', train_task=cp['task'],
                                 seed=cp['seed'], method=method, path=str(path)))
        print('[Built]', experiment, base, cp['task'], cp['seed'], flush=True)
    r.write(folder / f'{base}_build_summary.json', summaries)
    r.tsv(folder / f'{base}_module_audit.tsv', all_modules)
    return variants


def finalize(experiment):
    r.summarize(experiment, completion_status='validating_additional_deliverables')
    folder = r.OUT / experiment
    build_rows = list(itertools.chain.from_iterable(r.read(folder / f'{b}_build_summary.json') for b in r.BASES))
    module_rows = list(itertools.chain.from_iterable(r.load_tsv(folder / f'{b}_module_audit.tsv') for b in r.BASES))
    r.tsv(folder / 'budget.tsv', build_rows)
    if experiment == 'E2':
        r.tsv(folder / 'hns_runtime.tsv', [dict(base=x['base'], task=x['task'], seed=x['seed'],
            all_module_seconds=x['hns_all_svd_edit_rebuild_seconds'],
            od_module_seconds=x['hns_od_svd_edit_rebuild_seconds'], calibration_samples=0,
            forward_batches=0, backward_batches=0, timing='one GPU synchronized SVD/HNS/rebuild; excludes IO/audits')
            for x in build_rows if x['method'] == 'hns_od'])
    integrity = r.read(folder / 'artifact_integrity_audit.json')
    actual = {(x['base'], x['train_task'], int(x['seed']), x['method']): x
              for x in integrity['artifacts'] if x['experiment'] == experiment}
    for expected in build_rows:
        saved = actual[expected['base'], expected['task'], int(expected['seed']), expected['method']]
        assert saved['weights_sha256'] == expected['weights_sha256']
        assert saved['config_sha256'] == expected['config_sha256']
    integrity.update(edited_adapters=len(build_rows), selected_module_rows=len(module_rows),
                     max_reconstruction_relative_error=max(float(m['reconstruction_relative_error']) for m in module_rows),
                     max_direction_relative_error=max(float(m['direction_relative_error']) for m in module_rows))
    r.write(folder / 'artifact_integrity_audit.json', integrity)
    rows = r.load_tsv(folder / 'checkpoint_results.tsv')
    stats = r.load_tsv(folder / 'paired_ci.tsv')
    with (folder / 'report.md').open('a') as handle:
        if experiment == 'E2':
            for scope, methods in [('o_proj/down_proj', ('hns_od', 'ss_smooth_abs', 'spectral_surgery_grad_direction')),
                                   ('all seven', ('hns_f4_s1', 'ss_smooth_abs_all'))]:
                handle.write(f'\nModule-matched comparison: {scope}.\n\n| Method | Target | Off | FG |\n| --- | ---: | ---: | ---: |\n')
                for m in methods:
                    scores = [np.mean([float(x[k]) for x in rows if x['method'] == m]) for k in r.METRICS]
                    handle.write(f'| {m} | ' + ' | '.join(f'{v:.4f}' for v in scores) + ' |\n')
            handle.write('\nBudget: SS uses128 examples and64 forward/backward batches per checkpoint/config. '
                'HNS uses0 data and0 forward/backward passes. Timings in budget.tsv distinguish synchronized '
                'all-module SVD/edit/rebuild time, SS gradient/edit time, and build/save/audit overhead. '
                'Scope and measurement boundaries differ; no claim of equal compute budgets.\n')
            hns_rows = [x for x in build_rows if x['method'] == 'hns_od']
            handle.write('\n| Editor timing | Mean seconds |\n| --- | ---: |\n')
            for label, values in [
                ('HNS all7: SVD/edit/rebuild', [x['hns_all_svd_edit_rebuild_seconds'] for x in hns_rows]),
                ('HNS o/down: SVD/edit/rebuild', [x['hns_od_svd_edit_rebuild_seconds'] for x in hns_rows]),
                *[(m + ': original total_edit_seconds including load/calibration/save',
                   [x['edit_seconds'] for x in build_rows if x['method'] == m])
                  for m in ('ss_smooth_abs', 'ss_smooth_abs_all')]]:
                handle.write(f'| {label} | {np.mean(values):.4f} |\n')
            handle.write('\nDeduplicated per-checkpoint HNS timings are in hns_runtime.tsv; all individual SS timings and overheads are in budget.tsv.\n')
        else:
            effects, contrasts = [], {
                'shape_main': dict(hns_f4_s1=.5, mag_only=-.5, shape_only=.5, recon_0=-.5),
                'magnitude_main': dict(hns_f4_s1=.5, shape_only=-.5, mag_only=.5, recon_0=-.5),
                'interaction': dict(hns_f4_s1=1, mag_only=-1, shape_only=-1, recon_0=1)}
            checkpoints = sorted({x['checkpoint'] for x in rows})
            ix = {(x['checkpoint'], x['method']): x for x in rows}
            draw = np.random.default_rng(r.BOOT_SEED).integers(0, len(checkpoints), (r.NBOOT, len(checkpoints)))
            for name, weights in contrasts.items():
                for metric in r.METRICS:
                    diff = np.array([sum(w * float(ix[cp, m][metric]) for m, w in weights.items()) for cp in checkpoints])
                    lo, hi = np.quantile(diff[draw].mean(1), [.025, .975])
                    favorable = -diff if metric == 'forgetting_gap' else diff
                    effects.append(dict(effect=name, metric=metric, mean_pp=float(diff.mean()), ci_low_pp=float(lo),
                        ci_high_pp=float(hi), wins=int((favorable > 1e-10).sum()), ties=int((abs(favorable) <= 1e-10).sum()),
                        losses=int((favorable < -1e-10).sum()), bootstrap_resamples=r.NBOOT, unit='checkpoint'))
            r.tsv(folder / 'factorial_effects.tsv', effects)
            handle.write('\nFactorial main effects and interaction, pp (descriptive checkpoint intervals):\n\n'
                         '| Effect | Metric | Mean [95% CI] |\n| --- | --- | ---: |\n')
            for e in effects:
                handle.write(f'| {e["effect"]} | {e["metric"]} | {e["mean_pp"]:+.4f} [{e["ci_low_pp"]:+.4f}, {e["ci_high_pp"]:+.4f}] |\n')
            handle.write('\nNo independent spectral-shape benefit is claimed unless the relevant shape-effect interval excludes0.\n')
    if experiment == 'E3':
        plot_relationship(module_rows)
    report = folder / 'report.md'
    report.write_text(report.read_text().replace('Status: validating_additional_deliverables.', 'Status: complete.', 1))
    summary = r.read(folder / 'summary.json')
    summary['status'] = 'complete'
    summary['outputs_sha256'] = {name: r.sha(folder / name) for name in
        ('report.md', 'checkpoint_results.tsv', 'paired_ci.tsv', 'artifact_integrity_audit.json', 'coverage_audit.json')}
    r.write(folder / 'summary.json', summary)


def isolated_build(base, experiment):
    folder = r.OUT / experiment
    r.run_command([sys.executable, __file__, '--build', experiment, '--base', base],
                  folder / f'runtime/logs/{base}_build.log', folder / f'commands_{base}.json')
    return r.read(folder / f'{base}_built_variants.json')


def plot_relationship(module_rows):
    os.environ['MPLCONFIGDIR'] = str(r.OUT / 'E3/runtime/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modules = [m for m in module_rows if m['method'] == 'mag_only']
    old = {(x['base'], x['task'], int(x['seed']), x['module']): float(x['energy_matching_gain'])
           for x in r.load_tsv(r.ENERGY / 'old_scalar_energy_audit.tsv')}
    joined = []
    for m in modules:
        x = dict(m)
        x['scalar_e_gain'] = old[m['base'], m['task'], int(m['seed']), m['module']]
        joined.append(x)
    folder = r.OUT / 'E3'
    r.tsv(folder / 'module_scaling_relationship.tsv', joined)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for family in ALL_MODULES:
        ms = [m for m in joined if m['family'] == family]
        axes[0].scatter([float(m['participation_ratio_l1']) / int(m['rank']) for m in ms],
                        [float(m['fro_ratio_hns']) for m in ms], s=3, alpha=.25, label=family)
        axes[1].scatter([m['scalar_e_gain'] for m in ms], [float(m['fro_ratio_hns']) for m in ms], s=3, alpha=.25)
    x = np.linspace(0, 1, 200)
    axes[0].plot(x, np.sqrt(x), color='black', lw=1, label='exact-flat sqrt(PR1/r)')
    axes[0].set(xlabel='PR1 / rank', ylabel='HNS Frobenius ratio')
    axes[1].set(xlabel='Scalar-E gain (calibration data)', ylabel='HNS Frobenius ratio (no data)')
    axes[0].legend(fontsize=6)
    fig.savefig(folder / 'module_scaling_relationship.png', dpi=200)
    fig.savefig(folder / 'module_scaling_relationship.pdf')
    plt.close(fig)
    groups = {'pooled': joined}
    for m in joined:
        groups.setdefault(f'{m["base"]}/{m["task"]}/seed{m["seed"]}', []).append(m)
    correlations = []
    for name, ms in groups.items():
        x = [m['scalar_e_gain'] for m in ms]
        y = [float(m['fro_ratio_hns']) for m in ms]
        correlations.append(dict(group=name, modules=len(ms), spearman=correlation(average_ranks(x), average_ranks(y)),
            pearson=correlation(x, y), uncertainty='descriptive; modules are correlated; no IID module inference'))
    r.tsv(folder / 'scalar_e_correlations.tsv', correlations)


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    ranked = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        end = i + 1
        while end < len(order) and values[order[end]] == values[order[i]]:
            end += 1
        ranked[order[i:end]] = (i + end - 1) / 2 + 1
        i = end
    return ranked


def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pipeline():
    r.require_slurm()
    devices = os.getenv('CUDA_VISIBLE_DEVICES', '').split(',')
    assert len(devices) == 2
    from rebuttal_iclr27_finish import refresh_p0, publish
    refresh_p0()
    publish(['E1', 'E0'])
    for experiment in ('E2', 'E3'):
        stage_started = time.monotonic()
        prepare(experiment)
        states = []
        for base, device in zip(r.BASES, devices):
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = device
            env.pop('CUBLAS_WORKSPACE_CONFIG', None)
            log = r.OUT / experiment / f'runtime/{base}_worker.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open('a')
            process = subprocess.Popen([sys.executable, __file__, '--worker', experiment, '--base', base],
                cwd=r.ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            states.append((base, process, handle))
        failures = []
        for base, process, handle in states:
            code = process.wait(); handle.close()
            if code:
                failures.append((base, code))
        assert not failures, (experiment, failures)
        finalize(experiment)
        summary = r.read(r.OUT / experiment / 'summary.json')
        elapsed = time.monotonic() - stage_started
        summary['resource_usage'] = dict(allocated_gpus=2, stage_wall_seconds=elapsed,
            allocated_gpu_seconds=2*elapsed, boundary='entire stage including construction/evaluation/CPU audits; not kernel-active time')
        r.write(r.OUT / experiment / 'summary.json', summary)
        publish([experiment])
    r.write(r.OUT / 'e2_e3_complete.json', dict(status='complete', completed_utc=r.now(), job_id=os.getenv('SLURM_JOB_ID')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline', action='store_true')
    p.add_argument('--worker', choices=('E2', 'E3'))
    p.add_argument('--build', choices=('E2', 'E3'))
    p.add_argument('--base', choices=r.BASES)
    args = p.parse_args()
    if args.pipeline:
        pipeline()
    elif args.worker:
        r.worker(args.worker, args.base, variant_builder=lambda base: isolated_build(base, args.worker))
    elif args.build:
        variants = build(args.base, args.build)
        r.write(r.OUT / args.build / f'{args.base}_built_variants.json', variants)
