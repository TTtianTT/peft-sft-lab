#!/usr/bin/env python3
"""Fixed 18-checkpoint, modulewise raw-functional-energy controls; no training.

Stages are deliberately serial: Scalar-E (both bases), Flat-E (both bases),
fixed-metric analysis, sample transitions, existing functional boundary results.
All inference/scoring is delegated to the unchanged main experiment scripts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch

from finetune.spectral_edit.io import load_lora_state_dict, save_lora_state_dict
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma
from build_hns_step_grid_2x4 import collect_pairs, factor_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/hns_energy_matched_20260914'
LATEST = ROOT / 'reports/functional_hns_three_seed_20260914'
ACT = ROOT / 'reports/functional_activation_three_seed_20260914'
SOURCE = ROOT / 'reports/posthoc_flat_dghard_three_seed_20260913/source_manifest.json'
BASES = ('Qwen3-8B', 'Llama-3.1-8B-Instruct')
TASKS = ('magicoder', 'metamath', 'tulu', 'commonsense')
COUNTS = dict(zip(TASKS, (164, 1319, 541, 22419)))
CONTROLS = ('scalar_e', 'flat_e')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tsv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(buf.getvalue())
    tmp.replace(path)


def energy_match(s, hns, q):
    """Raw q, per module, no floor, no nuclear renormalization."""
    s, hns, q = (np.asarray(x, dtype=np.float64) for x in (s, hns, q))
    if s.shape != hns.shape or s.shape != q.shape:
        raise ValueError('Spectrum/q shape mismatch')
    if not all(np.isfinite(x).all() for x in (s, hns, q)):
        raise ValueError('Nonfinite spectrum/q')
    if any((x < 0).any() for x in (s, hns, q)):
        raise ValueError('Negative spectrum/q')
    numerator, denominator = np.dot(hns*hns, q), np.dot(s*s, q)
    if denominator == 0:
        if numerator != 0:
            raise ValueError('Cannot match nonzero reference from zero-energy spectrum')
        return np.zeros_like(s), 0.
    gain = float(np.sqrt(numerator / denominator))
    return s * gain, gain


def fro_match(s, hns):
    """Scalar-F matches the observed HNS spectrum, never ideal ExactFlat."""
    s, hns = (np.asarray(x, dtype=np.float64) for x in (s, hns))
    if s.shape != hns.shape or not np.isfinite(s).all() or not np.isfinite(hns).all():
        raise ValueError('Invalid Scalar-F spectrum')
    if (s < 0).any() or (hns < 0).any() or np.linalg.norm(s)==0:
        raise ValueError('Invalid Scalar-F source')
    gain=float(np.linalg.norm(hns)/np.linalg.norm(s))
    return s*gain,gain


def stats(s, q, moment, scale):
    e = s*s*q
    g = moment*s[:, None]*s[None, :]
    return dict(raw_fpr=float(e.sum()**2 / np.dot(e, e)),
                full_moment_pr=float(np.trace(g)**2 / np.square(g).sum()),
                functional_energy=float(e.sum()*scale**2))


def cells():
    with (LATEST/'matrix.tsv').open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT/'manifest.json'
    if manifest_path.exists():
        assert read(manifest_path)['source_manifest_sha256'] == sha(SOURCE)
        return
    source = read(SOURCE)['checkpoints']
    assert len(source) == 18
    assert {(c['base'], c['task'], c['seed']) for c in source} == {
        (b,t,s) for b in BASES for t in TASKS[:3] for s in (42,43,44)}
    assert read(LATEST/'artifact_integrity_audit.json')['status'] == 'pass'
    assert read(LATEST/'summary.json')['status'] == 'complete'
    code = read(LATEST/'manifest.json')['code_sha256']
    for path, digest in code.items():
        assert sha(ROOT/path) == digest, ('Evaluation code changed', path)
    activation_audit = []
    for c in source:
        assert sha(Path(c['source'])/'adapter_model.safetensors') == c['source_sha256']
        assert sha(Path(c['source'])/'adapter_config.json') == c['config_sha256']
        ampath = ACT/'activations'/c['base']/c['task']/f'seed{c["seed"]}.json'
        am = read(ampath)
        assert am['source_sha256'] == c['source_sha256']
        assert sha(am['npz']) == am['npz_sha256']
        activation_audit.append(dict(base=c['base'], task=c['task'], seed=c['seed'],
            manifest=str(ampath), manifest_sha256=sha(ampath), npz_sha256=am['npz_sha256']))
    refs = [r for r in cells() if r['method'] in ('base','original_lora','hns_f4_s1')]
    assert len(refs) == 152
    for r in refs:
        folder = Path(r['metrics_path']).parent
        assert sha(r['metrics_path']) == r['metrics_sha256']
        assert (folder/'COMPLETE').exists() and (folder/'scored.jsonl').exists()
        for method in CONTROLS:
            link = OUT/'eval'/method/r['base']/r['eval_task']/folder.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(folder.resolve(), target_is_directory=True)
            else:
                assert link.resolve() == folder.resolve()
        r.update(predictions_sha256=sha(folder/'predictions.jsonl'), scored_sha256=sha(folder/'scored.jsonl'))
    # Quantify why the old Frobenius scalar is not a usable energy control.
    old_audit = []
    for c in source:
        am = read(ACT/'activations'/c['base']/c['task']/f'seed{c["seed"]}.json')
        z = np.load(am['npz'])
        hmeta = read(Path(c['hns_path'])/'spectral_edit_meta.json')['module_stats']
        q = np.diagonal(z['coordinate_second_moment_sum'], axis1=-2, axis2=-1)/z['token_counts'].sum()
        for j,name in enumerate(z['names'].tolist()):
            s = z['sigma'][j].astype(np.float64)
            h = np.asarray(hmeta[name]['sigma_after'], dtype=np.float32).astype(np.float64)
            ef = np.dot(h*h,q[j])
            gf = np.linalg.norm(h)/np.linalg.norm(s)
            _, ge = energy_match(s,h,q[j])
            old_audit.append(dict(base=c['base'],task=c['task'],seed=c['seed'],module=name,
                fro_matching_gain=gf, energy_matching_gain=ge,
                old_fro_scalar_energy_over_hns=float(gf*gf*np.dot(s*s,q[j])/ef)))
    tsv(OUT/'old_scalar_energy_audit.tsv', old_audit)
    write(manifest_path, dict(status='prepared',created_utc=datetime.now(timezone.utc).isoformat(),
        source_manifest=str(SOURCE),source_manifest_sha256=sha(SOURCE),checkpoints=source,
        activation_audit=activation_audit,reference_cells=refs,code_sha256=code,
        latest_summary_sha256=sha(LATEST/'summary.json'),retraining=False,
        order=['scalar_e','flat_e','fpr','recovery_retention','functional_boundary'],
        reference='HNS 4+1; full scope; strength1; nuclear-preserving',
        matching='each module, sum(s_i^2 raw_q_i); controls not nuclear-preserving',
        statistical_plan=dict(metrics=['raw_fpr','full_moment_pr','functional_energy'],
            fpr_aggregation='module median',energy_aggregation='sum over modules; ratio to same source',
            models=['Energy','FPR','Energy+FPR'],cluster='source checkpoint',
            cv='leave-one-source-checkpoint-out, within-checkpoint centered outcomes/predictors',
            uncertainty='source-checkpoint bootstrap; seed20260914; 2000 resamples',
            matched_energy='analytically equal; never fit numerical roundoff as Energy information',
            recovery_macro='checkpoint equal; eight commonsense tasks equal then four families equal',
            zero_denominator='undefined/null, not zero; report coverage'),
        training_seed_caveat='Retain all labels42/43/44. Existing audit flags seed42 Llama recipe/provenance differences; not strict iid training replications.'))
    print('[Prepared] 18 checkpoints, 152 reference cells reused', flush=True)


def build(base, method, device='cuda'):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    summaries = []
    for c in read(OUT/'manifest.json')['checkpoints']:
        if c['base'] != base:
            continue
        dest = OUT/'adapters'/base/c['task']/f'seed{c["seed"]}'/method
        meta_path = dest/'energy_match_meta.json'
        if meta_path.exists():
            meta = read(meta_path)
            assert sha(dest/'adapter_model.safetensors') == meta['weights_sha256']
            summaries.append(meta['summary'])
            continue
        if dest.exists():
            raise FileExistsError(f'Incomplete adapter; refusing overwrite: {dest}')
        ampath = ACT/'activations'/base/c['task']/f'seed{c["seed"]}.json'
        am = read(ampath)
        z = np.load(am['npz'])
        names = z['names'].tolist()
        n = int(z['token_counts'].sum())
        moments = z['coordinate_second_moment_sum']/n
        q = np.diagonal(moments, axis1=-2, axis2=-1)
        state, fmt = load_lora_state_dict(c['source'])
        pairs = collect_pairs(state)
        assert sorted(pairs) == names
        hmeta = read(Path(c['hns_path'])/'spectral_edit_meta.json')['module_stats']
        output, modules = dict(state), {}
        with torch.inference_mode():
            for j,name in enumerate(names):
                ka,a = pairs[name]['A']; kb,b = pairs[name]['B']
                u,s,v,_ = lowrank_svd_from_ba(b.to(device),a.to(device))
                assert hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() == str(z['basis_sha256'][j]), (name,'Cached basis mismatch')
                orig = s.double().cpu().numpy()
                assert np.allclose(orig,z['sigma'][j],rtol=1e-6,atol=1e-8)
                h = np.asarray(hmeta[name]['sigma_after'],dtype=np.float32).astype(np.float64)
                target,gain = (fro_match(orig,h) if method=='scalar_f' else
                    energy_match(orig if method=='scalar_e' else np.ones_like(orig),h,q[j]))
                target_tensor = torch.as_tensor(target,device=device,dtype=s.dtype)
                bn,an = rebuild_ba_from_uv_sigma(u,v,target_tensor)
                bs,ass = bn.to(b.dtype),an.to(a.dtype)
                # Inspect saved matrix in the frozen source coordinates, not a new
                # SVD basis (Flat has degenerate singular values).
                # V/U from fp32 QR are only approximately orthonormal. Using
                # U.T D V naively injects basis Gram roundoff into tiny edited
                # directions. Recover coefficients with the finite-basis dual,
                # and use B.T B for the true left/output Gram instead.
                vd = v.double()
                coordinates = ass.double()@vd.T@torch.linalg.inv(vd@vd.T)
                m = torch.as_tensor(moments[j],device=device,dtype=torch.float64)
                href_e = float(np.dot(h*h,q[j])*float(z['scales'][j])**2)
                target_e = float(np.dot(target*target,q[j])*float(z['scales'][j])**2)
                save_gain = 1.
                # Casting factors to their original storage dtype can change
                # the energy slightly. Correct only that numerical discrepancy
                # with a common module scalar, never an extra shape edit.
                for _ in range(5):
                    output_gram = bs.double().T@bs.double()
                    saved_e = float(torch.trace(output_gram@coordinates@m@coordinates.T).item()*float(z['scales'][j])**2)
                    if abs(saved_e/target_e-1) < 1e-7: break
                    correction = float(np.sqrt(target_e/saved_e))
                    save_gain *= correction
                    bs = (bs.double()*correction).to(b.dtype)
                output_gram = bs.double().T@bs.double()
                saved_e = float(torch.trace(output_gram@coordinates@m@coordinates.T).item()*float(z['scales'][j])**2)
                assert abs(save_gain-1)<1e-4, (name,'Excessive save roundoff correction',save_gain)
                output[ka],output[kb] = ass.cpu(),bs.cpu()
                error = factor_error(bs.float(),ass.float(),(bn*save_gain).float(),an.float())
                assert error < 1e-5, (name,'Saved reconstruction',error)
                saved_energy_error = abs(saved_e/target_e-1)
                assert saved_energy_error < 1e-5, (name,'Saved energy mismatch',saved_energy_error)
                saved_fro_sq=float(torch.trace(output_gram@(ass.double()@ass.double().T)).item())
                saved_fro_error=abs(np.sqrt(saved_fro_sq/np.dot(h,h))-1)
                if method=='scalar_f':
                    assert saved_fro_error < 1e-5, (name,'Saved HNS Frobenius mismatch',saved_fro_error)
                scale = float(z['scales'][j])
                values = stats(target,q[j],moments[j],scale)
                modules[name] = dict(**values, sigma_after=target.tolist(),source_sigma=orig.tolist(),
                    q=q[j].tolist(),scale=scale,gain=gain,hns_functional_energy=href_e,
                    source_functional_energy=float(np.dot(orig*orig,q[j])*scale**2),
                    source_basis_sha256=str(z['basis_sha256'][j]),
                    analytic_energy_relative_error=abs(values['functional_energy']/target_e-1),
                    saved_energy_relative_error=saved_energy_error,saved_update_relative_error=error,
                    saved_hns_fro_relative_error=saved_fro_error if method=='scalar_f' else None,
                    storage_roundoff_common_gain=save_gain,
                    nuclear_ratio=float(target.sum()/orig.sum()))
                if (j+1)%64==0:
                    print('[Build]',base,c['task'],c['seed'],method,j+1,'/',len(names),flush=True)
        # Copy only the evaluation skeleton, never checkpoints/training state.
        dest.mkdir(parents=True)
        for filename in ('adapter_config.json','README.md'):
            src = Path(c['source'])/filename
            if src.exists(): shutil.copy2(src,dest/filename)
        save_lora_state_dict(str(dest),output,fmt)
        assert sha(dest/'adapter_config.json') == c['config_sha256']
        rows = list(modules.values())
        summary = dict(base=base,task=c['task'],seed=c['seed'],method=method,path=str(dest),
            raw_fpr=float(np.median([r['raw_fpr'] for r in rows])),
            full_moment_pr=float(np.median([r['full_moment_pr'] for r in rows])),
            functional_energy=sum(r['functional_energy'] for r in rows),
            source_functional_energy=sum(r['source_functional_energy'] for r in rows),
            max_saved_energy_relative_error=max(r['saved_energy_relative_error'] for r in rows),
            max_analytic_energy_relative_error=max(r['analytic_energy_relative_error'] for r in rows),
            max_storage_roundoff_common_gain_error=max(abs(r['storage_roundoff_common_gain']-1) for r in rows))
        write(meta_path,dict(status='pass',summary=summary,module_stats=modules,
            source_sha256=c['source_sha256'],activation_npz_sha256=am['npz_sha256'],
            weights_sha256=sha(dest/'adapter_model.safetensors'),preserve_nuclear_norm=False,
            matching='HNS 4+1 Frobenius, modulewise' if method=='scalar_f' else 'raw q, modulewise',
            builder_sha256=sha(__file__)))
        summaries.append(summary)
        print('[Built]',base,c['task'],c['seed'],method,'max saved E error',summary['max_saved_energy_relative_error'],flush=True)
    write(OUT/f'{base}_{method}_build_summary.json',summaries)
    variants = [dict(label=f'{r["task"]}__seed{r["seed"]}__{method}',path=r['path'],
        train_task=r['task'],seed=r['seed'],method=method) for r in summaries]
    oldvm = read(LATEST/f'{base}_variant_manifest.json')
    write(OUT/f'{base}_{method}_variant_manifest.json',dict(base=base,base_model=oldvm['base_model'],
        task_config=oldvm['task_config'],variants=variants,status='complete'))


def run_command(argv, log, env=None):
    log = Path(log); log.parent.mkdir(parents=True,exist_ok=True)
    history_path = OUT/f'commands_{os.getenv("HNS_ENERGY_WORKER", "orchestrator")}.json'
    history = read(history_path) if history_path.exists() else []
    entry = dict(argv=argv,log=str(log),started_utc=datetime.now(timezone.utc).isoformat())
    history.append(entry); write(history_path,history)
    print('[Start]', ' '.join(argv), 'log',log,flush=True)
    with log.open('w') as f:
        result = subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    entry.update(returncode=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
    write(history_path,history)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {log}')


def archived_probe_manifest(current, archived_generation):
    """Recover the manifest actually used, not today's expanded manifest.

    Adapter IDs are assigned from list position by the unchanged evaluator.
    Keeping all editing arms preserves its five-adapter batching blocks too.
    """
    order = archived_generation['variant_order']
    by_label = {v['label']: v for v in current['variants']}
    if len(by_label) != len(current['variants']) or len(set(order)) != len(order):
        raise ValueError('Duplicate adapter labels in reuse probe')
    if not set(order).issubset(by_label):
        raise ValueError('Archived probe adapter missing from current manifest')
    if archived_generation['configuration']['max_num_seqs'] != 4096:
        raise ValueError('Unexpected historical probe configuration')
    return {**current, 'variants': [by_label[label] for label in order]}


def reuse_probe_plans(base):
    # The extension's Llama manifest was later expanded from 30 to 45 arms.
    # Its archived short probe records the original 30-arm order exactly.
    roots = [('extension', LATEST)]
    if base == BASES[0]:
        PILOT = ROOT/'reports/functional_hns_pilot_2x3_seed42_20260914'
        roots.append(('seed42_pilot', PILOT))
    plans = []
    for name, root in roots:
        archived = root/'batch_probe'/base/'seqs4096'/'generation_manifest.json'
        manifest = archived_probe_manifest(read(root/f'{base}_variant_manifest.json'), read(archived))
        plans.append((name, manifest, archived.parent))
    return plans


def evaluate(base, method):
    vm_path = OUT/f'{base}_{method}_variant_manifest.json'
    vm = read(vm_path)
    def generate(manifest_path, output, tasks, seqs, limit=None):
        cmd = [sys.executable,str(ROOT/'scripts/eval_forgetting_matrix_vllm.py'),
            '--base_model',vm['base_model'],'--variant_manifest',str(manifest_path),
            '--config',vm['task_config'],'--output_dir',str(output),'--tasks',*tasks,
            '--max_model_len','4096','--gpu_memory_utilization','0.94',
            '--max_num_seqs',str(seqs),'--max_num_batched_tokens','65536',
            '--adapter_block_size','5','--max_lora_rank','16',
            '--prompt_chunk_short','4096','--prompt_chunk_long','1024','--seed','42']
        if limit: cmd += ['--diagnostic_max_samples',str(limit)]
        run_command(cmd,output/f'generate_{"_".join(tasks)}.log')
    # v2 deliberately does not reuse v1's failed probes or incompatible audit.
    # Reproduce historical SHORT probes (4096), not the full long-task cap.
    # Hash-check ALL final reference cells, including refreshed Llama seed42;
    # live compatibility coverage is explicitly recorded, never extrapolated
    # into a claim that every cached sample was regenerated.
    probe_audit = OUT/f'{base}_reuse_probe_v2_audit.json'
    if not probe_audit.exists():
        refs = [r for r in read(OUT/'manifest.json')['reference_cells'] if r['base']==base]
        for r in refs:
            folder = Path(r['metrics_path']).parent
            for filename, key in [('predictions.jsonl','predictions_sha256'),
                                  ('scored.jsonl','scored_sha256'),('metrics.json','metrics_sha256')]:
                assert sha(folder/filename)==r[key], ('Reference cache changed',folder,filename)
        check = []
        for name, historical_vm, old_probe in reuse_probe_plans(base):
            probe = OUT/'probe_v2'/base/name
            probevm = OUT/f'{base}_{name}_probe_v2_variant_manifest.json'
            write(probevm,historical_vm)
            generate(probevm,probe,list(TASKS),4096,32)
            labels = {'base'} | {v['label'] for v in historical_vm['variants']
                                    if v['method'] in ('original_lora','hns_f4_s1')}
            for r in refs:
                folder = Path(r['metrics_path']).parent
                if folder.name not in labels: continue
                new = list(map(json.loads,(probe/r['eval_task']/folder.name/'predictions.jsonl').read_text().splitlines()))
                for reference_name, reference_folder in [('historical_short',old_probe/r['eval_task']/folder.name),
                                                           ('final_full',folder)]:
                    ref = {x['id']:x for x in map(json.loads,(reference_folder/'predictions.jsonl').read_text().splitlines())}
                    mismatches = sum(x['token_ids']!=ref[x['id']]['token_ids'] for x in new)
                    check.append(dict(probe=name,reference=reference_name,eval_task=r['eval_task'],
                        label=folder.name,samples=len(new),token_mismatches=mismatches))
        changed = sum(r['token_mismatches'] for r in check)
        write(probe_audit,dict(version=2,status='pass' if changed==0 else 'incompatible',
            token_mismatches=changed,checks=check,hash_verified_final_cells=len(refs),
            live_probe_training_seeds=[42,43,44] if base==BASES[0] else [43,44],
            refreshed_llama_seed42='final cache integrity verified; not the obsolete pilot cache',
            max_num_seqs=4096,main_evaluation_protocol_changed=False))
        if changed:
            raise RuntimeError('Latest prediction cache incompatible with reproduction probe. Stop; do not mix scores or re-infer cached methods silently.')
    assert read(probe_audit)['status']=='pass'
    dest = OUT/'eval'/method/base
    for tasks,seqs in [(list(TASKS[:3]),2048),(['commonsense'],4096)]:
        if not all((dest/t/v['label']/'COMPLETE').exists() for t in tasks for v in vm['variants']):
            generate(vm_path,dest,tasks,seqs)
        run_command([sys.executable,str(ROOT/'scripts/score_forgetting_matrix.py'),
            '--matrix_dir',str(dest),'--workers','32'],dest/f'score_{tasks[0]}.log')
    assert all(read(dest/t/v['label']/'metrics.json')['samples']==COUNTS[t] for t in TASKS for v in vm['variants'])
    write(OUT/f'{base}_{method}_complete.json',dict(status='complete',new_cells=36,checkpoint_count=9))


def worker(base, method):
    os.environ.update(PYTHONPATH=f'{ROOT}/src:{ROOT}/scripts',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_HOME='/dataset1/zailong/cache/peft-sft-lab/huggingface',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_BATCH_INVARIANT='1',PYTHONHASHSEED='42',
        VLLM_DISABLE_COMPILE_CACHE='1',TORCH_CUDNN_SDPA_DEPRIORITIZED='1')
    os.environ['HNS_ENERGY_WORKER'] = base+'_'+method
    # ZMQ Unix socket paths must fit sockaddr_un's 107-character limit.
    tmp = Path(tempfile.mkdtemp(prefix=f'he-{os.environ.get("SLURM_JOB_ID",os.getpid())}-',dir='/tmp'))
    os.environ.update(TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),VLLM_CACHE_ROOT=str(tmp/'vllm'),
        TORCHINDUCTOR_CACHE_DIR=str(tmp/'inductor'),TRITON_CACHE_DIR=str(tmp/'triton'))
    env = os.environ.copy(); env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    run_command([sys.executable,__file__,'--build','--base',base,'--method',method,'--output_dir',str(OUT)],
        OUT/f'build_{base}_{method}.log',env)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    evaluate(base,method)


def pipeline():
    prepare()
    devices = os.getenv('CUDA_VISIBLE_DEVICES','0,1').split(',')
    if len(devices) != 2:
        raise RuntimeError(f'This pipeline requires exactly two allocated GPUs: {devices}')
    try:
        for method in CONTROLS:
            write(OUT/'worker_status.json',dict(phase=method,gpu_count=2,job_id=os.getenv('SLURM_JOB_ID')))
            processes = []
            for base,device in zip(BASES,devices):
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = device
                log = OUT/f'worker_{base}_{method}.log'
                handle = log.open('w')
                proc = subprocess.Popen([sys.executable,__file__,'--worker','--base',base,'--method',method],
                    cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                processes.append((proc,handle,base,log))
            failures = []
            for proc,handle,base,log in processes:
                code = proc.wait(); handle.close()
                if code: failures.append(f'{base}: exit{code}: {log}')
            if failures: raise RuntimeError('; '.join(failures))
        for stage in ('fpr','recovery','boundary','report'):
            write(OUT/'worker_status.json',dict(phase=stage,job_id=os.getenv('SLURM_JOB_ID')))
            run_command([sys.executable,str(ROOT/'scripts/analyze_hns_energy_matched.py'),'--stage',stage],OUT/f'{stage}.log')
        write(OUT/'worker_status.json',dict(phase='complete',job_id=os.getenv('SLURM_JOB_ID')))
    except BaseException as e:
        write(OUT/'worker_status.json',dict(phase='failed',error=str(e),job_id=os.getenv('SLURM_JOB_ID')))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true'); p.add_argument('--build',action='store_true')
    p.add_argument('--pipeline',action='store_true'); p.add_argument('--worker',action='store_true'); p.add_argument('--base',choices=BASES)
    p.add_argument('--method',choices=(*CONTROLS,'scalar_f')); p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--output_dir',type=Path,default=OUT)
    args=p.parse_args()
    OUT=args.output_dir.resolve()
    if args.prepare: prepare()
    if args.build: build(args.base,args.method,args.device)
    if args.worker: worker(args.base,args.method)
    if args.pipeline: pipeline()
