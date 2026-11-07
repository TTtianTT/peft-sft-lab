#!/usr/bin/env python3
"""Round-two shared protocol, audited cache views, builders and batched workers."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

import rebuttal_iclr27 as r
import rebuttal_iclr27_e4 as e4
from rebuttal_iclr27_e2_e3 import (average_ranks, correlation, operations,
                                 save_adapter, spectral_edit_in_builder, CALIBRATION, ALL_MODULES)

STD = ('original_lora', 'recon_0', 'hns_f4_s1', 'flat_nuclear')
SCOPES = ('R2-O1', 'R2-O2', 'R2-O3', 'R2-O4', 'R2-O5', 'R2-O6', 'R2-O7', 'R2-O8',
          'R2-O9', 'R2-O10', 'R2-O11', 'R2-O12')
CONFIGS = {'recon_0': (0, 0, 0.), 'hns_f4_s1': (4, 1, 1.),
           'recon_0_unbal': (0, 0, 0.), 'hns_unbal': (4, 1, 1.),
           'hns_f4_s0': (4, 0, 1.), 'hns_f0_s5': (0, 5, 1.), 'hns_f4_s1_l05': (4, 1, .5)}


def folder(exp):
    return r.OUT / exp


def label(cp, method):
    return cp['condition'] + '__' + method


def check(exp):
    p = folder(exp)
    lock = r.read(p / 'protocol.lock.json')
    assert r.sha(p / 'protocol.md') == lock['protocol_sha256']
    for path, digest in lock['files'].items():
        assert r.sha(r.ROOT / path) == digest, ('Frozen dependency changed', path)
    return lock


def require_complete(exp):
    p = folder(exp)
    data = r.read(p / 'summary.json')
    assert data['status'] == 'complete', (exp, data['status'])
    for name in ('report.md', 'checkpoint_results.tsv', 'paired_ci.tsv'):
        assert (p / name).stat().st_size
    for name in ('coverage_audit.json', 'artifact_integrity_audit.json'):
        assert r.read(p / name)['status'] == 'pass'
    for name, digest in data.get('outputs_sha256', {}).items():
        assert r.sha(p / name) == digest


def initialize_round():
    p = r.OUT / 'R2-plan.json'
    if p.exists():
        return r.read(p)
    for exp in ('E0', 'E1', 'E2', 'E3', 'E4'):
        require_complete(exp)
    start = datetime.now(timezone.utc)
    obj = dict(status='running', base_commit='9f23dfc', started_utc=start.isoformat(),
        deadline_utc=(start + timedelta(hours=48)).isoformat(), max_gpus=2,
        stages=list(SCOPES), o11_author_approval=None,
        budget_rule='A then B; C starts only if its prespecified estimate plus reserve fits remaining wall budget; no test-score-dependent omissions')
    r.write(p, obj)
    return obj


def remaining_hours():
    return (datetime.fromisoformat(initialize_round()['deadline_utc']) - datetime.now(timezone.utc)).total_seconds()/3600


def base_reference(base, task):
    selected = [x for x in r.reference_cells() if x['method'] == 'base' and x['base'] == base and x['eval_task'] == task]
    return dict(selected[0]) if selected else None


def source_record(path, **kwargs):
    path = Path(path)
    cfg = r.read(path / 'adapter_config.json')
    assert cfg['peft_type'] == 'LORA' and not cfg.get('use_dora') and not cfg.get('use_rslora')
    assert cfg['lora_alpha']/cfg['r'] == 2
    return dict(source=str(path), source_sha256=r.sha(path / 'adapter_model.safetensors'),
        config_sha256=r.sha(path / 'adapter_config.json'), rank=int(cfg['r']), alpha=int(cfg['lora_alpha']), **kwargs)


def first_wave_cells(exp, cp, method):
    p = r.OUT / exp
    if exp == 'main':
        rows = r.reference_cells()
        return {x['eval_task']: {**x} for x in rows if x['base'] == cp['base'] and
                x['train_task'] == cp['task'] and int(x['seed'] or -1) == cp['seed'] and x['method'] == method}
    if exp in ('para', 'spectral_surgery'):
        vm = r.read(r.BASELINES / exp / f'{cp["base"]}_variant_manifest.json')
        v = next(v for v in vm['variants'] if v['train_task'] == cp['task'] and int(v['seed']) == cp['seed'] and v['method'] == method)
        return {t: dict(metrics_path=str(r.BASELINES / exp / 'eval' / cp['base'] / t / v['label'] / 'metrics.json')) for t in r.TASKS}
    rows = r.load_tsv(p / 'evaluation_cells.tsv')
    if cp.get('old_condition'):
        selected = [x for x in rows if x.get('condition') == cp['old_condition'] and x['method'] == method]
    else:
        selected = [x for x in rows if x['base'] == cp['base'] and x['train_task'] == cp['task'] and
                    int(x['seed']) == cp['seed'] and x['method'] == method]
    assert len(selected) == 4, (exp, cp['condition'], method, len(selected))
    return {x['eval_task']: dict(x) for x in selected}


def attach_cache(cp, method, exp):
    refs = first_wave_cells(exp, cp, method)
    assert set(refs) == set(r.TASKS)
    cp.setdefault('cache', {})[method] = refs


def freeze(exp, cps, methods, detail, expected_cells, statistics='checkpoint', extra_files=()):
    initialize_round()
    p = folder(exp); p.mkdir(parents=True, exist_ok=True)
    if (p / 'protocol.lock.json').exists():
        return check(exp)
    assert len({x['condition'] for x in cps}) == len(cps)
    for cp in cps:
        assert r.sha(Path(cp['source']) / 'adapter_model.safetensors') == cp['source_sha256']
        assert r.sha(Path(cp['source']) / 'adapter_config.json') == cp['config_sha256']
        cp['methods'] = list(methods)
        cp['new_methods'] = [m for m in methods if m not in cp.get('cache', {})]
    files = dict(r.read(r.OUT / 'E1/protocol.lock.json')['files'])
    for name in ('scripts/rebuttal_iclr27_r2.py', 'scripts/rebuttal_iclr27_r2_a.py',
                 'scripts/rebuttal_iclr27_e4.py', 'scripts/rebuttal_iclr27_e2_e3.py',
                 'src/finetune/spectral_edit/flat.py', 'src/finetune/spectral_edit/cli.py',
                 'src/finetune/spectral_edit/hooks.py', 'src/finetune/spectral_edit/calib.py',
                 'src/finetune/spectral_edit/edit_strategies.py', *extra_files):
        if (r.ROOT / name).is_file():
            files[name] = r.sha(r.ROOT / name)
    body = f'''# {exp}: locked protocol

Frozen UTC: {r.now()}. Source round-one commit9f23dfc. All configurations and conditions are fixed before new test inference. All outcomes will be reported.
Use the existing functional_hns_three_seed_20260914 evaluator, native tokenizer/chat template, prompts/parsers/scorers and test items unchanged. HumanEval164 pass@1; GSM8K1319 strict; IFEval541 prompt strict; Commonsense22419 items and8 equally weighted subtasks. Greedy seed42, batch invariant, max_model_len4096, adapter block5, long max_num_seqs2048/CS4096, batched tokens65536, prompt chunks4096/1024, FLASH_ATTN, async off, prefix cache off, GPU memory0.94. max_lora_rank16 for ranks<=16,32 for rank32,64 for rank64. Same inference exception as E4b.
Target is the source task family. Off equally averages the other3 families. FG clips max(Base-family minus edited-family,0) within each off family before averaging. Scores/differences in percent/percentage points.
Inference and CUDA construction only inside Slurm. Global maximum2 GPUs; at most two1-GPU workers in a2-GPU allocation. Builders exit before exclusive-process vLLM starts. No retraining in tierA.
Source weights/config SHA256, serialized adapter SHA256, finite spectra, all-module coverage, nuclear budget and relative reconstruction error are audited. fp32 compact SVD, stock float64 HNS map, spectrum cast to fp32 before factorization, source dtype cast at exactly the stock writeback point. Numeric audit tolerance5e-5 is fixed; preserve native rank/alpha/scaling2. No threshold changes or test-based selection.
Reused cells must pass metrics/prediction/scored hashes where recorded and independent ordered per-sample input-hash agreement with Base. Cached cells are linked, never overwritten. New Base cells are required for new model sizes.
Statistics: {statistics}. Paired resamples20000, seed{r.BOOT_SEED}, percentile95%; all metrics reported. Checkpoint intervals are descriptive on shared test sets, not IID training runs. Single-seed conditions use paired item intervals and exact McNemar repairs/damage, resampling Commonsense within each of8 subtasks. An interval including0 is not an equivalence proof. No setting is selected by test performance.

{detail}
'''
    (p / 'protocol.md').write_text(body)
    obj = dict(status='frozen', created_utc=r.now(), files=files, protocol_sha256=r.sha(p/'protocol.md'),
        checkpoints=cps, methods=list(methods), expected_new_cells=expected_cells,
        statistics=statistics, inference_seed=42, bootstrap_resamples=20000, bootstrap_seed=r.BOOT_SEED,
        deadline_utc=initialize_round()['deadline_utc'])
    r.write(p/'protocol.lock.json', obj)
    return obj


def link_cell(destination, target):
    destination, target = Path(destination), Path(target).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.symlink_to(target, target_is_directory=True)
    assert destination.resolve() == target


def base_inputs(cps):
    result = {}
    for cp in cps:
        for task in r.TASKS:
            key = cp['base'], task
            if key not in result:
                ref = base_reference(cp['base'], task) or base_reference('Qwen3-8B', task)
                result[key] = r.input_hashes(Path(ref['metrics_path']).parent)
    return result


def prime_cache(exp):
    p = folder(exp); lock = check(exp); cps = lock['checkpoints']
    bases = base_inputs(cps); audited = []; memo = {}
    for cp in cps:
        view = p/'runtime/eval'/cp['condition']
        r.write(p/f'{cp["condition"]}_variant_manifest.json', dict(base=cp['base'],
            variants=[dict(label=label(cp,m),method=m,path=cp['source'],train_task=cp['task'],seed=cp['seed']) for m in lock['methods']]))
        for task in r.TASKS:
            ref = base_reference(cp['base'], task)
            if ref:
                target = Path(ref['metrics_path']).parent
                row = dict(base=cp['base'], train_task=cp['task'], seed=cp['seed'], method='base', eval_task=task,
                           condition=cp['condition'], metrics_path=ref['metrics_path'], metrics_sha256=ref['metrics_sha256'])
                key = str(target.resolve())
                if key not in memo: memo[key] = r.audit_cell(row, bases)
                audited.append({**memo[key], **row})
                link_cell(view/task/'base', target)
            for method, refs in cp.get('cache', {}).items():
                old = refs[task]; target = Path(old['metrics_path']).parent
                key = str(target.resolve())
                row = dict(base=cp['base'], train_task=cp['task'], seed=cp['seed'], method=method, eval_task=task,
                           condition=cp['condition'], metrics_path=str(target/'metrics.json'))
                if key not in memo:
                    memo[key] = r.audit_cell(row, bases, {k:v for k,v in old.items() if k.endswith('_sha256')})
                audited.append({**memo[key], **row})
                link_cell(view/task/label(cp,method), target)
    r.tsv(p/'cache_input_audit.tsv', audited)
    r.write(p/'cache_audit.json', dict(status='pass', cells=len(audited), unique_cells=len(memo),
        sample_input_mismatches=0, audit_sha256=r.sha(p/'cache_input_audit.tsv')))


def env_setup(exp, worker):
    p = folder(exp)
    os.environ.update(PYTHONPATH=f'{r.ROOT}/src:{r.ROOT}/scripts', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', VLLM_BATCH_INVARIANT='1',
        VLLM_USE_FLASHINFER_SAMPLER='0', PYTHONHASHSEED='42', VLLM_DISABLE_COMPILE_CACHE='1',
        TORCH_CUDNN_SDPA_DEPRIORITIZED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        HF_HOME='/dataset1/zailong/cache/peft-sft-lab/huggingface',
        HF_DATASETS_CACHE=str(p/'runtime/datasets_cache'), MPLCONFIGDIR=str(p/'runtime/matplotlib'))
    temp = Path(tempfile.mkdtemp(prefix=f'r2-{os.getenv("SLURM_JOB_ID","cpu")}-{worker}-',dir='/tmp'))
    os.environ.update(TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp), VLLM_CACHE_ROOT=str(temp/'vllm'),
        TORCHINDUCTOR_CACHE_DIR=str(temp/'inductor'), TRITON_CACHE_DIR=str(temp/'triton'))


def split_checkpoints(cps, worker):
    fresh = [c for c in cps if c['new_methods']]
    return [c for i,c in enumerate(fresh) if i % 2 == worker]


def spectrum(sigma, method):
    from finetune.spectral_edit.posthoc_hns import HNSEditConfig, hybrid_newton_schulz_singular_values
    from finetune.spectral_edit.flat import flat_spectrum
    if method == 'flat_nuclear': return flat_spectrum(sigma, method)
    fast, stable, strength = CONFIGS[method]
    return hybrid_newton_schulz_singular_values(sigma, HNSEditConfig(
        fast_steps=fast, stable_steps=stable, hns_strength=strength, preserve_nuclear_norm=True))[0]


def writeback(u, vh, sigma, method):
    from finetune.spectral_edit.svd import rebuild_ba_from_uv_sigma
    if method in ('recon_0_unbal', 'hns_unbal'):
        return (u * sigma.unsqueeze(0)).contiguous(), vh.contiguous()
    return rebuild_ba_from_uv_sigma(u, vh, sigma)


def spectral_statistics(original, target):
    s, t = original.double(), target.double()
    energy = t.square().sum()
    return dict(participation_rank=float(energy.square()/(len(t)*t.pow(4).sum())),
        nuclear_participation_rank=float(t.sum().square()/(len(t)*energy)),
        top1_energy=float(t.square().max()/energy), fro_ratio=float(t.norm()/s.norm()))


def build(exp, worker):
    r.require_slurm(); env_setup(exp, worker); os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
    lock = check(exp); p = folder(exp)
    torch, collect, error, load, svd, rebuild, hns, Config = operations()
    artifacts, modules, variants = [], [], []
    for cp in split_checkpoints(lock['checkpoints'],worker):
        src = Path(cp['source'])
        assert r.sha(src/'adapter_model.safetensors') == cp['source_sha256']
        assert r.sha(src/'adapter_config.json') == cp['config_sha256']
        state, fmt = load(str(src)); pairs = collect(state)
        assert len(pairs) > 0
        # A common basis for every newly written method in this condition.
        decomp = {}
        with torch.inference_mode():
            for name, pair in sorted(pairs.items()):
                decomp[name] = svd(pair['B'][1].cuda(), pair['A'][1].cuda())[:3]
        for method in cp['new_methods']:
            variant = dict(label=label(cp,method), train_task=cp['task'], seed=cp['seed'], method=method,
                           condition=cp['condition'], base=cp['base'], rank=cp['rank'])
            if method == 'original_lora':
                variant['path'] = str(src)
                artifacts.append(dict(**variant, weights_sha256=cp['source_sha256'], config_sha256=cp['config_sha256'], modules=len(pairs)))
                variants.append(variant); continue
            dest = p/'runtime/adapters'/cp['condition']/method
            marker = dest/'r2_integrity.json'
            if marker.exists():
                info = r.read(marker)
                assert r.sha(dest/'adapter_model.safetensors') == info['artifact']['weights_sha256']
                artifacts.append(info['artifact']); modules.extend(info['modules'])
            else:
                started = time.monotonic()
                output = dict(state); rows = []; budget = {}
                if method == 'ss_grad_direction_all':
                    name, data = CALIBRATION[cp['task']]
                    argv = [sys.executable,'-m','finetune.spectral_edit.cli','edit','--base_model',cp['base_model'],
                        '--lora_path',str(src),'--out_dir',str(dest),'--mode','gd','--target_modules',*ALL_MODULES,
                        '--calib_samples','128','--calib_batch_size','2','--calib_dataset',name,'--calib_dataset_path',data,
                        '--calib_split','train','--calib_shuffle','--calib_seed','42','--seed','42',
                        '--grad_norm','mean_abs','--preserve_energy','l1','--update_mode','multiplicative',
                        '--asymmetric_update','--eta_suppress','2.0','--eta_enhance','0.2']
                    if not (dest/'spectral_edit_meta.json').exists():
                        spectral_edit_in_builder(argv,p/f'runtime/logs/{cp["condition"]}_{method}_calib.log',p/f'calibration_commands_{worker}.json')
                    metadata = r.read(dest/'spectral_edit_meta.json')
                    assert metadata['meta']['calib_samples_used'] == 128 and set(metadata['sigma_stats']) == set(pairs)
                    output, _ = load(str(dest))
                    budget = dict(calibration_samples=128, forward_batches=64, backward_batches=64,
                        calibration_sha256=r.sha(data), gradient_seconds=metadata['meta']['gradient_seconds'],
                        edit_seconds=metadata['meta']['total_edit_seconds'])
                with torch.inference_mode():
                    for name, pair in sorted(pairs.items()):
                        ka,a=pair['A'];kb,b=pair['B'];u,s,vh=decomp[name]
                        assert len(s) == cp['rank']
                        if method == 'ss_grad_direction_all':
                            bn,an=output[kb].cuda(),output[ka].cuda()
                            un,t,vn,_=svd(bn,an);bt,at=rebuild(un,vn,t)
                            ud,vd=u.double(),vh.double();bd,ad=bn.double(),an.double()
                            left=ud@torch.linalg.solve(ud.T@ud,ud.T@bd)
                            right=(ad@vd.T)@torch.linalg.solve(vd@vd.T,vd)
                            direction_error=max(error(bd,ad,left,ad),error(bd,ad,bd,right))
                        else:
                            t=spectrum(s,method);bt,at=writeback(u,vh,t,method)
                            bn,an=bt.to(b.dtype),at.to(a.dtype)
                            output[kb],output[ka]=bn.cpu(),an.cpu()
                            direction_error=0.
                        assert torch.isfinite(t).all() and bool((t>=0).all())
                        relative=error(bn.double(),an.double(),bt.double(),at.double())
                        source_error=error(bn.double(),an.double(),b.cuda().double(),a.cuda().double())
                        _,ss,_,_=svd(bn,an)
                        nuclear_error=abs(float(ss.double().sum()/s.double().sum())-1)
                        assert max(relative,nuclear_error,direction_error)<5e-5,(name,method,relative,nuclear_error,direction_error)
                        if method.startswith('recon_0'): assert source_error<5e-5
                        assert bn.dtype==b.dtype and an.dtype==a.dtype
                        rows.append(dict(condition=cp['condition'],base=cp['base'],task=cp['task'],seed=cp['seed'],method=method,
                            module=name,rank=len(s),dtype_A=str(an.dtype),dtype_B=str(bn.dtype),
                            reconstruction_relative_error=relative, source_relative_error=source_error,
                            nuclear_relative_error=nuclear_error,direction_relative_error=direction_error,
                            **spectral_statistics(s,t)))
                if method != 'ss_grad_direction_all': save_adapter(src,dest,output,fmt)
                saved,_=load(str(dest))
                assert saved.keys()==output.keys() and all(torch.equal(saved[k],v) and saved[k].dtype==v.dtype for k,v in output.items())
                assert r.sha(dest/'adapter_config.json')==cp['config_sha256']
                artifact=dict(**variant,path=str(dest),source_sha256=cp['source_sha256'],
                    weights_sha256=r.sha(dest/'adapter_model.safetensors'), config_sha256=cp['config_sha256'],
                    modules=len(rows),seconds=time.monotonic()-started,job_id=os.getenv('SLURM_JOB_ID'),**budget)
                r.write(marker,dict(artifact=artifact,modules=rows))
                artifacts.append(artifact);modules.extend(rows)
            variant['path']=str(dest);variants.append(variant)
        del decomp,state
        torch.cuda.empty_cache()
        print('[Built]',exp,cp['condition'],flush=True)
    r.write(p/f'worker{worker}_build.json',dict(variants=variants,artifacts=artifacts))
    r.tsv(p/f'worker{worker}_module_audit.tsv',modules)


def worker(exp, slot):
    r.require_slurm();env_setup(exp,slot);lock=check(exp);p=folder(exp)
    assert r.read(p/'cache_audit.json')['status']=='pass'
    status=p/f'worker{slot}_status.json'
    if exp=='R2-O1' and slot==0:
        # Validation is used only as a plotted covariate; neither r64 condition is selected.
        r.run_command([sys.executable,str(r.ROOT/'scripts/rebuttal_iclr27_r2_a.py'),'--validate-r64'],
            p/'runtime/logs/r64_validation.log',p/'validation_commands.json')
    r.write(status,dict(phase='build',updated_utc=r.now(),job_id=os.getenv('SLURM_JOB_ID'),gpu_count=1))
    r.run_command([sys.executable,__file__,'--build',exp,'--slot',str(slot)],p/f'runtime/logs/worker{slot}_build.log',p/f'commands_{slot}.json')
    vm=r.read(p/f'worker{slot}_build.json')['variants']
    cps={c['condition']:c for c in lock['checkpoints']}
    groups={}
    for v in vm:
        maxrank=max(16,int(v['rank']))
        groups.setdefault((v['base'],maxrank),[]).append(v)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    for (base,maxrank),variants in groups.items():
        cp=cps[variants[0]['condition']]
        manifest=p/f'runtime/manifests/worker{slot}_{base}_r{maxrank}.json'
        config=r.REF/f'{base}_task_config.json'
        if not config.exists():config=r.REF/'Qwen3-8B_task_config.json'
        r.write(manifest,dict(base=base,base_model=cp['base_model'],task_config=str(config),variants=variants))
        dest=p/f'runtime/waves/worker{slot}_{base}_r{maxrank}'
        for task in r.TASKS:
            ref=base_reference(base,task)
            if ref:link_cell(dest/task/'base',Path(ref['metrics_path']).parent)
        for tasks,seqs in ((r.TASKS[:3],2048),(r.TASKS[3:],4096)):
            r.write(status,dict(phase='evaluate',base=base,max_lora_rank=maxrank,tasks=tasks,updated_utc=r.now(),job_id=os.getenv('SLURM_JOB_ID'),gpu_count=1))
            if not all((dest/t/v['label']/'COMPLETE').exists() for t in tasks for v in variants):
                argv=[sys.executable,str(r.ROOT/'scripts/eval_forgetting_matrix_vllm.py'),
                    '--base_model',cp['base_model'],'--variant_manifest',str(manifest),'--config',str(config),
                    '--output_dir',str(dest),'--tasks',*tasks,'--max_model_len','4096','--gpu_memory_utilization','0.94',
                    '--max_num_seqs',str(seqs),'--max_num_batched_tokens','65536','--adapter_block_size','5',
                    '--max_lora_rank',str(maxrank),'--prompt_chunk_short','4096','--prompt_chunk_long','1024','--seed','42']
                r.run_command(argv,p/f'runtime/logs/worker{slot}_{base}_r{maxrank}_{tasks[0]}_generate.log',p/f'commands_{slot}.json')
            r.run_command([sys.executable,str(r.ROOT/'scripts/score_forgetting_matrix.py'),'--matrix_dir',str(dest),'--workers','32'],
                p/f'runtime/logs/worker{slot}_{base}_r{maxrank}_{tasks[0]}_score.log',p/f'commands_{slot}.json')
        for v in variants:
            for task in r.TASKS:
                link_cell(p/'runtime/eval'/v['condition']/task/v['label'],dest/task/v['label'])
                link_cell(p/'runtime/eval'/v['condition']/task/'base',dest/task/'base')
    r.write(status,dict(phase='complete',updated_utc=r.now(),job_id=os.getenv('SLURM_JOB_ID'),gpu_count=1))


def launch_workers(exp):
    r.require_slurm();devices=os.getenv('CUDA_VISIBLE_DEVICES','').split(',');assert len(devices)==2
    prime_cache(exp)
    states=[]
    for slot,device in enumerate(devices):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device;env.pop('CUBLAS_WORKSPACE_CONFIG',None)
        path=folder(exp)/f'runtime/logs/worker{slot}.log';path.parent.mkdir(parents=True,exist_ok=True)
        handle=path.open('a');proc=subprocess.Popen([sys.executable,__file__,'--worker',exp,'--slot',str(slot)],cwd=r.ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        states.append((slot,proc,handle))
    failed=[]
    for slot,proc,handle in states:
        code=proc.wait();handle.close()
        if code:failed.append((slot,code))
    if failed:raise RuntimeError((exp,'worker failures',failed))


def audit_results(exp):
    p=folder(exp);lock=check(exp);bases=base_inputs(lock['checkpoints']);rows=[]
    for cp in lock['checkpoints']:
        view=p/'runtime/eval'/cp['condition']
        for method in ('base',*lock['methods']):
            for task in r.TASKS:
                name='base' if method=='base' else label(cp,method)
                row=dict(base=cp['base'],train_task=cp['task'],seed=cp['seed'],method=method,condition=cp['condition'],eval_task=task,
                         metrics_path=str(view/task/name/'metrics.json'))
                rows.append(r.audit_cell(row,bases))
    fresh=sum(4*len(c['new_methods'])+(4 if base_reference(c['base'],'magicoder') is None else 0) for c in lock['checkpoints'])
    assert fresh==lock['expected_new_cells'],(exp,fresh,lock['expected_new_cells'])
    builds=[r.read(p/f'worker{i}_build.json') for i in (0,1)]
    artifacts=list(itertools.chain.from_iterable(x['artifacts'] for x in builds))
    for a in artifacts:
        path=Path(a['path']);assert r.sha(path/'adapter_model.safetensors')==a['weights_sha256']
        assert r.sha(path/'adapter_config.json')==a['config_sha256']
    modules=list(itertools.chain.from_iterable(r.load_tsv(p/f'worker{i}_module_audit.tsv') for i in (0,1)))
    r.tsv(p/'evaluation_cells.tsv',rows)
    r.write(p/'coverage_audit.json',dict(status='pass',conditions=len(lock['checkpoints']),new_cells=fresh,
        expected_new_cells=lock['expected_new_cells'],audited_cells=len(rows),sample_input_mismatches=0))
    r.write(p/'artifact_integrity_audit.json',dict(status='pass',artifacts=artifacts,edited_module_rows=len(modules),
        max_reconstruction_relative_error=max((float(x['reconstruction_relative_error']) for x in modules),default=0),
        max_nuclear_relative_error=max((float(x['nuclear_relative_error']) for x in modules),default=0)))
    return rows,modules


def points(cells,cps):
    lookup={(x['condition'],x['method'],x['eval_task']):float(x['score']) for x in cells}
    result=[]
    for cp in cps:
        off=[t for t in r.TASKS if t!=cp['task']]
        methods=sorted({x['method'] for x in cells if x['condition']==cp['condition']})
        for method in methods:
            scores={t:lookup[cp['condition'],method,t] for t in r.TASKS}
            result.append(dict(condition=cp['condition'],checkpoint=cp['condition'],base=cp['base'],task=cp['task'],seed=cp['seed'],
                rank=cp['rank'],method=method,**scores,target=scores[cp['task']],off_score=float(np.mean([scores[t] for t in off])),
                forgetting_gap=float(np.mean([max(lookup[cp['condition'],'base',t]-scores[t],0) for t in off]))))
    return result


def contrast(rows,weights,name):
    ix={(x['checkpoint'],x['method']):x for x in rows};cps=sorted({x['checkpoint'] for x in rows})
    draws=np.random.default_rng(r.BOOT_SEED).integers(0,len(cps),(r.NBOOT,len(cps)));out=[]
    for metric in r.METRICS:
        diff=np.array([sum(w*ix[cp,m][metric] for m,w in weights.items()) for cp in cps])
        low,high=np.quantile(diff[draws].mean(1),[.025,.975]);favorable=-diff if metric=='forgetting_gap' else diff
        out.append(dict(comparison=name,metric=metric,difference_pp=float(diff.mean()),ci_low_pp=float(low),ci_high_pp=float(high),
            n_checkpoints=len(cps),wins=int((favorable>1e-10).sum()),ties=int((abs(favorable)<=1e-10).sum()),losses=int((favorable< -1e-10).sum()),
            bootstrap_resamples=r.NBOOT,uncertainty='descriptive paired checkpoint bootstrap; shared fixed test set; not IID training runs'))
    return out


def paired(rows):
    groups={'overall':rows}
    for key in ('base','task'):
        for value in sorted({x[key] for x in rows}):groups[f'{key}/{value}']=[x for x in rows if x[key]==value]
    out=[]
    for group,subset in groups.items():
        methods=sorted({x['method'] for x in subset})
        for left,right in itertools.permutations(methods,2):
            if left=='base':continue
            out.extend(dict(group=group,left=left,right=right,**x) for x in contrast(subset,{left:1,right:-1},f'{left} minus {right}'))
    return out


def item_statistics(exp,cps):
    old=(e4.OUT,e4.BASE,e4.source_for);rows=[];stats=[]
    try:
        e4.OUT=folder(exp)
        for cp in cps:
            e4.BASE=cp['base'];e4.source_for=lambda condition,cp=cp:(cp['task'],Path(cp['source']),cp['rank'])
            a,b=e4.item_statistics(cp['condition'])
            for x in a:x.update(seed=cp['seed'],rank=cp['rank'],checkpoint=cp['condition'])
            for x in b:x.update(seed=cp['seed'],rank=cp['rank'],base=cp['base'],task=cp['task'])
            rows.extend(a);stats.extend(b)
    finally:e4.OUT,e4.BASE,e4.source_for=old
    return rows,stats


def finalize(exp,extra=None):
    p=folder(exp);lock=check(exp);cells,modules=audit_results(exp)
    if lock['statistics']=='item':rows,stats=item_statistics(exp,lock['checkpoints'])
    else:rows=points(cells,lock['checkpoints']);stats=paired(rows)
    r.tsv(p/'checkpoint_results.tsv',rows);r.tsv(p/'paired_ci.tsv',stats);r.tsv(p/'module_audit.tsv',modules)
    lines=[f'# {exp}: completed results','','Status: complete. All prespecified conditions are reported.','',
        '| Condition | Method | Target | Off | FG |','| --- | --- | ---: | ---: | ---: |']
    for x in rows:
        if x['method']!='base':lines.append(f'| {x["condition"]} | {x["method"]} | {x["target"]:.4f} | {x["off_score"]:.4f} | {x["forgetting_gap"]:.4f} |')
    lines += ['', 'Target/Off are percentages; FG/differences are pp. Off averages3 non-target families equally; FG clips each family before averaging. Commonsense is an equal8-subtask macro.','',
        'All20,000-draw paired intervals and wins/ties/losses or exact McNemar repair/damage counts are in paired_ci.tsv. '
        +('Single-seed, paired question-level intervals; no training-run inference.' if lock['statistics']=='item' else 'Checkpoint intervals are descriptive on shared tests; not IID training-run CIs.'),
        '', 'Source/config/weight SHA256, module coverage, reconstruction errors and ordered per-sample input hashes passed. '
        'All CUDA/model work ran through Slurm; at most2 allocated GPUs. Builders exited before vLLM began. '
        'Commands, timings, job IDs and logs are in commands_*.json. See protocol.md for fixed settings.']
    if extra:extra(exp,lock,rows,stats,modules,lines)
    (p/'report.md').write_text('\n'.join(lines)+'\n')
    summary=dict(status='complete',completed_utc=r.now(),new_cells=lock['expected_new_cells'],conditions=len(lock['checkpoints']),
        statistics=lock['statistics'],bootstrap_resamples=r.NBOOT,job_id=os.getenv('SLURM_JOB_ID'),gpu_count=2,
        outputs_sha256={n:r.sha(p/n) for n in ('report.md','checkpoint_results.tsv','paired_ci.tsv','coverage_audit.json','artifact_integrity_audit.json')})
    r.write(p/'summary.json',summary)
    update_completion(exp,summary)


def update_completion(exp,summary):
    p=r.OUT/'R2-summary.json';data=r.read(p) if p.exists() else dict(round=2,base_commit='9f23dfc',experiments={})
    data['experiments'][exp]=summary;data['updated_utc']=r.now();r.write(p,data)
    target=r.OUT/'completion_report.md';text=target.read_text();marker=f'<!-- {exp} -->'
    if marker not in text:
        with target.open('a') as h:
            h.write(f'\n{marker}\n\n## Round 2: {exp}\n\nStatus: {summary["status"]}; new cells: {summary.get("new_cells",0)}. '
                    f'[Full report]({exp}/report.md), [summary]({exp}/summary.json). Protocol, coverage and artifact audits are in the same directory.\n')


def publish(exp):
    require_complete(exp)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=r.ROOT,text=True).strip()
    assert not staged,'Refusing to absorb pre-existing staged work'
    paths=[str(folder(exp).relative_to(r.ROOT)),'reports/rebuttal_iclr27/completion_report.md','reports/rebuttal_iclr27/R2-summary.json']
    for name in ('R2-plan.json','R2-submission.json','R2-publication.json'):
        if (r.OUT/name).exists():paths.append(str((r.OUT/name).relative_to(r.ROOT)))
    subprocess.run(['git','add','--',*paths],cwd=r.ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=r.ROOT).returncode:
        subprocess.run(['git','commit','-m',f'Report ICLR27 second-round {exp}'],cwd=r.ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip()
    env=os.environ.copy();env['GIT_SSH_COMMAND']='ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes'
    proc=subprocess.run(['git','push','git@github.com:TTtianTT/peft-sft-lab.git','refactor/sft-chat-template'],cwd=r.ROOT,env=env,capture_output=True,text=True)
    r.write(r.OUT/'R2-publication.json',dict(experiment=exp,commit=commit,status='pushed' if proc.returncode==0 else 'push_pending',push_returncode=proc.returncode,updated_utc=r.now()))
    print('[Published]',exp,commit,'push_returncode',proc.returncode,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',choices=SCOPES);p.add_argument('--worker',choices=SCOPES);p.add_argument('--slot',type=int,choices=(0,1),default=0)
    a=p.parse_args()
    if a.build:build(a.build,a.slot)
    elif a.worker:worker(a.worker,a.slot)
