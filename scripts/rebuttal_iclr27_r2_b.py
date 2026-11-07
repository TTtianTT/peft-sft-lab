#!/usr/bin/env python3
"""Tier B fixed-recipe training and audited main-protocol evaluation."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import rebuttal_iclr27 as r
import rebuttal_iclr27_e4 as e4
import rebuttal_iclr27_r2 as q

EXPERIMENTS=('R2-O5','R2-O6','R2-O7','R2-O8')
LR={'magicoder':1e-5,'metamath':2e-4}
EXPECTED={'R2-O5':64,'R2-O6':32,'R2-O7':40,'R2-O8':32}


def model_path(base):
    if base=='Qwen3-8B':return e4.MODEL
    info=r.read(r.OUT/f'R2-assets/{base}.json');assert info['status']=='complete'
    return info['path']


def run_specs(exp):
    if exp=='R2-O5':items=[(e4.BASE,t,s,16,'lora') for t in LR for s in (43,44)]
    elif exp=='R2-O6':items=[(e4.BASE,'metamath',42,rank,'lora') for rank in (8,32)]
    elif exp=='R2-O7':items=[(base,'metamath',42,16,'lora') for base in ('Qwen3-1.7B','Qwen3-4B')]
    elif exp=='R2-O8':items=[(e4.BASE,t,42,16,'loraplus') for t in LR]
    else:raise ValueError(exp)
    return [dict(base=b,base_model=model_path(b),task=t,seed=s,rank=rank,alpha=2*rank,peft_method=method,
        lr=LR[t],epochs=e4.EPOCHS[t],condition=f'{b}_{t}_{method}_r{rank}_seed{s}') for b,t,s,rank,method in items]


def prepare(exp):
    p=q.folder(exp);p.mkdir(parents=True,exist_ok=True)
    path=p/'training.lock.json'
    if path.exists():return training_lock(exp)
    specs=run_specs(exp);data=r.check_lock('E4')['datasets']
    files=dict(r.read(r.OUT/'E4/protocol.lock.json')['files'])
    for name in ('scripts/rebuttal_iclr27_r2.py','scripts/rebuttal_iclr27_r2_a.py','scripts/rebuttal_iclr27_r2_b.py'):
        files[name]=r.sha(r.ROOT/name)
    for spec in specs:
        if spec['base']!=e4.BASE:
            name=f'reports/rebuttal_iclr27/R2-assets/{spec["base"]}.json';files[name]=r.sha(r.ROOT/name)
    extra={
      'R2-O5':'Two tasks magicoder/metamath x seeds43/44, fixed E4 val-best recipe. Merge seed42 E4 val-best to report3 seeds/task mean and sample SD (ddof1). Report paired checkpoint descriptive intervals as well as per-condition question intervals; shared tests, not IID test sets.',
      'R2-O6':'MetaMath ranks8/32, alpha16/64, seed42. Merge E4 rank16 val-best and predeclared rank64 final. Plot ranks8/16/32/64; no rank chosen by test results.',
      'R2-O7':'Qwen3-1.7B and Qwen3-4B MetaMath rank16 alpha32 seed42. LR2e-4 directly transferred from8B without model-specific tuning. New Base four-family cells for each model. Official pinned model revisions and SHA256 in ../R2-assets/.',
      'R2-O8':'LoRA+ uses --peft_method loraplus and the repository default --loraplus_lr_ratio20.0, explicitly audited; MetaMath and Magicoder seed42/rank16/alpha32. DoRA is excluded because its separate magnitude vector changes BA-only editing semantics; PiSSA changes the base residual, so it is not a directly comparable BA-only edit.'}[exp]
    common=(q.folder('R2-O1')/'protocol.md').read_text().split('All12 E4a')[0]
    common=common.replace('# R2-O1: locked protocol',f'# {exp}: locked protocol')
    text=common+f'''\n{extra}\n
No learning-rate search. Fixed rates: Magicoder1e-5, MetaMath2e-4; selected by E4 training-only validation. Final100% checkpoint is predeclared for every new training (both E4 winners were100%). All new adapters receive Original LoRA, Recon-0, HNS4+1 strength1 nuclear all7, Flat-Nuclear. All results reported.
Use exactly E4 held-out/excluded datasets and hashes, no validation/test leakage. Model/Trainer seed is the requested43/44 or42; dataset_seed remains42 to isolate initialization/order seed while preserving the fixed source subset. Epochs Magicoder1 / MetaMath3. Chat non_thinking; sequence4096; all7 LoRA modules; dropout0.05; bf16; gradient checkpointing; microbatch16; global batch32; padding8; cosine; fused AdamW betas0.9/0.999; weight decay0; clip1. Preserve native rank and alpha2r/scaling2.
First preprocess with the model tokenizer to determine actual filtered examples/optimizer steps. Explicit warmup_steps=ceil(10% actual total), never warmup_ratio. Verify training_args.json and on_train_begin warmup_schedule_audit.json total/actual/requested steps. Save half and final, but select final without validation or tests. Verify run_args.json seed/rank/alpha/LR/LoRA+ratio and dataset hashes. No new hyperparameter is selected by test outcomes.
Every source/config and edited adapter SHA, native rank/alpha/dtype, reconstruction, module coverage, cache input hashes checked. {EXPECTED[exp]} new main-protocol cells. Single seed conditions use paired question-level bootstrap20000 and exact McNemar repairs/damage, as E4. O5 additionally reports3 independent training seeds/task sample SD and paired checkpoint summaries. Intervals remain descriptive on shared fixed tests.
\nPredeclared runs:\n'''
    text+='\n'.join(f'- {s["condition"]}: LR{s["lr"]:g}, epochs{s["epochs"]}, rank{s["rank"]}, alpha{s["alpha"]}, seed{s["seed"]}, method{s["peft_method"]}.' for s in specs)+'\n'
    (p/'protocol.md').write_text(text)
    obj=dict(created_utc=r.now(),files=files,protocol_sha256=r.sha(p/'protocol.md'),datasets=data,runs=specs,
        expected_trainings=len(specs),expected_new_cells=EXPECTED[exp],methods=q.STD,
        selected_checkpoint='final, predeclared; no new validation selection')
    r.write(path,obj);return obj


def training_lock(exp):
    p=q.folder(exp);lock=r.read(p/'training.lock.json')
    assert r.sha(p/'protocol.md')==lock['protocol_sha256']
    for name,digest in lock['files'].items():assert r.sha(r.ROOT/name)==digest,('Frozen training dependency changed',name)
    for obj in lock['datasets'].values():
        assert r.sha(obj['train_path'])==obj['train_sha256']
        assert r.sha(obj['validation_path'])==obj['validation_sha256']
    return lock


def training_argv(exp,spec,path,warmup,save_steps):
    # Reuse E4 argument construction, overriding only predeclared model/seed/rank/LR/method.
    old=e4.MODEL;e4.MODEL=spec['base_model']
    try:cmd=e4.training_argv(spec['task'],path,spec['rank'],spec['lr'],warmup,save_steps)
    finally:e4.MODEL=old
    cmd[cmd.index('--seed')+1]=str(spec['seed'])
    cmd[cmd.index('--peft_method')+1]=spec['peft_method']
    if spec['peft_method']=='loraplus':cmd+=['--loraplus_lr_ratio','20.0']
    return cmd


def train_worker(exp,slot):
    r.require_slurm();q.env_setup(exp,slot);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    lock=training_lock(exp);p=q.folder(exp);audits=[]
    for i,spec in enumerate(lock['runs']):
        if i%2!=slot:continue
        name=spec['condition'];path=p/'runtime/training'/name;prep=p/'runtime/preprocessing'/name
        status=p/f'training_worker{slot}_status.json'
        r.write(status,dict(phase='preprocessing',condition=name,updated_utc=r.now(),job_id=os.getenv('SLURM_JOB_ID'),gpu_count=1))
        if not (prep/'preprocessing_audit.json').exists():
            r.run_command(training_argv(exp,spec,prep,0,1)+['--preprocess_only'],p/f'runtime/logs/{name}_preprocess.log',p/f'training_commands_{slot}.json')
        total=int(r.read(prep/'preprocessing_audit.json')['total_optimizer_steps']);warmup=math.ceil(.1*total)
        r.write(status,dict(phase='training',condition=name,total_steps=total,warmup_steps=warmup,updated_utc=r.now(),job_id=os.getenv('SLURM_JOB_ID'),gpu_count=1))
        if not (path/'adapter_model.safetensors').exists():
            r.run_command(training_argv(exp,spec,path,warmup,math.ceil(total/2)),p/f'runtime/logs/{name}_train.log',p/f'training_commands_{slot}.json')
        args=r.read(path/'training_args.json');actual=r.read(path/'warmup_schedule_audit.json');run=r.read(path/'run_args.json')
        assert args['warmup_steps']==actual['warmup_steps']==actual['explicit_steps_requested']==warmup
        assert actual['total_optimizer_steps']==actual['estimated_total_optimizer_steps']==total
        assert run['seed']==spec['seed'] and run['dataset_seed']==42
        assert run['r']==spec['rank'] and run['lora_alpha']==spec['alpha'] and run['lr']==spec['lr']
        assert run['peft_method']==spec['peft_method']
        if spec['peft_method']=='loraplus':assert run['loraplus_lr_ratio']==20.
        cps=sorted(path.glob('checkpoint-*'),key=lambda x:int(x.name.split('-')[-1]));assert len(cps)==2
        assert r.read(cps[-1]/'trainer_state.json')['global_step']==total
        audits.append(dict(**spec,source=str(path),total_steps=total,warmup_steps=warmup,
            job_id=os.getenv('SLURM_JOB_ID'),gpu_count=1,weights_sha256=r.sha(path/'adapter_model.safetensors'),
            config_sha256=r.sha(path/'adapter_config.json'),training_args_sha256=r.sha(path/'training_args.json'),
            warmup_audit_sha256=r.sha(path/'warmup_schedule_audit.json'),run_args_sha256=r.sha(path/'run_args.json')))
        r.write(p/f'training_worker{slot}_audit.json',audits)
        print('[Trained]',exp,name,flush=True)
    r.write(p/f'training_worker{slot}_status.json',dict(phase='complete',updated_utc=r.now(),job_id=os.getenv('SLURM_JOB_ID')))


def launch_training(exp):
    devices=os.getenv('CUDA_VISIBLE_DEVICES','').split(',');assert len(devices)==2;r.require_slurm()
    states=[]
    for slot,device in enumerate(devices):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device
        path=q.folder(exp)/f'runtime/logs/training_worker{slot}.log';path.parent.mkdir(parents=True,exist_ok=True)
        h=path.open('a');proc=subprocess.Popen([sys.executable,__file__,'--train-worker',exp,'--slot',str(slot)],cwd=r.ROOT,env=env,stdout=h,stderr=subprocess.STDOUT);states.append((proc,h))
    codes=[]
    for proc,h in states:codes.append(proc.wait());h.close()
    assert not any(codes),(exp,codes)


def freeze_evaluation(exp):
    p=q.folder(exp);lock=training_lock(exp)
    if (p/'protocol.lock.json').exists():return q.check(exp)
    audits=sum([r.read(p/f'training_worker{i}_audit.json') for i in (0,1)],[]);assert len(audits)==lock['expected_trainings']
    r.write(p/'training_audit.json',audits);cps=[]
    for audit in audits:
        keys=('base','base_model','task','seed','rank','alpha','lr','epochs','peft_method','condition')
        values={k:audit[k] for k in keys if k not in ('rank','alpha')}
        cp=q.source_record(audit['source'],**values)
        assert cp['source_sha256']==audit['weights_sha256'] and cp['config_sha256']==audit['config_sha256']
        cps.append(cp)
    if exp=='R2-O5':
        for task in LR:
            selected=r.read(e4.OUT/f'{task}_selection.json')['selected']
            cp=q.source_record(selected['path'],base=e4.BASE,base_model=e4.MODEL,task=task,seed=42,
                condition=f'{e4.BASE}_{task}_lora_r16_seed42',old_condition=f'{task}_val_best_r16',lr=LR[task],peft_method='lora')
            for m in q.STD:q.attach_cache(cp,m,'E4')
            cps.append(cp)
    if exp=='R2-O6':
        for rank in (16,64):
            source=(r.read(e4.OUT/'metamath_selection.json')['selected']['path'] if rank==16 else r.read(e4.OUT/'rank64_training_audit.json')['final'])
            old='metamath_val_best_r16' if rank==16 else 'metamath_r64'
            cp=q.source_record(source,base=e4.BASE,base_model=e4.MODEL,task='metamath',seed=42,
                condition=f'{e4.BASE}_metamath_lora_r{rank}_seed42',old_condition=old,lr=LR['metamath'],peft_method='lora')
            for m in q.STD:q.attach_cache(cp,m,'E4')
            cps.append(cp)
    for cp in cps:cp['methods']=list(q.STD);cp['new_methods']=[m for m in q.STD if m not in cp.get('cache',{})]
    obj=dict(status='frozen',created_utc=r.now(),files=lock['files'],protocol_sha256=lock['protocol_sha256'],
        checkpoints=cps,methods=q.STD,expected_new_cells=lock['expected_new_cells'],statistics='item',
        bootstrap_resamples=r.NBOOT,bootstrap_seed=r.BOOT_SEED,training_lock_sha256=r.sha(p/'training.lock.json'))
    r.write(p/'protocol.lock.json',obj);return obj


def extra(exp,lock,rows,stats,modules,lines):
    p=q.folder(exp);audit=r.read(p/'training_audit.json')
    lines+=['','All new sources use E4 validation-selected fixed learning rates; no new LR tuning. Warmup is explicit10% steps, verified in saved arguments and actual Trainer schedule. '
        'New trainings use a predeclared final checkpoint. See training_audit.json and training_commands_*.json for hashes, actual steps, GPU time/logs.']
    if exp=='R2-O5':
        aggregate=[]
        for task in LR:
            for method in ('base',*q.STD):
                subset=[x for x in rows if x['task']==task and x['method']==method];assert len(subset)==3
                assert sorted(x['seed'] for x in subset)==[42,43,44]
                aggregate.append(dict(task=task,method=method,seeds=3,
                    **{f'{metric}_{stat}':float(fn([x[metric] for x in subset])) for metric in r.METRICS
                        for stat,fn in [('mean',np.mean),('sd',lambda values:np.std(values,ddof=1))]}))
        r.tsv(p/'three_seed_mean_sd.tsv',aggregate)
        checkpoints=q.paired(rows)
        stats+= [dict(statistical_unit='checkpoint; descriptive shared tests',**x) for x in checkpoints];r.tsv(p/'paired_ci.tsv',stats)
        lines+=['','## Three training seeds','','| Task | Method | Target mean ± SD | Off mean ± SD | FG mean ± SD |',
                '| --- | --- | ---: | ---: | ---: |']
        for x in aggregate:
            lines.append('| '+x['task']+' | '+x['method']+' | '+' | '.join(f'{x[m+"_mean"]:.4f} ± {x[m+"_sd"]:.4f}' for m in r.METRICS)+' |')
        lines+=['','SD is across3 training seeds, with a fixed data subset and shared test sets; it does not measure independent test-set uncertainty. '
            'Per-seed paired item intervals and checkpoint descriptive intervals are both included with their statistical units.']
    if exp in ('R2-O6','R2-O7'):
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data=[]
        for cp in sorted(lock['checkpoints'],key=lambda c:(c['rank'],c['base'])):
            subset={x['method']:x for x in rows if x['condition']==cp['condition']}
            data.append(dict(condition=cp['condition'],base=cp['base'],rank=cp['rank'],
                target_gain=subset['hns_f4_s1']['target']-subset['recon_0']['target'],
                fg_reduction=subset['recon_0']['forgetting_gap']-subset['hns_f4_s1']['forgetting_gap'],
                flat_minus_hns_target=subset['flat_nuclear']['target']-subset['hns_f4_s1']['target']))
        name='rank_curve' if exp=='R2-O6' else 'model_size_curve';r.tsv(p/f'{name}.tsv',data)
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained');x=np.arange(len(data))
        for metric,marker in [('target_gain','o'),('fg_reduction','s'),('flat_minus_hns_target','^')]:ax.plot(x,[d[metric] for d in data],marker+'-',label=metric)
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks(x,[str(d['rank']) if exp=='R2-O6' else d['base'] for d in data]);ax.set_ylabel('pp');ax.legend(fontsize=8)
        fig.savefig(p/f'{name}.png',dpi=180);fig.savefig(p/f'{name}.pdf');plt.close(fig)
        lines+=['',f'![{name}]({name}.png)','',
            'Single seed42; per-condition paired question-level intervals. '+('Ranks16/64 reuse E4 audited cells.' if exp=='R2-O6' else 'Small-model learning rate was transferred from8B without separate tuning; both Base results were newly evaluated.')]
    if exp=='R2-O8':lines+=['','LoRA+ uses the repository default LR ratio20.0, explicitly supplied and checked in run_args.json. '
        'DoRA carries an independent magnitude vector and PiSSA rewrites the base residual, so both differ from editing BA alone and are excluded as prespecified.']


def pipeline():
    r.require_slurm()
    for prior in q.SCOPES[:4]:q.require_complete(prior)
    for exp in EXPERIMENTS:
        if (q.folder(exp)/'summary.json').exists():q.require_complete(exp);continue
        prepare(exp);launch_training(exp);freeze_evaluation(exp);q.launch_workers(exp);q.finalize(exp,extra);q.publish(exp)
        print('[Complete]',exp,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--pipeline',action='store_true');p.add_argument('--train-worker',choices=EXPERIMENTS);p.add_argument('--slot',type=int,choices=(0,1),default=0)
    a=p.parse_args()
    if a.prepare:
        for exp in EXPERIMENTS:prepare(exp)
    elif a.train_worker:train_worker(a.train_worker,a.slot)
    elif a.pipeline:pipeline()
