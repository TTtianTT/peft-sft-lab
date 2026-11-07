#!/usr/bin/env python3
"""Budget-gated tier C; O11 always requires explicit author approval."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import rebuttal_iclr27 as r
import rebuttal_iclr27_e4 as e4
import rebuttal_iclr27_r2 as q
import rebuttal_iclr27_r2_b as b

EXPERIMENTS=('R2-O9','R2-O10')
ESTIMATES={'R2-O9':10.,'R2-O10':5.,'R2-O12':2.}


def specs(exp):
    if exp=='R2-O9':
        base='Llama-3.1-8B-Instruct';model=str(Path('/dataset1/zailong/models')/base)
        return [dict(base=base,base_model=model,task=t,seed=42,rank=16,alpha=32,peft_method='lora',
            lr=e4.LRS[t]*m,epochs=e4.EPOCHS[t],multiplier=m,
            condition=f'{base}_{t}_lr{m:g}') for t in b.LR for m in (.5,1.,2.)]
    if exp=='R2-O10':
        base='Qwen3-14B'
        return [dict(base=base,base_model=b.model_path(base),task='metamath',seed=42,rank=16,alpha=32,
            peft_method='lora',lr=2e-4,epochs=3,condition=f'{base}_metamath_lora_r16_seed42')]
    raise ValueError(exp)


def prepare(exp):
    p=q.folder(exp);p.mkdir(parents=True,exist_ok=True)
    if (p/'training.lock.json').exists():return b.training_lock(exp)
    runs=specs(exp);files=dict(r.read(r.OUT/'R2-O8/training.lock.json')['files'])
    name='scripts/rebuttal_iclr27_r2_c.py';files[name]=r.sha(r.ROOT/name)
    if exp=='R2-O10':
        name='reports/rebuttal_iclr27/R2-assets/Qwen3-14B.json';files[name]=r.sha(r.ROOT/name)
    common=(q.folder('R2-O5')/'protocol.md').read_text().split('Two tasks magicoder')[0]
    common=common.replace('# R2-O5: locked protocol',f'# {exp}: locked protocol')
    detail=('Repeat E4a on Llama-3.1-8B-Instruct with the same E4 held-out/excluded datasets and training recipe. '
        'LR grids Magicoder1e-5/2e-5/4e-5 and MetaMath5e-5/1e-4/2e-4. '
        'Six trainings; save50% and100%, validate all12 candidates. Select one/task solely by E4 validation score: '
        'highest score, then closest LR to1x in abs(log2 multiplier), then lower multiplier, then earlier fraction. '
        'This explicitly uses the E4 sequence4096/global-batch32 recipe for both tasks; it is not presented as a literal reproduction of older incomplete Llama model cards. '
        'Magicoder validation200 MBPP, MetaMath500 held-out training questions. Never select on main tests. '
        'Evaluate two selected sources x4 adapters x4 families=32 cells, cached Llama Base.' if exp=='R2-O9' else
        'Qwen3-14B MetaMath seed42, fixed E4-selected LR2e-4 transferred without retuning. '
        'One training; predeclared final checkpoint, rank16/alpha32. '
        'Before training run a vLLM fit check inside Slurm with main max_model_len4096, maxrank16, batch-invariant, TP1 on one B300; '
        'record physical GPU memory and engine initialization. This tests the stricter one-GPU case within the two-GPU global limit. '
        'If the fixed recipe cannot fit, report the capacity failure rather than change batches/precision/inference protocol. '
        'New Base and four adapters x4 families=20 cells.')
    text=common+'\n'+detail+'''

Explicit warmup_steps=ceil(10% actual total) after tokenizer/filtering preprocessing. Audit both saved training_args and actual Trainer warmup_schedule. E4 recipe: all7 modules, dropout.05, scaling2, bf16, gradient checkpointing, sequence4096, microbatch16, global32, padding8, cosine, fused AdamW betas.9/.999, weight_decay0, clip1, epochs Magicoder1/MetaMath3. Model/Trainer seed42 and dataset subset seed42. Save half/final. Same main eval prompts/templates/parsers/scorers; greedy seed42, maxmodel4096, block5, maxrank16, batch-invariant. Four methods Original LoRA, Recon-0, HNS4+1 strength1 nuclear all7, Flat-Nuclear. Single-seed paired question intervals20000 and exact McNemar repairs/damage, all Target/Off/FG; no training-run uncertainty claim. Cache/sample hashes, weight/config SHA256, coverage and reconstruction audit required. All configurations reported.
Budget rule: start only when prespecified runtime estimate plus1h reserve fits the remaining48h wall budget, independent of test performance. O11 is not authorized until the author explicitly confirms it.
'''
    text+='\n'.join(f'- {s["condition"]}: LR{s["lr"]:g}, epochs{s["epochs"]}, r16/alpha32.' for s in runs)+'\n'
    (p/'protocol.md').write_text(text)
    obj=dict(created_utc=r.now(),protocol_sha256=r.sha(p/'protocol.md'),files=files,
        datasets=r.read(e4.OUT/'protocol.lock.json')['datasets'],runs=runs,expected_trainings=len(runs),
        expected_new_cells=32 if exp=='R2-O9' else 20,methods=q.STD,runtime_estimate_hours=ESTIMATES[exp])
    r.write(p/'training.lock.json',obj);return obj


def preflight14():
    r.require_slurm();q.env_setup('R2-O10','fit');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    spec=specs('R2-O10')[0]
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv,noheader'],text=True)
    from vllm import LLM, SamplingParams
    llm=LLM(model=spec['base_model'],tensor_parallel_size=1,enable_lora=True,max_lora_rank=16,max_loras=5,max_cpu_loras=5,
        max_model_len=4096,gpu_memory_utilization=.94,attention_backend='FLASH_ATTN',async_scheduling=False,
        enable_prefix_caching=False,disable_custom_all_reduce=True,max_num_seqs=2048,max_num_batched_tokens=65536,seed=42)
    output=llm.generate(['Hello'],SamplingParams(temperature=0,max_tokens=8,seed=42),use_tqdm=False)
    assert output[0].outputs[0].token_ids
    r.write(q.folder('R2-O10')/'capacity_preflight.json',dict(status='pass',job_id=os.getenv('SLURM_JOB_ID'),
        devices=devices,tensor_parallel_size=1,max_model_len=4096,max_lora_rank=16,global_gpu_limit=2,
        tested='main engine initialization and greedy generation on one B300; training memory audited by actual fixed-recipe run'))


def train_worker(exp,slot):
    r.require_slurm();lock=b.training_lock(exp);p=q.folder(exp)
    if exp=='R2-O10' and slot==0 and not (p/'capacity_preflight.json').exists():
        r.run_command([sys.executable,__file__,'--preflight14'],p/'runtime/logs/capacity_preflight.log',p/'capacity_commands.json')
    b.train_worker(exp,slot)
    if not (p/f'training_worker{slot}_audit.json').exists():r.write(p/f'training_worker{slot}_audit.json',[])
    if exp=='R2-O9':
        old=(e4.OUT,e4.MODEL);e4.OUT=p;e4.MODEL=lock['runs'][0]['base_model'];candidates=[]
        try:
            for i,spec in enumerate(lock['runs']):
                if i%2!=slot:continue
                path=p/'runtime/training'/spec['condition']
                half=sorted(path.glob('checkpoint-*'),key=lambda c:int(c.name.split('-')[-1]))[0]
                for fraction,adapter in ((.5,half),(1.,path)):
                    tag=f'{spec["condition"]}_fraction{fraction:g}'
                    score,digest=e4.validate(spec['task'],adapter,tag)
                    candidates.append(dict(**spec,fraction=fraction,path=str(adapter),validation_score=score,
                        metric_sha256=digest,weights_sha256=r.sha(adapter/'adapter_model.safetensors'),config_sha256=r.sha(adapter/'adapter_config.json')))
        finally:e4.OUT,e4.MODEL=old
        r.write(p/f'worker{slot}_validation_candidates.json',candidates)


def train(exp):
    r.require_slurm();devices=os.getenv('CUDA_VISIBLE_DEVICES','').split(',');assert len(devices)==2
    states=[]
    for slot,device in enumerate(devices):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device
        path=q.folder(exp)/f'runtime/logs/train_worker{slot}.log';path.parent.mkdir(parents=True,exist_ok=True);h=path.open('a')
        proc=subprocess.Popen([sys.executable,__file__,'--train-worker',exp,'--slot',str(slot)],cwd=r.ROOT,env=env,stdout=h,stderr=subprocess.STDOUT);states.append((proc,h))
    codes=[]
    for proc,h in states:codes.append(proc.wait());h.close()
    assert not any(codes),(exp,codes)


def freeze_evaluation(exp):
    p=q.folder(exp);lock=b.training_lock(exp)
    if (p/'protocol.lock.json').exists():return q.check(exp)
    audits=sum([r.read(p/f'training_worker{i}_audit.json') for i in (0,1)],[]);assert len(audits)==lock['expected_trainings']
    r.write(p/'training_audit.json',audits)
    if exp=='R2-O9':
        candidates=sum([r.read(p/f'worker{i}_validation_candidates.json') for i in (0,1)],[]);assert len(candidates)==12
        r.tsv(p/'validation_candidates.tsv',candidates);selected=[]
        for task in b.LR:
            best=e4.choose_candidate([x for x in candidates if x['task']==task]);selected.append(best)
        r.write(p/'selection.json',dict(selected=selected,rule='E4 prespecified validation-only tie rule'))
    else:selected=[dict(**x,path=x['source']) for x in audits]
    cps=[]
    for x in selected:
        cp=q.source_record(x['path'],base=x['base'],base_model=x['base_model'],task=x['task'],seed=42,
            condition=x['base']+'_'+x['task']+'_val_best' if exp=='R2-O9' else x['condition'])
        assert cp['source_sha256']==x['weights_sha256'] and cp['config_sha256']==x['config_sha256']
        cp['methods']=list(q.STD);cp['new_methods']=list(q.STD);cps.append(cp)
    obj=dict(status='frozen',created_utc=r.now(),files=lock['files'],protocol_sha256=lock['protocol_sha256'],
        checkpoints=cps,methods=q.STD,expected_new_cells=lock['expected_new_cells'],statistics='item',
        bootstrap_resamples=r.NBOOT,bootstrap_seed=r.BOOT_SEED,training_lock_sha256=r.sha(p/'training.lock.json'))
    r.write(p/'protocol.lock.json',obj)


def extra(exp,lock,rows,stats,modules,lines):
    lines+=['','Single seed42; paired question-level intervals, not uncertainty over training runs. Explicit warmup verified against actual schedule; all candidate/training artifacts hashed.']
    if exp=='R2-O9':
        selected=r.read(q.folder(exp)/'selection.json')['selected']
        lines+=['','All12 candidate validation scores are in validation_candidates.tsv; exactly one checkpoint/task was selected by the predeclared validation-only rule.']
        for x in selected:lines.append(f'- {x["task"]}: LR{x["lr"]:g}, fraction{x["fraction"]:g}, validation{x["validation_score"]:.4f}.')
    else:lines+=['','Qwen3-14B LR2e-4 was transferred from8B without model-specific tuning. A one-B300 vLLM fit check passed before the single fixed-recipe training. '
        'Capacity details are in capacity_preflight.json; total allocated GPUs remained2.']


def pipeline():
    r.require_slurm()
    for exp in (*q.SCOPES[:4],*b.EXPERIMENTS):q.require_complete(exp)
    decisions=[]
    for exp in EXPERIMENTS:
        if (q.folder(exp)/'summary.json').exists():q.require_complete(exp);continue
        hours=q.remaining_hours()
        if hours<ESTIMATES[exp]+1:
            decisions.append(dict(experiment=exp,status='not_started_budget',remaining_hours=hours,estimate_hours=ESTIMATES[exp]));continue
        prepare(exp);train(exp);freeze_evaluation(exp);q.launch_workers(exp);q.finalize(exp,extra);q.publish(exp)
        decisions.append(dict(experiment=exp,status='complete',remaining_hours=q.remaining_hours()))
        r.write(r.OUT/'R2-C-budget.json',decisions)
    # Approval is never inferred from silence or elapsed time.
    decisions.append(dict(experiment='R2-O11',status='awaiting_author_confirmation',approval=q.initialize_round()['o11_author_approval']))
    # O11 cannot hold up already-authorized independent-target work.
    if q.remaining_hours()>=ESTIMATES['R2-O12']+1:
        import rebuttal_iclr27_r2_o12 as auxiliary
        auxiliary.pipeline()
        decisions.append(dict(experiment='R2-O12',status='complete',remaining_hours=q.remaining_hours()))
    else:
        decisions.append(dict(experiment='R2-O12',status='not_started_budget',remaining_hours=q.remaining_hours(),estimate_hours=ESTIMATES['R2-O12']))
    r.write(r.OUT/'R2-C-budget.json',decisions)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--pipeline',action='store_true');p.add_argument('--preflight14',action='store_true');p.add_argument('--train-worker',choices=EXPERIMENTS);p.add_argument('--slot',type=int,choices=(0,1),default=0)
    a=p.parse_args()
    if a.prepare:
        for exp in EXPERIMENTS:prepare(exp)
    elif a.preflight14:preflight14()
    elif a.train_worker:train_worker(a.train_worker,a.slot)
    elif a.pipeline:pipeline()
