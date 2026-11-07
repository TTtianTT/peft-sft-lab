#!/usr/bin/env python3
"""Independent auxiliary targets; main Target/Off/FG retain the original protocol."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import rebuttal_iclr27 as r
import rebuttal_iclr27_e4 as e4
import rebuttal_iclr27_r2 as q
import rebuttal_iclr27_r2_a as a

EXP='R2-O12';P=q.folder(EXP);DEPS=P/'runtime/deps'
sys.path.insert(0,str(DEPS))


def environment(slot):
    q.env_setup(EXP,slot);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['PYTHONPATH']=f'{DEPS}:{r.ROOT}/src:{r.ROOT}/scripts'
    os.environ['XDG_CACHE_HOME']=str(P/'runtime/cache')
    if (P/'assets.json').exists():os.environ['MBPP_OVERRIDE_PATH']=r.read(P/'assets.json')['mbppplus']['path']


def math_correct(gold,prediction):
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    configs=[LatexExtractionConfig(),ExprExtractionConfig()]
    answer=parse('$'+str(gold)+'$',extraction_config=configs,extraction_mode='first_match')
    candidate=parse(str(prediction),extraction_config=configs,extraction_mode='first_match')
    return bool(answer and candidate and verify(answer,candidate))


def code_solution(text,entrypoint):
    from evalplus.sanitize import sanitize
    return sanitize(str(text),entrypoint=entrypoint)


def frozen_adapter(cp,method):
    if method=='original_lora':return cp['source']
    exp={'recon_0':'E1','flat_nuclear':'E0','hns_f4_s1':None}[method]
    if exp is None:
        return next(x['hns_path'] for x in r.source_rows() if x['source']==cp['source'])
    vm=r.read(r.OUT/exp/f'{cp["base"]}_variant_manifest.json')
    return next(v['path'] for v in vm['variants'] if v['train_task']==cp['task'] and int(v['seed'])==cp['seed'] and v['method']==method)


def prepare():
    if (P/'protocol.lock.json').exists():return q.check(EXP)
    assets=r.read(P/'assets.json');assert assets['status']=='complete'
    assert r.read(P/'parser_unit_tests.json')['status']=='pass'
    cps=[c for c in a.cohort() if c['task'] in ('magicoder','metamath')];assert len(cps)==12
    artifacts=[]
    previous={x['path']:x for x in r.read(r.OUT/'E0/artifact_integrity_audit.json')['artifacts']}
    original={x['source']:x for x in r.source_rows()}
    for cp in cps:
        q.attach_cache(cp,'flat_nuclear','E0');cp['methods']=list(q.STD);cp['new_methods']=[];cp['adapter_paths']={}
        for method in q.STD:
            path=Path(frozen_adapter(cp,method));cp['adapter_paths'][method]=str(path)
            weights=r.sha(path/'adapter_model.safetensors');config=r.sha(path/'adapter_config.json')
            if method=='original_lora':assert weights==original[cp['source']]['source_sha256']
            elif method=='hns_f4_s1':assert weights==original[cp['source']]['hns_weights_sha256']
            else:assert weights==previous[str(path)]['weights_sha256'] and config==previous[str(path)]['config_sha256']
            artifacts.append(dict(condition=cp['condition'],method=method,path=str(path),weights_sha256=weights,config_sha256=config))
    files=dict(r.read(q.folder('R2-O4')/'protocol.lock.json')['files'])
    for name in ('scripts/rebuttal_iclr27_r2_o12.py','tests/test_rebuttal_r2_o12.py','reports/rebuttal_iclr27/R2-O12/assets.json','reports/rebuttal_iclr27/R2-O12/parser_unit_tests.json'):
        files[name]=r.sha(r.ROOT/name)
    text='''# R2-O12: locked independent auxiliary targets

Only the original18-source cohort's12 Magicoder/MetaMath checkpoints: two bases x two tasks x seeds42/43/44. These sources were never selected using MBPP validation. No E4/R2 training-selected checkpoint is included. Reuse Original LoRA, Recon-0(E1), HNS(main) and Flat-Nuclear(E0) adapter files after SHA256 checks; no rebuilding/training/tuning.
Additional target tests only: MetaMath sources on official HuggingFaceH4/MATH-500 test500; Magicoder sources on official EvalPlus MBPP+v0.2.0 all378. Four methods x12 sources=48 adapter cells, plus two bases x two auxiliary benchmarks=4 new Base cells. All52 cells reported. Data revisions, raw SHA256, scorer versions and source hashes in assets.json. Dataset read only after protocol/parser unit tests fixed; no hyperparameter selection from these tests.
Keep main functional_hns_three_seed_20260914 Target/Off/FG definitions and cached cells unchanged; report independent_target separately. Thus the auxiliary benchmark does not silently replace a family in Off/FG or alter the main protocol. Reaudit ordered per-sample input hashes for all reused main cells and all new auxiliary cells.
Auxiliary generation uses the same native non_thinking chat rendering, greedy seed42, batch-invariant, max_model_len4096, adapter block5, max_lora_rank16, max_num_seqs2048, batched tokens65536, FLASH_ATTN, GPU memory.94, async/prefix cache off; max_new_tokens512 as the original target tasks. One model engine per GPU. All GPU work and test scoring through Slurm; total at most2 GPUs.
MATH user instruction is the official dataset eval.yaml boxed-answer prompt followed by the problem. Scoring uses math-verify0.9.0 with default LatexExtractionConfig and ExprExtractionConfig, first_match, gold wrapped in math delimiters; verify exact mathematical equivalence with the library defaults. Empty/unparseable answers fail, no LLM judge. MBPP+ prompt asks for complete Python code for the official problem prompt. Postprocess with unmodified EvalPlus0.3.1 sanitize(entrypoint); official evaluate(dataset=mbpp, base_only=False, parallel32, version=v0.2.0) defaults, pass only if both base and plus status pass. Do not substitute base-only MBPP scoring. Unit tests cover nested/last boxed answers, wrong/empty/fraction/multiple answers and fenced Python entrypoint/dependency handling before any model test inference.
Paired checkpoint bootstrap20000, seed20261106,95% descriptive intervals and favorable wins/ties/losses, shared fixed tests, not IID training-run CIs. Also provide exact McNemar repair/damage and paired question-level bootstrap20000 separately per source/auxiliary benchmark. All main Target/Off/FG comparisons retained. Weight/config SHA, reused reconstruction/coverage audits, input/score/row coverage hashes and new predictions all checked.
Sources: https://huggingface.co/datasets/HuggingFaceH4/MATH-500 ; https://github.com/evalplus/evalplus ; https://github.com/huggingface/Math-Verify . The auxiliary benchmarks have separate prompts/scorers by necessity; the four main benchmarks remain unchanged.
'''
    (P/'protocol.md').write_text(text)
    lock=dict(status='frozen',created_utc=r.now(),files=files,protocol_sha256=r.sha(P/'protocol.md'),checkpoints=cps,
        artifacts=artifacts,methods=q.STD,expected_new_cells=52,statistics='checkpoint plus per-condition item',assets=assets)
    r.write(P/'protocol.lock.json',lock);return lock


MATH_INSTRUCTION="Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{ANSWER}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering."


def benchmark_data(task):
    assets=r.read(P/'assets.json')
    if task=='metamath':
        path=Path(assets['math500']['path']);assert r.sha(path)==assets['math500']['sha256']
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        records=[dict(id=str(x.get('unique_id',f'math500-{i}')),problem=x['problem'],gold=x['answer']) for i,x in enumerate(rows)]
        users=[MATH_INSTRUCTION+'\n\n'+x['problem'] for x in rows]
    else:
        from evalplus.data import get_mbpp_plus
        problems=get_mbpp_plus(version='v0.2.0');keys=sorted(problems)
        records=[dict(id=k,problem=problems[k]['prompt'],entry_point=problems[k]['entry_point']) for k in keys]
        users=['Write a complete Python solution for the following problem. Return only the Python code.\n\n'+problems[k]['prompt'] for k in keys]
    assert len(records)==(500 if task=='metamath' else 378)
    return records,users


def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));temporary.replace(path)


def generate(base):
    r.require_slurm();environment(base);lock=q.check(EXP);cps=[c for c in lock['checkpoints'] if c['base']==base]
    from finetune.eval.generation import load_eval_tokenizer,render_chat_prompt
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    tokenizer=load_eval_tokenizer(base_model=cps[0]['base_model'],adapter_dir=None)
    llm=LLM(model=cps[0]['base_model'],tensor_parallel_size=1,enable_lora=True,max_lora_rank=16,max_loras=5,max_cpu_loras=24,
        max_model_len=4096,gpu_memory_utilization=.94,attention_backend='FLASH_ATTN',async_scheduling=False,enable_prefix_caching=False,
        disable_custom_all_reduce=True,max_num_seqs=2048,max_num_batched_tokens=65536,seed=42)
    adapter_ids={(c['condition'],m):i for i,(c,m) in enumerate(itertools.product(cps,q.STD),start=1)}
    for task in ('magicoder','metamath'):
        records,users=benchmark_data(task)
        prompts=[render_chat_prompt(tokenizer=tokenizer,base_model=cps[0]['base_model'],user_content=user,chat_template_mode='non_thinking') for user in users]
        items=[('base',None)]
        items += [(q.label(c,m),LoRARequest(q.label(c,m),adapter_ids[c['condition'],m],c['adapter_paths'][m])) for c in cps if c['task']==task for m in q.STD]
        for start in range(0,len(items),5):
            block=[x for x in items[start:start+5] if not (P/'runtime/aux'/base/task/x[0]/'predictions.jsonl').exists()]
            if not block:continue
            inputs=[prompt for _,request in block for prompt in prompts]
            requests=[request for _,request in block for prompt in prompts]
            output=llm.generate(inputs,SamplingParams(temperature=0,max_tokens=512,seed=42),lora_request=requests,use_tqdm=False)
            for i,(name,_) in enumerate(block):
                chunk=output[i*len(records):(i+1)*len(records)];assert len(chunk)==len(records)
                rows=[dict(**rec,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),prediction_text=x.outputs[0].text,
                    token_ids=x.outputs[0].token_ids,finish_reason=x.outputs[0].finish_reason) for rec,prompt,x in zip(records,prompts,chunk)]
                write_jsonl(P/'runtime/aux'/base/task/name/'predictions.jsonl',rows)
            print('[Aux generated]',base,task,start,flush=True)


def score(base):
    r.require_slurm();environment(base);lock=q.check(EXP)
    from evalplus.evaluate import evaluate
    for task in ('magicoder','metamath'):
        paths=sorted((P/'runtime/aux'/base/task).glob('*/predictions.jsonl'))
        assert len(paths)==13,(base,task,len(paths))
        for path in paths:
            dest=path.parent
            if (dest/'COMPLETE').exists():continue
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            if task=='metamath':bits=[math_correct(x['gold'],x['prediction_text']) for x in rows]
            else:
                solutions=[dict(task_id=x['id'],solution=code_solution(x['prediction_text'],x['entry_point'])) for x in rows]
                sample=dest/'solutions.jsonl';write_jsonl(sample,solutions)
                evaluate(dataset='mbpp',samples=str(sample),base_only=False,parallel=32,version='v0.2.0')
                result=r.read(dest/'solutions_eval_results.json')['eval']
                assert set(result)=={x['id'] for x in rows}
                bits=[]
                for row in rows:
                    entry=result[row['id']];assert len(entry)==1
                    bits.append(entry[0]['base_status']=='pass' and entry[0]['plus_status']=='pass')
            write_jsonl(dest/'scored.jsonl',[dict(**x,correct=bool(bit)) for x,bit in zip(rows,bits)])
            r.write(dest/'metrics.json',dict(samples=len(rows),correct=sum(bits),accuracy=float(np.mean(bits)),primary_metric='accuracy'))
            (dest/'COMPLETE').write_text(r.now()+'\n')


def worker(base):
    r.require_slurm();environment(base)
    for mode in ('--generate','--score'):
        r.run_command([sys.executable,__file__,mode,base],P/f'runtime/logs/{base}_{mode[2:]}.log',P/f'commands_{base}.json')


def finalize():
    lock=q.check(EXP);bases=q.base_inputs(lock['checkpoints']);maincells=[];aux=[];stats=[]
    for cp in lock['checkpoints']:
        for method in ('base',*q.STD):
            for task in r.TASKS:
                name='base' if method=='base' else q.label(cp,method)
                maincells.append(r.audit_cell(dict(condition=cp['condition'],base=cp['base'],train_task=cp['task'],seed=cp['seed'],
                    method=method,eval_task=task,metrics_path=str(P/'runtime/eval'/cp['condition']/task/name/'metrics.json')),bases))
    points=q.points(maincells,lock['checkpoints']);stats=q.paired(points)
    for cp in lock['checkpoints']:
        matrices=[];ids=None;reference=None
        for method in ('base',*q.STD):
            name='base' if method=='base' else q.label(cp,method);dest=P/'runtime/aux'/cp['base']/cp['task']/name
            assert (dest/'COMPLETE').exists();raw=[json.loads(x) for x in (dest/'predictions.jsonl').read_text().splitlines()]
            scored=[json.loads(x) for x in (dest/'scored.jsonl').read_text().splitlines()]
            assert len(raw)==len(scored)==(500 if cp['task']=='metamath' else 378)
            current=[(x['id'],hashlib.sha256(json.dumps({k:v for k,v in x.items() if k not in r.OUTPUT_FIELDS},sort_keys=True).encode()).hexdigest()) for x in raw]
            if reference is None:reference=current
            assert current==reference and [x['id'] for x in raw]==[x['id'] for x in scored]
            bits=np.array([int(x['correct']) for x in scored]);assert np.isin(bits,[0,1]).all()
            metric=r.read(dest/'metrics.json');assert abs(metric['accuracy']-float(bits.mean()))<1e-12
            matrices.append(bits)
            aux.append(dict(condition=cp['condition'],base=cp['base'],task=cp['task'],seed=cp['seed'],method=method,
                independent_target=float(bits.mean()*100),samples=len(bits),metrics_path=str(dest/'metrics.json'),
                **{f'{name}_sha256':r.sha(dest/f'{name}.jsonl') for name in ('predictions','scored')},metrics_sha256=r.sha(dest/'metrics.json')))
        bits=np.array(matrices).T;patterns=((np.arange(32)[:,None]>>np.arange(5))&1).astype(float)
        counts=np.bincount(bits@(2**np.arange(5)),minlength=32)
        draws=np.random.default_rng(r.BOOT_SEED).multinomial(len(bits),counts/len(bits),size=r.NBOOT)@patterns/len(bits)*100
        for i,j in itertools.combinations(range(5),2):
            delta=draws[:,j]-draws[:,i];low,high=np.quantile(delta,[.025,.975])
            repairs=int(((bits[:,j]==1)&(bits[:,i]==0)).sum());damage=int(((bits[:,j]==0)&(bits[:,i]==1)).sum())
            stats.append(dict(condition=cp['condition'],left=('base',*q.STD)[j],right=('base',*q.STD)[i],
                metric='independent_target',statistical_unit='question; fixed checkpoint',difference_pp=float((bits[:,j]-bits[:,i]).mean()*100),
                ci_low_pp=float(low),ci_high_pp=float(high),repaired=repairs,damaged=damage,mcnemar_exact_p=e4.mcnemar_exact(repairs,damage),bootstrap_resamples=r.NBOOT))
    ix={(x['condition'],x['method']):x for x in aux}
    for row in points:row['independent_target']=ix[row['condition'],row['method']]['independent_target']
    old=r.METRICS
    try:
        r.METRICS=('independent_target',)
        stats += [dict(statistical_unit='checkpoint; descriptive shared tests',**x) for x in q.paired(points)]
    finally:r.METRICS=old
    for artifact in lock['artifacts']:
        path=Path(artifact['path']);assert r.sha(path/'adapter_model.safetensors')==artifact['weights_sha256']
        assert r.sha(path/'adapter_config.json')==artifact['config_sha256']
    unique={x['metrics_path'] for x in aux};assert len(unique)==52
    r.tsv(P/'evaluation_cells.tsv',aux);r.tsv(P/'main_reused_cells.tsv',maincells);r.tsv(P/'checkpoint_results.tsv',points);r.tsv(P/'paired_ci.tsv',stats)
    r.write(P/'coverage_audit.json',dict(status='pass',conditions=12,new_cells=52,expected_new_cells=52,ordered_input_mismatches=0,main_reused_cells=len(maincells)))
    previous={e:r.read(r.OUT/e/'artifact_integrity_audit.json') for e in ('E0','E1')}
    r.write(P/'artifact_integrity_audit.json',dict(status='pass',artifacts=lock['artifacts'],
        reconstruction_audit_references={e:dict(path=f'../{e}/artifact_integrity_audit.json',sha256=r.sha(r.OUT/e/'artifact_integrity_audit.json'),status=x['status']) for e,x in previous.items()},
        original_main_hns_audit_sha256=r.sha(r.REF/'artifact_integrity_audit.json')))
    lines=['# R2-O12: independent auxiliary targets','','Status: complete. No checkpoint/hyperparameter was selected on these tests.',
        '','| Source task | Method | Auxiliary target | Main Target | Main Off | Main FG |','| --- | --- | ---: | ---: | ---: | ---: |']
    for task in ('magicoder','metamath'):
        for method in ('base',*q.STD):
            subset=[x for x in points if x['task']==task and x['method']==method]
            lines.append('| '+task+' | '+method+' | '+' | '.join(f'{np.mean([x[m] for x in subset]):.4f}' for m in ('independent_target',*r.METRICS))+' |')
    lines+=['','Auxiliary target is MBPP+378 for Magicoder and MATH-500500 for MetaMath. Main Target/Off/FG reuse the unchanged four-family protocol after ordered input-hash audits. '
        'Original main-source checkpoints were never selected on MBPP validation; E4/R2-selected sources are excluded. '
        'Official EvalPlus base+plus pass and math-verify scoring; parser unit tests passed before inference. '
        'All20,000 paired checkpoint intervals are descriptive on shared tests, not IID training-run CIs. Per-condition item bootstrap/McNemar repairs/damage are separately labeled in paired_ci.tsv. '
        'All52 new auxiliary cells, cached main cells and artifact hashes passed coverage/integrity audits. All evaluation/scoring ran through Slurm with2 GPUs total.']
    (P/'report.md').write_text('\n'.join(lines)+'\n')
    summary=dict(status='complete',completed_utc=r.now(),new_cells=52,conditions=12,job_id=os.getenv('SLURM_JOB_ID'),gpu_count=2,
        outputs_sha256={n:r.sha(P/n) for n in ('report.md','checkpoint_results.tsv','paired_ci.tsv','coverage_audit.json','artifact_integrity_audit.json')})
    r.write(P/'summary.json',summary);q.update_completion(EXP,summary)


def pipeline():
    r.require_slurm();prepare();q.prime_cache(EXP);devices=os.getenv('CUDA_VISIBLE_DEVICES','').split(',');assert len(devices)==2
    states=[]
    for base,device in zip(r.BASES,devices):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device
        log=P/f'runtime/logs/{base}_worker.log';log.parent.mkdir(parents=True,exist_ok=True);h=log.open('a')
        proc=subprocess.Popen([sys.executable,__file__,'--worker',base],cwd=r.ROOT,env=env,stdout=h,stderr=subprocess.STDOUT);states.append((proc,h))
    codes=[]
    for proc,h in states:codes.append(proc.wait());h.close()
    assert not any(codes),codes
    finalize();q.publish(EXP)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--pipeline',action='store_true');p.add_argument('--worker',choices=r.BASES);p.add_argument('--generate',choices=r.BASES);p.add_argument('--score',choices=r.BASES)
    args=p.parse_args()
    if args.prepare:prepare()
    elif args.pipeline:pipeline()
    elif args.worker:worker(args.worker)
    elif args.generate:generate(args.generate)
    elif args.score:score(args.score)
