#!/usr/bin/env python3
"""Cache-first Section 5 package. CPU audit/statistics first; Scalar-F last.

No training, calibration or metric search. Large artifacts stay in ignored
runtime/ or the existing source locations; committed tables contain counts.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import hns_energy_matched as hns
from analyze_hns_energy_matched import macro, transition_metrics, table, load_tsv

ROOT=hns.ROOT
DEST=ROOT/'reports/analysis_section5'
OLD=hns.OUT
LATEST=hns.LATEST
RUNTIME=DEST/'runtime'
METHODS=('hns_f4_s1','scalar_e','flat_e')
PAIRS=tuple(itertools.combinations(METHODS,2))
SEED=20260914
NBOOT=2000
CS=('arc_challenge','arc_easy','boolq','hellaswag','openbookqa','piqa','siqa','winogrande')
BENCHES=(*hns.TASKS[:3],*CS)
RR_METRICS=('recovery_rate','retention_rate','new_success_rate','new_damage_rate')


def dump(name,obj): hns.write(DEST/name,obj)
def save(name,rows): hns.tsv(DEST/name,rows)
def family(benchmark): return 'commonsense' if benchmark in CS else benchmark
def archived(name): return OLD/name if (OLD/name).exists() else DEST/name


def defined_mean(values):
    x=np.asarray(values,dtype=float); good=np.isfinite(x)
    num=np.where(good,x,0).sum(0); den=good.sum(0)
    return np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0)


def eight_counts(b,l,e):
    b,l,e=(np.asarray(x,dtype=np.int64) for x in (b,l,e))
    if b.shape!=l.shape or b.shape!=e.shape or any(not np.isin(x,[0,1]).all() for x in (b,l,e)):
        raise ValueError('Correctness bits must be aligned and binary')
    c=np.bincount(4*b+2*l+e,minlength=8)
    return {f'n_{i:03b}':int(c[i]) for i in range(8)}


def counts_rates(c):
    """Last axis is B,L,H,S,F 32-state counts; rates use original definitions."""
    patterns=np.arange(32)
    b=(patterns>>4)&1; l=(patterns>>3)&1
    total=c.sum(-1)
    values=[]
    for bit in (2,1,0):
        e=(patterns>>bit)&1
        def ratio(mask,den):
            num=c[...,mask].sum(-1); denom=c[...,den].sum(-1)
            return np.divide(num,denom,out=np.full_like(num,np.nan,dtype=float),where=denom>0)
        values.append(np.stack([ratio((b==1)&(l==0)&(e==1),(b==1)&(l==0)),
            ratio((b==0)&(l==1)&(e==1),(b==0)&(l==1)),
            np.divide(c[...,(b==0)&(l==0)&(e==1)].sum(-1),total),
            np.divide(c[...,(b==1)&(l==1)&(e==0)].sum(-1),total)],axis=-1))
    return np.stack(values,axis=-2)


def interval(values):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return None,None,0
    q=np.quantile(x,[.025,.975])
    return float(q[0]),float(q[1]),len(x)


def audit():
    DEST.mkdir(parents=True,exist_ok=True)
    old=hns.read(OLD/'manifest.json'); final=hns.read(OLD/'final_audit.json')
    assert final['status']=='pass' and old['status']=='complete'
    assert len(old['checkpoints'])==18
    assert {(c['base'],c['task'],c['seed']) for c in old['checkpoints']}=={
        (b,t,s) for b in hns.BASES for t in hns.TASKS[:3] for s in (42,43,44)}
    assert hns.sha(OLD.with_suffix('.md'))==final['report_sha256']
    for name,digest in final['outputs'].items(): assert hns.sha(OLD/name)==digest,(name,'changed')
    for name,digest in old['code_sha256'].items(): assert hns.sha(ROOT/name)==digest,(name,'changed evaluation code')
    paths=[]; sources=[]
    for cp in old['checkpoints']:
        for file,key in [('adapter_model.safetensors','source_sha256'),('adapter_config.json','config_sha256')]:
            assert hns.sha(Path(cp['source'])/file)==cp[key]
        ampath=hns.ACT/'activations'/cp['base']/cp['task']/f'seed{cp["seed"]}.json'
        am=hns.read(ampath); assert am['source_sha256']==cp['source_sha256']
        assert hns.sha(am['npz'])==am['npz_sha256']
        sources.append({k:cp[k] for k in ('base','task','seed','source','source_sha256','config_sha256','hns_path','module_count')})
        sources[-1].update(activation_manifest=str(ampath),activation_sha256=hns.sha(ampath),
            activation_npz=am['npz'],activation_npz_sha256=am['npz_sha256'],
            hns_weights_sha256=hns.sha(Path(cp['hns_path'])/'adapter_model.safetensors'),
            hns_metadata_sha256=hns.sha(Path(cp['hns_path'])/'spectral_edit_meta.json'))
        for method in ('scalar_e','flat_e'):
            adapter=OLD/'adapters'/cp['base']/cp['task']/f'seed{cp["seed"]}'/method
            meta=hns.read(adapter/'energy_match_meta.json')
            assert hns.sha(adapter/'adapter_model.safetensors')==meta['weights_sha256']
            assert meta['source_sha256']==cp['source_sha256']
            assert meta['activation_npz_sha256']==am['npz_sha256']
            assert hns.sha(adapter/'adapter_config.json')==cp['config_sha256']
            sources[-1][method]=dict(path=str(adapter),weights_sha256=meta['weights_sha256'],
                metadata_sha256=hns.sha(adapter/'energy_match_meta.json'))
    for row in load_tsv(OLD/'evaluation_matrix.tsv'):
        if row['method'] not in (*METHODS,'base','original_lora'): continue
        folder=Path(row['metrics_path']).parent
        for file,key in [('metrics.json','metrics_sha256'),('predictions.jsonl','predictions_sha256'),('scored.jsonl','scored_sha256')]:
            expected=row.get(key)
            if expected: assert hns.sha(folder/file)==expected,(folder,file)
        paths.append(dict(**row,actual_directory=str(folder.resolve())))
    # All primary caches must have the same ordered sample content as Base.
    # Ignore prediction/scoring-only fields: original input fields are fixed.
    base_inputs={}; checked=[]
    for row in paths:
        folder=Path(row['actual_directory']); key=row['base'],row['eval_task']
        inputs=[]
        with (folder/'predictions.jsonl').open() as f:
            for line in f:
                r=json.loads(line)
                inputs.append({k:v for k,v in r.items() if k not in ('prediction_text','token_ids','finish_reason')})
        digest=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if row['method']=='base': base_inputs[key]=digest
        checked.append(dict(base=row['base'],eval_task=row['eval_task'],method=row['method'],
            train_task=row['train_task'],seed=row['seed'],sample_input_sha256=digest,samples=len(inputs)))
        assert len(inputs)==hns.COUNTS[row['eval_task']]
    for row in checked: assert row['sample_input_sha256']==base_inputs[row['base'],row['eval_task']]
    save('cache_input_audit.tsv',checked); save('evaluation_sources.tsv',paths); dump('source_checkpoints.json',sources)
    names=['energy_matched_checkpoint_results.tsv','energy_matched_grouped_results.tsv',
        'energy_matched_paired_statistics.tsv','raw_results.tsv','raw_results.json','fpr_statistics.tsv',
        'fpr_statistics.json','fpr_checkpoint_spearman.tsv','fpr_grouped_checkpoint_spearman.tsv',
        'fpr_regression_design.tsv','fpr_regression_predictions.tsv','functional_boundary_summary.tsv',
        'functional_boundary_counterexamples.tsv','recovery_retention_by_checkpoint_benchmark.tsv',
        'recovery_retention_by_checkpoint_family.tsv','recovery_retention_scatter_data.tsv',
        'recovery_retention_summary.tsv','recovery_retention_paired_statistics.tsv','old_scalar_energy_audit.tsv']
    reused=[]
    for name in names:
        shutil.copy2(OLD/name,DEST/name)
        reused.append(dict(output=name,source=str(OLD/name),sha256=hns.sha(OLD/name),operation='reused unchanged'))
    # Recompute the analytic matching independently from q/spectra; cross-check
    # saved-factor errors against the already verified builder metadata hash.
    energy=[]
    checkpoints={(c['base'],c['task'],c['seed']):c for c in old['checkpoints']}
    hmeta={key:hns.read(Path(c['hns_path'])/'spectral_edit_meta.json')['module_stats']
           for key,c in checkpoints.items()}
    with (OLD/'energy_matched_module_audit.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            s,sigma,q=(np.asarray(json.loads(r[k])) for k in ('sigma_after','source_sigma','q'))
            ht=np.asarray(hmeta[r['base'],r['task'],int(r['seed'])][r['module']]['sigma_after'],dtype=np.float32).astype(float)
            error=float(abs(np.dot(s*s,q)/np.dot(ht*ht,q)-1))
            shape=float(np.ptp(s/sigma)) if r['method']=='scalar_e' else float(np.ptp(s))
            assert error<1e-7 and float(r['saved_energy_relative_error'])<1e-5
            energy.append({k:r[k] for k in ('base','task','seed','method','module')})
            energy[-1].update(analytic_energy_relative_error=error,saved_energy_relative_error=float(r['saved_energy_relative_error']),
                shape_spread=shape,source_basis_sha256=r['source_basis_sha256'],nuclear_ratio=float(r['nuclear_ratio']))
    assert len(energy)==8568
    save('module_energy_audit.tsv',energy)
    dump('provenance.json',dict(status='pass',source_job=1070,source_report=str(OLD.with_suffix('.md')),
        source_report_sha256=final['report_sha256'],reused=reused,reference_protocol=str(LATEST),
        evaluation_code_sha256=old['code_sha256'],cache_cells_verified=len(paths),ordered_input_audit='pass',
        module_rows=len(energy),max_analytic_energy_error=max(r['analytic_energy_relative_error'] for r in energy),
        max_saved_energy_error=max(r['saved_energy_relative_error'] for r in energy),
        scope='2 bases x 3 tasks x seeds42/43/44',training=False))
    print('[CPU audit complete]',len(paths),'cache cells',len(energy),'module controls',flush=True)


def recovery():
    groups=defaultdict(list)
    path=OLD/'recovery_retention_sample_bits.tsv.gz'
    assert hns.sha(path)==hns.read(OLD/'recovery_complete.json')['sample_bits_sha256']
    with gzip.open(path,'rt') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            groups[r['checkpoint'],r['benchmark']].append([int(r[k]) for k in ('base_correct','lora_correct',*METHODS)])
    assert len(groups)==198
    atomic=[]; conditional=[]; boot_by_cp=defaultdict(dict); point_by_cp=defaultdict(dict)
    rng=np.random.default_rng(SEED)
    def paired_rows(point,boots,meta):
        for i,j in itertools.combinations(range(3),2):
            for k,metric in enumerate(RR_METRICS):
                low,high,defined=interval(boots[:,i,k]-boots[:,j,k])
                conditional.append(dict(**meta,comparison=METHODS[i]+' minus '+METHODS[j],metric=metric,
                    mean_difference=float(point[i,k]-point[j,k]) if np.isfinite(point[i,k]-point[j,k]) else None,
                    ci_low=low,ci_high=high,bootstrap_defined=defined,bootstrap_resamples=NBOOT,
                    uncertainty='paired test samples conditional on one fixed source checkpoint'))
    for (cp,bench),bits in sorted(groups.items()):
        bits=np.asarray(bits,dtype=np.int64); code=bits@np.array([16,8,4,2,1])
        counts=np.bincount(code,minlength=32); boots=counts_rates(rng.multinomial(len(bits),counts/len(bits),size=NBOOT))
        point=counts_rates(counts); point_by_cp[cp][bench]=point; boot_by_cp[cp][bench]=boots
        b,t,seed=cp.split('/'); seed=int(seed.removeprefix('seed'))
        for i,method in enumerate(METHODS):
            atomic.append(dict(checkpoint=cp,base=b,task=t,seed=seed,benchmark=bench,family=family(bench),method=method,
                **eight_counts(bits[:,0],bits[:,1],bits[:,i+2]),
                **transition_metrics(bits[:,0],bits[:,1],bits[:,i+2])))
        paired_rows(point,boots,dict(checkpoint=cp,benchmark=bench,level='atomic_benchmark'))
    fam=[]; cp_rows=[]
    for cp in sorted(point_by_cp):
        fp={}; fb={}
        for bench in hns.TASKS:
            sub=[s for s in BENCHES if family(s)==bench]
            fp[bench]=defined_mean([point_by_cp[cp][s] for s in sub])
            fb[bench]=defined_mean([boot_by_cp[cp][s] for s in sub])
            paired_rows(fp[bench],fb[bench],dict(checkpoint=cp,benchmark=bench,level='family_macro'))
            for method in METHODS:
                subset=[r for r in atomic if r['checkpoint']==cp and r['method']==method and r['family']==bench]
                row=macro(subset,dict(checkpoint=cp,base=subset[0]['base'],task=subset[0]['task'],seed=subset[0]['seed'],
                    method=method,benchmark=bench))
                row.update({f'n_{i:03b}':sum(r[f'n_{i:03b}'] for r in subset) for i in range(8)})
                fam.append(row)
        for role in ('all','target','off_task'):
            src=cp.split('/')[1]
            benches=[b for b in hns.TASKS if role=='all' or (role=='target' and b==src) or (role=='off_task' and b!=src)]
            paired_rows(defined_mean([fp[b] for b in benches]),defined_mean([fb[b] for b in benches]),
                dict(checkpoint=cp,benchmark=role,level='checkpoint_macro'))
            for method in METHODS:
                subset=[r for r in fam if r['checkpoint']==cp and r['method']==method and r['benchmark'] in benches]
                row=macro(subset,dict(checkpoint=cp,method=method,role=role))
                row.update({f'n_{i:03b}':sum(r[f'n_{i:03b}'] for r in subset) for i in range(8)})
                cp_rows.append(row)
    save('transitions8_checkpoint_benchmark.tsv',atomic); save('transitions8_checkpoint_family.tsv',fam)
    save('transitions8_checkpoint_macro.tsv',cp_rows); save('recovery_paired_sample_ci.tsv',conditional)
    pooled=[]; paired=[]
    def summarize(rows,label):
        ix={(r['checkpoint'],r['method']):r for r in rows}; cps=sorted({r['checkpoint'] for r in rows})
        for method in METHODS:
            selected=[ix[cp,method] for cp in cps]
            row=macro(selected,dict(group=label,method=method))
            row.update({f'n_{i:03b}':sum(r[f'n_{i:03b}'] for r in selected) for i in range(8)})
            pooled.append(row)
        for left,right in PAIRS:
            for metric in RR_METRICS:
                d=[ix[cp,left][metric]-ix[cp,right][metric] for cp in cps
                    if ix[cp,left][metric] is not None and ix[cp,right][metric] is not None]
                x=np.asarray(d); ci=interval(x[rng.integers(0,len(x),size=(NBOOT,len(x)))].mean(1)) if len(x) else (None,None,0)
                paired.append(dict(group=label,comparison=left+' minus '+right,metric=metric,n_checkpoints=len(cps),
                    n_defined=len(x),mean_difference=float(x.mean()) if len(x) else None,ci_low=ci[0],ci_high=ci[1],
                    uncertainty='source-checkpoint bootstrap on fixed test set; descriptive run heterogeneity'))
    for bench in BENCHES: summarize([r for r in atomic if r['benchmark']==bench],'benchmark/'+bench)
    for bench in hns.TASKS: summarize([r for r in fam if r['benchmark']==bench],'family/'+bench)
    for role in ('all','target','off_task'): summarize([r for r in cp_rows if r['role']==role],'macro/'+role)
    for base in hns.BASES: summarize([r for r in cp_rows if r['role']=='all' and r['checkpoint'].startswith(base+'/')],'macro/base/'+base)
    for task in hns.TASKS[:3]: summarize([r for r in cp_rows if r['role']=='all' and r['checkpoint'].split('/')[1]==task],'macro/source/'+task)
    save('transitions8_summary.tsv',pooled); save('recovery_paired_checkpoint_ci.tsv',paired)
    dump('transitions8_summary.json',pooled)
    dump('recovery_statistics_protocol.json',dict(status='complete',source_bits=str(path),source_bits_sha256=hns.sha(path),
        transitions='n_BLE: three binary correctness bits; all eight cells explicitly stored, including zero counts',
        recovery='n_101/(n_100+n_101)',retention='n_011/(n_010+n_011)',
        new_success='n_001',new_damage='n_110',zero_denominator='NA, not zero; defined counts reported',
        supplemental_prevalences='new_success_rate=n_001/n_samples; new_damage_rate=n_110/n_samples, not conditional opportunity rates',
        macro='equal CS subbenchmarks, equal four families, equal 18 checkpoints; target/off separately',
        checkpoint_bootstrap='existing protocol: resample paired source checkpoints; fixed test set; not iid training claims',
        sample_bootstrap='2000 multinomial draws of the joint 32 states B,L,H,S,F per atomic benchmark and checkpoint; edits share each draw. Benchmarks independently resampled for each checkpoint macro. No aggregate sample CI across checkpoints: test questions are shared across runs.',
        sample_bootstrap_baseline='each draw recomputes opportunity-set denominators; undefined draw omitted, coverage output',
        seed=SEED,resamples=NBOOT,new_inference_calls=0,atomic_rows=len(atomic),conditional_ci_rows=len(conditional)))
    print('[Eight transitions / paired CIs complete]',len(atomic),flush=True)


def performance_pairs(rows,prefix):
    paired=[]; rng=np.random.default_rng(SEED)
    methods=tuple(dict.fromkeys(r['method'] for r in rows))
    ix={(r['checkpoint'],r['method']):r for r in rows}
    differences=[]
    for left,right in itertools.combinations(methods,2):
        for cp in sorted({r['checkpoint'] for r in rows}):
            a,b=ix[cp,left],ix[cp,right]
            differences.append(dict(checkpoint=cp,base=a['base'],task=a['task'],seed=a['seed'],
                comparison=left+' minus '+right,**{k:a[k]-b[k] for k in ('target','off_score','forgetting_gap')}))
    selectors={'overall':lambda r:True}
    selectors.update({'base/'+b:lambda r,b=b:r['base']==b for b in hns.BASES})
    selectors.update({'source/'+t:lambda r,t=t:r['task']==t for t in hns.TASKS[:3]})
    selectors.update({'base_source/'+b+'/'+t:lambda r,b=b,t=t:r['base']==b and r['task']==t for b in hns.BASES for t in hns.TASKS[:3]})
    for group,select in selectors.items():
        for left,right in itertools.combinations(methods,2):
            cps=sorted({r['checkpoint'] for r in rows if select(r)})
            for outcome in ('target','off_score','forgetting_gap'):
                d=np.array([ix[cp,left][outcome]-ix[cp,right][outcome] for cp in cps])
                low,high,_=interval(d[rng.integers(0,len(d),size=(NBOOT,len(d)))].mean(1))
                paired.append(dict(group=group,comparison=left+' minus '+right,outcome=outcome,n_checkpoints=len(cps),mean_difference=float(d.mean()),
                    ci_low=low,ci_high=high,uncertainty='source-checkpoint bootstrap; paired variants, fixed test set; subgroup CIs unadjusted descriptive'))
    save(prefix,paired)
    save(prefix.replace('_ci.tsv','_checkpoint_differences.tsv'),differences)
    dump(prefix.replace('_ci.tsv','_protocol.json'),dict(seed=SEED,resamples=NBOOT,
        estimator='equal-checkpoint mean paired method difference; fixed benchmark/family weights',
        unit='source checkpoint, with all edited variants kept together',
        resampling='existing protocol: checkpoints sampled with replacement within each reported group; not additionally stratified by seed/base/task',
        groups=list(selectors),interval='percentile 2.5/97.5; subgroup intervals unadjusted descriptive',
        uncertainty='fixed test set, source-checkpoint heterogeneity; not token/module replication or combined test-and-training uncertainty'))


def fpr_audit():
    """Independently verify LOCO with training-only feature scaling."""
    rows=hns.read(archived('raw_results.json'))
    predictions=load_tsv(archived('fpr_regression_predictions.tsv'))
    saved={(r['cohort'],r['pr_definition'],r['model'],r['outcome'],r['checkpoint'],r['method']):r['loco_prediction'] for r in predictions}
    cohorts={'A_all_versions':rows,'B_edited_only':[r for r in rows if r['method']!='original_lora'],
        'C_energy_matched':[r for r in rows if r['method'] in METHODS]}
    folds=[]; error=0.; checked=0
    from analyze_hns_energy_matched import OUTCOMES,center_by_checkpoint
    for cohort,subset in cohorts.items():
        ids=np.array([r['checkpoint'] for r in subset]); cps=sorted(set(ids))
        for holdout in cps:
            train=ids!=holdout; test=~train
            folds.append(dict(cohort=cohort,holdout_checkpoint=holdout,training_checkpoints=json.dumps([cp for cp in cps if cp!=holdout]),
                train_arms=int(train.sum()),test_arms=int(test.sum()),source_checkpoint_overlap=0,
                outcome_baseline='within-heldout-checkpoint zero deviation, not deployable absolute prediction'))
            for pr in ('raw_fpr','full_moment_pr'):
                for model,features in [('Energy',('log_energy_ratio',)),('FPR',(pr,)),('Energy+FPR',('log_energy_ratio',pr))]:
                    x=center_by_checkpoint(subset,[[r[f] for f in features] for r in subset])
                    scales=x[train].std(0); x=np.divide(x,scales,out=np.zeros_like(x),where=scales>1e-10)
                    for outcome in OUTCOMES:
                        y=center_by_checkpoint(subset,[r[outcome] for r in subset])
                        beta=np.linalg.lstsq(x[train],y[train],rcond=1e-10)[0]; pred=x[test]@beta
                        for r,value in zip([r for r,t in zip(subset,test) if t],pred):
                            delta=abs(float(value)-saved[cohort,pr,model,outcome,r['checkpoint'],r['method']])
                            error=max(error,delta); checked+=1
    assert error<1e-7,(error,'LOCO changed with train-only scale')
    save('fpr_fold_membership.tsv',folds)
    dump('fpr_leaveout_audit.json',dict(status='pass',folds=len(folds),predictions_checked=checked,
        max_train_only_scaling_prediction_error=error,overlapping_source_checkpoints=0,
        baseline='conditional within-checkpoint zero deviation; test outcome/group centering disclosed, not absolute deployment prediction',
        note='train-only scaling independently reproduces the existing unregularized linear LOCO predictions; no new model/metric selection'))


def scalar_audit():
    legacy=ROOT/'reports/hns_forgetting_20260911/git_artifacts/run_metadata'
    found=[]
    for slug in ('qwen','llama'):
        gp=legacy/f'generation_manifest__formal_{slug}.json'
        gm=hns.read(gp); vm=hns.read(gm['variant_manifest'])
        selected=[v for v in vm['variants'] if v.get('method')=='common_per_module' or 'common_per_module' in v['label']]
        found.append(dict(generation_manifest=str(gp),variant_manifest=gm['variant_manifest'],
            generation_manifest_sha256=hns.sha(gp),matching='actual HNS aligned spectrum, not ideal ExactFlat',
            historical_configuration=gm['configuration'],
            canonical_configuration=dict(gpu_memory_utilization=.94,long_max_num_seqs=2048,cs_max_num_seqs=4096,adapter_block_size=5),
            candidate_labels=[v['label'] for v in selected],training_seeds=[42],
            eligible_for_unified_18_checkpoint_results=False,reason='only three relevant seed42 tasks per base; historical memory .92/maxseq256/block8 differ from final .94/2048-or-4096/block5; old Llama cache not final uniform wave'))
    dump('scalar_f_prior_results_audit.json',dict(status='incomplete under required protocol',sources=found,
        exactflat_warning='src/finetune/spectral_edit/ablations.py scalar_shrink matches ideal nuclear-preserving ExactFlat; not Scalar-F here',
        accepted_definition='gamma_F = norm(actual HNS 4+1 t)/norm(source sigma), per module; keep source spectrum ratios and LoRA scaling',
        branch_search=['current untracked workspace and actual report paths','main','origin/archive/rebuttal-analysis-suite'],
        required_new_cells=72,priority='last, after all CPU cache-only work'))


def prepare_scalar_runtime():
    RUNTIME.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OLD/'manifest.json',RUNTIME/'manifest.json')
    for b in hns.BASES:
        shutil.copy2(OLD/f'{b}_reuse_probe_v2_audit.json',RUNTIME/f'{b}_reuse_probe_v2_audit.json')
        for row in hns.read(OLD/'manifest.json')['reference_cells']:
            if row['base']!=b: continue
            folder=Path(row['metrics_path']).parent
            link=RUNTIME/'eval/scalar_f'/b/row['eval_task']/folder.name
            link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists(): link.symlink_to(folder.resolve(),target_is_directory=True)
    dump('scalar_f_evaluation_protocol.json',dict(reference_protocol=str(LATEST),main_arguments='unchanged long2048/CS4096, block5, maxrank16, tokens4096/512/2048/8 as task-config, greedy seed42',
        reuse_probe='reuse passed job1070 v2 environment compatibility and recheck all reference hashes; no new full inference for cached arms',
        new_inference_cells=72,no_training=True,artifact_directory=str(RUNTIME),max_gpus=2))


def scalar_gpu():
    from importlib.metadata import version
    expected=hns.read(LATEST/'manifest.json')['versions']
    actual={name:version(name) for name in expected}
    assert actual==expected,('Evaluation library versions changed',actual,expected)
    for c in hns.read(RUNTIME/'manifest.json')['checkpoints']:
        meta=hns.read(Path(c['hns_path'])/'spectral_edit_meta.json')['meta']
        assert (meta['fast_steps'],meta['stable_steps'],meta['hns_strength'])==(4,1,1)
        assert meta['preserve_nuclear_norm'] and meta['scope']=='all_modules'
    dump('scalar_f_environment_audit.json',dict(status='pass',expected=expected,actual=actual,
        all_hns_targets='all_modules 4+1 strength1 nuclear-preserving',job_id=os.environ.get('SLURM_JOB_ID')))
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(',')
    if len(devices)!=2: raise RuntimeError('Scalar-F requires exactly two allocated B300s')
    processes=[]
    for b,d in zip(hns.BASES,devices):
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=d
        handle=(RUNTIME/f'{b}_worker.log').open('w')
        cmd=[sys.executable,str(ROOT/'scripts/hns_energy_matched.py'),'--worker','--base',b,'--method','scalar_f','--output_dir',str(RUNTIME)]
        processes.append((subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT),handle,b))
    failed=[]
    for proc,handle,b in processes:
        code=proc.wait(); handle.close()
        if code: failed.append((b,code))
    if failed: raise RuntimeError(f'Scalar-F failed: {failed}')


def scalar_results():
    rows=hns.read(OLD/'raw_results.json'); selected=[r for r in rows if r['method'] in METHODS]
    matrix=load_tsv(OLD/'evaluation_matrix.tsv'); base={(r['base'],r['eval_task']):float(r['score']) for r in matrix if r['method']=='base'}
    modules=[]; cache_sources=[]
    input_ref={(r['base'],r['eval_task']):r['sample_input_sha256'] for r in load_tsv(DEST/'cache_input_audit.tsv') if r['method']=='base'}
    for b in hns.BASES:
        assert hns.read(RUNTIME/f'{b}_scalar_f_complete.json')['status']=='complete'
        for r in hns.read(RUNTIME/f'{b}_scalar_f_build_summary.json'):
            r['checkpoint']=f'{b}/{r["task"]}/seed{r["seed"]}'
            for bench in hns.TASKS:
                p=RUNTIME/'eval/scalar_f'/b/bench/f'{r["task"]}__seed{r["seed"]}__scalar_f'
                m=hns.read(p/'metrics.json'); assert m['samples']==hns.COUNTS[bench]
                r[bench]=100*m[m['primary_metric']]
                if bench=='commonsense':
                    for name,values in m['per_task'].items(): r['score_'+name]=100*values['accuracy']
                inputs=[]
                with (p/'predictions.jsonl').open() as f:
                    for line in f:
                        pred=json.loads(line)
                        inputs.append({k:v for k,v in pred.items() if k not in ('prediction_text','token_ids','finish_reason')})
                digest=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                assert digest==input_ref[b,bench],(r['checkpoint'],bench,'Scalar-F input differs')
                cache_sources.append(dict(checkpoint=r['checkpoint'],base=b,eval_task=bench,method='scalar_f',
                    directory=str(p.resolve()),sample_input_sha256=digest,metrics_sha256=hns.sha(p/'metrics.json'),
                    predictions_sha256=hns.sha(p/'predictions.jsonl'),scored_sha256=hns.sha(p/'scored.jsonl')))
            r['target']=r[r['task']]; off=[t for t in hns.TASKS if t!=r['task']]
            r['off_score']=float(np.mean([r[t] for t in off])); r['forgetting_gap']=float(np.mean([max(base[b,t]-r[t],0) for t in off]))
            meta=hns.read(Path(r['path'])/'energy_match_meta.json')
            assert hns.sha(Path(r['path'])/'adapter_model.safetensors')==meta['weights_sha256']
            for name,v in meta['module_stats'].items():
                sig=np.asarray(v['source_sigma']); target=np.asarray(v['sigma_after'])
                assert np.ptp(target/sig)<1e-12 and v['saved_hns_fro_relative_error']<1e-5
                modules.append(dict(base=b,task=r['task'],seed=r['seed'],module=name,gain=v['gain'],
                    storage_roundoff_common_gain=v['storage_roundoff_common_gain'],
                    effective_common_gain=v['gain']*v['storage_roundoff_common_gain'],
                    saved_hns_fro_relative_error=v['saved_hns_fro_relative_error'],functional_energy_over_hns=v['functional_energy']/v['hns_functional_energy'],
                    source_basis_sha256=v['source_basis_sha256'],nuclear_ratio=v['nuclear_ratio']))
            selected.append(r)
    save('scalar_f_evaluation_sources.tsv',cache_sources)
    assert len(cache_sources)==72 and len(modules)==4284 and len(selected)==72
    save('scalar_f_module_audit.tsv',modules); save('four_method_checkpoint_results.tsv',selected)
    commands=[]
    for b in hns.BASES:
        commands.extend(hns.read(RUNTIME/f'commands_{b}_scalar_f.json'))
    dump('scalar_f_actual_commands.json',commands)
    performance_pairs(selected,'four_method_paired_ci.tsv')
    dump('scalar_f_complete.json',dict(status='complete',checkpoints=18,new_evaluation_cells=72,job_id=os.environ.get('SLURM_JOB_ID'),
        sample_inputs_identical_to_final_base=True,
        module_rows=len(modules),max_saved_fro_relative_error=max(r['saved_hns_fro_relative_error'] for r in modules)))


def report():
    provenance=hns.read(DEST/'provenance.json'); rows=hns.read(archived('raw_results.json'))
    full=(DEST/'scalar_f_complete.json').exists()
    if full: rows=load_tsv(DEST/'four_method_checkpoint_results.tsv')+[r for r in rows if r['method']=='original_lora']
    methods=('original_lora',*METHODS,*(['scalar_f'] if full else []))
    means=[]
    for method in methods:
        s=[r for r in rows if r['method']==method]
        means.append([method,len(s),*[float(np.mean([r[k] for r in s])) for k in ('target','off_score','forgetting_gap')]])
    counts=load_tsv(DEST/'transitions8_summary.tsv')
    pairs=load_tsv(DEST/'all_energy_method_paired_ci.tsv')
    rr=[r for r in counts if r['group']=='macro/all']
    fpr=hns.read(archived('fpr_statistics.json'))
    selected=[r for r in fpr if r['kind']=='incremental_fpr_over_energy' and r['outcome'] in ('target_gain','off_gain')]
    scalar=load_tsv(DEST/'four_method_paired_ci.tsv') if full else []
    rr_pairs=load_tsv(DEST/'recovery_paired_checkpoint_ci.tsv')
    primary=load_tsv(DEST/'energy_matched_checkpoint_results.tsv')
    old_energy=load_tsv(DEST/'old_scalar_energy_audit.tsv')
    old_ratios=np.array([r['old_fro_scalar_energy_over_hns'] for r in old_energy])
    boundary=load_tsv(DEST/'functional_boundary_summary.tsv')
    ep={(r['comparison'],r['outcome']):r for r in pairs if r.get('group','overall')=='overall'}
    hflat_off=ep['hns_f4_s1 minus flat_e','off_score']
    dump('method_definitions.json',dict(reference='actual HNS 4+1, all_modules, strength1, nuclear norm preserved',
        q='raw diagonal frozen Base source-direction second moment; no floor',
        energy='c^2 * sum(s_i^2*q_i), c is unchanged original LoRA scaling',
        scalar_e='sigma_i * sqrt(sum(t_H^2*q)/sum(sigma^2*q)); no nuclear restoration',
        flat_e='sqrt(sum(t_H^2*q)/sum(q)) on all source rank directions; no nuclear restoration',
        scalar_f='sigma_i * norm(actual t_H)/norm(sigma), modulewise; no nuclear restoration; not energy-matched',
        raw_fpr='sum(s^2*q)^2 / sum((s^2*q)^2)',
        full_moment_pr='trace(G)^2 / sum(G^2), G=diag(s)*M*diag(s), M=raw frozen Base coordinate second moment',
        predictor_aggregation='module median PR; sum module functional energies; log edited/source energy ratio',
        performance='Target source family; Off mean of other three families; eight CS benchmarks equal within CS',
        fg='mean max(Base_family - Edited_family,0) on three off families, percentage points',
        scope='all 18 source checkpoints, including seeds42/43/44; no training or indicator search'))
    scalar_conclusion=[]
    scalar_validation=[]
    if full:
        sp={r['outcome']:r for r in scalar if r['comparison']=='hns_f4_s1 minus scalar_f' and r.get('group','overall')=='overall'}
        scalar_conclusion=[f"7. 新补Scalar-F的18个checkpoint上，HNS−Scalar-F Target {sp['target']['mean_difference']:+.3f}pp（CI [{sp['target']['ci_low']:+.3f},{sp['target']['ci_high']:+.3f}]），Off {sp['off_score']['mean_difference']:+.3f}pp（CI [{sp['off_score']['ci_low']:+.3f},{sp['off_score']['ci_high']:+.3f}]）。这是同Frobenius而不是同functional energy比较，不能直接解释为单独谱形状的因果贡献。"]
        fmods=load_tsv(DEST/'scalar_f_module_audit.tsv'); fc=hns.read(DEST/'scalar_f_complete.json')
        ratios=np.array([r['functional_energy_over_hns'] for r in fmods])
        scalar_validation=[f"Scalar-F新增作业{fc['job_id']}已完成72个cells/18个checkpoint；全部输入内容/顺序与最终Base缓存一致。4284个module最大保存后Frobenius相对误差{fc['max_saved_fro_relative_error']:.3e}。它的逐module functional-energy/HNS比值min {ratios.min():.6f}、median {np.median(ratios):.6f}、max {ratios.max():.6f}，再次明确Frobenius matching不等于Energy matching。四方法成绩、完整八项CS分数及配对CI见four_method_checkpoint_results.tsv / four_method_paired_ci.tsv。",'']
    text=['# Paper Section 5 — Analysis data and audit','',
        '状态：'+('complete；Scalar-F也已完成。' if full else 'CPU数据完整；Scalar-F最后优先级待评测。'),'',
        '## 1. 来源、复用范围与协议','',
        f'主要来源是作业1070（2026-09-14 21:22完成，exit0）及 `{OLD}`。18个源checkpoint全部保留：Qwen3-8B/Llama-3.1-8B-Instruct × magicoder/metamath/tulu × seed42/43/44。未重新训练。当前分支refactor/sft-chat-template；其他未提交训练数据修改未纳入。','',
        f'复用之前核验源权重/config、HNS权重/metadata、activation NPZ、旧最终报告与56个输出、评测代码版本；对所有{provenance["cache_cells_verified"]}个必要cache cells逐样本原始输入内容和顺序计算哈希并与同Base/benchmark的Base比较。结果pass，样本数HumanEval164/GSM8K1319/IFEval541/CS22419一致。来源及真实resolve路径见evaluation_sources.tsv；源checkpoint见source_checkpoints.json；代码SHA见provenance.json。','',
        '最终统一协议来自functional_hns_three_seed_20260914；使用其最终Llama重评缓存，不混入旧pilot。long max_num_seqs2048，CS4096；adapter block5，maxrank16，max_model_len4096，max_batched_tokens65536，greedy seed42，原task-config/prompt/chat/parser/评分实现不变。1070 live兼容性短测Qwen覆盖42/43/44，Llama覆盖43/44；Llama42最终缓存做完整性核验，不将旧pilot短测当最终成绩。','',
        '操作分类：同能量成绩/FPR原始表、相关性、LOCO设计矩阵及预测全部复用，不新增推理；逐module解析能量核验、八类转换、更多粒度的配对CI为CPU重算；Scalar-F属于唯一新增评测（72 cells），不重复已有方法推理。详见provenance.json和scalar_f_evaluation_protocol.json。','',
        '## 2. 方法定义和同能量结果','',
        '固定源SVD方向，q_i=E[(v_iᵀh)²]来自冻结Base cache的原始二阶矩，无floor。令EH=Σt_H²q；Scalar-E=σ√(EH/Σσ²q)，Flat-E每方向=√(EH/Σq)。原LoRA scaling c保持，实际能量是c²EH；controls匹配后不恢复nuclear norm。HNS固定全module4+1、strength1且保持原nuclear budget。','',
        f'Implementation audit：旧Scalar/common_per_module/实际HNS版ScalarShrink匹配Frobenius，不等价于Scalar-E；理想ExactFlat版ScalarShrink也不等价。对这18个checkpoint，Frobenius匹配的逐module functional-energy/HNS比值为min {old_ratios.min():.6f}、median {np.median(old_ratios):.6f}、max {old_ratios.max():.6f}，详见old_scalar_energy_audit.tsv。因此不能将旧Scalar成绩当作同能量对照。作业1070已有严格逐module Scalar-E/Flat-E，全部直接复用，不重复评测。','',
        f'独立核验8568个control/module行：最大解析相对能量误差={provenance["max_analytic_energy_error"]:.3e}，最大保存后相对误差={provenance["max_saved_energy_error"]:.3e}（门限1e-5）。module_energy_audit.tsv记录误差、shape spread、源方向hash和核范数比；保存误差来自1070经过权重hash核验的factor审计，不冒称本次重新GPU SVD。','',
        table(['Method','n','Target %','Off %','FG pp'],means),'',
        '逐checkpoint完整表energy_matched_checkpoint_results.tsv；Base/source分组energy_matched_grouped_results.tsv。Off为另外三family等权；CS为八子benchmark等权。FG为其他三family max(Base−edited,0)等权；FG截断不替代Target/Off分析。','',
        table(['Comparison','Outcome','Mean pp','95% CI low','95% CI high'],[[r[k] for k in ('comparison','outcome','mean_difference','ci_low','ci_high')] for r in pairs if r.get('group','overall')=='overall']),
        '完整逐checkpoint方法差值及按Base/source/base×source的配对CI见all_energy_method_paired_checkpoint_differences.tsv和all_energy_method_paired_ci.tsv。分组区间未校正多重比较，只作预先固定分组的描述，不用于选择有利结论。','',
        '逐checkpoint同能量成绩（完整18×3；单位为百分数/pp）：','',
        table(['Checkpoint','Method','Target','Off','FG'],[[r[k] for k in ('checkpoint','method','target','off_score','forgetting_gap')] for r in primary]),'',
        '## 3. FPR增量解释力','',
        '固定Raw FPR/module median、Full-moment PR/module median及Energy（逐module sum，predictor=log(edited/source energy)）。A LoRA+全部7版本126 arms；B edited-only108；C HNS/Scalar-E/Flat-E54；每层18个source checkpoint。Scalar-F是新增机制baseline，未事后加到既有FPR cohort以改变预先固定分析。','',
        '指标定义固定：e_i=s_i²q_i，Raw FPR=(Σe_i)²/Σe_i²；M=E[(Vᵀh)(Vᵀh)ᵀ]，G=diag(s)Mdiag(s)，Full-moment PR=tr(G)²/||G||F²；Energy=c²Σe_i。两种PR对公共谱缩放不变；Full-moment保留坐标交叉二阶矩，不重新选择指标或module聚合。','',
        '每个checkpoint内中心化特征和outcome，OLS；LOCO整组留出一个源checkpoint，该源所有版本均不进入训练。预测的是留出checkpoint内相对偏差，不是未知checkpoint的绝对性能；基线为该checkpoint内零偏差（该中心由测试组定义，故属于条件性的组内排序任务，不是无标签部署预测）。测试组中心化和全表标准差是既有协议，线性无正则OLS的标准差缩放不改变预测；没有把测试变体用于训练拟合或挑指标。Spearman逐checkpoint后等权汇总；C Energy解析设为完全相等，秩0，不利用存储舍入拟合。','',
        table(['Layer','FPR','Outcome','Δin-sample R²','ΔLOCO R²','95% low','95% high'],[[r[k] for k in ('cohort','pr_definition','outcome','delta_r2','delta_loco_r2','delta_loco_r2_ci_low','delta_loco_r2_ci_high')] for r in selected]),'',
        '完整Energy-only/FPR-only/Energy+FPR结果与所有四family/八CS分项见fpr_statistics.tsv/json；逐checkpoint相关性、设计矩阵、留出预测和raw_results.tsv/json均附。fpr_fold_membership.tsv显式记录54个整checkpoint留出fold；fpr_leaveout_audit.json独立用仅训练数据的标准差复核全部LOCO预测，source重叠为0。FPR的稳定额外留出解释力证据不足，尤其同能量Target/Off的Raw和Full-moment留出R²均为负；不能由训练内R²增加推断泛化收益。','',
        '## 4. 八类逐题转换与Recovery/Retention','',
        'n_BLE按Base、LoRA、Edited三位正确性表示，000/001/010/011/100/101/110/111全部存储。Recovery=n101/(n100+n101)，Retention=n011/(n010+n011)；New Success=n001，New Damage=n110。Retention不是Off accuracy。','',
        '分母为对应机会集合；零分母为NA而不是0。macro先八CS等权→四family等权→18checkpoint等权；target/off分别保存。Counts跨checkpoint求和是checkpoint×sample事件数，不是唯一题目数；宏率不是这些总count的比值。补充new_success_rate/new_damage_rate均除以该benchmark全部样本数，表示prevalence，不是条件机会率。每一级defined/coverage均保存。旧统一协议中retention一词曾指Off性能，这里明确只用条件逐题Retention，不继承那一词义。','',
        table(['Method','Recovery %','Retention %','New Success count','New Damage count','Recovery n','Retention n'],[[r['method'],100*r['recovery_rate'],100*r['retention_rate'],r['new_success'],r['new_damage'],r['recovery_set_size'],r['retention_set_size']] for r in rr]),'',
        '固定macro的配对source-checkpoint CI（差值为百分点；逐benchmark区间另附TSV）：','',
        table(['Comparison','Metric','Mean pp','95% low','95% high'],[[r['comparison'],r['metric'],100*r['mean_difference'],100*r['ci_low'],100*r['ci_high']] for r in rr_pairs if r['group']=='macro/all' and r['metric'] in ('recovery_rate','retention_rate')]),'',
        '逐benchmark宏汇总（其八类counts均在同名TSV）：','',
        table(['Benchmark','Method','Recovery %','Retention %','NS','ND','Recovery n','Retention n'],[[r['group'],r['method'],None if r['recovery_rate'] is None else 100*r['recovery_rate'],None if r['retention_rate'] is None else 100*r['retention_rate'],r['new_success'],r['new_damage'],r['recovery_set_size'],r['retention_set_size']] for r in counts if r['group'].startswith('benchmark/')]),'',
        '八类计数：transitions8_checkpoint_benchmark.tsv（594行）、transitions8_checkpoint_family.tsv、transitions8_checkpoint_macro.tsv、transitions8_summary.tsv。所有方法对的逐benchmark/overall checkpoint bootstrap CI见recovery_paired_checkpoint_ci.tsv；逐checkpoint/benchmark及family、target/off宏汇总的paired sample CI见recovery_paired_sample_ci.tsv。scatter输入recovery_retention_scatter_data.tsv复用。','',
        '统计不确定性分开报告：既有2000次source-checkpoint paired bootstrap（seed20260914）描述固定测试集上的run异质性，不能视模块/token/多个编辑版本为独立重复；部分Llama seed42训练recipe provenance有保留，因此不称严格iid重复。新增conditional test-sample bootstrap每个checkpoint/benchmark联合抽样B,L,H,S,F的32态计数，方法共享每次抽样并重算分母；空集合draw丢弃并记coverage。跨benchmark分层独立抽样后按固定macro组合。因为多个checkpoint共用同一题目，不提供把checkpoint样本独立拼池的测试样本CI，也不将两类区间混为一种总体不确定性。','',
        '## 5. Scalar-F审计与补齐','',
        'src/finetune/spectral_edit/mechanism.py和旧common_per_module匹配观测HNS；src/finetune/spectral_edit/ablations.py的ScalarShrink匹配理想ExactFlatNuclear，不能混为本项。旧common_per_module在本次训练任务范围仅六个seed42源checkpoint且是历史评测wave；已有权重/预测实际路径已追溯，见scalar_f_prior_results_audit.json，未因文件名不同重跑符合协议的结果。当前最终统一wave确无完整Scalar-F，因此补齐18个checkpoint。','',
        'Scalar-F使用各module真实HNS4+1 metadata中的t_H，γF=||t_H||₂/||σ||₂，保持原谱比例、source方向和scaling；不恢复nuclear norm。与Scalar-E使用相同balanced reconstruction与存储审计；它不匹配functional energy。大权重/预测仅在ignored runtime/，轻量误差表scalar_f_module_audit.tsv和四方法成绩/配对CI提交。','',
        *scalar_validation,
        table(['Scalar-F comparison','Outcome','Mean pp','CI low','CI high'],[[r[k] for k in ('comparison','outcome','mean_difference','ci_low','ci_high')] for r in scalar if 'scalar_f' in r['comparison'] and r.get('group','overall')=='overall']) if full else 'Scalar-F评测尚未完成。','',
        '现有Functional-HNS边界数据（全部复用，未搜索alpha或新增版本；PR为checkpoint内module median后18checkpoint等权）：','',
        table(['Method','n','Raw FPR','Full-moment PR','Energy/source','Target %','Off %'],[[r[k] for k in ('method','n','raw_fpr','full_moment_pr','energy_ratio','target','off_score')] for r in boundary]),'',
        '## 6. 客观结论 / Paper-ready summary','',
        '1. 同能量下HNS−Scalar-E Target −0.616pp、Off +0.322pp，本目录checkpoint CI均跨0；不能声称HNS全面明显优于逐模块缩放。FG差−0.181pp，CI不跨0，但受截断定义限制。',
        f"2. HNS与Flat-E接近；HNS Off {hflat_off['mean_difference']:+.3f}pp（本目录CI [{hflat_off['ci_low']:+.3f},{hflat_off['ci_high']:+.3f}]），Target差区间跨0；不显著不等于等效，也不能声称形状完全无效。",
        '3. Scalar-E在PR完全不变下仍有平均Target +4.702pp/Off +2.918pp vs LoRA，说明PR不能独自编码幅度效应；不能据此给出缩放对收益的因果归因比例。',
        '4. 三层固定分析中FPR稳定附加解释力证据不足；控制能量后Raw/Full-moment PR都未产生正的Target/Off留出R²。',
        '5. HNS同时恢复部分旧能力并保留部分新能力（Recovery62.351%、Retention62.288%），不是完全回到Base；但Recovery/Retention均未证明优于Scalar-E。',
        '6. 现有Functional-HNS/Functional-Flat的54配对中，更高Raw FPR但Target更差22例、Off更差37例；不支持最大化FPR必然提升adapter，也不把能量同时变化的比较解释为FPR因果效应。','',
        *scalar_conclusion,'',
        '能量匹配限定于固定Base calibration输入轨迹，不保证edited模型所有真实输入下输出变化强度完全一样。Scalar-F作为补充baseline结果原样附表，不修改上述固定分析规则。','',
        '## 7. 复现与文件索引','',
        'CPU：`PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/complete_analysis_section5.py --stage cpu`。Scalar-F：`sbatch slurm/analysis_section5_scalar_f.slurm`（最多2 B300，两个Base并行）。完成导出：同脚本`--stage finalize`。只重生成报告可用`--stage report`。独立复核FPR留出：`--stage fpr_audit`，原始cache不在时自动使用本目录归档的轻量raw_results和regression_predictions，不需要activation/权重/完整预测。','',
        '复用FPR重新计算的原始入口是`PYTHONPATH=src:scripts .../python scripts/analyze_hns_energy_matched.py --stage fpr`（必须已有两种control完成标志）；本次没有重算或搜索指标。所有路径/哈希见provenance.json、source_checkpoints.json、evaluation_sources.tsv。方法和bootstrap实现随脚本提交，测试tests/test_analysis_section5.py。','']
    (DEST/'report.md').write_text('\n'.join(text).rstrip()+'\n')
    dump('package_audit.json',dict(status='complete' if full else 'CPU complete; Scalar-F pending',scope_checkpoints=18,
        report_sha256=hns.sha(DEST/'report.md'),created_utc=datetime.now(timezone.utc).isoformat(),
        analysis_code_sha256={name:hns.sha(ROOT/name) for name in (
            'scripts/complete_analysis_section5.py','scripts/hns_energy_matched.py',
            'scripts/analyze_hns_energy_matched.py','slurm/analysis_section5_scalar_f.slurm',
            'tests/test_hns_energy_matched.py','tests/test_analysis_section5.py')},
        files={p.name:dict(bytes=p.stat().st_size,sha256=hns.sha(p)) for p in DEST.iterdir() if p.is_file() and p.name!='package_audit.json'}))


def cpu():
    audit(); recovery(); fpr_audit(); scalar_audit()
    rows=[r for r in hns.read(OLD/'raw_results.json') if r['method'] in METHODS]
    performance_pairs(rows,'all_energy_method_paired_ci.tsv'); prepare_scalar_runtime(); report()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['cpu','fpr_audit','scalar_gpu','finalize','report'],required=True)
    args=p.parse_args()
    if args.stage=='finalize': scalar_results(); report()
    else: globals()[args.stage]()
