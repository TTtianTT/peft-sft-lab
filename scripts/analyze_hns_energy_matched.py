#!/usr/bin/env python3
"""Fixed-metric paired analysis and cache-only recovery/retention audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
from datetime import datetime, timezone
import os

import numpy as np

from hns_energy_matched import OUT, ROOT, LATEST, BASES, TASKS, CONTROLS, COUNTS, read, write, tsv, sha, cells

SEED = 20260914
BOOTSTRAPS = 2000
MATCHED = ('hns_f4_s1','scalar_e','flat_e')
NAMES = dict(original_lora='LoRA',hns_f4_s1='HNS 4+1',scalar_e='Scalar-E',flat_e='Flat-E',
    functional_hns_a05='Functional-HNS α=.5',functional_hns_a10='Functional-HNS α=1',
    functional_flat='Protected Functional Flat')
FIELDS = dict(magicoder='correct',metamath='correct_strict',tulu='prompt_strict_passed',commonsense='correct')
OUTCOMES = ('target_gain','off_gain','fg_reduction',*('gain_'+t for t in TASKS),
    *('gain_'+s for s in ('arc_challenge','arc_easy','boolq','hellaswag','openbookqa','piqa','siqa','winogrande')))


def checkpoint(row):
    return f'{row["base"]}/{row["task"]}/seed{row["seed"]}'


def finite_mean(values):
    good = [v for v in values if v is not None]
    return float(np.mean(good)) if good else None


def table(headers, rows):
    def fmt(x):
        if x is None: return 'NA'
        if isinstance(x,(float,np.floating)): return f'{x:.4f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(fmt(x) for x in r)+' |' for r in rows]])


def collect_results():
    for b in BASES:
        for m in CONTROLS:
            assert read(OUT/f'{b}_{m}_complete.json')['status']=='complete', 'Both controls must finish before analysis'
    rows = read(LATEST/'summary.json')['rows']
    exported = defaultdict(list)
    with gzip.open(LATEST/'functional_pr_module_results.tsv.gz','rt') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            exported[r['base'],r['task'],int(r['seed']),r['method']].append(r)
    matrix = cells()
    base_scores = {(r['base'],r['eval_task']):float(r['score']) for r in matrix if r['method']=='base'}
    for row in rows:
        modules = exported[row['base'],row['task'],row['seed'],row['method']]
        assert modules
        row['raw_fpr']=float(np.median([float(r['functional_pr']) for r in modules]))
        row['full_moment_pr']=float(np.median([float(r['cov_functional_pr']) for r in modules]))
        row['functional_energy']=sum(float(r['functional_energy_after']) for r in modules)
        row['source_functional_energy']=sum(float(r['functional_energy_before']) for r in modules)
    for base in BASES:
        for method in CONTROLS:
            for row in read(OUT/f'{base}_{method}_build_summary.json'):
                for bench in TASKS:
                    folder = OUT/'eval'/method/base/bench/f'{row["task"]}__seed{row["seed"]}__{method}'
                    metric=read(folder/'metrics.json')
                    row[bench]=100*metric[metric['primary_metric']]
                    matrix.append(dict(base=base,train_task=row['task'],seed=row['seed'],method=method,
                        eval_task=bench,score=row[bench],samples=metric['samples'],metric=metric['primary_metric'],
                        metrics_path=str(folder/'metrics.json'),metrics_sha256=sha(folder/'metrics.json'),
                        predictions_sha256=sha(folder/'predictions.jsonl'),scored_sha256=sha(folder/'scored.jsonl'),origin='new_energy_matched'))
                row['target']=row[row['task']]
                off=[t for t in TASKS if t!=row['task']]
                row['off_score']=float(np.mean([row[t] for t in off]))
                row['forgetting_gap']=float(np.mean([max(base_scores[base,t]-row[t],0) for t in off]))
                rows.append(row)
    module_rows=[]
    for base in BASES:
        for method in CONTROLS:
            for row in read(OUT/f'{base}_{method}_build_summary.json'):
                meta=read(Path(row['path'])/'energy_match_meta.json')
                for module,values in meta['module_stats'].items():
                    record=dict(base=base,task=row['task'],seed=row['seed'],method=method,module=module,**values)
                    for key,value in list(record.items()):
                        if isinstance(value,list): record[key]=json.dumps(value,separators=(',',':'))
                    module_rows.append(record)
    tsv(OUT/'energy_matched_module_audit.tsv',module_rows)
    metric_paths = {(r['base'],r['train_task'],int(r['seed'] or 0),r['method'],r['eval_task']):r['metrics_path'] for r in matrix}
    index={(r['base'],r['task'],r['seed'],r['method']):r for r in rows}
    for row in rows:
        row['checkpoint']=checkpoint(row)
        old=index[row['base'],row['task'],row['seed'],'original_lora']
        hns=index[row['base'],row['task'],row['seed'],'hns_f4_s1']
        row['target_gain']=row['target']-old['target']
        row['off_gain']=row['off_score']-old['off_score']
        row['fg_reduction']=old['forgetting_gap']-row['forgetting_gap']
        row['energy_ratio']=row['functional_energy']/row['source_functional_energy']
        row['log_energy_ratio']=float(np.log(row['energy_ratio']))
        # A controlled equality must not become a predictor through floating point noise.
        if row['method'] in MATCHED:
            assert np.isclose(row['functional_energy'],hns['functional_energy'],rtol=1e-7)
            row['log_energy_ratio']=float(np.log(hns['functional_energy']/hns['source_functional_energy']))
        for bench in TASKS:
            row['gain_'+bench]=row[bench]-old[bench]
        path=metric_paths[row['base'],row['task'],row['seed'],row['method'],'commonsense']
        sub=read(path)['per_task']
        oldsub=read(metric_paths[row['base'],row['task'],row['seed'],'original_lora','commonsense'])['per_task']
        for key,value in sub.items():
            row['score_'+key]=100*value['accuracy']
            row['gain_'+key]=100*(value['accuracy']-oldsub[key]['accuracy'])
    assert len(rows)==126 and len({r['checkpoint'] for r in rows})==18
    write(OUT/'raw_results.json',rows)
    tsv(OUT/'raw_results.tsv',rows); tsv(OUT/'evaluation_matrix.tsv',matrix)
    primary=[r for r in rows if r['method'] in MATCHED]
    tsv(OUT/'energy_matched_checkpoint_results.tsv',primary)
    groups=[]
    selectors={'overall':lambda r:True}
    selectors.update({'base/'+b:lambda r,b=b:r['base']==b for b in BASES})
    selectors.update({'task/'+t:lambda r,t=t:r['task']==t for t in TASKS[:3]})
    selectors.update({'base_task/'+b+'/'+t:lambda r,b=b,t=t:r['base']==b and r['task']==t for b in BASES for t in TASKS[:3]})
    for group,select in selectors.items():
        for method in MATCHED:
            subset=[r for r in primary if select(r) and r['method']==method]
            entry=dict(group=group,method=method,n=len(subset))
            for key in ('target','off_score','forgetting_gap','raw_fpr','full_moment_pr','energy_ratio',*OUTCOMES):
                entry[key]=float(np.mean([r[key] for r in subset]))
            groups.append(entry)
    tsv(OUT/'energy_matched_grouped_results.tsv',groups)
    pairs=[]
    rng=np.random.default_rng(SEED)
    for method in CONTROLS:
        selected=[r for r in primary if r['method']==method]
        for key in ('target','off_score','forgetting_gap',*OUTCOMES):
            diff=np.array([index[r['base'],r['task'],r['seed'],'hns_f4_s1'][key]-r[key] for r in selected])
            ci=np.quantile(diff[rng.integers(0,18,size=(BOOTSTRAPS,18))].mean(1),[.025,.975])
            pairs.append(dict(comparison='HNS minus '+method,outcome=key,n_checkpoints=18,
                mean_difference=float(diff.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                positive=int((diff>1e-9).sum()),tie=int((abs(diff)<=1e-9).sum()),negative=int((diff< -1e-9).sum())))
    tsv(OUT/'energy_matched_paired_statistics.tsv',pairs)
    return rows


def center_by_checkpoint(rows, values):
    values=np.asarray(values,dtype=float)
    out=values.copy()
    ids=np.array([r['checkpoint'] for r in rows])
    for cp in sorted(set(ids)):
        mask=ids==cp; out[mask]-=values[mask].mean(axis=0)
    return out


def within_rho(x,y):
    # Fixed 1e-8 rounding ties are far below meaningful feature precision.
    rx=rankdata(np.round(x,8)); ry=rankdata(np.round(y,8))
    if np.ptp(rx)==0 or np.ptp(ry)==0: return None
    return float(np.corrcoef(rx,ry)[0,1])


def rankdata(values):
    """Average ranks of exact ties, implemented locally to avoid new deps."""
    _,inverse,counts=np.unique(np.asarray(values),return_inverse=True,return_counts=True)
    starts=np.cumsum(counts)-counts
    averages=starts+(counts+1)/2
    return averages[inverse]


def regression(rows, features, outcome):
    x=center_by_checkpoint(rows,[[r[f] for f in features] for r in rows])
    y=center_by_checkpoint(rows,[r[outcome] for r in rows])
    scales=x.std(0)
    x=np.divide(x,scales,out=np.zeros_like(x),where=scales>1e-10)
    beta=np.linalg.lstsq(x,y,rcond=1e-10)[0]
    predicted=x@beta
    sst=float(y@y)
    r2=float(1-np.square(y-predicted).sum()/sst) if sst>1e-15 else None
    cp_ids=np.array([r['checkpoint'] for r in rows]); cv=np.zeros(len(rows))
    records=[]
    for cp in sorted(set(cp_ids)):
        test=cp_ids==cp
        coef=np.linalg.lstsq(x[~test],y[~test],rcond=1e-10)[0]
        cv[test]=x[test]@coef
    cv_r2=float(1-np.square(y-cv).sum()/sst) if sst>1e-15 else None
    for i,r in enumerate(rows):
        records.append(dict(checkpoint=r['checkpoint'],method=r['method'],outcome=outcome,
            features='+'.join(features),observed_within_checkpoint=float(y[i]),
            fitted_within_checkpoint=float(predicted[i]),loco_prediction=float(cv[i]),
            **{f'centered_{f}':float(x[i,j]) for j,f in enumerate(features)}))
    return dict(r2=r2,loco_r2=cv_r2,rank=int(np.linalg.matrix_rank(x,tol=1e-10)),
        coefficients={f:float(beta[j]) for j,f in enumerate(features)}),records,x,y


def fixed_analysis(rows, prefix=''):
    statistics=[]; predictions=[]; correlations=[]; designs=[]
    cohorts={'A_all_versions':rows,'B_edited_only':[r for r in rows if r['method']!='original_lora'],
        'C_energy_matched':[r for r in rows if r['method'] in MATCHED]}
    if not set(MATCHED)<=set(r['method'] for r in rows):
        del cohorts['C_energy_matched']
    for cohort,subset in cohorts.items():
        cps=sorted({r['checkpoint'] for r in subset})
        assert len(cps)==18
        rng=np.random.default_rng(SEED)
        bycp={cp:[r for r in subset if r['checkpoint']==cp] for cp in cps}
        for outcome in OUTCOMES:
            for feature in ('log_energy_ratio','raw_fpr','full_moment_pr'):
                per=[]
                for cp in cps:
                    group=bycp[cp]
                    rho=within_rho([r[feature] for r in group],[r[outcome] for r in group])
                    per.append(rho)
                    correlations.append(dict(cohort=prefix+cohort,checkpoint=cp,feature=feature,outcome=outcome,
                        spearman=rho,n_arms=len(group)))
                valid=np.array([v for v in per if v is not None])
                ci=[None,None]
                if len(valid):
                    boot=valid[rng.integers(0,len(valid),size=(BOOTSTRAPS,len(valid)))].mean(1)
                    ci=list(map(float,np.quantile(boot,[.025,.975])))
                statistics.append(dict(cohort=prefix+cohort,kind='mean_within_checkpoint_spearman',
                    feature=feature,outcome=outcome,n_checkpoints=18,n_defined=len(valid),n_arms=len(subset),
                    mean_rho=float(valid.mean()) if len(valid) else None,ci_low=ci[0],ci_high=ci[1]))
            for pr in ('raw_fpr','full_moment_pr'):
                models={'Energy':('log_energy_ratio',),'FPR':(pr,),'Energy+FPR':('log_energy_ratio',pr)}
                fitted={}
                for model,features in models.items():
                    result,pred,x,y=regression(subset,features,outcome)
                    fitted[model]=(result,np.array([r['loco_prediction'] for r in pred]))
                    predictions.extend(dict(cohort=prefix+cohort,pr_definition=pr,model=model,**r) for r in pred)
                    designs.extend(dict(cohort=prefix+cohort,pr_definition=pr,model=model,checkpoint=r['checkpoint'],
                        method=r['method'],outcome=outcome,**{f:float(x[i,j]) for j,f in enumerate(features)}) for i,r in enumerate(subset))
                    statistics.append(dict(cohort=prefix+cohort,kind='paired_linear_regression',
                        pr_definition=pr,model=model,outcome=outcome,n_checkpoints=18,n_arms=len(subset),
                        r2=result['r2'],loco_r2=result['loco_r2'],rank=result['rank'],coefficients=json.dumps(result['coefficients'])))
                # Incremental explanatory and held-out power, cluster bootstrap.
                x=center_by_checkpoint(subset,[[r['log_energy_ratio'],r[pr]] for r in subset])
                y=center_by_checkpoint(subset,[r[outcome] for r in subset])
                scales=x.std(0); x=np.divide(x,scales,out=np.zeros_like(x),where=scales>1e-10)
                masks=[np.flatnonzero(np.array([r['checkpoint'] for r in subset])==cp) for cp in cps]
                increments=[]; heldout_increments=[]
                for draw in rng.integers(0,18,size=(BOOTSTRAPS,18)):
                    idx=np.concatenate([masks[j] for j in draw]); xb=x[idx]; yb=y[idx]
                    den=float(yb@yb)
                    if den<=1e-15: continue
                    pe=xb[:,:1]@np.linalg.lstsq(xb[:,:1],yb,rcond=1e-10)[0]
                    pb=xb@np.linalg.lstsq(xb,yb,rcond=1e-10)[0]
                    increments.append(float((np.square(yb-pe).sum()-np.square(yb-pb).sum())/den))
                    energy_cv=fitted['Energy'][1][idx]; combined_cv=fitted['Energy+FPR'][1][idx]
                    heldout_increments.append(float((np.square(yb-energy_cv).sum()-np.square(yb-combined_cv).sum())/den))
                ci=list(map(float,np.quantile(increments,[.025,.975]))) if increments else [None,None]
                cvci=list(map(float,np.quantile(heldout_increments,[.025,.975]))) if heldout_increments else [None,None]
                e,b=fitted['Energy'][0],fitted['Energy+FPR'][0]
                statistics.append(dict(cohort=prefix+cohort,kind='incremental_fpr_over_energy',pr_definition=pr,
                    outcome=outcome,n_checkpoints=18,n_arms=len(subset),
                    delta_r2=b['r2']-e['r2'] if b['r2'] is not None else None,
                    delta_loco_r2=b['loco_r2']-e['loco_r2'] if b['loco_r2'] is not None else None,
                    delta_r2_ci_low=ci[0],delta_r2_ci_high=ci[1],
                    delta_loco_r2_ci_low=cvci[0],delta_loco_r2_ci_high=cvci[1]))
    return statistics,predictions,correlations,designs


def fpr():
    rows=collect_results()
    statistics,predictions,correlations,designs=fixed_analysis(rows)
    tsv(OUT/'fpr_statistics.tsv',statistics); write(OUT/'fpr_statistics.json',statistics)
    tsv(OUT/'fpr_regression_predictions.tsv',predictions)
    tsv(OUT/'fpr_checkpoint_spearman.tsv',correlations)
    tsv(OUT/'fpr_regression_design.tsv',designs)
    grouped_correlations=[]
    rng=np.random.default_rng(SEED)
    for cohort in ('A_all_versions','B_edited_only','C_energy_matched'):
        for feature in ('log_energy_ratio','raw_fpr','full_moment_pr'):
            for outcome in OUTCOMES:
                group_rows=[r for r in correlations if r['cohort']==cohort and r['feature']==feature and r['outcome']==outcome]
                for group,select in {**{'base/'+b:lambda r,b=b:r['checkpoint'].startswith(b+'/') for b in BASES},
                    **{'task/'+t:lambda r,t=t:r['checkpoint'].split('/')[1]==t for t in TASKS[:3]}}.items():
                    selected=[r for r in group_rows if select(r)]
                    valid=np.array([r['spearman'] for r in selected if r['spearman'] is not None])
                    ci=np.quantile(valid[rng.integers(0,len(valid),size=(BOOTSTRAPS,len(valid)))].mean(1),[.025,.975]) if len(valid) else [None,None]
                    grouped_correlations.append(dict(cohort=cohort,group=group,feature=feature,outcome=outcome,
                        n_checkpoints=len(selected),n_defined=len(valid),mean_rho=float(valid.mean()) if len(valid) else None,
                        ci_low=float(ci[0]) if ci[0] is not None else None,ci_high=float(ci[1]) if ci[1] is not None else None,
                        positive=int((valid>1e-9).sum()),zero=int((abs(valid)<=1e-9).sum()),negative=int((valid< -1e-9).sum())))
    tsv(OUT/'fpr_grouped_checkpoint_spearman.tsv',grouped_correlations)
    # Reanalyze, but never pool, the older flat/norm baseline cohort. Its Llama
    # scores are not interchangeable with the final refreshed main experiment.
    legacy_path=ROOT/'reports/functional_activation_three_seed_20260914/diagonal_per_seed.tsv'
    if legacy_path.exists():
        legacy=[]
        for r in load_tsv(legacy_path):
            row=dict(base=r['base'],task=r['train_task'],seed=int(r['seed']),checkpoint=r['checkpoint'],method=r['method'],
                raw_fpr=float(r['functional_participation']),full_moment_pr=float(r['functional_cov_participation']),
                energy_ratio=float(r['functional_energy_ratio'])**2,
                log_energy_ratio=float(np.log(float(r['functional_energy_ratio'])**2)),
                target_gain=float(r['target_gain']),off_gain=float(r['off_gain']),fg_reduction=float(r['forgetting_reduction']))
            row.update({'gain_'+t:float(r['score_'+t]) for t in TASKS})
            row['score_manifest']=r['score_manifest'];row['label']=r['label']
            legacy.append(row)
        lix={(r['checkpoint'],r['method']):r for r in legacy}
        for row in legacy:
            orig=lix[row['checkpoint'],'original_lora']
            for t in TASKS:
                row['gain_'+t]-=orig['gain_'+t] if row['method']!='original_lora' else 0
            path=Path(row['score_manifest']).parent/'commonsense'/row['label']/'metrics.json'
            oldpath=Path(orig['score_manifest']).parent/'commonsense'/orig['label']/'metrics.json'
            sub=read(path)['per_task'];oldsub=read(oldpath)['per_task']
            for key,value in sub.items(): row['gain_'+key]=100*(value['accuracy']-oldsub[key]['accuracy'])
        for row in legacy:
            if row['method']=='original_lora':
                for t in TASKS: row['gain_'+t]=0.
        tsv(OUT/'legacy_fpr_raw_results.tsv',legacy)
        ls,lp,lc,ld=fixed_analysis(legacy,prefix='legacy_')
        tsv(OUT/'legacy_fpr_statistics.tsv',ls);write(OUT/'legacy_fpr_statistics.json',ls)
        tsv(OUT/'legacy_fpr_regression_predictions.tsv',lp)
        tsv(OUT/'legacy_fpr_checkpoint_spearman.tsv',lc)
        tsv(OUT/'legacy_fpr_regression_design.tsv',ld)
        write(OUT/'legacy_analysis_provenance.json',dict(note='Older LoRA/HNS/Flat-Fro/Flat-Nuclear cohort separately reanalyzed on all18 checkpoints; never pooled with latest Llama reevaluation.',
            original_analysis=str(ROOT/'reports/promising_spectral_metrics_three_seed_20260914'),
            original_analysis_sha256=sha(ROOT/'reports/promising_spectral_metrics_three_seed_20260914/functional_features.tsv'),
            historical_diagonal=str(legacy_path),historical_diagonal_sha256=sha(legacy_path)))
    write(OUT/'fpr_complete.json',dict(status='complete',checkpoints=18,arms=126,
        cohort_arms=dict(A_all_versions=126,B_edited_only=108,C_energy_matched=54),
        bootstrap_resamples=BOOTSTRAPS,seed=SEED,metrics=['raw_fpr','full_moment_pr','functional_energy'],
        energy_matched_predictor_rank=0,notes='Within-checkpoint centered LOCO predicts relative arm deviations, not absolute scores. No metric search or hyperparameter tuning. CI descriptive; no iid training-run claim.'))


def keyed_correct(path, field):
    result={}
    with Path(path).open() as f:
        for line in f:
            r=json.loads(line); identity=str(r['id'])
            if identity in result: raise ValueError(f'Duplicate id: {path}: {identity}')
            assert field in r and isinstance(r[field],bool)
            # Keep pairing evidence while dropping large prediction text/tokens.
            source={k:r[k] for k in ('id','subtask','dataset_source','problem_prompt','entry_point',
                'question','prompt','answer','gold','instruction_id_list','kwargs','choices') if k in r}
            sig=json.dumps(source,sort_keys=True,ensure_ascii=False)
            result[identity]=(r[field],str(r.get('subtask','all')),sig)
    return result


def transition_metrics(b,l,e):
    b,l,e=(np.asarray(x,dtype=bool) for x in (b,l,e))
    rec=b&~l; ret=~b&l; bothwrong=~b&~l; bothright=b&l
    nr,nt,ns,nd=(int(mask.sum()) for mask in (rec,ret,bothwrong,bothright))
    recovered=int((rec&e).sum()); retained=int((ret&e).sum())
    success=int((bothwrong&e).sum()); damage=int((bothright&~e).sum())
    return dict(samples=len(b),recovery_set_size=nr,retention_set_size=nt,
        new_success_set_size=ns,new_damage_set_size=nd,recovered=recovered,retained=retained,
        recovery_rate=recovered/nr if nr else None,retention_rate=retained/nt if nt else None,
        new_success=success,new_damage=damage,new_success_rate=success/len(b),new_damage_rate=damage/len(b),
        new_success_conditional_rate=success/ns if ns else None,new_damage_conditional_rate=damage/nd if nd else None)


def macro(entries, identity):
    result=dict(identity)
    for key in ('recovery_rate','retention_rate','new_success_rate','new_damage_rate',
        'new_success_conditional_rate','new_damage_conditional_rate'):
        result[key]=finite_mean([r[key] for r in entries])
        result[key+'_defined']=sum(r[key] is not None for r in entries)
    result['n_units']=len(entries)
    for key in ('samples','recovery_set_size','retention_set_size','new_success_set_size','new_damage_set_size',
        'recovered','retained','new_success','new_damage'):
        result[key]=sum(r[key] for r in entries)
    return result


def recovery():
    assert read(OUT/'fpr_complete.json')['status']=='complete'
    matrix=[]
    with (OUT/'evaluation_matrix.tsv').open() as f:
        matrix=list(csv.DictReader(f,delimiter='\t'))
    paths={(r['base'],r['train_task'],int(r['seed'] or 0),r['method'],r['eval_task']):Path(r['metrics_path']).parent/'scored.jsonl' for r in matrix}
    rows=read(OUT/'raw_results.json')
    # The three requested methods are mandatory; optional baselines must not
    # delay this stage. Existing functional variants are summarized next.
    methods=list(MATCHED)
    atomic=[]; families=[]; audit=[]
    # Per-item bit patterns allow paper rechecking without re-inference.
    item_path=OUT/'recovery_retention_sample_bits.tsv.gz'
    with gzip.open(item_path,'wt',compresslevel=5) as f:
        writer=csv.writer(f,delimiter='\t')
        writer.writerow(['checkpoint','benchmark','id','base_correct','lora_correct',*methods])
        for base in BASES:
            for bench in TASKS:
                bp=paths[base,'base',0,'base',bench]
                bd=keyed_correct(bp,FIELDS[bench])
                assert len(bd)==COUNTS[bench]
                for task in TASKS[:3]:
                    for seed in (42,43,44):
                        cp=f'{base}/{task}/seed{seed}'
                        lp=paths[base,task,seed,'original_lora',bench]
                        ld=keyed_correct(lp,FIELDS[bench]); ids=list(bd)
                        assert set(ld)==set(bd)
                        all_ed={}
                        for method in methods:
                            ep=paths[base,task,seed,method,bench]
                            ed=keyed_correct(ep,FIELDS[bench]); assert set(ed)==set(bd)
                            assert all(bd[i][1:]==ld[i][1:]==ed[i][1:] for i in ids), ('Sample content mismatch',cp,bench,method)
                            all_ed[method]=ed
                            audit.append(dict(checkpoint=cp,eval_task=bench,method=method,
                                base_scored_sha256=sha(bp),lora_scored_sha256=sha(lp),edited_scored_sha256=sha(ep),samples=len(ids)))
                            strata=sorted({bd[i][1] for i in ids})
                            per=[]
                            for stratum in strata:
                                selected=[i for i in ids if bd[i][1]==stratum]
                                benchmark=stratum if bench=='commonsense' else bench
                                r=dict(base=base,task=task,seed=seed,checkpoint=cp,method=method,
                                    family=bench,benchmark=benchmark,role='target' if task==bench else 'off_task',
                                    **transition_metrics([bd[i][0] for i in selected],[ld[i][0] for i in selected],[ed[i][0] for i in selected]))
                                atomic.append(r); per.append(r)
                            families.append(macro(per,dict(base=base,task=task,seed=seed,checkpoint=cp,method=method,
                                benchmark=bench,role='target' if task==bench else 'off_task')))
                        for i in ids:
                            benchmark=bd[i][1] if bench=='commonsense' else bench
                            writer.writerow([cp,benchmark,i,int(bd[i][0]),int(ld[i][0]),*[int(all_ed[m][i][0]) for m in methods]])
                        print('[Recovery]',cp,bench,flush=True)
    tsv(OUT/'recovery_retention_by_checkpoint_benchmark.tsv',atomic)
    tsv(OUT/'recovery_retention_by_checkpoint_family.tsv',families)
    grouped=[]
    for method in methods:
        for benchmark in sorted({r['benchmark'] for r in atomic}):
            selected=[r for r in atomic if r['method']==method and r['benchmark']==benchmark]
            grouped.append(macro(selected,dict(group='benchmark/'+benchmark,method=method)))
    cp_macro=[]
    for cp in sorted({r['checkpoint'] for r in families}):
        for method in methods:
            for role in ('all','target','off_task'):
                selected=[r for r in families if r['checkpoint']==cp and r['method']==method and (role=='all' or r['role']==role)]
                cp_macro.append(macro(selected,dict(checkpoint=cp,method=method,role=role)))
    for method in methods:
        for role in ('all','target','off_task'):
            selected=[r for r in cp_macro if r['method']==method and r['role']==role]
            grouped.append(macro(selected,dict(group='macro/'+role,method=method)))
        for b in BASES:
            selected=[r for r in cp_macro if r['method']==method and r['role']=='all' and r['checkpoint'].startswith(b+'/')]
            grouped.append(macro(selected,dict(group='macro/base/'+b,method=method)))
        for t in TASKS[:3]:
            selected=[r for r in cp_macro if r['method']==method and r['role']=='all' and r['checkpoint'].split('/')[1]==t]
            grouped.append(macro(selected,dict(group='macro/source/'+t,method=method)))
    tsv(OUT/'recovery_retention_summary.tsv',grouped)
    tsv(OUT/'recovery_retention_scatter_data.tsv',cp_macro)
    # Paired differences use the same opportunity sets, one weight per checkpoint.
    ix={(r['checkpoint'],r['method'],r['role']):r for r in cp_macro}
    paired=[]; rng=np.random.default_rng(SEED)
    for method in CONTROLS:
        for role in ('all','target','off_task'):
            for metric in ('recovery_rate','retention_rate','new_success_rate','new_damage_rate'):
                diffs=[]
                for cp in sorted({r['checkpoint'] for r in cp_macro}):
                    a,b=ix[cp,'hns_f4_s1',role][metric],ix[cp,method,role][metric]
                    if a is not None and b is not None: diffs.append(a-b)
                d=np.array(diffs)
                ci=np.quantile(d[rng.integers(0,len(d),size=(BOOTSTRAPS,len(d)))].mean(1),[.025,.975]) if len(d) else [None,None]
                paired.append(dict(comparison='HNS minus '+method,role=role,metric=metric,n_defined=len(d),
                    mean_difference=float(d.mean()) if len(d) else None,ci_low=float(ci[0]) if ci[0] is not None else None,
                    ci_high=float(ci[1]) if ci[1] is not None else None))
    tsv(OUT/'recovery_retention_paired_statistics.tsv',paired)
    tsv(OUT/'recovery_retention_cache_audit.tsv',audit)
    write(OUT/'recovery_complete.json',dict(status='complete',checkpoints=18,methods=len(methods),
        atomic_records=len(atomic),family_records=len(families),new_inference_calls=0,
        macro='Eight commonsense subbenchmarks equal, then four families equal, then 18 checkpoints equal. Undefined opportunity rates stay NA; coverage reported.',
        sample_bits_sha256=sha(item_path)))


def boundary():
    assert read(OUT/'recovery_complete.json')['status']=='complete'
    rows=read(OUT/'raw_results.json')
    selected=[r for r in rows if r['method'] in ('hns_f4_s1','functional_hns_a05','functional_hns_a10','functional_flat')]
    tsv(OUT/'functional_boundary_checkpoint_results.tsv',selected)
    summaries=[]
    for method in ('hns_f4_s1','functional_hns_a05','functional_hns_a10','functional_flat'):
        group=[r for r in selected if r['method']==method]
        summaries.append(dict(method=method,n=18,**{key:float(np.mean([r[key] for r in group])) for key in
            ('raw_fpr','full_moment_pr','energy_ratio','target','off_score','forgetting_gap')}))
    tsv(OUT/'functional_boundary_summary.tsv',summaries)
    ix={(r['checkpoint'],r['method']):r for r in selected}
    counterexamples=[]
    for cp in sorted({r['checkpoint'] for r in selected}):
        h=ix[cp,'hns_f4_s1']
        for method in ('functional_hns_a05','functional_hns_a10','functional_flat'):
            r=ix[cp,method]
            counterexamples.append(dict(checkpoint=cp,method=method,
                raw_fpr_delta=r['raw_fpr']-h['raw_fpr'],full_moment_pr_delta=r['full_moment_pr']-h['full_moment_pr'],
                target_delta=r['target']-h['target'],off_delta=r['off_score']-h['off_score'],
                higher_fpr_worse_target=r['raw_fpr']>h['raw_fpr']+1e-8 and r['target']<h['target']-1e-8,
                higher_fpr_worse_off=r['raw_fpr']>h['raw_fpr']+1e-8 and r['off_score']<h['off_score']-1e-8))
    tsv(OUT/'functional_boundary_counterexamples.tsv',counterexamples)
    write(OUT/'boundary_complete.json',dict(status='complete',new_adapters=0,new_inference=0,
        functional_hns_e='Not run: low-priority optional extension excluded.'))


def load_tsv(path):
    with Path(path).open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
    for row in rows:
        for key,value in list(row.items()):
            if value=='': row[key]=None; continue
            try:
                row[key]=int(value) if value.lstrip('+-').isdigit() else float(value)
            except ValueError:
                pass
    return rows


def report():
    assert read(OUT/'boundary_complete.json')['status']=='complete'
    rows=read(OUT/'raw_results.json'); stats=read(OUT/'fpr_statistics.json')
    matched=[r for r in rows if r['method'] in MATCHED]
    groups=load_tsv(OUT/'energy_matched_grouped_results.tsv')
    pairs=load_tsv(OUT/'energy_matched_paired_statistics.tsv')
    rr=load_tsv(OUT/'recovery_retention_summary.tsv')
    rrp=load_tsv(OUT/'recovery_retention_paired_statistics.tsv')
    bounds=load_tsv(OUT/'functional_boundary_summary.tsv')
    legacy_stats=read(OUT/'legacy_fpr_statistics.json')
    legacy_selected=[r for r in legacy_stats if r['kind']=='paired_linear_regression' and r['outcome'] in ('target_gain','off_gain')]
    grouped_fpr=load_tsv(OUT/'fpr_grouped_checkpoint_spearman.tsv')
    old=load_tsv(OUT/'old_scalar_energy_audit.tsv')
    ratios=np.array([float(r['old_fro_scalar_energy_over_hns']) for r in old])
    selected=[r for r in stats if r['kind']=='paired_linear_regression' and r['outcome'] in ('target_gain','off_gain','fg_reduction')]
    increment=[r for r in stats if r['kind']=='incremental_fpr_over_energy' and r['outcome'] in ('target_gain','off_gain')]
    benchmark_increment=[r for r in stats if r['kind']=='incremental_fpr_over_energy' and r['outcome'].startswith('gain_')]
    conclusions=[]
    for method in CONTROLS:
        entries={r['outcome']:r for r in pairs if r['comparison']=='HNS minus '+method}
        a,b=entries['target'],entries['off_score']
        conclusions.append(f"18 个 checkpoint 上，HNS 相对 {NAMES[method]} 的平均 Target 差为 {float(a['mean_difference']):+.3f} pp（checkpoint bootstrap 95% CI {float(a['ci_low']):+.3f}, {float(a['ci_high']):+.3f}），Off 差为 {float(b['mean_difference']):+.3f} pp（{float(b['ci_low']):+.3f}, {float(b['ci_high']):+.3f}）；不将点估计自动解释为显著优势。")
    conclusions.append('匹配逐模块 functional energy 后，三种方法的 Energy-only within-checkpoint 模型秩为 0；同能量内的性能差异不能由这个冻结缓存能量标量解释，但也不证明特定谱机制具有因果优势。')
    for pr in ('raw_fpr','full_moment_pr'):
        inc=[r for r in increment if r['cohort']=='C_energy_matched' and r['pr_definition']==pr]
        vals=', '.join(f"{r['outcome']} ΔLOCO R²={r['delta_loco_r2']:+.4f}" for r in inc)
        conclusions.append(f'{pr} 在同能量对照中的附加留出解释能力：{vals}。负留出 R² 表示不如 checkpoint 内零偏差预测；不能凭训练内 R² 增加宣称稳定泛化。')
    a={r['metric']:r for r in rrp if r['comparison']=='HNS minus scalar_e' and r['role']=='all'}
    hrr=next(r for r in rr if r['group']=='macro/all' and r['method']=='hns_f4_s1')
    conclusions.append(f"固定四-family macro 下，HNS Recovery={float(hrr['recovery_rate'])*100:.3f}%，Retention={float(hrr['retention_rate'])*100:.3f}%，原始 New Success 事件数={hrr['new_success']}；HNS−Scalar-E Recovery={float(a['recovery_rate']['mean_difference'])*100:+.3f} pp，Retention={float(a['retention_rate']['mean_difference'])*100:+.3f} pp。非零Retention意味着保留部分Base错误而LoRA正确的样本，不能把模型等同于Base；但不据此声称全面保留所有新能力。")
    cex=load_tsv(OUT/'functional_boundary_counterexamples.tsv')
    nt=sum(r['higher_fpr_worse_target']=='True' for r in cex); no=sum(r['higher_fpr_worse_off']=='True' for r in cex)
    conclusions.append(f'现有 54 个 Functional-HNS/Functional-Flat vs HNS 配对中，Raw FPR 更高但 Target 更低有 {nt} 个，Raw FPR 更高但 Off 更低有 {no} 个；若存在反例即不支持“FPR 越高 adapter 必然越好”，不新增 α 搜索。')
    ix={(r['checkpoint'],r['method']):r for r in rows}
    scalar=[r for r in rows if r['method']=='scalar_e']
    pr_error=max(abs(r['raw_fpr']-ix[r['checkpoint'],'original_lora']['raw_fpr']) for r in scalar)
    full_error=max(abs(r['full_moment_pr']-ix[r['checkpoint'],'original_lora']['full_moment_pr']) for r in scalar)
    conclusions.append(f"18 个 Scalar-E 与各自LoRA的Raw FPR / Full-moment PR最大差分别为 {pr_error:.2e} / {full_error:.2e}，但Scalar-E的平均Target/Off gain为 {np.mean([r['target_gain'] for r in scalar]):+.3f} / {np.mean([r['off_gain'] for r in scalar]):+.3f} pp。这验证公共缩放下PR不变，PR不能独自编码更新强度效应。")
    lines=['# HNS / LoRA-Norm：同功能能量机制对照与配对分析','',
        '状态：complete。2 Base × 3 source task × seed42/43/44，18 checkpoints 全部保留。禁止训练已遵守；固定 HNS 4+1，不搜索新指标/α，不运行 Functional-HNS-E。','',
        '## 1. Implementation audit','',
        '已有 `ScalarShrink`（`src/finetune/spectral_edit/mechanism.py:84`）和 `common_per_module`（`scripts/build_forgetting_common_basis_controls.py`）的 gain 为 ‖tᴴ‖₂/‖σ‖₂，匹配逐模块 Frobenius norm；global scalar、Flat-Fro 与 Flat-Nuclear 也不等价于 Scalar-E/Flat-E。不能复用其成绩充当同能量对照。','',
        f'在本次 {len(old)} 个 module/checkpoint 对上，旧 Fro scalar 的 functional energy / HNS 比值中位数={np.median(ratios):.6f}，范围=[{ratios.min():.6f}, {ratios.max():.6f}]。完整逐模块审计见 `old_scalar_energy_audit.tsv`。','',
        'Scalar-E 使用原 σ；Flat-E 使用全 1 谱；逐模块公共 gain=√[Σ(tᴴ)²q/Σs²q]。q 直接来自 frozen Base 源方向 activation cache，无 floor。controls 不保持 nuclear norm；原 HNS 仍保持。源 U/V 哈希必须完全相同，balanced reconstruction、原 rank/alpha/scaling/config 保留；保存后源坐标能量误差<1e-5。有限存储dtype舍入用额外极小公共标量校正，不做形状编辑；有限基核验显式处理Gram矩阵。每module的解析谱、q、保存误差和舍入校正记录在 `energy_matched_module_audit.tsv` 及adapter metadata中。','',
        'Base/LoRA/HNS 最终三 seed 的 152 个完整评测 cells 严格复用并核验prediction/scored/metrics哈希。新增 Scalar-E/Flat-E 各 72 cells，共 144；v2兼容性检查恢复历史短测generation_manifest中的完整adapter顺序、ID与5-arm blocks，短测统一4096，Qwen seed42 pilot与seed43/44 extension分开；逐token同时核验历史短测与对应最终缓存。Llama live短测覆盖seed43/44，其刷新后的seed42仅核验最终缓存完整性，不混入旧pilot；具体覆盖见各Base的reuse_probe_v2_audit.json。v1失败短测保留作审计而非成绩。少量兼容性probe不是新性能实验。完整主评测仍为long2048 / commonsense4096；主benchmark、样本数、prompt/chat、parser、greedy generation seed42、tokens及评分代码不变。','',
        '重要限制：现有训练审计对部分 seed42 Llama 的 seed/recipe 一致性有保留；全部保留但 bootstrap 与 mean±SD 是现有 checkpoint 的描述性不确定性，不能称为严格 iid 的相同 recipe 重复。功能能量只对固定任务 calibration/Base 输入轨迹成立，不代表 edited 模型实际全局功能输出强度完全一致。','',
        '## 2. Energy-matched results','',
        table(['Base','Source','Seed','Method','Target','Off','FG','Raw FPR','Full-moment PR','Energy/source'],
            [[r['base'],r['task'],r['seed'],NAMES[r['method']],*[r[k] for k in ('target','off_score','forgetting_gap','raw_fpr','full_moment_pr','energy_ratio')]] for r in matched]),'',
        'Off=其他三类benchmark百分成绩等权平均；FG=其他三类 max(Base−edited,0) 等权平均。Commonsense 为八子任务等权。FG 越低越好，且 Qwen 的 FG=0 截断不替代 Target/Off 分析。','',
        table(['Group','Method','n','Target','Off','FG'],[[r['group'],NAMES[r['method']],r['n'],*[r[k] for k in ('target','off_score','forgetting_gap')]] for r in groups]),'',
        '配对差与 checkpoint bootstrap（95%，2000次，seed20260914；非因果、非 iid 训练重复）：','',
        table(['Comparison','Outcome','Mean HNS−control','CI low','CI high','W/T/L'],[[r['comparison'],r['outcome'],r['mean_difference'],r['ci_low'],r['ci_high'],f"{r['positive']}/{r['tie']}/{r['negative']}"] for r in pairs if r['outcome'] in ('target','off_score','forgetting_gap')]),'',
        '## 3. FPR analysis','',
        '预先固定 Raw FPR、Full-moment PR、Energy。FPR 为 module median，Energy 为逐模块 c²Σs²q 总和；Energy predictor 固定为 log(edited/source energy)。A=LoRA+全部7版本（126 arms）；B=去除LoRA（108 arms）；C=HNS/Scalar-E/Flat-E（54 arms）；每层始终18个source checkpoint。旧谱相关性文件原样保留并记录哈希，不混入最终 Llama 重评成绩。','',
        '每个 source 内中心化 predictors/outcomes，固定线性 OLS，不调超参数；Spearman 先在每个 checkpoint 内计算，再等权平均。Bootstrap 按 checkpoint 抽样，不能把多个编辑版本算作独立训练。LOCO=留一个 source checkpoint，评估该 checkpoint 内各版本相对偏差（不是未知 checkpoint 的绝对性能预测）。同能量 C 的 Energy predictor 解析设为相等，不利用保存/浮点噪声拟合。','',
        table(['Layer','FPR definition','Outcome','Model','In-sample R²','LOCO R²','Rank'],[[r['cohort'],r['pr_definition'],r['outcome'],r['model'],r['r2'],r['loco_r2'],r['rank']] for r in selected]),'',
        table(['Layer','FPR definition','Outcome','ΔR² above Energy','ΔLOCO R²','ΔLOCO CI low','ΔLOCO CI high'],[[r['cohort'],r['pr_definition'],r['outcome'],r['delta_r2'],r['delta_loco_r2'],r['delta_loco_r2_ci_low'],r['delta_loco_r2_ci_high']] for r in increment]),'',
        '四family与八commonsense分项的附加留出解释力（相同固定模型，不挑最高相关性）：','',
        table(['Layer','FPR','Benchmark gain','ΔLOCO R²','CI low','CI high'],[[r['cohort'],r['pr_definition'],r['outcome'],r['delta_loco_r2'],r['delta_loco_r2_ci_low'],r['delta_loco_r2_ci_high']] for r in benchmark_increment]),'',
        '同能量 C 的按 Base/source-task 配对 Spearman，检查符号与关系是否稳定（不筛 checkpoint）：','',
        table(['Group','Feature','Outcome','Mean within-source ρ','CI low','CI high','Defined','Positive/Zero/Negative'],[[r['group'],r['feature'],r['outcome'],r['mean_rho'],r['ci_low'],r['ci_high'],r['n_defined'],f"{r['positive']}/{r['zero']}/{r['negative']}"] for r in grouped_fpr if r['cohort']=='C_energy_matched' and r['feature']!='log_energy_ratio' and r['outcome'] in ('target_gain','off_gain')]),'',
        '保留原始 LoRA/HNS/Flat-Fro/Flat-Nuclear 分析：使用已有18-checkpoint activation统计重新做相同固定指标分析，独立成legacy cohort，不与本次Llama最终重评成绩混池。','',
        table(['Legacy layer','FPR','Outcome','Model','R²','LOCO R²'],[[r['cohort'],r['pr_definition'],r['outcome'],r['model'],r['r2'],r['loco_r2']] for r in legacy_selected]),'',
        '全部四family与八commonsense分项、逐checkpoint Spearman（含undefined）、设计矩阵、原始表、每个留出预测与完整统计见 `raw_results.tsv`、`fpr_statistics.tsv/json`、`fpr_checkpoint_spearman.tsv`、`fpr_regression_design.tsv`、`fpr_regression_predictions.tsv`。不筛 seed，不修改指标。Spearman 使用固定1e-8小数舍入区分数值噪声与真实并列。训练内增量R²天然非负，不据此单独判断 FPR 有用；留出值、配对分项和跨Base/task异质性共同限制结论。','',
        '## 4. Recovery–Retention analysis','',
        '只读取已有 scored/prediction caches；新增推理次数=0。样本按 id 及原始题目/标签内容核验。Recovery=P(Edited对|Base对、LoRA错)；Retention=P(Edited对|Base错、LoRA对)。New Success=三者(0,0,1)，New Damage=(1,1,0)，报告原始数与四个机会集合大小。','',
        '固定macro：每checkpoint先将Commonsense八子任务等权，再将四families等权，最后18checkpoint等权；target/off另列。集合为空为NA，不能置0；defined/coverage保存在数据表中。summary里的counts是样本机会/事件的和，rate是固定macro而非这些和的比值。','',
        table(['Group','Method','Recovery','Retention','New Success','New Damage','Recovery n','Retention n','Both-wrong n','Both-right n'],[[r['group'],NAMES[r['method']],*[r[k] for k in ('recovery_rate','retention_rate','new_success','new_damage','recovery_set_size','retention_set_size','new_success_set_size','new_damage_set_size')]] for r in rr]),'',
        table(['Comparison','Role','Metric','Mean difference','CI low','CI high','n'],[[r['comparison'],r['role'],r['metric'],r['mean_difference'],r['ci_low'],r['ci_high'],r['n_defined']] for r in rrp]),'',
        '完整逐checkpoint × benchmark表：`recovery_retention_by_checkpoint_benchmark.tsv`；family表与macro：`recovery_retention_by_checkpoint_family.tsv`、`recovery_retention_summary.tsv`；论文scatter/table输入：`recovery_retention_scatter_data.tsv`。逐样本正确性bit表：`recovery_retention_sample_bits.tsv.gz`；源cache哈希：`recovery_retention_cache_audit.tsv`。','',
        '## 5. Functional-HNS boundary','',
        '只整理已有 α=.5/1/Protected Functional Flat；不继续开发/搜索，也不新增 Functional-HNS-E。以下固定18-checkpoint比较同时报告FPR、Energy及性能，不能把能量一起变化的结果解释为FPR的因果效应。','',
        table(['Method','n','Raw FPR','Full-moment PR','Energy/source','Target','Off','FG'],[[NAMES[r['method']],r['n'],*[r[k] for k in ('raw_fpr','full_moment_pr','energy_ratio','target','off_score','forgetting_gap')]] for r in bounds]),'',
        f'现有配对中，higher Raw FPR / worse Target={nt}/54，higher Raw FPR / worse Off={no}/54。逐checkpoint全部反例及非反例保留于 `functional_boundary_counterexamples.tsv`。','',
        '## 6. Main conclusions / Paper-ready summary','',
        *[f'{i}. {text}' for i,text in enumerate(conclusions,1)],'',
        '本报告不声称 FPR 是可靠优化目标，不把同能量残差直接解释为 functional concentration 的因果贡献，不根据结果筛选 checkpoint 或改HNS配置。若比较区间跨0或留出解释力不稳定，应原样报告证据不足。','',
        '## Reproducibility','',
        '实现：`scripts/hns_energy_matched.py`、`scripts/analyze_hns_energy_matched.py`；提交：`slurm/hns_energy_matched.slurm`（两张B300，两Base并行，各stage设置barrier）；`manifest.json`固定source/checkpoint、activation/cache、代码哈希与分析规则。`commands_*.json`记录真实命令及返回码。全部数据相对本报告同名目录。','']
    destination=OUT.with_suffix('.md')
    destination.write_text('\n'.join(lines))
    manifest=read(OUT/'manifest.json')
    manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        job_id=os.getenv('SLURM_JOB_ID'),gpu_count=2,report_sha256=sha(destination),
        analysis_script_sha256=sha(__file__))
    write(OUT/'manifest.json',manifest)
    write(OUT/'final_audit.json',dict(status='pass',checkpoints=18,primary_arms=54,total_arms=126,
        source_manifest_sha256=read(OUT/'manifest.json')['source_manifest_sha256'],
        report_sha256=sha(destination),retraining=False,
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ('.tsv','.gz','.json')
            and p.name not in ('final_audit.json','worker_status.json') and not p.name.startswith('commands_')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',required=True,choices=['fpr','recovery','boundary','report'])
    args=p.parse_args(); globals()[args.stage]()
