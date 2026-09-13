#!/usr/bin/env python3
"""Test proposed concentration, spatial-structure and frozen-activation diagnostics."""
import argparse
import csv
import gzip
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import analyze_spectral_performance_forgetting_three_seed as lib

PREVIOUS=lib.OUT
OUT=lib.ROOT/'reports/promising_spectral_metrics_three_seed_20260914'
PROPOSALS={
    'shannon_eff_rank':'median module exp(H(s/sum(s))), unnormalized 1..16',
    'r2_participation':'median module sum(s)^2/sum(s^2)',
    'stable_rank':'median module sum(s^2)/max(s)^2, unnormalized 1..16',
    'flatness':'median exp(mean(log(s)))/mean(s)',
    'gini':'median singular-value Gini; zero is flat',
    'decay_slope':'median OLS slope of log(s_i) against i=1..16',
    'erank_energy':'median exp(H(s^2/sum(s^2))), 1..16',
    'entropy_gap':'median H(p)-H(q)',
    'hill_tail_contrast':'median mean_{i<=8} log(lambda_i/lambda_9), lambda=s^2; 1/(alpha_hill-1), zero at flat endpoint',
    'd_flat_l1':'median sum(abs(s-mean(s)))/sum(s)',
    'soft_rank':'median mean(s)/max(s)',
    'head_tail':'median log(sum(s[:4])/sum(s[12:16]))',
    'erank_iqr':'IQR of module shannon effective ranks',
    'erank_depth_slope':'OLS slope of layer-median erank against raw layer index',
    'erank_depth_slope_normalized':'same slope against depth layer/(layers-1)',
    'erank_attn_mlp_gap':'median attention erank minus median MLP erank',
    'gamma_param':'global sum(w*beta), w=source_s^2/sum(source_s^2); equals previous scalar_fit',
    'eta_param':'1-gamma^2/sum(w*beta^2); equals previous shape_residual in common U/V',
    'fro_ratio':'previous aggregate global Frobenius ratio',
    'original_head1_retention':'previous actual original leading-direction energy retention',
    'previous_mean_entropy_rank':'previous mean module effective rank/16; aggregation comparator',
}
INCOMPLETE={
    'alpha_hill':'1+8/sum_{i<=8} log(lambda_i/lambda_9); undefined for flat/near-flat modules',
    'alpha_hill_k4':'same estimator k=4', 'alpha_hill_k12':'same estimator k=12',
    'alpha_weighted_hill':'median alpha_hill*log10(sigma1_scaled^2); Hill-based diagnostic, not official WeightWatcher PL fit',
    'alpha_iqr':'IQR of module alpha_hill, defined only if every module is nondegenerate',
    'kurtosis':'median population excess kurtosis of 16 singular values; flat spectrum undefined',
    'decay_family':'median R2(log(s)~log(i))-R2(log(s)~i); flat spectrum undefined',
}
FUNCTIONAL={
    'functional_participation':'median 1/sum(pi^2), pi=source_response_energy*beta^2 / module total',
    'functional_erank':'median exp(H(pi))',
    'functional_top1':'median max(pi)',
    'functional_erank_iqr':'IQR of module exp(H(pi))',
    'functional_attn_mlp_gap':'attention median functional erank minus MLP median',
    'functional_energy_ratio':'sqrt(sum source_response_energy*beta^2 / sum source_response_energy)',
    'gamma_functional':'sum(wE*beta), wE=source_response_energy/sum(source_response_energy)',
    'eta_functional':'1-gamma_functional^2/sum(wE*beta^2)',
}
BASE_FEATURES={f'base_{side}_overlap_k{k}':f'global parameter-energy fraction projected into pretrained top-{k} {side} subspace; randomized-SVD diagnostic'
    for k in (4,16,32) for side in ('left','right','bilateral')}

def setup():
    OUT.mkdir(exist_ok=True)
    lib.OUT=OUT; lib.CORE=dict(PROPOSALS)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'proposals':PROPOSALS,'incomplete_diagnostics':INCOMPLETE,'functional':FUNCTIONAL,
        'selection':'Proposed user metrics evaluated without accepting supplied correlations as evidence; no performance-selected aggregation.',
        'aggregation':'module medians; IQR separately; attention-MLP gap explicit. Raw parameter and energy weights include original LoRA scaling.',
        'precision':'singular shape normalized by module maximum; relative spread below 1e-5 treated as flat. Shape features rounded to 1e-5. Hill mean log contrast <=1e-4 is undefined, never capped.',
        'validation':'same grouped statistics / nested seed-base-task CV as prior study, separate cohorts, edited-only sensitivity. Incomplete alpha/kurtosis/family excluded from complete-case CV.',
        'functional_scope':'six seed42 source checkpoints; frozen-base training-distribution cache 256 samples, max512 tokens. No fabricated seeds43/44 activations.',
        'independent_unit':'18 source checkpoints (6 functional); repeated waves not independent replications','gpu_count':0}
    if not (OUT/'analysis_plan.json').exists():
        lib.write(OUT/'analysis_plan.json',plan)

def module_metrics(s):
    # A common relative resolution removes numerical fluctuations at equal-gain endpoints.
    s=s/s[:,0,None]
    flat=(s[:,0]-s[:,-1])<=1e-5
    s[flat]=1.
    n=s.shape[1]; p=s/s.sum(1)[:,None]; q=s*s/(s*s).sum(1)[:,None]
    hp=-(p*np.log(p)).sum(1); hq=-(q*np.log(q)).sum(1)
    x=np.arange(1,n+1,dtype=float); xc=x-x.mean(); z=np.log(x); zc=z-z.mean()
    logs=np.log(s); yc=logs-logs.mean(1)[:,None]; total=(yc*yc).sum(1)
    slope=yc@xc/(xc@xc); powerslope=yc@zc/(zc@zc)
    preference=np.divide((powerslope**2)*(zc@zc)-(slope**2)*(xc@xc),total,out=np.full(len(s),np.nan),where=total>1e-10)
    central=s-s.mean(1)[:,None]; variance=(central**2).mean(1)
    kurtosis=np.divide((central**4).mean(1),variance**2,out=np.full(len(s),np.nan),where=variance>1e-10)-3
    metrics={'shannon_eff_rank':np.exp(hp),'r2_participation':s.sum(1)**2/(s*s).sum(1),
        'stable_rank':(s*s).sum(1),'flatness':np.exp(logs.mean(1))/s.mean(1),
        'gini':2*(np.sort(s,axis=1)*x).sum(1)/(n*s.sum(1))-(n+1)/n,
        'decay_slope':slope,'erank_energy':np.exp(hq),'entropy_gap':hp-hq,
        'd_flat_l1':np.abs(s-s.mean(1)[:,None]).sum(1)/s.sum(1),'soft_rank':s.mean(1),
        'head_tail':np.log(s[:,:4].sum(1)/s[:,12:].sum(1)),
        'kurtosis':kurtosis,'decay_family':preference}
    for k in (4,8,12):
        contrast=2*(logs[:,:k]-logs[:,k,None]).mean(1)
        alpha=np.divide(1.,contrast,out=np.full(len(s),np.nan),where=contrast>1e-4)+1
        metrics['alpha_hill' if k==8 else f'alpha_hill_k{k}']=alpha
        if k==8:
            metrics['hill_tail_contrast']=contrast
    return metrics

def median_complete(a):
    return float(np.median(a)) if np.isfinite(a).all() else None

def compute_parameter_features(rows):
    audit=lib.read(PREVIOUS/'spectrum_audit.json')['adapters']; amap={r['path']:r for r in audit}
    features={}; module_rows=[]
    for i,path in enumerate(dict.fromkeys(r['path'] for r in rows),1):
        z=np.load(amap[path]['cache']); s=z['s']; names=z['names'].tolist(); masks=lib.scope_masks(names)
        values=module_metrics(s.copy()); values['alpha_weighted_hill']=values['alpha_hill']*np.log10(s[:,0]**2)
        f={k:median_complete(v) for k,v in values.items()}
        e=values['shannon_eff_rank']; layer=np.array([int(n.split('.layers.')[1].split('.')[0]) for n in names])
        lx=np.unique(layer); ey=np.array([np.median(e[layer==v]) for v in lx]); lc=lx-lx.mean()
        f['erank_iqr']=float(np.quantile(e,.75)-np.quantile(e,.25))
        f['alpha_iqr']=float(np.quantile(values['alpha_hill'],.75)-np.quantile(values['alpha_hill'],.25)) if np.isfinite(values['alpha_hill']).all() else None
        f['erank_depth_slope']=float(lc@(ey-ey.mean())/(lc@lc))
        f['erank_depth_slope_normalized']=f['erank_depth_slope']*float(lx.max())
        f['erank_attn_mlp_gap']=float(np.median(e[masks['attention']])-np.median(e[masks['mlp']]))
        f['alpha_valid_module_fraction']=float(np.isfinite(values['alpha_hill']).mean())
        f['kurtosis_valid_module_fraction']=float(np.isfinite(values['kurtosis']).mean())
        f['decay_family_valid_module_fraction']=float(np.isfinite(values['decay_family']).mean())
        features[path]={k:round(v,5) if v is not None else None for k,v in f.items()}
        for j,n in enumerate(names):
            module_rows.append({'path':path,'module':n,**{k:float(v[j]) if np.isfinite(v[j]) else None for k,v in values.items()}})
        if i%40==0:
            print(f'[Parameters] {i}/282',flush=True)
    extended=[]
    for r in rows:
        f=dict(features[r['path']],gamma_param=r['scalar_fit'],eta_param=r['shape_residual'],
            fro_ratio=r['fro_ratio'],original_head1_retention=r['original_head1_retention'],previous_mean_entropy_rank=r['entropy_rank'])
        extended.append(dict(r,**f))
    for c,cp in dict.fromkeys((r['cohort'],r['checkpoint']) for r in extended):
        group=[r for r in extended if (r['cohort'],r['checkpoint'])==(c,cp)]; baseline,=[r for r in group if r['baseline']]
        for r in group:
            for f in (*PROPOSALS,*INCOMPLETE):
                r['delta__'+f]=r[f]-baseline[f] if r[f] is not None and baseline[f] is not None else None
    lib.write(OUT/'features.json',extended); lib.table(OUT/'features.tsv',extended); lib.table(OUT/'module_metrics.tsv',module_rows)
    return extended

def functional_features(rows):
    root=Path(lib.read(lib.ROOT/'reports/hns_mechanism_geometry_20260910/geometry_audit.json')['source'])
    manifest=lib.read(root/'source_manifest.json')
    source_audit=lib.read(lib.SOURCE/'source_manifest.json')['checkpoints']
    cache_meta=[]; case_stats={}; case_hns={}; case_energy={}
    sp=lib.ROOT/'reports/hns_release_20260912/data/mechanism/module_spectra.csv'
    dp=lib.ROOT/'reports/hns_release_20260912/data/mechanism/direction_response.csv.gz'
    with sp.open() as f:
        for r in csv.DictReader(f):
            key=(r['base_model'],r['task']); case_stats.setdefault(key,{})[r['module']]=np.array([float(r[f'lora_s{i}']) for i in range(1,17)])
            case_hns.setdefault(key,{})[r['module']]=np.array([float(r[f'hns_s{i}']) for i in range(1,17)])
    with gzip.open(dp,'rt') as f:
        for r in csv.DictReader(f):
            if r['adapter']!='lora':
                continue
            key=(r['base_model'],r['task']); n='base_model.model.model.'+r['module']
            case_energy.setdefault(key,{}).setdefault(n,np.full(16,np.nan))[int(r['direction'])-1]=float(r['response_energy'])
    available={}
    for c in source_audit:
        if c['seed']!=42:
            continue
        key=(c['base'],c['task'])
        activation_root=Path(manifest['task_overrides'].get(':'.join(key),str(Path(manifest['models'][c['base']])/c['task'])))
        ap=activation_root/'activation/activation_analysis.json'; a=lib.read(ap)
        assert str(Path(a['lora_path']).resolve())==c['source']
        assert a['samples']==256 and a['max_seq_len']==512 and a['seed']==42
        assert 'frozen' in a['trajectory'].lower()
        names=sorted(case_stats[key]); orig=np.array([case_stats[key][n] for n in names]); energy=np.array([case_energy[key][n] for n in names])
        assert len(names)==c['module_count'] and np.isfinite(energy).all() and (energy>0).all()
        available[c['source']]=(names,orig,energy,np.array([case_hns[key][n] for n in names]),a['hns_path'])
        cache_meta.append({'base':c['base'],'train_task':c['task'],'seed':42,'source':c['source'],'activation_manifest':str(ap),
            'activation_manifest_sha256':lib.sha(ap),'dataset_path':a['dataset_path'],'samples':a['samples'],'max_seq_len':a['max_seq_len'],
            'sample_indices_sha256':__import__('hashlib').sha256(json.dumps(a['sample_indices']).encode()).hexdigest()})
    adapted={}; checks=[]
    for path,source in dict.fromkeys((r['path'],r['source']) for r in rows if r['source'] in available):
        names,orig,energy,cached_hns,cached_hns_path=available[source]; p=Path(path)
        if path==source:
            target=orig.copy()
        elif (p/'common_basis_meta.json').exists():
            meta=lib.read(p/'common_basis_meta.json'); build=lib.read(p.parent/'manifest.json'); label=meta['variant']
            if label.startswith('common_global_'):
                gamma=float(label.removeprefix('common_global_').replace('p','.')); target=orig*gamma
            elif label=='common_per_module':
                target=orig*np.array([build['module_stats'][n]['per_module_gamma'] for n in names])[:,None]
            elif label=='common_hns':
                assert Path(build['source_hns']).resolve()==Path(cached_hns_path).resolve()
                target=cached_hns.copy()
            else:
                raise ValueError(label)
        elif (p/'posthoc_meta.json').exists():
            m=lib.read(p/'posthoc_meta.json'); target=np.array([[m['module_stats'][n]['target_level']]*16 for n in names])
        else:
            m=lib.read(p/'spectral_edit_meta.json'); target=np.array([m['module_stats'][n]['sigma_after'] for n in names])
        audit=next(x for x in lib.read(PREVIOUS/'spectrum_audit.json')['adapters'] if x['path']==path)
        z=np.load(audit['cache']); saved=z['s']/2
        error=float(np.max(np.linalg.norm(saved-np.sort(target,axis=1)[:,::-1],axis=1)/np.linalg.norm(saved,axis=1)))
        assert error<.005,(path,error)
        beta=target/orig; predicted=energy*beta**2; pi=predicted/predicted.sum(1)[:,None]
        er=np.exp(-(pi*np.log(pi)).sum(1)); masks=lib.scope_masks(names)
        gamma=float(np.sum(energy*beta)/energy.sum()); ratio=float(predicted.sum()/energy.sum())
        f={'functional_participation':float(np.median(1/(pi*pi).sum(1))),'functional_erank':float(np.median(er)),
            'functional_top1':float(np.median(pi.max(1))),'functional_erank_iqr':float(np.quantile(er,.75)-np.quantile(er,.25)),
            'functional_attn_mlp_gap':float(np.median(er[masks['attention']])-np.median(er[masks['mlp']])),
            'functional_energy_ratio':float(np.sqrt(ratio)), 'gamma_functional':gamma,'eta_functional':float(max(0,1-gamma*gamma/ratio))}
        adapted[path]={k:round(v,5) for k,v in f.items()}
        checks.append({'path':path,'max_target_vs_actual_spectrum_error':error})
    result=[dict(r,**adapted[r['path']]) for r in rows if r['source'] in available]
    for cohort,cp in dict.fromkeys((r['cohort'],r['checkpoint']) for r in result):
        group=[r for r in result if (r['cohort'],r['checkpoint'])==(cohort,cp)]; b,=[r for r in group if r['baseline']]
        for r in group:
            for f in FUNCTIONAL:
                r['delta__'+f]=r[f]-b[f]
    lib.write(OUT/'functional_features.json',result); lib.table(OUT/'functional_features.tsv',result)
    lib.write(OUT/'functional_audit.json',{'status':'pass','source_checkpoints':6,'seed':42,'unique_adapters':len(adapted),'rows':len(result),
        'projection':'frozen-base cached diagonal response energies reweighted by intended shared-U/V spectrum beta^2; saved-spectrum validation; no new activations',
        'source_manifests':cache_meta,'inputs':[{'path':str(p),'sha256':lib.sha(p)} for p in (sp,dp,root/'source_manifest.json')],
        'saved_spectrum_checks':checks,'limitations':'training-distribution trajectories, six original seed42 sources; no raw hidden-state covariance or seeds43/44'} )
    print('[Functional]',len(result),'rows',len(adapted),'adapters, six sources',flush=True)
    return result

def base_features(rows):
    """Approximate pretrained top subspaces, reused across all seeds and interventions."""
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    cache=OUT/'base_subspaces'; cache.mkdir(exist_ok=True)
    source_manifest=lib.read(lib.SOURCE/'source_manifest.json')['checkpoints']
    old_audit={r['path']:r for r in lib.read(PREVIOUS/'spectrum_audit.json')['adapters']}
    features={}; audit=[]; started=time.monotonic(); completed=0
    for base in dict.fromkeys(r['base'] for r in rows):
        c=next(r for r in source_manifest if r['base']==base)
        model=Path(c['base_model']); index_path=model/'model.safetensors.index.json'; ix=lib.read(index_path)
        basis={}; nlist=sorted(c['module_shapes'])
        for i,n in enumerate(nlist):
            weight_key='model.layers.'+n.split('.layers.',1)[1]+'.weight'
            shard=model/ix['weight_map'][weight_key]
            key=hashlib.sha256((str(shard)+weight_key+str(shard.stat().st_size)+str(shard.stat().st_mtime_ns)+'q96n12-v1').encode()).hexdigest()[:24]
            p=cache/f'{key}.npz'
            if p.exists():
                z=np.load(p); u,v,s,residual=z['u'],z['v'],z['s'],z['residual']
                q,niter=int(z['q']),int(z['niter'])
            else:
                with safe_open(str(shard),framework='pt',device='cpu') as f:
                    w=f.get_tensor(weight_key).float()
                for q,niter in ((96,12),(128,24),(192,36)):
                    torch.manual_seed(20260914+i)
                    with torch.inference_mode():
                        ut,st,vt=torch.svd_lowrank(w,q=min(q,min(w.shape)),niter=niter)
                        residual=np.array([float(torch.sqrt(((w@vt[:,:k]-ut[:,:k]*st[:k])**2).sum()+((w.T@ut[:,:k]-vt[:,:k]*st[:k])**2).sum())/torch.linalg.vector_norm(st[:k])) for k in (4,16,32)])
                    if residual[1]<.01 and residual[2]<.02:
                        break
                u,v,s=ut[:,:32].numpy(),vt[:,:32].numpy(),st[:33].numpy()
                np.savez_compressed(p,u=u,v=v,s=s,residual=residual,q=q,niter=niter)
                del w,ut,vt,st
            assert residual[1]<.01 and residual[2]<.02,(base,n,residual)
            basis[n]=(torch.from_numpy(u),torch.from_numpy(v))
            audit.append({'base':base,'module':n,'weight_key':weight_key,'shard':str(shard),
                'cache':str(p),'q':q,'power_iterations':niter,'dual_relative_residual_k4':float(residual[0]),
                'dual_relative_residual_k16':float(residual[1]),'dual_relative_residual_k32':float(residual[2]),
                'sigma1':float(s[0]),'relative_boundary_gap_k32':float((s[31]-s[32])/s[31])})
            completed+=1
            lib.write(OUT/'progress.json',{'stage':'base_subspaces','modules':completed,'total_modules':476,'elapsed_seconds':time.monotonic()-started,'gpu_count':0})
            if (i+1)%16==0:
                print('[Base subspace]',base,i+1,len(nlist),'elapsed',round(time.monotonic()-started,1),flush=True)
        paths=list(dict.fromkeys(r['path'] for r in rows if r['base']==base))
        for i,path in enumerate(paths):
            state=lib.pairs(path); names=sorted(state); assert names==nlist
            sums={f:0. for f in BASE_FEATURES}; total=0.
            z=np.load(old_audit[path]['cache']); actual_f2=(z['s']**2).sum(1)/4
            for j,n in enumerate(names):
                a,b=state[n]['A'],state[n]['B']; u,v=basis[n]
                left=u.T@b; right=a@v; aa=a@a.T; bb=b.T@b
                total+=float(actual_f2[j])
                for k in (4,16,32):
                    l,r=left[:k],right[:,:k]
                    sums[f'base_left_overlap_k{k}']+=float(torch.trace(l@aa@l.T))
                    sums[f'base_right_overlap_k{k}']+=float(torch.trace(r.T@bb@r))
                    sums[f'base_bilateral_overlap_k{k}']+=float((l@r).square().sum())
            features[path]={f:round(max(0,x/total),6) for f,x in sums.items()}
            assert all(-1e-5<=x<=1+1e-5 for x in features[path].values())
            del state
            if (i+1)%32==0:
                print('[Base overlap]',base,i+1,len(paths),flush=True)
        del basis
    result=[dict(r,**features[r['path']]) for r in rows]
    for cohort,cp in dict.fromkeys((r['cohort'],r['checkpoint']) for r in result):
        group=[r for r in result if (r['cohort'],r['checkpoint'])==(cohort,cp)]; b,=[r for r in group if r['baseline']]
        for r in group:
            for f in BASE_FEATURES:
                r['delta__'+f]=r[f]-b[f]
    lib.write(OUT/'base_features.json',result); lib.table(OUT/'base_features.tsv',result)
    lib.write(OUT/'base_subspace_audit.json',{'status':'pass','method':'CPU randomized SVD, q96 power12; adapt q128/power24 or q192/power36 if residual exceeds gate',
        'k':[4,16,32],'modules':len(audit),'unique_adapters':len(features),'source_checkpoints':18,
        'limitations':'approximate pretrained principal subspace; boundary degeneracy may affect k32; k4/k16/k32 sensitivity reported, no exact-SVD claim',
        'module_audit':audit,'gpu_count':0})
    print('[Base complete]',len(audit),'pretrained modules',len(features),'adapters',flush=True)
    return result

def fdr(results,family_key='family'):
    for group in dict.fromkeys((r['cohort'],r['outcome'],r.get(family_key,'core')) for r in results):
        selected=[r for r in results if (r['cohort'],r['outcome'],r.get(family_key,'core'))==group and r.get('permutation_p') is not None]
        selected.sort(key=lambda r:r['permutation_p']); previous=1.; n=len(selected)
        for i in reversed(range(n)):
            previous=min(previous,selected[i]['permutation_p']*n/(i+1)); selected[i]['fdr_q']=float(previous)

def base_stability():
    """Second randomization / more iterations on predefined layers, without outcome access."""
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    audit=lib.read(OUT/'base_subspace_audit.json'); source=lib.read(lib.SOURCE/'source_manifest.json')['checkpoints']
    results=[]
    for base in dict.fromkeys(r['base'] for r in audit['module_audit']):
        entries=[r for r in audit['module_audit'] if r['base']==base]
        layers=[int(r['module'].split('.layers.')[1].split('.')[0]) for r in entries]; chosen={0,max(layers)//2}
        for i,r in enumerate(entries):
            layer=int(r['module'].split('.layers.')[1].split('.')[0])
            if layer not in chosen:
                continue
            z=np.load(r['cache']); u=torch.from_numpy(z['u']); v=torch.from_numpy(z['v'])
            with safe_open(r['shard'],framework='pt',device='cpu') as f:
                w=f.get_tensor(r['weight_key']).float()
            torch.manual_seed(20260914+10000+i)
            with torch.inference_mode():
                ua,sa,va=torch.svd_lowrank(w,q=min(128,min(w.shape)),niter=24)
            residual=float(torch.sqrt(((w@va[:,:16]-ua[:,:16]*sa[:16])**2).sum()+((w.T@ua[:,:16]-va[:,:16]*sa[:16])**2).sum())/torch.linalg.vector_norm(sa[:16]))
            item={'base':base,'module':r['module'],'second_q':128,'second_iterations':24,'second_dual_residual_k16':residual}
            for k in (4,16,32):
                item[f'left_rms_sine_k{k}']=float(np.sqrt(max(0,1-float((u[:,:k].T@ua[:,:k]).square().sum())/k)))
                item[f'right_rms_sine_k{k}']=float(np.sqrt(max(0,1-float((v[:,:k].T@va[:,:k]).square().sum())/k)))
            results.append(item); del w,ua,va,sa
            print('[Base stability]',len(results),base,r['module'],flush=True)
    lib.write(OUT/'base_stability_audit.json',{'status':'complete','sampled_modules':len(results),
        'sampling':'layer0 and floor(max_layer/2), all seven projection types, both bases; before looking at overlap-outcome associations',
        'comparison':'primary cached randomized SVD versus independent random seed q128 power24', 'modules':results})

def candidate_models(two=False):
    if not two:
        return [((f,),degree,ridge) for f in lib.CORE for degree in (1,2) for ridge in (.01,1.,10.)]
    return [((shape,strength),degree,ridge)
        for shape in ('erank_attn_mlp_gap','erank_iqr','erank_depth_slope','hill_tail_contrast','head_tail')
        for strength in ('fro_ratio','original_head1_retention') for degree in (1,2) for ridge in (.01,1.,10.)]

def source_correlations(rows,features):
    import itertools
    sources=[r for r in rows if r['cohort']=='joint_flat_hns' and r['baseline']]
    for r in sources:
        h=next(x for x in rows if x['cohort']=='joint_flat_hns' and x['checkpoint']==r['checkpoint'] and x['method']=='hns_f4_s1')
        r.update(hns_target_gain=h['target_gain'],hns_off_gain=h['off_gain'],hns_forgetting_reduction=h['forgetting_reduction'])
    groups=[np.array([i for i,r in enumerate(sources) if (r['base'],r['train_task'])==g]) for g in dict.fromkeys((r['base'],r['train_task']) for r in sources)]
    base_groups=[np.array([i for i,r in enumerate(sources) if r['base']==b]) for b in dict.fromkeys(r['base'] for r in sources)]
    permutations=list(itertools.permutations(range(3)))
    index=np.tile(np.arange(18),(6**6,1))
    for j,ix in enumerate(groups):
        p=np.array([permutations[(i//(6**j))%6] for i in range(6**6)])
        index[:,ix]=ix[p]
    result=[]
    for f in features:
        x=np.array([r[f] for r in sources],dtype=float)
        if not np.isfinite(x).all():
            continue
        xr=lib.center(x,groups,True)
        for k in ('target','off_score','forgetting_gap','hns_target_gain','hns_off_gain','hns_forgetting_reduction'):
            y=np.array([r[k] for r in sources]); yr=lib.center(y,groups,True); rho=lib.corr(xr,yr)
            den=np.linalg.norm(xr)*np.linalg.norm(yr)
            null=(yr[index]@xr)/den if den>1e-12 else np.full(len(index),np.nan)
            p=None if rho is None else float((np.abs(null)>=abs(rho)-1e-12).mean())
            result.append({'cohort':'source18','outcome':k,'feature':f,'family':'source','sources':18,
                'raw_spearman':lib.corr(lib.ranks(x),lib.ranks(y)),
                'within_base_spearman':lib.corr(lib.center(x,base_groups,True),lib.center(y,base_groups,True)),
                'within_base_task_spearman':rho,'within_base_task_pearson':lib.corr(lib.center(x,groups),lib.center(y,groups)),
                'permutation_p':p,'exact_seed_permutations':len(index)})
    fdr(result); lib.write(OUT/'source_correlations.json',result); lib.table(OUT/'source_correlations.tsv',result)
    return sources,result

def source_predict(train,test,feature,ridge):
    """Known Base×Task training-group means plus one centered spectral covariate."""
    keys=list(dict.fromkeys((r['base'],r['train_task']) for r in train)); means={}
    xc,yc=[],[]; outcomes=('target','forgetting_gap')
    for key in keys:
        group=[r for r in train if (r['base'],r['train_task'])==key]
        mx=float(np.mean([r[feature] for r in group])); my=np.mean([[r[k] for k in outcomes] for r in group],axis=0)
        means[key]=(mx,my)
    for r in train:
        mx,my=means[(r['base'],r['train_task'])]; xc.append(r[feature]-mx); yc.append(np.array([r[k] for k in outcomes])-my)
    xc=np.array(xc); yc=np.array(yc); sx=max(float(np.sqrt(np.mean(xc*xc))),1e-8)
    sy=np.maximum(np.std(yc,axis=0),.01); z=xc/sx
    coefficient=z@(yc/sy)/(z@z+ridge)
    predictions=[]; baseline=[]
    for r in test:
        mx,my=means[(r['base'],r['train_task'])]
        baseline.append(my); predictions.append(my+(r[feature]-mx)/sx*coefficient*sy)
    return np.array(predictions),np.array(baseline),sy

def source_cv(sources,features):
    specs=[(f,ridge) for f in features if np.isfinite(np.array([r[f] for r in sources],dtype=float)).all() and np.std([r[f] for r in sources])>1e-7 for ridge in (.01,1.,10.)]
    folds=[]; predictions=[]
    for seed in (42,43,44):
        train=[r for r in sources if r['seed']!=seed]; test=[r for r in sources if r['seed']==seed]
        losses=[]
        for f,ridge in specs:
            inner=[]
            for held in train:
                a=[r for r in train if r['checkpoint']!=held['checkpoint']]
                pred,base,sy=source_predict(a,[held],f,ridge)
                truth=np.array([[held['target'],held['forgetting_gap']]])
                inner.append(float(np.mean(((pred-truth)/sy)**2)))
            losses.append(float(np.mean(inner)))
        chosen=int(np.argmin(losses)); f,ridge=specs[chosen]
        pred,base,sy=source_predict(train,test,f,ridge)
        folds.append({'held_seed':seed,'feature':f,'ridge':ridge,'inner_joint_loss':losses[chosen],
            'train_checkpoints':[r['checkpoint'] for r in train],'test_checkpoints':[r['checkpoint'] for r in test],
            'train_residual_sd':sy.tolist()})
        for i,r in enumerate(test):
            predictions.append({'checkpoint':r['checkpoint'],'base':r['base'],'train_task':r['train_task'],'seed':seed,
                'feature':f,'ridge':ridge,'target':r['target'],'predicted_target':float(pred[i,0]),'group_mean_target':float(base[i,0]),
                'forgetting_gap':r['forgetting_gap'],'predicted_forgetting_gap':float(pred[i,1]),'group_mean_forgetting_gap':float(base[i,1])})
    summary={'sources':18,'design':'nested leave-seed-out; inner leave-source-out; train-only Base×Task means + one centered feature; linear only',
        'feature_selection':{f:sum(r['feature']==f for r in folds) for f in dict.fromkeys(r['feature'] for r in folds)}}
    for k in ('target','forgetting_gap'):
        y=np.array([r[k] for r in predictions]); p=np.array([r['predicted_'+k] for r in predictions]); b=np.array([r['group_mean_'+k] for r in predictions])
        summary[k+'_rmse']=float(np.sqrt(np.mean((y-p)**2)))
        summary[k+'_group_mean_rmse']=float(np.sqrt(np.mean((y-b)**2)))
        summary[k+'_skill_vs_group_mean']=float(1-np.sum((y-p)**2)/np.sum((y-b)**2))
        summary[k+'_r2']=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))
    lib.write(OUT/'source_cv_summary.json',summary); lib.write(OUT/'source_cv_folds.json',folds); lib.table(OUT/'source_cv_predictions.tsv',predictions)
    print('[Source seed CV]',summary,flush=True)
    return summary

def analyze(rows):
    primary=dict(PROPOSALS)
    if (OUT/'base_features.json').exists():
        extra={(r['cohort'],r['checkpoint'],r['method']):r for r in lib.read(OUT/'base_features.json')}
        rows=[extra[(r['cohort'],r['checkpoint'],r['method'])] for r in rows]
        primary.update({f:d for f,d in BASE_FEATURES.items() if f.endswith('k16')})
        secondary=[f for f in BASE_FEATURES if not f.endswith('k16')]
    else:
        secondary=[]
    lib.CORE=dict(primary,**INCOMPLETE)
    results=lib.correlations(rows,list(primary)+secondary)
    edited=lib.correlations([dict(r,cohort=r['cohort']+'__edited_only') for r in rows if not r['baseline']],list(primary)+secondary,seed=20260915)
    incomplete=[]
    for f in INCOMPLETE:
        valid=[r for r in rows if r[f] is not None]
        if valid:
            rec=lib.correlations(valid,[f],seed=20260916)
            for x in rec:
                x['missing_rows']=sum(r['cohort']==x['cohort'] for r in rows)-x['rows']
            incomplete.extend(rec)
    results.extend(incomplete); fdr(results)
    lib.write(OUT/'correlations.json',results); lib.table(OUT/'correlations.tsv',results)
    lib.write(OUT/'edited_only_correlations.json',edited); lib.table(OUT/'edited_only_correlations.tsv',edited)
    sources,sc=source_correlations(rows,list(primary)+secondary+list(INCOMPLETE)); source_cv(sources,list(primary))
    lib.CORE=dict(primary); lib.candidates=candidate_models
    lib.cross_validate(rows)
    functional=lib.read(OUT/'functional_features.json'); lib.CORE=dict(FUNCTIONAL)
    fc=lib.correlations(functional,list(FUNCTIONAL),seed=20260917)
    fe=lib.correlations([dict(r,cohort=r['cohort']+'__edited_only') for r in functional if not r['baseline']],list(FUNCTIONAL),seed=20260918)
    lib.write(OUT/'functional_correlations.json',fc); lib.table(OUT/'functional_correlations.tsv',fc)
    lib.write(OUT/'functional_edited_only_correlations.json',fe); lib.table(OUT/'functional_edited_only_correlations.tsv',fe)
    lib.CORE=dict(primary)
    # Functional CV only compares six complete source checkpoints; no full-three-seed claim.
    functional_core={k:FUNCTIONAL[k] for k in ('functional_participation','functional_erank','functional_energy_ratio')}
    functional_core.update({k:PROPOSALS[k] for k in ('erank_attn_mlp_gap','fro_ratio','original_head1_retention')})
    def f_candidates(two=False):
        if not two:
            return [((f,),d,lam) for f in functional_core for d in (1,2) for lam in (.01,1.,10.)]
        return [((f,'functional_energy_ratio'),d,lam) for f in ('functional_participation','functional_erank') for d in (1,2) for lam in (.01,1.,10.)]
    param_out=lib.OUT; lib.OUT=OUT/'functional_cv'; lib.OUT.mkdir(exist_ok=True)
    lib.CORE=functional_core; lib.candidates=f_candidates
    lib.cross_validate([r for r in functional if r['cohort']=='joint_flat_hns'])
    lib.OUT=param_out; lib.CORE=primary
    report(rows)

def fmt(x,d=3):
    return lib.fmt(x,d)

def figures(source_result,corr):
    import os,sys
    sys.path.insert(0,str(PREVIOUS/'plot_dependencies'))
    os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache'); os.environ['XDG_CACHE_HOME']=str(OUT/'font_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=OUT/'figures'; figdir.mkdir(exist_ok=True)
    selected=['shannon_eff_rank','head_tail','hill_tail_contrast','erank_iqr','erank_depth_slope','erank_attn_mlp_gap','fro_ratio','original_head1_retention']
    if any(r['feature']=='base_bilateral_overlap_k16' for r in source_result):
        selected+=['base_left_overlap_k16','base_right_overlap_k16','base_bilateral_overlap_k16']
    fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
    for ax,outcome in zip(axes,('target','forgetting_gap')):
        data=[next(r for r in source_result if r['feature']==f and r['outcome']==outcome) for f in selected]
        y=np.arange(len(data)); ax.barh(y-.17,[r['raw_spearman'] if r['raw_spearman'] is not None else 0 for r in data],height=.32,label='Raw')
        ax.barh(y+.17,[r['within_base_task_spearman'] if r['within_base_task_spearman'] is not None else 0 for r in data],height=.32,label='Within Base x Task')
        ax.set(yticks=y,yticklabels=selected if outcome=='target' else [],xlabel='Spearman rho',xlim=(-1,1),title=outcome+'; original 18 LoRA')
        ax.axvline(0,c='grey',lw=.8); ax.invert_yaxis()
    axes[0].legend(); fig.suptitle('Only 3 source seeds per Base x Task; raw correlations can be confounded')
    for ext in ('png','pdf'):
        fig.savefig(figdir/f'source_confounds.{ext}',dpi=180)
    plt.close(fig)
    latest=[r for r in corr if r['cohort']=='joint_flat_hns' and r['feature'] in selected]
    lookup={(r['feature'],r['outcome']):r for r in latest}
    edited=lib.read(OUT/'edited_only_correlations.json'); elookup={(r['feature'],r['outcome']):r for r in edited if r['cohort']=='joint_flat_hns__edited_only'}
    fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
    for ax,outcome in zip(axes,('target_gain','forgetting_reduction')):
        y=np.arange(len(selected))
        ax.barh(y-.17,[lookup[(f,outcome)]['within_spearman'] or 0 for f in selected],height=.32,label='LoRA + interventions')
        ax.barh(y+.17,[elookup[(f,outcome)]['within_spearman'] or 0 for f in selected],height=.32,label='Interventions only')
        ax.set(yticks=y,yticklabels=selected if outcome=='target_gain' else [],xlabel='Within-checkpoint Spearman rho',xlim=(-1,1),title=outcome)
        ax.axvline(0,c='grey',lw=.8); ax.invert_yaxis()
    axes[0].legend(); fig.suptitle('Latest matched evaluation wave; source-checkpoint grouping')
    for ext in ('png','pdf'):
        fig.savefig(figdir/f'intervention_sensitivity.{ext}',dpi=180)
    plt.close(fig)

def report(rows):
    if (OUT/'base_features.json').exists():
        rows=lib.read(OUT/'base_features.json')
    corr=lib.read(OUT/'correlations.json'); edited=lib.read(OUT/'edited_only_correlations.json')
    sc=lib.read(OUT/'source_correlations.json'); cv=lib.read(OUT/'cv_summary.json'); rankings=lib.read(OUT/'cv_rankings.json')
    source_summary=lib.read(OUT/'source_cv_summary.json'); functional=lib.read(OUT/'functional_correlations.json')
    source_lookup={(r['feature'],r['outcome']):r for r in sc}
    lookup={(r['cohort'],r['feature'],r['outcome']):r for r in corr}
    elookup={(r['cohort'].removesuffix('__edited_only'),r['feature'],r['outcome']):r for r in edited}
    fl={(r['cohort'],r['feature'],r['outcome']):r for r in functional}
    primary=list(PROPOSALS)+([f for f in BASE_FEATURES if f.endswith('k16')] if (OUT/'base_features.json').exists() else [])
    figures(sc,corr)
    source_rows=[r for r in rows if r['cohort']=='joint_flat_hns' and r['baseline']]
    selected=[r for r in rows if r['cohort']=='joint_flat_hns']
    quantities=primary+['target','off_score','forgetting_gap','target_gain','off_gain','forgetting_reduction']
    mean_rows=[]
    for base,task,method in dict.fromkeys((r['base'],r['train_task'],r['method']) for r in selected):
        group=[r for r in selected if (r['base'],r['train_task'],r['method'])==(base,task,method)]
        assert sorted(r['seed'] for r in group)==[42,43,44]
        for q in quantities:
            values=np.array([r[q] for r in group]); mean_rows.append({'base':base,'train_task':task,'method':method,'quantity':q,
                'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),**{f'seed{r["seed"]}':r[q] for r in group}})
    lib.table(OUT/'three_seed_mean_sd.tsv',mean_rows)
    def source_cell(f,k,key='within_base_task_spearman'):
        return fmt(source_lookup[(f,k)].get(key)) if (f,k) in source_lookup else '—'
    def edit_cell(cohort,f,k,only=False):
        table=elookup if only else lookup
        return fmt(table[(cohort,f,k)]['within_spearman']) if (cohort,f,k) in table else '—'
    source_info=source_lookup[('erank_iqr','target')]
    fedit=lib.read(OUT/'functional_edited_only_correlations.json')
    fel={(r['cohort'].removesuffix('__edited_only'),r['feature'],r['outcome']):r for r in fedit}
    lines=['# 有希望的谱量追加试验：三种子、两模型、三任务（2026-09-14）','',
        '追加检验用户提出的重尾、头尾结构、跨模块/跨层分布、attention–MLP 差异、功能参与率及 Base 主子空间 overlap。参数谱复用实际 adapter compact SVD 缓存；原始 downstream / forgetting 评测不变，各评测波次分开。数据仍是 18 个源 checkpoint、282 adapters、450 条方法记录；独立训练分组是 18，而不是 module 或 token 数量。','',
        '**本轮有值得保留的探索性信号，但没有找到在跨模型、跨任务留出后，稳定同时预测 downstream 和 forgetting 的最优谱量。** 新结构量提供了不同的描述轴；关联较大不代表增量预测已成立。','',
        f'未编辑源 checkpoint 的 erank_iqr 与 task score：去除 Base×Task 后 ρ={fmt(source_info["within_base_task_spearman"])}, 精确组内 seed 置换 p={fmt(source_info["permutation_p"],4)}, FDR q={fmt(source_info.get("fdr_q"))}，多重比较后不足以确认。erank_depth_slope 与绝对 FG 的组内 ρ={source_cell("erank_depth_slope","forgetting_gap")}, q={source_cell("erank_depth_slope","forgetting_gap","fdr_q")}。这些是有限 n=18、每组3种子的候选关联，尚未验证为选择规则。','',
        f'erank_attn_mlp_gap 与绝对 FG 的 raw ρ={source_cell("erank_attn_mlp_gap","forgetting_gap","raw_spearman")}，去除 Base×Task 后变为 {source_cell("erank_attn_mlp_gap","forgetting_gap")}。head_tail 与 HNS target gain 的 raw ρ={source_cell("head_tail","hns_target_gain","raw_spearman")}，去除组间差异后为 {source_cell("head_tail","hns_target_gain")}。在当前18-source数据上不能据 raw 相关声称它们不受混杂或已复现用户表中的最佳指标。','',
        f'源 checkpoint 的嵌套留 seed 单量选择未超过 training-group mean：Target RMSE={fmt(source_summary["target_rmse"])} vs {fmt(source_summary["target_group_mean_rmse"])} pp；FG RMSE={fmt(source_summary["forgetting_gap_rmse"])} vs {fmt(source_summary["forgetting_gap_group_mean_rmse"])} pp。即使整体 Target R²较高，也主要来自已知模型/任务均值，而非谱量增益。','',
        f'功能 PR 在六个 seed42 源上的最新 within-checkpoint ρ：ΔTarget={fmt(fl[("joint_flat_hns","functional_participation","target_gain")]["within_spearman"])}, FG reduction={fmt(fl[("joint_flat_hns","functional_participation","forgetting_reduction")]["within_spearman"])}；edited-only 后为 {fmt(fel[("joint_flat_hns","functional_participation","target_gain")]["within_spearman"])}/{fmt(fel[("joint_flat_hns","functional_participation","forgetting_reduction")]["within_spearman"])}。在这批有限干预中 PR、entropy 与 top1 share 给出相同（或反向）的 outcome 秩相关，尚不能声称 full-distribution 功能量更优。','',
        '实际用途：保留 erank_iqr / depth slope 作为空间分布候选，保留 functional PR/entropy 作为真实 activation 加权诊断，同时保留 Fro ratio 与原始 head retention 作幅度/方向对照。Base overlap 是另一轴的近似诊断；当前留出结果没有支持它成为跨模型通用最优量。', '',
        '**阅读时区分三个问题：未编辑源 checkpoint 的质量；某个源 checkpoint 是否更适合 HNS；同一 checkpoint 的已编辑方法谁更好。它们需要不同的 outcome 和统计分组。** 用户表中的已有 ρ 不作复现事实，以下全部重新计算。','',
        '## 定义与数值约定','',
        '本轮 concentration/shape 量按 module median 聚合，erank/stable rank 不除以 16；上轮的 mean erank/16 单独作为 aggregation comparator。IQR、层斜率和 attention–MLP gap 独立报告。原始 rank=16、alpha=32、scaling=2，不改变 adapters。', '',
        lib.md_table(['谱量','本轮定义'],[[f,d] for f,d in dict(PROPOSALS,**INCOMPLETE,**{f:BASE_FEATURES[f] for f in primary if f in BASE_FEATURES}).items()]),'',
        r'$\lambda_i=(c\sigma_i)^2$，Hill 使用 $k=8$，并给出 $k=4,12$ 敏感性。$\alpha=1+1/C_{tail}$，$C_{tail}=\frac1k\sum_{i\le k}\log(\lambda_i/\lambda_{k+1})$；完全平坦时 C=0、alpha 无穷/未定义，不能加 epsilon 或截断成一个看似正常的有限指数。','',
        'rank16 只有很少谱点，本轮 alpha_hill 是固定 k 的 LoRA 谱诊断，不是证实了 heavy-tail 分布，也不是 WeightWatcher 官方的 power-law fit。alpha_weighted_hill 明确使用 Hill 估计与 scaled operator norm 的联合量。[WeightWatcher 官方说明](https://weightwatcher.ai/fine_tuned.html) 也指出低秩更新的谱拟合存在 small-n 限制。', '',
        '相对谱跨度 ≤1e-5 时按平坦端点处理；Hill 的 mean-log contrast ≤1e-4 时记录 None。Kurtosis 与 decay-family 在零方差谱上也未定义。Checkpoint 层 alpha/kurtosis/family 只有所有 modules 定义良好时才汇总，不默默删除难拟合 modules；valid-module fraction 逐记录保存。finite tail-contrast 保留 flat endpoint 0，可进入完整范围 CV。所有量化与判定阈值用于保存误差控制，没有通过成绩挑选阈值。','',
        r'核 participation $r_2=(\sum\sigma)^2/\sum\sigma^2$ 与固定核预算的 flat-Fro ratio 满足 $\gamma_F^2=r_2/r$；但逐模块代数关系并不自动让不同跨模块聚合方式完全等价。$\gamma_{param}$ 和 $\eta_{param}$ 的权重明确是全模型参数谱能量权重，分别复用上轮 scalar_fit / shape_residual，不能当成独立的新发现。','',
        '## 18 个原始 LoRA：源谱量与质量 / 遗忘 / HNS 收益','',
        'Raw Spearman 跨所有模型任务，仅用于显示混杂。Within Base×Task Spearman 在六个组内部各自对三个 source seeds 排名、去均值后聚合。精确 p 枚举 6^6=46656 个组内 seed 排列；按 outcome 对全部源谱候选做 BH FDR。这里绝对 FG 的正相关表示更多遗忘；HNS FG reduction 的正相关表示恢复更多。', '',
        lib.md_table(['谱量','Target raw ρ','Target 去组间 ρ','Target q','FG raw ρ','FG 去组间 ρ','HNS ΔTarget 去组间 ρ','HNS FG reduction 去组间 ρ'],[
            [f,source_cell(f,'target','raw_spearman'),source_cell(f,'target'),source_cell(f,'target','fdr_q'),
                source_cell(f,'forgetting_gap','raw_spearman'),source_cell(f,'forgetting_gap'),source_cell(f,'hns_target_gain'),source_cell(f,'hns_forgetting_reduction')]
            for f in primary+list(INCOMPLETE)]), '',
        '![Source confounding](promising_spectral_metrics_three_seed_20260914/figures/source_confounds.png)','',
        '## 源 checkpoint 的留 seed 预测','',
        '只使用 18 个未编辑 LoRA，外层留 seed42/43/44；每折训练集内留 source checkpoint，选择一个线性谱量与 ridge∈{0.01,1,10}。训练 Base×Task 组均值是显式 baseline，再拟合组内中心化的单谱协变量。Feature/目标中心、尺度、baseline 均只使用训练 seeds。联合目标为 target 与绝对 FG，训练残差 SD 标准化后等权。', '',
        lib.md_table(['任务','谱量模型 RMSE','训练组均值 RMSE','Skill vs 组均值','整体 R²（含组别信息）'],[
            [k,fmt(source_summary[k+'_rmse']),fmt(source_summary[k+'_group_mean_rmse']),fmt(source_summary[k+'_skill_vs_group_mean']),fmt(source_summary[k+'_r2'])]
            for k in ('target','forgetting_gap')]),'',
        '折内选量：'+json.dumps(source_summary['feature_selection'],ensure_ascii=False)+'。完整逐 seed 预测与折内训练/测试 source 清单见 source_cv_predictions.tsv / source_cv_folds.json。整体 R² 可被模型/任务均值主导，判断谱量是否增益应看 Skill vs group mean。','',
        '## 同一 source 内的谱编辑关联','',
        '与上一轮一致：checkpoint 内含 ties 的 ranks，source-cluster bootstrap 2000 次，checkpoint 内置换 2000 次、BH FDR。完整数据同时展示 raw / within / 控制 log Frobenius 的 partial Pearson。', '',
        lib.md_table(['谱量','最新 ΔTarget ρ','最新 FG reduction ρ','最新 edited-only ΔTarget ρ','最新 edited-only FG ρ','HNS target网格 edited-only ρ','HNS retention网格 edited-only FG ρ'],[
            [f,edit_cell('joint_flat_hns',f,'target_gain'),edit_cell('joint_flat_hns',f,'forgetting_reduction'),
                edit_cell('joint_flat_hns',f,'target_gain',True),edit_cell('joint_flat_hns',f,'forgetting_reduction',True),
                edit_cell('hns_grid_target',f,'target_gain',True),edit_cell('hns_grid_retention',f,'forgetting_reduction',True)] for f in primary]),'',
        '![Intervention sensitivity](promising_spectral_metrics_three_seed_20260914/figures/intervention_sensitivity.png)','',
        '未定义 Hill/kurtosis/decay-family 的记录不进入完整范围候选 CV；仅对各量自己的有限子集计算补充关联，并明确报告 missing_rows。不同子集的相关大小不可直接作为最佳量排名。完全平坦与不同标量幅度会有相同 shape/functional PR，强关联也不等于能排序 Flat-Fro、Flat-Nuclear、HNS。','',
        '## 谱量能否预测未见干预','',
        '外层按 seed/base/task 完全留出源 checkpoint，内层留 source 选择 feature、一次/二次曲线、ridge。单量完整候选为上表可定义的参数/结构量以及 primary Base overlap k16；双量仅固定的五种结构/头尾量 × {Fro ratio,H1}。每批次使用自己的 LoRA baseline，目标 ΔTarget 与 FG reduction 等权标准化（只有 diagonal 的 HNS 网格仅预测 target）。HNS 主对照始终固定4+1，多配置只作为观测网格，不给每个 checkpoint 挑最佳设置。', '',
        lib.md_table(['批次','模型','留出','Skill vs train mean','Target R²','FG reduction R²','折内选量'],[
            [r['cohort'],r['model'],r['split'],fmt(r['skill_vs_train_mean']),fmt(r.get('target_gain_r2')),fmt(r.get('forgetting_reduction_r2')),json.dumps(r['feature_selection'],ensure_ascii=False)] for r in cv]),'',
        lib.md_table(['批次','候选族','全数据 inner winner（探索性）','次数','ridge','inner loss'],[
            [r['cohort'],r['model'],' + '.join(r['ranking'][0]['features']),r['ranking'][0]['degree'],r['ranking'][0]['ridge'],fmt(r['ranking'][0]['inner_cv_loss'])] for r in rankings]),'',
        '## 功能参与率与 entropy：六个 seed42 源 checkpoint 的诊断','',
        lib.md_table(['功能量','定义'],list(FUNCTIONAL.items())), '',
        '功能谱来自 frozen pretrained-base 的训练分布固定256样本、最长512 tokens；输入源路径与 source audit 逐一核验。保存的 response_energy 是 sigma_i²q_i，重新加权 beta_i² 得到不同谱编辑的方向能量。q 来自 cached source direction basis，不能把修改后重新排序的 spectrum 直接与它相乘。使用原 U/V 方向序列的目标谱并核验实际保存 spectrum，属于固定轨迹、共享 U/V 的诊断近似，未进行新的 forward inference。', '',
        lib.md_table(['功能量','最新 ΔTarget ρ','最新 FG reduction ρ','scalar ΔTarget ρ','scalar FG reduction ρ'],[
            [f,*[fmt(fl[(c,f,k)]['within_spearman']) for c,k in [('joint_flat_hns','target_gain'),('joint_flat_hns','forgetting_reduction'),('scalar_common_basis_seed42','target_gain'),('scalar_common_basis_seed42','forgetting_reduction')]]] for f in FUNCTIONAL]), '',
        '完整 edited-only 与六源 Base/task 留出 CV 在 functional_edited_only_correlations.tsv 和 functional_cv/。这里无 seed43/44 functional 数据，无法称作完整三种子功能谱验证；训练分布功能谱也不等价于 off-task 分布的损伤风险。', '',
        '## Base-subspace overlap 的计算审计','',
        '对两 Base 共476个被 LoRA 修改的 pretrained projection matrices 计算 top-k 主子空间，k∈{4,16,32}，k16 为主候选，k4/k32 为敏感性。left/right/bilateral 均按实际 adapter 更新参数能量全模型聚合。CPU randomized SVD q96、12次 power iteration，必要时增加到q128/24或q192/36，k16双侧相对残差<0.01、k32<0.02才通过。它是近似主子空间，不是 exact SVD；残差与k32边界gap逐module报告，k边界接近时解释需谨慎。', '',
        'Base 权重路径从原始 source manifest 的 base_model 与 safetensors index 解析，LoRA 权重路径复用已审计 manifest。用低秩因子直接计算 ||U0ᵀBA||、||BAV0||、||U0ᵀBAV0||；不构造稠密 LoRA BA，不修改原 Base 或 LoRA。', '',
        '## 完整逐种子主表','',
        lib.md_table(['Base','Task','Seed','Method','erank median','erank IQR','attn−MLP gap','depth slope','head_tail','F-ratio','Base bilateral k16','Target','Off','FG'],[
            [r['base'],r['train_task'],r['seed'],r['method'],fmt(r['shannon_eff_rank']),fmt(r['erank_iqr']),fmt(r['erank_attn_mlp_gap']),fmt(r['erank_depth_slope']),
                fmt(r['head_tail']),fmt(r['fro_ratio']),fmt(r.get('base_bilateral_overlap_k16'),5),fmt(r['target']),fmt(r['off_score']),fmt(r['forgetting_gap'])] for r in selected]), '',
        '## 三种子 mean ± sample SD','',
        lib.md_table(['Base','Task','Method','erank IQR','attn−MLP gap','head_tail','Target','FG'],[
            [base,task,method,*[next(fmt(r['mean'])+' ± '+fmt(r['sample_sd']) for r in mean_rows if (r['base'],r['train_task'],r['method'],r['quantity'])==(base,task,method,q)) for q in ('erank_iqr','erank_attn_mlp_gap','head_tail','target','forgetting_gap')]]
            for base,task,method in dict.fromkeys((r['base'],r['train_task'],r['method']) for r in selected)]), '',
        '全部核心谱量和性能/遗忘量的 seed42/43/44 值及 mean ± sample SD 在 three_seed_mean_sd.tsv。旧 seed42 training recipe 与新 seeds43/44 不同，两处历史 Llama seed42 标签未完全验证；18-source 分析与 seed预测仍是现有产物的回顾性检验。','',
        '## 重现与完整产物','',
        '```bash','/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/try_promising_spectral_metrics_three_seed.py --stage all','```','',
        'analysis_plan.json 记录定义、precision、模型候选与限制。本轮全程 CPU，torch 4 threads，GPU数0。绘图复用上一轮 isolated Matplotlib installation，不更改训练环境。','',
        '- features.tsv/json：全部450条新参数/结构谱量、原始score与source/adapter路径。',
        '- base_features.tsv/json、base_subspace_audit.json：450条overlap与476个Base modules 的计算诊断；大basis cache保留本地并从Git忽略。',
        '- base_weight_provenance.json：从 source manifest 解析的两Base权重、config和safetensors index的13个输入文件SHA256。',
        '- module_metrics.tsv：实际282个adapter逐module新谱量，非独立统计样本。',
        '- source_correlations.tsv/json：18源 checkpoint 品质、HNS收益的 raw / within 相关和精确置换。',
        '- source_cv_*：训练组均值对照、逐seed预测、折内源列表与选择。',
        '- correlations.tsv/json、edited_only_correlations.tsv/json：各wave、各outcome的完整关联、CI、FDR、partial。',
        '- cv_*：完整候选、内层选择、外层逐checkpoint预测及归一化系数。',
        '- functional_*、functional_cv/：六source cached功能谱、校验、完整关联和留出诊断。',
        '- three_seed_mean_sd.tsv、figures/*.png/pdf：逐seed统计与可导出科学图。','']
    if (OUT/'base_stability_audit.json').exists():
        stability=lib.read(OUT/'base_stability_audit.json')
        location=lines.index('## 完整逐种子主表')
        extra=['独立随机种子、更强 q128/power24 在预定义两层×七类投影×两Base（28 modules）复查：', '',
            lib.md_table(['k','左子空间 RMS sin(angle) 平均 / 最大','右子空间 RMS sin(angle) 平均 / 最大'],[
                [k,fmt(float(np.mean([r[f'left_rms_sine_k{k}'] for r in stability['modules']])),5)+' / '+fmt(max(r[f'left_rms_sine_k{k}'] for r in stability['modules']),5),
                    fmt(float(np.mean([r[f'right_rms_sine_k{k}'] for r in stability['modules']])),5)+' / '+fmt(max(r[f'right_rms_sine_k{k}'] for r in stability['modules']),5)] for k in (4,16,32)]), '',
            '该复查支持采样 modules 的近似稳定性，并不能证明所有 matrices 的 top-k 完全精确；k32 比 k4/k16 更依赖边界谱。','']
        lines[location:location]=extra
    path=lib.ROOT/'reports/promising_spectral_metrics_three_seed_20260914.md'; path.write_text('\n'.join(lines))
    inputs=[PREVIOUS/'manifest.json',PREVIOUS/'features.json',PREVIOUS/'spectrum_audit.json',OUT/'analysis_plan.json',Path(lib.__file__)]
    if (OUT/'base_weight_provenance.json').exists():
        inputs.append(OUT/'base_weight_provenance.json')
    manifest={'status':'complete','report':str(path),'completed_utc':datetime.now(timezone.utc).isoformat(),
        'source_checkpoints':18,'unique_adapters':282,'evaluation_rows':450,'core_parameter_features':len(primary),
        'incomplete_diagnostics':len(INCOMPLETE),'functional_source_checkpoints':6,'functional_seed':42,
        'gpu_count':0,'input_hashes':[{'path':str(p),'sha256':lib.sha(p)} for p in inputs], 'script_sha256':lib.sha(__file__),
        'report_sha256':lib.sha(path)}
    lib.write(OUT/'manifest.json',manifest)
    audit_results(rows)
    print('[Report complete]',path,flush=True)

def audit_results(rows):
    previous=lib.read(PREVIOUS/'features.json'); lookup={(r['cohort'],r['checkpoint'],r['label']):r for r in previous}
    assert len(rows)==450 and len(set(r['source'] for r in rows))==18 and len(set(r['path'] for r in rows))==282
    assert len({(r['cohort'],r['checkpoint'],r['label']) for r in rows})==450
    for r in rows:
        old=lookup[(r['cohort'],r['checkpoint'],r['label'])]
        for field in ('target','target_gain','off_score','off_gain','forgetting_gap','forgetting_reduction'):
            if field in old:
                assert r[field]==old[field]
        assert r['gamma_param']==old['scalar_fit'] and r['eta_param']==old['shape_residual']
        if r['method'] in ('flat_fro','flat_nuclear'):
            assert r['shannon_eff_rank']==r['r2_participation']==r['stable_rank']==16.
            assert r['d_flat_l1']==r['erank_iqr']==r['erank_attn_mlp_gap']==r['hill_tail_contrast']==0.
            assert r['alpha_hill'] is None and r['kurtosis'] is None and r['decay_family'] is None
        if 'base_left_overlap_k16' in r:
            for k in (4,16,32):
                left,right,both=[r[f'base_{s}_overlap_k{k}'] for s in ('left','right','bilateral')]
                assert 0<=both<=min(left,right)+1e-6 and max(left,right)<=1+1e-6
            for s in ('left','right','bilateral'):
                assert r[f'base_{s}_overlap_k4']<=r[f'base_{s}_overlap_k16']+1e-6<=r[f'base_{s}_overlap_k32']+2e-6
    f=lib.read(OUT/'functional_features.json'); assert len(f)==138 and len(set(r['source'] for r in f))==6
    assert all(r['seed']==42 and 1<=r['functional_participation']<=16 and 1<=r['functional_erank']<=16 for r in f)
    for fold in lib.read(OUT/'source_cv_folds.json'):
        assert set(fold['train_checkpoints']).isdisjoint(fold['test_checkpoints'])
        assert all(f'seed{fold["held_seed"]}' not in c for c in fold['train_checkpoints'])
    if (OUT/'base_subspace_audit.json').exists():
        b=lib.read(OUT/'base_subspace_audit.json'); assert b['modules']==476
        assert all(r['dual_relative_residual_k16']<.01 and r['dual_relative_residual_k32']<.02 for r in b['module_audit'])
    integrity={'status':'pass','rows':450,'source_checkpoints':18,'unique_adapters':282,
        'prior_performance_unchanged':'pass','alias_gamma_eta':'pass','flat_endpoints_and_undefined_diagnostics':'pass',
        'base_projector_bounds_and_nested_k':'pass','functional_scope_and_bounds':'pass','source_cv_seed_disjointness':'pass',
        'synthetic_shape_scale_invariance':'pass','gpu_count':0}
    lib.write(OUT/'result_audit.json',integrity)
    lib.write(OUT/'progress.json',{'stage':'complete','result_audit':'pass','gpu_count':0})
    manifest=lib.read(OUT/'manifest.json'); manifest.update(result_audit='pass',script_sha256=lib.sha(__file__))
    manifest['output_hashes']={str(p.relative_to(OUT)):lib.sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.tsv') and p.name!='manifest.json'}
    lib.write(OUT/'manifest.json',manifest)
    print('[Audit PASS]',integrity,flush=True)

def base_weight_provenance():
    sources=lib.read(lib.SOURCE/'source_manifest.json')['checkpoints']
    records=[]
    for model_path in dict.fromkeys(r['base_model'] for r in sources):
        root=Path(model_path)
        index=root/'model.safetensors.index.json'
        paths=[root/'config.json',index,*[root/s for s in sorted(set(lib.read(index)['weight_map'].values()))]]
        records.extend({'path':str(path),'sha256':lib.sha(path)} for path in paths)
    lib.write(OUT/'base_weight_provenance.json',records)
    print('[Base input hashes]',len(records),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['features','base','stability','provenance','analyze','report','all'],default='all'); args=p.parse_args()
    setup()
    if args.stage in ('features','all'):
        rows=compute_parameter_features(lib.read(PREVIOUS/'features.json')); functional_features(rows)
    if args.stage in ('base','all'):
        base_features(lib.read(OUT/'features.json'))
    if args.stage in ('stability','all'):
        base_stability()
    if args.stage in ('provenance','all'):
        base_weight_provenance()
    if args.stage in ('analyze','all'):
        analyze(lib.read(OUT/'features.json'))
    if args.stage=='report':
        report(lib.read(OUT/'features.json'))

if __name__=='__main__':
    main()
