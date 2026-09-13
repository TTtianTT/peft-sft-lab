#!/usr/bin/env python3
"""Retrospective, checkpoint-grouped spectral analysis; never trains or evaluates a model."""
import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from finetune.spectral_edit.io import (load_lora_state_dict, parse_lora_ab_key,
    find_adapter_weight_file, get_scaling_for_module)
from finetune.spectral_edit.svd import lowrank_svd_from_ba

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/spectral_performance_forgetting_three_seed_20260914'
SOURCE = ROOT / 'reports/posthoc_flat_dghard_three_seed_20260913'
FRESH = ROOT / 'reports/posthoc_flat_dghard_forgetting_three_seed_20260913'
LEGACY = ROOT / 'reports/hns_forgetting_20260911/git_artifacts/run_metadata'
TASKS = ('magicoder', 'metamath', 'tulu', 'commonsense')
CORE = {
    'entropy_rank': 'mean exp(-sum p log p)/r; p=s/sum(s)',
    'energy_entropy_rank': 'mean exp(-sum q log q)/r; q=s^2/sum(s^2)',
    'stable_rank': 'mean sum(s^2)/(r*max(s)^2)',
    'participation_rank': 'mean sum(s^2)^2/(r*sum(s^4))',
    'nuclear_participation_rank': 'mean sum(s)^2/(r*sum(s^2))',
    'top1_energy': 'mean max(s)^2/sum(s^2)',
    'top4_energy': 'mean sum(sorted(s)[:4]^2)/sum(s^2)',
    'log_condition': 'mean log(max(s)/min(s))',
    'log_fro': 'log sqrt(sum_modules sum(s^2)); s already includes LoRA scaling',
    'log_nuclear': 'log sum_modules sum(s); s already includes LoRA scaling',
    'log_head_rss': 'log sqrt(sum_modules max(s)^2); not global operator norm',
    'fro_ratio': 'global Frobenius / source global Frobenius',
    'nuclear_ratio': 'global sum nuclear / source global sum nuclear',
    'head_ratio': 'RSS of module operator norms / source RSS',
    'original_head1_retention': 'sum ||U_source[:,:1]^T D_new||_F^2 / sum source_s[:1]^2',
    'original_head4_retention': 'sum ||U_source[:,:4]^T D_new||_F^2 / sum source_s[:4]^2',
    'scalar_fit': '<D_new,D_source>_F / ||D_source||_F^2',
    'shape_residual': '1-<D_new,D_source>_F^2/(||D_new||_F^2*||D_source||_F^2)',
    'displacement': '||D_new-D_source||_F / ||D_source||_F',
}
SHAPE = list(CORE)[:8]
SCOPES = ('attention', 'mlp', 'early', 'middle', 'late', 'q_proj', 'k_proj',
    'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')

def read(p):
    return json.loads(Path(p).read_text())

def write(p, data):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def table(p, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'features': CORE,
        'secondary_scopes': SCOPES, 'secondary_features': ['entropy_rank','top1_energy','log_fro','original_head1_retention'],
        'outcomes': ['target_gain','off_gain','forgetting_reduction'],
        'independent_unit': '18 source checkpoints; modules and repeated evaluations are not independent replicates',
        'design': 'Separate evaluation waves. Baseline differences within checkpoint and cohort. DG identity and 0+0 controls excluded from fits. Shape and relative dimensionless features rounded at 1e-5; norm ratios within 1e-5 of 1 restored to exact invariant to avoid numerical noise becoming rank evidence.',
        'inference': 'Source-cluster bootstrap CI, within-checkpoint permutation p, BH FDR per cohort/outcome/feature family; raw correlations diagnostic only.',
        'validation': 'Nested grouped CV, leave seed/base/task out. Inner source-checkpoint folds select feature, linear/quadratic degree and ridge. Train-only outcome normalization. No target/HNS configuration chosen on held-out results.',
        'limitations': 'Retrospective association, not causality. Original seed42 training recipes differ from replication seeds; two Llama seed42 labels not fully verified.',
        'command': '/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/analyze_spectral_performance_forgetting_three_seed.py --stage all'}
    if not (OUT/'analysis_plan.json').exists():
        write(OUT/'analysis_plan.json', plan)
    else:
        existing=read(OUT/'analysis_plan.json'); existing['features']=CORE
        write(OUT/'analysis_plan.json',existing)
    sources = read(SOURCE/'source_manifest.json')['checkpoints']
    canonical_specs=read(FRESH/'Qwen3-8B_task_config.json')['evaluation_tasks']
    sample_counts={'magicoder':164,'metamath':1319,'tulu':541,'commonsense':22419}
    srcmap = {(c['base'],c['task'],c['seed']): c for c in sources}
    rows, inputs, protocols = [], set(), []
    def add_wave(cohort, base, seed, vp, sp, gp, full, allowed=None, seed_in_variants=False, cfg_override=None):
        inputs.update(map(str, (vp,sp,gp)))
        vm, sm, gm = read(vp), read(sp), read(gp)
        cfg_path=Path(cfg_override) if cfg_override else Path(vm['task_config']) if 'task_config' in vm else LEGACY/'experiment_config.json'
        specs=read(cfg_path)['evaluation_tasks']; inputs.add(str(cfg_path))
        assert all(specs[t['task']]==canonical_specs[t['task']] for t in gm['tasks']), ('Evaluation specifications differ',cohort,base,seed)
        for t in gm['tasks']:
            assert t['samples']==sample_counts[t['task']]
            assert t['max_tokens']==specs[t['task']]['max_new_tokens']
        # Generation manifest is authoritative for paths; copied release manifests may retain original path.
        assert Path(vm['base_model']).resolve() == Path(gm['base_model']).resolve()
        protocols.append({'cohort':cohort,'base':base,'seed':seed,'generation_manifest':str(gp),
            'evaluation_config':str(cfg_path),'tasks':gm['tasks'],'configuration':gm['configuration']})
        scored = {(r['task'],r['variant']): 100*r[r['primary_metric']] for r in sm['records']}
        assert all(r['samples']==sample_counts[r['task']] for r in sm['records'])
        if full:
            assert set(t['task'] for t in gm['tasks']) == set(TASKS)
        for v in vm['variants']:
            task = v.get('train_task',v['label'].split('__')[0])
            if task not in TASKS[:3]:
                continue
            if allowed and v.get('method') not in allowed:
                continue
            s = v['seed'] if seed_in_variants else seed
            src = srcmap[(base,task,s)]
            path = str(Path(v['path']).resolve())
            method = v.get('method',v['label'].split('__')[-1])
            # Grid method fields are generic; label defines the actual setting.
            if cohort.startswith('hns_grid'):
                method = v['label'].split('__')[-1]
            baseline = method in ('original_lora','common_lora')
            control = method in ('hns_f0_s0',)
            if baseline:
                if method == 'original_lora':
                    assert path == src['source']
                else:
                    assert str(Path(read(Path(path)/'common_basis_meta.json')['source_lora']).resolve()) == src['source']
            elif (Path(path)/'spectral_edit_meta.json').exists():
                assert str(Path(read(Path(path)/'spectral_edit_meta.json')['meta']['source_lora']).resolve()) == src['source']
            elif (Path(path)/'posthoc_meta.json').exists():
                meta = read(Path(path)/'posthoc_meta.json')
                assert str(Path(meta['source']).resolve()) == src['source']
            elif (Path(path)/'common_basis_meta.json').exists():
                assert str(Path(read(Path(path)/'common_basis_meta.json')['source_lora']).resolve()) == src['source']
            else:
                raise ValueError(('Unresolved source',path))
            values = {t:scored[(t,v['label'])] for t in (TASKS if full else (task,))}
            row = {'cohort':cohort,'base':base,'train_task':task,'seed':s,'method':method,
                'label':v['label'],'checkpoint':f'{base}/{task}/seed{s}', 'path':path,
                'source':src['source'],'baseline':baseline,'control':control,'target':values[task],
                'score_manifest':str(sp), 'variant_manifest':str(vp), 'generation_manifest':str(gp),'evaluation_config':str(cfg_path)}
            row.update({f'score_{k}':val for k,val in values.items()})
            if full:
                off = [t for t in TASKS if t!=task]
                row['off_score'] = float(np.mean([values[t] for t in off]))
                row['forgetting_gap'] = float(np.mean([max(0,scored[(t,'base')]-values[t]) for t in off]))
            rows.append(row)
    for base in dict.fromkeys(c['base'] for c in sources):
        add_wave('joint_flat_hns',base,None,FRESH/f'{base}_variant_manifest.json',
            FRESH/f'eval/{base}/score_manifest.json',FRESH/f'eval/{base}/generation_manifest.json',True,
            {'original_lora','flat_fro','flat_nuclear','hns_f4_s1'},True)
    for base, seed in dict.fromkeys((c['base'],c['seed']) for c in sources):
        c = next(c for c in sources if c['base']==base and c['seed']==seed)
        add_wave('hns_grid_target',base,seed,c['source_variant_manifest'],c['source_score_manifest'],c['source_generation_manifest'],False)
        if seed != 42:
            # Existing generation manifests found under the run recorded by the source audit.
            gp = Path(c['source_generation_manifest']).parent.parent.parent/'forgetting/eval/generation_manifest.json'
            gm = read(gp)
            # run_hns_seed_forgetting_b300.sh passes this override, not the grid variant manifest's 3-task config.
            inputs.add(str(ROOT/'scripts/run_hns_seed_forgetting_b300.sh'))
            add_wave('hns_grid_retention',base,seed,gm['variant_manifest'],gp.parent/'score_manifest.json',gp,True,cfg_override=gp.parent.parent/'task_config.json')
    for base, slug in [('Qwen3-8B','qwen'),('Llama-3.1-8B-Instruct','llama')]:
        gp = LEGACY/f'generation_manifest__formal_{slug}.json'
        gm = read(gp)
        add_wave('scalar_common_basis_seed42',base,42,gm['variant_manifest'],LEGACY/f'score_manifest__formal_{slug}.json',gp,True,
            {'common_lora','common_hns','common_per_module','global_0p25','global_0p40','global_0p55','global_0p70','global_0p85'})
    assert len(rows)==450, len(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r['cohort'],r['checkpoint']),[]).append(r)
    for group in groups.values():
        baseline, = [r for r in group if r['baseline']]
        for r in group:
            r['target_gain'] = r['target']-baseline['target']
            if 'off_score' in r:
                r['off_gain'] = r['off_score']-baseline['off_score']
                r['forgetting_reduction'] = baseline['forgetting_gap']-r['forgetting_gap']
    # Compare independently recomputed aggregates against published fresh and HNS retention tables.
    for cohort, refs in [('joint_flat_hns',read(FRESH/'summary.json')['checkpoint_rows']),
                         ('hns_grid_retention',read(ROOT/'reports/hns_seed_forgetting_20260913.json')['rows'])]:
        for r in [x for x in rows if x['cohort']==cohort]:
            setting = ('LoRA' if r['method']=='original_lora' else r['method'].removeprefix('hns_f').replace('_s','+'))
            ref = next(x for x in refs if x['base']==r['base'] and x['train_task']==r['train_task'] and x['seed']==r['seed'] and (x.get('method')==r['method'] or x.get('setting')==setting))
            assert all(abs(r[k]-ref[k])<1e-9 for k in ('target','off_score','forgetting_gap'))
    write(OUT/'inventory.json',rows)
    table(OUT/'inventory.tsv',rows)
    manifest = {'status':'inventory_complete','source_checkpoints':18,'evaluation_rows':len(rows),
        'benchmark_cells':sum(sum('score_'+t in r for t in TASKS) for r in rows),
        'evaluated_examples_in_selected_cells':sum(sum(sample_counts[t] for t in TASKS if 'score_'+t in r) for r in rows),
        'evaluation_specification_audit':'pass; all evaluated task specs equal canonical specs, all metric-record and generation sample counts / token budgets checked; retention config override resolved from original worker script',
        'unique_adapter_paths':len({r['path'] for r in rows}),'cohorts':{c:sum(r['cohort']==c for r in rows) for c in dict.fromkeys(r['cohort'] for r in rows)},
        'input_manifests':[{'path':p,'sha256':sha(p)} for p in sorted(inputs|{str(SOURCE/'source_manifest.json')})],
        'protocols':protocols,'independent_unit':'source checkpoint','gpu_count':0,'script_sha256':sha(__file__)}
    write(OUT/'manifest.json',manifest)
    print('Inventory', manifest['cohorts'], 'unique adapters',manifest['unique_adapter_paths'],flush=True)
    return sources,rows

def pairs(path):
    state,_ = load_lora_state_dict(str(path))
    result = {}
    for key,val in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed:
            prefix,which,adapter = parsed
            assert adapter is None, (key,adapter)
            result.setdefault(prefix,{})[which] = val.float()
    assert result and all(set(x)=={'A','B'} for x in result.values())
    return result

def scope_masks(names):
    layers = np.array([int(re.search(r'\.layers\.(\d+)\.',n)[1]) for n in names])
    thirds = np.minimum(2,3*layers//(int(layers.max())+1))
    masks = {'all':np.ones(len(names),dtype=bool),'attention':np.array(['.self_attn.' in n for n in names]),
        'mlp':np.array(['.mlp.' in n for n in names])}
    masks.update({name:thirds==i for i,name in enumerate(('early','middle','late'))})
    masks.update({name:np.array([n.endswith('.'+name) for n in names]) for name in SCOPES[5:]})
    return masks

def feature_values(s, source_s, head, inner, masks):
    result={}
    for scope,mask in masks.items():
        a,orig = s[mask],source_s[mask]
        l1,l2 = a.sum(1),np.square(a).sum(1)
        p,q = a/l1[:,None],a*a/l2[:,None]
        rank = a.shape[1]
        f = {'entropy_rank':float(np.exp(-(p*np.log(np.maximum(p,1e-300))).sum(1)).mean()/rank),
            'energy_entropy_rank':float(np.exp(-(q*np.log(np.maximum(q,1e-300))).sum(1)).mean()/rank),
            'stable_rank':float((l2/(rank*a[:,0]**2)).mean()),
            'participation_rank':float((l2*l2/(rank*(a**4).sum(1))).mean()),
            'nuclear_participation_rank':float((l1*l1/(rank*l2)).mean()),
            'top1_energy':float(q[:,0].mean()),'top4_energy':float(q[:,:4].sum(1).mean()),
            'log_condition':float(np.log(a[:,0]/np.maximum(a[:,-1],1e-300)).mean()),
            'log_fro':float(np.log(np.sqrt(l2.sum()))),'log_nuclear':float(np.log(l1.sum())),
            'log_head_rss':float(np.log(np.linalg.norm(a[:,0]))),
            'fro_ratio':float(np.sqrt(l2.sum()/(orig**2).sum())),
            'nuclear_ratio':float(l1.sum()/orig.sum()),'head_ratio':float(np.linalg.norm(a[:,0])/np.linalg.norm(orig[:,0])),
            'original_head1_retention':float(head[mask,0].sum()/(orig[:,:1]**2).sum()),
            'original_head4_retention':float(head[mask,1].sum()/(orig[:,:4]**2).sum()),
            'scalar_fit':float(inner[mask].sum()/(orig**2).sum()),
            'shape_residual':float(max(0,1-inner[mask].sum()**2/(l2.sum()*(orig**2).sum()))),
            'displacement':float(np.sqrt(max(0,l2.sum()+(orig**2).sum()-2*inner[mask].sum())/(orig**2).sum()))}
        if scope!='all':
            f = {k:f[k] for k in ('entropy_rank','top1_energy','log_fro','original_head1_retention')}
        # Enforced budgets are mathematically constant. Tiny SVD/save roundoff must not
        # yield artificial rank correlations of a preserved norm against the outcome.
        source_fro=float(np.linalg.norm(orig)); source_nuclear=float(orig.sum())
        if abs(np.exp(f['log_fro'])/source_fro-1)<1e-5:
            f['log_fro']=float(np.log(source_fro))
            if 'fro_ratio' in f:
                f['fro_ratio']=1.
        if 'nuclear_ratio' in f and abs(f['nuclear_ratio']-1)<1e-5:
            f['nuclear_ratio']=1.
            f['log_nuclear']=float(np.log(source_nuclear))
        for k,val in f.items():
            # Float32 saves cannot identify meaningful invariant differences at 1e-7.
            if k in SHAPE or k in CORE and not k.startswith('log_'):
                val = round(val,5)
            result[k if scope=='all' else f'{scope}__{k}'] = val
    return result

def extract(sources, rows):
    torch.set_num_threads(4)
    cache_dir = OUT/'spectra'; cache_dir.mkdir(exist_ok=True)
    checkpoint_features, audit, modules = {},[],[]
    done = 0; started = time.monotonic()
    for c in sources:
        subset = [r for r in rows if r['source']==c['source']]
        paths = list(dict.fromkeys(r['path'] for r in subset))
        orig_pairs = pairs(c['source']); names = sorted(orig_pairs)
        assert len(names)==c['module_count']
        cfg = c['adapter_config']
        original={}; orig_s=[]; scaling=[]
        with torch.inference_mode():
            for n in names:
                a,b = orig_pairs[n]['A'],orig_pairs[n]['B']
                u,s,vh,v = lowrank_svd_from_ba(b,a)
                original[n]=(u,s,vh)
                scale=get_scaling_for_module(cfg,n)
                orig_s.append(s.double().numpy()*scale); scaling.append(scale)
        orig_s=np.array(orig_s); masks=scope_masks(names)
        for path in paths:
            weight,_ = find_adapter_weight_file(path)
            digest=sha(weight); config_hash=sha(Path(path)/'adapter_config.json')
            assert config_hash==c['config_sha256']
            key=hashlib.sha256((path+digest+c['source_sha256']+'actual-v1').encode()).hexdigest()[:24]
            cached=cache_dir/f'{key}.npz'
            is_source=path==c['source']
            if is_source:
                assert digest==c['source_sha256'], ('Source changed since 18-checkpoint audit',path)
            if cached.exists():
                z=np.load(cached); s,head,inner=z['s'],z['head'],z['inner']
            else:
                current=orig_pairs if is_source else pairs(path)
                assert sorted(current)==names
                actual_s=[]; head=[]; inner=[]
                with torch.inference_mode():
                    for i,n in enumerate(names):
                        u,original_sigma,vh = original[n]
                        a,b=current[n]['A'],current[n]['B']
                        if is_source:
                            spectrum=original_sigma
                            projection=torch.diag(original_sigma)
                            h1=float(original_sigma[0].double().square())
                            h4=float(original_sigma[:4].double().square().sum())
                        else:
                            spectrum=lowrank_svd_from_ba(b,a)[1]
                            left=u.T @ b
                            projection=left @ (a @ vh.T)
                            # Row projection norm includes any components outside original right subspace.
                            aa=a @ a.T
                            h1=float((left[:1] @ aa @ left[:1].T).double().trace())
                            h4=float((left[:4] @ aa @ left[:4].T).double().trace())
                        scale=scaling[i]
                        actual_s.append(spectrum.double().numpy()*scale)
                        head.append([max(0,h1)*scale*scale,max(0,h4)*scale*scale])
                        inner.append(float((projection.diag().double()*original_sigma.double()).sum())*scale*scale)
                s,head,inner=np.array(actual_s),np.array(head),np.array(inner)
                np.savez_compressed(cached,s=s,head=head,inner=inner,names=np.array(names))
                del current
            assert np.isfinite(s).all() and (s>=0).all()
            features=feature_values(s,orig_s,head,inner,masks)
            checkpoint_features[path]=features
            # Verify cached build spectra against actual saved update, including direction-order sorting.
            metadata_path=Path(path)/'spectral_edit_meta.json'
            max_error=None
            if not is_source and metadata_path.exists() and not (Path(path)/'common_basis_meta.json').exists():
                meta=read(metadata_path)
                intended=np.array([np.sort(meta['module_stats'][n]['sigma_after'])[::-1]*scaling[i] for i,n in enumerate(names)])
                max_error=float(np.max(np.linalg.norm(s-intended,axis=1)/np.maximum(np.linalg.norm(intended,axis=1),1e-20)))
                assert max_error<0.01, (path,max_error)
            elif not is_source and (Path(path)/'posthoc_meta.json').exists():
                meta=read(Path(path)/'posthoc_meta.json')
                intended=np.array([[meta['module_stats'][n]['target_level']*scaling[i]]*16 for i,n in enumerate(names)])
                max_error=float(np.max(np.linalg.norm(s-intended,axis=1)/np.linalg.norm(intended,axis=1)))
                assert max_error<0.01
            audit.append({'checkpoint':subset[0]['checkpoint'],'path':path,'weight_sha256':digest,'config_sha256':config_hash,
                'modules':len(names),'max_saved_vs_intended_spectrum_relative_error':max_error,'cache':str(cached)})
            if is_source:
                for i,n in enumerate(names):
                    modules.append({'checkpoint':subset[0]['checkpoint'],'base':c['base'],'train_task':c['task'],'seed':c['seed'],
                        'module':n,'nuclear':float(s[i].sum()),'fro':float(np.linalg.norm(s[i])),'sigma1':float(s[i,0]),
                        'entropy_rank':float(np.exp(-(s[i]/s[i].sum()*np.log(s[i]/s[i].sum())).sum())),
                        'top1_energy':float(s[i,0]**2/(s[i]**2).sum())})
            done+=1
            print(f'[Spectrum] {done}/{len(set(r["path"] for r in rows))} {subset[0]["checkpoint"]} {Path(path).name}; elapsed={time.monotonic()-started:.1f}s',flush=True)
            write(OUT/'progress.json',{'stage':'extract','completed_adapters':done,'elapsed_seconds':time.monotonic()-started})
        del original, orig_pairs
    enriched=[]
    for r in rows:
        enriched.append(dict(r,**checkpoint_features[r['path']]))
    grouped={}
    for r in enriched:
        grouped.setdefault((r['cohort'],r['checkpoint']),[]).append(r)
    for group in grouped.values():
        baseline,=[r for r in group if r['baseline']]
        for r in group:
            for f in checkpoint_features[r['path']]:
                r['delta__'+f]=r[f]-baseline[f]
    write(OUT/'features.json',enriched); table(OUT/'features.tsv',enriched)
    write(OUT/'spectrum_audit.json',{'status':'pass','unique_adapters':len(audit),'module_svd_count':sum(r['modules'] for r in audit),'adapters':audit})
    table(OUT/'source_module_spectra.tsv',modules)
    manifest=read(OUT/'manifest.json'); manifest['status']='features_complete'; write(OUT/'manifest.json',manifest)
    return enriched

def ranks(x):
    """Average ranks, including exact ties and constants."""
    x=np.asarray(x); order=np.argsort(x,kind='stable'); result=np.empty(len(x),dtype=float)
    start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:
            end+=1
        result[order[start:end]]=(start+end-1)/2+1
        start=end
    return result

def corr(x,y):
    x=np.asarray(x)-np.mean(x); y=np.asarray(y)-np.mean(y)
    den=np.linalg.norm(x)*np.linalg.norm(y)
    return None if den<1e-12 else float(np.dot(x,y)/den)

def center(x,groups,rank=False):
    result=np.array(x,dtype=float)
    for ix in groups:
        a=ranks(result[ix]) if rank else result[ix]
        result[ix]=a-a.mean()
    return result

def correlations(rows, feature_names, n_boot=2000, n_perm=2000, seed=20260914):
    rng=np.random.default_rng(seed); results=[]
    for cohort in dict.fromkeys(r['cohort'] for r in rows):
        data=[r for r in rows if r['cohort']==cohort and not r['control']]
        groups=[np.array([i for i,r in enumerate(data) if r['checkpoint']==c]) for c in dict.fromkeys(r['checkpoint'] for r in data)]
        x=np.array([[r[f] for f in feature_names] for r in data]); xc=np.column_stack([center(x[:,j],groups) for j in range(x.shape[1])])
        xr=np.column_stack([center(x[:,j],groups,True) for j in range(x.shape[1])])
        norm=center(np.array([r['log_fro'] for r in data]),groups)
        xp=xc.copy()
        if norm@norm>1e-12:
            xp-=np.outer(norm,norm@xc/(norm@norm))
        # Cluster bootstrap keeps the entire intervention series of each sampled source.
        boot_counts=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=n_boot)
        perm_index=np.tile(np.arange(len(data)),(n_perm,1))
        for ix in groups:
            perm_index[:,ix]=ix[np.argsort(rng.random((n_perm,len(ix))),axis=1)]
        outcomes=[k for k in ('target_gain','off_gain','forgetting_reduction') if k in data[0]]
        for outcome in outcomes:
            y=np.array([r[outcome] for r in data]); yc=center(y,groups); yr=center(y,groups,True)
            yp=yc-norm*(norm@yc/(norm@norm)) if norm@norm>1e-12 else yc
            within=np.array([corr(xr[:,j],yr) if corr(xr[:,j],yr) is not None else np.nan for j in range(x.shape[1])])
            xx=np.array([(xr[ix]**2).sum(0) for ix in groups])
            xy=np.array([(xr[ix]*yr[ix,None]).sum(0) for ix in groups])
            yy=np.array([(yr[ix]**2).sum() for ix in groups])
            num=boot_counts@xy; den=np.sqrt((boot_counts@xx)*(boot_counts@yy)[:,None])
            boot=np.divide(num,den,out=np.full_like(num,np.nan),where=den>1e-12)
            permnum=yr[perm_index]@xr
            permden=np.linalg.norm(yr)*np.linalg.norm(xr,axis=0)
            permcorr=np.divide(permnum,permden[None,:],out=np.full_like(permnum,np.nan),where=permden[None,:]>1e-12)
            for j,f in enumerate(feature_names):
                valid=boot[:,j][np.isfinite(boot[:,j])]
                ci=np.quantile(valid,[.025,.975]).tolist() if len(valid)>n_boot*.8 else [None,None]
                per=[corr(ranks(x[ix,j]),ranks(y[ix])) for ix in groups]
                per=[z for z in per if z is not None]
                rec={'cohort':cohort,'outcome':outcome,'feature':f,'family':'core' if f in CORE else 'scope',
                    'rows':len(data),'source_clusters':len(groups),'raw_pearson':corr(x[:,j],y),
                    'raw_spearman':corr(ranks(x[:,j]),ranks(y)), 'within_pearson':corr(xc[:,j],yc),
                    'within_spearman':None if not np.isfinite(within[j]) else float(within[j]),
                    'cluster_ci_low':ci[0],'cluster_ci_high':ci[1],
                    'partial_pearson_logFro':corr(xp[:,j],yp),
                    'median_checkpoint_spearman':float(np.median(per)) if per else None,
                    'positive_checkpoints':sum(z>0 for z in per),'negative_checkpoints':sum(z<0 for z in per),
                    'nonconstant_checkpoints':len(per),
                    'permutation_p':None if not np.isfinite(within[j]) else float((1+(np.abs(permcorr[:,j])>=abs(within[j])-1e-12).sum())/(n_perm+1))}
                results.append(rec)
            for family in ('core','scope'):
                selected=[r for r in results if r['cohort']==cohort and r['outcome']==outcome and r['family']==family and r['permutation_p'] is not None]
                ordered=sorted(selected,key=lambda r:r['permutation_p']); m=len(ordered); prev=1.
                for i in reversed(range(m)):
                    prev=min(prev,ordered[i]['permutation_p']*m/(i+1)); ordered[i]['fdr_q']=float(prev)
            print(f'[Stats] {cohort} {outcome}: {len(feature_names)} features, {len(groups)} source clusters',flush=True)
    return results

def design(data, spec):
    fs,degree,lam=spec
    a=np.array([[r['delta__'+f] for f in fs] for r in data])
    return np.column_stack((a,a*a)) if degree==2 else a

def train_predict(train,test,spec,outcomes):
    a=design(train,spec); b=design(test,spec)
    sy=np.maximum(np.std([[r[k] for k in outcomes] for r in train],axis=0),1e-6)
    sx=np.maximum(np.sqrt(np.mean(a*a,axis=0)),1e-8)
    a=a/sx; b=b/sx
    y=np.array([[r[k] for k in outcomes] for r in train])/sy
    counts={c:sum(r['checkpoint']==c for r in train) for c in dict.fromkeys(r['checkpoint'] for r in train)}
    w=np.array([1/counts[r['checkpoint']] for r in train]); w=w/w.mean()
    lam=spec[2]
    coef=np.linalg.solve(a.T@(w[:,None]*a)+lam*np.eye(a.shape[1]),a.T@(w[:,None]*y))
    return (b@coef)*sy,sy,coef,sx

def candidates(two=False):
    if not two:
        return [((f,),d,lam) for f in CORE for d in (1,2) for lam in (.01,1.,10.)]
    # Declared small shape+strength family, no interactions or module-search in confirmatory CV.
    return [((s,n),d,lam) for s in ('entropy_rank','stable_rank','top1_energy','shape_residual')
        for n in ('log_fro','original_head1_retention','scalar_fit') for d in (1,2) for lam in (.01,1.,10.)]

def select_spec(train,outcomes,specs):
    groups=list(dict.fromkeys(r['checkpoint'] for r in train)); losses=[]
    for spec in specs:
        loss=[]
        for checkpoint in groups:
            inner_train=[r for r in train if r['checkpoint']!=checkpoint]
            inner_test=[r for r in train if r['checkpoint']==checkpoint]
            pred,sy,_,_=train_predict(inner_train,inner_test,spec,outcomes)
            truth=np.array([[r[k] for k in outcomes] for r in inner_test])
            loss.append(float(np.mean(((truth-pred)/sy)**2)))
        losses.append(float(np.mean(loss)))
    best=int(np.argmin(losses))
    return specs[best],losses[best],sorted([{'features':s[0],'degree':s[1],'ridge':s[2],'inner_cv_loss':loss} for s,loss in zip(specs,losses)],key=lambda r:r['inner_cv_loss'])

def cross_validate(rows):
    summaries,folds,predictions,rankings=[],[],[],[]
    for cohort in dict.fromkeys(r['cohort'] for r in rows):
        data=[r for r in rows if r['cohort']==cohort and not r['control'] and not r['baseline']]
        # Two-outcome objective always rewards target and forgetting equally after train-only normalization.
        outcomes=['target_gain','forgetting_reduction'] if 'forgetting_reduction' in data[0] else ['target_gain']
        for model in ('single','shape_plus_strength'):
            specs=candidates(model!='single')
            full_spec,full_loss,ranked=select_spec(data,outcomes,specs)
            _,full_sy,full_coef,full_sx=train_predict(data,data,full_spec,outcomes)
            rankings.append({'cohort':cohort,'model':model,'outcomes':outcomes,'ranking':ranked,
                'full_fit_feature_rms':full_sx.tolist(),'full_fit_outcome_sd':full_sy.tolist(),
                'full_fit_normalized_coefficients':full_coef.tolist()})
            print(f'[CV] {cohort} {model} full inner winner {full_spec}, loss={full_loss:.3f}',flush=True)
            for split in ('seed','base','train_task'):
                levels=list(dict.fromkeys(r[split] for r in data))
                if len(levels)<2:
                    continue
                collected=[]
                for held in levels:
                    train=[r for r in data if r[split]!=held]; test=[r for r in data if r[split]==held]
                    spec,inner_loss,_=select_spec(train,outcomes,specs)
                    pred,sy,coef,sx=train_predict(train,test,spec,outcomes)
                    mean=np.mean([[r[k] for k in outcomes] for r in train],axis=0)
                    truth=np.array([[r[k] for k in outcomes] for r in test])
                    fold={'cohort':cohort,'model':model,'split':split,'held_out':held,'train_clusters':len(set(r['checkpoint'] for r in train)),
                        'test_clusters':len(set(r['checkpoint'] for r in test)),'features':spec[0],'degree':spec[1],'ridge':spec[2],
                        'inner_cv_loss':inner_loss,'test_normalized_mse':float(np.mean(((pred-truth)/sy)**2)),
                        'train_feature_rms':sx.tolist(),'train_outcome_sd':sy.tolist(),'normalized_coefficients':coef.tolist(),
                        'zero_normalized_mse':float(np.mean((truth/sy)**2)),
                        'train_mean_normalized_mse':float(np.mean(((mean-truth)/sy)**2))}
                    folds.append(fold)
                    for i,r in enumerate(test):
                        rec={k:r[k] for k in ('cohort','checkpoint','base','train_task','seed','method')}
                        rec.update({'model':model,'split':split,'selected_features':'+'.join(spec[0]),'degree':spec[1],'ridge':spec[2]})
                        for j,k in enumerate(outcomes):
                            rec[k]=r[k]; rec['predicted_'+k]=float(pred[i,j]); rec['train_mean_'+k]=float(mean[j]); rec['train_sd_'+k]=float(sy[j])
                        predictions.append(rec); collected.append(rec)
                    print(f'[CV fold] {cohort}/{model} leave {split}={held}: {spec[0]} mse={fold["test_normalized_mse"]:.3f}',flush=True)
                fs=[f for f in folds if f['cohort']==cohort and f['model']==model and f['split']==split]
                summary={'cohort':cohort,'model':model,'split':split,'rows':len(collected),'clusters':len(set(r['checkpoint'] for r in collected)),
                    'normalized_mse':float(np.mean([f['test_normalized_mse'] for f in fs])),
                    'zero_normalized_mse':float(np.mean([f['zero_normalized_mse'] for f in fs])),
                    'train_mean_normalized_mse':float(np.mean([f['train_mean_normalized_mse'] for f in fs])),
                    'feature_selection':{f:sum('+'.join(z['features'])==f for z in fs) for f in dict.fromkeys('+'.join(z['features']) for z in fs)}}
                summary['skill_vs_zero']=1-summary['normalized_mse']/summary['zero_normalized_mse']
                summary['skill_vs_train_mean']=1-summary['normalized_mse']/summary['train_mean_normalized_mse']
                for k in outcomes:
                    truth=np.array([r[k] for r in collected]); pred=np.array([r['predicted_'+k] for r in collected])
                    summary[k+'_rmse']=float(np.sqrt(np.mean((truth-pred)**2)))
                    summary[k+'_r2']=float(1-np.sum((truth-pred)**2)/np.sum((truth-truth.mean())**2)) if np.std(truth)>1e-12 else None
                    summary[k+'_pearson']=corr(truth,pred)
                summaries.append(summary)
    write(OUT/'cv_rankings.json',rankings); write(OUT/'cv_folds.json',folds)
    table(OUT/'cv_predictions.tsv',predictions); write(OUT/'cv_summary.json',summaries); table(OUT/'cv_summary.tsv',summaries)
    return summaries,rankings

def grouped_diagnostics(rows):
    selected=[r for r in rows if r['cohort']=='joint_flat_hns']; sources=[r for r in selected if r['baseline']]
    source_corr=[]
    # Six base/task strata removed; only within-three-seed variation remains (18 sources, 12 residual df).
    groups=[np.array([i for i,r in enumerate(sources) if (r['base'],r['train_task'])==g]) for g in dict.fromkeys((r['base'],r['train_task']) for r in sources)]
    for f in CORE:
        for outcome in ('target','off_score','forgetting_gap'):
            x=np.array([r[f] for r in sources]); y=np.array([r[outcome] for r in sources])
            source_corr.append({'feature':f,'outcome':outcome,'sources':18,'raw_spearman':corr(ranks(x),ranks(y)),
                'within_base_task_pearson':corr(center(x,groups),center(y,groups)),
                'within_base_task_spearman':corr(center(x,groups,True),center(y,groups,True)),
                'interpretation':'exploratory n=18, only 3 seeds per stratum; not a validated predictor'})
    table(OUT/'source_checkpoint_correlations.tsv',source_corr)
    strata=[]
    for cohort in dict.fromkeys(r['cohort'] for r in rows):
        cohort_data=[r for r in rows if r['cohort']==cohort and not r['control']]
        for field in ('seed','base','train_task'):
            for value in dict.fromkeys(r[field] for r in cohort_data):
                data=[r for r in cohort_data if r[field]==value]
                groups=[np.array([i for i,r in enumerate(data) if r['checkpoint']==g]) for g in dict.fromkeys(r['checkpoint'] for r in data)]
                for f in CORE:
                    for outcome in ('target_gain','off_gain','forgetting_reduction'):
                        if outcome not in data[0]:
                            continue
                        strata.append({'cohort':cohort,'field':field,'value':value,'clusters':len(groups), 'feature':f,'outcome':outcome,
                            'within_spearman':corr(center(np.array([r[f] for r in data]),groups,True),center(np.array([r[outcome] for r in data]),groups,True))})
    table(OUT/'stratified_correlations.tsv',strata)
    means=[]
    for base,task,method in dict.fromkeys((r['base'],r['train_task'],r['method']) for r in selected):
        group=[r for r in selected if (r['base'],r['train_task'],r['method'])==(base,task,method)]
        assert sorted(r['seed'] for r in group)==[42,43,44]
        for f in (*CORE,'target','target_gain','off_score','off_gain','forgetting_gap','forgetting_reduction'):
            a=np.array([r[f] for r in group]); means.append({'base':base,'train_task':task,'method':method,'quantity':f,
                'n':3,'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1)),**{f'seed{r["seed"]}':r[f] for r in group}})
    table(OUT/'three_seed_mean_sd.tsv',means)
    # Same complete-flat shape, deliberately different budgets: a direct sufficiency counterexample.
    pairs=[]
    for cp in dict.fromkeys(r['checkpoint'] for r in selected):
        fro=next(r for r in selected if r['checkpoint']==cp and r['method']=='flat_fro')
        nuc=next(r for r in selected if r['checkpoint']==cp and r['method']=='flat_nuclear')
        pairs.append({'checkpoint':cp,'base':fro['base'],'train_task':fro['train_task'],'seed':fro['seed'],
            'flat_fro_entropy_rank':fro['entropy_rank'],'flat_nuclear_entropy_rank':nuc['entropy_rank'],
            'fro_ratio_fro':fro['fro_ratio'],'fro_ratio_nuclear':nuc['fro_ratio'],
            'target_fro_minus_nuclear':fro['target']-nuc['target'],
            'off_fro_minus_nuclear':fro['off_score']-nuc['off_score'],
            'gap_fro_minus_nuclear':fro['forgetting_gap']-nuc['forgetting_gap']})
    table(OUT/'flat_budget_pairs.tsv',pairs)
    return source_corr,means,pairs

def analyze(rows):
    feature_names=list(CORE)+[f'{scope}__{f}' for scope in SCOPES for f in ('entropy_rank','top1_energy','log_fro','original_head1_retention')]
    result=correlations(rows,feature_names)
    write(OUT/'correlations.json',result); table(OUT/'correlations.tsv',result)
    edited=[dict(r,cohort=r['cohort']+'__edited_only') for r in rows if not r['baseline']]
    sensitivity=correlations(edited,list(CORE),seed=20260915)
    write(OUT/'edited_only_correlations.json',sensitivity); table(OUT/'edited_only_correlations.tsv',sensitivity)
    source_corr,means,pairs=grouped_diagnostics(rows)
    summaries,rankings=cross_validate(rows)
    report(rows,result,summaries,rankings,source_corr,means,pairs)
    audit_results(rows)
    manifest=read(OUT/'manifest.json'); manifest.update({'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':sha(__file__),'core_features':len(CORE),'secondary_features':48,'bootstrap_replicates':2000,'permutation_replicates':2000})
    write(OUT/'manifest.json',manifest); print('Complete',OUT,flush=True)

def plots(rows,result,summaries):
    import os
    import sys
    sys.path.insert(0,str(OUT/'plot_dependencies'))
    os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache')
    os.environ['XDG_CACHE_HOME']=str(OUT/'font_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=OUT/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    cohorts=['joint_flat_hns','hns_grid_retention','scalar_common_basis_seed42']
    fig,axes=plt.subplots(1,3,figsize=(12,9),layout='constrained')
    for ax,cohort in zip(axes,cohorts):
        values=np.array([[next(r['within_spearman'] for r in result if r['cohort']==cohort and r['feature']==f and r['outcome']==k) for k in ('target_gain','off_gain','forgetting_reduction')] for f in CORE],dtype=float)
        im=ax.imshow(values,vmin=-1,vmax=1,cmap='RdBu',aspect='auto')
        ax.set_xticks(range(3),['Target gain','Off-task gain','FG reduction'],rotation=30,ha='right')
        ax.set_yticks(range(len(CORE)),list(CORE) if cohort==cohorts[0] else [])
        ax.set_title(cohort.replace('_','\n'))
        for i in range(len(CORE)):
            for j in range(3):
                if np.isfinite(values[i,j]):
                    ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(values[i,j])>.65 else 'black')
    fig.colorbar(im,ax=axes,label='Within-checkpoint Spearman rho',shrink=.6)
    fig.suptitle('Baseline included; waves kept separate; source checkpoints are the statistical clusters')
    fig.savefig(figures/'within_checkpoint_correlations.png'); fig.savefig(figures/'within_checkpoint_correlations.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    selected=[r for r in rows if r['cohort']=='joint_flat_hns' and not r['baseline']]
    for method,color in zip(('flat_fro','flat_nuclear','hns_f4_s1'),('tab:orange','tab:blue','tab:green')):
        data=[r for r in selected if r['method']==method]
        for ax,outcome in zip(axes,('target_gain','forgetting_reduction')):
            ax.scatter([r['original_head1_retention'] for r in data],[r[outcome] for r in data],label=method,c=color,alpha=.8)
            ax.axhline(0,c='grey',lw=.7); ax.set_xlabel('Original leading-direction energy retention'); ax.set_ylabel(outcome+' (pp)')
    axes[0].legend(); fig.suptitle('54 interventions on 18 source checkpoints; association is not causality')
    fig.savefig(figures/'head_retention_scatter.png'); fig.savefig(figures/'head_retention_scatter.pdf'); plt.close(fig)
    data=[r for r in summaries if r['cohort']=='joint_flat_hns']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,metric in zip(axes,('skill_vs_zero','skill_vs_train_mean')):
        for i,model in enumerate(('single','shape_plus_strength')):
            d=[next(r for r in data if r['model']==model and r['split']==s) for s in ('seed','base','train_task')]
            ax.bar(np.arange(3)+(i-.5)*.36,[r[metric] for r in d],width=.36,label=model)
        ax.set_xticks(range(3),['Leave seed out','Leave base out','Leave task out']); ax.axhline(0,c='black',lw=.8)
        ax.set_ylabel(metric); ax.set_title('Positive = better held-out joint prediction')
    axes[0].legend(); fig.suptitle('Nested checkpoint-grouped CV, target + FG objective, latest strict cohort')
    fig.savefig(figures/'nested_cv_skill.png'); fig.savefig(figures/'nested_cv_skill.pdf'); plt.close(fig)

def fmt(x,d=3):
    return '—' if x is None else f'{x:.{d}f}'

def md_table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])

def report(rows,result,summaries,rankings,source_corr,means,pairs):
    plots(rows,result,summaries)
    lookup={(r['cohort'],r['feature'],r['outcome']):r for r in result}
    sensitivity=read(OUT/'edited_only_correlations.json')
    edited_lookup={(r['cohort'].removesuffix('__edited_only'),r['feature'],r['outcome']):r for r in sensitivity}
    def rho(cohort,f,k,edited=False):
        return fmt((edited_lookup if edited else lookup)[(cohort,f,k)]['within_spearman'])
    primary_cv=next(r for r in summaries if (r['cohort'],r['model'],r['split'])==('joint_flat_hns','single','seed'))
    lines=['# 三种子 LoRA 谱量与 downstream / forgetting 的回顾性分析（2026-09-14）','',
        '本报告重算实际保存 adapter 的 compact SVD，分析 18 个 source checkpoints、282 个 adapter、450 条 checkpoint × 方法记录。只读已有训练与评测产物；没有训练、生成或占用 GPU。', '',
        '选中的记录对应 1206 个 benchmark cells、6293220 条既有评测样例（包括不同评测批次对同一 adapter 的重复评测）。样例数量用于说明数据覆盖，独立训练分组仍只有 18 个 source checkpoints。','',
        '**目前没有找到经跨模型、跨任务验证、能同时预测 downstream 与 forgetting 的统一最优单谱量。** 原始主导方向能量保留量 H1 是值得保留的关联候选；entropy effective rank 在同模型/任务范围内的留种子预测中更常被选择。两种结论分别是关联与预测，不应混为一谈。','',
        f'最新严格对照中，H1 与 ΔTarget / FG reduction 的 within-checkpoint ρ 为 {rho("joint_flat_hns","original_head1_retention","target_gain")} / {rho("joint_flat_hns","original_head1_retention","forgetting_reduction")}。去掉未编辑 LoRA 后变为 {rho("joint_flat_hns","original_head1_retention","target_gain",True)} / {rho("joint_flat_hns","original_head1_retention","forgetting_reduction",True)}；HNS 网格 edited-only 的 FG ρ 为 {rho("hns_grid_retention","original_head1_retention","forgetting_reduction",True)}。强关联一部分来自编辑与未编辑的区别，尚不能可靠用于排序全部已编辑方法。','',
        f'最新批次嵌套留 seed 的单量选择，在三折中均选 entropy effective rank；聚合 Target R²={fmt(primary_cv["target_gain_r2"])}, FG R²={fmt(primary_cv["forgetting_reduction_r2"])}。跨 base 的泛化失败，FG 改善的个体差异仍预测不佳。双量模型在最新批次没有稳定解决这一问题。','',
        '分析方案先写入 `analysis_plan.json`。候选筛选属于回顾性探索，嵌套交叉验证中的测试 seed/base/task 不参与特征、次数或正则选择。源 checkpoint 是统计分组单位；module 数量不增加独立样本数。','',
        '## 数据与协议','',
        md_table(['批次','checkpoint 数','记录数','拟合干预数','用途'],[
            ['joint_flat_hns',18,72,54,'最新同批次 LoRA / Flat-Fro / Flat-Nuclear / 固定 HNS 4+1'],
            ['hns_grid_target',18,198,162,'LoRA + 0+0 + 九个 HNS 设置；只有 diagonal 性能'],
            ['hns_grid_retention',12,132,108,'seed43/44，完整四个 benchmark，九个 HNS 设置'],
            ['scalar_common_basis_seed42',6,48,42,'共同 U/V 表示，scalar 与 per-module norm-matched 控制']]),'',
        '各批次使用各自 manifest 中的原始成绩和 LoRA 对照，不合并不同 adapter block / batch 的原始得分。最新批次 HNS 统一为 4+1；网格中的全部设置仅用于研究谱量的变化，不代表为每个 checkpoint 选择最佳 HNS。DG-Hard 已证明是 identity，不作为新的独立观测；0+0 重建控制保留在完整数据中但排除拟合。旧 scalar 批次的 common_lora 是其同表示对照，原始表示重复项排除。','',
        'Target 是对应训练任务的现有主指标（HumanEval pass@1 / GSM8K strict accuracy / IFEval strict prompt-level accuracy）。Off-task 是另外三个 benchmark 家族成绩的等权平均；Commonsense 内部复用原 metric。', '',
        r'$FG=\frac13\sum_{b\ne task}\max(0,Score_{Base,b}-Score_{adapter,b})$。$\Delta T=T_{adapter}-T_{LoRA}$；$\Delta O=O_{adapter}-O_{LoRA}$；$R_{FG}=FG_{LoRA}-FG_{adapter}$。三个变化指标均以正数为改善，单位 pp。FG 在 0 处截断，因此应同时查看未截断的 off-task gain。','',
        '18 个 checkpoint 的实际 source 路径来自原三种子审计，逐一验证每个派生 adapter 的 source 元数据和 adapter_config 哈希。seed42 的历史训练 recipe 与 seeds43/44 不完全一致，两处 Llama seed42 的实际训练 seed 未完全验证；相关结果不能声称为完全同 recipe 的三种子结论。','',
        '## 谱量定义','',
        r'模块实际更新 $\Delta W=cBA$，$c=\alpha/r$，奇异值记作 $s_i=c\sigma_i$；这里全部 $r=16,c=2$。令 $p_i=s_i/\sum s_i$、$q_i=s_i^2/\sum s_i^2$。形状量取所有 LoRA modules 的等权平均；绝对幅度量先在 modules 间按定义聚合。','',
        md_table(['量','定义 / 聚合'],[[f,definition] for f,definition in CORE.items()]),'',
        '推荐先理解六类量：entropy effective rank（平坦度）、stable rank（相对最大奇异值的能量）、top-k energy（头部集中度）、整体 Frobenius 幅度、原始头部方向能量保留量、以及最佳 scalar 拟合后的形状残差。', '',
        r'若希望将头部保留量改写为正向指标，可定义 $A_{head}=1-H_1$（主导方向衰减量，源 LoRA 为 0；不截断）。在共享 U/V 的理想编辑中，$H_1=\sum_m c_m^2t_{m,1}^2/\sum_m c_m^2\sigma_{m,1}^2$，其中索引 1 指源最大的奇异方向。正向写法不会增加信息或改善预测，只改变相关符号。建议同时保留平坦度、Fro ratio 和 A_head，防止把谱形状与预算混在一个量中。','',
        '有效秩采用 Roy–Vetterli 的奇异值概率熵定义：[原始论文资料](https://infoscience.epfl.ch/bitstreams/9a5f3153-5c5d-4845-ab41-962296ec93ac/download)。Stable rank 定义参考 [primary research paper](https://arxiv.org/abs/1507.02268)。Energy entropy 和参与率、相对原始方向的保留量作为本实验定义的诊断量，不能都称作 Roy–Vetterli effective rank。','',
        r'Original-head retention 使用源 $U_{m,0}$ 的前 $k$ 个左奇异方向：$H_k=\frac{\sum_m\|U_{m,0}[:,1:k]^\top\Delta W_m\|_F^2}{\sum_m\sum_{i\le k}s_{m,0,i}^2}$。在本实验共享 U/V 的理想谱编辑中就是 $\sum t_{m,i\le k}^2/\sum s_{m,0,i\le k}^2$，但实际计算包含保存后的投影误差。它允许大于 1；谱形状本身的 top-k energy 总是按修改后谱重新排序。HNS 的 sigma_after 不保证有序，不能误把数组第一项当修改后最大奇异值。','',
        r'最佳 scalar 拟合 $\gamma=\langle\Delta W_{new},\Delta W_0\rangle/\|\Delta W_0\|_F^2$，残差 $R=1-\langle\Delta W_{new},\Delta W_0\rangle^2/(\|\Delta W_{new}\|_F^2\|\Delta W_0\|_F^2)$。全局 Frobenius 指所有不同参数矩阵平方范数之和开根号；head RSS 只是 module operator norm 的平方和，不称为整个网络的 operator norm。','',
        '数值审计修正已写入 analysis_plan.json：形状和相对无量纲指标按 1e-5 取整；norm ratio 距离 1 小于 1e-5 时恢复为严格不变量，log norm 同步使用 source 的 log norm。原始未量化奇异值保留在 npz 中。实际保存谱对目标谱的最大相对误差为 6.23e-6。此修正用于消除保持核预算时浮点误差导致的伪秩相关，不通过性能选择精度。核参与率与平坦谱距离/CV 存在代数等价关系；top1 energy 与 module stable rank 互为函数，这些候选量不视为独立机制。另有 attention / MLP、深度三段、七类 projection 共 12 个预定义 scope，每个只分析四个量，作为探索性附表。','',
        '## 同一 checkpoint 内的相关性','',
        '表中是 within-checkpoint Spearman：每个 source checkpoint 的干预序列内分别计算含 ties 的平均秩、去均值后聚合。95% CI 为 2000 次 source-cluster bootstrap；p 为 2000 次 checkpoint 内 outcome 置换，BH FDR 分批次、outcome、core/scope 家族校正。置换是关联诊断，不是随机干预实验的因果 p 值。', '',
        md_table(['谱量','最新 ΔTarget ρ','最新 FG reduction ρ','95% CI（FG）','FDR q（FG）','网格 ΔTarget ρ','网格 FG reduction ρ','scalar ΔTarget ρ','scalar FG reduction ρ'],[
            [f,fmt(lookup[('joint_flat_hns',f,'target_gain')]['within_spearman']),fmt(lookup[('joint_flat_hns',f,'forgetting_reduction')]['within_spearman']),
             '['+fmt(lookup[('joint_flat_hns',f,'forgetting_reduction')]['cluster_ci_low'])+', '+fmt(lookup[('joint_flat_hns',f,'forgetting_reduction')]['cluster_ci_high'])+']',
             fmt(lookup[('joint_flat_hns',f,'forgetting_reduction')].get('fdr_q')),
             fmt(lookup[('hns_grid_target',f,'target_gain')]['within_spearman']),fmt(lookup[('hns_grid_retention',f,'forgetting_reduction')]['within_spearman']),
             fmt(lookup[('scalar_common_basis_seed42',f,'target_gain')]['within_spearman']),fmt(lookup[('scalar_common_basis_seed42',f,'forgetting_reduction')]['within_spearman'])] for f in CORE]),'',
        '完整 Pearson、Spearman、raw / within 对比、partial Pearson（控制 log Frobenius 幅度）、每 checkpoint 的方向计数以及 scope 结果见 `correlations.tsv`。控制幅度后的相关用于检查形状是否提供额外信息；在固定核预算等约束下仍可能存在强共线性，不能作因果解释。分 seed/base/task 结果见 `stratified_correlations.tsv`。','',
        '### 只比较已编辑方法的敏感性检查','',
        '排除各 checkpoint 的 LoRA baseline，并重新在干预序列内排名。此表回答“哪个已编辑 adapter 更好”，主表同时包含编辑与未编辑的比较。全部 19 个量的 CI、FDR、partial 结果在 edited_only_correlations.tsv。','',
        md_table(['谱量','最新 ΔTarget ρ','最新 FG ρ','HNS target 网格 ΔTarget ρ','HNS retention 网格 FG ρ','scalar ΔTarget ρ','scalar FG ρ'],[
            [f,rho('joint_flat_hns',f,'target_gain',True),rho('joint_flat_hns',f,'forgetting_reduction',True),
                rho('hns_grid_target',f,'target_gain',True),rho('hns_grid_retention',f,'forgetting_reduction',True),
                rho('scalar_common_basis_seed42',f,'target_gain',True),rho('scalar_common_basis_seed42',f,'forgetting_reduction',True)]
            for f in ('entropy_rank','stable_rank','top1_energy','fro_ratio','original_head1_retention','shape_residual')]),'',
        '最新 edited-only 的关联在 core 家族 FDR 下没有足够证据；HNS target 网格中 H1 和 shape residual 对 target 仍有较弱关联，retention 网格中 FG 几乎没有进一步排序信号。Scalar 只有 6 个源 checkpoint，cluster bootstrap CI 较宽；即便置换 q 较小，也不能声称已验证跨 checkpoint 通用性。','',
        'FG 的截断限制了可辨别性：最新 54 个编辑条件有 31 个 FG=0，18 个 source 中有 8 个在三种编辑方法间 FG 完全相同；HNS retention 网格 108 条编辑条件有 72 个 FG=0，12 个 source 中有 7 个在九种 HNS 设置间 FG 完全相同。零相关不能证明谱量完全无用；应同时看 off-task gain 与各 benchmark 的分项指标。','',
        '![Within-checkpoint correlations](spectral_performance_forgetting_three_seed_20260914/figures/within_checkpoint_correlations.png)','',
        '## 能否找到统一的最佳谱量','',
        '单量候选为 19 个 core features，允许一次或二次曲线，ridge ∈ {0.01,1,10}。双量候选仅为预定义的四种形状量 × 三种幅度/保留量。输入是相对各批次 LoRA 的谱量变化，拟合不带截距；没有编辑时预测变化为 0。训练集内按 source checkpoint 做 leave-one-checkpoint-out，选择 feature / 次数 / ridge；外层分别完全留出一个 seed、base 或训练任务。', '',
        '目标是预测 ΔTarget 和 FG reduction，按训练集各 outcome 的 SD 标准化后等权最小化 MSE；只有 target 的网格使用单 outcome。Baseline 同时报告预测 0（没有改善）和训练集平均改善。Skill=1−MSE_model/MSE_baseline，正数表示优于该对照；R² 为所有外层预测聚合后的结果。不同折的最佳量可变化，因此全数据 inner winner 只能是范围内候选，不能直接宣称跨模型通用。','',
        md_table(['批次','模型','外层留出','Skill vs 0','Skill vs train mean','Target R²','FG R²','折内选出的量'],[
            [r['cohort'],r['model'],r['split'],fmt(r['skill_vs_zero']),fmt(r['skill_vs_train_mean']),fmt(r.get('target_gain_r2')),
                fmt(r.get('forgetting_reduction_r2')),json.dumps(r['feature_selection'],ensure_ascii=False)] for r in summaries]),'',
        md_table(['批次','候选族','全数据 inner winner（探索性）','次数','ridge','inner joint loss'],[
            [r['cohort'],r['model'],' + '.join(r['ranking'][0]['features']),r['ranking'][0]['degree'],r['ranking'][0]['ridge'],fmt(r['ranking'][0]['inner_cv_loss'])] for r in rankings]),'',
        '![Nested validation](spectral_performance_forgetting_three_seed_20260914/figures/nested_cv_skill.png)','',
        '![Original head retention](spectral_performance_forgetting_three_seed_20260914/figures/head_retention_scatter.png)','',
        '## Flat 预算检查：相同平坦度是否足够','',
        f'18 对 Flat-Fro / Flat-Nuclear 的 entropy_rank 都为 1.00000；同样完全平坦，但 target 有 {sum(abs(r["target_fro_minus_nuclear"])>1e-9 for r in pairs)}/18 对不同，off-task 有 {sum(abs(r["off_fro_minus_nuclear"])>1e-9 for r in pairs)}/18 对不同。仅靠平坦度无法区分这两类 adapter；差异与预算/更新幅度一起存在。平均 Flat-Fro − Flat-Nuclear：target {np.mean([r["target_fro_minus_nuclear"] for r in pairs]):+.3f} pp，off-task {np.mean([r["off_fro_minus_nuclear"] for r in pairs]):+.3f} pp，FG {np.mean([r["gap_fro_minus_nuclear"] for r in pairs]):+.3f} pp。逐种子配对见 `flat_budget_pairs.tsv`。','',
        '## 最新同批次完整逐种子表','',
        'erank 与 stable rank 已除以 16。H1 是源头部方向保留量，F-ratio 相对原始 LoRA。Target/Off/FG 单位 % / pp。', '',
        md_table(['Base','Task','Seed','Method','erank/16','stable/16','F-ratio','H1','Target','Off','FG'],[
            [r['base'],r['train_task'],r['seed'],r['method'],fmt(r['entropy_rank'],5),fmt(r['stable_rank'],5),fmt(r['fro_ratio']),
                fmt(r['original_head1_retention']),fmt(r['target']),fmt(r['off_score']),fmt(r['forgetting_gap'])] for r in rows if r['cohort']=='joint_flat_hns']), '',
        '## 3-seed mean ± sample SD','',
        md_table(['Base','Task','Method','Target','FG','erank/16','F-ratio','H1'],[
            [base,task,method,*[next(fmt(m['mean'])+' ± '+fmt(m['sample_sd']) for m in means if (m['base'],m['train_task'],m['method'],m['quantity'])==(base,task,method,q)) for q in ('target','forgetting_gap','entropy_rank','fro_ratio','original_head1_retention')]]
            for base,task,method in dict.fromkeys((r['base'],r['train_task'],r['method']) for r in rows if r['cohort']=='joint_flat_hns')]),'',
        '全部 19 个谱量、六个性能/遗忘量的 3-seed mean ± sample SD 与 seed42/43/44 原值在 `three_seed_mean_sd.tsv`。18 个未编辑源 checkpoint 本身的跨 checkpoint 相关与去除 Base×Task 六组均值后的相关在 `source_checkpoint_correlations.tsv`：每组仅三种子，无法独立验证“最合适”预测量。','',
        md_table(['未编辑 source 谱量','Target raw ρ','Target 去 Base×Task ρ','FG raw ρ','FG 去 Base×Task ρ'],[
            [f,*[next(fmt(z[key]) for z in source_corr if z['feature']==f and z['outcome']==outcome)
                for outcome,key in [('target','raw_spearman'),('target','within_base_task_spearman'),('forgetting_gap','raw_spearman'),('forgetting_gap','within_base_task_spearman')]]]
            for f in ('entropy_rank','stable_rank','top1_energy','log_fro','log_nuclear','log_head_rss')]),'',
        '这里 FG 是绝对遗忘量，正相关表示更多遗忘，和上文 FG reduction 的改善方向相反。此表单独研究 18 个未编辑源 checkpoint，不能用干预数量放大其样本量。不同 benchmark 的 raw target 分数不可直接作为跨任务统一质量标尺，raw 相关仅展示混杂现象。','',
        '## 重现、manifest 与产物','',
        '```bash','/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/analyze_spectral_performance_forgetting_three_seed.py --stage extract',
        '/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/analyze_spectral_performance_forgetting_three_seed.py --stage analyze','```','',
        '绘图依赖安装在本报告目录的 `plot_dependencies`，不修改训练环境；Matplotlib 3.11.2。CPU 使用 4 torch threads；cache 为实际 adapter 权重 SHA256 + source SHA256 + 路径联合寻址。共享 compact SVD，不显式构建密集 BA。原始头部投影也通过低秩因子计算。', '',
        '- `manifest.json`：所有输入 manifest 路径/哈希、评测配置、数据规模、状态。',
        '- `inventory.tsv/json`：450 条记录及所有 adapter / source / score / generation / variant manifest 真实路径。',
        '- `features.tsv/json`：450 条全谱量与相对 LoRA 变化、每个 benchmark 的原始性能。',
        '- `spectrum_audit.json`、`spectra/*.npz`：282 adapters 的权重/config 哈希、module 数、实际 SVD、源头部投影、目标谱误差。',
        '- `source_module_spectra.tsv`：4284 个源 modules 的描述性谱数据。',
        '- `correlations.tsv/json`、`stratified_correlations.tsv`、`source_checkpoint_correlations.tsv`：完整相关与稳健性。',
        '- `edited_only_correlations.tsv/json`：排除 baseline 后的完整关联检查。',
        '- `cv_rankings.json`、`cv_folds.json`、`cv_predictions.tsv`、`cv_summary.tsv/json`：所有候选、折内选择、逐 seed/checkpoint 干预外层预测。',
        '- `three_seed_mean_sd.tsv`、`flat_budget_pairs.tsv`：完整逐种子和三种子统计。',
        '- `figures/*.png/pdf`：可导出科学图。','']
    (ROOT/'reports/spectral_performance_forgetting_three_seed_20260914.md').write_text('\n'.join(lines))

def audit_results(rows):
    manifest=read(OUT/'manifest.json')
    assert len(rows)==450 and len(set(r['checkpoint'] for r in rows))==18
    assert len({(r['cohort'],r['checkpoint'],r['label']) for r in rows})==450
    assert len(set(r['path'] for r in rows))==282
    assert manifest['benchmark_cells']==1206
    assert manifest['evaluated_examples_in_selected_cells']==6293220
    unchanged=[]
    for item in manifest['input_manifests']:
        assert sha(item['path'])==item['sha256'],item['path']
        unchanged.append(item['path'])
    for c in read(SOURCE/'source_manifest.json')['checkpoints']:
        weight,_=find_adapter_weight_file(c['source'])
        assert sha(weight)==c['source_sha256']
    for r in rows:
        if r['method']=='flat_fro':
            assert r['fro_ratio']==1.
        if r['method']=='flat_nuclear' or r['cohort'].startswith('hns_grid'):
            assert r['nuclear_ratio']==1.,r['path']
        if r['baseline']:
            assert all(r['delta__'+f]==0 for f in CORE)
    predictions=list(csv.DictReader((OUT/'cv_predictions.tsv').open(),delimiter='\t'))
    truth_lookup={(r['cohort'],r['checkpoint'],r['method']):r for r in rows}
    for pred in predictions:
        actual=truth_lookup[(pred['cohort'],pred['checkpoint'],pred['method'])]
        assert not actual['baseline'] and not actual['control']
        for k in ('target_gain','forgetting_reduction'):
            if pred.get(k):
                assert abs(float(pred[k])-actual[k])<1e-12
    for fold in read(OUT/'cv_folds.json'):
        data=[r for r in rows if r['cohort']==fold['cohort'] and not r['baseline'] and not r['control']]
        field,value=fold['split'],fold['held_out']
        train=[r for r in data if r[field]!=value]; test=[r for r in data if r[field]==value]
        assert set(r['checkpoint'] for r in train).isdisjoint(r['checkpoint'] for r in test)
        assert len(set(r['checkpoint'] for r in train))==fold['train_clusters']
        assert len(set(r['checkpoint'] for r in test))==fold['test_clusters']
        outcomes=['target_gain','forgetting_reduction'] if 'forgetting_reduction' in data[0] else ['target_gain']
        sd=np.maximum(np.std([[r[k] for k in outcomes] for r in train],axis=0),1e-6)
        assert np.allclose(fold['train_outcome_sd'],sd,rtol=0,atol=1e-12)
    result={'status':'pass','coverage_rows':450,'source_checkpoints':18,'unique_adapters':282,
        'benchmark_cells':1206,'evaluated_examples':6293220,'unchanged_input_manifests':len(unchanged),
        'source_hash_audit':'pass','norm_invariants':'pass','baseline_deltas':'pass',
        'cv_truth_join':'pass','cv_source_disjointness':'pass','cv_train_only_normalization':'pass',
        'synthetic_checks':'tied ranks / constant correlation / identity / Flat-Fro / Flat-Nuclear passed',
        'gpu_count':0}
    write(OUT/'result_audit.json',result)
    write(OUT/'progress.json',{'stage':'complete','timestamp_utc':datetime.now(timezone.utc).isoformat(),'result_audit':'pass','gpu_count':0})
    manifest.update({'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':sha(__file__),'result_audit':'pass','core_features':19,'secondary_features':48,
        'bootstrap_replicates':2000,'permutation_replicates':2000,
        'report':str(ROOT/'reports/spectral_performance_forgetting_three_seed_20260914.md'),
        'output_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.tsv','.txt') and p.name not in ('manifest.json',)}})
    write(OUT/'manifest.json',manifest)
    print('Final result audit PASS',result,flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['inventory','extract','analyze','report','all'],default='all'); args=p.parse_args()
    if args.stage in ('inventory','extract','all'):
        sources,rows=inventory()
    if args.stage in ('extract','all'):
        rows=extract(sources,rows)
    if args.stage in ('analyze','all'):
        analyze(read(OUT/'features.json'))
    if args.stage=='report':
        rows=read(OUT/'features.json'); source_corr,means,pairs=grouped_diagnostics(rows)
        report(rows,read(OUT/'correlations.json'),read(OUT/'cv_summary.json'),read(OUT/'cv_rankings.json'),source_corr,means,pairs)
        audit_results(rows)

if __name__=='__main__':
    main()
