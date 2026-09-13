#!/usr/bin/env python3
"""Complete all functional diagnostics and rerun grouped three-seed statistics."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import analyze_spectral_performance_forgetting_three_seed as lib
import try_promising_spectral_metrics_three_seed as prior
from collect_functional_activation_three_seed import OUT, PRIOR, SOURCE, read, write, sha

SPECTRA=lib.OUT
FUNCTIONAL=dict(prior.FUNCTIONAL,**{
    'functional_cov_participation':'median trace(G)^2/trace(G@G), G=diag(scaled target spectrum) C diag(scaled target spectrum)',
    'functional_cov_erank':'median exp(entropy(normalized eigenvalues of G)); includes cross-direction second moments',
    'functional_cov_top1':'median largest eigenvalue(G)/trace(G)',
    'functional_offdiag_fraction':'median ||G-diag(G)||F^2/||G||F^2; cross-direction energy structure',
})
PARAM=dict(prior.PROPOSALS,**{k:v for k,v in prior.BASE_FEATURES.items() if k.endswith('k16')})
SOURCE_FEATURES=list(PARAM)+[f for f in prior.BASE_FEATURES if not f.endswith('k16')]+list(prior.INCOMPLETE)+list(FUNCTIONAL)

def configure():
    lib.OUT=OUT; prior.OUT=OUT
    if not (OUT/'analysis_plan.json').exists():
        write(OUT/'analysis_plan.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'18 sources / 282 adapters / 450 separate-wave conditions',
            'functional_definitions':FUNCTIONAL,'source_feature_pool':SOURCE_FEATURES,
            'statistics':'within-source ranks, source-cluster bootstrap2000, permutation2000; source quality exact46656 within-Base×Task seed permutations; BH FDR per outcome',
            'validation':'nested leave seed/base/task; inner leave source; linear/quadratic ridge .01/1/10, train-only normalization; known-group source means as quality baseline',
            'sampling_seed':42,'training_seeds':[42,43,44],'max_gpus_collection':2,
            'note':'exploratory follow-up, not preregistered'})

def features():
    write(OUT/'progress.json',{'status':'computing_features','updated_utc':datetime.now(timezone.utc).isoformat()})
    rows=read(PRIOR/'base_features.json'); cps=read(SOURCE)['checkpoints']
    cases=read(OUT/'collection_manifest.json')['cases']; available={}; inputs=[]; checks=[]
    for case in cases:
        audit=read(OUT/'activations'/case['base']/case['task']/'case_audit.json'); assert audit['status']=='complete'
        for c in case['checkpoints']:
            p=OUT/'activations'/c['base']/c['task']/f'seed{c["seed"]}.json'; a=read(p)
            assert a['status']=='complete' and a['source']==c['source'] and a['source_sha256']==c['source_sha256']
            assert sha(a['npz'])==a['npz_sha256'] and a['samples']==256 and a['sampling_seed']==42
            z=np.load(a['npz']); ns=z['names'].tolist(); s=z['sigma'].astype(float); scale=z['scales']; cov=z['coordinate_second_moment_sum']
            assert len(ns)==c['module_count'] and cov.shape==(len(ns),16,16)
            diag=np.diagonal(cov,axis1=-2,axis2=-1)
            assert np.isfinite(cov).all() and (diag>0).all()
            eig=np.linalg.eigvalsh(cov); assert np.min(eig/eig[:,-1,None])> -1e-6
            available[c['source']]=(ns,s,scale,cov,a)
            inputs.append({'path':str(p),'sha256':sha(p),'npz':a['npz'],'npz_sha256':a['npz_sha256']})
    assert len(available)==18
    collection=read(OUT/'collection_manifest.json'); collection['status']='complete'; collection['source_checkpoints_collected']=18
    collection['completed_utc']=datetime.now(timezone.utc).isoformat(); write(OUT/'collection_manifest.json',collection)
    old_hns={}
    with (lib.ROOT/'reports/hns_release_20260912/data/mechanism/module_spectra.csv').open() as f:
        for r in csv.DictReader(f):
            old_hns.setdefault((r['base_model'],r['task']),{})[r['module']]=np.array([float(r[f'hns_s{i}']) for i in range(1,17)])
    spectrum={r['path']:r for r in read(SPECTRA/'spectrum_audit.json')['adapters']}; adapted={}; modules=[]
    for path,source in dict.fromkeys((r['path'],r['source']) for r in rows):
        ns,orig,scales,cov,a=available[source]; p=Path(path)
        if path==source: target=orig.copy()
        elif (p/'common_basis_meta.json').exists():
            meta=read(p/'common_basis_meta.json'); build=read(p.parent/'manifest.json'); label=meta['variant']
            if label.startswith('common_global_'):
                gamma=float(label.removeprefix('common_global_').replace('p','.')); target=orig*gamma
            elif label=='common_per_module': target=orig*np.array([build['module_stats'][n]['per_module_gamma'] for n in ns])[:,None]
            elif label=='common_hns':
                assert a['training_seed']==42
                old=next(r for r in read(PRIOR/'functional_audit.json')['source_manifests'] if r['source']==source)
                assert Path(build['source_hns']).resolve()==Path(read(old['activation_manifest'])['hns_path']).resolve()
                target=np.array([old_hns[(a['base'],a['task'])][n] for n in ns])
            else: raise ValueError(label)
        elif (p/'posthoc_meta.json').exists():
            meta=read(p/'posthoc_meta.json'); target=np.array([[meta['module_stats'][n]['target_level']]*16 for n in ns])
        else:
            meta=read(p/'spectral_edit_meta.json'); target=np.array([meta['module_stats'][n]['sigma_after'] for n in ns])
        z=np.load(spectrum[path]['cache']); assert z['names'].tolist()==ns
        saved=z['s']/scales[:,None]
        error=float(np.max(np.linalg.norm(saved-np.sort(target,axis=1)[:,::-1],axis=1)/np.linalg.norm(saved,axis=1)))
        assert error<.005,(path,error)
        qsum=np.diagonal(cov,axis1=-2,axis2=-1); base_energy=qsum*(orig*scales[:,None])**2
        energy=qsum*(target*scales[:,None])**2; pi=energy/energy.sum(1)[:,None]
        beta=target/orig; gamma=float((base_energy*beta).sum()/base_energy.sum()); ratio=float(energy.sum()/base_energy.sum())
        er=np.exp(-(pi*np.log(pi)).sum(1)); masks=lib.scope_masks(ns)
        gain=target*scales[:,None]; g=cov*gain[:,:,None]*gain[:,None,:]
        eigen=np.linalg.eigvalsh(g).clip(min=0); eigpi=eigen/eigen.sum(1)[:,None]
        cov_er=np.exp(-(eigpi*np.log(eigpi.clip(min=1e-300))).sum(1)); cov_pr=1/(eigpi*eigpi).sum(1)
        offdiag=1-np.sum(energy*energy,1)/np.sum(g*g,(1,2))
        assert np.all(cov_pr<=1/(pi*pi).sum(1)+1e-5)
        f={'functional_participation':float(np.median(1/(pi*pi).sum(1))),'functional_erank':float(np.median(er)),
            'functional_top1':float(np.median(pi.max(1))),'functional_erank_iqr':float(np.quantile(er,.75)-np.quantile(er,.25)),
            'functional_attn_mlp_gap':float(np.median(er[masks['attention']])-np.median(er[masks['mlp']])),
            'functional_energy_ratio':float(np.sqrt(ratio)),'gamma_functional':gamma,'eta_functional':float(max(0,1-gamma*gamma/ratio)),
            'functional_cov_participation':float(np.median(cov_pr)),'functional_cov_erank':float(np.median(cov_er)),
            'functional_cov_top1':float(np.median(eigpi[:,-1])),'functional_offdiag_fraction':float(np.median(offdiag))}
        adapted[path]={k:round(v,5) for k,v in f.items()}
        checks.append({'path':path,'source':source,'saved_target_spectrum_relative_error':error})
        for j,n in enumerate(ns):
            modules.append({'path':path,'source':source,'module':n,'functional_participation':float(1/(pi[j]*pi[j]).sum()),
                'functional_erank':float(er[j]),'functional_top1':float(pi[j].max()),'functional_cov_participation':float(cov_pr[j]),
                'functional_cov_erank':float(cov_er[j]),'functional_cov_top1':float(eigpi[j,-1]),'functional_offdiag_fraction':float(offdiag[j])})
    complete=[dict(r,**adapted[r['path']]) for r in rows]
    for co,cp in dict.fromkeys((r['cohort'],r['checkpoint']) for r in complete):
        group=[r for r in complete if (r['cohort'],r['checkpoint'])==(co,cp)]; baseline,=[r for r in group if r['baseline']]
        for r in group:
            for f in FUNCTIONAL: r['delta__'+f]=r[f]-baseline[f]
    lib.write(OUT/'features.json',complete); lib.table(OUT/'features.tsv',complete); lib.table(OUT/'functional_module_metrics.tsv',modules)
    # Compare the recomputed seed42 quantities with the previous cache-based calculation.
    old={(r['cohort'],r['checkpoint'],r['label']):r for r in read(PRIOR/'functional_features.json')}; comparisons=[]
    for r in complete:
        key=(r['cohort'],r['checkpoint'],r['label'])
        if key in old:
            for f in prior.FUNCTIONAL:
                comparisons.append({'checkpoint':r['checkpoint'],'cohort':r['cohort'],'label':r['label'],'feature':f,
                    'previous':old[key][f],'recollected':r[f],'difference':r[f]-old[key][f]})
    lib.table(OUT/'seed42_cache_comparison.tsv',comparisons)
    lib.write(OUT/'feature_audit.json',{'status':'pass','source_checkpoints':18,'unique_adapters':282,'rows':450,
        'seed_counts':{str(s):sum(r['seed']==s for r in complete) for s in (42,43,44)},'features':FUNCTIONAL,
        'collection_inputs':inputs,'saved_spectrum_checks':checks,'gpu_count_analysis':0,
        'seed42_cache_max_absolute_difference':{f:max(abs(r['difference']) for r in comparisons if r['feature']==f) for f in prior.FUNCTIONAL},
        'approximation':'fixed frozen-base trajectory; intended source-direction spectrum validated against saved factors; bf16 projected coordinates',
        'covariance':'uncentered second moments; full G eigenvalues supplement diagonal direction energy, not centered statistical covariance'})
    print('[Features complete] 18 sources / 282 adapters / 450 rows; diagonal + full second moments',flush=True)
    return complete

def candidates(two=False):
    shapes=('functional_participation','functional_erank','functional_cov_participation','functional_cov_erank',
        'functional_attn_mlp_gap','functional_erank_iqr','erank_iqr','erank_depth_slope')
    if not two: return [((f,),d,l) for f in lib.CORE for d in (1,2) for l in (.01,1.,10.)]
    return [((f,strength),d,l) for f in shapes for strength in ('functional_energy_ratio','fro_ratio') for d in (1,2) for l in (.01,1.,10.)]

def analyze(rows):
    write(OUT/'progress.json',{'status':'grouped_statistics_and_nested_cv','updated_utc':datetime.now(timezone.utc).isoformat()})
    configure(); lib.CORE=dict(FUNCTIONAL)
    fc=lib.correlations(rows,list(FUNCTIONAL),seed=20260919)
    fe=lib.correlations([dict(r,cohort=r['cohort']+'__edited_only') for r in rows if not r['baseline']],list(FUNCTIONAL),seed=20260920)
    lib.write(OUT/'correlations.json',fc); lib.table(OUT/'correlations.tsv',fc)
    lib.write(OUT/'edited_only_correlations.json',fe); lib.table(OUT/'edited_only_correlations.tsv',fe)
    seedwise=lib.correlations([dict(r,cohort=r['cohort']+'__seed'+str(r['seed'])) for r in rows],list(FUNCTIONAL),seed=20260921)
    lib.write(OUT/'per_seed_correlations.json',seedwise); lib.table(OUT/'per_seed_correlations.tsv',seedwise)
    sources,sc=prior.source_correlations(rows,SOURCE_FEATURES)
    prior.source_cv(sources,list(PARAM)+list(FUNCTIONAL))
    core={k:FUNCTIONAL[k] for k in ('functional_participation','functional_erank','functional_cov_participation','functional_cov_erank',
        'functional_energy_ratio','functional_attn_mlp_gap','functional_erank_iqr')}
    core.update({k:PARAM[k] for k in ('erank_iqr','erank_depth_slope','fro_ratio','original_head1_retention')})
    lib.CORE=core; lib.candidates=candidates; lib.cross_validate(rows)
    report(rows)

def report(rows):
    configure(); fc=read(OUT/'correlations.json'); fe=read(OUT/'edited_only_correlations.json'); sc=read(OUT/'source_correlations.json')
    cv=read(OUT/'cv_summary.json'); folds=read(OUT/'cv_folds.json'); src_cv=read(OUT/'source_cv_summary.json')
    main=[r for r in rows if r['cohort']=='joint_flat_hns']; groups=[]
    for base,task,method in dict.fromkeys((r['base'],r['train_task'],r['method']) for r in main):
        a=[r for r in main if (r['base'],r['train_task'],r['method'])==(base,task,method)]; assert sorted(r['seed'] for r in a)==[42,43,44]
        g={'base':base,'task':task,'method':method,'n':3}
        for f in (*FUNCTIONAL,'target','off_score','forgetting_gap','target_gain','forgetting_reduction'):
            values=[r[f] for r in sorted(a,key=lambda r:r['seed'])]
            g.update({f+'_seed42':values[0],f+'_seed43':values[1],f+'_seed44':values[2],f+'_mean':float(np.mean(values)),f+'_sample_sd':float(np.std(values,ddof=1))})
        groups.append(g)
    lib.table(OUT/'three_seed_mean_sd.tsv',groups); lib.table(OUT/'diagonal_per_seed.tsv',main)
    figures(fc,fe)
    pr_target=next(r for r in fc if r['cohort']=='joint_flat_hns' and r['feature']=='functional_participation' and r['outcome']=='target_gain')
    pr_fg=next(r for r in fc if r['cohort']=='joint_flat_hns' and r['feature']=='functional_participation' and r['outcome']=='forgetting_reduction')
    edited_target=next(r for r in fe if r['cohort']=='joint_flat_hns__edited_only' and r['feature']=='functional_participation' and r['outcome']=='target_gain')
    edited_fg=next(r for r in fe if r['cohort']=='joint_flat_hns__edited_only' and r['feature']=='functional_participation' and r['outcome']=='forgetting_reduction')
    fmt=lib.fmt; table=lib.md_table
    lines=['# 完整三种子 activation 与功能谱重算（2026-09-14）','',
        '已采集两Base×三Task×三个训练seed的全部18个源checkpoint；补齐原缺失的12组seed43/44，并重采6组seed42用于旧缓存核验及补存16×16方向二阶矩。两张B300分别处理一个Base，每worker只见一张GPU，array最多两个worker。没有重新训练，没有修改adapter，没有重跑或替换原始性能/遗忘成绩。','',
        '每个Base×Task使用原activation manifest的同一256条训练分布样本、相同顺序与chat rendering、最长512 tokens。sampling seed始终42，它与LoRA训练seed42/43/44是不同概念；冻结Base轨迹相同，因此一次前向可以同时投影到三个source的V方向。','',
        '参数/结构谱量沿用上一轮审计产物；本轮12个功能量在全部282个实际adapter、450条分波次方法记录上完整计算。独立训练单位仍只有18，重复波次和大量module/token不能当作独立训练重复。DG-Hard已被此前逐token核验证实为identity，故沿用LoRA功能谱，不作为新增独立方法拟合。','',
        '**重算结论：补齐seed43/44后，功能PR/entropy的编辑前后关联仍在；但还没有找到能跨模型、跨任务稳定同时预测性能和遗忘的通用谱量。**','',
        '最新matched wave包含LoRA与三种编辑时，functional PR与ΔTarget / FG reduction的within-source ρ='+lib.fmt(pr_target['within_spearman'])+' / '+lib.fmt(pr_fg['within_spearman'])+'，FDR q='+lib.fmt(pr_target['fdr_q'],6)+' / '+lib.fmt(pr_fg['fdr_q'],6)+'。只比较已编辑adapter后，ρ降为'+lib.fmt(edited_target['within_spearman'])+' / '+lib.fmt(edited_fg['within_spearman'])+'，q='+lib.fmt(edited_target['fdr_q'])+' / '+lib.fmt(edited_fg['fdr_q'])+'，尚不能稳定排序Flat-Fro、Flat-Nuclear、HNS。','',
        'full-moment PR/entropy包含方向间相关性，但没有明显改善最新wave的已编辑方法排序。未编辑source的full-moment PR与绝对FG在Base×Task组内ρ=0.730（更大表示更多遗忘），全候选FDR q=0.184，仍是探索性风险信号；这个问题与编辑前后改善的比较不同。','',
        '源checkpoint质量的嵌套留seed谱量选择：Target RMSE='+lib.fmt(src_cv['target_rmse'])+' vs training-group mean '+lib.fmt(src_cv['target_group_mean_rmse'])+' pp；FG RMSE='+lib.fmt(src_cv['forgetting_gap_rmse'])+' vs '+lib.fmt(src_cv['forgetting_gap_group_mean_rmse'])+' pp。尚未超过组均值对照，干预的跨Base/Task留出也未支持通用双outcome预测。','',
        '## 本轮定义与数值检查','',
        table(['量','定义'],[[k,v] for k,v in FUNCTIONAL.items()]),'',
        r'$C=\sum_{\mathrm{nonpad\ tokens}}(V^\top x)(V^\top x)^\top$ 为未中心化二阶矩。旧diagonal定义使用 $\pi_i=t_i^2 C_{ii}/\sum_jt_j^2C_{jj}$；新增full-moment版本用 $G=\operatorname{diag}(ct)C\operatorname{diag}(ct)$ 的归一化特征值计算PR/entropy。它能包含方向间相关性，但仍是固定Base轨迹诊断，不能代表编辑后所有hidden states。','',
        '模型与V方向投影bf16，坐标乘积fp32、跨batch累加fp64；关闭TF32。保存每条样本的16方向能量、全二阶矩、Base输出能量、token数量、来源权重和basis哈希。完整moment diagonal与逐sample sum检查、PSD检查、保存adapter实际谱与目标谱检查通过。Flat谱的Hill/kurtosis等数学未定义项继续保持None，不能为了“补全”伪造数值。','',
        '## 全部18源checkpoint：质量、遗忘与HNS收益','',
        'Within Base×Task为六个组内分别对三个seed排名、中心化后合并相关，精确枚举46656种组内seed排列。FDR按outcome对参数与功能候选一起校正。绝对FG越小越好；FG reduction越大越好。','',
        table(['Feature','Target raw ρ','Target within ρ','Target q','FG raw ρ','FG within ρ','FG q','HNS ΔTarget within ρ','HNS FG reduction within ρ'],[
            [f,*[fmt(next(r for r in sc if r['feature']==f and r['outcome']==o).get(k)) for o,k in
                [('target','raw_spearman'),('target','within_base_task_spearman'),('target','fdr_q'),('forgetting_gap','raw_spearman'),
                 ('forgetting_gap','within_base_task_spearman'),('forgetting_gap','fdr_q'),('hns_target_gain','within_base_task_spearman'),('hns_forgetting_reduction','within_base_task_spearman')]]]
            for f in ('erank_iqr','erank_depth_slope','erank_attn_mlp_gap','head_tail',*FUNCTIONAL)]),'',
        '## 同checkpoint内的干预比较：包含LoRA / edited-only','',
        table(['Wave','Feature','ΔTarget ρ / q','FG reduction ρ / q','edited ΔTarget ρ / q','edited FG reduction ρ / q'],[
            [co,f,*[(fmt(next((r for r in rec if r['cohort']==co+suff and r['feature']==f and r['outcome']==out),{}).get('within_spearman'))+' / '+fmt(next((r for r in rec if r['cohort']==co+suff and r['feature']==f and r['outcome']==out),{}).get('fdr_q')))
                for rec,suff,out in [(fc,'','target_gain'),(fc,'','forgetting_reduction'),(fe,'__edited_only','target_gain'),(fe,'__edited_only','forgetting_reduction')]]]
            for co in dict.fromkeys(r['cohort'] for r in rows) for f in FUNCTIONAL]),'',
        '## 原六source与完整三种子的覆盖对照','',
        '该对照区分seed42重采引起的数值差异与增加seed43/44后统计变化。各seed的完整相关在per_seed_correlations.tsv/json。','',
        table(['Feature','Outcome','旧seed42 ρ','重采seed42 ρ','完整18-source ρ'],[
            [f,out,fmt(next((r for r in read(PRIOR/'functional_correlations.json') if r['cohort']=='joint_flat_hns' and r['feature']==f and r['outcome']==out),{}).get('within_spearman')),
                fmt(next(r for r in read(OUT/'per_seed_correlations.json') if r['cohort']=='joint_flat_hns__seed42' and r['feature']==f and r['outcome']==out)['within_spearman']),
                fmt(next(r for r in fc if r['cohort']=='joint_flat_hns' and r['feature']==f and r['outcome']==out)['within_spearman'])]
            for f in ('functional_participation','functional_erank','functional_cov_participation','functional_cov_erank') for out in ('target_gain','forgetting_reduction')]),'',
        '## 嵌套留出验证','',
        '同checkpoint干预CV排除baseline与0+0 identity control；内层leave-source选择单量/双量、linear/quadratic、ridge；外层leave seed/base/task。各wave独立，不把不同协议波次混为同一性能表。完整系数、训练尺度、选择、每checkpoint预测见cv_*。','',
        table(['Wave','Model','Leave','Target R²','FG reduction R²','Skill vs train mean'],[[r['cohort'],r['model'],r['split'],fmt(r.get('target_gain_r2')),fmt(r.get('forgetting_reduction_r2')),fmt(r.get('skill_vs_train_mean'))] for r in cv]),'',
        '未编辑source的质量预测使用training-only Base×Task组均值加一个谱量，嵌套leave-seed；参数与功能量同池选择。','',
        '```json\n'+json.dumps(src_cv,indent=2)+'\n```','',
        '## 采集吞吐与旧seed42缓存核验','',
        table(['Base','Task','Samples','Tokens','Batch','Seconds','Pilot batch/status/peak GiB'],[[c['base'],c['task'],a['samples'],a['tokens'],a['batch_size'],fmt(a['seconds'],1),'; '.join(str(p['batch_size'])+'/'+p['status']+'/'+fmt(p.get('peak_allocated_gib'),1) for p in a['pilots'])]
            for c in read(OUT/'collection_manifest.json')['cases'] for a in [read(OUT/'activations'/c['base']/c['task']/'case_audit.json')]]),'',
        '旧缓存与本轮重采的全部138条seed42方法记录、八量逐值对照在seed42_cache_comparison.tsv；相同协议下bf16 kernel、batch padding和累计顺序会带来数值差异，不以旧数值替换新采集。','',
        table(['Feature','max |new−old|'],[[f,fmt(v,6)] for f,v in read(OUT/'feature_audit.json')['seed42_cache_max_absolute_difference'].items()]),'',
        '## 完整逐种子 diagonal 表','',
        table(['Base','Task','Seed','Method','diag PR','diag erank','cov PR','cov erank','func gap','func energy ratio','Target','Off','FG'],[
            [r['base'],r['train_task'],r['seed'],r['method'],*[fmt(r[f]) for f in ('functional_participation','functional_erank','functional_cov_participation','functional_cov_erank','functional_attn_mlp_gap','functional_energy_ratio','target','off_score','forgetting_gap')]] for r in main]),'',
        '## 三种子 mean ± sample SD','',
        table(['Base','Task','Method','diag PR','cov PR','cov erank','Target','FG'],[[g['base'],g['task'],g['method'],*[fmt(g[f+'_mean'])+' ± '+fmt(g[f+'_sample_sd']) for f in ('functional_participation','functional_cov_participation','functional_cov_erank','target','forgetting_gap')]] for g in groups]),'',
        '全部12量与性能/遗忘的seed42/43/44和sample SD在three_seed_mean_sd.tsv。历史seed42训练recipe与43/44不同，两处Llama历史seed42训练标签未完全验证；新增activation不消除该训练来源限制。','',
        '## 命令与产物','',
        '```bash\n/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/collect_functional_activation_three_seed.py --prepare\nsbatch slurm/functional_activation_three_seed.slurm\n/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/analyze_functional_activation_three_seed.py --stage all\n```','',
        'collection_manifest.json保存原manifest解析的全部source、dataset、sample indices与哈希；activations/<Base>/<Task>/seed{42,43,44}.npz/json为18份完整moment缓存；case_audit.json保存短测试、batch与耗时。features.tsv/json保存450条完整功能量及原始成绩、路径；functional_module_metrics.tsv为67116个module条件；correlations/edited_only/source_correlations与cv_*保存全部统计；diagonal_per_seed.tsv与three_seed_mean_sd.tsv保存完整逐seed及统计表。manifest.json记录输入/输出SHA256。','']
    p=lib.ROOT/'reports/functional_activation_three_seed_20260914.md'; p.write_text('\n'.join(lines))
    audit(rows,p)
    print('[Complete]',p,flush=True)

def figures(fc,fe):
    import os,sys
    sys.path.insert(0,str(SPECTRA/'plot_dependencies'))
    os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache'); os.environ['XDG_CACHE_HOME']=str(OUT/'font_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=('functional_participation','functional_erank','functional_cov_participation','functional_cov_erank',
        'functional_top1','functional_attn_mlp_gap','functional_energy_ratio')
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    y=np.arange(len(selected))
    for ax,out in zip(axes,('target_gain','forgetting_reduction')):
        for records,suffix,shift,label in ((fc,'',-.17,'LoRA + edits'),(fe,'__edited_only',.17,'Edits only')):
            values=[next(r for r in records if r['cohort']=='joint_flat_hns'+suffix and r['feature']==f and r['outcome']==out)['within_spearman'] for f in selected]
            ax.barh(y+shift,[0 if v is None else v for v in values],height=.32,label=label)
        ax.set(yticks=y,yticklabels=selected if out=='target_gain' else [],xlim=(-1,1),xlabel='Within-source Spearman rho',title=out)
        ax.axvline(0,c='grey',lw=.8); ax.invert_yaxis()
    axes[0].legend(); fig.suptitle('All 18 sources: matched task/forgetting evaluation; functional activation moments')
    d=OUT/'figures'; d.mkdir(exist_ok=True)
    for suffix in ('png','pdf'): fig.savefig(d/f'functional_intervention_sensitivity.{suffix}',dpi=180)
    plt.close(fig)

def audit(rows,report_path):
    assert len(rows)==450 and len(set(r['source'] for r in rows))==18 and len(set(r['path'] for r in rows))==282
    previous={(r['cohort'],r['checkpoint'],r['label']):r for r in read(PRIOR/'base_features.json')}
    for r in rows:
        assert all(np.isfinite(r[f]) for f in FUNCTIONAL)
        old=previous[(r['cohort'],r['checkpoint'],r['label'])]
        assert all(r[k]==old[k] for k in ('target','target_gain','off_score','off_gain','forgetting_gap','forgetting_reduction') if k in old)
        assert 1<=r['functional_cov_participation']<=r['functional_participation']+1e-5<=16.00002
        assert 1<=r['functional_cov_erank']<=16.00001
    for fold in read(OUT/'source_cv_folds.json'):
        assert set(fold['train_checkpoints']).isdisjoint(fold['test_checkpoints'])
    for fold in read(OUT/'cv_folds.json'):
        data=[r for r in rows if r['cohort']==fold['cohort'] and not r['baseline'] and not r['control']]
        field=fold['split']; held=fold['held_out']
        train=[r for r in data if r[field]!=held]; test=[r for r in data if r[field]==held]
        assert set(r['checkpoint'] for r in train).isdisjoint(r['checkpoint'] for r in test)
        assert len(set(r['checkpoint'] for r in train))==fold['train_clusters']
        assert len(set(r['checkpoint'] for r in test))==fold['test_clusters']
        outcomes=['target_gain','forgetting_reduction'] if 'forgetting_reduction' in train[0] else ['target_gain']
        expected=np.maximum(np.std([[r[k] for k in outcomes] for r in train],axis=0),1e-6)
        assert np.allclose(fold['train_outcome_sd'],expected,rtol=0,atol=1e-12)
    write(OUT/'result_audit.json',{'status':'pass','rows':450,'unique_adapters':282,'source_checkpoints':18,
        'all_functional_features_finite':'pass','original_performance_unchanged':'pass','functional_moment_bounds':'pass',
        'source_cv_disjointness':'pass','intervention_cv_disjointness_and_train_scales':'pass','gpu_limit':2})
    write(OUT/'progress.json',{'status':'complete','updated_utc':datetime.now(timezone.utc).isoformat()})
    inputs=[PRIOR/'manifest.json',PRIOR/'base_features.json',SPECTRA/'spectrum_audit.json',SOURCE,OUT/'collection_manifest.json',OUT/'analysis_plan.json']
    write(OUT/'manifest.json',{'status':'complete','result_audit':'pass','source_checkpoints':18,'unique_adapters':282,'rows':450,
        'functional_features':FUNCTIONAL,'source_feature_pool':SOURCE_FEATURES,'report':str(report_path),'report_sha256':sha(report_path),
        'input_hashes':[{'path':str(p),'sha256':sha(p)} for p in inputs],
        'script_hashes':{str(p):sha(p) for p in [Path(__file__),lib.ROOT/'scripts/collect_functional_activation_three_seed.py',Path(lib.__file__),Path(prior.__file__)]},
        'output_hashes':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.tsv','.npz','.png','.pdf') and p.name!='manifest.json' and 'mpl_cache' not in p.parts and 'font_cache' not in p.parts},
        'max_gpus_collection':2,'gpu_count_analysis':0,'completed_utc':datetime.now(timezone.utc).isoformat()})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['features','analyze','report','all'],default='all'); args=parser.parse_args()
    configure()
    if args.stage in ('features','all'): features()
    if args.stage in ('analyze','all'): analyze(read(OUT/'features.json'))
    if args.stage=='report': report(read(OUT/'features.json'))
