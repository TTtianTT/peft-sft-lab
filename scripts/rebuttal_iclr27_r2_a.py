#!/usr/bin/env python3
"""Sequential, predeclared tier-A second-round rebuttal experiments."""
from __future__ import annotations
import argparse
import itertools
import json
import os
from pathlib import Path
import numpy as np
import rebuttal_iclr27 as r
import rebuttal_iclr27_e4 as e4
import rebuttal_iclr27_r2 as q
from rebuttal_iclr27_e2_e3 import average_ranks, correlation

EXPERIMENTS=('R2-O1','R2-O2','R2-O3','R2-O4')


def cohort():
    rows=[]
    for old in r.source_rows():
        cp=q.source_record(old['source'],base=old['base'],task=old['task'],seed=int(old['seed']),
            condition=f'{old["base"]}_{old["task"]}_seed{old["seed"]}',
            base_model=r.read(r.REF/f'{old["base"]}_variant_manifest.json')['base_model'])
        assert cp['source_sha256']==old['source_sha256'] and cp['config_sha256']==old['config_sha256']
        q.attach_cache(cp,'original_lora','main');q.attach_cache(cp,'hns_f4_s1','main');q.attach_cache(cp,'recon_0','E1')
        rows.append(cp)
    return rows


def prepare_o1():
    cps=[]
    for task in ('magicoder','metamath'):
        selection=r.read(e4.OUT/f'{task}_selection.json')
        for row in selection['candidates']:
            cp=q.source_record(row['path'],base=e4.BASE,base_model=e4.MODEL,task=task,seed=42,
                condition=f'{task}_r16_lr{row["multiplier"]:g}_fraction{row["fraction"]:g}',
                multiplier=row['multiplier'],lr=row['lr'],fraction=row['fraction'],validation_score=row['validation_score'])
            assert cp['source_sha256']==row['weights_sha256'] and cp['config_sha256']==row['config_sha256']
            if row['path']==selection['selected']['path']:
                cp['old_condition']=f'{task}_val_best_r16'
                for method in q.STD:q.attach_cache(cp,method,'E4')
            cps.append(cp)
    old=r.read(e4.OUT/'rank64_training_audit.json')
    for fraction,key in ((.5,'half'),(1.,'final')):
        cp=q.source_record(old[key],base=e4.BASE,base_model=e4.MODEL,task='metamath',seed=42,
            condition=f'metamath_r64_lr2_fraction{fraction:g}',multiplier=2.,lr=2e-4,fraction=fraction,validation_score=None)
        if fraction==1:
            assert cp['source_sha256']==old['weights_sha256']
            cp['old_condition']='metamath_r64'
            for method in q.STD:q.attach_cache(cp,method,'E4')
        cps.append(cp)
    return q.freeze('R2-O1',cps,q.STD,
        'All12 E4a rank16 candidates and both rank64 checkpoints are reported. Reuse2 val-best and rank64-final cells; '
        '11 new conditions x4 methods x4 families=176 new cells. No retraining or model selection. '
        'For r64 half and final, run the same E4 held-out500 MetaMath validation questions only as descriptive covariates. '
        'Plot HNS minus Recon-0 Target gain and Recon-0 minus HNS FG, sorted LR and training fraction, with validation scores. '
        'Spearman correlations pooled and by task/rank are descriptive, with tied ranks averaged; never infer independent training runs.',176,'item')


def prepare(exp):
    if (q.folder(exp)/'protocol.lock.json').exists():return q.check(exp)
    if exp=='R2-O1':return prepare_o1()
    cps=cohort()
    if exp=='R2-O2':
        methods=('original_lora','recon_0','hns_f4_s1','recon_0_unbal','hns_unbal')
        detail='Unbalanced factors B=U diag(sigma), A=Vh versus stock balanced B=U sqrt(sigma), A=sqrt(sigma)Vh; '
        detail+='same fp32 compact SVD, float64 HNS map, spectrum fp32 and native-factor cast. '
        detail+='Compare HNS-unbal minus Recon-unbal, balanced HNS minus Recon, their paired difference-in-differences, and both direct factorization differences. '
        detail+='Report exact full generated-token-sequence equality for each question and family, including mismatch counts; no tolerance or text-normalization substitution.'
        return q.freeze(exp,cps,methods,detail,144)
    if exp=='R2-O3':
        for cp in cps:
            for m in ('hns_od','ss_smooth_abs','ss_smooth_abs_all'):q.attach_cache(cp,m,'E2')
            q.attach_cache(cp,'spectral_surgery_grad_direction','spectral_surgery')
        methods=('original_lora','recon_0','hns_f4_s1','hns_od','ss_smooth_abs','ss_smooth_abs_all',
                 'spectral_surgery_grad_direction','ss_grad_direction_all')
        detail='Spectral Surgery grad_direction all7 q/k/v/o/gate/up/down_proj; original implementation and E2 defaults unchanged. '
        detail+='calib128, batch2, seed42, shuffled train split, start0; grad_norm mean_abs, l1 preservation, multiplicative, asymmetric, eta_suppress2, eta_enhance0.2, sigma_clip_min0. '
        detail+='Same task calibration files/formatters as E2, zero test calibration/tuning; full2 strategies x2 module scopes table. '
        detail+='Original source https://github.com/TTtianTT/peft-sft-lab at139bfd7; source audit/defaults in ../literature_audit.json and ../E2/protocol.md; paper https://arxiv.org/abs/2603.03995. '
        detail+='Report128 examples,64 forward/backward batches and measured editor time; HNS requires zero data/forward/backward.'
        return q.freeze(exp,cps,methods,detail,72)
    if exp=='R2-O4':
        methods=('original_lora','recon_0','hns_f4_s1','hns_f4_s0','hns_f0_s5','hns_f4_s1_l05')
        detail='Prespecified HNS fast4/stable0 strength1; fast0/stable5 strength1; fast4/stable1 strength0.5. '
        detail+='All7 modules; nuclear restoration; stock coefficients FAST(3.4445,-4.775,2.0315), STABLE(2,-1.5,0.5); eps1e-7. '
        detail+='Strength interpolation follows stock function ordering (HNS then nuclear restore then interpolate). '
        detail+='Report every variant, normalized participation rank (sum sigma^2)^2/(r sum sigma^4), top1 energy max(sigma^2)/sum(sigma^2), Frobenius ratio to original, per module/checkpoint and aggregate. '
        detail+='Compare all to Recon and main4+1; relate effects descriptively to achieved flatness; no stage or dose selected on tests.'
        return q.freeze(exp,cps,methods,detail,216)
    raise ValueError(exp)


def validate_r64():
    r.require_slurm();q.env_setup('R2-O1','validation');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    lock=q.check('R2-O1');old=e4.OUT;e4.OUT=q.folder('R2-O1');rows=[]
    try:
        for cp in lock['checkpoints']:
            if cp['rank']==64:
                score,sha=e4.validate('metamath',cp['source'],cp['condition'])
                rows.append(dict(condition=cp['condition'],validation_score=score,metrics_sha256=sha))
    finally:e4.OUT=old
    r.tsv(q.folder('R2-O1')/'rank64_validation.tsv',rows)


def spearman_permutation(x,y,seed):
    x,y=average_ranks(np.asarray(x)),average_ranks(np.asarray(y))
    value=correlation(x,y)
    if value is None:return None,None
    rng=np.random.default_rng(seed)
    null=[correlation(x,rng.permutation(y)) for _ in range(20000)]
    return value,float((1+sum(abs(v)>=abs(value)-1e-12 for v in null))/(len(null)+1))


def o1_extra(exp,lock,rows,stats,modules,lines):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=q.folder(exp);ix={(x['condition'],x['method']):x for x in rows}
    validation={x['condition']:float(x['validation_score']) for x in r.load_tsv(p/'rank64_validation.tsv')}
    curve=[]
    for cp in lock['checkpoints']:
        h=ix[cp['condition'],'hns_f4_s1'];re=ix[cp['condition'],'recon_0']
        curve.append(dict(**{k:cp[k] for k in ('condition','task','rank','multiplier','lr','fraction')},
            validation_score=validation.get(cp['condition'],cp['validation_score']),
            target_gain=h['target']-re['target'],fg_reduction=re['forgetting_gap']-h['forgetting_gap']))
    r.tsv(p/'training_state_curve.tsv',curve)
    correlations=[]
    groups={'pooled_all':curve,'rank16':[x for x in curve if x['rank']==16]}
    for task in ('magicoder','metamath'):groups[task+'_rank16']=[x for x in curve if x['task']==task and x['rank']==16]
    for group,subset in groups.items():
        for metric in ('target_gain','fg_reduction'):
            rho,pvalue=spearman_permutation([x['validation_score'] for x in subset],[x[metric] for x in subset],r.BOOT_SEED)
            correlations.append(dict(group=group,n=len(subset),metric=metric,spearman=rho,
                permutation_p=pvalue,permutations=20000,interpretation='descriptive, correlated checkpoints; pooled task scores are not commensurate'))
    r.tsv(p/'validation_gain_spearman.tsv',correlations)
    fig,axes=plt.subplots(2,1,figsize=(15,8),layout='constrained')
    for ax,task in zip(axes,('magicoder','metamath')):
        subset=sorted([x for x in curve if x['task']==task],key=lambda x:(x['rank'],x['multiplier'],x['fraction']))
        xpos=np.arange(len(subset));ax.axhline(0,color='gray',lw=.8)
        ax.plot(xpos,[x['target_gain'] for x in subset],'o-',label='HNS - Recon-0 Target')
        ax.plot(xpos,[x['fg_reduction'] for x in subset],'s-',label='Recon-0 - HNS FG')
        ax.set_xticks(xpos,[f'r{x["rank"]} LR{x["multiplier"]:g}x {int(100*x["fraction"])}%\nval={100*x["validation_score"]:.1f}' for x in subset])
        ax.set_ylabel('Gain / reduction (pp)');ax.set_title(task+'; single seed42');ax.legend()
    fig.savefig(p/'training_state_curve.png',dpi=180);fig.savefig(p/'training_state_curve.pdf');plt.close(fig)
    lines+=['','## Training state curve','','![Training state curve](training_state_curve.png)','',
        'All12 rank16 candidates and both rank64 checkpoints are shown. Validation is never used to select reported test conditions. '
        'See validation_gain_spearman.tsv for tied-rank Spearman correlations and descriptive permutation p-values. '
        'Pooled task validation scores have different meanings; within-task rank16 comparisons are the primary correlations.']
    for row in correlations:lines.append(f'- {row["group"]}, {row["metric"]}: rho={row["spearman"]}, n={row["n"]}.')


def token_consistency(exp,lock):
    p=q.folder(exp);out=[]
    for cp in lock['checkpoints']:
        for left,right in (('hns_unbal','hns_f4_s1'),('recon_0_unbal','recon_0'),
                           ('hns_unbal','recon_0_unbal'),('hns_f4_s1','recon_0')):
            for task in r.TASKS:
                a=p/'runtime/eval'/cp['condition']/task/q.label(cp,left)/'predictions.jsonl'
                b=p/'runtime/eval'/cp['condition']/task/q.label(cp,right)/'predictions.jsonl'
                records_a=[json.loads(x) for x in a.read_text().splitlines()]
                records_b=[json.loads(x) for x in b.read_text().splitlines()]
                assert len(records_a)==len(records_b)==r.COUNTS[task]
                same=0
                for x,y in zip(records_a,records_b):
                    assert x['id']==y['id'] and 'token_ids' in x and 'token_ids' in y
                    same+=x['token_ids']==y['token_ids']
                out.append(dict(condition=cp['condition'],base=cp['base'],task=cp['task'],seed=cp['seed'],
                    eval_task=task,left=left,right=right,total=len(records_a),identical=same,changed=len(records_a)-same,
                    full_token_sequence_agreement=same/len(records_a)))
    r.tsv(p/'token_consistency.tsv',out)
    return out


def a_extra(exp,lock,rows,stats,modules,lines):
    p=q.folder(exp)
    if exp=='R2-O1':return o1_extra(exp,lock,rows,stats,modules,lines)
    if exp=='R2-O2':
        diffs=[]
        groups={'overall':rows}
        for key in ('base','task'):
            for val in sorted({x[key] for x in rows}):groups[f'{key}/{val}']=[x for x in rows if x[key]==val]
        for group,subset in groups.items():
            diffs.extend(dict(group=group,**x) for x in q.contrast(subset,
                {'hns_unbal':1,'recon_0_unbal':-1,'hns_f4_s1':-1,'recon_0':1},'unbalanced gain minus balanced gain'))
        stats+=diffs;r.tsv(p/'paired_ci.tsv',stats)
        tokens=token_consistency(exp,lock)
        lines+=['','## Factorization comparisons','','The paired difference-in-differences directly compares the HNS-minus-Recon gain under the two writeback choices. '
            'All three metrics and subgroup CIs are in paired_ci.tsv. Exact full-token-sequence agreement, including changed-item counts, is in token_consistency.tsv. '
            'A CI including zero alone does not prove equivalent gains.']
        for x in diffs:
            if x['group']=='overall':lines.append(f'- {x["metric"]} gain difference: {x["difference_pp"]:+.4f} [{x["ci_low_pp"]:+.4f}, {x["ci_high_pp"]:+.4f}] pp.')
        for left,right in (('hns_unbal','hns_f4_s1'),('recon_0_unbal','recon_0')):
            subset=[x for x in tokens if x['left']==left and x['right']==right]
            lines.append(f'- {left} vs {right}: {sum(x["identical"] for x in subset)}/{sum(x["total"] for x in subset)} exact token-sequence matches (item-weighted).')
    if exp=='R2-O3':
        lines+=['','## Strategy and module scope','','| Scope | Method | Target | Off | FG |','| --- | --- | ---: | ---: | ---: |']
        for scope,methods in [('o/down',('hns_od','spectral_surgery_grad_direction','ss_smooth_abs')),
                              ('all7',('hns_f4_s1','ss_grad_direction_all','ss_smooth_abs_all'))]:
            for method in methods:
                subset=[x for x in rows if x['method']==method]
                lines.append(f'| {scope} | {method} | '+ ' | '.join(f'{np.mean([x[m] for x in subset]):.4f}' for m in r.METRICS)+' |')
        builds=[r.read(p/f'worker{i}_build.json') for i in (0,1)]
        budget=[x for b in builds for x in b['artifacts'] if x['method']=='ss_grad_direction_all']
        r.tsv(p/'calibration_budget.tsv',budget)
        lines+=['','SS uses128 calibration examples and64 forward/backward batches per adapter; zero test examples. '
            'Measured calibration/editor/GPU wall seconds are in calibration_budget.tsv; HNS requires0 examples and0 forward/backward passes (SVD timings in E2). '
            'Scope is matched in each row group; data and compute budgets are explicitly reported rather than claimed equal.']
    if exp=='R2-O4':
        summary=[]
        for (condition,method),it in itertools.groupby(sorted(modules,key=lambda x:(x['condition'],x['method'])),key=lambda x:(x['condition'],x['method'])):
            values=list(it);summary.append(dict(condition=condition,method=method,n_modules=len(values),
                **{metric:float(np.mean([float(x[metric]) for x in values])) for metric in ('participation_rank','top1_energy','fro_ratio')}))
        for cp in lock['checkpoints']:
            hmeta=r.read(Path(next(x for x in r.source_rows() if x['source']==cp['source'])['hns_path'])/'spectral_edit_meta.json')['module_stats']
            # Main4+1 statistics use the exact cached adapter metadata.
            metrics=[]
            for x in hmeta.values():
                s=np.asarray(x['sigma_before'],float);t=np.asarray(x['sigma_after'],float)
                metrics.append((t.dot(t)**2/(len(t)*np.sum(t**4)),max(t*t)/t.dot(t),np.linalg.norm(t)/np.linalg.norm(s)))
            summary.append(dict(condition=cp['condition'],method='hns_f4_s1',n_modules=len(metrics),
                **dict(zip(('participation_rank','top1_energy','fro_ratio'),np.mean(metrics,axis=0).tolist()))))
        r.tsv(p/'spectral_statistics.tsv',summary)
        lines+=['','## Achieved spectra','','| Method | Participation rank | Top1 energy | Fro ratio |','| --- | ---: | ---: | ---: |']
        for method in sorted({x['method'] for x in summary}):
            subset=[x for x in summary if x['method']==method]
            lines.append('| '+method+' | '+' | '.join(f'{np.mean([x[m] for x in subset]):.6f}' for m in ('participation_rank','top1_energy','fro_ratio'))+' |')
        lines+=['','All variants are reported. Spectral statistics describe achieved flatness; paired_ci.tsv contains performance comparisons to Recon-0 and4+1. '
            'These measurements support assessing convergence and dose; no independent performance benefit is attributed to a stage without the paired evidence.']


def pipeline():
    r.require_slurm()
    for exp in EXPERIMENTS:
        if (q.folder(exp)/'summary.json').exists():q.require_complete(exp);continue
        prepare(exp);q.launch_workers(exp);q.finalize(exp,a_extra);q.publish(exp)
        print('[Complete]',exp,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--pipeline',action='store_true');p.add_argument('--validate-r64',action='store_true')
    a=p.parse_args()
    if a.prepare:
        for exp in EXPERIMENTS:prepare(exp)
    elif a.validate_r64:validate_r64()
    elif a.pipeline:pipeline()
