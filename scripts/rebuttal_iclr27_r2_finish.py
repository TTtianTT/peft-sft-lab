#!/usr/bin/env python3
"""Verify expected second-round outputs and publish an honest completion index."""
import os
from pathlib import Path
import subprocess
import numpy as np
import rebuttal_iclr27 as r
import rebuttal_iclr27_r2 as q


def main():
    r.require_slurm();entries={};all_rows=[];summaries=[]
    factor=r.read(q.folder('R2-O2')/'factorization_update_audit.json');assert factor['status']=='pass'
    for exp in q.SCOPES:
        path=q.folder(exp)
        if (path/'summary.json').exists():
            q.require_complete(exp);summary=r.read(path/'summary.json');entries[exp]=dict(status='complete',summary=summary)
            rows=r.load_tsv(path/'checkpoint_results.tsv')
            for method in sorted({x['method'] for x in rows}):
                selected=[x for x in rows if x['method']==method]
                all_rows.append(dict(experiment=exp,method=method,conditions=len(selected),
                    **{metric:float(np.mean([float(x[metric]) for x in selected])) for metric in r.METRICS}))
            summaries.append(summary)
        elif exp=='R2-O11':
            entries[exp]=dict(status='awaiting_author_confirmation',message='Not executed; no approval inferred from silence.')
        else:
            entries[exp]=dict(status='not_started_budget',message='Optional C-tier stage was not started within the fixed wall-clock budget.')
    for exp in q.SCOPES[:8]:assert entries[exp]['status']=='complete',(exp,entries[exp])
    decisions=r.read(r.OUT/'R2-C-budget.json')
    for decision in decisions:
        exp=decision['experiment']
        if entries[exp]['status']!='complete':entries[exp]=decision
    new_cells=sum(s['new_cells'] for s in summaries)
    trainings=0
    for exp in q.SCOPES:
        path=q.folder(exp)/'training_audit.json'
        if path.exists():trainings+=len(r.read(path))
    result=dict(status='authorized_completed_with_explicit_optional_omissions',updated_utc=r.now(),
        experiments=entries,total_new_cells=new_cells,total_new_trainings=trainings,
        o11_author_approval=q.initialize_round()['o11_author_approval'],global_gpu_limit=2,
        submission_sha256=r.sha(r.OUT/'R2-submission.json'),budget_decisions_sha256=r.sha(r.OUT/'R2-C-budget.json'),factorization_update_audit=factor)
    r.write(r.OUT/'R2-completion_index.json',result);r.tsv(r.OUT/'R2-method_summary.tsv',all_rows)
    lines=['# Second-round rebuttal completion','','All A/B experiments passed output, coverage and integrity audits. '
        'Optional stages and pending author confirmation are listed explicitly.','',
        f'New training runs: {trainings}; new cells: {new_cells}; global GPU limit:2.',
        '','| Experiment | Status | New cells | Report |','| --- | --- | ---: | --- |']
    for exp,entry in entries.items():
        completed=entry['status']=='complete';n=entry.get('summary',{}).get('new_cells',0)
        lines.append(f'| {exp} | {entry["status"]} | {n} | '+(f'[{exp}]({exp}/report.md)' if completed else entry.get('message','See R2-C-budget.json'))+' |')
    lines+=['','All methods/configurations are reported; none were chosen using main or auxiliary test performance. '
        'Main Target/Off/FG preserve the functional_hns_three_seed_20260914 protocol and caches after sample-input-hash audits. '
        'Checkpoint CIs are descriptive shared-test intervals; single-seed conditions use paired question intervals and exact McNemar repair/damage. '
        'See each report for inference/capacity exceptions (rank32/64) and any limits on interpretation.',
        '',f'Balanced/unbalanced represented-update audit: {factor["compared_module_pairs"]} module pairs, maximum relative difference{factor["max_relative_update_difference"]:.8g}, fixed tolerance5e-5. [Audit](R2-O2/factorization_update_audit.json).',
        '','## Aggregate values','','| Experiment | Method | Target | Off | FG |','| --- | --- | ---: | ---: | ---: |']
    for row in all_rows:lines.append('| '+row['experiment']+' | '+row['method']+' | '+' | '.join(f'{row[m]:.4f}' for m in r.METRICS)+' |')
    lines+=['','These aggregates pool only the prespecified conditions within each experiment and do not imply comparable training-run uncertainty across experiments. '
        'Use paired_ci.tsv and subgroup results for the actual contrasts. O11 was not executed without explicit author authorization.']
    report=r.OUT/'R2-completion_report.md';report.write_text('\n'.join(lines)+'\n')
    root=r.OUT/'completion_report.md';text=root.read_text()
    marker='<!-- round-two-final-index -->'
    if marker not in text:
        with root.open('a') as h:h.write(f'\n{marker}\n\n## Second-round completion index\n\n{trainings} new trainings, {new_cells} new cells. '
            '[Audited second-round index](R2-completion_report.md). O11 remains unexecuted pending explicit author confirmation; all optional omissions are documented.\n')
    plan=q.initialize_round();plan['status']='authorized_work_complete_o11_pending';plan['finished_utc']=r.now();r.write(r.OUT/'R2-plan.json',plan)
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=r.ROOT,text=True).strip()
    names=['R2-completion_index.json','R2-completion_report.md','R2-method_summary.tsv','completion_report.md','R2-C-budget.json','R2-plan.json','R2-submission.json','R2-publication.json',
           'R2-O2/factorization_update_audit.json','R2-O2/factorization_update_audit.tsv']
    subprocess.run(['git','add','--',*[str((r.OUT/n).relative_to(r.ROOT)) for n in names if (r.OUT/n).exists()]],cwd=r.ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=r.ROOT).returncode:
        subprocess.run(['git','commit','-m','Index audited second-round rebuttal completion and optional omissions'],cwd=r.ROOT,check=True)
    env=os.environ.copy();env['GIT_SSH_COMMAND']='ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes'
    subprocess.run(['git','push','git@github.com:TTtianTT/peft-sft-lab.git','refactor/sft-chat-template'],cwd=r.ROOT,env=env,check=True)
    print('Verified and published second round:',new_cells,'new cells,',trainings,'trainings.',flush=True)


if __name__=='__main__':main()
