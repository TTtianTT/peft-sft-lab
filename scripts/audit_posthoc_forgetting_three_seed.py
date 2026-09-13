#!/usr/bin/env python3
"""Verify complete four-benchmark coverage and exact original DG identity."""
import json
import itertools
from pathlib import Path
from prepare_posthoc_forgetting_three_seed import ROOT, OUT, DIAGONAL, BASES, TASKS, COUNTS, sha
from summarize_posthoc_forgetting_three_seed import atomic, main as publish


def rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    print('[Audit] Checking frozen evaluation code, datasets, and adapter fingerprints',flush=True)
    for path,expected in manifest['code_sha256'].items():
        assert sha(ROOT/path)==expected, f'Evaluation code changed: {path}'
    for dataset in manifest['datasets']:
        assert sha(dataset['path'])==dataset['sha256'], f'Dataset changed: {dataset["path"]}'
    for adapter in manifest['adapters']:
        folder=Path(adapter['path'])
        assert sha(folder/'adapter_model.safetensors')==adapter['weight_sha256'], f'Adapter weights changed: {folder}'
        assert sha(folder/'adapter_config.json')==adapter['config_sha256'], f'Adapter config changed: {folder}'
    atomic(OUT/'artifact_integrity_audit.json',json.dumps(dict(status='pass',
        evaluation_code_files=len(manifest['code_sha256']),datasets=len(manifest['datasets']),
        adapters=len(manifest['adapters'])),indent=2)+'\n')
    checks=[]
    samples=token_mismatches=metric_mismatches=0
    diagonal_changes=[]
    diagonal_cells=0
    coverage=[]
    for base,_ in BASES:
        print(f'[Audit] Coverage and identity: {base}',flush=True)
        root=OUT/'eval'/base
        generation=json.loads((root/'generation_manifest.json').read_text())
        assert {t['task']:t['samples'] for t in generation['tasks']}==COUNTS
        records=json.loads((root/'score_manifest.json').read_text())['records']
        vm=json.loads((OUT/f'{base}_variant_manifest.json').read_text())
        expected={(task,label) for task in TASKS for label in ['base']+[v['label'] for v in vm['variants']]}
        assert {(r['task'],r['variant']) for r in records}==expected and len(records)==184
        scores={(r['task'],r['variant']):r[r['primary_metric']] for r in records}
        for task in TASKS:
            base_rows=list(rows(root/task/'base'/'predictions.jsonl'))
            assert len(base_rows)==COUNTS[task]
            assert len({str(r['id']) for r in base_rows})==COUNTS[task]
            inputs=[{k:v for k,v in r.items() if k not in ('prediction_text','token_ids','finish_reason')} for r in base_rows]
            for label in ['base']+[v['label'] for v in vm['variants']]:
                count=0
                for expected_input,actual in itertools.zip_longest(inputs,rows(root/task/label/'predictions.jsonl')):
                    assert expected_input is not None and actual is not None, (base,task,label)
                    actual_input={k:v for k,v in actual.items() if k not in ('prediction_text','token_ids','finish_reason')}
                    assert actual_input==expected_input, (base,task,label,actual['id'])
                    count+=1
                metric=next(r for r in records if r['task']==task and r['variant']==label)
                assert count==metric['samples']==COUNTS[task]
                coverage.append(dict(base=base,eval_task=task,label=label,samples=count,
                    input_records_match_base=True,unique_ids=True))
        previous=json.loads((DIAGONAL/'eval/joint'/base/'score_manifest.json').read_text())['records']
        for old in previous:
            key=old['task'],old['variant']
            diagonal_cells+=1
            before=old[old['primary_metric']]
            if abs(scores[key]-before)>1e-12:
                diagonal_changes.append(dict(base=base,eval_task=key[0],variant=key[1],
                    previous=before*100,current=scores[key]*100,delta_pp=(scores[key]-before)*100))
        for variant in vm['variants']:
            if variant['method']!='original_lora':
                continue
            lora=variant['label']
            dg=lora.replace('__original_lora','__dg_hard')
            for task in TASKS:
                count=changed=0
                left=root/task/lora/'predictions.jsonl'
                right=root/task/dg/'predictions.jsonl'
                for a,b in itertools.zip_longest(rows(left),rows(right)):
                    assert a is not None and b is not None
                    assert a['id']==b['id']
                    assert {k:v for k,v in a.items() if k not in ('prediction_text','token_ids','finish_reason')}=={k:v for k,v in b.items() if k not in ('prediction_text','token_ids','finish_reason')}
                    changed+=a['token_ids']!=b['token_ids']
                    count+=1
                assert count==COUNTS[task]
                difference=abs(scores[task,lora]-scores[task,dg])>1e-12
                checks.append(dict(base=base,train_task=variant['train_task'],seed=variant['seed'],
                    eval_task=task,samples=count,token_mismatches=changed,metric_mismatch=difference,
                    lora_score=scores[task,lora]*100,dg_score=scores[task,dg]*100))
                samples+=count;token_mismatches+=changed;metric_mismatches+=difference
    result=dict(status='pass' if not token_mismatches and not metric_mismatches else 'fail',
        pairs=len(checks),samples=samples,token_mismatches=token_mismatches,
        metric_mismatches=metric_mismatches,checks=checks)
    atomic(OUT/'identity_audit.json',json.dumps(result,indent=2)+'\n')
    atomic(OUT/'coverage_audit.json',json.dumps(dict(status='pass',cells=len(coverage),
        samples=sum(r['samples'] for r in coverage),checks=coverage),indent=2)+'\n')
    atomic(OUT/'diagonal_audit.json',json.dumps(dict(cells=diagonal_cells,changed_metrics=len(diagonal_changes),changes=diagonal_changes),indent=2)+'\n')
    publish()
    print(json.dumps({k:result[k] for k in ('status','pairs','samples','token_mismatches','metric_mismatches')},indent=2))
    if result['status']!='pass':
        raise SystemExit(2)


if __name__=='__main__':
    main()
