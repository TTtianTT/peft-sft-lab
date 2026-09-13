#!/usr/bin/env python3
"""Prepare a uniform comparison after observed identity/legacy generation drift."""
import json
from audit_posthoc_three_seed import OUT
from summarize_posthoc_flat_dghard_three_seed import stats


def main():
    audit = json.loads((OUT / 'source_manifest.json').read_text())
    flat = json.loads((OUT / 'flat_summary.json').read_text())
    assert flat['status'] == 'complete' and len(flat['rows']) == 18
    initial = []
    for row in flat['rows']:
        r = dict(row)
        dest = OUT / 'eval/dg' / r['base']
        assert (dest / 'generation_manifest.json').is_file()
        sm = json.loads((dest / 'score_manifest.json').read_text())
        label = f"{r['task']}__seed{r['seed']}__dg_hard"
        rec = next(c for c in sm['records'] if c['variant'] == label and c['task'] == r['task'])
        assert rec['samples'] == {'magicoder': 164, 'metamath': 1319, 'tulu': 541}[r['task']]
        r['DG-Hard'] = 100 * rec[rec['primary_metric']]
        initial.append(r)
    (OUT / 'dg_initial_summary.json').write_text(json.dumps({'status': 'complete', 'rows': initial,
        'overall': stats(initial, ['LoRA', 'Flat-Fro', 'Flat-Nuclear', 'DG-Hard', 'HNS']),
        'note': 'Independent initial evaluation; LoRA/HNS historical references. Identity adapters exhibited legacy generation drift. Main final table uses the uniform joint evaluation instead.'}, indent=2) + '\n')
    for base in sorted({r['base'] for r in audit['checkpoints']}):
        variants = []
        subset = [r for r in audit['checkpoints'] if r['base'] == base]
        for r in subset:
            parent = OUT / 'adapters' / base / f"seed{r['seed']}" / r['task']
            for method, path in [('original_lora', r['source']), ('flat_fro', parent / 'flat_fro'),
                ('flat_nuclear', parent / 'flat_nuclear'), ('dg_hard', parent / 'dg_hard'), ('hns_f4_s1', r['hns_path'])]:
                variants.append({'label': f"{r['task']}__seed{r['seed']}__{method}", 'path': str(path),
                    'method': method, 'seed': r['seed'], 'train_task': r['task']})
        (OUT / f'{base}_joint_variant_manifest.json').write_text(json.dumps({'status': 'complete',
            'base_model': subset[0]['base_model'], 'task_config': subset[0]['task_config'],
            'adapter_block_size': 5, 'reason': 'Each checkpoint has all five methods in one batch, with exact original LoRA/DG identity controls.',
            'variants': variants}, indent=2) + '\n')
    print('Initial DG results retained; joint manifests ready: 90 cells, all five methods/checkpoint.')


if __name__ == '__main__':
    main()
