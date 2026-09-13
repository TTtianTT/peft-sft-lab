#!/usr/bin/env python3
"""Compare initial DG predictions against the exact archived LoRA generation."""
import json
from pathlib import Path
from audit_posthoc_three_seed import OUT


def main():
    sources = json.loads((OUT / 'source_manifest.json').read_text())['checkpoints']
    rows = []
    for source in sources:
        magic = next(r for r in sources if r['base'] == source['base'] and r['seed'] == source['seed'] and r['task'] == 'magicoder')
        legacy_root = Path(magic['references']['LoRA']['record']['results_path']).parent.parent.parent
        old = legacy_root / source['task'] / f"{source['task']}__original_lora" / 'predictions.jsonl'
        new = OUT / 'eval/dg' / source['base'] / source['task'] / f"{source['task']}__seed{source['seed']}__dg_hard" / 'predictions.jsonl'
        pp, nn = [{r['id']: r for r in map(json.loads, p.read_text().splitlines())} for p in (old, new)]
        assert pp.keys() == nn.keys()
        fields = {'magicoder': ('problem_prompt', 'entry_point'), 'metamath': ('question', 'gold'), 'tulu': ('prompt', 'instruction_id_list', 'kwargs')}[source['task']]
        assert all(pp[k][field] == nn[k][field] for k in pp for field in fields)
        metric = json.loads((new.parent / 'metrics.json').read_text())
        initial_score = metric[metric['primary_metric']] * 100
        rows.append({**{k: source[k] for k in ('base', 'task', 'seed')}, 'legacy_predictions': str(old),
            'initial_dg_predictions': str(new), 'samples': len(pp),
            'token_mismatches': sum(pp[k]['token_ids'] != nn[k]['token_ids'] for k in pp),
            'text_mismatches': sum(pp[k]['prediction_text'] != nn[k]['prediction_text'] for k in pp),
            'input_records_identical': True, 'legacy_LoRA_score': source['references']['LoRA']['score_percent'],
            'initial_DG_score': initial_score, 'score_delta_pp': initial_score - source['references']['LoRA']['score_percent']})
    result = {'status': 'complete', 'rows': rows, 'token_mismatches': sum(r['token_mismatches'] for r in rows),
        'samples': sum(r['samples'] for r in rows), 'explanation': 'Exact weights and identical benchmark input records. Generation drift observed relative to legacy scheduling. Mechanism not established; use joint controls for final comparison.'}
    (OUT / 'identity_legacy_drift_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k != 'rows'})


if __name__ == '__main__':
    main()
