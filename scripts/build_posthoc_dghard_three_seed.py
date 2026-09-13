#!/usr/bin/env python3
"""Build original DG-Hard only after the complete Flat summary exists."""
import json
import shutil
from pathlib import Path
import torch
from audit_posthoc_three_seed import OUT, sha
from build_hns_step_grid_2x4 import collect_pairs
from finetune.spectral_edit.dg_hard import dg_hard_spectrum
from finetune.spectral_edit.io import load_lora_state_dict, save_lora_state_dict
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def build(row):
    root = OUT / 'adapters' / row['base'] / f"seed{row['seed']}" / row['task']
    manifest = root / 'dg_manifest.json'
    if manifest.is_file():
        result = json.loads(manifest.read_text())
        assert result['source_sha256'] == row['source_sha256']
        return result
    state, fmt = load_lora_state_dict(row['source'])
    edited = dict(state)
    pairs = collect_pairs(state)
    stats = {}
    with torch.inference_mode():
        for prefix, pair in sorted(pairs.items()):
            ka, a = pair['A']; kb, b = pair['B']
            u, s, vh, _ = lowrank_svd_from_ba(b, a)
            t, st = dg_hard_spectrum(s, b.shape[0], a.shape[1])
            if not st['unchanged']:
                bn, an = rebuild_ba_from_uv_sigma(u, vh, t)
                edited[ka], edited[kb] = an.to(a.dtype).contiguous(), bn.to(b.dtype).contiguous()
            # An identity transform preserves the original factors exactly.
            stats[prefix] = st
    dest = root / 'dg_hard'
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir()
    for p in Path(row['source']).iterdir():
        if p.is_file() and (p.name == 'adapter_config.json' or p.name.startswith(('tokenizer', 'special_tokens', 'added_tokens', 'vocab', 'merges', 'chat_template'))):
            shutil.copy2(p, dest / p.name)
    save_lora_state_dict(str(dest), edited, fmt)
    assert sha(dest / 'adapter_config.json') == row['config_sha256']
    saved, _ = load_lora_state_dict(str(dest))
    assert saved.keys() == state.keys() and all(torch.equal(saved[k], v) for k, v in edited.items())
    values = list(stats.values())
    before_fro = sum(s['fro_before']**2 for s in values)**.5
    after_fro = sum(s['fro_retained']**2 for s in values)**.5
    before_nuc = sum(s['nuclear_before'] for s in values)
    after_nuc = sum(s['nuclear_retained'] for s in values)
    summary = {'modules': len(values), 'retained_rank_min': min(s['retained_rank'] for s in values),
        'retained_rank_max': max(s['retained_rank'] for s in values),
        'unchanged_module_fraction': sum(s['unchanged'] for s in values) / len(values),
        'threshold_min': min(s['threshold'] for s in values), 'threshold_max': max(s['threshold'] for s in values),
        'fro_before': before_fro, 'fro_retained': after_fro, 'nuclear_before': before_nuc, 'nuclear_retained': after_nuc,
        'retained_fro_ratio': after_fro / before_fro if before_fro else 1.,
        'retained_nuclear_ratio': after_nuc / before_nuc if before_nuc else 1.,
        'identity': all(s['unchanged'] for s in values),
        'source_factors_bytewise_equal': all(torch.equal(saved[k].contiguous().view(torch.uint8), v.contiguous().view(torch.uint8)) for k, v in state.items())}
    metadata = {'method': 'dg_hard', 'original_algorithm': True, 'active_spectrum_adaptation': False,
        'source': row['source'], 'source_sha256': row['source_sha256'], 'summary': summary, 'module_stats': stats}
    (dest / 'posthoc_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    result = {'status': 'complete', **{k: row[k] for k in ('base', 'task', 'seed')},
        'source_sha256': row['source_sha256'], 'source': row['source'], 'summary': summary,
        'variant': {'label': f"{row['task']}__seed{row['seed']}__dg_hard", 'path': str(dest),
            'train_task': row['task'], 'seed': row['seed'], 'method': 'dg_hard'}}
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    print(f"[DG] {row['base']}/{row['task']}/seed{row['seed']}: {summary}", flush=True)
    return result


def main():
    torch.set_num_threads(4)
    flat = json.loads((OUT / 'flat_summary.json').read_text())
    assert flat['status'] == 'complete' and len(flat['rows']) == 18
    audit = json.loads((OUT / 'source_manifest.json').read_text())
    results = [(row, build(row)) for row in audit['checkpoints']]
    for base in sorted({r['base'] for r, _ in results}):
        subset = [(r, m) for r, m in results if r['base'] == base]
        (OUT / f'{base}_dg_variant_manifest.json').write_text(json.dumps({
            'status': 'complete', 'base_model': subset[0][0]['base_model'], 'task_config': subset[0][0]['task_config'],
            'variants': [m['variant'] for _, m in subset]}, indent=2) + '\n')
    (OUT / 'dg_build_complete.json').write_text(json.dumps({'status': 'complete', 'checkpoints': 18, 'results': [m for _, m in results]}, indent=2) + '\n')


if __name__ == '__main__':
    main()
