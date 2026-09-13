#!/usr/bin/env python3
"""Build all Flat adapters using audited sources and the shared HNS SVD/I/O."""
import json
import shutil
from pathlib import Path
import torch
from audit_posthoc_three_seed import OUT, sha
from build_hns_step_grid_2x4 import collect_pairs
from finetune.spectral_edit.flat import flat_spectrum
from finetune.spectral_edit.io import load_lora_state_dict, save_lora_state_dict
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def build(row):
    root = OUT / 'adapters' / row['base'] / f"seed{row['seed']}" / row['task']
    manifest = root / 'flat_manifest.json'
    if manifest.is_file():
        result = json.loads(manifest.read_text())
        assert result['source_sha256'] == row['source_sha256']
        return result
    root.mkdir(parents=True, exist_ok=True)
    state, fmt = load_lora_state_dict(row['source'])
    pairs = collect_pairs(state)
    states = {m: dict(state) for m in ('flat_fro', 'flat_nuclear')}
    stats = {m: {} for m in states}
    with torch.inference_mode():
        for prefix, pair in sorted(pairs.items()):
            ka, a = pair['A']; kb, b = pair['B']
            u, s, vh, _ = lowrank_svd_from_ba(b, a)
            for method in states:
                t = flat_spectrum(s, method)
                bn, an = rebuild_ba_from_uv_sigma(u, vh, t)
                bs, ass = bn.to(b.dtype), an.to(a.dtype)
                _, saved_s, _, _ = lowrank_svd_from_ba(bs, ass)
                before = s.norm() if method == 'flat_fro' else s.sum()
                target = t.norm() if method == 'flat_fro' else t.sum()
                saved = saved_s.norm() if method == 'flat_fro' else saved_s.sum()
                err = float((target - before).abs() / before.clamp_min(1e-12))
                saved_err = float((saved - before).abs() / before.clamp_min(1e-12))
                assert err < 2e-6, (prefix, method, err)
                assert saved_err < 5e-3, (prefix, method, saved_err)
                assert ass.shape == a.shape and bs.shape == b.shape
                states[method][ka], states[method][kb] = ass.contiguous(), bs.contiguous()
                stats[method][prefix] = {'rank': len(s), 'singular_values_before': s.tolist(),
                    'target_level': float(t[0]), 'fro_before': float(s.norm()), 'fro_target': float(t.norm()),
                    'fro_saved': float(saved_s.norm()), 'nuclear_before': float(s.sum()),
                    'nuclear_target': float(t.sum()), 'nuclear_saved': float(saved_s.sum()),
                    'budget_relative_error': err, 'saved_budget_relative_error': saved_err}
    variants = []
    for method, edited in states.items():
        dest = root / method
        if dest.exists():
            raise FileExistsError(dest)
        dest.mkdir()
        # Copy adapter/tokenizer files only; never copy Trainer optimizer/checkpoints.
        for p in Path(row['source']).iterdir():
            if p.is_file() and (p.name == 'adapter_config.json' or p.name.startswith(('tokenizer', 'special_tokens', 'added_tokens', 'vocab', 'merges', 'chat_template'))):
                shutil.copy2(p, dest / p.name)
        save_lora_state_dict(str(dest), edited, fmt)
        assert sha(dest / 'adapter_config.json') == row['config_sha256']
        # Reload the actual artifact, rather than checking only tensors in memory.
        reloaded, _ = load_lora_state_dict(str(dest))
        assert reloaded.keys() == state.keys()
        assert all(torch.equal(reloaded[k], v) for k, v in edited.items())
        metadata = {'method': method, 'source': row['source'], 'source_sha256': row['source_sha256'],
            'balanced_reconstruction': True, 'module_stats': stats[method]}
        (dest / 'posthoc_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
        variants.append({'label': f"{row['task']}__seed{row['seed']}__{method}", 'path': str(dest),
            'train_task': row['task'], 'seed': row['seed'], 'method': method,
            'max_budget_relative_error': max(s['budget_relative_error'] for s in stats[method].values()),
            'max_saved_budget_relative_error': max(s['saved_budget_relative_error'] for s in stats[method].values())})
    result = {'status': 'complete', 'source_sha256': row['source_sha256'], 'source': row['source'], 'variants': variants}
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    print(f"[Build] {row['base']}/{row['task']}/seed{row['seed']}: {variants}", flush=True)
    return result


def main():
    torch.set_num_threads(4)
    audit = json.loads((OUT / 'source_manifest.json').read_text())
    results = [(row, build(row)) for row in audit['checkpoints']]
    for base in sorted({r['base'] for r, _ in results}):
        subset = [(r, m) for r, m in results if r['base'] == base]
        # All six source protocols must use the same benchmark data and token limits.
        for task in ('magicoder', 'metamath', 'tulu'):
            specs = [r['evaluation_spec'] for r, _ in subset if r['task'] == task]
            assert all(s == specs[0] for s in specs)
        (OUT / f'{base}_flat_variant_manifest.json').write_text(json.dumps({
            'status': 'complete', 'base_model': subset[0][0]['base_model'],
            'task_config': subset[0][0]['task_config'],
            'variants': [v for _, m in subset for v in m['variants']]}, indent=2) + '\n')
    (OUT / 'flat_build_complete.json').write_text(json.dumps({'status': 'complete', 'checkpoints': 18, 'adapters': 36, 'results': [m for _, m in results]}, indent=2) + '\n')


if __name__ == '__main__':
    main()
