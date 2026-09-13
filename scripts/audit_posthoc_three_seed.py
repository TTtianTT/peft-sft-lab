#!/usr/bin/env python3
"""Resolve sources from the completed HNS manifests and audit all 18 adapters."""
import hashlib
import json
from pathlib import Path

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key, find_adapter_weight_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/posthoc_flat_dghard_three_seed_20260913'
RUN = Path('/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912')
BASES = [('Qwen3-8B', 'qwen3_8b'), ('Llama-3.1-8B-Instruct', 'llama31_8b')]
TASKS = ('magicoder', 'metamath', 'tulu')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    rows = []
    for base, slug in BASES:
        for seed in (42, 43, 44):
            if seed == 42:
                folder = ROOT / 'reports/hns_release_20260912/data/step_grid/manifests'
                vp, sp, gp = [folder / f'{slug}_{kind}_manifest.json' for kind in ('variant', 'score', 'generation')]
            else:
                folder = RUN / base / f'seed{seed}/hns_step_grid'
                vp = folder / f'adapters/{base}/variant_manifest.json'
                sp, gp = [folder / f'eval/{kind}_manifest.json' for kind in ('score', 'generation')]
            vm, sm, gm = [json.loads(p.read_text()) for p in (vp, sp, gp)]
            cfg_path = Path(vm['task_config'])
            cfg = json.loads(cfg_path.read_text())
            for task in TASKS:
                source = next(v for v in vm['variants'] if v['label'] == f'{task}__original_lora')
                hns = next(v for v in vm['variants'] if v['label'] == f'{task}__hns_f4_s1')
                cp = Path(source['path']).resolve()
                assert cp == Path(next(c['lora'] for c in cfg['checkpoints'] if c['base'] == base and c['train_task'] == task)).resolve()
                ac = json.loads((cp / 'adapter_config.json').read_text())
                assert ac['r'] == 16 and ac['lora_alpha'] == 32
                hm = json.loads((Path(hns['path']) / 'spectral_edit_meta.json').read_text())['meta']
                assert hm['fast_steps'] == 4 and hm['stable_steps'] == 1
                assert hm['scope'] == 'all_modules' and hm['preserve_nuclear_norm'] is True and hm['hns_strength'] == 1.0
                assert Path(hm['source_lora']).resolve() == cp
                assert sha(Path(hns['path']) / 'adapter_config.json') == sha(cp / 'adapter_config.json')
                state, fmt = load_lora_state_dict(str(cp))
                pairs = {}
                for key, tensor in state.items():
                    parsed = parse_lora_ab_key(key)
                    if parsed:
                        prefix, which, adapter = parsed
                        pairs.setdefault((prefix, adapter), {})[which] = tensor
                assert pairs
                shapes = {}
                for (prefix, adapter), pair in pairs.items():
                    assert set(pair) == {'A', 'B'}, prefix
                    a, b = pair['A'], pair['B']
                    assert a.shape[0] == b.shape[1] == 16
                    assert a.isfinite().all() and b.isfinite().all()
                    shapes[prefix] = {'A': list(a.shape), 'B': list(b.shape), 'dtype': str(a.dtype)}
                spec = cfg['evaluation_tasks'][task]
                references = {}
                for name, label in [('LoRA', source['label']), ('HNS', hns['label'])]:
                    rec = next(r for r in sm['records'] if r['task'] == task and r['variant'] == label)
                    assert rec['samples'] == spec['samples']
                    references[name] = {'score_percent': 100 * rec[rec['primary_metric']], 'record': rec}
                weights, _ = find_adapter_weight_file(str(cp))
                run_args_path = cp / 'run_args.json'
                training_evidence = None
                if run_args_path.is_file():
                    run_args = json.loads(run_args_path.read_text())
                    training_evidence = {'path': str(run_args_path), 'sha256': sha(run_args_path),
                        'seed': run_args.get('seed'), 'dataset_seed': run_args.get('dataset_seed')}
                if seed in (43, 44):
                    assert training_evidence is not None and training_evidence['seed'] == seed
                rows.append({'base': base, 'task': task, 'seed': seed, 'source': str(cp),
                    'source_variant_manifest': str(vp), 'source_score_manifest': str(sp),
                    'source_generation_manifest': str(gp), 'task_config': str(cfg_path),
                    'base_model': vm['base_model'], 'hns_path': hns['path'], 'hns_metadata': hm,
                    'source_sha256': sha(weights), 'config_sha256': sha(cp / 'adapter_config.json'),
                    'adapter_config': ac, 'module_count': len(pairs), 'module_shapes': shapes,
                    'evaluation_spec': spec, 'generation_configuration': gm['configuration'],
                    'references': references, 'training_cli_seed_evidence': training_evidence,
                    'seed_evidence': 'historical label; original training seed not fully verified' if seed == 42 and base.startswith('Llama') and task != 'tulu' else 'supported by existing three-seed audit'})
                print(f'[Audit] {base}/{task}/seed{seed}: {cp}, modules={len(pairs)}', flush=True)
    assert len(rows) == 18 and len({r['source'] for r in rows}) == 18
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'source_manifest.json').write_text(json.dumps({'status': 'complete', 'hns_representative': '4+1, all modules, strength=1, nuclear preservation', 'checkpoints': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
