#!/usr/bin/env python3
"""Freeze the existing five-method adapters and HNS forgetting protocol."""
import json
import importlib.metadata
from pathlib import Path
import torch
from audit_posthoc_three_seed import ROOT, OUT as DIAGONAL, BASES, sha
from finetune.spectral_edit.io import load_lora_state_dict

OUT = ROOT / 'reports/posthoc_flat_dghard_forgetting_three_seed_20260913'
REPORT = OUT.with_suffix('.md')
TASKS = ('magicoder', 'metamath', 'tulu', 'commonsense')
METHODS = ('original_lora', 'flat_fro', 'flat_nuclear', 'dg_hard', 'hns_f4_s1')
NAMES = ('LoRA', 'Flat-Fro', 'Flat-Nuclear', 'DG-Hard', 'HNS 4+1')
COUNTS = dict(magicoder=164, metamath=1319, tulu=541, commonsense=22419)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    source = json.loads((DIAGONAL / 'source_manifest.json').read_text())
    assert len(source['checkpoints']) == 18
    reference = ROOT / 'configs/hns_forgetting_2x4_20260911.json'
    reference_cfg = json.loads(reference.read_text())
    cs = reference_cfg['evaluation_tasks']['commonsense']
    datasets = []
    for task in TASKS[:3]:
        spec = reference_cfg['evaluation_tasks'][task]
        path = Path(spec['dataset'])
        assert path.is_file(), path
        datasets.append(dict(task=task, path=str(path), sha256=sha(path)))
    cs_counts = {}
    for task in cs['tasks']:
        path = Path(cs['reference_predictions_root']) / task / 'predictions.jsonl'
        assert path.is_file(), path
        with path.open() as handle:
            cs_counts[task] = sum(bool(line.strip()) for line in handle)
        datasets.append(dict(task='commonsense', subtask=task, samples=cs_counts[task],
                             path=str(path), sha256=sha(path)))
    assert sum(cs_counts.values()) == COUNTS['commonsense']
    adapters = []
    for base, _ in BASES:
        previous = DIAGONAL / f'{base}_joint_variant_manifest.json'
        vm = json.loads(previous.read_text())
        assert len(vm['variants']) == 45
        cfg = json.loads(Path(vm['task_config']).read_text())
        for task in TASKS[:3]:
            assert cfg['evaluation_tasks'][task] == reference_cfg['evaluation_tasks'][task]
        cfg['evaluation_tasks']['commonsense'] = cs
        config_path = OUT / f'{base}_task_config.json'
        config_path.write_text(json.dumps(cfg, indent=2) + '\n')
        for start in range(0, 45, 5):
            block = vm['variants'][start:start+5]
            assert tuple(v['method'] for v in block) == METHODS
            assert len({(v['train_task'], v['seed']) for v in block}) == 1
            row = next(r for r in source['checkpoints'] if r['base'] == base
                       and r['task'] == block[0]['train_task'] and r['seed'] == block[0]['seed'])
            assert block[0]['path'] == row['source'] and block[4]['path'] == row['hns_path']
            for variant in block:
                path = Path(variant['path'])
                assert (path / 'adapter_model.safetensors').is_file()
                assert sha(path / 'adapter_config.json') == row['config_sha256']
                weight_sha = sha(path / 'adapter_model.safetensors')
                if variant['method'] == 'original_lora':
                    assert weight_sha == row['source_sha256'], (base,variant['label'])
                if variant['method'] == 'dg_hard':
                    original,_=load_lora_state_dict(row['source'])
                    saved,_=load_lora_state_dict(str(path))
                    assert original.keys()==saved.keys()
                    assert all(torch.equal(v.contiguous().view(torch.uint8),saved[k].contiguous().view(torch.uint8)) for k,v in original.items())
                    del original,saved
                adapters.append(dict(base=base, **variant, weight_sha256=weight_sha,
                                     config_sha256=row['config_sha256'],
                                     source_factors_bytewise_equal=True if variant['method']=='dg_hard' else None))
        vm.update(task_config=str(config_path), source_variant_manifest=str(previous),
                  evaluation_tasks=list(TASKS), diagonal_only=False)
        (OUT / f'{base}_variant_manifest.json').write_text(json.dumps(vm, indent=2)+'\n')
        probe = dict(vm, variants=vm['variants'][:5])
        (OUT / f'{base}_probe_manifest.json').write_text(json.dumps(probe, indent=2)+'\n')
    code = ['scripts/eval_forgetting_matrix_vllm.py', 'scripts/score_forgetting_matrix.py',
            'scripts/eval_commonsense_8tasks.py', 'src/finetune/eval/prompting.py',
            'src/finetune/eval/eval_humaneval.py', 'src/finetune/eval/eval_gsm8k.py',
            'src/finetune/eval/eval_ifeval.py', 'src/finetune/eval/generation.py']
    code = [p for p in code if (ROOT / p).is_file()]
    orchestration=['scripts/prepare_posthoc_forgetting_three_seed.py',
        'scripts/run_posthoc_forgetting_three_seed.py','scripts/summarize_posthoc_forgetting_three_seed.py',
        'scripts/audit_posthoc_forgetting_three_seed.py','slurm/posthoc_forgetting_three_seed.slurm']
    audit = dict(status='ready', checkpoint_count=18, adapter_count=90,
        expected_cells=368, cells_per_base=184, evaluation_tasks=list(TASKS),
        expected_samples=COUNTS, commonsense_subtask_counts=cs_counts,
        source_manifest=str(DIAGONAL/'source_manifest.json'),
        source_manifest_sha256=sha(DIAGONAL/'source_manifest.json'),
        reference_config=str(reference), reference_config_sha256=sha(reference),
        hns_representative=source['hns_representative'], datasets=datasets, adapters=adapters,
        code_sha256={p:sha(ROOT/p) for p in code},
        orchestration_sha256={p:sha(ROOT/p) for p in orchestration},
        versions={p:importlib.metadata.version(p) for p in ('vllm','torch','transformers','peft','safetensors','human-eval')},
        protocol=dict(off_task='Equal weight over the other three benchmark families',
            commonsense='Equal weight over eight subtask accuracies',
            forgetting_gap='Mean of max(Base_score - adapter_score, 0) over off-task families',
            adapter_block_size=5, inference_seed=42, training_seeds=[42,43,44],
            gpu_limit=2, retraining=False))
    (OUT/'manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ('status','checkpoint_count','adapter_count','expected_cells','commonsense_subtask_counts')},indent=2))


if __name__ == '__main__':
    main()
