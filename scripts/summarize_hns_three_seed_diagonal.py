#!/usr/bin/env python3
"""Export original and replication diagonal scores without pooling different recipes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912')
SETTINGS = ['LoRA', '0+0', '2+0', '2+1', '2+2', '4+0', '4+1', '4+2', '8+0', '8+1', '8+2']
BASES = [('Qwen3-8B', 'qwen3_8b'), ('Llama-3.1-8B-Instruct', 'llama31_8b')]
TASKS = [('magicoder', 'HumanEval'), ('metamath', 'GSM8K'), ('tulu', 'IFEval'), ('commonsense', 'Commonsense-8')]


def collect():
    rows = []
    for base, slug in BASES:
        manifests = {42: ROOT / f'reports/hns_release_20260912/data/step_grid/manifests/{slug}_score_manifest.json'}
        manifests.update({seed: RUN / base / f'seed{seed}/hns_step_grid/eval/score_manifest.json' for seed in (43, 44)})
        scores = {}
        for seed, path in manifests.items():
            records = json.loads(path.read_text())['records']
            scores[seed] = {(r['task'], r['variant']): r[r['primary_metric']] * 100 for r in records}
        for task, benchmark in TASKS:
            for seed in (42, 43, 44):
                row = {'model': base, 'task': task, 'benchmark': benchmark, 'seed': seed}
                for setting in SETTINGS:
                    if setting == 'LoRA':
                        label = f'{task}__original_lora'
                    else:
                        fast, stable = setting.split('+')
                        label = f'{task}__hns_f{fast}_s{stable}'
                    value = scores[seed].get((task, label))
                    if task != 'commonsense' or seed == 42:
                        assert value is not None, (base, task, seed, label)
                    row[setting] = value
                rows.append(row)
    return rows


if __name__ == '__main__':
    rows = collect()
    columns = ['model', 'task', 'benchmark', 'seed', *SETTINGS]
    print('| 模型 | 本任务基准 | Training seed | ' + ' | '.join(SETTINGS) + ' |')
    print('|---|---|---|' + '---:|' * len(SETTINGS))
    for row in rows:
        label = str(row['seed']) + ('（原始）' if row['seed'] == 42 else '')
        cells = [row['model'], row['benchmark'], label] + [f'{row[s]:.2f}' if row[s] is not None else '—' for s in SETTINGS]
        print('| ' + ' | '.join(cells) + ' |')
