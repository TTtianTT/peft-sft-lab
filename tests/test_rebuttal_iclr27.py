"""Guard the nonlinear forgetting metric, pairing, and input-identity audit."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'rebuttal_iclr27', Path(__file__).resolve().parents[1] / 'scripts/rebuttal_iclr27.py')
reb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reb)


def test_forgetting_clips_each_off_family_before_averaging():
    cells = [dict(base='Qwen3-8B', method='base', eval_task=t, score=50.) for t in reb.TASKS]
    scores = dict(magicoder=70., metamath=40., tulu=60., commonsense=50.)
    cells += [dict(base='Qwen3-8B', train_task='magicoder', seed=42, method='edit',
                   eval_task=t, score=scores[t]) for t in reb.TASKS]
    row, = reb.checkpoint_results(cells)
    assert row['target'] == 70
    assert row['off_score'] == 50
    assert row['forgetting_gap'] == pytest.approx(10 / 3)


def test_bootstrap_pairs_checkpoint_ids_and_counts_lower_fg_as_win():
    rows = []
    for b in reb.BASES:
        for t in reb.TASKS[:3]:
            for s in (42, 43, 44):
                for method, gain in [('lora', 0), ('edit', 1)]:
                    rows.append(dict(base=b, task=t, seed=s, checkpoint=f'{b}/{t}/{s}',
                        method=method, target=s + gain, off_score=s + 2 * gain,
                        forgetting_gap=10 - gain))
    stats = reb.paired(list(reversed(rows)))
    r = next(r for r in stats if r['group'] == 'overall' and r['left'] == 'edit'
             and r['right'] == 'lora' and r['metric'] == 'forgetting_gap')
    assert (r['difference_pp'], r['ci_low_pp'], r['ci_high_pp']) == (-1, -1, -1)
    assert (r['wins'], r['ties'], r['losses']) == (18, 0, 0)
    assert r['bootstrap_resamples'] == 20000


def test_input_hash_ignores_outputs_but_detects_gold_changes(tmp_path):
    import json
    path = tmp_path / 'predictions.jsonl'
    row = dict(id='1', question='q', gold='42', prediction_text='42', token_ids=[1], finish_reason='stop')
    path.write_text(json.dumps(row) + '\n')
    first = reb.input_hashes(tmp_path)
    row.update(prediction_text='43', token_ids=[2])
    path.write_text(json.dumps(row) + '\n')
    assert reb.input_hashes(tmp_path) == first
    row['gold'] = '43'
    path.write_text(json.dumps(row) + '\n')
    assert reb.input_hashes(tmp_path) != first
