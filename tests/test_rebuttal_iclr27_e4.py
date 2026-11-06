import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rebuttal_iclr27_e4 as e4


def test_exact_mcnemar_binomial_tail():
    assert e4.mcnemar_exact(12, 2) == pytest.approx(212/16384)
    assert e4.mcnemar_exact(2, 12) == e4.mcnemar_exact(12, 2)
    assert e4.mcnemar_exact(0, 0) == 1
    assert e4.mcnemar_exact(10, 10) == 1


def test_selection_uses_validation_and_predeclared_ties_only():
    candidates = [dict(validation_score=.5, multiplier=lr, fraction=fraction, test_score=1-lr/10)
                  for lr in (.5, 1., 2.) for fraction in (.5, 1.)]
    selected = e4.choose_candidate(candidates)
    assert (selected['multiplier'], selected['fraction']) == (1, .5)
    candidates[0]['validation_score'] = .6
    assert e4.choose_candidate(candidates) is candidates[0]


def test_item_bootstrap_preserves_pairing_and_commonsense_macro(tmp_path, monkeypatch):
    monkeypatch.setattr(e4, 'OUT', tmp_path)
    monkeypatch.setattr(e4, 'source_for', lambda condition: ('magicoder', tmp_path, 16))
    monkeypatch.setattr(e4.r, 'NBOOT', 1000)
    condition = 'magicoder_val_best_r16'
    variants = [dict(method=m, label=m) for m in e4.METHODS[1:]]
    (tmp_path / f'{condition}_variant_manifest.json').write_text(json.dumps(dict(variants=variants)))
    for family in e4.r.TASKS:
        for method in e4.METHODS:
            path = tmp_path / 'runtime/eval' / condition / family / method
            path.mkdir(parents=True)
            rows = []
            groups = [(f's{i}', 1 if i == 0 else 10) for i in range(8)] if family == 'commonsense' else [(family, 4)]
            for sub, count in groups:
                for i in range(count):
                    bit = family == 'commonsense' and sub == 's0' and method in ('recon_0', 'hns_f4_s1')
                    key = {'magicoder': 'correct', 'metamath': 'correct_strict', 'tulu': 'prompt_strict_passed', 'commonsense': 'correct'}[family]
                    rows.append(dict(id=f'{sub}/{i}', subtask=sub, **{key: bit}))
            (path / 'scored.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    rows, paired = e4.item_statistics(condition)
    recon = next(x for x in rows if x['method'] == 'recon_0')
    assert recon['commonsense'] == 12.5  # Equal tasks; not pooled1/71.
    assert recon['off_score'] == pytest.approx(12.5/3)
    equal = next(x for x in paired if x['left'] == 'hns_f4_s1' and x['right'] == 'recon_0' and x['benchmark'] == 'Target')
    assert (equal['difference_pp'], equal['ci_low_pp'], equal['ci_high_pp']) == (0, 0, 0)
    assert equal['mcnemar_exact_p'] == 1
    assert equal['repaired'] == equal['damaged'] == 0
