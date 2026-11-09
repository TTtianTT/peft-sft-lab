#!/usr/bin/env python3
"""AUXILIARY, read-only CPU diagnostic of R2-O7 GSM8K outputs (Qwen3-1.7B / Qwen3-4B).

It does not modify any prediction, scored file, metric or the main strict scorer. The main
O7 numbers remain the protocol GSM8K strict scores. This script only re-reads the saved
scored.jsonl files and reports:
  * the main strict score recomputed with the unchanged main extractor (must equal metrics.json);
  * output-format statistics (literal '<answer>' placeholder copied after '####', missing '####',
    finish_reason=length truncation, thinking tags);
  * an AUXILIARY placeholder-tolerant parse that only removes a literal '<answer>'/'</answer>'
    tag after the final '####' before applying the same first-line + _norm rule.
The auxiliary parse is a diagnostic of answer formatting, not a replacement metric.
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from finetune.eval.eval_gsm8k import _extract_answer, _norm  # noqa: E402  (main extractor, unchanged)

O7 = ROOT / 'reports/rebuttal_iclr27/R2-O7'
OUT = O7 / 'aux_gsm8k_format_diagnostic'
TAG = re.compile(r'</?answer>', re.IGNORECASE)


def aux_extract(text):
    if '####' not in text:
        return _extract_answer(text)
    tail = TAG.sub('', text.split('####')[-1]).strip()
    return tail.splitlines()[0].strip() if tail else ''


def numeric_equal(a, b):
    try:
        return abs(float(_norm(a).lstrip('$').rstrip('.')) - float(_norm(b))) < 1e-9
    except ValueError:
        return False


def context_8b():
    cells = [x for x in csv.DictReader(open(O7.parent / 'R2-O11/evaluation_cells.tsv'), delimiter='\t') if x['eval_task'] == 'metamath']
    agg = {}
    for cell in cells:
        scored = Path(cell['metrics_path']).parent / 'scored.jsonl'
        assert hashlib.sha256(scored.read_bytes()).hexdigest() == cell['scored_sha256'], scored
        a = agg.setdefault((cell['base'], cell['method']), [0, 0, 0, 0])
        for x in map(json.loads, scored.open()):
            t, g = x['prediction_text'], x['gold']
            a[0] += 1
            a[1] += _norm(_extract_answer(t)) == _norm(g)
            a[2] += _norm(aux_extract(t)) == _norm(g) or numeric_equal(aux_extract(t), g)
            a[3] += '####' in t and bool(TAG.search(t.split('####')[-1]))
    return [dict(base=b, method=m, items=a[0], main_strict_pct=100 * a[1] / a[0], aux_placeholder_tolerant_pct=100 * a[2] / a[0],
                 literal_answer_tag_after_marker_pct=100 * a[3] / a[0]) for (b, m), a in sorted(agg.items())]


def main():
    cells = [x for x in csv.DictReader(open(O7 / 'evaluation_cells.tsv'), delimiter='\t') if x['eval_task'] == 'metamath']
    rows, per_item = [], {}
    for cell in cells:
        folder = Path(cell['metrics_path']).parent
        scored = folder / 'scored.jsonl'
        assert hashlib.sha256(scored.read_bytes()).hexdigest() == cell['scored_sha256'], scored
        items = [json.loads(line) for line in scored.open()]
        metrics = json.load(open(cell['metrics_path']))
        n = len(items)
        strict = sum(_norm(_extract_answer(x['prediction_text'])) == _norm(x['gold']) for x in items)
        assert strict == sum(bool(x['correct_strict']) for x in items), folder
        assert abs(strict / n - metrics['strict_accuracy']) < 1e-12, folder
        aux = [_norm(aux_extract(x['prediction_text'])) == _norm(x['gold']) or numeric_equal(aux_extract(x['prediction_text']), x['gold']) for x in items]
        tails = [x['prediction_text'].split('####')[-1] for x in items if '####' in x['prediction_text']]
        key = (cell['base'], cell['method'])
        per_item[key] = aux
        rows.append(dict(
            base=cell['base'], method=cell['method'], samples=n,
            main_strict_pct=100 * strict / n,
            aux_placeholder_tolerant_pct=100 * sum(aux) / n,
            aux_minus_main_pp=100 * (sum(aux) - strict) / n,
            has_hash_marker_pct=100 * len(tails) / n,
            literal_answer_tag_after_marker_pct=100 * sum(bool(TAG.search(t)) for t in tails) / n,
            finish_length_pct=100 * sum(x.get('finish_reason') == 'length' for x in items) / n,
            think_tag_pct=100 * sum('<think>' in x['prediction_text'] for x in items) / n,
            mean_output_tokens=sum(len(x['token_ids']) for x in items) / n,
            scored_sha256=cell['scored_sha256']))
    OUT.mkdir(exist_ok=True)
    with open(OUT / 'gsm8k_format_diagnostic.tsv', 'w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    lines = ['# R2-O7 GSM8K output-format diagnostic (AUXILIARY)', '',
             'Read-only re-analysis of saved O7 GSM8K outputs. The protocol metric is unchanged: main O7 Target remains '
             'GSM8K strict (first line after the final `####`, `_norm`, exact match). The prompt asks for the final line '
             '`#### <answer>`; the auxiliary column only strips a literal `<answer>`/`</answer>` tag after the final '
             '`####` and then applies the same rule (plus numeric equality). It is a formatting diagnostic, '
             'not a replacement score, and was defined after seeing the outputs.', '',
             '| Base | Method | Main strict | Aux tag-tolerant | Aux - main | `####` present | literal `<answer>` after `####` | finish=length | mean tokens |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for x in rows:
        lines.append(f"| {x['base']} | {x['method']} | {x['main_strict_pct']:.2f} | {x['aux_placeholder_tolerant_pct']:.2f} | "
                     f"{x['aux_minus_main_pp']:+.2f} | {x['has_hash_marker_pct']:.1f} | {x['literal_answer_tag_after_marker_pct']:.1f} | "
                     f"{x['finish_length_pct']:.2f} | {x['mean_output_tokens']:.0f} |")
    lines += ['', 'Recomputed main strict equals each metrics.json exactly; scored.jsonl SHA256 matches evaluation_cells.tsv.']
    ctx = context_8b()
    with open(OUT / 'gsm8k_format_context_8b.tsv', 'w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(ctx[0]), delimiter='\t')
        w.writeheader()
        w.writerows(ctx)
    lines += ['', '## Context: same diagnostic on the main 8B GSM8K cells', '',
              'Pooled over the 9 main checkpoints per base (3 tasks x 3 seeds; the R2-O11 cell set, which links the E1 cached cells). '
              'Item-pooled descriptive rates only; use the protocol paired CIs for inference. At 8B the literal placeholder is rare; '
              'the remaining strict-vs-auxiliary gap is mostly currency/decimal formatting such as `$18` or `18.00`, which the '
              'auxiliary numeric-equality rule accepts. HNS remains above Recon-0 and original LoRA under both parses.', '',
              '| Base | Method | Items | Main strict | Aux tag-tolerant | literal `<answer>` after `####` |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for x in ctx:
        lines.append(f"| {x['base']} | {x['method']} | {x['items']} | {x['main_strict_pct']:.2f} | {x['aux_placeholder_tolerant_pct']:.2f} | {x['literal_answer_tag_after_marker_pct']:.1f} |")
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
