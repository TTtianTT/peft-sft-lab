#!/usr/bin/env python3
"""R2-O11: validation-selected global gamma on the fixed 18 main checkpoints.

Author-approved on 2026-11-09 after the C pipeline finished. Every candidate
adapter is gamma * Recon-0 on the shared fp32 compact SVD basis (balanced
writeback), so gamma=1.0 is exactly the audited E1 Recon-0 adapter. One gamma per
checkpoint is selected only from independent validation data; the main tests are
never used for selection, and every result is reported.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

import rebuttal_iclr27 as r
import rebuttal_iclr27_r2 as q
import rebuttal_iclr27_r2_a as a
from rebuttal_iclr27_e2_e3 import operations, save_adapter

EXP = 'R2-O11'
P = q.folder(EXP)
GAMMAS = (.3, .4, .5, .6, .7, .85, 1.)
DATA = P / 'validation_data'
RAW = P / 'runtime/data/raw'
SVAMP = dict(repo='ChilleD/SVAMP', revision='5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef',
             files=('svamp_train-00000-of-00001.parquet', 'svamp_test-00000-of-00001.parquet'))
IFLIKE = dict(repo='argilla/ifeval-like-data', config='filtered', revision='56505d1771e36fa5aa07aa9a1b39008fe60a8487',
              files=('ifeval_like_filtered.parquet',))
IFEVAL_TEST = Path('/dataset1/zailong/data/peft-sft-lab/cross-task-mechanism/ifeval-train.jsonl')
TULU_TRAIN = Path('/dataset1/zailong/data/peft-sft-lab/cross-task-mechanism/tulu-3-sft-personas-instruction-following-train.parquet')
MAGICODER_TRAIN = Path('/dataset1/zailong/data/peft-sft-lab/magicoder-train.parquet')
METAMATH_TRAIN = Path('/dataset1/zailong/data/peft-sft-lab/metamathqa-train.parquet')
VALIDATION_N = 500
METRIC = dict(magicoder='pass@1', metamath='accuracy_strict', tulu='prompt_level_strict_acc')
MAX_NEW = dict(magicoder=512, metamath=512, tulu=2048)
SLOT_BASE = {0: 'Qwen3-8B', 1: 'Llama-3.1-8B-Instruct'}
LOCKED = ('scripts/rebuttal_iclr27_r2_o11.py', 'scripts/rebuttal_iclr27_r2.py', 'scripts/rebuttal_iclr27_r2_a.py',
          'scripts/rebuttal_iclr27.py', 'scripts/rebuttal_iclr27_e2_e3.py', 'scripts/rebuttal_iclr27_e4.py',
          'src/finetune/eval/eval_mbpp.py', 'src/finetune/eval/eval_gsm8k.py', 'src/finetune/eval/eval_ifeval.py',
          'src/finetune/spectral_edit/svd.py', 'src/finetune/data/base.py')


def norm(text):
    return re.sub(r'\s+', ' ', str(text)).strip().lower()


def number(value):
    x = float(value)
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else format(x, 'g')


def prepare_data():
    """CPU-only: build the three fixed validation sets and contamination audits."""
    import pandas as pd
    DATA.mkdir(parents=True, exist_ok=True)
    audit = {}
    # MetaMath: SVAMP (ASDiv-A derived, not a GSM8K/MATH augmentation).
    svamp = pd.concat([pd.read_parquet(RAW / f) for f in SVAMP['files']], ignore_index=True)
    assert len(svamp) == 1000
    rows = []
    for x in svamp.to_dict('records'):
        body = str(x['Body']).strip()
        body = body if body.endswith(('.', '?', '!')) else body + '.'
        rows.append(dict(question=f'{body} {str(x["Question"]).strip()}', answer='#### ' + number(x['Answer']), svamp_id=x['ID']))
    queries = {norm(v) for v in pd.read_parquet(METAMATH_TRAIN, columns=['query'])['query']}
    clean = [x for x in rows if norm(x['question']) not in queries]
    order = np.random.default_rng(42).permutation(len(clean))[:VALIDATION_N]
    val = pd.DataFrame([clean[i] for i in order])
    val.to_parquet(DATA / 'metamath_svamp500.parquet', index=False)
    audit['metamath'] = dict(source=SVAMP, pool=len(rows), exact_overlap_with_full_metamathqa_query=len(rows) - len(clean),
                             selected=len(val), raw_sha256={f: r.sha(RAW / f) for f in SVAMP['files']})
    # Tulu: IFEval-format synthetic prompts, restricted to instruction ids of the main IFEval scorer set.
    test = [json.loads(line) for line in IFEVAL_TEST.read_text().splitlines() if line.strip()]
    from finetune.eval.eval_ifeval import check_instruction_strict
    # Keep ids that occur in the main IFEval set AND are implemented by the unchanged main scorer.
    allowed = {i for x in test for i in x['instruction_id_list']
               if 'unknown instruction id' not in str(check_instruction_strict(inst_id=i, resp='x', prompt='', kwargs={}).detail)}
    banned = {norm(x['prompt']) for x in test} | {norm(v) for v in pd.read_parquet(TULU_TRAIN, columns=['prompt'])['prompt']}
    pool = pd.read_parquet(RAW / IFLIKE['files'][0])
    eligible, reasons = [], dict(unparsed=0, unsupported_id=0, overlap=0, duplicate=0)
    seen = set()
    for x in pool.to_dict('records'):
        ids = re.findall(r"'([^']+)'", str(x['instruction_id_list']))
        try:
            kwargs = [{k: v for k, v in d.items() if v is not None} for d in json.loads(x['kwargs'])]
        except Exception:
            reasons['unparsed'] += 1; continue
        if not ids or len(ids) != len(kwargs):
            reasons['unparsed'] += 1; continue
        if not set(ids) <= allowed:
            reasons['unsupported_id'] += 1; continue
        key = norm(x['prompt'])
        if key in banned:
            reasons['overlap'] += 1; continue
        if key in seen:
            reasons['duplicate'] += 1; continue
        seen.add(key)
        eligible.append(dict(key=int(x['key']), prompt=x['prompt'], instruction_id_list=ids, kwargs=kwargs))
    order = np.random.default_rng(42).permutation(len(eligible))[:VALIDATION_N]
    with (DATA / 'tulu_ifevallike500.jsonl').open('w') as h:
        for i in order:
            h.write(json.dumps(eligible[i], ensure_ascii=False) + '\n')
    audit['tulu'] = dict(source=IFLIKE, pool=len(pool), eligible=len(eligible), excluded=reasons, selected=len(order),
                         allowed_instruction_ids=sorted(allowed), raw_sha256=r.sha(RAW / IFLIKE['files'][0]),
                         excluded_against=[str(IFEVAL_TEST), str(TULU_TRAIN)])
    # Magicoder: the fixed E4 MBPP200 validation set, audited against the full local Magicoder table.
    e4data = r.read(r.OUT / 'E4/protocol.lock.json')['datasets']['magicoder']
    mbpp = pd.read_parquet(e4data['validation_path'])
    instructions = {norm(v) for v in pd.read_parquet(MAGICODER_TRAIN, columns=['instruction'])['instruction']}
    audit['magicoder'] = dict(source=e4data['validation_path'], sha256=e4data['validation_sha256'], selected=len(mbpp),
        exact_overlap_with_full_magicoder_instruction=int(sum(norm(v) in instructions for v in mbpp['prompt'])))
    audit['limitations'] = ('Exact-text audits only use local training tables. The seed42 Qwen sources were trained elsewhere from '
        'MetaMathQA-395K json / tulu-3-sft parquet copies, so their exact rows cannot be re-derived; SVAMP and argilla '
        'ifeval-like are external sources not built from GSM8K/MATH or Tulu personas. MBPP200 comes from MBPP sanitized test.')
    r.write(P / 'validation_data_audit.json', audit)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('allowed_instruction_ids', 'source')}
                      for k, v in audit.items() if isinstance(v, dict)}, indent=1))


def data_paths():
    return dict(magicoder=Path(r.read(r.OUT / 'E4/protocol.lock.json')['datasets']['magicoder']['validation_path']),
                metamath=DATA / 'metamath_svamp500.parquet', tulu=DATA / 'tulu_ifevallike500.jsonl')


def recon_paths():
    out = {}
    for base in SLOT_BASE.values():
        for x in r.read(r.OUT / f'E1/{base}_build_summary.json'):
            if x['method'] == 'recon_0':
                out[base, x['task'], int(x['seed'])] = x
    return out


def cohort():
    cps = a.cohort(); recon = recon_paths()
    for cp in cps:
        x = recon[cp['base'], cp['task'], cp['seed']]
        assert x['source_sha256'] == cp['source_sha256']
        cp['recon_path'] = x['path']; cp['recon_sha256'] = x['weights_sha256']
    assert len(cps) == 18
    return cps


def lock_validation():
    if (P / 'validation.lock.json').exists():
        return check_validation()
    q.initialize_round()
    approval = dict(approved=True, by='paper author, in Claude Code session', approved_local_date='2026-11-09',
                    recorded_utc=r.now(), note='Explicit instruction to run O11 after O10/O12 finished.')
    plan = r.read(r.OUT / 'R2-plan.json'); plan['o11_author_approval'] = approval; r.write(r.OUT / 'R2-plan.json', plan)
    cps = cohort(); paths = data_paths()
    text = f'''# R2-O11: validation-selected global gamma (locked before any O11 inference)

Author approval: {approval['approved_local_date']} ({approval['by']}). Frozen UTC {r.now()}.

Question (reviewer 8Euc, "appropriate scaling"): can one global scale of the update, chosen on independent validation data,
match the data-free nuclear-preserving flattening? This differs from E3 Global-gamma (data-free mean Frobenius ratio).

Cohort: the fixed 18 main checkpoints (Qwen3-8B / Llama-3.1-8B-Instruct x magicoder/metamath/tulu x seed42/43/44). No retraining.
Candidates: gamma in {list(GAMMAS)}. Candidate = gamma * Recon-0: shared fp32 compact SVD of BA, singular values multiplied by
gamma, balanced writeback B=U sqrt(gamma sigma), A=sqrt(gamma sigma) Vh, cast to source dtype, native rank/alpha/config preserved.
gamma=1.0 is the audited E1 Recon-0 adapter itself (weights SHA256 checked). Audit: relative reconstruction error, error to
gamma*source update and nuclear ratio to gamma*||sigma||_1 all below the fixed 5e-5, every module.

Validation (target task only; one score per checkpoint x gamma, 126 runs):
- magicoder: E4 MBPP200 (sanitized MBPP test, seed42 subset), unchanged eval_mbpp.py pass@1, max_new_tokens512.
- metamath: SVAMP{VALIDATION_N} ({SVAMP['repo']}@{SVAMP['revision'][:12]}, train+test pool, seed42 subset), formatted as
  GSM8K-style '#### answer'; unchanged eval_gsm8k.py strict accuracy, max_new_tokens512. The E4 MetaMath held-out set is NOT used:
  it was drawn from the same 50K MetaMathQA subset the old sources trained on.
- tulu: {VALIDATION_N} prompts from {IFLIKE['repo']} (config {IFLIKE['config']}@{IFLIKE['revision'][:12]}), only instruction ids present in the main IFEval set and implemented by the
  unchanged main scorer, exact prompts of IFEval541 and of the local Tulu personas IF training table removed; unchanged eval_ifeval.py
  prompt-level strict accuracy (--fail_on_unknown), max_new_tokens2048.
All: non_thinking chat template as E4/O9 validation, vLLM TP1, max_model_len4096, FLASH_ATTN, flashinfer sampler off, seed42,
batch-invariant environment, inside Slurm, at most 2 GPUs (one worker per base model).

Selection rule (fixed now): highest validation score; ties broken by larger gamma (smaller intervention, closer to Recon-0).
Never use HumanEval/GSM8K/IFEval/Commonsense test scores. All 126 validation scores are reported.

Test: selected adapter = method gamma_val on the four unchanged main families. If gamma=1.0 is selected, gamma_val is the
E1 Recon-0 adapter and its audited cached cells are reused (no new inference). Compare with original_lora, recon_0 (E1 cache)
and hns_f4_s1 (main cache). Statistics: paired checkpoint bootstrap 20000, seed{r.BOOT_SEED}, descriptive on shared tests,
wins/ties/losses; Target/Off/FG definitions unchanged. All outcomes are reported, including any unfavourable to HNS.
Validation budget is recorded separately (126 validation runs; HNS uses none).
'''
    (P / 'validation_protocol.md').write_text(text)
    obj = dict(created_utc=r.now(), approval=approval, gammas=list(GAMMAS), selection='max validation; tie -> larger gamma',
        files={n: r.sha(r.ROOT / n) for n in LOCKED}, protocol_sha256=r.sha(P / 'validation_protocol.md'),
        data={t: dict(path=str(v), sha256=r.sha(v)) for t, v in paths.items()},
        data_audit_sha256=r.sha(P / 'validation_data_audit.json'), metrics=METRIC, max_new_tokens=MAX_NEW,
        checkpoints=[{k: cp[k] for k in ('condition', 'base', 'base_model', 'task', 'seed', 'rank', 'source', 'source_sha256',
                                          'config_sha256', 'recon_path', 'recon_sha256')} for cp in cps])
    r.write(P / 'validation.lock.json', obj)
    return obj


def check_validation():
    lock = r.read(P / 'validation.lock.json')
    assert r.sha(P / 'validation_protocol.md') == lock['protocol_sha256']
    for name, digest in lock['files'].items():
        assert r.sha(r.ROOT / name) == digest, ('Frozen O11 dependency changed', name)
    for obj in lock['data'].values():
        assert r.sha(obj['path']) == obj['sha256']
    return lock


def candidate_dir(cp, gamma):
    return Path(cp['recon_path']) if gamma == 1. else P / 'runtime/adapters' / cp['condition'] / f'gamma{gamma:g}'


def build(slot):
    r.require_slurm(); q.env_setup(EXP, f'build{slot}'); os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    lock = check_validation(); torch, collect, error, load, svd, rebuild, _, _ = operations()
    artifacts, modules = [], []
    for cp in [c for c in lock['checkpoints'] if c['base'] == SLOT_BASE[slot]]:
        src = Path(cp['source'])
        assert r.sha(src / 'adapter_model.safetensors') == cp['source_sha256']
        assert r.sha(Path(cp['recon_path']) / 'adapter_model.safetensors') == cp['recon_sha256']
        state, fmt = load(str(src)); pairs = collect(state); decomp = {}
        with torch.inference_mode():
            for name, pair in sorted(pairs.items()):
                decomp[name] = svd(pair['B'][1].cuda(), pair['A'][1].cuda())[:3]
        for gamma in GAMMAS[:-1]:
            dest = candidate_dir(cp, gamma); marker = dest / 'o11_integrity.json'
            if marker.exists():
                info = r.read(marker); assert r.sha(dest / 'adapter_model.safetensors') == info['artifact']['weights_sha256']
                artifacts.append(info['artifact']); modules.extend(info['modules']); continue
            started = time.monotonic(); output = dict(state); rows = []
            with torch.inference_mode():
                for name, pair in sorted(pairs.items()):
                    ka, A = pair['A']; kb, B = pair['B']; u, s, vh = decomp[name]
                    assert len(s) == cp['rank']
                    t = s * gamma; bt, at = q.writeback(u, vh, t, 'recon_0')
                    bn, an = bt.to(B.dtype), at.to(A.dtype); output[kb], output[ka] = bn.cpu(), an.cpu()
                    relative = error(bn.double(), an.double(), bt.double(), at.double())
                    source_error = error(bn.double(), an.double(), gamma * B.cuda().double(), A.cuda().double())
                    _, ss, _, _ = svd(bn, an)
                    nuclear_error = abs(float(ss.double().sum() / (gamma * s.double().sum())) - 1)
                    assert torch.isfinite(t).all() and max(relative, nuclear_error, source_error) < 5e-5, (name, gamma, relative, nuclear_error, source_error)
                    rows.append(dict(condition=cp['condition'], base=cp['base'], task=cp['task'], seed=cp['seed'], method=f'gamma{gamma:g}',
                        module=name, rank=len(s), dtype_A=str(an.dtype), dtype_B=str(bn.dtype), gamma=gamma,
                        reconstruction_relative_error=relative, source_relative_error=source_error, nuclear_relative_error=nuclear_error,
                        **q.spectral_statistics(s, t)))
            save_adapter(src, dest, output, fmt)
            saved, _ = load(str(dest))
            assert saved.keys() == output.keys() and all(torch.equal(saved[k], v) and saved[k].dtype == v.dtype for k, v in output.items())
            assert r.sha(dest / 'adapter_config.json') == cp['config_sha256']
            artifact = dict(condition=cp['condition'], base=cp['base'], train_task=cp['task'], seed=cp['seed'], rank=cp['rank'],
                gamma=gamma, path=str(dest), source_sha256=cp['source_sha256'], weights_sha256=r.sha(dest / 'adapter_model.safetensors'),
                config_sha256=cp['config_sha256'], modules=len(rows), seconds=time.monotonic() - started, job_id=os.getenv('SLURM_JOB_ID'))
            r.write(marker, dict(artifact=artifact, modules=rows)); artifacts.append(artifact); modules.extend(rows)
        del decomp, state; torch.cuda.empty_cache(); print('[Built]', cp['condition'], flush=True)
    r.write(P / f'candidates{slot}_build.json', dict(artifacts=artifacts))
    r.tsv(P / f'candidates{slot}_module_audit.tsv', modules)


def validate(slot):
    r.require_slurm(); q.env_setup(EXP, f'val{slot}'); os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    lock = check_validation(); paths = {t: Path(v['path']) for t, v in lock['data'].items()}
    r.run_command([sys.executable, __file__, '--build', '--slot', str(slot)], P / f'runtime/logs/candidates{slot}_build.log',
                  P / f'validation_commands_{slot}.json')
    built = {(x['condition'], x['gamma']): x for x in r.read(P / f'candidates{slot}_build.json')['artifacts']}
    rows = []
    for cp in [c for c in lock['checkpoints'] if c['base'] == SLOT_BASE[slot]]:
        for gamma in GAMMAS:
            adapter = candidate_dir(cp, gamma)
            weights = r.sha(adapter / 'adapter_model.safetensors')
            assert weights == (cp['recon_sha256'] if gamma == 1. else built[cp['condition'], gamma]['weights_sha256'])
            dest = P / 'runtime/validation' / cp['condition'] / f'gamma{gamma:g}'
            task = cp['task']
            if not (dest / 'metrics.json').exists():
                module = dict(magicoder='finetune.eval.eval_mbpp', metamath='finetune.eval.eval_gsm8k', tulu='finetune.eval.eval_ifeval')[task]
                cmd = [sys.executable, '-m', module, '--base_model', cp['base_model'], '--adapter_dir', str(adapter),
                       '--output_dir', str(dest), '--dataset_path', str(paths[task]), '--chat_template_mode', 'non_thinking',
                       '--use_vllm', '--tensor_parallel_size', '1', '--vllm_max_model_len', '4096', '--vllm_attention_backend', 'FLASH_ATTN',
                       '--vllm_disable_flashinfer_sampler', '--max_new_tokens', str(MAX_NEW[task]), '--seed', '42']
                if task == 'magicoder': cmd += ['--eval_n_workers', '16']
                if task == 'tulu': cmd += ['--fail_on_unknown']
                r.run_command(cmd, P / f'runtime/logs/{cp["condition"]}_gamma{gamma:g}_validation.log', P / f'validation_commands_{slot}.json')
            metric = r.read(dest / 'metrics.json')
            total = metric.get('total', metric.get('total_prompts'))
            assert total == (200 if task == 'magicoder' else VALIDATION_N), (cp['condition'], gamma, total)
            rows.append(dict(condition=cp['condition'], base=cp['base'], task=task, seed=cp['seed'], gamma=gamma,
                             validation_score=float(metric[METRIC[task]]), metric=METRIC[task], n=total, adapter=str(adapter),
                             weights_sha256=weights, metrics_sha256=r.sha(dest / 'metrics.json')))
            print('[Validated]', cp['condition'], gamma, rows[-1]['validation_score'], flush=True)
    r.write(P / f'validation_worker{slot}.json', rows)


def parallel(flag):
    devices = os.getenv('CUDA_VISIBLE_DEVICES', '').split(','); assert len(devices) == 2
    states = []
    for slot, device in enumerate(devices):
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = device; env.pop('CUBLAS_WORKSPACE_CONFIG', None)
        log = P / f'runtime/logs/{flag.strip("-")}{slot}.log'; log.parent.mkdir(parents=True, exist_ok=True); h = log.open('a')
        states.append((slot, subprocess.Popen([sys.executable, __file__, flag, '--slot', str(slot)], cwd=r.ROOT, env=env,
                                              stdout=h, stderr=subprocess.STDOUT), h))
    failed = []
    for slot, proc, h in states:
        code = proc.wait(); h.close()
        if code: failed.append((slot, code))
    if failed: raise RuntimeError((EXP, flag, failed))


def select():
    lock = check_validation()
    rows = sum([r.read(P / f'validation_worker{i}.json') for i in (0, 1)], [])
    assert len(rows) == 18 * len(GAMMAS)
    r.tsv(P / 'validation_candidates.tsv', rows)
    chosen = []
    for cp in lock['checkpoints']:
        mine = [x for x in rows if x['condition'] == cp['condition']]
        best = min(mine, key=lambda x: (-x['validation_score'], -x['gamma']))
        chosen.append(best)
    r.write(P / 'selection.json', dict(rule='max validation score; tie -> larger gamma', selected=chosen, created_utc=r.now()))
    return chosen


def freeze_test():
    if (P / 'protocol.lock.json').exists():
        return q.check(EXP)
    vlock = check_validation(); chosen = {x['condition']: x for x in r.read(P / 'selection.json')['selected']}
    cps = cohort(); new = 0
    for cp in cps:
        pick = chosen[cp['condition']]; cp['gamma'] = pick['gamma']; cp['gamma_path'] = pick['adapter']
        cp['gamma_weights_sha256'] = pick['weights_sha256']
        if pick['gamma'] == 1.:
            cp['cache']['gamma_val'] = q.first_wave_cells('E1', cp, 'recon_0')
        else:
            new += 4
        cp.pop('recon_path', None); cp.pop('recon_sha256', None)
    detail = (f'O11 test phase. Validation design, candidates, data and selection rule were locked before any O11 inference in '
              f'validation_protocol.md (sha256 {vlock["protocol_sha256"]}); selection.json records the chosen gamma per checkpoint '
              f'from validation only. Methods: original_lora/hns_f4_s1 (main cache), recon_0 (E1 cache), gamma_val (selected gamma*Recon-0; '
              f'gamma=1 reuses Recon-0 cells). {new} new test cells.')
    return q.freeze(EXP, cps, ('original_lora', 'recon_0', 'hns_f4_s1', 'gamma_val'), detail, new,
                    extra_files=('scripts/rebuttal_iclr27_r2_o11.py', 'reports/rebuttal_iclr27/R2-O11/validation.lock.json',
                                 'reports/rebuttal_iclr27/R2-O11/selection.json'))


def eval_worker(slot):
    lock = q.check(EXP); modules = r.load_tsv(P / f'candidates{slot}_module_audit.tsv')
    variants, artifacts = [], []
    for cp in [c for c in lock['checkpoints'] if c['base'] == SLOT_BASE[slot] and 'gamma_val' in c['new_methods']]:
        path = Path(cp['gamma_path']); assert r.sha(path / 'adapter_model.safetensors') == cp['gamma_weights_sha256']
        v = dict(label=q.label(cp, 'gamma_val'), train_task=cp['task'], seed=cp['seed'], method='gamma_val', condition=cp['condition'],
                 base=cp['base'], rank=cp['rank'], path=str(path))
        variants.append(v); artifacts.append(dict(**v, gamma=cp['gamma'], weights_sha256=cp['gamma_weights_sha256'], config_sha256=cp['config_sha256']))
    r.write(P / f'worker{slot}_build.json', dict(variants=variants, artifacts=artifacts))
    r.tsv(P / f'worker{slot}_module_audit.tsv', modules)
    original = r.run_command
    def skip_build(argv, log, history):
        if '--build' in argv:
            return None  # candidates were built and audited in the validation phase
        return original(argv, log, history)
    r.run_command = skip_build
    q.worker(EXP, slot)


def extra(exp, lock, rows, stats, modules, lines):
    sel = r.read(P / 'selection.json')['selected']
    lines += ['', 'Selected gamma per checkpoint (validation only; tie -> larger gamma). All 126 validation scores: validation_candidates.tsv.', '',
              '| Checkpoint | Selected gamma | Validation score |', '| --- | ---: | ---: |']
    lines += [f'| {x["condition"]} | {x["gamma"]:g} | {x["validation_score"]:.4f} |' for x in sel]
    lines += ['', 'gamma_val is gamma*Recon-0 on the shared SVD basis; gamma=1 means the Recon-0 adapter and its cached cells. '
              'Validation sets: MBPP200 (Magicoder), SVAMP500 (MetaMath), argilla ifeval-like 500 (Tulu); see validation_protocol.md and '
              'validation_data_audit.json. This is a validation-tuned scaling control and differs from the data-free E3 Global-gamma. '
              'HNS uses no validation data; gamma_val uses 7 validation runs per checkpoint.']


def pipeline():
    r.require_slurm()
    lock_validation()
    if not all((P / f'validation_worker{i}.json').exists() for i in (0, 1)):
        parallel('--validate')
    if not (P / 'selection.json').exists():
        select()
    freeze_test()
    q.prime_cache(EXP)
    parallel('--eval-worker')
    q.finalize(EXP, extra)
    q.publish(EXP)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('--prepare-data', '--lock', '--build', '--validate', '--eval-worker', '--pipeline'):
        p.add_argument(flag, action='store_true')
    p.add_argument('--slot', type=int, choices=(0, 1), default=0)
    x = p.parse_args()
    if x.prepare_data: prepare_data()
    elif x.lock: print(json.dumps({k: v for k, v in lock_validation().items() if k != 'checkpoints'}, indent=1)[:3000])
    elif x.build: build(x.slot)
    elif x.validate: validate(x.slot)
    elif x.eval_worker: eval_worker(x.slot)
    elif x.pipeline: pipeline()
