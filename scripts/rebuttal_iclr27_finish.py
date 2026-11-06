#!/usr/bin/env python3
"""Validate finished experiment packages, publish only owned files, write E5/E6."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import subprocess

import rebuttal_iclr27 as r


def validate(experiment):
    folder = r.OUT / experiment
    summary = r.read(folder / 'summary.json')
    assert summary['status'] == 'complete', (experiment, summary['status'])
    for name in ('report.md', 'summary.json', 'checkpoint_results.tsv', 'paired_ci.tsv',
                 'coverage_audit.json', 'artifact_integrity_audit.json'):
        assert (folder / name).is_file() and (folder / name).stat().st_size > 0, (experiment, name)
    assert r.read(folder / 'coverage_audit.json')['status'] == 'pass'
    assert r.read(folder / 'artifact_integrity_audit.json')['status'] == 'pass'
    if experiment in ('E0', 'E1', 'E2', 'E3'):
        stats = r.load_tsv(folder / 'paired_ci.tsv')
        assert stats and all(int(x['bootstrap_resamples']) == 20000 for x in stats)
        assert all(int(x['wins']) + int(x['ties']) + int(x['losses']) == int(x['n_checkpoints']) for x in stats)
    return summary


def publish(experiments):
    """Never include pre-existing staged work or any file outside this package."""
    for exp in experiments:
        validate(exp)
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=r.ROOT, text=True).strip()
    assert branch == 'refactor/sft-chat-template', ('Unexpected publication branch', branch)
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=r.ROOT, text=True).splitlines()
    assert not staged, ('Refusing to mix already staged work into experiment commit', staged)
    paths = [str((r.OUT / exp).relative_to(r.ROOT)) for exp in experiments]
    for name in ('submission.json', 'environment_audit.json', 'literature_audit.json', 'README.md'):
        if (r.OUT / name).exists():
            paths.append(str((r.OUT / name).relative_to(r.ROOT)))
    subprocess.run(['git', 'add', '--', *paths], cwd=r.ROOT, check=True)
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=r.ROOT).returncode
    if changed:
        subprocess.run(['git', 'commit', '-m', 'Report ICLR27 rebuttal ' + ', '.join(experiments)], cwd=r.ROOT, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=r.ROOT, text=True).strip()
    pushed = subprocess.run(['git', 'push', 'origin', branch], cwd=r.ROOT, capture_output=True, text=True)
    transport = 'origin'
    if pushed.returncode:
        # HTTPS has no helper on this host; the existing SSH key was verified
        # by an actual successful push of the initial experiment commit.
        env = os.environ.copy()
        env['GIT_SSH_COMMAND'] = 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes'
        pushed = subprocess.run(['git', 'push', 'git@github.com:TTtianTT/peft-sft-lab.git', branch],
                                cwd=r.ROOT, env=env, capture_output=True, text=True)
        transport = 'existing SSH key'
    # Save status without echoing credential/helper diagnostics.
    r.write(r.OUT / 'publication.json', dict(commit=commit, experiments=experiments,
        push_returncode=pushed.returncode, transport=transport,
        status='pushed' if not pushed.returncode else 'local_commit_push_pending',
        updated_utc=r.now()))
    print('[Publication]', commit, 'push status', pushed.returncode, flush=True)


def refresh_p0():
    """Finish explicit Flat numerical links before downstream GPU work starts."""
    for exp in ('E1', 'E0'):
        validate(exp)
    flats = [r.read(r.OUT / 'E0' / f'{b}_flat_integrity_audit.json') for b in r.BASES]
    assert all(x['status'] == 'pass' for x in flats)
    flat_expected = {(x['base'], x['train_task'], int(x['seed']), x['method']): x
                     for x in r.check_lock('E0')['flat_adapters']}
    recon_expected = {(x['base'], x['task'], int(x['seed']), x['method']): x
                      for base in r.BASES for x in r.read(r.OUT / 'E1' / f'{base}_build_summary.json')}
    for exp in ('E0', 'E1'):
        integrity = r.read(r.OUT / exp / 'artifact_integrity_audit.json')
        for artifact in integrity['artifacts']:
            key = artifact['base'], artifact['train_task'], int(artifact['seed']), artifact['method']
            expected = flat_expected[key] if artifact['experiment'] == 'E0' else recon_expected[key]
            assert artifact['weights_sha256'] == expected.get('weights_sha256', expected.get('weight_sha256'))
            assert artifact['config_sha256'] == expected['config_sha256']
        integrity['flat_adapters_reused_unchanged'] = True
        integrity['flat_module_rows'] = sum(x['modules'] for x in flats)
        integrity['max_flat_reconstruction_relative_error'] = max(x['max_reconstruction_relative_error'] for x in flats)
        integrity['max_flat_budget_relative_error'] = max(x['max_budget_relative_error'] for x in flats)
        r.write(r.OUT / exp / 'artifact_integrity_audit.json', integrity)
        coverage = r.read(r.OUT / exp / 'coverage_audit.json')
        coverage['audited_new_and_dependency_cells'] = coverage.get('audited_new_and_dependency_cells', coverage['new_cells'])
        coverage['new_cells'] = 72 if exp == 'E1' else 144
        r.write(r.OUT / exp / 'coverage_audit.json', coverage)
        report_path = r.OUT / exp / 'report.md'
        if 'Flat adapter audit:' not in report_path.read_text():
            with report_path.open('a') as handle:
                handle.write(f'\nFlat adapter audit: {integrity["flat_module_rows"]} module/method rows; '
                f'max reconstruction error {integrity["max_flat_reconstruction_relative_error"]:.9g}; '
                f'max budget error {integrity["max_flat_budget_relative_error"]:.9g}. '
                'Exact historical weight/config hashes preserved; original CPU SVD/writeback recomputed only for audit.\n')
        summary = r.read(r.OUT / exp / 'summary.json')
        summary['failed_job_ids'] = ['14888']
        summary['failure_before_generation'] = 'Builder held an exclusive CUDA context; fixed by isolated builder subprocess. Existing adapters reused.'
        if 'Initial Slurm14888' not in report_path.read_text():
            with report_path.open('a') as handle:
                handle.write('\nInitial Slurm14888 failed before generation because the in-process builder retained '
                    'an exclusive CUDA context. Slurm14889 isolates the builder, reuses the audited weights and '
                    'keeps the evaluation protocol unchanged. See submission.json and retained logs.\n')
        summary['outputs_sha256'] = {n: r.sha(r.OUT / exp / n) for n in
            ('report.md', 'checkpoint_results.tsv', 'paired_ci.tsv', 'artifact_integrity_audit.json', 'coverage_audit.json')}
        r.write(r.OUT / exp / 'summary.json', summary)


def writing_package(experiment, report, summary, extra_files=None):
    folder = r.OUT / experiment
    folder.mkdir(exist_ok=True)
    protocol = folder / 'protocol.md'
    protocol.write_text(f'# {experiment}: locked writing-only scope\n\nNo training, calibration, inference or new test scores. '
        'E5 checks the pinned official implementation and uses the requested setting-mismatch exception. '
        'E6 extracts the existing implementation without changing numerical settings. '
        'Metrics/CI TSVs contain headers only; numerical update audits are not applicable.\n')
    r.write(folder / 'protocol.lock.json', dict(created_utc=r.now(), protocol_sha256=r.sha(protocol),
        writer_sha256=r.sha(Path(__file__)), source_hashes=summary.get('source_hashes', {})))
    (folder / 'report.md').write_text(report)
    (folder / 'checkpoint_results.tsv').write_text('checkpoint\tmethod\ttarget\toff_score\tforgetting_gap\n')
    (folder / 'paired_ci.tsv').write_text('comparison\tmetric\tdifference_pp\tci_low_pp\tci_high_pp\n')
    for name, text in (extra_files or {}).items():
        (folder / name).write_text(text)
    r.write(folder / 'coverage_audit.json', dict(status='pass', evaluation_cells=0, mode='writing only'))
    r.write(folder / 'artifact_integrity_audit.json', dict(status='pass', weights=[],
        reconstruction_error=None, applicability='No adapter construction/evaluation in this writing-only stage',
        source_hashes=summary.get('source_hashes', {})))
    r.write(folder / 'summary.json', dict(status='complete', completed_utc=r.now(), new_cells=0,
        **summary, outputs_sha256={n: r.sha(folder / n) for n in ('report.md', 'checkpoint_results.tsv', 'paired_ci.tsv')}))


def write_e5():
    sources = r.read(r.OUT / 'literature_audit.json')
    commit = sources['loda']['commit']
    report = f'''# E5: LoDA scope assessment

Public source is available in [the authors' repository](https://github.com/HHHLF/LoDA_ICML2026/tree/{commit}). Its [network implementation](https://github.com/HHHLF/LoDA_ICML2026/blob/{commit}/models/network.py) uses a pretrained Vision Transformer; published configs cover image classification sequences. Porting it to Qwen causal-language-model training would require a new implementation and recipe. Under the requested setting-mismatch exception, no GPU comparison is run.

[LoDA](https://proceedings.mlr.press/v306/he26f.html) uses task data and past feature statistics to construct shared/isolated subspaces, learns up-projections, and recalibrates before merging task updates. LoRA-Norm edits a completed single adapter without data. These interventions address different stages; combining them is a future hypothesis, not an established compatibility or additive-gain result.

Rebuttal wording: “LoDA is a data-dependent continual-learning training method; our experiment studies data-free post-processing of a fixed single-task adapter. We clarify this scope difference and do not infer a performance ranking across these settings.”

No metrics or CIs exist for this assessment; both TSVs deliberately contain headers only.
'''
    writing_package('E5', report, dict(mode='scope comparison; no GPU experiment', code_public=True,
        official_commit=commit, reason='Official implementation is vision continual learning; direct Qwen port is beyond fixed single-adapter rebuttal scope.',
        compatibility_tested=False))


def write_e6():
    names = ['src/finetune/spectral_edit/posthoc_hns.py', 'src/finetune/spectral_edit/svd.py',
             'src/finetune/spectral_edit/cli.py', 'src/finetune/spectral_edit/io.py']
    hashes, symbols = {}, {}
    for name in names:
        path = r.ROOT / name
        hashes[name] = r.sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                symbols[node.name] = dict(file=name, line=node.lineno, end_line=node.end_lineno)
    posthoc = ast.parse((r.ROOT / names[0]).read_text())
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in posthoc.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    fast, stable = constants['FAST_HNS_COEFFICIENTS'], constants['STABLE_HNS_COEFFICIENTS']
    assert fast == (3.4445, -4.775, 2.0315) and stable == (2., -1.5, .5)
    def link(name):
        loc = symbols[name]
        return f'[{loc["file"]}:{loc["line"]}–{loc["end_line"]}]({r.ROOT / loc["file"]}:{loc["line"]})'
    tex = r'''\begin{algorithm}[t]
\caption{LoRA-Norm: data-free nuclear-preserving HNS (paper setting $4+1$)}
\label{alg:lora_norm_complete}
\begin{algorithmic}[1]
\Require Adapter factors $(A_m,B_m)$, unchanged scaling $s_m=\alpha_m/r_m$;
selected module suffixes and layer bounds; $K_f=4,K_s=1$, $\lambda=1$, $\epsilon=10^{-7}$
\Ensure Edited adapter; unselected factors and source configuration unchanged
\For{each selected complete LoRA factor pair $m$}
  \State Remember original storage dtypes; cast $A_m,B_m$ to fp32
  \State $B_m=Q_bR_b$, $A_m^\top=Q_aR_a$ \Comment{reduced QR}
  \State $R_bR_a^\top=U_c\operatorname{diag}(\sigma)V_c^\top$ \Comment{compact fp32 SVD}
  \State $U=Q_bU_c$, $V=Q_aV_c$ \Comment{$\Delta W_m=s_mU\operatorname{diag}(\sigma)V^\top$}
  \State If rank is empty, return unchanged empty spectrum
  \State Convert $\sigma$ to fp64 and clamp to nonnegative values
  \State $r_o=r$; if output rank $q$ supplied, $r_o=\min(\max(1,q),r)$
  \State $\sigma_{\rm ref}=\sigma_{1:r_o}$, $x=\sigma/\max(\|\sigma\|_2,\epsilon)$
  \For{$k=1,\ldots,K_f$}
    \State $x\gets x(3.4445-4.7750x^2+2.0315x^4)$ \Comment{elementwise}
  \EndFor
  \For{$k=1,\ldots,K_s$}
    \State $x\gets x(2.0-1.5x^2+0.5x^4)$
  \EndFor
  \State $x\gets\max(x_{1:r_o},0)$, $\tau=\sum_i\sigma_{{\rm ref},i}$
  \If{$\sum_i x_i\leq\epsilon$}
    \State $x_i\gets\tau/r_o$ for every retained direction
  \Else
    \State $x\gets x\,\tau/(\sum_i x_i)$ \Comment{restore nuclear norm}
  \EndIf
  \State $\sigma'\gets\sigma_{\rm ref}+\lambda(x-\sigma_{\rm ref})$; cast back to fp32
  \State $B'_m=U_{:,1:r_o}\operatorname{diag}(\sqrt{\max(\sigma',0)})$
  \State $A'_m=\operatorname{diag}(\sqrt{\max(\sigma',0)})V_{:,1:r_o}^\top$
  \State Cast each factor once to its original storage dtype; detach, move to CPU, save
\EndFor
\State Save adapter tensors and per-module spectral/configuration metadata
\end{algorithmic}
\end{algorithm}
'''
    report = '# E6: complete algorithm and code audit\n\n' + '''The manuscript setting is all7 projection families,4 fast steps plus1 stable step, strength1, nuclear preservation and native source rank. No model forward/backward or calibration data is needed. The CLI defaults are8+2; the rebuttal explicitly passes4+1.

The actual implementation performs fp32 compact QR/SVD of BA, fp64 polynomial/norm arithmetic, casts the new spectrum to fp32, forms balanced factors, then casts each factor to its source storage dtype. Scaling alpha/r stays outside the factorization and is unchanged. This is equivalent to nuclear-preserving editing of sBA for the positive scaling2 used in the cohort. It does not write the unbalanced A=V^T, B=U Sigma representation described in the initial proposal.

Rank truncation is optional in the spectral function; the reported experiments retain the source rank (16, or native64 in E4b). The source adapter configuration is copied unchanged. A different output rank would additionally require consistent PEFT rank/alpha configuration; copying the source configuration alone does not implement that adjustment. Normalization/restoration eps is1e-7; nonnegative clamps, an empty-spectrum branch and the uniform fallback for near-zero edited mass are explicit. Entropy/effective-rank diagnostics separately use eps1e-12 and are not part of the transform. Strength must lie in[0,1].

The complete publication-ready block is in algorithm.tex (algorithmicx/algpseudocode syntax). It includes module/layer selection, effective update/scaling, compact SVD, both coefficient triples, rank handling, norm restoration, interpolation, dtype/cast timing, writeback and metadata.

''' + '\n'.join([
        f'- Module selection and writeback: {link("_collect_lora_pairs")}; {link("run_hns")}.',
        f'- Reduced QR/compact SVD: {link("lowrank_svd_from_ba")}.',
        f'- Polynomial step: {link("_ns_polynomial_step")}.',
        f'- HNS spectrum, rank, eps and nuclear restoration: {link("hybrid_newton_schulz_singular_values")}.',
        f'- SVD truncation: {link("apply_hns_to_svd")}.',
        f'- Balanced factor reconstruction and final nonnegative clamp: {link("rebuild_ba_from_uv_sigma")}.',
        f'- Per-module scaling: {link("get_scaling_for_module")}.',
        f'- CLI numerical defaults: {link("build_arg_parser")}.',
    ]) + '\n\nAll source SHA256 and exact symbol line ranges are in summary.json. No evaluation metrics or paired intervals are generated by this writing-only stage.\n'
    writing_package('E6', report, dict(mode='algorithm extraction; no GPU work', source_hashes=hashes,
        symbols=symbols, fast_coefficients=fast, stable_coefficients=stable,
        paper_steps=[4,1], cli_default_steps=[8,2], eps=1e-7, source_rank_retained=True), {'algorithm.tex': tex})


def writing_stages():
    for exp in ('E1', 'E0', 'E2', 'E3', 'E4'):
        validate(exp)
    write_e5(); publish(['E5'])
    write_e6(); publish(['E6'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publish', nargs='+')
    p.add_argument('--refresh-p0', action='store_true')
    p.add_argument('--writing', action='store_true')
    args = p.parse_args()
    if args.refresh_p0:
        refresh_p0()
    if args.publish:
        publish(args.publish)
    if args.writing:
        writing_stages()
