#!/usr/bin/env python3
"""Validate complete diagonal cells and export Flat-first/final summaries."""
import argparse
import json
import statistics
from pathlib import Path
from audit_posthoc_three_seed import OUT, ROOT
TASK_NAMES = {'magicoder': 'Magicoder', 'metamath': 'MetaMath', 'tulu': 'Tulu'}


def table(rows, methods):
    lines = ['| Base | Task | Seed | ' + ' | '.join(methods) + ' |',
        '| ---- | ---- | ---: | ' + ' | '.join('---:' for _ in methods) + ' |']
    for r in rows:
        lines.append('| ' + ' | '.join([r['base'], TASK_NAMES[r['task']], str(r['seed'])] + [f"{r[m]:.2f}" for m in methods]) + ' |')
    return lines


def stats(rows, methods):
    result = {}
    for method in methods:
        values = [r[method] for r in rows]
        entry = {'mean': statistics.mean(values),
            'delta_vs_LoRA_pp': statistics.mean(r[method] - r['LoRA'] for r in rows),
            'delta_vs_HNS_pp': statistics.mean(r[method] - r['HNS'] for r in rows)}
        for ref in ('LoRA', 'HNS'):
            delta = [r[method] - r[ref] for r in rows]
            entry[f'vs_{ref}_win_tie_loss'] = [sum(d > 1e-8 for d in delta), sum(abs(d) <= 1e-8 for d in delta), sum(d < -1e-8 for d in delta)]
        result[method] = entry
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('flat', 'final'), required=True)
    args = p.parse_args()
    audit = json.loads((OUT / 'source_manifest.json').read_text())
    methods = ['LoRA', 'Flat-Fro', 'Flat-Nuclear'] + (['DG-Hard'] if args.phase == 'final' else []) + ['HNS']
    rows = []
    historical = []
    for source in audit['checkpoints']:
        r = {k: source[k] for k in ('base', 'task', 'seed')}
        r.update({k: v['score_percent'] for k, v in source['references'].items()})
        historical.append(dict(r))
        selections = [('Flat-Fro', 'flat', 'flat_fro'), ('Flat-Nuclear', 'flat', 'flat_nuclear')]
        if args.phase == 'final':
            selections = [(display, 'joint', method) for display, method in [('LoRA', 'original_lora'), ('Flat-Fro', 'flat_fro'), ('Flat-Nuclear', 'flat_nuclear'), ('DG-Hard', 'dg_hard'), ('HNS', 'hns_f4_s1')]]
        for display, phase, method in selections:
            dest = OUT / 'eval' / phase / r['base']
            gm = json.loads((dest / 'generation_manifest.json').read_text())
            assert gm['status'] == 'generation_complete'
            sm = json.loads((dest / 'score_manifest.json').read_text())
            label = f"{r['task']}__seed{r['seed']}__{method}"
            cell = dest / r['task'] / label
            assert (cell / 'COMPLETE').is_file()
            record = next(v for v in sm['records'] if v['task'] == r['task'] and v['variant'] == label)
            assert record['samples'] == source['evaluation_spec']['samples']
            assert record['primary_metric'] == source['references']['LoRA']['record']['primary_metric']
            assert sum(1 for line in (cell / 'scored.jsonl').open() if line.strip()) == record['samples']
            r[display] = 100 * record[record['primary_metric']]
        rows.append(r)
    assert len(rows) == 18
    base_order = {b: i for i, b in enumerate(dict.fromkeys(r['base'] for r in audit['checkpoints']))}
    task_order = {t: i for i, t in enumerate(TASK_NAMES)}
    rows.sort(key=lambda r: (base_order[r['base']], task_order[r['task']], r['seed']))
    identity_comparisons = []
    if args.phase == 'final':
        for r in rows:
            dest = OUT / 'eval/joint' / r['base'] / r['task']
            cells = [dest / f"{r['task']}__seed{r['seed']}__{method}" for method in ('original_lora', 'dg_hard')]
            pp, dd = [{v['id']: v for v in map(json.loads, (c / 'predictions.jsonl').read_text().splitlines())} for c in cells]
            assert pp.keys() == dd.keys()
            identity_comparisons.append({**{k: r[k] for k in ('base', 'task', 'seed')},
                'samples': len(pp), 'token_mismatches': sum(pp[k]['token_ids'] != dd[k]['token_ids'] for k in pp),
                'score_delta_pp': r['DG-Hard'] - r['LoRA']})
    overall = stats(rows, methods)
    groups = {}
    for dimension in ('base', 'task'):
        groups[dimension] = {key: stats([r for r in rows if r[dimension] == key], methods) for key in sorted({r[dimension] for r in rows})}
    seed_stats = []
    for base, task in sorted({(r['base'], r['task']) for r in rows}):
        subset = [r for r in rows if (r['base'], r['task']) == (base, task)]
        assert sorted(r['seed'] for r in subset) == [42, 43, 44]
        seed_stats.append({'base': base, 'task': task, 'methods': {m: {'mean': statistics.mean(r[m] for r in subset), 'sample_sd': statistics.stdev(r[m] for r in subset)} for m in methods}})
    probe_comparisons = []
    for phase in ('flat', 'dg') if args.phase == 'final' else ('flat',):
        for base in sorted({r['base'] for r in rows}):
            dest = OUT / 'eval' / phase / base
            vm = json.loads((OUT / f'{base}_{phase}_variant_manifest.json').read_text())
            completed_probes = sorted(p for p in (OUT / 'batch_probe' / phase / base).glob('seqs*') if p.is_dir() and (p / 'generation_manifest.json').is_file())
            assert completed_probes
            probe = completed_probes[-1]
            for v in vm['variants']:
                partial = probe / v['train_task'] / v['label'] / 'predictions.jsonl'
                full = dest / v['train_task'] / v['label'] / 'predictions.jsonl'
                pp = {r['id']: r for r in map(json.loads, partial.read_text().splitlines())}
                ff = {r['id']: r for r in map(json.loads, full.read_text().splitlines())}
                mismatches = sum(pp[k]['token_ids'] != ff[k]['token_ids'] for k in pp)
                probe_comparisons.append({'phase': phase, 'base': base, 'label': v['label'], 'samples': len(pp), 'token_mismatches': mismatches})
    result = {'status': 'complete', 'phase': args.phase, 'rows': rows, 'overall': overall, 'per_group': groups, 'three_seed': seed_stats, 'batch_probe_token_comparisons': probe_comparisons,
        'historical_reference_rows': historical, 'joint_identity_comparisons': identity_comparisons}
    path = OUT / ('flat_summary.json' if args.phase == 'flat' else 'final_summary.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Post-hoc Flat / DG-Hard 三组训练运行 diagonal evaluation', '',
        f"状态：{args.phase} 阶段全部完成。18 checkpoints；Flat-Fro / Flat-Nuclear 各18个完整 diagonal cells" + ('；DG-Hard 18个完整 diagonal cells。' if args.phase == 'final' else '。') + '成绩为百分比，差值为百分点。', '',
        'HNS 统一代表配置：4+1、all modules、strength=1、preserve_nuclear_norm=true、输出 rank16。不使用旧模型卡或逐 checkpoint 最优值。' + ('最终主表的全部五种方法来自本次 joint 统一复评；历史 LoRA/HNS 参考成绩仅保留在 source manifest 和先行 Flat/DG 快照中。' if args.phase == 'final' else 'LoRA / HNS 从三种子报告指定的统一 step-grid score manifests 读取。'), '',
        '协议：完全复用 scripts/eval_forgetting_matrix_vllm.py 和 scripts/score_forgetting_matrix.py；HumanEval strict_continuation chat / pass@1（164）；GSM8K strict accuracy（1319）；IFEval prompt strict（541）。non_thinking chat、greedy temperature0/top_p1、推理 seed42，生成上限分别512/512/2048，max_model_len4096。VLLM_BATCH_INVARIANT=1、FLASH_ATTN、async_scheduling=false、prefix caching=false。', '',
        '统计边界：原始42是历史归档标签；Llama Magicoder/MetaMath 原始训练 seed 未完整核实。原始与43/44训练 recipe 存在报告已记录的差异，因此下列 mean±sample SD 是三组现有训练运行的描述统计，不能声称为严格同 recipe 的三训练种子重复。跨 benchmark 的总平均是18 cells 等权宏平均，不能解释为统一指标或合并样本准确率。', '',
        '## 逐 checkpoint', '', *table(rows, methods), '',
        '## 18 checkpoints 宏平均和胜负', '',
        '| Method | Mean | Δ vs LoRA | Δ vs HNS | vs LoRA W/T/L | vs HNS W/T/L |',
        '| --- | ---: | ---: | ---: | --- | --- |']
    for m, s in overall.items():
        lines.append(f"| {m} | {s['mean']:.4f} | {s['delta_vs_LoRA_pp']:+.4f} | {s['delta_vs_HNS_pp']:+.4f} | {'/'.join(map(str,s['vs_LoRA_win_tie_loss']))} | {'/'.join(map(str,s['vs_HNS_win_tie_loss']))} |")
    for dimension, grouped in groups.items():
        lines += ['', f'## Per-{dimension} 平均', '', '| Group | ' + ' | '.join(methods) + ' |', '| --- | ' + ' | '.join('---:' for _ in methods) + ' |']
        for key, ss in grouped.items():
            lines.append('| ' + TASK_NAMES.get(key, key) + ' | ' + ' | '.join(f"{ss[m]['mean']:.4f}" for m in methods) + ' |')
    lines += ['', '## 3-seed mean ± sample SD（ddof=1）', '', '| Base | Task | ' + ' | '.join(methods) + ' |', '| --- | --- | ' + ' | '.join('---:' for _ in methods) + ' |']
    for s in seed_stats:
        lines.append('| ' + s['base'] + ' | ' + TASK_NAMES[s['task']] + ' | ' + ' | '.join(f"{s['methods'][m]['mean']:.2f} ± {s['methods'][m]['sample_sd']:.2f}" for m in methods) + ' |')
    build = json.loads((OUT / 'flat_build_complete.json').read_text())
    lines += ['', '## 数值审计', '', 'Flat-Fro: t=||σ||₂/√r；Flat-Nuclear: t=||σ||₁/r。两者均使用 shared compact SVD 和 B=U√t、A=√t Vᵀ。所有 source config 字节保持一致，未改变 scaling、rank、alpha、target modules 或 base model；保存权重重载后逐 tensor 验证。', '']
    for m in ('flat_fro', 'flat_nuclear'):
        vs = [v for c in build['results'] for v in c['variants'] if v['method'] == m]
        lines.append(f"- {m}: max target budget relative error={max(v['max_budget_relative_error'] for v in vs):.9g}; max saved budget relative error={max(v['max_saved_budget_relative_error'] for v in vs):.9g}.")
    lines += ['', f"已有短测试与完整评测的逐 token 重复性检查：{sum(s['token_mismatches'] for s in probe_comparisons)} / {sum(s['samples'] for s in probe_comparisons)} 个重复样本输出不同；详细计数见 summary JSON。"]
    lines += ['', 'HumanEval 日志存在 multiprocessing 临时目录清理回调的 NFS `.nfs*` busy 异常，既有 HNS seed-eval 日志也出现相同现象。检查 human_eval/execution.py 确认测试结果先写入 manager result 再清理；主评分正常完成。汇总逐 cell 验证全部164条执行结果和 scored rows，无遗漏；未修改执行 timeout、parser 或 metric。']
    if args.phase == 'final':
        dg = json.loads((OUT / 'dg_build_complete.json').read_text())
        lines += ['', '## 统一比较与 identity 检查', '',
            '严格先完成 Flat 36 cells 和先行汇总，再实现、构建、完成独立 DG 18 cells。独立 DG 权重逐字节等于原始 LoRA，但与历史 LoRA predictions 出现生成漂移（Llama seed44/Tulu 98/541 token 序列不同、主指标差−0.9242pp）。因此追加 joint 统一复评：每个 checkpoint 的 LoRA / Flat-Fro / Flat-Nuclear / DG-Hard / 固定 HNS 放入同一个五-adapter batch。不预设漂移具体原因，不把 identity 的执行差异当作算法效果；未更改 prompt、生成参数、parser 或 metric。', '',
            f"最终同批 LoRA / DG 对照：token 序列不同 {sum(s['token_mismatches'] for s in identity_comparisons)} / {sum(s['samples'] for s in identity_comparisons)} 个样本；成绩不同的 checkpoints={sum(abs(s['score_delta_pp']) > 1e-8 for s in identity_comparisons)} / 18。逐 checkpoint 检查见 final_summary.json。", '',
            '先行 Flat 汇总：flat_summary.md/json；独立 DG 与历史参考结果：dg_initial_summary.json；joint 完整结果：eval/joint/<base>/{generation_manifest,score_manifest,commands}.json。主表不把历史成绩与 joint 成绩拼接。']
        drift = json.loads((OUT / 'identity_legacy_drift_audit.json').read_text())
        lines += ['', f"历史漂移审计：{drift['token_mismatches']} / {drift['samples']} 个样本 token 序列不同，全部 benchmark 输入记录一致；逐 checkpoint 的旧/新 predictions 路径、分数、差值见 identity_legacy_drift_audit.json。"]
        lines += ['', '## 原始 DG-Hard 审计', '', 'τ=ω(β)·median(full singular spectrum), β=min(m,n)/max(m,n); ω(β)=λ*(β)/√μβ，其中 μβ 为 Marchenko–Pastur 分布中位数。完整 spectrum 长度 min(m,n)，补上低秩表示中省略的零；严格保留 σ>τ。未采用 active-spectrum median。', '',
            '| Base | Task | Seed | Retained rank min/max | Unchanged module fraction | Threshold min/max | Retained Fro / before (ratio) | Retained nuclear / before (ratio) | Identity |',
            '| --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- |']
        for c in dg['results']:
            s = c['summary']
            lines.append(f"| {c['base']} | {TASK_NAMES[c['task']]} | {c['seed']} | {s['retained_rank_min']}/{s['retained_rank_max']} | {s['unchanged_module_fraction']:.6f} | {s['threshold_min']:.6g}/{s['threshold_max']:.6g} | {s['fro_retained']:.6g}/{s['fro_before']:.6g} ({s['retained_fro_ratio']:.9g}) | {s['nuclear_retained']:.6g}/{s['nuclear_before']:.6g} ({s['retained_nuclear_ratio']:.9g}) | {s['identity']} |")
        lines += ['', 'DG 的逐 module threshold、median、aspect ratio、保留谱和 norm 位于每个 dg_hard/posthoc_meta.json。identity 模块直接保留原始 A/B，以避免 SVD 重构舍入改变 identity baseline。DG-Hard 仍通过同一脚本重新生成、评分，不以 LoRA 成绩填充。', '',
            '原论文：[Donoho & Gavish, The Optimal Hard Threshold for Singular Values is 4/√3](https://arxiv.org/abs/1305.5870)。']
        lines += ['', 'Fro 总量为各 module Fro² 之和开根号；nuclear 总量为各 module nuclear norm 之和，均针对未乘 scaling 的 BA。原始 scaling 保持不变；norm ratios 同样适用于 scaled updates。MP density 为 sqrt((b-x)(x-a))/(2πβx)，按作者原始补充代码的归一化计算；Gauss–Legendre 积分加二分求中位数，逐步加倍积分阶数至收敛，不使用三次多项式近似。', '',
            '原始补充代码（作者代码镜像）：[optimal_SVHT_coef.m](https://raw.githubusercontent.com/bwbrunton/dmd-neuro/master/optimal_SVHT_coef.m)。']
    lines += ['', '## 运行顺序与命令', '', '最多两个单 GPU Slurm worker（array=0-1%2，gres=gpu:1）；每个 worker 一个 base，所有训练任务和种子串行/同引擎批量评测。先做 max_num_seqs2048 的256样本/任务探测，再完整评测；OOM 才依次降为1024/512/256。max_num_batched_tokens65536、prompt chunk1024；独立 Flat/DG adapter block6，joint block5。joint 五-adapter batch 的并发规模在成功的六-adapter Flat 短测试范围内，沿用2048，无额外探测。实际选定配置见 generation_manifest，探测日志见 batch_probe。', '',
        '```bash', 'PYTHONPATH=src OMP_NUM_THREADS=4 /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/audit_posthoc_three_seed.py',
        'PYTHONPATH=src /dataset1/zailong/envs/peft-sft-lab/bin/python -m pytest -q tests/test_posthoc_flat.py',
        'PYTHONPATH=src:scripts OMP_NUM_THREADS=4 /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/build_posthoc_flat_three_seed.py',
        'sbatch slurm/posthoc_flat_three_seed.slurm',
        'PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/summarize_posthoc_flat_dghard_three_seed.py --phase flat', '```', '',
        '逐条真实 generation/scoring argv 见 eval/<phase>/<base>/commands.json；Slurm 提交回执见 submission.json。']
    submissions = json.loads((OUT / 'submission.json').read_text())
    lines += ['', '实际 Slurm 提交：`' + json.dumps(submissions, ensure_ascii=False) + '`。首次 job970 因 node01 /tmp 满在引擎启动阶段失败，没有生成评测结果；job972 改用 workspace 临时缓存路径后重试。']
    if args.phase == 'final':
        lines += ['', '```bash', 'PYTHONPATH=src /dataset1/zailong/envs/peft-sft-lab/bin/python -m pytest -q tests/test_posthoc_dg_hard.py', 'PYTHONPATH=src:scripts OMP_NUM_THREADS=4 /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/build_posthoc_dghard_three_seed.py', 'sbatch slurm/posthoc_dghard_three_seed.slurm', 'PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/audit_posthoc_identity_drift.py', 'PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/prepare_posthoc_joint_three_seed.py', 'sbatch slurm/posthoc_joint_three_seed.slurm', 'PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/summarize_posthoc_flat_dghard_three_seed.py --phase final', '```']
    lines += ['', '## Source / adapter 路径和 manifest', '', f'完整 source audit: `{OUT / "source_manifest.json"}`。Flat build: `flat_build_complete.json`。先行 Flat 结果快照: `flat_summary.json` / `flat_summary.md`。所有结果 JSON 保留未四舍五入分数。', '', '| Base | Task | Seed | Source checkpoint | Adapter parent |', '| --- | --- | ---: | --- | --- |']
    for r in audit['checkpoints']:
        parent = OUT / 'adapters' / r['base'] / f"seed{r['seed']}" / r['task']
        lines.append(f"| {r['base']} | {r['task']} | {r['seed']} | `{r['source']}` | `{parent}` |")
    lines += ['', '每个 Adapter parent 下为 `flat_fro/`、`flat_nuclear/`' + ('、`dg_hard/`' if args.phase == 'final' else '') + '。Variant manifests: `<base>_flat_variant_manifest.json`' + (' / `<base>_dg_variant_manifest.json`。' if args.phase == 'final' else '。'), '', 'Source manifest 同时记录原始 variant/score/generation/task config 路径、SHA256、完整 adapter config、module shapes 和 LoRA/HNS metric records。']
    if args.phase == 'final':
        lines += ['', '最终统一复评 manifests：`<base>_joint_variant_manifest.json`（每个 base 45 entries，按 checkpoint 五种方法分组）。全部 GPU 作业已完成并释放资源；Slurm 实际分配均为每个 worker 一张 B300，最多两个同时运行。']
    if (OUT / 'experiment_manifest.json').is_file():
        provenance = json.loads((OUT / 'experiment_manifest.json').read_text())
        lines += ['', f'总实验 manifest：`{OUT / "experiment_manifest.json"}`，包含全部 artifact 索引、评测/谱编辑代码 SHA256、包版本和 GPU 限制。', '', '实际包版本：`' + json.dumps(provenance['packages']) + '`。']
    content = '\n'.join(lines) + '\n'
    (ROOT / 'reports/posthoc_flat_dghard_three_seed_20260913.md').write_text(content)
    if args.phase == 'flat':
        (OUT / 'flat_summary.md').write_text(content)
    print('\n'.join(table(rows, methods)))
    print(json.dumps(overall, indent=2))


if __name__ == '__main__':
    main()
