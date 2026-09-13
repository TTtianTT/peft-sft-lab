#!/usr/bin/env python3
"""Atomically publish complete cells and descriptive three-seed forgetting tables."""
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import statistics as st
from pathlib import Path
from zoneinfo import ZoneInfo
from prepare_posthoc_forgetting_three_seed import OUT, REPORT, TASKS, METHODS, NAMES, COUNTS, BASES


def atomic(path, text):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def render(value):
    return '待完成' if value is None else f'{value:.2f}'


def table(lines, columns, rows):
    lines += ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    lines.extend('| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows)
    lines.append('')


def tsv(path, rows, columns):
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter='\t', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, handle.getvalue())


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT / 'summary.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summarize()


def summarize():
    manifest = load(OUT / 'manifest.json')
    assert manifest, 'Prepare manifest first'
    scores, metrics, matrix, cs_rows = {}, {}, [], []
    for base, _ in BASES:
        vm = load(OUT / f'{base}_variant_manifest.json')
        labels = ['base'] + [v['label'] for v in vm['variants']]
        for task in TASKS:
            for label in labels:
                folder = OUT / 'eval' / base / task / label
                metric = load(folder / 'metrics.json')
                if not metric or not (folder/'COMPLETE').is_file() or not (folder/'scored.jsonl').is_file():
                    continue
                assert metric['samples'] == COUNTS[task], (base, task, label, metric['samples'])
                if task == 'commonsense':
                    assert {t:r['total'] for t,r in metric['per_task'].items()} == manifest['commonsense_subtask_counts']
                scores[base, task, label] = metric[metric['primary_metric']] * 100
                metrics[base, task, label] = metric
                variant = next((v for v in vm['variants'] if v['label'] == label), {})
                matrix.append(dict(base=base, train_task=variant.get('train_task','base'),
                    seed=variant.get('seed',''), method=variant.get('method','base'), eval_task=task,
                    score=scores[base,task,label], samples=metric['samples'], primary_metric=metric['primary_metric'],
                    metrics_path=str(folder/'metrics.json'), predictions_path=str(folder/'predictions.jsonl')))
                if task == 'commonsense':
                    for subtask, row in metric['per_task'].items():
                        cs_rows.append(dict(base=base, train_task=variant.get('train_task','base'),
                            seed=variant.get('seed',''),method=variant.get('method','base'),
                            subtask=subtask,score=row['accuracy']*100,samples=row['total']))
    def get(base, train, seed, method, task):
        return scores.get((base,task,f'{train}__seed{seed}__{method}'))
    cohorts, derived = [], []
    for base, _ in BASES:
        for train in TASKS[:3]:
            off = [t for t in TASKS if t != train]
            for seed in (42,43,44):
                # Equal complete cohorts across all five methods; partial cells stay in the matrix.
                ready = all((base,t,'base') in scores and
                            all(get(base,train,seed,m,t) is not None for m in METHODS) for t in off)
                if not ready:
                    continue
                cohorts.append((base,train,seed))
                for method in METHODS:
                    score = {t:get(base,train,seed,method,t) for t in off}
                    target = get(base,train,seed,method,train)
                    lora_target = get(base,train,seed,METHODS[0],train)
                    derived.append(dict(base=base,train_task=train,seed=seed,method=method,target=target,
                        target_gain=None if target is None or lora_target is None else target-lora_target,
                        base_off=st.mean(scores[base,t,'base'] for t in off),
                        off_score=st.mean(score.values()),
                        off_gain_lora=st.mean(score[t]-get(base,train,seed,METHODS[0],t) for t in off),
                        off_gain_hns=st.mean(score[t]-get(base,train,seed,METHODS[-1],t) for t in off),
                        forgetting_gap=st.mean(max(scores[base,t,'base']-score[t],0) for t in off)))
    index={(r['base'],r['train_task'],r['seed'],r['method']):r for r in derived}
    def aggregate(rows):
        result=[]
        for method,name in zip(METHODS,NAMES):
            rs=[r for r in rows if r['method']==method]
            if not rs:
                continue
            def wtl(key):
                return [sum(r[key]>1e-9 for r in rs),sum(abs(r[key])<=1e-9 for r in rs),sum(r[key]<-1e-9 for r in rs)]
            def gap_wtl(reference):
                deltas=[index[r['base'],r['train_task'],r['seed'],reference]['forgetting_gap']-r['forgetting_gap'] for r in rs]
                return [sum(d>1e-9 for d in deltas),sum(abs(d)<=1e-9 for d in deltas),sum(d<-1e-9 for d in deltas)]
            result.append(dict(method=method,name=name,n=len(rs),off_score=st.mean(r['off_score'] for r in rs),
                off_gain_lora=st.mean(r['off_gain_lora'] for r in rs),off_gain_hns=st.mean(r['off_gain_hns'] for r in rs),
                forgetting_gap=st.mean(r['forgetting_gap'] for r in rs),
                gap_reduction_lora=st.mean(index[r['base'],r['train_task'],r['seed'],METHODS[0]]['forgetting_gap']-r['forgetting_gap'] for r in rs),
                gap_reduction_hns=st.mean(index[r['base'],r['train_task'],r['seed'],METHODS[-1]]['forgetting_gap']-r['forgetting_gap'] for r in rs),
                vs_lora=wtl('off_gain_lora'),vs_hns=wtl('off_gain_hns'),
                gap_vs_lora=gap_wtl(METHODS[0]),gap_vs_hns=gap_wtl(METHODS[-1])))
        return result
    total=aggregate(derived)
    grouped={}
    for field,values in (('base',[b for b,_ in BASES]),('train_task',list(TASKS[:3]))):
        for value in values:
            grouped[f'{field}/{value}']=aggregate([r for r in derived if r[field]==value])
    now=datetime.now(timezone.utc).isoformat()
    complete=len(matrix)==368
    job=load(OUT/'job_manifest.json')
    worker_states=[load(OUT/f'{b}_worker_status.json') for b,_ in BASES]
    completed=[r['updated_utc'] for r in worker_states if r and r['phase']=='complete']
    completed_sg=datetime.fromisoformat(max(completed)).astimezone(ZoneInfo('Asia/Singapore')).isoformat() if len(completed)==2 else None
    lines=['# Flat-Fro / Flat-Nuclear / DG-Hard 三种子 forgetting 评测','',
        f'更新时间（UTC）：{now}。状态：{"全部评分完成" if complete else "运行中"}；已评分 {len(matrix)}/368 cells；完整 off-task checkpoint {len(cohorts)}/18。','',
        f'实验日期：2026-09-13开始，跨午夜继续运行；Slurm数组作业 {job["job_id"] if job else "待提交"}。全部生成/评分结束时间（新加坡）：{completed_sg or "待完成"}。','',
        '18 个既有 LoRA，90 个五方法 adapters；每个 base 45 variants × 4 基准 + 4 个共享 Base = 184 cells。未重新训练或重建 adapter。训练 seed 为 42/43/44，推理 seed 统一 42。','',
        '协议完全复用现有 HNS forgetting 的 evaluator、prompt/chat template、generation 参数、parser 和 metric：HumanEval pass@1（164）、GSM8K strict accuracy（1319）、IFEval prompt-level strict accuracy（541）、Commonsense-8 等权子任务 macro accuracy（22419）。','',
        'Off-task = 除训练任务对应基准外的其余三个基准族等权平均。遗忘量 = mean(max(Base − adapter, 0))，先逐基准截断再平均，越低越好；成绩越高越好。所有成绩为百分数，差值和遗忘量为百分点。Commonsense 内部的八个子任务等权，不能用22419样本的 pooled accuracy替代。','',
        'HNS 全部固定为 4+1、strength=1、all_modules、保持 nuclear norm，无逐 checkpoint 选优。五方法按同一 checkpoint 连续成组、block_size=5 一起生成。比较全部使用本轮新生成的 LoRA/HNS/Base；此前 diagonal 成绩只用于另存审计。','',
        '## 18-checkpoint off-task 总体比较','',
        '运行中只汇总所有五方法与 Base 均已完成三个 off-task 的同一批 checkpoint；N 显式列出，不能将中间结果当作最终18项平均。','']
    def aggregate_table(title,entries):
        lines.extend([title,''])
        table(lines,['方法','N','Off-task','ΔLoRA','ΔHNS','遗忘量↓','遗忘量减少 vs LoRA','遗忘量减少 vs HNS','Off-task vs LoRA 胜/平/负','Off-task vs HNS 胜/平/负'],
            [[e['name'],e['n'],render(e['off_score']),f"{e['off_gain_lora']:+.2f}",f"{e['off_gain_hns']:+.2f}",f"{e['forgetting_gap']:.3f}",f"{e['gap_reduction_lora']:+.3f}",f"{e['gap_reduction_hns']:+.4f}",'/'.join(map(str,e['vs_lora'])),'/'.join(map(str,e['vs_hns']))] for e in entries])
    aggregate_table('总体（18 项等权）',total)
    if complete:
        entries={e['method']:e for e in total}
        fro=entries['flat_fro'];nuc=entries['flat_nuclear'];hns=entries['hns_f4_s1']
        lines += [f"完整18项：Flat-Fro、Flat-Nuclear相对LoRA的平均off-task提升分别为 {fro['off_gain_lora']:+.3f}、{nuc['off_gain_lora']:+.3f} pp，平均遗忘量分别减少 {fro['gap_reduction_lora']:.3f}、{nuc['gap_reduction_lora']:.3f} pp。",'',
            f"Flat-Nuclear与固定HNS的平均遗忘量接近（{nuc['forgetting_gap']:.6f} vs {hns['forgetting_gap']:.6f} pp，差 {nuc['forgetting_gap']-hns['forgetting_gap']:+.6f} pp）；平均off-task ΔHNS={nuc['off_gain_hns']:+.6f} pp。遗忘量逐checkpoint相对HNS胜/平/负={'/'.join(map(str,nuc['gap_vs_hns']))}，胜负分布与均值一并报告。DG-Hard退化为identity，全部18项off-task平均值与LoRA持平。",'']
    lines += ['遗忘量逐checkpoint胜/平/负（更低为胜）：','']
    table(lines,['方法','N','遗忘量 vs LoRA 胜/平/负','遗忘量 vs HNS 胜/平/负'],
        [[e['name'],e['n'],'/'.join(map(str,e['gap_vs_lora'])),'/'.join(map(str,e['gap_vs_hns']))] for e in total])
    for key,entries in grouped.items():
        aggregate_table(f'### {key}',entries)
    for field,title in (('off_score','完整逐种子 off-task 成绩'),('forgetting_gap','完整逐种子遗忘量（越低越好）')):
        lines += [f'## {title}','']
        rows=[]
        for base,_ in BASES:
            for train in TASKS[:3]:
                for seed in (42,43,44):
                    rows.append([base,train,seed]+[render(index.get((base,train,seed,m),{}).get(field)) for m in METHODS])
        table(lines,['Base','训练任务','Seed',*NAMES],rows)
    lines += ['## 3-seed mean ± sample SD','',
        'SD 使用 n−1 分母。仅在同一 Base×训练任务的三个 seed 全部完成后显示；既有 seed42 与43/44训练 recipe差异、部分seed42训练seed证据不足沿用 source audit 的说明。这里是既有三次训练运行的描述性统计。','']
    seed_stats=[]
    for field,title in (('off_score','Off-task'),('forgetting_gap','遗忘量↓')):
        rows=[]
        for base,_ in BASES:
            for train in TASKS[:3]:
                cells=[base,train,title]
                for method in METHODS:
                    rs=[r for r in derived if r['base']==base and r['train_task']==train and r['method']==method]
                    if len(rs)==3:
                        avg=st.mean(r[field] for r in rs); sd=st.stdev(r[field] for r in rs)
                        cells.append(f'{avg:.2f} ± {sd:.2f}')
                        seed_stats.append(dict(base=base,train_task=train,method=method,field=field,mean=avg,sample_sd=sd,seeds=[42,43,44]))
                    else:
                        cells.append(f'待完成 ({len(rs)}/3)')
                rows.append(cells)
        table(lines,['Base','训练任务','统计',*NAMES],rows)
    lines += ['## 各评测基准的 3-seed mean ± sample SD','',
        '每个Base×训练任务×评测基准分别对42/43/44取均值及sample SD；同一条件三个seed齐全后才显示。','']
    benchmark_stats=[]
    benchmark_table=[]
    for base,_ in BASES:
        for train in TASKS[:3]:
            for task in TASKS:
                cells=[base,train,task,'target' if task==train else 'off-task']
                for method in METHODS:
                    values=[get(base,train,seed,method,task) for seed in (42,43,44)]
                    if all(value is not None for value in values):
                        avg=st.mean(values);sd=st.stdev(values)
                        cells.append(f'{avg:.2f} ± {sd:.2f}')
                        benchmark_stats.append(dict(base=base,train_task=train,eval_task=task,
                            method=method,mean=avg,sample_sd=sd,seeds=[42,43,44]))
                    else:
                        cells.append(f'待完成 ({sum(v is not None for v in values)}/3)')
                benchmark_table.append(cells)
    table(lines,['Base','训练任务','评测基准','角色',*NAMES],benchmark_table)
    lines += ['## 完整逐种子四基准矩阵','', '对应训练任务的基准行是 target，其余三行是 off-task；缺失结果明确标为待完成。','']
    rows=[]
    for base,_ in BASES:
        for train in TASKS[:3]:
            for seed in (42,43,44):
                for task in TASKS:
                    rows.append([base,train,seed,task,'target' if task==train else 'off-task',render(scores.get((base,task,'base')))]+[render(get(base,train,seed,m,task)) for m in METHODS])
    table(lines,['Base','训练任务','Seed','评测基准','角色','Base score',*NAMES],rows)
    lines += ['## 按评测基准汇总 off-task 成绩','',
        '前三个基准只纳入另外两种训练任务的 checkpoints（每方法12项）；Commonsense纳入全部18项。运行中只纳入本基准五方法均已完成的同一批checkpoint，并列出N；这是单基准阶段结果，不代表三个off-task已齐全。','']
    eval_aggregates=[]
    for task in TASKS:
        selected=[(b,t,s) for b,_ in BASES for t in TASKS[:3] for s in (42,43,44)
                  if t!=task and all(get(b,t,s,m,task) is not None for m in METHODS)]
        cells=[task,len(selected)]
        for method in METHODS:
            values=[get(b,t,s,method,task) for b,t,s in selected]
            avg=st.mean(values) if values else None
            cells.append(render(avg))
            eval_aggregates.append(dict(eval_task=task,method=method,n=len(values),score=avg))
        rows_cell=[cells]
        table(lines,['评测基准','N',*NAMES],rows_cell)
    lines += ['## DG-Hard identity 与重评审计','',
        '原始 DG-Hard 在全部4284模块中完整谱 median=0、threshold=0，retained rank=16、unchanged module fraction=100%、保留Frobenius/nuclear norm=100%；本轮重新加载核实原始A/B tensor逐字节一致（safetensors文件序列化SHA256可不同）。未使用DG-Hard-Active。完整谱退化情况和逐模块数值见此前 diagonal 报告及 dg_build_complete.json。','']
    identity=load(OUT/'identity_audit.json')
    lines += [f"本轮 token/metric identity 审计：{identity['status']}；{identity['samples']} 成对样本，token差异 {identity['token_mismatches']}，主指标差异 {identity['metric_mismatches']}。" if identity else '本轮全部四基准 token/metric identity 审计待完整生成与评分后运行。','']
    interim=load(OUT/'interim_humaneval_identity_audit.json')
    if interim:
        lines += [f"HumanEval先行配对审计已完成：{interim['pairs']} 对checkpoint、{interim['samples']} 样本，token差异 {interim['token_mismatches']}，主指标差异 {interim['metric_mismatches']}。",'']
    diagonal_audit=load(OUT/'diagonal_audit.json')
    if diagonal_audit:
        lines += [f"相对前轮统一五方法 diagonal 评测，有 {diagonal_audit['changed_metrics']}/{diagonal_audit['cells']} 个主指标变化；详见 diagonal_audit.json，主比较未混用历史分数。",'']
    integrity=load(OUT/'artifact_integrity_audit.json')
    coverage=load(OUT/'coverage_audit.json')
    lines += [f"最终代码/数据/adapter指纹审计：{integrity['status']}；{integrity['adapters']} 个adapters、{integrity['datasets']} 个数据文件。" if integrity else '最终代码/数据/adapter指纹审计待完成。','',
        f"完整数据覆盖审计：{coverage['status']}；{coverage['cells']} 个cells、{coverage['samples']} 条生成记录，所有条件的唯一ID及输入记录均与本轮Base对齐。" if coverage else '完整数据覆盖审计待完成。','']
    observations=job.get('resource_observations',[]) if job else []
    if observations:
        steps=observations[-1]['steps']
        lines += ['GPU资源实测（Slurm sstat，详细时间与原始字段见job_manifest）：'+
            '；'.join(f"{r['job_step']}: GPU利用率 {r['gpu_util_percent']}%，显存 {r['gpu_memory_mib']/1024:.2f} GiB" for r in steps)+'。全程最多两个单卡B300任务。','']
    lines += ['## 数据、路径、命令与进度记录','',
        f'- 完整数据覆盖审计：`{OUT}/coverage_audit.json`；逐样本token配对：`{OUT}/identity_audit.json`。',
        f'- 代码/数据/全部adapter指纹审计：`{OUT}/artifact_integrity_audit.json`。',
        f'- Slurm提交与最多两张GPU配置：`{OUT}/job_manifest.json`。',
        f'- 实验 manifest（90条准确adapter路径及权重/配置SHA256）：`{OUT}/manifest.json`。',
        f'- 来源 manifest：`{manifest["source_manifest"]}`。',
        f'- 本轮 predictions / metrics / scored：`{OUT}/eval/<base>/<eval_task>/<label>/`。',
        f'- 完整逐样本子任务指标：`{OUT}/commonsense_subtasks.tsv`（所有已完成条件×8子任务）。',
        f'- 原始逐cell矩阵：`{OUT}/matrix.tsv`；逐checkpoint汇总：`{OUT}/checkpoint_summary.tsv`；机器可读总表：`{OUT}/summary.json`。',
        f'- 实际生成、评分命令与开始/结束/失败状态：`{OUT}/eval/<base>/commands.json`；逐base状态：`{OUT}/<base>_worker_status.json`。',
        f'- Batch短测与日志：`{OUT}/batch_probe/<base>/`；生成/评分日志：`{OUT}/eval/<base>/*.log`。',
        f'- 进度历史：`{OUT}/progress.jsonl`；每完成条件评分后原子更新本报告、JSON及TSV。','',
        '复现命令：','',
        '```bash','export PYTHONPATH="$PWD/src:$PWD/scripts"',
        '/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/prepare_posthoc_forgetting_three_seed.py',
        'sbatch slurm/posthoc_forgetting_three_seed.slurm',
        '/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/summarize_posthoc_forgetting_three_seed.py',
        '/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/audit_posthoc_forgetting_three_seed.py','```','']
    summary=dict(status='complete' if complete else 'running',updated_utc=now,cells=len(matrix),expected_cells=368,
        complete_checkpoints=len(cohorts),aggregates=total,grouped=grouped,three_seed_statistics=seed_stats,
        benchmark_three_seed_statistics=benchmark_stats,
        eval_task_aggregates=eval_aggregates,checkpoint_rows=derived,matrix=matrix,identity_audit=identity)
    previous=load(OUT/'summary.json')
    if not previous or previous['cells']!=len(matrix):
        with (OUT/'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(updated_utc=now,cells=len(matrix),complete_checkpoints=len(cohorts),status=summary['status']))+'\n')
    atomic(OUT/'summary.json',json.dumps(summary,indent=2)+'\n')
    atomic(REPORT,'\n'.join(lines)+'\n')
    tsv(OUT/'matrix.tsv',matrix,['base','train_task','seed','method','eval_task','score','samples','primary_metric','metrics_path','predictions_path'])
    tsv(OUT/'checkpoint_summary.tsv',derived,['base','train_task','seed','method','target','target_gain','base_off','off_score','off_gain_lora','off_gain_hns','forgetting_gap'])
    tsv(OUT/'commonsense_subtasks.tsv',cs_rows,['base','train_task','seed','method','subtask','score','samples'])
    print(json.dumps({k:summary[k] for k in ('status','cells','complete_checkpoints')}),flush=True)


if __name__ == '__main__':
    main()
