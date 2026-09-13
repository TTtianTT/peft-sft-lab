#!/usr/bin/env python3
"""Summarize the completed seed43/44 forgetting matrices (descriptive, no CI)."""
import json
import statistics
from pathlib import Path

RUN = Path('/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912')
BASES = ['Qwen3-8B', 'Llama-3.1-8B-Instruct']
TRAIN = ['magicoder', 'metamath', 'tulu']
TASKS = [*TRAIN, 'commonsense']
SETTINGS = ['LoRA', '0+0', '2+0', '2+1', '2+2', '4+0', '4+1', '4+2', '8+0', '8+1', '8+2']


def main():
    rows, matrix, audits = [], [], []
    for base in BASES:
        for seed in (43, 44):
            root = RUN / base / f'seed{seed}'
            records = json.loads((root/'forgetting/eval/score_manifest.json').read_text())['records']
            assert len(records) == 136
            scores = {(r['task'], r['variant']): r[r['primary_metric']]*100 for r in records}
            assert len(scores) == 136
            previous = json.loads((root/'hns_step_grid/eval/score_manifest.json').read_text())['records']
            for old in previous:
                key = (old['task'], old['variant'])
                before = old[old['primary_metric']]*100
                if abs(scores[key]-before) > 1e-9:
                    audits.append(dict(base=base, seed=seed, task=key[0], variant=key[1], old=before, new=scores[key]))
            for task in TASKS:
                matrix.append(dict(base=base, seed=seed, train_task='base', setting='Base', eval_task=task, score=scores[task,'base']))
            for train in TRAIN:
                off = [t for t in TASKS if t != train]
                for setting in SETTINGS:
                    label = f'{train}__original_lora' if setting == 'LoRA' else f'{train}__hns_f{setting.split("+")[0]}_s{setting.split("+")[1]}'
                    score = {task:scores[task,label] for task in TASKS}
                    for task in TASKS:
                        matrix.append(dict(base=base, seed=seed, train_task=train, setting=setting, eval_task=task, score=score[task]))
                    lora = f'{train}__original_lora'
                    control = f'{train}__hns_f0_s0'
                    rows.append(dict(base=base, seed=seed, train_task=train, setting=setting,
                        target=score[train], target_gain=score[train]-scores[train,lora],
                        base_off=statistics.mean(scores[t,'base'] for t in off),
                        off_score=statistics.mean(score[t] for t in off),
                        off_gain=statistics.mean(score[t]-scores[t,lora] for t in off),
                        off_gain_over_control=statistics.mean(score[t]-scores[t,control] for t in off),
                        forgetting_gap=statistics.mean(max(scores[t,'base']-score[t],0) for t in off)))
    def mean(rs,k):
        return statistics.mean(r[k] for r in rs)
    lines = ['# 新增 training seeds 的 HNS 遗忘评测汇总', '',
        '日期：2026-09-13。四组均完成，136 cells/组，共544项；最后一组在20:30:19（新加坡时间）完成。', '',
        '训练任务为 Magicoder、MetaMath、Tulu；评测包含 HumanEval、GSM8K、IFEval、Commonsense-8。没有 Commonsense 新训练 checkpoint。', '',
        'Off-task 为其余三个基准族的等权平均；Commonsense 内部为八子任务等权 macro。遗忘量 = 各 off-task 上 max(Base−adapter, 0) 的均值，越低越好。先逐基准截断，再平均。所有差值单位为百分点。这里只提供描述性结果，未计算 paired CI 或显著性。', '',
        '## 全部固定设置的总体结果', '',
        '等权平均12个模型×训练任务×seed checkpoint。这不是12个独立 training seeds。', '',
        '| 设置 | 本任务 ΔLoRA | Off-task ΔLoRA | Off-task Δ0+0 | Off-task 胜/平/负 | 平均遗忘量 |',
        '|---|---:|---:|---:|---|---:|']
    aggregates=[]
    for setting in SETTINGS:
        rs = [r for r in rows if r['setting']==setting]
        win=sum(r['off_gain']>1e-9 for r in rs)
        tie=sum(abs(r['off_gain'])<=1e-9 for r in rs)
        entry=dict(setting=setting,target_gain=mean(rs,'target_gain'),off_gain=mean(rs,'off_gain'),off_gain_over_control=mean(rs,'off_gain_over_control'),forgetting_gap=mean(rs,'forgetting_gap'),wins=win,ties=tie,losses=12-win-tie)
        aggregates.append(entry)
        lines.append(f"| {setting} | {entry['target_gain']:+.2f} | {entry['off_gain']:+.2f} | {entry['off_gain_over_control']:+.2f} | {win}/{tie}/{12-win-tie} | {entry['forgetting_gap']:.2f} |")
    lines += ['', '## 六组 checkpoint 的两-seed均值', '',
        '| 模型 | 训练任务 | LoRA off-task | LoRA 遗忘量 | 4+0 off-task Δ | 4+0 遗忘量 | 4+1 off-task Δ | 4+1 遗忘量 | 8+2 off-task Δ | 8+2 遗忘量 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for base in BASES:
        for task in TRAIN:
            select=lambda setting:[r for r in rows if r['base']==base and r['train_task']==task and r['setting']==setting]
            lora=select('LoRA')
            cells=[base,task,f"{mean(lora,'off_score'):.2f}",f"{mean(lora,'forgetting_gap'):.2f}"]
            for setting in ('4+0','4+1','8+2'):
                rs=select(setting)
                cells += [f"{mean(rs,'off_gain'):+.2f}",f"{mean(rs,'forgetting_gap'):.2f}"]
            lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## 全部设置的六组 off-task 增益均值', '',
        '| 模型 | 训练任务 | '+' | '.join(SETTINGS[1:])+' |',
        '|---|---|'+'---:|'*10]
    for base in BASES:
        for task in TRAIN:
            cells=[base,task]
            for setting in SETTINGS[1:]:
                rs=[r for r in rows if r['base']==base and r['train_task']==task and r['setting']==setting]
                cells.append(f"{mean(rs,'off_gain'):+.2f}")
            lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## 逐-seed 锚点结果', '',
        '| 模型 | 训练任务 | seed | 设置 | 本任务 ΔLoRA | Off-task ΔLoRA | Off-task Δ0+0 | 遗忘量 |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        if r['setting'] in ('LoRA','4+0','4+1','8+2'):
            lines.append(f"| {r['base']} | {r['train_task']} | {r['seed']} | {r['setting']} | {r['target_gain']:+.2f} | {r['off_gain']:+.2f} | {r['off_gain_over_control']:+.2f} | {r['forgetting_gap']:.2f} |")
    lines += ['', '## 审计与结论边界', '',
        f'本轮新生成的本任务/Base36项×4组与此前 diagonal 结果相比，有{len(audits)}项主指标变化。详情在 summary JSON 的 diagonal_audit。不要把两个评测批次的单点值混用。', '',
        '仅两个新增 training seeds；原始seed42未混入。训练配置差异见 hns_three_seed_diagonal_20260913.md。', '',
        '0+0 重构控制必须保留；HNS相对LoRA的收益不等于纯谱形状收益。当前未包含 norm-matched/calibrated scalar 或 ExactFlatNuclear，因此不能证明优于这些对照，也不能区分尺度与谱形状贡献。', '',
        '所有HNS设置均在同一test set上报告，描述性最优不等于独立验证的最优；本轮未进行显著性检验。', '',
        '原始评测数据：/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/<base>/seed<43|44>/forgetting/eval/。',
        '机器可读汇总：hns_seed_forgetting_20260913.json；包括544项score矩阵、132条逐checkpoint/setting汇总和推理重评差异。']
    print(json.dumps(dict(report='\n'.join(lines)+'\n',aggregates=aggregates,rows=rows,matrix=matrix,diagonal_audit=audits),ensure_ascii=False))


if __name__ == '__main__':
    main()
