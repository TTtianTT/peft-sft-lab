# Flat-Fro / Flat-Nuclear / DG-Hard 三种子 forgetting 评测

更新时间（UTC）：2026-09-13T17:22:20.906673+00:00。状态：全部评分完成；已评分 368/368 cells；完整 off-task checkpoint 18/18。

实验日期：2026-09-13开始，跨午夜继续运行；Slurm数组作业 992。全部生成/评分结束时间（新加坡）：2026-09-14T01:20:46.537283+08:00。

18 个既有 LoRA，90 个五方法 adapters；每个 base 45 variants × 4 基准 + 4 个共享 Base = 184 cells。未重新训练或重建 adapter。训练 seed 为 42/43/44，推理 seed 统一 42。

协议完全复用现有 HNS forgetting 的 evaluator、prompt/chat template、generation 参数、parser 和 metric：HumanEval pass@1（164）、GSM8K strict accuracy（1319）、IFEval prompt-level strict accuracy（541）、Commonsense-8 等权子任务 macro accuracy（22419）。

Off-task = 除训练任务对应基准外的其余三个基准族等权平均。遗忘量 = mean(max(Base − adapter, 0))，先逐基准截断再平均，越低越好；成绩越高越好。所有成绩为百分数，差值和遗忘量为百分点。Commonsense 内部的八个子任务等权，不能用22419样本的 pooled accuracy替代。

HNS 全部固定为 4+1、strength=1、all_modules、保持 nuclear norm，无逐 checkpoint 选优。五方法按同一 checkpoint 连续成组、block_size=5 一起生成。比较全部使用本轮新生成的 LoRA/HNS/Base；此前 diagonal 成绩只用于另存审计。

## 18-checkpoint off-task 总体比较

运行中只汇总所有五方法与 Base 均已完成三个 off-task 的同一批 checkpoint；N 显式列出，不能将中间结果当作最终18项平均。

总体（18 项等权）

| 方法 | N | Off-task | ΔLoRA | ΔHNS | 遗忘量↓ | 遗忘量减少 vs LoRA | 遗忘量减少 vs HNS | Off-task vs LoRA 胜/平/负 | Off-task vs HNS 胜/平/负 |
|---|---|---|---|---|---|---|---|---|---|
| LoRA | 18 | 69.54 | +0.00 | -3.21 | 2.71 | +0.00 | -2.34 | 0/18/0 | 3/0/15 |
| Flat-Fro | 18 | 72.21 | +2.67 | -0.53 | 0.56 | +2.15 | -0.19 | 15/0/3 | 3/0/15 |
| Flat-Nuclear | 18 | 72.66 | +3.13 | -0.08 | 0.36 | +2.35 | +0.00 | 14/0/4 | 8/0/10 |
| DG-Hard | 18 | 69.54 | +0.00 | -3.21 | 2.71 | +0.00 | -2.34 | 0/18/0 | 3/0/15 |
| HNS 4+1 | 18 | 72.74 | +3.21 | +0.00 | 0.37 | +2.34 | +0.00 | 15/0/3 | 0/18/0 |

遗忘量逐checkpoint胜/平/负（更低为胜）：

| 方法 | N | 遗忘量 vs LoRA 胜/平/负 | 遗忘量 vs HNS 胜/平/负 |
|---|---|---|---|
| LoRA | 18 | 0/18/0 | 0/4/14 |
| Flat-Fro | 18 | 14/4/0 | 2/8/8 |
| Flat-Nuclear | 18 | 14/4/0 | 2/11/5 |
| DG-Hard | 18 | 0/18/0 | 0/4/14 |
| HNS 4+1 | 18 | 14/4/0 | 0/18/0 |

### base/Qwen3-8B

| 方法 | N | Off-task | ΔLoRA | ΔHNS | 遗忘量↓ | 遗忘量减少 vs LoRA | 遗忘量减少 vs HNS | Off-task vs LoRA 胜/平/负 | Off-task vs HNS 胜/平/负 |
|---|---|---|---|---|---|---|---|---|---|
| LoRA | 9 | 77.27 | +0.00 | -3.22 | 2.20 | +0.00 | -2.20 | 0/9/0 | 3/0/6 |
| Flat-Fro | 9 | 80.02 | +2.74 | -0.47 | 0.05 | +2.16 | -0.05 | 7/0/2 | 2/0/7 |
| Flat-Nuclear | 9 | 80.52 | +3.25 | +0.03 | 0.00 | +2.20 | +0.00 | 6/0/3 | 5/0/4 |
| DG-Hard | 9 | 77.27 | +0.00 | -3.22 | 2.20 | +0.00 | -2.20 | 0/9/0 | 3/0/6 |
| HNS 4+1 | 9 | 80.49 | +3.22 | +0.00 | 0.00 | +2.20 | +0.00 | 6/0/3 | 0/9/0 |

### base/Llama-3.1-8B-Instruct

| 方法 | N | Off-task | ΔLoRA | ΔHNS | 遗忘量↓ | 遗忘量减少 vs LoRA | 遗忘量减少 vs HNS | Off-task vs LoRA 胜/平/负 | Off-task vs HNS 胜/平/负 |
|---|---|---|---|---|---|---|---|---|---|
| LoRA | 9 | 61.80 | +0.00 | -3.20 | 3.21 | +0.00 | -2.48 | 0/9/0 | 0/0/9 |
| Flat-Fro | 9 | 64.40 | +2.60 | -0.60 | 1.07 | +2.14 | -0.34 | 8/0/1 | 1/0/8 |
| Flat-Nuclear | 9 | 64.81 | +3.00 | -0.19 | 0.73 | +2.49 | +0.01 | 8/0/1 | 3/0/6 |
| DG-Hard | 9 | 61.80 | +0.00 | -3.20 | 3.21 | +0.00 | -2.48 | 0/9/0 | 0/0/9 |
| HNS 4+1 | 9 | 65.00 | +3.20 | +0.00 | 0.73 | +2.48 | +0.00 | 9/0/0 | 0/9/0 |

### train_task/magicoder

| 方法 | N | Off-task | ΔLoRA | ΔHNS | 遗忘量↓ | 遗忘量减少 vs LoRA | 遗忘量减少 vs HNS | Off-task vs LoRA 胜/平/负 | Off-task vs HNS 胜/平/负 |
|---|---|---|---|---|---|---|---|---|---|
| LoRA | 6 | 69.59 | +0.00 | -3.88 | 3.14 | +0.00 | -2.46 | 0/6/0 | 0/0/6 |
| Flat-Fro | 6 | 73.15 | +3.55 | -0.33 | 0.77 | +2.37 | -0.09 | 6/0/0 | 1/0/5 |
| Flat-Nuclear | 6 | 73.23 | +3.64 | -0.24 | 0.64 | +2.51 | +0.05 | 6/0/0 | 1/0/5 |
| DG-Hard | 6 | 69.59 | +0.00 | -3.88 | 3.14 | +0.00 | -2.46 | 0/6/0 | 0/0/6 |
| HNS 4+1 | 6 | 73.47 | +3.88 | +0.00 | 0.68 | +2.46 | +0.00 | 6/0/0 | 0/6/0 |

### train_task/metamath

| 方法 | N | Off-task | ΔLoRA | ΔHNS | 遗忘量↓ | 遗忘量减少 vs LoRA | 遗忘量减少 vs HNS | Off-task vs LoRA 胜/平/负 | Off-task vs HNS 胜/平/负 |
|---|---|---|---|---|---|---|---|---|---|
| LoRA | 6 | 63.36 | +0.00 | -5.57 | 4.52 | +0.00 | -4.16 | 0/6/0 | 0/0/6 |
| Flat-Fro | 6 | 67.87 | +4.50 | -1.07 | 0.81 | +3.71 | -0.45 | 6/0/0 | 0/0/6 |
| Flat-Nuclear | 6 | 68.87 | +5.50 | -0.07 | 0.40 | +4.13 | -0.03 | 6/0/0 | 3/0/3 |
| DG-Hard | 6 | 63.36 | +0.00 | -5.57 | 4.52 | +0.00 | -4.16 | 0/6/0 | 0/0/6 |
| HNS 4+1 | 6 | 68.94 | +5.57 | +0.00 | 0.36 | +4.16 | +0.00 | 6/0/0 | 0/6/0 |

### train_task/tulu

| 方法 | N | Off-task | ΔLoRA | ΔHNS | 遗忘量↓ | 遗忘量减少 vs LoRA | 遗忘量减少 vs HNS | Off-task vs LoRA 胜/平/负 | Off-task vs HNS 胜/平/负 |
|---|---|---|---|---|---|---|---|---|---|
| LoRA | 6 | 75.66 | +0.00 | -0.17 | 0.46 | +0.00 | -0.41 | 0/6/0 | 3/0/3 |
| Flat-Fro | 6 | 75.62 | -0.04 | -0.21 | 0.09 | +0.37 | -0.03 | 3/0/3 | 2/0/4 |
| Flat-Nuclear | 6 | 75.89 | +0.24 | +0.07 | 0.06 | +0.41 | -0.00 | 2/0/4 | 4/0/2 |
| DG-Hard | 6 | 75.66 | +0.00 | -0.17 | 0.46 | +0.00 | -0.41 | 0/6/0 | 3/0/3 |
| HNS 4+1 | 6 | 75.82 | +0.17 | +0.00 | 0.05 | +0.41 | +0.00 | 3/0/3 | 0/6/0 |

## 完整逐种子 off-task 成绩

| Base | 训练任务 | Seed | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS 4+1 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | magicoder | 42 | 77.80 | 81.35 | 81.00 | 77.80 | 81.33 |
| Qwen3-8B | magicoder | 43 | 77.76 | 80.82 | 81.03 | 77.76 | 81.19 |
| Qwen3-8B | magicoder | 44 | 78.55 | 81.02 | 80.75 | 78.55 | 81.26 |
| Qwen3-8B | metamath | 42 | 71.29 | 73.87 | 75.20 | 71.29 | 74.92 |
| Qwen3-8B | metamath | 43 | 69.67 | 74.88 | 76.53 | 69.67 | 76.65 |
| Qwen3-8B | metamath | 44 | 63.51 | 73.93 | 74.75 | 63.51 | 74.26 |
| Qwen3-8B | tulu | 42 | 86.02 | 85.04 | 85.53 | 86.02 | 85.36 |
| Qwen3-8B | tulu | 43 | 85.12 | 85.19 | 85.12 | 85.12 | 84.95 |
| Qwen3-8B | tulu | 44 | 85.74 | 84.07 | 84.80 | 85.74 | 84.48 |
| Llama-3.1-8B-Instruct | magicoder | 42 | 60.20 | 64.19 | 63.70 | 60.20 | 64.81 |
| Llama-3.1-8B-Instruct | magicoder | 43 | 61.09 | 65.36 | 66.62 | 61.09 | 65.94 |
| Llama-3.1-8B-Instruct | magicoder | 44 | 62.16 | 66.12 | 66.30 | 62.16 | 66.31 |
| Llama-3.1-8B-Instruct | metamath | 42 | 57.82 | 61.14 | 63.24 | 57.82 | 63.15 |
| Llama-3.1-8B-Instruct | metamath | 43 | 58.32 | 61.14 | 61.63 | 58.32 | 62.09 |
| Llama-3.1-8B-Instruct | metamath | 44 | 59.58 | 62.25 | 61.84 | 59.58 | 62.55 |
| Llama-3.1-8B-Instruct | tulu | 42 | 63.73 | 65.76 | 66.69 | 63.73 | 66.40 |
| Llama-3.1-8B-Instruct | tulu | 43 | 67.57 | 67.94 | 67.74 | 67.57 | 67.74 |
| Llama-3.1-8B-Instruct | tulu | 44 | 65.76 | 65.70 | 65.50 | 65.76 | 66.00 |

## 完整逐种子遗忘量（越低越好）

| Base | 训练任务 | Seed | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS 4+1 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | magicoder | 42 | 1.90 | 0.00 | 0.00 | 1.90 | 0.00 |
| Qwen3-8B | magicoder | 43 | 1.99 | 0.00 | 0.00 | 1.99 | 0.00 |
| Qwen3-8B | magicoder | 44 | 1.11 | 0.00 | 0.00 | 1.11 | 0.00 |
| Qwen3-8B | metamath | 42 | 1.80 | 0.20 | 0.00 | 1.80 | 0.00 |
| Qwen3-8B | metamath | 43 | 3.52 | 0.00 | 0.00 | 3.52 | 0.00 |
| Qwen3-8B | metamath | 44 | 9.52 | 0.20 | 0.00 | 9.52 | 0.00 |
| Qwen3-8B | tulu | 42 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Qwen3-8B | tulu | 43 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Qwen3-8B | tulu | 44 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Llama-3.1-8B-Instruct | magicoder | 42 | 5.04 | 1.05 | 1.54 | 5.04 | 0.84 |
| Llama-3.1-8B-Instruct | magicoder | 43 | 4.69 | 2.09 | 0.68 | 4.69 | 1.73 |
| Llama-3.1-8B-Instruct | magicoder | 44 | 4.14 | 1.48 | 1.60 | 4.14 | 1.54 |
| Llama-3.1-8B-Instruct | metamath | 42 | 4.71 | 2.21 | 1.32 | 4.71 | 1.82 |
| Llama-3.1-8B-Instruct | metamath | 43 | 4.21 | 1.39 | 0.43 | 4.21 | 0.37 |
| Llama-3.1-8B-Instruct | metamath | 44 | 3.36 | 0.89 | 0.62 | 3.36 | 0.00 |
| Llama-3.1-8B-Instruct | tulu | 42 | 1.81 | 0.20 | 0.34 | 1.81 | 0.32 |
| Llama-3.1-8B-Instruct | tulu | 43 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Llama-3.1-8B-Instruct | tulu | 44 | 0.96 | 0.33 | 0.00 | 0.96 | 0.00 |

## 3-seed mean ± sample SD

SD 使用 n−1 分母。仅在同一 Base×训练任务的三个 seed 全部完成后显示；既有 seed42 与43/44训练 recipe差异、部分seed42训练seed证据不足沿用 source audit 的说明。这里是既有三次训练运行的描述性统计。

| Base | 训练任务 | 统计 | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS 4+1 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | magicoder | Off-task | 78.04 ± 0.45 | 81.07 ± 0.27 | 80.93 ± 0.16 | 78.04 ± 0.45 | 81.26 ± 0.07 |
| Qwen3-8B | metamath | Off-task | 68.16 ± 4.10 | 74.23 ± 0.57 | 75.49 ± 0.93 | 68.16 ± 4.10 | 75.28 ± 1.23 |
| Qwen3-8B | tulu | Off-task | 85.63 ± 0.46 | 84.76 ± 0.61 | 85.15 ± 0.37 | 85.63 ± 0.46 | 84.93 ± 0.44 |
| Llama-3.1-8B-Instruct | magicoder | Off-task | 61.15 ± 0.98 | 65.22 ± 0.97 | 65.54 ± 1.60 | 61.15 ± 0.98 | 65.68 ± 0.78 |
| Llama-3.1-8B-Instruct | metamath | Off-task | 58.57 ± 0.91 | 61.51 ± 0.64 | 62.24 ± 0.88 | 58.57 ± 0.91 | 62.60 ± 0.53 |
| Llama-3.1-8B-Instruct | tulu | Off-task | 65.68 ± 1.92 | 66.47 ± 1.27 | 66.64 ± 1.12 | 65.68 ± 1.92 | 66.72 ± 0.91 |

| Base | 训练任务 | 统计 | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS 4+1 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | magicoder | 遗忘量↓ | 1.67 ± 0.48 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.67 ± 0.48 | 0.00 ± 0.00 |
| Qwen3-8B | metamath | 遗忘量↓ | 4.95 ± 4.05 | 0.14 ± 0.12 | 0.00 ± 0.00 | 4.95 ± 4.05 | 0.00 ± 0.00 |
| Qwen3-8B | tulu | 遗忘量↓ | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Llama-3.1-8B-Instruct | magicoder | 遗忘量↓ | 4.62 ± 0.45 | 1.54 ± 0.52 | 1.27 ± 0.52 | 4.62 ± 0.45 | 1.37 ± 0.47 |
| Llama-3.1-8B-Instruct | metamath | 遗忘量↓ | 4.09 ± 0.69 | 1.49 ± 0.67 | 0.79 ± 0.47 | 4.09 ± 0.69 | 0.73 ± 0.96 |
| Llama-3.1-8B-Instruct | tulu | 遗忘量↓ | 0.92 ± 0.90 | 0.18 ± 0.17 | 0.11 ± 0.19 | 0.92 ± 0.90 | 0.11 ± 0.18 |

## 各评测基准的 3-seed mean ± sample SD

每个Base×训练任务×评测基准分别对42/43/44取均值及sample SD；同一条件三个seed齐全后才显示。

| Base | 训练任务 | 评测基准 | 角色 | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS 4+1 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | magicoder | magicoder | target | 64.23 ± 1.27 | 75.20 ± 2.31 | 74.39 ± 1.22 | 64.23 ± 1.27 | 74.59 ± 0.35 |
| Qwen3-8B | magicoder | metamath | off-task | 83.52 ± 0.84 | 87.64 ± 0.26 | 88.88 ± 0.44 | 83.52 ± 0.84 | 88.91 ± 0.36 |
| Qwen3-8B | magicoder | tulu | off-task | 66.91 ± 1.11 | 71.60 ± 0.95 | 70.36 ± 0.53 | 66.91 ± 1.11 | 71.35 ± 0.37 |
| Qwen3-8B | magicoder | commonsense | off-task | 83.67 ± 0.12 | 83.96 ± 0.10 | 83.53 ± 0.05 | 83.67 ± 0.12 | 83.53 ± 0.11 |
| Qwen3-8B | metamath | magicoder | off-task | 54.27 ± 12.01 | 66.46 ± 1.06 | 70.53 ± 2.88 | 54.27 ± 12.01 | 69.51 ± 3.17 |
| Qwen3-8B | metamath | metamath | target | 84.05 ± 0.09 | 87.79 ± 0.62 | 87.54 ± 0.58 | 84.05 ± 0.09 | 87.47 ± 0.68 |
| Qwen3-8B | metamath | tulu | off-task | 67.04 ± 0.53 | 71.72 ± 0.49 | 71.35 ± 1.21 | 67.04 ± 0.53 | 71.72 ± 0.98 |
| Qwen3-8B | metamath | commonsense | off-task | 83.17 ± 0.23 | 84.49 ± 0.17 | 84.60 ± 0.09 | 83.17 ± 0.23 | 84.60 ± 0.10 |
| Qwen3-8B | tulu | magicoder | off-task | 83.54 ± 1.61 | 80.89 ± 1.96 | 81.91 ± 1.41 | 83.54 ± 1.61 | 81.30 ± 1.41 |
| Qwen3-8B | tulu | metamath | off-task | 88.96 ± 0.23 | 88.86 ± 0.55 | 88.98 ± 0.56 | 88.96 ± 0.23 | 88.96 ± 0.72 |
| Qwen3-8B | tulu | tulu | target | 66.97 ± 0.83 | 69.99 ± 1.17 | 71.66 ± 0.56 | 66.97 ± 0.83 | 71.41 ± 0.91 |
| Qwen3-8B | tulu | commonsense | off-task | 84.39 ± 0.11 | 84.55 ± 0.00 | 84.55 ± 0.08 | 84.39 ± 0.11 | 84.54 ± 0.08 |
| Llama-3.1-8B-Instruct | magicoder | magicoder | target | 54.27 ± 1.06 | 54.27 ± 1.06 | 54.88 ± 0.61 | 54.27 ± 1.06 | 54.88 ± 1.22 |
| Llama-3.1-8B-Instruct | magicoder | metamath | off-task | 61.94 ± 4.99 | 66.24 ± 3.73 | 66.62 ± 3.98 | 61.94 ± 4.99 | 67.40 ± 3.30 |
| Llama-3.1-8B-Instruct | magicoder | tulu | off-task | 51.08 ± 3.70 | 57.92 ± 2.52 | 58.53 ± 1.39 | 51.08 ± 3.70 | 58.16 ± 1.99 |
| Llama-3.1-8B-Instruct | magicoder | commonsense | off-task | 70.43 ± 1.17 | 71.52 ± 1.32 | 71.47 ± 0.98 | 70.43 ± 1.17 | 71.49 ± 0.97 |
| Llama-3.1-8B-Instruct | metamath | magicoder | off-task | 54.67 ± 0.70 | 55.69 ± 1.27 | 55.49 ± 4.27 | 54.67 ± 0.70 | 56.30 ± 4.58 |
| Llama-3.1-8B-Instruct | metamath | metamath | target | 75.56 ± 1.47 | 78.49 ± 1.60 | 79.35 ± 1.35 | 75.56 ± 1.47 | 79.45 ± 1.24 |
| Llama-3.1-8B-Instruct | metamath | tulu | off-task | 56.75 ± 2.50 | 58.47 ± 0.95 | 60.14 ± 0.47 | 56.75 ± 2.50 | 60.38 ± 1.98 |
| Llama-3.1-8B-Instruct | metamath | commonsense | off-task | 64.30 ± 0.66 | 70.37 ± 1.16 | 71.09 ± 1.25 | 64.30 ± 0.66 | 71.11 ± 1.27 |
| Llama-3.1-8B-Instruct | tulu | magicoder | off-task | 63.62 ± 0.70 | 63.01 ± 0.70 | 62.80 ± 1.06 | 63.62 ± 0.70 | 62.80 ± 0.61 |
| Llama-3.1-8B-Instruct | tulu | metamath | off-task | 60.50 ± 4.00 | 64.52 ± 3.23 | 65.68 ± 2.14 | 60.50 ± 4.00 | 65.81 ± 1.67 |
| Llama-3.1-8B-Instruct | tulu | tulu | target | 63.15 ± 0.11 | 64.63 ± 0.28 | 64.26 ± 0.95 | 63.15 ± 0.11 | 65.19 ± 0.28 |
| Llama-3.1-8B-Instruct | tulu | commonsense | off-task | 72.94 ± 1.43 | 71.88 ± 0.94 | 71.44 ± 0.98 | 72.94 ± 1.43 | 71.54 ± 1.00 |

## 完整逐种子四基准矩阵

对应训练任务的基准行是 target，其余三行是 off-task；缺失结果明确标为待完成。

| Base | 训练任务 | Seed | 评测基准 | 角色 | Base score | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS 4+1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | magicoder | 42 | magicoder | target | 66.46 | 65.24 | 76.83 | 75.61 | 65.24 | 75.00 |
| Qwen3-8B | magicoder | 42 | metamath | off-task | 85.75 | 83.93 | 87.34 | 89.39 | 83.93 | 88.63 |
| Qwen3-8B | magicoder | 42 | tulu | off-task | 69.69 | 65.80 | 72.64 | 70.06 | 65.80 | 71.72 |
| Qwen3-8B | magicoder | 42 | commonsense | off-task | 82.68 | 83.66 | 84.07 | 83.56 | 83.66 | 83.65 |
| Qwen3-8B | magicoder | 43 | magicoder | target | 66.46 | 62.80 | 76.22 | 73.17 | 62.80 | 74.39 |
| Qwen3-8B | magicoder | 43 | metamath | off-task | 85.75 | 82.56 | 87.79 | 88.63 | 82.56 | 88.78 |
| Qwen3-8B | magicoder | 43 | tulu | off-task | 69.69 | 66.91 | 70.79 | 70.98 | 66.91 | 71.35 |
| Qwen3-8B | magicoder | 43 | commonsense | off-task | 82.68 | 83.80 | 83.88 | 83.48 | 83.80 | 83.45 |
| Qwen3-8B | magicoder | 44 | magicoder | target | 66.46 | 64.63 | 72.56 | 74.39 | 64.63 | 74.39 |
| Qwen3-8B | magicoder | 44 | metamath | off-task | 85.75 | 84.08 | 87.79 | 88.63 | 84.08 | 89.31 |
| Qwen3-8B | magicoder | 44 | tulu | off-task | 69.69 | 68.02 | 71.35 | 70.06 | 68.02 | 70.98 |
| Qwen3-8B | magicoder | 44 | commonsense | off-task | 82.68 | 83.56 | 83.93 | 83.55 | 83.56 | 83.48 |
| Qwen3-8B | metamath | 42 | magicoder | off-task | 66.46 | 64.02 | 65.85 | 68.29 | 64.02 | 67.68 |
| Qwen3-8B | metamath | 42 | metamath | target | 85.75 | 84.15 | 87.64 | 88.17 | 84.15 | 88.17 |
| Qwen3-8B | metamath | 42 | tulu | off-task | 69.69 | 66.73 | 71.35 | 72.64 | 66.73 | 72.46 |
| Qwen3-8B | metamath | 42 | commonsense | off-task | 82.68 | 83.13 | 84.40 | 84.66 | 83.13 | 84.62 |
| Qwen3-8B | metamath | 43 | magicoder | off-task | 66.46 | 57.93 | 67.68 | 73.78 | 57.93 | 73.17 |
| Qwen3-8B | metamath | 43 | metamath | target | 85.75 | 84.00 | 87.26 | 87.04 | 84.00 | 86.81 |
| Qwen3-8B | metamath | 43 | tulu | off-task | 69.69 | 67.65 | 72.27 | 71.16 | 67.65 | 72.09 |
| Qwen3-8B | metamath | 43 | commonsense | off-task | 82.68 | 83.42 | 84.69 | 84.65 | 83.42 | 84.69 |
| Qwen3-8B | metamath | 44 | magicoder | off-task | 66.46 | 40.85 | 65.85 | 69.51 | 40.85 | 67.68 |
| Qwen3-8B | metamath | 44 | metamath | target | 85.75 | 84.00 | 88.48 | 87.41 | 84.00 | 87.41 |
| Qwen3-8B | metamath | 44 | tulu | off-task | 69.69 | 66.73 | 71.53 | 70.24 | 66.73 | 70.61 |
| Qwen3-8B | metamath | 44 | commonsense | off-task | 82.68 | 82.96 | 84.40 | 84.50 | 82.96 | 84.49 |
| Qwen3-8B | tulu | 42 | magicoder | off-task | 66.46 | 84.76 | 82.32 | 83.54 | 84.76 | 82.93 |
| Qwen3-8B | tulu | 42 | metamath | off-task | 85.75 | 89.01 | 88.25 | 88.55 | 89.01 | 88.70 |
| Qwen3-8B | tulu | 42 | tulu | target | 69.69 | 67.84 | 69.32 | 71.16 | 67.84 | 70.79 |
| Qwen3-8B | tulu | 42 | commonsense | off-task | 82.68 | 84.29 | 84.55 | 84.49 | 84.29 | 84.46 |
| Qwen3-8B | tulu | 43 | magicoder | off-task | 66.46 | 81.71 | 81.71 | 81.10 | 81.71 | 80.49 |
| Qwen3-8B | tulu | 43 | metamath | off-task | 85.75 | 89.16 | 89.31 | 89.61 | 89.16 | 89.76 |
| Qwen3-8B | tulu | 43 | tulu | target | 69.69 | 66.17 | 71.35 | 72.27 | 66.17 | 72.46 |
| Qwen3-8B | tulu | 43 | commonsense | off-task | 82.68 | 84.51 | 84.54 | 84.64 | 84.51 | 84.61 |
| Qwen3-8B | tulu | 44 | magicoder | off-task | 66.46 | 84.15 | 78.66 | 81.10 | 84.15 | 80.49 |
| Qwen3-8B | tulu | 44 | metamath | off-task | 85.75 | 88.70 | 89.01 | 88.78 | 88.70 | 88.40 |
| Qwen3-8B | tulu | 44 | tulu | target | 69.69 | 66.91 | 69.32 | 71.53 | 66.91 | 70.98 |
| Qwen3-8B | tulu | 44 | commonsense | off-task | 82.68 | 84.36 | 84.55 | 84.51 | 84.36 | 84.54 |
| Llama-3.1-8B-Instruct | magicoder | 42 | magicoder | target | 52.44 | 53.66 | 53.05 | 54.88 | 53.66 | 54.88 |
| Llama-3.1-8B-Instruct | magicoder | 42 | metamath | off-task | 62.40 | 56.25 | 61.94 | 62.17 | 56.25 | 63.61 |
| Llama-3.1-8B-Instruct | magicoder | 42 | tulu | off-task | 61.92 | 55.27 | 60.63 | 58.60 | 55.27 | 60.44 |
| Llama-3.1-8B-Instruct | magicoder | 42 | commonsense | off-task | 71.40 | 69.08 | 70.00 | 70.34 | 69.08 | 70.37 |
| Llama-3.1-8B-Instruct | magicoder | 43 | magicoder | target | 52.44 | 55.49 | 54.88 | 54.27 | 55.49 | 56.10 |
| Llama-3.1-8B-Instruct | magicoder | 43 | metamath | off-task | 62.40 | 63.99 | 68.08 | 67.85 | 63.99 | 68.92 |
| Llama-3.1-8B-Instruct | magicoder | 43 | tulu | off-task | 61.92 | 48.24 | 55.64 | 59.89 | 48.24 | 56.75 |
| Llama-3.1-8B-Instruct | magicoder | 43 | commonsense | off-task | 71.40 | 71.02 | 72.36 | 72.11 | 71.02 | 72.15 |
| Llama-3.1-8B-Instruct | magicoder | 44 | magicoder | target | 52.44 | 53.66 | 54.88 | 55.49 | 53.66 | 53.66 |
| Llama-3.1-8B-Instruct | magicoder | 44 | metamath | off-task | 62.40 | 65.58 | 68.69 | 69.83 | 65.58 | 69.67 |
| Llama-3.1-8B-Instruct | magicoder | 44 | tulu | off-task | 61.92 | 49.72 | 57.49 | 57.12 | 49.72 | 57.30 |
| Llama-3.1-8B-Instruct | magicoder | 44 | commonsense | off-task | 71.40 | 71.19 | 72.20 | 71.96 | 71.19 | 71.94 |
| Llama-3.1-8B-Instruct | metamath | 42 | magicoder | off-task | 52.44 | 54.27 | 56.71 | 60.37 | 54.27 | 61.59 |
| Llama-3.1-8B-Instruct | metamath | 42 | metamath | target | 62.40 | 77.26 | 79.98 | 80.89 | 77.26 | 80.82 |
| Llama-3.1-8B-Instruct | metamath | 42 | tulu | off-task | 61.92 | 54.16 | 57.67 | 59.70 | 54.16 | 58.23 |
| Llama-3.1-8B-Instruct | metamath | 42 | commonsense | off-task | 71.40 | 65.03 | 69.04 | 69.65 | 65.03 | 69.65 |
| Llama-3.1-8B-Instruct | metamath | 43 | magicoder | off-task | 52.44 | 54.27 | 54.27 | 52.44 | 54.27 | 53.66 |
| Llama-3.1-8B-Instruct | metamath | 43 | metamath | target | 62.40 | 74.83 | 76.80 | 78.77 | 74.83 | 78.39 |
| Llama-3.1-8B-Instruct | metamath | 43 | tulu | off-task | 61.92 | 56.93 | 58.23 | 60.63 | 56.93 | 60.81 |
| Llama-3.1-8B-Instruct | metamath | 43 | commonsense | off-task | 71.40 | 63.75 | 70.94 | 71.82 | 63.75 | 71.81 |
| Llama-3.1-8B-Instruct | metamath | 44 | magicoder | off-task | 52.44 | 55.49 | 56.10 | 53.66 | 55.49 | 53.66 |
| Llama-3.1-8B-Instruct | metamath | 44 | metamath | target | 62.40 | 74.60 | 78.70 | 78.39 | 74.60 | 79.15 |
| Llama-3.1-8B-Instruct | metamath | 44 | tulu | off-task | 61.92 | 59.15 | 59.52 | 60.07 | 59.15 | 62.11 |
| Llama-3.1-8B-Instruct | metamath | 44 | commonsense | off-task | 71.40 | 64.11 | 71.14 | 71.80 | 64.11 | 71.89 |
| Llama-3.1-8B-Instruct | tulu | 42 | magicoder | off-task | 52.44 | 62.80 | 62.20 | 63.41 | 62.80 | 62.80 |
| Llama-3.1-8B-Instruct | tulu | 42 | metamath | off-task | 62.40 | 57.09 | 64.29 | 66.26 | 57.09 | 65.96 |
| Llama-3.1-8B-Instruct | tulu | 42 | tulu | target | 61.92 | 63.22 | 64.70 | 63.22 | 63.22 | 65.43 |
| Llama-3.1-8B-Instruct | tulu | 42 | commonsense | off-task | 71.40 | 71.29 | 70.81 | 70.39 | 71.29 | 70.45 |
| Llama-3.1-8B-Instruct | tulu | 43 | magicoder | off-task | 52.44 | 64.02 | 63.41 | 63.41 | 64.02 | 63.41 |
| Llama-3.1-8B-Instruct | tulu | 43 | metamath | off-task | 62.40 | 64.90 | 67.85 | 67.48 | 64.90 | 67.40 |
| Llama-3.1-8B-Instruct | tulu | 43 | tulu | target | 61.92 | 63.22 | 64.88 | 64.51 | 63.22 | 65.25 |
| Llama-3.1-8B-Instruct | tulu | 43 | commonsense | off-task | 71.40 | 73.79 | 72.55 | 72.32 | 73.79 | 72.42 |
| Llama-3.1-8B-Instruct | tulu | 44 | magicoder | off-task | 52.44 | 64.02 | 63.41 | 61.59 | 64.02 | 62.20 |
| Llama-3.1-8B-Instruct | tulu | 44 | metamath | off-task | 62.40 | 59.51 | 61.41 | 63.31 | 59.51 | 64.06 |
| Llama-3.1-8B-Instruct | tulu | 44 | tulu | target | 61.92 | 63.03 | 64.33 | 65.06 | 63.03 | 64.88 |
| Llama-3.1-8B-Instruct | tulu | 44 | commonsense | off-task | 71.40 | 73.73 | 72.28 | 71.60 | 73.73 | 71.75 |

## 按评测基准汇总 off-task 成绩

前三个基准只纳入另外两种训练任务的 checkpoints（每方法12项）；Commonsense纳入全部18项。运行中只纳入本基准五方法均已完成的同一批checkpoint，并列出N；这是单基准阶段结果，不代表三个off-task已齐全。

| 评测基准 | N | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS 4+1 |
|---|---|---|---|---|---|---|
| magicoder | 12 | 64.02 | 66.51 | 67.68 | 64.02 | 67.48 |

| 评测基准 | N | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS 4+1 |
|---|---|---|---|---|---|---|
| metamath | 12 | 73.73 | 76.81 | 77.54 | 73.73 | 77.77 |

| 评测基准 | N | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS 4+1 |
|---|---|---|---|---|---|---|
| tulu | 12 | 60.44 | 64.93 | 65.10 | 60.44 | 65.40 |

| 评测基准 | N | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS 4+1 |
|---|---|---|---|---|---|---|
| commonsense | 18 | 76.48 | 77.80 | 77.78 | 76.48 | 77.80 |

## DG-Hard identity 与重评审计

原始 DG-Hard 在全部4284模块中完整谱 median=0、threshold=0，retained rank=16、unchanged module fraction=100%、保留Frobenius/nuclear norm=100%；本轮重新加载核实原始A/B tensor逐字节一致（safetensors文件序列化SHA256可不同）。未使用DG-Hard-Active。完整谱退化情况和逐模块数值见此前 diagonal 报告及 dg_build_complete.json。

本轮 token/metric identity 审计：pass；439974 成对样本，token差异 0，主指标差异 0。

HumanEval先行配对审计已完成：18 对checkpoint、2952 样本，token差异 0，主指标差异 0。

相对前轮统一五方法 diagonal 评测，有 0/96 个主指标变化；详见 diagonal_audit.json，主比较未混用历史分数。

最终代码/数据/adapter指纹审计：pass；90 个adapters、11 个数据文件。

完整数据覆盖审计：pass；368 个cells、2248756 条生成记录，所有条件的唯一ID及输入记录均与本轮Base对齐。

GPU资源实测（Slurm sstat，详细时间与原始字段见job_manifest）：992_0.batch: GPU利用率 97%，显存 260.16 GiB；992_1.batch: GPU利用率 97%，显存 259.46 GiB。全程最多两个单卡B300任务。

## 数据、路径、命令与进度记录

- 完整数据覆盖审计：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/coverage_audit.json`；逐样本token配对：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/identity_audit.json`。
- 代码/数据/全部adapter指纹审计：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/artifact_integrity_audit.json`。
- Slurm提交与最多两张GPU配置：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/job_manifest.json`。
- 实验 manifest（90条准确adapter路径及权重/配置SHA256）：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/manifest.json`。
- 来源 manifest：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/source_manifest.json`。
- 本轮 predictions / metrics / scored：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/eval/<base>/<eval_task>/<label>/`。
- 完整逐样本子任务指标：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/commonsense_subtasks.tsv`（所有已完成条件×8子任务）。
- 原始逐cell矩阵：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/matrix.tsv`；逐checkpoint汇总：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/checkpoint_summary.tsv`；机器可读总表：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/summary.json`。
- 实际生成、评分命令与开始/结束/失败状态：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/eval/<base>/commands.json`；逐base状态：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/<base>_worker_status.json`。
- Batch短测与日志：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/batch_probe/<base>/`；生成/评分日志：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/eval/<base>/*.log`。
- 进度历史：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_forgetting_three_seed_20260913/progress.jsonl`；每完成条件评分后原子更新本报告、JSON及TSV。

复现命令：

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts"
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/prepare_posthoc_forgetting_three_seed.py
sbatch slurm/posthoc_forgetting_three_seed.slurm
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/summarize_posthoc_forgetting_three_seed.py
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/audit_posthoc_forgetting_three_seed.py
```

