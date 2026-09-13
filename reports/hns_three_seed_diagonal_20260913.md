# 三个训练运行的本任务结果与配置审计

日期：2026-09-13。成绩单位均为百分比。原始 checkpoint 使用 2026-09-12 release 的统一实现 step-grid 成绩，不混入 Hugging Face 模型卡中的旧评测成绩。43/44 使用已完成的 hns_step_grid/eval，不使用正在运行的遗忘评测。

发布前二次审计更正：下表 Warmup 是 CLI 请求比例，不是实际生效值。12 次新增运行保存的 training_args.json 均为 warmup_steps=0，且无 warmup_ratio；Transformers 5.16.1 的 TrainingArguments 接口不接受 warmup_ratio，训练脚本按签名过滤参数时丢弃该请求。因此这些训练实际没有 warmup，这是与原始 recipe 的额外差异。Magicoder 50K 为截断过滤前样本数，实际训练 Qwen 49,936 / Llama 49,941；实际更新步数分别为 Magicoder 1561、Qwen MetaMath 4689、Llama MetaMath 198、Tulu 470。

“42（原始）”是原始运行的归档标签；Qwen Magicoder/MetaMath/Tulu 和 Llama Tulu 的本地卡片/参数支持 seed42，Llama Magicoder/MetaMath 的完整原始 seed/Trainer 参数尚未在本次审计中核实。不能只用推理 seed42 推断训练 seed42。

0+0 为 SVD 重构控制；其余为 all-module HNS，保留核范数、strength=1、输出 rank16。— 表示没有新增对应训练 checkpoint，不是零分。

## 完整大表

| 模型 | 本任务基准 | Training seed | LoRA | 0+0 | 2+0 | 2+1 | 2+2 | 4+0 | 4+1 | 4+2 | 8+0 | 8+1 | 8+2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | HumanEval | 42（原始） | 66.46 | 67.68 | 75.61 | 74.39 | 76.22 | 74.39 | 75.00 | 75.61 | 75.00 | 75.00 | 75.61 |
| Qwen3-8B | HumanEval | 43 | 62.80 | 64.02 | 74.39 | 75.00 | 74.39 | 73.78 | 74.39 | 75.61 | 73.78 | 75.61 | 75.61 |
| Qwen3-8B | HumanEval | 44 | 64.63 | 64.63 | 72.56 | 72.56 | 73.17 | 72.56 | 74.39 | 73.78 | 76.22 | 73.78 | 75.00 |
| Qwen3-8B | GSM8K | 42（原始） | 84.15 | 84.53 | 88.25 | 88.25 | 88.25 | 88.25 | 88.02 | 88.17 | 88.40 | 88.55 | 88.48 |
| Qwen3-8B | GSM8K | 43 | 84.00 | 83.93 | 87.41 | 87.72 | 86.96 | 87.19 | 86.81 | 87.19 | 87.11 | 87.19 | 87.11 |
| Qwen3-8B | GSM8K | 44 | 84.00 | 84.08 | 87.49 | 88.02 | 87.57 | 87.49 | 87.41 | 87.26 | 87.72 | 87.87 | 87.57 |
| Qwen3-8B | IFEval | 42（原始） | 67.65 | 67.65 | 70.06 | 70.79 | 70.06 | 71.35 | 70.24 | 70.43 | 70.61 | 70.43 | 68.95 |
| Qwen3-8B | IFEval | 43 | 66.17 | 68.58 | 72.64 | 73.38 | 72.46 | 72.46 | 72.46 | 71.72 | 71.35 | 73.20 | 71.90 |
| Qwen3-8B | IFEval | 44 | 66.91 | 67.84 | 71.16 | 71.16 | 71.72 | 72.46 | 70.98 | 71.53 | 70.43 | 72.09 | 70.98 |
| Qwen3-8B | Commonsense-8 | 42（原始） | 90.79 | 90.79 | 90.23 | 90.30 | 90.30 | 90.31 | 90.32 | 90.27 | 90.30 | 90.26 | 90.27 |
| Qwen3-8B | Commonsense-8 | 43 | — | — | — | — | — | — | — | — | — | — | — |
| Qwen3-8B | Commonsense-8 | 44 | — | — | — | — | — | — | — | — | — | — | — |
| Llama-3.1-8B-Instruct | HumanEval | 42（原始） | 53.66 | 54.27 | 53.66 | 54.27 | 53.66 | 53.66 | 54.88 | 53.66 | 54.27 | 54.88 | 53.66 |
| Llama-3.1-8B-Instruct | HumanEval | 43 | 55.49 | 55.49 | 57.93 | 57.93 | 56.71 | 55.49 | 55.49 | 54.88 | 54.88 | 56.10 | 54.27 |
| Llama-3.1-8B-Instruct | HumanEval | 44 | 53.66 | 54.88 | 54.88 | 53.66 | 55.49 | 54.27 | 53.66 | 54.88 | 56.71 | 53.66 | 55.49 |
| Llama-3.1-8B-Instruct | GSM8K | 42（原始） | 77.10 | 77.33 | 80.89 | 80.82 | 80.29 | 80.89 | 80.67 | 81.05 | 80.67 | 80.14 | 80.82 |
| Llama-3.1-8B-Instruct | GSM8K | 43 | 75.36 | 75.21 | 79.15 | 78.47 | 77.94 | 79.15 | 78.39 | 78.62 | 79.30 | 78.32 | 78.32 |
| Llama-3.1-8B-Instruct | GSM8K | 44 | 74.60 | 75.21 | 79.15 | 78.85 | 77.63 | 79.15 | 79.15 | 77.94 | 79.30 | 78.85 | 78.85 |
| Llama-3.1-8B-Instruct | IFEval | 42（原始） | 63.22 | 63.59 | 65.06 | 65.06 | 64.51 | 65.43 | 64.88 | 65.06 | 64.70 | 64.33 | 65.25 |
| Llama-3.1-8B-Instruct | IFEval | 43 | 62.85 | 63.96 | 65.25 | 64.70 | 63.03 | 63.03 | 65.25 | 62.11 | 63.77 | 63.59 | 63.59 |
| Llama-3.1-8B-Instruct | IFEval | 44 | 63.03 | 65.06 | 63.96 | 63.59 | 63.96 | 64.14 | 64.88 | 63.96 | 64.70 | 64.51 | 65.25 |
| Llama-3.1-8B-Instruct | Commonsense-8 | 42（原始） | 87.96 | 87.95 | 87.41 | 87.56 | 87.58 | 87.52 | 87.52 | 87.53 | 87.46 | 87.66 | 87.58 |
| Llama-3.1-8B-Instruct | Commonsense-8 | 43 | — | — | — | — | — | — | — | — | — | — | — |
| Llama-3.1-8B-Instruct | Commonsense-8 | 44 | — | — | — | — | — | — | — | — | — | — | — |

## 新增 seeds43/44 的实际保存训练参数

已逐项比较六组 run_args.json：43/44 除 seed、输出路径外无差异。每个训练任务单 GPU；下表 micro × accumulation = global batch。

| 模型 | 训练任务 | 样本（截断过滤前） | Epoch | 最大序列长 | Micro × 累积 = Global batch | LR | Warmup 请求值（实际均0） | Scheduler | Adam β2 |
|---|---|---:|---:|---:|---|---|---:|---|---:|
| Qwen3-8B | Magicoder | 50K | 1 | 4096 | 16 × 2 = 32 | 2e-5 | 0.05 | cosine | 0.999 |
| Qwen3-8B | MetaMath | 50K | 3 | 4096 | 16 × 2 = 32 | 1e-4 | 0.05 | cosine | 0.999 |
| Qwen3-8B | Tulu | 全量 29,980 | 2 | 4096 | 16 × 8 = 128 | 4e-4 | 0.03 | cosine | 0.999 |
| Llama-3.1-8B-Instruct | Magicoder | 50K | 1 | 4096 | 16 × 2 = 32 | 2e-5 | 0.05 | cosine | 0.999 |
| Llama-3.1-8B-Instruct | MetaMath | 50K | 3 | 1024 | 64 × 12 = 768 | 1e-4 | 0.10 | cosine_with_min_lr | 0.95 |
| Llama-3.1-8B-Instruct | Tulu | 全量 29,980 | 2 | 1024 | 64 × 2 = 128 | 2e-4 | 0.10 | cosine_with_min_lr | 0.95 |

公共设置：LoRA r16/alpha32/dropout0.05/all-linear；bf16、gradient checkpointing、weight_decay0、grad_clip1、Adam β1=0.9、padding multiple8。Llama 数学/指令 minimum LR ratio=0.01。聊天 SFT，Qwen non_thinking，Llama auto。固定数据子集 seed42；新增全局训练 seed 与 Trainer data_seed 均为43/44。

## 与原始运行的差异及证据边界

| 模型/任务 | 原始记录能支持的配置 | 新增运行的差异 / 未核实项 |
|---|---|---|
| Qwen / Magicoder | 模型卡：50K、epoch1、seq4096、GBS32、LR2e-5、r16、seed42 | 上述主要参数保留；原始 micro-batch、warmup、β、padding 和完整 Trainer 配置未由本次找到的卡片证明一致 |
| Qwen / MetaMath | run_args.json：50K、3ep、seq4096、GBS32、LR1e-4、warmup0.05、cosine、β2=.999、micro1/累积32、dataset_seed42 | 新增 micro16/累积2；padding8；原始数据文件 JSON、新增本地 parquet，子集身份尚未逐项核对；训练 seed42→43/44 |
| Qwen / Tulu | run_args.json：2ep、seq4096、GBS128、LR4e-4、warmup0.03、cosine、β2=.999、micro1/累积128；run_config：29,980 条 | 新增 micro16/累积8；padding8；数据路径不同，逐条身份尚未核对；训练 seed42→43/44 |
| Llama / Magicoder | 本地源 HNS8p2 模型卡：50K、源epoch1、seq4096、GBS32、LR2e-5、r16/alpha32 | 主要记录参数保留；原始 micro-batch、warmup、β2、dropout、训练框架与完整 seed 设置未在本次卡片审计中证明一致 |
| Llama / MetaMath | LoRA 模型卡仅给出 MetaMath 50K、r16 | 新增使用仓库 Llama-matching math profile：3ep、seq1024、GBS768、LR1e-4 等；与原始全部参数是否一致仍未知，不能将 profile 当作原始运行日志 |
| Llama / Tulu | LoRA 模型卡明确 seq1024、GBS128、r16、seed42 | 新增保留上述参数；2ep、LR2e-4、warmup.1、β2=.95 等来自仓库 profile，原始未完整核实 |

micro-batch 和梯度累积改变，即使 global batch 不变，也可能因可变长度样本的 loss 加权、dropout RNG 消耗及数值舍入产生差异。保持 dataset_seed 并不自动证明不同数据文件产生相同子集。

原始 TrainingArguments seed/data_seed 未完整核实；新增运行显式传入二者。不要把“默认可能是42”写成对所有旧 checkpoint 的事实审计结论。

## 评测配置差异

三组均取统一 step-grid 的 greedy 主指标：HumanEval pass@1（164）、GSM8K strict accuracy（1319）、IFEval prompt strict（541）。不是每个 checkpoint 评测三次，而是三个训练运行各评测一次。

- 原始 release 的最终稳定推理 token budget=65,536。新增 Qwen43/44 与 Llama44 为65,536；Llama43 沿用先前成功完成的131,072配置。新增 max_num_seqs=1024。
- Base 是不经过训练的参考，不应作为独立 training seed 成绩。Llama IFEval Base 在新增两组间差一个正确 prompt，推理的逐条完全一致性未建立。
- 模型卡的旧成绩与 release 中的统一重评成绩不同，例如 Qwen Magicoder LoRA 卡片67.07、这里66.46。不能择优混用不同评测协议/实现的结果。

## 统计使用建议

保留本表逐运行分数，seeds43/44 可按同一新增配置报告 mean ± sample SD；原始42单列为历史参考。当前不应将三组均值/标准差宣称为严格同配置的三-training-seed 稳健性结果。SD 不是 CI；训练 seed 重复和 item-level paired bootstrap CI 回答不同问题。

## 可追溯数据

- 原始：reports/hns_release_20260912/data/step_grid/manifests/{qwen3_8b,llama31_8b}_score_manifest.json。
- 新增：/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/<base>/seed<43|44>/hns_step_grid/eval/score_manifest.json。
- 新增训练参数：同一 seed 目录 lora/<task>/run_args.json、training_args.bin。
- 原始配置证据：/dataset1/zailong/models/spectral-surgery/ 对应 LoRA 的 run_args.json/run_config.json/README.md；Llama Magicoder 本次使用对应源 HNS8p2 README 的 Training 部分。
- 表格重建：scripts/summarize_hns_three_seed_diagonal.py。
