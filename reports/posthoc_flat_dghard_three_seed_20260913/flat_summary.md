# Post-hoc Flat / DG-Hard 三组训练运行 diagonal evaluation

状态：flat 阶段全部完成。18 checkpoints；Flat-Fro / Flat-Nuclear 各18个完整 diagonal cells。成绩为百分比，差值为百分点。

HNS 统一代表配置：4+1、all modules、strength=1、preserve_nuclear_norm=true、输出 rank16。LoRA / HNS 均从三种子报告指定的统一 step-grid score manifests 读取，不使用旧模型卡或逐 checkpoint 最优值。

协议：完全复用 scripts/eval_forgetting_matrix_vllm.py 和 scripts/score_forgetting_matrix.py；HumanEval strict_continuation chat / pass@1（164）；GSM8K strict accuracy（1319）；IFEval prompt strict（541）。non_thinking chat、greedy temperature0/top_p1、推理 seed42，生成上限分别512/512/2048，max_model_len4096。VLLM_BATCH_INVARIANT=1、FLASH_ATTN、async_scheduling=false、prefix caching=false。

统计边界：原始42是历史归档标签；Llama Magicoder/MetaMath 原始训练 seed 未完整核实。原始与43/44训练 recipe 存在报告已记录的差异，因此下列 mean±sample SD 是三组现有训练运行的描述统计，不能声称为严格同 recipe 的三训练种子重复。跨 benchmark 的总平均是18 cells 等权宏平均，不能解释为统一指标或合并样本准确率。

## 逐 checkpoint

| Base | Task | Seed | LoRA | Flat-Fro | Flat-Nuclear | HNS |
| ---- | ---- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | magicoder | 42 | 66.46 | 76.83 | 75.61 | 75.00 |
| Qwen3-8B | metamath | 42 | 84.15 | 87.64 | 88.17 | 88.02 |
| Qwen3-8B | tulu | 42 | 67.65 | 69.32 | 71.16 | 70.24 |
| Qwen3-8B | magicoder | 43 | 62.80 | 76.22 | 73.17 | 74.39 |
| Qwen3-8B | metamath | 43 | 84.00 | 87.26 | 87.04 | 86.81 |
| Qwen3-8B | tulu | 43 | 66.17 | 71.35 | 72.27 | 72.46 |
| Qwen3-8B | magicoder | 44 | 64.63 | 72.56 | 74.39 | 74.39 |
| Qwen3-8B | metamath | 44 | 84.00 | 88.48 | 87.41 | 87.41 |
| Qwen3-8B | tulu | 44 | 66.91 | 69.32 | 71.53 | 70.98 |
| Llama-3.1-8B-Instruct | magicoder | 42 | 53.66 | 53.05 | 54.88 | 54.88 |
| Llama-3.1-8B-Instruct | metamath | 42 | 77.10 | 79.98 | 80.89 | 80.67 |
| Llama-3.1-8B-Instruct | tulu | 42 | 63.22 | 64.70 | 63.22 | 64.88 |
| Llama-3.1-8B-Instruct | magicoder | 43 | 55.49 | 54.88 | 54.27 | 55.49 |
| Llama-3.1-8B-Instruct | metamath | 43 | 75.36 | 76.80 | 78.77 | 78.39 |
| Llama-3.1-8B-Instruct | tulu | 43 | 62.85 | 64.88 | 64.51 | 65.25 |
| Llama-3.1-8B-Instruct | magicoder | 44 | 53.66 | 54.88 | 55.49 | 53.66 |
| Llama-3.1-8B-Instruct | metamath | 44 | 74.60 | 78.70 | 78.39 | 79.15 |
| Llama-3.1-8B-Instruct | tulu | 44 | 63.03 | 64.33 | 65.06 | 64.88 |

## 18 checkpoints 宏平均和胜负

| Method | Mean | Δ vs LoRA | Δ vs HNS | vs LoRA W/T/L | vs HNS W/T/L |
| --- | ---: | ---: | ---: | --- | --- |
| LoRA | 68.0982 | +0.0000 | -3.9544 | 0/18/0 | 0/2/16 |
| Flat-Fro | 71.7311 | +3.6329 | -0.3216 | 16/0/2 | 5/0/13 |
| Flat-Nuclear | 72.0139 | +3.9157 | -0.0387 | 16/1/1 | 9/3/6 |
| HNS | 72.0526 | +3.9544 | +0.0000 | 16/2/0 | 0/18/0 |

## Per-base 平均

| Group | LoRA | Flat-Fro | Flat-Nuclear | HNS |
| --- | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | 64.3295 | 65.7985 | 66.1649 | 66.3605 |
| Qwen3-8B | 71.8669 | 77.6636 | 77.8629 | 77.7448 |

## Per-task 平均

| Group | LoRA | Flat-Fro | Flat-Nuclear | HNS |
| --- | ---: | ---: | ---: | ---: |
| magicoder | 59.4512 | 64.7358 | 64.6341 | 64.6341 |
| metamath | 79.8711 | 83.1438 | 83.4471 | 83.4091 |
| tulu | 64.9723 | 67.3136 | 67.9606 | 68.1146 |

## 3-seed mean ± sample SD（ddof=1）

| Base | Task | LoRA | Flat-Fro | Flat-Nuclear | HNS |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | magicoder | 54.27 ± 1.06 | 54.27 ± 1.06 | 54.88 ± 0.61 | 54.67 ± 0.93 |
| Llama-3.1-8B-Instruct | metamath | 75.69 ± 1.28 | 78.49 ± 1.60 | 79.35 ± 1.35 | 79.40 ± 1.16 |
| Llama-3.1-8B-Instruct | tulu | 63.03 ± 0.18 | 64.63 ± 0.28 | 64.26 ± 0.95 | 65.00 ± 0.21 |
| Qwen3-8B | magicoder | 64.63 ± 1.83 | 75.20 ± 2.31 | 74.39 ± 1.22 | 74.59 ± 0.35 |
| Qwen3-8B | metamath | 84.05 ± 0.09 | 87.79 ± 0.62 | 87.54 ± 0.58 | 87.41 ± 0.61 |
| Qwen3-8B | tulu | 66.91 ± 0.74 | 69.99 ± 1.17 | 71.66 ± 0.56 | 71.23 ± 1.13 |

## 数值审计

Flat-Fro: t=||σ||₂/√r；Flat-Nuclear: t=||σ||₁/r。两者均使用 shared compact SVD 和 B=U√t、A=√t Vᵀ。所有 source config 字节保持一致，未改变 scaling、rank、alpha、target modules 或 base model；保存权重重载后逐 tensor 验证。
- flat_fro: max target budget relative error=1.19209048e-07; max saved budget relative error=4.93352218e-07.
- flat_nuclear: max target budget relative error=1.4863501e-07; max saved budget relative error=5.83788676e-07.

已有短测试与完整评测的逐 token 重复性检查：0 / 8112 个重复样本输出不同；详细计数见 summary JSON。

HumanEval 日志存在 multiprocessing 临时目录清理回调的 NFS `.nfs*` busy 异常，既有 HNS seed-eval 日志也出现相同现象。检查 human_eval/execution.py 确认测试结果先写入 manager result 再清理；主评分正常完成。汇总逐 cell 验证全部164条执行结果和 scored rows，无遗漏；未修改执行 timeout、parser 或 metric。

## 运行顺序与命令

最多两个单 GPU Slurm worker（array=0-1%2，gres=gpu:1）；每个 worker 一个 base，所有训练任务和种子串行/同引擎批量评测。先做 max_num_seqs2048 的256样本/任务探测，再完整评测；OOM 才依次降为1024/512/256。max_num_batched_tokens65536、adapter block6、prompt chunk1024，使六个 Flat adapters 可在同一 batch 中评测。实际选定配置见 generation_manifest，探测日志见 batch_probe。

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/audit_posthoc_three_seed.py
PYTHONPATH=src /dataset1/zailong/envs/peft-sft-lab/bin/python -m pytest -q tests/test_posthoc_flat.py
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/build_posthoc_flat_three_seed.py
sbatch slurm/posthoc_flat_three_seed.slurm
PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/summarize_posthoc_flat_dghard_three_seed.py --phase flat
```

逐条真实 generation/scoring argv 见 eval/<phase>/<base>/commands.json；Slurm 提交回执见 submission.json。

实际 Slurm 提交：`{"flat": {"job_id": 972, "command": ["sbatch", "slurm/posthoc_flat_three_seed.slurm"], "max_gpus": 2, "gpus_per_worker": 1, "failed_attempts": [{"job_id": 970, "reason": "node01 /tmp full during torch.compile; no predictions generated"}]}}`。首次 job970 因 node01 /tmp 满在引擎启动阶段失败，没有生成评测结果；job972 改用 workspace 临时缓存路径后重试。

## Source / adapter 路径和 manifest

完整 source audit: `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/source_manifest.json`。Flat build: `flat_build_complete.json`。先行 Flat 结果快照: `flat_summary.json` / `flat_summary.md`。所有结果 JSON 保留未四舍五入分数。

| Base | Task | Seed | Source checkpoint | Adapter parent |
| --- | --- | ---: | --- | --- |
| Qwen3-8B | magicoder | 42 | `/dataset1/zailong/models/spectral-surgery/Qwen3-8B-Magicoder-50K-LoRA-E1` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Qwen3-8B/seed42/magicoder` |
| Qwen3-8B | metamath | 42 | `/dataset1/zailong/models/spectral-surgery/Qwen3-8B-MetaMathQA-50K-LoRA` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Qwen3-8B/seed42/metamath` |
| Qwen3-8B | tulu | 42 | `/dataset1/zailong/models/spectral-surgery/Qwen3-8B-InstructionFollowing-LoRA` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Qwen3-8B/seed42/tulu` |
| Qwen3-8B | magicoder | 43 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Qwen3-8B/seed43/lora/magicoder` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Qwen3-8B/seed43/magicoder` |
| Qwen3-8B | metamath | 43 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Qwen3-8B/seed43/lora/metamath` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Qwen3-8B/seed43/metamath` |
| Qwen3-8B | tulu | 43 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Qwen3-8B/seed43/lora/tulu` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Qwen3-8B/seed43/tulu` |
| Qwen3-8B | magicoder | 44 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Qwen3-8B/seed44/lora/magicoder` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Qwen3-8B/seed44/magicoder` |
| Qwen3-8B | metamath | 44 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Qwen3-8B/seed44/lora/metamath` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Qwen3-8B/seed44/metamath` |
| Qwen3-8B | tulu | 44 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Qwen3-8B/seed44/lora/tulu` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Qwen3-8B/seed44/tulu` |
| Llama-3.1-8B-Instruct | magicoder | 42 | `/dataset1/zailong/models/spectral-surgery/Llama-3.1-8B-Instruct-Magicoder-50K-LoRA-E1` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Llama-3.1-8B-Instruct/seed42/magicoder` |
| Llama-3.1-8B-Instruct | metamath | 42 | `/dataset1/zailong/models/spectral-surgery/Llama-3.1-8B-Instruct-MetaMathQA-50K-LoRA` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Llama-3.1-8B-Instruct/seed42/metamath` |
| Llama-3.1-8B-Instruct | tulu | 42 | `/dataset1/zailong/models/spectral-surgery/Llama-3.1-8B-Instruct-InstructionFollowing-LoRA` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Llama-3.1-8B-Instruct/seed42/tulu` |
| Llama-3.1-8B-Instruct | magicoder | 43 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Llama-3.1-8B-Instruct/seed43/lora/magicoder` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Llama-3.1-8B-Instruct/seed43/magicoder` |
| Llama-3.1-8B-Instruct | metamath | 43 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Llama-3.1-8B-Instruct/seed43/lora/metamath` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Llama-3.1-8B-Instruct/seed43/metamath` |
| Llama-3.1-8B-Instruct | tulu | 43 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Llama-3.1-8B-Instruct/seed43/lora/tulu` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Llama-3.1-8B-Instruct/seed43/tulu` |
| Llama-3.1-8B-Instruct | magicoder | 44 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Llama-3.1-8B-Instruct/seed44/lora/magicoder` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Llama-3.1-8B-Instruct/seed44/magicoder` |
| Llama-3.1-8B-Instruct | metamath | 44 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Llama-3.1-8B-Instruct/seed44/lora/metamath` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Llama-3.1-8B-Instruct/seed44/metamath` |
| Llama-3.1-8B-Instruct | tulu | 44 | `/dataset1/zailong/runs/peft-sft-lab/hns-training-seeds-3task-20260912/Llama-3.1-8B-Instruct/seed44/lora/tulu` | `/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/adapters/Llama-3.1-8B-Instruct/seed44/tulu` |

每个 Adapter parent 下为 `flat_fro/`、`flat_nuclear/`。Variant manifests: `<base>_flat_variant_manifest.json`。

Source manifest 同时记录原始 variant/score/generation/task config 路径、SHA256、完整 adapter config、module shapes 和 LoRA/HNS metric records。
