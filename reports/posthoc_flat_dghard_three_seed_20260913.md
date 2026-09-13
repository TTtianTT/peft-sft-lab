# Post-hoc Flat / DG-Hard 三组训练运行 diagonal evaluation

状态：final 阶段全部完成。18 checkpoints；Flat-Fro / Flat-Nuclear 各18个完整 diagonal cells；DG-Hard 18个完整 diagonal cells。成绩为百分比，差值为百分点。

HNS 统一代表配置：4+1、all modules、strength=1、preserve_nuclear_norm=true、输出 rank16。不使用旧模型卡或逐 checkpoint 最优值。最终主表的全部五种方法来自本次 joint 统一复评；历史 LoRA/HNS 参考成绩仅保留在 source manifest 和先行 Flat/DG 快照中。

协议：完全复用 scripts/eval_forgetting_matrix_vllm.py 和 scripts/score_forgetting_matrix.py；HumanEval strict_continuation chat / pass@1（164）；GSM8K strict accuracy（1319）；IFEval prompt strict（541）。non_thinking chat、greedy temperature0/top_p1、推理 seed42，生成上限分别512/512/2048，max_model_len4096。VLLM_BATCH_INVARIANT=1、FLASH_ATTN、async_scheduling=false、prefix caching=false。

统计边界：原始42是历史归档标签；Llama Magicoder/MetaMath 原始训练 seed 未完整核实。原始与43/44训练 recipe 存在报告已记录的差异，因此下列 mean±sample SD 是三组现有训练运行的描述统计，不能声称为严格同 recipe 的三训练种子重复。跨 benchmark 的总平均是18 cells 等权宏平均，不能解释为统一指标或合并样本准确率。

## 逐 checkpoint

| Base | Task | Seed | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS |
| ---- | ---- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | Magicoder | 42 | 65.24 | 76.83 | 75.61 | 65.24 | 75.00 |
| Qwen3-8B | Magicoder | 43 | 62.80 | 76.22 | 73.17 | 62.80 | 74.39 |
| Qwen3-8B | Magicoder | 44 | 64.63 | 72.56 | 74.39 | 64.63 | 74.39 |
| Qwen3-8B | MetaMath | 42 | 84.15 | 87.64 | 88.17 | 84.15 | 88.17 |
| Qwen3-8B | MetaMath | 43 | 84.00 | 87.26 | 87.04 | 84.00 | 86.81 |
| Qwen3-8B | MetaMath | 44 | 84.00 | 88.48 | 87.41 | 84.00 | 87.41 |
| Qwen3-8B | Tulu | 42 | 67.84 | 69.32 | 71.16 | 67.84 | 70.79 |
| Qwen3-8B | Tulu | 43 | 66.17 | 71.35 | 72.27 | 66.17 | 72.46 |
| Qwen3-8B | Tulu | 44 | 66.91 | 69.32 | 71.53 | 66.91 | 70.98 |
| Llama-3.1-8B-Instruct | Magicoder | 42 | 53.66 | 53.05 | 54.88 | 53.66 | 54.88 |
| Llama-3.1-8B-Instruct | Magicoder | 43 | 55.49 | 54.88 | 54.27 | 55.49 | 56.10 |
| Llama-3.1-8B-Instruct | Magicoder | 44 | 53.66 | 54.88 | 55.49 | 53.66 | 53.66 |
| Llama-3.1-8B-Instruct | MetaMath | 42 | 77.26 | 79.98 | 80.89 | 77.26 | 80.82 |
| Llama-3.1-8B-Instruct | MetaMath | 43 | 74.83 | 76.80 | 78.77 | 74.83 | 78.39 |
| Llama-3.1-8B-Instruct | MetaMath | 44 | 74.60 | 78.70 | 78.39 | 74.60 | 79.15 |
| Llama-3.1-8B-Instruct | Tulu | 42 | 63.22 | 64.70 | 63.22 | 63.22 | 65.43 |
| Llama-3.1-8B-Instruct | Tulu | 43 | 63.22 | 64.88 | 64.51 | 63.22 | 65.25 |
| Llama-3.1-8B-Instruct | Tulu | 44 | 63.03 | 64.33 | 65.06 | 63.03 | 64.88 |

## 18 checkpoints 宏平均和胜负

| Method | Mean | Δ vs LoRA | Δ vs HNS | vs LoRA W/T/L | vs HNS W/T/L |
| --- | ---: | ---: | ---: | --- | --- |
| LoRA | 68.0402 | +0.0000 | -4.1248 | 0/18/0 | 0/1/17 |
| Flat-Fro | 71.7311 | +3.6909 | -0.4339 | 16/0/2 | 5/0/13 |
| Flat-Nuclear | 72.0139 | +3.9737 | -0.1510 | 16/1/1 | 8/4/6 |
| DG-Hard | 68.0402 | +0.0000 | -4.1248 | 0/18/0 | 0/1/17 |
| HNS | 72.1650 | +4.1248 | +0.0000 | 17/1/0 | 0/18/0 |

## Per-base 平均

| Group | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | 64.3284 | 65.7985 | 66.1649 | 64.3284 | 66.5067 |
| Qwen3-8B | 71.7520 | 77.6636 | 77.8629 | 71.7520 | 77.8232 |

## Per-task 平均

| Group | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Magicoder | 59.2480 | 64.7358 | 64.6341 | 59.2480 | 64.7358 |
| MetaMath | 79.8079 | 83.1438 | 83.4471 | 79.8079 | 83.4597 |
| Tulu | 65.0647 | 67.3136 | 67.9606 | 65.0647 | 68.2994 |

## 3-seed mean ± sample SD（ddof=1）

| Base | Task | LoRA | Flat-Fro | Flat-Nuclear | DG-Hard | HNS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | Magicoder | 54.27 ± 1.06 | 54.27 ± 1.06 | 54.88 ± 0.61 | 54.27 ± 1.06 | 54.88 ± 1.22 |
| Llama-3.1-8B-Instruct | MetaMath | 75.56 ± 1.47 | 78.49 ± 1.60 | 79.35 ± 1.35 | 75.56 ± 1.47 | 79.45 ± 1.24 |
| Llama-3.1-8B-Instruct | Tulu | 63.15 ± 0.11 | 64.63 ± 0.28 | 64.26 ± 0.95 | 63.15 ± 0.11 | 65.19 ± 0.28 |
| Qwen3-8B | Magicoder | 64.23 ± 1.27 | 75.20 ± 2.31 | 74.39 ± 1.22 | 64.23 ± 1.27 | 74.59 ± 0.35 |
| Qwen3-8B | MetaMath | 84.05 ± 0.09 | 87.79 ± 0.62 | 87.54 ± 0.58 | 84.05 ± 0.09 | 87.47 ± 0.68 |
| Qwen3-8B | Tulu | 66.97 ± 0.83 | 69.99 ± 1.17 | 71.66 ± 0.56 | 66.97 ± 0.83 | 71.41 ± 0.91 |

## 数值审计

Flat-Fro: t=||σ||₂/√r；Flat-Nuclear: t=||σ||₁/r。两者均使用 shared compact SVD 和 B=U√t、A=√t Vᵀ。所有 source config 字节保持一致，未改变 scaling、rank、alpha、target modules 或 base model；保存权重重载后逐 tensor 验证。

- flat_fro: max target budget relative error=1.19209048e-07; max saved budget relative error=4.93352218e-07.
- flat_nuclear: max target budget relative error=1.4863501e-07; max saved budget relative error=5.83788676e-07.

已有短测试与完整评测的逐 token 重复性检查：0 / 12168 个重复样本输出不同；详细计数见 summary JSON。

HumanEval 日志存在 multiprocessing 临时目录清理回调的 NFS `.nfs*` busy 异常，既有 HNS seed-eval 日志也出现相同现象。检查 human_eval/execution.py 确认测试结果先写入 manager result 再清理；主评分正常完成。汇总逐 cell 验证全部164条执行结果和 scored rows，无遗漏；未修改执行 timeout、parser 或 metric。

## 统一比较与 identity 检查

严格先完成 Flat 36 cells 和先行汇总，再实现、构建、完成独立 DG 18 cells。独立 DG 权重逐字节等于原始 LoRA，但与历史 LoRA predictions 出现生成漂移（Llama seed44/Tulu 98/541 token 序列不同、主指标差−0.9242pp）。因此追加 joint 统一复评：每个 checkpoint 的 LoRA / Flat-Fro / Flat-Nuclear / DG-Hard / 固定 HNS 放入同一个五-adapter batch。不预设漂移具体原因，不把 identity 的执行差异当作算法效果；未更改 prompt、生成参数、parser 或 metric。

最终同批 LoRA / DG 对照：token 序列不同 0 / 12144 个样本；成绩不同的 checkpoints=0 / 18。逐 checkpoint 检查见 final_summary.json。

先行 Flat 汇总：flat_summary.md/json；独立 DG 与历史参考结果：dg_initial_summary.json；joint 完整结果：eval/joint/<base>/{generation_manifest,score_manifest,commands}.json。主表不把历史成绩与 joint 成绩拼接。

历史漂移审计：645 / 12144 个样本 token 序列不同，全部 benchmark 输入记录一致；逐 checkpoint 的旧/新 predictions 路径、分数、差值见 identity_legacy_drift_audit.json。

## 原始 DG-Hard 审计

τ=ω(β)·median(full singular spectrum), β=min(m,n)/max(m,n); ω(β)=λ*(β)/√μβ，其中 μβ 为 Marchenko–Pastur 分布中位数。完整 spectrum 长度 min(m,n)，补上低秩表示中省略的零；严格保留 σ>τ。未采用 active-spectrum median。

| Base | Task | Seed | Retained rank min/max | Unchanged module fraction | Threshold min/max | Retained Fro / before (ratio) | Retained nuclear / before (ratio) | Identity |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| Qwen3-8B | Magicoder | 42 | 16/16 | 1.000000 | 0/0 | 4.18651/4.18651 (1) | 118.24/118.24 (1) | True |
| Qwen3-8B | MetaMath | 42 | 16/16 | 1.000000 | 0/0 | 18.5199/18.5199 (1) | 627.713/627.713 (1) | True |
| Qwen3-8B | Tulu | 42 | 16/16 | 1.000000 | 0/0 | 24.2385/24.2385 (1) | 1020.35/1020.35 (1) | True |
| Qwen3-8B | Magicoder | 43 | 16/16 | 1.000000 | 0/0 | 2.98692/2.98692 (1) | 100.707/100.707 (1) | True |
| Qwen3-8B | MetaMath | 43 | 16/16 | 1.000000 | 0/0 | 20.7336/20.7336 (1) | 777.44/777.44 (1) | True |
| Qwen3-8B | Tulu | 43 | 16/16 | 1.000000 | 0/0 | 25.1637/25.1637 (1) | 1146.82/1146.82 (1) | True |
| Qwen3-8B | Magicoder | 44 | 16/16 | 1.000000 | 0/0 | 2.97311/2.97311 (1) | 100.217/100.217 (1) | True |
| Qwen3-8B | MetaMath | 44 | 16/16 | 1.000000 | 0/0 | 20.6926/20.6926 (1) | 777.909/777.909 (1) | True |
| Qwen3-8B | Tulu | 44 | 16/16 | 1.000000 | 0/0 | 25.8174/25.8174 (1) | 1153.93/1153.93 (1) | True |
| Llama-3.1-8B-Instruct | Magicoder | 42 | 16/16 | 1.000000 | 0/0 | 3.44694/3.44694 (1) | 115.011/115.011 (1) | True |
| Llama-3.1-8B-Instruct | MetaMath | 42 | 16/16 | 1.000000 | 0/0 | 14.4983/14.4983 (1) | 673.089/673.089 (1) | True |
| Llama-3.1-8B-Instruct | Tulu | 42 | 16/16 | 1.000000 | 0/0 | 11.2335/11.2335 (1) | 516.527/516.527 (1) | True |
| Llama-3.1-8B-Instruct | Magicoder | 43 | 16/16 | 1.000000 | 0/0 | 2.47617/2.47617 (1) | 89.5759/89.5759 (1) | True |
| Llama-3.1-8B-Instruct | MetaMath | 43 | 16/16 | 1.000000 | 0/0 | 4.65314/4.65314 (1) | 196.486/196.486 (1) | True |
| Llama-3.1-8B-Instruct | Tulu | 43 | 16/16 | 1.000000 | 0/0 | 11.3327/11.3327 (1) | 518.841/518.841 (1) | True |
| Llama-3.1-8B-Instruct | Magicoder | 44 | 16/16 | 1.000000 | 0/0 | 2.48652/2.48652 (1) | 89.3398/89.3398 (1) | True |
| Llama-3.1-8B-Instruct | MetaMath | 44 | 16/16 | 1.000000 | 0/0 | 4.67818/4.67818 (1) | 198.322/198.322 (1) | True |
| Llama-3.1-8B-Instruct | Tulu | 44 | 16/16 | 1.000000 | 0/0 | 11.3431/11.3431 (1) | 517.996/517.996 (1) | True |

DG 的逐 module threshold、median、aspect ratio、保留谱和 norm 位于每个 dg_hard/posthoc_meta.json。identity 模块直接保留原始 A/B，以避免 SVD 重构舍入改变 identity baseline。DG-Hard 仍通过同一脚本重新生成、评分，不以 LoRA 成绩填充。

原论文：[Donoho & Gavish, The Optimal Hard Threshold for Singular Values is 4/√3](https://arxiv.org/abs/1305.5870)。

Fro 总量为各 module Fro² 之和开根号；nuclear 总量为各 module nuclear norm 之和，均针对未乘 scaling 的 BA。原始 scaling 保持不变；norm ratios 同样适用于 scaled updates。MP density 为 sqrt((b-x)(x-a))/(2πβx)，按作者原始补充代码的归一化计算；Gauss–Legendre 积分加二分求中位数，逐步加倍积分阶数至收敛，不使用三次多项式近似。

原始补充代码（作者代码镜像）：[optimal_SVHT_coef.m](https://raw.githubusercontent.com/bwbrunton/dmd-neuro/master/optimal_SVHT_coef.m)。

## 运行顺序与命令

最多两个单 GPU Slurm worker（array=0-1%2，gres=gpu:1）；每个 worker 一个 base，所有训练任务和种子串行/同引擎批量评测。先做 max_num_seqs2048 的256样本/任务探测，再完整评测；OOM 才依次降为1024/512/256。max_num_batched_tokens65536、prompt chunk1024；独立 Flat/DG adapter block6，joint block5。joint 五-adapter batch 的并发规模在成功的六-adapter Flat 短测试范围内，沿用2048，无额外探测。实际选定配置见 generation_manifest，探测日志见 batch_probe。

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/audit_posthoc_three_seed.py
PYTHONPATH=src /dataset1/zailong/envs/peft-sft-lab/bin/python -m pytest -q tests/test_posthoc_flat.py
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/build_posthoc_flat_three_seed.py
sbatch slurm/posthoc_flat_three_seed.slurm
PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/summarize_posthoc_flat_dghard_three_seed.py --phase flat
```

逐条真实 generation/scoring argv 见 eval/<phase>/<base>/commands.json；Slurm 提交回执见 submission.json。

实际 Slurm 提交：`{"flat": {"job_id": 972, "command": ["sbatch", "slurm/posthoc_flat_three_seed.slurm"], "max_gpus": 2, "gpus_per_worker": 1, "failed_attempts": [{"job_id": 970, "reason": "node01 /tmp full during torch.compile; no predictions generated"}]}, "dg": {"job_id": 979, "command": ["sbatch", "slurm/posthoc_dghard_three_seed.slurm"], "max_gpus": 2, "gpus_per_worker": 1, "flat_summary_required": true}, "joint": {"job_id": 981, "command": ["sbatch", "slurm/posthoc_joint_three_seed.slurm"], "max_gpus": 2, "gpus_per_worker": 1, "reason": "Observed identity/legacy generation drift; five methods/checkpoint in one batch"}}`。首次 job970 因 node01 /tmp 满在引擎启动阶段失败，没有生成评测结果；job972 改用 workspace 临时缓存路径后重试。

```bash
PYTHONPATH=src /dataset1/zailong/envs/peft-sft-lab/bin/python -m pytest -q tests/test_posthoc_dg_hard.py
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/build_posthoc_dghard_three_seed.py
sbatch slurm/posthoc_dghard_three_seed.slurm
PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/audit_posthoc_identity_drift.py
PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/prepare_posthoc_joint_three_seed.py
sbatch slurm/posthoc_joint_three_seed.slurm
PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/summarize_posthoc_flat_dghard_three_seed.py --phase final
```

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

每个 Adapter parent 下为 `flat_fro/`、`flat_nuclear/`、`dg_hard/`。Variant manifests: `<base>_flat_variant_manifest.json` / `<base>_dg_variant_manifest.json`。

Source manifest 同时记录原始 variant/score/generation/task config 路径、SHA256、完整 adapter config、module shapes 和 LoRA/HNS metric records。

最终统一复评 manifests：`<base>_joint_variant_manifest.json`（每个 base 45 entries，按 checkpoint 五种方法分组）。全部 GPU 作业已完成并释放资源；Slurm 实际分配均为每个 worker 一张 B300，最多两个同时运行。

总实验 manifest：`/dataset1/zailong/workspace/peft-sft-lab/reports/posthoc_flat_dghard_three_seed_20260913/experiment_manifest.json`，包含全部 artifact 索引、评测/谱编辑代码 SHA256、包版本和 GPU 限制。

实际包版本：`{"torch": "2.11.0+cu128", "vllm": "0.25.1+cu129", "transformers": "5.16.1", "peft": "0.20.0", "numpy": "2.3.5", "safetensors": "0.8.0", "human-eval": "1.0.3"}`。

## Git 归档

上传范围、解压方法和本地保留文件见 [GIT_ARTIFACTS.md](posthoc_flat_dghard_three_seed_20260913/GIT_ARTIFACTS.md)。逐样本 JSONL、生成/评分日志及 Slurm 日志以无损 `.gz` 保存，原文件和压缩文件的 SHA256 见 `git_artifact_manifest.json`；adapter 权重及复制的 `tokenizer.json` 保留在 manifest 记录的本地路径。
