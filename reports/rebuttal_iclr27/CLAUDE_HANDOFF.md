# LoRA-Norm ICLR 2027 #15844：Claude Code 完整接手文档

> 快照：2026-11-08 18:30:52 +08:00。所有时间默认 Asia/Singapore（UTC+8）；JSON 的 `*_utc` 字段需要加 8 小时。
> 这是交接任务，现有 Slurm 作业保持运行。请先读本文件，再刷新队列；不要重复提交已经排好的 GPU 实验。
> 数值来自已保存的结果表，而非最初方案的预期。交接时第一轮 E0–E6、第二轮 R2-O1–O8 均已核验完成；R2-O9 正在训练。

## 1. 接手后先做什么

1. 查看 `squeue -u "$USER"`。本快照唯一运行的 GPU 作业为 **15429，C 档，2 GPU，node03**。最终 CPU 审计 **15431** 已排好依赖；不要另起一套 C 档实验。
2. 监控 `reports/rebuttal_iclr27/R2-O9/training_commands_0.json` 和 `_1.json` 中**最后一条实际命令的 log 路径**。当前是带 `.job15429.3.log` 后缀的日志，旧的 `_train.log` 已因节点故障中断，不能用旧日志估计进度。
3. 按现有 C 档流水线完成 O9 → O10 → O12；O11 没有得到作者确认，禁止执行。预算不足时允许按已锁规则跳过 C 档后续项，不能按测试分数决定跳过。
4. 核对 O7/O8 的负结果，并在最终解读中如实保留。不要把后续结论写成“所有规模、所有训练策略均有效”。
5. 每项结束核对四个主输出、覆盖率和数值审计，再更新进度、commit/push。最终 CPU 作业会生成第二轮完成索引；仍需人工阅读结论、检查异常和推送状态。

## 2. 项目位置、环境和任务背景

| 项目 | 值 |
| --- | --- |
| 仓库 | `TTtianTT/peft-sft-lab` |
| 工作目录 | `/dataset1/zailong/workspace/peft-sft-lab` |
| 分支 | `refactor/sft-chat-template` |
| 第一轮起点 | `139bfd7` |
| 第一轮完成索引提交 / 第二轮基点 | `9f23dfc` |
| 交接前最新实验结果提交 | `898df832ddfdc4ca0fb0e76356d0d89b77482917`（R2-O8；流水线后续会继续自动提交） |
| Python | `/dataset1/zailong/envs/peft-sft-lab/bin/python` |
| 模型/数据常用根目录 | `/dataset1/zailong/models`、`/dataset1/zailong/data/peft-sft-lab` |
| 统一结果根目录 | `reports/rebuttal_iclr27/` |

进入环境：

```bash
cd /dataset1/zailong/workspace/peft-sft-lab
export PYTHONPATH="$PWD/src:$PWD/scripts"
```

论文研究 **LoRA-Norm：对已经训练好的单任务 LoRA adapter 做无需数据的谱后处理**，希望提高目标能力并缓解非目标基准的遗忘。不是重新训练一个完整模型，也不使用测试数据选编辑参数。

正文主方法在所有七类投影上编辑 LoRA 更新：`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`。对更新做紧凑 SVD，保持左右奇异子空间，仅变换奇异值，用 HNS（Hybrid Newton–Schulz）迭代使谱趋于展平，再恢复原 nuclear norm。固定配置为 fast4 + stable1、strength1、all_modules。

这两轮补实验主要回应以下审稿意见：

| 审稿人 | 主要问题 | 对应实验 |
| --- | --- | --- |
| XJ3f | 相比精确展平是否必要；重建/因子表示的数值效应；SS 模块公平性 | E0/E1/E2、O2/O3 |
| FzQ5 | Fro 与 nuclear 的差别；SS 策略/模块/预算；训练 recipe、rank、模型大小；两阶段必要性；完整算法 | E0/E2/E4/E6、O3–O10 |
| 8Euc | 协议不统一；源 adapter 欠训练/过拟合；只是缩放还是谱形状；appropriate scaling | 统一主协议、E3/E4、O1/O5/O6/O8、待批准 O11 |

**叙事已经改变**：现有 rank16 主队列中，HNS 和精确 Flat-Nuclear 很接近，不能把“HNS 明显优于精确 nuclear 展平”当作结论。更稳妥的贡献是“保持 nuclear norm 的谱展平”这一类变换，HNS 为其中一种实现；其对弱谱尾的区别需要结果支持。O7/O8 还出现了实质负结果，跨规模/策略的普适性不成立。

## 3. 所有接手操作必须遵守的限制

### 3.1 集群与 GPU

- **所有 GPU 训练、评测、推理和 CUDA adapter 构建都必须通过 `srun` 或 `sbatch`。不得在 login 节点直接执行这些操作。**
- **全局最多 2 张 GPU，跨本用户所有项目/作业合计。**不是每个实验 2 张，也不是每个 agent 2 张。
- 每次提交新 GPU 作业前都必须运行 `squeue -u "$USER"` 并检查现有 GPU 作业。允许一个 2 GPU 作业，或两个 1 GPU 作业。当前 15429 已占 2 GPU，不能再开一个 GPU 工作负载。
- 不使用 `nohup`、后台 shell、screen 等绕开 Slurm。只读结果分析、文档和小型 CPU 单元测试可以在 login 节点执行；较大数值审计已放到 0 GPU Slurm 作业。
- node 使用 exclusive-process GPU 模式。**构建 adapter 的 CUDA 子进程必须退出后，才能启动 vLLM。**不能让父进程先持有 CUDA context 再启动 vLLM EngineCore。第一轮 14888 就因此在生成前失败。

### 3.2 统一评测及统计

主协议必须沿用 `reports/functional_hns_three_seed_20260914/`，不得另起可比较性不明的新波次或改主基准的 prompt/chat template/parser/scorer。

| 代码任务名 | 基准族 | 主指标 / 样本数 |
| --- | --- | --- |
| `magicoder` | HumanEval | greedy pass@1，164 |
| `metamath` | GSM8K | strict accuracy，1319 |
| `tulu` | IFEval | prompt-level strict，541 |
| `commonsense` | Commonsense-8 | 22419 题，八个子任务等权 macro |

推理固定：greedy、seed42、batch-invariant、`max_model_len=4096`、adapter block5、FLASH_ATTN、async off、prefix cache off、GPU memory utilization0.94、batched tokens65536；长任务 `max_num_seqs=2048`、Commonsense4096；prompt chunk short4096/long1024。最大 LoRA rank 正常16；r32/r64 条件分别用32/64，是已预先声明的必要例外。

主队列为两模型 `Qwen3-8B` / `Llama-3.1-8B-Instruct` × 三个训练任务 magicoder/metamath/tulu × seed42/43/44，共18个固定 source checkpoint；E0–E3、O2–O4不重新训练它们。新训练另按 E4、O5–O10 的明确方案统计。

令单个 checkpoint 的源任务为 t，四基准族分数为 S，原始 Base 分数为 B：

```text
Target = S_t
Off = mean(S_j for j != t)                        # 三个非目标族等权
FG = mean(max(B_j - S_j, 0) for j != t)            # 先逐族截断，再平均
```

分数用百分数，差值/FG 用百分点（pp）。不可用 `max(mean(B-S),0)` 代替 FG，也不可按 Commonsense 总题数给四族加权。

- 原18个 checkpoint 的比较：paired checkpoint bootstrap，20000次，95% percentile CI，bootstrap seed20261106，附有利方向的胜/平/负。**共享固定测试集，这些是描述性区间，不是 IID 训练运行的置信区间。**CI 跨0不证明等价。
- E4、O1/O6–O10 等每条件单 seed：题目级 paired bootstrap20000次；二元指标 exact McNemar，附修复/破坏数；Commonsense 按八子任务分别重采样后 macro。Off/FG 传播联合题目重采样。不要把这些区间写成训练运行不确定性。
- O5 有每任务3个训练 seed，另报告 mean ± sample SD（ddof1）及描述性 checkpoint 配对比较。
- 原因子、编辑因子和 Base 的输入必须逐样本、按 ID 和顺序做哈希校验。不能只检查题数。还需核对 adapter/config SHA256、重建误差、完整模块覆盖。
- 所有配置在新推理/训练之前写入 `protocol.md` 并锁 SHA。全部预设条件都报告；不得用测试结果选择方法、强度、LR、模型条件或是否公开。
- 每个实验目录都应有 `report.md`、`summary.json`、`checkpoint_results.tsv`、`paired_ci.tsv`、`coverage_audit.json`、`artifact_integrity_audit.json`。写作类 E5/E6 的 TSV 仅表头是有意为之。

### 3.3 真实代码中的 HNS 与重建

**主实现是 balanced 写回，初始用户方案中把它写成 unbalanced 的描述已经被纠正。不要为了“符合原提案”去改主实现。**

```text
native scaling s = alpha/r，主队列为2，配置保持不变
compact fp32 QR/SVD：BA = U diag(sigma) Vh
fp64 谱运算，sigma/||sigma||2；eps=1e-7
fast 系数 (3.4445, -4.7750, 2.0315)，执行4次
stable 系数 (2.0, -1.5, 0.5)，执行1次
每次 f(x)=x*(a+b*x²+c*x⁴)
恢复原 sum(sigma)，再按 strength 在原谱和编辑谱间插值
新谱 cast fp32；balanced B'=U sqrt(sigma')，A'=sqrt(sigma') Vh
最后分别 cast 回原 factor dtype，保存原 rank/alpha/config
```

计算 BA 并保留正 scaling2，与对 sBA 做保持 nuclear norm 的同类编辑等价；主代码不物化 dense ΔW。HNS CLI 默认是8+2，实验必须显式使用4+1。Recon-0 调用同一路径，谱恒等（strength0），因此仍会受到 SVD/因子表示/有限精度的数值效应。O2 才另外做 unbalanced：B=U diag(sigma)、A=Vh。

## 4. 已完成第一轮：E0–E6

第一轮新做 **696个主协议 cell、7次训练**，已经完成并重新核验所有输出 SHA。固定18主队列没有重训；E4 是新训练。完整索引在 [completion_report.md](completion_report.md) 的第一轮部分，机器可读索引为 `completion_summary.json`。

| ID | 问题 / 做了什么 | 新 cell | 状态 |
| --- | --- | --- | --- |
| E1 | 18主 checkpoint 的 Recon-0；以它为基线重算 HNS/各对照 | 72 | 完成 |
| E0 | 原 Slurm992 的 Flat-Nuclear/Flat-Fro adapter 原样复用到主协议，先核 SHA | 144 | 完成 |
| E2 | HNS-o/down、SS-smooth_abs-o/down、SS-smooth_abs-all7 | 216 | 完成 |
| E3 | 数据无关 Mag-only、Shape-only、Global-γ；2×2 幅度/形状析因 | 216 | 完成 |
| E4 | Qwen8B 两任务3LR训练6次并选验证集最优；MetaMath r64训练1次 | 48 | 完成 |
| E5 | LoDA 的代码/设定可比性核对 | 0 | 完成，按设定差距例外未移植 |
| E6 | 与实现逐行一致的完整算法框 | 0 | 完成，见 `E6/algorithm.tex` |

第一轮运行记录：14888（2GPU，00:03:03）在生成前因 CUDA context 冲突失败；修复为独立 builder 子进程后，14889（2GPU，01:50:08）完成E1/E0，14892（2GPU，03:43:36）完成E2/E3，14893（2GPU，12:46:10）完成E4并接着整理E5/E6。各作业串行，最大并发2GPU。详细命令/日志见第一轮 `submission.json`、各实验 `commands_*.json`、`logs/rebuttal27-*.out/err`；最终第一轮索引提交为 `9f23dfc`。

### 4.1 18主 checkpoint 的统一绝对均值

| 方法 | Target ↑ | Off ↑ | FG ↓ |
| --- | --- | --- | --- |
| flat_e | 72.273 | 72.566 | 0.396 |
| flat_fro | 71.731 | 72.210 | 0.577 |
| flat_nuclear | 72.014 | 72.664 | 0.383 |
| global_gamma | 70.540 | 71.430 | 1.219 |
| hns_f4_s1 | 72.083 | 72.792 | 0.344 |
| mag_only | 70.565 | 71.436 | 1.121 |
| original_lora | 67.998 | 69.551 | 2.708 |
| para_e90 | 68.778 | 70.013 | 2.371 |
| para_e95 | 68.582 | 69.636 | 2.695 |
| para_e99 | 68.442 | 69.778 | 2.564 |
| recon_0 | 68.477 | 69.766 | 2.576 |
| scalar_e | 72.700 | 72.470 | 0.525 |
| shape_only | 71.898 | 72.212 | 0.548 |
| spectral_surgery_grad_direction | 68.764 | 68.282 | 3.723 |

核心配对结论（95% 描述性 checkpoint CI，单位pp）：

| 比较 | Target | Off | FG |
| --- | --- | --- | --- |
| HNS − 原LoRA | +4.086 [2.683,5.642] | +3.240 [1.975,4.636] | −2.364 [−3.476,−1.457] |
| Recon-0 − 原LoRA | +0.479 [0.028,0.930] | +0.215 [−0.010,0.435] | −0.132 [−0.324,0.065] |
| HNS − Recon-0 | +3.607 [2.078,5.346] | +3.026 [1.761,4.363] | −2.232 [−3.259,−1.376] |
| Flat-Nuclear − HNS | −0.070 [−0.431,0.295] | −0.127 [−0.336,0.079] | +0.038 [−0.113,0.182] |
| Flat-Nuclear − Flat-Fro | +0.283 [−0.372,0.901] | +0.454 [0.123,0.809] | −0.195 [−0.416,−0.005] |

因此重建本身确有数值影响，但 HNS 相对 Recon-0 的主队列增益仍存在；nuclear 恢复比 Fro 恢复在 Off/FG 更有支持。不能声称 HNS 优于精确 Flat-Nuclear，也不能把跨0写成严格等价。

旧9月13日波次 HNS Target=72.16 与正文主波次不是同一批结果。**后续统一使用72.0834（正文四舍五入72.08）。**E0 的两 Flat 方法跨旧/新波次879948个样本比较中 token/text/correctness 变化数均为0；这不代表所有方法/因子化均逐 token 可复现。

### 4.2 E2 与 SS 的公平性

SS 原实现来自本仓库公开实现，论文 https://arxiv.org/abs/2603.03995，已按引用源核 SHA；见 `literature_audit.json`。smooth_abs 固定默认值：core_frac/noise_frac=.2/.2，amp1.25，sup.8，mid1，min_core_k1，temperature.35，center_q.5，align_mid=true；grad_norm=mean_abs，preserve_energy=l1，sigma_clip_min0；训练集 calib128，batch2，shuffle seed42，start0。没有测试调参。

相同 o/down 范围，HNS-od − smooth_abs 的 Target/Off/FG 为 +1.432 [0.715,2.383] / +1.481 [0.984,1.974] / −1.285 [−1.762,−0.832]。相同all7范围，HNS − smooth_abs-all 为 +3.804 [2.544,5.063] / +3.989 [2.623,5.423] / −2.973 [−4.150,−1.952]。

SS 每 adapter 用128个训练校准样本、64个前向/反向 batch；HNS 用0数据、0模型前向/反向，只有 SVD/谱编辑。记录的是不同实际预算，**不是声称两者计算预算相等**。E2 记录 HNS/SS 时间；O3 补全 grad_direction-all。

### 4.3 E3：幅度/形状析因，已经有结果

共享同一个 SVD 基底。令原谱σ、保持 nuclear 的 HNS 谱σ_H：Recon-0=σ；Mag-only=σ·||σ_H||2/||σ||2；Shape-only=σ_H·||σ||2/||σ_H||2；HNS=σ_H。Global-γ 为整个 adapter 一个 γ，取逐模块 Fro 比值的平均，**没有验证集调参**。

```text
形状主效应 = 0.5*((HNS−Mag-only)+(Shape-only−Recon-0))
幅度主效应 = 0.5*((HNS−Shape-only)+(Mag-only−Recon-0))
交互效应 = HNS−Mag-only−Shape-only+Recon-0
```

| 效应 | 指标 | 差值 [95% CI] pp |
| --- | --- | --- |
| shape_main | target | +2.470 [+1.309, +3.703] |
| shape_main | off_score | +1.901 [+1.057, +2.786] |
| shape_main | forgetting_gap | -1.402 [-2.095, -0.821] |
| magnitude_main | target | +1.137 [+0.565, +1.761] |
| magnitude_main | off_score | +1.125 [+0.651, +1.606] |
| magnitude_main | forgetting_gap | -0.830 [-1.196, -0.525] |
| interaction | target | -1.903 [-3.333, -0.712] |
| interaction | off_score | -1.091 [-1.873, -0.339] |
| interaction | forgetting_gap | +1.251 [+0.637, +1.965] |

这个固定设计中两个主效应均有支持，但交互非0，不能写成所有任务/配置下“彼此独立、普适”的贡献。严格展平的缩放系数 γ_m=||σ||1/(sqrt(r)||σ||2)=sqrt(PR1/r)，其中 PR1=(||σ||1/||σ||2)²。注意它不同于能量 participation rank。逐模块图、Scalar-E 系数相关性在 `E3/module_scaling_relationship.*` 与 `E3/scalar_e_correlations.tsv`。

**E3 的未调参 Global-γ 不等于 O11 的验证集选优 γ。**9月10日内部报告曾发现校准的全局标量与 HNS 接近；不能宣称现有 E3 已排除所有 appropriate scaling 对照。

### 4.4 E4：训练充分性、显式 warmup、rank64

已完成 Qwen8B rank16 两任务×LR倍率{.5,1,2}，每次保存约50%和100% checkpoint，共12个候选；验证集选择，不用主测试集。选中 Magicoder LR1e-5/100%，MBPP151/200=.755；MetaMath LR2e-4/100%，held-out432/500=.864。平局按距1x的abs(log2倍率)、再较低倍率、再较早checkpoint排序。

MetaMath 验证题从原训练源预留500个 numeric-answer 问题，匹配问题组从训练排除；固定训练表47063条。Magicoder 沿用固定50K训练源，验证为固定 seed 抽取的200条 MBPP，与 HumanEval 主测试分开。E4 数据路径/哈希是后续 B/C 的权威来源：`E4/protocol.lock.json` 的 `datasets`。

所有新训练都纠正旧 seed43/44 的 warmup 未生效问题：用显式 `warmup_steps=ceil(.1*实际总步数)`，不再用被 Transformers5.16.1 移除且被参数过滤的 warmup_ratio。Qwen Magicoder1561步/warmup157；MetaMath4413步/warmup442。必须同时核 `training_args.json` 与 `warmup_schedule_audit.json`（on_train_begin 的真实计划）。

| 条件 | 方法 | Target | Off | FG |
| --- | --- | --- | --- | --- |
| magicoder_val_best_r16 | original_lora | 64.024 | 77.563 | 2.126 |
| magicoder_val_best_r16 | recon_0 | 64.024 | 77.979 | 1.728 |
| magicoder_val_best_r16 | hns_f4_s1 | 73.171 | 80.557 | 0.000 |
| magicoder_val_best_r16 | flat_nuclear | 73.171 | 80.497 | 0.000 |
| metamath_val_best_r16 | original_lora | 84.761 | 67.076 | 5.925 |
| metamath_val_best_r16 | recon_0 | 84.534 | 68.042 | 4.964 |
| metamath_val_best_r16 | hns_f4_s1 | 89.158 | 74.277 | 0.000 |
| metamath_val_best_r16 | flat_nuclear | 88.324 | 73.827 | 0.000 |
| metamath_r64 | original_lora | 85.292 | 63.360 | 9.581 |
| metamath_r64 | recon_0 | 85.444 | 63.075 | 9.866 |
| metamath_r64 | hns_f4_s1 | 89.234 | 74.837 | 0.000 |
| metamath_r64 | flat_nuclear | 89.083 | 74.577 | 0.185 |

r64 用已选 MetaMath LR2e-4，alpha128/scaling2，预先指定 final；只有45/16128个谱方向低于模块 Fro 的0.005（0.279%），HNS vs Flat-Nuclear 三指标的题目级 CI 仍跨0，不支持 r64 上 HNS 优势的强结论。见 `E4/module_tail_audit.tsv` 与尾部图。

E4 可选的2x-LR额外测试和4B当时未做；第二轮 O1/O7 已补相关覆盖。原18队列 seed43/44 的旧训练 warmup=0，不能描述为与 E4 新 recipe 完全一致的独立重复。

### 4.5 E5/E6

LoDA 有公开代码（`https://github.com/HHHLF/LoDA_ICML2026`），但发布的是 ViT 图像持续学习实现，直接移植到 Qwen 单 adapter 语言模型需要另写方法和 recipe；按用户许可的设定差距例外，仅写比较，不做 GPU 移植，也不声称已验证可叠加。第二轮明确不再做 LoDA/DoRA/PiSSA/tulu新训练。

E6 的算法框在 `E6/algorithm.tex`；`E6/report.md`、`summary.json` 含真实源码行号/SHA，写作应从这里引用，不从最初提案抄错因子化或 fp32/fp64 时机。

## 5. 已完成第二轮：A/B档 O1–O8

第二轮截至本快照已完成 **776个新 cell、10次新训练**。加第一轮共1472个新 cell、17次已完成新训练；这不含重复使用缓存的 cell，也不把尚未完成的 Llama 训练计入。

| 实验 | 目的 / 条件 | 新 cell | 新训练 | 结果提交 |
| --- | --- | --- | --- | --- |
| O1 | 源 checkpoint 的 LR×训练阶段曲线：11个新条件×4方法×4族 | 176 | 0 | `6d8d047` |
| O2 | 主18队列两种 unbalanced 写回，和balanced主结果配对 | 144 | 0 | `ba5d233` |
| O3 | SS-grad_direction-all7，补齐2策略×2模块范围 | 72 | 0 | `9f46036` |
| O4 | HNS4+0、0+5、4+1 λ=.5，均nuclear/all7 | 216 | 0 | `a239067` |
| O5 | Qwen8B两任务补seed43/44，合并E4 seed42 | 64 | 4 | `b7840ab` |
| O6 | Qwen8B MetaMath rank8/32，合并E4 rank16/64 | 32 | 2 | `01473db` |
| O7 | Qwen3-1.7B/4B MetaMath，含新Base | 40 | 2 | `cff6047` |
| O8 | Qwen8B两任务 LoRA+，默认LR ratio20 | 32 | 2 | `898df83` |

每个目录都通过 `rebuttal_iclr27_r2.require_complete()` 复核（包含输出SHA），并已 commit/push。原始预测/权重仍在忽略的 runtime 目录，本地必须保留。

### 5.1 O1：增益不只出现在某一种训练阶段

完整视图共14个条件：rank16的12个候选 + r64半程/最终；其中3个条件复用 E4，所以新评测11个。图 `R2-O1/training_state_curve.png/pdf` 标注LR、训练进度和验证分数。所有14个条件的 HNS−Recon Target 点估计为正：Magicoder约+4.268～+9.146pp；MetaMath约+3.487～+5.231pp。不能把14个相关 checkpoint 当成14次独立训练。

rank16内验证分数与 Target 增益的 Spearman：Magicoder rho=0（n6），MetaMath rho≈.0882（n6）；MetaMath验证分数与FG降幅rho≈.4058。原始表和描述性 permutation p 值见 `validation_gain_spearman.tsv`。跨任务混合验证分数不可直接同义比较；不要从小样本相关性过度断言无关系。

### 5.2 O2：收益在两种因子化下都保留，但 token 不完全一致

| 对比 | 指标 | 差值 [95% CI] pp | 胜/平/负 |
| --- | --- | --- | --- |
| hns_f4_s1 − recon_0 | target | +3.607 [+2.078, +5.346] | 16/0/2 |
| hns_f4_s1 − recon_0 | off_score | +3.026 [+1.761, +4.363] | 15/0/3 |
| hns_f4_s1 − recon_0 | forgetting_gap | -2.232 [-3.259, -1.376] | 14/4/0 |
| hns_unbal − recon_0_unbal | target | +3.732 [+2.339, +5.256] | 17/0/1 |
| hns_unbal − recon_0_unbal | off_score | +3.002 [+1.755, +4.354] | 15/0/3 |
| hns_unbal − recon_0_unbal | forgetting_gap | -2.290 [-3.392, -1.405] | 14/4/0 |

unbalanced增益减balanced增益的差中差：Target +.126 [−.357,.617]，Off −.024 [−.262,.219]，FG −.059 [−.263,.147]pp，均跨0。正确表述是未发现清晰的收益差异，不能声称统计等价或“完全排除数值效应”。

精确完整 token 序列一致率：HNS为416923/439974≈94.76%；Recon为415849/439974≈94.52%。见 `token_consistency.tsv`。另0GPU作业15441对8568个模块对审计表示的更新，最大相对差 **1.74918e-7**，小于预锁5e-5，详见 `factorization_update_audit.json/tsv`。

### 5.3 O3：完整 SS 2×2

| 模块范围 | 方法 | Target | Off | FG |
| --- | --- | --- | --- | --- |
| o/down | HNS-od | 69.909 | 70.947 | 1.510 |
| o/down | SS-grad_direction | 68.764 | 68.282 | 3.723 |
| o/down | SS-smooth_abs | 68.477 | 69.466 | 2.795 |
| all7 | HNS | 72.083 | 72.792 | .344 |
| all7 | SS-grad_direction-all | 67.278 | 65.974 | 5.891 |
| all7 | SS-smooth_abs-all | 68.279 | 68.803 | 3.317 |

grad_direction-all 参数与原OD配置相同，仅模块范围改变：calib128/batch2/seed42/shuffle train；mean_abs、l1、multiplicative、asymmetric、eta_suppress2/eta_enhance.2。SS-all并未改善本主队列结果。数据/GPU时间预算见 `R2-O3/calibration_budget.tsv`，paired CI 见同目录。

### 5.4 O4：阶段/强度曲线

| 变体 | Target | Off | FG |
| --- | --- | --- | --- |
| recon_0 | 68.477 | 69.766 | 2.576 |
| hns_f4_s1 | 72.083 | 72.792 | 0.344 |
| hns_f4_s0 | 72.272 | 72.607 | 0.433 |
| hns_f0_s5 | 72.067 | 72.660 | 0.341 |
| hns_f4_s1_l05 | 70.668 | 71.781 | 0.979 |

对应平均能量 participation rank / top1 energy / Fro比值：4+0为.908153/.093086/.772265；0+5为.995093/.063938/.762727；4+1为.999260/.065025/.762387；半强度为.499850/.320962/.830723。这里 PR 公式为 `(sum σ²)²/(r*sum σ⁴)`，与E3的PR1不是同一个定义。数值来自本实验完整模块审计，不能与初始计划里其他聚合口径的预估特征均值直接拼接。

效果不是“两阶段必然额外带来性能”，0+5也接近展平且性能接近；不要反过来把任何两个阶段差别描述成已经证明的严格等价。具体差别看 `paired_ci.tsv` 中对4+1的比较。

### 5.5 O5：每任务三seed

固定 Magicoder LR1e-5、MetaMath2e-4，final checkpoint，不对seed43/44重新选LR；dataset_seed42，Trainer/model seed43/44。训练长度/数据/显式warmup与E4一致。合并 E4 seed42得到：

| 任务 | 方法 | Target mean ± SD | Off mean ± SD | FG mean ± SD |
| --- | --- | --- | --- | --- |
| magicoder | original_lora | 64.431 ± 0.352 | 77.587 ± 0.068 | 2.089 ± 0.077 |
| magicoder | recon_0 | 65.041 ± 0.931 | 77.900 ± 0.284 | 1.793 ± 0.278 |
| magicoder | hns_f4_s1 | 73.374 ± 0.352 | 80.557 ± 0.003 | 0.000 ± 0.000 |
| magicoder | flat_nuclear | 73.984 ± 0.931 | 80.783 ± 0.248 | 0.000 ± 0.000 |
| metamath | original_lora | 84.862 ± 0.463 | 68.498 ± 2.287 | 4.515 ± 2.348 |
| metamath | recon_0 | 84.711 ± 0.306 | 68.838 ± 1.578 | 4.172 ± 1.641 |
| metamath | hns_f4_s1 | 88.021 ± 1.101 | 73.944 ± 0.303 | 0.203 ± 0.352 |
| metamath | flat_nuclear | 87.895 ± 0.374 | 73.865 ± 0.778 | 0.203 ± 0.352 |

完整 checkpoint级/每seed题目级区间见 `R2-O5/paired_ci.tsv`。仅3seed，且共享测试集，不夸大统计精度。

### 5.6 O6：rank8/16/32/64

均 Qwen8B MetaMath seed42，LR2e-4，alpha2r；r16/r64缓存来自E4，r8/r32新训练。下表为点估计，单seed题目CI在原表：

| Rank | HNS − Recon Target | Recon FG − HNS FG | Flat-Nuclear − HNS Target |
| --- | --- | --- | --- |
| 8 | 3.412 | 1.244 | -0.607 |
| 16 | 4.625 | 4.964 | -0.834 |
| 32 | 3.487 | 11.346 | 0.227 |
| 64 | 3.791 | 9.866 | -0.152 |

四rank中 HNS相对Recon Target均正，FG下降；HNS vs Flat-Nuclear 的方向不固定，不能拿单个rank点估计宣称普遍优越。

### 5.7 O7/O8：必须保留的负结果和限制

O7 小模型直接沿用8B验证选的MetaMath LR2e-4，未单独调参。O8 LoRA+比例20为仓库默认值，仍沿用E4的task LR。以下是完整点估计，绝不能从最终汇总中删掉：

| 实验 | 条件 | 方法 | Target | Off | FG |
| --- | --- | --- | --- | --- | --- |
| R2-O7 | Qwen3-1.7B_metamath_lora_r16_seed42 | base | 4.018 | 63.715 | 0.000 |
| R2-O7 | Qwen3-1.7B_metamath_lora_r16_seed42 | original_lora | 65.201 | 44.458 | 19.257 |
| R2-O7 | Qwen3-1.7B_metamath_lora_r16_seed42 | recon_0 | 65.125 | 44.022 | 19.693 |
| R2-O7 | Qwen3-1.7B_metamath_lora_r16_seed42 | hns_f4_s1 | 5.231 | 51.186 | 12.529 |
| R2-O7 | Qwen3-1.7B_metamath_lora_r16_seed42 | flat_nuclear | 5.686 | 51.304 | 12.411 |
| R2-O7 | Qwen3-4B_metamath_lora_r16_seed42 | base | 64.139 | 68.088 | 0.000 |
| R2-O7 | Qwen3-4B_metamath_lora_r16_seed42 | original_lora | 54.284 | 63.479 | 5.016 |
| R2-O7 | Qwen3-4B_metamath_lora_r16_seed42 | recon_0 | 56.634 | 63.574 | 5.124 |
| R2-O7 | Qwen3-4B_metamath_lora_r16_seed42 | hns_f4_s1 | 85.974 | 61.142 | 7.563 |
| R2-O7 | Qwen3-4B_metamath_lora_r16_seed42 | flat_nuclear | 85.595 | 61.576 | 7.128 |
| R2-O8 | Qwen3-8B_magicoder_loraplus_r16_seed42 | base | 66.463 | 79.369 | 0.000 |
| R2-O8 | Qwen3-8B_magicoder_loraplus_r16_seed42 | original_lora | 70.122 | 76.854 | 2.814 |
| R2-O8 | Qwen3-8B_magicoder_loraplus_r16_seed42 | recon_0 | 71.341 | 77.222 | 2.461 |
| R2-O8 | Qwen3-8B_magicoder_loraplus_r16_seed42 | hns_f4_s1 | 78.049 | 81.002 | 0.000 |
| R2-O8 | Qwen3-8B_magicoder_loraplus_r16_seed42 | flat_nuclear | 78.049 | 80.565 | 0.000 |
| R2-O8 | Qwen3-8B_metamath_loraplus_r16_seed42 | base | 85.747 | 72.941 | 0.000 |
| R2-O8 | Qwen3-8B_metamath_loraplus_r16_seed42 | original_lora | 80.440 | 38.250 | 34.692 |
| R2-O8 | Qwen3-8B_metamath_loraplus_r16_seed42 | recon_0 | 80.895 | 38.789 | 34.153 |
| R2-O8 | Qwen3-8B_metamath_loraplus_r16_seed42 | hns_f4_s1 | 79.757 | 40.560 | 32.381 |
| R2-O8 | Qwen3-8B_metamath_loraplus_r16_seed42 | flat_nuclear | 79.227 | 40.661 | 32.281 |

HNS−Recon 的题目级95% CI（FG负值有利；Off/FG不具有单个二元McNemar计数）：

| 实验 | 条件 | 指标：HNS − Recon | 差值 [95% CI] pp | 修复题数 | 破坏题数 |
| --- | --- | --- | --- | --- | --- |
| R2-O7 | Qwen3-1.7B_metamath_lora_r16_seed42 | Target | -59.894 [-62.625, -57.165] | 14 | 804 |
| R2-O7 | Qwen3-1.7B_metamath_lora_r16_seed42 | Off | +7.164 [+4.470, +9.817] |  |  |
| R2-O7 | Qwen3-1.7B_metamath_lora_r16_seed42 | FG | -7.164 [-9.817, -4.470] |  |  |
| R2-O7 | Qwen3-4B_metamath_lora_r16_seed42 | Target | +29.340 [+26.685, +31.994] | 411 | 24 |
| R2-O7 | Qwen3-4B_metamath_lora_r16_seed42 | Off | -2.433 [-5.423, +0.533] |  |  |
| R2-O7 | Qwen3-4B_metamath_lora_r16_seed42 | FG | +2.439 [-0.505, +4.531] |  |  |
| R2-O8 | Qwen3-8B_magicoder_loraplus_r16_seed42 | Target | +6.707 [+1.220, +12.195] | 16 | 5 |
| R2-O8 | Qwen3-8B_magicoder_loraplus_r16_seed42 | Off | +3.780 [+2.551, +5.018] |  |  |
| R2-O8 | Qwen3-8B_magicoder_loraplus_r16_seed42 | FG | -2.461 [-3.705, -1.111] |  |  |
| R2-O8 | Qwen3-8B_metamath_loraplus_r16_seed42 | Target | -1.137 [-2.654, +0.379] | 43 | 58 |
| R2-O8 | Qwen3-8B_metamath_loraplus_r16_seed42 | Off | +1.771 [-0.699, +4.275] |  |  |
| R2-O8 | Qwen3-8B_metamath_loraplus_r16_seed42 | FG | -1.771 [-4.275, +0.699] |  |  |

需要接手者特别关注：

1. **Qwen3-1.7B**：Recon GSM8K65.125%，HNS仅5.231%，Target跌约59.894pp；Flat-Nuclear同样降到5.686%。Off上升、FG下降并不能抵消此目标能力崩溃。Base strict仅4.018%也很异常，需读原始输出排查格式/strict抽取、截断、模型revision/tokenizer、权重等，但原因尚未查明。不得直接认定是bug或丢弃结果。
2. **Qwen3-4B**：HNS Target56.634→85.974%，但Off63.574→61.142%，FG5.124→7.563，尤其HumanEval下降。不能描述成三指标一起改善。
3. **MetaMath LoRA+**：源adapter遗忘很大（Recon FG34.153）；HNS Target80.895→79.757下降，Off38.789→40.560小幅恢复，FG仍32.381。Magicoder LoRA+则有明显收益。不能声称对训练策略无条件鲁棒。
4. “完整性审计通过”只说明文件/输入/指标重算/数值覆盖通过，**不等于负结果已被科学解释**。可先做只读输出诊断；主strict scorer/template不改。若做额外宽松解析诊断，要标成辅助诊断，不替换原主结果，也不能据此选方法超参。

## 6. 当前阶段和已经排好的作业

| 作业 | 内容 | GPU | 状态与时间（新加坡） |
| --- | --- | --- | --- |
| 15389 | 第二轮A：O1→O4 | 2 | 完成；11/7 19:03:51→11/8 00:40:42，5:36:51 |
| 15390 | 第二轮B：O5→O8 | 2 | 完成；11/8 00:40:42→15:32:57，14:52:15 |
| 15441 | O2表示更新的CPU审计 | 0 | 完成；11/8 00:40:42→00:41:44 |
| **15429** | C：O9→O10→O12（O11未批准） | **2** | **运行；目前node03；本次Start=11/8 16:40:36，TimeLimit20h** |
| 15431 | 最终CPU审计、索引、Git推送 | 0 | 已排队，依赖 `afterok:15429:15441` |

### 6.1 C档发生过一次节点故障

`sacct -D -j15429` 显示第一次尝试11/8 15:32:57→16:27:39，在node02发生 `NODE_FAIL`（exit1:0），不是实验结果失败。Slurm `Restarts=1`，同一job ID于16:40:36在node03重新启动。前次一部分日志/命令记录没有returncode就结束，后次新命令改用 `.job15429.3.log`。**不要把旧log的388/781步当成当前停滞，也不要为了它再提交一个C作业。**

接手时当前两GPU都在最前面的 Llama Magicoder训练（0.5x、1x LR），已有约半程以上进度。精确快照：

| worker | 当前 Magicoder LR | 最近步数（快照） | 实际日志 |
| --- | --- | --- | --- |
| 0 | 1e-05 | 1304/1561 | /dataset1/zailong/workspace/peft-sft-lab/reports/rebuttal_iclr27/R2-O9/runtime/logs/Llama-3.1-8B-Instruct_magicoder_lr0.5_train.job15429.3.log |
| 1 | 2e-05 | 1307/1561 | /dataset1/zailong/workspace/peft-sft-lab/reports/rebuttal_iclr27/R2-O9/runtime/logs/Llama-3.1-8B-Instruct_magicoder_lr1_train.job15429.3.log |

流水线跳过已有最终adapter，但对未完成训练没有自动加 `--resume_from_checkpoint`；本次重排实际从头重新训练。输出目录可能有第一次中断留下的checkpoint。若未来保存数量/最后global_step断言异常，应先核日志与残留，不要无记录地删除或把旧状态当新训练完成。

### 6.2 时间和预算

第二轮48h预算从11/7 18:42:37开始，墙钟截止 **11/9 18:42:37**（`R2-plan.json.deadline_utc=2026-11-09T10:42:37Z`）。但本次 C Slurm allocation 本身会在 **11/9 12:40:36** 达20h限制，先注意这个更早的限制。

每个可选C阶段开始前要求 `remaining_hours >= 预估时长 + 1h reserve`；预估 O9=10h、O10=5h、O12=2h。节点故障已消耗约1h，ETA须按新日志更新。此前估计11/9上午至中午完成是条件性估计，不是已完成承诺；不要据旧 ETA 判断异常。

训练实测参考：Qwen8B Magicoder单次约3.1h，MetaMath约2.45h；1.7B/4B MetaMath约1.42h/2.02h。两个GPU并行，不能把这些GPU上单次时长简单当总wall。A档608cell/5.61h≈108个cell/小时（两GPU合计）。

## 7. 未来实验：具体做什么、何时能做

### O9（正在运行）：Llama上重做验证集选优

- Llama-3.1-8B-Instruct，magicoder与metamath，各3档LR，共6次训练；seed42、r16/alpha32；50%和100%各存一次，12个候选全做验证。
- LR：Magicoder{1e-5,2e-5,4e-5}；MetaMath{5e-5,1e-4,2e-4}。
- 使用E4相同训练排除/验证数据；Magicoder MBPP200、MetaMath held-out500；同样平局规则，只按验证选择每任务1个val-best。
- **当前实现明确用E4的4096序列长度、GBS32、micro16、cosine、Adam(.9,.999)、Magicoder1epoch/MetaMath3epochs。**它不同于旧seed43/44复现实验采用的Llama math profile（seq1024/GBS768）；旧seed42模型卡本身没有完整配方。协议已写明这一差别，不要混称“与旧源适配器完全同recipe”。
- 每个选中source构建Original/Recon/HNS/Flat-Nuclear，4族，共32cell，复用Llama Base；单seed题目区间。
- worker0依次训练Magicoder0.5x、Magicoder2x、MetaMath1x；worker1依次Magicoder1x、MetaMath0.5x、MetaMath2x。各worker训练完后在本worker做其半程/最终验证。
- 原因/参数权威文件：`R2-O9/protocol.md`、`training.lock.json`；选优结果将写 `selection.json`、`validation_candidates.tsv`。

### O10（尚未开始）：Qwen3-14B MetaMath

- 官方14B模型已经下载、固定revision、SHA记录在 `R2-assets/Qwen3-14B.json`，实际文件位于同目录runtime/models。
- O9完成后、预算足够时，先做Slurm内 vLLM fit check：TP1、一张B300、main4096/maxrank16等固定设置；若能在一张卡初始化和生成，则自然满足最多两卡限制。
- 一次训练：seed42、r16/alpha32、MetaMath LR2e-4从8B沿用，final checkpoint预先指定；没有为14B重新调参。Recipe同E4，显式warmup10%。
- 新Base加4adapter×4族，共20cell。若固定recipe显存不够，记录容量失败，不能悄悄改变microbatch、精度或评测协议后把它算同配置。
- `capacity_preflight.json` 是预检结果；当前尚未产生，不应对外宣称14B已验证可跑。

### O11（禁止擅自开始）：验证集选优的全局γ

- 用户明确要求先和作者确认；之前已询问，**截至本快照没有批准**。`o11_author_approval=null`。不把“继续”或查询进度当作批准这个可选对照。
- 若明确批准：18主checkpoint分别从γ∈{.3,.4,.5,.6,.7,.85,1.0}中仅按独立验证分数选一个，再跑4族，共72testcell，另加validation成本；所有结果必须公开，不能不利就撤掉。
- 这与E3的data-free平均Fro比值Global-γ不同。旧内部报告发现调过的全局标量可能与HNS打平，是重要且可能不利的对照。
- **当前 C driver 只写了“等待批准”的占位，并没有O11完整执行器。**仅改 `R2-plan.json` 的 approval 字段不会自动执行；真批准后还要明确适用于旧18source的独立验证数据（尤其tulu）、冻结选择规则、实现新runner、检查预算与GPU并发。
- `rebuttal_iclr27_r2_finish.py` 的若干文案也按O11未执行写死。若之后批准并执行，要同步真实状态/最终文案；不要在当前未批准时去篡改已锁C脚本。

### O12（已准备但未推理）：独立target基准

- 只用原18主队列里magicoder/metamath的12个source，这批没用MBPP选模型；**不纳入E4/O5等MBPP-selected sources**。
- MetaMath source加测MATH-500（500）；Magicoder source加测官方EvalPlus MBPP+v0.2.0（378）；4方法×12=48adaptercell，另2base×2auxbenchmark=4Basecell，共52。
- 官方数据/库固定：Math-Verify0.9.0、latex2sympy2_extended1.11.0、ANTLR4.13.2；EvalPlus0.3.1，MBPP+v0.2.0。安装在 `R2-O12/runtime/deps/`，没有升级第一轮主Python环境。`assets.json` 记录数据revision/SHA和评分源码SHA。
- 用官方EvalPlus sanitize(entrypoint)，官方base+plus均pass才计正确；Math-Verify用锁定的默认Latex/Expr extraction +数学等价核验。新增parser的10项单测已通过（另3项R2核心单测，共13）；500个MATH gold均可解析。MBPP/793官方只有3个base输入、0个plus输入，原样保留，不人为过滤。
- 额外target有独立必要的prompt/scorer，已在protocol固定；greedy/seed42/maxmodel4096/block5/maxrank16等不变，max_new_tokens512。四个主基准及主Target/Off/FG **不被替换**，继续用已审计缓存；aux分数另记 `independent_target`。
- 20000次checkpoint描述性配对CI，另附每condition题目bootstrap/McNemar。不得把混合aux/main表当成新正文统一协议。
- O12在O10之后按预算判断执行。预先有协议和单测不代表已经评测完成。

## 8. 代码、结果和数据入口

以下路径均相对仓库根目录，除非写了绝对路径：

| 文件/目录 | 作用 |
| --- | --- |
| `reports/functional_hns_three_seed_20260914/` | 唯一主评测协议、配置、缓存、matrix、主HNS权重审计 |
| `reports/analysis_section5/source_checkpoints.json` | 18主source路径/seed/config/weights/HNS/激活缓存哈希 |
| `reports/posthoc_flat_dghard_forgetting_three_seed_20260913/` | 旧Slurm992的Flat adapter和旧波次；E0只复用其已核SHA的adapter |
| `reports/posthoc_baselines_para_spectral_surgery/` | 旧PARA、SS-grad_direction-o/down缓存 |
| `reports/hns_energy_matched_20260914/` | Scalar-E/Flat-E缓存 |
| `reports/hns_posthoc_lora_normalization_20260910/` | 早期数值/标量对照内部材料，不能混入统一主波次数值 |
| `reports/hns_training_seeds_3task_20260913.md` | 旧seed43/44训练warmup=0等审计背景 |
| `reports/rebuttal_iclr27/completion_report.md` | 第一轮总结+第二轮各阶段完成链接；开头“queue empty”是第一轮历史快照 |
| `reports/rebuttal_iclr27/completion_summary.json` | 第一轮最终哈希/完成性索引 |
| `reports/rebuttal_iclr27/R2-summary.json` | 第二轮已完成各项summary |
| `reports/rebuttal_iclr27/R2-plan.json` | 第二轮预算截止、O11批准状态 |
| `reports/rebuttal_iclr27/R2-submission.json` | 作业ID/依赖/命令/日志；`state`是早期提交快照，**不是实时状态** |
| `reports/rebuttal_iclr27/R2-publication.json` | 最近一次自动commit/push结果；未提交的本文件更新通常正常 |
| `reports/rebuttal_iclr27/R2-C-budget.json` | C每阶段完成/预算跳过记录；O9未完成时可能尚不存在 |
| `scripts/rebuttal_iclr27.py` | 第一轮共用、主cache审计、SHA/TSV工具 |
| `scripts/rebuttal_iclr27_e2_e3.py` | E2/E3编辑、预算、相关性、共同SVD helpers |
| `scripts/rebuttal_iclr27_e4.py` | E4训练/验证/选优/题目级bootstrap与McNemar；后续复用 |
| `scripts/rebuttal_iclr27_r2.py` | 第二轮共用协议锁、缓存view、builder、分组vLLM worker、统计、自动publish |
| `scripts/rebuttal_iclr27_r2_a.py` | A档O1→O4顺序执行和特定图/审计 |
| `scripts/rebuttal_iclr27_r2_b.py` | B档O5→O8训练/评测 |
| `scripts/rebuttal_iclr27_r2_c.py` | 正在运行的C driver，O9/O10预算门控，O11占位，O12调用 |
| `scripts/rebuttal_iclr27_r2_o12.py` | auxiliary benchmark完整生成/评分/审计 |
| `scripts/rebuttal_iclr27_r2_assets.py` / `...r2_o12_assets.py` | 已完成官方模型/数据下载与固定版本记录，不需重复下载 |
| `scripts/rebuttal_iclr27_r2_factor_audit.py` | 已完成的0GPU factorization represented-update审计 |
| `scripts/rebuttal_iclr27_r2_finish.py` | 最终核对O1–O8、可选状态、0GPU审计并写总索引/推送 |
| `slurm/rebuttal_iclr27_r2_{a,b,c,finish,factor_audit}.slurm` | 对应作业脚本；已提交，不能当作待提交清单 |
| `scripts/eval_forgetting_matrix_vllm.py` / `score_forgetting_matrix.py` | 未改的主推理/评分入口 |
| `src/finetune/spectral_edit/{posthoc_hns,svd,cli}.py` | HNS谱函数、balanced重建、模块选择/CLI |
| `src/finetune/train_sft_peft.py` | warmup显式步骤/actual Trainer audit、LoRA+实现入口 |
| `tests/test_rebuttal_r2.py` / `test_rebuttal_r2_o12.py` | 3项核心+10项辅助parser单测 |

E4训练数据和验证集都在 `E4/runtime/data/`；六个rank16训练的半程/final、r64两阶段在 `E4/runtime/training/`，O1复用了它们。具体路径通过 `E4/{task}_selection.json`、`rank64_training_audit.json` 和各 `protocol.lock.json` 读取，不猜路径。

新模型目录为 `R2-assets/runtime/models/Qwen3-1.7B`、`Qwen3-4B`、`Qwen3-14B`。官方模型信息在对应同名JSON；**Qwen3.5-4B不是Qwen3-4B，不可替代。**runtime权重、jsonl预测、optimizer/checkpoint文件一般被Git忽略，但恢复/复核会用到，不能清理。

## 9. 常用安全检查与故障恢复

### 9.1 只读状态/完成性检查

```bash
squeue -u "$USER"
scontrol show job -d 15429
sacct -D -j 15389,15390,15429,15431,15441 -o JobID,State,Elapsed,Start,End,ExitCode
tail -n 40 logs/rebuttal27-r2-c-15429.err
```

不要只看顶层.out：不少输出被重定向到每项 `runtime/logs`。训练看 `training_commands_*.json` 最后一条的 `log`；重跑会带后缀。`training_worker*_status.json` 的时间是阶段开始时间，不是每步心跳。最终adapter文件存在也不是完整实验完成；必须检查summary和审计。

验证已完成包（仅CPU读结果，不启动模型）：

```bash
PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python -c '
import rebuttal_iclr27_r2 as q
for i in range(1, 9):
    q.require_complete(f"R2-O{i}")
print("O1-O8 reports, coverage, integrity and stored output SHA256 verified")
'
```

新阶段完成后把范围扩到实际完成项，不要用未完成项的不存在summary当作失败。统计脚本用依赖哈希断言防止协议漂移；失败时先解释哪个文件变了，不要直接禁用assert或静默重写锁。

### 9.2 恢复原则

- 15429仍在运行时，不能从login运行 `--pipeline`、`--train-worker`、`--build` 或评测入口；也不能再提交第二个2GPU作业。
- 若失败/timeout，先确认Slurm状态、GPU进程已退出、失败的是训练/生成/评分/审计哪层，再在原授权范围内恢复。已有已核验缓存/adapter可以复用，避免重复训练和重复cell。
- `--pipeline`有部分恢复能力（跳过complete实验/已有最终source/COMPLETE生成），**不等于所有中断训练均自动resume**。看训练argv和检查点，必要修复要留记录。
- 主builder必须独立CUDA子进程退出。SS的原CLI在builder同一进程运行，防止exclusive context二次占用；不能随意把它改为又起另一个CUDA子进程。
- 修复源码若在protocol锁中，会触发锁校验。保存原SHA/失败日志/变更原因，核实是否只是调度/数值审计修复；不能悄悄把超参、数据或测试解析改了还声称同一协议。
- 若15429最终失败，15431的afterok依赖可能变成DependencyNeverSatisfied；恢复成功后要核对并修正依赖，不能漏掉最终索引。当前不用更改。

### 9.3 自动提交与用户已有工作

流水线每项完成后会自动 `git add` 该实验目录、更新索引、commit并通过SSH推送。它要求没有其他已暂存文件；**不要长时间留staged改动，与自动publish抢Git索引。**修改源码也可能使正在运行的后续锁失败。

开始本任务前已存在且仍需保留的用户改动包括：

- `src/finetune/data/registry.py`
- `tests/test_sft_preprocessing.py`
- `src/peft_sft_lab.egg-info/PKG-INFO`、`SOURCES.txt`
- 大量更早的未跟踪日志/实验脚本；例如 `src/finetune/data/commonsense170k.py` 等并非本轮新写。

不要 `git add .`、reset、clean，或把这些用户工作混入rebuttal提交。`R2-publication.json` 的未暂存更新是本轮自动推送元数据，应与用户修改区别对待。

HTTPS远程push没有可用凭据；已有可工作的推送方式：

```bash
git push git@github.com:TTtianTT/peft-sft-lab.git refactor/sft-chat-template
```

无需重新配置/打印token。`R2-publication.json` 的 `push_returncode=0,status=pushed` 才说明最近结果已推送；Git分支ahead可能是SSH直推未刷新tracking，不代表push失败。

## 10. 接手后的完成标准与论文表述

剩余实际工作：监控并核验O9；按已锁预算继续O10/O12；O11维持待批准；读O7/O8原始输出做必要诊断；检查15431最终索引/推送；给作者一份含正负结果、条件、CI口径、失败记录和代码提交号的总结。

最终脚本应产生 `R2-completion_report.md`、`R2-completion_index.json`、`R2-method_summary.tsv`，并更新根 `completion_report.md`。这些文件现在未必存在，不能提前引用为已完成结果。第一轮及第二轮每阶段四个主输出都应可复核；raw runtime保留。

可用的当前叙事：主18队列上HNS相对Recon有收益；精确nuclear展平与HNS未发现可靠优势差；nuclear恢复优于Fro恢复的Off/FG结果有支持；E3固定析因设计同时显示形状与幅度作用及非零交互；训练充分性、训练阶段、rank和三seed有正面证据；SS模块/策略补齐后仍不及HNS。

必须同时保留的限制：因子化会改变token；跨0不证明等价；高rank不证明HNS优于Flat；小模型和MetaMath LoRA+有实质失败/权衡；8B的LR未经重新调参直接迁移到其他规模；单seed题目CI不是训练runCI；验证调参的全局γ尚未做，不能用data-free γ代替回答；LoDA可组合仅是未来假设。

**不能为了让rebuttal更好看而选择性重跑、删条件或把上述负结果藏进“异常无效”。有新解释或修复，就保留原始结果、说明证据和协议差别。**
