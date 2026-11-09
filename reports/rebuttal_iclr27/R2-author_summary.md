# 第二轮 rebuttal 实验：给作者的总结（2026-11-09）

> 所有数字取自已提交的 `report.md` / `paired_ci.tsv` / `summary.json`；本文不引入新的主指标。
> 索引：[R2-completion_report.md](R2-completion_report.md)；最终索引提交 `52346cb`，O7 诊断 `68d731b`。

## 0. 状态

- R2-O1–O12 **全部完成**，均通过 `require_complete`（输出 SHA、覆盖、样本输入哈希、重建误差审计），已推送到 `refactor/sft-chat-template`。
- 第二轮合计 **952 个新 cell、17 次新训练**；全程经 Slurm，全局 ≤2 GPU。
- O11 在作者明确批准后执行（批准记录见 `R2-plan.json`、`R2-C-budget.json`）。

## 1. 统计口径（写回复时请统一沿用）

- 主指标：Target = 训练任务对应族；Off = 其余 3 族等权均值；FG = 各非目标族 max(Base − 编辑后, 0) 的均值。HumanEval164 pass@1、GSM8K1319 strict、IFEval541 prompt-strict、Commonsense 8 子任务 macro（22419 题）。
- 多 checkpoint 条件（主 18 队列、O2–O4、O5、O9、O11、O12）：**配对 checkpoint bootstrap**，20000 次，seed 20261106。这是共享固定测试集上的**描述性**区间，**不是**训练运行 IID 的置信区间。
- 单 seed 条件（O1、O6–O8、O10 等）：题目级配对 bootstrap + 二元指标 exact McNemar（附修复/破坏数）；Commonsense 按子任务重采样后 macro。不能写成训练层面的不确定性。
- 区间跨 0 **不等于**等价；只能说“未发现清晰差异”。
- 没有任何设置是按测试分数选的；所有条件全部报告。

## 2. 正结果

| 实验 | 内容 | 关键结果（HNS − Recon-0，pp，[95% 区间]，胜/平/负） |
| --- | --- | --- |
| 主 18 队列（O2/O3/O4 共用） | Qwen3-8B / Llama-3.1-8B × 3 任务 × 3 seed | Target +3.61 [2.08, 5.35] 16/0/2；Off +3.03 [1.76, 4.36]；FG −2.23 [−3.26, −1.38] |
| O1 | LR × 训练阶段 14 个 checkpoint | 14 个条件 HNS − Recon Target 点估计全为正（Magicoder +4.3～+9.1，MetaMath +3.5～+5.2）；14 个相关 checkpoint，非独立训练 |
| O2 | unbalanced 写回 | hns_unbal − recon_unbal Target +3.73 [2.34, 5.26]；与 balanced 增益差中差均跨 0（未发现差异，非等价）；更新审计最大相对差 1.75e-7 |
| O3 | SS 完整 2×2 | HNS 全面优于四种 SS，例如 HNS − SS-grad_direction-all：Target +4.81 [2.78, 6.92]，FG −5.55 |
| O4 | 阶段/强度 | 4+0、0+5、4+1 结果接近；λ=0.5 较弱（4+1 − 4+1λ.5 Target +1.40 [0.42, 2.48]） |
| O5 | Qwen-8B 每任务 3 seed | MetaMath HNS 88.02±1.10 vs Recon 84.71±0.31；Magicoder 73.37±0.35 vs 65.04±0.93（仅 3 seed） |
| O6 | rank 8/16/32/64 | 四个 rank 的 HNS − Recon Target 均为正（+3.4～+4.6），FG 均下降 |
| O9 | Llama 新训练（Magicoder LR1e-5、MetaMath LR5e-5） | MetaMath Target +5.08 [2.88, 7.35]，Off +4.16，FG −3.96；Magicoder Off +3.72 [2.11, 5.34]、FG −3.21，但 Target −1.83 [−7.93, 4.88] |
| O12 | 独立目标基准（MBPP+、MATH-500） | +4.95 [3.63, 6.43] 12/0/0；MBPP+ +3.00 [2.51, 3.53]，MATH-500 +6.90 [5.30, 8.60] |

## 3. 负结果与限制（不能删，回复中需如实呈现）

### 3.1 O11：验证集选优的全局 γ 几乎追平 HNS（对“只是缩放吗”的直接回答）

- 设计：γ·Recon-0，γ ∈ {.3,.4,.5,.6,.7,.85,1}，每个 checkpoint 用训练任务验证集选（Magicoder→MBPP200，MetaMath→SVAMP500，Tulu→argilla ifeval-like 500），平票取较大 γ；126 个验证分数全部公开。
- **HNS − γ_val（18 checkpoint）**：Target +0.28 [−0.71, 1.72] 7/3/8；Off +0.31 [−0.09, 0.72] 14/0/4；FG −0.11 [−0.26, 0.03]。
  - Qwen3-8B：Off +0.67 [0.27, 1.17] 9/0/0（偏向 HNS）；Target +0.89 [−0.74, 3.57]。
  - Llama-8B：三个指标的点估计都在 ±0.35 以内，区间均跨 0。
- γ_val 相对 Recon-0：Target +3.33 [2.13, 4.70]，FG −2.12，与 HNS 的增益量级相同。
- **重要限制**：18 个 checkpoint 中有 **11 个选中网格下界 γ=0.3**，所有 MetaMath 均如此，且验证分数随 γ 单调上升。这说明预锁网格的下界在起作用，真实最优缩放可能更小；按协议不扩网格，也不按测试分数调整。
- 可辩护的表述：主队列中 HNS 的收益大部分可以由“**用验证数据**调出的统一收缩”复现，而 HNS **不用任何数据**（γ_val 每个 checkpoint 需 7 次验证推理），仅在 Qwen 的 Off 上略占优势。**不能**再声称谱形状的收益已被证明超出缩放。

### 3.2 O7：小模型（单 seed，LR 直接沿用 8B，未调参）

| 模型 | 方法 | Target（主指标，GSM8K strict） | Off | FG |
| --- | --- | ---: | ---: | ---: |
| 1.7B | Base / LoRA / Recon / HNS / Flat | 4.02 / 65.20 / 65.13 / **5.23** / 5.69 | 63.7 / 44.5 / 44.0 / 51.2 / 51.3 | 0 / 19.3 / 19.7 / 12.5 / 12.4 |
| 4B | Base / LoRA / Recon / HNS / Flat | 64.14 / 54.28 / 56.63 / **85.97** / 85.60 | 68.1 / 63.5 / 63.6 / 61.1 / 61.6 | 0 / 5.0 / 5.1 / 7.6 / 7.1 |

- 主结果不变：1.7B HNS − Recon Target **−59.89** [−62.63, −57.16]（修复/破坏 14/804）；4B HNS 的 Off −2.43 [−5.42, 0.53]、FG +2.44，其中 HumanEval −23.17 [−31.71, −15.24]。
- **辅助诊断**（只读、事后定义，不替代主指标；见 `R2-O7/aux_gsm8k_format_diagnostic/`）：GSM8K 提示要求最后一行写 `#### <answer>`。1.7B HNS 有 92% 的输出把字面量 `<answer>` 抄进答案行（如 `#### <answer>\n18`），strict 解析因此失败；1.7B Base 只有 31.5% 的输出带 `####`。只去掉字面 `<answer>` 标签、其余规则不变时：
  - 1.7B：Base 30.3、LoRA 74.6、Recon 74.3、HNS 77.9、Flat 78.0；
  - 4B：Base 81.4、LoRA 83.9、Recon 83.7、HNS 88.2、Flat 87.9（4B LoRA 有 34% 抄占位符，HNS 只有 1%）。
  - 结论：O7 两个方向的 GSM8K 巨幅波动（−60 / +29 pp）主要来自**提示占位符与 strict 解析的格式交互**，而非数学能力本身。这一判断是事后诊断，主表仍需原样报告。4B 的 HumanEval −23 pp 不在本诊断范围内，仍是负结果。
  - 8B 主队列中占位符抄写率 ≤3%；在两种解析下，HNS 都高于 Recon 和原始 LoRA（Llama strict 70.8 vs 66.6，aux 80.7 vs 74.8；Qwen strict 88.4 vs 85.4，aux 91.5 vs 87.8）。Llama 的 strict 与 aux 差距主要是 `$N` / `N.00` 这类货币格式。

### 3.3 O8：LoRA+（单 seed，比例 20，沿用 E4 LR）

- Magicoder：HNS − Recon Target +6.71 [1.22, 12.20]，Off +3.78，FG −2.46，方向为正。
- **MetaMath**：源 adapter 遗忘严重（Recon FG 34.15，Off 38.8 vs Base 72.9）。HNS Target −1.14 [−2.65, 0.38]，Off +1.77 [−0.70, 4.27]，FG 仍为 32.38。在严重损伤的源上，HNS 只能小幅恢复。不能声称 HNS 对训练策略无条件鲁棒。

### 3.4 O10：Qwen3-14B MetaMath（单 seed，修正 1：32768 batched tokens）

- 各方法 Target / Off / FG：Base 88.25 / 80.71 / 0；LoRA 86.13 / 71.73 / 8.98；Recon 86.20 / 71.84 / 8.87；HNS 85.82 / 76.75 / 4.15；Flat 86.58 / 76.29 / 4.54。
- HNS − Recon：Target −0.38 [−2.27, 1.52]（未发现差异），Off +4.90 [2.40, 7.45]，FG −4.72 [−7.20, −2.17]。
- 14B 上 LoRA 训练本身使 GSM8K 低于 Base；HNS 的作用主要是减轻遗忘，没有提升目标任务。Flat − HNS Target +0.76 [0.15, 1.36]。

### 3.5 HNS 与精确 Flat-Nuclear

- O12 HNS − Flat：Target +0.02（主）、独立目标 +0.20 [−0.30, 0.61]；O6 中不同 rank 的方向不固定；O10 中 Flat 的 Target 略高。
- 叙事应写成“**保持 nuclear norm 的谱展平**这一类变换”，HNS 是其中一种高效实现，**不**宣称 HNS 优于精确展平。

## 4. 故障与修正记录

| 时间 (+08) | 作业 | 事件 | 处理 |
| --- | --- | --- | --- |
| 11/8 15:32–16:27 | 15429 第一次尝试 | node02 `NODE_FAIL`（节点故障，非实验失败） | Slurm 自动在 node03 重启（16:40）；新日志带 `.job15429.3` 后缀 |
| 11/8 23:37–23:39 | 15429 | O10 预声明 fit check 失败：在 65536 batched tokens 下，Inductor 内核 `cudaErrorIllegalAddress`（14B gate_up 65536×2×17408 = 2.28e9 元素，超出 int32 索引范围；权重只占 28 GiB，不是显存不足） | 在任何 14B 训练/推理前，经作者批准修正为 **32768，仅限 O10**，Base 和四个方法一致；`R2-O10/protocol.md` Amendment 1 与 training lock 已记录；失败日志已入库 |
| 11/9 00:42–05:16 | 16484 | C 档重跑 O10 → O12，完成 | 原依赖作业 15431 改为依赖 16484 |
| 11/9 11:57–14:16 | 16810 | O11 完成（126 次验证 + 72 个测试 cell） | — |
| 11/9 14:17 | 16952 | 最终索引按 O11 已完成的实际状态更新，并推送 | finish 脚本的修改提交于 `993440e` |
| O11 锁定前 | — | Tulu 验证数据中出现主评分器不支持的指令类型（constrained_response） | 在锁定和推理前收紧为仅使用评分器支持的类型，排除 75 行；参考答案通过率 420/500 |

O11 的验证集说明：MetaMath 没有使用 E4 的 held-out 500 题（与训练用的 MetaMathQA 50K 同源，可能被旧 adapter 见过），改用外部 SVAMP500。三套验证集与本地训练数据、IFEval541 的精确文本重叠均为 0；但 seed42 的 Qwen adapter 在另一台机器上训练，无法逐行重建其训练数据，因此该检查不覆盖它们，已写入协议。

## 5. 提交

| 提交 | 内容 |
| --- | --- |
| `a8bf47c` | R2-O9 报告 |
| `060a429` | O10 修正 1（32768） |
| `ebfd4c3` | R2-O10 报告 |
| `fef347a` | R2-O12 报告 |
| `2d8f36f` | 第一版完成索引（O11 待批准） |
| `58f4e6a` / `0972e51` | O11 协议锁定 / Tulu 验证集 |
| `a0936f6` | R2-O11 报告 |
| `993440e` / `52346cb` | finish 脚本修正 / 最终索引（含 O11） |
| `68d731b` | O7 GSM8K 格式辅助诊断 |
| 本文件 | 作者总结 |

## 6. 写回复的建议

1. **核心论点**：data-free 的 nuclear 保持谱展平，在主 8B 队列、多 LR/阶段/rank、独立目标基准（O12）上稳定减轻遗忘并提升或保持目标性能。
2. **对 8Euc（只是缩放吗）**：如实报告 O11。验证集调出的统一收缩可以达到与 HNS 相近的效果。HNS 的价值在于不需要验证数据、不需要逐 checkpoint 搜索；**不要**声称谱形状的增益已超出缩放。
3. **对 FzQ5（规模/训练策略）**：报告 O7、O8 MetaMath、O10 Target 的负或零结果；O7 附上格式诊断，但要明确标注为事后辅助分析。
4. **对 XJ3f（与精确展平的关系）**：HNS 与 Flat-Nuclear 没有可靠差异，两者同属一类变换。
