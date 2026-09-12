# PIQA signed-gate 756 复核与下一步决策

复核日期：2026-09-10。原运行结果保持不变；本次只读取已有产物并进行 CPU 复算，没有启动新的 GPU 作业。

## 决策

结束本轮 PIQA 单方向 accuracy 扫描，不反转 signed predictor，不依据 locked validation 重新选方向或剂量。下一轮转向 Qwen MetaMath，但先做 **reward-gradient 测量与三条预定义整体编辑路径的可辨识性检查**，通过后才进入模块/方向选择。

## 已核对的事实

- Slurm 756：COMPLETED，exit 0:0，01:44:15，单 GPU。
- Validation：55/55 个版本（含 LoRA），每个版本 512 个唯一且一致的题目 ID；无 invalid output；correct 字段与预测字母/金标准一致。
- signal/dose/validation 的 128/128/512 个题目 ID 不重叠。dose 与 validation 的新 LoRA 输出均与原来源预测逐题一致。
- 已有 48 个单方向 adapter 的存盘谱核验为 PASS，最大相对谱误差 9.05e-7，其他模块张量没有改动。此检查不等价于推理数值不变性检查。
- Full HNS：453/512，LoRA：460/512；9 修复、16 破坏，净 -1.3672 pp。原 paired bootstrap CI [-3.3203,+0.5859] pp，exact McNemar p=0.2295。

## 汇总表之外的发现

### 1. 单方向效果只由一到两道题支撑

| 剂量 | 零翻转方向 | 一题翻转方向 | 涉及的独立题目 |
|---|---:|---:|---:|
| 10% | 13/24 | 11/24 | 1 |
| 100% | 13/24 | 11/24 | 2 |

10% 的 11 次翻转全部是 `How do you cut a circle ruffle?` 的修复。
100% 的 11 次翻转由同题的 9 次修复，以及 `How can you spread butter if it is cool or too hard to spread?` 的 2 次破坏组成。

事后敏感性分析：去掉 circle-ruffle 一题，100% 剂量的 Spearman 从 -0.4141 变成 +0.0653；10% 的所有方向效用变为零，相关系数无法定义。该分析仅说明结果被单题主导，不能用于重新选策略或作新的验证检验。

Spearman permutation p=0.045 没有单独解决这些方向共用题目、结构依赖及题目抽样不确定性的问题。主要 matched contrast 的区间跨零，不能建立反向选择规律。

### 2. 剂量选择没有得到有效剂量证据

dose split 的 predicted-suppress 组在两个剂量的平均 accuracy gain 都是零。两档弱正 matched contrast 都来自 retain 组同一道题的一次破坏。最终选 10% 是预注册的平局规则，不能称为经效用验证的最佳剂量。

### 3. 不能把失败已经归因为 margin 梯度样本外反号

signal split 的 restricted first-token/parser agreement 和 global first-token choice fraction 均为 128/128。至少在此 split，格式或选项 token 不一致不是主要解释。

所选 24 个方向在原 signal 的前后 64 题中，预测符号全部一致，Spearman=0.9913。但两半均参与原方向选择，因此这是偏乐观的内部诊断，不是新的泛化证据。

信号集绝对 margin 中位数为 6.8125，只有 9/128 题的绝对 margin 小于 1。所选方向的全剂量预测**平均** margin 改变量绝对值中位数为 0.001455。这两个统计量粒度不同，不能计算逐题翻转概率，但提示全样本平均 margin 与边界附近 accuracy 变化的尺度差异。

已有有限干预文件只保存生成答案，没有保存编辑后的连续 margin，也没有 validation 上重新估计的 margin gradient。因此目前无法区分：

1. 梯度不能预测同题有限编辑后的 margin；
2. margin sensitivity 在新题目上变化；
3. margin 能预测，但改善集中在本来就会做对的题，未转化为 accuracy；
4. 极小效应中的数值实现影响。

最稳妥结论是：**当前首 token margin 排名没有建立可用的真实 accuracy 预测能力，且本轮单方向干预任务效应过于稀疏；失败原因尚未定位。**

### 4. 单题组合效应不足以解释机制

AB、retain block、scalar 修复同一题，无法归因于 signed selection、谱形状或协同。Scalar gamma=0.999863565，即总 LoRA norm 只下降约 0.01364%。

构造器对编辑模块重建 SVD 因子，scalar 对全部模块重建；运行使用 bf16。协议中没有零剂量 SVD 重构对照或同路径数值差分核验。因此应在下轮测量器的 smoke test 中加入这些检查。当前证据并不能断言这一题是数值伪影。

零翻转时 empirical bootstrap [0,0] 是重采样的退化结果，不代表总体效应精确为零。512 个独立题目零翻转的单侧 95% 二项上界仍约为 0.5834% 的翻转概率；不能据此建立 ±0.5 pp 总体等效性。

## 下一轮：先验证测量器，再做 task-aware selection

### 第一步：小规模数值检查

在新测量器中使用显式的谱门控，保持原 LoRA 基线计算路径，门控增量为零时不重新分解/替换全部因子。

- 原始 LoRA 与零门控分支比较；如使用导出 adapter，再加 zero-dose SVD 重构对照。
- 8–16 条短样本上，对相同固定序列的 log probability 做自动梯度与有限差分比较；使用两个以上步长，避免 bf16 舍入淹没差分。
- 同时保存连续 score、梯度和实现精度，核验门控方向、LoRA scaling、响应 token mask。
- 梯度数值核验的对象是可微 log probability，不是离散 verifier reward 的单样本数值导数。

这属于测量器检查，不是 PIQA 重新筛选。若额外检查旧边界题，只作为已观察题的数值诊断。

### 第二步：下一批主要计算预算限定为 512 条 baseline rollout

Checkpoint：Qwen3-8B MetaMath。

- 128 道校准题，每题固定 4 条独立采样，总共 512 条。
- 预先划为两个 64 题组，检验三条**事先定义**路径的信号稳定性；后续验证题独立锁定。
- 数据检查排除训练精确重叠；记录已有 benchmark 的使用历史，不声称这些题是从未使用过的新 benchmark。
- 一个固定随机解码协议，例如 temperature=0.7、top_p=1、无 top-k、无额外 repetition penalty；log probability 必须使用相同 temperature 和实际采样分布。保持已有 prompt、答案解析及最大生成长度，记录 EOS 和截断。
- 用现有答案 verifier 得到二元奖励；全对/全错组保留在总体平均中，不为了得到 mixed outcomes 事后筛题。
- 使用同题 leave-one-out reward baseline，对完整响应序列 log probability 的和求谱门控梯度，不对每条响应除以长度。

定义谱路径为 sigma(lambda)=sigma_LoRA+lambda*d。对每条固定路径估计：

\[
\widehat g_d=\frac1{NK}\sum_{x,k}(R_{xk}-\bar R_{x,-k})
\left\langle\nabla_\sigma\log\pi_{\sigma,T}(y_{xk}|x),d\right\rangle.
\]

三条路径：Full HNS、HeadOnly、与对应整体范数目标匹配的 ScalarShrink。先在**每条 rollout**内聚合方向梯度，再以题目为单位估计不确定性；不把几千个方向或响应 token 当独立样本。

报告：正确率、mixed-outcome 题数、截断/解析情况、三条路径的梯度与题目 bootstrap 区间、两个独立题组上的信号符号，以及信号是否被极少数题支配。全对/全错比例高或区间宽时结论是信息不足，不强制输出新 adapter。

### 第三步：只有信号可辨识，才进行有限干预与 where selection

1. 先对预定义整体路径做少量、独立题目的随机解码 reward 验证，采用预先固定的小剂量，并保留原 Full HNS 剂量用于检查局部近似边界。一个梯度即使稳定为负也可验证，不以“必须预测 HNS 正收益”作为通过条件。
2. 首要检验是 reward gradient 能否预测相同随机解码协议下的有限效应；greedy 是后续单列终点。既有 greedy HNS 正收益不能当作随机 reward 梯度必须为正的真值。
3. 若整体路径上都不能得到可复现的预测关系，暂停 reward-gradient 选方向路线，区分采样方差、梯度实现与有限剂量非线性，不直接扩大到大量单方向。
4. 若通过，再做 8 个预定义结构块（4 个层段 × attention/MLP），先提出集合、实际联合验证，再考虑更细的方向。保留 LoRA 和 Full HNS 两个整体候选，不假定单模块效用可加。

此阶段顺序是：先检验 whether/how strongly 的测量基础，再解决 where，避免再次用几乎不改变任务输出的单方向扫描消耗验证预算。

## 依据与产物

- 原运行根目录：`/dataset1/zailong/runs/peft-sft-lab/hns-signed-gates-20260910/qwen_commonsense_piqa`。
- 本次逐题 CPU 复算：`reports/hns_signed_gate_review_20260910.json`。
- REINFORCE/RLOO 的 sequence-level 估计及 leave-one-out baseline 依据：[Back to Basics](https://arxiv.org/html/2402.14740v2)。将它用于 HNS 谱门控是拟验证的方案，论文没有证明它可以预测本项目的有限干预收益。
