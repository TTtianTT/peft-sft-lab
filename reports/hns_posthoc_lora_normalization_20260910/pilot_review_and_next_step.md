# Slurm 768 复核与下一步建议

日期：2026-09-10。只读取已有结果并进行 CPU 配对复核；未提交 GPU 作业。本文不修改原实验协议、终点或停止规则，后续方案尚未执行。

## 决策

停止 HNS 导出的逐模块 scalar normalization 分支，保留 Full HNS 的谱编辑研究。下一项优先工作应是同一数学 checkpoint 上的数值实现控制，确认 HNS 相对逐模块范数匹配 scalar 的优势是否对实现方式稳健。暂不展开方向选择、跨 checkpoint transfer 或 data-conditioned normalization。

## 已有证据

- Full HNS vs PerModule：严格评分修复 29 题、破坏 10 题，净增 19/512 = 3.7109 pp；配对区间 [1.37, 6.05] pp，exact McNemar p=0.00337785。
- Numeric：修复 29、破坏 6，净增 23/512 = 4.4922 pp。
- PerModule vs Global，以及 PerModule vs shuffled-family mean，均未支持正向的模块分配价值。
- 这些结果排除了“本轮匹配的逐模块 Frobenius 范数足以复现 HNS 表现”的解释；不能排除其他 scalar 剂量，也不能单独排除因子表示和数值计算差异。

## CPU 事后敏感性分析

按题目 ID 对齐 direct-dose 与本轮 validation。两次 LoRA 或 HNS 的 extracted prediction 发生变化的题目并集为 22 题。

| 子集 | 题数 | HNS 相对 PerModule：Strict 修复/破坏 | Strict 净增 | Numeric 修复/破坏 | Numeric 净增 |
|---|---:|---:|---:|---:|---:|
| 全部 validation | 512 | 29/10 | 19，+3.7109 pp | 29/6 | 23，+4.4922 pp |
| 排除 22 道已观察到的不稳定题 | 490 | 24/9 | 15，+3.0612 pp | 24/6 | 18，+3.6735 pp |
| 22 道不稳定题 | 22 | 5/1 | 4 | 5/0 | 5 |

这说明当前优势并非完全由已观察到的跨运行不稳定题支撑。子集是在看到输出后定义的，不能将其 p 值或区间作为新的确认性推断。PerModule 没有同条件重复，因此也没有证明其余题目或该方法本身数值稳定。

以此前 HNS 输出替换当前 HNS、仍使用当前 PerModule，仅作描述性比较时，Strict 净增仍为 15/512 = 2.9297 pp，Numeric 为 18/512 = 3.5156 pp。这不是同次运行的受控比较，也不是独立复现。

## 三个解释边界

1. **表示方式不一致。** 本轮 scalar 直接缩放原始 B，A 完全不变；HNS 加载已有重构 adapter。相同目标范数没有控制 A/B 表示及实际低精度计算。此前 alpha=0 重构能够改变正确状态，故这一项需要直接控制。
2. **固定范数匹配系数不是调优 scalar。** Global gamma=0.6088122867。它适合机制对照，但未回答 HNS 是否优于经过相同校准预算选择的全局 gamma。
3. **题目复用。** 协议明确复用了 direct-dose 的 256/512 划分。本轮没有从 validation 选择系数，但它不是新样本上的独立确认；连续根据同一集合的结果调整研究假设，也使项目层面的确认性解释受限。

97.5%–97.9% 的答案一致率不能转换成通用的“低于 1 pp 都是数值噪声”阈值，也不能证明大于该值的差异必然是谱形状效应。当前配对区间主要反映题目抽样不确定性，不包含完整的引擎运行间或因子表示不确定性。

## 推荐的下一步：先做实现控制，再决定是否确认性评测

### A. 小样本数值诊断

在已有题目上固定模型、adapter、dtype、prompt、解码、请求顺序和批处理配置。可以包含已知不稳定题与预先随机抽取的题，但此集合只用于调试。

- 同一 adapter 在新进程中重复，记录完整输出 token ID、正确状态和实际运行配置；更换标签/加载顺序检查是否有调度或缓存因素。
- HNS 与 matched scalar 使用共同的奇异向量基底和因子构造。一个明确设计是各模块共同使用 A=V^T，只通过 B=U diag(d) 改变谱，统一处理 PEFT scaling；固定本轮 HNS 目标，不顺便修改迭代次数或剂量。
- 同时保留原始 LoRA、零谱编辑重构 LoRA、原始因子 scalar，测量纯因子重构的行为差异。若重构影响与主效应相当，应先解决表示/精度问题。
- 共同路径仍不能保证处理相关的舍入误差消失；必要时用精度可控的 LoRA 分支或第二种共同表示做诊断。提高精度的诊断不能冒称已经复现实际 bf16 部署策略。

当前 eval_gsm8k_adapter_set.py 设置了 greedy SamplingParams 的 seed，但没有显式配置确定性调度或 batch invariance；启动脚本同样没有设置相应开关。这是待检查因素，尚未证明是本次不一致的具体原因。vLLM 官方明确默认不保证可复现，可从离线确定性调度或 batch invariance 开始检查，但必须验证当前版本、Qwen 和 LoRA 路径的实际行为，不能只靠开关宣称已解决：

- https://docs.vllm.ai/en/latest/usage/reproducibility/
- https://docs.vllm.ai/en/stable/features/batch_invariance/

### B. 数值诊断通过后，只做两个明确比较

| 比较 | 意义 |
|---|---|
| 同一路径 Full HNS vs 同一路径 PerModule | 核验逐模块范数之外的谱编辑效果 |
| Full HNS vs 校准集选定的 global scalar | 核验实用价值是否超出简单剂量选择 |

第二项的 scalar 候选范围、选择规则、平局规则事先固定，包含 gamma=1；严格评分为主，numeric 为次。新确认阶段使用未参与此前方法选择的数据，不能将已经看过的 GSM8K 题重新划分后称为未见验证。没有新数据时，可以进行同题实现复核，但只作该范围内的解释。

重复运行首先用于检查实现稳定性，不能把同一题的多个输出当成独立样本扩大 n。两项主要比较及成功/停止标准应预先明确，保留“不确定”结果，不按 p 值反复追加样本。

## 结果如何决定研究路线

- 两个比较均得到稳健正证据：继续以 post-hoc spectral balancing/refinement 为方法定位，之后才考虑跨 checkpoint 复现。
- HNS 优于 matched scalar，但未优于校准 scalar：保留谱形状的机制观测，收紧实用优越性主张。
- 优势随表示或运行配置明显改变：优先报告实现依赖和机制边界，不继续构造方向级 predictor。
- 区间仍宽：报告当前预算下不可辨识，不将不显著解释为等效。

“LoRA Normalization”不能代替这些检验。本轮失败的是特定的谱驱动逐模块缩放规则；Full HNS 的谱重分配仍有值得确认的信号，但还没有建立通用归一化方法。

## 依据

- 本目录 pilot_protocol.md、pilot_result.md、pilot_audit.json。
- hns-posthoc-scaling-pilot-20260910/qwen_metamath/eval/validation 下各版本 predictions.jsonl。
- hns-metamath-direct-dose-20260910/eval/validation 下 LoRA 与 Full HNS predictions.jsonl。
- scripts/eval_gsm8k_adapter_set.py、scripts/run_qwen3_posthoc_spectral_scaling_pilot.sh。
