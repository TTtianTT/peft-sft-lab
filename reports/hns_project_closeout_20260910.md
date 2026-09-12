# HNS 项目收束建议：截至作业 773

日期：2026-09-10。本文件依据已有报告、审计与构造代码整理研究结论和后续决策；未提交新 GPU 作业，也不改变各轮停止规则。

## 当前建议

暂停 HNS 的新方法扩展，进入结论整理与写作阶段。现有证据尚未建立 HNS 相对合理校准 global scalar 的额外实用价值，也未建立可靠的 task-aware 模块或方向选择方法。不要继续在同一批 GSM8K 题上追加 scalar 剂量、谱映射、局部梯度或方向筛选来追求显著性。

本结论适用于已审计的 Qwen MetaMath 设置。它不是对所有模型与任务上的 HNS 效果的普遍否定，也不是 HNS 与 scalar 的等效性证明。

## 可保留的贡献与解释边界

| 问题 | 可支持的陈述 | 尚不能支持的陈述 |
|---|---|---|
| 功能强度 | 既有受控 hidden-state 测量中，功能能量集中度高于原始谱集中度；HNS 降低极端更新 | 在所有实际生成轨迹上普遍成立；能量集中意味着该方向应被压制 |
| 任务效用 | suppression 的下游效用依赖任务，Commonsense 提供反例 | dominant modes 普遍属于冗余或噪声 |
| 谱形状 | 共同基底下 HNS 相对 PerModule 有未解析的正向趋势 | HNS 已被证明凭借形状稳定胜过 scalar |
| 实用比较 | 校准 scalar 在此次同题数值复核中获得接近 HNS 的表现 | HNS/scalar 等效、scalar 普遍更优，或 scalar 最优剂量已定位 |
| 数值实现 | 相同 adapter 在 32 题诊断中可跨进程重复；更换因子表示可改变生成 | 32 题的变化率是总体发生率；保存误差为零意味着运行时函数完全一致 |
| 效用预测 | 已测试的信号与预算没有给出可用方向选择器 | 所有 reward-gradient、局部因果信号或更大样本方法均不可能有效 |

旧 2×4 functional 测量使用固定的 pretrained-base hidden-state trajectory。写作时保留该口径，不将其替换成完整的 free-running LoRA trajectory 结论。

## 作业 773 的关键结果

- Common HNS − PerModule：Strict +1.3672 pp，95% CI [-0.7813,+3.5156]；Numeric +1.9531 pp，CI [0,+4.1016]，exact McNemar p=0.0987。因此 numeric 的区间端点为零不构成已经确认的正效应。
- Common HNS − calibrated global scalar：Strict +0.3906 pp，CI [-1.5625,+2.3438]；Numeric -0.1953 pp，CI [-1.7578,+1.5625]。
- Common HNS − zero rebuild：Strict +3.3203 pp，CI [+0.5859,+6.0547]。该比较同时包含幅度与谱形状变化，不能独立证明形状贡献。
- gamma=0.4 在 256 题校准集上从预设六点网格选择，且位于网格下界；没有测得 scalar 最优点，也不能据此推断 gamma=0 或任何更小 gamma 必然更好。
- 所有题目此前均包含在完整 1,319 题 GSM8K 评测中。本次属于同题数值与基线复核，不是独立确认。

HNS 未证明优越性足以支持停止方法扩展；它与“已经排除了任何有实际意义的 HNS 优势”不同。上述区间仍允许约 +2.34 pp 的严格准确率优势，当前没有进行等效性或非劣效性检验。

## 数值审计的限定

build_metamath_common_basis_controls.py 的 max_saved_update_relative_error 比较保存 dtype 的新因子与当前构造的新目标因子，不是原始 LoRA 与重构 LoRA 的运行时等价检验。该值为 0.0 不意味着此前的 HNS adapter、投影后的 common HNS 与实际 bf16 算子完全相同。

Common HNS 对既有 HNS 的最大基底对齐误差为 8.40e-4。共用基底明确了本轮谱对照，但同时重构了表示并近似对齐了旧 HNS。由于引擎设置与表示也有变化，不能将前后收益差值按比例归因为某一种数值因素。

32 题诊断包含先前已观察到的 22 道不稳定题，属于富集诊断集合。22/32 等 token 一致率用于证明现象存在，不用于估计一般题目的发生率。

## 下一阶段的具体产出

1. 整理一个主结果表：原始 LoRA、共同基底零编辑、共同基底 PerModule、共同基底 HNS、校准 scalar，并附配对修复/破坏、区间和数值口径。
2. 将“使用强度”和“任务效用”分开陈述；总结失败预测器的目标、样本量、实际非零信号和停止原因。
3. 将保存更新等价、运行重复性、因子表示敏感性分开审计。保留共同基底与确定性运行的可复用代码。
4. 写成机制与评测研究，而非目前尚未建立的 LoRA Normalization 新方法。稿件能否形成完整论文，仍需核查与既有工作的区别及证据范围；当前没有承诺发表或通用方法贡献。

可用工作标题：**Revisiting Post-hoc Spectral Editing of LoRA: Scale, Shape, and Numerical Representation**。

可用于摘要的核心段落：

> 我们研究 LoRA 后训练谱编辑的收益来源与失效边界。受控 hidden-state 测量显示，参数谱的集中性可被输入对齐进一步放大，且 HNS 能降低极端更新，但这些强度指标不足以预测下游收益。对 Qwen MetaMath 的进一步数值复核发现，因子表示会影响低精度生成，共同基底下的谱形状特异性收益仍未解析，校准后的全局 LoRA 缩放获得与 HNS 接近的表现。结果强调，评价谱编辑需要同时控制适配强度、因子表示及效用目标，且不支持从功能主导性直接推断方向应被抑制。

## 若以后继续方法研究：只保留一个有区别的新问题

可选问题是：**冻结的 HNS 在没有目标任务标签用于校准时，能否优于同样冻结的简单 scalar 默认规则？** 这是新的成本与迁移目标，不是用新指标重新宣布本次优越性检验成功。

若要检验，先冻结方法、scalar 默认值、任务选择标准、统计终点、最小实际收益和总预算，再使用未参与当前方法开发的 checkpoint 与数据。最小对照包含 base-only（gamma=0）、原始 LoRA（gamma=1）、冻结 scalar、冻结 HNS。base-only 用来判断是否获得了适配价值，不能在缺少结果时推断它已优于 LoRA。

若增加目标任务校准，HNS 与 scalar 应使用同等标签/评测预算；不能把依赖广泛目标任务试验选出的 HNS 设置描述为开发过程不使用数据。新的 checkpoint 检验迁移，不能自动使已经参与假设形成的 GSM8K 数据变成独立未见数据。

默认不启动此实验。只有“目标标签缺失或校准成本重要”本身是实际需求，且可以在固定预算内回答问题，才值得重开。若没有这样的需求，完成现有研究的整理就是下一步。

## 相关背景

全局 scalar 在数学上对应预训练权重与适配后权重的线性插值：

\[
W_0+\gamma\Delta W=(1-\gamma)W_0+\gamma(W_0+\Delta W).
\]

权重插值已有相关研究，例如 [WiSE-FT](https://arxiv.org/abs/2109.01903) 在视觉模型中研究其鲁棒性。该工作不能证明本项目的 LoRA 实验结论，但说明仅将缩放重新命名为 normalization 不足以建立新颖性。

## 原始依据

- reports/hns_posthoc_lora_normalization_20260910/common_basis_numeric_result.md
- reports/hns_posthoc_lora_normalization_20260910/common_basis_numeric_audit.json
- reports/hns_posthoc_lora_normalization_20260910/common_basis_numeric_protocol.md
- scripts/build_metamath_common_basis_controls.py
- reports/hns_mechanism_stage_conclusion_20260910.md
- reports/hns_2x4_mechanism_final_20260910.md

## 写作包

已停止新增实验并生成以下写作阶段产物：

- `reports/hns_paper_package_20260910/main_results.tsv`：统一的 2×4 谱、functional、modification、六组干预和 HeadFro 主表。
- `reports/hns_paper_package_20260910/claim_evidence_matrix.tsv`：逐项区分 supported、partially supported、unresolved 与 not established。
- `reports/hns_paper_package_20260910/failed_predictors.tsv`：集中记录相关、localization、utility、signed gate、reward-gradient、dose 和 scalar 对照的失败边界。
- `reports/hns_paper_package_20260910/paper_outline.md`：标题、中心论点、摘要草稿、章节结构、主图表和措辞约束。

这些文件不改变任何原始统计终点，也没有把不显著结果解释为等效性证据。
