# HNS 阶段性机制结论：截至 MetaMath direct-dose 作业 766

日期：2026-09-10。本文件整合已有实验的证据边界，不更改各轮预注册终点、原始运行产物或停止规则。复核只使用 CPU；不提交新 GPU 实验，不进入 module/direction localization。

## 当前结论

已有受控 hidden-state 测量显示，LoRA 谱方向的 functional energy 比参数谱更加集中，HNS 能降低 extreme modification。但使用强度本身不足以识别 excessive 与 task-essential directions，现阶段也没有建立可靠的 task-aware 谱方向选择规则。

MetaMath 的历史完整评测提供 HNS 有益的证据。最新小预算剂量实验中，HeadOnly、逐模块范数匹配 scalar 和 Full HNS 的收益点估计接近；HeadOnly 的形状特异性增益尚未被识别。整体幅度变化是一种有竞争力的解释，但目前不能给出缩放与谱形状的因果贡献比例。

## 作业 766 的独立复核

- Slurm：COMPLETED，exit 0:0，单 GPU，00:06:13。
- 256 条校准题、512 条锁定验证题，排除 reward-gradient pilot 的 128 题。
- 校准选择 HeadOnly alpha=1.0；所有 leave-one-question-out 子集仍选 1.0，但题目 bootstrap 中仅 62.04% 选择该剂量。
- 验证的四个版本各有 512 个唯一且一致的题目 ID。

| 版本 | 严格正确数 | 数值等价正确数 |
|---|---:|---:|
| 原 LoRA | 428 | 434 |
| HeadOnly alpha=1 | 440 | 456 |
| 逐模块 matched scalar | 437 | 450 |
| Full HNS | 441 | 455 |

HeadOnly-minus-scalar：严格指标 +3/512 = +0.586 pp，配对区间 [-1.37,+2.54] pp；数值等价指标 +6/512 = +1.172 pp，区间 [-0.59,+2.93] pp。两者均未解析出正负方向。

严格指标下，HeadOnly 修复/破坏 40/28 题，scalar 修复/破坏 35/26 题。共同修复 32 题、共同破坏 20 题。净收益 12 与 9 的比值是 75%，但干预对应的逐题变化并不完全相同，不能将净收益比值当作机制归因。

## 应收紧的表述

1. **“Scalar 解释了约 75% 的收益”只能是点估计的描述。** 严格指标中两个 vs-LoRA 区间都跨零，收益比值的分母并未稳定识别；此外该比例不是 mediation 或可加因果分解。推荐写成“matched scalar 获得与 HeadOnly 接近的收益点估计；谱形状的额外收益尚未确定”。

2. **Scalar 的范围必须写清楚。** 本轮每个模块有自己的 gamma_m(alpha)，逐模块匹配同剂量 HeadOnly 的 Frobenius norm，保留模块内部谱形状。它也改变了模块之间的相对幅度，不能简写为“一个全局 LoRA 系数就解释了结果”。

3. **不显著不等于等效。** 严格 HeadOnly-minus-scalar 区间仍容许约 +2.5 pp 的额外收益；本轮没有证明两者等价，也没有证明 shape 没有效果。数值等价作为次要终点支持收益方向，但不能替代预注册严格终点来宣布主要检验成功。

4. **校准流程完成，不等于校准策略已改善性能。** 选出的 alpha=1 与固定全剂量 HeadOnly 相同，且没有显示优于 Full HNS。可以报告这次剂量选择及其锁定结果，不能据此宣称获得了优于固定 heuristic 的自适应策略。

## 重建对照的意义

alpha=0 重建的最大相对模块误差为 8.32e-6，但校准集上有 6 道题的严格正确状态改变：1 修复、5 破坏，净 -1.56 pp。数值等价终点仍有 5 道状态改变，因此并非全部来自答案格式。

该结果说明存盘矩阵的近似等价不足以保证当前低精度、分解因子计算和自回归 greedy 路径的行为等价。这里的 8.32e-6 不是实际运行时每层浮点算子的误差界，不能简单解释成“模型对大小为 8.32e-6 的纯矩阵扰动必然敏感”。

HeadOnly 和 matched scalar 共用构造方式，减少了原始因子与重构因子的系统差别，但不能保证各剂量的舍入及生成分叉完全抵消。零重建只测了校准集，不能拿它的 -1.56 pp 去校正验证集效应；重建和谱编辑也未被证明可加。

## 与此前证据的关系

- 历史 Qwen MetaMath HeadOnly+FroRestore 的正结果仍应保留：它是一个保持 Frobenius norm 的谱编辑取得收益的观测。
- 该结果与本轮 matched scalar 比较不是同一检验。旧 scalar 匹配 Full HNS 的范数，本轮匹配相应剂量 HeadOnly 的范数；数据子集与数值对照也不同。
- 因此，新结果既不自动推翻历史 norm-preserving 正点估计，也不支持继续称其为已经排除实现因素的“纯 head-shape 因果证明”。
- PIQA 首 token margin predictor 未建立 accuracy 预测能力；其单方向结果仅由一两道题支撑。
- MetaMath reward-gradient pilot 在当前预算下不可辨识，且跨策略概率口径曾有混淆；这属于未取得可用信号，不是梯度理论或所有任务奖励敏感度的普遍反证。

## 收束决定

遵循停止条件，不追加 GPU，不做 localization。阶段产出定位为 HNS 的机制边界与因果控制研究：建立“使用强度”和“干预效用”必须分开处理的证据，保留谱形状、逐模块幅度和数值实现因素尚未分清的结论。当前没有证据支持对单个 dominant direction 给出可泛化的 excessive/essential 标签。

可用于摘要的表述：

> HNS 能在受控 hidden states 上抑制集中的 LoRA modifications，但这种抑制并不充分决定任务收益。MetaMath 的有限剂量干预呈现正向趋势，逐模块范数匹配的 scalar control 获得接近的收益点估计。当前尚未识别出谱形状的额外贡献，也未建立可靠的方向级效用预测器。微小的因子重构差异能够改变 greedy 结果，进一步限制了将下游差异专门归因于谱形状的解释。

依据：`hns_2x4_mechanism_final_20260910.md`、`hns_signed_gate_review_20260910.md`、`hns_metamath_reward_gradient_pilot_20260910.md`、`hns_metamath_direct_dose_result_20260910.md`、`hns_metamath_direct_dose_audit_20260910.json`。
