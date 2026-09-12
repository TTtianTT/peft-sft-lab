# HNS 在 Qwen3-8B Magicoder 上的机制分析

日期：2026-09-08

## 核心结论

当前证据支持下面这条较为克制的机制链：

> LoRA 更新的奇异谱被少数头部方向支配；HNS 在保留 LoRA 已学习奇异向量的同时重平衡奇异值，主要收益来自抑制头部方向，而不是统一减小 LoRA 强度或单独抬升尾部。该变化显著降低更新对真实隐藏状态的极端扰动，并主要通过中层、尤其注意力 Q/K/V 路径影响代码生成。它改善的是自由生成得到可执行正确程序的概率，而不等价于提高某个参考答案的 teacher-forced likelihood。

这一结论目前是单模型、单次确定性评测上的机制证据；仍需多 seed / 更多任务确认统计稳健性。

## 1. 参数层：HNS 改变谱值，而非重新学习方向

令 LoRA 更新为

\[
\Delta W=BA=U\operatorname{diag}(\sigma)V^\top.
\]

在 Qwen3-8B Magicoder 的 252 个 LoRA 投影上：

| 指标 | LoRA | HNS |
|---|---:|---:|
| top-1 energy share | 75.74% | 6.28% |
| top-4 energy share | 90.62% | 25.07% |
| effective rank（最大 16） | 9.66 | 16.00 |

HNS 相对原 LoRA 奇异基的平均非对角能量比例为 `5.39e-7`，投影残差为 `8.80e-5`。因此变化几乎完全可以解释为同一组已学习方向上的奇异值重分配，而不是方向旋转。

但“谱越平越好”不是充分条件：Llama-3.1 CommonSense 的 top-1 energy share 同样从 52.51% 降到 6.25%，macro accuracy 却下降 0.19 pp。原始 top-1 dominance 与收益的探索性 Spearman 相关仅为 0.37。因此更准确的命题是：头部支配提供了 HNS 可干预的对象，任务与层级兼容性决定干预是否有益。

## 2. 因果谱对照：主要有效成分是 head suppression

以下对照直接由观测到的 LoRA/HNS 谱构造：ScalarShrink 保留 LoRA 谱形状并匹配 HNS Frobenius norm；ShapeOnly 保留 HNS 相对谱形状但恢复 LoRA Frobenius norm；HeadOnly 只接受 HNS 对奇异值的降低；TailOnly 只接受 HNS 对奇异值的升高。

| 变体 | HumanEval | MBPP | 平均 | 相对 LoRA |
|---|---:|---:|---:|---:|
| LoRA | 77.44% | 71.98% | 74.71% | +0.00 pp |
| ScalarShrink | 78.66% | 71.60% | 75.13% | +0.42 pp |
| ShapeOnly | 82.32% | 73.54% | 77.93% | +3.22 pp |
| HeadOnly | 82.93% | 72.76% | 77.84% | +3.13 pp |
| TailOnly | 76.83% | 72.37% | 74.60% | -0.11 pp |
| HNS | 82.93% | 73.93% | 78.43% | +3.72 pp |

这排除了两个简单解释：

1. 不是纯粹的“LoRA 太强，所以整体缩小就行”：ScalarShrink 只恢复 0.42 pp，而 ShapeOnly 在不降低 LoRA Frobenius norm 时仍恢复 3.22 pp。
2. 不是“放大尾部方向”单独起作用：TailOnly 基本无收益；HeadOnly 已恢复完整 HNS 增益的主要部分。尾部抬升可能只提供协同，尤其 HNS 相比 HeadOnly 在 MBPP 多通过 3 题。

成对样本上，HNS 相比 LoRA 的 HumanEval gain/loss 为 16/7，MBPP 为 16/11；精确双侧 McNemar/binomial 检验分别为 `p=0.0931` 和 `p=0.4421`。方向一致但单个测试集功效不足，不能宣称单基准 0.05 显著。

## 3. 模块和层级定位

| 干预范围 | HumanEval | MBPP | 主要解读 |
|---|---:|---:|---|
| LoRA | 127/164 | 185/257 | 基线 |
| 全模块 HNS | 136/164 | 190/257 | +9 / +5 |
| Attention only | 131/164 | 189/257 | 两任务都稳定获益 |
| MLP only | 131/164 | 183/257 | HumanEval 获益、MBPP 退化 |
| QKV only | 130/164 | 190/257 | MBPP 完全复现全 HNS |
| O-proj only | 124/164 | 184/257 | 单独应用有害 |
| Gate+Up only | 127/164 | 184/257 | 基本无效 |
| Down only | 128/164 | 185/257 | 基本无效 |
| layers 0--11 | 129/164 | 188/257 | 更偏 MBPP |
| layers 12--23 | 132/164 | 186/257 | 更偏 HumanEval |
| layers 24--35 | 126/164 | 184/257 | 局部应用有害 |

因此 HNS 不是模块可交换的通用平滑。最稳定的单一定位结论是 Attention/QKV；MLP 需要跨 Gate/Up/Down 协同，O-proj 和 late-only 是明确的负对照。全模块超过任一单组，也说明存在跨模块组合效应。

## 4. 任务梯度：可用于定位，但不能单独预测最终收益

对每个 LoRA 奇异方向计算精确谱梯度：

\[
\frac{\partial L}{\partial \sigma_i}
=u_i^\top\frac{\partial L}{\partial \Delta W}v_i,
\qquad
\Delta L_{\mathrm{1st}}
\approx\sum_i\frac{\partial L}{\partial\sigma_i}
(\sigma_i^{\mathrm{HNS}}-\sigma_i).
\]

定义 compatibility 为上述一阶损失变化的负值。64 个 Magicoder 样本、22,551 个监督 token 的结果为：

- 只有 layer 16、17、18 的层平均 compatibility 为正，分别为 `2.27e-5`、`1.97e-5`、`5.64e-6`。
- 14 个被选模块中 12 个位于 layers 12--23；layer 17 一层占 4 个。
- 梯度选择的 14 模块 HNS 得到 HumanEval 129、MBPP 185，只比 LoRA 多 2/0，明显低于全 HNS 的 136/190。
- 每一种模块类型的全层平均 compatibility 都为负；这和 HNS 提高参考答案 NLL 的观察一致，却与 pass@1 改善并存。

因此梯度更适合作为“哪里对任务敏感/哪里短期风险低”的定位器，而不是 HNS 收益的完整解释。代码执行正确率涉及自回归轨迹和离散测试结果，一阶 teacher-forced CE 无法覆盖跨模块、高阶和解码路径效应。

## 5. 激活响应：HNS 降低极端扰动，但尺度降低并非充分解释

在 256 个固定 Magicoder 样本上，以同一 frozen-base hidden trajectory 测量

\[
\|\Delta W h\|_F/\|W h\|_F.
\]

HNS 在 36/36 层都降低了 p99 响应；各层 `HNS p99 / LoRA p99` 的平均值为 0.175。LoRA 最大 p99 在 layer 22（0.0366），HNS 最大值在 layer 25（0.0047）。七类投影的 paired sample 中，HNS 响应均为 100% 更低。

这说明 HNS 强烈抑制了由谱头部造成的极端更新响应。不过 ShapeOnly 在恢复 LoRA Frobenius norm 后仍有 +3.22 pp，因此“响应更小”只是伴随机制，不是充分因果解释；关键仍是各奇异方向之间的相对平衡。

## 6. 注意力与生成轨迹

在 LoRA→HNS 的 gain、loss、stable-pass、stable-fail 样本上，对 canonical solution 做 teacher forcing：

- 全 HNS 在中层普遍提高归一化注意力熵并降低 top-1 attention mass，表现为注意力更分散，而不是更尖锐。
- 该全局效应不能稳定区分 gain 与 loss：HumanEval 中 gain 的熵增加略小于 loss，MBPP 中方向相反。
- QKV-only 在中层轻微增加 target-token 对 prompt 的 attention mass（HumanEval gain `+0.0030`，MBPP gain `+0.0021`），与 QKV-only 的行为收益方向一致，但幅度小且 loss 样本也存在，暂时只能作为候选机制。
- 在 layers 16--18 的 32 个 heads 中，layer 16 head 25 在两个基准上都呈现较一致的 gain-vs-loss 模式：prompt mass 增加、熵降低、top-1 mass 增加；layer 18 head 8 次之。样本量每类最多 8 个，属于待因果验证的 head 候选，而非已证实解释。
- 完整 HNS 对参考答案 NLL 的影响在所有类别都为正（更差）：HumanEval gain/loss 分别 `+0.565/+0.666`，MBPP gain/loss 分别 `+1.053/+1.175`。这否定了“收益来自更好拟合唯一参考程序”的解释。一个题目存在许多等价正确程序，pass@1 与 canonical-solution NLL 本就不是同一目标。

## 7. 论文中建议使用的主张

建议主张：

> HNS is a direction-preserving spectral rebalancing operator. On code adaptation, its gain is causally concentrated in suppressing dominant LoRA singular modes, rather than globally shrinking the adapter or independently amplifying its tail. This rebalancing reduces extreme feature perturbations and acts primarily through middle-layer and QKV pathways, while improving execution-level generation without improving the likelihood of a single reference solution.

不建议主张：

- “谱越平，性能一定越好”：已有 CommonSense 反例。
- “任务梯度可以直接挑出最优 HNS 模块”：14-module 版本只得到较小收益。
- “HNS 让注意力更集中”：全局统计显示相反。
- “当前 HumanEval/MBPP 已统计显著”：单次评测功效不足。

## 8. 优先级最高的后续实验

1. **多 seed / 多 checkpoint 确认 HeadOnly 与 ShapeOnly。** 对至少 3 个训练 seed 重复 LoRA、HeadOnly、ShapeOnly、HNS；主比较使用 paired bootstrap/McNemar，并把 seed 作为独立单位。这是当前最重要的论文补强。
2. **HNS 强度 × 模块范围二维图。** 对全模块、QKV、中层 QKV 扫描 5--7 个强度点，验证是否存在更稳健的 HNS 变体，而不是只比较一个固定强度。
3. **针对候选 head 的因果干预。** 对 layer 16 head 25、layer 18 head 8 做 attention-output patching 或 head ablation，并配相同层随机 head 对照；分别在 gain/loss 样本上测 pass flip。
4. **自由生成轨迹而非 canonical teacher forcing。** 对 LoRA/HNS 使用相同题目生成各自程序，比较首个 divergence token 前后的 logit margin、entropy、prompt attention 和 residual-stream patching，直接连接注意力变化与最终 pass/fail。
5. **验证失败边界。** 在 Llama-3.1 CommonSense 失败 checkpoint 上重复模块定位和任务梯度；检验其是否缺少 16--18 类正 compatibility 区域，或 HeadOnly 本身也失败。该实验最能把“何时 HNS work”写成可证伪条件。

## 产物位置

- 完整谱/因果控制/真实激活：`/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-mechanism-20260908`
- 模块定位与任务梯度：`/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/magicoder-hns-localization-20260908`
- 注意力与 hidden-state 对比：`/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/magicoder-hns-interpretability-20260908`
