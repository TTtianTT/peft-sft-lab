# HNS 的谱几何、功能能量与任务效用：论文机制分析草稿

日期：2026-09-10。数学推导结合已有 2×4 缓存数据与作业 773 的结果；本轮仅运行 CPU 分析，没有新模型推理或下游评测。

## 论文主张

HNS 可解释为保留已学习输入/输出子空间的谱增益重分配。接近平坦的谱降低最坏方向的增益，并在保核范数条件下降低总参数能量。输入与原始 dominant directions 的对齐，使这种变化在固定 hidden states 上表现为更强的功能收缩。然而，输出变化的强度不能决定任务收益的符号；若原更新幅度超过任务需要，收缩可能有益，若承载必要修正则可能有害。

这一解释兼容“相对 LoRA 有效”和“尚未证明优于校准 scalar”两个结果。不应把它写成 HNS 已识别有害方向、所有 dominant modes 都过拟合，或谱形状具有普遍独立收益。

## 1. 实际算子与理想化端点

采用紧致 SVD，设秩为 r 且非零奇异值为正：

\[
D=\Delta W=U\operatorname{diag}(\sigma)V^\top,\quad
D_H=U\operatorname{diag}(t)V^\top.
\]

PEFT 的固定正 scaling 可吸收进 sigma 和 t。HNS 保持 U、V，因而保留线性输入与输出子空间；当所有 t_i>0 时也保持秩。这不等同于保留所有任务信息或所有下游行为。

核对当前 Qwen MetaMath 的实际 spectral_edit_meta.json，其设置为 4 次 fast + 1 次 stable，系数分别为 (3.4445,-4.775,2.0315) 与 (2,-1.5,0.5)，保核范数、rank=16、strength=1。不要将代码类当前默认的 8+2 误写为这张 checkpoint 的实验设置。

每次更新在标量谱上应用 f(x)=ax+bx^3+cx^5；开始用 Frobenius norm 归一化，最后恢复核范数。有限多项式可能非单调，不能无条件宣称每步都满足 majorization、每个样本的修改量都减小，或输出谱精确相等。

设 S=sum_i sigma_i，mu=S/r。理想平坦端点是

\[
D_{\rm flat}=\mu UV^\top.
\]

UV^T 是紧致 SVD 对应的 partial isometry：在 span(V) 内等比例映射到 span(U)，而不是整个高维空间的正交矩阵。它与极分解相关；经典背景可引用 [Higham, 1986](https://eprints.maths.manchester.ac.uk/694/)。下面的定标推导是直接的线性代数，不应包装为新的通用学习理论。

**最近的等增益算子。** 在固定 U,V 的 cUV^T 家族中，

\[
\arg\min_{c\ge0}\|D-cUV^\top\|_F^2
=\arg\min_c\sum_i(\sigma_i-c)^2=S/r.
\]

因此，平均奇异值的定标同时给出与原更新最接近的等增益端点和核范数保持。有限 HNS 可以从这一端点理解，但不等于精确求解上述问题。

**最坏方向增益。** 在 t_i>=0、sum_i t_i=S 条件下，

\[
\|D_t\|_2=\max_i t_i\ge S/r,\qquad
\|D_t\|_F^2=\sum_i t_i^2\ge S^2/r.
\]

两者在 t_i=S/r 时同时达到下界。它解释了谱平坦化的确定性效果：给定奇异值总和，降低最高方向增益和总参数能量。它不保证任务损失下降，也不保证真实输出协方差变白；非各向同性输入仍可导致高度各向异性的输出。

## 2. 谱形状与幅度收缩为何耦合

令 p_i=sigma_i/S，则

\[
\|D\|_F^2=S^2\sum_i p_i^2.
\]

因此保核范数时，降低 sum_i p_i^2 必然降低 Frobenius norm。定义基于奇异值概率 p 的二阶有效秩

\[
r_2=(\sum_i\sigma_i)^2/\sum_i\sigma_i^2=1/\sum_i p_i^2,
\]

则理想平坦端点的范数缩放为

\[
\gamma_F=\frac{\|D_{\rm flat}\|_F}{\|D\|_F}
=\sqrt{r_2/r}.
\]

该 r_2 不等于先前报告中基于 Shannon entropy 的 effective rank，也不等于 stable rank。不要混用。

这说明 HNS 在数学上具有谱集中度相关的收缩作用；谱集中时，平坦化导致更强的范数下降。它不证明收缩剂量最优，也不证明逐模块的此种分配优于 global scalar，后者已被实验直接检验而未得到支持。

## 3. 为什么真实输入会加强功能收缩

固定 hidden-state 分布，令 C=E[hh^T]（未中心化二阶矩），q_i=v_i^T C v_i。由于 U 的列正交，

\[
\mathcal E(D)=E\|Dh\|^2=\sum_i\sigma_i^2q_i.
\]

不需要假设 C 在 V 基底下对角化。只有在计算这个输出欧氏能量的 trace 时，输出方向正交性使交叉项消失。

令 beta_i=t_i/sigma_i，w_i=sigma_i^2 q_i / mathcal E(D)，则

\[
\boxed{\frac{\mathcal E(D_H)}{\mathcal E(D)}=\sum_iw_i\beta_i^2.}
\]

这是对固定 h 的精确恒等式。若原来的功能能量主要集中于 beta_i<1 的方向，少数方向的收缩就可能显著降低总体功能能量；tail 方向即使相对增长很大，也未必在此分布上占据同等权重。

同理，原始 top-1 功能占比高于其参数能量占比的条件是

\[
q_1>\frac{\sum_i\sigma_i^2q_i}{\sum_i\sigma_i^2}.
\]

因此 observed amplification 可以解释为输入二阶矩向原 dominant direction 对齐，但对齐本身没有有害/有益标签。

**参数范数匹配不是功能强度匹配。** 取 gamma_F^2=sum_i t_i^2/sum_i sigma_i^2，则

\[
\mathcal E(D_H)-\mathcal E(\gamma_FD)
=\sum_iq_i(t_i^2-\gamma_F^2\sigma_i^2),
\]

其中括号项的无权重总和为零，加权总和通常不为零。若输入对齐强的方向主要被压制，HNS 可以比 Frobenius-matched scalar 产生更低的功能能量。这也解释了保持 Frobenius norm 的 ShapeOnly 仍可降低 extreme modification 的可能性；它不是对 p99 单调下降的定理。

本项目原 2×4 的 h 来自固定 pretrained-base trajectory，不能将以上缓存分析写成完整 LoRA/HNS 自由生成轨迹的结论。

## 4. 不必假装 HNS 是 scalar：可以精确分解二者差异

在固定 h 上，求每个模块最接近 HNS 输出的 scalar：

\[
\gamma_C^*=\arg\min_\gamma E\|D_Hh-\gamma Dh\|^2
=\frac{\sum_i\sigma_i t_iq_i}{\sum_i\sigma_i^2q_i}
=\sum_iw_i\beta_i.
\]

定义 D_perp=D_H-gamma_C^*D，则 E[(Dh)^T D_perp h]=0，且

\[
E\|D_\perp h\|^2
=\mathcal E(D)\operatorname{Var}_w(\beta),
\]

\[
\boxed{\eta_{\rm shape}
=\frac{E\|D_\perp h\|^2}{E\|D_Hh\|^2}
=1-\frac{(\sum_iw_i\beta_i)^2}{\sum_iw_i\beta_i^2}.}
\]

这个量衡量 scalar 无法复现的输出几何成分，而不是谱形状解释了多少任务收益。gamma_C^* 也不是 accuracy-optimal gamma，不可当作已验证的选择器。

它还区分三个经常混淆的 scalar：

- gamma_F：参数 Frobenius norm 匹配。
- gamma_E=sqrt(sum_i w_i beta_i^2)：固定输入分布上的功能能量匹配。
- gamma_C^*=sum_i w_i beta_i：固定输入分布上的输出最小二乘拟合。

后二者满足 gamma_C^*<=gamma_E，但通常都不等于前者；任何一个都未必等于下游任务的最优剂量。

## 5. 本轮零 GPU 的补充结果

读取已有 module_spectra.csv 与 direction_response.csv，按原始奇异方向对齐，共 1,904 个模块、8 个 checkpoint。缓存能量与 E_i beta_i^2 的最大相对差异为 9.47e-8，支持字段对齐与代数一致性；由于缓存本来就按这些量构造，这不是独立的机制验证。

归一化后的 HNS 谱在 1,904/1,904 模块满足原始谱对它的 majorization，容差为 1e-6。核范数最大相对差异约 3.99e-6。这是样本内数值性质，不是有限 HNS 对任意矩阵的理论保证，模块数也不能当作独立实验样本数。

下表均为**每个 checkpoint 内的模块中位数**，而非 pooled/global 系数：

| Checkpoint | gamma_F | gamma_E | gamma_C^* | eta_shape |
|---|---:|---:|---:|---:|
| Qwen Code | 0.611 | 0.219 | 0.191 | 23.3% |
| Qwen Math | 0.673 | 0.357 | 0.277 | 35.1% |
| Qwen Tulu | 0.752 | 0.512 | 0.415 | 34.5% |
| Qwen Commonsense | 0.640 | 0.505 | 0.351 | 51.8% |
| Llama Code | 0.777 | 0.375 | 0.313 | 23.6% |
| Llama Math | 0.873 | 0.675 | 0.593 | 21.5% |
| Llama Tulu | 0.909 | 0.737 | 0.677 | 17.0% |
| Llama Commonsense | 0.752 | 0.544 | 0.434 | 38.0% |

Qwen MetaMath 中，HNS 谱相对等奇异值端点的距离 ||t-mean(t)||/||t|| 的模块中位数为 1.46%；原功能能量落在被收缩方向上的比例中位数为 98.0%。后者不是“移除了 98% 能量”。

最有用的解释是：该 checkpoint 中参数范数匹配 gamma_F 与功能强度匹配 gamma_E 明显不同，因此 norm-matched scalar 没有控制所有计算层面的强度差异。另一方面，eta_shape=35.1% 表明 HNS 输出也不能被每个模块的最优 scalar 完全复现。内部输出不同而 accuracy 点估计接近并不矛盾；这些输出差异对最终任务的相关性尚未测得。

不要将 gamma_E 的模块中位数 0.357 与在 GSM8K 校准得到的 global gamma=0.4 的接近，解释成预测最优剂量成功：输入轨迹、聚合方式与目标均不同，且这是事后观察。

## 6. 为什么收缩有时改善任务：可证伪的条件模型

假设某个方向的任务目标幅度为 a，当前幅度为 sigma，输入二阶矩为 q>0；定义一个解释性的局部平方误差模型

\[
\ell(\sigma)=\tfrac12q(\sigma-a)^2.
\]

压制 d>0，令 t=sigma-d，则

\[
\ell(t)-\ell(\sigma)
=q[-d(\sigma-a)+\tfrac12d^2].
\]

所以此次有限剂量压制有益当且仅当

\[
\boxed{a<\sigma-d/2=(\sigma+t)/2.}
\]

q 决定效应的权重，却不决定其正负。高 sigma、高 q 的方向既可能幅度过量，也可能恰好是任务所需。更重要的是，**task-essential 与 amplitude-excessive 并非互斥**：a>0 的必要方向，仍可能因为 sigma 过大而受益于适量收缩。

例如 sigma=10、t=6、q=1。在 a=7 时，该方向对任务有用，但误差从 4.5 降到 0.5；在 a=10 时，相同干预使误差从 0 升到 8。两者具有完全相同的原始谱与使用强度。这构成“仅凭使用强度不能识别压制符号”的明确反例。

这是条件模型，不是已测得的神经网络真实目标幅度，也不是 greedy accuracy 的精确损失形式。真实任务的 a 未知，现有实验未证明 HNS 能估计它。因此最合适的收益解释是“与缓解过强适配的假设一致”，而不是“已证明去除了过拟合噪声”。

## 7. 多模块与序列决策为何破坏简单预测

对一个可微任务目标 L，以及所有谱参数的有限扰动 delta，局部有

\[
\Delta L\approx g^T\delta+\tfrac12\delta^TH\delta.
\]

分模块后，二阶项包括 delta_m^T H_mn delta_n，故模块单独编辑的效应不能一般性相加。完整非线性网络中还有更高阶项。这给出 non-additivity 的可能结构解释，不说明已识别了具体交互矩阵。

第一阶项也需要任务相关的符号信息。单模块处

\[
\partial L/\partial\sigma_i
=E[(u_i^T\nabla_yL)(v_i^Th)],
\]

而功能能量是 sigma_i^2 E[(v_i^T h)^2]。前者包含下游任务的导数与正负号，后者是非负二阶量。Fisher/KL 类型的二阶敏感度能描述变化大小或局部风险，却同样不自动给出 reward 改善的符号。

普通 SFT 的 gradient 对应其自身目标，不是生成正确率；greedy accuracy 本身也不是可随意使用 Taylor 展开的光滑目标。上述式子只适用于明确的可微替代目标或满足可微条件的期望目标，不构成先前 margin/reward-gradient predictor 应当成功的保证。

## 8. 数值实现如何进入机制解释

在实际低精度计算中，fl(B fl(Ah)) 不仅由实数矩阵 BA 决定。即使两个因子表示对应相同或近似相同的更新，舍入路径也可能不同。在共享生成前缀上，若 logit 的最大扰动为 epsilon，原 top-1 margin 大于 2 epsilon 是保持 argmax token 的充分条件；小 margin 时该保证消失，一次 token 分叉又会改变后续 hidden states。

这解释了“同一表示可重复”和“不同等价表示改变生成”可以同时发生。作业 773 的富集 32 题诊断支持该现象存在，不能用于估计总体发生率。此前 mixed-factor 对照与 common-basis 对照的差异也不能按数值相减，直接分配成某项因素的因果贡献比例。

## 9. 与相关工作的关系

- [Higham, Computing the Polar Decomposition—with Applications, 1986](https://eprints.maths.manchester.ac.uk/694/)：为等增益 partial-isometry 解释提供经典极分解背景；不提供 HNS 下游收益的保证。
- [Wortsman et al., Robust Fine-tuning of Zero-shot Models](https://arxiv.org/abs/2109.01903)：视觉模型中的权重插值为“减弱适配、靠近原模型”的解释提供相关背景；不是本项目任务上的因果证据。
- [Shuttleworth et al., LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228)：研究 merged weights 的 intruder dimensions 与遗忘。该对象是 W0+Delta W 的谱，不能直接将其等同于这里 Delta W 的 dominant directions；本项目未证明 HNS 的收益来自移除 intruder dimensions。
- [Li and Tsuchiya, Muon with Finite Newton-Schulz, 2026](https://arxiv.org/abs/2608.26288)：强调有限 NS 与精确 polar map 的区别。其结论属于优化器训练理论，不能移植为对已训练 LoRA 做一次谱编辑的性能保证。

## 10. 可用于英文论文的机制段落

**Spectral balancing and functional attenuation.** HNS preserves the singular subspaces of the LoRA update while redistributing gains within them. A useful idealization is the scaled partial isometry (||Delta W||_*/r) UV^T. Among spectra with the same nuclear norm, this equal-gain endpoint minimizes both the operator norm and the Frobenius norm. Spectral balancing therefore couples reduced directional amplification with a reduction in parameter-space energy. The finite HNS map need not attain this endpoint exactly, although the stored spectra in our experiments are close to it.

**Input alignment separates parameter scale from functional scale.** For a fixed input second moment C, the functional energy is sum_i sigma_i^2 v_i^T C v_i. Consequently, attenuating directions with large input alignment can reduce functional energy substantially more than a Frobenius-matched scalar. Our cached frozen-base trajectories exhibit this distinction: for Qwen MetaMath, the module-median parameter-norm and functional-energy matching coefficients are 0.673 and 0.357, respectively. These quantities describe intervention geometry and do not estimate the accuracy-optimal scaling coefficient.

**Task-dependent utility.** Reduced functional modification is insufficient to guarantee improved task performance. In a conditional quadratic model, suppressing a direction is beneficial only when its original amplitude exceeds the task-required amplitude by enough to compensate for the finite intervention. A direction may therefore be task-essential and simultaneously over-amplified. Our controlled downstream recheck does not establish a shape-specific advantage over calibrated global scaling. The evidence supports spectral balancing and functional attenuation as mechanisms of the intervention, while leaving their task-specific contribution to performance unresolved.

## Reproduction and artifacts

CPU command:

```bash
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/analyze_hns_mechanism_geometry.py --source /dataset1/zailong/runs/peft-sft-lab/hns-2x4-allmodules-20260909 --output reports/hns_mechanism_geometry_20260910
```

- module_geometry.tsv：逐模块几何指标。
- checkpoint_geometry.tsv：模块中位数与四分位数；不是置信区间。
- geometry_audit.json：来源、数值一致性、统计口径。
- 可选 --plot 需要 matplotlib；当前环境未安装该依赖，本轮没有生成图像文件。可直接用上述表格绘制 parameter/functional scale 对照图，或在具备该依赖的环境中运行同一脚本。

所有新增指标都是事后描述性分析。本轮没有用它们选择方向、剂量或 checkpoint，也没有将它们解释为已验证的下游收益预测器。
