# HNS 与对比方法：论文 Method 部分写作报告

日期：2026-09-13。依据当前仓库源码和 2026-09-12 HNS release 撰写。

本文提供算法定义、实现细节、对照设计及英文写作草稿。它不是新的实验结果报告，也不将已有趋势写成普遍理论保证。新增 seeds 的配置审计见 [三运行结果与配置报告](hns_three_seed_diagonal_20260913.md)。

## 1. 方法定位与论文组织

建议方法名使用 **Post-hoc Hybrid Newton–Schulz Spectral Editing (HNS)**；Spectral Surgery 可作为项目或框架名称。Hybrid 指先后使用两组多项式系数，不是把两种训练算法混合。

一句话定义：HNS 是一种无需额外训练数据、梯度或再训练的 LoRA 后处理操作，在每个已训练适配器的奇异子空间内重新分配谱增益，并恢复原始核范数。

这里的 data-free 指固定超参数后生成编辑适配器的操作。若根据验证集选择迭代数，该选择过程仍使用数据；若在测试集上挑最高分设置，则只能报告描述性最优，不能称为无数据选择或独立验证的最优配置。

建议 Method 主文按照以下顺序组织：LoRA 参数化 → 紧致 SVD → 两阶段谱迭代 → 核范数恢复与因子重构 → 主要尺度/形状对照。机制定位、signed standardization、剂量及额外 PEFT 实现放 appendix；训练 recipes、数据、推理和统计放 Experimental Setup。

| 类型 | 本项目方法 | 论文中的角色 |
|---|---|---|
| 来源与参考 | Base、原始 LoRA | 不编辑参考与任务适配基线 |
| 主方法 | HNS | 训练后的谱编辑 |
| 数值参考 | SVD 0+0 / Zero rebuild | 检查因子重构与表示敏感性 |
| 尺度对照 | Per-module norm-matched scalar、Global norm-matched scalar、Calibrated scalar | 区分谱形状与整体更新强度 |
| 机制拆分 | ShapeOnly、HeadOnly、TailOnly、HeadOnly+FroRestore | 检验增益重分配的组成部分 |
| 替代谱变换 | ExactFlat、TopShrink、TailLift、Temperature、signed standardization | 检验迭代与简单谱变换的差异 |
| 因果定位诊断 | Functional/Raw/random Top-K、F×C、module utility | 机制分析，不是全模块主方法 |
| 训练期 PEFT | LoRA+、PiSSA、AdaLoRA | 仓库支持的训练方法；不能仅据代码支持声称完成当前 HNS 全矩阵比较 |

## 2. 统一符号与 LoRA 来源

对模块 m，冻结的预训练权重为 W_m^0 ∈ R^{d_out×d_in}，训练后的 LoRA 因子为 B_m ∈ R^{d_out×r}、A_m ∈ R^{r×d_in}。

\[
D_m=B_mA_m,\qquad
W_m=W_m^0+s_mD_m,\qquad s_m=\alpha_m/r_m.
\]

本报告把未乘 PEFT scaling 的 BA 记为 D，把实际生效更新记为 ΔW=sD。当前标准 LoRA 的 r16、alpha32 对应 s=2；不应在写公式时漏掉 scaling，也不能重复乘两次。普通 LoRA 冻结原模型并训练低秩增量，见 [LoRA 原论文](https://arxiv.org/abs/2106.09685)。

HNS 的实际分解对象是 **训练后的 BA**，不是 W^0、A 单独、B 单独、训练梯度或优化器 momentum。对固定正 s、固定输出秩，保持 BA 核范数也保持实际更新 ΔW 的核范数。编辑不会改变冻结的 W^0。

设紧致 SVD 为

\[
D_m=U_m\operatorname{diag}(\sigma_m)V_m^\top,
\qquad \sigma_{mi}\ge0.
\]

U,V 在编辑中保持原有列/方向配对。有限多项式未必保持谱排序，因此编辑后的系数应理解为**原始奇异基底中对齐的非负增益**；其重新排序的多重集才是通常按降序表示的奇异值。不能把编辑后的第 i 个最大奇异值与原始第 i 个方向任意重新配对。

当前主网格不改变存储 rank16。若输出增益均严格为正，非零秩与输入/输出子空间保持；零方向、数值截断与 dtype 舍入需另行检查。因此更稳妥的描述是 preserves the learned singular subspaces / retains the adapter rank budget，而不是无条件 preserves every task-relevant direction。

## 3. 无需构造稠密更新的紧致 SVD

源码：`src/finetune/spectral_edit/svd.py::lowrank_svd_from_ba`。

先对两个细长矩阵作 reduced QR：

\[
B=Q_BR_B,\qquad A^\top=Q_AR_A.
\]

对 r×r 核矩阵作 SVD：

\[
M=R_BR_A^\top=\widetilde U\operatorname{diag}(\sigma)\widetilde V^\top,
\quad U=Q_B\widetilde U,\quad V=Q_A\widetilde V.
\]

因此只需处理 LoRA 因子和小核矩阵，不物化 d_out×d_in 的 BA。当前 QR/SVD 在 float32 中执行，谱迭代内部转换为 float64；最终因子转换回来源适配器的存储 dtype。

每模块分解与重构的算术量级为 O((d_out+d_in)r²+r³)，谱迭代为 O((K_f+K_s)r)，工作存储量级 O((d_out+d_in)r+r²)。这是算法量级，不是实测 wall-clock；不能把整个方法写成 O(Kr)，因为 QR/SVD 的成本仍然存在。多种迭代设置可复用一次来源分解，当前 grid builder 即如此实现。

## 4. HNS 谱变换

源码：`src/finetune/spectral_edit/posthoc_hns.py`。

### 4.1 Frobenius 归一化

\[
x^{(0)}=\frac{\sigma}{\max(\|\sigma\|_2,\epsilon)},\qquad \epsilon=10^{-7}.
\]

这里向量的 ℓ2 norm 对应 D 的 Frobenius norm，不是 D 的 operator norm。不能写成用最大奇异值归一化。

### 4.2 两阶段五次多项式

定义逐方向变换

\[
f_{a,b,c}(x)=ax+bx^3+cx^5=x(a+bx^2+cx^4).
\]

先执行 K_f 次 fast 阶段，再执行 K_s 次 stable 阶段：

\[
x\leftarrow f_f(x),\quad
(a_f,b_f,c_f)=(3.4445,-4.7750,2.0315),
\]

\[
x\leftarrow f_s(x),\quad
(a_s,b_s,c_s)=(2,-1.5,0.5).
\]

fast 系数与 [Muon 作者实现](https://github.com/KellerJordan/Muon) 中的 quintic Newton–Schulz 变换相关。**HNS 不是 Muon 优化器**：Muon 在训练时处理更新方向；这里把相关多项式用于完成训练的适配器谱，并随后恢复核范数。两阶段组合、stable 系数和后处理设置以本项目源码为准；不能只凭 Muon 引用认定全部组合都是其原始算法，也不宜在没有进一步原始文献核对时给出精确首创归属。

为何需要区分两个阶段：fast 多项式在零附近斜率为3.4445，对小正增益具有较强放大作用，但 f_f(1)=0.701，并不把1保持为固定点。stable 多项式满足 f_s(1)=1、f_s'(1)=0，可用于在单位增益附近精化。这些是多项式的局部性质，不是对任意输入、任意次数的全局收敛证明。

矩阵等价式为

\[
X\leftarrow aX+b(XX^\top)X+c(XX^\top)^2X.
\]

若 X=U diag(x)V^T、U/V 列正交，则上述操作恰等价于 x_i←f(x_i)。实现直接迭代 r 个谱值，避免稠密矩阵迭代。

### 4.3 非负处理与核范数恢复

两阶段结束后，记 q=max(x,0)，恢复来源谱总和 S=Σ_iσ_i：

\[
\widehat\sigma_i=q_i\frac{S}{\sum_jq_j}.
\]

源码在 Σq≤ε 时使用 S/r 的常数填充作为退化保护；全零来源因此仍得到全零目标。正常分支采用上式。这里保护的是核范数

\[
\|D_H\|_*=\sum_i\widehat\sigma_i=\sum_i\sigma_i=\|D\|_*,
\]

不是 Frobenius norm，也不是 operator norm。

### 4.4 可选强度与输出秩

源码提供 strength λ∈[0,1]：

\[
\sigma_i^{(\lambda)}=(1-\lambda)\sigma_i+\lambda\widehat\sigma_i.
\]

当两个端点非负且核范数相同时，插值也保持核范数。λ=0 为来源谱、λ=1 为完整编辑；当前统一 grid 的 λ=1，不应把强度插值实验写成所有主结果的默认流程。

库还支持 output_rank 截断：先用完整来源谱归一化和迭代，再保留原始排序中的前 r_out 个方向，恢复**保留部分来源谱**的核范数。此时不能声称保持截断前完整更新的核范数。当前主网格没有使用该功能。

### 4.5 重构为可直接加载的 LoRA

统一 step-grid 和核心 causal controls 使用 balanced 因子：

\[
B_H=U\operatorname{diag}(\sqrt{\sigma^{(\lambda)}}),\qquad
A_H=\operatorname{diag}(\sqrt{\sigma^{(\lambda)}})V^\top.
\]

保持原 adapter config 与 PEFT scaling，保存新的 A/B，随后可用原 LoRA 推理路径加载。没有新增 trainable parameters、推理分支或在线路由；也可将更新合并进权重，合并操作本身不是本报告新增实验。

**实现分支不能混写：** common-basis scalar/遗忘控制和 signed standardization 使用 A'=V^T、B'=U diag(c) 的单边表示。它与 balanced 表示在精确算术下给出相同 BA，但有限精度下不保证逐 token 等价。

### 4.6 伪代码

```text
Input: trained LoRA factors {(B_m, A_m)}, fixed (K_f, K_s), λ=1
For each adapted module m:
    Obtain (U, σ, Vᵀ) by QR of B and Aᵀ and SVD of the r×r core
    x ← σ / max(||σ||₂, ε)
    Repeat K_f times: x ← x * (3.4445 - 4.7750*x² + 2.0315*x⁴)
    Repeat K_s times: x ← x * (2 - 1.5*x² + 0.5*x⁴)
    q ← max(x, 0)
    If sum(q) > ε: τ ← q * sum(σ)/sum(q)
    Else: τ_i ← sum(σ)/r
    t ← (1-λ)*σ + λ*τ
    B'_m ← U diag(sqrt(t)); A'_m ← diag(sqrt(t)) Vᵀ
    Cast factors back to source storage dtype
Return adapter with edited factors and unchanged scaling/config
```

## 5. 能证明什么，不能证明什么

### 5.1 理想平坦端点

设 S=Σσ_i，μ=S/r。理想等增益更新为

\[
D_{flat}=\mu UV^\top.
\]

UV^T 是在已学习子空间之间作用的 partial isometry，不是整个高维空间的方形正交矩阵。它在 span(V) 内等比例映射到 span(U)。

对任意非负目标 t、Σt_i=S，由最大值界和 Cauchy–Schwarz：

\[
\max_it_i\ge S/r,\qquad \sum_it_i^2\ge S^2/r.
\]

常数谱同时最小化该约束下的 operator norm 与 Frobenius norm。固定 U,V 的 cUV^T 家族中，最小化与来源 D 的平方 Frobenius 距离也给出 c=S/r。

这说明平衡谱会把谱形状改变和更新能量收缩耦合起来。但是有限次 HNS **不是精确 ExactFlat**；没有证明每一次多项式迭代都降低 operator/Frobenius norm，也没有证明每一步 majorization 或每个样本输出变化单调下降。

### 5.2 形状与尺度的耦合

记 p_i=σ_i/S，则 ||D||_F²=S²Σp_i²。保核范数时，降低二阶谱集中度才意味着 Frobenius norm 降低。定义二阶有效秩

\[
r_2=\frac{(\sum_i\sigma_i)^2}{\sum_i\sigma_i^2},\qquad
\frac{\|D_{flat}\|_F}{\|D\|_F}=\sqrt{r_2/r}.
\]

代码中报告的 effective rank 则是 Shannon entropy 版本：exp(-Σp_i log p_i)。它不等于 r2，也不等于 stable rank=||D||_F²/||D||_op²。论文应给出定义，不混用名称。

### 5.3 功能能量与任务表现

固定 hidden-state 分布，C=E[hh^T] 为未中心化二阶矩，q_i=v_i^T C v_i，则

\[
\mathcal E(D)=E\|Dh\|_2^2=\sum_i\sigma_i^2q_i.
\]

实际更新能量还要乘 s²。等式不要求 C 在 V 基底中对角化；它依赖 U 列正交。输入对齐使参数谱强度和功能输出强度不同，因此 Frobenius-matched scalar 未必匹配功能能量。

但这里的 C 是固定分布/固定轨迹。编辑全模型后 hidden states 会变化，不能将固定输入恒等式当作端到端网络能量定律。功能修改减少也不保证任务损失下降：方向可能承载任务必要修正。当前证据不支持“HNS 自动识别有害方向”“dominant directions 都是过拟合”或“谱平坦化普遍提高泛化”。

## 6. 主尺度与形状对照：精确定义

以下每模块 σ 为来源 LoRA 谱，τ 为 HNS 在**同一来源基底中对齐**的目标增益。为简洁省略 m。所有方法保持方向，只改增益；分母使用数值保护。

| 方法 | 目标增益 t | 受控量 / 要回答的问题 |
|---|---|---|
| Original LoRA | σ | 原始学习结果，不重构 |
| SVD 0+0 | σ，仅重新分解重构 | 因子表示与数值变化能否解释部分收益 |
| Per-module scalar / ScalarShrink | σ·||τ||₂/||σ||₂ | 每模块匹配 HNS Frobenius norm，保持来源谱形状 |
| ShapeOnly | τ·||σ||₂/||τ||₂ | 保持 HNS 形状，恢复来源 Frobenius norm |
| HeadOnly | min(σ_i,τ_i) | 仅接受 HNS 对来源方向的抑制 |
| TailOnly | max(σ_i,τ_i) | 仅接受 HNS 对来源方向的放大 |
| HeadOnly+FroRestore | q·||σ||₂/||q||₂，q=min(σ,τ) | 抑制方向的相对形状改变是否超越能量收缩 |

HeadOnly/TailOnly 不是固定 top-k 切片，也不是按均值 μ 做阈值。二者按 HNS 是否压制该方向定义，满足逐方向 min(σ,τ)+max(σ,τ)-σ=τ。然而下游分数不是线性的，不能把两个 accuracy 增益相加当作 HNS 增益。

ScalarShrink 这个名字在代码中有两种目标，务必区分：`mechanism.py` 匹配**observed HNS** Frobenius norm；`ablations.py` 匹配**理想 ExactFlat** Frobenius norm S/√r。论文建议分别命名 HNS-Fro-matched scalar 和 Flat-Fro-matched scalar。

### 6.1 Global norm-matched scalar

所有模块共用 γ：

\[
\gamma_F=\sqrt{\frac{\sum_m\|\tau_m\|_2^2}{\sum_m\|\sigma_m\|_2^2}},\qquad t_m=\gamma_F\sigma_m.
\]

这匹配未缩放 BA 拼接后的参数能量。若 s_m 不同、想匹配实际 ΔW 总能量，公式必须在分子分母中加入 s_m²。当前 r/alpha 相同的比较无需改变数值，但应说明匹配对象。

### 6.2 Shuffled module scaling

在同一 module type 内随机置换 per-module γ_m，再全局修正以匹配总 Frobenius norm。它检验收益是否依赖 γ 与真实模块的对应关系。置换 seeds 是**控制随机性重复**，不是新增 training seeds。具体分层和修正规则应引用对应 pilot manifest，不外推为当前全部 checkpoint 协议。

### 6.3 Calibrated global scalar

令 t_m=γσ_m。需区分两个已经做过的选择协议：

- Qwen MetaMath common-basis 定向实验：γ∈{1,.85,.70,.60,.50,.40}，校准集选 target strict accuracy 最优，然后在 locked validation 上报告。
- 原始 2×4 遗忘控制：γ∈{.25,.40,.55,.70,.85,1}，要求 target score 不比 HNS 低超过1 pp，然后在可行集合内最大化 off-task retention。

第二种是在相同报告数据上选择和评测，是探索性上界，不是独立验证的 scalar 方法。两种协议不能混为同一 baseline，也不能把 oracle retention 选择描述成 data-free。当前新增 seeds 的遗忘作业尚未包含 scalar bank，不能声称已有新三运行 scalar 对照。

### 6.4 对齐 observed HNS 的原因

HNS 接近平坦时，对 HNS 再做独立 SVD 可能在近简并子空间内旋转方向。控制构造使用

\[
C_{align}=U^\top B_HA_HV,
\]

读取其对角线作为 τ，检查 off-diagonal energy 与投影残差。当前 causal builder 默认最大基底误差5e-3；具体实验以命令及 manifest 为准。微小负对角值做非负裁剪。不能用两个独立 SVD 的排序结果直接构造 min/max 而不核对方向。

## 7. 简单替代谱变换与剂量消融

源码：`src/finetune/spectral_edit/ablations.py`。设 μ=S/r。

| 方法 | 定义 | 与 HNS 的差异 |
|---|---|---|
| ExactFlatNuclear | t_i=μ | 直接到达保核范数的等增益端点，无有限步多项式残差 |
| Flat-Fro-matched scalar | t=σ·(S/√r)/||σ||₂ | 匹配 ExactFlat 能量但不改变来源形状 |
| TopShrink | t_i=min(σ_i,μ) | 仅将高于均值者压到均值；不是 observed-HNS HeadOnly |
| TailLift | t_i=max(σ_i,μ) | 仅将低于均值者抬到均值；不是 observed-HNS TailOnly |
| Temperature | t_i=S·σ_i^θ/Σ_jσ_j^θ | θ=1 为来源；θ=0 为常数谱；0<θ<1 压缩谱动态范围 |

θ>0 时源码让原零系数保持零；θ=0 直接填均值，因此零方向行为不同。这些实现不代表已完成所有 checkpoint 的全量比较，已有 pilot/full-set 的范围必须从结果 manifest 核对。

Head dose 使用 t(η)=σ+η[min(σ,τ)-σ]，η∈{0,.25,.5,1}。对应 dose-matched scalar 是 t_scalar(η)=σ·||t(η)||₂/||σ||₂。它提供形状干预的剂量—反应对照；η 与 HNS 全谱 strength λ 不同，不混用符号。

## 8. Centered signed standardization

源码：`scripts/build_2x4_spectral_standardization.py` 与 `scripts/build_2x4_restored_standardization.py`。

用 population variance，即 correction=0：

\[
\mu=\frac1r\sum_i\sigma_i,\quad
v=\frac1r\sum_i(\sigma_i-\mu)^2,\quad
z_i=\frac{\sigma_i-\mu}{\sqrt v}.
\]

Direct 分别使用 c=(σ-μ)/v 或 c=z，不裁剪、不取绝对值、不恢复 norm。居中后出现负系数，因此它们不是通常意义的非负奇异值，而是来源 U,V 基底上的 **signed gains**：

\[
D_{signed}=U\operatorname{diag}(c)V^\top.
\]

用 B'=U diag(c)、A'=V^T 重构，不能对负系数使用非负 sqrt 重构。负增益使相应方向的作用符号反转，因此该方法不仅改变谱集中度，也改变原始方向的作用方向。

Norm-restored 分支保留 z 的符号：

\[
c_F=z\frac{T_F}{\|z\|_2},\qquad
c_*=z\frac{T_*}{\sum_i|z_i|}.
\]

目标 T 为来源 LoRA 或 HNS 的 Frobenius/nuclear norm，得到四种组合。核范数必须用绝对值和，不能使用居中后的 Σz=0。若正分母差异是唯一变化，恢复同一 norm 后 `(σ-μ)/v` 与 `(σ-μ)/sqrt(v)` 数学等价，代码显式检查该等价误差。

当 v≤ε，builder 拒绝退化谱，不应把此分支写成使用任意 epsilon z-score。它们是负结果替代方法，适合 appendix；不能称为对模型 activations 做 whitening，也不能据其失败证明一切谱标准化都失败。

## 9. 模块定位与 utility：不属于主算法的步骤

FunctionalTop-K、RawTop-K、uniform random、matched layer/type random：只将选中25%/50%模块替换为 observed all-module HNS，其他模块维持原 LoRA。它们检验干预位置，而不是给 HNS 加一个在线选择器。

Functional concentration 来自固定输入下 σ_i²q_i 的能量占比；Raw concentration 仅使用 σ_i²。F×C 干预按 module type×layer quartile 分层，对功能集中度 F 和梯度 compatibility 的层内 rank corners 作同 quota 对照。High-C 是层内相对高，不等于绝对 compatibility>0。

Module utility 则比较单模块编辑前后 held-out teacher-forced SFT NLL；正 utility 表示该 surrogate loss 改善。它不是生成 benchmark accuracy，也不保证多个单模块 utility 可加。缓存激活、梯度和 held-out 样本是机制诊断的数据需求，**不是固定 HNS 操作的数据需求**。

精确取样、F/C 评分和配额应放入对应机制实验 protocol；不应凭本报告的概括宣称这些诊断已形成可靠选模块算法。现有结果未建立 FunctionalTop-K 普遍优于 matched random。

## 10. 仓库支持的其他训练期 PEFT 方法

以下来自 `src/finetune/peft_builders.py` 和训练入口。HNS release 的 all_methods_main_table 主要列的是后处理对照，而不是这些训练方法的完整2×4基准。

### 10.1 LoRA+

LoRA+ 对 A/B 采用不同学习率，见 [LoRA+ 原论文](https://arxiv.org/abs/2402.12354)。仓库使用普通 LoraConfig，优化器分组令 lr_A=lr、lr_B=ρlr；CLI 默认ρ=20，embedding groups 可使用单独基准 LR。它改变训练优化，不是对完成训练的谱作 scalar edit。默认值不等于所有已运行实验的值，写论文前需查具体 run_args。

### 10.2 PiSSA

PiSSA 用预训练 W 的 principal components 初始化低秩可训练部分，将剩余部分作为冻结 residual，见 [PiSSA 原论文](https://arxiv.org/abs/2404.02948)。仓库通过 PEFT 的 init_lora_weights=pissa 或 pissa_niter_16 调用实现。PiSSA 分解的是预训练权重用于初始化，HNS 分解的是训练后的增量用于后处理；时间点和对象都不同。公平推理还需确认 residual/base 与 adapter 的组合或转换，不能把 PiSSA 因子直接加到未经处理的原 base 后宣称等价。

### 10.3 AdaLoRA

AdaLoRA 在训练中自适应分配低秩预算，见 [AdaLoRA 原论文](https://arxiv.org/abs/2303.10512)。仓库调用 AdaLoraConfig，CLI 默认 init_r32、target_r16；builder 默认 tinit≈0.10T、tfinal≈0.50T、deltaT≈0.01T，并作合法区间保护。配置中的 beta1=beta2=.85 是 AdaLoRA 重要性估计参数，不能误写为 Adam β。

### 10.4 状态与公平性边界

这三种方法在当前仓库有实现，但本报告未核实它们已完成与当前 HNS 相同任务、相同训练预算、相同评测协议的全部矩阵。DoRA 未见当前训练入口支持；本报告不将其列为已做基线。

若未来比较训练期 PEFT，应控制 base、任务数据与子集、监督 token 格式、GBS、epoch/token预算、序列长、有效 rank 与学习率调优预算，并核实训练配置。HNS 的主要配对比较应从同一保存 LoRA 来源出发，这比拿不同训练来源的适配器互相比更可解释。

## 11. 当前实验设置在 Method 中如何表述

- 原始主网格：Qwen3-8B、Llama-3.1-8B-Instruct × Magicoder、MetaMath、Tulu、Commonsense。每个 base/task 一个保存来源 LoRA。
- HNS 主网格：K_f∈{2,4,8}、K_s∈{0,1,2}；all modules、λ=1、保核范数、来源 rank16；额外0+0控制。
- 库默认8+2、原始部分保存 HNS4+1/8+2、统一重建 grid 是不同概念，须对每个 panel 明示实际来源。不能把8+2写成所有论文结果的唯一设置。
- HumanEval chat pass@1、GSM8K strict accuracy、IFEval prompt strict、Commonsense-8 equal-subtask macro；token limits分别512、512、2048、8。来源协议与 prompt/解析器版本要归档。
- 新增43/44仅覆盖三个非 Commonsense 训练任务。原始42与新增配置存在执行/来源核实差异，不写成所有 checkpoint 都有严格同配置三个 seeds。
- 不同机制、标准化及重评 panel 使用了不同评测批次；报告分数时不能把不同批次逐 cell 相减。原始统一 release 和模型卡旧分数也不能择优混用。
- 20,000 paired item bootstrap 和 Holm 等属于具体原始对照分析的统计协议；本报告没有新算 CI。item CI 不是 training-seed CI，SD 也不是 CI。

原始遗忘指标：target 是训练任务的基准得分；off-task 为其他三类基准等权平均。每一项 clipped forgetting 为 max(base_score-adapter_score,0)。用 retention 增益与 forgetting-gap 减少分别报告，不把正迁移全部称为“恢复原本遗忘”。Commonsense macro 先在八子任务内等权，再在四 benchmark families 间等权，不直接拼接全部题目算微平均。

## 12. 可用于论文的英文方法草稿

以下草稿只描述可核实操作。它不预先指定某个 test-set 最优 setting，也不声称识别有害方向。

### 12.1 Post-hoc spectral editing

We study post-hoc editing of a trained low-rank adapter. For an adapted module, let W⁰ denote the frozen pretrained weight and let D=BA be the learned low-rank product, so that the effective weight is W=W⁰+sD with the original positive LoRA scaling s. Unlike initialization-based or training-time adaptation methods, our procedure operates on the completed adapter and leaves the pretrained weight unchanged. Once its hyperparameters are fixed, the editing operation requires no examples, gradients, or additional optimization.

We compute the compact decomposition D=U diag(σ)Vᵀ without materializing the dense product. Reduced QR decompositions of B and Aᵀ give B=Q_B R_B and Aᵀ=Q_A R_A. We then decompose the small core R_B R_Aᵀ and recover the corresponding left and right singular bases. This implementation uses float32 for the decompositions and float64 internally for the spectral iteration.

Our Hybrid Newton–Schulz (HNS) editor transforms gains in the learned singular basis. Starting from x=σ/max(||σ||₂,ε), we apply K_f elementwise quintic steps with coefficients (3.4445,-4.7750,2.0315), followed by K_s steps with coefficients (2,-1.5,0.5). Each step has the form x←ax+bx³+cx⁵. These spectral steps are equivalent to a quintic matrix transformation while preserving the paired singular directions, but operate only on the low-dimensional gain vector. The fast coefficients are related to the Newton–Schulz transformation used in Muon; our method uses the transformation as an adapter post-processing operation rather than as a training optimizer. [Muon implementation](https://github.com/KellerJordan/Muon).

After the two stages, we clamp the gains to be nonnegative and rescale them to match the sum of the original singular values. Denoting the resulting vector by τ, this yields Σ_iτ_i=Σ_iσ_i and therefore preserves the nuclear norm of both D and its fixed-scaled effective update. A degenerate near-zero output sum is handled by the constant vector with this same total mass. An optional strength parameter λ interpolates between σ and τ; our full-edit comparisons use λ=1 and retain the source adapter rank budget.

We reconstruct balanced factors B'=U diag(√τ) and A'=diag(√τ)Vᵀ, cast them to the source storage dtype, and retain the original adapter scaling and configuration. The edited adapter is loaded through the same inference path as the source LoRA. The per-module arithmetic cost is O((d_in+d_out)r²+r³+(K_f+K_s)r), dominated by compact decomposition and reconstruction rather than dense model-weight processing.

### 12.2 Geometric interpretation

An idealized equal-gain endpoint is D_flat=(S/r)UVᵀ, where S=Σ_iσ_i. Among nonnegative spectra with total mass S, the constant spectrum minimizes both the largest gain and the squared Frobenius norm. Spectral balancing under a fixed nuclear norm thus couples redistribution of directional gains with a reduction in parameter-space energy. Finite HNS iterations need not attain this endpoint exactly and are not assumed to monotonically reduce every norm or every sample-wise output change.

For a fixed hidden-state second moment C=E[hhᵀ], the adapter output energy is Σ_iσ_i²v_iᵀCv_i. This distinguishes parameter-space strength from functional strength on a particular input distribution. Neither reduced energy nor preserved singular subspaces guarantees improved task performance: attenuation may remove excessive amplification or weaken task-essential corrections. We therefore evaluate both target-task quality and cross-task retention and do not assume a universal relationship between spectral flattening and generalization.

### 12.3 Scale and shape controls

To distinguish changes in shape from changes in strength, we construct controls in the original adapter basis. A per-module scalar multiplies σ by ||τ||₂/||σ||₂, matching the HNS Frobenius norm without changing the source spectral shape. Conversely, ShapeOnly rescales τ to restore the source Frobenius norm. HeadOnly and TailOnly accept only the coordinate-wise decreases and increases proposed by HNS, respectively: min(σ,τ) and max(σ,τ). Their names refer to suppressed and amplified source directions, not to a fixed top-k partition. We additionally restore the source Frobenius norm after HeadOnly to test the remaining shape intervention at matched strength.

We compare against global scalar controls where included in the experiment-specific protocol, and distinguish norm-matched scalars from data-calibrated or retention-selected scalars. The selection data, candidate grid, and target-feasibility criterion are reported separately. A scalar selected and evaluated on the same retention benchmarks is treated as an exploratory upper bound, not as an independently validated baseline.

Finally, an unchanged-spectrum SVD reconstruction control measures representation sensitivity. The main step-grid uses balanced square-root factors, whereas certain common-basis and signed-gain controls use a one-sided factorization. These representations are equivalent in exact arithmetic but may differ after finite-precision storage and inference; we therefore keep original-factor and reconstruction references explicit in the relevant comparisons.

## 13. 写入论文前的核对清单

1. 给每个主结果 panel 明示实际 (K_f,K_s)、来源 adapter、rank、strength、scope 和 factorization，而非照抄库默认。
2. 主文引用 LoRA 和 Muon；涉及 PiSSA/LoRA+/AdaLoRA 时引用各自原论文，并仅把已完成同协议实验的方法称为 evaluated baselines。
3. 明确主方法无再训练，但验证选参和机制缓存不属于 data-free 操作。
4. 不声称保持 Frobenius norm；它是重要的被改变因素，故需要 scalar/ShapeOnly 对照。
5. 不把 finite HNS 写成精确 whitened/orthogonal adapter；UVᵀ 是子空间 partial isometry，固定输入输出协方差不必各向同性。
6. 对 signed standardization 使用 signed gains 与绝对值核范数定义。
7. 对新增 seeds 保留配置边界，不用 item bootstrap CI 替代 training variance。
8. 本轮新增遗忘结果和 scalar comparison 未完成核实前，不更新为已证实的三-seed结论。

## 14. 源码与结果入口

| 内容 | 入口 |
|---|---|
| HNS 配置与谱迭代 | src/finetune/spectral_edit/posthoc_hns.py |
| QR-core SVD、balanced 重构 | src/finetune/spectral_edit/svd.py |
| 保存来源配置/scaling | src/finetune/spectral_edit/io.py；scripts/build_hns_step_grid_2x4.py |
| 核心 causal controls 与对齐 | src/finetune/spectral_edit/mechanism.py；scripts/build_hns_causal_controls.py |
| ExactFlat/Temperature 等 | src/finetune/spectral_edit/ablations.py |
| common-basis scalar 控制 | scripts/build_forgetting_common_basis_controls.py |
| HeadOnly 恢复与剂量 | scripts/build_headonly_fro_restore.py；scripts/build_metamath_head_dose_adapters.py |
| signed standardization | scripts/build_2x4_spectral_standardization.py；scripts/build_2x4_restored_standardization.py |
| 训练期 PEFT | src/finetune/peft_builders.py；src/finetune/train_sft_peft.py |
| 已尝试方法及分 panel 结果 | reports/hns_release_20260912/main/all_methods_main_table.md |
| 谱几何与固定输入推导 | reports/hns_release_20260912/main/mechanism_geometry_for_paper.md |
| 机制与限制 | reports/hns_release_20260912/main/mechanism_main_report.md |
| 原始遗忘控制与选择边界 | reports/hns_release_20260912/main/forgetting_main_report.md |
| 统一迭代网格 | reports/hns_release_20260912/rebuttal/hns_step_grid_report.md |

本报告中的算法公式以本地实现为依据；英文草稿中的几何界和能量式是注明条件的直接推导，未将外部优化器文献的性能结论转移到本项目。
