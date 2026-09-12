# HNS 机制验证最终报告：2 个 base × 4 类任务

更新时间：2026-09-10

## 结论先行

当前证据支持一个**收紧后的机制故事**，但不支持最强版本的单因果叙事：

> LoRA 的 dominant singular directions 在真实 hidden states 上会被进一步放大；HNS 确实系统性降低 extreme modification。这个变化在代码和数学任务上通常有益，但“functional concentration 高”或“p99 被压低”本身都不足以判定该方向是 excessive 还是 task-essential。最终收益还取决于 task、layer/module type、谱尺度、覆盖率以及多模块非线性交互。

最可靠的结果是：

1. **alignment amplification 成立。** 8/8 checkpoint 的 functional top-1 share 均高于 raw top-1 share，增幅为 3.5–27.2 pp。
2. **modification suppression 成立。** all-module HNS 在 8/8 checkpoint 都降低 per-layer p99，HNS/LoRA 比值为 0.175–0.581。
3. **数学任务最稳定。** Qwen MetaMath 的 HNS、ShapeOnly、HeadOnly+FrobeniusRestore 均有稳健正收益；Llama MetaMath 的 Full HNS 也通过全局 Holm 校正，但其 HeadOnly+FroRestore 只剩 noise-level 正点估计。
4. **Head suppression 是重要成分，但不是普适充分条件。** MetaMath 在两个 base 上均有 `HeadOnly > TailOnly, ScalarShrink`；Llama Tulu 也满足。但 Qwen Tulu 的所有单因素干预都下降，只有 Full HNS 上升。
5. **functional concentration 只比 raw spectrum 略强，尚未成为稳定 predictor。** 八 checkpoint 上 Spearman 分别为 0.452 与 0.357，均不显著；去掉 Magicoder 后相关性崩塌或反向。
6. **FunctionalTop-K 没有稳定击败 RawTop-K 或 matched random；数据也不支持 F×C 的简单联合规则。** 在新的结构匹配四象限干预中，唯一通过 60 项全局 BH/Holm 的结果是 Llama Tulu 的 `Low-F/High-C Top-50`（+4.00 pp），而不是 `High-F/High-C`。
7. **teacher-forced utility 与 downstream utility 明显错配。** Gradient compatibility 能预测 Qwen Commonsense 的 SFT-NLL utility，但该任务 downstream 反而下降；Llama Tulu 则在所有被测 adapter set 上 NLL 变差、downstream 却上升。多模块非加性也不能忽略。

因此，推荐使用的表述是 **“dominant-mode suppression 是 HNS 的主要候选机制之一，但其效用由数据条件下的任务兼容性门控”**，而不是“谱越尖，HNS 越有效”。

## 实验口径与完整性

- Base：Qwen3-8B、Llama-3.1-8B-Instruct。
- Task：Magicoder/code、MetaMath/math、Tulu/instruction following、CommonsenseQA suite。
- 主 2×4 实验均为 **all modules**：每层 `q/k/v/o/gate/up/down_proj`；Qwen 252 个模块，Llama 224 个模块。
- 每个 checkpoint 固定抽样 256 条训练分布样本，使用同一 pretrained-base hidden-state trajectory。
- 每个 LoRA 模块 rank=16；六组干预只改奇异值，保持原始 `U,V`。
- 8/8 checkpoint 均有 256 个唯一 sample index、6 个 variant；direction row 数全部匹配预期。
- 最大 singular-basis off-diagonal error 为 `1.37e-6`。
- 下游推断：10,000 次 paired bootstrap、100,000 次 paired permutation；类别结果使用 exact McNemar，并报告全局 Holm 与 BH-FDR。
- 自动审计结果：`PASS`。唯一发现的数据版本差异是 Llama Commonsense 的旧 aggregate point estimates；本报告使用后续 fresh rerun 的 paired estimates。

## 八 checkpoint 主结果

表中 p99 ratio 是各层 `p99(HNS)/p99(LoRA)` 的汇总比值；gain 和 CI 单位均为百分点。Llama Commonsense 使用 fresh rerun。

| Base | Task | Raw top-1 | Functional top-1 | Raw r_eff | Func r_eff | HNS p99 ratio | HNS gain [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3 | Magicoder | 75.7% | 93.4% | 9.66 | 1.37 | 0.175 | +3.72 [+0.38,+7.15] |
| Qwen3 | MetaMath | 66.3% | 79.5% | 10.60 | 2.32 | 0.221 | +4.40 [+2.65,+6.22] |
| Qwen3 | Tulu | 53.3% | 62.4% | 11.79 | 3.36 | 0.382 | +1.81 [−0.35,+4.02] |
| Qwen3 | Commonsense | 63.7% | 67.3% | 9.69 | 3.09 | 0.398 | −0.44 [−0.80,−0.10] |
| Llama3.1 | Magicoder | 55.8% | 83.0% | 12.35 | 2.19 | 0.229 | +4.02 [+0.53,+7.46] |
| Llama3.1 | MetaMath | 34.3% | 50.6% | 13.91 | 5.55 | 0.581 | +3.26 [+1.44,+5.08] |
| Llama3.1 | Tulu | 31.1% | 50.3% | 14.36 | 6.16 | 0.422 | +2.07 [−0.34,+4.56] |
| Llama3.1 | Commonsense | 52.5% | 63.4% | 11.50 | 3.68 | 0.472 | −0.07 [−0.47,+0.31] |

显著性边界：

- **可按强证据陈述：** Qwen MetaMath HNS（Holm p=0.00043）与 Llama MetaMath HNS（Holm p=0.01496）。
- **可按 nominal positive evidence 陈述：** Qwen Magicoder、Llama Magicoder；二者 CI 排除零，但未通过 43 项全局 Holm，BH q 分别为 0.076 与 0.054。
- **只能按 noise-level / suggestive 陈述：** 两个 Tulu 的 Full HNS；CI 均跨零。
- Qwen Commonsense 为小幅负向，CI 排除零、BH q=0.0366，但未通过 Holm；Llama Commonsense fresh rerun 与零不可区分。

## 六组谱干预：收益来自哪里

下表为相对 LoRA 的 point gain（pp）；显著性应以上一节及 `paired_inference.tsv` 为准。

| Base / task | ScalarShrink | ShapeOnly | HeadOnly | TailOnly | Full HNS |
|---|---:|---:|---:|---:|---:|
| Qwen Magicoder | +0.42 | +3.22 | +3.13 | −0.11 | +3.72 |
| Qwen MetaMath | +0.99 | +3.87 | +3.11 | +1.06 | +4.40 |
| Qwen Tulu | −5.36 | −3.02 | −1.98 | −6.58 | +1.81 |
| Qwen Commonsense | +0.12 | −1.07 | −0.52 | −0.46 | −0.44 |
| Llama Magicoder | +4.22 | +3.00 | +4.44 | −0.61 | +4.02 |
| Llama MetaMath | +0.30 | +1.90 | +2.05 | +0.23 | +3.26 |
| Llama Tulu | +1.80 | +2.04 | +3.61 | −0.64 | +2.07 |
| Llama Commonsense | +0.39 | −0.67 | −0.27 | −0.65 | −0.07 |

解释：

- MetaMath 在两个 base 上都复现 `HeadOnly > TailOnly / ScalarShrink`，是最干净的跨 base dominant-head suppression 证据。
- Llama Tulu 也复现该排序，但 Qwen Tulu 不复现；Qwen Tulu 表现为强非加性：Full HNS 正，四个拆分干预全负。
- TailOnly 在 6/8 checkpoint 为负，且几乎不压低 modification（p99 ratio 约 1.00–1.03），不支持“单独抬 tail 是主要收益来源”。
- ScalarShrink 不是统一解释：Qwen Magicoder/MetaMath 上远弱于 HeadOnly；但 Llama Magicoder 的 ScalarShrink +4.22 pp，几乎等于 HeadOnly +4.44 pp，形成明确 scale confound。
- ShapeOnly 保持原 LoRA Frobenius norm，却仍把 p99 ratio 降至 0.313–0.684。它在代码/数学可正向、在 Commonsense 和 Qwen Tulu 可负向，说明 extreme modification 可以靠谱重分配降低，但“降低 extreme”并不自动带来性能。

## HeadOnly + FrobeniusRestore：8/8 checkpoint 的 scale-confound 控制

该控制先应用 HNS 中所有 `sigma_HNS < sigma_LoRA` 的 head suppression，再逐模块整体重缩放，使 Frobenius norm 恢复到原 LoRA。

| Checkpoint | Plain HeadOnly | HeadOnly+FroRestore [95% CI] | 全局校正后的结论 |
|---|---:|---:|---|
| Qwen Magicoder | +3.13 | +1.41 [−1.72,+4.55] | noise；恢复 norm 后明显衰减 |
| Qwen MetaMath | +3.11 | +3.18 [+1.67,+4.78] | Holm p=0.0045，稳健正向 |
| Qwen Tulu | −1.98 | −4.97 [−7.54,−2.50] | Holm p=0.0045，稳健负向 |
| Qwen Commonsense | −0.52 | −0.77 [−1.09,−0.44] | Holm p=0.00048，稳健负向 |
| Llama Magicoder | +4.44 | +3.52 [+1.16,+5.99] | BH q=0.0126，未过 Holm |
| Llama MetaMath | +2.05 | +1.44 [−0.08,+2.96] | noise |
| Llama Tulu | +3.61 | +1.57 [−0.72,+3.87] | noise |
| Llama Commonsense | −0.27 | −0.49 [−0.82,−0.16] | BH q=0.0098，未过 Holm |

与 Plain HeadOnly 的直接 paired contrast 中，只有 Qwen Tulu 的 FroRestore 显著更差（−2.99 pp，95% CI [−5.09,−0.91]，8 项 Holm/BH p=0.0379）；其余 7 项直接差异均不显著。

因此，这个控制把“纯 head-shape 因果证据”主要限定到 Qwen MetaMath。Llama Magicoder 保留了 FDR-level 正效应，但其 ScalarShrink 本身也有 +4.22 pp，说明 shape 与 scale 两条路径同时存在；Qwen Tulu 则说明恢复原范数可能把 head-shape 干预推入明显有害区间。

## Parameter → functional → modification 链条

### 描述性链条成立

8/8 checkpoint 都满足 `functional top-1 > raw top-1`：

- Qwen：Magicoder +17.6 pp、MetaMath +13.2 pp、Tulu +9.0 pp、Commonsense +3.5 pp。
- Llama：Magicoder +27.2 pp、MetaMath +16.4 pp、Tulu +19.2 pp、Commonsense +10.9 pp。

因此 hidden-state alignment 的确会进一步放大 dominant singular direction。HNS 同时在所有 checkpoint 把 per-layer p99 ratio 压到 0.175–0.581，完成了“谱集中 → functional 集中 → extreme modification 被抑制”的描述性闭环。

### 但跨 checkpoint predictor 很弱

| Predictor | Spearman rho with HNS gain | exact permutation p |
|---|---:|---:|
| Raw top-1 | 0.357 | 0.389 |
| Functional top-1 | 0.452 | 0.267 |
| Functional effective rank | −0.476 | 0.243 |
| p99 suppression | 0.571 | 0.151 |
| Alignment amplification `functional−raw`（额外诊断） | 0.667 | 0.083 |

Functional 指标方向上比 raw 更接近预期，但 n=8 时没有一个达到显著。特别是 leave-one-task-out：去掉 Magicoder 后 raw rho=−0.029、functional rho=−0.086、p99 suppression rho=0.143。故不能写成“functional concentration 已被证明比 raw spectrum 更能预测 HNS gain”；最多只能写成**有方向一致的初步趋势，且 alignment amplification 比 concentration 本身更值得后续验证**。

## Functional localization

干预是：对选中模块复制 observed all-module HNS 的精确 A/B tensor，未选模块保持 LoRA。每个 scope 比较 FunctionalTop-K、RawTop-K、3 个 uniform random 和 3 个按 `module type × layer quartile` 匹配的 random。

| Case | Scope | Functional | Raw | Uniform random mean [range] | Matched random mean [range] |
|---|---:|---:|---:|---:|---:|
| Qwen Magicoder | 25% | −0.50 | −0.19 | +0.82 [−0.11,+1.58] | −1.06 [−1.69,−0.47] |
| Qwen Magicoder | 50% | +2.91 | +2.22 | +0.87 [+0.08,+1.72] | −0.24 [−0.69,+0.28] |
| Qwen Commonsense | 25% | +0.03 | +0.02 | −0.09 [−0.16,−0.03] | −0.33 [−0.45,−0.21] |
| Qwen Commonsense | 50% | −0.14 | −0.21 | −0.27 [−0.35,−0.19] | −0.21 [−0.25,−0.14] |
| Llama Tulu | 25% | +1.74 | +0.50 | +0.83 [+0.22,+1.23] | +1.40 [+0.84,+2.23] |
| Llama Tulu | 50% | +1.37 | +2.09 | +1.32 [+0.86,+1.84] | +1.95 [+1.43,+2.25] |

关键统计：

- Magicoder FunctionalTop-50 为 +2.91 pp，95% CI [−0.30,+6.30]，paired p=0.094；比 RawTop-50 高 +0.69 pp，但两者直接比较 p=0.537。
- Qwen Commonsense FunctionalTop-50 为 −0.14 pp，CI [−0.44,+0.15]；与零及 RawTop-50 均不可区分。
- Llama Tulu FunctionalTop-25 为 +1.74 pp，CI [−0.51,+4.01]；比 RawTop-25 高 +1.24 pp，直接比较 p=0.170。50% 时反而 Raw 和 matched random 更高。

结论：确实观察到 task-dependent sign 和 Magicoder 的 coverage threshold（25% 负、50% 正），但没有证据表明 FunctionalTop-K 稳定优于 RawTop-K 或 matched random。Llama Tulu 中 matched random 很强，Magicoder 中 matched random 为负，说明 layer/type 分布及跨模块协同是不可忽略的因果组成。

## Functional concentration × gradient compatibility：结构匹配四象限干预

为了直接检验“高 functional concentration 只有在与任务梯度兼容时才值得压制”，在三个代表 checkpoint 上按 `module type × layer quartile` 分层，并在每个层内分别取 F 与 C 的 rank corners。所有四象限都使用完全相同的分层 quota；25%/50% 分别编辑 Qwen 的 63/126 个模块、Llama 的 56/112 个模块。这里的 High-C 是**层内相对高**，不等价于绝对 `C>0`。30 个生成 adapter 已逐张量核对：被选模块精确等于 observed all-module HNS，未选模块精确等于 LoRA，mismatch=0。

相对 LoRA 的 downstream gain（pp）：

| Case | Scope | High-F/High-C | High-F/Low-C | Low-F/High-C | Low-F/Low-C | C-only Top-K |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Magicoder | 25% | +0.89 | +0.25 | +1.00 | −0.03 | +0.69 |
| Qwen Magicoder | 50% | +0.64 | +0.80 | −0.25 | +1.91 | +1.14 |
| Qwen Commonsense | 25% | −0.02 | −0.13 | −0.19 | −0.09 | −0.13 |
| Qwen Commonsense | 50% | −0.24 | −0.05 | −0.36 | −0.19 | −0.15 |
| Llama Tulu | 25% | +0.14 | +0.37 | +2.21 | +1.30 | +1.52 |
| Llama Tulu | 50% | +1.94 | +1.49 | **+4.00** | +1.33 | +1.51 |

统计裁决：

- 60 个 comparison 统一校正后，唯一通过 Holm/BH 的 vs-LoRA 干预是 Llama Tulu `Low-F/High-C Top-50`：+4.00 pp，95% CI [+1.94,+6.13]，Holm=BH=0.012。
- 在 Llama Tulu 的 High-C 条件内，提高 F 反而降低 point estimate：25% 时 `HH−LH=−2.07 pp`，50% 时同为 `−2.07 pp`；两者 nominal p<0.05，但未通过全局 BH。
- Qwen Magicoder 没有一个四象限 contrast 显著，且 Top-50 最好的是 Low-F/Low-C；Qwen Commonsense 各组几乎全为负或零。

因此，数据不支持简单的 `High-F × High-C ⇒ high utility` 判据。新的、更窄结论是：**compatibility 在某些任务上能定位有益区域，但高 functional concentration 既非必要条件，也可能在兼容区域内成为负向筛选因素。** 三个任务仍呈现清楚的 task-dependent sign：Magicoder 弱正且高度依赖组合，Commonsense 负/零，Llama Tulu 明显正。

## Module-level utility：能否识别 excessive 与 task-essential directions

定义：

\[
U_m = \operatorname{NLL}(\mathrm{LoRA})-
\operatorname{NLL}(\text{only module }m\text{ copied from observed HNS}).
\]

正值表示单独编辑模块 `m` 改善 held-out SFT NLL。Gradient compatibility 使用随机排列中的前 64 条训练样本，utility 使用后续 64 条；二者不重叠。Magicoder 因 3 条样本在截断后没有监督 token，实际使用 61 条。

### Utility 分布

| Case | Modules / samples | Mean U | Median U | U>0 | CI>0 / CI<0 |
|---|---:|---:|---:|---:|---:|
| Qwen Magicoder | 252 / 61 | −2.43e−4 | −2.24e−4 | 8.3% | 0 / 24 |
| Qwen Commonsense | 252 / 64 | +1.32e−2 | +4.07e−3 | 60.7% | 80 / 41 |
| Llama Tulu | 224 / 64 | +4.24e−5 | +6.68e−5 | 56.2% | 3 / 4 |

这个结果非常重要：Magicoder 的 Full HNS 下游为正，但几乎所有 single-module HNS 都让 held-out SFT NLL 变差；Qwen Commonsense 则相反，许多单模块编辑改善 SFT NLL，而 Full HNS 的 commonsense benchmark 略降。说明 `U_m` 是**局部、teacher-forced、训练分布 loss utility**，不是 downstream generation/accuracy utility；多模块非加性与目标指标错配都很强。

### 单变量 Spearman

| Case | Functional F | Raw | Gradient compatibility C | p99 contribution |
|---|---:|---:|---:|---:|
| Qwen Magicoder | −0.177 | −0.230 | +0.196 | −0.174 |
| Qwen Commonsense | +0.058 | +0.002 | **+0.869** | −0.016 |
| Llama Tulu | −0.213 | −0.157 | +0.073 | −0.325 |
| Pooled, within-case z(U) | −0.068 | −0.095 | +0.295 | −0.201 |

模块并非独立样本，表中 permutation p 应视为 exploratory；不能把名义显著性当成 checkpoint-level replication。

### Predictive model

Ridge 模型包含 layer 与 module-type fixed effects。报告 out-of-fold R²：

| Case | Structure | +F | +C | +p99 | Full | Full leave-layer-quartile-out |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Magicoder | 0.063 | 0.009 | 0.133 | −0.002 | 0.170 | −0.048 |
| Qwen Commonsense | −0.013 | 0.012 | **0.858** | −0.066 | 0.849 | **0.735** |
| Llama Tulu | 0.236 | 0.231 | 0.161 | 0.217 | 0.201 | 0.236 |
| Pooled | 0.023 | 0.026 | 0.344 | 0.029 | 0.353 | 0.301 |

`F` 没有稳定增量预测价值。`C` 在 Qwen Commonsense 极强，说明一阶 loss-gradient compatibility 可以跨 held-out 样本预测单模块 NLL 方向；但它在 Magicoder 和 Llama Tulu 不稳定，而且并不能预测下游 benchmark 的任务级符号。当前不存在一个已验证的、跨任务通用的简单判据。

### Adapter-set NLL 与非加性核对

对 Full HNS、FunctionalTop-50、C-only Top-50、High-F/High-C Top-50 使用与 single-module utility 相同的 held-out 样本重新测量 teacher-forced NLL：

| Case | Full HNS U | FunctionalTop-50 U | C-only Top-50 U | High-F/High-C Top-50 U | Downstream 方向 |
|---|---:|---:|---:|---:|---|
| Qwen Magicoder | −0.18175 | −0.07333 | −0.01982 | −0.02717 | 多数正向 |
| Qwen Commonsense | +2.34276 | +2.24289 | +3.03127 | +2.83425 | 负/零 |
| Llama Tulu | −0.04003 | −0.02913 | −0.01324 | −0.01356 | 正向 |

`U>0` 表示 NLL 改善。三者均显示 SFT-NLL 和 downstream 的符号可直接相反：Qwen Commonsense 的 NLL 大幅改善但 benchmark 下降；Magicoder/Llama Tulu 的 NLL 变差但 generation benchmark 上升。这说明 C 的强预测力是真实的，却只针对 teacher-forced surrogate，不能被解读成 downstream compatibility。

集合效应也不是单模块 utility 的简单加和。定义 interaction 为 `observed set U − Σ single-module U`：

- Qwen Magicoder Full HNS interaction = −0.12053，95% CI [−0.20672,−0.03391]，为显著协同伤害。
- Qwen Commonsense C-only Top-50 = −1.6395 [−2.433,−0.840]，High-F/High-C Top-50 = −1.2605 [−2.028,−0.515]，均为显著 antagonism。
- Llama Tulu 的四组 interaction 均为负且 Full HNS 出现 `ΣU>0`、`observed U<0` 的符号翻转，但 CI 跨零。

这解释了为什么 single-module 回归无法直接给出可靠 intervention policy：模块编辑之间存在实际的跨层/跨模块交互，且优化 surrogate 与最终评测目标不一致。

## 对四条核心假设的裁决

1. **“参数谱集中只有在 hidden states 强激活 dominant directions 时，才形成严重 functional concentration。”——支持描述性部分。** 8/8 都发生 alignment amplification；Magicoder amplification 最大且是正例。但“只有在”仍缺少 direction-level counterfactual，当前数据证明相关链条，不证明必要性。
2. **“主要收益来自压 dominant modes，而非整体 shrink 或单独抬 tail。”——部分支持。** MetaMath 最支持，TailOnly 总体弱/负；但 Llama Magicoder 存在 ScalarShrink confound，Qwen Tulu 是非加性例外，8/8 HeadFro 中只有 Qwen MetaMath 给出 Holm-robust 的纯 shape 正点估计。
3. **“完全谱平坦不充分，收益依赖 functional compatibility。”——支持前半句，但 compatibility 定义需要收紧。** Commonsense、Qwen Tulu 拆分、localization 和四象限干预都否定“越平越好”；然而 SFT-gradient compatibility 只能预测局部 NLL，不能统一预测 downstream，且 Llama Tulu 的最佳组是 Low-F/High-C。
4. **“Magicoder 正例、Commonsense 负例，MetaMath/Tulu 验证跨任务。”——大体成立但需要细化。** 两个 base 的 Magicoder/MetaMath 都正，Commonsense 都负或零；Tulu 两个 Full HNS 都为正点估计，但只有 Llama 的 HeadOnly 有较强证据，Qwen Tulu 机制不同。

## 两个 base × 四任务的共性

- Full HNS gain 的 task-level 符号跨 base 一致：code/math/IF 为正点估计，commonsense 为负或零。
- MetaMath 是跨 base 最一致的 head-suppression regime。
- TailOnly 在两个 base 的 code/Tulu/commonsense 上普遍无益；tail enhancement 不是核心正机制。
- HellaSwag 是稳定 failure category：Qwen HNS −0.88 pp、Llama HNS −1.87 pp，均经 category BH-FDR 显著。失败更像 continuation/task-format compatibility，而不是所有 commonsense 能力一起下降。
- Base 仍会改变 confound：Llama Magicoder 对整体缩放高度敏感；Qwen Tulu 显示联合谱阈值；Llama Tulu 更接近 HeadOnly-positive regime。

## 新发现的 failure conditions

1. **Task-essential dominant mode：** Commonsense 中压 head/改 shape 可降低 modification，却不提升 accuracy；dominant direction 可能承载必要决策边界。
2. **Scale confound：** Llama Magicoder 的 ScalarShrink 足以复现几乎全部收益；plain HeadOnly 不能直接等同于 shape mechanism。
3. **Coverage threshold / non-additivity：** Magicoder FunctionalTop-25 为负而 Top-50 为正；Qwen Tulu 只有 Full HNS 正。
4. **Layer/type structural confound：** matched random 在 Llama Tulu 很强、在 Magicoder 很差；不能只按 F 排序解释定位结果。
5. **Surrogate mismatch：** held-out SFT NLL utility 与 generation/accuracy 的 task-level 符号可相反。
6. **Correlation fragility：** pooled n=8 的 functional correlation 由 Magicoder 点强烈驱动，leave-one-Magicoder-out 后崩塌。
7. **Category-specific incompatibility：** HellaSwag 在两个 base 上稳定受损，提示 continuation-style evaluation 是独立 failure regime。
8. **Compatibility-objective mismatch：** 一阶训练 loss gradient 可以很好地预测 held-out SFT NLL，却与最终 generation/accuracy 的符号相反；“兼容”必须针对真正的 downstream objective 定义。
9. **F×C 非单调：** Llama Tulu 中 Low-F/High-C Top-50 是唯一全局显著定位干预，高 F 在 High-C 条件内反而降低点估计；functional concentration 不能直接当作 intervention priority。
10. **Set non-additivity：** 若干 adapter set 的 observed utility 显著低于 single-module utility 之和；逐模块打分后直接 Top-K 组合不可靠。

## 最终可讲与不可讲

可以讲：

- HNS 跨 base 地降低 functional concentration 和 extreme LoRA modification。
- 数学任务提供了最强的 dominant-head suppression 因果证据。
- functional alignment 比 raw spectrum 更接近下游效应，但目前只是趋势。
- HNS 是否有效由 task-conditioned compatibility、scope 和结构位置共同决定。
- 现有 gradient compatibility 是有效的局部 NLL 指标，但不是 downstream utility 的通用替代变量。

暂时不能讲：

- “functional top-1 已被证明比 raw top-1 更好地预测 HNS gain。”
- “HeadOnly 普遍优于 ScalarShrink。”
- “找到高 functional concentration 模块并压制即可稳定提升。”
- “High-F 与 High-C 同时成立的模块应优先干预。”
- “单模块 SFT-NLL utility 可直接替代 downstream utility。”

## 结果位置与复现

- 主 2×4 谱、functional spectrum、modification 与 benchmark：`/dataset1/zailong/runs/peft-sft-lab/hns-2x4-allmodules-20260909/`
- 配对统计、category CI、leave-out correlation、HeadFro：`/dataset1/zailong/runs/peft-sft-lab/hns-mechanism-first-batch-20260909/`
- Functional localization：`/dataset1/zailong/runs/peft-sft-lab/hns-functional-localization-20260909/`
- Module utility：`/dataset1/zailong/runs/peft-sft-lab/hns-module-utility-20260909/`
- 自动审计：`reports/hns_2x4_mechanism_audit_20260910.json`

重新运行完整性审计：

```bash
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/audit_hns_mechanism_results.py \
  --aggregate_root /dataset1/zailong/runs/peft-sft-lab/hns-2x4-allmodules-20260909 \
  --statistics_root /dataset1/zailong/runs/peft-sft-lab/hns-mechanism-first-batch-20260909/statistics \
  --localization_root /dataset1/zailong/runs/peft-sft-lab/hns-functional-localization-20260909 \
  --utility_root /dataset1/zailong/runs/peft-sft-lab/hns-module-utility-20260909 \
  --output reports/hns_2x4_mechanism_audit_20260910.json
```
