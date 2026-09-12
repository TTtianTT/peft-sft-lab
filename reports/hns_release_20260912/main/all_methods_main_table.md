# HNS 项目：全部已尝试谱编辑与缩放方法汇总

更新时间：2026-09-12

## Main table A：统一 2 bases × 4 tasks 配对机制评测

表中均为相对各自原始 LoRA 的百分点变化（pp）。Q-Code/Q-Math/Q-IF/Q-CS 分别为
Qwen3-8B 的 Magicoder/MetaMath/Tulu/Commonsense checkpoint；L-* 为相同任务的
Llama-3.1-8B-Instruct checkpoint。所有方法编辑 all modules，保持原 LoRA 的奇异向量
`U,V`，只修改 rank-16 谱。该 panel 内数字可直接横向比较。

| 方法 | Q-Code | Q-Math | Q-IF | Q-CS | L-Code | L-Math | L-IF | L-CS | 8-cell mean | >LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| ScalarShrink | +0.42 | +0.99 | −5.36 | +0.12 | +4.22 | +0.30 | +1.80 | +0.39 | +0.36 | 7/8 |
| ShapeOnly | +3.22 | +3.87 | −3.02 | −1.07 | +3.00 | +1.90 | +2.04 | −0.67 | +1.16 | 5/8 |
| HeadOnly | +3.13 | +3.11 | −1.98 | −0.52 | +4.44 | +2.05 | +3.61 | −0.27 | +1.70 | 5/8 |
| TailOnly | −0.11 | +1.06 | −6.58 | −0.46 | −0.61 | +0.23 | −0.64 | −0.65 | −0.97 | 2/8 |
| HeadOnly + FroRestore | +1.41 | +3.18 | −4.97 | −0.77 | +3.52 | +1.44 | +1.57 | −0.49 | +0.61 | 5/8 |
| **Full HNS（参照）** | **+3.72** | **+4.40** | **+1.81** | **−0.44** | **+4.02** | **+3.26** | **+2.07** | **−0.07** | **+2.35** | **6/8** |

解释边界：Full HNS 的均值最高；HeadOnly 是最强的单因素拆分，MetaMath 在两个 base 上都满足
`HeadOnly > TailOnly, ScalarShrink`。但是 HeadOnly+FroRestore 只有 Qwen MetaMath 保留全局
Holm 校正后的正效应。TailOnly 在 6/8 为负。ScalarShrink 在 Llama Magicoder 很强，说明某些
checkpoint 的收益主要可能来自 scale，而不是谱形状。Qwen Tulu 的 Full HNS 为正、所有拆分均为负，
显示明显的组合/非线性效应。

## Main table B：标准化分支（独立 full-set 2×4 评测）

下表同样是相对该次运行中 LoRA 的 pp 变化，但评测运行和 Panel A 不同，不能把单个 cell 与
Panel A 作逐点差值。该 panel 的目的，是完整记录所有作为 HNS 替代方案尝试过的标准化方法。

| 方法 | Q-Code | Q-Math | Q-IF | Q-CS | L-Code | L-Math | L-IF | L-CS | 8-cell mean | >LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct `(sigma-mu)/Var` | −66.46 | −84.15 | −47.87 | −90.79 | −53.66 | −77.10 | −48.99 | −87.96 | −69.62 | 0/8 |
| Direct `(sigma-mu)/Std` | −66.46 | −83.47 | −20.88 | −1.71 | −53.66 | −74.67 | −53.98 | −87.93 | −55.34 | 0/8 |
| Std + restore LoRA Fro | −1.22 | −1.66 | +1.85 | −2.14 | +3.05 | −2.27 | +0.18 | −3.02 | −0.65 | 3/8 |
| Std + restore LoRA nuclear | +0.61 | −1.81 | +1.48 | −2.39 | +0.61 | −3.26 | −1.30 | −3.36 | −1.18 | 3/8 |
| **Std + restore HNS Fro** | **+6.71** | **+0.46** | **+2.59** | **−2.36** | **+0.61** | **−2.19** | **+0.55** | **−3.29** | **+0.39** | **5/8** |
| Std + restore HNS nuclear | +0.61 | −1.81 | +1.48 | −2.41 | −0.61 | −2.57 | −0.93 | −3.29 | −1.19 | 2/8 |
| **Full HNS（同运行参照）** | **+8.54** | **+4.02** | **+3.51** | **−0.41** | **+0.61** | **+3.34** | **+1.11** | **−0.37** | **+2.54** | **6/8** |

Direct standardization 的失败主要来自尺度爆炸：总 adapter Frobenius norm 可达到原 LoRA 的
3.30–1251.73 倍。恢复尺度后模型恢复可用，但最强的 `restore HNS Fro` 仍在 8/8 cell 低于
同运行 HNS，平均低 2.16 pp。因此 centered signed standardization 不适合作为 HNS 替代。

## Main table C：Qwen MetaMath 定向数值/缩放控制

这些方法只在固定 GSM8K 子集上做过，因此用单独 panel 报告。主列是 512 题 locked validation
上的 strict accuracy；不同实验批次之间不直接比较。

| 实验批次 | 方法 | Strict score | Delta vs untouched LoRA | 结论 |
|---|---|---:|---:|---|
| 模块 scaling pilot | Untouched LoRA | 83.20 | 0.00 | 基线 |
| 模块 scaling pilot | GlobalNormMatched | 84.18 | +0.98 | 小幅正点估计 |
| 模块 scaling pilot | PerModuleSpectralScale | 83.20 | 0.00 | 没有优于全局缩放 |
| 模块 scaling pilot | ShuffledScale，3 seeds mean | 84.05 | +0.85 | 真正的模块对应关系没有优势 |
| 模块 scaling pilot | Full HNS | 86.91 | +3.71 | 但该批次有 factor/path 数值混杂 |
| common-basis recheck | Untouched LoRA | 83.40 | 0.00 | 基线 |
| common-basis recheck | Zero rebuild | 82.81 | −0.59 | 微小重构误差可改变生成 |
| common-basis recheck | Calibrated global scalar, gamma=0.40 | 85.74 | +2.34 | 校准集从固定网格选出 |
| common-basis recheck | PerModule norm-matched scalar | 84.77 | +1.37 | 低于 calibrated scalar |
| common-basis recheck | Common-basis HNS | 86.13 | +2.73 | 相对 calibrated scalar 仅 +0.39，CI 跨零 |
| direct-dose | HeadOnly, alpha=1 | — | +2.34 | 95% CI [−0.78,+5.47] |
| direct-dose | Per-module matched scalar | — | +1.76 | 捕获约 75% HeadOnly 正点估计 |
| direct-dose | HeadOnly − matched scalar | — | +0.59 | 95% CI [−1.37,+2.54]，不可辨识 |

因此，逐模块 scaling/打乱 scaling 分支没有显示可迁移的方法价值；在统一 `U,V` 表示后，HNS
也没有证明优于合理校准的单一全局 scalar。这个结论仅针对 Qwen MetaMath 当前设置。

## 具体实现

对每个 LoRA 模块写作

\[
\Delta W_m=B_mA_m=U_m\operatorname{diag}(\sigma_m)V_m^\top,
\]

并把 observed HNS 在同一基底中的谱记为 \(\tau_m\)。除 direct standardization 外，控制方法
保持 `U,V` 不变，只替换谱：

| 方法 | 实际谱变换 |
|---|---|
| Full HNS（参照） | 先令 \(x=\sigma/\|\sigma\|_2\)，逐方向迭代 \(x\leftarrow x(a+bx^2+cx^4)\)：8 次 `(3.4445,-4.7750,2.0315)`，再 2 次 `(2,-1.5,.5)`；最后恢复原 LoRA nuclear norm |
| ScalarShrink | \(\sigma'_m=c_m\sigma_m,\ c_m=\|\tau_m\|_2/\|\sigma_m\|_2\)；逐模块匹配 HNS Frobenius norm |
| ShapeOnly | \(\sigma'_m=\tau_m\|\sigma_m\|_2/\|\tau_m\|_2\)；保留 HNS shape，恢复 LoRA Frobenius norm |
| HeadOnly | \(\sigma'_m=\min(\sigma_m,\tau_m)\)，逐方向只接受 HNS 的压制 |
| TailOnly | \(\sigma'_m=\max(\sigma_m,\tau_m)\)，逐方向只接受 HNS 的提升 |
| HeadOnly+FroRestore | \(q_m=\min(\sigma_m,\tau_m),\ \sigma'_m=q_m\|\sigma_m\|_2/\|q_m\|_2\) |
| GlobalNormMatched | 所有模块用同一 \(\gamma\)，其中 \(\gamma^2=\sum_m\|\tau_m\|_2^2/\sum_m\|\sigma_m\|_2^2\) |
| PerModuleSpectralScale | \(\sigma'_m=\gamma_m\sigma_m,\ \gamma_m=\|\tau_m\|_2/\|\sigma_m\|_2\) |
| ShuffledScale | 在相同 module type 内随机置换 \(\gamma_m\)，再作一次全局修正以匹配总 Frobenius norm |
| Calibrated global scalar | \(\sigma'_m=\gamma\sigma_m\)；Qwen MetaMath 固定网格 `{1,.85,.70,.60,.50,.40}`，校准集选 strict 最优，选中 0.40 |
| Direct standardization | \(c_i=(\sigma_i-\mu)/\mathrm{Var}(\sigma)\) 或 \((\sigma_i-\mu)/\mathrm{Std}(\sigma)\)；不裁剪、不取绝对值、不恢复 norm |
| Restored standardization | 先 \(z_i=(\sigma_i-\mu)/\mathrm{Std}(\sigma)\)，再令 \(c=z\,T/\|z\|\)，其中目标 \(T\) 为 LoRA/HNS 的 Frobenius 或 nuclear norm |
| Head dose | \(\sigma_m(\alpha)=\sigma_m+\alpha[\min(\sigma_m,\tau_m)-\sigma_m]\)，测试 \(\alpha\in\{0,.25,.5,1\}\) |
| Dose-matched scalar | 对每个剂量和模块用 \(\gamma_m(\alpha)=\|\sigma_m(\alpha)\|_2/\|\sigma_m\|_2\) 缩放原谱 |
| Zero rebuild | 不改 \(\sigma\)，仅统一重构为 `A=Vh, B=U*diag(sigma)`，检查因子表示敏感性 |

Standardization 中居中后约 73.2% 系数为负，因此这些是原 `U,V` 基底中的 signed coefficients，
严格说不再是非负奇异值。

## 代码入口

- HNS：`src/finetune/spectral_edit/posthoc_hns.py`
- ScalarShrink / ShapeOnly / HeadOnly / TailOnly：`scripts/build_hns_causal_controls.py` 与
  `src/finetune/spectral_edit/mechanism.py`
- HeadOnly+FroRestore：`scripts/build_headonly_fro_restore.py`
- Direct standardization：`scripts/build_2x4_spectral_standardization.py`
- Norm-restored standardization：`scripts/build_2x4_restored_standardization.py`
- Common-basis global/per-module/HNS controls：`scripts/build_metamath_common_basis_controls.py`
- Head-dose 与 dose-matched scalar：`scripts/build_metamath_head_dose_adapters.py`
- Functional/Raw/random localization：`scripts/build_functional_localization_adapters.py`

## 不应塞进方法主表的 localization 诊断

FunctionalTop-K、RawTop-K、uniform random、matched layer/type random，以及 F×C 四象限，都是
“只把选中 25%/50% 模块替换成 observed HNS”的因果定位诊断，不是完整 adapter 方法。结果没有显示
FunctionalTop-K 稳定优于 RawTop-K 或 matched random；唯一通过全局校正的定位结果是 Llama Tulu
`Low-F/High-C Top-50` 的 +4.00 pp。这些结果适合放机制 appendix，而不是和全模块方法混入论文主表。

## 总结

在不把 calibrated scalar 当作主要比较对象时，Full HNS 是当前跨 8 checkpoint 平均表现最好的
方法（Panel A +2.35 pp）；其次是 HeadOnly（+1.70 pp）和 ShapeOnly（+1.16 pp）。但没有一个
非 HNS 方法普遍可靠：TailOnly 明显失败，centered standardization 灾难性失败，norm-restored
standardization 只得到很弱的平均收益，逐模块 scaling 也没有优于全局或打乱缩放。
