## Functional PR 的具体实现

### 定义与 Functional-HNS 的关系

对一个 LoRA 线性模块，实际更新为 $\Delta W=cBA=cU\operatorname{diag}(s)V^\top$，其中 $c$ 保持原 adapter scaling，本实验为 $32/16=2$。原 LoRA 的 $s=\sigma$，编辑后 $s=t$。令 $h_a$ 为 calibration 中第 $a$ 个有效 token 在该模块的输入 hidden state，$z_a=V^\top h_a$。缓存以下**未中心化二阶矩**，不减均值：

$$
M=\sum_{a=1}^{N}z_az_a^\top,\qquad
C_V=M/N,\qquad q_i=(C_V)_{ii}=\frac1N\sum_a(v_i^\top h_a)^2.
$$

方向 $i$ 的输出更新能量及其占比为

$$
E_i=(cs_i)^2q_i,\qquad
\pi_i=\frac{E_i}{\sum_jE_j}.
$$

报告字段 `functional_pr`（Raw PR）的定义是

$$
\boxed{\mathrm{Functional\ PR}=\frac{1}{\sum_i\pi_i^2}
=\frac{(\sum_i s_i^2q_i)^2}{\sum_i(s_i^2q_i)^2}}.
$$

公共 scaling $c^2$ 在占比中抵消；绝对能量统计仍包含 $c^2$。总能量满足 $\mathbb E\|\Delta Wh\|_2^2=\sum_iE_i$，因为 $U$ 的列正交。PR 衡量能量有效分布在多少个**源 SVD 方向**上，非零总能量时介于 1 与方向数 $r$ 之间；本实验 $r=16$。它不同于参数谱的 $r_2=(\sum_i\sigma_i)^2/\sum_i\sigma_i^2$，也不是只用 $\sigma_i^2$ 的参数能量 participation ratio。

Functional-HNS 使用 $g_i=\sigma_i(\bar q_i/\operatorname{median}(q))^{\alpha/2}$ 进行编辑，最后恢复参数 nuclear mass；PR 本身不是优化目标。编辑后 $t$ 仍按原 $V$ 的方向顺序与 $q$ 配对，不重新排序 functional gains。本报告的编辑后 PR 使用同一个冻结 Base activation 缓存，不代表编辑后模型自身轨迹上的实测 PR。

### Activation 实际如何采集

- 实现位于 [collect_functional_activation_three_seed.py](../../scripts/collect_functional_activation_three_seed.py) 的 `run()` 与模块 forward hook；输入渲染复用 [measure_lora_modification.py](../../scripts/measure_lora_modification.py) 的 `render()`。各任务从对应训练 SFT 数据固定取 256 个样本，sampling seed=42、最长 512 tokens，chat template 包含已有 assistant 响应，`add_generation_prompt=False`，Qwen 关闭 thinking。
- 模型为冻结 pretrained Base，`eval()`、`torch.inference_mode()`、`use_cache=False`；不加载 LoRA 到 forward 轨迹。hook 读取 LoRA 所对应线性层的 `inputs[0]`，与各 checkpoint 自己的 $V^\top$ 投影。seed43/44 的 $V$ 不同，因此必须各自采集方向统计；同一 Base/Task 的三个 seed 共享完全相同的输入与 Base forward。
- `z = F.linear(hidden, joined_Vh).float()`；用 `attention_mask` 把 padding 投影置零；将 batch/token 维展平为 `flat`，累积 `flat.T @ flat`。所有有效 prompt、assistant 及模板 token 都计入；不只取最后 token，也不只取 assistant token。$N$ 是 256 条截断后序列的有效 token 数总和，因此按 token 加权，长样本贡献更多。
- Base 权重与 $V^\top$ 投影使用 bf16；投影结果转 fp32，矩阵乘积 fp32，累积矩阵 fp64。TF32 关闭。源 compact SVD 的 fp32 $V^\top$ SHA256 必须与构建 adapter 时逐模块完全一致，再使用同样的 bf16 投影。batch 短测和 OOM 回退不改变取样；OOM 时回滚已累积矩阵，避免重复计数。
- NPZ 保存 `names`、`sigma`、`scales`、`coordinate_second_moment_sum`（每模块 16×16 的 $M$）、`per_sample_coordinate_energy`、`token_counts`、`sample_indices`、`basis_sha256`。逐方向矩阵对角线与逐样本能量之和核验相对误差小于 $2\times10^{-5}$。保存的是源方向投影二阶矩，不是完整 hidden covariance；不能用于任意新 $V$。

### Raw PR、Protected PR 与 Full-moment PR

`protected_functional_pr` 用 $\bar q_i=\max(q_i,0.1\operatorname{median}(q))$ 替换 $q_i$ 后计算同一 PR。Raw PR 始终使用未加下限的真实缓存 $q$。Protected Exact Functional Flat 满足 $t_i^2\bar q_i$ 相等，因此 Protected PR 为 16；Raw PR 只有在下限未改变方向能量时才必然为 16。当前数据各项结果以真实计算值为准。

`cov_functional_pr`（Full-moment PR）额外考虑不同方向 activation 的相关性。令

$$
G=\operatorname{diag}(s)C_V\operatorname{diag}(s),\quad
\lambda_i=\operatorname{eigval}(G)_i,\quad
\omega_i=\lambda_i/\sum_j\lambda_j,
$$

$$
\mathrm{Full\!\!\!-moment\ PR}=1/\sum_i\omega_i^2.
$$

实际代码用 $M$ 代替 $C_V$，公共 $N$ 和 $c^2$ 抵消。特征值使用 `np.linalg.eigvalsh(G).clip(min=0)`，去除数值舍入导致的负值。它等于输出更新未中心化二阶矩非零特征谱的 participation ratio，区别于 Raw PR 的方向能量占比。非零总能量下 Full-moment PR 不大于 Raw PR，因为 $\operatorname{tr}(G^2)=\sum_iG_{ii}^2+\sum_{i\ne j}G_{ij}^2$。缓存 $C_V$ 为对角矩阵时二者相等。

`functional_erank` 同时报告 $\exp(-\sum_i\pi_i\log\pi_i)$；`cov_functional_erank` 对 $\omega$ 使用相同定义。当前源谱与各编辑谱能量均正，因此 Raw entropy 的实现直接使用 `pi*np.log(pi)`；Full-moment entropy 为避免零特征值的 log(0)，log 输入裁剪为最小 $10^{-300}$，零项仍乘零。

### 与本次运行一致的 NumPy 计算

实现为 `scripts/prepare_functional_hns_seed_extension.py`（本地实现） 的 `functional_stats()`；seed42 pilot 的同名函数采用相同计算。以下展示每模块核心运算，`s` 为所评测 adapter 的 16 个 singular gains：

```python
z = np.load(activation_npz)
M = z['coordinate_second_moment_sum'][module_index].astype(np.float64)
N = z['token_counts'].sum()
q = np.diag(M) / N
s = np.asarray(singular_gains, dtype=np.float64)

energy = s**2 * q
pi = energy / energy.sum()
functional_pr = float(1 / np.square(pi).sum())
functional_erank = float(np.exp(-(pi * np.log(pi)).sum()))

q_protected = np.maximum(q, 0.1 * np.median(q))
protected_energy = s**2 * q_protected
protected_pi = protected_energy / protected_energy.sum()
protected_functional_pr = float(1 / np.square(protected_pi).sum())

G = M * s[:, None] * s[None, :]
eig = np.linalg.eigvalsh(G).clip(min=0)
omega = eig / eig.sum()
cov_functional_pr = float(1 / np.square(omega).sum())
cov_functional_erank = float(
    np.exp(-(omega * np.log(omega.clip(min=1e-300))).sum())
)
absolute_functional_energy = float(energy.sum() * scaling**2)
```

同一模块归一化前确认 $q$、谱及能量有限且总能量为正；Functional-HNS 对非正 $\operatorname{median}(q)$ 直接报错，不伪造 activation fallback。$\operatorname{median}$ 对 rank16 的中间两个值取平均；torch 使用 `quantile(q, .5)` 与 NumPy 对齐。

### 模块汇总、结果文件与可复核性

单 checkpoint 的 `functional_pr`、Protected PR、Full-moment PR 和 entropy 均为**全部 LoRA 模块的 median**，各模块等权，不是先拼接全部方向再算 PR。跨 checkpoint 方法均值是 18 个 checkpoint median 的算术平均；Base×Task 的 mean±SD 是 seed42/43/44 三个 checkpoint median 的均值与 **sample SD（ddof=1）**。

绝对 functional energy 跨模块求和；`functional_rms_ratio` 为 $\sqrt{\sum_m E'_m/\sum_m E_m}$，`functional_squared_energy_ratio` 为其平方。固定 nuclear budget 不固定 Frobenius norm 或 functional energy；因此 PR 变化与剂量变化共存，不能仅凭这些结果断言 PR 是收益的独立原因。

`functional_pr_module_results.tsv.gz` 保存全部 90 个 adapter/对照的逐模块 $s,q,\bar q$、方向能量、三类 PR、entropy 和源缓存 SHA；`evaluation_metrics_snapshot.json.gz` 保存最终全部 368 项指标原文；`evaluation_commands.json` 保存两 Base 的实际评测/评分命令。它们由 `scripts/export_functional_hns_results.py`（本地实现） 从已经完成的结果生成，并核对逐模块重新计算后的 median 与 `summary.json` 一致。完整模型权重、原始逐样本预测和 activation NPZ 保留在报告所列路径，Git 中上传结果表、指标快照、审计与哈希，不复制这些大文件。
