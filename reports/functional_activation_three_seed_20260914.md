# 完整三种子 activation 与功能谱重算（2026-09-14）

已采集两Base×三Task×三个训练seed的全部18个源checkpoint；补齐原缺失的12组seed43/44，并重采6组seed42用于旧缓存核验及补存16×16方向二阶矩。两张B300分别处理一个Base，每worker只见一张GPU，array最多两个worker。没有重新训练，没有修改adapter，没有重跑或替换原始性能/遗忘成绩。

每个Base×Task使用原activation manifest的同一256条训练分布样本、相同顺序与chat rendering、最长512 tokens。sampling seed始终42，它与LoRA训练seed42/43/44是不同概念；冻结Base轨迹相同，因此一次前向可以同时投影到三个source的V方向。

参数/结构谱量沿用上一轮审计产物；本轮12个功能量在全部282个实际adapter、450条分波次方法记录上完整计算。独立训练单位仍只有18，重复波次和大量module/token不能当作独立训练重复。DG-Hard已被此前逐token核验证实为identity，故沿用LoRA功能谱，不作为新增独立方法拟合。

**重算结论：补齐seed43/44后，功能PR/entropy的编辑前后关联仍在；但还没有找到能跨模型、跨任务稳定同时预测性能和遗忘的通用谱量。**

最新matched wave包含LoRA与三种编辑时，functional PR与ΔTarget / FG reduction的within-source ρ=0.616 / 0.696，FDR q=0.000600 / 0.000600。只比较已编辑adapter后，ρ降为0.149 / 0.302，q=0.573 / 0.169，尚不能稳定排序Flat-Fro、Flat-Nuclear、HNS。

full-moment PR/entropy包含方向间相关性，但没有明显改善最新wave的已编辑方法排序。未编辑source的full-moment PR与绝对FG在Base×Task组内ρ=0.730（更大表示更多遗忘），全候选FDR q=0.184，仍是探索性风险信号；这个问题与编辑前后改善的比较不同。

源checkpoint质量的嵌套留seed谱量选择：Target RMSE=1.220 vs training-group mean 1.184 pp；FG RMSE=3.097 vs 2.131 pp。尚未超过组均值对照，干预的跨Base/Task留出也未支持通用双outcome预测。

## 本轮定义与数值检查

| 量 | 定义 |
| --- | --- |
| functional_participation | median 1/sum(pi^2), pi=source_response_energy*beta^2 / module total |
| functional_erank | median exp(H(pi)) |
| functional_top1 | median max(pi) |
| functional_erank_iqr | IQR of module exp(H(pi)) |
| functional_attn_mlp_gap | attention median functional erank minus MLP median |
| functional_energy_ratio | sqrt(sum source_response_energy*beta^2 / sum source_response_energy) |
| gamma_functional | sum(wE*beta), wE=source_response_energy/sum(source_response_energy) |
| eta_functional | 1-gamma_functional^2/sum(wE*beta^2) |
| functional_cov_participation | median trace(G)^2/trace(G@G), G=diag(scaled target spectrum) C diag(scaled target spectrum) |
| functional_cov_erank | median exp(entropy(normalized eigenvalues of G)); includes cross-direction second moments |
| functional_cov_top1 | median largest eigenvalue(G)/trace(G) |
| functional_offdiag_fraction | median ||G-diag(G)||F^2/||G||F^2; cross-direction energy structure |

$C=\sum_{\mathrm{nonpad\ tokens}}(V^\top x)(V^\top x)^\top$ 为未中心化二阶矩。旧diagonal定义使用 $\pi_i=t_i^2 C_{ii}/\sum_jt_j^2C_{jj}$；新增full-moment版本用 $G=\operatorname{diag}(ct)C\operatorname{diag}(ct)$ 的归一化特征值计算PR/entropy。它能包含方向间相关性，但仍是固定Base轨迹诊断，不能代表编辑后所有hidden states。

模型与V方向投影bf16，坐标乘积fp32、跨batch累加fp64；关闭TF32。保存每条样本的16方向能量、全二阶矩、Base输出能量、token数量、来源权重和basis哈希。完整moment diagonal与逐sample sum检查、PSD检查、保存adapter实际谱与目标谱检查通过。Flat谱的Hill/kurtosis等数学未定义项继续保持None，不能为了“补全”伪造数值。

## 全部18源checkpoint：质量、遗忘与HNS收益

Within Base×Task为六个组内分别对三个seed排名、中心化后合并相关，精确枚举46656种组内seed排列。FDR按outcome对参数与功能候选一起校正。绝对FG越小越好；FG reduction越大越好。

| Feature | Target raw ρ | Target within ρ | Target q | FG raw ρ | FG within ρ | FG q | HNS ΔTarget within ρ | HNS FG reduction within ρ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| erank_iqr | 0.163 | 0.713 | 0.576 | 0.286 | -0.091 | 0.955 | -0.305 | -0.365 |
| erank_depth_slope | 0.744 | -0.045 | 0.957 | -0.413 | -0.639 | 0.218 | -0.087 | -0.365 |
| erank_attn_mlp_gap | -0.413 | 0.223 | 0.932 | 0.575 | -0.183 | 0.934 | -0.087 | 0.091 |
| head_tail | 0.410 | 0.178 | 0.932 | -0.049 | 0.091 | 0.955 | 0.000 | 0.365 |
| functional_participation | 0.191 | -0.178 | 0.932 | -0.364 | 0.639 | 0.218 | 0.348 | 0.365 |
| functional_erank | 0.202 | -0.267 | 0.932 | -0.372 | 0.548 | 0.331 | 0.479 | 0.274 |
| functional_top1 | -0.189 | 0.267 | 0.932 | 0.370 | -0.548 | 0.331 | -0.479 | -0.274 |
| functional_erank_iqr | 0.225 | -0.312 | 0.932 | -0.274 | 0.639 | 0.218 | 0.305 | 0.365 |
| functional_attn_mlp_gap | -0.028 | -0.624 | 0.691 | -0.327 | -0.091 | 0.955 | -0.044 | 0.183 |
| functional_energy_ratio | — | — | — | — | — | — | — | — |
| gamma_functional | — | — | — | — | — | — | — | — |
| eta_functional | — | — | — | — | — | — | — | — |
| functional_cov_participation | 0.213 | -0.312 | 0.932 | -0.338 | 0.730 | 0.184 | 0.261 | 0.456 |
| functional_cov_erank | 0.216 | -0.178 | 0.932 | -0.331 | 0.730 | 0.184 | 0.435 | 0.456 |
| functional_cov_top1 | -0.216 | 0.178 | 0.932 | 0.355 | -0.456 | 0.539 | -0.261 | -0.183 |
| functional_offdiag_fraction | 0.395 | -0.401 | 0.885 | -0.215 | 0.091 | 0.955 | 0.435 | 0.365 |

## 同checkpoint内的干预比较：包含LoRA / edited-only

| Wave | Feature | ΔTarget ρ / q | FG reduction ρ / q | edited ΔTarget ρ / q | edited FG reduction ρ / q |
| --- | --- | --- | --- | --- | --- |
| joint_flat_hns | functional_participation | 0.616 / 0.001 | 0.696 / 0.001 | 0.149 / 0.573 | 0.302 / 0.169 |
| joint_flat_hns | functional_erank | 0.616 / 0.001 | 0.696 / 0.001 | 0.149 / 0.573 | 0.302 / 0.169 |
| joint_flat_hns | functional_top1 | -0.616 / 0.001 | -0.654 / 0.001 | -0.149 / 0.573 | -0.168 / 0.386 |
| joint_flat_hns | functional_erank_iqr | 0.188 / 0.183 | 0.418 / 0.004 | -0.149 / 0.573 | 0.235 / 0.266 |
| joint_flat_hns | functional_attn_mlp_gap | 0.146 / 0.296 | 0.289 / 0.038 | 0.099 / 0.627 | -0.168 / 0.386 |
| joint_flat_hns | functional_energy_ratio | -0.633 / 0.001 | -0.671 / 0.001 | -0.257 / 0.384 | -0.291 / 0.169 |
| joint_flat_hns | gamma_functional | -0.633 / 0.001 | -0.671 / 0.001 | -0.257 / 0.384 | -0.291 / 0.169 |
| joint_flat_hns | eta_functional | 0.633 / 0.001 | 0.671 / 0.001 | 0.257 / 0.384 | 0.291 / 0.169 |
| joint_flat_hns | functional_cov_participation | 0.599 / 0.001 | 0.696 / 0.001 | 0.099 / 0.627 | 0.302 / 0.169 |
| joint_flat_hns | functional_cov_erank | 0.616 / 0.001 | 0.696 / 0.001 | 0.149 / 0.573 | 0.302 / 0.169 |
| joint_flat_hns | functional_cov_top1 | -0.599 / 0.001 | -0.654 / 0.001 | -0.099 / 0.627 | -0.168 / 0.386 |
| joint_flat_hns | functional_offdiag_fraction | 0.670 / 0.001 | 0.675 / 0.001 | 0.297 / 0.384 | 0.235 / 0.266 |
| hns_grid_target | functional_participation | 0.389 / 0.001 | — / — | 0.188 / 0.106 | — / — |
| hns_grid_target | functional_erank | 0.383 / 0.001 | — / — | 0.180 / 0.106 | — / — |
| hns_grid_target | functional_top1 | -0.391 / 0.001 | — / — | -0.191 / 0.106 | — / — |
| hns_grid_target | functional_erank_iqr | 0.008 / 0.921 | — / — | -0.141 / 0.126 | — / — |
| hns_grid_target | functional_attn_mlp_gap | -0.028 / 0.803 | — / — | -0.112 / 0.169 | — / — |
| hns_grid_target | functional_energy_ratio | -0.361 / 0.001 | — / — | -0.154 / 0.106 | — / — |
| hns_grid_target | gamma_functional | -0.362 / 0.001 | — / — | -0.155 / 0.106 | — / — |
| hns_grid_target | eta_functional | 0.346 / 0.001 | — / — | 0.136 / 0.126 | — / — |
| hns_grid_target | functional_cov_participation | 0.340 / 0.001 | — / — | 0.121 / 0.155 | — / — |
| hns_grid_target | functional_cov_erank | 0.429 / 0.001 | — / — | 0.244 / 0.054 | — / — |
| hns_grid_target | functional_cov_top1 | -0.369 / 0.001 | — / — | -0.158 / 0.106 | — / — |
| hns_grid_target | functional_offdiag_fraction | 0.361 / 0.001 | — / — | 0.151 / 0.126 | — / — |
| hns_grid_retention | functional_participation | 0.366 / 0.001 | 0.299 / 0.003 | 0.197 / 0.148 | 0.003 / 0.979 |
| hns_grid_retention | functional_erank | 0.387 / 0.001 | 0.290 / 0.004 | 0.228 / 0.124 | -0.010 / 0.979 |
| hns_grid_retention | functional_top1 | -0.344 / 0.001 | -0.315 / 0.003 | -0.167 / 0.165 | -0.028 / 0.979 |
| hns_grid_retention | functional_erank_iqr | -0.026 / 0.864 | 0.207 / 0.031 | -0.177 / 0.165 | -0.038 / 0.979 |
| hns_grid_retention | functional_attn_mlp_gap | 0.009 / 0.940 | 0.275 / 0.007 | -0.081 / 0.439 | 0.068 / 0.979 |
| hns_grid_retention | functional_energy_ratio | -0.337 / 0.002 | -0.264 / 0.007 | -0.154 / 0.170 | 0.051 / 0.979 |
| hns_grid_retention | gamma_functional | -0.338 / 0.002 | -0.261 / 0.007 | -0.156 / 0.170 | 0.056 / 0.979 |
| hns_grid_retention | eta_functional | 0.329 / 0.002 | 0.267 / 0.007 | 0.143 / 0.189 | -0.047 / 0.979 |
| hns_grid_retention | functional_cov_participation | 0.334 / 0.002 | 0.284 / 0.004 | 0.154 / 0.170 | -0.019 / 0.979 |
| hns_grid_retention | functional_cov_erank | 0.382 / 0.001 | 0.309 / 0.003 | 0.221 / 0.124 | 0.019 / 0.979 |
| hns_grid_retention | functional_cov_top1 | -0.360 / 0.001 | -0.283 / 0.005 | -0.186 / 0.165 | 0.022 / 0.979 |
| hns_grid_retention | functional_offdiag_fraction | 0.391 / 0.001 | 0.341 / 0.003 | 0.232 / 0.124 | 0.067 / 0.979 |
| scalar_common_basis_seed42 | functional_participation | 0.181 / 0.350 | 0.196 / 0.237 | 0.156 / 0.500 | 0.150 / 0.445 |
| scalar_common_basis_seed42 | functional_erank | 0.181 / 0.350 | 0.196 / 0.237 | 0.156 / 0.500 | 0.150 / 0.445 |
| scalar_common_basis_seed42 | functional_top1 | -0.181 / 0.350 | -0.196 / 0.237 | -0.156 / 0.500 | -0.150 / 0.445 |
| scalar_common_basis_seed42 | functional_erank_iqr | 0.014 / 0.968 | 0.317 / 0.114 | 0.017 / 0.961 | 0.374 / 0.088 |
| scalar_common_basis_seed42 | functional_attn_mlp_gap | -0.014 / 0.968 | 0.317 / 0.114 | 0.017 / 0.961 | 0.374 / 0.088 |
| scalar_common_basis_seed42 | functional_energy_ratio | -0.590 / 0.003 | -0.601 / 0.003 | -0.437 / 0.078 | -0.445 / 0.027 |
| scalar_common_basis_seed42 | gamma_functional | -0.578 / 0.003 | -0.601 / 0.003 | -0.419 / 0.078 | -0.445 / 0.027 |
| scalar_common_basis_seed42 | eta_functional | 0.117 / 0.563 | 0.197 / 0.237 | 0.023 / 0.961 | 0.106 / 0.520 |
| scalar_common_basis_seed42 | functional_cov_participation | 0.181 / 0.350 | 0.196 / 0.237 | 0.156 / 0.500 | 0.150 / 0.445 |
| scalar_common_basis_seed42 | functional_cov_erank | 0.181 / 0.350 | 0.196 / 0.237 | 0.156 / 0.500 | 0.150 / 0.445 |
| scalar_common_basis_seed42 | functional_cov_top1 | -0.181 / 0.350 | -0.196 / 0.237 | -0.156 / 0.500 | -0.150 / 0.445 |
| scalar_common_basis_seed42 | functional_offdiag_fraction | 0.181 / 0.350 | 0.196 / 0.237 | 0.156 / 0.500 | 0.150 / 0.445 |

## 原六source与完整三种子的覆盖对照

该对照区分seed42重采引起的数值差异与增加seed43/44后统计变化。各seed的完整相关在per_seed_correlations.tsv/json。

| Feature | Outcome | 旧seed42 ρ | 重采seed42 ρ | 完整18-source ρ |
| --- | --- | --- | --- | --- |
| functional_participation | target_gain | 0.622 | 0.622 | 0.616 |
| functional_participation | forgetting_reduction | 0.700 | 0.700 | 0.696 |
| functional_erank | target_gain | 0.622 | 0.622 | 0.616 |
| functional_erank | forgetting_reduction | 0.700 | 0.700 | 0.696 |
| functional_cov_participation | target_gain | — | 0.622 | 0.599 |
| functional_cov_participation | forgetting_reduction | — | 0.700 | 0.696 |
| functional_cov_erank | target_gain | — | 0.622 | 0.616 |
| functional_cov_erank | forgetting_reduction | — | 0.700 | 0.696 |

## 嵌套留出验证

同checkpoint干预CV排除baseline与0+0 identity control；内层leave-source选择单量/双量、linear/quadratic、ridge；外层leave seed/base/task。各wave独立，不把不同协议波次混为同一性能表。完整系数、训练尺度、选择、每checkpoint预测见cv_*。

| Wave | Model | Leave | Target R² | FG reduction R² | Skill vs train mean |
| --- | --- | --- | --- | --- | --- |
| joint_flat_hns | single | seed | 0.151 | -0.057 | 0.071 |
| joint_flat_hns | single | base | -3.204 | -3.559 | -0.897 |
| joint_flat_hns | single | train_task | -0.424 | -1.375 | -0.317 |
| joint_flat_hns | shape_plus_strength | seed | 0.454 | -0.096 | 0.194 |
| joint_flat_hns | shape_plus_strength | base | -2.381 | -1.303 | -0.666 |
| joint_flat_hns | shape_plus_strength | train_task | -0.604 | -1.514 | -0.121 |
| hns_grid_target | single | seed | 0.149 | — | 0.163 |
| hns_grid_target | single | base | -3.051 | — | -0.743 |
| hns_grid_target | single | train_task | -0.107 | — | 0.047 |
| hns_grid_target | shape_plus_strength | seed | 0.254 | — | 0.332 |
| hns_grid_target | shape_plus_strength | base | -9.366 | — | -1.434 |
| hns_grid_target | shape_plus_strength | train_task | -1.111 | — | -0.246 |
| hns_grid_retention | single | seed | 0.196 | 0.076 | 0.146 |
| hns_grid_retention | single | base | -1.840 | -1.160 | -0.342 |
| hns_grid_retention | single | train_task | -0.355 | -4.649 | -0.404 |
| hns_grid_retention | shape_plus_strength | seed | 0.174 | 0.078 | 0.135 |
| hns_grid_retention | shape_plus_strength | base | -5.266 | -0.176 | -1.561 |
| hns_grid_retention | shape_plus_strength | train_task | -9.080 | -6.993 | -3.966 |
| scalar_common_basis_seed42 | single | base | -0.431 | -0.510 | 0.027 |
| scalar_common_basis_seed42 | single | train_task | -0.013 | -0.620 | 0.081 |
| scalar_common_basis_seed42 | shape_plus_strength | base | -0.626 | -0.457 | -0.071 |
| scalar_common_basis_seed42 | shape_plus_strength | train_task | -0.045 | -0.608 | 0.083 |

未编辑source的质量预测使用training-only Base×Task组均值加一个谱量，嵌套leave-seed；参数与功能量同池选择。

```json
{
  "sources": 18,
  "design": "nested leave-seed-out; inner leave-source-out; train-only Base\u00d7Task means + one centered feature; linear only",
  "feature_selection": {
    "functional_attn_mlp_gap": 1,
    "erank_depth_slope_normalized": 1,
    "erank_iqr": 1
  },
  "target_rmse": 1.2202919560065233,
  "target_group_mean_rmse": 1.1835818172339712,
  "target_skill_vs_group_mean": -0.06299427994356033,
  "target_r2": 0.9836346454612208,
  "forgetting_gap_rmse": 3.0974956179394613,
  "forgetting_gap_group_mean_rmse": 2.130959846181704,
  "forgetting_gap_skill_vs_group_mean": -1.1128607167590379,
  "forgetting_gap_r2": -0.6774572217317434
}
```

## 采集吞吐与旧seed42缓存核验

| Base | Task | Samples | Tokens | Batch | Seconds | Pilot batch/status/peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | magicoder | 256 | 104523 | 256 | 16.8 | 64/pass/18.7; 128/pass/22.0; 256/pass/28.5 |
| Qwen3-8B | metamath | 256 | 62944 | 256 | 11.3 | 64/pass/18.7; 128/pass/22.0; 256/pass/28.5 |
| Qwen3-8B | tulu | 256 | 82337 | 256 | 11.4 | 64/pass/18.7; 128/pass/22.0; 256/pass/28.5 |
| Llama-3.1-8B-Instruct | magicoder | 256 | 103865 | 256 | 13.4 | 64/pass/18.8; 128/pass/22.4; 256/pass/29.7 |
| Llama-3.1-8B-Instruct | metamath | 256 | 59165 | 256 | 10.3 | 64/pass/18.8; 128/pass/22.4; 256/pass/29.7 |
| Llama-3.1-8B-Instruct | tulu | 256 | 82360 | 256 | 10.4 | 64/pass/18.8; 128/pass/22.4; 256/pass/29.7 |

旧缓存与本轮重采的全部138条seed42方法记录、八量逐值对照在seed42_cache_comparison.tsv；相同协议下bf16 kernel、batch padding和累计顺序会带来数值差异，不以旧数值替换新采集。

| Feature | max |new−old| |
| --- | --- |
| functional_participation | 0.000310 |
| functional_erank | 0.000160 |
| functional_top1 | 0.000010 |
| functional_erank_iqr | 0.004690 |
| functional_attn_mlp_gap | 0.001010 |
| functional_energy_ratio | 0.000020 |
| gamma_functional | 0.000010 |
| eta_functional | 0.000010 |

## 完整逐种子 diagonal 表

| Base | Task | Seed | Method | diag PR | diag erank | cov PR | cov erank | func gap | func energy ratio | Target | Off | FG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | magicoder | 42 | original_lora | 1.045 | 1.143 | 1.036 | 1.121 | 0.157 | 1.000 | 65.244 | 77.798 | 1.900 |
| Qwen3-8B | magicoder | 42 | flat_fro | 2.423 | 4.812 | 1.997 | 3.901 | 1.606 | 0.289 | 76.829 | 81.351 | 0.000 |
| Qwen3-8B | magicoder | 42 | flat_nuclear | 2.423 | 4.812 | 1.997 | 3.901 | 1.606 | 0.139 | 75.610 | 81.002 | 0.000 |
| Qwen3-8B | magicoder | 42 | hns_f4_s1 | 2.436 | 4.878 | 2.019 | 3.936 | 1.703 | 0.141 | 75.000 | 81.333 | 0.000 |
| Qwen3-8B | metamath | 42 | original_lora | 1.315 | 1.797 | 1.246 | 1.642 | 0.798 | 1.000 | 84.155 | 71.293 | 1.799 |
| Qwen3-8B | metamath | 42 | flat_fro | 6.610 | 9.980 | 4.179 | 7.422 | 2.382 | 0.375 | 87.642 | 73.866 | 0.203 |
| Qwen3-8B | metamath | 42 | flat_nuclear | 6.610 | 9.980 | 4.179 | 7.422 | 2.382 | 0.214 | 88.173 | 75.200 | 0.000 |
| Qwen3-8B | metamath | 42 | hns_f4_s1 | 6.785 | 10.064 | 4.265 | 7.502 | 2.183 | 0.214 | 88.173 | 74.920 | 0.000 |
| Qwen3-8B | tulu | 42 | original_lora | 2.201 | 3.241 | 1.936 | 2.888 | 0.982 | 1.000 | 67.837 | 86.018 | 0.000 |
| Qwen3-8B | tulu | 42 | flat_fro | 9.904 | 12.451 | 7.058 | 10.352 | 0.735 | 0.579 | 69.316 | 85.037 | 0.000 |
| Qwen3-8B | tulu | 42 | flat_nuclear | 9.904 | 12.451 | 7.058 | 10.352 | 0.735 | 0.425 | 71.165 | 85.527 | 0.000 |
| Qwen3-8B | tulu | 42 | hns_f4_s1 | 10.025 | 12.473 | 7.165 | 10.388 | 0.744 | 0.423 | 70.795 | 85.363 | 0.000 |
| Qwen3-8B | magicoder | 43 | original_lora | 1.077 | 1.223 | 1.052 | 1.164 | 0.280 | 1.000 | 62.805 | 77.759 | 1.986 |
| Qwen3-8B | magicoder | 43 | flat_fro | 2.670 | 5.360 | 2.104 | 4.179 | 2.391 | 0.298 | 76.220 | 80.821 | 0.000 |
| Qwen3-8B | magicoder | 43 | flat_nuclear | 2.670 | 5.360 | 2.104 | 4.179 | 2.391 | 0.159 | 73.171 | 81.028 | 0.000 |
| Qwen3-8B | magicoder | 43 | hns_f4_s1 | 2.676 | 5.348 | 2.138 | 4.244 | 2.595 | 0.160 | 74.390 | 81.191 | 0.000 |
| Qwen3-8B | metamath | 43 | original_lora | 1.672 | 2.464 | 1.555 | 2.203 | 0.876 | 1.000 | 84.003 | 69.666 | 3.523 |
| Qwen3-8B | metamath | 43 | flat_fro | 7.794 | 10.917 | 5.067 | 8.373 | 2.172 | 0.397 | 87.263 | 74.881 | 0.000 |
| Qwen3-8B | metamath | 43 | flat_nuclear | 7.794 | 10.917 | 5.067 | 8.373 | 2.172 | 0.244 | 87.036 | 76.531 | 0.000 |
| Qwen3-8B | metamath | 43 | hns_f4_s1 | 7.945 | 11.012 | 5.073 | 8.443 | 1.980 | 0.244 | 86.808 | 76.650 | 0.000 |
| Qwen3-8B | tulu | 43 | original_lora | 2.611 | 3.934 | 2.332 | 3.517 | 1.096 | 1.000 | 66.174 | 85.125 | 0.000 |
| Qwen3-8B | tulu | 43 | flat_fro | 10.804 | 13.067 | 7.445 | 10.785 | 0.404 | 0.607 | 71.349 | 85.186 | 0.000 |
| Qwen3-8B | tulu | 43 | flat_nuclear | 10.804 | 13.067 | 7.445 | 10.785 | 0.404 | 0.468 | 72.274 | 85.117 | 0.000 |
| Qwen3-8B | tulu | 43 | hns_f4_s1 | 10.918 | 13.179 | 7.500 | 10.876 | 0.416 | 0.466 | 72.458 | 84.954 | 0.000 |
| Qwen3-8B | magicoder | 44 | original_lora | 1.062 | 1.186 | 1.039 | 1.131 | 0.184 | 1.000 | 64.634 | 78.553 | 1.111 |
| Qwen3-8B | magicoder | 44 | flat_fro | 2.591 | 5.289 | 2.165 | 4.289 | 1.995 | 0.298 | 72.561 | 81.024 | 0.000 |
| Qwen3-8B | magicoder | 44 | flat_nuclear | 2.591 | 5.289 | 2.165 | 4.289 | 1.995 | 0.157 | 74.390 | 80.745 | 0.000 |
| Qwen3-8B | magicoder | 44 | hns_f4_s1 | 2.644 | 5.356 | 2.157 | 4.374 | 2.275 | 0.158 | 74.390 | 81.257 | 0.000 |
| Qwen3-8B | metamath | 44 | original_lora | 1.636 | 2.428 | 1.512 | 2.207 | 0.992 | 1.000 | 84.003 | 63.515 | 9.522 |
| Qwen3-8B | metamath | 44 | flat_fro | 7.467 | 10.742 | 4.900 | 8.284 | 2.335 | 0.398 | 88.476 | 73.930 | 0.203 |
| Qwen3-8B | metamath | 44 | flat_nuclear | 7.467 | 10.742 | 4.900 | 8.284 | 2.335 | 0.244 | 87.415 | 74.750 | 0.000 |
| Qwen3-8B | metamath | 44 | hns_f4_s1 | 7.664 | 10.797 | 4.965 | 8.378 | 2.176 | 0.244 | 87.415 | 74.259 | 0.000 |
| Qwen3-8B | tulu | 44 | original_lora | 2.396 | 3.662 | 2.096 | 3.255 | 0.932 | 1.000 | 66.913 | 85.738 | 0.000 |
| Qwen3-8B | tulu | 44 | flat_fro | 10.462 | 13.047 | 7.483 | 10.699 | 0.894 | 0.601 | 69.316 | 84.071 | 0.000 |
| Qwen3-8B | tulu | 44 | flat_nuclear | 10.462 | 13.047 | 7.483 | 10.699 | 0.894 | 0.460 | 71.534 | 84.796 | 0.000 |
| Qwen3-8B | tulu | 44 | hns_f4_s1 | 10.690 | 13.150 | 7.508 | 10.785 | 1.006 | 0.458 | 70.980 | 84.476 | 0.000 |
| Llama-3.1-8B-Instruct | magicoder | 42 | original_lora | 1.255 | 1.694 | 1.221 | 1.600 | 0.873 | 1.000 | 53.659 | 60.202 | 5.039 |
| Llama-3.1-8B-Instruct | magicoder | 42 | flat_fro | 4.565 | 8.603 | 3.922 | 7.591 | 3.227 | 0.299 | 53.049 | 64.189 | 1.052 |
| Llama-3.1-8B-Instruct | magicoder | 42 | flat_nuclear | 4.565 | 8.603 | 3.922 | 7.591 | 3.227 | 0.158 | 54.878 | 63.701 | 1.540 |
| Llama-3.1-8B-Instruct | magicoder | 42 | hns_f4_s1 | 4.771 | 8.735 | 3.978 | 7.744 | 3.152 | 0.160 | 54.878 | 64.807 | 0.838 |
| Llama-3.1-8B-Instruct | metamath | 42 | original_lora | 3.437 | 5.432 | 3.105 | 5.010 | 0.596 | 1.000 | 77.255 | 57.819 | 4.713 |
| Llama-3.1-8B-Instruct | metamath | 42 | flat_fro | 11.919 | 13.887 | 9.490 | 12.336 | -0.352 | 0.640 | 79.985 | 61.139 | 2.206 |
| Llama-3.1-8B-Instruct | metamath | 42 | flat_nuclear | 11.919 | 13.887 | 9.490 | 12.336 | -0.352 | 0.546 | 80.895 | 63.241 | 1.323 |
| Llama-3.1-8B-Instruct | metamath | 42 | hns_f4_s1 | 12.034 | 13.924 | 9.650 | 12.448 | -0.279 | 0.544 | 80.819 | 63.153 | 1.817 |
| Llama-3.1-8B-Instruct | tulu | 42 | original_lora | 3.874 | 6.359 | 3.381 | 5.781 | 1.827 | 1.000 | 63.216 | 63.726 | 1.809 |
| Llama-3.1-8B-Instruct | tulu | 42 | flat_fro | 11.993 | 13.905 | 9.262 | 12.354 | 1.665 | 0.447 | 64.695 | 65.764 | 0.200 |
| Llama-3.1-8B-Instruct | tulu | 42 | flat_nuclear | 11.993 | 13.905 | 9.262 | 12.354 | 1.665 | 0.349 | 63.216 | 66.689 | 0.338 |
| Llama-3.1-8B-Instruct | tulu | 42 | hns_f4_s1 | 12.136 | 14.006 | 9.334 | 12.447 | 1.732 | 0.347 | 65.434 | 66.404 | 0.319 |
| Llama-3.1-8B-Instruct | magicoder | 43 | original_lora | 1.240 | 1.645 | 1.193 | 1.534 | 0.849 | 1.000 | 55.488 | 61.085 | 4.686 |
| Llama-3.1-8B-Instruct | magicoder | 43 | flat_fro | 4.014 | 7.616 | 3.380 | 6.668 | 3.349 | 0.304 | 54.878 | 65.361 | 2.095 |
| Llama-3.1-8B-Instruct | magicoder | 43 | flat_nuclear | 4.014 | 7.616 | 3.380 | 6.668 | 3.349 | 0.181 | 54.268 | 66.616 | 0.678 |
| Llama-3.1-8B-Instruct | magicoder | 43 | hns_f4_s1 | 4.142 | 7.863 | 3.438 | 6.779 | 3.542 | 0.182 | 56.098 | 65.938 | 1.725 |
| Llama-3.1-8B-Instruct | metamath | 43 | original_lora | 1.681 | 2.440 | 1.435 | 2.082 | 1.045 | 1.000 | 74.829 | 58.317 | 4.215 |
| Llama-3.1-8B-Instruct | metamath | 43 | flat_fro | 5.009 | 8.289 | 3.102 | 6.078 | 2.566 | 0.378 | 76.801 | 61.144 | 1.388 |
| Llama-3.1-8B-Instruct | metamath | 43 | flat_nuclear | 5.009 | 8.289 | 3.102 | 6.078 | 2.566 | 0.289 | 78.772 | 61.628 | 0.431 |
| Llama-3.1-8B-Instruct | metamath | 43 | hns_f4_s1 | 5.068 | 8.334 | 3.137 | 6.158 | 2.624 | 0.285 | 78.393 | 62.092 | 0.370 |
| Llama-3.1-8B-Instruct | tulu | 43 | original_lora | 3.544 | 5.849 | 3.192 | 5.460 | 2.469 | 1.000 | 63.216 | 67.571 | 0.000 |
| Llama-3.1-8B-Instruct | tulu | 43 | flat_fro | 11.614 | 13.709 | 9.197 | 12.233 | 1.576 | 0.449 | 64.880 | 67.940 | 0.000 |
| Llama-3.1-8B-Instruct | tulu | 43 | flat_nuclear | 11.614 | 13.709 | 9.197 | 12.233 | 1.576 | 0.349 | 64.510 | 67.738 | 0.000 |
| Llama-3.1-8B-Instruct | tulu | 43 | hns_f4_s1 | 11.743 | 13.819 | 9.246 | 12.271 | 1.425 | 0.346 | 65.250 | 67.745 | 0.000 |
| Llama-3.1-8B-Instruct | magicoder | 44 | original_lora | 1.222 | 1.597 | 1.181 | 1.514 | 0.899 | 1.000 | 53.659 | 62.164 | 4.139 |
| Llama-3.1-8B-Instruct | magicoder | 44 | flat_fro | 4.283 | 7.727 | 3.420 | 6.842 | 3.888 | 0.301 | 54.878 | 66.124 | 1.479 |
| Llama-3.1-8B-Instruct | magicoder | 44 | flat_nuclear | 4.283 | 7.727 | 3.420 | 6.842 | 3.888 | 0.176 | 55.488 | 66.300 | 1.602 |
| Llama-3.1-8B-Instruct | magicoder | 44 | hns_f4_s1 | 4.373 | 7.902 | 3.548 | 6.931 | 3.823 | 0.177 | 53.659 | 66.306 | 1.540 |
| Llama-3.1-8B-Instruct | metamath | 44 | original_lora | 1.673 | 2.442 | 1.423 | 2.022 | 0.879 | 1.000 | 74.602 | 59.581 | 3.357 |
| Llama-3.1-8B-Instruct | metamath | 44 | flat_fro | 4.705 | 8.089 | 3.192 | 6.325 | 3.093 | 0.383 | 78.696 | 62.254 | 0.887 |
| Llama-3.1-8B-Instruct | metamath | 44 | flat_nuclear | 4.705 | 8.089 | 3.192 | 6.325 | 3.093 | 0.295 | 78.393 | 61.845 | 0.616 |
| Llama-3.1-8B-Instruct | metamath | 44 | hns_f4_s1 | 4.896 | 8.244 | 3.288 | 6.394 | 3.082 | 0.290 | 79.151 | 62.551 | 0.000 |
| Llama-3.1-8B-Instruct | tulu | 44 | original_lora | 3.463 | 5.765 | 3.264 | 5.376 | 2.393 | 1.000 | 63.031 | 65.757 | 0.960 |
| Llama-3.1-8B-Instruct | tulu | 44 | flat_fro | 11.176 | 13.545 | 8.535 | 11.963 | 1.608 | 0.445 | 64.325 | 65.703 | 0.329 |
| Llama-3.1-8B-Instruct | tulu | 44 | flat_nuclear | 11.176 | 13.545 | 8.535 | 11.963 | 1.608 | 0.345 | 65.065 | 65.497 | 0.000 |
| Llama-3.1-8B-Instruct | tulu | 44 | hns_f4_s1 | 11.266 | 13.613 | 8.655 | 12.017 | 1.549 | 0.343 | 64.880 | 66.004 | 0.000 |

## 三种子 mean ± sample SD

| Base | Task | Method | diag PR | cov PR | cov erank | Target | FG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | magicoder | original_lora | 1.061 ± 0.016 | 1.042 ± 0.008 | 1.139 ± 0.022 | 64.228 ± 1.269 | 1.666 ± 0.483 |
| Qwen3-8B | magicoder | flat_fro | 2.561 ± 0.126 | 2.089 ± 0.085 | 4.123 ± 0.200 | 75.203 ± 2.308 | 0.000 ± 0.000 |
| Qwen3-8B | magicoder | flat_nuclear | 2.561 ± 0.126 | 2.089 ± 0.085 | 4.123 ± 0.200 | 74.390 ± 1.220 | 0.000 ± 0.000 |
| Qwen3-8B | magicoder | hns_f4_s1 | 2.585 ± 0.130 | 2.104 ± 0.075 | 4.185 ± 0.225 | 74.593 ± 0.352 | 0.000 ± 0.000 |
| Qwen3-8B | metamath | original_lora | 1.541 ± 0.197 | 1.438 ± 0.167 | 2.017 ± 0.325 | 84.054 ± 0.088 | 4.948 ± 4.054 |
| Qwen3-8B | metamath | flat_fro | 7.290 ± 0.611 | 4.715 ± 0.472 | 8.026 ± 0.525 | 87.794 ± 0.621 | 0.136 ± 0.117 |
| Qwen3-8B | metamath | flat_nuclear | 7.290 ± 0.611 | 4.715 ± 0.472 | 8.026 ± 0.525 | 87.541 ± 0.579 | 0.000 ± 0.000 |
| Qwen3-8B | metamath | hns_f4_s1 | 7.465 ± 0.605 | 4.768 ± 0.439 | 8.108 ± 0.526 | 87.465 ± 0.684 | 0.000 ± 0.000 |
| Qwen3-8B | tulu | original_lora | 2.403 ± 0.205 | 2.121 ± 0.199 | 3.220 ± 0.316 | 66.975 ± 0.834 | 0.000 ± 0.000 |
| Qwen3-8B | tulu | flat_fro | 10.390 ± 0.454 | 7.329 ± 0.235 | 10.612 ± 0.229 | 69.994 ± 1.174 | 0.000 ± 0.000 |
| Qwen3-8B | tulu | flat_nuclear | 10.390 ± 0.454 | 7.329 ± 0.235 | 10.612 ± 0.229 | 71.657 ± 0.565 | 0.000 ± 0.000 |
| Qwen3-8B | tulu | hns_f4_s1 | 10.544 ± 0.464 | 7.391 ± 0.196 | 10.683 ± 0.259 | 71.411 ± 0.912 | 0.000 ± 0.000 |
| Llama-3.1-8B-Instruct | magicoder | original_lora | 1.239 ± 0.016 | 1.198 ± 0.021 | 1.549 ± 0.045 | 54.268 ± 1.056 | 4.621 ± 0.454 |
| Llama-3.1-8B-Instruct | magicoder | flat_fro | 4.287 ± 0.275 | 3.574 ± 0.302 | 7.034 ± 0.490 | 54.268 ± 1.056 | 1.542 ± 0.524 |
| Llama-3.1-8B-Instruct | magicoder | flat_nuclear | 4.287 ± 0.275 | 3.574 ± 0.302 | 7.034 ± 0.490 | 54.878 ± 0.610 | 1.273 ± 0.517 |
| Llama-3.1-8B-Instruct | magicoder | hns_f4_s1 | 4.429 ± 0.318 | 3.655 ± 0.285 | 7.151 ± 0.519 | 54.878 ± 1.220 | 1.368 ± 0.468 |
| Llama-3.1-8B-Instruct | metamath | original_lora | 2.264 ± 1.016 | 1.988 ± 0.967 | 3.038 ± 1.708 | 75.562 ± 1.471 | 4.095 ± 0.686 |
| Llama-3.1-8B-Instruct | metamath | flat_fro | 7.211 ± 4.080 | 5.261 ± 3.663 | 8.246 ± 3.544 | 78.494 ± 1.602 | 1.494 ± 0.666 |
| Llama-3.1-8B-Instruct | metamath | flat_nuclear | 7.211 ± 4.080 | 5.261 ± 3.663 | 8.246 ± 3.544 | 79.353 ± 1.348 | 0.790 ± 0.471 |
| Llama-3.1-8B-Instruct | metamath | hns_f4_s1 | 7.333 ± 4.072 | 5.358 ± 3.717 | 8.333 ± 3.565 | 79.454 ± 1.241 | 0.729 ± 0.961 |
| Llama-3.1-8B-Instruct | tulu | original_lora | 3.627 ± 0.218 | 3.279 ± 0.096 | 5.539 ± 0.214 | 63.155 ± 0.107 | 0.923 ± 0.905 |
| Llama-3.1-8B-Instruct | tulu | flat_fro | 11.595 ± 0.409 | 8.998 ± 0.402 | 12.183 ± 0.200 | 64.633 ± 0.282 | 0.176 ± 0.166 |
| Llama-3.1-8B-Instruct | tulu | flat_nuclear | 11.595 ± 0.409 | 8.998 ± 0.402 | 12.183 ± 0.200 | 64.264 ± 0.949 | 0.113 ± 0.195 |
| Llama-3.1-8B-Instruct | tulu | hns_f4_s1 | 11.715 ± 0.435 | 9.078 ± 0.369 | 12.245 ± 0.217 | 65.188 ± 0.282 | 0.106 ± 0.184 |

全部12量与性能/遗忘的seed42/43/44和sample SD在three_seed_mean_sd.tsv。历史seed42训练recipe与43/44不同，两处Llama历史seed42训练标签未完全验证；新增activation不消除该训练来源限制。

## 命令与产物

```bash
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/collect_functional_activation_three_seed.py --prepare
sbatch slurm/functional_activation_three_seed.slurm
/dataset1/zailong/envs/peft-sft-lab/bin/python scripts/analyze_functional_activation_three_seed.py --stage all
```

collection_manifest.json保存原manifest解析的全部source、dataset、sample indices与哈希；activations/<Base>/<Task>/seed{42,43,44}.npz/json为18份完整moment缓存；case_audit.json保存短测试、batch与耗时。features.tsv/json保存450条完整功能量及原始成绩、路径；functional_module_metrics.tsv为67116个module条件；correlations/edited_only/source_correlations与cv_*保存全部统计；diagonal_per_seed.tsv与three_seed_mean_sd.tsv保存完整逐seed及统计表。manifest.json记录输入/输出SHA256。
