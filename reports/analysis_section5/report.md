# Paper Section 5 — Analysis data and audit

状态：complete；Scalar-F也已完成。

## 1. 来源、复用范围与协议

主要来源是作业1070（2026-09-14 21:22完成，exit0）及 `/dataset1/zailong/workspace/peft-sft-lab/reports/hns_energy_matched_20260914`。18个源checkpoint全部保留：Qwen3-8B/Llama-3.1-8B-Instruct × magicoder/metamath/tulu × seed42/43/44。未重新训练。当前分支refactor/sft-chat-template；其他未提交训练数据修改未纳入。

复用之前核验源权重/config、HNS权重/metadata、activation NPZ、旧最终报告与56个输出、评测代码版本；对所有296个必要cache cells逐样本原始输入内容和顺序计算哈希并与同Base/benchmark的Base比较。结果pass，样本数HumanEval164/GSM8K1319/IFEval541/CS22419一致。来源及真实resolve路径见evaluation_sources.tsv；源checkpoint见source_checkpoints.json；代码SHA见provenance.json。

最终统一协议来自functional_hns_three_seed_20260914；使用其最终Llama重评缓存，不混入旧pilot。long max_num_seqs2048，CS4096；adapter block5，maxrank16，max_model_len4096，max_batched_tokens65536，greedy seed42，原task-config/prompt/chat/parser/评分实现不变。1070 live兼容性短测Qwen覆盖42/43/44，Llama覆盖43/44；Llama42最终缓存做完整性核验，不将旧pilot短测当最终成绩。

操作分类：同能量成绩/FPR原始表、相关性、LOCO设计矩阵及预测全部复用，不新增推理；逐module解析能量核验、八类转换、更多粒度的配对CI为CPU重算；Scalar-F属于唯一新增评测（72 cells），不重复已有方法推理。详见provenance.json和scalar_f_evaluation_protocol.json。

## 2. 方法定义和同能量结果

固定源SVD方向，q_i=E[(v_iᵀh)²]来自冻结Base cache的原始二阶矩，无floor。令EH=Σt_H²q；Scalar-E=σ√(EH/Σσ²q)，Flat-E每方向=√(EH/Σq)。原LoRA scaling c保持，实际能量是c²EH；controls匹配后不恢复nuclear norm。HNS固定全module4+1、strength1且保持原nuclear budget。

Implementation audit：旧Scalar/common_per_module/实际HNS版ScalarShrink匹配Frobenius，不等价于Scalar-E；理想ExactFlat版ScalarShrink也不等价。对这18个checkpoint，Frobenius匹配的逐module functional-energy/HNS比值为min 0.294708、median 2.883298、max 14.773126，详见old_scalar_energy_audit.tsv。因此不能将旧Scalar成绩当作同能量对照。作业1070已有严格逐module Scalar-E/Flat-E，全部直接复用，不重复评测。

独立核验8568个control/module行：最大解析相对能量误差=8.882e-16，最大保存后相对误差=7.519e-06（门限1e-5）。module_energy_audit.tsv记录误差、shape spread、源方向hash和核范数比；保存误差来自1070经过权重hash核验的factor审计，不冒称本次重新GPU SVD。

| Method | n | Target % | Off % | FG pp |
| --- | --- | --- | --- | --- |
| original_lora | 18 | 67.9980 | 69.5512 | 2.7082 |
| hns_f4_s1 | 18 | 72.0834 | 72.7915 | 0.3441 |
| scalar_e | 18 | 72.6996 | 72.4696 | 0.5247 |
| flat_e | 18 | 72.2733 | 72.5661 | 0.3957 |
| scalar_f | 18 | 70.6430 | 71.5579 | 1.1111 |

逐checkpoint完整表energy_matched_checkpoint_results.tsv；Base/source分组energy_matched_grouped_results.tsv。Off为另外三family等权；CS为八子benchmark等权。FG为其他三family max(Base−edited,0)等权；FG截断不替代Target/Off分析。

| Comparison | Outcome | Mean pp | 95% CI low | 95% CI high |
| --- | --- | --- | --- | --- |
| hns_f4_s1 minus scalar_e | target | -0.6162 | -1.6399 | 0.2488 |
| hns_f4_s1 minus scalar_e | off_score | 0.3219 | -0.1246 | 0.8284 |
| hns_f4_s1 minus scalar_e | forgetting_gap | -0.1806 | -0.3461 | -0.0371 |
| hns_f4_s1 minus flat_e | target | -0.1898 | -0.6158 | 0.2593 |
| hns_f4_s1 minus flat_e | off_score | 0.2254 | 0.1037 | 0.3564 |
| hns_f4_s1 minus flat_e | forgetting_gap | -0.0516 | -0.1381 | 0.0252 |
| scalar_e minus flat_e | target | 0.4264 | -0.5006 | 1.3940 |
| scalar_e minus flat_e | off_score | -0.0965 | -0.5589 | 0.3796 |
| scalar_e minus flat_e | forgetting_gap | 0.1290 | 0.0224 | 0.2698 |
完整逐checkpoint方法差值及按Base/source/base×source的配对CI见all_energy_method_paired_checkpoint_differences.tsv和all_energy_method_paired_ci.tsv。分组区间未校正多重比较，只作预先固定分组的描述，不用于选择有利结论。

逐checkpoint同能量成绩（完整18×3；单位为百分数/pp）：

| Checkpoint | Method | Target | Off | FG |
| --- | --- | --- | --- | --- |
| Qwen3-8B/magicoder/seed42 | hns_f4_s1 | 75.0000 | 81.3332 | 0 |
| Qwen3-8B/magicoder/seed43 | hns_f4_s1 | 74.3902 | 81.1914 | 0 |
| Qwen3-8B/magicoder/seed44 | hns_f4_s1 | 74.3902 | 81.2572 | 0 |
| Qwen3-8B/metamath/seed42 | hns_f4_s1 | 88.1729 | 74.9204 | 0 |
| Qwen3-8B/metamath/seed43 | hns_f4_s1 | 86.8082 | 76.6504 | 0 |
| Qwen3-8B/metamath/seed44 | hns_f4_s1 | 87.4147 | 74.2593 | 0 |
| Qwen3-8B/tulu/seed42 | hns_f4_s1 | 70.7948 | 85.3627 | 0 |
| Qwen3-8B/tulu/seed43 | hns_f4_s1 | 72.4584 | 84.9537 | 0 |
| Qwen3-8B/tulu/seed44 | hns_f4_s1 | 70.9797 | 84.4759 | 0 |
| Llama-3.1-8B-Instruct/magicoder/seed42 | hns_f4_s1 | 54.8780 | 64.7929 | 0.9097 |
| Llama-3.1-8B-Instruct/magicoder/seed43 | hns_f4_s1 | 55.4878 | 66.0280 | 1.6020 |
| Llama-3.1-8B-Instruct/magicoder/seed44 | hns_f4_s1 | 53.6585 | 66.5840 | 1.3555 |
| Llama-3.1-8B-Instruct/metamath/seed42 | hns_f4_s1 | 80.6672 | 63.3380 | 1.6901 |
| Llama-3.1-8B-Instruct/metamath/seed43 | hns_f4_s1 | 78.3927 | 62.2925 | 0.2465 |
| Llama-3.1-8B-Instruct/metamath/seed44 | hns_f4_s1 | 78.9992 | 62.4981 | 0.0616 |
| Llama-3.1-8B-Instruct/tulu/seed42 | hns_f4_s1 | 64.8799 | 66.4154 | 0.3284 |
| Llama-3.1-8B-Instruct/tulu/seed43 | hns_f4_s1 | 65.2495 | 67.6859 | 0 |
| Llama-3.1-8B-Instruct/tulu/seed44 | hns_f4_s1 | 64.8799 | 66.2083 | 0 |
| Qwen3-8B/magicoder/seed42 | scalar_e | 75.6098 | 81.6482 | 0.0000 |
| Qwen3-8B/metamath/seed42 | scalar_e | 86.9598 | 73.3800 | 0.4065 |
| Qwen3-8B/tulu/seed42 | scalar_e | 73.0129 | 84.1305 | 0.0000 |
| Qwen3-8B/magicoder/seed43 | scalar_e | 76.8293 | 80.8311 | 0.0000 |
| Qwen3-8B/metamath/seed43 | scalar_e | 86.1259 | 74.3887 | 0.0000 |
| Qwen3-8B/tulu/seed43 | scalar_e | 71.7190 | 85.4800 | 0.0000 |
| Qwen3-8B/magicoder/seed44 | scalar_e | 77.4390 | 81.6134 | 0.0000 |
| Qwen3-8B/metamath/seed44 | scalar_e | 85.4435 | 72.8677 | 1.0163 |
| Qwen3-8B/tulu/seed44 | scalar_e | 71.5342 | 83.9158 | 0.0000 |
| Qwen3-8B/magicoder/seed42 | flat_e | 75.6098 | 81.4047 | 0.0000 |
| Qwen3-8B/metamath/seed42 | flat_e | 88.2487 | 74.1715 | 0.0000 |
| Qwen3-8B/tulu/seed42 | flat_e | 69.8706 | 85.3077 | 0.0000 |
| Qwen3-8B/magicoder/seed43 | flat_e | 75.0000 | 80.8352 | 0.0000 |
| Qwen3-8B/metamath/seed43 | flat_e | 87.3389 | 76.5332 | 0.0000 |
| Qwen3-8B/tulu/seed43 | flat_e | 73.7523 | 84.8238 | 0.0000 |
| Qwen3-8B/magicoder/seed44 | flat_e | 75.0000 | 81.2920 | 0.0000 |
| Qwen3-8B/metamath/seed44 | flat_e | 87.6422 | 74.4767 | 0.0000 |
| Qwen3-8B/tulu/seed44 | flat_e | 71.3494 | 84.1313 | 0.0000 |
| Llama-3.1-8B-Instruct/magicoder/seed42 | scalar_e | 56.0976 | 65.1533 | 0.9031 |
| Llama-3.1-8B-Instruct/metamath/seed42 | scalar_e | 78.6202 | 60.7657 | 2.6364 |
| Llama-3.1-8B-Instruct/tulu/seed42 | scalar_e | 66.3586 | 66.4628 | 0.6843 |
| Llama-3.1-8B-Instruct/magicoder/seed43 | scalar_e | 57.9268 | 66.1368 | 1.4787 |
| Llama-3.1-8B-Instruct/metamath/seed43 | scalar_e | 78.7718 | 62.8080 | 0.5941 |
| Llama-3.1-8B-Instruct/tulu/seed43 | scalar_e | 63.5860 | 67.2912 | 0.0000 |
| Llama-3.1-8B-Instruct/magicoder/seed44 | scalar_e | 59.7561 | 67.0538 | 1.1707 |
| Llama-3.1-8B-Instruct/metamath/seed44 | scalar_e | 78.8476 | 63.0510 | 0.5543 |
| Llama-3.1-8B-Instruct/tulu/seed44 | scalar_e | 63.9556 | 67.4753 | 0.0000 |
| Llama-3.1-8B-Instruct/magicoder/seed42 | flat_e | 53.0488 | 64.3678 | 1.0567 |
| Llama-3.1-8B-Instruct/metamath/seed42 | flat_e | 80.9704 | 62.5913 | 2.2335 |
| Llama-3.1-8B-Instruct/tulu/seed42 | flat_e | 64.6950 | 65.9905 | 0.3205 |
| Llama-3.1-8B-Instruct/magicoder/seed43 | flat_e | 57.9268 | 66.1056 | 1.2323 |
| Llama-3.1-8B-Instruct/metamath/seed43 | flat_e | 78.0136 | 61.8467 | 0.4929 |
| Llama-3.1-8B-Instruct/tulu/seed43 | flat_e | 63.5860 | 67.8232 | 0.0000 |
| Llama-3.1-8B-Instruct/magicoder/seed44 | flat_e | 54.8780 | 66.4998 | 1.3555 |
| Llama-3.1-8B-Instruct/metamath/seed44 | flat_e | 78.9234 | 61.9401 | 0.4313 |
| Llama-3.1-8B-Instruct/tulu/seed44 | flat_e | 65.0647 | 66.0492 | 0.0000 |

## 3. FPR增量解释力

固定Raw FPR/module median、Full-moment PR/module median及Energy（逐module sum，predictor=log(edited/source energy)）。A LoRA+全部7版本126 arms；B edited-only108；C HNS/Scalar-E/Flat-E54；每层18个source checkpoint。Scalar-F是新增机制baseline，未事后加到既有FPR cohort以改变预先固定分析。

指标定义固定：e_i=s_i²q_i，Raw FPR=(Σe_i)²/Σe_i²；M=E[(Vᵀh)(Vᵀh)ᵀ]，G=diag(s)Mdiag(s)，Full-moment PR=tr(G)²/||G||F²；Energy=c²Σe_i。两种PR对公共谱缩放不变；Full-moment保留坐标交叉二阶矩，不重新选择指标或module聚合。

每个checkpoint内中心化特征和outcome，OLS；LOCO整组留出一个源checkpoint，该源所有版本均不进入训练。预测的是留出checkpoint内相对偏差，不是未知checkpoint的绝对性能；基线为该checkpoint内零偏差（该中心由测试组定义，故属于条件性的组内排序任务，不是无标签部署预测）。测试组中心化和全表标准差是既有协议，线性无正则OLS的标准差缩放不改变预测；没有把测试变体用于训练拟合或挑指标。Spearman逐checkpoint后等权汇总；C Energy解析设为完全相等，秩0，不利用存储舍入拟合。

| Layer | FPR | Outcome | Δin-sample R² | ΔLOCO R² | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- | --- |
| A_all_versions | raw_fpr | target_gain | 0.0884 | 0.0201 | -0.2344 | 0.1290 |
| A_all_versions | full_moment_pr | target_gain | 0.1115 | 0.0477 | -0.2156 | 0.1657 |
| A_all_versions | raw_fpr | off_gain | 0.0480 | 0.0327 | -0.0446 | 0.1432 |
| A_all_versions | full_moment_pr | off_gain | 0.0692 | 0.0559 | -0.0387 | 0.1976 |
| B_edited_only | raw_fpr | target_gain | 0.0224 | 0.0155 | -0.0439 | 0.0865 |
| B_edited_only | full_moment_pr | target_gain | 0.0106 | -0.0022 | -0.0496 | 0.0423 |
| B_edited_only | raw_fpr | off_gain | 0.0815 | 0.0092 | -0.2204 | 0.1729 |
| B_edited_only | full_moment_pr | off_gain | 0.0279 | -0.0402 | -0.1834 | 0.0563 |
| C_energy_matched | raw_fpr | target_gain | 0.0004 | -0.0767 | -0.1649 | -0.0363 |
| C_energy_matched | full_moment_pr | target_gain | 0.0019 | -0.0826 | -0.1981 | -0.0300 |
| C_energy_matched | raw_fpr | off_gain | 0.1063 | -0.0319 | -0.5470 | 0.1992 |
| C_energy_matched | full_moment_pr | off_gain | 0.0963 | -0.0426 | -0.5517 | 0.1733 |

完整Energy-only/FPR-only/Energy+FPR结果与所有四family/八CS分项见fpr_statistics.tsv/json；逐checkpoint相关性、设计矩阵、留出预测和raw_results.tsv/json均附。fpr_fold_membership.tsv显式记录54个整checkpoint留出fold；fpr_leaveout_audit.json独立用仅训练数据的标准差复核全部LOCO预测，source重叠为0。FPR的稳定额外留出解释力证据不足，尤其同能量Target/Off的Raw和Full-moment留出R²均为负；不能由训练内R²增加推断泛化收益。

## 4. 八类逐题转换与Recovery/Retention

n_BLE按Base、LoRA、Edited三位正确性表示，000/001/010/011/100/101/110/111全部存储。Recovery=n101/(n100+n101)，Retention=n011/(n010+n011)；New Success=n001，New Damage=n110。Retention不是Off accuracy。

分母为对应机会集合；零分母为NA而不是0。macro先八CS等权→四family等权→18checkpoint等权；target/off分别保存。Counts跨checkpoint求和是checkpoint×sample事件数，不是唯一题目数；宏率不是这些总count的比值。补充new_success_rate/new_damage_rate均除以该benchmark全部样本数，表示prevalence，不是条件机会率。每一级defined/coverage均保存。旧统一协议中retention一词曾指Off性能，这里明确只用条件逐题Retention，不继承那一词义。

| Method | Recovery % | Retention % | New Success count | New Damage count | Recovery n | Retention n |
| --- | --- | --- | --- | --- | --- | --- |
| hns_f4_s1 | 62.3514 | 62.2881 | 4553 | 3163 | 27973 | 22345 |
| scalar_e | 62.8758 | 63.2198 | 3893 | 3462 | 27973 | 22345 |
| flat_e | 62.4302 | 62.4637 | 4487 | 3200 | 27973 | 22345 |

固定macro的配对source-checkpoint CI（差值为百分点；逐benchmark区间另附TSV）：

| Comparison | Metric | Mean pp | 95% low | 95% high |
| --- | --- | --- | --- | --- |
| hns_f4_s1 minus scalar_e | recovery_rate | -0.5244 | -2.5873 | 1.7698 |
| hns_f4_s1 minus scalar_e | retention_rate | -0.9317 | -2.6571 | 0.9693 |
| hns_f4_s1 minus flat_e | recovery_rate | -0.0788 | -0.6267 | 0.5080 |
| hns_f4_s1 minus flat_e | retention_rate | -0.1756 | -0.8208 | 0.4791 |
| scalar_e minus flat_e | recovery_rate | 0.4456 | -1.8100 | 2.3324 |
| scalar_e minus flat_e | retention_rate | 0.7561 | -1.0625 | 2.2918 |

逐benchmark宏汇总（其八类counts均在同名TSV）：

| Benchmark | Method | Recovery % | Retention % | NS | ND | Recovery n | Retention n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark/magicoder | hns_f4_s1 | 57.9488 | 68.2708 | 70 | 44 | 379 | 468 |
| benchmark/magicoder | scalar_e | 58.1403 | 69.6314 | 67 | 40 | 379 | 468 |
| benchmark/magicoder | flat_e | 58.4759 | 67.1680 | 71 | 46 | 379 | 468 |
| benchmark/metamath | hns_f4_s1 | 62.9831 | 70.7371 | 693 | 487 | 2141 | 2532 |
| benchmark/metamath | scalar_e | 64.9782 | 71.3237 | 691 | 529 | 2141 | 2532 |
| benchmark/metamath | flat_e | 62.2292 | 71.7503 | 653 | 478 | 2141 | 2532 |
| benchmark/tulu | hns_f4_s1 | 67.2095 | 55.3193 | 226 | 218 | 1089 | 708 |
| benchmark/tulu | scalar_e | 69.1293 | 55.2110 | 216 | 244 | 1089 | 708 |
| benchmark/tulu | flat_e | 67.7935 | 55.6855 | 224 | 246 | 1089 | 708 |
| benchmark/arc_challenge | hns_f4_s1 | 60.4183 | 68.0366 | 204 | 151 | 1333 | 1133 |
| benchmark/arc_challenge | scalar_e | 57.8715 | 67.8697 | 167 | 182 | 1333 | 1133 |
| benchmark/arc_challenge | flat_e | 60.2940 | 68.4172 | 208 | 153 | 1333 | 1133 |
| benchmark/arc_easy | hns_f4_s1 | 61.5172 | 69.1322 | 155 | 123 | 1545 | 1327 |
| benchmark/arc_easy | scalar_e | 60.2433 | 68.6915 | 132 | 156 | 1545 | 1327 |
| benchmark/arc_easy | flat_e | 60.8895 | 68.8478 | 154 | 129 | 1545 | 1327 |
| benchmark/boolq | hns_f4_s1 | 73.3635 | 37.0115 | 198 | 303 | 1874 | 1524 |
| benchmark/boolq | scalar_e | 71.2211 | 40.2441 | 194 | 278 | 1874 | 1524 |
| benchmark/boolq | flat_e | 72.4838 | 37.7407 | 195 | 298 | 1874 | 1524 |
| benchmark/hellaswag | hns_f4_s1 | 57.8392 | 50.2692 | 2474 | 1337 | 12234 | 7570 |
| benchmark/hellaswag | scalar_e | 54.6599 | 50.7927 | 1870 | 1522 | 12234 | 7570 |
| benchmark/hellaswag | flat_e | 57.5125 | 50.2474 | 2434 | 1335 | 12234 | 7570 |
| benchmark/openbookqa | hns_f4_s1 | 62.1140 | 60.5033 | 93 | 47 | 312 | 500 |
| benchmark/openbookqa | scalar_e | 61.6971 | 62.8955 | 94 | 42 | 312 | 500 |
| benchmark/openbookqa | flat_e | 62.9676 | 61.6611 | 87 | 43 | 312 | 500 |
| benchmark/piqa | hns_f4_s1 | 64.6126 | 50.6373 | 101 | 129 | 2075 | 1746 |
| benchmark/piqa | scalar_e | 61.0423 | 53.5546 | 92 | 108 | 2075 | 1746 |
| benchmark/piqa | flat_e | 64.6805 | 51.0954 | 107 | 139 | 2075 | 1746 |
| benchmark/siqa | hns_f4_s1 | 58.6765 | 45.6172 | 157 | 187 | 1736 | 1857 |
| benchmark/siqa | scalar_e | 58.2562 | 48.9738 | 140 | 158 | 1736 | 1857 |
| benchmark/siqa | flat_e | 58.8681 | 45.8116 | 167 | 198 | 1736 | 1857 |
| benchmark/winogrande | hns_f4_s1 | 51.5713 | 57.3935 | 182 | 137 | 3255 | 2980 |
| benchmark/winogrande | scalar_e | 49.0529 | 60.6821 | 230 | 203 | 3255 | 2980 |
| benchmark/winogrande | flat_e | 52.0815 | 58.1862 | 187 | 135 | 3255 | 2980 |

八类计数：transitions8_checkpoint_benchmark.tsv（594行）、transitions8_checkpoint_family.tsv、transitions8_checkpoint_macro.tsv、transitions8_summary.tsv。所有方法对的逐benchmark/overall checkpoint bootstrap CI见recovery_paired_checkpoint_ci.tsv；逐checkpoint/benchmark及family、target/off宏汇总的paired sample CI见recovery_paired_sample_ci.tsv。scatter输入recovery_retention_scatter_data.tsv复用。

统计不确定性分开报告：既有2000次source-checkpoint paired bootstrap（seed20260914）描述固定测试集上的run异质性，不能视模块/token/多个编辑版本为独立重复；部分Llama seed42训练recipe provenance有保留，因此不称严格iid重复。新增conditional test-sample bootstrap每个checkpoint/benchmark联合抽样B,L,H,S,F的32态计数，方法共享每次抽样并重算分母；空集合draw丢弃并记coverage。跨benchmark分层独立抽样后按固定macro组合。因为多个checkpoint共用同一题目，不提供把checkpoint样本独立拼池的测试样本CI，也不将两类区间混为一种总体不确定性。

## 5. Scalar-F审计与补齐

src/finetune/spectral_edit/mechanism.py和旧common_per_module匹配观测HNS；src/finetune/spectral_edit/ablations.py的ScalarShrink匹配理想ExactFlatNuclear，不能混为本项。旧common_per_module在本次训练任务范围仅六个seed42源checkpoint且是历史评测wave；已有权重/预测实际路径已追溯，见scalar_f_prior_results_audit.json，未因文件名不同重跑符合协议的结果。当前最终统一wave确无完整Scalar-F，因此补齐18个checkpoint。

Scalar-F使用各module真实HNS4+1 metadata中的t_H，γF=||t_H||₂/||σ||₂，保持原谱比例、source方向和scaling；不恢复nuclear norm。与Scalar-E使用相同balanced reconstruction与存储审计；它不匹配functional energy。大权重/预测仅在ignored runtime/，轻量误差表scalar_f_module_audit.tsv和四方法成绩/配对CI提交。

Scalar-F新增作业1074已完成72个cells/18个checkpoint；全部输入内容/顺序与最终Base缓存一致。4284个module最大保存后Frobenius相对误差3.054e-06。它的逐module functional-energy/HNS比值min 0.294708、median 2.883298、max 14.773126，再次明确Frobenius matching不等于Energy matching。四方法成绩、完整八项CS分数及配对CI见four_method_checkpoint_results.tsv / four_method_paired_ci.tsv。

| Scalar-F comparison | Outcome | Mean pp | CI low | CI high |
| --- | --- | --- | --- | --- |
| hns_f4_s1 minus scalar_f | target | 1.4405 | 0.3529 | 2.4396 |
| hns_f4_s1 minus scalar_f | off_score | 1.2336 | 0.5579 | 1.9086 |
| hns_f4_s1 minus scalar_f | forgetting_gap | -0.7670 | -1.3169 | -0.3070 |
| scalar_e minus scalar_f | target | 2.0567 | 1.0458 | 3.2149 |
| scalar_e minus scalar_f | off_score | 0.9117 | 0.3728 | 1.4125 |
| scalar_e minus scalar_f | forgetting_gap | -0.5864 | -1.0281 | -0.2030 |
| flat_e minus scalar_f | target | 1.6303 | 0.5176 | 2.6904 |
| flat_e minus scalar_f | off_score | 1.0082 | 0.3501 | 1.6553 |
| flat_e minus scalar_f | forgetting_gap | -0.7153 | -1.2208 | -0.2972 |

现有Functional-HNS边界数据（全部复用，未搜索alpha或新增版本；PR为checkpoint内module median后18checkpoint等权）：

| Method | n | Raw FPR | Full-moment PR | Energy/source | Target % | Off % |
| --- | --- | --- | --- | --- | --- | --- |
| hns_f4_s1 | 18 | 7.3451 | 5.3923 | 0.0972 | 72.0834 | 72.7915 |
| functional_hns_a05 | 18 | 12.7281 | 8.8312 | 0.0636 | 72.1080 | 72.5863 |
| functional_hns_a10 | 18 | 15.9820 | 12.1128 | 0.0536 | 71.5957 | 72.2840 |
| functional_flat | 18 | 16.0000 | 12.1303 | 0.0535 | 71.3217 | 72.3405 |

## 6. 客观结论 / Paper-ready summary

1. 同能量下HNS−Scalar-E Target −0.616pp、Off +0.322pp，本目录checkpoint CI均跨0；不能声称HNS全面明显优于逐模块缩放。FG差−0.181pp，CI不跨0，但受截断定义限制。
2. HNS与Flat-E接近；HNS Off +0.225pp（本目录CI [+0.104,+0.356]），Target差区间跨0；不显著不等于等效，也不能声称形状完全无效。
3. Scalar-E在PR完全不变下仍有平均Target +4.702pp/Off +2.918pp vs LoRA，说明PR不能独自编码幅度效应；不能据此给出缩放对收益的因果归因比例。
4. 三层固定分析中FPR稳定附加解释力证据不足；控制能量后Raw/Full-moment PR都未产生正的Target/Off留出R²。
5. HNS同时恢复部分旧能力并保留部分新能力（Recovery62.351%、Retention62.288%），不是完全回到Base；但Recovery/Retention均未证明优于Scalar-E。
6. 现有Functional-HNS/Functional-Flat的54配对中，更高Raw FPR但Target更差22例、Off更差37例；不支持最大化FPR必然提升adapter，也不把能量同时变化的比较解释为FPR因果效应。

7. 新补Scalar-F的18个checkpoint上，HNS−Scalar-F Target +1.440pp（CI [+0.353,+2.440]），Off +1.234pp（CI [+0.558,+1.909]）。这是同Frobenius而不是同functional energy比较，不能直接解释为单独谱形状的因果贡献。

能量匹配限定于固定Base calibration输入轨迹，不保证edited模型所有真实输入下输出变化强度完全一样。Scalar-F作为补充baseline结果原样附表，不修改上述固定分析规则。

## 7. 复现与文件索引

CPU：`PYTHONPATH=src:scripts /dataset1/zailong/envs/peft-sft-lab/bin/python scripts/complete_analysis_section5.py --stage cpu`。Scalar-F：`sbatch slurm/analysis_section5_scalar_f.slurm`（最多2 B300，两个Base并行）。完成导出：同脚本`--stage finalize`。只重生成报告可用`--stage report`。独立复核FPR留出：`--stage fpr_audit`，原始cache不在时自动使用本目录归档的轻量raw_results和regression_predictions，不需要activation/权重/完整预测。

复用FPR重新计算的原始入口是`PYTHONPATH=src:scripts .../python scripts/analyze_hns_energy_matched.py --stage fpr`（必须已有两种control完成标志）；本次没有重算或搜索指标。所有路径/哈希见provenance.json、source_checkpoints.json、evaluation_sources.tsv。方法和bootstrap实现随脚本提交，测试tests/test_analysis_section5.py。
