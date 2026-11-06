# E1: unified main-protocol results

Status: complete.

18 fixed source checkpoints, no retraining. Scores are %; differences and FG are percentage points. Off averages the three non-target benchmark families equally; FG clips each family before averaging. Commonsense-8 uses equal subtask weights.

| Method | Target | Off | FG |
| --- | ---: | ---: | ---: |
| flat_e | 72.2733 | 72.5661 | 0.3957 |
| flat_fro | 71.7311 | 72.2102 | 0.5772 |
| flat_nuclear | 72.0139 | 72.6640 | 0.3825 |
| hns_f4_s1 | 72.0834 | 72.7915 | 0.3441 |
| original_lora | 67.9980 | 69.5512 | 2.7082 |
| para_e90 | 68.7778 | 70.0126 | 2.3706 |
| para_e95 | 68.5823 | 69.6357 | 2.6954 |
| para_e99 | 68.4421 | 69.7776 | 2.5637 |
| recon_0 | 68.4769 | 69.7658 | 2.5757 |
| scalar_e | 72.6996 | 72.4696 | 0.5247 |
| spectral_surgery_grad_direction | 68.7637 | 68.2820 | 3.7225 |

| Comparison | Metric | Difference [95% CI] pp | Win/tie/loss |
| --- | --- | ---: | ---: |
| flat_e − original_lora | target | +4.2753 [+2.7739, +5.9648] | 17/0/1 |
| flat_e − original_lora | off_score | +3.0149 [+1.7457, +4.4395] | 15/0/3 |
| flat_e − original_lora | forgetting_gap | -2.3125 [-3.4179, -1.4194] | 14/4/0 |
| flat_e − recon_0 | target | +3.7963 [+2.1976, +5.6517] | 15/1/2 |
| flat_e − recon_0 | off_score | +2.8004 [+1.5276, +4.1735] | 14/0/4 |
| flat_e − recon_0 | forgetting_gap | -2.1800 [-3.2090, -1.3402] | 14/4/0 |
| flat_fro − original_lora | target | +3.7331 [+2.1549, +5.5589] | 16/0/2 |
| flat_fro − original_lora | off_score | +2.6590 [+1.5072, +3.9672] | 16/0/2 |
| flat_fro − original_lora | forgetting_gap | -2.1309 [-3.2124, -1.2886] | 14/4/0 |
| flat_fro − recon_0 | target | +3.2541 [+1.5974, +5.2091] | 15/1/2 |
| flat_fro − recon_0 | off_score | +2.4445 [+1.2774, +3.6998] | 14/0/4 |
| flat_fro − recon_0 | forgetting_gap | -1.9984 [-2.9992, -1.2026] | 14/4/0 |
| flat_nuclear − original_lora | target | +4.0159 [+2.5734, +5.5513] | 16/1/1 |
| flat_nuclear − original_lora | off_score | +3.1128 [+1.8373, +4.5585] | 14/0/4 |
| flat_nuclear − original_lora | forgetting_gap | -2.3257 [-3.4420, -1.4230] | 14/4/0 |
| flat_nuclear − recon_0 | target | +3.5370 [+1.9899, +5.2439] | 16/0/2 |
| flat_nuclear − recon_0 | off_score | +2.8983 [+1.6496, +4.2805] | 14/0/4 |
| flat_nuclear − recon_0 | forgetting_gap | -2.1932 [-3.2164, -1.3552] | 14/4/0 |
| hns_f4_s1 − original_lora | target | +4.0855 [+2.6830, +5.6421] | 16/1/1 |
| hns_f4_s1 − original_lora | off_score | +3.2403 [+1.9751, +4.6365] | 15/0/3 |
| hns_f4_s1 − original_lora | forgetting_gap | -2.3641 [-3.4756, -1.4568] | 14/4/0 |
| hns_f4_s1 − recon_0 | target | +3.6065 [+2.0782, +5.3461] | 16/0/2 |
| hns_f4_s1 − recon_0 | off_score | +3.0258 [+1.7605, +4.3633] | 15/0/3 |
| hns_f4_s1 − recon_0 | forgetting_gap | -2.2316 [-3.2588, -1.3755] | 14/4/0 |
| original_lora − recon_0 | target | -0.4789 [-0.9302, -0.0279] | 6/1/11 |
| original_lora − recon_0 | off_score | -0.2145 [-0.4346, +0.0103] | 6/0/12 |
| original_lora − recon_0 | forgetting_gap | +0.1325 [-0.0647, +0.3243] | 5/4/9 |
| para_e90 − original_lora | target | +0.7798 [+0.0514, +1.5457] | 11/3/4 |
| para_e90 − original_lora | off_score | +0.4614 [+0.0891, +0.8547] | 13/0/5 |
| para_e90 − original_lora | forgetting_gap | -0.3375 [-0.6228, -0.0806] | 11/4/3 |
| para_e90 − recon_0 | target | +0.3008 [-0.5297, +1.1281] | 9/0/9 |
| para_e90 − recon_0 | off_score | +0.2469 [-0.1366, +0.6293] | 12/0/6 |
| para_e90 − recon_0 | forgetting_gap | -0.2050 [-0.4246, +0.0149] | 11/4/3 |
| para_e95 − original_lora | target | +0.5844 [+0.0926, +1.0979] | 12/0/6 |
| para_e95 − original_lora | off_score | +0.0845 [-0.3014, +0.4869] | 8/0/10 |
| para_e95 − original_lora | forgetting_gap | -0.0127 [-0.3084, +0.2970] | 7/4/7 |
| para_e95 − recon_0 | target | +0.1054 [-0.4564, +0.6948] | 9/1/8 |
| para_e95 − recon_0 | off_score | -0.1300 [-0.5019, +0.2602] | 8/0/10 |
| para_e95 − recon_0 | forgetting_gap | +0.1198 [-0.1308, +0.4249] | 6/4/8 |
| para_e99 − original_lora | target | +0.4441 [+0.0933, +0.7908] | 10/1/7 |
| para_e99 − original_lora | off_score | +0.2264 [-0.0314, +0.5451] | 11/0/7 |
| para_e99 − original_lora | forgetting_gap | -0.1444 [-0.4151, +0.0818] | 9/4/5 |
| para_e99 − recon_0 | target | -0.0348 [-0.5088, +0.4822] | 6/3/9 |
| para_e99 − recon_0 | off_score | +0.0118 [-0.2838, +0.3430] | 8/0/10 |
| para_e99 − recon_0 | forgetting_gap | -0.0119 [-0.2119, +0.1661] | 5/5/8 |
| recon_0 − original_lora | target | +0.4789 [+0.0279, +0.9302] | 11/1/6 |
| recon_0 − original_lora | off_score | +0.2145 [-0.0103, +0.4346] | 12/0/6 |
| recon_0 − original_lora | forgetting_gap | -0.1325 [-0.3243, +0.0647] | 9/4/5 |
| scalar_e − original_lora | target | +4.7017 [+3.0920, +6.5830] | 18/0/0 |
| scalar_e − original_lora | off_score | +2.9184 [+1.7084, +4.1563] | 15/0/3 |
| scalar_e − original_lora | forgetting_gap | -2.1835 [-3.1916, -1.3460] | 14/4/0 |
| scalar_e − recon_0 | target | +4.2227 [+2.4904, +6.3093] | 16/0/2 |
| scalar_e − recon_0 | off_score | +2.7039 [+1.4657, +3.9197] | 15/0/3 |
| scalar_e − recon_0 | forgetting_gap | -2.0510 [-2.9859, -1.2587] | 14/4/0 |
| spectral_surgery_grad_direction − original_lora | target | +0.7657 [+0.2639, +1.3117] | 12/1/5 |
| spectral_surgery_grad_direction − original_lora | off_score | -1.2692 [-2.4241, -0.1904] | 5/0/13 |
| spectral_surgery_grad_direction − original_lora | forgetting_gap | +1.0144 [+0.0421, +2.0754] | 4/3/11 |
| spectral_surgery_grad_direction − recon_0 | target | +0.2868 [-0.2810, +0.9534] | 9/2/7 |
| spectral_surgery_grad_direction − recon_0 | off_score | -1.4838 [-2.5988, -0.4534] | 4/0/14 |
| spectral_surgery_grad_direction − recon_0 | forgetting_gap | +1.1469 [+0.2152, +2.2005] | 3/3/12 |

20,000 paired checkpoint bootstrap draws; all comparisons and model/task subgroups in paired_ci.tsv. These are descriptive intervals on shared test sets, not IID training-run confidence intervals. A CI containing zero is inconclusive, not evidence of equivalence. No configuration was selected by test scores.

Recon-0 calls the actual HNS builder with an identity spectrum (strength0), preserving its balanced factors, fp32 SVD and cast timing. The proposal described unbalanced factors; this report follows actual main-table code. All source scalings remain2. No optional alternate-factorization runs are claimed.

Maximum Recon-0 relative update error to source: 1.20773946e-05; to identity reconstruction: 0. All new and reused cells pass ordered per-sample input hash checks. Weight/config hashes and coverage are audited.

Slurm jobs: 14889; two allocated GPUs total, one worker per model. Actual commands, times and output paths are in commands_*.json; source/config/code locks in protocol.lock.json.

HNS versus Recon-0, target: higher; +3.6065 [+2.0782, +5.3461] pp.

HNS versus Recon-0, off_score: higher; +3.0258 [+1.7605, +4.3633] pp.

HNS versus Recon-0, forgetting_gap: lower; -2.2316 [-3.2588, -1.3755] pp.

Flat adapter audit: 8568 module/method rows; max reconstruction error 0; max budget error 5.96046448e-07. Exact historical weight/config hashes preserved; original CPU SVD/writeback recomputed only for audit.

Initial Slurm14888 failed before generation because the in-process builder retained an exclusive CUDA context. Slurm14889 isolates the builder, reuses the audited weights and keeps the evaluation protocol unchanged. See submission.json and retained logs.
