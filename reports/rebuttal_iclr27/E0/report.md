# E0: unified main-protocol results

Status: complete.

18 fixed source checkpoints, no retraining. Scores are %; differences and FG are percentage points. Off averages the three non-target benchmark families equally; FG clips each family before averaging. Commonsense-8 uses equal subtask weights.

| Method | Target | Off | FG |
| --- | ---: | ---: | ---: |
| flat_e | 72.2733 | 72.5661 | 0.3957 |
| flat_fro | 71.7311 | 72.2102 | 0.5772 |
| flat_nuclear | 72.0139 | 72.6640 | 0.3825 |
| hns_f4_s1 | 72.0834 | 72.7915 | 0.3441 |
| original_lora | 67.9980 | 69.5512 | 2.7082 |
| recon_0 | 68.4769 | 69.7658 | 2.5757 |
| scalar_e | 72.6996 | 72.4696 | 0.5247 |

| Comparison | Metric | Difference [95% CI] pp | Win/tie/loss |
| --- | --- | ---: | ---: |
| flat_e − hns_f4_s1 | target | +0.1898 [-0.2659, +0.6436] | 12/0/6 |
| flat_e − hns_f4_s1 | off_score | -0.2254 [-0.3606, -0.0967] | 5/0/13 |
| flat_e − hns_f4_s1 | forgetting_gap | +0.0516 [-0.0278, +0.1407] | 2/12/4 |
| flat_e − original_lora | target | +4.2753 [+2.7739, +5.9648] | 17/0/1 |
| flat_e − original_lora | off_score | +3.0149 [+1.7457, +4.4395] | 15/0/3 |
| flat_e − original_lora | forgetting_gap | -2.3125 [-3.4179, -1.4194] | 14/4/0 |
| flat_fro − hns_f4_s1 | target | -0.3524 [-0.8695, +0.1982] | 5/0/13 |
| flat_fro − hns_f4_s1 | off_score | -0.5813 [-0.8786, -0.3191] | 3/0/15 |
| flat_fro − hns_f4_s1 | forgetting_gap | +0.2331 [+0.0877, +0.4092] | 1/8/9 |
| flat_fro − original_lora | target | +3.7331 [+2.1549, +5.5589] | 16/0/2 |
| flat_fro − original_lora | off_score | +2.6590 [+1.5072, +3.9672] | 16/0/2 |
| flat_fro − original_lora | forgetting_gap | -2.1309 [-3.2124, -1.2886] | 14/4/0 |
| flat_nuclear − hns_f4_s1 | target | -0.0695 [-0.4306, +0.2952] | 8/4/6 |
| flat_nuclear − hns_f4_s1 | off_score | -0.1275 [-0.3359, +0.0786] | 8/0/10 |
| flat_nuclear − hns_f4_s1 | forgetting_gap | +0.0384 [-0.1131, +0.1817] | 2/11/5 |
| flat_nuclear − original_lora | target | +4.0159 [+2.5734, +5.5513] | 16/1/1 |
| flat_nuclear − original_lora | off_score | +3.1128 [+1.8373, +4.5585] | 14/0/4 |
| flat_nuclear − original_lora | forgetting_gap | -2.3257 [-3.4420, -1.4230] | 14/4/0 |
| hns_f4_s1 − original_lora | target | +4.0855 [+2.6830, +5.6421] | 16/1/1 |
| hns_f4_s1 − original_lora | off_score | +3.2403 [+1.9751, +4.6365] | 15/0/3 |
| hns_f4_s1 − original_lora | forgetting_gap | -2.3641 [-3.4756, -1.4568] | 14/4/0 |
| original_lora − hns_f4_s1 | target | -4.0855 [-5.6421, -2.6830] | 1/1/16 |
| original_lora − hns_f4_s1 | off_score | -3.2403 [-4.6365, -1.9751] | 3/0/15 |
| original_lora − hns_f4_s1 | forgetting_gap | +2.3641 [+1.4568, +3.4756] | 0/4/14 |
| recon_0 − hns_f4_s1 | target | -3.6065 [-5.3461, -2.0782] | 2/0/16 |
| recon_0 − hns_f4_s1 | off_score | -3.0258 [-4.3633, -1.7605] | 3/0/15 |
| recon_0 − hns_f4_s1 | forgetting_gap | +2.2316 [+1.3755, +3.2588] | 0/4/14 |
| recon_0 − original_lora | target | +0.4789 [+0.0279, +0.9302] | 11/1/6 |
| recon_0 − original_lora | off_score | +0.2145 [-0.0103, +0.4346] | 12/0/6 |
| recon_0 − original_lora | forgetting_gap | -0.1325 [-0.3243, +0.0647] | 9/4/5 |
| scalar_e − hns_f4_s1 | target | +0.6162 [-0.2671, +1.6097] | 10/0/8 |
| scalar_e − hns_f4_s1 | off_score | -0.3219 [-0.8224, +0.1294] | 10/0/8 |
| scalar_e − hns_f4_s1 | forgetting_gap | +0.1806 [+0.0385, +0.3482] | 3/9/6 |
| scalar_e − original_lora | target | +4.7017 [+3.0920, +6.5830] | 18/0/0 |
| scalar_e − original_lora | off_score | +2.9184 [+1.7084, +4.1563] | 15/0/3 |
| scalar_e − original_lora | forgetting_gap | -2.1835 [-3.1916, -1.3460] | 14/4/0 |

20,000 paired checkpoint bootstrap draws; all comparisons and model/task subgroups in paired_ci.tsv. These are descriptive intervals on shared test sets, not IID training-run confidence intervals. A CI containing zero is inconclusive, not evidence of equivalence. No configuration was selected by test scores.

Recon-0 calls the actual HNS builder with an identity spectrum (strength0), preserving its balanced factors, fp32 SVD and cast timing. The proposal described unbalanced factors; this report follows actual main-table code. All source scalings remain2. No optional alternate-factorization runs are claimed.

Maximum Recon-0 relative update error to source: 1.20773946e-05; to identity reconstruction: 0. All new and reused cells pass ordered per-sample input hash checks. Weight/config hashes and coverage are audited.

Slurm jobs: 14889; two allocated GPUs total, one worker per model. Actual commands, times and output paths are in commands_*.json; source/config/code locks in protocol.lock.json.

Historical Slurm992 adapters were reused byte-for-byte. Wave-change counts: {"samples": 879948, "token_changes": 0, "text_changes": 0, "correctness_changes": 0}. See wave_consistency_audit.tsv for checkpoint/subtask details.

Flat adapter audit: 8568 module/method rows; max reconstruction error 0; max budget error 5.96046448e-07. Exact historical weight/config hashes preserved; original CPU SVD/writeback recomputed only for audit.

Initial Slurm14888 failed before generation because the in-process builder retained an exclusive CUDA context. Slurm14889 isolates the builder, reuses the audited weights and keeps the evaluation protocol unchanged. See submission.json and retained logs.
