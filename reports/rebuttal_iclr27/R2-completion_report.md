# Second-round rebuttal completion

All A/B experiments passed output, coverage and integrity audits. Optional stages and pending author confirmation are listed explicitly.

New training runs: 17; new cells: 952; global GPU limit:2.

| Experiment | Status | New cells | Report |
| --- | --- | ---: | --- |
| R2-O1 | complete | 176 | [R2-O1](R2-O1/report.md) |
| R2-O2 | complete | 144 | [R2-O2](R2-O2/report.md) |
| R2-O3 | complete | 72 | [R2-O3](R2-O3/report.md) |
| R2-O4 | complete | 216 | [R2-O4](R2-O4/report.md) |
| R2-O5 | complete | 64 | [R2-O5](R2-O5/report.md) |
| R2-O6 | complete | 32 | [R2-O6](R2-O6/report.md) |
| R2-O7 | complete | 40 | [R2-O7](R2-O7/report.md) |
| R2-O8 | complete | 32 | [R2-O8](R2-O8/report.md) |
| R2-O9 | complete | 32 | [R2-O9](R2-O9/report.md) |
| R2-O10 | complete | 20 | [R2-O10](R2-O10/report.md) |
| R2-O11 | complete | 72 | [R2-O11](R2-O11/report.md) |
| R2-O12 | complete | 52 | [R2-O12](R2-O12/report.md) |

All methods/configurations are reported; none were chosen using main or auxiliary test performance. Main Target/Off/FG preserve the functional_hns_three_seed_20260914 protocol and caches after sample-input-hash audits. Checkpoint CIs are descriptive shared-test intervals; single-seed conditions use paired question intervals and exact McNemar repair/damage. See each report for inference/capacity exceptions (rank32/64) and any limits on interpretation.

Balanced/unbalanced represented-update audit: 8568 module pairs, maximum relative difference1.749179e-07, fixed tolerance5e-5. [Audit](R2-O2/factorization_update_audit.json).

## Aggregate values

| Experiment | Method | Target | Off | FG |
| --- | --- | ---: | ---: | ---: |
| R2-O1 | base | 77.4825 | 75.6962 | 0.0000 |
| R2-O1 | flat_nuclear | 82.1354 | 77.5537 | 0.0510 |
| R2-O1 | hns_f4_s1 | 82.1241 | 77.4941 | 0.1016 |
| R2-O1 | original_lora | 76.3554 | 72.9826 | 3.0721 |
| R2-O1 | recon_0 | 76.6497 | 73.0933 | 2.9408 |
| R2-O2 | base | 66.4730 | 69.9931 | 0.0000 |
| R2-O2 | hns_f4_s1 | 72.0834 | 72.7915 | 0.3441 |
| R2-O2 | hns_unbal | 72.1878 | 72.6307 | 0.3974 |
| R2-O2 | original_lora | 67.9980 | 69.5512 | 2.7082 |
| R2-O2 | recon_0 | 68.4769 | 69.7658 | 2.5757 |
| R2-O2 | recon_0_unbal | 68.4554 | 69.6285 | 2.6876 |
| R2-O3 | base | 66.4730 | 69.9931 | 0.0000 |
| R2-O3 | hns_f4_s1 | 72.0834 | 72.7915 | 0.3441 |
| R2-O3 | hns_od | 69.9092 | 70.9472 | 1.5098 |
| R2-O3 | original_lora | 67.9980 | 69.5512 | 2.7082 |
| R2-O3 | recon_0 | 68.4769 | 69.7658 | 2.5757 |
| R2-O3 | spectral_surgery_grad_direction | 68.7637 | 68.2820 | 3.7225 |
| R2-O3 | ss_grad_direction_all | 67.2777 | 65.9738 | 5.8911 |
| R2-O3 | ss_smooth_abs | 68.4773 | 69.4657 | 2.7952 |
| R2-O3 | ss_smooth_abs_all | 68.2792 | 68.8028 | 3.3166 |
| R2-O4 | base | 66.4730 | 69.9931 | 0.0000 |
| R2-O4 | hns_f0_s5 | 72.0670 | 72.6598 | 0.3409 |
| R2-O4 | hns_f4_s0 | 72.2719 | 72.6070 | 0.4325 |
| R2-O4 | hns_f4_s1 | 72.0834 | 72.7915 | 0.3441 |
| R2-O4 | hns_f4_s1_l05 | 70.6675 | 71.7815 | 0.9786 |
| R2-O4 | original_lora | 67.9980 | 69.5512 | 2.7082 |
| R2-O4 | recon_0 | 68.4769 | 69.7658 | 2.5757 |
| R2-O5 | base | 76.1051 | 76.1553 | 0.0000 |
| R2-O5 | flat_nuclear | 80.9393 | 77.3238 | 0.1016 |
| R2-O5 | hns_f4_s1 | 80.6976 | 77.2503 | 0.1016 |
| R2-O5 | original_lora | 74.6466 | 73.0425 | 3.3019 |
| R2-O5 | recon_0 | 74.8756 | 73.3691 | 2.9827 |
| R2-O6 | base | 85.7468 | 72.9414 | 0.0000 |
| R2-O6 | flat_nuclear | 88.1539 | 74.4500 | 0.0462 |
| R2-O6 | hns_f4_s1 | 88.4951 | 74.6405 | 0.0000 |
| R2-O6 | original_lora | 84.7801 | 65.8461 | 7.1105 |
| R2-O6 | recon_0 | 84.6664 | 66.1027 | 6.8554 |
| R2-O7 | base | 34.0788 | 65.9015 | 0.0000 |
| R2-O7 | flat_nuclear | 45.6406 | 56.4402 | 9.7694 |
| R2-O7 | hns_f4_s1 | 45.6027 | 56.1638 | 10.0458 |
| R2-O7 | original_lora | 59.7422 | 53.9685 | 12.1362 |
| R2-O7 | recon_0 | 60.8795 | 53.7979 | 12.4085 |
| R2-O8 | base | 76.1051 | 76.1553 | 0.0000 |
| R2-O8 | flat_nuclear | 78.6377 | 60.6131 | 16.1403 |
| R2-O8 | hns_f4_s1 | 78.9031 | 60.7811 | 16.1906 |
| R2-O8 | original_lora | 75.2808 | 57.5516 | 18.7527 |
| R2-O8 | recon_0 | 76.1180 | 58.0056 | 18.3070 |
| R2-O9 | base | 57.4174 | 63.6388 | 0.0000 |
| R2-O9 | flat_nuclear | 66.0982 | 64.6596 | 1.1083 |
| R2-O9 | hns_f4_s1 | 66.3652 | 64.6055 | 0.9058 |
| R2-O9 | original_lora | 64.4730 | 60.3247 | 4.6248 |
| R2-O9 | recon_0 | 64.7400 | 60.6650 | 4.4939 |
| R2-O10 | base | 88.2487 | 80.7075 | 0.0000 |
| R2-O10 | flat_nuclear | 86.5807 | 76.2913 | 4.5394 |
| R2-O10 | hns_f4_s1 | 85.8226 | 76.7464 | 4.1459 |
| R2-O10 | original_lora | 86.1259 | 71.7319 | 8.9756 |
| R2-O10 | recon_0 | 86.2017 | 71.8419 | 8.8656 |
| R2-O11 | base | 66.4730 | 69.9931 | 0.0000 |
| R2-O11 | gamma_val | 71.8077 | 72.4800 | 0.4542 |
| R2-O11 | hns_f4_s1 | 72.0834 | 72.7915 | 0.3441 |
| R2-O11 | original_lora | 67.9980 | 69.5512 | 2.7082 |
| R2-O11 | recon_0 | 68.4769 | 69.7658 | 2.5757 |
| R2-O12 | base | 66.7612 | 69.8970 | 0.0000 |
| R2-O12 | flat_nuclear | 74.0406 | 71.0490 | 0.5460 |
| R2-O12 | hns_f4_s1 | 74.0216 | 71.2621 | 0.4888 |
| R2-O12 | original_lora | 69.5724 | 66.4960 | 3.8190 |
| R2-O12 | recon_0 | 69.8133 | 66.6642 | 3.6933 |

These aggregates pool only the prespecified conditions within each experiment and do not imply comparable training-run uncertainty across experiments. Use paired_ci.tsv and subgroup results for the actual contrasts. O11 was executed after explicit author approval (see R2-plan.json and R2-O11/validation_protocol.md).
