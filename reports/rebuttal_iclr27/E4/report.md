# E4: validation-selected sources and rank64

Status: complete.

Single training seed42. All intervals are paired question-level intervals, not checkpoint or training-run intervals. Six predeclared validation candidates per task; both full candidate tables are published. The r64 source uses the selected MetaMath LR and a predeclared final checkpoint. Optional runs omitted.

| Condition | Method | Target | Off | FG |
| --- | --- | ---: | ---: | ---: |
| magicoder_val_best_r16 | base | 66.4634 | 79.3692 | 0.0000 |
| magicoder_val_best_r16 | original_lora | 64.0244 | 77.5627 | 2.1262 |
| magicoder_val_best_r16 | recon_0 | 64.0244 | 77.9787 | 1.7281 |
| magicoder_val_best_r16 | hns_f4_s1 | 73.1707 | 80.5574 | 0.0000 |
| magicoder_val_best_r16 | flat_nuclear | 73.1707 | 80.4974 | 0.0000 |
| metamath_val_best_r16 | base | 85.7468 | 72.9414 | 0.0000 |
| metamath_val_best_r16 | original_lora | 84.7612 | 67.0764 | 5.9255 |
| metamath_val_best_r16 | recon_0 | 84.5337 | 68.0421 | 4.9645 |
| metamath_val_best_r16 | hns_f4_s1 | 89.1585 | 74.2772 | 0.0000 |
| metamath_val_best_r16 | flat_nuclear | 88.3245 | 73.8269 | 0.0000 |
| metamath_r64 | base | 85.7468 | 72.9414 | 0.0000 |
| metamath_r64 | original_lora | 85.2919 | 63.3601 | 9.5813 |
| metamath_r64 | recon_0 | 85.4435 | 63.0750 | 9.8664 |
| metamath_r64 | hns_f4_s1 | 89.2343 | 74.8373 | 0.0000 |
| metamath_r64 | flat_nuclear | 89.0826 | 74.5768 | 0.1848 |

| Condition | Comparison | Metric | Difference [95% CI] pp | Repairs/damage |
| --- | --- | --- | ---: | ---: |
| magicoder_val_best_r16 | recon_0 − original_lora | Target | +0.0000 [-1.8293, +1.8293] | 1/1 |
| magicoder_val_best_r16 | recon_0 − original_lora | Off | +0.4160 [-0.3238, +1.1587] | NA/NA |
| magicoder_val_best_r16 | recon_0 − original_lora | FG | -0.3981 [-1.0900, +0.3302] | NA/NA |
| magicoder_val_best_r16 | hns_f4_s1 − original_lora | Target | +9.1463 [+1.2195, +17.0732] | 29/14 |
| magicoder_val_best_r16 | hns_f4_s1 − original_lora | Off | +2.9947 [+1.7558, +4.2323] | NA/NA |
| magicoder_val_best_r16 | hns_f4_s1 − original_lora | FG | -2.1262 [-3.3236, -0.9397] | NA/NA |
| magicoder_val_best_r16 | hns_f4_s1 − recon_0 | Target | +9.1463 [+1.8293, +17.0732] | 29/14 |
| magicoder_val_best_r16 | hns_f4_s1 − recon_0 | Off | +2.5786 [+1.3110, +3.8561] | NA/NA |
| magicoder_val_best_r16 | hns_f4_s1 − recon_0 | FG | -1.7281 [-2.9666, -0.6571] | NA/NA |
| magicoder_val_best_r16 | flat_nuclear − original_lora | Target | +9.1463 [+1.2195, +17.0732] | 29/14 |
| magicoder_val_best_r16 | flat_nuclear − original_lora | Off | +2.9347 [+1.6604, +4.2158] | NA/NA |
| magicoder_val_best_r16 | flat_nuclear − original_lora | FG | -2.1262 [-3.2967, -0.8924] | NA/NA |
| magicoder_val_best_r16 | flat_nuclear − recon_0 | Target | +9.1463 [+1.2195, +17.0732] | 29/14 |
| magicoder_val_best_r16 | flat_nuclear − recon_0 | Off | +2.5186 [+1.2119, +3.8421] | NA/NA |
| magicoder_val_best_r16 | flat_nuclear − recon_0 | FG | -1.7281 [-2.9255, -0.5954] | NA/NA |
| magicoder_val_best_r16 | flat_nuclear − hns_f4_s1 | Target | +0.0000 [-2.4390, +2.4390] | 2/2 |
| magicoder_val_best_r16 | flat_nuclear − hns_f4_s1 | Off | -0.0600 [-0.6330, +0.5012] | NA/NA |
| magicoder_val_best_r16 | flat_nuclear − hns_f4_s1 | FG | +0.0000 [-0.1848, +0.4313] | NA/NA |
| metamath_val_best_r16 | recon_0 − original_lora | Target | -0.2274 [-0.9098, +0.5307] | 10/13 |
| metamath_val_best_r16 | recon_0 − original_lora | Off | +0.9657 [+0.2151, +1.8188] | NA/NA |
| metamath_val_best_r16 | recon_0 − original_lora | FG | -0.9610 [-1.7748, -0.1848] | NA/NA |
| metamath_val_best_r16 | hns_f4_s1 − original_lora | Target | +4.3973 [+2.5777, +6.2168] | 106/48 |
| metamath_val_best_r16 | hns_f4_s1 − original_lora | Off | +7.2008 [+4.4198, +10.0234] | NA/NA |
| metamath_val_best_r16 | hns_f4_s1 − original_lora | FG | -5.9255 [-8.3893, -2.6423] | NA/NA |
| metamath_val_best_r16 | hns_f4_s1 − recon_0 | Target | +4.6247 [+2.8810, +6.4443] | 106/45 |
| metamath_val_best_r16 | hns_f4_s1 − recon_0 | Off | +6.2351 [+3.4641, +9.0582] | NA/NA |
| metamath_val_best_r16 | hns_f4_s1 − recon_0 | FG | -4.9645 [-7.4304, -1.8171] | NA/NA |
| metamath_val_best_r16 | flat_nuclear − original_lora | Target | +3.5633 [+1.7437, +5.3829] | 100/53 |
| metamath_val_best_r16 | flat_nuclear − original_lora | Off | +6.7505 [+3.9412, +9.5633] | NA/NA |
| metamath_val_best_r16 | flat_nuclear − original_lora | FG | -5.9255 [-8.1677, -2.5502] | NA/NA |
| metamath_val_best_r16 | flat_nuclear − recon_0 | Target | +3.7908 [+2.0470, +5.6103] | 99/49 |
| metamath_val_best_r16 | flat_nuclear − recon_0 | Off | +5.7848 [+2.9987, +8.6329] | NA/NA |
| metamath_val_best_r16 | flat_nuclear − recon_0 | FG | -4.9645 [-7.2004, -1.7396] | NA/NA |
| metamath_val_best_r16 | flat_nuclear − hns_f4_s1 | Target | -0.8340 [-1.5921, -0.0758] | 8/19 |
| metamath_val_best_r16 | flat_nuclear − hns_f4_s1 | Off | -0.4503 [-1.4299, +0.4106] | NA/NA |
| metamath_val_best_r16 | flat_nuclear − hns_f4_s1 | FG | +0.0000 [-0.4313, +1.0963] | NA/NA |
| metamath_r64 | recon_0 − original_lora | Target | +0.1516 [-0.5307, +0.8340] | 11/9 |
| metamath_r64 | recon_0 − original_lora | Off | -0.2851 [-1.1679, +0.5315] | NA/NA |
| metamath_r64 | recon_0 − original_lora | FG | +0.2851 [-0.5027, +1.1522] | NA/NA |
| metamath_r64 | hns_f4_s1 − original_lora | Target | +3.9424 [+2.2745, +5.6103] | 88/36 |
| metamath_r64 | hns_f4_s1 − original_lora | Off | +11.4772 [+8.6055, +14.3758] | NA/NA |
| metamath_r64 | hns_f4_s1 − original_lora | FG | -9.5813 [-12.2431, -6.4019] | NA/NA |
| metamath_r64 | hns_f4_s1 − recon_0 | Target | +3.7908 [+2.1228, +5.4587] | 86/36 |
| metamath_r64 | hns_f4_s1 − recon_0 | Off | +11.7623 [+8.8238, +14.7121] | NA/NA |
| metamath_r64 | hns_f4_s1 − recon_0 | FG | -9.8664 [-12.4848, -6.7510] | NA/NA |
| metamath_r64 | flat_nuclear − original_lora | Target | +3.7908 [+2.1228, +5.4587] | 87/37 |
| metamath_r64 | flat_nuclear − original_lora | Off | +11.2168 [+8.3330, +14.0914] | NA/NA |
| metamath_r64 | flat_nuclear − original_lora | FG | -9.3965 [-12.1166, -6.2737] | NA/NA |
| metamath_r64 | flat_nuclear − recon_0 | Target | +3.6391 [+1.9712, +5.3071] | 86/38 |
| metamath_r64 | flat_nuclear − recon_0 | Off | +11.5018 [+8.5644, +14.3930] | NA/NA |
| metamath_r64 | flat_nuclear − recon_0 | FG | -9.6816 [-12.3841, -6.6023] | NA/NA |
| metamath_r64 | flat_nuclear − hns_f4_s1 | Target | -0.1516 [-0.8340, +0.5307] | 10/12 |
| metamath_r64 | flat_nuclear − hns_f4_s1 | Off | -0.2605 [-1.1790, +0.6478] | NA/NA |
| metamath_r64 | flat_nuclear − hns_f4_s1 | FG | +0.1848 [-0.4065, +0.7394] | NA/NA |

20,000 paired item bootstrap draws; CS subtask macro preserved. McNemar exact tests and repair/damage counts for every atomic binary benchmark/subtask are in paired_ci.tsv. Off/FG are macro statistics and have no binary McNemar test. All main-protocol inputs hash-match final Base. Warmup settings and actual Trainer totals are verified for all seven trainings; see training audits.

Module spectra, minima and counts below0.005 are in module_tail_audit.tsv. Code/config/weights/dataset hashes and commands are archived. No test-set choice or IID training-run claim.

Slurm job 14893; two GPUs allocated, at most two1-GPU workers at any time.
