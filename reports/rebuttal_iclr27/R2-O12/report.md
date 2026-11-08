# R2-O12: independent auxiliary targets

Status: complete. No checkpoint/hyperparameter was selected on these tests.

| Source task | Method | Auxiliary target | Main Target | Main Off | Main FG |
| --- | --- | ---: | ---: | ---: | ---: |
| magicoder | base | 64.6825 | 59.4512 | 72.3337 | 0.0000 |
| magicoder | original_lora | 61.8166 | 59.3496 | 69.6176 | 3.1312 |
| magicoder | recon_0 | 61.5961 | 59.6545 | 69.6239 | 3.1877 |
| magicoder | hns_f4_s1 | 64.5944 | 64.6341 | 73.5311 | 0.6445 |
| magicoder | flat_nuclear | 64.4621 | 64.6341 | 73.2321 | 0.6667 |
| metamath | base | 44.9000 | 74.0713 | 67.4604 | 0.0000 |
| metamath | original_lora | 43.4333 | 79.7953 | 63.3745 | 4.5069 |
| metamath | recon_0 | 43.3667 | 79.9722 | 63.7045 | 4.1988 |
| metamath | hns_f4_s1 | 50.2667 | 83.4091 | 68.9931 | 0.3330 |
| metamath | flat_nuclear | 50.0000 | 83.4471 | 68.8658 | 0.4252 |

Auxiliary target is MBPP+378 for Magicoder and MATH-500500 for MetaMath. Main Target/Off/FG reuse the unchanged four-family protocol after ordered input-hash audits. Original main-source checkpoints were never selected on MBPP validation; E4/R2-selected sources are excluded. Official EvalPlus base+plus pass and math-verify scoring; parser unit tests passed before inference. All20,000 paired checkpoint intervals are descriptive on shared tests, not IID training-run CIs. Per-condition item bootstrap/McNemar repairs/damage are separately labeled in paired_ci.tsv. All52 new auxiliary cells, cached main cells and artifact hashes passed coverage/integrity audits. All evaluation/scoring ran through Slurm with2 GPUs total.
