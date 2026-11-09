# R2-O7 GSM8K output-format diagnostic (AUXILIARY)

Read-only re-analysis of saved O7 GSM8K outputs. The protocol metric is unchanged: main O7 Target remains GSM8K strict (first line after the final `####`, `_norm`, exact match). The prompt asks for the final line `#### <answer>`; the auxiliary column only strips a literal `<answer>`/`</answer>` tag after the final `####` and then applies the same rule (plus numeric equality). It is a formatting diagnostic, not a replacement score, and was defined after seeing the outputs.

| Base | Method | Main strict | Aux tag-tolerant | Aux - main | `####` present | literal `<answer>` after `####` | finish=length | mean tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | base | 4.02 | 30.25 | +26.23 | 31.5 | 31.1 | 1.21 | 200 |
| Qwen3-1.7B | original_lora | 65.20 | 74.60 | +9.40 | 98.5 | 12.7 | 0.15 | 125 |
| Qwen3-1.7B | recon_0 | 65.13 | 74.30 | +9.17 | 98.5 | 11.5 | 0.00 | 124 |
| Qwen3-1.7B | hns_f4_s1 | 5.23 | 77.86 | +72.63 | 92.9 | 92.3 | 0.00 | 132 |
| Qwen3-1.7B | flat_nuclear | 5.69 | 78.01 | +72.33 | 92.1 | 91.6 | 0.00 | 132 |
| Qwen3-4B | base | 64.14 | 81.35 | +17.21 | 89.2 | 16.1 | 0.68 | 211 |
| Qwen3-4B | original_lora | 54.28 | 83.85 | +29.57 | 100.0 | 33.9 | 0.08 | 124 |
| Qwen3-4B | recon_0 | 56.63 | 83.70 | +27.07 | 100.0 | 30.6 | 0.00 | 124 |
| Qwen3-4B | hns_f4_s1 | 85.97 | 88.17 | +2.20 | 100.0 | 1.2 | 0.00 | 125 |
| Qwen3-4B | flat_nuclear | 85.60 | 87.87 | +2.27 | 100.0 | 1.3 | 0.00 | 125 |

Recomputed main strict equals each metrics.json exactly; scored.jsonl SHA256 matches evaluation_cells.tsv.

## Context: same diagnostic on the main 8B GSM8K cells

Pooled over the 9 main checkpoints per base (3 tasks x 3 seeds; the R2-O11 cell set, which links the E1 cached cells). Item-pooled descriptive rates only; use the protocol paired CIs for inference. At 8B the literal placeholder is rare; the remaining strict-vs-auxiliary gap is mostly currency/decimal formatting such as `$18` or `18.00`, which the auxiliary numeric-equality rule accepts. HNS remains above Recon-0 and original LoRA under both parses.

| Base | Method | Items | Main strict | Aux tag-tolerant | literal `<answer>` after `####` |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | base | 11871 | 62.40 | 80.67 | 0.0 |
| Llama-3.1-8B-Instruct | gamma_val | 11871 | 70.80 | 80.65 | 0.0 |
| Llama-3.1-8B-Instruct | hns_f4_s1 | 11871 | 70.82 | 80.70 | 0.0 |
| Llama-3.1-8B-Instruct | original_lora | 11871 | 65.93 | 74.31 | 0.1 |
| Llama-3.1-8B-Instruct | recon_0 | 11871 | 66.59 | 74.83 | 0.1 |
| Qwen3-8B | base | 11871 | 85.75 | 91.05 | 2.9 |
| Qwen3-8B | gamma_val | 11871 | 87.57 | 90.67 | 0.1 |
| Qwen3-8B | hns_f4_s1 | 11871 | 88.44 | 91.52 | 0.2 |
| Qwen3-8B | original_lora | 11871 | 85.51 | 87.97 | 0.1 |
| Qwen3-8B | recon_0 | 11871 | 85.41 | 87.83 | 0.1 |
