# Sweep configuration summary

## Sources and defaults
- Training defaults: src/finetune/train_sft_peft.py (build_arg_parser)
- Sweep grid: src/finetune/sweep/make_grid.py
- Train profiles: src/finetune/train_profiles.py
- PEFT method specifics: src/finetune/peft_builders.py
- Task datasets/formatting: src/finetune/data/*.py
- Base-model specific defaults: none found in code; tokenizer uses base_model

## Assumptions
- assumed_num_gpus: 8
- gradient_accumulation_steps formula:
  - if global_train_batch_size is set: ceil(target / (per_device_bs * num_gpus))
  - otherwise: config.gradient_accumulation_steps
- effective_global_batch_size formula: grad_accum * per_device_bs * num_gpus

## Train profiles
| profile | max_seq_len | warmup_ratio | min_lr_ratio | weight_decay | grad_clip | bf16 | fp16 | adam_beta1 | adam_beta2 | global_train_batch_size | lr_scheduler_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_code_ift | 4096 | 0.1 | 0.01 | 0.0 | 1.0 | true | false | 0.9 | 0.95 | 192 | cosine_with_min_lr |
| paper_default_ift | 2048 | 0.1 | 0.01 | 0.0 | 1.0 | true | false | 0.9 | 0.95 | 256 | cosine_with_min_lr |
| paper_math_ift | 1024 | 0.1 | 0.01 | 0.0 | 1.0 | true | false | 0.9 | 0.95 | 768 | cosine_with_min_lr |

### Task -> profile (auto)
- default profile when task not mapped: paper_default_ift
- code -> paper_code_ift
- magicoder -> paper_code_ift
- math -> paper_math_ift
- metamath -> paper_math_ift
- metamathqa -> paper_math_ift

## Task plugins
| task | dataset_id | split | prompt_formatting |
| --- | --- | --- | --- |
| metamath | meta-math/MetaMathQA | train | Uses first present of query/original_question/question/instruction/prompt as instruction and response/answer/output/solution as response; formats with '### Instruction' + '### Response'. |
| magicoder | ise-uiuc/Magicoder-Evol-Instruct-110K | train | Uses first present of instruction/prompt/query/problem as instruction and response/output/answer/completion as response; formats with '### Instruction' + '### Response'. |
| alpaca | tatsu-lab/alpaca | train | If text is present, uses text as-is; otherwise uses instruction + optional input + output, formatted with '### Instruction' + '### Response'. |
| csqa | tau/commonsense_qa | train | Formats question + choices (A-E) and expects a single-letter answer; wraps with '### Instruction' + '### Response'. |

## PEFT method specifics
- lora: LoraConfig without init_lora_weights (peft default).
- pissa: LoraConfig with init_lora_weights=pissa_init_mode.
- adalora: AdaLoraConfig with total_step=max_steps, tinit=0.10, tfinal=0.80, deltaT=0.01.
- loraplus: LoraConfig like lora; optimizer uses LoRA+ param grouping.

## Resolved runs
- base_models: meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1
- tasks: metamath, magicoder, alpaca, csqa
- peft_methods: lora, loraplus, adalora, pissa

### Task: metamath
- dataset_id: meta-math/MetaMathQA
- dataset_split: train

| base_model | tokenizer_model | peft_method | train_profile | max_seq_len | max_steps | per_device_bs | grad_accum | global_bs_target | assumed_num_gpus | effective_global_bs | grad_accum_formula | lr | warmup_ratio | min_lr_ratio | weight_decay | adam_betas | optimizer | lr_scheduler_type | precision | grad_clip | grad_checkpointing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | lora | paper_math_ift | 1024 | 200 | 1 | 96 | 768 | 8 | 768 | ceil(768 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | loraplus | paper_math_ift | 1024 | 200 | 1 | 96 | 768 | 8 | 768 | ceil(768 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (torch.optim) with LoRA+ param groups | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | adalora | paper_math_ift | 1024 | 200 | 1 | 96 | 768 | 8 | 768 | ceil(768 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | pissa | paper_math_ift | 1024 | 200 | 1 | 96 | 768 | 8 | 768 | ceil(768 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | lora | paper_math_ift | 1024 | 200 | 1 | 96 | 768 | 8 | 768 | ceil(768 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | loraplus | paper_math_ift | 1024 | 200 | 1 | 96 | 768 | 8 | 768 | ceil(768 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (torch.optim) with LoRA+ param groups | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | adalora | paper_math_ift | 1024 | 200 | 1 | 96 | 768 | 8 | 768 | ceil(768 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | pissa | paper_math_ift | 1024 | 200 | 1 | 96 | 768 | 8 | 768 | ceil(768 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |

| base_model | peft_method | target_modules | r | lora_alpha | lora_dropout | init_lora_weights | loraplus_lr_ratio | loraplus_lr_embedding | loraplus_lr_assignment | adalora_init_r | adalora_target_r | adalora_total_step | adalora_tinit | adalora_tfinal | adalora_deltaT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-2-7b-hf | lora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | - | - | - | - | - | - |
| meta-llama/Llama-2-7b-hf | loraplus | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | 20.0 | - | lora_A=lr(0.0002), lora_B=lr*ratio(0.0002*20.0), lora_embedding_A=lr, lora_embedding_B=lr*ratio, other=lr | - | - | - | - | - | - |
| meta-llama/Llama-2-7b-hf | adalora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | 12 | 8 | 200 | 20 | 160 | 2 |
| meta-llama/Llama-2-7b-hf | pissa | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | pissa | - | - | - | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | lora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | loraplus | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | 20.0 | - | lora_A=lr(0.0002), lora_B=lr*ratio(0.0002*20.0), lora_embedding_A=lr, lora_embedding_B=lr*ratio, other=lr | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | adalora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | 12 | 8 | 200 | 20 | 160 | 2 |
| mistralai/Mistral-7B-v0.1 | pissa | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | pissa | - | - | - | - | - | - | - | - | - |

### Task: magicoder
- dataset_id: ise-uiuc/Magicoder-Evol-Instruct-110K
- dataset_split: train

| base_model | tokenizer_model | peft_method | train_profile | max_seq_len | max_steps | per_device_bs | grad_accum | global_bs_target | assumed_num_gpus | effective_global_bs | grad_accum_formula | lr | warmup_ratio | min_lr_ratio | weight_decay | adam_betas | optimizer | lr_scheduler_type | precision | grad_clip | grad_checkpointing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | lora | paper_code_ift | 4096 | 200 | 1 | 24 | 192 | 8 | 192 | ceil(192 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | loraplus | paper_code_ift | 4096 | 200 | 1 | 24 | 192 | 8 | 192 | ceil(192 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (torch.optim) with LoRA+ param groups | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | adalora | paper_code_ift | 4096 | 200 | 1 | 24 | 192 | 8 | 192 | ceil(192 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | pissa | paper_code_ift | 4096 | 200 | 1 | 24 | 192 | 8 | 192 | ceil(192 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | lora | paper_code_ift | 4096 | 200 | 1 | 24 | 192 | 8 | 192 | ceil(192 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | loraplus | paper_code_ift | 4096 | 200 | 1 | 24 | 192 | 8 | 192 | ceil(192 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (torch.optim) with LoRA+ param groups | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | adalora | paper_code_ift | 4096 | 200 | 1 | 24 | 192 | 8 | 192 | ceil(192 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | pissa | paper_code_ift | 4096 | 200 | 1 | 24 | 192 | 8 | 192 | ceil(192 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |

| base_model | peft_method | target_modules | r | lora_alpha | lora_dropout | init_lora_weights | loraplus_lr_ratio | loraplus_lr_embedding | loraplus_lr_assignment | adalora_init_r | adalora_target_r | adalora_total_step | adalora_tinit | adalora_tfinal | adalora_deltaT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-2-7b-hf | lora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | - | - | - | - | - | - |
| meta-llama/Llama-2-7b-hf | loraplus | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | 20.0 | - | lora_A=lr(0.0002), lora_B=lr*ratio(0.0002*20.0), lora_embedding_A=lr, lora_embedding_B=lr*ratio, other=lr | - | - | - | - | - | - |
| meta-llama/Llama-2-7b-hf | adalora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | 12 | 8 | 200 | 20 | 160 | 2 |
| meta-llama/Llama-2-7b-hf | pissa | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | pissa | - | - | - | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | lora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | loraplus | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | 20.0 | - | lora_A=lr(0.0002), lora_B=lr*ratio(0.0002*20.0), lora_embedding_A=lr, lora_embedding_B=lr*ratio, other=lr | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | adalora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | 12 | 8 | 200 | 20 | 160 | 2 |
| mistralai/Mistral-7B-v0.1 | pissa | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | pissa | - | - | - | - | - | - | - | - | - |

### Task: alpaca
- dataset_id: tatsu-lab/alpaca
- dataset_split: train

| base_model | tokenizer_model | peft_method | train_profile | max_seq_len | max_steps | per_device_bs | grad_accum | global_bs_target | assumed_num_gpus | effective_global_bs | grad_accum_formula | lr | warmup_ratio | min_lr_ratio | weight_decay | adam_betas | optimizer | lr_scheduler_type | precision | grad_clip | grad_checkpointing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | lora | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | loraplus | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (torch.optim) with LoRA+ param groups | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | adalora | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | pissa | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | lora | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | loraplus | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (torch.optim) with LoRA+ param groups | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | adalora | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | pissa | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |

| base_model | peft_method | target_modules | r | lora_alpha | lora_dropout | init_lora_weights | loraplus_lr_ratio | loraplus_lr_embedding | loraplus_lr_assignment | adalora_init_r | adalora_target_r | adalora_total_step | adalora_tinit | adalora_tfinal | adalora_deltaT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-2-7b-hf | lora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | - | - | - | - | - | - |
| meta-llama/Llama-2-7b-hf | loraplus | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | 20.0 | - | lora_A=lr(0.0002), lora_B=lr*ratio(0.0002*20.0), lora_embedding_A=lr, lora_embedding_B=lr*ratio, other=lr | - | - | - | - | - | - |
| meta-llama/Llama-2-7b-hf | adalora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | 12 | 8 | 200 | 20 | 160 | 2 |
| meta-llama/Llama-2-7b-hf | pissa | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | pissa | - | - | - | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | lora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | loraplus | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | 20.0 | - | lora_A=lr(0.0002), lora_B=lr*ratio(0.0002*20.0), lora_embedding_A=lr, lora_embedding_B=lr*ratio, other=lr | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | adalora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | 12 | 8 | 200 | 20 | 160 | 2 |
| mistralai/Mistral-7B-v0.1 | pissa | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | pissa | - | - | - | - | - | - | - | - | - |

### Task: csqa
- dataset_id: tau/commonsense_qa
- dataset_split: train

| base_model | tokenizer_model | peft_method | train_profile | max_seq_len | max_steps | per_device_bs | grad_accum | global_bs_target | assumed_num_gpus | effective_global_bs | grad_accum_formula | lr | warmup_ratio | min_lr_ratio | weight_decay | adam_betas | optimizer | lr_scheduler_type | precision | grad_clip | grad_checkpointing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | lora | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | loraplus | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (torch.optim) with LoRA+ param groups | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | adalora | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-7b-hf | pissa | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | lora | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | loraplus | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (torch.optim) with LoRA+ param groups | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | adalora | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |
| mistralai/Mistral-7B-v0.1 | mistralai/Mistral-7B-v0.1 | pissa | paper_default_ift | 2048 | 200 | 1 | 32 | 256 | 8 | 256 | ceil(256 / (1 * 8)) | 0.0002 | 0.1 | 0.01 | 0.0 | 0.9,0.95 | AdamW (transformers default) | cosine_with_min_lr | bf16 | 1.0 | false |

| base_model | peft_method | target_modules | r | lora_alpha | lora_dropout | init_lora_weights | loraplus_lr_ratio | loraplus_lr_embedding | loraplus_lr_assignment | adalora_init_r | adalora_target_r | adalora_total_step | adalora_tinit | adalora_tfinal | adalora_deltaT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-2-7b-hf | lora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | - | - | - | - | - | - |
| meta-llama/Llama-2-7b-hf | loraplus | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | 20.0 | - | lora_A=lr(0.0002), lora_B=lr*ratio(0.0002*20.0), lora_embedding_A=lr, lora_embedding_B=lr*ratio, other=lr | - | - | - | - | - | - |
| meta-llama/Llama-2-7b-hf | adalora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | 12 | 8 | 200 | 20 | 160 | 2 |
| meta-llama/Llama-2-7b-hf | pissa | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | pissa | - | - | - | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | lora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | loraplus | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | 20.0 | - | lora_A=lr(0.0002), lora_B=lr*ratio(0.0002*20.0), lora_embedding_A=lr, lora_embedding_B=lr*ratio, other=lr | - | - | - | - | - | - |
| mistralai/Mistral-7B-v0.1 | adalora | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | not set | - | - | - | 12 | 8 | 200 | 20 | 160 | 2 |
| mistralai/Mistral-7B-v0.1 | pissa | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 16 | 32 | 0.05 | pissa | - | - | - | - | - | - | - | - | - |
