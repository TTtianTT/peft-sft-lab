# Code / HumanEval Recipe

This repo now supports:

- SFT training on `ise-uiuc/Magicoder-Evol-Instruct-110K`
- Local code SFT files in the same normalized style as the new math/IF pipeline
- HumanEval evaluation on Hugging Face `openai/openai_humaneval`
- Checkpoint-by-checkpoint evaluation for `meta-llama/Llama-3.1-8B-Instruct`
- A second raw-completion code path via `--sft_format raw_completion`

## Training Data Format

The `code` task accepts four input shapes.

### 1. Native Magicoder format

```json
{
  "instruction": "Write a Python function that returns 42.",
  "response": "def answer():\n    return 42"
}
```

This is the native shape used by `ise-uiuc/Magicoder-Evol-Instruct-110K`.

### 2. Repo-native chat SFT format

```json
{
  "prompt": [
    {"role": "user", "content": "Write a Python function that returns 42."}
  ],
  "completion": [
    {"role": "assistant", "content": "def answer():\n    return 42"}
  ]
}
```

Use this if you want a normalized local format consistent with the new math-data pipeline.

### 3. Conversation format

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function that returns 42."},
    {"role": "assistant", "content": "def answer():\n    return 42"}
  ]
}
```

`conversations` is also accepted as a field name.

### 4. Simple single-turn fallback

```json
{
  "prompt": "Write a Python function that returns 42.",
  "response": "def answer():\n    return 42"
}
```

`output`, `answer`, `completion`, and `solution` are also accepted in place of `response`.

## Recommended Data Usage

For your first run, there are two clean options.

### Option A. Sample 50K directly from Hugging Face

Use the official `ise-uiuc/Magicoder-Evol-Instruct-110K` training split and let the trainer do a deterministic shuffle + truncate:

- `--max_train_samples 50000`
- `--dataset_seed 42`

This is the lightest workflow and is sufficient if you do not need to freeze a separate local copy.

### Option B. Freeze a local 50K sample

If you want the exact sample on disk for reuse across runs, write a local `.json`, `.jsonl`, or `.parquet` file in one of the accepted formats above and pass:

- `--dataset_path /abs/path/to/magicoder_50k.jsonl`

## Recommended First Run

For your first `meta-llama/Llama-3.1-8B-Instruct` experiment, use:

- task: `code`
- train dataset: `ise-uiuc/Magicoder-Evol-Instruct-110K`
- eval dataset: `openai/openai_humaneval`
- train profile: `paper_code_ift_2ep`
- epochs: `2`
- metric to watch: `pass@1`

Example:

```bash
accelerate launch --num_processes 8 -m finetune.train_sft_peft \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --task code \
  --peft_method lora \
  --output_dir runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/profile-paper_code_ift_2ep/rank-16/seed42 \
  --train_profile paper_code_ift_2ep \
  --per_device_train_batch_size 1 \
  --lr 2e-4 \
  --r 16 \
  --max_train_samples 50000 \
  --dataset_seed 42 \
  --bf16 \
  --gradient_checkpointing \
  --seed 42
```

`save_strategy="epoch"` is already enabled in the training pipeline, so a 2-epoch run will automatically save:

- one checkpoint at the end of epoch 1
- one checkpoint at the end of epoch 2
- the final adapter in the run root

## Single-GPU 32GB Commands

If you are limited to one GPU with about 32GB VRAM, prefer QLoRA and do not use `paper_code_ift_2ep` directly.

Reason:

- `paper_code_ift_2ep` forces `max_seq_len=4096`
- on a single 32GB card, `Llama-3.1-8B-Instruct` is much safer with `--use_qlora` and a shorter sequence length such as `2048`

### Smoke test

```bash
accelerate launch --num_processes 1 -m finetune.train_sft_peft \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --task code \
  --peft_method lora \
  --dataset_path /root/autodl-tmp/magicoder-train.jsonl \
  --output_dir runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/smoke-qlora-2048/seed42 \
  --max_steps 8 \
  --max_train_samples 128 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_len 1024 \
  --lr 2e-4 \
  --warmup_ratio 0.1 \
  --min_lr_ratio 0.01 \
  --lr_scheduler_type cosine_with_min_lr \
  --weight_decay 0.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --grad_clip 1.0 \
  --r 16 \
  --use_qlora \
  --bf16 \
  --gradient_checkpointing \
  --seed 42
```

### Full 50K run

```bash
accelerate launch --num_processes 1 -m finetune.train_sft_peft \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --task code \
  --peft_method lora \
  --dataset_path /root/autodl-tmp/magicoder-train.jsonl \
  --output_dir runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/single-gpu-qlora-50k-2ep/seed42 \
  --num_train_epochs 2 \
  --max_train_samples 50000 \
  --dataset_seed 42 \
  --per_device_train_batch_size 1 \
  --global_train_batch_size 192 \
  --max_seq_len 2048 \
  --lr 2e-4 \
  --warmup_ratio 0.1 \
  --min_lr_ratio 0.01 \
  --lr_scheduler_type cosine_with_min_lr \
  --weight_decay 0.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --grad_clip 1.0 \
  --r 16 \
  --use_qlora \
  --bf16 \
  --gradient_checkpointing \
  --seed 42
```

If your `magicoder-train.jsonl` is already a fixed 50K subset, remove `--max_train_samples 50000`.

## HumanEval Evaluation

`finetune.eval.eval_humaneval` now defaults to Hugging Face `openai/openai_humaneval`.

The expected eval fields are:

- `task_id`
- `prompt`
- `canonical_solution`
- `test`
- `entry_point`

Run final-adapter evaluation like this:

```bash
python -m finetune.eval.eval_humaneval \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter_dir runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/profile-paper_code_ift_2ep/rank-16/seed42 \
  --output_dir eval/humaneval-llama31-code-lora-final \
  --use_vllm \
  --tensor_parallel_size 8 \
  --max_new_tokens 256
```

Evaluate a specific epoch checkpoint by pointing `--adapter_dir` at its `checkpoint-*` directory:

```bash
python -m finetune.eval.eval_humaneval \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter_dir runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/profile-paper_code_ift_2ep/rank-16/seed42/checkpoint-<step> \
  --output_dir eval/humaneval-llama31-code-lora-epochX \
  --use_vllm \
  --tensor_parallel_size 8 \
  --max_new_tokens 256
```

If you want to evaluate a local HumanEval clone or subset instead of the default Hugging Face test split, pass:

```bash
--dataset_path /abs/path/to/openai_humaneval_snapshot_or_file --split test
```

## Raw-Completion Variant

If you want a second pipeline closer to completion-style HumanEval, train with:

- `--sft_format raw_completion`
- `--train_profile paper_code_raw_2ep`

Example:

```bash
accelerate launch --num_processes 1 -m finetune.train_sft_peft \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --task code \
  --peft_method lora \
  --dataset_path /root/autodl-tmp/magicoder-train.jsonl \
  --output_dir runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/raw-completion-50k-2ep/seed42 \
  --train_profile paper_code_raw_2ep \
  --sft_format raw_completion \
  --max_train_samples 50000 \
  --dataset_seed 42 \
  --per_device_train_batch_size 1 \
  --lr 2e-4 \
  --r 16 \
  --bf16 \
  --gradient_checkpointing \
  --seed 42
```

Evaluate that raw-completion adapter on HumanEval with:

```bash
python -m finetune.eval.eval_humaneval \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter_dir runs/meta-llama-Llama-3.1-8B-Instruct/code/lora/raw-completion-50k-2ep/seed42 \
  --output_dir eval/humaneval-llama31-code-lora-raw \
  --prompt_style raw \
  --use_vllm \
  --tensor_parallel_size 1 \
  --max_new_tokens 256
```

## Batch Evaluation Workflow

For a full run directory containing the final adapter and `checkpoint-*` subdirectories, use:

```bash
python scripts/eval_runs.py \
  --runs_root runs \
  --task code \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --output_dir eval_results/code_llama31_2ep \
  --use_vllm \
  --tensor_parallel_size 8 \
  --max_new_tokens 256 \
  --timeout_s 3.0
```

This will discover:

- the final adapter
- epoch checkpoints under the same run
- any other matching code adapters for the same base model

and will write per-run `metrics.json`, logs, plus a summarized `summary.csv`.
