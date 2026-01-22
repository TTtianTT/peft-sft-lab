# peft-sft-lab

Minimal, reproducible PEFT SFT lab: **2 base models × 4 tasks × LoRA-family variants**, single-node multi-GPU training via `accelerate`, plus simple sweep utilities.

## Quickstart

### 0) Install

Install PyTorch for your CUDA first (see https://pytorch.org). Then:

```bash
cd peft-sft-lab
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 1) Configure Accelerate (once)

```bash
accelerate config
```

Multi-GPU smoke test (saves an adapter):

```bash
accelerate launch --num_processes 2 -m finetune.train_sft_peft \
  --task alpaca --peft_method lora --output_dir runs/smoke-alpaca-lora \
  --max_steps 10 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --bf16 --gradient_checkpointing
```

## Train: one run

```bash
accelerate launch --num_processes 4 -m finetune.train_sft_peft \
  --base_model mistralai/Mistral-7B-v0.3 \
  --task csqa \
  --peft_method adalora \
  --output_dir runs/mistral-csqa-adalora \
  --max_steps 200 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr 2e-4 \
  --bf16 \
  --gradient_checkpointing
```

### LoRA+ (optimizer-only)

```bash
accelerate launch --num_processes 4 -m finetune.train_sft_peft \
  --task alpaca --peft_method loraplus --output_dir runs/mistral-alpaca-loraplus \
  --max_steps 200 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --lr 2e-4 --loraplus_lr_ratio 20.0 --bf16 --gradient_checkpointing
```

### QLoRA (4-bit)

```bash
accelerate launch --num_processes 4 -m finetune.train_sft_peft \
  --task alpaca --peft_method lora --use_qlora \
  --output_dir runs/qlora-alpaca \
  --max_steps 100 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --lr 2e-4 --bf16 --gradient_checkpointing
```

## Sweeps

Generate a `configs.jsonl` grid:

```bash
python -m finetune.sweep.make_grid --output configs.jsonl
```

Run the grid serially (each job uses `accelerate launch`):

```bash
python -m finetune.sweep.run_grid --configs_jsonl configs.jsonl --num_processes 4
```

Or use bash helpers:

```bash
./scripts/launch_one.sh
./scripts/launch_grid.sh configs.jsonl
```

## Evaluate

All evaluators save `metrics.json` and (when applicable) generations under the given `--output_dir`.

### GSM8K (strict match)

```bash
python -m finetune.eval.eval_gsm8k \
  --base_model mistralai/Mistral-7B-v0.3 \
  --adapter_dir runs/mistral-math-lora \
  --output_dir eval/gsm8k-mistral-math-lora
```

### HumanEval (pass@1, minimal)

```bash
python -m finetune.eval.eval_humaneval \
  --base_model mistralai/Mistral-7B-v0.3 \
  --adapter_dir runs/mistral-code-lora \
  --output_dir eval/humaneval-mistral-code-lora
```

Note: this evaluator executes generated code in a subprocess with a timeout.

### IFEval (rule-based, minimal)

```bash
python -m finetune.eval.eval_ifeval \
  --base_model mistralai/Mistral-7B-v0.3 \
  --adapter_dir runs/mistral-alpaca-lora \
  --output_dir eval/ifeval-mistral-alpaca-lora
```

### CommonsenseQA (A/B/C/D/E accuracy)

```bash
python -m finetune.eval.eval_csqa \
  --base_model mistralai/Mistral-7B-v0.3 \
  --adapter_dir runs/mistral-csqa-lora \
  --output_dir eval/csqa-mistral-csqa-lora
```

## Spectral edit (LoRA)

Smoke test: edit one metamath adapter, evaluate a small GSM8K slice, and verify outputs.

```bash
python -m finetune.spectral_edit.cli edit \
  --base_model meta-llama/Llama-3.1-8B \
  --lora_path runs/meta-llama-Llama-3.1-8B/metamath/lora/profile-default/rank-16/seed42 \
  --out_dir runs/edited/metamath/lora/seed42/smooth_abs \
  --mode smooth_abs \
  --calib_samples 8 \
  --calib_batch_size 2

python -m finetune.eval.eval_gsm8k \
  --base_model meta-llama/Llama-3.1-8B \
  --adapter_dir runs/edited/metamath/lora/seed42/smooth_abs \
  --output_dir eval/gsm8k-metamath-smooth-abs \
  --max_samples 32

ls runs/edited/metamath/lora/seed42/smooth_abs/adapter_model.safetensors \
   runs/edited/metamath/lora/seed42/smooth_abs/spectral_edit_meta.json
```

## Output structure

Each training run writes:

- `run_args.json`: exact CLI args used
- `requirements-freeze.txt`: `pip freeze` snapshot (best-effort)
- `train.log`: stdout+file logger copy
- PEFT adapter files (e.g. `adapter_model.safetensors`, `adapter_config.json`)

## Common errors

- **Gated model / auth required** (e.g. Llama-3.1): run `huggingface-cli login` and accept the model terms on Hugging Face.
- **Dataset not found / missing fields**: check the dataset card; this repo prints the expected columns for each task.
- **PiSSA not supported**: upgrade PEFT (`pip install -U 'peft>=0.11.0'`).
- **OOM**:
  - Use `--use_qlora`, lower `--max_seq_len`, reduce batch size, increase gradient accumulation.
  - Turn on `--gradient_checkpointing`.
- **AdaLoRA schedule mismatch**: AdaLoRA requires `total_step == max_steps`; this repo enforces it automatically.
