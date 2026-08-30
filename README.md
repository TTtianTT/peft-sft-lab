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

Instruction-following SFT is also supported with `--task if` (training on `allenai/tulu-3-sft-personas-instruction-following`).

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

For a Qwen3-8B run that matches the Llama 3.1 math recipe, explicitly use the
non-thinking chat template for both SFT and GSM8K evaluation:

```bash
PYTHONPATH=src accelerate launch --num_processes 8 \
  -m finetune.train_sft_peft \
  --base_model Qwen/Qwen3-8B \
  --task math \
  --peft_method lora \
  --output_dir runs_refactor_data_20260121/Qwen-Qwen3-8B/math/lora/profile-paper_math_ift_3ep/rank-16/seed42 \
  --train_profile paper_math_ift_3ep \
  --per_device_train_batch_size 2 \
  --lr 1e-4 \
  --max_train_samples 50000 \
  --r 16 \
  --seed 42 \
  --sft_format chat \
  --chat_template_mode non_thinking

PYTHONPATH=src python -m finetune.eval.eval_gsm8k \
  --base_model Qwen/Qwen3-8B \
  --adapter_dir runs_refactor_data_20260121/Qwen-Qwen3-8B/math/lora/profile-paper_math_ift_3ep/rank-16/seed42 \
  --output_dir eval_results/qwen3-8b-metamath-lora-gsm8k \
  --use_vllm \
  --tensor_parallel_size 8 \
  --max_new_tokens 256 \
  --seed 42 \
  --chat_template_mode non_thinking
```

### HumanEval (pass@1, minimal)

```bash
python -m finetune.eval.eval_humaneval \
  --base_model mistralai/Mistral-7B-v0.3 \
  --adapter_dir runs/mistral-code-lora \
  --output_dir eval/humaneval-mistral-code-lora
```

Note: this evaluator executes generated code in a subprocess with a timeout.

See `reports/code_humaneval_recipe.md` for the recommended 2-epoch `meta-llama/Llama-3.1-8B-Instruct` setup, supported local code-data formats, and checkpoint evaluation workflow.

### MBPP (chat pass@1)

The MBPP evaluator defaults to the sanitized MBPP test split and LlamaFactory-compatible
Llama-3 chat prompting: one user task followed by an assistant code generation, with no
system message. The user prompt includes MBPP's first public test, following the standard
MBPP protocol so the required function name is visible. It runs generated Python against
the full MBPP test list in a temporary directory.

```bash
PYTHONPATH=src python -m finetune.eval.eval_mbpp \
  --base_model /root/autodl-tmp/Llama-3.1-8B-Instruct \
  --adapter_dir /root/autodl-tmp/llamafactory_outputs/llama31_magicoder_lora_r16 \
  --output_dir eval/mbpp-llama31-magicoder-chat-lora \
  --use_vllm \
  --tensor_parallel_size 1 \
  --vllm_attention_backend FLASH_ATTN \
  --vllm_disable_flashinfer_sampler
```

Use `--dataset_path` with a local parquet/json/jsonl snapshot when Hugging Face access is unavailable.

### IFEval (rule-based, minimal)

```bash
python -m finetune.eval.eval_ifeval \
  --base_model mistralai/Mistral-7B-v0.3 \
  --adapter_dir runs/mistral-alpaca-lora \
  --output_dir eval/ifeval-mistral-alpaca-lora
```

### IFBench (official scorer)

```bash
python -m finetune.eval.eval_ifbench \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter_dir runs/meta-llama-Llama-3.1-8B-Instruct/if/lora/profile-paper_if_tulu_2ep/rank-16/seed42 \
  --official_eval_root external/IFBench \
  --output_dir eval/ifbench-llama31-if-lora \
  --use_vllm \
  --tensor_parallel_size 8 \
  --max_new_tokens 2048
```

See `reports/instruction_following_ifbench_recipe.md` for the recommended 2-epoch Llama-3.1-8B-Instruct setup and supported local data formats.

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

Post-hoc HNS experiment: flatten one LoRA checkpoint with a Muon-style Hybrid Newton-Schulz pass, restore its nuclear norm, and then evaluate it directly without retraining or calibration data.

```bash
python -m finetune.spectral_edit.cli hns \
  --lora_path runs/meta-llama-Llama-3.1-8B/metamath/lora/profile-default/rank-16/seed42 \
  --out_dir runs/edited/metamath/lora/seed42/hns_8plus2 \
  --target_modules down_proj o_proj \
  --output_rank 16

python -m finetune.eval.eval_gsm8k \
  --base_model meta-llama/Llama-3.1-8B \
  --adapter_dir runs/edited/metamath/lora/seed42/hns_8plus2 \
  --output_dir eval/gsm8k-metamath-hns-8plus2 \
  --max_samples 32

cat runs/edited/metamath/lora/seed42/hns_8plus2/spectral_edit_meta.json
```

### Calibration importance + test-time HNS

The two-stage commonsense workflow keeps HNS as a fixed binary edit. Stage 1
ranks LoRA modules with calibration saliency `mean(|sigma * dL/dsigma|)`, keeps
a high-importance budget, and rejects modules whose fixed HNS direction has
non-positive first-order compatibility. Stage 2 uses unlabeled choice
permutations on each test task to select a task-specific subset and falls back
to the original LoRA when entropy/consistency does not improve or the KL trust
region is exceeded.

```bash
# Stage 1: supervised calibration localization and fixed HNS proposal.
PYTHONPATH=src python -m finetune.spectral_edit.cli sensitivity-hns \
  --base_model /path/to/Llama-3.1-8B-Instruct \
  --lora_path runs/Llama-3.1-8B-Instruct/commonsense170k/lora-2ep/seed42 \
  --out_dir runs/Llama-3.1-8B-Instruct/commonsense170k/sensitivity-hns-8plus2/seed42 \
  --target_modules all_modules \
  --calib_dataset commonsense170k \
  --calib_dataset_path /path/to/commonsense_170k.parquet \
  --calib_samples 256 \
  --calib_batch_size 2 \
  --calib_shuffle \
  --sft_format chat \
  --chat_template_mode auto \
  --fast_steps 8 \
  --stable_steps 2

# Stage 2: label-free task-level module routing and exact candidate validation.
python scripts/build_commonsense_tthns_adapters.py \
  --base_model /path/to/Llama-3.1-8B-Instruct \
  --lora_path runs/Llama-3.1-8B-Instruct/commonsense170k/lora-2ep/seed42 \
  --calibration_hns_path runs/Llama-3.1-8B-Instruct/commonsense170k/sensitivity-hns-8plus2/seed42 \
  --out_root runs/Llama-3.1-8B-Instruct/commonsense170k/tthns-8plus2/seed42 \
  --tasks all \
  --selection_samples 64 \
  --num_permutations 4 \
  --chat_template_mode auto

# Evaluate each task with its selected adapter and aggregate the suite.
python scripts/eval_commonsense_tthns.py \
  --base_model /path/to/Llama-3.1-8B-Instruct \
  --adapters_root runs/Llama-3.1-8B-Instruct/commonsense170k/tthns-8plus2/seed42 \
  --output_dir eval/Llama-3.1-8B-Instruct/commonsense170k/tthns-8plus2-chat/seed42 \
  --tasks all \
  --backend vllm \
  --chat_template_mode auto
```

### Code TT-HNS (Qwen + Llama)

The code pipeline uses the same two-stage safety rule without reading HumanEval
solutions or tests. Magicoder responses supervise calibration localization;
HumanEval problem prompts then rank the fixed HNS directions using normalized
next-token entropy and consistency between two equivalent prompt wrappers. The
candidate is kept only when it improves the unlabeled objective inside a KL
trust region; otherwise the output adapter is restored to the original LoRA.

Set the real adapter directories from your runs, then launch either or both
model families sequentially:

```bash
QWEN_LORA_PATH=/path/to/qwen3-code-lora \
LLAMA_LORA_PATH=/path/to/llama31-code-lora \
MODEL_FAMILIES="qwen llama" \
bash scripts/run_code_tthns_qwen_llama.sh
```

Useful overrides include `QWEN_BASE_MODEL`, `LLAMA_BASE_MODEL`,
`MAGICODER_PATH`, `HUMANEVAL_PATH`, `CALIB_SAMPLES`, `SELECTION_SAMPLES`, and
the two `*_OUT_ROOT` variables. Qwen uses `chat_template_mode=non_thinking`;
Llama uses `auto`. The launcher writes the calibration adapter, final TT-HNS
adapter, LoRA/TT-HNS HumanEval results, and `test_time_hns_meta.json` under each
output root.

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
