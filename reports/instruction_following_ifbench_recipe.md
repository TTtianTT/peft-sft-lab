# Instruction Following / IFBench Recipe

This repo now supports:

- SFT training on `allenai/tulu-3-sft-personas-instruction-following`
- Official-score evaluation on `allenai/IFBench_test`

## Training Data Format

The `if` task accepts three input shapes.

### 1. Native Tulu format

```json
{
  "id": "personas_IF_xxx",
  "prompt": "Provide a short answer ...",
  "messages": [
    {"role": "user", "content": "Provide a short answer ..."},
    {"role": "assistant", "content": "Here is the answer ..."}
  ],
  "constraints": ["length constraints:number of words"]
}
```

This is the format used by `allenai/tulu-3-sft-personas-instruction-following`. The loader splits the last assistant message into the supervised completion.

### 2. Repo-native chat SFT format

```json
{
  "prompt": [
    {"role": "user", "content": "Provide a short answer ..."}
  ],
  "completion": [
    {"role": "assistant", "content": "Here is the answer ..."}
  ]
}
```

Use this if you want a normalized local format consistent with the new math-data pipeline.

### 3. Simple single-turn fallback

```json
{
  "prompt": "Provide a short answer ...",
  "response": "Here is the answer ..."
}
```

`output` and `answer` are also accepted in place of `response`.

## Recommended First Run

For your first Llama-3.1-8B-Instruct run, use:

- task: `if`
- train dataset: `allenai/tulu-3-sft-personas-instruction-following`
- eval dataset: `allenai/IFBench_test`
- train profile: `paper_if_tulu_2ep`
- epochs: `2`
- max sequence length: `4096`
- metric to watch: `ifbench_prompt_loose_accuracy`

Example:

```bash
accelerate launch --num_processes 8 -m finetune.train_sft_peft \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --task if \
  --peft_method lora \
  --output_dir runs/meta-llama-Llama-3.1-8B-Instruct/if/lora/profile-paper_if_tulu_2ep/rank-16/seed42 \
  --train_profile paper_if_tulu_2ep \
  --per_device_train_batch_size 1 \
  --global_train_batch_size 128 \
  --lr 2e-4 \
  --r 16 \
  --bf16 \
  --gradient_checkpointing \
  --seed 42
```

`save_strategy="epoch"` is already enabled in the training pipeline, so you will automatically get one checkpoint at the end of epoch 1 and one at the end of epoch 2 under `checkpoint-*`. The final adapter is also saved to the run root.

## Official IFBench Evaluation

Clone the official evaluator once:

```bash
git clone https://github.com/allenai/IFBench.git external/IFBench
```

Then evaluate any adapter or checkpoint:

```bash
python -m finetune.eval.eval_ifbench \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter_dir runs/meta-llama-Llama-3.1-8B-Instruct/if/lora/profile-paper_if_tulu_2ep/rank-16/seed42/checkpoint-<epoch1_or_epoch2_step> \
  --official_eval_root external/IFBench \
  --output_dir eval/ifbench-llama31-if-lora-epoch1 \
  --use_vllm \
  --tensor_parallel_size 8 \
  --max_new_tokens 2048
```

The evaluator writes:

- `responses.jsonl`: generated `prompt/response` pairs
- `official_ifbench/input_data.jsonl`: exported IFBench prompt file
- `official_ifbench/input_response_data.jsonl`: exported response file
- `official_ifbench/input_response_data-eval_results_strict.jsonl`: official strict results
- `official_ifbench/input_response_data-eval_results_loose.jsonl`: official loose results
- `metrics.json`: summarized metrics for this repo

`metrics.json` uses the official scorer outputs and exposes:

- `ifbench_prompt_loose_accuracy`
- `ifbench_prompt_strict_accuracy`
- `ifbench_instruction_loose_accuracy`
- `ifbench_instruction_strict_accuracy`

The paper-level number to compare first is `ifbench_prompt_loose_accuracy`.

## Optional Wrapper

`scripts/train_with_eval.py` now supports this setup too. For IFBench runs, pass `--official_eval_root` and set a realistic `--eval_max_new_tokens`, for example:

```bash
python scripts/train_with_eval.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --task if \
  --peft_method lora \
  --output_dir runs/meta-llama-Llama-3.1-8B-Instruct/if/lora/profile-paper_if_tulu_2ep/rank-16/seed42 \
  --train_profile paper_if_tulu_2ep \
  --official_eval_root external/IFBench \
  --eval_max_new_tokens 2048 \
  --tensor_parallel_size 8
```
