from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT finetuning with PEFT (LoRA/PiSSA/AdaLoRA/LoRA+).")

    # Core
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="HF model id (e.g. meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1).",
    )
    parser.add_argument("--task", type=str, required=True, help="Task plugin name (e.g. math, code, alpaca, csqa).")
    parser.add_argument(
        "--peft_method",
        type=str,
        required=True,
        choices=["lora", "pissa", "adalora", "loraplus"],
        help="PEFT method: lora, pissa (PiSSA init), adalora, or loraplus (optimizer only).",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write adapter + logs.")

    # Training budget
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    # Optim
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Sequence / reproducibility
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    # Precision / memory
    mp = parser.add_mutually_exclusive_group()
    mp.add_argument("--bf16", action="store_true")
    mp.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_qlora", action="store_true", help="4-bit QLoRA (bitsandbytes).")

    # Modules
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules for LoRA-family adapters.",
    )

    # LoRA / PiSSA
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--pissa_init_mode",
        type=str,
        default="pissa",
        help='PiSSA init mode: "pissa" or "pissa_niter_16".',
    )

    # AdaLoRA
    parser.add_argument("--init_r", type=int, default=12)
    parser.add_argument("--target_r", type=int, default=8)

    # LoRA+
    parser.add_argument("--loraplus_lr_ratio", type=float, default=20.0)
    parser.add_argument(
        "--loraplus_lr_embedding",
        type=float,
        default=None,
        help="Optional base LR for embedding LoRA params (lora_embedding_A/B).",
    )

    return parser


def _maybe_enable_gradient_checkpointing(model: object, enabled: bool) -> None:
    if not enabled:
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def main() -> None:
    args = build_arg_parser().parse_args()

    from finetune.data.registry import get_task_plugin
    from finetune.peft_builders import (
        build_loraplus_param_groups,
        build_peft_config,
        check_peft_method_support,
    )
    from finetune.utils import (
        best_effort_pip_freeze,
        format_oom_hint,
        is_rank0,
        parse_csv_list,
        save_json,
        seed_everything,
        setup_logging,
    )

    output_dir = str(Path(args.output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("Starting run: base_model=%s task=%s peft_method=%s", args.base_model, args.task, args.peft_method)

    seed_everything(args.seed)

    if is_rank0():
        save_json(Path(output_dir) / "run_args.json", vars(args))
        best_effort_pip_freeze(output_dir)

    check_peft_method_support(args.peft_method)

    try:
        task = get_task_plugin(args.task)
        train_ds = task.load("train")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset for task={args.task}: {exc}\n"
            "Check that you have `datasets` installed and the dataset is available."
        ) from exc

    if len(train_ds) == 0:
        raise RuntimeError(f"Loaded empty dataset for task={args.task}.")

    # Map to a single text field for SFTTrainer.
    try:
        train_ds = train_ds.map(
            lambda ex: {"text": task.format_example(ex)},
            remove_columns=train_ds.column_names,
            desc=f"Formatting {args.task} examples",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed while formatting examples for task={args.task}: {exc}\n"
            f"Dataset columns were: {getattr(train_ds, 'column_names', None)}"
        ) from exc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch

    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    quantization_config = None
    device_map = None
    if args.use_qlora:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "QLoRA requested but BitsAndBytesConfig could not be imported.\n"
                "Install bitsandbytes (Linux + NVIDIA recommended):\n"
                "  pip install -U bitsandbytes\n"
                f"Original error: {exc}"
            ) from exc

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = {"": local_rank}

    logger.info("Loading model (qlora=%s)...", args.use_qlora)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype if not args.use_qlora else None,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    _maybe_enable_gradient_checkpointing(model, args.gradient_checkpointing)

    if args.use_qlora:
        try:
            from peft import prepare_model_for_kbit_training
        except Exception as exc:
            raise RuntimeError(
                "QLoRA requested but prepare_model_for_kbit_training is unavailable.\n"
                "Install/upgrade peft:\n  pip install -U peft\n"
                f"Original error: {exc}"
            ) from exc

        # Signature differs slightly between peft versions.
        try:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing
            )
        except TypeError:
            model = prepare_model_for_kbit_training(model)

    target_modules = parse_csv_list(args.target_modules)
    peft_config = build_peft_config(
        peft_method="lora" if args.peft_method == "loraplus" else args.peft_method,
        target_modules=target_modules,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        pissa_init_mode=args.pissa_init_mode,
        init_r=args.init_r,
        target_r=args.target_r,
        max_steps=args.max_steps,
    )

    try:
        from peft import get_peft_model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import peft.get_peft_model: {exc}\nInstall: pip install -U peft"
        ) from exc

    model = get_peft_model(model, peft_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    from transformers import TrainingArguments, get_scheduler

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=max(1, min(50, args.max_steps // 10)),
        save_strategy="no",
        evaluation_strategy="no",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    try:
        from trl import SFTTrainer
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import trl.SFTTrainer: {exc}\nInstall: pip install -U trl"
        ) from exc

    import inspect

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )
    sig = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        raise RuntimeError(
            "Unsupported trl.SFTTrainer signature (expected `tokenizer` or `processing_class`)."
        )

    optimizers = None
    if args.peft_method == "loraplus":
        grouping = build_loraplus_param_groups(
            model=model,
            lr=args.lr,
            lr_ratio=args.loraplus_lr_ratio,
            weight_decay=args.weight_decay,
            lr_embedding=args.loraplus_lr_embedding,
        )
        logger.info("LoRA+ param grouping: %s", grouping.summary)
        optimizer = torch.optim.AdamW(grouping.param_groups)
        warmup_steps = int(args.warmup_ratio * args.max_steps)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=args.max_steps,
        )
        optimizers = (optimizer, lr_scheduler)

    trainer = SFTTrainer(**trainer_kwargs, **({} if optimizers is None else {"optimizers": optimizers}))

    try:
        trainer.train()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda oom" in msg:
            logger.error(format_oom_hint())
        raise

    if is_rank0():
        logger.info("Saving adapter + tokenizer to %s", output_dir)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
