from __future__ import annotations

import argparse
import inspect
import math
import os
from pathlib import Path
from typing import Any


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
    parser.add_argument(
        "--train_profile",
        type=str,
        default=None,
        help="Recipe profile (e.g. paper_code_ift, paper_math_ift, paper_default_ift, or auto).",
    )

    # Training budget
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument(
        "--global_train_batch_size",
        type=int,
        default=None,
        help="If set, overrides gradient_accumulation_steps based on world size.",
    )

    # Optim
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--min_lr_ratio", type=float, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=None, help="Max grad norm (clip).")

    # Sequence / reproducibility
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="If set, downsample the training set after a deterministic shuffle.",
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=42,
        help="Seed for deterministic dataset shuffling before downsampling.",
    )

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
    parser.add_argument("--adalora_total_steps", type=int, default=1500)
    parser.add_argument("--adalora_init_warmup_steps", type=int, default=500)
    parser.add_argument("--adalora_final_warmup_steps", type=int, default=500)
    parser.add_argument("--adalora_deltaT", type=int, default=None)

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


def _format_sft_text(example: dict[str, Any]) -> str:
    text = example.get("text", "")
    if isinstance(text, str):
        return text
    return str(text)


def _get_world_size() -> int:
    for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS", "PMI_SIZE"):
        value = os.environ.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return 1


def _get_scheduler_func(lr_scheduler_type: str):
    try:
        from transformers import optimization
    except Exception:
        return None

    mapping = getattr(optimization, "TYPE_TO_SCHEDULER_FUNCTION", None)
    if mapping is not None:
        func = None
        if hasattr(optimization, "SchedulerType"):
            try:
                key = optimization.SchedulerType(lr_scheduler_type)
            except Exception:
                key = None
            if key is not None:
                func = mapping.get(key)
        if func is None:
            func = mapping.get(lr_scheduler_type)
        if func is not None:
            return func

    func_name = f"get_{lr_scheduler_type}_schedule_with_warmup"
    return getattr(optimization, func_name, None)


def _resolve_min_lr_kwargs(
    lr_scheduler_type: str,
    min_lr_ratio: float | None,
    base_lr: float | None = None,
) -> dict[str, float]:
    if min_lr_ratio is None:
        return {}
    func = _get_scheduler_func(lr_scheduler_type)
    if func is None:
        return {}
    params = inspect.signature(func).parameters
    if "min_lr_ratio" in params:
        return {"min_lr_ratio": min_lr_ratio}
    if "min_lr_rate" in params:
        return {"min_lr_rate": min_lr_ratio}
    if "min_lr" in params and base_lr is not None:
        return {"min_lr": base_lr * min_lr_ratio}
    return {}


def _effective_global_batch_size(
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> int:
    return max(1, per_device_train_batch_size) * max(1, gradient_accumulation_steps) * max(1, world_size)


def _estimate_steps_per_epoch(
    dataset_size: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> int:
    effective_gbs = _effective_global_batch_size(
        per_device_train_batch_size, gradient_accumulation_steps, world_size
    )
    return max(1, math.ceil(dataset_size / effective_gbs))


def _resolve_adalora_schedule(
    *,
    total_steps: int,
    init_warmup_steps: int,
    final_warmup_steps: int,
    delta_t: int | None,
) -> dict[str, int]:
    if total_steps <= 0:
        raise ValueError("--adalora_total_steps must be > 0.")
    if init_warmup_steps < 0 or final_warmup_steps < 0:
        raise ValueError("--adalora_init_warmup_steps and --adalora_final_warmup_steps must be >= 0.")
    if init_warmup_steps + final_warmup_steps > total_steps:
        raise ValueError("AdaLoRA warmup steps exceed total_steps; pruning window would be negative.")

    tinit = init_warmup_steps
    tfinal = final_warmup_steps
    budget_steps = total_steps - tinit - tfinal
    if budget_steps <= 0:
        raise ValueError("AdaLoRA warmup steps leave no budgeting window; decrease warmups or increase total_steps.")
    resolved_delta = delta_t if delta_t is not None else max(1, int(0.01 * total_steps))
    resolved_delta = min(resolved_delta, budget_steps)
    return {
        "total_step": total_steps,
        "tinit": tinit,
        "tfinal": tfinal,
        "deltaT": resolved_delta,
    }


def main() -> None:
    args = build_arg_parser().parse_args()

    from finetune.data.base import first_present
    from finetune.data.registry import get_task_plugin
    from finetune.peft_builders import (
        build_loraplus_param_groups,
        build_peft_config,
        check_peft_method_support,
    )
    from finetune.train_profiles import pick_profile, profile_overrides
    from finetune.utils import (
        best_effort_pip_freeze,
        format_oom_hint,
        is_rank0,
        parse_csv_list,
        save_json,
        seed_everything,
        setup_logging,
    )

    profile = pick_profile(args.task, args.train_profile)
    if profile:
        for key, value in profile_overrides(profile).items():
            setattr(args, key, value)
        args.train_profile = profile.name

    if not args.lr_scheduler_type:
        args.lr_scheduler_type = "linear"
    if args.min_lr_ratio is not None and args.lr_scheduler_type == "linear":
        args.lr_scheduler_type = "cosine_with_min_lr"

    output_dir = str(Path(args.output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("Starting run: base_model=%s task=%s peft_method=%s", args.base_model, args.task, args.peft_method)
    if profile:
        logger.info("Applied train_profile=%s", profile.name)

    seed_everything(args.seed)

    if args.global_train_batch_size is not None:
        world_size = _get_world_size()
        denom = args.per_device_train_batch_size * world_size
        if denom <= 0:
            raise ValueError("per_device_train_batch_size * world_size must be > 0.")
        target = args.global_train_batch_size
        grad_accum = target / denom
        grad_accum_int = max(1, math.ceil(grad_accum))
        if grad_accum_int != grad_accum and is_rank0():
            logger.warning(
                "global_train_batch_size=%s not divisible by per_device_train_batch_size*world_size=%s; "
                "using gradient_accumulation_steps=%s.",
                target,
                denom,
                grad_accum_int,
            )
        args.gradient_accumulation_steps = grad_accum_int
        if is_rank0():
            effective = grad_accum_int * denom
            logger.info(
                "Computed gradient_accumulation_steps=%s for global_train_batch_size=%s (effective=%s).",
                grad_accum_int,
                target,
                effective,
            )

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

    if args.task == "alpaca":
        def _normalize_field(value):
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            return str(value).strip()

        def _alpaca_is_valid(example):
            text = example.get("text")
            if isinstance(text, str) and text.strip():
                return True
            instruction = _normalize_field(example.get("instruction"))
            output = _normalize_field(example.get("output"))
            return bool(instruction) and bool(output)

        before_count = len(train_ds)
        train_ds = train_ds.filter(_alpaca_is_valid, desc="Filtering alpaca examples")
        after_count = len(train_ds)
        filtered = before_count - after_count
        if is_rank0():
            if filtered:
                logger.info(
                    "Filtered %d invalid alpaca examples (kept %d/%d).",
                    filtered,
                    after_count,
                    before_count,
                )
            else:
                logger.info("No invalid alpaca examples filtered (kept %d).", after_count)
        if after_count == 0:
            raise RuntimeError("All alpaca examples were filtered out; check dataset fields.")

    if args.task in {"metamath", "metamathqa", "math"}:
        def _metamath_is_valid(example):
            instruction = first_present(
                example,
                ["query", "original_question", "question", "instruction", "prompt"],
            )
            response = first_present(example, ["response", "answer", "output", "solution"])
            return bool(instruction) and bool(response)

        before_count = len(train_ds)
        train_ds = train_ds.filter(_metamath_is_valid, desc="Filtering metamath examples")
        after_count = len(train_ds)
        filtered = before_count - after_count
        if is_rank0():
            if filtered:
                logger.info(
                    "Filtered %d invalid metamath examples (kept %d/%d).",
                    filtered,
                    after_count,
                    before_count,
                )
            else:
                logger.info("No invalid metamath examples filtered (kept %d).", after_count)
        if after_count == 0:
            raise RuntimeError("All metamath examples were filtered out; check dataset fields.")

    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max_train_samples must be > 0.")
        before_count = len(train_ds)
        if before_count > args.max_train_samples:
            train_ds = train_ds.shuffle(seed=args.dataset_seed)
            train_ds = train_ds.select(range(args.max_train_samples))
            after_count = len(train_ds)
            if is_rank0():
                logger.info(
                    "Downsampled train dataset from %d to %d using seed=%d.",
                    before_count,
                    after_count,
                    args.dataset_seed,
                )
        else:
            if is_rank0():
                logger.info(
                    "max_train_samples=%d >= dataset size (%d); no downsampling applied.",
                    args.max_train_samples,
                    before_count,
                )

    adalora_schedule = None
    if args.peft_method == "adalora":
        adalora_schedule = _resolve_adalora_schedule(
            total_steps=args.adalora_total_steps,
            init_warmup_steps=args.adalora_init_warmup_steps,
            final_warmup_steps=args.adalora_final_warmup_steps,
            delta_t=args.adalora_deltaT,
        )
        if args.max_steps != adalora_schedule["total_step"]:
            logger.info(
                "Overriding max_steps from %s to %s for AdaLoRA schedule.",
                args.max_steps,
                adalora_schedule["total_step"],
            )
            args.max_steps = adalora_schedule["total_step"]

        world_size = _get_world_size()
        denom = max(1, args.per_device_train_batch_size * world_size)
        target_epochs = args.num_train_epochs if args.num_train_epochs is not None else 3.0
        desired_effective_gbs = (len(train_ds) * target_epochs) / args.max_steps
        grad_accum = max(1, math.ceil(desired_effective_gbs / denom))
        if grad_accum != args.gradient_accumulation_steps:
            logger.info(
                "Adjusting gradient_accumulation_steps from %s to %s for AdaLoRA %s-step schedule.",
                args.gradient_accumulation_steps,
                grad_accum,
                args.max_steps,
            )
            args.gradient_accumulation_steps = grad_accum
        args.global_train_batch_size = None

    world_size = _get_world_size()
    effective_gbs = _effective_global_batch_size(
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        world_size,
    )
    steps_per_epoch = _estimate_steps_per_epoch(
        len(train_ds),
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        world_size,
    )
    if args.peft_method == "adalora":
        effective_epochs = (args.max_steps * effective_gbs) / len(train_ds)
        logger.info(
            "Dataset size=%d | effective_global_batch_size=%d | steps_per_epoch=%d | max_steps=%d | "
            "approx_epochs=%.3f",
            len(train_ds),
            effective_gbs,
            steps_per_epoch,
            args.max_steps,
            effective_epochs,
        )
        if adalora_schedule:
            budget_steps = (
                adalora_schedule["total_step"]
                - adalora_schedule["tinit"]
                - adalora_schedule["tfinal"]
            )
            logger.info(
                "AdaLoRA schedule: total_step=%d, tinit=%d, tfinal=%d, deltaT=%d, budget_steps=%d.",
                adalora_schedule["total_step"],
                adalora_schedule["tinit"],
                adalora_schedule["tfinal"],
                adalora_schedule["deltaT"],
                budget_steps,
            )
    else:
        if args.num_train_epochs is not None:
            estimated_total_steps = max(1, math.ceil(steps_per_epoch * args.num_train_epochs))
            logger.info(
                "Dataset size=%d | effective_global_batch_size=%d | steps_per_epoch=%d | "
                "num_train_epochs=%.3f | est_total_steps=%d",
                len(train_ds),
                effective_gbs,
                steps_per_epoch,
                args.num_train_epochs,
                estimated_total_steps,
            )
        else:
            logger.info(
                "Dataset size=%d | effective_global_batch_size=%d | steps_per_epoch=%d | max_steps=%d",
                len(train_ds),
                effective_gbs,
                steps_per_epoch,
                args.max_steps,
            )

    if is_rank0():
        save_json(Path(output_dir) / "run_args.json", vars(args))
        best_effort_pip_freeze(output_dir)

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
        adalora_total_steps=adalora_schedule["total_step"] if adalora_schedule else None,
        adalora_tinit=adalora_schedule["tinit"] if adalora_schedule else None,
        adalora_tfinal=adalora_schedule["tfinal"] if adalora_schedule else None,
        adalora_deltaT=adalora_schedule["deltaT"] if adalora_schedule else None,
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

    use_max_steps = args.peft_method == "adalora" or args.num_train_epochs is None
    if use_max_steps:
        total_steps_for_logging = max(1, args.max_steps)
    else:
        total_steps_for_logging = max(1, math.ceil(steps_per_epoch * args.num_train_epochs))

    training_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=max(1, min(50, total_steps_for_logging // 10)),
        save_strategy="epoch",
        save_safetensors=True,
        evaluation_strategy="no",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    if use_max_steps:
        training_kwargs["max_steps"] = args.max_steps
    else:
        training_kwargs["num_train_epochs"] = args.num_train_epochs
    if args.grad_clip is not None:
        training_kwargs["max_grad_norm"] = args.grad_clip
    min_lr_kwargs = _resolve_min_lr_kwargs(
        args.lr_scheduler_type,
        args.min_lr_ratio,
        base_lr=args.lr,
    )
    if args.min_lr_ratio is not None and not min_lr_kwargs and is_rank0():
        logger.warning(
            "min_lr_ratio=%s ignored because scheduler %s does not support min_lr_* kwargs.",
            args.min_lr_ratio,
            args.lr_scheduler_type,
        )
    if min_lr_kwargs:
        training_kwargs["lr_scheduler_kwargs"] = min_lr_kwargs
    training_sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" not in training_sig.parameters and "eval_strategy" in training_sig.parameters:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    training_kwargs = {k: v for k, v in training_kwargs.items() if k in training_sig.parameters}
    training_args = TrainingArguments(**training_kwargs)
    if is_rank0():
        save_json(Path(output_dir) / "training_args.json", training_args.to_dict())

    try:
        from trl import SFTTrainer
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import trl.SFTTrainer: {exc}\nInstall: pip install -U trl"
        ) from exc

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
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
    if "dataset_text_field" in sig.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in sig.parameters:
        trainer_kwargs["formatting_func"] = _format_sft_text
    if "max_seq_length" in sig.parameters:
        trainer_kwargs["max_seq_length"] = args.max_seq_len
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in sig.parameters}

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
        optimizer = torch.optim.AdamW(
            grouping.param_groups,
            betas=(args.adam_beta1, args.adam_beta2),
        )
        warmup_steps = int(args.warmup_ratio * args.max_steps)
        scheduler_specific_kwargs = dict(min_lr_kwargs) if min_lr_kwargs else None
        lr_scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=args.max_steps,
            scheduler_specific_kwargs=scheduler_specific_kwargs,
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
