from __future__ import annotations

import argparse
import json
from pathlib import Path

from finetune.utils import parse_csv_list


def _slug(s: str) -> str:
    return s.replace("/", "-").replace(":", "-")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a configs.jsonl sweep grid.")
    p.add_argument("--output", type=str, default="configs.jsonl")

    p.add_argument(
        "--base_models",
        type=str,
        default="meta-llama/Llama-2-7b-hf,mistralai/Mistral-7B-v0.1",
        help="Comma-separated HF model ids.",
    )
    p.add_argument("--tasks", type=str, default="math,code,alpaca,csqa")
    p.add_argument("--peft_methods", type=str, default="lora,pissa,adalora,loraplus")
    p.add_argument("--seeds", type=str, default="42")

    # Shared training args (written to each config).
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--bf16", action="store_true", help="Include --bf16 in configs.")
    p.add_argument("--fp16", action="store_true", help="Include --fp16 in configs.")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_qlora", action="store_true")

    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # LoRA / PiSSA
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--pissa_init_mode", type=str, default="pissa")

    # AdaLoRA
    p.add_argument("--init_r", type=int, default=12)
    p.add_argument("--target_r", type=int, default=8)

    # LoRA+
    p.add_argument("--loraplus_lr_ratio", type=float, default=20.0)
    p.add_argument("--loraplus_lr_embedding", type=float, default=None)

    p.add_argument(
        "--runs_root",
        type=str,
        default="runs",
        help="Root folder for output_dir entries in generated configs.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    base_models = parse_csv_list(args.base_models)
    tasks = parse_csv_list(args.tasks)
    peft_methods = parse_csv_list(args.peft_methods)
    seeds = [int(s) for s in parse_csv_list(args.seeds)]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shared = {
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_seq_len": args.max_seq_len,
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "use_qlora": bool(args.use_qlora),
        "target_modules": args.target_modules,
        "r": args.r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "pissa_init_mode": args.pissa_init_mode,
        "init_r": args.init_r,
        "target_r": args.target_r,
        "loraplus_lr_ratio": args.loraplus_lr_ratio,
        "loraplus_lr_embedding": args.loraplus_lr_embedding,
    }

    with out_path.open("w", encoding="utf-8") as f:
        for base_model in base_models:
            for task in tasks:
                for peft_method in peft_methods:
                    for seed in seeds:
                        model_slug = _slug(base_model)
                        out_dir = str(
                            Path(args.runs_root)
                            / model_slug
                            / task
                            / peft_method
                            / f"seed{seed}"
                        )
                        cfg = {
                            "base_model": base_model,
                            "task": task,
                            "peft_method": peft_method,
                            "seed": seed,
                            "output_dir": out_dir,
                            **shared,
                        }
                        f.write(json.dumps(cfg, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

