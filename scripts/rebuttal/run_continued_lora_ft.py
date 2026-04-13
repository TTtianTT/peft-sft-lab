#!/usr/bin/env python3
"""
P0-A: Continued LoRA Fine-Tuning Baseline.

Loads a pre-trained LoRA adapter, continues fine-tuning ALL LoRA parameters
on calibration data (same data used by spectral editing), then saves the
updated adapter for downstream evaluation.

This answers the reviewer question: "Why not just continue fine-tuning
on the calibration data instead of doing spectral surgery?"

Usage:
    python scripts/rebuttal/run_continued_lora_ft.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_dir /path/to/lora/adapter \
        --out_dir /path/to/output \
        --task math \
        --calib_samples 128 \
        --steps 50 \
        --lr 5e-5 \
        --seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src/ to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.spectral_edit.calib import (
    build_calib_formatter,
    load_calibration_split,
    make_calib_batch,
    sample_calibration_examples,
)

# ============================================================================
# Constants
# ============================================================================

TASK_TO_CALIB_DATASET = {
    "math": ("gsm8k", "main"),
    "metamath": ("gsm8k", "main"),
    "code": ("ise-uiuc/Magicoder-Evol-Instruct-110K", None),
    "magicoder": ("ise-uiuc/Magicoder-Evol-Instruct-110K", None),
    "alpaca": ("tatsu-lab/alpaca", None),
    "csqa": ("tau/commonsense_qa", None),
    "commonsense_qa": ("tau/commonsense_qa", None),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def continued_lora_ft(args) -> Dict:
    """Run continued LoRA fine-tuning on calibration data."""
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Requires CUDA GPU.")

    # ---- Load model + adapter ----
    print(f"[Info] Loading base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
        cache_dir=args.cache_dir,
    ).to(device)

    print(f"[Info] Loading LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=True).to(device)
    model.config.use_cache = False
    if getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[Info] Gradient checkpointing enabled")
    model.train()

    # Freeze base, unfreeze LoRA
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Trainable parameters: {trainable_params:,}")

    # ---- Load calibration data ----
    calib_dataset = args.calib_dataset
    calib_config = args.calib_config
    if calib_dataset is None:
        task_key = args.task.lower()
        if task_key not in TASK_TO_CALIB_DATASET:
            raise ValueError(f"Unknown task '{args.task}'. Provide --calib_dataset explicitly.")
        calib_dataset, calib_config = TASK_TO_CALIB_DATASET[task_key]

    formatter, _ = build_calib_formatter(calib_dataset, args.calib_text_fields)
    ds_split = load_calibration_split(
        calib_dataset, calib_config, args.calib_split, cache_dir=args.cache_dir
    )

    calib_seed = args.calib_seed if args.calib_seed is not None else args.seed
    calib_examples = sample_calibration_examples(
        ds_split, args.calib_samples, args.calib_shuffle, calib_seed, args.calib_start
    )
    ncal = len(calib_examples)
    print(f"[Info] Calibration examples: {ncal}")

    # ---- Held-out proxy split ----
    proxy_examples = []
    if args.proxy_samples > 0:
        proxy_examples = sample_calibration_examples(
            ds_split, args.proxy_samples, args.calib_shuffle, calib_seed,
            calib_start=args.calib_samples,  # offset past edit set
        )
        print(f"[Info] Proxy (held-out) examples: {len(proxy_examples)}")

    # ---- Build batches ----
    bs = max(1, args.calib_batch_size)
    batches = []
    for i in range(0, ncal, bs):
        batch_ex = calib_examples[i : i + bs]
        input_ids, attn_mask, labels = make_calib_batch(tok, batch_ex, formatter, add_eos=True)
        batches.append((input_ids.to(device), attn_mask.to(device), labels.to(device)))

    if not batches:
        raise RuntimeError("No calibration batches built.")

    # ---- Compute pre-training losses ----
    def _compute_loss(examples):
        total_l, n_b = 0.0, 0
        model.eval()
        with torch.no_grad():
            for i in range(0, len(examples), bs):
                b_ex = examples[i : i + bs]
                ids, mask, lab = make_calib_batch(tok, b_ex, formatter, add_eos=True)
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=lab.to(device))
                total_l += float(out.loss.item())
                n_b += 1
        model.train()
        return total_l / max(1, n_b)

    edit_loss_before = _compute_loss(calib_examples)
    proxy_loss_before = _compute_loss(proxy_examples) if proxy_examples else None
    print(f"[Pre-train] edit_loss={edit_loss_before:.4f}", end="")
    if proxy_loss_before is not None:
        print(f", proxy_loss={proxy_loss_before:.4f}")
    else:
        print()

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- Training loop ----
    total_steps = args.steps
    loss_history = []
    t0 = time.time()

    print(f"[Train] Starting continued FT: {total_steps} steps, lr={args.lr}, bs={bs}")

    for step in range(1, total_steps + 1):
        # Cycle through batches
        batch_idx = (step - 1) % len(batches)
        input_ids, attn_mask, labels = batches[batch_idx]

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )

        optimizer.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if step % max(1, total_steps // 10) == 0 or step == total_steps or step == 1:
            elapsed = time.time() - t0
            print(f"  [Step {step}/{total_steps}] loss={loss_val:.4f}  elapsed={elapsed:.1f}s")

    elapsed_total = time.time() - t0
    print(f"[Train] Done. Final loss={loss_history[-1]:.4f}, total time={elapsed_total:.1f}s")

    # ---- Save adapter ----
    os.makedirs(args.out_dir, exist_ok=True)
    # Use PEFT's save method to get proper adapter files
    model.save_pretrained(args.out_dir)
    print(f"[Save] Adapter saved to: {args.out_dir}")

    # ---- Compute post-training losses ----
    edit_loss_after = _compute_loss(calib_examples)
    proxy_loss_after = _compute_loss(proxy_examples) if proxy_examples else None
    print(f"[Post-train] edit_loss={edit_loss_after:.4f}", end="")
    if proxy_loss_after is not None:
        print(f", proxy_loss={proxy_loss_after:.4f}")
    else:
        print()

    # ---- Save metadata ----
    meta = {
        "method": "continued_lora_ft",
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir),
        "task": args.task,
        "calib_dataset": calib_dataset,
        "calib_config": calib_config,
        "calib_split": args.calib_split,
        "calib_samples": args.calib_samples,
        "calib_samples_used": ncal,
        "calib_batch_size": bs,
        "calib_seed": calib_seed,
        "calib_start": args.calib_start,
        "proxy_samples": len(proxy_examples),
        "steps": total_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "trainable_params": trainable_params,
        "loss_initial": loss_history[0] if loss_history else None,
        "loss_final": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "edit_loss_before": edit_loss_before,
        "edit_loss_after": edit_loss_after,
        "proxy_loss_before": proxy_loss_before,
        "proxy_loss_after": proxy_loss_after,
        "elapsed_seconds": elapsed_total,
    }

    with open(os.path.join(args.out_dir, "continued_ft_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Cleanup ----
    try:
        model.to("cpu")
        base.to("cpu")
    except Exception:
        pass
    del model, base, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continued LoRA Fine-Tuning Baseline (P0-A)")

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True,
                        help="Task name: math, code, alpaca, csqa")

    # Training hyperparameters
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Calibration data
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--calib_batch_size", type=int, default=2)
    parser.add_argument("--calib_dataset", type=str, default=None)
    parser.add_argument("--calib_config", type=str, default=None)
    parser.add_argument("--calib_split", type=str, default="train")
    parser.add_argument("--calib_text_fields", type=str, nargs="*", default=None)
    parser.add_argument("--calib_shuffle", action="store_true")
    parser.add_argument("--calib_seed", type=int, default=None)
    parser.add_argument("--calib_start", type=int, default=0)

    # Proxy (held-out) evaluation
    parser.add_argument("--proxy_samples", type=int, default=128,
                        help="Number of held-out proxy examples (0 to disable)")

    # Gradient checkpointing (for code task OOM)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce memory")

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    continued_lora_ft(args)


if __name__ == "__main__":
    main()
