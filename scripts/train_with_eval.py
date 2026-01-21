#!/usr/bin/env python3
"""
Training wrapper with pre/post evaluation hooks.

Runs:
1. Pre-training evaluation on base model (no adapter)
2. LoRA training using the existing training pipeline
3. Post-training evaluation on the trained adapter

Example usage:
    python scripts/train_with_eval.py \
        --base_model Qwen/Qwen3-8B \
        --task csqa \
        --peft_method lora \
        --output_dir runs/Qwen-Qwen3-8B/csqa/lora/profile-paper_csqa_3ep/rank-16/seed42 \
        --train_profile paper_csqa_3ep \
        --tensor_parallel_size 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LoRA with pre/post CSQA evaluation."
    )
    # Core training args (passed through to train_sft_peft.py)
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--peft_method", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--train_profile", type=str, default=None)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    # Evaluation args
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--eval_max_new_tokens", type=int, default=8)
    p.add_argument("--skip_pre_eval", action="store_true", help="Skip pre-training evaluation")
    p.add_argument("--skip_post_eval", action="store_true", help="Skip post-training evaluation")
    p.add_argument("--skip_training", action="store_true", help="Skip training (eval only)")
    p.add_argument("--use_vllm", action="store_true", default=True, help="Use vLLM for evaluation (default: True)")
    p.add_argument("--no_vllm", action="store_true", help="Use transformers instead of vLLM for eval")

    # Training args passthrough
    p.add_argument("--num_processes", type=int, default=8, help="Number of GPUs for training")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--global_train_batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true")

    return p.parse_args()


def run_evaluation(
    *,
    base_model: str,
    adapter_dir: str | None,
    output_dir: Path,
    task: str,
    tensor_parallel_size: int,
    max_new_tokens: int,
    seed: int,
    use_vllm: bool,
) -> dict[str, Any]:
    """Run evaluation and return metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"

    # Map task to eval module
    eval_modules = {
        "csqa": "finetune.eval.eval_csqa",
        "commonsenseqa": "finetune.eval.eval_csqa",
        "metamath": "finetune.eval.eval_gsm8k",
        "gsm8k": "finetune.eval.eval_gsm8k",
        "magicoder": "finetune.eval.eval_humaneval",
        "humaneval": "finetune.eval.eval_humaneval",
        "alpaca": "finetune.eval.eval_ifeval",
        "ifeval": "finetune.eval.eval_ifeval",
    }

    eval_module = eval_modules.get(task.lower())
    if not eval_module:
        raise ValueError(f"Unknown task for evaluation: {task}")

    cmd = [
        sys.executable, "-m", eval_module,
        "--base_model", base_model,
        "--output_dir", str(output_dir),
        "--max_new_tokens", str(max_new_tokens),
        "--seed", str(seed),
    ]

    if use_vllm:
        cmd.extend(["--use_vllm", "--tensor_parallel_size", str(tensor_parallel_size)])

    if adapter_dir is not None:
        cmd.extend(["--adapter_dir", adapter_dir])

    print(f"\n{'='*60}")
    print(f"Running evaluation: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"Evaluation failed! Check log: {log_path}")
        # Try to read any partial metrics
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            return json.loads(metrics_path.read_text())
        return {"error": f"evaluation failed with code {proc.returncode}"}

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {"error": "metrics.json not found"}


def run_training(
    *,
    base_model: str,
    task: str,
    peft_method: str,
    output_dir: str,
    train_profile: str | None,
    num_processes: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int | None,
    global_train_batch_size: int,
    lr: float,
    num_train_epochs: float | None,
    max_steps: int | None,
    r: int,
    seed: int,
    bf16: bool,
    gradient_checkpointing: bool,
) -> int:
    """Run training and return exit code."""
    cmd = [
        "accelerate", "launch",
        "--num_processes", str(num_processes),
        "-m", "finetune.train_sft_peft",
        "--base_model", base_model,
        "--task", task,
        "--peft_method", peft_method,
        "--output_dir", output_dir,
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--global_train_batch_size", str(global_train_batch_size),
        "--lr", str(lr),
        "--r", str(r),
        "--seed", str(seed),
    ]

    if train_profile:
        cmd.extend(["--train_profile", train_profile])

    if gradient_accumulation_steps is not None:
        cmd.extend(["--gradient_accumulation_steps", str(gradient_accumulation_steps)])

    if num_train_epochs is not None:
        cmd.extend(["--num_train_epochs", str(num_train_epochs)])
    elif max_steps is not None:
        cmd.extend(["--max_steps", str(max_steps)])
    # If neither is specified, let the profile handle it

    if bf16:
        cmd.append("--bf16")

    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")

    print(f"\n{'='*60}")
    print(f"Running training: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    proc = subprocess.run(cmd)
    return proc.returncode


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    """Pretty-print metrics to stdout."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*60}\n")


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_vllm = args.use_vllm and not args.no_vllm

    # Collect all results
    results = {
        "base_model": args.base_model,
        "task": args.task,
        "peft_method": args.peft_method,
        "train_profile": args.train_profile,
        "r": args.r,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    # 1. Pre-training evaluation (base model, no adapter)
    if not args.skip_pre_eval:
        print("\n" + "="*60)
        print("  PHASE 1: Pre-training evaluation (base model)")
        print("="*60)

        pre_eval_dir = output_dir / "eval_pre"
        pre_metrics = run_evaluation(
            base_model=args.base_model,
            adapter_dir=None,
            output_dir=pre_eval_dir,
            task=args.task,
            tensor_parallel_size=args.tensor_parallel_size,
            max_new_tokens=args.eval_max_new_tokens,
            seed=args.seed,
            use_vllm=use_vllm,
        )
        results["pre_eval"] = pre_metrics
        print_metrics("PRE-TRAINING EVALUATION (Base Model)", pre_metrics)

    # 2. Training
    if not args.skip_training:
        print("\n" + "="*60)
        print("  PHASE 2: Training")
        print("="*60)

        train_rc = run_training(
            base_model=args.base_model,
            task=args.task,
            peft_method=args.peft_method,
            output_dir=str(output_dir),
            train_profile=args.train_profile,
            num_processes=args.num_processes,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            global_train_batch_size=args.global_train_batch_size,
            lr=args.lr,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            r=args.r,
            seed=args.seed,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
        )

        if train_rc != 0:
            print(f"\nTraining failed with exit code {train_rc}")
            results["training_status"] = "failed"
            results["training_exit_code"] = train_rc
            # Save partial results
            (output_dir / "eval_summary.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False)
            )
            return train_rc

        results["training_status"] = "success"
        print("\nTraining completed successfully!")

    # 3. Post-training evaluation (with adapter)
    if not args.skip_post_eval:
        print("\n" + "="*60)
        print("  PHASE 3: Post-training evaluation (with adapter)")
        print("="*60)

        # Check adapter exists
        adapter_path = output_dir / "adapter_model.safetensors"
        if not adapter_path.exists():
            print(f"Warning: adapter not found at {adapter_path}")
            if args.skip_training:
                print("Skipping post-eval since no adapter exists and training was skipped")
            else:
                results["post_eval"] = {"error": "adapter not found"}
        else:
            post_eval_dir = output_dir / "eval_post"
            post_metrics = run_evaluation(
                base_model=args.base_model,
                adapter_dir=str(output_dir),
                output_dir=post_eval_dir,
                task=args.task,
                tensor_parallel_size=args.tensor_parallel_size,
                max_new_tokens=args.eval_max_new_tokens,
                seed=args.seed,
                use_vllm=use_vllm,
            )
            results["post_eval"] = post_metrics
            print_metrics("POST-TRAINING EVALUATION (With Adapter)", post_metrics)

    # 4. Summary
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)

    if "pre_eval" in results and "post_eval" in results:
        pre = results["pre_eval"]
        post = results["post_eval"]

        # Find the main metric (accuracy for CSQA)
        metric_keys = ["accuracy", "accuracy_strict", "pass@1", "ifeval_strict_accuracy"]
        for mk in metric_keys:
            if mk in pre and mk in post:
                pre_val = pre[mk]
                post_val = post[mk]
                delta = post_val - pre_val
                delta_pct = (delta / pre_val * 100) if pre_val != 0 else 0
                print(f"  {mk}:")
                print(f"    Base model:    {pre_val:.4f}")
                print(f"    With adapter:  {post_val:.4f}")
                print(f"    Delta:         {delta:+.4f} ({delta_pct:+.2f}%)")
                break

    # Save summary
    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to: {summary_path}")
    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
