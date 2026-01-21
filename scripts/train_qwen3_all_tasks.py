#!/usr/bin/env python3
"""
Batch training script for Qwen3-8B on all 4 tasks with LoRA and LoRA+.

Runs 8 training jobs total (4 tasks × 2 methods) with:
- Pre-training evaluation on base model
- Training with checkpoint saving
- Post-training evaluation on trained adapter
- Results logged to CSV

Usage:
    python scripts/train_qwen3_all_tasks.py --tensor_parallel_size 8 --num_processes 8

    # Dry run to see what will be executed
    python scripts/train_qwen3_all_tasks.py --dry_run

    # Run specific tasks/methods only
    python scripts/train_qwen3_all_tasks.py --tasks csqa metamath --methods lora
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskConfig:
    name: str
    profile: str
    eval_module: str
    metric_key: str
    max_new_tokens: int
    eval_split: str | None = None


TASKS: dict[str, TaskConfig] = {
    "csqa": TaskConfig(
        name="csqa",
        profile="paper_csqa_3ep",
        eval_module="finetune.eval.eval_csqa",
        metric_key="accuracy",
        max_new_tokens=8,
    ),
    "metamath": TaskConfig(
        name="metamath",
        profile="paper_math_ift_3ep",
        eval_module="finetune.eval.eval_gsm8k",
        metric_key="accuracy_strict",
        max_new_tokens=256,
    ),
    "magicoder": TaskConfig(
        name="magicoder",
        profile="paper_code_ift_3ep",
        eval_module="finetune.eval.eval_humaneval",
        metric_key="pass@1",
        max_new_tokens=256,
    ),
    "alpaca": TaskConfig(
        name="alpaca",
        profile="paper_alpaca_3ep",
        eval_module="finetune.eval.eval_ifeval",
        metric_key="ifeval_strict_accuracy",
        max_new_tokens=256,
        eval_split="train",
    ),
}

METHODS = ["lora", "loraplus"]
BASE_MODEL = "Qwen/Qwen3-8B"
BASE_MODEL_DIR = "Qwen-Qwen3-8B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Qwen3-8B on all tasks with LoRA/LoRA+")
    p.add_argument("--base_model", type=str, default=BASE_MODEL)
    p.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                   choices=list(TASKS.keys()), help="Tasks to run")
    p.add_argument("--methods", nargs="+", default=METHODS,
                   choices=METHODS, help="Methods to run")
    p.add_argument("--tensor_parallel_size", type=int, default=8)
    p.add_argument("--num_processes", type=int, default=8)
    p.add_argument("--r", type=int, default=16, help="LoRA rank")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--results_csv", type=str, default="eval_results/qwen3_8b_pre_post.csv")
    p.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    p.add_argument("--skip_pre_eval", action="store_true", help="Skip pre-training evaluation")
    p.add_argument("--skip_post_eval", action="store_true", help="Skip post-training evaluation")
    p.add_argument("--skip_training", action="store_true", help="Skip training (eval only)")
    p.add_argument("--no_vllm", action="store_true", help="Use transformers instead of vLLM")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--continue_from", type=str, default=None,
                   help="Continue from task:method (e.g., 'metamath:lora')")
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sanitize_model_dir(model_id: str) -> str:
    """Convert model ID to directory name."""
    return model_id.replace("/", "-")


def run_evaluation(
    *,
    base_model: str,
    adapter_dir: str | None,
    output_dir: Path,
    task_config: TaskConfig,
    tensor_parallel_size: int,
    seed: int,
    use_vllm: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Run evaluation and return metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"

    cmd = [
        sys.executable, "-m", task_config.eval_module,
        "--base_model", base_model,
        "--output_dir", str(output_dir),
        "--max_new_tokens", str(task_config.max_new_tokens),
        "--seed", str(seed),
    ]

    if use_vllm:
        cmd.extend(["--use_vllm", "--tensor_parallel_size", str(tensor_parallel_size)])

    if task_config.eval_split:
        cmd.extend(["--split", task_config.eval_split])

    if adapter_dir is not None:
        cmd.extend(["--adapter_dir", adapter_dir])

    print(f"  CMD: {' '.join(cmd)}")

    if dry_run:
        return {"dry_run": True}

    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"  ERROR: Evaluation failed! Check: {log_path}")
        return {"error": f"exit code {proc.returncode}"}

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        return _read_json(metrics_path)
    return {"error": "metrics.json not found"}


def run_training(
    *,
    base_model: str,
    task: str,
    method: str,
    output_dir: str,
    profile: str,
    num_processes: int,
    per_device_train_batch_size: int,
    r: int,
    seed: int,
    dry_run: bool,
) -> int:
    """Run training and return exit code."""
    cmd = [
        "accelerate", "launch",
        "--num_processes", str(num_processes),
        "-m", "finetune.train_sft_peft",
        "--base_model", base_model,
        "--task", task,
        "--peft_method", method,
        "--output_dir", output_dir,
        "--train_profile", profile,
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--r", str(r),
        "--seed", str(seed),
        "--bf16",
    ]

    print(f"  CMD: {' '.join(cmd)}")

    if dry_run:
        return 0

    proc = subprocess.run(cmd)
    return proc.returncode


def append_to_csv(
    csv_path: Path,
    row: dict[str, Any],
) -> None:
    """Append a row to the results CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "base_model",
        "task",
        "method",
        "profile",
        "rank",
        "seed",
        "run_dir",
        "pre_metric_name",
        "pre_metric_value",
        "pre_total",
        "post_metric_name",
        "post_metric_value",
        "post_total",
        "delta",
        "delta_pct",
        "training_status",
    ]

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_summary(
    task: str,
    method: str,
    pre_metrics: dict[str, Any],
    post_metrics: dict[str, Any],
    metric_key: str,
) -> tuple[float | None, float | None]:
    """Print summary and return (delta, delta_pct)."""
    print(f"\n  {'='*50}")
    print(f"  SUMMARY: {task} / {method}")
    print(f"  {'='*50}")

    pre_val = pre_metrics.get(metric_key)
    post_val = post_metrics.get(metric_key)

    if pre_val is not None:
        print(f"  Pre-training ({metric_key}):  {pre_val:.4f}")
    else:
        print(f"  Pre-training: {pre_metrics}")

    if post_val is not None:
        print(f"  Post-training ({metric_key}): {post_val:.4f}")
    else:
        print(f"  Post-training: {post_metrics}")

    delta = None
    delta_pct = None
    if pre_val is not None and post_val is not None:
        delta = post_val - pre_val
        delta_pct = (delta / pre_val * 100) if pre_val != 0 else 0
        print(f"  Delta: {delta:+.4f} ({delta_pct:+.2f}%)")

    print(f"  {'='*50}\n")
    return delta, delta_pct


def main() -> int:
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    results_csv = Path(args.results_csv)
    use_vllm = not args.no_vllm
    model_dir = _sanitize_model_dir(args.base_model)

    # Build list of (task, method) pairs
    pairs = [(t, m) for t in args.tasks for m in args.methods]

    # Handle --continue_from
    if args.continue_from:
        try:
            cont_task, cont_method = args.continue_from.split(":")
            skip_until = (cont_task, cont_method)
            new_pairs = []
            found = False
            for p in pairs:
                if p == skip_until:
                    found = True
                if found:
                    new_pairs.append(p)
            if not found:
                print(f"ERROR: --continue_from '{args.continue_from}' not found in run list")
                return 1
            pairs = new_pairs
            print(f"Continuing from {args.continue_from}, {len(pairs)} runs remaining")
        except ValueError:
            print("ERROR: --continue_from format should be 'task:method' (e.g., 'metamath:lora')")
            return 1

    total = len(pairs)
    print(f"\n{'='*60}")
    print(f"  Qwen3-8B Training: {total} runs")
    print(f"  Tasks: {args.tasks}")
    print(f"  Methods: {args.methods}")
    print(f"  Results CSV: {results_csv}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("*** DRY RUN MODE ***\n")

    # Cache for base model pre-eval (avoid re-running for same task)
    base_eval_cache: dict[str, dict[str, Any]] = {}

    for idx, (task_name, method) in enumerate(pairs, 1):
        task_config = TASKS[task_name]

        print(f"\n{'#'*60}")
        print(f"# [{idx}/{total}] Task: {task_name} | Method: {method}")
        print(f"# Profile: {task_config.profile}")
        print(f"{'#'*60}\n")

        # Output directory
        output_dir = (
            runs_dir / model_dir / task_name / method
            / f"profile-{task_config.profile}" / f"rank-{args.r}" / f"seed{args.seed}"
        )

        # --- Phase 1: Pre-training evaluation ---
        pre_metrics: dict[str, Any] = {}
        if not args.skip_pre_eval:
            print(f"[{idx}/{total}] Phase 1: Pre-training evaluation (base model)")

            # Check cache
            if task_name in base_eval_cache:
                print(f"  Using cached base model evaluation for {task_name}")
                pre_metrics = base_eval_cache[task_name]
            else:
                pre_eval_dir = output_dir / "eval_pre"
                pre_metrics = run_evaluation(
                    base_model=args.base_model,
                    adapter_dir=None,
                    output_dir=pre_eval_dir,
                    task_config=task_config,
                    tensor_parallel_size=args.tensor_parallel_size,
                    seed=args.seed,
                    use_vllm=use_vllm,
                    dry_run=args.dry_run,
                )
                base_eval_cache[task_name] = pre_metrics

                if not args.dry_run:
                    print(f"  Pre-eval result: {task_config.metric_key}={pre_metrics.get(task_config.metric_key)}")

        # --- Phase 2: Training ---
        training_status = "skipped"
        if not args.skip_training:
            print(f"\n[{idx}/{total}] Phase 2: Training {method.upper()}")

            rc = run_training(
                base_model=args.base_model,
                task=task_name,
                method=method,
                output_dir=str(output_dir),
                profile=task_config.profile,
                num_processes=args.num_processes,
                per_device_train_batch_size=args.per_device_train_batch_size,
                r=args.r,
                seed=args.seed,
                dry_run=args.dry_run,
            )

            if args.dry_run:
                training_status = "dry_run"
            elif rc != 0:
                print(f"  ERROR: Training failed with exit code {rc}")
                training_status = "failed"
            else:
                training_status = "success"
                print(f"  Training completed successfully!")

        # --- Phase 3: Post-training evaluation ---
        post_metrics: dict[str, Any] = {}
        if not args.skip_post_eval and training_status in ("success", "dry_run", "skipped"):
            print(f"\n[{idx}/{total}] Phase 3: Post-training evaluation (with adapter)")

            adapter_path = output_dir / "adapter_model.safetensors"
            if args.dry_run or adapter_path.exists() or args.skip_training:
                post_eval_dir = output_dir / "eval_post"
                post_metrics = run_evaluation(
                    base_model=args.base_model,
                    adapter_dir=str(output_dir),
                    output_dir=post_eval_dir,
                    task_config=task_config,
                    tensor_parallel_size=args.tensor_parallel_size,
                    seed=args.seed,
                    use_vllm=use_vllm,
                    dry_run=args.dry_run,
                )

                if not args.dry_run:
                    print(f"  Post-eval result: {task_config.metric_key}={post_metrics.get(task_config.metric_key)}")
            else:
                print(f"  Skipping post-eval: adapter not found at {adapter_path}")
                post_metrics = {"error": "adapter not found"}

        # --- Summary and CSV logging ---
        if not args.dry_run:
            delta, delta_pct = print_summary(
                task_name, method, pre_metrics, post_metrics, task_config.metric_key
            )

            # Save summary JSON
            summary = {
                "timestamp": datetime.now().isoformat(),
                "base_model": args.base_model,
                "task": task_name,
                "method": method,
                "profile": task_config.profile,
                "rank": args.r,
                "seed": args.seed,
                "pre_eval": pre_metrics,
                "post_eval": post_metrics,
                "training_status": training_status,
            }
            summary_path = output_dir / "eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

            # Append to CSV
            csv_row = {
                "timestamp": datetime.now().isoformat(),
                "base_model": args.base_model,
                "task": task_name,
                "method": method,
                "profile": task_config.profile,
                "rank": args.r,
                "seed": args.seed,
                "run_dir": str(output_dir),
                "pre_metric_name": task_config.metric_key,
                "pre_metric_value": pre_metrics.get(task_config.metric_key),
                "pre_total": pre_metrics.get("total"),
                "post_metric_name": task_config.metric_key,
                "post_metric_value": post_metrics.get(task_config.metric_key),
                "post_total": post_metrics.get("total"),
                "delta": delta,
                "delta_pct": delta_pct,
                "training_status": training_status,
            }
            append_to_csv(results_csv, csv_row)
            print(f"  Results appended to: {results_csv}")

    print(f"\n{'='*60}")
    print(f"  ALL RUNS COMPLETED!")
    print(f"  Results: {results_csv}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
