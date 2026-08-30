#!/usr/bin/env python3
"""Evaluate task-specific test-time HNS adapters and aggregate eight-task metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_commonsense_8tasks import TASKS, _parse_tasks  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapters_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument(
        "--chat_template_mode",
        choices=("auto", "thinking", "non_thinking"),
        default="auto",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--request_batch_size", type=int, default=256)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_model_len", type=int, default=2048)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--vllm_attention_backend", default="FLASH_ATTN")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    selected_tasks = _parse_tasks(args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_results = {}

    for task_name in selected_tasks:
        adapter_dir = args.adapters_root / task_name
        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing task-specific adapter: {adapter_dir}")
        task_output = args.output_dir / task_name
        summary_path = task_output / "summary.json"
        if not (args.skip_existing and summary_path.is_file()):
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "eval_commonsense_8tasks.py"),
                "--base_model",
                args.base_model,
                "--adapter_dir",
                str(adapter_dir),
                "--output_dir",
                str(task_output),
                "--tasks",
                task_name,
                "--backend",
                args.backend,
                "--chat_template_mode",
                args.chat_template_mode,
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--request_batch_size",
                str(args.request_batch_size),
                "--tensor_parallel_size",
                str(args.tensor_parallel_size),
                "--vllm_max_model_len",
                str(args.vllm_max_model_len),
                "--vllm_gpu_memory_utilization",
                str(args.vllm_gpu_memory_utilization),
                "--vllm_attention_backend",
                args.vllm_attention_backend,
                "--dtype",
                args.dtype,
                "--seed",
                str(args.seed),
            ]
            if args.max_samples is not None:
                command.extend(["--max_samples", str(args.max_samples)])
            print(f"\n[{task_name}] {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)

        task_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        task_results[task_name] = task_summary["tasks"][task_name]

    macro = sum(result["accuracy"] for result in task_results.values()) / len(task_results)
    correct = sum(result["correct"] for result in task_results.values())
    total = sum(result["total"] for result in task_results.values())
    summary = {
        "method": "calibration_importance_plus_test_time_hns",
        "base_model": args.base_model,
        "adapters_root": str(args.adapters_root),
        "chat_template_mode": args.chat_template_mode,
        "macro_accuracy": macro,
        "micro_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "tasks": task_results,
        "seed": args.seed,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("task", "accuracy", "correct", "total", "invalid", "split"))
        for task_name, result in task_results.items():
            writer.writerow(
                (
                    task_name,
                    result["accuracy"],
                    result["correct"],
                    result["total"],
                    result["invalid"],
                    result["split"],
                )
            )
        writer.writerow(("macro_average", macro, "", "", "", ""))

    print("\nTask                 Accuracy     Correct/Total  Invalid")
    print("-" * 62)
    for task_name, result in task_results.items():
        print(
            f"{task_name:<20} {result['accuracy']:>9.6f}  "
            f"{result['correct']:>7}/{result['total']:<7}  {result['invalid']:>7}"
        )
    print("-" * 62)
    print(f"{'macro_average':<20} {macro:>9.6f}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
