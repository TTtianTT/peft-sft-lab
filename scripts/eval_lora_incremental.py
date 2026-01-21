#!/usr/bin/env python3
"""
Incremental LoRA adapter evaluation script.

Rules:
1. Skip AdaLoRA entirely (paths containing /adalora/)
2. Skip checkpoint directories (checkpoint-*)
3. Only evaluate directories with adapter_model.safetensors
4. Deduplicate using existing results.csv (by adapter_dir)
5. Append new results to the same CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    eval_module: str
    metric_key: str
    max_new_tokens: int
    split: str | None = None
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaseModelSpec:
    model_id: str
    model_dir: str


@dataclass(frozen=True)
class AdapterSpec:
    adapter_dir: Path
    base_model_id: str
    base_model_dir: str
    task: str
    method: str
    profile: str
    rank: str
    seed: str


TASK_SPECS: list[TaskSpec] = [
    TaskSpec(
        name="csqa",
        eval_module="finetune.eval.eval_csqa",
        metric_key="accuracy",
        max_new_tokens=8,
    ),
    TaskSpec(
        name="metamath",
        eval_module="finetune.eval.eval_gsm8k",
        metric_key="accuracy_strict",
        max_new_tokens=256,
    ),
    TaskSpec(
        name="magicoder",
        eval_module="finetune.eval.eval_humaneval",
        metric_key="pass@1",
        max_new_tokens=256,
    ),
    TaskSpec(
        name="alpaca",
        eval_module="finetune.eval.eval_ifeval",
        metric_key="ifeval_strict_accuracy",
        max_new_tokens=256,
        split="train",
    ),
]

TASK_ORDER = {spec.name: idx for idx, spec in enumerate(TASK_SPECS)}
TASK_BY_NAME = {spec.name: spec for spec in TASK_SPECS}

BASE_MODELS: list[BaseModelSpec] = [
    BaseModelSpec(model_id="meta-llama/Llama-3.1-8B", model_dir="meta-llama-Llama-3.1-8B"),
    BaseModelSpec(model_id="mistralai/Mistral-7B-v0.3", model_dir="mistralai-Mistral-7B-v0.3"),
]

BASE_MODEL_DIR_TO_ID = {spec.model_dir: spec.model_id for spec in BASE_MODELS}
BASE_MODEL_ID_TO_DIR = {spec.model_id: spec.model_dir for spec in BASE_MODELS}

# Methods to evaluate (explicitly exclude adalora)
LORA_METHODS = {"lora", "loraplus", "pissa"}

TASK_ALIASES = {
    "gsm8k": "metamath",
    "humaneval": "magicoder",
    "ifeval": "alpaca",
    "csqa": "csqa",
    "math": "metamath",
    "code": "magicoder",
    "instruction": "alpaca",
    "instruction_following": "alpaca",
    "instruction-following": "alpaca",
    "commonsense": "csqa",
    "metamath": "metamath",
    "magicoder": "magicoder",
    "alpaca": "alpaca",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Incremental LoRA evaluation (skip adalora, checkpoints; dedupe & append)."
    )
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--results_csv", type=str, required=True, help="Path to results.csv (read existing, append new)")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true", help="Print what would be evaluated without running")
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_method(value: str | None) -> str:
    if not value:
        return ""
    v = value.strip().lower()
    aliases = {
        "lora+": "loraplus",
        "lora_plus": "loraplus",
        "lora-plus": "loraplus",
    }
    return aliases.get(v, v)


def _sanitize_tag(value: str) -> str:
    value = value.strip().replace(os.sep, "-")
    return "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in value).strip("-")


def _normalize_task(value: str | None) -> str:
    if not value:
        return ""
    raw = value.strip().lower()
    lookup = raw.replace("-", "_")
    return TASK_ALIASES.get(lookup, raw)


def _is_checkpoint_dir(path: Path) -> bool:
    """Check if any part of the path is a checkpoint directory."""
    for part in path.parts:
        if re.match(r"^checkpoint-\d+$", part):
            return True
    return False


def _is_adalora_path(path: Path) -> bool:
    """Check if path contains /adalora/."""
    return "/adalora/" in str(path) or "\\adalora\\" in str(path)


def _infer_adapter(adapter_dir: Path, runs_root: Path) -> tuple[AdapterSpec | None, list[str]]:
    errors: list[str] = []
    config_path = adapter_dir / "adapter_config.json"

    # Skip adalora paths
    if _is_adalora_path(adapter_dir):
        return None, []

    # Skip checkpoint directories
    if _is_checkpoint_dir(adapter_dir):
        return None, []

    try:
        rel_parts = adapter_dir.relative_to(runs_root).parts
    except ValueError:
        errors.append(f"adapter not under runs dir: {adapter_dir}")
        return None, errors

    base_model_dir = rel_parts[0] if len(rel_parts) >= 1 else ""
    task = rel_parts[1] if len(rel_parts) >= 2 else ""
    method = rel_parts[2] if len(rel_parts) >= 3 else ""

    profile = ""
    rank = ""
    seed = ""
    for part in rel_parts[3:]:
        if part.startswith(("profile-", "profile_")):
            profile = part
        elif part.startswith(("rank-", "rank_")):
            rank = part
        elif part.startswith("seed"):
            seed = part

    run_args = _read_json(adapter_dir / "run_args.json")
    config = _read_json(config_path) if config_path.exists() else {}
    base_model_id = str(config.get("base_model_name_or_path") or run_args.get("base_model") or "").strip()
    target_by_dir = base_model_dir in BASE_MODEL_DIR_TO_ID
    target_by_id = base_model_id in BASE_MODEL_ID_TO_DIR

    if not config_path.exists():
        if target_by_dir or target_by_id:
            errors.append(f"missing adapter_config.json: {adapter_dir}")
            return None, errors
        return None, []

    if not target_by_dir and not target_by_id:
        return None, []

    # Double-check peft_type is not ADALORA
    peft_type = str(config.get("peft_type", "")).upper()
    if peft_type == "ADALORA":
        return None, []

    if not task:
        task = str(run_args.get("task", "")).strip()
    task = _normalize_task(task)
    if not method:
        method = str(run_args.get("peft_method", "")).strip()
    method = _normalize_method(method)

    if not profile:
        profile_val = str(run_args.get("train_profile", "")).strip()
        if profile_val:
            profile = f"profile-{profile_val}"
    if not rank and run_args.get("r") is not None:
        rank = f"rank-{run_args['r']}"
    if not seed and run_args.get("seed") is not None:
        seed = f"seed{run_args['seed']}"

    if not base_model_id and base_model_dir:
        base_model_id = BASE_MODEL_DIR_TO_ID.get(base_model_dir, "")
    if not base_model_dir and base_model_id:
        base_model_dir = BASE_MODEL_ID_TO_DIR.get(base_model_id, "")

    if not task:
        errors.append(f"missing task in path or run_args.json: {adapter_dir}")
    if not method:
        errors.append(f"missing method in path or run_args.json: {adapter_dir}")
    if not base_model_id:
        errors.append(f"missing base_model id in adapter_config.json: {adapter_dir}")
    if not base_model_dir:
        errors.append(f"missing base_model dir for adapter: {adapter_dir}")

    if errors:
        return None, errors

    if method not in LORA_METHODS:
        return None, []

    if task not in TASK_BY_NAME:
        errors.append(f"unknown task '{task}' for adapter: {adapter_dir}")
        return None, errors

    profile = profile or "profile-unknown"
    rank = rank or "rank-unknown"
    seed = seed or "seed-unknown"

    return (
        AdapterSpec(
            adapter_dir=adapter_dir,
            base_model_id=base_model_id,
            base_model_dir=base_model_dir,
            task=task,
            method=method,
            profile=profile,
            rank=rank,
            seed=seed,
        ),
        [],
    )


def discover_adapters(runs_root: Path) -> list[AdapterSpec]:
    """Discover all valid adapters (excluding adalora and checkpoints)."""
    errors: list[str] = []
    adapters: list[AdapterSpec] = []
    for model_path in sorted(runs_root.rglob("adapter_model.safetensors")):
        adapter_dir = model_path.parent
        spec, spec_errors = _infer_adapter(adapter_dir, runs_root)
        if spec_errors:
            errors.extend(spec_errors)
            continue
        if spec is None:
            continue
        adapters.append(spec)

    if errors:
        msg = "Adapter discovery errors:\n" + "\n".join(f"- {err}" for err in errors)
        raise RuntimeError(msg)
    return adapters


def load_existing_results(results_csv: Path) -> set[str]:
    """Load existing adapter_dir values from results.csv for deduplication."""
    completed: set[str] = set()
    if not results_csv.exists():
        return completed

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            adapter_dir = row.get("adapter_dir", "").strip()
            if adapter_dir:
                # Normalize path for comparison
                completed.add(str(Path(adapter_dir).resolve()))
    return completed


def git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True)
        return out.strip()
    except Exception:
        return ""


def progress_bar(idx: int, total: int, label: str) -> str:
    width = 28
    filled = int(width * idx / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {idx}/{total} {label}"


def run_eval(
    *,
    task: TaskSpec,
    base_model_id: str,
    adapter_dir: Path | None,
    output_dir: Path,
    tensor_parallel_size: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"
    cmd = [
        sys.executable,
        "-m",
        task.eval_module,
        "--base_model",
        base_model_id,
        "--output_dir",
        str(output_dir),
        "--use_vllm",
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--max_new_tokens",
        str(task.max_new_tokens),
        "--seed",
        str(seed),
    ]
    if task.split:
        cmd.extend(["--split", task.split])
    if task.extra_args:
        cmd.extend(task.extra_args)
    if adapter_dir is not None:
        cmd.extend(["--adapter_dir", str(adapter_dir)])

    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluation failed: {' '.join(cmd)} (see {log_path})")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    runs_root = (root / args.runs_dir).resolve()
    results_csv = Path(args.results_csv).resolve()
    out_dir = results_csv.parent

    # Discover all valid adapters
    print("Discovering adapters...")
    adapters = discover_adapters(runs_root)
    print(f"Found {len(adapters)} valid adapters (excluding adalora and checkpoints)")

    # Load existing results for deduplication
    print("Loading existing results for deduplication...")
    completed_adapters = load_existing_results(results_csv)
    print(f"Found {len(completed_adapters)} already evaluated adapter paths")

    # Filter out already completed adapters
    pending_adapters = [
        a for a in adapters
        if str(a.adapter_dir.resolve()) not in completed_adapters
    ]
    print(f"Pending: {len(pending_adapters)} adapters to evaluate")

    if args.dry_run:
        print("\n=== DRY RUN: Would evaluate the following adapters ===")
        for adapter in pending_adapters:
            print(f"  {adapter.method:10} | {adapter.task:10} | {adapter.adapter_dir}")
        return 0

    if not pending_adapters:
        print("Nothing new to evaluate. All adapters already in results.csv.")
        return 0

    # Sort by task order, then method, profile, rank, seed
    pending_adapters.sort(
        key=lambda x: (
            x.base_model_id,
            TASK_ORDER.get(x.task, 999),
            x.method,
            x.profile,
            x.rank,
            x.seed,
            str(x.adapter_dir),
        )
    )

    commit = git_commit(root)
    temperature = 0.0
    top_p = 1.0

    fieldnames = [
        "base_model_id",
        "base_model_dir",
        "task",
        "adapter_dir",
        "method",
        "profile",
        "rank",
        "seed",
        "metric_name",
        "metric_value",
        "num_examples",
        "max_new_tokens",
        "temperature",
        "top_p",
        "seed_value",
        "tensor_parallel_size",
        "backend",
        "output_dir",
        "git_commit",
    ]

    # Determine if we need to write header (new file)
    write_header = not results_csv.exists() or results_csv.stat().st_size == 0

    total_runs = len(pending_adapters)
    completed = 0

    # Open in append mode
    with results_csv.open("a", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for adapter in pending_adapters:
            task = TASK_BY_NAME[adapter.task]
            completed += 1
            label = f"{adapter.base_model_id} | {task.name} | {adapter.method}"
            print(progress_bar(completed, total_runs, label))

            output_dir_path = (
                out_dir
                / "runs"
                / task.name
                / adapter.base_model_dir
                / _sanitize_tag(adapter.method)
                / _sanitize_tag(adapter.profile)
                / _sanitize_tag(adapter.rank)
                / _sanitize_tag(adapter.seed)
            )

            try:
                run_eval(
                    task=task,
                    base_model_id=adapter.base_model_id,
                    adapter_dir=adapter.adapter_dir,
                    output_dir=output_dir_path,
                    tensor_parallel_size=args.tensor_parallel_size,
                    seed=args.seed,
                )
            except RuntimeError as e:
                print(f"  ERROR: {e}")
                continue

            metrics_path = output_dir_path / "metrics.json"
            metrics = _read_json(metrics_path)
            if task.metric_key not in metrics:
                print(f"  WARNING: missing metric '{task.metric_key}' in {metrics_path}")
                continue

            writer.writerow(
                {
                    "base_model_id": adapter.base_model_id,
                    "base_model_dir": adapter.base_model_dir,
                    "task": task.name,
                    "adapter_dir": str(adapter.adapter_dir),
                    "method": adapter.method,
                    "profile": adapter.profile,
                    "rank": adapter.rank,
                    "seed": adapter.seed,
                    "metric_name": task.metric_key,
                    "metric_value": metrics.get(task.metric_key),
                    "num_examples": metrics.get("total"),
                    "max_new_tokens": task.max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed_value": args.seed,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "backend": "vllm",
                    "output_dir": str(output_dir_path),
                    "git_commit": commit,
                }
            )
            csv_f.flush()

    print(f"\nFinished {completed} evaluations. Results appended to: {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
