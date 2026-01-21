#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
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

LORA_METHODS = {"lora", "loraplus", "adalora", "pissa"}

# Methods that vLLM does NOT support (will use transformers backend instead)
VLLM_UNSUPPORTED_METHODS = {"adalora"}

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
    p = argparse.ArgumentParser(description="Evaluate all base+LoRA runs with vLLM (strict order).")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=42)
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


def _infer_adapter(adapter_dir: Path, runs_root: Path) -> tuple[AdapterSpec | None, list[str]]:
    errors: list[str] = []
    config_path = adapter_dir / "adapter_config.json"

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
    use_vllm: bool = True,
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
        "--max_new_tokens",
        str(task.max_new_tokens),
        "--seed",
        str(seed),
    ]
    if use_vllm:
        cmd.extend(["--use_vllm", "--tensor_parallel_size", str(tensor_parallel_size)])
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
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    adapters = discover_adapters(runs_root)
    adapters_by_base: dict[str, list[AdapterSpec]] = {spec.model_id: [] for spec in BASE_MODELS}
    skipped = 0
    for adapter in adapters:
        if adapter.base_model_id in adapters_by_base:
            adapters_by_base[adapter.base_model_id].append(adapter)
        else:
            skipped += 1

    for base_id, items in adapters_by_base.items():
        items.sort(
            key=lambda x: (
                TASK_ORDER.get(x.task, 999),
                x.method,
                x.profile,
                x.rank,
                x.seed,
                str(x.adapter_dir),
            )
        )

    results_path = out_dir / "results.csv"
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

    total_runs = 0
    for base_spec in BASE_MODELS:
        total_runs += len(TASK_SPECS) + len(adapters_by_base.get(base_spec.model_id, []))

    completed = 0
    with results_path.open("w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        for base_spec in BASE_MODELS:
            base_out_root = out_dir / "runs"
            for task in TASK_SPECS:
                completed += 1
                label = f"{base_spec.model_id} | {task.name} | base"
                print(progress_bar(completed, total_runs, label))
                output_dir = base_out_root / task.name / base_spec.model_dir / "base"
                run_eval(
                    task=task,
                    base_model_id=base_spec.model_id,
                    adapter_dir=None,
                    output_dir=output_dir,
                    tensor_parallel_size=args.tensor_parallel_size,
                    seed=args.seed,
                )
                metrics_path = output_dir / "metrics.json"
                metrics = _read_json(metrics_path)
                if task.metric_key not in metrics:
                    raise RuntimeError(f"missing metric '{task.metric_key}' in {metrics_path}")

                writer.writerow(
                    {
                        "base_model_id": base_spec.model_id,
                        "base_model_dir": base_spec.model_dir,
                        "task": task.name,
                        "adapter_dir": "",
                        "method": "base",
                        "profile": "",
                        "rank": "",
                        "seed": "",
                        "metric_name": task.metric_key,
                        "metric_value": metrics.get(task.metric_key),
                        "num_examples": metrics.get("total"),
                        "max_new_tokens": task.max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "seed_value": args.seed,
                        "tensor_parallel_size": args.tensor_parallel_size,
                        "backend": "vllm",
                        "output_dir": str(output_dir),
                        "git_commit": commit,
                    }
                )
                csv_f.flush()

            for adapter in adapters_by_base.get(base_spec.model_id, []):
                task = TASK_BY_NAME[adapter.task]
                completed += 1
                # Fall back to transformers backend for methods vLLM doesn't support
                adapter_use_vllm = adapter.method not in VLLM_UNSUPPORTED_METHODS
                backend_label = "vllm" if adapter_use_vllm else "transformers"
                label = f"{base_spec.model_id} | {task.name} | {adapter.method} ({backend_label})"
                print(progress_bar(completed, total_runs, label))
                output_dir = (
                    base_out_root
                    / task.name
                    / base_spec.model_dir
                    / _sanitize_tag(adapter.method)
                    / _sanitize_tag(adapter.profile)
                    / _sanitize_tag(adapter.rank)
                    / _sanitize_tag(adapter.seed)
                )
                run_eval(
                    task=task,
                    base_model_id=base_spec.model_id,
                    adapter_dir=adapter.adapter_dir,
                    output_dir=output_dir,
                    tensor_parallel_size=args.tensor_parallel_size,
                    seed=args.seed,
                    use_vllm=adapter_use_vllm,
                )
                metrics_path = output_dir / "metrics.json"
                metrics = _read_json(metrics_path)
                if task.metric_key not in metrics:
                    raise RuntimeError(f"missing metric '{task.metric_key}' in {metrics_path}")

                writer.writerow(
                    {
                        "base_model_id": base_spec.model_id,
                        "base_model_dir": base_spec.model_dir,
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
                        "backend": backend_label,
                        "output_dir": str(output_dir),
                        "git_commit": commit,
                    }
                )
                csv_f.flush()

    print(f"Finished {completed} runs. Results: {results_path}")
    if skipped:
        print(f"Skipped {skipped} adapter(s) for non-target base models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
