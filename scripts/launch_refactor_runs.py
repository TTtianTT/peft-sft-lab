#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TaskConfig:
    task: str
    profile: str
    lr: float
    per_device_train_batch_size: int
    max_train_samples: int | None


@dataclass(frozen=True)
class RunSpec:
    base_model: str
    model_slug: str
    task: str
    method: str
    profile: str
    rank: int
    seed: int
    lr: float
    per_device_train_batch_size: int
    max_train_samples: int | None
    output_dir: Path


DEFAULT_BASE_MODELS = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B",
]
DEFAULT_TASKS = ["math", "csqa", "code"]
DEFAULT_METHODS = ["lora", "loraplus"]
DEFAULT_RANKS = [16]
DEFAULT_SEEDS = [42]
DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs_refactor_data_20260121"

TASK_CONFIGS: dict[str, TaskConfig] = {
    "math": TaskConfig(
        task="math",
        profile="paper_math_ift_3ep",
        lr=1e-4,
        per_device_train_batch_size=2,
        max_train_samples=50000,
    ),
    "csqa": TaskConfig(
        task="csqa",
        profile="paper_csqa_3ep",
        lr=2e-4,
        per_device_train_batch_size=2,
        max_train_samples=None,
    ),
    "code": TaskConfig(
        task="code",
        profile="paper_code_ift_3ep",
        lr=2e-4,
        per_device_train_batch_size=1,
        max_train_samples=50000,
    ),
}

TASK_ALIASES = {
    "metamath": "math",
    "metamathqa": "math",
    "magicoder": "code",
    "commonsenseqa": "csqa",
}


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in _parse_csv_list(value):
        try:
            out.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid integer list item: {item!r}") from exc
    return out


def _slug_model(model_id: str) -> str:
    return model_id.replace("/", "-").replace(":", "-")


def _get_git_commit() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _normalize_tasks(tasks: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for task in tasks:
        key = TASK_ALIASES.get(task.strip().lower(), task.strip().lower())
        if key not in TASK_CONFIGS:
            known = ", ".join(sorted(TASK_CONFIGS))
            raise ValueError(f"Unknown task {task!r}. Known tasks: {known}")
        normalized.append(key)
    return normalized


def _iter_runs(
    *,
    base_models: list[str],
    tasks: list[str],
    methods: list[str],
    ranks: list[int],
    seeds: list[int],
    runs_root: Path,
) -> Iterable[RunSpec]:
    for method in methods:
        for base_model in base_models:
            model_slug = _slug_model(base_model)
            for task in tasks:
                cfg = TASK_CONFIGS[task]
                for rank in ranks:
                    for seed in seeds:
                        output_dir = (
                            runs_root
                            / model_slug
                            / cfg.task
                            / method
                            / f"profile-{cfg.profile}"
                            / f"rank-{rank}"
                            / f"seed{seed}"
                        )
                        yield RunSpec(
                            base_model=base_model,
                            model_slug=model_slug,
                            task=cfg.task,
                            method=method,
                            profile=cfg.profile,
                            rank=rank,
                            seed=seed,
                            lr=cfg.lr,
                            per_device_train_batch_size=cfg.per_device_train_batch_size,
                            max_train_samples=cfg.max_train_samples,
                            output_dir=output_dir,
                        )


def _build_command(run: RunSpec, num_processes: int, accelerate_config: str | None) -> list[str]:
    cmd = ["accelerate", "launch", "--num_processes", str(num_processes)]
    if accelerate_config:
        cmd += ["--config_file", accelerate_config]
    cmd += [
        "-m", "finetune.train_sft_peft",
        "--base_model", run.base_model,
        "--task", run.task,
        "--peft_method", run.method,
        "--output_dir", str(run.output_dir),
        "--train_profile", run.profile,
        "--per_device_train_batch_size", str(run.per_device_train_batch_size),
        "--lr", str(run.lr),
        "--r", str(run.rank),
        "--seed", str(run.seed),
    ]
    if run.max_train_samples is not None:
        cmd += ["--max_train_samples", str(run.max_train_samples)]
    return cmd


def _write_manifest(run: RunSpec, command: str, git_commit: str) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "git_commit": git_commit,
        "base_model_id": run.base_model,
        "task": run.task,
        "method": run.method,
        "profile": run.profile,
        "seed": run.seed,
        "rank": run.rank,
        "command": command,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = run.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch refactor-data training runs (LoRA first, then LoRA+)."
    )
    p.add_argument(
        "--base_models",
        type=str,
        default=",".join(DEFAULT_BASE_MODELS),
        help="Comma-separated HF model ids.",
    )
    p.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated tasks (math, csqa, code).",
    )
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods (lora, loraplus).",
    )
    p.add_argument(
        "--ranks",
        type=str,
        default=",".join(str(r) for r in DEFAULT_RANKS),
        help="Comma-separated LoRA ranks.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated random seeds.",
    )
    p.add_argument(
        "--runs_root",
        type=str,
        default=str(DEFAULT_RUNS_ROOT),
        help="Root output directory for runs.",
    )
    p.add_argument("--num_processes", type=int, default=8, help="Accelerate GPU processes.")
    p.add_argument("--accelerate_config", type=str, default=None, help="Optional accelerate config file.")
    p.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    p.add_argument("--continue_on_error", action="store_true", help="Continue after a failed run.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    base_models = _parse_csv_list(args.base_models)
    tasks = _normalize_tasks(_parse_csv_list(args.tasks))
    requested_methods = [m.strip().lower() for m in _parse_csv_list(args.methods)]
    unknown_methods = [m for m in requested_methods if m not in DEFAULT_METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}. Supported: {DEFAULT_METHODS}")
    methods = [m for m in DEFAULT_METHODS if m in requested_methods]
    if not methods:
        raise ValueError(f"No valid methods selected from: {requested_methods}")

    ranks = _parse_int_list(args.ranks)
    seeds = _parse_int_list(args.seeds)
    runs_root = Path(args.runs_root).resolve()

    run_specs = list(
        _iter_runs(
            base_models=base_models,
            tasks=tasks,
            methods=methods,
            ranks=ranks,
            seeds=seeds,
            runs_root=runs_root,
        )
    )

    total = len(run_specs)
    print(f"Planned runs: {total}")
    print(f"Runs root: {runs_root}")
    if args.dry_run:
        print("*** DRY RUN MODE ***")

    git_commit = _get_git_commit()

    for idx, run in enumerate(run_specs, 1):
        cmd = _build_command(run, args.num_processes, args.accelerate_config)
        cmd_str = shlex.join(cmd)
        print(f"[{idx}/{total}] {cmd_str}")

        if args.dry_run:
            continue

        try:
            _write_manifest(run, cmd_str, git_commit)
        except FileExistsError:
            raise RuntimeError(
                f"Output directory already exists: {run.output_dir}\n"
                "Refusing to reuse old runs. Pick a new --runs_root."
            )

        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"Run failed with exit_code={rc}")
            if not args.continue_on_error:
                return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
