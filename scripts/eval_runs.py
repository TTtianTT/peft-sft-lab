#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TaskSpec:
    task_dir: str
    dataset: str
    eval_script: Path
    primary_metric: str
    metric_order: tuple[str, ...]
    supports_split: bool
    supports_timeout: bool


BASE_MODEL_DIRS = [
    "meta-llama-Llama-3.1-8B",
    "mistralai-Mistral-7B-v0.3",
]

BASE_MODEL_ID_FALLBACK = {
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "mistralai-Mistral-7B-v0.3": "mistralai/Mistral-7B-v0.3",
}


TASK_SPECS = {
    "csqa": TaskSpec(
        task_dir="csqa",
        dataset="csqa",
        eval_script=Path("src/finetune/eval/eval_csqa.py"),
        primary_metric="accuracy",
        metric_order=("accuracy", "correct", "total"),
        supports_split=True,
        supports_timeout=False,
    ),
    "metamath": TaskSpec(
        task_dir="metamath",
        dataset="gsm8k",
        eval_script=Path("src/finetune/eval/eval_gsm8k.py"),
        primary_metric="accuracy_strict",
        metric_order=("accuracy_strict", "correct", "total"),
        supports_split=True,
        supports_timeout=False,
    ),
    "magicoder": TaskSpec(
        task_dir="magicoder",
        dataset="humaneval",
        eval_script=Path("src/finetune/eval/eval_humaneval.py"),
        primary_metric="pass@1",
        metric_order=("pass@1", "passed", "total"),
        supports_split=False,
        supports_timeout=True,
    ),
    "alpaca": TaskSpec(
        task_dir="alpaca",
        dataset="ifeval",
        eval_script=Path("src/finetune/eval/eval_ifeval.py"),
        primary_metric="ifeval_strict_accuracy",
        metric_order=("ifeval_strict_accuracy", "ifeval_avg_score", "avg_checks_per_example", "total"),
        supports_split=True,
        supports_timeout=False,
    ),
}

TASK_ALIASES = {
    "gsm8k": "metamath",
    "humaneval": "magicoder",
    "ifeval": "alpaca",
    "csqa": "csqa",
    "math": "metamath",
    "code": "magicoder",
    "instruction": "alpaca",
    "instruction_following": "alpaca",
    "commonsense": "csqa",
}


@dataclass
class AdapterInfo:
    adapter_dir: Path
    method: str
    rank: str
    seed: str
    profile: str


@dataclass
class RunRequest:
    task_spec: TaskSpec
    task_dir: str
    base_model_dir: str
    base_model_id: str
    method: str
    rank: str
    seed: str
    profile: str
    adapter_dir: Path | None


@dataclass
class RunResult:
    data: dict[str, Any]


ADAPTER_MODEL_FILES = [
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
]


def parse_list(val: str | None) -> list[str]:
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


def normalize_rank(value: str) -> str:
    value = value.strip()
    if value.startswith("rank-") or value.startswith("rank_"):
        return value[5:]
    return value


def normalize_seed(value: str) -> str:
    value = value.strip()
    if value.startswith("seed-") or value.startswith("seed_"):
        return value[5:]
    if value.startswith("seed"):
        return value[4:]
    return value


def sanitize_tag(text: str) -> str:
    text = text.strip().replace(os.sep, "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_base_model_id(runs_root: Path, base_model_dir: str) -> str:
    search_root = runs_root / base_model_dir
    for cfg in search_root.rglob("run_config.json"):
        data = read_json(cfg)
        base_model = data.get("base_model")
        if isinstance(base_model, str) and base_model:
            return base_model
    return BASE_MODEL_ID_FALLBACK.get(base_model_dir, base_model_dir)


def discover_task_dirs(runs_root: Path, base_model_dir: str) -> list[str]:
    base_dir = runs_root / base_model_dir
    if not base_dir.is_dir():
        return []
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir()])


def is_adapter_dir(path: Path) -> bool:
    if not (path / "adapter_config.json").is_file():
        return False
    for name in ADAPTER_MODEL_FILES:
        if (path / name).is_file():
            return True
    return False


def parse_adapter_metadata(adapter_dir: Path, task_root: Path) -> AdapterInfo:
    rel = adapter_dir.relative_to(task_root)
    parts = rel.parts

    method = parts[0] if parts else "unknown"
    rank = ""
    seed = ""
    profile = ""

    for part in parts:
        if part.startswith("rank-"):
            rank = part[len("rank-") :]
        elif part.startswith("seed"):
            seed = part[len("seed") :]
        elif part.startswith("profile-") or part.startswith("profile_"):
            profile = part

    run_cfg = read_json(adapter_dir / "run_config.json")
    if isinstance(run_cfg.get("peft_method"), str):
        method = str(run_cfg["peft_method"])

    return AdapterInfo(adapter_dir=adapter_dir, method=method, rank=rank, seed=seed, profile=profile)


def discover_adapters(runs_root: Path, base_model_dir: str, task_dir: str) -> list[AdapterInfo]:
    task_root = runs_root / base_model_dir / task_dir
    if not task_root.is_dir():
        return []

    adapters: list[AdapterInfo] = []
    for cfg in task_root.rglob("adapter_config.json"):
        adapter_dir = cfg.parent
        if not is_adapter_dir(adapter_dir):
            continue
        adapters.append(parse_adapter_metadata(adapter_dir, task_root))

    adapters.sort(key=lambda a: str(a.adapter_dir))
    return adapters


def resolve_task_filter(values: Iterable[str]) -> list[str]:
    resolved = []
    for raw in values:
        key = raw.strip().lower()
        if not key:
            continue
        task_dir = TASK_ALIASES.get(key, key)
        resolved.append(task_dir)
    return resolved


def build_run_id(req: RunRequest) -> str:
    parts = [req.method]
    if req.profile:
        parts.append(req.profile)
    if req.rank:
        parts.append(f"rank-{req.rank}")
    if req.seed:
        parts.append(f"seed{req.seed}")
    return sanitize_tag("_".join(parts))


def build_eval_command(req: RunRequest, args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [sys.executable, str(req.task_spec.eval_script), "--base_model", req.base_model_id]
    if req.adapter_dir is not None:
        cmd += ["--adapter_dir", str(req.adapter_dir)]
    cmd += ["--output_dir", str(output_dir)]

    if args.max_samples is not None:
        cmd += ["--max_samples", str(args.max_samples)]
    if args.max_new_tokens is not None:
        cmd += ["--max_new_tokens", str(args.max_new_tokens)]
    if args.dtype is not None:
        cmd += ["--dtype", args.dtype]
    if args.eval_seed is not None:
        cmd += ["--seed", str(args.eval_seed)]
    if args.use_vllm:
        cmd += ["--use_vllm"]
        cmd += ["--tensor_parallel_size", str(args.tensor_parallel_size)]
    if args.split is not None and req.task_spec.supports_split:
        cmd += ["--split", args.split]
    if args.timeout_s is not None and req.task_spec.supports_timeout:
        cmd += ["--timeout_s", str(args.timeout_s)]

    return cmd


def run_eval(
    *,
    req: RunRequest,
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
) -> RunResult:
    run_id = build_run_id(req)
    run_dir = output_root / "runs" / req.task_dir / req.base_model_dir / run_id
    log_path = output_root / "logs" / req.task_dir / req.base_model_dir / f"{run_id}.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_eval_command(req, args, run_dir)
    env = os.environ.copy()
    # Ensure eval scripts can import the src/ package layout without editable installs.
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(
        os.pathsep
    )

    start = time.time()
    error = ""
    metrics: dict[str, Any] = {}

    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("Command: " + " ".join(cmd) + "\n")
        log_f.flush()

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if proc.returncode != 0:
                error = f"exit_code={proc.returncode}"
        except Exception as exc:
            error = f"exception={exc}"

    elapsed = time.time() - start
    metrics_path = run_dir / "metrics.json"
    if not error:
        if metrics_path.is_file():
            metrics = read_json(metrics_path)
        else:
            error = "missing_metrics"

    result = {
        "task": req.task_spec.dataset,
        "task_dir": req.task_dir,
        "base_model": req.base_model_id,
        "base_model_dir": req.base_model_dir,
        "method": req.method,
        "rank": req.rank,
        "seed": req.seed,
        "profile": req.profile,
        "adapter_path": str(req.adapter_dir) if req.adapter_dir else "",
        "run_id": run_id,
        "output_dir": str(run_dir),
        "log_path": str(log_path),
        "elapsed_s": round(elapsed, 3),
        "error": error,
        "metrics": metrics,
        "metric": metrics.get(req.task_spec.primary_metric, "") if metrics else "",
    }
    return RunResult(data=result)


def select_base_models(
    available: list[str],
    base_model_ids: dict[str, str],
    filters: list[str],
) -> list[str]:
    if not filters:
        return available
    selected = []
    filter_set = {f.strip() for f in filters if f.strip()}
    for base_dir in available:
        if base_dir in filter_set or base_model_ids.get(base_dir) in filter_set:
            selected.append(base_dir)
    return selected


def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No runs executed."

    def fmt(val: Any) -> str:
        if isinstance(val, float):
            return f"{val:.6f}"
        return str(val) if val is not None else ""

    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(fmt(row.get(col, ""))))

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    lines = [header, sep]
    for row in rows:
        lines.append(" | ".join(fmt(row.get(col, "")).ljust(widths[col]) for col in columns))
    return "\n".join(lines)


def write_summary(output_root: Path, results: list[RunResult], metric_columns: list[str]) -> None:
    summary_path = output_root / "summary.csv"
    results_path = output_root / "results.jsonl"

    base_cols = [
        "task",
        "task_dir",
        "base_model",
        "base_model_dir",
        "method",
        "rank",
        "seed",
        "profile",
        "adapter_path",
        "metric",
    ]
    cols = base_cols + metric_columns + ["error", "output_dir", "log_path", "elapsed_s"]

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for res in results:
            row = {k: res.data.get(k, "") for k in base_cols}
            metrics = res.data.get("metrics", {}) or {}
            for key in metric_columns:
                row[key] = metrics.get(key, "")
            row["error"] = res.data.get("error", "")
            row["output_dir"] = res.data.get("output_dir", "")
            row["log_path"] = res.data.get("log_path", "")
            row["elapsed_s"] = res.data.get("elapsed_s", "")
            writer.writerow(row)

    with results_path.open("w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res.data, ensure_ascii=False) + "\n")


def collect_metric_columns(results: list[RunResult]) -> list[str]:
    metric_keys: list[str] = []
    seen = set()

    def add(key: str):
        if key not in seen:
            seen.add(key)
            metric_keys.append(key)

    for spec in TASK_SPECS.values():
        for key in spec.metric_order:
            add(key)

    for res in results:
        metrics = res.data.get("metrics", {}) or {}
        for key in metrics.keys():
            if key not in seen:
                add(key)

    return metric_keys


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Batch-evaluate discovered LoRA-family adapters under runs/ using existing eval scripts."
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default=str(repo_root / "runs"),
        help="Root of runs/ directory.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(repo_root / "eval_results"),
        help="Base output directory (timestamped subdir will be created).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Full output directory (overrides output_root + timestamp).",
    )
    parser.add_argument("--task", type=str, default=None, help="Task filter (comma-separated).")
    parser.add_argument("--base_model", type=str, default=None, help="Base model filter (comma-separated).")
    parser.add_argument("--method", type=str, default=None, help="Method filter (comma-separated).")
    parser.add_argument("--rank", type=str, default=None, help="Rank filter (comma-separated).")
    parser.add_argument("--seed", type=str, default=None, help="Seed filter (comma-separated).")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit max eval samples.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override max_new_tokens.")
    parser.add_argument("--dtype", type=str, default=None, choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--eval_seed", type=int, default=None, help="Override eval seed.")
    parser.add_argument("--split", type=str, default=None, help="Dataset split override (tasks that support it).")
    parser.add_argument("--timeout_s", type=float, default=None, help="HumanEval timeout override.")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM.")

    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.is_dir():
        print(f"runs_root not found: {runs_root}")
        return 1

    base_model_ids = {base_dir: infer_base_model_id(runs_root, base_dir) for base_dir in BASE_MODEL_DIRS}
    available_base_models = [b for b in BASE_MODEL_DIRS if (runs_root / b).is_dir()]

    base_filters = parse_list(args.base_model)
    selected_base_models = select_base_models(available_base_models, base_model_ids, base_filters)

    task_filters = resolve_task_filter(parse_list(args.task))
    method_filters = {m.lower() for m in parse_list(args.method)}
    rank_filters = {normalize_rank(r) for r in parse_list(args.rank)}
    seed_filters = {normalize_seed(s) for s in parse_list(args.seed)}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(args.output_root) / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    run_requests: list[RunRequest] = []

    # Discovery phase: gather adapter runs (and vanilla baselines) under runs/.
    for base_dir in selected_base_models:
        base_model_id = base_model_ids[base_dir]
        task_dirs = discover_task_dirs(runs_root, base_dir)
        for task_dir in task_dirs:
            if task_dir not in TASK_SPECS:
                continue
            if task_filters and task_dir not in task_filters:
                continue

            task_spec = TASK_SPECS[task_dir]
            adapters = discover_adapters(runs_root, base_dir, task_dir)

            for adapter in adapters:
                if method_filters and adapter.method.lower() not in method_filters:
                    continue
                if rank_filters and adapter.rank not in rank_filters:
                    continue
                if seed_filters and adapter.seed not in seed_filters:
                    continue
                run_requests.append(
                    RunRequest(
                        task_spec=task_spec,
                        task_dir=task_dir,
                        base_model_dir=base_dir,
                        base_model_id=base_model_id,
                        method=adapter.method,
                        rank=adapter.rank,
                        seed=adapter.seed,
                        profile=adapter.profile,
                        adapter_dir=adapter.adapter_dir,
                    )
                )

            include_vanilla = not method_filters or "vanilla" in method_filters
            if include_vanilla:
                run_requests.append(
                    RunRequest(
                        task_spec=task_spec,
                        task_dir=task_dir,
                        base_model_dir=base_dir,
                        base_model_id=base_model_id,
                        method="vanilla",
                        rank="",
                        seed="",
                        profile="",
                        adapter_dir=None,
                    )
                )

    if not run_requests:
        print("No runs matched the provided filters.")
        print(f"Output directory created at: {output_root}")
        return 0

    # Execution phase: run eval scripts and capture metrics/errors.
    total = len(run_requests)
    for idx, req in enumerate(run_requests, start=1):
        print(
            f"[{idx}/{total}] task={req.task_spec.dataset} base={req.base_model_dir} method={req.method}"
            f" rank={req.rank or '-'} seed={req.seed or '-'}"
        )
        res = run_eval(req=req, args=args, repo_root=repo_root, output_root=output_root)
        results.append(res)

    # Reporting phase: write summary artifacts and print a concise table.
    metric_columns = collect_metric_columns(results)
    write_summary(output_root, results, metric_columns)

    rows = []
    for res in results:
        row = {
            "task": res.data.get("task", ""),
            "base_model": res.data.get("base_model", ""),
            "method": res.data.get("method", ""),
            "rank": res.data.get("rank", ""),
            "seed": res.data.get("seed", ""),
            "adapter_path": res.data.get("adapter_path", ""),
            "metric": res.data.get("metric", ""),
            "error": res.data.get("error", ""),
        }
        rows.append(row)

    table_cols = ["task", "base_model", "method", "rank", "seed", "adapter_path", "metric", "error"]
    print("\nSummary")
    print(render_table(rows, table_cols))
    print(f"\nSaved summary to: {output_root / 'summary.csv'}")
    print(f"Saved detailed results to: {output_root / 'results.jsonl'}")

    error_count = sum(1 for r in results if r.data.get("error"))
    if error_count:
        print(f"Runs with errors: {error_count}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
