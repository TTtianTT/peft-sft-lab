#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

import eval_runs


COMPARE_FIELDS = {
    "csqa": ("prediction_text", "prediction_letter", "correct"),
    "metamath": ("prediction_text", "prediction_extracted", "correct"),
    "magicoder": ("completion", "passed", "error"),
    "alpaca": ("output", "score", "all_passed"),
}

OUTPUT_FILE_CANDIDATES = ("predictions.jsonl", "outputs.jsonl")


@dataclass
class BackendRun:
    backend: str
    output_dir: Path
    log_path: Path
    metrics: dict[str, Any]
    error: str


def find_output_file(run_dir: Path) -> Path | None:
    for name in OUTPUT_FILE_CANDIDATES:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    jsonl_files = sorted(run_dir.glob("*.jsonl"))
    if len(jsonl_files) == 1:
        return jsonl_files[0]
    return None


def compact_value(value: Any, limit: int = 200) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...<truncated>"
    return value


def diff_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    keys = sorted(set(a.keys()) | set(b.keys()))
    for key in keys:
        va = a.get(key)
        vb = b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            if abs(delta) > 1e-12:
                diff[key] = {"transformers": va, "vllm": vb, "delta": delta}
        else:
            if va != vb:
                diff[key] = {"transformers": va, "vllm": vb}
    return diff


def compare_outputs(task_dir: str, tf_dir: Path, vllm_dir: Path, max_examples: int) -> dict[str, Any]:
    tf_file = find_output_file(tf_dir)
    vllm_file = find_output_file(vllm_dir)

    if tf_file is None or vllm_file is None:
        return {
            "error": "missing_output_file",
            "transformers_file": str(tf_file) if tf_file else "",
            "vllm_file": str(vllm_file) if vllm_file else "",
        }

    fields = COMPARE_FIELDS.get(task_dir)
    total = 0
    diffs = 0
    samples: list[dict[str, Any]] = []

    try:
        with tf_file.open("r", encoding="utf-8") as fa, vllm_file.open("r", encoding="utf-8") as fb:
            for idx, (la, lb) in enumerate(zip_longest(fa, fb, fillvalue=None)):
                if la is None or lb is None:
                    diffs += 1
                    total += 1
                    if len(samples) < max_examples:
                        samples.append(
                            {
                                "index": idx,
                                "reason": "line_count_mismatch",
                                "transformers": bool(la is not None),
                                "vllm": bool(lb is not None),
                            }
                        )
                    continue

                total += 1
                rec_a = json.loads(la)
                rec_b = json.loads(lb)
                field_diffs: dict[str, Any] = {}

                if fields:
                    for field in fields:
                        va = rec_a.get(field)
                        vb = rec_b.get(field)
                        if va != vb:
                            field_diffs[field] = {
                                "transformers": compact_value(va),
                                "vllm": compact_value(vb),
                            }
                else:
                    if rec_a != rec_b:
                        field_diffs["record"] = {
                            "transformers": compact_value(rec_a),
                            "vllm": compact_value(rec_b),
                        }

                if field_diffs:
                    diffs += 1
                    if len(samples) < max_examples:
                        sample = {"index": idx, "fields": field_diffs}
                        for key in ("task_id", "question", "prompt"):
                            if key in rec_a:
                                sample[key] = rec_a.get(key)
                                break
                        samples.append(sample)
    except Exception as exc:
        return {
            "error": f"compare_failed: {exc}",
            "transformers_file": str(tf_file),
            "vllm_file": str(vllm_file),
        }

    return {
        "transformers_file": str(tf_file),
        "vllm_file": str(vllm_file),
        "total": total,
        "differences": diffs,
        "sample_diffs": samples,
    }


def build_eval_command(
    *,
    task_spec: eval_runs.TaskSpec,
    base_model_id: str,
    output_dir: Path,
    args: argparse.Namespace,
    use_vllm: bool,
) -> list[str]:
    cmd = [sys.executable, str(task_spec.eval_script), "--base_model", base_model_id, "--output_dir", str(output_dir)]

    if args.max_samples is not None:
        cmd += ["--max_samples", str(args.max_samples)]
    if args.max_new_tokens is not None:
        cmd += ["--max_new_tokens", str(args.max_new_tokens)]
    if args.dtype is not None:
        cmd += ["--dtype", args.dtype]
    if args.eval_seed is not None:
        cmd += ["--seed", str(args.eval_seed)]
    if args.split is not None and task_spec.supports_split:
        cmd += ["--split", args.split]
    if args.timeout_s is not None and task_spec.supports_timeout:
        cmd += ["--timeout_s", str(args.timeout_s)]

    if use_vllm:
        cmd += ["--use_vllm", "--tensor_parallel_size", str(args.tensor_parallel_size)]

    if args.adapter_dir is not None:
        cmd += ["--adapter_dir", str(args.adapter_dir)]

    return cmd


def run_backend(
    *,
    backend: str,
    task_spec: eval_runs.TaskSpec,
    base_model_id: str,
    output_root: Path,
    repo_root: Path,
    base_model_dir: str,
    task_dir: str,
    args: argparse.Namespace,
) -> BackendRun:
    run_dir = output_root / "runs" / task_dir / base_model_dir / backend
    log_path = output_root / "logs" / task_dir / base_model_dir / f"{backend}.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_eval_command(
        task_spec=task_spec,
        base_model_id=base_model_id,
        output_dir=run_dir,
        args=args,
        use_vllm=(backend == "vllm"),
    )

    env = os.environ.copy()
    # Ensure eval scripts can import the src/ package layout without editable installs.
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

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

    metrics_path = run_dir / "metrics.json"
    if not error:
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as exc:
                error = f"metrics_read_failed={exc}"
        else:
            error = "missing_metrics"

    return BackendRun(backend=backend, output_dir=run_dir, log_path=log_path, metrics=metrics, error=error)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Compare Transformers vs vLLM inference outputs/metrics for each base model and task."
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default=str(repo_root / "runs"),
        help="Root of runs/ directory (used to infer base_model ids).",
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
    parser.add_argument("--max_samples", type=int, default=None, help="Limit max eval samples.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override max_new_tokens.")
    parser.add_argument("--dtype", type=str, default=None, choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--eval_seed", type=int, default=None, help="Override eval seed.")
    parser.add_argument("--split", type=str, default=None, help="Dataset split override (tasks that support it).")
    parser.add_argument("--timeout_s", type=float, default=None, help="HumanEval timeout override.")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Tensor parallel size for vLLM.")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Optional adapter directory to compare.")
    parser.add_argument("--max_diff_examples", type=int, default=5, help="Max example diffs to include per task.")

    args = parser.parse_args()
    if args.adapter_dir is not None:
        args.adapter_dir = str(Path(args.adapter_dir).expanduser().resolve())

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.is_dir():
        print(f"runs_root not found: {runs_root}")
        return 1

    base_model_ids = {
        base_dir: eval_runs.infer_base_model_id(runs_root, base_dir) for base_dir in eval_runs.BASE_MODEL_DIRS
    }
    available_base_models = [b for b in eval_runs.BASE_MODEL_DIRS if (runs_root / b).is_dir()]
    base_filters = eval_runs.parse_list(args.base_model)
    selected_base_models = eval_runs.select_base_models(available_base_models, base_model_ids, base_filters)

    task_filters = eval_runs.resolve_task_filter(eval_runs.parse_list(args.task))
    task_dirs = [t for t in eval_runs.TASK_SPECS.keys() if not task_filters or t in task_filters]

    if args.adapter_dir is not None and len(selected_base_models) > 1:
        print("--adapter_dir requires selecting a single base model via --base_model.")
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(args.output_root) / f"backend_compare_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    report_entries: list[dict[str, Any]] = []
    rows = []

    for base_dir in selected_base_models:
        base_model_id = base_model_ids[base_dir]
        for task_dir in task_dirs:
            task_spec = eval_runs.TASK_SPECS[task_dir]
            print(f"Running {task_spec.dataset} for {base_dir} (transformers vs vLLM)")

            tf_run = run_backend(
                backend="transformers",
                task_spec=task_spec,
                base_model_id=base_model_id,
                output_root=output_root,
                repo_root=repo_root,
                base_model_dir=base_dir,
                task_dir=task_dir,
                args=args,
            )
            vllm_run = run_backend(
                backend="vllm",
                task_spec=task_spec,
                base_model_id=base_model_id,
                output_root=output_root,
                repo_root=repo_root,
                base_model_dir=base_dir,
                task_dir=task_dir,
                args=args,
            )

            metrics_diff = diff_metrics(tf_run.metrics, vllm_run.metrics)
            output_diff = compare_outputs(task_dir, tf_run.output_dir, vllm_run.output_dir, args.max_diff_examples)

            entry = {
                "task": task_spec.dataset,
                "task_dir": task_dir,
                "base_model": base_model_id,
                "base_model_dir": base_dir,
                "transformers": {
                    "output_dir": str(tf_run.output_dir),
                    "log_path": str(tf_run.log_path),
                    "metrics": tf_run.metrics,
                    "error": tf_run.error,
                },
                "vllm": {
                    "output_dir": str(vllm_run.output_dir),
                    "log_path": str(vllm_run.log_path),
                    "metrics": vllm_run.metrics,
                    "error": vllm_run.error,
                },
                "metrics_diff": metrics_diff,
                "output_diff": output_diff,
            }
            report_entries.append(entry)

            metric_key = task_spec.primary_metric
            tf_metric = tf_run.metrics.get(metric_key, "")
            vllm_metric = vllm_run.metrics.get(metric_key, "")
            delta = ""
            if isinstance(tf_metric, (int, float)) and isinstance(vllm_metric, (int, float)):
                delta = vllm_metric - tf_metric

            rows.append(
                {
                    "task": task_spec.dataset,
                    "base_model": base_model_id,
                    "metric": metric_key,
                    "transformers": tf_metric,
                    "vllm": vllm_metric,
                    "delta": delta,
                    "output_diffs": output_diff.get("differences", ""),
                    "total": output_diff.get("total", ""),
                    "error": tf_run.error or vllm_run.error or output_diff.get("error", ""),
                }
            )

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "entries": report_entries,
    }

    report_path = output_root / "diff_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    table_cols = ["task", "base_model", "metric", "transformers", "vllm", "delta", "output_diffs", "total", "error"]
    table = eval_runs.render_table(rows, table_cols) if rows else "No comparisons executed."
    text_report = "Backend comparison summary\n" + table + f"\n\nFull report: {report_path}\n"

    text_path = output_root / "diff_report.txt"
    text_path.write_text(text_report, encoding="utf-8")

    print(text_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
