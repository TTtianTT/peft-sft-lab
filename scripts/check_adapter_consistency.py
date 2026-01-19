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


BASE_MODEL_DIRS = [
    "meta-llama-Llama-3.1-8B",
    "mistralai-Mistral-7B-v0.3",
]

OUTPUT_FILE_CANDIDATES = ("predictions.jsonl", "outputs.jsonl")

COMPARE_FIELDS = {
    "csqa": {"text": "prediction_text", "extracted": ["prediction_letter", "correct"]},
    "metamath": {"text": "prediction_text", "extracted": ["prediction_extracted", "correct"]},
    "magicoder": {"text": "completion", "extracted": ["passed", "error"]},
    "alpaca": {"text": "output", "extracted": ["score", "all_passed"]},
}

ID_FIELDS = ("task_id", "question", "prompt")


@dataclass
class BackendRun:
    backend: str
    output_dir: Path
    log_path: Path
    metrics: dict[str, Any]
    error: str


def compact_value(value: Any, limit: int = 200) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...<truncated>"
    return value


def values_equal(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def find_output_file(run_dir: Path) -> Path | None:
    for name in OUTPUT_FILE_CANDIDATES:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    jsonl_files = sorted(run_dir.glob("*.jsonl"))
    if len(jsonl_files) == 1:
        return jsonl_files[0]
    return None


def diff_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    keys = sorted(set(a.keys()) | set(b.keys()))
    for key in keys:
        va = a.get(key)
        vb = b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = float(vb) - float(va)
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

    fields = COMPARE_FIELDS.get(task_dir, {})
    text_field = fields.get("text")
    extracted_fields = fields.get("extracted", [])

    total = 0
    text_matches = 0
    extracted_matches = 0
    samples: list[dict[str, Any]] = []

    try:
        with tf_file.open("r", encoding="utf-8") as fa, vllm_file.open("r", encoding="utf-8") as fb:
            for idx, (la, lb) in enumerate(zip_longest(fa, fb, fillvalue=None)):
                if la is None or lb is None:
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

                rec_a = json.loads(la)
                rec_b = json.loads(lb)
                total += 1

                text_match = True
                extracted_match = True
                sample: dict[str, Any] = {"index": idx}

                for key in ID_FIELDS:
                    if key in rec_a:
                        sample[key] = compact_value(rec_a.get(key))
                        break

                if text_field:
                    va = rec_a.get(text_field)
                    vb = rec_b.get(text_field)
                    text_match = values_equal(va, vb)
                    if not text_match:
                        sample["text"] = {
                            "transformers": compact_value(va),
                            "vllm": compact_value(vb),
                        }

                extracted_diffs: dict[str, Any] = {}
                for field in extracted_fields:
                    va = rec_a.get(field)
                    vb = rec_b.get(field)
                    if not values_equal(va, vb):
                        extracted_match = False
                        extracted_diffs[field] = {
                            "transformers": compact_value(va),
                            "vllm": compact_value(vb),
                        }

                if not extracted_match:
                    sample["extracted"] = extracted_diffs

                if text_match:
                    text_matches += 1
                if extracted_match:
                    extracted_matches += 1

                if (not text_match or not extracted_match) and len(samples) < max_examples:
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
        "text_matches": text_matches,
        "text_mismatches": total - text_matches,
        "extracted_matches": extracted_matches,
        "extracted_mismatches": total - extracted_matches,
        "sample_mismatches": samples,
    }


def build_eval_command(
    *,
    task_spec: eval_runs.TaskSpec,
    base_model_id: str,
    adapter_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    use_vllm: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(task_spec.eval_script),
        "--base_model",
        base_model_id,
        "--adapter_dir",
        str(adapter_dir),
        "--output_dir",
        str(output_dir),
    ]

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

    return cmd


def run_backend(
    *,
    backend: str,
    task_spec: eval_runs.TaskSpec,
    base_model_id: str,
    adapter_dir: Path,
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
        adapter_dir=adapter_dir,
        output_dir=run_dir,
        args=args,
        use_vllm=(backend == "vllm"),
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(
        os.pathsep
    )

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


def select_adapter(
    *,
    runs_root: Path,
    base_model_dir: str,
    task_dir: str | None,
    method: str,
    rank: str | None,
    seed: str | None,
    profile_contains: str | None,
) -> tuple[str, eval_runs.AdapterInfo] | None:
    if task_dir:
        task_candidates = [task_dir]
    else:
        task_candidates = list(eval_runs.TASK_SPECS.keys())

    rank_value = normalize_rank(rank) if rank else None
    seed_value = normalize_seed(seed) if seed else None

    for candidate in task_candidates:
        adapters = eval_runs.discover_adapters(runs_root, base_model_dir, candidate)
        filtered = []
        for adapter in adapters:
            if method and adapter.method.lower() != method.lower():
                continue
            if rank_value and adapter.rank != rank_value:
                continue
            if seed_value and adapter.seed != seed_value:
                continue
            if profile_contains and profile_contains not in adapter.profile:
                continue
            filtered.append(adapter)

        if filtered:
            filtered.sort(key=lambda a: str(a.adapter_dir))
            return candidate, filtered[0]

    return None


def infer_task_from_adapter(runs_root: Path, base_model_dir: str, adapter_dir: Path) -> str | None:
    try:
        rel = adapter_dir.relative_to(runs_root / base_model_dir)
    except ValueError:
        return None
    if not rel.parts:
        return None
    return rel.parts[0]


def resolve_adapter_dir(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Check Transformers vs vLLM consistency for one LoRA adapter per base model."
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default=str(repo_root / "runs"),
        help="Root of runs/ directory (used for adapter discovery).",
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
    parser.add_argument("--task", type=str, default=None, help="Optional task to target (e.g., csqa).")
    parser.add_argument("--method", type=str, default="lora", help="Adapter method to select.")
    parser.add_argument("--rank", type=str, default=None, help="Adapter rank filter.")
    parser.add_argument("--seed", type=str, default=None, help="Adapter seed filter.")
    parser.add_argument("--profile_contains", type=str, default=None, help="Adapter profile substring filter.")
    parser.add_argument("--adapter_llama", type=str, default=None, help="Explicit adapter path for Llama 3.1 8B.")
    parser.add_argument("--adapter_mistral", type=str, default=None, help="Explicit adapter path for Mistral 7B v0.3.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit max eval samples.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override max_new_tokens.")
    parser.add_argument("--dtype", type=str, default=None, choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--eval_seed", type=int, default=None, help="Override eval seed.")
    parser.add_argument("--split", type=str, default=None, help="Dataset split override (tasks that support it).")
    parser.add_argument("--timeout_s", type=float, default=None, help="HumanEval timeout override.")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Tensor parallel size for vLLM.")
    parser.add_argument("--max_examples", type=int, default=5, help="Max mismatch examples to include.")

    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.is_dir():
        print(f"runs_root not found: {runs_root}")
        return 1

    task_filters = eval_runs.resolve_task_filter(eval_runs.parse_list(args.task))
    if len(task_filters) > 1:
        print("Please specify at most one task for this consistency check.")
        return 1

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(args.output_root) / f"consistency_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    report_entries: list[dict[str, Any]] = []
    rows = []

    for base_dir in BASE_MODEL_DIRS:
        base_model_id = eval_runs.infer_base_model_id(runs_root, base_dir)

        explicit_adapter = None
        if base_dir == "meta-llama-Llama-3.1-8B" and args.adapter_llama:
            explicit_adapter = resolve_adapter_dir(args.adapter_llama)
        if base_dir == "mistralai-Mistral-7B-v0.3" and args.adapter_mistral:
            explicit_adapter = resolve_adapter_dir(args.adapter_mistral)

        if explicit_adapter:
            if not eval_runs.is_adapter_dir(explicit_adapter):
                print(f"Invalid adapter directory: {explicit_adapter}")
                return 1
            task_dir = infer_task_from_adapter(runs_root, base_dir, explicit_adapter)
            if task_filters:
                requested = task_filters[0]
                if task_dir and task_dir != requested:
                    print(f"Adapter task mismatch: {explicit_adapter} is under {task_dir}, requested {requested}")
                    return 1
                task_dir = requested
            if not task_dir:
                if not task_filters:
                    print(f"Unable to infer task for adapter: {explicit_adapter}. Use --task.")
                    return 1
                task_dir = task_filters[0]
            adapter_info = eval_runs.AdapterInfo(
                adapter_dir=explicit_adapter,
                method=args.method,
                rank=normalize_rank(args.rank) if args.rank else "",
                seed=normalize_seed(args.seed) if args.seed else "",
                profile=args.profile_contains or "",
            )
        else:
            selected = select_adapter(
                runs_root=runs_root,
                base_model_dir=base_dir,
                task_dir=task_filters[0] if task_filters else None,
                method=args.method,
                rank=args.rank,
                seed=args.seed,
                profile_contains=args.profile_contains,
            )
            if not selected:
                print(f"No adapter found for {base_dir} (method={args.method}).")
                return 1
            task_dir, adapter_info = selected

        if task_dir not in eval_runs.TASK_SPECS:
            print(f"Unsupported task directory for adapter: {task_dir}")
            return 1

        task_spec = eval_runs.TASK_SPECS[task_dir]
        print(
            f"Evaluating {base_dir} adapter at {adapter_info.adapter_dir} for task {task_spec.dataset}"
        )

        tf_run = run_backend(
            backend="transformers",
            task_spec=task_spec,
            base_model_id=base_model_id,
            adapter_dir=adapter_info.adapter_dir,
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
            adapter_dir=adapter_info.adapter_dir,
            output_root=output_root,
            repo_root=repo_root,
            base_model_dir=base_dir,
            task_dir=task_dir,
            args=args,
        )

        metrics_diff = diff_metrics(tf_run.metrics, vllm_run.metrics)
        output_diff = compare_outputs(task_dir, tf_run.output_dir, vllm_run.output_dir, args.max_examples)

        entry = {
            "task": task_spec.dataset,
            "task_dir": task_dir,
            "base_model": base_model_id,
            "base_model_dir": base_dir,
            "adapter_dir": str(adapter_info.adapter_dir),
            "adapter_method": adapter_info.method,
            "adapter_rank": adapter_info.rank,
            "adapter_seed": adapter_info.seed,
            "adapter_profile": adapter_info.profile,
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
            delta = float(vllm_metric) - float(tf_metric)

        total = output_diff.get("total", "")
        text_matches = output_diff.get("text_matches", "")
        extracted_matches = output_diff.get("extracted_matches", "")

        rows.append(
            {
                "base_model": base_model_id,
                "task": task_spec.dataset,
                "adapter_dir": str(adapter_info.adapter_dir),
                "text_matches": text_matches,
                "extracted_matches": extracted_matches,
                "total": total,
                "metric": metric_key,
                "transformers": tf_metric,
                "vllm": vllm_metric,
                "delta": delta,
                "error": tf_run.error or vllm_run.error or output_diff.get("error", ""),
            }
        )

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "entries": report_entries,
    }

    report_path = output_root / "consistency_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    table_cols = [
        "base_model",
        "task",
        "adapter_dir",
        "text_matches",
        "extracted_matches",
        "total",
        "metric",
        "transformers",
        "vllm",
        "delta",
        "error",
    ]
    table = eval_runs.render_table(rows, table_cols) if rows else "No comparisons executed."

    lines: list[str] = ["Adapter consistency summary", table, ""]
    for entry in report_entries:
        output_diff = entry.get("output_diff", {}) or {}
        base_model = entry.get("base_model", "")
        task = entry.get("task", "")
        adapter_dir = entry.get("adapter_dir", "")
        text_matches = output_diff.get("text_matches", "")
        extracted_matches = output_diff.get("extracted_matches", "")
        total = output_diff.get("total", "")
        lines.append(f"Base model: {base_model}")
        lines.append(f"Task: {task}")
        lines.append(f"Adapter: {adapter_dir}")
        lines.append(f"Text matches: {text_matches} / {total}")
        lines.append(f"Extracted matches: {extracted_matches} / {total}")
        sample_mismatches = output_diff.get("sample_mismatches", []) or []
        if sample_mismatches:
            lines.append("Examples:")
            for sample in sample_mismatches:
                lines.append(json.dumps(sample))
        lines.append("")

    lines.append(f"Full report: {report_path}")
    text_report = "\n".join(lines) + "\n"
    text_path = output_root / "consistency_report.txt"
    text_path.write_text(text_report, encoding="utf-8")

    print(text_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
