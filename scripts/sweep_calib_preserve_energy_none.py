#!/usr/bin/env python3
"""
Sweep calib_samples with preserve_energy=none for LoRA adapters only.

This script runs scripts/run_lm_eval_harness_spectral_edits.py for:
  --calib_samples: 32, 64, 128
  --adapter_types: lora
  --preserve_energy: none
  --no-keep_edited_adapter (temporary edited adapters only)

Outputs:
  - per-run logs under <out_dir>/logs/
  - a results.jsonl summary at <out_dir>/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


# ============================================================================
# Constants
# ============================================================================

POLICIES = ["abs_select", "smooth_abs", "random_index", "grad_direction"]
CALIB_SAMPLES = [32, 64, 128]
PRESERVE_ENERGY = "none"
ADAPTER_TYPES = ["lora"]
DEFAULT_TASKS = ["math", "code", "alpaca", "csqa"]
DEFAULT_SEEDS = [42]
KNOWN_MODEL_PREFIXES = ("meta-llama", "Qwen")


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class SummaryRecord:
    calib_samples: int
    preserve_energy: str
    model: str
    task: str
    metric: Optional[str]
    baseline: Optional[float]
    abs_select: Optional[float]
    smooth_abs: Optional[float]
    random_index: Optional[float]
    grad_direction: Optional[float]
    status: str
    log_path: str
    seed: int


# ============================================================================
# Utilities
# ============================================================================

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def write_jsonl_atomic(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp_path, path)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def read_tail(path: Path, max_bytes: int = 20000) -> str:
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def format_cmd(cmd: List[str], env_prefix: Optional[Dict[str, str]] = None) -> str:
    cmd_str = shlex.join(cmd)
    if not env_prefix:
        return cmd_str
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_prefix.items())
    return f"{prefix} {cmd_str}"


def record_key(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        rec.get("calib_samples"),
        rec.get("preserve_energy"),
        rec.get("model"),
        rec.get("task"),
        rec.get("seed"),
    )


def parse_summary(out_root: Path) -> List[Dict[str, Any]]:
    summary_path = out_root / "summary.json"
    data = read_json(summary_path)
    if not data:
        return []
    if isinstance(data, dict):
        data = data.get("records", [])
    if not isinstance(data, list):
        return []
    return data


def looks_like_model_tag(name: str) -> bool:
    return name.startswith(KNOWN_MODEL_PREFIXES)


def has_task_subdir(root: Path, tasks: List[str]) -> bool:
    try:
        for child in root.iterdir():
            if child.is_dir() and child.name in tasks:
                return True
    except Exception:
        return False
    return False


def discover_models(runs_roots: List[Path], tasks: List[str]) -> List[str]:
    models: List[str] = []
    for root in runs_roots:
        if looks_like_model_tag(root.name) or has_task_subdir(root, tasks):
            models.append(root.name)
            continue
        try:
            for child in root.iterdir():
                if child.is_dir():
                    models.append(child.name)
        except Exception:
            continue
    return sorted(set(models))


def models_from_summary(summary_records: List[Dict[str, Any]]) -> List[str]:
    found = {
        rec.get("base_model_tag")
        for rec in summary_records
        if isinstance(rec, dict) and rec.get("base_model_tag")
    }
    return sorted(found)


def aggregate_metrics(
    records: List[Dict[str, Any]],
    model: str,
    task: str,
) -> Tuple[Dict[str, Optional[float]], Optional[str], str]:
    def collect_variant(variant: str) -> Tuple[List[float], List[str], List[str]]:
        vals = []
        keys = []
        errors = []
        for rec in records:
            if rec.get("base_model_tag") != model:
                continue
            if rec.get("task") != task:
                continue
            if rec.get("variant") != variant:
                continue
            if rec.get("error"):
                errors.append(rec.get("error"))
                continue
            if rec.get("metric_value") is not None:
                try:
                    vals.append(float(rec["metric_value"]))
                except Exception:
                    continue
            if rec.get("metric_key"):
                keys.append(rec["metric_key"])
        return vals, keys, errors

    variants = {
        "baseline": "baseline",
        "abs_select": "edited/abs_select",
        "smooth_abs": "edited/smooth_abs",
        "random_index": "edited/random_index",
        "grad_direction": "edited/grad_direction",
    }

    output: Dict[str, Optional[float]] = {}
    metric_keys: List[str] = []
    status = "ok"

    for key, variant in variants.items():
        vals, keys, errors = collect_variant(variant)
        if errors or not vals:
            status = "error"
            output[key] = None
        else:
            output[key] = sum(vals) / len(vals)
        metric_keys.extend(keys)

    metric = metric_keys[0] if metric_keys else None
    return output, metric, status


def build_out_root(base_out: Path, calib_samples: int, seeds: List[int], seed: int) -> Path:
    suffix = f"_calib{calib_samples}"
    if len(seeds) > 1:
        suffix += f"_seed{seed}"
    return Path(str(base_out) + suffix)


def build_runner_cmd(
    runs_roots: List[Path],
    out_root: Path,
    tasks: List[str],
    calib_samples: int,
    seed: int,
    policies: List[str],
    extra_args: List[str],
) -> List[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_lm_eval_harness_spectral_edits.py"),
        "--runs_roots",
        *[str(p) for p in runs_roots],
        "--out_root",
        str(out_root),
        "--policies",
        *policies,
        "--adapter_types",
        *ADAPTER_TYPES,
        "--use_vllm_lora",
        "--fallback_merge",
        "--calib_samples",
        str(calib_samples),
        "--preserve_energy",
        PRESERVE_ENERGY,
        "--no-keep_edited_adapter",
        "--seed",
        str(seed),
        "--tasks",
        *tasks,
    ]
    cmd.extend(extra_args)
    return cmd


def run_command(
    cmd: List[str],
    log_path: Path,
    timeout_s: Optional[int],
) -> Tuple[bool, Optional[str]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_text = format_cmd(cmd)
    write_text(log_path.with_suffix(".cmd.txt"), cmd_text + "\n")

    proc = None
    try:
        with open(log_path, "w") as log_file:
            log_file.write(f"# cmd: {cmd_text}\n")
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=REPO_ROOT,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout_s)
            except KeyboardInterrupt:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                raise
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                return False, f"runner timed out after {timeout_s} seconds"
    except Exception as exc:
        return False, f"runner failed to start: {exc}"

    if proc is None:
        return False, "runner failed to start"

    if proc.returncode != 0:
        log_tail = read_tail(log_path)
        error_msg = log_tail[-2000:] if log_tail else "Unknown error"
        return False, f"runner failed (code {proc.returncode}): {error_msg}"

    return True, None


def build_summary_records(
    summary_records: List[Dict[str, Any]],
    models: List[str],
    tasks: List[str],
    calib_samples: int,
    seed: int,
    log_path: Path,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not summary_records:
        for model in models:
            for task in tasks:
                records.append(
                    asdict(
                        SummaryRecord(
                            calib_samples=calib_samples,
                            preserve_energy=PRESERVE_ENERGY,
                            model=model,
                            task=task,
                            metric=None,
                            baseline=None,
                            abs_select=None,
                            smooth_abs=None,
                            random_index=None,
                            grad_direction=None,
                            status="error",
                            log_path=str(log_path),
                            seed=seed,
                        )
                    )
                )
        return records

    for model in models:
        for task in tasks:
            values, metric, status = aggregate_metrics(summary_records, model, task)
            records.append(
                asdict(
                    SummaryRecord(
                        calib_samples=calib_samples,
                        preserve_energy=PRESERVE_ENERGY,
                        model=model,
                        task=task,
                        metric=metric,
                        baseline=values.get("baseline"),
                        abs_select=values.get("abs_select"),
                        smooth_abs=values.get("smooth_abs"),
                        random_index=values.get("random_index"),
                        grad_direction=values.get("grad_direction"),
                        status=status,
                        log_path=str(log_path),
                        seed=seed,
                    )
                )
            )
    return records


# ============================================================================
# Main
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep calib_samples with preserve_energy=none for LoRA adapters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs_roots", type=Path, nargs="+", help="Adapter runs roots.")
    group.add_argument("--run_root", action="append", type=Path, help="Single runs root (repeatable).")
    p.add_argument("--out_dir", type=Path, required=True, help="Base output directory for logs/results.")
    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=DEFAULT_TASKS,
        choices=DEFAULT_TASKS,
        help="Tasks to run.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Spectral edit seeds.")
    p.add_argument(
        "--calib_samples",
        type=int,
        nargs="+",
        default=CALIB_SAMPLES,
        help="Calibration sample counts to sweep.",
    )
    p.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=POLICIES,
        choices=POLICIES,
        help="Spectral edit policies.",
    )
    p.add_argument("--models", type=str, nargs="+", default=None, help="Optional model tags for aggregation.")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs already marked ok in results.jsonl.",
    )
    p.add_argument(
        "--runner_timeout_s",
        type=int,
        default=None,
        help="Optional timeout (seconds) for each run_lm_eval invocation.",
    )
    p.add_argument("--dry_run", action="store_true", help="Print planned commands only.")
    return p


def main() -> None:
    parser = build_arg_parser()
    args, extra_args = parser.parse_known_args()

    runs_roots = args.runs_roots or args.run_root or []
    runs_roots = [p.resolve() for p in runs_roots]
    for root in runs_roots:
        if not root.exists():
            print(f"[ERROR] Runs root does not exist: {root}")
            sys.exit(1)

    tasks = args.tasks
    seeds = args.seeds
    calib_samples_list = args.calib_samples
    policies = args.policies

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_models = args.models or discover_models(runs_roots, tasks)
    if not base_models:
        print("[ERROR] Failed to determine model tags. Use --models to specify.")
        sys.exit(1)

    print("=" * 70)
    print("Sweep: calib_samples preserve_energy=none")
    print("=" * 70)
    print(f"Runs roots: {', '.join(str(r) for r in runs_roots)}")
    print(f"Output dir: {out_dir}")
    print(f"Tasks: {tasks}")
    print(f"Seeds: {seeds}")
    print(f"Calib samples: {calib_samples_list}")
    print(f"Policies: {policies}")
    print(f"Models: {base_models}")
    if args.runner_timeout_s:
        print(f"Runner timeout: {args.runner_timeout_s}s")
    if extra_args:
        print(f"Extra args: {shlex.join(extra_args)}")
    print("=" * 70)

    results_path = out_dir / "results.jsonl"
    existing_records = read_jsonl(results_path)
    existing_ok = {record_key(r) for r in existing_records if r.get("status") == "ok"}

    for calib_samples in calib_samples_list:
        for seed in seeds:
            out_root = build_out_root(out_dir, calib_samples, seeds, seed)
            log_path = logs_dir / f"calib{calib_samples}_seed{seed}.log"

            cmd = build_runner_cmd(
                runs_roots=runs_roots,
                out_root=out_root,
                tasks=tasks,
                calib_samples=calib_samples,
                seed=seed,
                policies=policies,
                extra_args=extra_args,
            )

            if args.dry_run:
                print(f"[RUN] calib_samples={calib_samples} seed={seed} out_root={out_root}")
                print("  " + format_cmd(cmd))
                continue

            if args.resume:
                models_for_run = base_models
                expected_keys = {
                    (calib_samples, PRESERVE_ENERGY, model, task, seed)
                    for model in models_for_run
                    for task in tasks
                }
                if expected_keys and expected_keys.issubset(existing_ok):
                    print(f"[SKIP] calib_samples={calib_samples} seed={seed} (all results ok)")
                    continue

                summary_records = parse_summary(out_root)
                summary_models = models_from_summary(summary_records)
                if summary_models:
                    if args.models:
                        models_for_run = sorted(set(base_models).union(summary_models))
                    else:
                        models_for_run = summary_models
                    expected_keys = {
                        (calib_samples, PRESERVE_ENERGY, model, task, seed)
                        for model in models_for_run
                        for task in tasks
                    }
                if summary_records:
                    new_records = build_summary_records(
                        summary_records=summary_records,
                        models=models_for_run,
                        tasks=tasks,
                        calib_samples=calib_samples,
                        seed=seed,
                        log_path=log_path,
                    )
                    record_map = {record_key(r): r for r in existing_records}
                    for rec in new_records:
                        record_map[record_key(rec)] = rec
                    existing_records = list(record_map.values())
                    write_jsonl_atomic(results_path, existing_records)
                    existing_ok = {record_key(r) for r in existing_records if r.get("status") == "ok"}
                    if expected_keys and expected_keys.issubset(existing_ok):
                        print(f"[SKIP] calib_samples={calib_samples} seed={seed} (summary already ok)")
                        continue

            print(f"[RUN] calib_samples={calib_samples} seed={seed} out_root={out_root}")
            success, error = run_command(cmd, log_path, args.runner_timeout_s)
            if not success:
                print(
                    f"[WARN] runner failed for calib_samples={calib_samples} seed={seed}: {error}"
                )

            summary_records = parse_summary(out_root)
            if not summary_records:
                print(f"[WARN] summary.json missing or empty: {out_root}")

            models_for_run = base_models
            summary_models = models_from_summary(summary_records)
            if summary_models:
                if args.models:
                    models_for_run = sorted(set(base_models).union(summary_models))
                else:
                    models_for_run = summary_models

            new_records = build_summary_records(
                summary_records=summary_records,
                models=models_for_run,
                tasks=tasks,
                calib_samples=calib_samples,
                seed=seed,
                log_path=log_path,
            )

            record_map = {record_key(r): r for r in existing_records}
            for rec in new_records:
                record_map[record_key(rec)] = rec
            existing_records = list(record_map.values())
            write_jsonl_atomic(results_path, existing_records)
            existing_ok = {record_key(r) for r in existing_records if r.get("status") == "ok"}


if __name__ == "__main__":
    main()
