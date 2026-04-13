#!/usr/bin/env python3
"""
Evaluate edited adapters from multiseed results using EXISTING eval scripts.

Uses native vLLM LoRA loading (no merge needed). Runs up to N_GPUS evals
in parallel, one per GPU.

Usage:
    # Evaluate seed=42, 8 GPUs
    python scripts/rebuttal/run_eval_from_results.py \
        --results_jsonl outputs/rebuttal_exp/raw/multiseed/results.jsonl \
        --eval_root outputs/rebuttal_exp/raw/multiseed_eval \
        --seeds 42 --n_gpus 8

    # Evaluate all seeds, specific task
    python scripts/rebuttal/run_eval_from_results.py \
        --results_jsonl outputs/rebuttal_exp/raw/multiseed/results.jsonl \
        --eval_root outputs/rebuttal_exp/raw/multiseed_eval \
        --tasks math --n_gpus 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"

# ============================================================================
# Constants — aligned with run_refactor_lora_spectral_edit_repeat.py
# ============================================================================

BASE_MODEL_DIR_TO_ID = {
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
}

# Map our task names to the internal task names used by eval scripts
TASK_TO_INTERNAL = {
    "math": "metamath",
    "code": "magicoder",
    "alpaca": "alpaca",
    "csqa": "csqa",
}

# Eval script modules (same as repeat runner)
TASK_TO_EVAL_SCRIPT = {
    "metamath": "finetune.eval.eval_gsm8k",
    "magicoder": "finetune.eval.eval_humaneval",
    "alpaca": "finetune.eval.eval_ifeval",
    "csqa": "finetune.eval.eval_csqa",
}

# Primary metric key in metrics.json (actual keys from eval scripts)
TASK_TO_METRIC = {
    "metamath": "accuracy_strict",
    "magicoder": "pass@1",
    "alpaca": "prompt_level_strict_acc",
    "csqa": "accuracy",
}


@dataclass
class EvalRecord:
    timestamp: str
    base_model_id: str
    base_model_dir: str
    task: str
    method: str
    repeat_seed: int
    adapter_dir: str
    edited_adapter_dir: Optional[str]
    eval_output_dir: str
    metric_name: str
    metric_value: Optional[float]
    all_metrics: Optional[Dict]
    error: Optional[str]


class EvalWriter:
    """Thread-safe incremental results writer with resume support."""

    def __init__(self, out_root: Path):
        self.out_root = out_root
        self.jsonl_path = out_root / "eval_results.jsonl"
        self.csv_path = out_root / "eval_results.csv"
        out_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.existing_keys: set = set()
        self._load_existing()

    def _key(self, adapter_dir: str, method: str, seed: int) -> str:
        return f"{adapter_dir}|{method}|{seed}"

    def _load_existing(self):
        if self.jsonl_path.exists():
            try:
                with open(self.jsonl_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        key = self._key(
                            rec.get("adapter_dir", ""),
                            rec.get("method", ""),
                            rec.get("repeat_seed", 0),
                        )
                        self.existing_keys.add(key)
            except Exception:
                pass

    def is_done(self, adapter_dir: str, method: str, seed: int) -> bool:
        return self._key(adapter_dir, method, seed) in self.existing_keys

    def write(self, result: EvalRecord):
        with self._lock:
            key = self._key(result.adapter_dir, result.method, result.repeat_seed)
            if key in self.existing_keys:
                return
            self.existing_keys.add(key)
            d = asdict(result)
            d_csv = {
                **d,
                "all_metrics": json.dumps(d.get("all_metrics"))
                if d.get("all_metrics")
                else None,
            }
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(d) + "\n")
            exists = self.csv_path.exists()
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(d_csv.keys()))
                if not exists:
                    w.writeheader()
                w.writerow(d_csv)


class GPUPool:
    """Simple GPU allocator: hand out GPU IDs, return when done."""

    def __init__(self, n_gpus: int):
        self._q: Queue = Queue()
        for i in range(n_gpus):
            self._q.put(i)

    def acquire(self) -> int:
        return self._q.get()

    def release(self, gpu_id: int):
        self._q.put(gpu_id)


def run_single_eval(
    base_model_id: str,
    adapter_dir: Optional[str],
    task: str,
    output_dir: Path,
    gpu_id: int,
    seed: int = 42,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Run evaluation using existing eval scripts (native vLLM LoRA, no merge).
    Aligned with run_refactor_lora_spectral_edit_repeat.py::run_evaluation().
    """
    internal_task = TASK_TO_INTERNAL.get(task, task)
    eval_module = TASK_TO_EVAL_SCRIPT.get(internal_task)
    if not eval_module:
        return None, f"Unknown task: {task} (internal: {internal_task})"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already evaluated (metrics.json exists)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            return metrics, None
        except Exception:
            pass  # re-run if metrics.json is corrupted

    cmd = [
        sys.executable, "-m", eval_module,
        "--base_model", base_model_id,
        "--output_dir", str(output_dir),
        "--seed", str(seed),
    ]

    if adapter_dir:
        cmd.extend(["--adapter_dir", str(adapter_dir)])

    # Always use vLLM with tp=1 (one GPU per eval)
    cmd.append("--use_vllm")
    cmd.extend(["--tensor_parallel_size", "1"])

    # Task-specific args (aligned with repeat runner)
    if internal_task == "metamath":
        cmd.extend(["--max_new_tokens", "256"])
    elif internal_task == "magicoder":
        cmd.extend(["--max_new_tokens", "256", "--timeout_s", "3.0"])
    elif internal_task == "alpaca":
        cmd.extend(["--max_new_tokens", "2048", "--split", "train"])
    elif internal_task == "csqa":
        cmd.extend(["--max_new_tokens", "8"])

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "PYTHONPATH": str(SRC_DIR),
    }

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=str(REPO_ROOT),
            env=env,
        )

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "Unknown error"
            return None, f"Eval failed (code {result.returncode}): {error_msg}"

        if not metrics_path.exists():
            return None, "Eval completed but no metrics.json found"

        with open(metrics_path) as f:
            metrics = json.load(f)

        return metrics, None

    except subprocess.TimeoutExpired:
        return None, "Eval timed out after 2 hours"
    except Exception as e:
        return None, f"Eval failed with exception: {e}"


def process_job(
    job: Dict,
    idx: int,
    total: int,
    writer: EvalWriter,
    gpu_pool: GPUPool,
    eval_root: Path,
) -> str:
    """Process a single eval job. Called from thread pool."""
    model_dir = job["base_model_dir"]
    task = job["task"]
    method = job["method"]
    seed = job["repeat_seed"]
    base_model_id = BASE_MODEL_DIR_TO_ID.get(model_dir)
    tag = f"{model_dir}_{task}_{method}_s{seed}"

    if not base_model_id:
        return f"[{idx+1}/{total}] SKIP {tag} (unknown model)"

    # Determine adapter path
    if method == "baseline":
        adapter_path = job["adapter_dir"]
    else:
        adapter_path = job.get("edited_adapter_dir")
        if not adapter_path or not Path(adapter_path).exists():
            return f"[{idx+1}/{total}] SKIP {tag} (no adapter)"

    eval_dir = eval_root / "eval_outputs" / tag
    internal_task = TASK_TO_INTERNAL.get(task, task)
    metric_name = TASK_TO_METRIC.get(internal_task, task)

    # Acquire GPU
    gpu_id = gpu_pool.acquire()
    try:
        print(f"  [{idx+1}/{total}] GPU{gpu_id} {model_dir}/{task}/{method}/s{seed}")

        t0 = time.time()
        metrics, error = run_single_eval(
            base_model_id=base_model_id,
            adapter_dir=adapter_path,
            task=task,
            output_dir=eval_dir,
            gpu_id=gpu_id,
            seed=seed,
        )
        elapsed = time.time() - t0

        if error:
            metric_value = None
            all_metrics = None
            msg = f"  [{idx+1}/{total}] GPU{gpu_id} FAILED {tag}: {error[:100]}"
        else:
            metric_value = metrics.get(metric_name, metrics.get("accuracy", None))
            if metric_value is not None:
                metric_value = float(metric_value)
            all_metrics = metrics
            val_str = f"{metric_value:.4f}" if metric_value is not None else "N/A"
            msg = f"  [{idx+1}/{total}] GPU{gpu_id} OK {tag}: {metric_name}={val_str} ({elapsed:.0f}s)"

        writer.write(EvalRecord(
            timestamp=datetime.now().isoformat(),
            base_model_id=base_model_id,
            base_model_dir=model_dir,
            task=task,
            method=method,
            repeat_seed=seed,
            adapter_dir=job["adapter_dir"],
            edited_adapter_dir=job.get("edited_adapter_dir"),
            eval_output_dir=str(eval_dir),
            metric_name=metric_name,
            metric_value=metric_value,
            all_metrics=all_metrics,
            error=error,
        ))

        print(msg)
        return msg

    finally:
        gpu_pool.release(gpu_id)


def main():
    parser = argparse.ArgumentParser(description="Parallel eval using existing eval scripts")
    parser.add_argument("--results_jsonl", type=str, required=True)
    parser.add_argument("--eval_root", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--n_gpus", type=int, default=8)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    writer = EvalWriter(eval_root)

    # Load editing results
    with open(args.results_jsonl) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Filter
    jobs = []
    for r in records:
        if r.get("error"):
            continue
        if args.seeds and r["repeat_seed"] not in args.seeds:
            continue
        if args.tasks and r["task"] not in args.tasks:
            continue
        if args.methods and r["method"] not in args.methods:
            continue
        if args.models and r["base_model_dir"] not in args.models:
            continue
        jobs.append(r)

    print(f"Total eval jobs: {len(jobs)}")
    if not jobs:
        print("Nothing to evaluate!")
        return

    # Check which are already done
    todo = [
        j
        for j in jobs
        if not writer.is_done(j["adapter_dir"], j["method"], j["repeat_seed"])
    ]
    print(f"Already done: {len(jobs) - len(todo)}, Todo: {len(todo)}")
    print(f"Using {args.n_gpus} GPUs in parallel")

    if not todo:
        print("All jobs already evaluated!")
    else:
        gpu_pool = GPUPool(args.n_gpus)

        with ThreadPoolExecutor(max_workers=args.n_gpus) as executor:
            futures = {}
            for i, job in enumerate(todo):
                f = executor.submit(
                    process_job, job, i, len(todo), writer, gpu_pool, eval_root
                )
                futures[f] = job

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    job = futures[f]
                    print(f"  [ERROR] {job['base_model_dir']}/{job['task']}/{job['method']}: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    if writer.jsonl_path.exists():
        with open(writer.jsonl_path) as f:
            all_evals = [json.loads(l) for l in f if l.strip()]

        groups = defaultdict(list)
        for e in all_evals:
            if e.get("metric_value") is not None:
                groups[(e["base_model_dir"], e["task"], e["method"])].append(
                    e["metric_value"]
                )

        print(
            f"\n{'Model':<28s} {'Task':<8s} {'Method':<20s} {'N':>3s} {'Metric':>8s}"
        )
        print("-" * 80)
        for model, task, method in sorted(groups.keys()):
            vals = groups[(model, task, method)]
            mean_v = sum(vals) / len(vals)
            print(
                f"{model:<28s} {task:<8s} {method:<20s} {len(vals):>3d} {mean_v:>8.4f}"
            )

    print(f"\nResults: {writer.jsonl_path}")


if __name__ == "__main__":
    main()
