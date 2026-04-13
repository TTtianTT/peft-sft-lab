#!/usr/bin/env python3
"""
Run continued_ft HP grid on priority settings with 8-GPU parallelism.

Grid: lr ∈ {1e-5, 5e-5} × steps ∈ {10, 20, 50} = 6 configs
Settings: Qwen/{math,alpaca}, Llama/{math,csqa} = 4 adapters
Total: 24 editing jobs → 8 GPUs parallel

Then evaluates all successful results.

Usage:
    python scripts/rebuttal/run_continued_ft_grid.py --n_gpus 8
    python scripts/rebuttal/run_continued_ft_grid.py --n_gpus 8 --eval_only
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"

# ============================================================================
# Priority settings
# ============================================================================

ADAPTERS = [
    {
        "base_model_id": "Qwen/Qwen3-8B",
        "base_model_dir": "Qwen-Qwen3-8B",
        "task": "math",
        "adapter_dir": "runs_refactor_data_20260121/Qwen-Qwen3-8B/math/lora/profile-paper_math_ift_3ep/rank-16/seed42",
    },
    {
        "base_model_id": "Qwen/Qwen3-8B",
        "base_model_dir": "Qwen-Qwen3-8B",
        "task": "alpaca",
        "adapter_dir": "runs_refactor_data_20260121/Qwen-Qwen3-8B/alpaca/lora/profile-paper_alpaca_3ep/rank-16/seed42",
    },
    {
        "base_model_id": "meta-llama/Llama-3.1-8B",
        "base_model_dir": "meta-llama-Llama-3.1-8B",
        "task": "math",
        "adapter_dir": "runs_refactor_data_20260121/meta-llama-Llama-3.1-8B/math/lora/profile-paper_math_ift_3ep/rank-16/seed42",
    },
    {
        "base_model_id": "meta-llama/Llama-3.1-8B",
        "base_model_dir": "meta-llama-Llama-3.1-8B",
        "task": "csqa",
        "adapter_dir": "runs_refactor_data_20260121/meta-llama-Llama-3.1-8B/csqa/lora/profile-paper_csqa_3ep/rank-16/seed42",
    },
]

LR_GRID = [1e-5, 5e-5]
STEPS_GRID = [10, 20, 50]

OUT_ROOT = Path("outputs/rebuttal_exp/raw/continued_ft_grid")
RESULTS_JSONL = OUT_ROOT / "results.jsonl"


class GPUPool:
    def __init__(self, n_gpus: int):
        self._q: Queue = Queue()
        for i in range(n_gpus):
            self._q.put(i)

    def acquire(self) -> int:
        return self._q.get()

    def release(self, gpu_id: int):
        self._q.put(gpu_id)


def _results_lock():
    return threading.Lock()

_lock = threading.Lock()


def load_done_keys() -> set:
    keys = set()
    if RESULTS_JSONL.exists():
        with open(RESULTS_JSONL) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                keys.add(rec.get("key", ""))
    return keys


def run_one_ft(
    adapter: Dict,
    lr: float,
    steps: int,
    gpu_id: int,
    seed: int = 42,
) -> Dict:
    """Run a single continued_ft job. Returns result dict."""
    tag = f"{adapter['base_model_dir']}_{adapter['task']}_lr{lr}_s{steps}"
    out_dir = OUT_ROOT / "adapters" / tag

    # Check if meta already exists
    meta_path = out_dir / "continued_ft_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return {
            "key": tag,
            "base_model_id": adapter["base_model_id"],
            "base_model_dir": adapter["base_model_dir"],
            "task": adapter["task"],
            "adapter_dir": adapter["adapter_dir"],
            "lr": lr,
            "steps": steps,
            "seed": seed,
            "out_dir": str(out_dir),
            "error": None,
            "meta": meta,
        }

    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_continued_lora_ft.py"),
        "--base_model", adapter["base_model_id"],
        "--adapter_dir", adapter["adapter_dir"],
        "--out_dir", str(out_dir),
        "--task", adapter["task"],
        "--calib_samples", "128",
        "--calib_batch_size", "2",
        "--steps", str(steps),
        "--lr", str(lr),
        "--seed", str(seed),
        "--calib_seed", str(seed),
        "--calib_shuffle",
        "--proxy_samples", "128",
    ]

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "PYTHONPATH": str(SRC_DIR),
    }

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
            cwd=str(REPO_ROOT), env=env,
        )
        if result.returncode != 0:
            error = result.stderr[-2000:] if result.stderr else "Unknown"
            return {
                "key": tag, "base_model_id": adapter["base_model_id"],
                "base_model_dir": adapter["base_model_dir"],
                "task": adapter["task"], "adapter_dir": adapter["adapter_dir"],
                "lr": lr, "steps": steps, "seed": seed,
                "out_dir": str(out_dir), "error": error, "meta": None,
            }

        meta = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        return {
            "key": tag, "base_model_id": adapter["base_model_id"],
            "base_model_dir": adapter["base_model_dir"],
            "task": adapter["task"], "adapter_dir": adapter["adapter_dir"],
            "lr": lr, "steps": steps, "seed": seed,
            "out_dir": str(out_dir), "error": None, "meta": meta,
        }

    except subprocess.TimeoutExpired:
        return {
            "key": tag, "base_model_id": adapter["base_model_id"],
            "base_model_dir": adapter["base_model_dir"],
            "task": adapter["task"], "adapter_dir": adapter["adapter_dir"],
            "lr": lr, "steps": steps, "seed": seed,
            "out_dir": str(out_dir), "error": "Timeout", "meta": None,
        }


def process_job(job, idx, total, gpu_pool):
    adapter, lr, steps = job
    gpu_id = gpu_pool.acquire()
    tag = f"{adapter['base_model_dir']}_{adapter['task']}_lr{lr}_s{steps}"
    try:
        print(f"  [{idx+1}/{total}] GPU{gpu_id} {tag}")
        t0 = time.time()
        result = run_one_ft(adapter, lr, steps, gpu_id)
        elapsed = time.time() - t0

        # Save result
        with _lock:
            with open(RESULTS_JSONL, "a") as f:
                # Don't save full meta in JSONL (too big), just key fields
                slim = {k: v for k, v in result.items() if k != "meta"}
                if result.get("meta"):
                    slim["loss_initial"] = result["meta"].get("loss_initial")
                    slim["loss_final"] = result["meta"].get("loss_final")
                    slim["edit_loss_before"] = result["meta"].get("edit_loss_before")
                    slim["edit_loss_after"] = result["meta"].get("edit_loss_after")
                    slim["proxy_loss_before"] = result["meta"].get("proxy_loss_before")
                    slim["proxy_loss_after"] = result["meta"].get("proxy_loss_after")
                f.write(json.dumps(slim) + "\n")

        if result.get("error"):
            print(f"  [{idx+1}/{total}] GPU{gpu_id} FAILED {tag}: {result['error'][:80]}")
        else:
            meta = result.get("meta", {}) or {}
            li = meta.get("loss_initial", "?")
            lf = meta.get("loss_final", "?")
            pb = meta.get("proxy_loss_before", "?")
            pa = meta.get("proxy_loss_after", "?")
            li_s = f"{li:.4f}" if isinstance(li, (int, float)) else li
            lf_s = f"{lf:.4f}" if isinstance(lf, (int, float)) else lf
            pb_s = f"{pb:.4f}" if isinstance(pb, (int, float)) else pb
            pa_s = f"{pa:.4f}" if isinstance(pa, (int, float)) else pa
            print(f"  [{idx+1}/{total}] GPU{gpu_id} OK {tag}: loss {li_s}→{lf_s}, proxy {pb_s}→{pa_s} ({elapsed:.0f}s)")
        return result
    finally:
        gpu_pool.release(gpu_id)


def write_eval_results_jsonl(results: List[Dict]):
    """Write results in the format expected by run_eval_from_results.py."""
    eval_jsonl = OUT_ROOT / "for_eval.jsonl"
    with open(eval_jsonl, "w") as f:
        for r in results:
            if r.get("error"):
                continue
            rec = {
                "base_model_id": r["base_model_id"],
                "base_model_dir": r["base_model_dir"],
                "task": r["task"],
                "adapter_dir": r["adapter_dir"],
                "method": f"continued_ft_lr{r['lr']}_s{r['steps']}",
                "repeat_seed": r["seed"],
                "edited_adapter_dir": r["out_dir"],
                "error": None,
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {eval_jsonl} for evaluation")
    return eval_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs = list(itertools.product(ADAPTERS, LR_GRID, STEPS_GRID))
    print(f"Total grid jobs: {len(jobs)} ({len(ADAPTERS)} settings × {len(LR_GRID)} lr × {len(STEPS_GRID)} steps)")

    if not args.eval_only:
        # Check which are already done
        done_keys = load_done_keys()
        todo = []
        for adapter, lr, steps in jobs:
            tag = f"{adapter['base_model_dir']}_{adapter['task']}_lr{lr}_s{steps}"
            meta_path = OUT_ROOT / "adapters" / tag / "continued_ft_meta.json"
            if tag not in done_keys and not meta_path.exists():
                todo.append((adapter, lr, steps))
            else:
                # Still record if meta exists but not in jsonl
                if meta_path.exists() and tag not in done_keys:
                    todo.append((adapter, lr, steps))

        print(f"Todo: {len(todo)}, Already done: {len(jobs) - len(todo)}")

        if todo:
            gpu_pool = GPUPool(args.n_gpus)
            with ThreadPoolExecutor(max_workers=args.n_gpus) as executor:
                futures = {}
                for i, job in enumerate(todo):
                    f = executor.submit(process_job, job, i, len(todo), gpu_pool)
                    futures[f] = job
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        job = futures[f]
                        print(f"  [ERROR] {job[0]['task']}/lr{job[1]}/s{job[2]}: {e}")

    # Collect all results
    results = []
    if RESULTS_JSONL.exists():
        with open(RESULTS_JSONL) as f:
            results = [json.loads(l) for l in f if l.strip()]

    ok = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]
    print(f"\nGrid complete: {len(ok)} OK, {len(err)} errors out of {len(jobs)} total")

    # Print summary table
    print(f"\n{'Setting':<35s} {'LR':>8s} {'Steps':>6s} {'Loss_i':>8s} {'Loss_f':>8s} {'Proxy_b':>8s} {'Proxy_a':>8s} {'ΔProxy':>8s}")
    print("-" * 110)
    for r in sorted(ok, key=lambda x: (x["base_model_dir"], x["task"], x["lr"], x["steps"])):
        tag = f"{r['base_model_dir']}/{r['task']}"
        li = r.get("loss_initial", None)
        lf = r.get("loss_final", None)
        pb = r.get("proxy_loss_before", None)
        pa = r.get("proxy_loss_after", None)
        dp = (pb - pa) if (pb is not None and pa is not None) else None
        fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
        print(f"{tag:<35s} {r['lr']:>8.0e} {r['steps']:>6d} {fmt(li):>8s} {fmt(lf):>8s} {fmt(pb):>8s} {fmt(pa):>8s} {fmt(dp):>8s}")

    # Write for_eval.jsonl
    eval_jsonl = write_eval_results_jsonl(ok)

    print(f"\nTo evaluate:")
    print(f"  python scripts/rebuttal/run_eval_from_results.py \\")
    print(f"    --results_jsonl {eval_jsonl} \\")
    print(f"    --eval_root {OUT_ROOT / 'eval'} \\")
    print(f"    --seeds {args.seed} --n_gpus {args.n_gpus}")


if __name__ == "__main__":
    main()
