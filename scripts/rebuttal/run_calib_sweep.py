#!/usr/bin/env python3
"""
P1-F: Calibration-Size Sweep with 8-GPU parallelism.

Runs editing methods across different calibration sizes (Ncal) to study
the effect of calibration data quantity on each method.

Phase 1 (edit): Parallel editing on 8 GPUs
Phase 2 (eval): Use run_eval_from_results.py with native vLLM LoRA

Ncal in {32, 64, 128, 256}
Methods: random_index, smooth_abs, grad_direction, continued_ft, sigma_only
Settings: Qwen/{math,alpaca,csqa}, Llama/{math,alpaca,csqa}
Seeds: 3 per configuration (42, 43, 44)

Usage:
    # Edit only (8 GPUs, ~45 min)
    python scripts/rebuttal/run_calib_sweep.py --n_gpus 8

    # Then evaluate
    python scripts/rebuttal/run_eval_from_results.py \
        --results_jsonl outputs/rebuttal_exp/raw/calib_sweep/for_eval.jsonl \
        --eval_root outputs/rebuttal_exp/raw/calib_sweep/eval \
        --seeds 42 43 44 --n_gpus 8
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"

# ============================================================================
# Priority settings (4 adapters)
# ============================================================================

ADAPTERS = [
    {
        "base_model_id": "Qwen/Qwen3-8B",
        "base_model_dir": "Qwen-Qwen3-8B",
        "task": "math",
        "adapter_dir": "runs_refactor_data_20260121/Qwen-Qwen3-8B/math/lora/profile-paper_math_ift_3ep/rank-16/seed42",
        "calib_dataset": "gsm8k",
        "calib_config": "main",
    },
    {
        "base_model_id": "Qwen/Qwen3-8B",
        "base_model_dir": "Qwen-Qwen3-8B",
        "task": "alpaca",
        "adapter_dir": "runs_refactor_data_20260121/Qwen-Qwen3-8B/alpaca/lora/profile-paper_alpaca_3ep/rank-16/seed42",
        "calib_dataset": "tatsu-lab/alpaca",
        "calib_config": None,
    },
    {
        "base_model_id": "Qwen/Qwen3-8B",
        "base_model_dir": "Qwen-Qwen3-8B",
        "task": "csqa",
        "adapter_dir": "runs_refactor_data_20260121/Qwen-Qwen3-8B/csqa/lora/profile-paper_csqa_3ep/rank-16/seed42",
        "calib_dataset": "tau/commonsense_qa",
        "calib_config": None,
    },
    {
        "base_model_id": "meta-llama/Llama-3.1-8B",
        "base_model_dir": "meta-llama-Llama-3.1-8B",
        "task": "math",
        "adapter_dir": "runs_refactor_data_20260121/meta-llama-Llama-3.1-8B/math/lora/profile-paper_math_ift_3ep/rank-16/seed42",
        "calib_dataset": "gsm8k",
        "calib_config": "main",
    },
    {
        "base_model_id": "meta-llama/Llama-3.1-8B",
        "base_model_dir": "meta-llama-Llama-3.1-8B",
        "task": "alpaca",
        "adapter_dir": "runs_refactor_data_20260121/meta-llama-Llama-3.1-8B/alpaca/lora/profile-paper_alpaca_3ep/rank-16/seed42",
        "calib_dataset": "tatsu-lab/alpaca",
        "calib_config": None,
    },
    {
        "base_model_id": "meta-llama/Llama-3.1-8B",
        "base_model_dir": "meta-llama-Llama-3.1-8B",
        "task": "csqa",
        "adapter_dir": "runs_refactor_data_20260121/meta-llama-Llama-3.1-8B/csqa/lora/profile-paper_csqa_3ep/rank-16/seed42",
        "calib_dataset": "tau/commonsense_qa",
        "calib_config": None,
    },
]

# Spectral edit method → CLI mode
METHOD_TO_MODE = {
    "random_index": "random_index",
    "smooth_abs": "smooth_abs",
    "grad_direction": "gd",
}

SPECTRAL_METHODS = ["random_index", "smooth_abs", "grad_direction"]
TRAIN_METHODS = ["continued_ft", "sigma_only"]
EDIT_METHODS = SPECTRAL_METHODS + TRAIN_METHODS

OUT_ROOT = Path("outputs/rebuttal_exp/raw/calib_sweep")
RESULTS_JSONL = OUT_ROOT / "results.jsonl"

_lock = threading.Lock()


class GPUPool:
    def __init__(self, n_gpus: int):
        self._q: Queue = Queue()
        for i in range(n_gpus):
            self._q.put(i)

    def acquire(self) -> int:
        return self._q.get()

    def release(self, gpu_id: int):
        self._q.put(gpu_id)


def _build_spectral_cmd(adapter: Dict, method: str, ncal: int, seed: int,
                         out_dir: Path) -> List[str]:
    """Build spectral edit CLI command."""
    mode = METHOD_TO_MODE[method]
    cmd = [
        sys.executable, "-m", "finetune.spectral_edit.cli", "edit",
        "--base_model", adapter["base_model_id"],
        "--lora_path", adapter["adapter_dir"],
        "--out_dir", str(out_dir),
        "--mode", mode,
        "--target_modules", "down_proj", "o_proj",
        "--calib_samples", str(ncal),
        "--calib_batch_size", "2",
        "--calib_dataset", adapter["calib_dataset"],
        "--seed", str(seed),
        "--calib_seed", str(seed),
        "--calib_shuffle",
        "--grad_norm", "mean_abs",
        "--preserve_energy", "l1",
    ]
    if adapter["calib_config"]:
        cmd.extend(["--calib_config", adapter["calib_config"]])

    # Method-specific args
    if method in ["random_index"]:
        cmd.extend(["--core_frac", "0.2", "--noise_frac", "0.2",
                     "--amp_factor", "1.25", "--sup_factor", "0.80"])
    elif method == "smooth_abs":
        cmd.extend(["--smooth_temperature", "0.35", "--smooth_center_q", "0.5",
                     "--amp_factor", "1.25", "--sup_factor", "0.80"])
    elif method == "grad_direction":
        cmd.extend(["--update_mode", "multiplicative", "--asymmetric_update",
                     "--eta_suppress", "2.0", "--eta_enhance", "0.2"])
    return cmd


def _build_continued_ft_cmd(adapter: Dict, ncal: int, seed: int,
                             out_dir: Path, steps: int, lr: float) -> List[str]:
    """Build continued_ft command."""
    return [
        sys.executable, str(SCRIPT_DIR / "run_continued_lora_ft.py"),
        "--base_model", adapter["base_model_id"],
        "--adapter_dir", adapter["adapter_dir"],
        "--out_dir", str(out_dir),
        "--task", adapter["task"],
        "--calib_samples", str(ncal),
        "--calib_batch_size", "2",
        "--steps", str(steps),
        "--lr", str(lr),
        "--seed", str(seed),
        "--calib_seed", str(seed),
        "--calib_shuffle",
        "--proxy_samples", "128",
    ]


def _build_sigma_only_cmd(adapter: Dict, ncal: int, seed: int,
                           out_dir: Path, steps: int, lr: float) -> List[str]:
    """Build sigma_only command."""
    return [
        sys.executable, str(SCRIPT_DIR / "run_sigma_only_train.py"),
        "--base_model", adapter["base_model_id"],
        "--adapter_dir", adapter["adapter_dir"],
        "--out_dir", str(out_dir),
        "--task", adapter["task"],
        "--calib_samples", str(ncal),
        "--calib_batch_size", "2",
        "--steps", str(steps),
        "--lr", str(lr),
        "--seed", str(seed),
        "--calib_seed", str(seed),
        "--calib_shuffle",
        "--preserve_energy", "none",
    ]


def _job_tag(adapter: Dict, method: str, ncal: int, seed: int) -> str:
    return f"{adapter['base_model_dir']}_{adapter['task']}_{method}_ncal{ncal}_s{seed}"


def _out_dir(adapter: Dict, method: str, ncal: int, seed: int) -> Path:
    return OUT_ROOT / "edited" / _job_tag(adapter, method, ncal, seed)


def _is_done(out_dir: Path, method: str) -> bool:
    """Check if edit output already exists."""
    if method in SPECTRAL_METHODS:
        return (out_dir / "adapter_model.safetensors").exists() or \
               (out_dir / "adapter_model.bin").exists()
    elif method == "continued_ft":
        return (out_dir / "continued_ft_meta.json").exists()
    elif method == "sigma_only":
        return (out_dir / "sigma_only_train_meta.json").exists()
    return False


def run_one_edit(
    adapter: Dict,
    method: str,
    ncal: int,
    seed: int,
    gpu_id: int,
    ft_steps: int = 50,
    ft_lr: float = 5e-5,
    sigma_steps: int = 50,
    sigma_lr: float = 5e-3,
) -> Dict:
    """Run a single editing job on a specific GPU."""
    out_dir = _out_dir(adapter, method, ncal, seed)
    tag = _job_tag(adapter, method, ncal, seed)

    # Skip if done
    if _is_done(out_dir, method):
        return {"tag": tag, "out_dir": str(out_dir), "error": None, "skipped": True}

    if method in SPECTRAL_METHODS:
        cmd = _build_spectral_cmd(adapter, method, ncal, seed, out_dir)
    elif method == "continued_ft":
        cmd = _build_continued_ft_cmd(adapter, ncal, seed, out_dir, ft_steps, ft_lr)
    elif method == "sigma_only":
        cmd = _build_sigma_only_cmd(adapter, ncal, seed, out_dir, sigma_steps, sigma_lr)
    else:
        return {"tag": tag, "out_dir": str(out_dir), "error": f"Unknown method: {method}"}

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "PYTHONPATH": f"{SRC_DIR}:{os.environ.get('PYTHONPATH', '')}",
        "PYTHONUNBUFFERED": "1",
    }

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
            cwd=str(REPO_ROOT), env=env,
        )
        if result.returncode != 0:
            error = result.stderr[-2000:] if result.stderr else "Unknown"
            return {"tag": tag, "out_dir": str(out_dir), "error": error, "skipped": False}
        return {"tag": tag, "out_dir": str(out_dir), "error": None, "skipped": False}
    except subprocess.TimeoutExpired:
        return {"tag": tag, "out_dir": str(out_dir), "error": "Timeout", "skipped": False}
    except Exception as e:
        return {"tag": tag, "out_dir": str(out_dir), "error": str(e), "skipped": False}


def write_for_eval_jsonl(adapters: List[Dict], methods: List[str],
                          ncals: List[int], seeds: List[int]) -> Path:
    """Write for_eval.jsonl for run_eval_from_results.py."""
    eval_jsonl = OUT_ROOT / "for_eval.jsonl"
    records = []
    for adapter in adapters:
        for ncal in ncals:
            for method in methods:
                for seed in seeds:
                    out_dir = _out_dir(adapter, method, ncal, seed)
                    if _is_done(out_dir, method):
                        records.append({
                            "base_model_id": adapter["base_model_id"],
                            "base_model_dir": adapter["base_model_dir"],
                            "task": adapter["task"],
                            "adapter_dir": adapter["adapter_dir"],
                            "method": f"{method}_ncal{ncal}",
                            "repeat_seed": seed,
                            "edited_adapter_dir": str(out_dir),
                            "error": None,
                        })
            # Baseline is Ncal-independent; reuse from multiseed eval

    with open(eval_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} records to {eval_jsonl}")
    return eval_jsonl


def main():
    parser = argparse.ArgumentParser(description="Calibration-Size Sweep (P1-F)")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--ncals", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["random_index", "smooth_abs", "grad_direction",
                                 "continued_ft", "sigma_only"])
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--base_seed", type=int, default=42)
    # FT / sigma hyperparams
    parser.add_argument("--ft_steps", type=int, default=50)
    parser.add_argument("--ft_lr", type=float, default=5e-5)
    parser.add_argument("--sigma_steps", type=int, default=50)
    parser.add_argument("--sigma_lr", type=float, default=5e-3)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    seeds = [args.base_seed + i for i in range(args.n_repeats)]
    methods = [m for m in args.methods if m != "baseline"]  # baseline needs no editing

    # Build job list
    jobs = list(itertools.product(ADAPTERS, methods, args.ncals, seeds))
    total = len(jobs)
    print(f"Total editing jobs: {total} "
          f"({len(ADAPTERS)} settings x {len(methods)} methods x "
          f"{len(args.ncals)} ncals x {len(seeds)} seeds)")

    if not args.eval_only:
        # Count already done
        already_done = sum(1 for a, m, n, s in jobs if _is_done(_out_dir(a, m, n, s), m))
        todo = total - already_done
        print(f"Already done: {already_done}, Todo: {todo}")

        if todo > 0:
            gpu_pool = GPUPool(args.n_gpus)
            ok_count, err_count, skip_count = 0, 0, 0

            def process_job(job_args):
                adapter, method, ncal, seed = job_args
                gpu_id = gpu_pool.acquire()
                tag = _job_tag(adapter, method, ncal, seed)
                try:
                    # Stagger model loading to avoid GPU contention
                    time.sleep(gpu_id * 3)
                    t0 = time.time()
                    result = run_one_edit(
                        adapter, method, ncal, seed, gpu_id,
                        ft_steps=args.ft_steps, ft_lr=args.ft_lr,
                        sigma_steps=args.sigma_steps, sigma_lr=args.sigma_lr,
                    )
                    elapsed = time.time() - t0
                    return result, elapsed
                finally:
                    gpu_pool.release(gpu_id)

            with ThreadPoolExecutor(max_workers=args.n_gpus) as executor:
                futures = {}
                for i, job in enumerate(jobs):
                    f = executor.submit(process_job, job)
                    futures[f] = (i, job)

                for f in as_completed(futures):
                    i, (adapter, method, ncal, seed) = futures[f]
                    tag = _job_tag(adapter, method, ncal, seed)
                    try:
                        result, elapsed = f.result()
                        if result.get("skipped"):
                            skip_count += 1
                        elif result.get("error"):
                            err_count += 1
                            print(f"  FAIL [{ok_count+err_count+skip_count}/{total}] "
                                  f"{tag}: {result['error'][:80]}")
                            with _lock:
                                with open(RESULTS_JSONL, "a") as fo:
                                    fo.write(json.dumps({
                                        "tag": tag, "error": result["error"][-1500:],
                                        **{k: adapter[k] for k in ["base_model_dir", "task"]},
                                        "method": method, "ncal": ncal, "seed": seed,
                                    }) + "\n")
                        else:
                            ok_count += 1
                            print(f"  OK   [{ok_count+err_count+skip_count}/{total}] "
                                  f"{tag} ({elapsed:.0f}s)")
                            with _lock:
                                with open(RESULTS_JSONL, "a") as fo:
                                    fo.write(json.dumps({
                                        "tag": tag, "error": None,
                                        "out_dir": result["out_dir"],
                                        **{k: adapter[k] for k in
                                           ["base_model_id", "base_model_dir", "task",
                                            "adapter_dir"]},
                                        "method": method, "ncal": ncal, "seed": seed,
                                    }) + "\n")
                    except Exception as e:
                        err_count += 1
                        print(f"  EXCEPTION {tag}: {e}")

            print(f"\nEditing done: {ok_count} OK, {err_count} errors, "
                  f"{skip_count} skipped (pre-existing)")
        else:
            print("All editing jobs already complete!")

    # Write for_eval.jsonl
    eval_jsonl = write_for_eval_jsonl(ADAPTERS, methods, args.ncals, seeds)

    # Summary
    n_done = sum(1 for a, m, n, s in jobs if _is_done(_out_dir(a, m, n, s), m))
    print(f"\nEditing complete: {n_done}/{total}")

    print(f"\nTo evaluate:")
    seeds_str = " ".join(str(s) for s in seeds)
    print(f"  python scripts/rebuttal/run_eval_from_results.py \\")
    print(f"    --results_jsonl {eval_jsonl} \\")
    print(f"    --eval_root {OUT_ROOT / 'eval'} \\")
    print(f"    --seeds {seeds_str} --n_gpus {args.n_gpus}")


if __name__ == "__main__":
    main()
