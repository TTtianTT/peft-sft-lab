#!/usr/bin/env python3
"""
P0-1: Random Mechanism Decomposition

Goal: Explain WHY random_index sometimes outperforms baseline.
Distinguish "spectral regularization effect" from "lucky random hit".

Design:
- For 4 core settings (Qwen/math, Qwen/alpaca, Llama/math, Llama/csqa),
  run 50 random_index seeds each.
- For each run, record spectral statistics before/after edit.
- Add 5 matched-budget controls:
  1) flatten_spectrum: sigma_new = mean(sigma0) * ones (L1-preserved)
  2) shrink_top_only: reduce top-k sigma by sup_factor (L1-preserved)
  3) suppress_tail_only: reduce bottom-k sigma by sup_factor (L1-preserved)
  4) permute_sigma: randomly permute sigma values (no net change)
  5) uniform_rescale: multiply all sigma by same random factor in [0.8, 1.25]
- Compute correlations: random gain vs spectral statistics
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ============================================================================
# Constants
# ============================================================================

BASE_MODEL_DIR_TO_ID = {
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
}

TASK_TO_CALIB = {
    "math": ("gsm8k", "main"),
    "alpaca": ("tatsu-lab/alpaca", None),
    "csqa": ("tau/commonsense_qa", None),
}

TASK_DIR_TO_LM_EVAL = {
    "math": "gsm8k",
    "alpaca": "ifeval",
    "csqa": "commonsense_qa",
}

TASK_CONFIGS = {
    "math": {"num_fewshot": 5, "gen_kwargs": "temperature=0,top_p=1", "gpu_mem": 0.95},
    "alpaca": {"num_fewshot": None, "gen_kwargs": "max_gen_toks=2048,temperature=0,top_p=1", "gpu_mem": 0.95},
    "csqa": {"num_fewshot": 0, "gen_kwargs": None, "gpu_mem": 0.85},
}

PRIORITY_SETTINGS = [
    ("Qwen-Qwen3-8B", "math"),
    ("Qwen-Qwen3-8B", "alpaca"),
    ("meta-llama-Llama-3.1-8B", "math"),
    ("meta-llama-Llama-3.1-8B", "csqa"),
]

RUNS_ROOT = REPO_ROOT / "runs_refactor_data_20260121"


def find_adapter(model_dir: str, task: str) -> Path:
    """Find the LoRA adapter path for a model/task combo."""
    profile_map = {
        "math": "paper_math_ift_3ep",
        "alpaca": "paper_alpaca_3ep",
        "csqa": "paper_csqa_3ep",
    }
    profile = profile_map[task]
    p = RUNS_ROOT / model_dir / task / "lora" / f"profile-{profile}" / "rank-16" / "seed42"
    if not p.exists():
        raise FileNotFoundError(f"Adapter not found: {p}")
    return p


# ============================================================================
# Phase 1: Editing with spectral statistics collection
# ============================================================================

def run_edit_and_collect_stats(
    base_model_id: str,
    adapter_dir: Path,
    out_dir: Path,
    mode: str,
    seed: int,
    calib_dataset: str,
    calib_config: Optional[str],
    calib_samples: int = 128,
    gpu_id: int = 0,
    custom_edit_fn: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run spectral edit on a single GPU and return metadata + spectral stats.
    For custom controls, we use a special wrapper.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if custom_edit_fn:
        # Custom control methods
        cmd = [
            sys.executable, str(SCRIPT_DIR / "p0_1_control_edit.py"),
            "--base_model", base_model_id,
            "--lora_path", str(adapter_dir),
            "--out_dir", str(out_dir),
            "--control_method", custom_edit_fn,
            "--calib_samples", str(calib_samples),
            "--calib_dataset", calib_dataset,
            "--seed", str(seed),
        ]
        if calib_config:
            cmd += ["--calib_config", calib_config]
    else:
        cmd = [
            sys.executable, "-m", "finetune.spectral_edit.cli", "edit",
            "--base_model", base_model_id,
            "--lora_path", str(adapter_dir),
            "--out_dir", str(out_dir),
            "--mode", mode,
            "--target_modules", "down_proj", "o_proj",
            "--calib_samples", str(calib_samples),
            "--calib_batch_size", "2",
            "--calib_dataset", calib_dataset,
            "--seed", str(seed),
            "--calib_seed", str(seed),
            "--calib_shuffle",
            "--grad_norm", "mean_abs",
            "--preserve_energy", "l1",
        ]
        if calib_config:
            cmd += ["--calib_config", calib_config]

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(SRC_DIR),
        timeout=600,
    )

    if result.returncode != 0:
        return {"error": result.stderr[-2000:] if result.stderr else "unknown error"}

    # Read metadata
    meta_path = out_dir / "spectral_edit_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"error": "no metadata file produced"}


def run_lm_eval(
    base_model_id: str,
    adapter_dir: Optional[Path],
    task: str,
    output_dir: Path,
    gpu_id: int = 0,
    tp: int = 1,
) -> Optional[float]:
    """Run lm-eval-harness and return the primary metric value."""
    env = os.environ.copy()

    lm_task = TASK_DIR_TO_LM_EVAL[task]
    tcfg = TASK_CONFIGS[task]

    if tp > 1:
        gpu_list = ",".join(str(gpu_id + i) for i in range(tp))
        env["CUDA_VISIBLE_DEVICES"] = gpu_list
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Build merged model path
    merged_dir = output_dir / "merged_model"

    # Step 1: Merge adapter into base model
    merge_script = f"""
import torch, shutil, json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "{base_model_id}"
adapter_dir = "{str(adapter_dir) if adapter_dir else ''}"
merged_dir = "{str(merged_dir)}"

tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
if adapter_dir:
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
model.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
del model
torch.cuda.empty_cache()
"""
    merge_result = subprocess.run(
        [sys.executable, "-c", merge_script],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if merge_result.returncode != 0:
        print(f"[WARN] Merge failed: {merge_result.stderr[-500:]}")
        return None

    # Step 2: Run lm-eval
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", f"pretrained={merged_dir},tensor_parallel_size={tp},gpu_memory_utilization={tcfg['gpu_mem']},dtype=float16",
        "--tasks", lm_task,
        "--batch_size", "auto",
        "--output_path", str(output_dir / "lm_eval_results"),
    ]
    if tcfg["num_fewshot"] is not None:
        cmd += ["--num_fewshot", str(tcfg["num_fewshot"])]
    if tcfg["gen_kwargs"]:
        cmd += ["--gen_kwargs", tcfg["gen_kwargs"]]

    eval_result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=1800,
    )

    # Clean up merged model to save disk
    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)

    if eval_result.returncode != 0:
        print(f"[WARN] lm-eval failed: {eval_result.stderr[-500:]}")
        return None

    # Parse result
    return parse_lm_eval_result(output_dir / "lm_eval_results", lm_task)


def parse_lm_eval_result(results_dir: Path, lm_task: str) -> Optional[float]:
    """Parse lm-eval-harness results JSON."""
    # lm-eval writes results under results_dir/<model_name>/results_*.json
    metric_keys = {
        "gsm8k": "exact_match,strict-match",
        "ifeval": "prompt_level_strict_acc,none",
        "commonsense_qa": "acc,none",
    }
    primary_key = metric_keys.get(lm_task, "acc,none")

    # Find the results JSON
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                try:
                    with open(os.path.join(root, f)) as fp:
                        data = json.load(fp)
                    results = data.get("results", {})
                    task_results = results.get(lm_task, {})
                    if primary_key in task_results:
                        return float(task_results[primary_key])
                    # Fallback: try any acc-like key
                    for k, v in task_results.items():
                        if "acc" in k or "exact_match" in k:
                            return float(v)
                except Exception:
                    continue
    return None


# ============================================================================
# Spectral statistics computation
# ============================================================================

def compute_spectral_stats(sigma: np.ndarray) -> Dict[str, float]:
    """Compute spectral statistics for a singular value vector."""
    sigma = np.abs(sigma)
    sigma = sigma[sigma > 1e-12]  # filter near-zero
    if len(sigma) == 0:
        return {}

    total = sigma.sum()
    r = len(sigma)

    # Effective rank (exponential of spectral entropy)
    p = sigma / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    eff_rank = np.exp(entropy)

    # Gini coefficient
    sorted_s = np.sort(sigma)
    n = len(sorted_s)
    cum = np.cumsum(sorted_s)
    gini = 1 - 2 * np.sum(cum) / (n * total) + 1/n

    # Top-k energy ratios
    sorted_desc = np.sort(sigma)[::-1]
    top1_ratio = sorted_desc[0] / total if total > 0 else 0
    top4_ratio = sorted_desc[:min(4, r)].sum() / total if total > 0 else 0

    # Tail mass (bottom 50%)
    tail_mass = sorted_desc[r//2:].sum() / total if total > 0 else 0

    return {
        "effective_rank": float(eff_rank),
        "spectral_entropy": float(entropy),
        "gini": float(gini),
        "top1_energy_ratio": float(top1_ratio),
        "top4_energy_ratio": float(top4_ratio),
        "tail_mass_ratio": float(tail_mass),
        "sigma_mean": float(sigma.mean()),
        "sigma_std": float(sigma.std()),
        "sigma_max": float(sigma.max()),
        "sigma_min": float(sigma.min()),
        "rank": int(r),
    }


def extract_sigma_from_meta(meta: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Extract before/after spectral stats from edit metadata."""
    sigma_stats = meta.get("sigma_stats", {})
    if not sigma_stats:
        return None, None

    all_sigma0 = []
    all_sigma_new = []

    for prefix, stats in sigma_stats.items():
        s0_sum = stats.get("sigma0_sum", 0)
        s_new_sum = stats.get("sigma_new_sum", 0)
        all_sigma0.append(s0_sum)
        all_sigma_new.append(s_new_sum)

    return (
        {"modules": len(sigma_stats), "total_sigma0_sum": sum(all_sigma0)},
        {"modules": len(sigma_stats), "total_sigma_new_sum": sum(all_sigma_new)},
    )


# ============================================================================
# Main experiment runner
# ============================================================================

def run_p0_1_experiment(args):
    """Run the full P0-1 experiment."""
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results_path = out_root / "results.jsonl"

    # Load existing results for resume
    existing_keys = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    key = f"{rec['model_dir']}|{rec['task']}|{rec['method']}|{rec['seed']}"
                    existing_keys.add(key)
        print(f"[Resume] Found {len(existing_keys)} existing results")

    # Build job list
    jobs = []

    for model_dir, task in PRIORITY_SETTINGS:
        base_model_id = BASE_MODEL_DIR_TO_ID[model_dir]
        adapter_dir = find_adapter(model_dir, task)
        calib_dataset, calib_config = TASK_TO_CALIB[task]

        # 50 random_index seeds
        for seed in range(args.n_random_seeds):
            key = f"{model_dir}|{task}|random_index|{seed}"
            if key not in existing_keys:
                jobs.append({
                    "model_dir": model_dir,
                    "base_model_id": base_model_id,
                    "task": task,
                    "adapter_dir": str(adapter_dir),
                    "method": "random_index",
                    "seed": seed,
                    "calib_dataset": calib_dataset,
                    "calib_config": calib_config,
                })

        # 5 matched-budget controls (1 seed each is enough since they're deterministic or low-variance)
        for control in ["flatten_spectrum", "shrink_top_only", "suppress_tail_only",
                        "permute_sigma", "uniform_rescale"]:
            for seed in range(min(5, args.n_random_seeds)):
                key = f"{model_dir}|{task}|{control}|{seed}"
                if key not in existing_keys:
                    jobs.append({
                        "model_dir": model_dir,
                        "base_model_id": base_model_id,
                        "task": task,
                        "adapter_dir": str(adapter_dir),
                        "method": control,
                        "seed": seed,
                        "calib_dataset": calib_dataset,
                        "calib_config": calib_config,
                    })

    print(f"[P0-1] Total jobs: {len(jobs)} (across {len(PRIORITY_SETTINGS)} settings)")

    if args.edit_only:
        run_edit_phase(jobs, out_root, args)
    elif args.eval_only:
        run_eval_phase(out_root, args)
    else:
        run_edit_phase(jobs, out_root, args)
        run_eval_phase(out_root, args)


def run_edit_phase(jobs, out_root, args):
    """Phase 1: Run all edits in parallel across GPUs."""
    n_gpus = args.n_gpus

    print(f"\n[Phase 1] Running {len(jobs)} editing jobs on {n_gpus} GPUs...")

    def edit_job(job, gpu_id):
        method = job["method"]
        seed = job["seed"]
        model_dir = job["model_dir"]
        task = job["task"]

        edit_dir = out_root / "edited" / model_dir / task / method / f"seed{seed}"
        edit_dir.mkdir(parents=True, exist_ok=True)

        is_control = method in ["flatten_spectrum", "shrink_top_only",
                                "suppress_tail_only", "permute_sigma", "uniform_rescale"]

        meta = run_edit_and_collect_stats(
            base_model_id=job["base_model_id"],
            adapter_dir=Path(job["adapter_dir"]),
            out_dir=edit_dir,
            mode="random_index" if not is_control else method,
            seed=seed,
            calib_dataset=job["calib_dataset"],
            calib_config=job["calib_config"],
            calib_samples=args.calib_samples,
            gpu_id=gpu_id,
            custom_edit_fn=method if is_control else None,
        )

        # Write result record
        result = {
            "timestamp": datetime.now().isoformat(),
            "model_dir": model_dir,
            "task": task,
            "method": method,
            "seed": seed,
            "adapter_dir": job["adapter_dir"],
            "edited_dir": str(edit_dir),
            "error": meta.get("error"),
            "meta": meta if "error" not in meta else None,
        }

        return result

    results = []
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        gpu_cycle = list(range(n_gpus))
        futures = {}
        for i, job in enumerate(jobs):
            gpu_id = gpu_cycle[i % n_gpus]
            future = executor.submit(edit_job, job, gpu_id)
            futures[future] = job

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)

                # Append to JSONL
                with open(out_root / "results.jsonl", "a") as f:
                    f.write(json.dumps(result) + "\n")

                status = "OK" if not result.get("error") else f"ERR: {result['error'][:80]}"
                print(f"  [{len(results)}/{len(jobs)}] {result['model_dir']}/{result['task']}/{result['method']}/s{result['seed']} => {status}")
            except Exception as e:
                print(f"  [ERROR] {e}")

    print(f"\n[Phase 1 done] {len(results)} edits completed")


def run_eval_phase(out_root, args):
    """Phase 2: Evaluate all edited adapters."""
    # Collect all edited adapters that need eval
    eval_jobs = []
    results_path = out_root / "results.jsonl"
    eval_results_path = out_root / "eval_results.jsonl"

    # Load existing eval results
    existing_evals = set()
    if eval_results_path.exists():
        with open(eval_results_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    key = f"{rec['model_dir']}|{rec['task']}|{rec['method']}|{rec['seed']}"
                    existing_evals.add(key)

    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("error"):
                        continue
                    key = f"{rec['model_dir']}|{rec['task']}|{rec['method']}|{rec['seed']}"
                    if key not in existing_evals:
                        eval_jobs.append(rec)

    print(f"\n[Phase 2] {len(eval_jobs)} evaluations to run (TP={args.tp})")

    if not eval_jobs:
        print("  Nothing to evaluate.")
        return

    # For evaluation, we need TP GPUs per job
    tp = args.tp
    n_gpus = args.n_gpus
    n_parallel = max(1, n_gpus // tp)

    def eval_job(job_rec, base_gpu):
        model_dir = job_rec["model_dir"]
        task = job_rec["task"]
        method = job_rec["method"]
        seed = job_rec["seed"]
        edited_dir = Path(job_rec["edited_dir"])
        base_model_id = BASE_MODEL_DIR_TO_ID[model_dir]

        eval_output = out_root / "eval_output" / model_dir / task / method / f"seed{seed}"
        eval_output.mkdir(parents=True, exist_ok=True)

        metric = run_lm_eval(
            base_model_id=base_model_id,
            adapter_dir=edited_dir,
            task=task,
            output_dir=eval_output,
            gpu_id=base_gpu,
            tp=tp,
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "model_dir": model_dir,
            "task": task,
            "method": method,
            "seed": seed,
            "metric_value": metric,
            "error": None if metric is not None else "eval_failed",
        }

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {}
        for i, job_rec in enumerate(eval_jobs):
            base_gpu = (i % n_parallel) * tp
            future = executor.submit(eval_job, job_rec, base_gpu)
            futures[future] = job_rec

        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                completed += 1

                with open(eval_results_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

                val = result.get("metric_value", "N/A")
                print(f"  [{completed}/{len(eval_jobs)}] {result['model_dir']}/{result['task']}/{result['method']}/s{result['seed']} => {val}")
            except Exception as e:
                completed += 1
                print(f"  [{completed}/{len(eval_jobs)}] ERROR: {e}")

    print(f"\n[Phase 2 done] {completed} evaluations completed")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="P0-1: Random Mechanism Decomposition")
    parser.add_argument("--out_root", type=str,
                        default=str(REPO_ROOT / "outputs" / "rebuttal_v2" / "p0_1_random_mechanism"))
    parser.add_argument("--n_random_seeds", type=int, default=50)
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel for eval")
    parser.add_argument("--edit_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    run_p0_1_experiment(args)


if __name__ == "__main__":
    main()
