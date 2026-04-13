#!/usr/bin/env python3
"""
Compute edit-set and proxy (held-out) losses for ALL methods.

For each (model, task, method, seed), loads the edited adapter and computes:
  - edit_loss: avg loss on the calibration subset used for editing
  - proxy_loss: avg loss on a held-out proxy subset

Combines with test eval metrics from eval_results.jsonl to produce a unified table:
  delta_edit, delta_proxy, delta_test per method.

Runs on 8 GPUs in parallel (one model load per GPU at a time).

Usage:
    python scripts/rebuttal/run_proxy_all_methods.py --n_gpus 8 --seed 42
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.spectral_edit.calib import (
    build_calib_formatter,
    load_calibration_split,
    make_calib_batch,
    sample_calibration_examples,
)

TASK_TO_CALIB_DATASET = {
    "math": ("gsm8k", "main"),
    "code": ("ise-uiuc/Magicoder-Evol-Instruct-110K", None),
    "alpaca": ("tatsu-lab/alpaca", None),
    "csqa": ("tau/commonsense_qa", None),
}

OUT_ROOT = Path("outputs/rebuttal_exp/raw/proxy_all_methods")
RESULTS_JSONL = OUT_ROOT / "proxy_results.jsonl"


class GPUPool:
    def __init__(self, n_gpus: int):
        self._q: Queue = Queue()
        for i in range(n_gpus):
            self._q.put(i)

    def acquire(self) -> int:
        return self._q.get()

    def release(self, gpu_id: int):
        self._q.put(gpu_id)


_lock = threading.Lock()


def compute_loss(model, tokenizer, examples, formatter, batch_size, device, max_seq_len=None):
    """Compute avg cross-entropy loss on examples."""
    total_loss, n_batches = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch_ex = examples[i: i + batch_size]
            input_ids, attn_mask, labels = make_calib_batch(
                tokenizer, batch_ex, formatter, add_eos=True, max_seq_len=max_seq_len
            )
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attn_mask.to(device),
                labels=labels.to(device),
            )
            total_loss += float(out.loss.item())
            n_batches += 1
    return total_loss / max(1, n_batches)


def process_one(
    record: Dict,
    edit_examples: List,
    proxy_examples: List,
    formatter,
    gpu_id: int,
    batch_size: int,
    max_seq_len: Optional[int],
    cache_dir: Optional[str],
) -> Dict:
    """Load one adapter, compute edit/proxy losses, return result."""
    device = f"cuda:{gpu_id}"
    base_model_id = record["base_model_id"]
    method = record["method"]

    adapter_path = record["adapter_dir"] if method == "baseline" else record.get("edited_adapter_dir")
    if not adapter_path or not Path(adapter_path).exists():
        return {"error": f"Adapter not found: {adapter_path}"}

    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map=None, cache_dir=cache_dir,
    ).to(device)

    model = PeftModel.from_pretrained(base, str(adapter_path)).to(device)
    model.eval()
    model.config.use_cache = False

    edit_loss = compute_loss(model, tok, edit_examples, formatter, batch_size, device, max_seq_len)
    proxy_loss = compute_loss(model, tok, proxy_examples, formatter, batch_size, device, max_seq_len)

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    return {"edit_loss": edit_loss, "proxy_loss": proxy_loss, "error": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed_results", type=str,
                        default="outputs/rebuttal_exp/raw/multiseed/results.jsonl")
    parser.add_argument("--eval_results", type=str,
                        default="outputs/rebuttal_exp/raw/multiseed_eval/eval_results.jsonl")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--proxy_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["baseline", "random_index", "grad_direction", "continued_ft", "sigma_only"])
    parser.add_argument("--tasks", type=str, nargs="+", default=["math", "alpaca", "csqa"])
    parser.add_argument("--models", type=str, nargs="+",
                        default=["Qwen-Qwen3-8B", "meta-llama-Llama-3.1-8B"])
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load multiseed results
    with open(args.multiseed_results) as f:
        all_records = [json.loads(l) for l in f if l.strip()]

    # Filter to relevant jobs
    jobs = []
    for r in all_records:
        if r.get("error"):
            continue
        if r["repeat_seed"] != args.seed:
            continue
        if r["method"] not in args.methods:
            continue
        if r["task"] not in args.tasks:
            continue
        if r["base_model_dir"] not in args.models:
            continue
        jobs.append(r)

    print(f"Total proxy jobs: {len(jobs)}")

    # Load existing results for resume
    done_keys = set()
    if RESULTS_JSONL.exists():
        with open(RESULTS_JSONL) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_keys.add(f"{rec['base_model_dir']}|{rec['task']}|{rec['method']}|{rec['seed']}")

    # Group by (model, task) to share calibration data loading
    from itertools import groupby
    jobs.sort(key=lambda x: (x["base_model_dir"], x["task"]))

    gpu_pool = GPUPool(args.n_gpus)

    # Pre-load calibration data per (model, task) — shared across methods
    calib_cache = {}

    for (model_dir, task), group_iter in groupby(jobs, key=lambda x: (x["base_model_dir"], x["task"])):
        group = list(group_iter)
        print(f"\n[Proxy] {model_dir}/{task}: {len(group)} methods")

        if (model_dir, task) not in calib_cache:
            calib_dataset, calib_config = TASK_TO_CALIB_DATASET[task]
            formatter, _ = build_calib_formatter(calib_dataset, None)
            ds_split = load_calibration_split(calib_dataset, calib_config, "train", cache_dir=args.cache_dir)
            edit_examples = sample_calibration_examples(
                ds_split, args.calib_samples, True, args.seed, calib_start=0
            )
            proxy_examples = sample_calibration_examples(
                ds_split, args.proxy_samples, True, args.seed, calib_start=args.calib_samples
            )
            calib_cache[(model_dir, task)] = (formatter, edit_examples, proxy_examples)
            print(f"  Loaded calib: edit={len(edit_examples)}, proxy={len(proxy_examples)}")

        formatter, edit_examples, proxy_examples = calib_cache[(model_dir, task)]

        # Process methods in parallel
        futures = {}
        with ThreadPoolExecutor(max_workers=args.n_gpus) as executor:
            for r in group:
                key = f"{r['base_model_dir']}|{r['task']}|{r['method']}|{args.seed}"
                if key in done_keys:
                    print(f"  SKIP {r['method']} (already done)")
                    continue

                gpu_id = gpu_pool.acquire()
                f = executor.submit(
                    process_one, r, edit_examples, proxy_examples,
                    formatter, gpu_id, args.batch_size, args.max_seq_len, args.cache_dir,
                )
                futures[f] = (r, gpu_id)

            for f in as_completed(futures):
                r, gpu_id = futures[f]
                gpu_pool.release(gpu_id)
                try:
                    result = f.result()
                    if result.get("error"):
                        print(f"  {r['method']}: ERROR {result['error'][:80]}")
                    else:
                        print(f"  {r['method']}: edit_loss={result['edit_loss']:.4f}, proxy_loss={result['proxy_loss']:.4f}")

                    rec = {
                        "base_model_dir": r["base_model_dir"],
                        "base_model_id": r["base_model_id"],
                        "task": r["task"],
                        "method": r["method"],
                        "seed": args.seed,
                        "adapter_dir": r["adapter_dir"],
                        "edited_adapter_dir": r.get("edited_adapter_dir"),
                        "edit_loss": result.get("edit_loss"),
                        "proxy_loss": result.get("proxy_loss"),
                        "error": result.get("error"),
                    }
                    with _lock:
                        with open(RESULTS_JSONL, "a") as f_out:
                            f_out.write(json.dumps(rec) + "\n")
                        done_keys.add(f"{r['base_model_dir']}|{r['task']}|{r['method']}|{args.seed}")
                except Exception as e:
                    print(f"  {r['method']}: EXCEPTION {e}")
                    gpu_pool.release(gpu_id)  # double release is safe for Queue

    # Load eval results and combine
    print("\n" + "=" * 100)
    print("PROXY FAITHFULNESS SUMMARY")
    print("=" * 100)

    eval_metrics = {}
    if Path(args.eval_results).exists():
        with open(args.eval_results) as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                if e.get("metric_value") is not None and e["repeat_seed"] == args.seed:
                    eval_metrics[(e["base_model_dir"], e["task"], e["method"])] = e["metric_value"]

    # Load proxy results
    proxy_data = {}
    if RESULTS_JSONL.exists():
        with open(RESULTS_JSONL) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("edit_loss") is not None:
                    proxy_data[(r["base_model_dir"], r["task"], r["method"])] = r

    # Print combined table
    print(f"\n{'Model':<28s} {'Task':<8s} {'Method':<16s} {'EditL':>8s} {'ProxyL':>8s} {'TestM':>8s} {'ΔEdit':>8s} {'ΔProxy':>8s} {'ΔTest':>8s}")
    print("-" * 120)

    for model_dir in args.models:
        for task in args.tasks:
            bl_proxy = proxy_data.get((model_dir, task, "baseline"))
            bl_test = eval_metrics.get((model_dir, task, "baseline"))

            for method in args.methods:
                p = proxy_data.get((model_dir, task, method))
                t = eval_metrics.get((model_dir, task, method))

                if p is None:
                    continue

                el = p["edit_loss"]
                pl = p["proxy_loss"]
                bl_el = bl_proxy["edit_loss"] if bl_proxy else None
                bl_pl = bl_proxy["proxy_loss"] if bl_proxy else None

                d_edit = (bl_el - el) if bl_el is not None else None
                d_proxy = (bl_pl - pl) if bl_pl is not None else None
                d_test = (t - bl_test) if (t is not None and bl_test is not None) else None

                fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
                fmtd = lambda v: f"{v:+.4f}" if v is not None else "N/A"

                print(f"{model_dir:<28s} {task:<8s} {method:<16s} {fmt(el):>8s} {fmt(pl):>8s} {fmt(t):>8s} {fmtd(d_edit):>8s} {fmtd(d_proxy):>8s} {fmtd(d_test):>8s}")

    print(f"\nProxy results: {RESULTS_JSONL}")


if __name__ == "__main__":
    main()
