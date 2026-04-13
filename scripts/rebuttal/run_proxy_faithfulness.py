#!/usr/bin/env python3
"""
P1-E: Proxy Faithfulness Experiment.

Tests whether calibration loss improvement predicts test metric improvement.

For each (model, task, method):
1. Split calibration pool: edit subset (for editing) + held-out proxy subset
2. Record: edit-set loss, held-out loss, final eval metric
3. Compute correlations: proxy vs test, edit-set vs test

Usage:
    python scripts/rebuttal/run_proxy_faithfulness.py \
        --runs_root /path/to/runs_refactor_data_20260121 \
        --out_root outputs/rebuttal_exp/raw/proxy_faithfulness \
        --calib_samples 128 \
        --proxy_samples 128
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
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

# ============================================================================
# Constants
# ============================================================================

BASE_MODEL_DIR_TO_ID = {
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
}

TASK_TO_CALIB_DATASET = {
    "math": ("gsm8k", "main"),
    "code": ("ise-uiuc/Magicoder-Evol-Instruct-110K", None),
    "alpaca": ("tatsu-lab/alpaca", None),
    "csqa": ("tau/commonsense_qa", None),
}

TASK_ALIASES = {"metamath": "math", "magicoder": "code", "general": "alpaca", "commonsense_qa": "csqa"}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss_on_examples(
    model,
    tokenizer,
    examples: List[dict],
    formatter,
    batch_size: int = 2,
    device: str = "cuda",
) -> float:
    """Compute average cross-entropy loss on a set of examples."""
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(examples), batch_size):
        batch_ex = examples[i : i + batch_size]
        input_ids, attn_mask, labels = make_calib_batch(tokenizer, batch_ex, formatter, add_eos=True)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            total_loss += float(out.loss.item())
            n_batches += 1

    return total_loss / max(1, n_batches)


def discover_adapters(runs_root: Path, tasks: Optional[List[str]] = None):
    """Discover final LoRA adapters."""
    adapters = []
    for root, dirs, files in os.walk(runs_root):
        root_path = Path(root)
        if "checkpoint-" in str(root_path):
            dirs.clear()
            continue
        has_weights = "adapter_model.safetensors" in files or "adapter_model.bin" in files
        has_config = "adapter_config.json" in files
        if not (has_weights and has_config):
            continue
        try:
            rel = root_path.relative_to(runs_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 6 or parts[2] != "lora":
            continue
        base_model_dir = parts[0]
        task = TASK_ALIASES.get(parts[1], parts[1])
        if tasks and task not in tasks:
            continue
        base_model_id = BASE_MODEL_DIR_TO_ID.get(base_model_dir)
        if not base_model_id:
            continue
        adapters.append({
            "adapter_dir": root_path,
            "task": task,
            "base_model_dir": base_model_dir,
            "base_model_id": base_model_id,
        })
    return adapters


def run_proxy_experiment(
    adapter_info: Dict,
    out_dir: Path,
    calib_samples: int,
    proxy_samples: int,
    batch_size: int,
    seed: int,
    cache_dir: Optional[str],
    methods: List[str],
) -> Dict:
    """
    For one adapter: compute loss on edit-set and proxy-set for each method.
    """
    device = "cuda"
    task = adapter_info["task"]
    base_model_id = adapter_info["base_model_id"]
    adapter_dir = adapter_info["adapter_dir"]

    calib_dataset, calib_config = TASK_TO_CALIB_DATASET[task]
    formatter, _ = build_calib_formatter(calib_dataset, None)
    ds_split = load_calibration_split(calib_dataset, calib_config, "train", cache_dir=cache_dir)

    # Split: first calib_samples for editing, next proxy_samples for held-out
    set_seed(seed)
    edit_examples = sample_calibration_examples(
        ds_split, calib_samples, True, seed, calib_start=0
    )
    proxy_examples = sample_calibration_examples(
        ds_split, proxy_samples, True, seed, calib_start=calib_samples
    )

    print(f"  Edit examples: {len(edit_examples)}, Proxy examples: {len(proxy_examples)}")

    results = {}

    for method in methods:
        print(f"  Computing losses for method={method}...")

        if method == "baseline":
            adapter_path = adapter_dir
        else:
            # Look for edited adapter from multiseed results
            # Convention: out_root/edited_adapters/{run_id}/{method}/repeat_00_seed{seed}/
            # For now, just compute on unedited as baseline
            adapter_path = adapter_dir

        # Load model
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

        edit_loss = compute_loss_on_examples(model, tok, edit_examples, formatter, batch_size, device)
        proxy_loss = compute_loss_on_examples(model, tok, proxy_examples, formatter, batch_size, device)

        results[method] = {
            "edit_loss": edit_loss,
            "proxy_loss": proxy_loss,
        }

        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Proxy Faithfulness (P1-E)")
    parser.add_argument("--runs_root", type=str,
                        default="/mnt/data/zailongtian/workspace/peft-sft-lab/runs_refactor_data_20260121")
    parser.add_argument("--out_root", type=str,
                        default="outputs/rebuttal_exp/raw/proxy_faithfulness")
    parser.add_argument("--multiseed_root", type=str,
                        default="outputs/rebuttal_exp/raw/multiseed",
                        help="Root of multiseed results (for edited adapters)")
    parser.add_argument("--tasks", type=str, nargs="+", default=["math", "code", "alpaca", "csqa"])
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--proxy_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    adapters = discover_adapters(runs_root, args.tasks)
    print(f"Found {len(adapters)} adapters")

    all_results = []

    for adapter_info in adapters:
        print(f"\n[Proxy] {adapter_info['base_model_dir']}/{adapter_info['task']}")

        adapter_out = out_root / adapter_info["base_model_dir"] / adapter_info["task"]
        adapter_out.mkdir(parents=True, exist_ok=True)

        # For proxy faithfulness, we compute losses for baseline (unedited)
        # The edited adapter losses should come from multiseed experiment
        results = run_proxy_experiment(
            adapter_info,
            adapter_out,
            calib_samples=args.calib_samples,
            proxy_samples=args.proxy_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            cache_dir=args.cache_dir,
            methods=["baseline"],
        )

        record = {
            "base_model_dir": adapter_info["base_model_dir"],
            "base_model_id": adapter_info["base_model_id"],
            "task": adapter_info["task"],
            "adapter_dir": str(adapter_info["adapter_dir"]),
            "calib_samples": args.calib_samples,
            "proxy_samples": args.proxy_samples,
            "seed": args.seed,
            **{f"{m}_{k}": v for m, losses in results.items() for k, v in losses.items()},
        }
        all_results.append(record)

        with open(adapter_out / "proxy_losses.json", "w") as f:
            json.dump(record, f, indent=2)

    # Save all
    with open(out_root / "all_proxy_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {out_root}")


if __name__ == "__main__":
    main()
