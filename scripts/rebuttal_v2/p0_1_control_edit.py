#!/usr/bin/env python3
"""
P0-1 Control Edit Methods.

Implements matched-budget spectral controls that do NOT use gradient information:
1) flatten_spectrum: sigma_new = mean(sigma0) (L1-preserved)
2) shrink_top_only: shrink top-k sigma by sup_factor (L1-preserved)
3) suppress_tail_only: shrink bottom-k sigma by sup_factor (L1-preserved)
4) permute_sigma: randomly permute sigma values (same total energy)
5) uniform_rescale: multiply all sigma by random factor in [0.8, 1.25], then L1-preserve

All controls use the same calibration pipeline (model load + SVD) but skip gradient computation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.spectral_edit.edit_strategies import preserve_spectral_energy
from finetune.spectral_edit.io import (
    ensure_local_lora_dir,
    get_scaling_for_module,
    layer_idx_from_module_prefix,
    load_adapter_config,
    load_lora_state_dict,
    parse_lora_ab_key,
    save_lora_state_dict,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_control_edit(
    sigma0: torch.Tensor,
    method: str,
    seed: int,
    core_frac: float = 0.2,
    noise_frac: float = 0.2,
    amp_factor: float = 1.25,
    sup_factor: float = 0.80,
) -> torch.Tensor:
    """Apply a control edit to singular values."""
    r = sigma0.numel()
    k_core = max(int(round(r * core_frac)), 1)
    k_noise = max(0, min(int(round(r * noise_frac)), r - k_core))

    if method == "flatten_spectrum":
        # Replace all sigma with their mean (perfectly flat spectrum)
        sigma_new = torch.full_like(sigma0, sigma0.mean().item())

    elif method == "shrink_top_only":
        # Shrink the top-k (largest) singular values by sup_factor
        order = torch.argsort(sigma0, descending=True)
        sigma_new = sigma0.clone()
        top_idx = order[:k_core]
        sigma_new[top_idx] = sigma_new[top_idx] * sup_factor

    elif method == "suppress_tail_only":
        # Shrink the bottom-k (smallest) singular values by sup_factor
        order = torch.argsort(sigma0, descending=True)
        sigma_new = sigma0.clone()
        tail_idx = order[-k_noise:] if k_noise > 0 else torch.tensor([], dtype=torch.long)
        if len(tail_idx) > 0:
            sigma_new[tail_idx] = sigma_new[tail_idx] * sup_factor

    elif method == "permute_sigma":
        # Randomly permute sigma values (same set of values, different assignment)
        torch.manual_seed(seed)
        perm = torch.randperm(r)
        sigma_new = sigma0[perm]

    elif method == "uniform_rescale":
        # Multiply all sigma by a random factor, then L1-preserve
        torch.manual_seed(seed)
        factor = 0.8 + 0.45 * torch.rand(1).item()  # uniform in [0.8, 1.25]
        sigma_new = sigma0 * factor

    else:
        raise ValueError(f"Unknown control method: {method}")

    # L1 energy preservation for all controls
    sigma_new = preserve_spectral_energy(sigma0, sigma_new, "l1")
    sigma_new = sigma_new.clamp_min(0.0)

    return sigma_new


def run_control_edit(args):
    """Run control edit without gradient computation."""
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lora_dir = ensure_local_lora_dir(args.lora_path, cache_dir=None)
    adapter_cfg = load_adapter_config(lora_dir)
    sd, fmt = load_lora_state_dict(lora_dir)

    if os.path.abspath(args.out_dir) != os.path.abspath(lora_dir):
        if os.path.exists(args.out_dir):
            shutil.rmtree(args.out_dir)
        shutil.copytree(lora_dir, args.out_dir)

    # Find LoRA pairs
    target_modules_set = {"down_proj", "o_proj"}
    pairs: Dict[str, dict] = {}

    for k, t in sd.items():
        parsed = parse_lora_ab_key(k)
        if not parsed:
            continue
        prefix, which, adapter = parsed
        suffix = prefix.split(".")[-1]
        if suffix not in target_modules_set:
            continue
        pairs.setdefault(prefix, {})
        pairs[prefix][which] = (k, t, adapter)

    selected = [p for p in pairs if "A" in pairs[p] and "B" in pairs[p]]
    print(f"[Control Edit] {args.control_method} on {len(selected)} modules, seed={args.seed}")

    sigma_stats = {}

    for prefix in selected:
        keyA, A_cpu, _ = pairs[prefix]["A"]
        keyB, B_cpu, _ = pairs[prefix]["B"]

        A = A_cpu.float()
        B = B_cpu.float()

        U, S, Vh, V = lowrank_svd_from_ba(B, A)
        sigma0 = S.detach()

        # Apply control edit
        sigma_new = apply_control_edit(
            sigma0, args.control_method, args.seed,
        )

        # Rebuild
        B_new, A_new = rebuild_ba_from_uv_sigma(U, Vh, sigma_new)
        A_new = A_new.to(dtype=A_cpu.dtype)
        B_new = B_new.to(dtype=B_cpu.dtype)

        sd[keyA] = A_new
        sd[keyB] = B_new

        sigma_stats[prefix] = {
            "mode": args.control_method,
            "r": int(sigma0.numel()),
            "sigma0_sum": float(sigma0.sum().item()),
            "sigma_new_sum": float(sigma_new.sum().item()),
            "sigma0_top1": float(sigma0.max().item()),
            "sigma_new_top1": float(sigma_new.max().item()),
            "sigma0_mean": float(sigma0.mean().item()),
            "sigma_new_mean": float(sigma_new.mean().item()),
            "sigma0_std": float(sigma0.std().item()),
            "sigma_new_std": float(sigma_new.std().item()),
        }

    save_lora_state_dict(args.out_dir, sd, fmt)

    meta = {
        "meta": {
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "control_method": args.control_method,
            "seed": args.seed,
            "calib_samples": args.calib_samples,
            "calib_dataset": args.calib_dataset,
        },
        "sigma_stats": sigma_stats,
    }

    with open(os.path.join(args.out_dir, "spectral_edit_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Control Edit] Saved to {args.out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--control_method", required=True,
                        choices=["flatten_spectrum", "shrink_top_only",
                                 "suppress_tail_only", "permute_sigma", "uniform_rescale"])
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--calib_dataset", type=str, default="gsm8k")
    parser.add_argument("--calib_config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_control_edit(args)


if __name__ == "__main__":
    main()
