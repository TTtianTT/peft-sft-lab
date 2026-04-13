#!/usr/bin/env python3
"""
P0-B: Sigma-Only Optimization Baseline.

Decomposes each LoRA (B, A) pair via SVD, freezes U and V, and optimizes
only the singular values (sigma) on calibration data. After optimization,
reconstructs new B, A from the optimized sigma and saves the adapter.

This answers the reviewer question: "Why use heuristic editing instead
of directly optimizing the singular values?"

Usage:
    python scripts/rebuttal/run_sigma_only_train.py \
        --base_model meta-llama/Llama-3.1-8B \
        --adapter_dir /path/to/lora/adapter \
        --out_dir /path/to/output \
        --task math \
        --calib_samples 128 \
        --steps 50 \
        --lr 5e-3 \
        --seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
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
from finetune.spectral_edit.edit_strategies import preserve_spectral_energy
from finetune.spectral_edit.io import (
    get_scaling_for_module,
    layer_idx_from_module_prefix,
    load_adapter_config,
    load_lora_state_dict,
    parse_lora_ab_key,
    save_lora_state_dict,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma

# ============================================================================
# Constants
# ============================================================================

TASK_TO_CALIB_DATASET = {
    "math": ("gsm8k", "main"),
    "metamath": ("gsm8k", "main"),
    "code": ("ise-uiuc/Magicoder-Evol-Instruct-110K", None),
    "magicoder": ("ise-uiuc/Magicoder-Evol-Instruct-110K", None),
    "alpaca": ("tatsu-lab/alpaca", None),
    "csqa": ("tau/commonsense_qa", None),
    "commonsense_qa": ("tau/commonsense_qa", None),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SigmaLoRAHook:
    """
    Replaces LoRA forward with sigma-parameterized forward.

    Instead of the standard LoRA: out = base(x) + scaling * B @ A @ x
    We do: out = base(x) + scaling * U @ diag(sigma) @ Vh @ x

    Only sigma is optimized. U and Vh are frozen.
    """

    def __init__(
        self,
        module: nn.Module,
        U: torch.Tensor,
        Vh: torch.Tensor,
        sigma_param: nn.Parameter,
        scaling: float,
    ):
        self.module = module
        self.U = U.detach()  # [d_out, r]
        self.Vh = Vh.detach()  # [r, d_in]
        self.sigma_param = sigma_param  # [r], trainable
        self.scaling = scaling
        self.handle = None

    def install(self):
        """Disable existing LoRA and install sigma-based forward hook."""
        # Disable the existing LoRA by setting its scaling to 0
        # We'll use a forward hook to add our sigma-parameterized delta
        if hasattr(self.module, 'lora_A') and hasattr(self.module, 'lora_B'):
            # Freeze and zero out existing LoRA
            for name in ['lora_A', 'lora_B']:
                lora_mod = getattr(self.module, name, None)
                if lora_mod is not None:
                    for p in lora_mod.parameters():
                        p.requires_grad_(False)

            # Set scaling to 0 to disable original LoRA forward
            if hasattr(self.module, 'scaling'):
                if isinstance(self.module.scaling, dict):
                    for key in self.module.scaling:
                        self.module.scaling[key] = 0.0
                else:
                    self.module.scaling = 0.0

        def _hook(module, inputs, output):
            x = inputs[0]
            # x: [batch, seq, d_in] or [batch, d_in]
            # Compute: scaling * x @ Vh^T @ diag(sigma) @ U^T
            # Which is: scaling * (U @ diag(sigma) @ Vh @ x^T)^T
            # More efficient: out += scaling * (x @ Vh.T) @ diag(sigma) @ U.T
            #   = scaling * ((x @ Vh.T) * sigma) @ U.T
            xVh = x.to(self.U.dtype) @ self.Vh.t()  # [..., r]
            xVh_sigma = xVh * self.sigma_param.unsqueeze(0) if x.dim() == 2 else \
                xVh * self.sigma_param
            delta = xVh_sigma @ self.U.t()  # [..., d_out]
            return output + self.scaling * delta.to(output.dtype)

        self.handle = self.module.register_forward_hook(_hook)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def sigma_only_train(args) -> Dict:
    """Run sigma-only optimization on calibration data."""
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Requires CUDA GPU.")

    # ---- Load adapter config and weights ----
    adapter_dir = args.adapter_dir
    adapter_cfg = load_adapter_config(adapter_dir)
    sd, fmt = load_lora_state_dict(adapter_dir)

    # ---- Parse LoRA pairs ----
    target_modules_set = set(args.target_modules)
    pairs: Dict[str, dict] = {}

    for k, t in sd.items():
        parsed = parse_lora_ab_key(k)
        if not parsed:
            continue
        prefix, which, adapter = parsed
        suffix = prefix.split(".")[-1]
        if suffix not in target_modules_set:
            continue

        li = layer_idx_from_module_prefix(prefix)
        if li is not None and not (args.layer_min <= li <= args.layer_max):
            continue

        pairs.setdefault(prefix, {})
        pairs[prefix][which] = (k, t, adapter)

    selected_prefixes = [p for p in pairs if "A" in pairs[p] and "B" in pairs[p]]
    if not selected_prefixes:
        raise RuntimeError("No matching LoRA (A,B) pairs found.")

    print(f"[Info] Selected LoRA modules for sigma optimization: {len(selected_prefixes)}")

    # ---- Load model ----
    print(f"[Info] Loading base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
        cache_dir=args.cache_dir,
    ).to(device)

    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True).to(device)
    model.eval()
    model.config.use_cache = False
    if getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[Info] Gradient checkpointing enabled")

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- SVD decomposition and sigma parameter creation ----
    name_to_module = dict(model.named_modules())
    sigma_params = []
    hooks: List[SigmaLoRAHook] = []
    sigma_info = {}  # prefix -> {U, Vh, sigma0, sigma_param}

    for prefix in selected_prefixes:
        keyA, A_cpu, adapterA = pairs[prefix]["A"]
        keyB, B_cpu, adapterB = pairs[prefix]["B"]

        A = A_cpu.to(device)
        B = B_cpu.to(device)

        U, S, Vh, V = lowrank_svd_from_ba(B, A)
        scaling = get_scaling_for_module(adapter_cfg, prefix)

        # Create trainable sigma parameter (float32 for stable optimization)
        sigma_param = nn.Parameter(S.clone().float().to(device))
        sigma_params.append(sigma_param)

        # Find the module
        if prefix not in name_to_module:
            candidates = [nm for nm in name_to_module if nm.endswith(prefix)]
            if not candidates:
                raise RuntimeError(f"Cannot find module '{prefix}' in model")
            module_name = candidates[0]
        else:
            module_name = prefix
        mod = name_to_module[module_name]

        # Install sigma hook
        hook = SigmaLoRAHook(mod, U.to(device), Vh.to(device), sigma_param, scaling)
        hook.install()
        hooks.append(hook)

        sigma_info[prefix] = {
            "U": U.detach().cpu(),
            "Vh": Vh.detach().cpu(),
            "sigma0": S.detach().cpu(),
            "sigma_param": sigma_param,
            "keyA": keyA,
            "keyB": keyB,
            "A_dtype": A_cpu.dtype,
            "B_dtype": B_cpu.dtype,
        }

    total_sigma_params = sum(p.numel() for p in sigma_params)
    print(f"[Info] Total sigma parameters: {total_sigma_params}")

    # ---- Load calibration data ----
    calib_dataset = args.calib_dataset
    calib_config = args.calib_config
    if calib_dataset is None:
        task_key = args.task.lower()
        if task_key not in TASK_TO_CALIB_DATASET:
            raise ValueError(f"Unknown task '{args.task}'. Provide --calib_dataset explicitly.")
        calib_dataset, calib_config = TASK_TO_CALIB_DATASET[task_key]

    formatter, _ = build_calib_formatter(calib_dataset, args.calib_text_fields)
    ds_split = load_calibration_split(
        calib_dataset, calib_config, args.calib_split, cache_dir=args.cache_dir
    )

    calib_seed = args.calib_seed if args.calib_seed is not None else args.seed
    calib_examples = sample_calibration_examples(
        ds_split, args.calib_samples, args.calib_shuffle, calib_seed, args.calib_start
    )
    ncal = len(calib_examples)
    print(f"[Info] Calibration examples: {ncal}")

    # Build batches
    bs = max(1, args.calib_batch_size)
    batches = []
    for i in range(0, ncal, bs):
        batch_ex = calib_examples[i : i + bs]
        input_ids, attn_mask, labels = make_calib_batch(tok, batch_ex, formatter, add_eos=True)
        batches.append((input_ids.to(device), attn_mask.to(device), labels.to(device)))

    if not batches:
        raise RuntimeError("No calibration batches built.")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(sigma_params, lr=args.lr, weight_decay=args.weight_decay)

    # ---- Training loop ----
    total_steps = args.steps
    loss_history = []
    t0 = time.time()

    print(f"[Train] Starting sigma-only optimization: {total_steps} steps, lr={args.lr}")

    for step in range(1, total_steps + 1):
        batch_idx = (step - 1) % len(batches)
        input_ids, attn_mask, labels = batches[batch_idx]

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optionally project sigma to preserve L1 energy after each step
        if args.preserve_energy != "none":
            with torch.no_grad():
                for prefix, info in sigma_info.items():
                    sigma0 = info["sigma0"].to(device)
                    sp = info["sigma_param"]
                    sp.data.clamp_min_(0.0)
                    sp.data.copy_(
                        preserve_spectral_energy(sigma0, sp.data, args.preserve_energy)
                    )

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if step % max(1, total_steps // 10) == 0 or step == total_steps or step == 1:
            elapsed = time.time() - t0
            print(f"  [Step {step}/{total_steps}] loss={loss_val:.4f}  elapsed={elapsed:.1f}s")

    elapsed_total = time.time() - t0
    print(f"[Train] Done. Final loss={loss_history[-1]:.4f}, total time={elapsed_total:.1f}s")

    # ---- Remove hooks ----
    for hook in hooks:
        hook.remove()

    # ---- Reconstruct B, A from optimized sigma ----
    import shutil
    if os.path.abspath(args.out_dir) != os.path.abspath(adapter_dir):
        if os.path.exists(args.out_dir):
            shutil.rmtree(args.out_dir)
        shutil.copytree(adapter_dir, args.out_dir)

    sigma_stats = {}
    for prefix, info in sigma_info.items():
        sigma_optimized = info["sigma_param"].detach().cpu().clamp_min(0.0)
        sigma0 = info["sigma0"]

        # Final energy preservation
        if args.preserve_energy != "none":
            sigma_optimized = preserve_spectral_energy(sigma0, sigma_optimized, args.preserve_energy)

        U = info["U"].float()
        Vh = info["Vh"].float()
        B_new, A_new = rebuild_ba_from_uv_sigma(U, Vh, sigma_optimized)

        sd[info["keyA"]] = A_new.to(dtype=info["A_dtype"]).detach().cpu()
        sd[info["keyB"]] = B_new.to(dtype=info["B_dtype"]).detach().cpu()

        # Verify reconstruction
        delta_orig = (info["sigma0"].float()).sum().item()
        delta_new = (sigma_optimized.float()).sum().item()
        sigma_stats[prefix] = {
            "sigma0_l1": delta_orig,
            "sigma_new_l1": delta_new,
            "sigma0_top1": float(sigma0.max().item()),
            "sigma_new_top1": float(sigma_optimized.max().item()),
            "sigma_change_l2": float(torch.norm(sigma_optimized - sigma0).item()),
        }

    save_lora_state_dict(args.out_dir, sd, fmt)
    print(f"[Save] Edited adapter saved to: {args.out_dir}")

    # ---- Save metadata ----
    meta = {
        "method": "sigma_only_train",
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "task": args.task,
        "target_modules": args.target_modules,
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "calib_dataset": calib_dataset,
        "calib_config": calib_config,
        "calib_split": args.calib_split,
        "calib_samples": args.calib_samples,
        "calib_samples_used": ncal,
        "calib_batch_size": bs,
        "calib_seed": calib_seed,
        "calib_start": args.calib_start,
        "steps": total_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "preserve_energy": args.preserve_energy,
        "seed": args.seed,
        "total_sigma_params": total_sigma_params,
        "loss_initial": loss_history[0] if loss_history else None,
        "loss_final": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "elapsed_seconds": elapsed_total,
        "sigma_stats": sigma_stats,
    }

    with open(os.path.join(args.out_dir, "sigma_only_train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Cleanup ----
    try:
        model.to("cpu")
        base.to("cpu")
    except Exception:
        pass
    del model, base, optimizer, sigma_params
    gc.collect()
    torch.cuda.empty_cache()

    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sigma-Only Train Baseline (P0-B)")

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--target_modules", type=str, nargs="+", default=["down_proj", "o_proj"])
    parser.add_argument("--layer_min", type=int, default=0)
    parser.add_argument("--layer_max", type=int, default=10**9)

    # Training hyperparameters
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--preserve_energy", type=str, choices=["none", "l1", "l2"], default="none",
                        help="Project sigma to preserve energy after each step")

    # Calibration data
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--calib_batch_size", type=int, default=2)
    parser.add_argument("--calib_dataset", type=str, default=None)
    parser.add_argument("--calib_config", type=str, default=None)
    parser.add_argument("--calib_split", type=str, default="train")
    parser.add_argument("--calib_text_fields", type=str, nargs="*", default=None)
    parser.add_argument("--calib_shuffle", action="store_true")
    parser.add_argument("--calib_seed", type=int, default=None)
    parser.add_argument("--calib_start", type=int, default=0)

    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce memory")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    sigma_only_train(args)


if __name__ == "__main__":
    main()
