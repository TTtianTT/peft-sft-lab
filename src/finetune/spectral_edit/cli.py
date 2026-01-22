"""Command-line interface for LoRA spectral editing."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .calib import build_calib_formatter, load_calibration_split, make_calib_batch, sample_calibration_examples
from .edit_strategies import EditConfig, apply_spectral_edit
from .hooks import HOOK_CTX, ModuleSpec, register_sigma_hooks, remove_hooks
from .io import (
    ensure_local_lora_dir,
    get_scaling_for_module,
    layer_idx_from_module_prefix,
    load_adapter_config,
    load_lora_state_dict,
    parse_lora_ab_key,
    save_lora_state_dict,
)
from .svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_edit(args) -> None:
    """Main editing function."""
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This tool requires a CUDA GPU for gradient computation.")

    lora_dir = ensure_local_lora_dir(args.lora_path, cache_dir=args.cache_dir)
    adapter_cfg = load_adapter_config(lora_dir)
    sd, fmt = load_lora_state_dict(lora_dir)

    if os.path.abspath(args.out_dir) != os.path.abspath(lora_dir):
        if os.path.exists(args.out_dir):
            shutil.rmtree(args.out_dir)
        shutil.copytree(lora_dir, args.out_dir)

    pairs: Dict[str, dict] = {}
    target_modules_set = set(args.target_modules)

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

    selected_prefixes = [p for p in pairs.keys() if "A" in pairs[p] and "B" in pairs[p]]
    if not selected_prefixes:
        raise RuntimeError("No matching LoRA (A,B) pairs found for given target_modules/layer range.")

    print(f"[Info] Selected LoRA modules: {len(selected_prefixes)}")
    for p in selected_prefixes[:5]:
        print(f"   - {p}")
    if len(selected_prefixes) > 5:
        print(f"   ... and {len(selected_prefixes) - 5} more")

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

    model = PeftModel.from_pretrained(base, lora_dir, is_trainable=True).to(device)
    model.eval()
    model.config.use_cache = False

    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    name_to_module = dict(model.named_modules())
    specs: Dict[str, ModuleSpec] = {}

    for prefix in selected_prefixes:
        keyA, A_cpu, adapterA = pairs[prefix]["A"]
        keyB, B_cpu, adapterB = pairs[prefix]["B"]
        adapter_name = adapterA if adapterA is not None else adapterB

        if prefix not in name_to_module:
            candidates = [nm for nm in name_to_module.keys() if nm.endswith(prefix)]
            if not candidates:
                raise RuntimeError(f"Cannot find module '{prefix}' in model")
            module_name = candidates[0]
        else:
            module_name = prefix

        mod = name_to_module[module_name]
        A = A_cpu.to(device)
        B = B_cpu.to(device)

        U, S, Vh, V = lowrank_svd_from_ba(B, A)
        scaling = get_scaling_for_module(adapter_cfg, prefix)

        specs[prefix] = ModuleSpec(
            module_prefix=prefix,
            module=mod,
            U=U.detach(),
            V=V.detach(),
            Vh=Vh.detach(),
            sigma0=S.detach().cpu(),
            scaling=scaling,
            adapter=adapter_name,
        )

    print(f"[Info] Built SVD specs for {len(specs)} modules.")

    handles = register_sigma_hooks(specs)

    calib_config = args.calib_config
    if calib_config is None and args.calib_dataset == "gsm8k":
        calib_config = "main"

    formatter, normalized_fields = build_calib_formatter(args.calib_dataset, args.calib_text_fields)
    ds_split = load_calibration_split(
        args.calib_dataset,
        calib_config,
        args.calib_split,
        cache_dir=args.cache_dir,
    )
    calib_seed = args.calib_seed if args.calib_seed is not None else args.seed
    calib_examples = sample_calibration_examples(
        ds_split,
        args.calib_samples,
        args.calib_shuffle,
        calib_seed,
        args.calib_start,
    )
    ncal = len(calib_examples)

    bs = max(1, args.calib_batch_size)
    total_loss = 0.0
    n_steps = 0

    HOOK_CTX.reset()

    for i in range(0, ncal, bs):
        batch_ex = calib_examples[i : i + bs]
        input_ids, attn_mask, labels = make_calib_batch(tok, batch_ex, formatter, add_eos=True)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        HOOK_CTX.attn_mask = attn_mask

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss
        total_loss += float(loss.item())
        n_steps += 1

        model.zero_grad(set_to_none=True)
        loss.backward()
        model.zero_grad(set_to_none=True)

        step_num = i // bs + 1
        total_steps = math.ceil(ncal / bs) if ncal else 0
        if total_steps == 0:
            break
        if step_num % 5 == 0 or step_num == total_steps:
            print(f"[Calib] step {step_num}/{total_steps} loss={loss.item():.4f}")

    remove_hooks(handles)
    HOOK_CTX.attn_mask = None

    if n_steps > 0:
        print(f"[Calib] avg loss: {total_loss / max(1, n_steps):.4f}")
    print(f"[Calib] total active tokens: {HOOK_CTX.total_active_tokens}")

    if not HOOK_CTX.gsum:
        raise RuntimeError("No gradients accumulated. Hooks may not have fired.")

    edit_config = EditConfig(
        mode=args.mode,
        core_frac=args.core_frac,
        noise_frac=args.noise_frac,
        amp_factor=args.amp_factor,
        sup_factor=args.sup_factor,
        mid_factor=args.mid_factor,
        min_core_k=args.min_core_k,
        smooth_temperature=args.smooth_temperature,
        smooth_center_q=args.smooth_center_q,
        smooth_align_mid=not args.no_smooth_align_mid,
        z_high=args.z_high,
        z_low=args.z_low,
        z_tau=args.z_tau,
        z_fallback_std=args.z_fallback_std,
        robust_z_high=args.robust_z_high,
        robust_z_low=args.robust_z_low,
        robust_z_tau=args.robust_z_tau,
        robust_fallback_sigma=args.robust_fallback_sigma,
        eta=args.eta,
        update_mode=args.update_mode,
        asymmetric_update=args.asymmetric_update,
        eta_suppress=args.eta_suppress,
        eta_enhance=args.eta_enhance,
        pos_power=args.pos_power,
        grad_norm=args.grad_norm,
        preserve_energy=args.preserve_energy,
        sigma_clip_min=args.sigma_clip_min,
    )

    sigma_stats = {}

    for prefix, spec in specs.items():
        sigma0 = spec.sigma0.clone()
        g = HOOK_CTX.gsum.get(prefix, None)
        if g is None:
            continue

        sigma_new, stats = apply_spectral_edit(sigma0, g, edit_config)

        U = spec.U.to(device)
        Vh = spec.Vh.to(device)
        sigma_new_gpu = sigma_new.to(device)

        B_new, A_new = rebuild_ba_from_uv_sigma(U, Vh, sigma_new_gpu)

        keyA, A_old, _ = pairs[prefix]["A"]
        keyB, B_old, _ = pairs[prefix]["B"]
        A_new = A_new.to(dtype=A_old.dtype).detach().cpu()
        B_new = B_new.to(dtype=B_old.dtype).detach().cpu()

        sd[keyA] = A_new
        sd[keyB] = B_new
        sigma_stats[prefix] = stats

    save_lora_state_dict(args.out_dir, sd, fmt)

    meta = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "target_modules": args.target_modules,
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "calib_dataset": args.calib_dataset,
        "calib_config": calib_config,
        "calib_split": args.calib_split,
        "calib_text_fields": normalized_fields,
        "calib_shuffle": args.calib_shuffle,
        "calib_seed": calib_seed,
        "calib_start": args.calib_start,
        "calib_samples": args.calib_samples,
        "calib_samples_used": ncal,
        "calib_batch_size": args.calib_batch_size,
        "mode": args.mode,
        "core_frac": args.core_frac,
        "noise_frac": args.noise_frac,
        "amp_factor": args.amp_factor,
        "sup_factor": args.sup_factor,
        "mid_factor": args.mid_factor,
        "grad_norm": args.grad_norm,
        "preserve_energy": args.preserve_energy,
        "seed": args.seed,
    }

    with open(os.path.join(args.out_dir, "spectral_edit_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "sigma_stats": sigma_stats}, f, indent=2)

    print(f"[Save] Edited adapter saved to: {args.out_dir}")

    try:
        model.to("cpu")
        base.to("cpu")
    except Exception:
        pass
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA Spectral Edit - Sensitivity-based spectral editing for LoRA adapters"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    edit_parser = subparsers.add_parser("edit", help="Edit a LoRA adapter using spectral manipulation")
    edit_parser.add_argument("--base_model", type=str, required=True, help="HuggingFace model ID for base model")
    edit_parser.add_argument("--lora_path", type=str, required=True, help="Path or HF ID for LoRA adapter")
    edit_parser.add_argument("--out_dir", type=str, required=True, help="Output directory for edited adapter")

    edit_parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["down_proj", "o_proj"],
        help="Module names to edit (default: down_proj o_proj)",
    )
    edit_parser.add_argument("--layer_min", type=int, default=0, help="Minimum layer index to edit")
    edit_parser.add_argument("--layer_max", type=int, default=10**9, help="Maximum layer index to edit")

    edit_parser.add_argument("--calib_samples", type=int, default=32, help="Number of calibration samples")
    edit_parser.add_argument("--calib_batch_size", type=int, default=2, help="Calibration batch size")

    edit_parser.add_argument(
        "--calib_dataset",
        type=str,
        default="gsm8k",
        help="Calibration dataset (default: gsm8k)",
    )
    edit_parser.add_argument(
        "--calib_config",
        type=str,
        default=None,
        help="Dataset config name (default: main for gsm8k)",
    )
    edit_parser.add_argument(
        "--calib_split",
        type=str,
        default="train",
        help="Dataset split for calibration (default: train)",
    )
    edit_parser.add_argument(
        "--calib_text_fields",
        type=str,
        nargs="*",
        default=None,
        help="Text field(s) for prompt/answer (override GSM8K default)",
    )
    edit_parser.add_argument(
        "--calib_shuffle",
        action="store_true",
        help="Shuffle calibration dataset before sampling",
    )
    edit_parser.add_argument(
        "--calib_seed",
        type=int,
        default=None,
        help="Seed for calibration shuffle (defaults to --seed)",
    )
    edit_parser.add_argument(
        "--calib_start",
        type=int,
        default=0,
        help="Start offset into calibration dataset",
    )

    edit_parser.add_argument(
        "--mode",
        "--edit_mode",
        type=str,
        choices=["abs_select", "smooth_abs", "double_smooth", "z_score", "robust_z", "random_index", "gd"],
        default="abs_select",
        help="Edit mode: abs_select, smooth_abs, double_smooth, z_score, robust_z, random_index, or gd",
    )
    edit_parser.add_argument("--core_frac", type=float, default=0.2, help="Fraction of dims to amplify")
    edit_parser.add_argument("--noise_frac", type=float, default=0.2, help="Fraction of dims to suppress")
    edit_parser.add_argument("--amp_factor", type=float, default=1.25, help="Amplification factor")
    edit_parser.add_argument("--sup_factor", type=float, default=0.80, help="Suppression factor")
    edit_parser.add_argument("--mid_factor", type=float, default=1.0, help="Scale factor for middle dims")
    edit_parser.add_argument("--min_core_k", type=int, default=1, help="Minimum number of core dims per module")

    edit_parser.add_argument(
        "--smooth_temperature",
        type=float,
        default=0.35,
        help="Smoothness for smooth_abs/double_smooth (larger=smoother, smaller=sharper)",
    )
    edit_parser.add_argument(
        "--smooth_center_q",
        type=float,
        default=0.5,
        help="Center quantile for smooth_abs (0.5=median)",
    )
    edit_parser.add_argument(
        "--no_smooth_align_mid",
        action="store_true",
        help="Disable aligning gate(center)=mid_factor in smooth_abs",
    )

    edit_parser.add_argument("--z_high", type=float, default=1.0, help="Z-score threshold for amplification")
    edit_parser.add_argument("--z_low", type=float, default=-0.5, help="Z-score threshold for suppression")
    edit_parser.add_argument("--z_tau", type=float, default=0.2, help="Temperature for z-score gating")
    edit_parser.add_argument(
        "--z_fallback_std",
        type=float,
        default=1e-6,
        help="Stddev floor that triggers z_score fallback",
    )

    edit_parser.add_argument(
        "--robust_z_high",
        type=float,
        default=1.0,
        help="Robust z-score threshold for amplification",
    )
    edit_parser.add_argument(
        "--robust_z_low",
        type=float,
        default=-0.5,
        help="Robust z-score threshold for suppression",
    )
    edit_parser.add_argument(
        "--robust_z_tau",
        type=float,
        default=0.2,
        help="Temperature for robust z-score gating",
    )
    edit_parser.add_argument(
        "--robust_fallback_sigma",
        type=float,
        default=1e-6,
        help="Sigma floor that triggers robust_z fallback",
    )

    edit_parser.add_argument("--eta", type=float, default=0.2, help="Learning rate (gd mode)")
    edit_parser.add_argument(
        "--update_mode",
        type=str,
        choices=["additive", "multiplicative"],
        default="multiplicative",
        help="Update mode (gd mode)",
    )
    edit_parser.add_argument(
        "--asymmetric_update",
        action="store_true",
        help="Use asymmetric step sizes (gd mode)",
    )
    edit_parser.add_argument("--eta_suppress", type=float, default=2.0, help="Step size for g>0 (gd mode)")
    edit_parser.add_argument("--eta_enhance", type=float, default=0.2, help="Step size for g<0 (gd mode)")
    edit_parser.add_argument("--pos_power", type=float, default=1.0, help="Nonlinearity power (gd mode)")

    edit_parser.add_argument(
        "--grad_norm",
        type=str,
        choices=["none", "mean_abs", "l2"],
        default="mean_abs",
        help="Gradient normalization method",
    )
    edit_parser.add_argument(
        "--preserve_energy",
        type=str,
        choices=["none", "l1", "l2"],
        default="l1",
        help="Energy preservation method",
    )
    edit_parser.add_argument(
        "--sigma_clip_min",
        type=float,
        default=0.0,
        help="Minimum sigma value after editing",
    )

    edit_parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for HF downloads")
    edit_parser.add_argument("--seed", type=int, default=0, help="Random seed")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "edit":
        run_edit(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
