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

from .calib import (
    build_calib_formatter,
    load_calibration_split,
    make_calib_batch,
    make_chat_calib_batch,
    sample_calibration_examples,
)
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
from .module_selection import score_module_gradient_batches, select_important_modules
from .posthoc_hns import (
    FAST_HNS_COEFFICIENTS,
    STABLE_HNS_COEFFICIENTS,
    HNSEditConfig,
    apply_hns_to_svd,
)
from .svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _copy_lora_tree(lora_dir: str, out_dir: str) -> None:
    """Copy a LoRA adapter directory unless editing in place."""
    if os.path.abspath(out_dir) == os.path.abspath(lora_dir):
        return
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(lora_dir, out_dir)


def _collect_lora_pairs(
    sd: Dict[str, torch.Tensor],
    target_modules: List[str],
    layer_min: int,
    layer_max: int,
) -> tuple[Dict[str, dict], List[str], List[str]]:
    """Collect LoRA A/B pairs filtered by suffix and layer range."""
    discovered_suffixes: set[str] = set()
    parsed_items: list[tuple[str, str, Optional[str], str, torch.Tensor]] = []

    for key, tensor in sd.items():
        parsed = parse_lora_ab_key(key)
        if not parsed:
            continue
        prefix, which, adapter = parsed
        suffix = prefix.split(".")[-1]
        discovered_suffixes.add(suffix)
        parsed_items.append((prefix, which, adapter, key, tensor))

    target_aliases = {item.lower() for item in target_modules}
    if "all" in target_aliases or "all_modules" in target_aliases:
        resolved_targets = sorted(discovered_suffixes)
    else:
        resolved_targets = list(dict.fromkeys(target_modules))

    target_modules_set = set(resolved_targets)
    pairs: Dict[str, dict] = {}

    for prefix, which, adapter, key, tensor in parsed_items:
        suffix = prefix.split(".")[-1]
        if suffix not in target_modules_set:
            continue

        layer_idx = layer_idx_from_module_prefix(prefix)
        if layer_idx is not None and not (layer_min <= layer_idx <= layer_max):
            continue

        pairs.setdefault(prefix, {})
        pairs[prefix][which] = (key, tensor, adapter)

    selected_prefixes = sorted([prefix for prefix in pairs.keys() if "A" in pairs[prefix] and "B" in pairs[prefix]])
    if not selected_prefixes:
        raise RuntimeError(
            "No matching LoRA (A,B) pairs found for the given target_modules/layer range. "
            f"Available suffixes: {sorted(discovered_suffixes)}"
        )

    return pairs, selected_prefixes, resolved_targets


def run_edit(args) -> None:
    """Main editing function."""
    set_seed(args.seed)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This tool requires a CUDA GPU for gradient computation.")

    lora_dir = ensure_local_lora_dir(args.lora_path, cache_dir=args.cache_dir)
    adapter_cfg = load_adapter_config(lora_dir)
    sd, fmt = load_lora_state_dict(lora_dir)

    _copy_lora_tree(lora_dir, args.out_dir)
    pairs, selected_prefixes, resolved_targets = _collect_lora_pairs(
        sd=sd,
        target_modules=args.target_modules,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
    )

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
        dataset_path=args.calib_dataset_path,
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
        "target_modules_requested": args.target_modules,
        "target_modules": resolved_targets,
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "calib_dataset": args.calib_dataset,
        "calib_dataset_path": args.calib_dataset_path,
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


def run_hns(args) -> None:
    """Apply post-hoc Hybrid Newton-Schulz editing to a LoRA adapter."""
    lora_dir = ensure_local_lora_dir(args.lora_path, cache_dir=args.cache_dir)
    adapter_cfg = load_adapter_config(lora_dir)
    sd, fmt = load_lora_state_dict(lora_dir)
    _copy_lora_tree(lora_dir, args.out_dir)

    pairs, selected_prefixes, resolved_targets = _collect_lora_pairs(
        sd=sd,
        target_modules=args.target_modules,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
    )

    hns_config = HNSEditConfig(
        fast_steps=args.fast_steps,
        stable_steps=args.stable_steps,
        fast_coefficients=tuple(float(x) for x in args.fast_coefficients),
        stable_coefficients=tuple(float(x) for x in args.stable_coefficients),
        preserve_nuclear_norm=not args.no_preserve_nuclear_norm,
        hns_strength=args.hns_strength,
        output_rank=args.output_rank,
        eps=args.eps,
    )

    module_stats: Dict[str, dict] = {}
    eff_before: list[float] = []
    eff_after: list[float] = []

    for prefix in selected_prefixes:
        keyA, A_old, adapterA = pairs[prefix]["A"]
        keyB, B_old, adapterB = pairs[prefix]["B"]
        adapter_name = adapterA if adapterA is not None else adapterB
        scaling = get_scaling_for_module(adapter_cfg, prefix)

        U, S, Vh, _ = lowrank_svd_from_ba(B_old, A_old)
        U_new, Vh_new, sigma_new, stats = apply_hns_to_svd(U, Vh, S, config=hns_config)
        B_new, A_new = rebuild_ba_from_uv_sigma(U_new, Vh_new, sigma_new)

        sd[keyA] = A_new.to(dtype=A_old.dtype).detach().cpu()
        sd[keyB] = B_new.to(dtype=B_old.dtype).detach().cpu()

        stats.update(
            {
                "module_suffix": prefix.split(".")[-1],
                "layer_index": layer_idx_from_module_prefix(prefix),
                "adapter_name": adapter_name,
                "scaling": float(scaling),
            }
        )
        module_stats[prefix] = stats
        eff_before.append(float(stats["effective_rank_before"]))
        eff_after.append(float(stats["effective_rank_after"]))

    save_lora_state_dict(args.out_dir, sd, fmt)

    summary = {
        "num_modules": len(module_stats),
        "mean_effective_rank_before": (sum(eff_before) / len(eff_before)) if eff_before else 0.0,
        "mean_effective_rank_after": (sum(eff_after) / len(eff_after)) if eff_after else 0.0,
        "mean_effective_rank_delta": ((sum(eff_after) - sum(eff_before)) / len(eff_before)) if eff_before else 0.0,
    }
    meta = {
        "method": "posthoc_hns",
        "lora_path": args.lora_path,
        "target_modules_requested": args.target_modules,
        "target_modules": resolved_targets,
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "output_rank": args.output_rank,
        "fast_steps": args.fast_steps,
        "stable_steps": args.stable_steps,
        "fast_coefficients": [float(x) for x in args.fast_coefficients],
        "stable_coefficients": [float(x) for x in args.stable_coefficients],
        "preserve_nuclear_norm": not args.no_preserve_nuclear_norm,
        "hns_strength": args.hns_strength,
        "eps": args.eps,
    }

    with open(os.path.join(args.out_dir, "spectral_edit_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "module_stats": module_stats}, f, indent=2)

    print(
        "[HNS] Edited "
        f"{len(module_stats)} modules | mean r_eff {summary['mean_effective_rank_before']:.4f} -> "
        f"{summary['mean_effective_rank_after']:.4f}"
    )
    print(f"[Save] Edited adapter saved to: {args.out_dir}")


def _auto_importance_module_budget(module_prefixes: List[str]) -> int:
    """Match the number of modules in the common down_proj+o_proj baseline."""
    layers = {
        layer_idx
        for prefix in module_prefixes
        if (layer_idx := layer_idx_from_module_prefix(prefix)) is not None
    }
    if layers:
        return min(len(module_prefixes), 2 * len(layers))
    return max(1, math.ceil(2 * len(module_prefixes) / 7))


def run_sensitivity_hns(args) -> None:
    """Select task-important, HNS-compatible LoRA modules using calibration CE."""
    set_seed(args.seed)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finetune.data.chat_sft import ensure_chat_template

    if not torch.cuda.is_available():
        raise RuntimeError("sensitivity-hns requires a CUDA GPU for calibration gradients.")
    device = "cuda"

    lora_dir = ensure_local_lora_dir(args.lora_path, cache_dir=args.cache_dir)
    adapter_cfg = load_adapter_config(lora_dir)
    sd, fmt = load_lora_state_dict(lora_dir)
    pairs, selected_prefixes, resolved_targets = _collect_lora_pairs(
        sd=sd,
        target_modules=args.target_modules,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
    )

    module_budget = args.module_budget
    if module_budget is None:
        module_budget = _auto_importance_module_budget(selected_prefixes)
    if module_budget < 1:
        raise ValueError("--module_budget must be >= 1")

    print(
        f"[Sensitivity-HNS] candidates={len(selected_prefixes)} "
        f"importance_budget={min(module_budget, len(selected_prefixes))}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, cache_dir=args.cache_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.sft_format == "chat":
        ensure_chat_template(tokenizer, args.base_model)

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype_by_name[args.dtype],
        low_cpu_mem_usage=True,
        device_map=None,
        cache_dir=args.cache_dir,
    ).to(device)
    model = PeftModel.from_pretrained(base, lora_dir, is_trainable=True).to(device)
    model.eval()
    model.config.use_cache = False
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in parameter_name)

    name_to_module = dict(model.named_modules())
    specs: Dict[str, ModuleSpec] = {}
    hns_sigmas: Dict[str, torch.Tensor] = {}
    hns_stats: Dict[str, dict] = {}
    hns_config = HNSEditConfig(
        fast_steps=args.fast_steps,
        stable_steps=args.stable_steps,
        fast_coefficients=tuple(float(x) for x in args.fast_coefficients),
        stable_coefficients=tuple(float(x) for x in args.stable_coefficients),
        preserve_nuclear_norm=not args.no_preserve_nuclear_norm,
        hns_strength=1.0,
        output_rank=None,
        eps=args.eps,
    )

    for prefix in selected_prefixes:
        _, A_cpu, adapter_a = pairs[prefix]["A"]
        _, B_cpu, adapter_b = pairs[prefix]["B"]
        adapter_name = adapter_a if adapter_a is not None else adapter_b

        module_name = prefix
        if module_name not in name_to_module:
            candidates = [name for name in name_to_module if name.endswith(prefix)]
            if not candidates:
                raise RuntimeError(f"Cannot find module {prefix!r} in model")
            module_name = candidates[0]

        U, sigma, Vh, V = lowrank_svd_from_ba(B_cpu.to(device), A_cpu.to(device))
        _, _, sigma_hns, stats = apply_hns_to_svd(U, Vh, sigma, config=hns_config)
        specs[prefix] = ModuleSpec(
            module_prefix=prefix,
            module=name_to_module[module_name],
            U=U.detach(),
            V=V.detach(),
            Vh=Vh.detach(),
            sigma0=sigma.detach().cpu(),
            scaling=get_scaling_for_module(adapter_cfg, prefix),
            adapter=adapter_name,
        )
        hns_sigmas[prefix] = sigma_hns.detach().cpu()
        hns_stats[prefix] = stats

    calib_config = args.calib_config
    if calib_config is None and args.calib_dataset == "gsm8k":
        calib_config = "main"
    formatter, normalized_fields = build_calib_formatter(args.calib_dataset, args.calib_text_fields)
    dataset = load_calibration_split(
        args.calib_dataset,
        calib_config,
        args.calib_split,
        cache_dir=args.cache_dir,
        dataset_path=args.calib_dataset_path,
    )
    calib_seed = args.calib_seed if args.calib_seed is not None else args.seed
    examples = sample_calibration_examples(
        dataset,
        args.calib_samples,
        args.calib_shuffle,
        calib_seed,
        args.calib_start,
    )
    if not examples:
        raise RuntimeError("Calibration selection produced zero examples")

    gradient_batches: Dict[str, list[torch.Tensor]] = {prefix: [] for prefix in selected_prefixes}
    handles = register_sigma_hooks(specs)
    total_loss = 0.0
    supervised_tokens = 0
    batch_count = 0
    HOOK_CTX.reset()
    try:
        batch_size = max(1, args.calib_batch_size)
        total_batches = math.ceil(len(examples) / batch_size)
        for start in range(0, len(examples), batch_size):
            raw_batch = examples[start : start + batch_size]
            if args.sft_format == "chat":
                input_ids, attention_mask, labels = make_chat_calib_batch(
                    tokenizer,
                    raw_batch,
                    formatter,
                    chat_template_mode=args.chat_template_mode,
                    max_seq_len=args.max_seq_len,
                )
            else:
                input_ids, attention_mask, labels = make_calib_batch(
                    tokenizer,
                    raw_batch,
                    formatter,
                    add_eos=True,
                )
                if input_ids.shape[1] > args.max_seq_len:
                    input_ids = input_ids[:, : args.max_seq_len]
                    attention_mask = attention_mask[:, : args.max_seq_len]
                    labels = labels[:, : args.max_seq_len]

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            HOOK_CTX.attn_mask = attention_mask
            HOOK_CTX.gsum = {}

            model.zero_grad(set_to_none=True)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            output.loss.backward()

            missing = [prefix for prefix in selected_prefixes if prefix not in HOOK_CTX.gsum]
            if missing:
                raise RuntimeError(
                    f"No singular-value gradient captured for {len(missing)} modules; first={missing[0]}"
                )
            for prefix in selected_prefixes:
                gradient_batches[prefix].append(HOOK_CTX.gsum[prefix].clone())

            total_loss += float(output.loss.item())
            supervised_tokens += int((labels != -100).sum().item())
            batch_count += 1
            model.zero_grad(set_to_none=True)
            if batch_count % 5 == 0 or batch_count == total_batches:
                print(
                    f"[Calibration] batch {batch_count}/{total_batches} "
                    f"loss={output.loss.item():.4f}"
                )
    finally:
        remove_hooks(handles)
        HOOK_CTX.attn_mask = None

    raw_scores = {
        prefix: score_module_gradient_batches(
            specs[prefix].sigma0,
            hns_sigmas[prefix],
            torch.stack(gradient_batches[prefix], dim=0),
        )
        for prefix in selected_prefixes
    }
    require_compatibility = args.selection_rule == "importance_compatible"
    chosen, annotated_scores = select_important_modules(
        raw_scores,
        module_budget=module_budget,
        require_positive_compatibility=require_compatibility,
        min_compatibility=args.min_compatibility,
    )
    _copy_lora_tree(lora_dir, args.out_dir)
    module_stats: Dict[str, dict] = {}
    for prefix in chosen:
        key_a, A_old, _ = pairs[prefix]["A"]
        key_b, B_old, _ = pairs[prefix]["B"]
        spec = specs[prefix]
        sigma_new = hns_sigmas[prefix].to(device)
        B_new, A_new = rebuild_ba_from_uv_sigma(spec.U, spec.Vh, sigma_new)
        sd[key_a] = A_new.to(dtype=A_old.dtype).detach().cpu()
        sd[key_b] = B_new.to(dtype=B_old.dtype).detach().cpu()

        stats = dict(hns_stats[prefix])
        stats.update(
            {
                "module_suffix": prefix.split(".")[-1],
                "layer_index": layer_idx_from_module_prefix(prefix),
                "importance": annotated_scores[prefix].importance,
                "compatibility": annotated_scores[prefix].compatibility,
                "hns_risk": annotated_scores[prefix].hns_risk,
            }
        )
        module_stats[prefix] = stats

    save_lora_state_dict(args.out_dir, sd, fmt)
    ranked_modules = sorted(
        selected_prefixes,
        key=lambda prefix: (-annotated_scores[prefix].importance, prefix),
    )
    selection_stats = {
        prefix: annotated_scores[prefix].to_dict() for prefix in ranked_modules
    }
    suffix_counts: Dict[str, int] = {}
    for prefix in chosen:
        suffix = prefix.split(".")[-1]
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

    meta = {
        "method": "calibration_sensitivity_hns",
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "target_modules_requested": args.target_modules,
        "target_modules": resolved_targets,
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "selection_rule": args.selection_rule,
        "module_budget": module_budget,
        "min_compatibility": args.min_compatibility,
        "calib_dataset": args.calib_dataset,
        "calib_dataset_path": args.calib_dataset_path,
        "calib_config": calib_config,
        "calib_split": args.calib_split,
        "calib_text_fields": normalized_fields,
        "calib_samples": args.calib_samples,
        "calib_samples_used": len(examples),
        "calib_batch_size": args.calib_batch_size,
        "calib_shuffle": args.calib_shuffle,
        "calib_seed": calib_seed,
        "calib_start": args.calib_start,
        "sft_format": args.sft_format,
        "chat_template_mode": args.chat_template_mode,
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
        "fast_steps": args.fast_steps,
        "stable_steps": args.stable_steps,
        "fast_coefficients": [float(x) for x in args.fast_coefficients],
        "stable_coefficients": [float(x) for x in args.stable_coefficients],
        "preserve_nuclear_norm": not args.no_preserve_nuclear_norm,
        "hns_strength": 1.0,
        "seed": args.seed,
    }
    summary = {
        "num_candidates": len(selected_prefixes),
        "num_importance_shortlisted": min(module_budget, len(selected_prefixes)),
        "num_selected": len(chosen),
        "num_compatibility_rejected": sum(
            score.rejection_reason == "non_positive_hns_compatibility"
            for score in annotated_scores.values()
        ),
        "selected_suffix_counts": suffix_counts,
        "average_calibration_loss": total_loss / max(1, batch_count),
        "supervised_tokens": supervised_tokens,
        "selected_modules": chosen,
    }
    with open(os.path.join(args.out_dir, "spectral_edit_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "meta": meta,
                "summary": summary,
                "module_selection": selection_stats,
                "module_stats": module_stats,
            },
            handle,
            indent=2,
        )

    print(
        f"[Sensitivity-HNS] selected {len(chosen)}/{len(selected_prefixes)} modules "
        f"after importance+compatibility filtering"
    )
    print(f"[Sensitivity-HNS] selected suffixes: {suffix_counts}")
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
        help="Module names to edit (default: down_proj o_proj). Use 'all_modules' to edit every LoRA matrix suffix found.",
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
        "--calib_dataset_path",
        type=str,
        default=None,
        help="Optional local calibration dataset root or file. Supports split-named .parquet/.json/.jsonl files.",
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

    hns_parser = subparsers.add_parser("hns", help="Apply post-hoc Hybrid Newton-Schulz editing to a LoRA adapter")
    hns_parser.add_argument("--lora_path", type=str, required=True, help="Path or HF ID for LoRA adapter")
    hns_parser.add_argument("--out_dir", type=str, required=True, help="Output directory for edited adapter")
    hns_parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["down_proj", "o_proj"],
        help="Module names to edit (default: down_proj o_proj). Use 'all_modules' to edit every LoRA matrix suffix found.",
    )
    hns_parser.add_argument("--layer_min", type=int, default=0, help="Minimum layer index to edit")
    hns_parser.add_argument("--layer_max", type=int, default=10**9, help="Maximum layer index to edit")
    hns_parser.add_argument(
        "--output_rank",
        type=int,
        default=None,
        help="Optional rank to keep after HNS refactorization. Defaults to the adapter's current rank.",
    )
    hns_parser.add_argument(
        "--fast_steps",
        type=int,
        default=8,
        help="Number of aggressive Muon-style Newton-Schulz steps.",
    )
    hns_parser.add_argument(
        "--stable_steps",
        type=int,
        default=2,
        help="Number of stable Newton-Schulz refinement steps.",
    )
    hns_parser.add_argument(
        "--fast_coefficients",
        type=float,
        nargs=3,
        default=list(FAST_HNS_COEFFICIENTS),
        help="Stage-1 coefficients a b c (default: 3.4445 -4.7750 2.0315).",
    )
    hns_parser.add_argument(
        "--stable_coefficients",
        type=float,
        nargs=3,
        default=list(STABLE_HNS_COEFFICIENTS),
        help="Stage-2 coefficients a b c (default: 2 -1.5 0.5).",
    )
    hns_parser.add_argument(
        "--no_preserve_nuclear_norm",
        action="store_true",
        help="Disable nuclear-norm restoration after HNS.",
    )
    hns_parser.add_argument(
        "--hns_strength",
        type=float,
        default=1.0,
        help="Interpolate from the original spectrum (0) to the full HNS edit (1).",
    )
    hns_parser.add_argument(
        "--eps",
        type=float,
        default=1e-7,
        help="Numerical epsilon used for normalization and norm restoration.",
    )
    hns_parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for HF downloads")

    sensitivity_hns_parser = subparsers.add_parser(
        "sensitivity-hns",
        help="Use calibration task importance and HNS compatibility to select modules for full HNS editing",
    )
    sensitivity_hns_parser.add_argument("--base_model", type=str, required=True)
    sensitivity_hns_parser.add_argument("--lora_path", type=str, required=True)
    sensitivity_hns_parser.add_argument("--out_dir", type=str, required=True)
    sensitivity_hns_parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["all_modules"],
        help="Candidate LoRA module suffixes. Defaults to every module present in the adapter.",
    )
    sensitivity_hns_parser.add_argument("--layer_min", type=int, default=0)
    sensitivity_hns_parser.add_argument("--layer_max", type=int, default=10**9)
    sensitivity_hns_parser.add_argument(
        "--module_budget",
        type=int,
        default=None,
        help="High-importance shortlist size. Default: 2 x number of transformer layers (down+o matched).",
    )
    sensitivity_hns_parser.add_argument(
        "--selection_rule",
        choices=("importance", "importance_compatible"),
        default="importance_compatible",
        help="Select by task importance alone or additionally require positive predicted HNS benefit.",
    )
    sensitivity_hns_parser.add_argument(
        "--min_compatibility",
        type=float,
        default=0.0,
        help="Minimum -<grad, HNS_delta> for shortlisted modules.",
    )

    sensitivity_hns_parser.add_argument("--calib_dataset", type=str, default="gsm8k")
    sensitivity_hns_parser.add_argument("--calib_dataset_path", type=str, default=None)
    sensitivity_hns_parser.add_argument("--calib_config", type=str, default=None)
    sensitivity_hns_parser.add_argument("--calib_split", type=str, default="train")
    sensitivity_hns_parser.add_argument("--calib_text_fields", type=str, nargs="*", default=None)
    sensitivity_hns_parser.add_argument("--calib_samples", type=int, default=256)
    sensitivity_hns_parser.add_argument("--calib_batch_size", type=int, default=2)
    sensitivity_hns_parser.add_argument("--calib_shuffle", action="store_true")
    sensitivity_hns_parser.add_argument("--calib_seed", type=int, default=None)
    sensitivity_hns_parser.add_argument("--calib_start", type=int, default=0)
    sensitivity_hns_parser.add_argument(
        "--sft_format",
        choices=("chat", "plain"),
        default="chat",
        help="Calibration rendering; chat is recommended to match chat SFT/evaluation.",
    )
    sensitivity_hns_parser.add_argument(
        "--chat_template_mode",
        choices=("auto", "thinking", "non_thinking"),
        default="auto",
    )
    sensitivity_hns_parser.add_argument("--max_seq_len", type=int, default=2048)
    sensitivity_hns_parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )

    sensitivity_hns_parser.add_argument("--fast_steps", type=int, default=8)
    sensitivity_hns_parser.add_argument("--stable_steps", type=int, default=2)
    sensitivity_hns_parser.add_argument(
        "--fast_coefficients",
        type=float,
        nargs=3,
        default=list(FAST_HNS_COEFFICIENTS),
    )
    sensitivity_hns_parser.add_argument(
        "--stable_coefficients",
        type=float,
        nargs=3,
        default=list(STABLE_HNS_COEFFICIENTS),
    )
    sensitivity_hns_parser.add_argument("--no_preserve_nuclear_norm", action="store_true")
    sensitivity_hns_parser.add_argument("--eps", type=float, default=1e-7)
    sensitivity_hns_parser.add_argument("--cache_dir", type=str, default=None)
    sensitivity_hns_parser.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "edit":
        run_edit(args)
    elif args.command == "hns":
        run_hns(args)
    elif args.command == "sensitivity-hns":
        run_sensitivity_hns(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
