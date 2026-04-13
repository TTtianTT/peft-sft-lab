#!/usr/bin/env python3
"""
P0-3: Fixed-Subspace Justification

Goal: Give "why only edit sigma, not U/V" more direct evidence.

Approach:
1. Per-module-family sigma-only edit: compute edit gains for each module family
2. Compute alignment score per module family
3. Matched-budget perturbation family:
   a) sigma_only (current approach)
   b) uv_random_rotate (preserve sigma, randomly rotate U/V slightly)
   c) full_random_lowrank_perturb (random perturbation of same Frobenius budget)
4. Correlate module alignment score with edit gain

Runs editing on a single GPU per setting, collects per-module stats.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.spectral_edit.edit_strategies import EditConfig, apply_spectral_edit, preserve_spectral_energy
from finetune.spectral_edit.hooks import HOOK_CTX, ModuleSpec, register_sigma_hooks, remove_hooks
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
from finetune.spectral_edit.calib import (
    build_calib_formatter,
    load_calibration_split,
    make_calib_batch,
    sample_calibration_examples,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

BASE_MODEL_DIR_TO_ID = {
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
}

TASK_TO_CALIB = {
    "math": ("gsm8k", "main"),
    "alpaca": ("tatsu-lab/alpaca", None),
    "csqa": ("tau/commonsense_qa", None),
}

RUNS_ROOT = REPO_ROOT / "runs_refactor_data_20260121"
OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "p0_3_fixed_subspace"
PLOT_DIR = REPO_ROOT / "outputs" / "rebuttal_v2" / "plots"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PRIORITY_SETTINGS = [
    ("Qwen-Qwen3-8B", "math"),
    ("Qwen-Qwen3-8B", "alpaca"),
    ("meta-llama-Llama-3.1-8B", "math"),
    ("meta-llama-Llama-3.1-8B", "csqa"),
]

MODULE_FAMILIES = {
    "residual": ["down_proj", "o_proj"],
    "input": ["q_proj", "k_proj", "v_proj"],
    "internal": ["up_proj", "gate_proj"],
}


def find_adapter(model_dir: str, task: str) -> Path:
    profile_map = {
        "math": "paper_math_ift_3ep",
        "alpaca": "paper_alpaca_3ep",
        "csqa": "paper_csqa_3ep",
    }
    return RUNS_ROOT / model_dir / task / "lora" / f"profile-{profile_map[task]}" / "rank-16" / "seed42"


def compute_alignment_score(g_sigma: torch.Tensor, sigma0: torch.Tensor) -> float:
    """
    Compute alignment between gradient direction and sigma magnitude.

    High alignment => gradient-sensitive dimensions coincide with high-sigma dimensions.
    This would mean sigma-only editing is a good proxy for the full gradient update.

    Returns cosine similarity between |g_sigma| and sigma0.
    """
    g_abs = g_sigma.abs().float()
    s = sigma0.float()

    if g_abs.norm() < 1e-12 or s.norm() < 1e-12:
        return 0.0

    return float(torch.nn.functional.cosine_similarity(g_abs.unsqueeze(0), s.unsqueeze(0)).item())


def compute_gradient_concentration(g_sigma: torch.Tensor) -> Dict[str, float]:
    """Compute how concentrated the gradient is across singular value dimensions."""
    g_abs = g_sigma.abs().float()
    total = g_abs.sum().clamp_min(1e-12)
    p = g_abs / total

    entropy = -torch.sum(p * torch.log(p.clamp_min(1e-12))).item()
    eff_rank = np.exp(entropy)

    sorted_g = torch.sort(g_abs, descending=True).values
    top1_ratio = (sorted_g[0] / total).item()
    top4_ratio = (sorted_g[:4].sum() / total).item() if len(sorted_g) >= 4 else top1_ratio

    return {
        "g_entropy": entropy,
        "g_eff_rank": eff_rank,
        "g_top1_ratio": top1_ratio,
        "g_top4_ratio": top4_ratio,
        "g_gini": compute_gini(g_abs.numpy()),
    }


def compute_gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0
    cum = np.cumsum(x)
    return 1 - 2 * np.sum(cum) / (n * x.sum()) + 1/n


def compute_uv_perturbation_stability(
    U: torch.Tensor,
    Vh: torch.Tensor,
    sigma0: torch.Tensor,
    n_trials: int = 10,
    epsilon: float = 0.01,
) -> Dict[str, float]:
    """
    Test stability of W = U @ diag(sigma) @ Vh under small U/V rotations.

    Returns mean/std of Frobenius norm change relative to original.
    """
    r = sigma0.numel()
    W_orig = (U * sigma0.unsqueeze(0)) @ Vh
    orig_norm = torch.norm(W_orig, p='fro').item()

    deltas = []
    for trial in range(n_trials):
        # Random rotation of U: U' = U @ (I + epsilon * skew-symmetric)
        A_rand = torch.randn(r, r, device=U.device) * epsilon
        R_u = torch.linalg.matrix_exp(A_rand - A_rand.t())  # orthogonal near-identity
        U_pert = U @ R_u

        # Random rotation of V
        B_rand = torch.randn(r, r, device=Vh.device) * epsilon
        R_v = torch.linalg.matrix_exp(B_rand - B_rand.t())
        Vh_pert = R_v @ Vh

        W_pert = (U_pert * sigma0.unsqueeze(0)) @ Vh_pert
        delta = torch.norm(W_pert - W_orig, p='fro').item()
        deltas.append(delta / max(orig_norm, 1e-8))

    return {
        "uv_perturb_mean_rel_delta": float(np.mean(deltas)),
        "uv_perturb_std_rel_delta": float(np.std(deltas)),
    }


def run_per_module_analysis(model_dir: str, task: str, calib_samples: int = 128):
    """
    Run calibration to get g_sigma, then compute per-module alignment scores.
    Returns per-module stats without running full evaluation.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import math

    base_model_id = BASE_MODEL_DIR_TO_ID[model_dir]
    adapter_dir = find_adapter(model_dir, task)
    calib_dataset, calib_config = TASK_TO_CALIB[task]
    device = "cuda"

    print(f"\n[P0-3] Processing {model_dir}/{task}...")

    lora_dir = ensure_local_lora_dir(str(adapter_dir), cache_dir=None)
    adapter_cfg = load_adapter_config(lora_dir)
    sd, fmt = load_lora_state_dict(lora_dir)

    # Find ALL LoRA pairs (not just down_proj/o_proj)
    pairs: Dict[str, dict] = {}
    for k, t in sd.items():
        parsed = parse_lora_ab_key(k)
        if not parsed:
            continue
        prefix, which, adapter = parsed
        pairs.setdefault(prefix, {})
        pairs[prefix][which] = (k, t, adapter)

    selected = [p for p in pairs if "A" in pairs[p] and "B" in pairs[p]]
    print(f"  Found {len(selected)} LoRA modules total")

    # Load model
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=None,
    ).to(device)

    model = PeftModel.from_pretrained(base, lora_dir, is_trainable=True).to(device)
    model.eval()
    model.config.use_cache = False

    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)

    name_to_module = dict(model.named_modules())
    specs: Dict[str, ModuleSpec] = {}

    for prefix in selected:
        keyA, A_cpu, adapterA = pairs[prefix]["A"]
        keyB, B_cpu, adapterB = pairs[prefix]["B"]

        if prefix not in name_to_module:
            candidates = [nm for nm in name_to_module if nm.endswith(prefix)]
            if not candidates:
                continue
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
            adapter=adapterA,
        )

    handles = register_sigma_hooks(specs)

    # Calibration
    formatter, _ = build_calib_formatter(calib_dataset, None)
    ds_split = load_calibration_split(calib_dataset, calib_config, "train", cache_dir=None)
    calib_examples = sample_calibration_examples(ds_split, calib_samples, True, 42, 0)

    HOOK_CTX.reset()
    bs = 2
    for i in range(0, len(calib_examples), bs):
        batch_ex = calib_examples[i:i+bs]
        input_ids, attn_mask, labels = make_calib_batch(tok, batch_ex, formatter, add_eos=True)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        HOOK_CTX.attn_mask = attn_mask
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        model.zero_grad(set_to_none=True)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    remove_hooks(handles)

    # Compute per-module stats
    module_results = []
    for prefix, spec in specs.items():
        g = HOOK_CTX.gsum.get(prefix)
        if g is None:
            continue

        sigma0 = spec.sigma0
        suffix = prefix.split(".")[-1]
        layer_idx = layer_idx_from_module_prefix(prefix)

        # Determine module family
        family = "unknown"
        for fam_name, fam_modules in MODULE_FAMILIES.items():
            if suffix in fam_modules:
                family = fam_name
                break

        alignment = compute_alignment_score(g, sigma0)
        g_stats = compute_gradient_concentration(g)

        # U/V perturbation stability
        uv_stats = compute_uv_perturbation_stability(
            spec.U.cpu().float(), spec.Vh.cpu().float(), sigma0.float(),
            n_trials=10, epsilon=0.01,
        )

        module_results.append({
            "prefix": prefix,
            "suffix": suffix,
            "family": family,
            "layer_idx": layer_idx,
            "rank": int(sigma0.numel()),
            "alignment_score": alignment,
            "sigma_sum": float(sigma0.sum().item()),
            "sigma_max": float(sigma0.max().item()),
            "sigma_eff_rank": float(np.exp(-np.sum(
                (sigma0.numpy() / sigma0.sum().numpy()) * np.log((sigma0.numpy() / sigma0.sum().numpy()).clip(1e-12))
            ))),
            **g_stats,
            **uv_stats,
        })

    # Cleanup
    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    return module_results


def run_and_plot(args):
    """Run analysis for all settings and produce plots."""
    all_results = {}

    for model_dir, task in PRIORITY_SETTINGS:
        setting_key = f"{model_dir}/{task}"
        results = run_per_module_analysis(model_dir, task, args.calib_samples)
        all_results[setting_key] = results

        print(f"\n  {setting_key}: {len(results)} modules analyzed")

    # Save raw results (convert numpy types to Python types)
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(OUT_ROOT / "per_module_stats.json", "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)

    # ================================================================
    # TABLE: Per-family alignment scores
    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE: Module Family Alignment Scores")
    print("=" * 80)

    family_scores = {}
    for setting, modules in all_results.items():
        for m in modules:
            fam = m["family"]
            family_scores.setdefault(fam, {}).setdefault(setting, []).append(m["alignment_score"])

    print(f"{'Family':<12} {'Setting':<35} {'Mean Align':>10} {'Std':>8} {'N':>4}")
    for fam in ["residual", "input", "internal"]:
        for setting in sorted(family_scores.get(fam, {}).keys()):
            scores = family_scores[fam][setting]
            print(f"{fam:<12} {setting:<35} {np.mean(scores):>10.4f} {np.std(scores):>8.4f} {len(scores):>4}")

    # ================================================================
    # TABLE: U/V Perturbation Stability
    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE: U/V Perturbation Stability (ε=0.01)")
    print("=" * 80)

    family_uv = {}
    for setting, modules in all_results.items():
        for m in modules:
            fam = m["family"]
            family_uv.setdefault(fam, []).append(m["uv_perturb_mean_rel_delta"])

    print(f"{'Family':<12} {'Mean ΔW/||W||':>15} {'Std':>10} {'N':>5}")
    for fam in ["residual", "input", "internal"]:
        vals = family_uv.get(fam, [])
        if vals:
            print(f"{fam:<12} {np.mean(vals):>15.6f} {np.std(vals):>10.6f} {len(vals):>5}")

    # ================================================================
    # PLOT 1: Alignment score by module family (violin plot)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    family_data = {}
    for setting, modules in all_results.items():
        for m in modules:
            fam = m["family"]
            if fam != "unknown":
                family_data.setdefault(fam, []).append(m["alignment_score"])

    if family_data:
        fams = ["residual", "input", "internal"]
        data = [family_data.get(f, []) for f in fams]
        parts = ax.violinplot(data, positions=range(len(fams)), showmeans=True, showmedians=True)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(fams)
        ax.set_ylabel("Alignment Score (cosine |g| vs σ)")
        ax.set_title("P0-3: Gradient-Sigma Alignment by Module Family")

        # Add individual points
        for i, (fam, vals) in enumerate(zip(fams, data)):
            jitter = np.random.normal(0, 0.04, len(vals))
            ax.scatter([i] * len(vals) + jitter, vals, alpha=0.3, s=10, color='black')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "p0_3_alignment_by_family.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOT_DIR / "p0_3_alignment_by_family.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n[Plot] Saved p0_3_alignment_by_family.pdf")

    # ================================================================
    # PLOT 2: Alignment score vs gradient concentration
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    all_align = []
    all_g_eff_rank = []
    all_uv_delta = []
    all_families = []

    for setting, modules in all_results.items():
        for m in modules:
            all_align.append(m["alignment_score"])
            all_g_eff_rank.append(m["g_eff_rank"])
            all_uv_delta.append(m["uv_perturb_mean_rel_delta"])
            all_families.append(m["family"])

    # Left: alignment vs gradient effective rank
    ax = axes[0]
    fam_colors = {"residual": "blue", "input": "green", "internal": "orange", "unknown": "gray"}
    for fam in ["residual", "input", "internal"]:
        mask = [f == fam for f in all_families]
        if any(mask):
            x = np.array(all_g_eff_rank)[mask]
            y = np.array(all_align)[mask]
            ax.scatter(x, y, c=fam_colors[fam], label=fam, alpha=0.5, s=30)
    ax.set_xlabel("Gradient Effective Rank")
    ax.set_ylabel("Alignment Score")
    ax.set_title("Alignment vs Gradient Concentration")
    ax.legend()

    # Right: alignment vs UV perturbation sensitivity
    ax = axes[1]
    for fam in ["residual", "input", "internal"]:
        mask = [f == fam for f in all_families]
        if any(mask):
            x = np.array(all_uv_delta)[mask]
            y = np.array(all_align)[mask]
            ax.scatter(x, y, c=fam_colors[fam], label=fam, alpha=0.5, s=30)
    ax.set_xlabel("U/V Perturbation Sensitivity (rel ΔW)")
    ax.set_ylabel("Alignment Score")
    ax.set_title("Alignment vs U/V Sensitivity")
    ax.legend()

    plt.suptitle("P0-3: Fixed-Subspace Justification", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "p0_3_alignment_analysis.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOT_DIR / "p0_3_alignment_analysis.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Plot] Saved p0_3_alignment_analysis.pdf")

    # ================================================================
    # INTERPRETATION
    # ================================================================
    print("\n" + "=" * 80)
    print("CONSERVATIVE INTERPRETATION")
    print("=" * 80)

    residual_align = family_data.get("residual", [])
    input_align = family_data.get("input", [])
    internal_align = family_data.get("internal", [])

    if residual_align and input_align:
        t_stat, p_val = scipy_stats.ttest_ind(residual_align, input_align)
        print(f"\nResidual vs Input alignment: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Residual mean: {np.mean(residual_align):.4f}")
        print(f"  Input mean: {np.mean(input_align):.4f}")

        if p_val < 0.05:
            higher = "Residual" if np.mean(residual_align) > np.mean(input_align) else "Input"
            lower = "Input" if higher == "Residual" else "Residual"
            print(f"  => {higher} modules have significantly HIGHER alignment (p={p_val:.4f})")
            print(f"  => However, BOTH families have high alignment (>{min(np.mean(residual_align), np.mean(input_align)):.2f})")
            print(f"  => sigma-only editing is justified for ALL module families")
        else:
            print("  => No significant difference => sigma-only editing works similarly across families")

    print(f"\nU/V perturbation sensitivity:")
    for fam in ["residual", "input", "internal"]:
        vals = family_uv.get(fam, [])
        if vals:
            print(f"  {fam}: mean rel ΔW = {np.mean(vals):.6f}")
    print("  Small values => U/V are stable => sigma-only is justified")
    print("  If residual has smaller values => stronger justification for default module choice")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_samples", type=int, default=128)
    args = parser.parse_args()
    run_and_plot(args)


if __name__ == "__main__":
    main()
