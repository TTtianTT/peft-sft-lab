#!/usr/bin/env python3
"""
Experiment 4: IFEval Anomaly Deep Dive

Explain WHY all perturbations improve IFEval by +8-12% despite hurting other tasks.

Part A: Spectral concentration comparison across tasks
  - Compare spectral entropy / effective rank of adapters across tasks
  - Is the Alpaca adapter's spectrum more concentrated?

Part B: Per-constraint-type analysis of IFEval
  - Categorize IFEval constraints and compare pass rates by method

Part C: Generation length analysis
  - Compare response length distributions across methods

Part D: Spectral entropy of ΔW before/after editing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from finetune.spectral_edit.io import (
    load_lora_state_dict, parse_lora_ab_key,
    layer_idx_from_module_prefix,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba

OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "exp4_ifeval_deep_dive"
PYTHON = sys.executable

ALL_ADAPTERS = {
    "meta-llama-Llama-3.1-8B/alpaca": str(REPO_ROOT / "runs/meta-llama-Llama-3.1-8B/alpaca/lora/profile-paper_alpaca_3ep/rank-16/seed42"),
    "meta-llama-Llama-3.1-8B/csqa": str(REPO_ROOT / "runs/meta-llama-Llama-3.1-8B/csqa/lora/profile-paper_csqa_3ep/rank-16/seed42"),
    "meta-llama-Llama-3.1-8B/math": str(REPO_ROOT / "runs/meta-llama-Llama-3.1-8B/metamath/lora/profile-paper_math_ift_3ep/rank-16/seed42"),
    "Qwen-Qwen3-8B/alpaca": str(REPO_ROOT / "runs/Qwen-Qwen3-8B/alpaca/lora/profile-paper_alpaca_3ep/rank-16/seed42"),
    "Qwen-Qwen3-8B/csqa": str(REPO_ROOT / "runs/Qwen-Qwen3-8B/csqa/lora/profile-paper_csqa_3ep/rank-16/seed42"),
    "Qwen-Qwen3-8B/math": str(REPO_ROOT / "runs/Qwen-Qwen3-8B/metamath/lora/profile-paper_math_ift_3ep/rank-16/seed42"),
}

TARGET_MODULES = ["down_proj", "o_proj"]


def part_a_spectral_concentration():
    """Compare spectral properties across task adapters."""
    print("=" * 80)
    print("Part A: Spectral Concentration Comparison")
    print("=" * 80)

    results = {}

    for skey, lora_path in ALL_ADAPTERS.items():
        sd, fmt = load_lora_state_dict(lora_path)

        target_set = set(TARGET_MODULES)
        pairs = {}
        for k, t in sd.items():
            parsed = parse_lora_ab_key(k)
            if not parsed:
                continue
            prefix, which, adapter = parsed
            suffix = prefix.split(".")[-1]
            if suffix not in target_set:
                continue
            pairs.setdefault(prefix, {})
            pairs[prefix][which] = (k, t, adapter)

        selected = [p for p in pairs if "A" in pairs[p] and "B" in pairs[p]]

        entropies = []
        eff_ranks = []
        top1_ratios = []
        gini_coeffs = []

        for prefix in selected:
            _, A_cpu, _ = pairs[prefix]["A"]
            _, B_cpu, _ = pairs[prefix]["B"]
            U, S, Vh, V = lowrank_svd_from_ba(B_cpu, A_cpu)
            sigma = S.numpy()

            # Normalize to probability distribution
            sigma_abs = np.abs(sigma)
            p = sigma_abs / (sigma_abs.sum() + 1e-12)

            # Shannon entropy
            entropy = -np.sum(p * np.log(p + 1e-12))
            entropies.append(entropy)

            # Effective rank (exponential of entropy)
            eff_rank = np.exp(entropy)
            eff_ranks.append(eff_rank)

            # Top-1 ratio
            top1_ratios.append(float(sigma_abs.max() / (sigma_abs.sum() + 1e-12)))

            # Gini coefficient
            n = len(sigma_abs)
            sorted_s = np.sort(sigma_abs)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_s) / (n * np.sum(sorted_s) + 1e-12)) - (n + 1) / n
            gini_coeffs.append(gini)

        results[skey] = {
            "n_modules": len(selected),
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
            "eff_rank_mean": float(np.mean(eff_ranks)),
            "eff_rank_std": float(np.std(eff_ranks)),
            "top1_ratio_mean": float(np.mean(top1_ratios)),
            "top1_ratio_std": float(np.std(top1_ratios)),
            "gini_mean": float(np.mean(gini_coeffs)),
            "gini_std": float(np.std(gini_coeffs)),
        }
        print(f"  {skey:40s} entropy={np.mean(entropies):.4f}  eff_rank={np.mean(eff_ranks):.2f}  "
              f"top1={np.mean(top1_ratios):.4f}  gini={np.mean(gini_coeffs):.4f}")

    # Check if alpaca adapters are more concentrated
    print("\n  Is Alpaca adapter more concentrated?")
    for model in ["meta-llama-Llama-3.1-8B", "Qwen-Qwen3-8B"]:
        alpaca = results.get(f"{model}/alpaca", {})
        others = [results.get(f"{model}/{t}", {}) for t in ["csqa", "math"] if f"{model}/{t}" in results]
        if alpaca and others:
            alpaca_ent = alpaca["entropy_mean"]
            other_ent = np.mean([o["entropy_mean"] for o in others])
            print(f"    {model}: alpaca entropy={alpaca_ent:.4f} vs others mean={other_ent:.4f} "
                  f"({'MORE concentrated' if alpaca_ent < other_ent else 'LESS concentrated'})")

    analysis_dir = OUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis_dir / "spectral_concentration.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def part_b_per_constraint_analysis():
    """Analyze IFEval results per constraint type."""
    print("\n" + "=" * 80)
    print("Part B: Per-Constraint-Type Analysis")
    print("=" * 80)

    # Find IFEval detailed results from prior experiments
    # Look for lm-eval output files that contain per-sample results
    ifeval_results_dirs = []
    for root_dir in [
        REPO_ROOT / "outputs" / "rebuttal_v2" / "p0_1_random_mechanism" / "edited",
        REPO_ROOT / "outputs" / "rebuttal_exp" / "raw" / "multiseed" / "edited_adapters",
        REPO_ROOT / "eval_results",
    ]:
        if root_dir.exists():
            for path in root_dir.rglob("results_*.json"):
                if "ifeval" in str(path).lower() or "alpaca" in str(path).lower():
                    ifeval_results_dirs.append(path)

    print(f"  Found {len(ifeval_results_dirs)} potential IFEval result files")

    # Parse IFEval detailed results
    constraint_data = defaultdict(lambda: defaultdict(list))

    for path in ifeval_results_dirs:
        try:
            with open(path) as f:
                data = json.load(f)
            results = data.get("results", {})
            if "ifeval" in results:
                ifeval_r = results["ifeval"]
                # Get all constraint-level metrics
                for k, v in ifeval_r.items():
                    if isinstance(v, (int, float)):
                        # Try to identify the method from the path
                        method = "unknown"
                        path_str = str(path)
                        for m in ["random_index", "smooth_abs", "grad_direction", "flatten_spectrum",
                                   "shrink_top_only", "suppress_tail_only", "permute_sigma", "uniform_rescale"]:
                            if m in path_str:
                                method = m
                                break
                        if method == "unknown" and "baseline" in path_str:
                            method = "baseline"
                        constraint_data[method][k].append(v)
        except Exception:
            continue

    if constraint_data:
        print("\n  IFEval metrics by method:")
        for method in sorted(constraint_data.keys()):
            print(f"\n    {method}:")
            for metric, vals in sorted(constraint_data[method].items()):
                print(f"      {metric}: {np.mean(vals):.4f}")

    analysis_dir = OUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis_dir / "per_constraint.json", "w") as f:
        json.dump({k: {mk: mv for mk, mv in v.items()} for k, v in constraint_data.items()}, f, indent=2, default=str)

    return constraint_data


def part_c_generation_length():
    """Analyze generation length from IFEval error taxonomy data."""
    print("\n" + "=" * 80)
    print("Part C: Generation Length Analysis")
    print("=" * 80)

    # Load P1-6 error taxonomy data
    taxonomy_path = REPO_ROOT / "outputs" / "rebuttal_v2" / "p1_6_ifeval_taxonomy" / "error_taxonomy.json"
    if not taxonomy_path.exists():
        print("  [SKIP] P1-6 error taxonomy not found")
        return {}

    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    # The taxonomy should have per-method error counts
    error_counts = taxonomy.get("error_counts", {})
    if error_counts:
        print("\n  Error type distribution:")
        print(f"  {'Method':<20} {'Empty':>8} {'Format':>8} {'MinWords':>8} {'Total':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for method in ["baseline", "random_index", "smooth_abs", "grad_direction"]:
            counts = error_counts.get(method, {})
            if counts:
                empty = counts.get("empty_response", 0)
                fmt = sum(v for k, v in counts.items() if "format" in k)
                minw = counts.get("below_word_minimum", 0)
                total = sum(counts.values())
                print(f"  {method:<20} {empty:>8} {fmt:>8} {minw:>8} {total:>8}")

    # Check for detailed per-sample data
    detail_path = REPO_ROOT / "outputs" / "rebuttal_v2" / "p1_6_ifeval_taxonomy"
    length_data = {}
    for method_dir in sorted(detail_path.iterdir()) if detail_path.exists() else []:
        if method_dir.is_dir():
            for f in method_dir.rglob("*.json"):
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                    if "response_lengths" in data:
                        length_data[method_dir.name] = data["response_lengths"]
                except Exception:
                    continue

    analysis_dir = OUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis_dir / "generation_length.json", "w") as f:
        json.dump({
            "error_counts": error_counts,
            "length_data_available": bool(length_data),
        }, f, indent=2)

    return error_counts


def plot_results():
    """Generate publication-quality plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = OUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = OUT_ROOT / "analysis"

    # Plot A: Spectral concentration comparison
    conc_path = analysis_dir / "spectral_concentration.json"
    if conc_path.exists():
        with open(conc_path) as f:
            conc = json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for ax, metric, label in zip(axes,
            ["entropy_mean", "eff_rank_mean", "top1_ratio_mean"],
            ["Shannon Entropy", "Effective Rank", "Top-1 Ratio"]):

            # Group by model
            for model in ["meta-llama-Llama-3.1-8B", "Qwen-Qwen3-8B"]:
                tasks = []
                vals = []
                for skey, data in sorted(conc.items()):
                    if model in skey:
                        tasks.append(skey.split("/")[1])
                        vals.append(data[metric])
                if tasks:
                    short_model = model.split("-")[-1]
                    x = np.arange(len(tasks))
                    ax.bar(x + (0.2 if "Qwen" in model else -0.2), vals,
                           width=0.35, label=short_model)
                    ax.set_xticks(x)
                    ax.set_xticklabels(tasks)

            ax.set_ylabel(label, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle("Spectral Concentration by Task Adapter", fontsize=12)
        fig.tight_layout()
        fig.savefig(plot_dir / "exp4_spectral_concentration.pdf", dpi=150)
        plt.close(fig)

    # Plot C: Error taxonomy stacked bar
    gen_path = analysis_dir / "generation_length.json"
    if gen_path.exists():
        with open(gen_path) as f:
            gen_data = json.load(f)

        error_counts = gen_data.get("error_counts", {})
        if error_counts:
            methods = ["baseline", "random_index", "smooth_abs", "grad_direction"]
            methods = [m for m in methods if m in error_counts]

            if methods:
                categories = ["empty_response", "below_word_minimum"]
                format_keys = [k for k in list(error_counts.values())[0].keys() if "format" in k]

                fig, ax = plt.subplots(figsize=(8, 5))
                x = np.arange(len(methods))
                width = 0.6

                bottom = np.zeros(len(methods))
                colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
                labels = ["Empty Response", "Below Min Words", "Format Violations", "Other"]

                for cat_idx, (cat, color, label) in enumerate(zip(
                    ["empty_response", "below_word_minimum"],
                    colors[:2], labels[:2]
                )):
                    vals = [error_counts.get(m, {}).get(cat, 0) for m in methods]
                    ax.bar(x, vals, width, bottom=bottom, color=color, label=label)
                    bottom += np.array(vals)

                # Format violations (sum of all format-related)
                fmt_vals = []
                for m in methods:
                    fmt_total = sum(error_counts.get(m, {}).get(k, 0) for k in format_keys)
                    fmt_vals.append(fmt_total)
                ax.bar(x, fmt_vals, width, bottom=bottom, color=colors[2], label=labels[2])
                bottom += np.array(fmt_vals)

                # Other
                other_vals = []
                known_keys = {"empty_response", "below_word_minimum"} | set(format_keys)
                for m in methods:
                    other = sum(v for k, v in error_counts.get(m, {}).items() if k not in known_keys)
                    other_vals.append(other)
                if any(v > 0 for v in other_vals):
                    ax.bar(x, other_vals, width, bottom=bottom, color=colors[3], label=labels[3])

                ax.set_xticks(x)
                ax.set_xticklabels(methods, rotation=20, ha="right")
                ax.set_ylabel("Error Count", fontsize=11)
                ax.set_title("IFEval Error Taxonomy by Method", fontsize=12)
                ax.legend(loc="upper right", fontsize=9)
                ax.grid(True, axis="y", alpha=0.3)
                fig.tight_layout()
                fig.savefig(plot_dir / "exp4_ifeval_error_stacked.pdf", dpi=150)
                plt.close(fig)

    print(f"\n[Saved] Plots to {plot_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: IFEval Anomaly Deep Dive")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["concentration", "constraints", "length", "plot", "all"])
    args = parser.parse_args()

    if args.phase in ("concentration", "all"):
        part_a_spectral_concentration()
    if args.phase in ("constraints", "all"):
        part_b_per_constraint_analysis()
    if args.phase in ("length", "all"):
        part_c_generation_length()
    if args.phase in ("plot", "all"):
        plot_results()


if __name__ == "__main__":
    main()
