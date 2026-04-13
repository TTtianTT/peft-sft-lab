#!/usr/bin/env python3
"""
P0-1 Sigma-Pattern Analysis: Compare spectral signatures of control vs guided edits.

No GPU needed. Works from metadata files only.

Produces:
1. Aggregate sigma-change table (Δentropy, Δeffective_rank, Δgini, Δtop1_ratio)
2. Per-method sigma-shape comparison plots
3. De-specialization hypothesis test:
   Does flatten/shrink_top produce similar sigma-space changes to random_index?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTROL_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "p0_1_random_mechanism" / "edited"
PRIOR_ROOT = REPO_ROOT / "outputs" / "rebuttal_exp" / "raw" / "multiseed" / "edited_adapters"
EVAL_CSV = REPO_ROOT / "outputs" / "rebuttal_exp" / "raw" / "multiseed_eval" / "eval_results.csv"
OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "p0_1_random_mechanism"
PLOT_DIR = REPO_ROOT / "outputs" / "rebuttal_v2" / "plots"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Map control dir names -> prior experiment dir prefixes
SETTINGS = [
    ("Qwen-Qwen3-8B", "math", "Qwen-Qwen3-8B_math_paper_math_ift_3ep_16_42"),
    ("Qwen-Qwen3-8B", "alpaca", "Qwen-Qwen3-8B_alpaca_paper_alpaca_3ep_16_42"),
    ("meta-llama-Llama-3.1-8B", "math", "meta-llama-Llama-3.1-8B_math_paper_math_ift_3ep_16_42"),
    ("meta-llama-Llama-3.1-8B", "csqa", "meta-llama-Llama-3.1-8B_csqa_paper_csqa_3ep_16_42"),
]

PRIOR_METHODS = ["random_index", "smooth_abs", "grad_direction"]


def compute_sigma_stats_from_meta(sigma_stats: dict) -> dict:
    """Aggregate sigma statistics across all modules in an edit."""
    module_deltas = []
    for mod_name, ms in sigma_stats.items():
        r = ms.get("r", 16)
        s0_sum = ms.get("sigma0_sum", 0)
        sn_sum = ms.get("sigma_new_sum", 0)
        s0_top1 = ms.get("sigma0_top1", 0)
        sn_top1 = ms.get("sigma_new_top1", 0)
        s0_mean = ms.get("sigma0_mean", 0)
        sn_mean = ms.get("sigma_new_mean", 0)
        s0_std = ms.get("sigma0_std", 0)
        sn_std = ms.get("sigma_new_std", 0)

        if s0_sum < 1e-12:
            continue

        # top1 energy ratio change
        top1_ratio_before = s0_top1 / s0_sum if s0_sum > 0 else 0
        top1_ratio_after = sn_top1 / sn_sum if sn_sum > 0 else 0

        # CV change (coefficient of variation ~ inverse of effective rank)
        cv_before = s0_std / s0_mean if s0_mean > 0 else 0
        cv_after = sn_std / sn_mean if sn_mean > 0 else 0

        # L1 preservation ratio
        l1_ratio = sn_sum / s0_sum if s0_sum > 0 else 1.0

        module_deltas.append({
            "top1_ratio_before": top1_ratio_before,
            "top1_ratio_after": top1_ratio_after,
            "delta_top1_ratio": top1_ratio_after - top1_ratio_before,
            "cv_before": cv_before,
            "cv_after": cv_after,
            "delta_cv": cv_after - cv_before,
            "l1_ratio": l1_ratio,
        })

    if not module_deltas:
        return {}

    arr = {k: np.array([m[k] for m in module_deltas]) for k in module_deltas[0]}
    return {
        "n_modules": len(module_deltas),
        "mean_top1_ratio_before": float(arr["top1_ratio_before"].mean()),
        "mean_top1_ratio_after": float(arr["top1_ratio_after"].mean()),
        "mean_delta_top1": float(arr["delta_top1_ratio"].mean()),
        "std_delta_top1": float(arr["delta_top1_ratio"].std()),
        "mean_cv_before": float(arr["cv_before"].mean()),
        "mean_cv_after": float(arr["cv_after"].mean()),
        "mean_delta_cv": float(arr["delta_cv"].mean()),
        "std_delta_cv": float(arr["delta_cv"].std()),
        "mean_l1_ratio": float(arr["l1_ratio"].mean()),
        "frac_top1_decreased": float((arr["delta_top1_ratio"] < 0).mean()),
        "frac_cv_decreased": float((arr["delta_cv"] < 0).mean()),
    }


def load_control_edits() -> list:
    """Load sigma stats from all control edit metadata files."""
    records = []
    for meta_path in CONTROL_ROOT.rglob("spectral_edit_meta.json"):
        try:
            with open(meta_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        meta = data.get("meta", {})
        sigma_stats = data.get("sigma_stats", {})
        if not sigma_stats:
            continue

        # Parse path: .../model_dir/task/control_method/seedN/
        parts = meta_path.relative_to(CONTROL_ROOT).parts
        if len(parts) >= 4:
            model_dir, task, method, seed_dir = parts[0], parts[1], parts[2], parts[3]
            seed = int(seed_dir.replace("seed", ""))
        else:
            model_dir = meta.get("base_model", "").replace("/", "-")
            task = meta.get("lora_path", "").split("/")[-4] if "lora_path" in meta else ""
            method = meta.get("control_method", "unknown")
            seed = meta.get("seed", 0)

        agg = compute_sigma_stats_from_meta(sigma_stats)
        if agg:
            records.append({
                "model_dir": model_dir,
                "task": task,
                "method": method,
                "seed": seed,
                **agg,
            })

    return records


def load_prior_edits() -> list:
    """Load sigma stats from prior rebuttal experiment metadata files."""
    records = []
    for model_dir, task, prior_prefix in SETTINGS:
        prior_dir = PRIOR_ROOT / prior_prefix
        if not prior_dir.exists():
            continue

        for method in PRIOR_METHODS:
            method_dir = prior_dir / method
            if not method_dir.exists():
                continue

            for repeat_dir in method_dir.iterdir():
                meta_path = repeat_dir / "spectral_edit_meta.json"
                if not meta_path.exists():
                    continue

                try:
                    with open(meta_path) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue

                sigma_stats = data.get("sigma_stats", {})
                if not sigma_stats:
                    continue

                # Parse seed from repeat dir name
                seed_str = repeat_dir.name.split("_seed")[-1] if "_seed" in repeat_dir.name else "0"
                seed = int(seed_str)

                agg = compute_sigma_stats_from_meta(sigma_stats)
                if agg:
                    records.append({
                        "model_dir": model_dir,
                        "task": task,
                        "method": method,
                        "seed": seed,
                        **agg,
                    })

    return records


def load_eval_data() -> dict:
    """Load existing eval metrics from prior experiments."""
    import csv
    eval_data = {}
    if EVAL_CSV.exists():
        with open(EVAL_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("metric_value"):
                    try:
                        val = float(row["metric_value"])
                    except (ValueError, TypeError):
                        continue
                    key = (row.get("base_model_dir", ""), row.get("task", ""), row.get("method", ""))
                    eval_data.setdefault(key, []).append(val)
    return eval_data


def analyze():
    control_records = load_control_edits()
    prior_records = load_prior_edits()
    all_records = control_records + prior_records
    eval_data = load_eval_data()

    print(f"[P0-1 Sigma Analysis] Control records: {len(control_records)}, Prior records: {len(prior_records)}")

    # Organize by setting and method
    by_setting = defaultdict(lambda: defaultdict(list))
    for rec in all_records:
        key = (rec["model_dir"], rec["task"])
        by_setting[key][rec["method"]].append(rec)

    # ================================================================
    # TABLE 1: Sigma-Space Changes Per Method
    # ================================================================
    print("\n" + "=" * 90)
    print("TABLE 1: Spectral Signature Changes Per Method (averaged across modules and seeds)")
    print("=" * 90)

    ALL_METHODS = ["random_index", "smooth_abs", "grad_direction",
                   "flatten_spectrum", "shrink_top_only", "suppress_tail_only",
                   "permute_sigma", "uniform_rescale"]

    header = f"{'Setting':<35} {'Method':<18} {'N':>3} {'Δtop1%':>8} {'Δcv':>8} {'L1rat':>6} {'top1↓%':>7}"
    print(header)
    print("-" * len(header))

    table_data = []
    for (model_dir, task), methods in sorted(by_setting.items()):
        setting = f"{model_dir}/{task}"
        for method in ALL_METHODS:
            recs = methods.get(method, [])
            if not recs:
                continue

            dt = np.array([r["mean_delta_top1"] for r in recs])
            dcv = np.array([r["mean_delta_cv"] for r in recs])
            l1r = np.array([r["mean_l1_ratio"] for r in recs])
            frac_dec = np.array([r["frac_top1_decreased"] for r in recs])

            row = {
                "setting": setting, "method": method, "n": len(recs),
                "delta_top1": float(dt.mean()), "delta_top1_std": float(dt.std()),
                "delta_cv": float(dcv.mean()), "delta_cv_std": float(dcv.std()),
                "l1_ratio": float(l1r.mean()),
                "frac_top1_decreased": float(frac_dec.mean()),
            }
            table_data.append(row)

            # Get eval metric if available
            eval_key = (model_dir, task, method)
            eval_vals = eval_data.get(eval_key, [])
            eval_str = f" eval={np.mean(eval_vals):.4f}" if eval_vals else ""

            print(f"{setting:<35} {method:<18} {len(recs):>3} {dt.mean():>+7.4f} "
                  f"{dcv.mean():>+7.3f} {l1r.mean():>5.3f} {frac_dec.mean()*100:>6.1f}%{eval_str}")

    # ================================================================
    # TABLE 2: Method-Level Aggregation
    # ================================================================
    print("\n" + "=" * 90)
    print("TABLE 2: Method-Level Sigma Change Summary (across all settings)")
    print("=" * 90)

    method_agg = defaultdict(list)
    for row in table_data:
        method_agg[row["method"]].append(row)

    print(f"{'Method':<18} {'Mean Δtop1%':>11} {'Mean Δcv':>10} {'Mean L1':>8} {'top1↓%':>8} {'Nset':>5}")
    print("-" * 65)

    method_summary = {}
    for method in ALL_METHODS:
        rows = method_agg.get(method, [])
        if not rows:
            continue
        dt = np.mean([r["delta_top1"] for r in rows])
        dcv = np.mean([r["delta_cv"] for r in rows])
        l1 = np.mean([r["l1_ratio"] for r in rows])
        frac = np.mean([r["frac_top1_decreased"] for r in rows])
        method_summary[method] = {"delta_top1": dt, "delta_cv": dcv, "l1_ratio": l1, "frac_top1_dec": frac}
        print(f"{method:<18} {dt:>+10.4f} {dcv:>+9.3f} {l1:>7.3f} {frac*100:>7.1f}% {len(rows):>5}")

    # ================================================================
    # DE-SPECIALIZATION HYPOTHESIS TEST
    # ================================================================
    print("\n" + "=" * 90)
    print("DE-SPECIALIZATION HYPOTHESIS TEST")
    print("=" * 90)

    ri = method_summary.get("random_index", {})
    fs = method_summary.get("flatten_spectrum", {})
    st = method_summary.get("shrink_top_only", {})
    ur = method_summary.get("uniform_rescale", {})
    ps = method_summary.get("permute_sigma", {})

    if ri and fs:
        print(f"\n  random_index:     Δtop1 = {ri['delta_top1']:+.4f}, Δcv = {ri['delta_cv']:+.3f}")
        print(f"  flatten_spectrum: Δtop1 = {fs['delta_top1']:+.4f}, Δcv = {fs['delta_cv']:+.3f}")
        if st:
            print(f"  shrink_top_only:  Δtop1 = {st['delta_top1']:+.4f}, Δcv = {st['delta_cv']:+.3f}")
        if ur:
            print(f"  uniform_rescale:  Δtop1 = {ur['delta_top1']:+.4f}, Δcv = {ur['delta_cv']:+.3f}")
        if ps:
            print(f"  permute_sigma:    Δtop1 = {ps['delta_top1']:+.4f}, Δcv = {ps['delta_cv']:+.3f}")

        # Direction match: does random_index move sigma in the same direction as flatten?
        ri_direction = "decreases" if ri["delta_top1"] < 0 else "increases"
        fs_direction = "decreases" if fs["delta_top1"] < 0 else "increases"

        print(f"\n  Random index {ri_direction} top-1 concentration (Δ={ri['delta_top1']:+.4f})")
        print(f"  Flatten spectrum {fs_direction} top-1 concentration (Δ={fs['delta_top1']:+.4f})")

        if np.sign(ri["delta_top1"]) == np.sign(fs["delta_top1"]):
            print("  => SAME DIRECTION: random_index acts like (mild) de-specialization")
            print("  => Supports: part of random gain may come from smoothing the spectrum")
        else:
            print("  => DIFFERENT DIRECTION: random_index does NOT simply de-specialize")
            print("  => Opposes de-specialization as primary mechanism")

        # Is random_index closer to permute or flatten in sigma space?
        if ps and fs:
            ri_vec = np.array([ri["delta_top1"], ri["delta_cv"]])
            fs_vec = np.array([fs["delta_top1"], fs["delta_cv"]])
            ps_vec = np.array([ps["delta_top1"], ps["delta_cv"]])

            dist_to_flatten = np.linalg.norm(ri_vec - fs_vec)
            dist_to_permute = np.linalg.norm(ri_vec - ps_vec)

            print(f"\n  Distance in (Δtop1, Δcv) space:")
            print(f"    random → flatten:  {dist_to_flatten:.4f}")
            print(f"    random → permute:  {dist_to_permute:.4f}")
            if dist_to_flatten < dist_to_permute:
                print(f"  => random_index is CLOSER to flatten (de-specialization)")
            else:
                print(f"  => random_index is CLOSER to permute (shuffling)")

    # ================================================================
    # PLOT 1: Sigma-change fingerprints per method (radar/bar chart)
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of Δtop1 by method
    ax = axes[0]
    methods_with_data = [m for m in ALL_METHODS if m in method_summary]
    x = np.arange(len(methods_with_data))
    vals = [method_summary[m]["delta_top1"] for m in methods_with_data]
    colors = {
        "random_index": "#4C72B0", "smooth_abs": "#55A868", "grad_direction": "#C44E52",
        "flatten_spectrum": "#8172B2", "shrink_top_only": "#CCB974",
        "suppress_tail_only": "#64B5CD", "permute_sigma": "#DA8BC3", "uniform_rescale": "#8C8C8C",
    }
    bar_colors = [colors.get(m, "gray") for m in methods_with_data]
    ax.bar(x, vals, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in methods_with_data], fontsize=8)
    ax.set_ylabel("Mean Δ top-1 energy ratio", fontsize=10)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title("(a) Top-1 Concentration Change", fontsize=11)

    # Bar chart of Δcv by method
    ax = axes[1]
    vals2 = [method_summary[m]["delta_cv"] for m in methods_with_data]
    ax.bar(x, vals2, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in methods_with_data], fontsize=8)
    ax.set_ylabel("Mean Δ CV (coefficient of variation)", fontsize=10)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title("(b) Spectral Spread Change", fontsize=11)

    plt.suptitle("P0-1: Spectral Signature of Edit Methods", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "p0_1_sigma_fingerprint.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOT_DIR / "p0_1_sigma_fingerprint.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n[Plot] Saved p0_1_sigma_fingerprint.pdf")

    # ================================================================
    # PLOT 2: Per-setting comparison (control vs guided methods)
    # ================================================================
    n_settings = len(by_setting)
    if n_settings > 0:
        fig, axes = plt.subplots(1, n_settings, figsize=(4.5 * n_settings, 5), squeeze=False)

        for s_idx, ((model_dir, task), methods) in enumerate(sorted(by_setting.items())):
            ax = axes[0, s_idx]
            present_methods = [m for m in ALL_METHODS if m in methods]

            x = np.arange(len(present_methods))
            dt_means = []
            dt_stds = []
            for m in present_methods:
                recs = methods[m]
                dt = [r["mean_delta_top1"] for r in recs]
                dt_means.append(np.mean(dt))
                dt_stds.append(np.std(dt))

            bc = [colors.get(m, "gray") for m in present_methods]
            ax.bar(x, dt_means, yerr=dt_stds, color=bc, alpha=0.8,
                   edgecolor="black", linewidth=0.5, capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels([m[:10] for m in present_methods], rotation=45, ha='right', fontsize=7)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylabel("Δ top-1 ratio")

            short_model = model_dir.split("-")[-1]
            ax.set_title(f"{short_model}/{task}", fontsize=10)

        plt.suptitle("P0-1: Top-1 Concentration Change by Method and Setting", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "p0_1_per_setting_sigma.pdf", bbox_inches="tight", dpi=150)
        plt.savefig(PLOT_DIR / "p0_1_per_setting_sigma.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Plot] Saved p0_1_per_setting_sigma.pdf")

    # ================================================================
    # PLOT 3: Scatter - Δtop1 vs eval metric (for methods with eval data)
    # ================================================================
    scatter_points = []
    for row in table_data:
        parts = row["setting"].split("/")
        model_dir, task = parts[0], parts[1]
        eval_key = (model_dir, task, row["method"])
        eval_vals = eval_data.get(eval_key, [])
        bl_key = (model_dir, task, "baseline")
        bl_vals = eval_data.get(bl_key, [])

        if eval_vals and bl_vals:
            bl_mean = np.mean(bl_vals)
            eval_mean = np.mean(eval_vals)
            delta_metric = eval_mean - bl_mean
            scatter_points.append({
                "method": row["method"],
                "delta_top1": row["delta_top1"],
                "delta_metric": delta_metric,
                "setting": row["setting"],
            })

    if scatter_points:
        fig, ax = plt.subplots(figsize=(8, 6))
        for method in ALL_METHODS:
            pts = [p for p in scatter_points if p["method"] == method]
            if pts:
                x = [p["delta_top1"] for p in pts]
                y = [p["delta_metric"] for p in pts]
                ax.scatter(x, y, c=colors.get(method, "gray"), label=method,
                           s=60, alpha=0.8, edgecolors='black', linewidth=0.5)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel("Mean Δ top-1 energy ratio (sigma change)", fontsize=11)
        ax.set_ylabel("Δ downstream metric (vs baseline)", fontsize=11)
        ax.set_title("P0-1: Sigma Change vs Downstream Effect", fontsize=12)
        ax.legend(fontsize=8, loc='best')

        # Correlation
        all_x = [p["delta_top1"] for p in scatter_points]
        all_y = [p["delta_metric"] for p in scatter_points]
        if len(all_x) >= 3:
            r, p_val = scipy_stats.pearsonr(all_x, all_y)
            ax.annotate(f"r = {r:.3f}, p = {p_val:.3f}", xy=(0.05, 0.95),
                        xycoords='axes fraction', fontsize=10, va='top')

        plt.tight_layout()
        plt.savefig(PLOT_DIR / "p0_1_sigma_vs_metric.pdf", bbox_inches="tight", dpi=150)
        plt.savefig(PLOT_DIR / "p0_1_sigma_vs_metric.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"[Plot] Saved p0_1_sigma_vs_metric.pdf")

    # ================================================================
    # CONSERVATIVE INTERPRETATION
    # ================================================================
    print("\n" + "=" * 90)
    print("CONSERVATIVE INTERPRETATION")
    print("=" * 90)

    if ri:
        if ri["delta_top1"] < -0.01:
            print("""
  1. random_index DECREASES spectral concentration (top-1 ratio drops).
     This is consistent with the de-specialization hypothesis.
     The adapter's dominant singular value is damped by random selection.

  2. However, the magnitude matters: if flatten does 10x more de-specialization
     but produces similar (or worse) downstream effects, then de-specialization
     is necessary but not sufficient — the editing process also matters.
""")
        elif ri["delta_top1"] > 0.01:
            print("""
  1. random_index INCREASES spectral concentration (top-1 ratio rises).
     This CONTRADICTS the simple de-specialization hypothesis.
     Random selection amplifies dominant singular values.

  2. If flatten (which forces de-specialization) produces DIFFERENT downstream
     effects, this confirms that the mechanism is not de-specialization.
     The L1 preservation + random perturbation creates a different effect.
""")
        else:
            print("""
  1. random_index causes NEGLIGIBLE change in spectral concentration.
     The editing is a small perturbation that mostly preserves the spectrum.

  2. This suggests the gain (if any) comes from a regularization-like effect
     rather than systematic restructuring of the spectrum.
""")

    print("  Key question for eval: Do control methods that match random_index's")
    print("  sigma fingerprint also match its downstream performance?")
    print("  If yes: the sigma pattern is predictive => the mechanism is spectral.")
    print("  If no: the sigma pattern is incidental => need to look elsewhere.")

    # Save results
    with open(OUT_ROOT / "sigma_analysis_summary.json", "w") as f:
        json.dump({
            "table_data": table_data,
            "method_summary": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in method_summary.items()},
            "scatter_points": scatter_points,
        }, f, indent=2)

    print(f"\n[Saved] {OUT_ROOT / 'sigma_analysis_summary.json'}")


if __name__ == "__main__":
    analyze()
