#!/usr/bin/env python3
"""
P0-1 Analysis: Random Mechanism Decomposition

Reads results from P0-1 experiments and produces:
1. Control comparison table (mean/std/win-rate for each control)
2. Scatter: Δmetric vs Δspectral_entropy
3. Correlation analysis: random gain vs original spectral concentration
4. Conservative interpretation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "p0_1_random_mechanism"
PLOT_DIR = REPO_ROOT / "outputs" / "rebuttal_v2" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load edit results and eval results."""
    edit_results = []
    eval_results = []

    edit_path = OUT_ROOT / "results.jsonl"
    eval_path = OUT_ROOT / "eval_results.jsonl"

    if edit_path.exists():
        with open(edit_path) as f:
            for line in f:
                if line.strip():
                    edit_results.append(json.loads(line))

    if eval_path.exists():
        with open(eval_path) as f:
            for line in f:
                if line.strip():
                    eval_results.append(json.loads(line))

    return edit_results, eval_results


def load_baseline_metrics():
    """Load baseline metrics from prior rebuttal experiments."""
    baselines = {}
    prior_path = REPO_ROOT / "outputs" / "rebuttal_exp" / "raw" / "multiseed_eval" / "eval_results.csv"

    if prior_path.exists():
        import csv
        with open(prior_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("method") == "baseline":
                    key = (row["base_model_dir"], row["task"])
                    val = float(row["metric_value"]) if row.get("metric_value") else None
                    if val is not None:
                        baselines.setdefault(key, []).append(val)

    # Also try JSONL
    prior_jsonl = REPO_ROOT / "outputs" / "rebuttal_exp" / "raw" / "multiseed" / "results.jsonl"
    if prior_jsonl.exists():
        with open(prior_jsonl) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("method") == "baseline" and rec.get("metric_value") is not None:
                        key = (rec.get("base_model_dir", ""), rec.get("task", ""))
                        baselines.setdefault(key, []).append(float(rec["metric_value"]))

    # Average baselines
    return {k: np.mean(v) for k, v in baselines.items()}


def compute_spectral_stats(sigma):
    """Compute spectral statistics for a sigma vector."""
    sigma = np.array(sigma)
    sigma = sigma[sigma > 1e-12]
    if len(sigma) == 0:
        return {}

    total = sigma.sum()
    r = len(sigma)

    p = sigma / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    eff_rank = np.exp(entropy)

    sorted_s = np.sort(sigma)
    cum = np.cumsum(sorted_s)
    gini = 1 - 2 * np.sum(cum) / (r * total) + 1/r

    sorted_desc = np.sort(sigma)[::-1]
    top1_ratio = sorted_desc[0] / total
    top4_ratio = sorted_desc[:min(4, r)].sum() / total
    tail_mass = sorted_desc[r//2:].sum() / total

    return {
        "effective_rank": float(eff_rank),
        "spectral_entropy": float(entropy),
        "gini": float(gini),
        "top1_energy_ratio": float(top1_ratio),
        "top4_energy_ratio": float(top4_ratio),
        "tail_mass_ratio": float(tail_mass),
    }


def analyze_and_plot(edit_results, eval_results, baselines):
    """Run full analysis and generate plots + tables."""

    # Merge edit and eval results
    eval_map = {}
    for rec in eval_results:
        key = (rec["model_dir"], rec["task"], rec["method"], rec["seed"])
        eval_map[key] = rec.get("metric_value")

    # Organize by setting and method
    by_setting_method = {}
    for rec in edit_results:
        if rec.get("error"):
            continue
        key = (rec["model_dir"], rec["task"])
        method = rec["method"]
        seed = rec["seed"]

        eval_key = (rec["model_dir"], rec["task"], method, seed)
        metric = eval_map.get(eval_key)

        by_setting_method.setdefault(key, {}).setdefault(method, []).append({
            "seed": seed,
            "metric": metric,
            "meta": rec.get("meta"),
        })

    # ================================================================
    # TABLE 1: Control comparison (mean/std/win-rate)
    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 1: Control Method Comparison")
    print("=" * 80)

    all_methods = ["random_index", "flatten_spectrum", "shrink_top_only",
                   "suppress_tail_only", "permute_sigma", "uniform_rescale"]

    header = f"{'Setting':<35} {'Method':<20} {'N':>3} {'Mean':>8} {'Std':>8} {'Delta%':>8} {'WinRate':>8}"
    print(header)
    print("-" * len(header))

    table_rows = []

    for (model_dir, task), methods_data in sorted(by_setting_method.items()):
        bl_key = (model_dir, task)
        bl_mean = baselines.get(bl_key, None)

        for method in all_methods:
            runs = methods_data.get(method, [])
            metrics = [r["metric"] for r in runs if r["metric"] is not None]

            if not metrics:
                continue

            arr = np.array(metrics)
            mean_val = arr.mean()
            std_val = arr.std()

            if bl_mean is not None and bl_mean > 0:
                delta_pct = (mean_val - bl_mean) / bl_mean * 100
                win_rate = np.mean(arr > bl_mean) * 100
            else:
                delta_pct = 0
                win_rate = 0

            row = {
                "setting": f"{model_dir}/{task}",
                "method": method,
                "n": len(metrics),
                "mean": mean_val,
                "std": std_val,
                "delta_pct": delta_pct,
                "win_rate": win_rate,
            }
            table_rows.append(row)
            print(f"{row['setting']:<35} {method:<20} {row['n']:>3} {mean_val:>8.4f} {std_val:>8.4f} {delta_pct:>+7.2f}% {win_rate:>7.1f}%")

    # Save table as CSV
    if table_rows:
        import csv
        csv_path = OUT_ROOT / "control_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=table_rows[0].keys())
            w.writeheader()
            w.writerows(table_rows)
        print(f"\n[Saved] {csv_path}")

    # ================================================================
    # TABLE 2: Aggregated summary across settings
    # ================================================================
    print("\n" + "=" * 80)
    print("TABLE 2: Aggregated Method Summary")
    print("=" * 80)

    method_agg = {}
    for row in table_rows:
        method = row["method"]
        method_agg.setdefault(method, {"deltas": [], "win_rates": []})
        method_agg[method]["deltas"].append(row["delta_pct"])
        method_agg[method]["win_rates"].append(row["win_rate"])

    print(f"{'Method':<20} {'Mean Δ%':>8} {'Std Δ%':>8} {'Mean WR':>8} {'Min Δ%':>8} {'Max Δ%':>8}")
    for method in all_methods:
        agg = method_agg.get(method)
        if not agg:
            continue
        d = np.array(agg["deltas"])
        wr = np.array(agg["win_rates"])
        print(f"{method:<20} {d.mean():>+7.2f}% {d.std():>7.2f}% {wr.mean():>7.1f}% {d.min():>+7.2f}% {d.max():>+7.2f}%")

    # ================================================================
    # PLOT 1: Δmetric distribution for random_index (violin/box)
    # ================================================================
    fig, axes = plt.subplots(1, len(by_setting_method), figsize=(4*len(by_setting_method), 5), squeeze=False)

    for idx, ((model_dir, task), methods_data) in enumerate(sorted(by_setting_method.items())):
        ax = axes[0, idx]
        bl_mean = baselines.get((model_dir, task), 0)

        method_metrics = {}
        for method in all_methods:
            runs = methods_data.get(method, [])
            metrics = [r["metric"] for r in runs if r["metric"] is not None]
            if metrics:
                method_metrics[method] = np.array(metrics) - bl_mean

        if method_metrics:
            labels = list(method_metrics.keys())
            data = [method_metrics[m] for m in labels]
            bp = ax.boxplot(data, labels=[m[:12] for m in labels], patch_artist=True)
            colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        short_model = model_dir.split("-")[-1]
        ax.set_title(f"{short_model}/{task}", fontsize=10)
        ax.set_ylabel("Δ metric (vs baseline)")
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle("P0-1: Control Method Comparison (Δ from baseline)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "p0_1_control_comparison_boxplot.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOT_DIR / "p0_1_control_comparison_boxplot.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n[Plot] Saved p0_1_control_comparison_boxplot.pdf")

    # ================================================================
    # PLOT 2: Random seed gain distribution (histogram)
    # ================================================================
    fig, axes = plt.subplots(1, len(by_setting_method), figsize=(4*len(by_setting_method), 4), squeeze=False)

    for idx, ((model_dir, task), methods_data) in enumerate(sorted(by_setting_method.items())):
        ax = axes[0, idx]
        bl_mean = baselines.get((model_dir, task), 0)

        random_runs = methods_data.get("random_index", [])
        metrics = [r["metric"] for r in random_runs if r["metric"] is not None]

        if metrics:
            deltas = np.array(metrics) - bl_mean
            ax.hist(deltas, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', label='baseline')
            ax.axvline(x=np.mean(deltas), color='orange', linestyle='-', label=f'mean={np.mean(deltas):.4f}')
            ax.set_xlabel("Δ metric")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

            # Win rate annotation
            wr = np.mean(deltas > 0) * 100
            ax.annotate(f"WR={wr:.0f}%", xy=(0.95, 0.95), xycoords='axes fraction',
                        ha='right', va='top', fontsize=9)

        short_model = model_dir.split("-")[-1]
        ax.set_title(f"{short_model}/{task} (N={len(metrics)})", fontsize=10)

    plt.suptitle("P0-1: Random Index Gain Distribution (50 seeds)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "p0_1_random_gain_distribution.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOT_DIR / "p0_1_random_gain_distribution.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Plot] Saved p0_1_random_gain_distribution.pdf")

    # ================================================================
    # INTERPRETATION
    # ================================================================
    print("\n" + "=" * 80)
    print("CONSERVATIVE INTERPRETATION")
    print("=" * 80)

    # Check if flatten/shrink_top replicate random gain
    for (model_dir, task), methods_data in sorted(by_setting_method.items()):
        bl_mean = baselines.get((model_dir, task), 0)
        setting = f"{model_dir}/{task}"

        random_metrics = [r["metric"] for r in methods_data.get("random_index", []) if r["metric"] is not None]
        flatten_metrics = [r["metric"] for r in methods_data.get("flatten_spectrum", []) if r["metric"] is not None]
        shrink_metrics = [r["metric"] for r in methods_data.get("shrink_top_only", []) if r["metric"] is not None]

        if random_metrics and bl_mean:
            random_delta = np.mean(random_metrics) - bl_mean
            print(f"\n{setting}:")
            print(f"  random_index mean Δ: {random_delta:+.4f} ({random_delta/bl_mean*100:+.2f}%)")

            if flatten_metrics:
                flatten_delta = np.mean(flatten_metrics) - bl_mean
                print(f"  flatten_spectrum Δ: {flatten_delta:+.4f} ({flatten_delta/bl_mean*100:+.2f}%)")
                if abs(flatten_delta) > 0 and np.sign(flatten_delta) == np.sign(random_delta):
                    print(f"  -> Flatten REPLICATES direction of random gain => supports de-specialization story")
                else:
                    print(f"  -> Flatten does NOT replicate random gain direction")

            if shrink_metrics:
                shrink_delta = np.mean(shrink_metrics) - bl_mean
                print(f"  shrink_top_only Δ: {shrink_delta:+.4f} ({shrink_delta/bl_mean*100:+.2f}%)")

    print("\n[Summary]")
    print("If flatten/shrink_top reproduce random gain: supports 'over-specialization / de-specialization' hypothesis")
    print("If no clear pattern: more consistent with 'flat/non-unique basin' interpretation")
    print("If uniform_rescale ~= random_index: the L1-preservation itself is the active ingredient")

    # Save full analysis
    analysis = {
        "table_rows": table_rows,
        "method_agg": {m: {"mean_delta": np.mean(a["deltas"]), "std_delta": np.std(a["deltas"]),
                           "mean_winrate": np.mean(a["win_rates"])}
                       for m, a in method_agg.items()},
    }
    with open(OUT_ROOT / "analysis_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)


def main():
    edit_results, eval_results = load_results()
    baselines = load_baseline_metrics()

    if not eval_results:
        print("[WARN] No evaluation results found. Run eval phase first.")
        print(f"  Edit results: {len(edit_results)}")
        if edit_results:
            print("  Generating edit-only analysis...")
    if not baselines:
        print("[WARN] No baseline metrics found. Using hardcoded values from prior rebuttal.")
        baselines = {
            ("Qwen-Qwen3-8B", "math"): 0.8379,
            ("Qwen-Qwen3-8B", "alpaca"): 0.4606,
            ("meta-llama-Llama-3.1-8B", "math"): 0.6852,
            ("meta-llama-Llama-3.1-8B", "csqa"): 0.8244,
        }

    analyze_and_plot(edit_results, eval_results, baselines)


if __name__ == "__main__":
    main()
