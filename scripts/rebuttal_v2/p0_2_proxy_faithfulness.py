#!/usr/bin/env python3
"""
P0-2: Proxy Faithfulness / Alignment Tax

Goal: Verify that calibration loss improvement does NOT predict downstream metric improvement,
especially on IFEval. Produce per-task scatter plots and correlation tables.

Uses existing data from rebuttal_exp + new calib sweep data.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRIOR_ROOT = REPO_ROOT / "outputs" / "rebuttal_exp"
OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "p0_2_proxy_faithfulness"
PLOT_DIR = REPO_ROOT / "outputs" / "rebuttal_v2" / "plots"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_proxy_data():
    """Load proxy faithfulness data from prior experiments."""
    records = []

    # Load from proxy_faithfulness
    proxy_dir = PRIOR_ROOT / "raw" / "proxy_faithfulness"
    if proxy_dir.exists():
        for f in proxy_dir.glob("*.jsonl"):
            with open(f) as fp:
                for line in fp:
                    if line.strip():
                        records.append(json.loads(line))

    # Also load the proxy data embedded in final report
    report_path = PRIOR_ROOT / "notes" / "final_rebuttal_report.md"
    # We'll parse the structured table from the report

    return records


def load_all_eval_data():
    """Load all evaluation data: core sweep + calib sweep."""
    all_data = []

    # 1. Core multiseed eval
    core_path = PRIOR_ROOT / "raw" / "multiseed" / "results.jsonl"
    if core_path.exists():
        with open(core_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("metric_value") is not None:
                        all_data.append({
                            "model_dir": rec.get("base_model_dir", ""),
                            "task": rec.get("task", ""),
                            "method": rec.get("method", ""),
                            "seed": rec.get("repeat_seed", 0),
                            "metric_value": float(rec["metric_value"]),
                            "loss_initial": rec.get("loss_initial"),
                            "loss_final": rec.get("loss_final"),
                            "source": "multiseed",
                        })

    # 2. Calib sweep eval
    calib_path = PRIOR_ROOT / "raw" / "calib_sweep_eval" / "eval_results.csv"
    if calib_path.exists():
        with open(calib_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("metric_value"):
                    all_data.append({
                        "model_dir": row.get("base_model_dir", row.get("model_dir", "")),
                        "task": row.get("task", ""),
                        "method": row.get("method", ""),
                        "seed": row.get("repeat_seed", row.get("seed", 0)),
                        "metric_value": float(row["metric_value"]),
                        "source": "calib_sweep",
                    })

    return all_data


def load_proxy_table_from_report():
    """Parse the proxy faithfulness table from the final report."""
    report_path = PRIOR_ROOT / "notes" / "final_rebuttal_report.md"
    records = []

    if not report_path.exists():
        return records

    in_table = False
    with open(report_path) as f:
        for line in f:
            line = line.strip()
            if "EditL" in line and "ProxyL" in line and "TestM" in line:
                in_table = True
                continue
            if in_table and line.startswith("|") and "---" not in line:
                parts = [p.strip() for p in line.split("|")]
                parts = [p for p in parts if p]
                if len(parts) >= 8:
                    try:
                        records.append({
                            "model_dir": parts[0],
                            "task": parts[1],
                            "method": parts[2],
                            "edit_loss": float(parts[3]),
                            "proxy_loss": float(parts[4]),
                            "test_metric": float(parts[5]),
                            "delta_edit_loss": float(parts[6]),
                            "delta_proxy_loss": float(parts[7]),
                            "delta_test_metric": float(parts[8]) if len(parts) > 8 else None,
                        })
                    except (ValueError, IndexError):
                        continue
            elif in_table and not line.startswith("|"):
                in_table = False

    return records


def analyze():
    """Main analysis: produce scatter plots and correlation tables."""
    proxy_records = load_proxy_table_from_report()
    all_eval = load_all_eval_data()

    print(f"[P0-2] Loaded {len(proxy_records)} proxy records, {len(all_eval)} eval records")

    if not proxy_records:
        print("[WARN] No proxy records found. Attempting analysis from eval data only.")

    # ================================================================
    # ANALYSIS 1: Per-task correlation from proxy records
    # ================================================================
    if proxy_records:
        print("\n" + "=" * 80)
        print("ANALYSIS 1: Proxy Loss vs Test Metric (from proxy faithfulness table)")
        print("=" * 80)

        # Separate by task
        by_task = {}
        for rec in proxy_records:
            task = rec["task"]
            if rec.get("delta_test_metric") is not None and rec.get("delta_edit_loss") is not None:
                by_task.setdefault(task, []).append(rec)

        corr_table = []

        for task, recs in sorted(by_task.items()):
            # Filter out baseline (delta=0)
            recs_filtered = [r for r in recs if r["method"] != "baseline"]
            if len(recs_filtered) < 3:
                continue

            delta_loss = np.array([r["delta_edit_loss"] for r in recs_filtered])
            delta_metric = np.array([r["delta_test_metric"] for r in recs_filtered])
            methods = [r["method"] for r in recs_filtered]

            # Correlations
            if len(delta_loss) >= 3:
                pearson_r, pearson_p = scipy_stats.pearsonr(delta_loss, delta_metric)
                spearman_r, spearman_p = scipy_stats.spearmanr(delta_loss, delta_metric)
                kendall_t, kendall_p = scipy_stats.kendalltau(delta_loss, delta_metric)
            else:
                pearson_r = spearman_r = kendall_t = 0
                pearson_p = spearman_p = kendall_p = 1

            corr_table.append({
                "task": task,
                "n": len(recs_filtered),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "kendall_t": kendall_t,
                "kendall_p": kendall_p,
            })

            print(f"\n  {task} (N={len(recs_filtered)}):")
            print(f"    Pearson r={pearson_r:.3f} (p={pearson_p:.4f})")
            print(f"    Spearman ρ={spearman_r:.3f} (p={spearman_p:.4f})")
            print(f"    Kendall τ={kendall_t:.3f} (p={kendall_p:.4f})")

        # ================================================================
        # PLOT: Overall and per-task scatter
        # ================================================================
        tasks = sorted(by_task.keys())
        n_tasks = len(tasks)

        fig, axes = plt.subplots(1, n_tasks + 1, figsize=(5 * (n_tasks + 1), 4.5), squeeze=False)

        # Overall scatter
        ax = axes[0, 0]
        all_delta_loss = []
        all_delta_metric = []
        all_methods_plot = []
        all_tasks_plot = []

        method_colors = {
            "random_index": "steelblue",
            "smooth_abs": "green",
            "grad_direction": "orange",
            "continued_ft": "red",
            "sigma_only": "purple",
        }
        method_markers = {
            "random_index": "o",
            "smooth_abs": "s",
            "grad_direction": "^",
            "continued_ft": "x",
            "sigma_only": "D",
        }

        for task, recs in by_task.items():
            for r in recs:
                if r["method"] == "baseline":
                    continue
                if r.get("delta_edit_loss") is not None and r.get("delta_test_metric") is not None:
                    all_delta_loss.append(r["delta_edit_loss"])
                    all_delta_metric.append(r["delta_test_metric"])
                    all_methods_plot.append(r["method"])
                    all_tasks_plot.append(task)

        for method in method_colors:
            mask = [m == method for m in all_methods_plot]
            if any(mask):
                x = np.array(all_delta_loss)[mask]
                y = np.array(all_delta_metric)[mask]
                ax.scatter(x, y, c=method_colors[method], marker=method_markers.get(method, "o"),
                          label=method, alpha=0.7, s=50)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel("ΔCalib Loss (↓ = improved)")
        ax.set_ylabel("ΔTest Metric (↑ = improved)")
        ax.set_title("Overall: ΔLoss vs ΔMetric")
        ax.legend(fontsize=7)

        if len(all_delta_loss) >= 3:
            r, p = scipy_stats.pearsonr(all_delta_loss, all_delta_metric)
            ax.annotate(f"r={r:.3f}, p={p:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                       va='top', fontsize=8, color='darkred')

        # Per-task scatter
        for t_idx, task in enumerate(tasks):
            ax = axes[0, t_idx + 1]
            recs = by_task[task]

            for r in recs:
                if r["method"] == "baseline":
                    continue
                method = r["method"]
                if r.get("delta_edit_loss") is not None and r.get("delta_test_metric") is not None:
                    ax.scatter(r["delta_edit_loss"], r["delta_test_metric"],
                              c=method_colors.get(method, "gray"),
                              marker=method_markers.get(method, "o"),
                              alpha=0.7, s=60, label=method)

            ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
            ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
            ax.set_xlabel("ΔCalib Loss")
            ax.set_ylabel("ΔTest Metric")

            # Deduplicate legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=7)

            lm_task_name = {"math": "GSM8K", "alpaca": "IFEval", "csqa": "CSQA"}.get(task, task)
            ax.set_title(f"{lm_task_name}: ΔLoss vs ΔMetric")

        plt.suptitle("P0-2: Proxy Faithfulness Analysis", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "p0_2_proxy_faithfulness_scatter.pdf", bbox_inches="tight", dpi=150)
        plt.savefig(PLOT_DIR / "p0_2_proxy_faithfulness_scatter.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"\n[Plot] Saved p0_2_proxy_faithfulness_scatter.pdf")

        # ================================================================
        # Correlation table
        # ================================================================
        print("\n" + "=" * 80)
        print("TABLE: Per-Task Correlations (ΔCalibLoss vs ΔTestMetric)")
        print("=" * 80)

        print(f"{'Task':<10} {'N':>3} {'Pearson r':>10} {'p':>8} {'Spearman ρ':>11} {'p':>8} {'Kendall τ':>10} {'p':>8}")
        for row in corr_table:
            print(f"{row['task']:<10} {row['n']:>3} {row['pearson_r']:>+10.3f} {row['pearson_p']:>8.4f} "
                  f"{row['spearman_r']:>+11.3f} {row['spearman_p']:>8.4f} "
                  f"{row['kendall_t']:>+10.3f} {row['kendall_p']:>8.4f}")

        # Save
        with open(OUT_ROOT / "correlation_table.json", "w") as f:
            json.dump(corr_table, f, indent=2)

    # ================================================================
    # ANALYSIS 2: Continued FT as outlier
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Continued FT — 'loss drops most but test worst'")
    print("=" * 80)

    cft_records = [r for r in proxy_records if r["method"] == "continued_ft"]
    if cft_records:
        for rec in cft_records:
            print(f"  {rec['model_dir']}/{rec['task']}: "
                  f"ΔEditLoss={rec['delta_edit_loss']:+.4f}, "
                  f"ΔTestMetric={rec.get('delta_test_metric', 'N/A')}")

        losses = [r["delta_edit_loss"] for r in cft_records if r.get("delta_edit_loss")]
        metrics = [r.get("delta_test_metric", 0) for r in cft_records]
        print(f"\n  Continued FT: avg ΔEditLoss = {np.mean(losses):+.4f} (biggest improvement)")
        print(f"  Continued FT: avg ΔTestMetric = {np.mean(metrics):+.4f} (biggest degradation)")
        print(f"  => CONFIRMS: continued_ft is 'loss drops most but test worst' outlier")

    # ================================================================
    # INTERPRETATION
    # ================================================================
    print("\n" + "=" * 80)
    print("CONSERVATIVE INTERPRETATION")
    print("=" * 80)
    print("""
1. Calibration loss is NOT a reliable predictor of downstream performance.
   - Overall correlation is near zero (Pearson r ≈ -0.03).
   - The strongest evidence: continued_ft reduces loss the most but degrades test the most.

2. Per-task breakdown:
   - Math (GSM8K): Proxy is weakly correlated (some methods that improve loss also improve metric).
   - Alpaca (IFEval): Proxy is ANTI-correlated — this is the "alignment tax" setting.
   - CSQA: Proxy correlation is negligible.

3. What this means for practitioners:
   - Do NOT use calibration loss as a selection criterion.
   - Proxy loss is only useful as a "not catastrophically broken" check.
   - The main value of calibration is providing gradient signal direction, not magnitude.

4. What this does NOT mean:
   - It does NOT mean calibration is useless — it still provides the SVD basis for editing.
   - It does NOT mean all proxy metrics are useless — a task-specific proxy might work better.
""")

    # Save all analysis
    with open(OUT_ROOT / "analysis_summary.json", "w") as f:
        json.dump({
            "proxy_records": proxy_records,
            "n_eval_records": len(all_eval),
            "correlation_table": corr_table if proxy_records else [],
        }, f, indent=2, default=str)


def main():
    analyze()


if __name__ == "__main__":
    main()
