#!/usr/bin/env python3
"""
Generate comprehensive rebuttal report from all experiment data.

Produces:
  1. Full 5-seed core sweep table (mean/std/median/min/max/win-rate)
  2. Guided-vs-random analysis (per-task + pooled, scatter plot, P(guided>random))
  3. Proxy-faithfulness correlations (Pearson/Spearman)
  4. Calibration-size sweep summary (if available)
  5. "Best rebuttal table" recommendation
  6. Unified CSV/JSON exports

Usage:
    python scripts/rebuttal/generate_final_report.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

EXP_ROOT = Path("outputs/rebuttal_exp")
EVAL_JSONL = EXP_ROOT / "raw" / "multiseed_eval" / "eval_results.jsonl"
PROXY_JSONL = EXP_ROOT / "raw" / "proxy_all_methods" / "proxy_results.jsonl"
CALIB_SWEEP_EVAL_JSONL = EXP_ROOT / "raw" / "calib_sweep_eval" / "eval_results.jsonl"

REPORT_DIR = EXP_ROOT / "notes"
TABLES_DIR = EXP_ROOT / "tables"
PLOTS_DIR = EXP_ROOT / "plots"
SUMMARIZED_DIR = EXP_ROOT / "summarized"

PRIORITY_SETTINGS = [
    ("Qwen-Qwen3-8B", "math"),
    ("Qwen-Qwen3-8B", "alpaca"),
    ("Qwen-Qwen3-8B", "csqa"),
    ("meta-llama-Llama-3.1-8B", "math"),
    ("meta-llama-Llama-3.1-8B", "alpaca"),
    ("meta-llama-Llama-3.1-8B", "csqa"),
]

CORE_METHODS = ["baseline", "random_index", "smooth_abs", "grad_direction",
                "continued_ft", "sigma_only"]

METHOD_DISPLAY = {
    "baseline": "Baseline (no edit)",
    "random_index": "Random Index",
    "smooth_abs": "Smooth Abs",
    "grad_direction": "Grad Direction",
    "continued_ft": "Continued FT",
    "sigma_only": "Sigma-Only Opt",
}


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def pearson_r(x, y):
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    mx, my = np.mean(x), np.mean(y)
    dx, dy = x - mx, y - my
    r = np.sum(dx * dy) / (np.sqrt(np.sum(dx**2)) * np.sqrt(np.sum(dy**2)) + 1e-12)
    # t-test for significance
    if abs(r) >= 1.0:
        return float(r), 0.0
    t = r * np.sqrt((n - 2) / (1 - r**2 + 1e-12))
    from scipy import stats as scipy_stats
    p = 2 * scipy_stats.t.sf(abs(t), n - 2)
    return float(r), float(p)


def spearman_rho(x, y):
    """Spearman rank correlation."""
    from scipy import stats as scipy_stats
    rho, p = scipy_stats.spearmanr(x, y)
    return float(rho), float(p)


# ============================================================================
# Section 1: Core sweep table
# ============================================================================

def build_core_sweep(eval_records: List[Dict]) -> Tuple[str, List[Dict]]:
    """Build full 5-seed core sweep table."""
    groups = defaultdict(list)
    for r in eval_records:
        if r.get("metric_value") is None:
            continue
        key = (r["base_model_dir"], r["task"], r["method"])
        groups[key].append(r["metric_value"])

    lines = ["## 1. Core Sweep: All Methods x All Settings x 5 Seeds\n"]
    rows = []

    for model, task in PRIORITY_SETTINGS:
        bl_vals = groups.get((model, task, "baseline"), [])
        bl_mean = statistics.mean(bl_vals) if bl_vals else None

        for method in CORE_METHODS:
            vals = groups.get((model, task, method), [])
            if not vals:
                continue
            n = len(vals)
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if n > 1 else 0.0
            median = statistics.median(vals)
            mn, mx = min(vals), max(vals)

            # Delta from baseline
            delta = (mean - bl_mean) if bl_mean is not None else None
            delta_pct = (delta / bl_mean * 100) if (bl_mean and delta is not None) else None

            # Win rate vs baseline mean
            if bl_mean is not None and method != "baseline":
                wins = sum(1 for v in vals if v >= bl_mean)
                win_rate = wins / n
            else:
                wins, win_rate = n, 1.0

            row = {
                "model": model, "task": task, "method": method,
                "n": n, "mean": mean, "std": std, "median": median,
                "min": mn, "max": mx,
                "delta": delta, "delta_pct": delta_pct,
                "win_rate": win_rate, "wins": wins,
            }
            rows.append(row)

    # Format table
    lines.append(f"| {'Model':<28s} | {'Task':<7s} | {'Method':<16s} | {'N':>2s} | "
                 f"{'Mean':>7s} | {'Std':>6s} | {'Med':>7s} | {'Min':>7s} | {'Max':>7s} | "
                 f"{'Delta':>7s} | {'D%':>6s} | {'WinR':>5s} |")
    lines.append("|" + "|".join(["---"] * 12) + "|")

    for row in rows:
        d = f"{row['delta']:+.4f}" if row['delta'] is not None else "—"
        dp = f"{row['delta_pct']:+.1f}%" if row['delta_pct'] is not None else "—"
        wr = f"{row['win_rate']:.0%}" if row['method'] != 'baseline' else "—"
        lines.append(
            f"| {row['model']:<28s} | {row['task']:<7s} | {row['method']:<16s} | "
            f"{row['n']:>2d} | {row['mean']:.4f} | {row['std']:.4f} | {row['median']:.4f} | "
            f"{row['min']:.4f} | {row['max']:.4f} | {d:>7s} | {dp:>6s} | {wr:>5s} |"
        )

    lines.append("")
    return "\n".join(lines), rows


# ============================================================================
# Section 2: Guided vs Random
# ============================================================================

def build_guided_vs_random(eval_records: List[Dict]) -> Tuple[str, List[Dict]]:
    """Guided vs random analysis with P(guided>random), per-task + pooled."""
    groups = defaultdict(list)
    for r in eval_records:
        if r.get("metric_value") is None:
            continue
        groups[(r["base_model_dir"], r["task"], r["method"])].append(
            (r["repeat_seed"], r["metric_value"])
        )

    lines = ["## 2. Guided vs Random Analysis\n"]
    lines.append("### Per-Setting Comparison\n")

    guided_methods = ["smooth_abs", "grad_direction"]
    random_method = "random_index"

    all_guided_deltas = []
    all_random_deltas = []
    comparison_rows = []

    for model, task in PRIORITY_SETTINGS:
        bl_vals = groups.get((model, task, "baseline"), [])
        rn_vals = groups.get((model, task, random_method), [])
        bl_mean = statistics.mean([v for _, v in bl_vals]) if bl_vals else None

        for gm in guided_methods:
            gd_vals = groups.get((model, task, gm), [])
            if not gd_vals or not rn_vals:
                continue

            # Match seeds where both guided and random have data
            gd_by_seed = {s: v for s, v in gd_vals}
            rn_by_seed = {s: v for s, v in rn_vals}
            common_seeds = sorted(set(gd_by_seed.keys()) & set(rn_by_seed.keys()))

            if not common_seeds:
                continue

            gd_matched = [gd_by_seed[s] for s in common_seeds]
            rn_matched = [rn_by_seed[s] for s in common_seeds]

            # Pairwise: guided > random?
            n_pairs = len(common_seeds)
            guided_wins = sum(1 for g, r in zip(gd_matched, rn_matched) if g > r)
            ties = sum(1 for g, r in zip(gd_matched, rn_matched) if g == r)
            p_gd_gt_rn = guided_wins / n_pairs if n_pairs > 0 else None

            # P(guided > baseline)
            if bl_mean is not None:
                p_gd_gt_bl = sum(1 for v in gd_matched if v > bl_mean) / len(gd_matched)
                p_rn_gt_bl = sum(1 for v in rn_matched if v > bl_mean) / len(rn_matched)
            else:
                p_gd_gt_bl = p_rn_gt_bl = None

            gd_mean = statistics.mean(gd_matched)
            rn_mean = statistics.mean(rn_matched)
            delta_gd = (gd_mean - bl_mean) if bl_mean else None
            delta_rn = (rn_mean - bl_mean) if bl_mean else None

            if delta_gd is not None:
                all_guided_deltas.append(delta_gd)
            if delta_rn is not None:
                all_random_deltas.append(delta_rn)

            comparison_rows.append({
                "model": model, "task": task, "guided_method": gm,
                "n_pairs": n_pairs,
                "guided_mean": gd_mean, "random_mean": rn_mean,
                "delta_guided": delta_gd, "delta_random": delta_rn,
                "P_guided_gt_random": p_gd_gt_rn,
                "P_guided_gt_baseline": p_gd_gt_bl,
                "P_random_gt_baseline": p_rn_gt_bl,
                "guided_wins": guided_wins, "ties": ties,
            })

    # Table
    lines.append(f"| {'Model':<28s} | {'Task':<7s} | {'Guided':<16s} | {'N':>2s} | "
                 f"{'G_mean':>7s} | {'R_mean':>7s} | {'ΔG':>7s} | {'ΔR':>7s} | "
                 f"{'P(G>R)':>7s} | {'P(G>BL)':>8s} | {'P(R>BL)':>8s} |")
    lines.append("|" + "|".join(["---"] * 11) + "|")

    for row in comparison_rows:
        dg = f"{row['delta_guided']:+.4f}" if row['delta_guided'] is not None else "—"
        dr = f"{row['delta_random']:+.4f}" if row['delta_random'] is not None else "—"
        pgr = f"{row['P_guided_gt_random']:.0%}" if row['P_guided_gt_random'] is not None else "—"
        pgb = f"{row['P_guided_gt_baseline']:.0%}" if row['P_guided_gt_baseline'] is not None else "—"
        prb = f"{row['P_random_gt_baseline']:.0%}" if row['P_random_gt_baseline'] is not None else "—"
        lines.append(
            f"| {row['model']:<28s} | {row['task']:<7s} | {row['guided_method']:<16s} | "
            f"{row['n_pairs']:>2d} | {row['guided_mean']:.4f} | {row['random_mean']:.4f} | "
            f"{dg:>7s} | {dr:>7s} | {pgr:>7s} | {pgb:>8s} | {prb:>8s} |"
        )

    # Pooled summary
    lines.append("")
    lines.append("### Pooled Summary\n")

    if all_guided_deltas and all_random_deltas:
        gd_pool_mean = statistics.mean(all_guided_deltas)
        rn_pool_mean = statistics.mean(all_random_deltas)
        # How often guided delta > random delta across settings
        n_settings = min(len(all_guided_deltas), len(all_random_deltas))
        gd_better = sum(1 for gd, rn in zip(all_guided_deltas, all_random_deltas) if gd > rn)
        lines.append(f"- Guided mean Δ (pooled): **{gd_pool_mean:+.4f}**")
        lines.append(f"- Random mean Δ (pooled): **{rn_pool_mean:+.4f}**")
        lines.append(f"- Settings where guided Δ > random Δ: **{gd_better}/{n_settings}**")
        lines.append(f"- Advantage (guided - random): **{gd_pool_mean - rn_pool_mean:+.4f}**")
    lines.append("")

    # Honest conclusion
    lines.append("### Interpretation\n")
    if comparison_rows:
        all_p = [r["P_guided_gt_random"] for r in comparison_rows if r["P_guided_gt_random"] is not None]
        mean_p = statistics.mean(all_p) if all_p else 0.5
        if mean_p > 0.7:
            lines.append("Gradient-guided selection shows a **consistent advantage** over random "
                         "index selection across settings.")
        elif mean_p > 0.55:
            lines.append("Gradient-guided selection shows a **modest advantage** over random "
                         "index selection. The benefit is setting-dependent and not always large.")
        else:
            lines.append("Gradient-guided selection shows **no clear advantage** over random "
                         "index selection. Both perform similarly, suggesting that *which* "
                         "singular values are modified matters less than the spectral editing "
                         "framework itself.")
    lines.append("")

    return "\n".join(lines), comparison_rows


# ============================================================================
# Section 3: Proxy faithfulness
# ============================================================================

def build_proxy_analysis(proxy_records: List[Dict], eval_records: List[Dict]) -> Tuple[str, Dict]:
    """Proxy faithfulness: correlations between loss improvements and test metric."""
    lines = ["## 3. Proxy Faithfulness Analysis\n"]

    # Get test metric for each (model, task, method) — seed=42 only
    test_by_key = {}
    for r in eval_records:
        if r.get("metric_value") is not None and r.get("repeat_seed") == 42:
            test_by_key[(r["base_model_dir"], r["task"], r["method"])] = r["metric_value"]

    # Build table: edit_loss, proxy_loss, test_metric for each method
    lines.append("### Loss and Metric Table (seed=42)\n")
    lines.append(f"| {'Model':<28s} | {'Task':<7s} | {'Method':<16s} | "
                 f"{'EditL':>8s} | {'ProxyL':>8s} | {'TestM':>8s} | "
                 f"{'ΔEditL':>8s} | {'ΔProxyL':>8s} | {'ΔTestM':>8s} |")
    lines.append("|" + "|".join(["---"] * 9) + "|")

    # Organize proxy data
    proxy_by_key = {}
    for r in proxy_records:
        if r.get("edit_loss") is not None:
            proxy_by_key[(r["base_model_dir"], r["task"], r["method"])] = r

    # Collect deltas for correlation
    delta_edit_list = []
    delta_proxy_list = []
    delta_test_list = []
    method_labels = []

    settings = sorted(set((r["base_model_dir"], r["task"]) for r in proxy_records))

    for model, task in settings:
        bl = proxy_by_key.get((model, task, "baseline"))
        bl_test = test_by_key.get((model, task, "baseline"))
        if not bl:
            continue

        for method in CORE_METHODS:
            p = proxy_by_key.get((model, task, method))
            t = test_by_key.get((model, task, method))
            if not p:
                continue

            el, pl = p["edit_loss"], p["proxy_loss"]
            bl_el, bl_pl = bl["edit_loss"], bl["proxy_loss"]

            # Loss decrease = improvement (flip sign for loss)
            d_edit = bl_el - el
            d_proxy = bl_pl - pl
            d_test = (t - bl_test) if (t is not None and bl_test is not None) else None

            fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
            fmtd = lambda v: f"{v:+.4f}" if v is not None else "N/A"

            lines.append(
                f"| {model:<28s} | {task:<7s} | {method:<16s} | "
                f"{fmt(el):>8s} | {fmt(pl):>8s} | {fmt(t):>8s} | "
                f"{fmtd(d_edit):>8s} | {fmtd(d_proxy):>8s} | {fmtd(d_test):>8s} |"
            )

            if d_test is not None and method != "baseline":
                delta_edit_list.append(d_edit)
                delta_proxy_list.append(d_proxy)
                delta_test_list.append(d_test)
                method_labels.append(f"{model[:5]}/{task}/{method[:8]}")

    lines.append("")

    # Correlations
    corr_results = {}
    if len(delta_test_list) >= 3:
        lines.append("### Correlations\n")
        de = np.array(delta_edit_list)
        dp = np.array(delta_proxy_list)
        dt = np.array(delta_test_list)

        try:
            r_edit, p_edit = pearson_r(de, dt)
            rho_edit, p_rho_edit = spearman_rho(de, dt)
            r_proxy, p_proxy = pearson_r(dp, dt)
            rho_proxy, p_rho_proxy = spearman_rho(dp, dt)

            lines.append("| Comparison | Pearson r | p-value | Spearman ρ | p-value |")
            lines.append("|---|---|---|---|---|")
            lines.append(f"| ΔEditLoss vs ΔTestMetric | {r_edit:.3f} | {p_edit:.4f} | "
                         f"{rho_edit:.3f} | {p_rho_edit:.4f} |")
            lines.append(f"| ΔProxyLoss vs ΔTestMetric | {r_proxy:.3f} | {p_proxy:.4f} | "
                         f"{rho_proxy:.3f} | {p_rho_proxy:.4f} |")
            lines.append(f"| ΔEditLoss vs ΔProxyLoss | "
                         f"{pearson_r(de, dp)[0]:.3f} | {pearson_r(de, dp)[1]:.4f} | "
                         f"{spearman_rho(de, dp)[0]:.3f} | {spearman_rho(de, dp)[1]:.4f} |")
            lines.append("")

            corr_results = {
                "pearson_edit_vs_test": {"r": r_edit, "p": p_edit},
                "spearman_edit_vs_test": {"rho": rho_edit, "p": p_rho_edit},
                "pearson_proxy_vs_test": {"r": r_proxy, "p": p_proxy},
                "spearman_proxy_vs_test": {"rho": rho_proxy, "p": p_rho_proxy},
                "n_datapoints": len(delta_test_list),
            }

            lines.append("### Interpretation\n")
            if abs(r_proxy) < 0.3 and abs(rho_proxy) < 0.3:
                lines.append("Proxy loss improvement is **not predictive** of downstream test "
                             "performance. This is expected: calibration loss is a local signal "
                             "that does not capture the full evaluation pipeline (generation, "
                             "task-specific scoring). The proxy split mainly confirms that "
                             "edits are not catastrophically overfitting.")
            elif abs(r_proxy) > 0.6:
                lines.append("Proxy loss shows **strong correlation** with test metric, "
                             "suggesting calibration loss is a reasonable surrogate for "
                             "this set of tasks.")
            else:
                lines.append("Proxy loss shows **moderate correlation** with test metric. "
                             "It provides some signal but should not be relied upon as a "
                             "sole proxy for downstream performance.")
        except Exception as e:
            lines.append(f"*Correlation computation failed: {e}*\n")

    lines.append("")
    return "\n".join(lines), corr_results


# ============================================================================
# Section 4: Continued FT analysis
# ============================================================================

def build_continued_ft_analysis(eval_records: List[Dict]) -> str:
    """Analyze continued_ft degradation pattern."""
    lines = ["## 4. Continued Fine-Tuning Analysis\n"]

    groups = defaultdict(list)
    for r in eval_records:
        if r.get("metric_value") is None:
            continue
        groups[(r["base_model_dir"], r["task"], r["method"])].append(r["metric_value"])

    lines.append("### Degradation Summary\n")
    lines.append("| Setting | BL Mean | CFT Mean | Delta | Delta% | Consistent? |")
    lines.append("|---|---|---|---|---|---|")

    all_deltas = []
    for model, task in PRIORITY_SETTINGS:
        bl = groups.get((model, task, "baseline"), [])
        cft = groups.get((model, task, "continued_ft"), [])
        if not bl or not cft:
            continue

        bl_m = statistics.mean(bl)
        cft_m = statistics.mean(cft)
        delta = cft_m - bl_m
        delta_pct = delta / bl_m * 100 if bl_m else 0
        all_deltas.append(delta_pct)

        # Is degradation consistent? (all seeds worse)
        consistent = all(v < bl_m for v in cft)
        c_str = "Yes" if consistent else "No"

        lines.append(f"| {model}/{task} | {bl_m:.4f} | {cft_m:.4f} | "
                     f"{delta:+.4f} | {delta_pct:+.1f}% | {c_str} |")

    if all_deltas:
        lines.append(f"\n**Overall**: mean degradation = {statistics.mean(all_deltas):+.1f}%, "
                     f"consistent across {sum(1 for d in all_deltas if d < 0)}/{len(all_deltas)} settings.")

    lines.append("")
    lines.append("### Why continued FT degrades\n")
    lines.append("Continued fine-tuning on the calibration subset causes classic **calibration "
                 "overfitting**: edit-set loss decreases sharply, but held-out proxy loss and "
                 "downstream test metrics degrade. The model memorizes calibration examples "
                 "rather than learning generalizable parameter adjustments. This supports the "
                 "core thesis: direct optimization of LoRA parameters on a small calibration "
                 "set is fundamentally limited, motivating the spectral (SVD-based) approach.\n")
    return "\n".join(lines)


# ============================================================================
# Section 5: Calibration-size sweep
# ============================================================================

def build_calib_sweep(calib_eval_records: List[Dict], core_eval_records: List[Dict]) -> Tuple[str, List[Dict]]:
    """Analyze effect of calibration size (Ncal) on performance."""
    lines = ["## 5. Calibration-Size Sweep (Ncal = 32, 64, 128, 256)\n"]

    if not calib_eval_records:
        lines.append("*No calibration sweep data available.*\n")
        return "\n".join(lines), []

    # Parse method_ncalXXX format
    groups = defaultdict(list)
    for r in calib_eval_records:
        if r.get("metric_value") is None:
            continue
        method_raw = r["method"]
        if "_ncal" not in method_raw:
            continue
        parts = method_raw.rsplit("_ncal", 1)
        method = parts[0]
        ncal = int(parts[1])
        key = (r["base_model_dir"], r["task"], method, ncal)
        groups[key].append(r["metric_value"])

    # Get baseline means from core eval
    bl_groups = defaultdict(list)
    for r in core_eval_records:
        if r.get("metric_value") is not None and r.get("method") == "baseline":
            bl_groups[(r["base_model_dir"], r["task"])].append(r["metric_value"])
    bl_means = {k: statistics.mean(v) for k, v in bl_groups.items()}

    settings = PRIORITY_SETTINGS
    methods_sweep = ["random_index", "smooth_abs", "grad_direction", "continued_ft", "sigma_only"]
    ncals = [32, 64, 128, 256]

    # Detailed table
    lines.append("### Per-Setting Results\n")
    lines.append(f"| {'Model':<28s} | {'Task':<7s} | {'Method':<16s} | {'Ncal':>4s} | "
                 f"{'N':>2s} | {'Mean':>7s} | {'Std':>6s} | {'Δ%':>7s} |")
    lines.append("|" + "|".join(["---"] * 8) + "|")

    sweep_rows = []
    for model, task in settings:
        bl_m = bl_means.get((model, task))
        for method in methods_sweep:
            for ncal in ncals:
                vals = groups.get((model, task, method, ncal), [])
                if not vals:
                    continue
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                delta_pct = ((mean - bl_m) / bl_m * 100) if bl_m else None
                dp = f"{delta_pct:+.1f}%" if delta_pct is not None else "—"

                row = {
                    "model": model, "task": task, "method": method,
                    "ncal": ncal, "n": len(vals), "mean": mean, "std": std,
                    "delta_pct": delta_pct,
                }
                sweep_rows.append(row)
                lines.append(
                    f"| {model:<28s} | {task:<7s} | {method:<16s} | {ncal:>4d} | "
                    f"{len(vals):>2d} | {mean:.4f} | {std:.4f} | {dp:>7s} |"
                )

    # Aggregated: mean delta% per method across Ncal values
    lines.append("")
    lines.append("### Aggregated: Mean Δ% by Method × Ncal\n")
    lines.append(f"| {'Method':<16s} |" + "|".join([f" {'Ncal='+str(n):>10s} " for n in ncals]) + f"| {'Range':>7s} |")
    lines.append("|" + "|".join(["---"] * (len(ncals) + 2)) + "|")

    for method in methods_sweep:
        parts = [f"{method:<16s}"]
        ncal_means = []
        for ncal in ncals:
            # Aggregate across settings
            deltas = [r["delta_pct"] for r in sweep_rows
                      if r["method"] == method and r["ncal"] == ncal and r["delta_pct"] is not None]
            if deltas:
                m = statistics.mean(deltas)
                ncal_means.append(m)
                parts.append(f"{m:+.2f}%")
            else:
                parts.append("—")
        rng = f"{max(ncal_means) - min(ncal_means):.2f}pp" if len(ncal_means) > 1 else "—"
        parts.append(rng)
        lines.append("| " + " | ".join(parts) + " |")

    # Interpretation
    lines.append("")
    lines.append("### Interpretation\n")

    # Check if performance varies significantly with Ncal
    method_ranges = {}
    for method in methods_sweep:
        ncal_avgs = []
        for ncal in ncals:
            deltas = [r["delta_pct"] for r in sweep_rows
                      if r["method"] == method and r["ncal"] == ncal and r["delta_pct"] is not None]
            if deltas:
                ncal_avgs.append(statistics.mean(deltas))
        if len(ncal_avgs) > 1:
            method_ranges[method] = max(ncal_avgs) - min(ncal_avgs)

    if method_ranges:
        max_range = max(method_ranges.values())
        mean_range = statistics.mean(method_ranges.values())
        if mean_range < 1.0:
            lines.append("Calibration size has **minimal effect** on spectral editing methods "
                         f"(mean range = {mean_range:.2f}pp across Ncal values). "
                         "This suggests the method is robust to calibration set size, "
                         "and even Ncal=32 provides sufficient signal for editing.")
        elif mean_range < 3.0:
            lines.append(f"Calibration size has a **moderate effect** (mean range = {mean_range:.2f}pp). "
                         "Larger calibration sets tend to provide slightly better results, "
                         "but the improvement is modest beyond Ncal=64.")
        else:
            lines.append(f"Calibration size has a **significant effect** (mean range = {mean_range:.2f}pp). "
                         "Larger calibration sets consistently improve results.")

    lines.append("")
    return "\n".join(lines), sweep_rows


# ============================================================================
# Section 6: Method ranking
# ============================================================================

def build_method_ranking(eval_records: List[Dict]) -> str:
    """Rank methods by aggregate performance."""
    lines = ["## 6. Method Ranking & Best Rebuttal Table\n"]

    groups = defaultdict(list)
    for r in eval_records:
        if r.get("metric_value") is None:
            continue
        groups[(r["base_model_dir"], r["task"], r["method"])].append(r["metric_value"])

    # Compute mean delta% for each method across settings
    method_deltas = defaultdict(list)
    for model, task in PRIORITY_SETTINGS:
        bl = groups.get((model, task, "baseline"), [])
        if not bl:
            continue
        bl_m = statistics.mean(bl)

        for method in CORE_METHODS:
            if method == "baseline":
                continue
            vals = groups.get((model, task, method), [])
            if not vals:
                continue
            delta_pct = (statistics.mean(vals) - bl_m) / bl_m * 100
            method_deltas[method].append(delta_pct)

    if method_deltas:
        lines.append("### Aggregate Δ% from Baseline (across settings)\n")
        lines.append("| Method | Mean Δ% | Std Δ% | Best | Worst | #Settings |")
        lines.append("|---|---|---|---|---|---|")

        ranked = sorted(method_deltas.items(),
                        key=lambda x: statistics.mean(x[1]), reverse=True)
        for method, deltas in ranked:
            m = statistics.mean(deltas)
            s = statistics.stdev(deltas) if len(deltas) > 1 else 0
            lines.append(f"| {method} | {m:+.2f}% | {s:.2f}% | "
                         f"{max(deltas):+.2f}% | {min(deltas):+.2f}% | {len(deltas)} |")

        lines.append("")

        # Best rebuttal table recommendation
        lines.append("### Recommended Rebuttal Table\n")
        lines.append("For the rebuttal, present the following comparison:\n")
        lines.append("1. **Baseline** (unedited LoRA adapter)")
        lines.append("2. **Continued FT** (show it degrades — motivates spectral approach)")

        best_spectral = ranked[0][0] if ranked[0][0] != "continued_ft" else (
            ranked[1][0] if len(ranked) > 1 else ranked[0][0])
        lines.append(f"3. **{best_spectral}** (best spectral method)")
        lines.append("4. **sigma_only** (optimization-based comparison)")
        lines.append(f"\nBest spectral method overall: **{best_spectral}** "
                     f"(mean Δ = {statistics.mean(method_deltas[best_spectral]):+.2f}%)\n")

    return "\n".join(lines)


# ============================================================================
# Scatter plot script
# ============================================================================

def write_scatter_plot_script():
    """Write a matplotlib script for guided-vs-random scatter plot."""
    script = '''#!/usr/bin/env python3
"""Generate guided-vs-random scatter plot."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_PATH = Path("outputs/rebuttal_exp/summarized/guided_vs_random.json")
OUT_PATH = Path("outputs/rebuttal_exp/plots/guided_vs_random_scatter.pdf")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    rows = json.load(f)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

for row in rows:
    dg = row.get("delta_guided")
    dr = row.get("delta_random")
    if dg is None or dr is None:
        continue
    label = f"{row['model'][:5]}/{row['task']}/{row['guided_method'][:6]}"
    ax.scatter(dr, dg, s=60, zorder=3)
    ax.annotate(label, (dr, dg), fontsize=6, ha="left", va="bottom")

# Diagonal
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1, label="y=x")
ax.set_xlabel("Δ Random Index (vs baseline)")
ax.set_ylabel("Δ Guided Method (vs baseline)")
ax.set_title("Guided vs Random: Per-Setting Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
'''
    path = PLOTS_DIR / "plot_guided_vs_random.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(script)
    return path


def write_calib_sweep_plot_script():
    """Write a matplotlib script for calibration-size line plot."""
    script = '''#!/usr/bin/env python3
"""Generate calibration-size sweep line plot."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

EVAL_PATH = Path("outputs/rebuttal_exp/raw/calib_sweep_eval/eval_results.jsonl")
OUT_PATH = Path("outputs/rebuttal_exp/plots/calib_sweep_lineplot.pdf")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if not EVAL_PATH.exists():
    print(f"No calib sweep eval data at {EVAL_PATH}")
    exit(0)

with open(EVAL_PATH) as f:
    records = [json.loads(l) for l in f if l.strip()]

# Parse method_ncalXXX format
groups = defaultdict(list)
for r in records:
    if r.get("metric_value") is None:
        continue
    method_raw = r["method"]
    if "_ncal" in method_raw:
        parts = method_raw.rsplit("_ncal", 1)
        method = parts[0]
        ncal = int(parts[1])
    else:
        continue
    key = (r["base_model_dir"], r["task"], method, ncal)
    groups[key].append(r["metric_value"])

# Aggregate
agg = {}
for (model, task, method, ncal), vals in groups.items():
    agg[(model, task, method, ncal)] = (np.mean(vals), np.std(vals) if len(vals) > 1 else 0)

# Get unique settings
settings = sorted(set((m, t) for m, t, _, _ in agg.keys()))
methods = sorted(set(meth for _, _, meth, _ in agg.keys()))
ncals = sorted(set(n for _, _, _, n in agg.keys()))

n_settings = len(settings)
fig, axes = plt.subplots(1, n_settings, figsize=(5 * n_settings, 4), sharey=False)
if n_settings == 1:
    axes = [axes]

for ax, (model, task) in zip(axes, settings):
    for method in methods:
        xs, ys, errs = [], [], []
        for ncal in ncals:
            if (model, task, method, ncal) in agg:
                m, s = agg[(model, task, method, ncal)]
                xs.append(ncal)
                ys.append(m)
                errs.append(s)
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker="o", label=method, capsize=3)
    ax.set_xlabel("Ncal")
    ax.set_ylabel("Test Metric")
    ax.set_title(f"{model[:12]}/{task}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle("Calibration-Size Sweep", y=1.02)
fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
'''
    path = PLOTS_DIR / "plot_calib_sweep.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(script)
    return path


# ============================================================================
# Export
# ============================================================================

def export_unified(eval_records, proxy_records, sweep_rows, gvr_rows, corr_results, calib_rows):
    """Export unified CSV and JSON."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIZED_DIR.mkdir(parents=True, exist_ok=True)

    # Unified eval results CSV
    if eval_records:
        csv_path = TABLES_DIR / "eval_results.csv"
        fields = ["base_model_dir", "task", "method", "repeat_seed", "metric_value"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in eval_records:
                if r.get("metric_value") is not None:
                    w.writerow(r)
        print(f"  {csv_path}")

    # Sweep summary
    if sweep_rows:
        csv_path = TABLES_DIR / "core_sweep_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"  {csv_path}")

    # Calib sweep summary
    if calib_rows:
        csv_path = TABLES_DIR / "calib_sweep_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=calib_rows[0].keys())
            w.writeheader()
            w.writerows(calib_rows)
        print(f"  {csv_path}")

    # Guided vs random
    if gvr_rows:
        with open(SUMMARIZED_DIR / "guided_vs_random.json", "w") as f:
            json.dump(gvr_rows, f, indent=2)
        print(f"  {SUMMARIZED_DIR / 'guided_vs_random.json'}")

    # Proxy correlations
    if corr_results:
        with open(SUMMARIZED_DIR / "proxy_correlations.json", "w") as f:
            json.dump(corr_results, f, indent=2)
        print(f"  {SUMMARIZED_DIR / 'proxy_correlations.json'}")

    # Full JSON dump
    all_data = {
        "core_sweep": sweep_rows,
        "guided_vs_random": gvr_rows,
        "proxy_correlations": corr_results,
        "calib_sweep": calib_rows,
    }
    with open(TABLES_DIR / "results.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"  {TABLES_DIR / 'results.json'}")


# ============================================================================
# Main
# ============================================================================

def main():
    for d in [REPORT_DIR, TABLES_DIR, PLOTS_DIR, SUMMARIZED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    eval_records = load_jsonl(EVAL_JSONL)
    proxy_records = load_jsonl(PROXY_JSONL)
    calib_sweep_records = load_jsonl(CALIB_SWEEP_EVAL_JSONL)

    print(f"  Eval records: {len(eval_records)}")
    print(f"  Proxy records: {len(proxy_records)}")
    print(f"  Calib sweep eval records: {len(calib_sweep_records)}")

    report_lines = ["# Spectral Surgery — Rebuttal Experiments Report\n"]
    report_lines.append(f"*Generated from {len(eval_records)} eval records, "
                        f"{len(proxy_records)} proxy records, "
                        f"{len(calib_sweep_records)} calib sweep records.*\n")

    # Section 1: Core sweep
    print("\n[1/6] Core sweep table...")
    sweep_text, sweep_rows = build_core_sweep(eval_records)
    report_lines.append(sweep_text)

    # Section 2: Guided vs random
    print("[2/6] Guided vs random...")
    gvr_text, gvr_rows = build_guided_vs_random(eval_records)
    report_lines.append(gvr_text)

    # Section 3: Proxy faithfulness
    print("[3/6] Proxy faithfulness...")
    proxy_text, corr_results = build_proxy_analysis(proxy_records, eval_records)
    report_lines.append(proxy_text)

    # Section 4: Continued FT
    print("[4/6] Continued FT analysis...")
    cft_text = build_continued_ft_analysis(eval_records)
    report_lines.append(cft_text)

    # Section 5: Calibration-size sweep
    print("[5/6] Calibration-size sweep...")
    calib_text, calib_rows = build_calib_sweep(calib_sweep_records, eval_records)
    report_lines.append(calib_text)

    # Section 6: Method ranking
    print("[6/6] Method ranking...")
    ranking_text = build_method_ranking(eval_records)
    report_lines.append(ranking_text)

    # Write report
    report_path = REPORT_DIR / "final_rebuttal_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport: {report_path}")

    # Export data
    print("\nExporting data...")
    export_unified(eval_records, proxy_records, sweep_rows, gvr_rows, corr_results, calib_rows)

    # Write plot scripts
    print("\nWriting plot scripts...")
    p1 = write_scatter_plot_script()
    p2 = write_calib_sweep_plot_script()
    print(f"  {p1}")
    print(f"  {p2}")

    # Repro commands
    print("\n" + "=" * 60)
    print("REPRODUCTION COMMANDS")
    print("=" * 60)
    print(f"# Generate plots:")
    print(f"python {p1}")
    print(f"python {p2}")
    print(f"\n# Re-generate this report:")
    print(f"python scripts/rebuttal/generate_final_report.py")


if __name__ == "__main__":
    main()
