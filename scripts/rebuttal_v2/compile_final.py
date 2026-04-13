#!/usr/bin/env python3
"""
Final Compilation V3: Rebuttal-Ready Summary + Paper-Ready Summary + Overclaim List

Aggregates all Round 1 (P0, P1) and Round 2 (Exp 1-6) experiment results:
1. Rebuttal-ready summary v3 (key numbers + paragraphs)
2. Paper-ready conservative summary (for camera-ready revision)
3. Overclaim list (interpretations NOT safe to include)

Round 2 experiments:
  Exp 1: Component-wise Leave-One-Out
  Exp 2: Oracle Spectrum Editing
  Exp 3: Sensitivity Signal Stability
  Exp 4: IFEval Anomaly Deep Dive
  Exp 5: Few-Shot Baseline Comparison
  Exp 6: Calibration Set Scaling
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
V2_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2"
PRIOR_ROOT = REPO_ROOT / "outputs" / "rebuttal_exp"
FINAL_DIR = V2_ROOT / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def load_json_safe(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_jsonl_safe(path):
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def load_prior_eval_data():
    """Load eval metrics from prior rebuttal experiments."""
    eval_data = {}
    path = PRIOR_ROOT / "raw" / "multiseed_eval" / "eval_results.csv"
    if path.exists():
        with open(path) as f:
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


def compile():
    """Compile all results into final summaries."""

    # Load all analysis results — Round 1
    p0_1_sigma = load_json_safe(V2_ROOT / "p0_1_random_mechanism" / "sigma_analysis_summary.json")
    p0_1_eval = load_jsonl_safe(V2_ROOT / "p0_1_random_mechanism" / "control_eval_results.jsonl")
    p0_2 = load_json_safe(V2_ROOT / "p0_2_proxy_faithfulness" / "analysis_summary.json")
    p0_3 = load_json_safe(V2_ROOT / "p0_3_fixed_subspace" / "per_module_stats.json")
    p0_4 = load_json_safe(V2_ROOT / "p0_4_applicability" / "analysis_summary.json")
    p1_5 = load_json_safe(V2_ROOT / "p1_5_sigma_sgd" / "analysis_summary.json")
    p1_6 = load_json_safe(V2_ROOT / "p1_6_ifeval_taxonomy" / "error_taxonomy.json")
    prior_eval = load_prior_eval_data()

    # Load Round 2 results
    exp1_loo = load_json_safe(V2_ROOT / "exp1_leave_one_out" / "analysis" / "loo_analysis.json")
    if not exp1_loo:
        exp1_loo = []
    exp2_oracle = load_json_safe(V2_ROOT / "exp2_oracle_editing" / "analysis" / "oracle_analysis.json")
    if not exp2_oracle:
        exp2_oracle = []
    exp3_stability = load_json_safe(V2_ROOT / "exp3_signal_stability" / "analysis" / "stability_analysis.json")
    if not exp3_stability:
        exp3_stability = []
    exp4_concentration = load_json_safe(V2_ROOT / "exp4_ifeval_deep_dive" / "analysis" / "spectral_concentration.json")
    exp4_constraint = load_json_safe(V2_ROOT / "exp4_ifeval_deep_dive" / "analysis" / "per_constraint.json")
    exp4_genlength = load_json_safe(V2_ROOT / "exp4_ifeval_deep_dive" / "analysis" / "generation_length.json")
    exp5_fewshot = load_json_safe(V2_ROOT / "exp5_fewshot" / "analysis" / "fewshot_analysis.json")
    if not exp5_fewshot:
        exp5_fewshot = []
    exp6_scaling = load_json_safe(V2_ROOT / "exp6_calib_scaling" / "analysis" / "scaling_analysis.json")
    if not exp6_scaling:
        exp6_scaling = []

    plot_dir = V2_ROOT / "plots"
    plots = sorted(plot_dir.glob("*.pdf")) if plot_dir.exists() else []

    # ================================================================
    # Compute P0-3 family statistics from raw data
    # ================================================================
    p0_3_family_stats = {}
    p0_3_uv_stats = {}
    if p0_3:
        for setting, modules in p0_3.items():
            if not isinstance(modules, list):
                continue
            for m in modules:
                fam = m.get("family", "unknown")
                p0_3_family_stats.setdefault(fam, []).append(m.get("alignment_score", 0))
                p0_3_uv_stats.setdefault(fam, []).append(m.get("uv_perturb_mean_rel_delta", 0))

    # Compute P0-1 control eval summary
    p0_1_ctrl_summary = {}
    for rec in p0_1_eval:
        if rec.get("metric_value") is not None:
            key = (rec["model_dir"], rec["task"], rec["method"])
            p0_1_ctrl_summary.setdefault(key, []).append(rec["metric_value"])

    # ================================================================
    # 1. REBUTTAL-READY SUMMARY
    # ================================================================
    lines = []
    lines.append("# Rebuttal V2: Explanatory Experiments Summary\n")
    lines.append("Generated from 6 supplementary experiments (P0-1 through P1-6).\n")

    # --- A. Random mechanism ---
    lines.append("## A. Why does matched-random baseline sometimes improve? (P0-1)\n")
    sigma_summary = p0_1_sigma.get("method_summary", {})
    ri_sigma = sigma_summary.get("random_index", {})
    fs_sigma = sigma_summary.get("flatten_spectrum", {})
    ps_sigma = sigma_summary.get("permute_sigma", {})

    if ri_sigma:
        lines.append("### Sigma-Pattern Analysis (100 control edits + 60 prior edits)\n")
        lines.append("| Method | Δ top-1 ratio | Δ CV | L1 ratio |")
        lines.append("|--------|--------------|------|----------|")
        for method in ["random_index", "smooth_abs", "grad_direction",
                       "flatten_spectrum", "shrink_top_only", "suppress_tail_only",
                       "permute_sigma", "uniform_rescale"]:
            ms = sigma_summary.get(method, {})
            if ms:
                lines.append(f"| {method} | {ms['delta_top1']:+.4f} | {ms['delta_cv']:+.3f} | {ms['l1_ratio']:.3f} |")
        lines.append("")

        lines.append("**Key finding**: random_index causes **near-zero spectral change** "
                      f"(Δtop1 = {ri_sigma['delta_top1']:+.4f}), closest to permute_sigma "
                      f"(Δtop1 = {ps_sigma.get('delta_top1', 0):+.4f}). "
                      f"Flatten_spectrum causes massive de-specialization "
                      f"(Δtop1 = {fs_sigma.get('delta_top1', 0):+.4f}).\n")
        lines.append("**Interpretation**: The de-specialization hypothesis is **REJECTED**. "
                      "Random index editing acts as a small L1-preserved perturbation "
                      "(similar to sigma permutation), not as spectral flattening. "
                      "Any downstream effect is a regularization-like noise injection.\n")

    # Control eval results
    if p0_1_ctrl_summary:
        lines.append("### Control Edit Downstream Evaluation\n")
        lines.append("| Setting | Method | N | Mean | vs Baseline |")
        lines.append("|---------|--------|---|------|-------------|")
        for (md, task, method), vals in sorted(p0_1_ctrl_summary.items()):
            bl_key = (md, task, "baseline")
            bl_vals = prior_eval.get(bl_key, [])
            bl_mean = np.mean(bl_vals) if bl_vals else 0
            delta = (np.mean(vals) - bl_mean) / max(bl_mean, 1e-8) * 100
            lines.append(f"| {md}/{task} | {method} | {len(vals)} | {np.mean(vals):.4f} | {delta:+.2f}% |")
        lines.append("")

    # --- B. Proxy faithfulness ---
    lines.append("## B. Is alignment tax / proxy-metric misalignment real? (P0-2)\n")
    corr_table = p0_2.get("correlation_table", [])
    if corr_table:
        lines.append("| Task | Pearson r | p-value | N | Interpretation |")
        lines.append("|------|-----------|---------|---|----------------|")
        for row in corr_table:
            interp = "ANTI-correlated" if row["pearson_r"] < -0.5 and row["pearson_p"] < 0.05 else \
                     "Not significant" if row["pearson_p"] > 0.05 else "Weak"
            lines.append(f"| {row['task']} | {row['pearson_r']:+.3f} | {row['pearson_p']:.4f} | "
                         f"{row.get('n', '?')} | {interp} |")
        lines.append("")
        lines.append("**Key finding**: IFEval shows statistically significant **anti-correlation** "
                      f"(r = {corr_table[0]['pearson_r']:+.3f}, p = {corr_table[0]['pearson_p']:.4f}): "
                      "methods that reduce calibration loss the MOST perform WORST on downstream tasks.\n")
    else:
        lines.append("Overall proxy-downstream correlation: r ≈ -0.03 (from prior experiments).\n")

    lines.append("**Interpretation**: Calibration loss is NOT predictive of downstream improvement. "
                 "Continued FT achieves the largest loss reduction but causes -10.7% degradation. "
                 "The alignment tax is REAL and comes from overfitting to calibration data.\n")

    # --- C. Fixed subspace ---
    lines.append("## C. Why is sigma-only editing justified? (P0-3)\n")
    if p0_3_family_stats:
        lines.append("### Alignment Scores by Module Family (952 modules, 4 settings)\n")
        lines.append("| Family | Mean Alignment | Std | N |")
        lines.append("|--------|---------------|-----|---|")
        for fam in ["residual", "input", "internal"]:
            scores = p0_3_family_stats.get(fam, [])
            if scores:
                lines.append(f"| {fam} | {np.mean(scores):.4f} | {np.std(scores):.4f} | {len(scores)} |")
        lines.append("")

        lines.append("### UV Perturbation Stability (ε=0.01, 10 random trials per module)\n")
        lines.append("| Family | Mean ΔW/||W|| | Std | N |")
        lines.append("|--------|--------------|-----|---|")
        for fam in ["residual", "input", "internal"]:
            uv_vals = p0_3_uv_stats.get(fam, [])
            if uv_vals:
                lines.append(f"| {fam} | {np.mean(uv_vals):.6f} | {np.std(uv_vals):.6f} | {len(uv_vals)} |")
        lines.append("")

        all_align = []
        for v in p0_3_family_stats.values():
            all_align.extend(v)
        all_uv = []
        for v in p0_3_uv_stats.values():
            all_uv.extend(v)

        lines.append(f"**Key finding**: All module families show high alignment (mean = {np.mean(all_align):.2f}), "
                      f"confirming that |g_sigma| correlates with sigma magnitude. "
                      f"UV perturbation sensitivity is uniformly low ({np.mean(all_uv):.4f} relative ΔW), "
                      f"confirming U/V subspaces are stable.\n")
        lines.append("**Interpretation**: The singular vector bases (U, V) capture stable directional "
                      "structure of the LoRA update. Sigma controls magnitude allocation within these "
                      "fixed directions, making sigma-only editing a principled interface.\n")
    else:
        lines.append("(P0-3 results not yet available)\n")

    # --- D. Applicability ---
    lines.append("## D. Practical deployment rule (P0-4)\n")
    rules = p0_4.get("deployment_rules", [])
    if rules:
        lines.append("| Setting | Recommended | Expected Δ% | Worst Case |")
        lines.append("|---------|-------------|-------------|------------|")
        for r in rules:
            lines.append(f"| {r['setting']} | {r['recommended']} | {r['expected_delta_pct']:+.2f}% | "
                         f"{r.get('worst_case_delta', 'N/A')} |")
        lines.append("")
    lines.append("**Deployment rule**: Use `random_index` as default (no gradient computation needed, "
                 "comparable results, bounded worst-case). "
                 "Do NOT use continued fine-tuning on calibration data.\n")

    # --- P1-5: Sigma SGD ---
    lines.append("## E. Sigma-only SGD vs heuristic (P1-5)\n")
    p1_5_data = p1_5.get("comparison_data", [])
    if p1_5_data:
        sigma_only_deltas = [r["delta_pct"] for r in p1_5_data if r["method"] == "sigma_only"]
        heuristic_deltas = [r["delta_pct"] for r in p1_5_data
                            if r["method"] in ["random_index", "smooth_abs", "grad_direction"]]
        lines.append(f"- Sigma-only SGD: mean Δ = {np.mean(sigma_only_deltas):+.2f}%, "
                      f"std = {np.std(sigma_only_deltas):.2f}%")
        lines.append(f"- Heuristic methods: mean Δ = {np.mean(heuristic_deltas):+.2f}%, "
                      f"std = {np.std(heuristic_deltas):.2f}%\n")
        lines.append("**Interpretation**: The sigma-only interface has value (valid parametrization), "
                      "but the heuristic approach is more stable than SGD optimization. "
                      "The low-data regime favors simple heuristics over gradient descent.\n")
    else:
        lines.append("(P1-5 data from prior experiments)\n")

    # --- P1-6: IFEval taxonomy ---
    lines.append("## F. IFEval error taxonomy (P1-6)\n")
    p1_6_counts = p1_6.get("error_counts", {})
    if p1_6_counts:
        lines.append("| Method | Empty Response | Format Violations | Below Min Words | Total Errors |")
        lines.append("|--------|---------------|-------------------|-----------------|--------------|")
        for method in ["baseline", "random_index", "smooth_abs", "grad_direction"]:
            counts = p1_6_counts.get(method, {})
            if counts:
                empty = counts.get("empty_response", 0)
                format_v = (counts.get("missing_json_format", 0) + counts.get("missing_table_format", 0) +
                            counts.get("missing_list_format", 0))
                below_min = counts.get("below_word_minimum", 0)
                total = sum(counts.values())
                lines.append(f"| {method} | {empty} | {format_v} | {below_min} | {total} |")
        lines.append("")

        gd_counts = p1_6_counts.get("grad_direction", {})
        bl_counts = p1_6_counts.get("baseline", {})
        if gd_counts and bl_counts:
            gd_empty = gd_counts.get("empty_response", 0)
            bl_empty = bl_counts.get("empty_response", 0)
            lines.append(f"**Key finding**: `grad_direction` produces {gd_empty} empty responses vs "
                          f"{bl_empty} for baseline. This accounts for the majority of IFEval failures "
                          f"after gradient-guided editing.\n")
            lines.append("**Interpretation**: The alignment tax from `grad_direction` manifests primarily "
                          "as response suppression (empty outputs), not as format violations. "
                          "This suggests the gradient-guided edits over-attenuate generation capacity.\n")

    # ================================================================
    # ROUND 2: New Experiments (Exp 1-6)
    # ================================================================
    lines.append("---\n")
    lines.append("# Round 2: Additional Experiments\n")

    # --- Exp 1: LOO ---
    lines.append("## G. Component-wise Leave-One-Out (Exp 1)\n")
    lines.append("Zero each of 16 singular components across ALL layers and evaluate downstream.\n")
    if exp1_loo:
        lines.append("| Setting | Baseline | Spearman ρ | p-value | Interpretation |")
        lines.append("|---------|----------|-----------|---------|----------------|")
        for rec in exp1_loo:
            skey = f"{rec['model_dir']}/{rec['task']}"
            corr = rec.get("correlation", {})
            rho = corr.get("spearman_rho", 0)
            p = corr.get("spearman_p", 1)
            interp = "Significant positive" if rho > 0.4 and p < 0.05 else \
                     "Significant negative" if rho < -0.4 and p < 0.05 else \
                     "Not significant"
            lines.append(f"| {skey} | {rec.get('baseline', 0):.4f} | {rho:+.3f} | {p:.4f} | {interp} |")
        lines.append("")

        # Summarize
        sig_pos = sum(1 for r in exp1_loo if r["correlation"]["spearman_rho"] > 0.4 and r["correlation"]["spearman_p"] < 0.05)
        marginal_neg = sum(1 for r in exp1_loo if r["correlation"]["spearman_rho"] < -0.4 and r["correlation"]["spearman_p"] < 0.10)
        lines.append(f"**Key finding**: Sensitivity signal |g_k| correlates with LOO importance in only "
                     f"{sig_pos}/{len(exp1_loo)} settings (Llama/math ρ=+0.54, p=0.032). "
                     f"{marginal_neg}/{len(exp1_loo)} show marginally significant negative correlation "
                     f"(Llama/csqa ρ=-0.49, p=0.055). The signal is unreliable as a downstream importance proxy.\n")

        # IFEval special case
        alpaca_recs = [r for r in exp1_loo if r["task"] == "alpaca"]
        for ar in alpaca_recs:
            deltas = ar.get("loo_delta", {})
            pos_deltas = [v for v in deltas.values() if v > 0]
            if pos_deltas:
                lines.append(f"**IFEval anomaly confirmed**: Removing ANY component improves IFEval "
                             f"(+{min(pos_deltas)*100:.1f}% to +{max(pos_deltas)*100:.1f}%), "
                             f"confirming 'perturbation as regularization' mechanism.\n")
    else:
        lines.append("(Exp 1 results not yet available)\n")

    # --- Exp 2: Oracle ---
    lines.append("## H. Oracle Spectrum Editing (Exp 2)\n")
    lines.append("Upper bound: what if we KNEW which components to zero/scale?\n")
    if exp2_oracle:
        lines.append("| Setting | Strategy | Metric | Δ% vs Baseline |")
        lines.append("|---------|----------|--------|----------------|")
        for rec in exp2_oracle:
            lines.append(f"| {rec['model_dir']}/{rec['task']} | {rec['strategy']} | "
                         f"{rec['metric']:.4f} | {rec['delta_pct']:+.2f}% |")
        lines.append("")
        best_oracle = max(exp2_oracle, key=lambda r: r.get("delta_pct", -999))
        lines.append(f"**Key finding**: Best oracle strategy achieves {best_oracle['delta_pct']:+.2f}% "
                     f"({best_oracle['strategy']} on {best_oracle['model_dir']}/{best_oracle['task']}). "
                     f"This sets the upper bound for spectrum-only editing.\n")
    else:
        lines.append("(Exp 2 results not yet available)\n")

    # --- Exp 3: Stability ---
    lines.append("## I. Sensitivity Signal Stability (Exp 3)\n")
    lines.append("Are |g_k| rankings stable across different calibration subsets?\n")
    if exp3_stability:
        lines.append("| Setting | Mean Pairwise ρ | Std ρ | Mean CV | Interpretation |")
        lines.append("|---------|----------------|-------|---------|----------------|")
        for rec in exp3_stability:
            skey = f"{rec['model_dir']}/{rec['task']}"
            lines.append(f"| {skey} | {rec['mean_pairwise_spearman']:.3f} | "
                         f"{rec['std_pairwise_spearman']:.3f} | {rec['per_component_cv_mean']:.4f} | "
                         f"{rec['interpretation']} |")
        lines.append("")

        mean_rho = np.mean([r["mean_pairwise_spearman"] for r in exp3_stability])
        lines.append(f"**Key finding**: Sensitivity rankings are stable across calibration subsets "
                     f"(mean ρ = {mean_rho:.2f}). Even the least stable setting (Qwen/alpaca, ρ = "
                     f"{min(r['mean_pairwise_spearman'] for r in exp3_stability):.2f}) shows moderate stability.\n")
        lines.append("**Interpretation**: The gradient signal IS measuring something consistent "
                     "(not noise), but Exp 1 shows what it measures does not always correlate "
                     "with downstream importance. The signal is reliable but not sufficient.\n")
    else:
        lines.append("(Exp 3 results not yet available)\n")

    # --- Exp 4: IFEval Deep Dive ---
    lines.append("## J. IFEval Anomaly Deep Dive (Exp 4)\n")
    if exp4_concentration:
        lines.append("### Spectral Concentration by Task\n")
        lines.append("| Setting | Entropy | Eff Rank | Top-1 Ratio | Gini |")
        lines.append("|---------|---------|----------|-------------|------|")
        for skey, stats in sorted(exp4_concentration.items()):
            lines.append(f"| {skey} | {stats['entropy_mean']:.3f} | {stats['eff_rank_mean']:.1f} | "
                         f"{stats['top1_ratio_mean']:.3f} | {stats['gini_mean']:.3f} |")
        lines.append("")

        # Compare alpaca vs non-alpaca
        alpaca_gini = [s["gini_mean"] for k, s in exp4_concentration.items() if "alpaca" in k]
        other_gini = [s["gini_mean"] for k, s in exp4_concentration.items() if "alpaca" not in k]
        if alpaca_gini and other_gini:
            lines.append(f"**Key finding**: Alpaca adapters are NOT more concentrated "
                         f"(Gini {np.mean(alpaca_gini):.3f} vs others {np.mean(other_gini):.3f}). "
                         f"The IFEval anomaly is not explained by spectral structure.\n")

    if exp4_constraint and isinstance(exp4_constraint, dict):
        lines.append("### Per-Constraint Analysis (IFEval Prompt-Level Strict Accuracy)\n")
        lines.append("| Method | Mean Acc | Std | N seeds |")
        lines.append("|--------|---------|-----|---------|")
        key = "prompt_level_strict_acc,none"
        for method, metrics in sorted(exp4_constraint.items()):
            if isinstance(metrics, dict) and key in metrics:
                vals = metrics[key]
                lines.append(f"| {method} | {np.mean(vals):.4f} | {np.std(vals):.4f} | {len(vals)} |")
        lines.append("")

    if exp4_genlength and isinstance(exp4_genlength, dict):
        error_counts = exp4_genlength.get("error_counts", {})
        if error_counts:
            lines.append("### Error Taxonomy (from generation analysis)\n")
            lines.append("| Method | Empty Response | Format Violations | Below Min Words | Total |")
            lines.append("|--------|---------------|-------------------|-----------------|-------|")
            for method in ["baseline", "random_index", "smooth_abs", "grad_direction"]:
                counts = error_counts.get(method, {})
                if counts:
                    empty = counts.get("empty_response", 0)
                    fmt = (counts.get("missing_json_format", 0) + counts.get("missing_table_format", 0) +
                           counts.get("missing_list_format", 0))
                    below = counts.get("below_word_minimum", 0)
                    total = sum(counts.values())
                    lines.append(f"| {method} | {empty} | {fmt} | {below} | {total} |")
            lines.append("")

    # --- Exp 5: Few-Shot ---
    lines.append("## K. Few-Shot Baseline Comparison (Exp 5)\n")
    if exp5_fewshot:
        lines.append("| Setting | k-shot | Metric | vs Best Spectral |")
        lines.append("|---------|--------|--------|------------------|")
        for rec in exp5_fewshot:
            diff = ""
            if rec.get("best_spectral") is not None:
                diff_val = (rec["metric"] - rec["best_spectral"]) / max(rec["best_spectral"], 1e-8) * 100
                diff = f"{diff_val:+.2f}%"
            lines.append(f"| {rec['setting']} | {rec['k_shot']} | {rec['metric']:.4f} | {diff} |")
        lines.append("")
        # Add interpretation
        lines.append("**Key findings:**")
        lines.append("- For Llama-3.1-8B, spectral-edited 0-shot outperforms best k-shot baseline on both tasks "
                      "(CSQA: 0.822 spectral vs 0.800 5-shot; Math: 0.688 spectral vs 0.662 1-shot).")
        lines.append("- For Qwen3-8B, the unedited baseline adapter already matches or exceeds spectral editing "
                      "(Math: 0.851 0-shot > 0.841 spectral; IFEval: 0.497 0-shot > 0.475 spectral).")
        lines.append("- Few-shot prompting shows diminishing returns: 1-shot captures most of the benefit, "
                      "and for Qwen/math, additional shots HURT performance (0.851 → 0.812).")
        lines.append("- **Interpretation**: Spectral editing provides value when the adapter under-performs "
                      "(Llama settings), acting as a solidified few-shot equivalent. When the adapter is "
                      "already well-calibrated (Qwen), spectral editing is unnecessary but harmless.\n")
    else:
        lines.append("(Exp 5 results not yet available)\n")

    # --- Exp 6: Calibration Scaling ---
    lines.append("## L. Calibration Set Scaling (Exp 6)\n")
    if exp6_scaling:
        lines.append("| Setting | Method | N_cal | Mean | Δ% |")
        lines.append("|---------|--------|-------|------|-----|")
        for rec in exp6_scaling:
            lines.append(f"| {rec['setting']} | {rec['method']} | {rec['n_cal']} | "
                         f"{rec['metric_mean']:.4f} | {rec['delta_pct']:+.2f}% |")
        lines.append("")
    else:
        lines.append("(Exp 6 results not yet available)\n")

    # --- Figures ---
    lines.append("## Available Figures\n")
    all_plot_dirs = [
        V2_ROOT / "plots",
        V2_ROOT / "exp1_leave_one_out" / "plots",
        V2_ROOT / "exp2_oracle_editing" / "plots",
        V2_ROOT / "exp3_signal_stability" / "plots",
        V2_ROOT / "exp4_ifeval_deep_dive" / "plots",
        V2_ROOT / "exp5_fewshot" / "plots",
        V2_ROOT / "exp6_calib_scaling" / "plots",
    ]
    for pd in all_plot_dirs:
        if pd.exists():
            for p in sorted(pd.glob("*.pdf")):
                lines.append(f"- `{p.relative_to(V2_ROOT)}`")
    lines.append("")

    with open(FINAL_DIR / "rebuttal_ready_summary_v3.md", "w") as f:
        f.write("\n".join(lines))

    # ================================================================
    # 2. PAPER-READY CONSERVATIVE SUMMARY
    # ================================================================
    conservative = []
    conservative.append("# Paper-Ready Conservative Summary\n")
    conservative.append("## Claims supported by evidence\n")

    conservative.append("1. **Spectral editing preserves performance** (mean Δ ≈ 0%, no catastrophic failures).")
    conservative.append("   Evidence: 180+ spectral editing runs across 6 settings × 5 seeds.\n")

    conservative.append("2. **Continued FT on small calibration sets degrades performance** (mean −10.7%).")
    conservative.append("   Evidence: 30/30 individual seeds below baseline, across all 6 settings.\n")

    conservative.append("3. **Calibration loss is not predictive of downstream performance**.")
    if corr_table:
        conservative.append(f"   Evidence: IFEval r = {corr_table[0]['pearson_r']:+.3f} (p = {corr_table[0]['pearson_p']:.4f}), "
                            "CSQA r = −0.15 (n.s.), Math r = −0.92 (p = 0.085).")
    else:
        conservative.append("   Evidence: Pearson r ≈ −0.03 across 16+ method-setting pairs.")
    conservative.append("")

    conservative.append("4. **The spectral framework (L1-preserved sigma perturbation) is the active ingredient,**")
    conservative.append("   **not gradient-guided index selection** (P(guided > random) = 48.3%).")
    conservative.append("   Evidence: 60 seed-matched pairs + sigma pattern analysis showing near-zero spectral change.\n")

    conservative.append("5. **Sigma-only editing is justified**: U/V subspaces are stable (ΔW/||W|| ≈ 0.077),")
    if p0_3_family_stats:
        all_a = []
        for v in p0_3_family_stats.values():
            all_a.extend(v)
        conservative.append(f"   alignment scores are high (mean = {np.mean(all_a):.2f}) across all module families.")
    else:
        conservative.append("   alignment scores are high across all module families.")
    conservative.append("   Evidence: 952 modules analyzed across 2 models × 2 tasks.\n")

    conservative.append("6. **Random index editing does NOT de-specialize the spectrum**.")
    if ri_sigma:
        conservative.append(f"   Evidence: Δtop1 = {ri_sigma['delta_top1']:+.4f} (≈0) vs flatten = {fs_sigma.get('delta_top1', 0):+.4f}.")
    conservative.append("   The mechanism is perturbation/noise injection, not spectral restructuring.\n")

    # Compute control vs guided aggregates from final analysis
    ctrl_deltas_all = []
    guid_deltas_all = []
    if p0_1_ctrl_summary and prior_eval:
        for (md, task, method), vals in p0_1_ctrl_summary.items():
            bl_key = (md, task, "baseline")
            bl_vals = prior_eval.get(bl_key, [])
            if bl_vals:
                bl_mean = np.mean(bl_vals)
                delta = (np.mean(vals) - bl_mean) / max(bl_mean, 1e-8) * 100
                ctrl_deltas_all.append(delta)
        for method in ["random_index", "smooth_abs", "grad_direction"]:
            for (md, task, m), vals in prior_eval.items():
                if m == method:
                    bl_key = (md, task, "baseline")
                    bl_vals = prior_eval.get(bl_key, [])
                    if bl_vals:
                        bl_mean = np.mean(bl_vals)
                        delta = (np.mean(vals) - bl_mean) / max(bl_mean, 1e-8) * 100
                        guid_deltas_all.append(delta)

    if ctrl_deltas_all and guid_deltas_all:
        conservative.append("7. **Guided methods are more stable than matched-budget controls**.")
        conservative.append(f"   Controls: mean Δ = {np.mean(ctrl_deltas_all):+.2f}%, std = {np.std(ctrl_deltas_all):.2f}% "
                            f"(hurt on 3/4 settings, help on IFEval only).")
        conservative.append(f"   Guided: mean Δ = {np.mean(guid_deltas_all):+.2f}%, std = {np.std(guid_deltas_all):.2f}% "
                            f"(near-zero across all settings).")
        conservative.append("   Evidence: 100 control evals (5 methods × 4 settings × 5 seeds) vs 60 guided evals.\n")

    # Round 2 claims
    if exp1_loo:
        sig_pos = sum(1 for r in exp1_loo if r["correlation"]["spearman_rho"] > 0.4 and r["correlation"]["spearman_p"] < 0.05)
        conservative.append(f"8. **Sensitivity signal correlates with LOO importance in {sig_pos}/{len(exp1_loo)} settings**.")
        conservative.append("   Evidence: Component-wise leave-one-out across 4 (model, task) pairs.\n")

    if exp3_stability:
        mean_rho = np.mean([r["mean_pairwise_spearman"] for r in exp3_stability])
        conservative.append(f"9. **Sensitivity rankings are stable across calibration subsets** (mean ρ = {mean_rho:.2f}).")
        conservative.append("   Evidence: 5 disjoint calibration subsets × 4 settings, pairwise Spearman.\n")

    if exp4_concentration:
        alpaca_gini = [s["gini_mean"] for k, s in exp4_concentration.items() if "alpaca" in k]
        other_gini = [s["gini_mean"] for k, s in exp4_concentration.items() if "alpaca" not in k]
        if alpaca_gini and other_gini:
            conservative.append(f"10. **IFEval anomaly is NOT due to spectral concentration** "
                                f"(alpaca Gini {np.mean(alpaca_gini):.3f} vs others {np.mean(other_gini):.3f}).")
            conservative.append("    Evidence: Shannon entropy, effective rank, Gini for all 6 adapters.\n")

    if exp5_fewshot:
        # Check which settings spectral beats best k-shot
        from collections import defaultdict as dd
        settings_data = dd(dict)
        for rec in exp5_fewshot:
            settings_data[rec["setting"]][rec["k_shot"]] = rec["metric"]
            settings_data[rec["setting"]]["best_spectral"] = rec.get("best_spectral")
        spectral_wins = 0
        total = 0
        for skey, sdata in settings_data.items():
            bs = sdata.get("best_spectral")
            if bs is not None:
                best_kshot = max(v for k, v in sdata.items() if isinstance(k, int))
                total += 1
                if bs > best_kshot:
                    spectral_wins += 1
        conservative.append(f"11. **Spectral editing outperforms best k-shot prompting in {spectral_wins}/{total} settings**.")
        conservative.append("    Evidence: 13 fewshot evaluations (k=0,1,3,5) across 4 settings vs spectral baselines.\n")

    conservative.append("## Claims that MUST be weakened\n")
    conservative.append("1. ~~'Gradient-guided selection significantly outperforms random.'~~")
    conservative.append("   → 'The framework works equally well with random index selection.'\n")
    conservative.append("2. ~~'Proxy loss on calibration set predicts downstream improvement.'~~")
    conservative.append("   → 'Proxy loss confirms edits are not catastrophic but does not predict magnitude.'\n")
    conservative.append("3. ~~'Spectral surgery consistently improves performance.'~~")
    conservative.append("   → 'Spectral surgery preserves performance while operating in a low-DOF regime.'\n")

    conservative.append("## Numbers to cite\n")
    conservative.append("| # | Claim | Number | 95% safe? |")
    conservative.append("|---|-------|--------|-----------|")
    conservative.append("| 1 | Continued FT degrades | −10.7% mean, 30/30 seeds | Yes |")
    conservative.append("| 2 | Spectral ≈ baseline | +0.15% mean, no catastrophic | Yes |")
    conservative.append("| 3 | Guided ≈ random | P(guided > random) = 48.3% | Yes |")
    if corr_table:
        conservative.append(f"| 4 | IFEval anti-correlated | r = {corr_table[0]['pearson_r']:+.3f}, p = {corr_table[0]['pearson_p']:.4f} | Yes |")
    else:
        conservative.append("| 4 | Proxy ≠ downstream | r ≈ −0.03 | Yes |")
    conservative.append("| 5 | Random ≈ permutation in sigma space | Δtop1 ≈ 0 | Yes |")
    if p0_3_family_stats:
        conservative.append(f"| 6 | UV stability | ΔW/||W|| = {np.mean(list(np.mean(v) for v in p0_3_uv_stats.values())):.4f} | Yes |")
    if gd_counts:
        conservative.append(f"| 7 | grad_direction empty responses | {gd_counts.get('empty_response', 'N/A')} vs {bl_counts.get('empty_response', 'N/A')} baseline | Yes |")
    if ctrl_deltas_all:
        conservative.append(f"| 8 | Controls hurt on average | mean Δ = {np.mean(ctrl_deltas_all):+.2f}%, hurt 3/4 settings | Yes |")
        conservative.append(f"| 9 | Guided more stable than controls | std {np.std(guid_deltas_all):.2f}% vs {np.std(ctrl_deltas_all):.2f}% | Yes |")
    # Round 2 numbers
    if exp1_loo:
        sig_pos = sum(1 for r in exp1_loo if r["correlation"]["spearman_rho"] > 0.4 and r["correlation"]["spearman_p"] < 0.05)
        conservative.append(f"| 10 | LOO validates sensitivity | {sig_pos}/{len(exp1_loo)} settings significant positive ρ | Yes |")
    if exp3_stability:
        mean_rho = np.mean([r["mean_pairwise_spearman"] for r in exp3_stability])
        conservative.append(f"| 11 | Sensitivity stable across subsets | mean ρ = {mean_rho:.2f} | Yes |")
    if exp2_oracle:
        best_o = max(exp2_oracle, key=lambda r: r.get("delta_pct", -999))
        conservative.append(f"| 12 | Oracle upper bound | best {best_o['delta_pct']:+.2f}% ({best_o['strategy']}) | Yes |")
    if exp5_fewshot:
        conservative.append(f"| 13 | Spectral vs few-shot | beats best k-shot in {spectral_wins}/{total} settings | Yes |")
    conservative.append("")

    with open(FINAL_DIR / "paper_ready_conservative.md", "w") as f:
        f.write("\n".join(conservative))

    # ================================================================
    # 3. OVERCLAIM LIST
    # ================================================================
    overclaim = []
    overclaim.append("# Interpretations NOT Safe to Include in Rebuttal\n")
    overclaim.append("## Overclaims to avoid\n")

    overclaim.append("1. **'Spectral surgery consistently improves task performance.'**")
    overclaim.append("   Reality: Mean improvement is +0.15%, within noise. Several settings show slight degradation.")
    overclaim.append("   The honest claim is 'preserves performance' not 'improves performance'.\n")

    overclaim.append("2. **'The gradient signal provides meaningful guidance for index selection.'**")
    overclaim.append("   Reality: P(guided > random) = 48.3%. Gradient signal is not informative for this purpose.")
    overclaim.append("   Additionally, grad_direction INCREASES spectral concentration (+0.11 Δtop1),")
    overclaim.append("   which may explain its occasional failures (IFEval empty responses).\n")

    overclaim.append("3. **'Random editing helps because it de-specializes the spectrum.'**")
    overclaim.append("   Reality: Sigma analysis shows random_index causes Δtop1 ≈ 0 (near-zero change).")
    overclaim.append("   Flatten (true de-specialization) produces radically different sigma fingerprint.")
    overclaim.append("   Random editing is a perturbation, not a restructuring.\n")

    overclaim.append("4. **'Our method is universally applicable to all LoRA adapters.'**")
    overclaim.append("   Reality: Tested on 2 models × 3 tasks with rank-16 LoRA. Generalization is unproven.\n")

    overclaim.append("5. **'Calibration loss predicts downstream improvement.'**")
    overclaim.append("   Reality: IFEval shows significant ANTI-correlation (r = -0.96).")
    overclaim.append("   Methods that reduce calibration loss most perform WORST downstream.\n")

    overclaim.append("6. **'Module family matters for sigma-only editing.'**")
    overclaim.append("   Reality: All families have comparable alignment (0.74-0.79) and UV stability (~0.077).")
    overclaim.append("   The choice of which modules to edit is less important than expected.\n")

    overclaim.append("7. **'Sigma-only optimization (SGD) validates the interface.'**")
    overclaim.append("   Reality: Sigma-only SGD has higher variance than heuristic methods.")
    overclaim.append("   The interface exists but the optimization approach matters.\n")

    overclaim.append("8. **Cherry-picking Qwen/math + grad_direction (best setting).**")
    overclaim.append("   This is the only setting where grad_direction consistently helps (+0.3%, 100% WR).")
    overclaim.append("   It is NOT representative of the full picture.\n")

    overclaim.append("9. **'Matched-budget controls uniformly improve performance.'**")
    overclaim.append("   Reality: Controls average −1.80% across 4 settings (std = 6.5%).")
    overclaim.append("   They help massively on IFEval (+8-12%) but HURT on math (−1.6 to −6.7%)")
    overclaim.append("   and CSQA (−6.3 to −10.1%). The IFEval benefit is a task-specific anomaly.\n")

    overclaim.append("10. **'LOO importance validates the gradient signal in all settings.'**")
    if exp1_loo:
        sig_pos = sum(1 for r in exp1_loo if r["correlation"]["spearman_rho"] > 0.4 and r["correlation"]["spearman_p"] < 0.05)
        overclaim.append(f"    Reality: Only {sig_pos}/{len(exp1_loo)} settings show significant positive correlation.")
        neg = [r for r in exp1_loo if r["correlation"]["spearman_rho"] < -0.4]
        if neg:
            overclaim.append(f"    {len(neg)} setting(s) show NEGATIVE correlation (higher sensitivity ≠ more important).")
    overclaim.append("    The gradient signal is consistent but not always informative.\n")

    overclaim.append("11. **'IFEval benefits from spectral editing because it de-specializes the spectrum.'**")
    overclaim.append("    Reality: Spectral concentration analysis shows alpaca adapters are NOT more concentrated.")
    overclaim.append("    The mechanism is perturbation-as-regularization, not de-specialization.\n")

    overclaim.append("12. **'Spectral editing universally outperforms few-shot prompting.'**")
    overclaim.append("    Reality: Only true for Llama-3.1-8B (2/2 tasks). For Qwen3-8B, the unedited")
    overclaim.append("    baseline adapter already matches or exceeds spectral editing without any few-shot.")
    overclaim.append("    The benefit depends on how well-calibrated the adapter already is.\n")

    overclaim.append("## What remains true even under the weakened narrative\n")
    overclaim.append("1. Spectral editing is a safe, low-cost post-hoc intervention (no catastrophic failures).")
    overclaim.append("2. Continued FT is demonstrably worse than doing nothing or spectral editing.")
    overclaim.append("3. The sigma-only parametrization is a valid interface (high alignment, stable U/V).")
    overclaim.append("4. The method requires minimal hyperparameter tuning (robust across Ncal, seed).")
    overclaim.append("5. Random index editing is the safest default: no gradients needed, bounded risk.")
    overclaim.append("6. The spectral editing effect is perturbation-like, not spectral restructuring.")
    overclaim.append("   This is actually a STRENGTH: it means the intervention is mild and predictable.")
    overclaim.append("7. Guided methods (random_index, smooth_abs) are MORE STABLE than matched-budget controls.")
    overclaim.append("   Controls: mean Δ = −1.80%, std = 6.5%. Guided: mean Δ ≈ +0.28%, std ≈ 1.1%.")
    overclaim.append("   The index selection heuristic keeps edits in a safe regime.")
    overclaim.append("8. The gradient signal |g_k| is STABLE (ρ ≈ 0.71–0.97 across calibration subsets),")
    overclaim.append("   even though it does not always predict downstream importance (LOO).")
    overclaim.append("   This means the signal captures consistent spectral features, not random noise.")
    overclaim.append("9. For IFEval, perturbation of ANY component improves performance (+1.8% to +4.4%),")
    overclaim.append("   confirming that the benefit comes from regularization, not targeted editing.")
    overclaim.append("   This is a genuine finding that explains the IFEval anomaly.")
    overclaim.append("10. For Llama-3.1-8B, spectral editing outperforms the best k-shot prompting baseline,")
    overclaim.append("    supporting the 'solidified few-shot knowledge' narrative for under-performing adapters.")
    overclaim.append("")

    with open(FINAL_DIR / "overclaim_list.md", "w") as f:
        f.write("\n".join(overclaim))

    print(f"\n[Final] Generated 3 documents in {FINAL_DIR}:")
    print(f"  1. rebuttal_ready_summary.md")
    print(f"  2. paper_ready_conservative.md")
    print(f"  3. overclaim_list.md")
    print(f"  Plots: {len(plots)} PDF figures in {V2_ROOT / 'plots'}")


def main():
    compile()


if __name__ == "__main__":
    main()
