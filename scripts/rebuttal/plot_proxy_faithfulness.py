#!/usr/bin/env python3
"""
Plotting: Proxy faithfulness correlation plots.

Generates scatter plots of:
- Held-out proxy loss improvement vs final test metric
- Edit-set loss improvement vs final test metric
- Pearson/Spearman correlation coefficients

Usage:
    python scripts/rebuttal/plot_proxy_faithfulness.py \
        --proxy_results outputs/rebuttal_exp/raw/proxy_faithfulness/all_proxy_results.json \
        --eval_results outputs/rebuttal_exp/raw/multiseed/results.jsonl \
        --out_dir outputs/rebuttal_exp/plots/proxy_faithfulness
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot proxy faithfulness")
    parser.add_argument("--proxy_results", type=str, required=True)
    parser.add_argument("--eval_results", type=str, required=True,
                        help="JSONL with eval metrics from multiseed experiment")
    parser.add_argument("--out_dir", type=str, default="outputs/rebuttal_exp/plots/proxy_faithfulness")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.proxy_results) as f:
        proxy_data = json.load(f)

    eval_records = []
    with open(args.eval_results) as f:
        for line in f:
            if line.strip():
                eval_records.append(json.loads(line))

    # Merge: for each (model, task), get baseline proxy loss and eval metrics per method
    proxy_by_mt = {}
    for p in proxy_data:
        key = (p["base_model_dir"], p["task"])
        proxy_by_mt[key] = p

    eval_by_mt_method = defaultdict(list)
    for r in eval_records:
        if r.get("error") or r.get("metric_value") is None:
            continue
        key = (r["base_model_dir"], r["task"], r["method"])
        eval_by_mt_method[key].append(r["metric_value"])

    # Simple scatter: baseline edit_loss vs mean eval metric across tasks
    points_edit = []
    points_proxy = []
    labels = []

    for (model, task), proxy in proxy_by_mt.items():
        bl_edit_loss = proxy.get("baseline_edit_loss")
        bl_proxy_loss = proxy.get("baseline_proxy_loss")
        if bl_edit_loss is None:
            continue

        bl_eval = eval_by_mt_method.get((model, task, "baseline"), [])
        if not bl_eval:
            continue
        bl_eval_mean = np.mean(bl_eval)

        points_edit.append((bl_edit_loss, bl_eval_mean))
        points_proxy.append((bl_proxy_loss, bl_eval_mean))
        labels.append(f"{model[:10]}/{task}")

    if points_edit:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, points, title in [
            (axes[0], points_edit, "Edit-set Loss vs Eval Metric"),
            (axes[1], points_proxy, "Proxy (Held-out) Loss vs Eval Metric"),
        ]:
            x = [p[0] for p in points]
            y = [p[1] for p in points]

            ax.scatter(x, y, alpha=0.7, s=50)
            for i, label in enumerate(labels):
                ax.annotate(label, (x[i], y[i]), fontsize=7, alpha=0.7)

            ax.set_xlabel("Loss")
            ax.set_ylabel("Eval Metric")
            ax.set_title(title)

            # Compute correlation if enough points
            if len(x) > 2:
                from scipy import stats as sp_stats
                r_pearson, p_pearson = sp_stats.pearsonr(x, y)
                r_spearman, p_spearman = sp_stats.spearmanr(x, y)
                ax.text(0.05, 0.95,
                        f"Pearson r={r_pearson:.3f} (p={p_pearson:.3f})\n"
                        f"Spearman ρ={r_spearman:.3f} (p={p_spearman:.3f})",
                        transform=ax.transAxes, fontsize=8, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        fig.savefig(out_dir / "proxy_correlation.pdf", bbox_inches="tight")
        plt.close(fig)
        print("Saved proxy_correlation.pdf")
    else:
        print("Not enough data for proxy correlation plots")

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
