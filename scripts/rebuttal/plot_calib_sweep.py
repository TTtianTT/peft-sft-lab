#!/usr/bin/env python3
"""
Plotting: Calibration-size sweep curves.

Generates performance vs Ncal curves per (model, task), one line per method.

Usage:
    python scripts/rebuttal/plot_calib_sweep.py \
        --summary_json outputs/rebuttal_exp/raw/calib_sweep/calib_sweep_summary.json \
        --out_dir outputs/rebuttal_exp/plots/calib_sweep
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "baseline": "black",
    "random_index": "gray",
    "smooth_abs": "tab:blue",
    "abs_select": "tab:orange",
    "grad_direction": "tab:green",
    "continued_ft": "tab:red",
    "sigma_only": "tab:purple",
}

METHOD_MARKERS = {
    "baseline": "s",
    "random_index": "x",
    "smooth_abs": "o",
    "abs_select": "^",
    "grad_direction": "D",
    "continued_ft": "v",
    "sigma_only": "P",
}


def main():
    parser = argparse.ArgumentParser(description="Plot calibration sweep curves")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/rebuttal_exp/plots/calib_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.summary_json) as f:
        summary = json.load(f)

    # Group by (model, task, method) -> list of (ncal, mean, std)
    curves = defaultdict(list)
    for s in summary:
        key = (s["model"], s["task"], s["method"])
        curves[key].append((s["ncal"], s["mean"], s["std"]))

    model_tasks = sorted(set((k[0], k[1]) for k in curves.keys()))

    for model, task in model_tasks:
        fig, ax = plt.subplots(figsize=(8, 5))

        methods_here = sorted(set(k[2] for k in curves if k[0] == model and k[1] == task))

        for method in methods_here:
            data = sorted(curves[(model, task, method)], key=lambda x: x[0])
            ncals = [d[0] for d in data]
            means = [d[1] for d in data]
            stds = [d[2] for d in data]

            color = METHOD_COLORS.get(method, "tab:gray")
            marker = METHOD_MARKERS.get(method, "o")

            ax.errorbar(ncals, means, yerr=stds, label=method,
                       color=color, marker=marker, capsize=3, linewidth=1.5, markersize=6)

        ax.set_xlabel("Ncal (calibration samples)")
        ax.set_ylabel("Metric")
        ax.set_title(f"{model} / {task}")
        ax.legend(fontsize=8, loc="best")
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"sweep_{model}_{task}.pdf"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
