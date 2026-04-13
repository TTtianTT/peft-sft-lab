#!/usr/bin/env python3
"""
Plotting: Multi-seed stability visualization.

Generates:
- Summary table (mean ± std) as a heatmap
- Win-rate table
- Boxplot per (model, task)

Usage:
    python scripts/rebuttal/plot_stability.py \
        --results_jsonl outputs/rebuttal_exp/raw/multiseed/results.jsonl \
        --out_dir outputs/rebuttal_exp/plots/stability
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


def load_records(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def group_results(records):
    groups = defaultdict(list)
    for r in records:
        if r.get("error") or r.get("metric_value") is None:
            continue
        key = (r["base_model_dir"], r["task"], r["method"])
        groups[key].append(r["metric_value"])
    return groups


def plot_boxplots(groups, out_dir: Path):
    """One boxplot per (model, task) showing all methods side by side."""
    model_tasks = sorted(set((k[0], k[1]) for k in groups.keys()))

    for model, task in model_tasks:
        methods = sorted(set(k[2] for k in groups if k[0] == model and k[1] == task))
        data = [groups.get((model, task, m), []) for m in methods]

        fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), 5))
        bp = ax.boxplot(data, labels=methods, patch_artist=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title(f"{model} / {task}", fontsize=12)
        ax.set_ylabel("Metric")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()

        fname = f"boxplot_{model}_{task}.pdf"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_summary_table(groups, out_dir: Path):
    """Heatmap-style table of mean ± std."""
    import statistics

    models = sorted(set(k[0] for k in groups))
    tasks = sorted(set(k[1] for k in groups))
    methods = sorted(set(k[2] for k in groups))

    for model in models:
        n_methods = len(methods)
        n_tasks = len(tasks)

        fig, ax = plt.subplots(figsize=(max(8, n_tasks * 2.5), max(4, n_methods * 0.6)))

        cell_text = []
        cell_colors = []
        for method in methods:
            row = []
            row_colors = []
            for task in tasks:
                vals = groups.get((model, task, method), [])
                if vals:
                    mean = statistics.mean(vals)
                    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    row.append(f"{mean:.4f}±{std:.4f}")
                    row_colors.append(mean)
                else:
                    row.append("N/A")
                    row_colors.append(0)
            cell_text.append(row)
            cell_colors.append(row_colors)

        # Normalize colors per task column
        arr = np.array(cell_colors)
        for j in range(arr.shape[1]):
            col = arr[:, j]
            vmin, vmax = col.min(), col.max()
            if vmax > vmin:
                arr[:, j] = (col - vmin) / (vmax - vmin)
            else:
                arr[:, j] = 0.5

        ax.set_axis_off()
        table = ax.table(
            cellText=cell_text,
            rowLabels=methods,
            colLabels=tasks,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Color cells
        cmap = plt.cm.RdYlGn
        for i in range(len(methods)):
            for j in range(len(tasks)):
                cell = table[i + 1, j]
                cell.set_facecolor(cmap(arr[i, j]))
                cell.set_alpha(0.6)

        ax.set_title(f"Mean±Std: {model}", fontsize=12, pad=20)
        plt.tight_layout()

        fname = f"summary_table_{model}.pdf"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_win_rate_table(groups, out_dir: Path):
    """Win rate table: P(method > baseline mean)."""
    import statistics

    models = sorted(set(k[0] for k in groups))
    tasks = sorted(set(k[1] for k in groups))
    methods = sorted(set(k[2] for k in groups if k[2] != "baseline"))

    for model in models:
        cell_text = []
        for method in methods:
            row = []
            for task in tasks:
                bl_vals = groups.get((model, task, "baseline"), [])
                method_vals = groups.get((model, task, method), [])
                if bl_vals and method_vals:
                    bl_mean = statistics.mean(bl_vals)
                    wins = sum(1 for v in method_vals if v > bl_mean)
                    row.append(f"{wins}/{len(method_vals)}")
                else:
                    row.append("N/A")
            cell_text.append(row)

        fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 2), max(3, len(methods) * 0.5)))
        ax.set_axis_off()
        table = ax.table(
            cellText=cell_text,
            rowLabels=methods,
            colLabels=tasks,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        ax.set_title(f"Win Rate vs Baseline: {model}", fontsize=12, pad=20)
        plt.tight_layout()

        fname = f"win_rate_{model}.pdf"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="Plot stability results")
    parser.add_argument("--results_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/rebuttal_exp/plots/stability")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.results_jsonl)
    print(f"Loaded {len(records)} records")

    groups = group_results(records)
    print(f"Groups: {len(groups)}")

    plot_boxplots(groups, out_dir)
    plot_summary_table(groups, out_dir)
    plot_win_rate_table(groups, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
