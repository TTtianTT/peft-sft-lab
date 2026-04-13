#!/usr/bin/env python3
"""
Auto-generate a markdown report summarizing all rebuttal experiments.

Usage:
    python scripts/rebuttal/generate_rebuttal_report.py \
        --exp_root outputs/rebuttal_exp \
        --out_file outputs/rebuttal_exp/notes/rebuttal_report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=str, default="outputs/rebuttal_exp")
    parser.add_argument("--out_file", type=str, default="outputs/rebuttal_exp/notes/rebuttal_report.md")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Spectral Surgery Rebuttal Experiments Report\n"]

    # ---- Multiseed stability ----
    multiseed = load_jsonl(exp_root / "raw" / "multiseed" / "results.jsonl")
    if multiseed:
        lines.append("## 1. Multi-Seed Stability (P0-C)\n")

        groups = defaultdict(list)
        for r in multiseed:
            if r.get("error") or r.get("metric_value") is None:
                continue
            key = (r["base_model_dir"], r["task"], r["method"])
            groups[key].append(r["metric_value"])

        # Build table
        models = sorted(set(k[0] for k in groups))
        tasks = sorted(set(k[1] for k in groups))
        methods = sorted(set(k[2] for k in groups))

        for model in models:
            lines.append(f"### {model}\n")
            header = "| Method | " + " | ".join(tasks) + " |"
            sep = "|" + "|".join(["---"] * (len(tasks) + 1)) + "|"
            lines.append(header)
            lines.append(sep)

            for method in methods:
                row = f"| {method} |"
                for task in tasks:
                    vals = groups.get((model, task, method), [])
                    if vals:
                        m = statistics.mean(vals)
                        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
                        row += f" {m:.4f}±{s:.4f} |"
                    else:
                        row += " N/A |"
                lines.append(row)
            lines.append("")

        # Win rates
        lines.append("### Win Rates (vs baseline mean)\n")
        for model in models:
            lines.append(f"**{model}**\n")
            header = "| Method | " + " | ".join(tasks) + " |"
            sep = "|" + "|".join(["---"] * (len(tasks) + 1)) + "|"
            lines.append(header)
            lines.append(sep)
            for method in methods:
                if method == "baseline":
                    continue
                row = f"| {method} |"
                for task in tasks:
                    bl = groups.get((model, task, "baseline"), [])
                    mv = groups.get((model, task, method), [])
                    if bl and mv:
                        bl_mean = statistics.mean(bl)
                        wins = sum(1 for v in mv if v > bl_mean)
                        row += f" {wins}/{len(mv)} |"
                    else:
                        row += " N/A |"
                lines.append(row)
            lines.append("")

    # ---- Guided vs Random ----
    gvr_path = exp_root / "summarized" / "guided_vs_random" / "summary.json"
    if gvr_path.exists():
        lines.append("## 2. Guided vs Random (P1-D)\n")
        with open(gvr_path) as f:
            gvr = json.load(f)

        header = "| Model | Task | Method | P(G>R) | P(G>BL) | Δ_guided | Δ_random |"
        lines.append(header)
        lines.append("|---|---|---|---|---|---|---|")
        for s in gvr:
            dg = f"{s['mean_delta_guided']:.4f}" if s.get('mean_delta_guided') is not None else "N/A"
            dr = f"{s['mean_delta_random']:.4f}" if s.get('mean_delta_random') is not None else "N/A"
            lines.append(
                f"| {s['model']} | {s['task']} | {s['guided_method']} | "
                f"{s['P_guided_gt_random']:.0%} | {s['P_guided_gt_baseline']:.0%} | "
                f"{dg} | {dr} |"
            )
        lines.append("")

    # ---- Calib sweep ----
    sweep_path = exp_root / "raw" / "calib_sweep" / "calib_sweep_summary.json"
    if sweep_path.exists():
        lines.append("## 3. Calibration-Size Sweep (P1-F)\n")
        with open(sweep_path) as f:
            sweep = json.load(f)

        header = "| Model | Task | Method | Ncal | Mean | Std |"
        lines.append(header)
        lines.append("|---|---|---|---|---|---|")
        for s in sweep:
            lines.append(
                f"| {s['model']} | {s['task']} | {s['method']} | "
                f"{s['ncal']} | {s['mean']:.4f} | {s['std']:.4f} |"
            )
        lines.append("")

    # Write
    with open(out_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to {out_file}")


if __name__ == "__main__":
    main()
