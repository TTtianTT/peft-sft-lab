#!/usr/bin/env python3
"""
P1-D: Guided vs Random Systematic Comparison.

Analyzes multi-seed results to compute:
- Per (model, task, Ncal, seed): delta_guided, delta_random, delta_diff
- P(guided > random) across seeds
- Pooled scatter plot data
- Summary tables

Usage:
    python scripts/rebuttal/analyze_guided_vs_random.py \
        --results_jsonl outputs/rebuttal_exp/raw/multiseed/results.jsonl \
        --out_dir outputs/rebuttal_exp/summarized/guided_vs_random
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import csv


def load_results(jsonl_path: str) -> List[Dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def analyze(records: List[Dict], guided_methods: List[str], random_method: str = "random_index"):
    """
    For each (model, task, seed), compute deltas relative to baseline.
    Then compute P(guided > random).
    """
    # Group by (model, task, method, seed)
    grouped = defaultdict(dict)  # (model, task, seed) -> {method: metric}
    for r in records:
        if r.get("error") or r.get("metric_value") is None:
            continue
        key = (r["base_model_dir"], r["task"], r["repeat_seed"])
        grouped[key][r["method"]] = r["metric_value"]

    comparisons = []
    for (model, task, seed), methods in grouped.items():
        baseline = methods.get("baseline")
        random_val = methods.get(random_method)

        for gm in guided_methods:
            guided_val = methods.get(gm)
            if guided_val is None:
                continue

            delta_guided = (guided_val - baseline) if baseline is not None else None
            delta_random = (random_val - baseline) if (random_val is not None and baseline is not None) else None

            comparisons.append({
                "model": model,
                "task": task,
                "seed": seed,
                "guided_method": gm,
                "baseline": baseline,
                "random": random_val,
                "guided": guided_val,
                "delta_guided": delta_guided,
                "delta_random": delta_random,
                "guided_better_than_random": (guided_val > random_val) if random_val is not None else None,
                "guided_better_than_baseline": (guided_val > baseline) if baseline is not None else None,
            })

    # Aggregate by (model, task, guided_method)
    agg_key = lambda c: (c["model"], c["task"], c["guided_method"])
    agg_groups = defaultdict(list)
    for c in comparisons:
        agg_groups[agg_key(c)].append(c)

    summary = []
    for (model, task, gm), items in agg_groups.items():
        n = len(items)
        wins_vs_random = sum(1 for c in items if c["guided_better_than_random"] is True)
        wins_vs_baseline = sum(1 for c in items if c["guided_better_than_baseline"] is True)
        n_valid_random = sum(1 for c in items if c["guided_better_than_random"] is not None)
        n_valid_baseline = sum(1 for c in items if c["guided_better_than_baseline"] is not None)

        delta_guided_vals = [c["delta_guided"] for c in items if c["delta_guided"] is not None]
        delta_random_vals = [c["delta_random"] for c in items if c["delta_random"] is not None]

        summary.append({
            "model": model,
            "task": task,
            "guided_method": gm,
            "n_seeds": n,
            "P_guided_gt_random": wins_vs_random / max(1, n_valid_random),
            "P_guided_gt_baseline": wins_vs_baseline / max(1, n_valid_baseline),
            "mean_delta_guided": statistics.mean(delta_guided_vals) if delta_guided_vals else None,
            "std_delta_guided": statistics.stdev(delta_guided_vals) if len(delta_guided_vals) > 1 else 0.0,
            "mean_delta_random": statistics.mean(delta_random_vals) if delta_random_vals else None,
            "std_delta_random": statistics.stdev(delta_random_vals) if len(delta_random_vals) > 1 else 0.0,
        })

    return comparisons, summary


def main():
    parser = argparse.ArgumentParser(description="Guided vs Random Analysis (P1-D)")
    parser.add_argument("--results_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/rebuttal_exp/summarized/guided_vs_random")
    parser.add_argument("--guided_methods", type=str, nargs="+",
                        default=["smooth_abs", "abs_select", "grad_direction"])
    parser.add_argument("--random_method", type=str, default="random_index")
    args = parser.parse_args()

    records = load_results(args.results_jsonl)
    print(f"Loaded {len(records)} records")

    comparisons, summary = analyze(records, args.guided_methods, args.random_method)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save comparisons
    with open(out_dir / "comparisons.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    # Save summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV summary
    if summary:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary[0].keys())
            w.writeheader()
            w.writerows(summary)

    # Print
    print(f"\n{'Model':<30} {'Task':<8} {'Method':<18} {'P(G>R)':>8} {'P(G>BL)':>8} "
          f"{'Δ_guided':>10} {'Δ_random':>10}")
    print("-" * 100)
    for s in sorted(summary, key=lambda x: (x["model"], x["task"], x["guided_method"])):
        dg = f"{s['mean_delta_guided']:.4f}" if s['mean_delta_guided'] is not None else "N/A"
        dr = f"{s['mean_delta_random']:.4f}" if s['mean_delta_random'] is not None else "N/A"
        print(f"{s['model']:<30} {s['task']:<8} {s['guided_method']:<18} "
              f"{s['P_guided_gt_random']:>8.2%} {s['P_guided_gt_baseline']:>8.2%} "
              f"{dg:>10} {dr:>10}")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
