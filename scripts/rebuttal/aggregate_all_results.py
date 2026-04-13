#!/usr/bin/env python3
"""
Aggregate all rebuttal experiment results into unified CSV/JSON.

Reads from:
- multiseed results
- calib sweep results
- proxy faithfulness results

Outputs unified tables for the rebuttal appendix.

Usage:
    python scripts/rebuttal/aggregate_all_results.py \
        --exp_root outputs/rebuttal_exp \
        --out_dir outputs/rebuttal_exp/summarized
"""

from __future__ import annotations

import argparse
import csv
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


def aggregate_group(values: List[float]) -> Dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": n,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if n > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate all rebuttal results")
    parser.add_argument("--exp_root", type=str, default="outputs/rebuttal_exp")
    parser.add_argument("--out_dir", type=str, default="outputs/rebuttal_exp/summarized")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Multiseed results ----
    multiseed = load_jsonl(exp_root / "raw" / "multiseed" / "results.jsonl")
    print(f"Multiseed records: {len(multiseed)}")

    # Group by (model, task, method)
    ms_groups = defaultdict(list)
    for r in multiseed:
        if r.get("error") or r.get("metric_value") is None:
            continue
        key = (r["base_model_dir"], r["task"], r["method"])
        ms_groups[key].append(r["metric_value"])

    ms_summary = []
    for (model, task, method), values in sorted(ms_groups.items()):
        agg = aggregate_group(values)
        ms_summary.append({
            "experiment": "multiseed",
            "model": model,
            "task": task,
            "method": method,
            **agg,
        })

    # ---- Calib sweep results ----
    sweep_records = []
    for ncal_dir in sorted(exp_root.glob("raw/calib_sweep/ncal_*")):
        sweep_records.extend(load_jsonl(ncal_dir / "results.jsonl"))
    print(f"Calib sweep records: {len(sweep_records)}")

    sw_groups = defaultdict(list)
    for r in sweep_records:
        if r.get("error") or r.get("metric_value") is None:
            continue
        key = (r["base_model_dir"], r["task"], r["method"], r["calib_samples"])
        sw_groups[key].append(r["metric_value"])

    sw_summary = []
    for (model, task, method, ncal), values in sorted(sw_groups.items()):
        agg = aggregate_group(values)
        sw_summary.append({
            "experiment": "calib_sweep",
            "model": model,
            "task": task,
            "method": method,
            "ncal": ncal,
            **agg,
        })

    # ---- Write unified outputs ----
    all_summary = ms_summary + sw_summary

    with open(out_dir / "unified_results.json", "w") as f:
        json.dump(all_summary, f, indent=2)

    if all_summary:
        # Get all keys
        all_keys = set()
        for s in all_summary:
            all_keys.update(s.keys())
        fieldnames = ["experiment", "model", "task", "method", "ncal",
                      "n", "mean", "std", "median", "min", "max"]
        fieldnames = [k for k in fieldnames if k in all_keys]

        with open(out_dir / "unified_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_summary)

    # ---- Print summary ----
    print(f"\n{'Experiment':<15} {'Model':<30} {'Task':<8} {'Method':<20} "
          f"{'N':>4} {'Mean':>8} {'Std':>8}")
    print("-" * 95)
    for s in all_summary[:50]:
        mean_s = f"{s['mean']:.4f}" if s['mean'] is not None else "N/A"
        std_s = f"{s['std']:.4f}" if s['std'] is not None else "N/A"
        ncal_s = f" N={s.get('ncal','')}" if s.get("ncal") else ""
        print(f"{s['experiment']:<15} {s['model']:<30} {s['task']:<8} "
              f"{s['method']:<20} {s['n']:>4} {mean_s:>8} {std_s:>8}{ncal_s}")

    if len(all_summary) > 50:
        print(f"  ... and {len(all_summary) - 50} more rows")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
