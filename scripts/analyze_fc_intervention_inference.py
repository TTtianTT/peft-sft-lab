#!/usr/bin/env python3
"""Paired inference for structure-matched functional x compatibility interventions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_functional_localization_inference import (
    CASES,
    binary_compare,
    ifeval_compare,
)
from analyze_hns_paired_inference import (
    add_multiple_testing_adjustments,
    load_binary_categories,
    load_ifeval,
    two_sided_permutation_p,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--permutation_samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root, output = Path(args.run_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for case, (kind, baseline_root) in CASES.items():
        manifest = json.loads((root / case / "adapters/manifest.json").read_text())
        roots = {row["label"]: root / case / "eval" / row["label"] for row in manifest["variants"]}
        roots["lora"] = baseline_root
        if kind == "tulu":
            data = {label: load_ifeval(path) for label, path in roots.items()}
            compare = ifeval_compare
        else:
            data = {label: load_binary_categories(kind, path) for label, path in roots.items()}
            compare = binary_compare

        comparisons = [("lora", label, "vs_lora") for label in roots if label != "lora"]
        for pct in (25, 50):
            comparisons.extend([
                (f"high_f_low_c_top{pct}", f"high_f_high_c_top{pct}", "C_at_high_F"),
                (f"low_f_low_c_top{pct}", f"low_f_high_c_top{pct}", "C_at_low_F"),
                (f"low_f_high_c_top{pct}", f"high_f_high_c_top{pct}", "F_at_high_C"),
                (f"low_f_low_c_top{pct}", f"high_f_low_c_top{pct}", "F_at_low_C"),
                (f"compatibility_top{pct}", f"high_f_high_c_top{pct}", "F_added_to_C"),
            ])
        for left_label, right_label, contrast in comparisons:
            left_score, right_score, delta, boot, null, units = compare(
                data[left_label], data[right_label], args.bootstrap_samples,
                args.permutation_samples, rng,
            )
            rows.append({
                "case": case, "contrast": contrast, "left": left_label, "right": right_label,
                "left_score": left_score, "right_score": right_score, "delta": delta,
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "paired_permutation_p": two_sided_permutation_p(null, delta),
                "paired_units": units,
            })
    add_multiple_testing_adjustments(rows, "paired_permutation_p", "paired_permutation_p")
    path = output / "fc_paired_inference.tsv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (output / "metadata.json").write_text(json.dumps({
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "seed": args.seed,
        "comparisons": len(rows),
        "multiple_testing_family": "all functional x compatibility comparisons",
    }, indent=2) + "\n")
    print(json.dumps({"comparisons": len(rows), "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
