#!/usr/bin/env python3
"""Direct paired comparison of HeadOnly against HeadOnly+FrobeniusRestore."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_functional_localization_inference import binary_compare, ifeval_compare
from analyze_hns_paired_inference import (
    add_multiple_testing_adjustments,
    load_binary_categories,
    load_ifeval,
    two_sided_permutation_p,
    variant_roots,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first_batch_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--permutation_samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for (model, task), (kind, paths) in variant_roots(Path(args.first_batch_root)).items():
        if "head_fro_restore" not in paths:
            continue
        if kind == "tulu":
            left, right = load_ifeval(paths["head_only"]), load_ifeval(paths["head_fro_restore"])
            result = ifeval_compare(left, right, args.bootstrap_samples, args.permutation_samples, rng)
        else:
            left = load_binary_categories(kind, paths["head_only"])
            right = load_binary_categories(kind, paths["head_fro_restore"])
            result = binary_compare(left, right, args.bootstrap_samples, args.permutation_samples, rng)
        left_score, right_score, delta, boot, null, units = result
        rows.append({
            "base_model": model, "task": task,
            "left": "head_only", "right": "head_fro_restore",
            "head_only_score": left_score, "head_fro_restore_score": right_score,
            "delta_restore_minus_head": delta,
            "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
            "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
            "paired_permutation_p": two_sided_permutation_p(null, delta),
            "paired_units": units,
        })
    add_multiple_testing_adjustments(rows, "paired_permutation_p", "paired_permutation_p")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"comparisons": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
