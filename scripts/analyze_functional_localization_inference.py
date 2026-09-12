#!/usr/bin/env python3
"""Paired bootstrap/randomization inference for functional localization edits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_hns_paired_inference import (
    bootstrap_binary_macro,
    bootstrap_ifeval,
    ifeval_arrays,
    ifeval_score,
    load_binary_categories,
    load_ifeval,
    permutation_binary_macro,
    permutation_ifeval,
    two_sided_permutation_p,
)


CASES = {
    "qwen_magicoder": (
        "magicoder",
        Path("/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-mechanism-20260908/eval/LoRA"),
    ),
    "qwen_commonsense": (
        "commonsense",
        Path("/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Qwen3-8B/commonsense/eval/lora"),
    ),
    "llama_tulu": (
        "tulu",
        Path("/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Llama-3.1-8B-Instruct/tulu/eval/lora"),
    ),
}


def binary_compare(left, right, boot_samples, perm_samples, rng):
    pairs = []
    for category in sorted(left):
        keys = sorted(left[category])
        if set(keys) != set(right[category]):
            raise RuntimeError(f"Paired keys differ for {category}")
        pairs.append((
            np.asarray([left[category][key] for key in keys], dtype=bool),
            np.asarray([right[category][key] for key in keys], dtype=bool),
        ))
    left_score = float(np.mean([a.mean() for a, _ in pairs]))
    right_score = float(np.mean([b.mean() for _, b in pairs]))
    delta = right_score - left_score
    boot = bootstrap_binary_macro(pairs, boot_samples, rng)
    null = permutation_binary_macro(pairs, perm_samples, rng)
    return left_score, right_score, delta, boot, null, sum(len(a) for a, _ in pairs)


def ifeval_compare(left_rows, right_rows, boot_samples, perm_samples, rng):
    if [str(row["key"]) for row in left_rows] != [str(row["key"]) for row in right_rows]:
        raise RuntimeError("IFEval prompt keys differ")
    left, right = ifeval_arrays(left_rows), ifeval_arrays(right_rows)
    left_score, right_score = ifeval_score(left), ifeval_score(right)
    delta = right_score - left_score
    boot = bootstrap_ifeval(left, right, boot_samples, rng)
    null = permutation_ifeval(left, right, perm_samples, rng)
    return left_score, right_score, delta, boot, null, len(left_rows)


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
        comparisons = [("lora", label) for label in roots if label != "lora"]
        comparisons.extend([
            (f"raw_top{pct}", f"functional_top{pct}") for pct in (25, 50)
        ])
        for left_label, right_label in comparisons:
            left_score, right_score, delta, boot, null, units = compare(
                data[left_label], data[right_label], args.bootstrap_samples,
                args.permutation_samples, rng,
            )
            rows.append({
                "case": case, "left": left_label, "right": right_label,
                "left_score": left_score, "right_score": right_score, "delta": delta,
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "paired_permutation_p": two_sided_permutation_p(null, delta),
                "paired_units": units,
            })
    with (output / "localization_paired_inference.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"comparisons": len(rows), "cases": list(CASES)}, indent=2))


if __name__ == "__main__":
    main()
