#!/usr/bin/env python3
"""Compare observed set-level HNS NLL utility with sums of single-module utilities."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CASES = ("qwen_magicoder", "qwen_commonsense", "llama_tulu")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--module_utility_root", required=True)
    parser.add_argument("--functional_localization_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.run_root)
    utility_root = Path(args.module_utility_root)
    functional_root = Path(args.functional_localization_root)
    rng = np.random.default_rng(args.seed)
    rows = []

    for case in CASES:
        module_rows = read_tsv(utility_root / "utility" / case / "module_utility.tsv")
        modules = {row["module"] for row in module_rows}
        module_examples = read_tsv(utility_root / "utility" / case / "module_utility_examples.tsv")
        single: dict[tuple[str, int], float] = {
            (row["module"], int(row["sample_index"])): float(row["utility"])
            for row in module_examples
        }
        selected = {
            "full_hns": modules,
            "functional_top50": set(json.loads(
                (functional_root / case / "adapters/functional_top50/functional_localization_meta.json").read_text()
            )["selected_modules"]),
        }
        for label in ("compatibility_top50", "high_f_high_c_top50"):
            selected[label] = set(json.loads(
                (root / case / "adapters" / label / "selection_meta.json").read_text()
            )["selected_modules"])

        nll_examples = read_tsv(root / case / "nll" / "adapter_nll_examples.tsv")
        observed: dict[str, dict[int, float]] = defaultdict(dict)
        for row in nll_examples:
            if row["label"] != "lora":
                observed[row["label"]][int(row["sample_index"])] = float(row["utility"])

        for label, chosen in selected.items():
            if label not in observed:
                raise RuntimeError(f"missing observed NLL label {case}/{label}")
            sample_ids = sorted(observed[label])
            additive = np.asarray([
                sum(single[(module, sample)] for module in chosen)
                for sample in sample_ids
            ])
            actual = np.asarray([observed[label][sample] for sample in sample_ids])
            interaction = actual - additive
            indices = rng.integers(0, len(sample_ids), size=(args.bootstrap_samples, len(sample_ids)))
            interaction_boot = interaction[indices].mean(1)
            actual_boot = actual[indices].mean(1)
            additive_mean = float(additive.mean())
            rows.append({
                "case": case,
                "label": label,
                "selected_modules": len(chosen),
                "samples": len(sample_ids),
                "observed_set_utility": float(actual.mean()),
                "observed_ci_low": float(np.quantile(actual_boot, 0.025)),
                "observed_ci_high": float(np.quantile(actual_boot, 0.975)),
                "sum_single_module_utility": additive_mean,
                "interaction_observed_minus_additive": float(interaction.mean()),
                "interaction_ci_low": float(np.quantile(interaction_boot, 0.025)),
                "interaction_ci_high": float(np.quantile(interaction_boot, 0.975)),
                "observed_to_additive_ratio": float(actual.mean() / additive_mean) if abs(additive_mean) > 1e-12 else "",
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
