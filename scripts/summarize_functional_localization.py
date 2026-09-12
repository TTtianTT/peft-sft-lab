#!/usr/bin/env python3
"""Summarize functional/raw/random HNS localization benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


BASELINE_ROOTS = {
    "qwen_magicoder": Path("/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-mechanism-20260908/eval/LoRA"),
    "qwen_commonsense": Path("/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Qwen3-8B/commonsense/eval/lora"),
    "llama_tulu": Path("/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909/Llama-3.1-8B-Instruct/tulu/eval/lora"),
}


def score(case: str, root: Path) -> tuple[float, dict[str, float]]:
    if case == "qwen_magicoder":
        human = json.loads((root / "humaneval/metrics.json").read_text())["pass@1"]
        mbpp = json.loads((root / "mbpp/metrics.json").read_text())["pass@1"]
        return (human + mbpp) / 2, {"humaneval": human, "mbpp": mbpp}
    if case == "qwen_commonsense":
        data = json.loads((root / "commonsense/summary.json").read_text())
        return float(data["macro_accuracy"]), {
            task: float(values["accuracy"]) for task, values in data["tasks"].items()
        }
    data = json.loads((root / "ifeval/metrics.json").read_text())
    endpoints = {
        "prompt_strict": float(data["prompt_level_strict_acc"]),
        "prompt_loose": float(data["prompt_level_loose_acc"]),
        "instruction_strict": float(data["inst_level_strict_acc"]),
        "instruction_loose": float(data["inst_level_loose_acc"]),
    }
    return sum(endpoints.values()) / len(endpoints), endpoints


def write_tsv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--case", required=True, choices=tuple(BASELINE_ROOTS))
    args = parser.parse_args()
    root = Path(args.run_root) / args.case
    manifest = json.loads((root / "adapters/manifest.json").read_text())
    baseline, baseline_categories = score(args.case, BASELINE_ROOTS[args.case])
    rows, category_rows = [], []
    for variant in manifest["variants"]:
        label = variant["label"]
        value, categories = score(args.case, root / "eval" / label)
        rows.append({
            "case": args.case,
            "label": label,
            "selection": variant["selection"],
            "scope_fraction": variant["scope_fraction_requested"],
            "selected_modules": variant["selected_count"],
            "seed": variant.get("seed", ""),
            "baseline_score": baseline,
            "score": value,
            "gain": value - baseline,
        })
        for category, category_value in categories.items():
            category_rows.append({
                "case": args.case,
                "label": label,
                "category": category,
                "baseline_score": baseline_categories[category],
                "score": category_value,
                "gain": category_value - baseline_categories[category],
            })
    write_tsv(root / "localization_summary.tsv", rows)
    write_tsv(root / "localization_categories.tsv", category_rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["selection"], row["scope_fraction"])].append(float(row["gain"]))
    group_rows = []
    for (selection, scope), gains in sorted(groups.items()):
        mean = sum(gains) / len(gains)
        variance = sum((value - mean) ** 2 for value in gains) / max(1, len(gains) - 1)
        group_rows.append({
            "case": args.case, "selection": selection, "scope_fraction": scope,
            "runs": len(gains), "mean_gain": mean,
            "sample_sd": math.sqrt(variance) if len(gains) > 1 else 0.0,
            "min_gain": min(gains), "max_gain": max(gains),
        })
    write_tsv(root / "localization_group_summary.tsv", group_rows)
    print(json.dumps({"case": args.case, "variants": len(rows), "baseline": baseline}, indent=2))


if __name__ == "__main__":
    main()
