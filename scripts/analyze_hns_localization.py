#!/usr/bin/env python3
"""Summarize HNS localization performance and Magicoder task-gradient scores."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    args = parser.parse_args()
    root = Path(args.run_root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    gradient_meta = json.loads(
        (root / "adapters" / "gradient-selected-hns" / "spectral_edit_meta.json").read_text(encoding="utf-8")
    )
    scores = gradient_meta["module_selection"]

    performance = sorted(
        summary,
        key=lambda row: (
            -(row["humaneval_pass@1"] if row["humaneval_pass@1"] is not None else -1),
            -(row["mbpp_pass@1"] if row["mbpp_pass@1"] is not None else -1),
        ),
    )
    performance_lines = ["rank\tlabel\thumaneval_pass@1\tmbpp_pass@1\tmean_pass@1"]
    for rank, row in enumerate(performance, start=1):
        human = row["humaneval_pass@1"]
        mbpp = row["mbpp_pass@1"]
        average = (human + mbpp) / 2 if human is not None and mbpp is not None else None
        performance_lines.append(f"{rank}\t{row['label']}\t{human}\t{mbpp}\t{average}")
    (root / "localization_performance.tsv").write_text(
        "\n".join(performance_lines) + "\n", encoding="utf-8"
    )

    suffix_groups: dict[str, list[dict]] = defaultdict(list)
    layer_groups: dict[int, list[dict]] = defaultdict(list)
    for prefix, score in scores.items():
        suffix_groups[prefix.rsplit(".", 1)[-1]].append(score)
        match = LAYER_RE.search(prefix)
        if match:
            layer_groups[int(match.group(1))].append(score)

    def aggregate(items: list[dict]) -> dict[str, float | int]:
        return {
            "modules": len(items),
            "mean_importance": mean([float(item["importance"]) for item in items]),
            "mean_compatibility": mean([float(item["compatibility"]) for item in items]),
            "positive_compatibility_fraction": mean(
                [float(item["compatibility"] > 0.0) for item in items]
            ),
            "mean_hns_risk": mean([float(item["hns_risk"]) for item in items]),
            "selected": sum(bool(item["selected"]) for item in items),
        }

    suffix_summary = {name: aggregate(items) for name, items in sorted(suffix_groups.items())}
    layer_summary = {str(layer): aggregate(items) for layer, items in sorted(layer_groups.items())}
    thirds = {
        "early_00_11": aggregate([item for layer, items in layer_groups.items() if layer < 12 for item in items]),
        "middle_12_23": aggregate([item for layer, items in layer_groups.items() if 12 <= layer < 24 for item in items]),
        "late_24_35": aggregate([item for layer, items in layer_groups.items() if layer >= 24 for item in items]),
    }
    analysis = {
        "gradient_summary": gradient_meta["summary"],
        "by_module_suffix": suffix_summary,
        "by_layer": layer_summary,
        "by_layer_third": thirds,
        "performance_ranking": performance,
    }
    (root / "hns_localization_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )

    suffix_lines = [
        "module\tmodules\tmean_importance\tmean_compatibility\tpositive_fraction\tmean_hns_risk\tselected"
    ]
    for name, row in sorted(suffix_summary.items(), key=lambda item: -float(item[1]["mean_compatibility"])):
        suffix_lines.append(
            f"{name}\t{row['modules']}\t{row['mean_importance']}\t{row['mean_compatibility']}\t"
            f"{row['positive_compatibility_fraction']}\t{row['mean_hns_risk']}\t{row['selected']}"
        )
    (root / "module_gradient_summary.tsv").write_text("\n".join(suffix_lines) + "\n", encoding="utf-8")

    layer_lines = [
        "layer\tmodules\tmean_importance\tmean_compatibility\tpositive_fraction\tmean_hns_risk\tselected"
    ]
    for layer, row in layer_summary.items():
        layer_lines.append(
            f"{layer}\t{row['modules']}\t{row['mean_importance']}\t{row['mean_compatibility']}\t"
            f"{row['positive_compatibility_fraction']}\t{row['mean_hns_risk']}\t{row['selected']}"
        )
    (root / "layer_gradient_summary.tsv").write_text("\n".join(layer_lines) + "\n", encoding="utf-8")
    print("\n".join(performance_lines))
    print("\n[Gradient by module]\n" + "\n".join(suffix_lines))
    print("\n[Gradient by layer third]\n" + json.dumps(thirds, indent=2))


if __name__ == "__main__":
    main()
