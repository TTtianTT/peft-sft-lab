#!/usr/bin/env python3
"""Aggregate diagonal 2-base x 4-task spectral-standardization scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("original_lora", "hns", "zscore_variance", "zscore_std")
TASKS = ("magicoder", "metamath", "tulu", "commonsense")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.run_root).resolve()
    rows = []
    for base in ("Qwen3-8B", "Llama-3.1-8B-Instruct"):
        score_manifest = json.loads((root / "eval" / base / "score_manifest.json").read_text())
        by_key = {(item["task"], item["variant"]): item for item in score_manifest["records"]}
        for task in TASKS:
            scores = {}
            for method in METHODS:
                label = f"{task}__{method}"
                item = by_key[(task, label)]
                metric = item["primary_metric"]
                scores[method] = float(item[metric])
            for method in METHODS:
                rows.append({
                    "base": base,
                    "task": task,
                    "method": method,
                    "primary_score": scores[method],
                    "gain_vs_lora": scores[method] - scores["original_lora"],
                    "difference_vs_hns": scores[method] - scores["hns"],
                })
    write_tsv(root / "summary.tsv", rows)
    payload = {"status": "complete", "rows": rows}
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
