#!/usr/bin/env python3
"""Aggregate diagonal 2x4 scores for norm-restored spectral standardization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = (
    "original_lora",
    "hns",
    "restore_lora_fro",
    "restore_lora_nuclear",
    "restore_hns_fro",
    "restore_hns_nuclear",
)
TASKS = ("magicoder", "metamath", "tulu", "commonsense")


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    rows = []
    for base in ("Qwen3-8B", "Llama-3.1-8B-Instruct"):
        score_manifest = json.loads((root / "eval" / base / "score_manifest.json").read_text())
        by_key = {(item["task"], item["variant"]): item for item in score_manifest["records"]}
        for task in TASKS:
            scores = {}
            for method in METHODS:
                item = by_key[(task, f"{task}__{method}")]
                scores[method] = float(item[item["primary_metric"]])
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
    (root / "summary.json").write_text(json.dumps({"status": "complete", "rows": rows}, indent=2) + "\n")
    print(f"[Done] wrote {len(rows)} rows to {root / 'summary.tsv'}")


if __name__ == "__main__":
    main()
