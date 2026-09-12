#!/usr/bin/env python3
"""Refresh aggregate benchmark point estimates from authoritative paired inference."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate_root", required=True)
    parser.add_argument("--paired_inference", required=True)
    args = parser.parse_args()
    root = Path(args.aggregate_root)
    paired_path = Path(args.paired_inference)

    paired_rows = [
        row for row in read_tsv(paired_path)
        if row["variant"] in {"scalar_shrink", "shape_only", "head_only", "tail_only", "hns"}
    ]
    lookup = {(row["base_model"], row["task"], row["variant"]): row for row in paired_rows}
    checkpoints = {(row["base_model"], row["task"]) for row in paired_rows}
    if len(lookup) != 40 or len(checkpoints) != 8:
        raise RuntimeError(f"Expected 40 variant rows over 8 checkpoints, got {len(lookup)}/{len(checkpoints)}")

    targets = ("summary.tsv", "cross_task_summary.tsv", "cross_base_task_comparison.tsv")
    for filename in targets:
        source = root / filename
        backup = root / filename.replace(".tsv", ".pre-paired-refresh.tsv")
        if not backup.exists():
            shutil.copy2(source, backup)

    summary = read_tsv(root / "summary.tsv")
    summary_changes = []
    for row in summary:
        key2 = (row["base_model"], row["task"])
        if key2 not in checkpoints:
            continue
        if row["label"] == "lora":
            source = lookup[key2 + ("hns",)]
            new_score, new_gain = float(source["lora_score"]), 0.0
        else:
            source = lookup[key2 + (row["label"],)]
            new_score, new_gain = float(source["variant_score"]), float(source["delta"])
        old_score, old_gain = float(row["primary_score"]), float(row["gain_vs_lora"])
        if abs(new_score - old_score) > 1e-12 or abs(new_gain - old_gain) > 1e-12:
            summary_changes.append({
                "base_model": row["base_model"], "task": row["task"], "variant": row["label"],
                "old_score": old_score, "new_score": new_score,
                "old_gain": old_gain, "new_gain": new_gain,
            })
        row["primary_score"], row["gain_vs_lora"] = new_score, new_gain
    write_tsv(root / "summary.tsv", summary)

    cross = read_tsv(root / "cross_task_summary.tsv")
    cross_changes = []
    for row in cross:
        key2 = (row["base_model"], row["task"])
        baseline = float(lookup[key2 + ("hns",)]["lora_score"])
        old_baseline = float(row["lora_score"])
        if abs(baseline - old_baseline) > 1e-12:
            cross_changes.append({
                "base_model": key2[0], "task": key2[1], "field": "lora_score",
                "old": old_baseline, "new": baseline,
            })
        row["lora_score"] = baseline
        for variant in ("scalar_shrink", "shape_only", "head_only", "tail_only", "hns"):
            field = f"{variant}_gain"
            old, new = float(row[field]), float(lookup[key2 + (variant,)]["delta"])
            if abs(new - old) > 1e-12:
                cross_changes.append({
                    "base_model": key2[0], "task": key2[1], "field": field,
                    "old": old, "new": new,
                })
            row[field] = new
    write_tsv(root / "cross_task_summary.tsv", cross)

    cross_lookup = {(row["base_model"], row["task"]): row for row in cross}
    comparison = read_tsv(root / "cross_base_task_comparison.tsv")
    for row in comparison:
        task = row["task"]
        qwen = cross_lookup[("Qwen3-8B", task)]
        llama = cross_lookup[("Llama-3.1-8B-Instruct", task)]
        row["qwen_hns_gain"], row["llama_hns_gain"] = qwen["hns_gain"], llama["hns_gain"]
        row["qwen_head_only_gain"], row["llama_head_only_gain"] = qwen["head_only_gain"], llama["head_only_gain"]
    write_tsv(root / "cross_base_task_comparison.tsv", comparison)

    manifest = {
        "method": "refresh aggregate benchmark scores from paired_inference.tsv",
        "paired_inference": str(paired_path.resolve()),
        "paired_variants": len(paired_rows),
        "summary_changes": summary_changes,
        "cross_task_changes": cross_changes,
        "note": "Spectral, functional, modification, and category source tables were not changed.",
    }
    (root / "paired_refresh_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"summary_changes": len(summary_changes), "cross_task_changes": len(cross_changes)}, indent=2))


if __name__ == "__main__":
    main()
