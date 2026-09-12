#!/usr/bin/env python3
"""Summarize diagonal full-evaluation results from the 2x4 HNS step grid."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


BASES = ("Qwen3-8B", "Llama-3.1-8B-Instruct")
TASKS = ("magicoder", "metamath", "tulu", "commonsense")
STEP_RE = re.compile(r"^(?P<task>[^_]+)__hns_f(?P<fast>\d+)_s(?P<stable>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_dir")
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def variant_info(task: str, variant: str) -> tuple[str, int | str, int | str]:
    if variant == "base":
        return "base", "", ""
    if variant == f"{task}__original_lora":
        return "original_lora", "", ""
    match = STEP_RE.match(variant)
    if match and match.group("task") == task:
        return "hns", int(match.group("fast")), int(match.group("stable"))
    return "other", "", ""


def display_name(row: dict) -> str:
    if row["method"] == "base":
        return "Base"
    if row["method"] == "original_lora":
        return "LoRA"
    if row["method"] == "hns":
        return f"HNS {row['fast_steps']}+{row['stable_steps']}"
    return row["variant"]


def main() -> None:
    args = parse_args()
    root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "summary"
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for base in BASES:
        score_path = root / "eval" / base / "score_manifest.json"
        if not score_path.is_file():
            print(f"[Missing] {score_path}")
            continue
        for record in json.loads(score_path.read_text())["records"]:
            task = record["task"]
            method, fast, stable = variant_info(task, record["variant"])
            if method == "other":
                continue
            metric = record["primary_metric"]
            rows.append({
                "base": base,
                "task": task,
                "variant": record["variant"],
                "method": method,
                "fast_steps": fast,
                "stable_steps": stable,
                "primary_metric": metric,
                "score": float(record[metric]),
                "samples": record["samples"],
            })

    lora_scores = {
        (row["base"], row["task"]): row["score"]
        for row in rows
        if row["method"] == "original_lora"
    }
    for row in rows:
        baseline = lora_scores.get((row["base"], row["task"]))
        row["delta_vs_lora_pp"] = "" if baseline is None else 100 * (row["score"] - baseline)
        row["display_name"] = display_name(row)
    rows.sort(key=lambda row: (row["base"], TASKS.index(row["task"]), row["display_name"]))
    fields = [
        "base", "task", "display_name", "variant", "method", "fast_steps", "stable_steps",
        "primary_metric", "score", "delta_vs_lora_pp", "samples",
    ]
    write_tsv(output / "summary.tsv", rows, fields)

    configurations = [(0, 0)] + [(f, s) for f in (2, 4, 8) for s in (0, 1, 2)]
    table_rows = []
    for base in BASES:
        for task in TASKS:
            selected = [row for row in rows if row["base"] == base and row["task"] == task]
            if not selected:
                continue
            by_name = {row["display_name"]: row["score"] for row in selected}
            hns_rows = [row for row in selected if row["method"] == "hns"]
            best = max(hns_rows, key=lambda row: row["score"])
            out = {
                "base": base,
                "task": task,
                "metric": selected[0]["primary_metric"],
                "Base": by_name.get("Base", ""),
                "LoRA": by_name.get("LoRA", ""),
            }
            for fast, stable in configurations:
                out[f"HNS_{fast}+{stable}"] = by_name.get(f"HNS {fast}+{stable}", "")
            out.update({
                "best_HNS": best["display_name"],
                "best_HNS_score": best["score"],
                "best_HNS_gain_pp": best["delta_vs_lora_pp"],
            })
            table_rows.append(out)
    table_fields = ["base", "task", "metric", "Base", "LoRA"] + [
        f"HNS_{fast}+{stable}" for fast, stable in configurations
    ] + ["best_HNS", "best_HNS_score", "best_HNS_gain_pp"]
    write_tsv(output / "main_table.tsv", table_rows, table_fields)

    aggregate = []
    for fast, stable in configurations:
        selected = [
            row for row in rows
            if row["method"] == "hns"
            and row["fast_steps"] == fast
            and row["stable_steps"] == stable
        ]
        if not selected:
            continue
        deltas = [float(row["delta_vs_lora_pp"]) for row in selected]
        aggregate.append({
            "configuration": f"{fast}+{stable}",
            "checkpoints": len(selected),
            "mean_gain_pp": sum(deltas) / len(deltas),
            "median_gain_pp": statistics.median(deltas),
            "wins": sum(value > 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
        })
    write_tsv(
        output / "step_aggregate.tsv",
        aggregate,
        ["configuration", "checkpoints", "mean_gain_pp", "median_gain_pp", "wins", "ties", "losses"],
    )

    spectral_rows = []
    for base in BASES:
        for task in TASKS:
            manifest_path = root / "adapters" / base / task / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            for item in manifest["variants"]:
                spectral_rows.append({
                    "base": base,
                    "task": task,
                    "configuration": f"{item['fast_steps']}+{item['stable_steps']}",
                    "fast_steps": item["fast_steps"],
                    "stable_steps": item["stable_steps"],
                    "modules": item["num_modules"],
                    "mean_effective_rank_before": item["mean_effective_rank_before"],
                    "mean_effective_rank_after": item["mean_effective_rank_after"],
                    "adapter_fro_before": item["adapter_fro_before"],
                    "adapter_fro_after": item["adapter_fro_after"],
                    "adapter_fro_ratio": item["adapter_fro_ratio"],
                    "max_saved_update_relative_error": item["max_saved_update_relative_error"],
                })
    write_tsv(
        output / "spectral_summary.tsv",
        spectral_rows,
        [
            "base", "task", "configuration", "fast_steps", "stable_steps", "modules",
            "mean_effective_rank_before", "mean_effective_rank_after", "adapter_fro_before",
            "adapter_fro_after", "adapter_fro_ratio", "max_saved_update_relative_error",
        ],
    )
    print(f"[Done] wrote {output}")


if __name__ == "__main__":
    main()
