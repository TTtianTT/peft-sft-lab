#!/usr/bin/env python3
"""Summarize one six-way MetaMath, Tulu, or Commonsense experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--task", choices=("metamath", "tulu", "commonsense"), required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    experiments = json.loads((root / "experiment_paths.json").read_text())
    rows: list[dict] = []
    categories: list[dict] = []
    for item in experiments:
        label = item["label"]
        if args.task == "metamath":
            path = root / "eval" / label / "gsm8k" / "metrics.json"
            metrics = json.loads(path.read_text()) if path.is_file() else {}
            score = metrics.get("accuracy_strict")
            row = {
                "label": label,
                "primary_score": score,
                "gsm8k_accuracy": score,
                "correct": metrics.get("correct"),
                "total": metrics.get("total"),
            }
        elif args.task == "tulu":
            path = root / "eval" / label / "ifeval" / "metrics.json"
            metrics = json.loads(path.read_text()) if path.is_file() else {}
            score_keys = (
                "prompt_level_strict_acc",
                "prompt_level_loose_acc",
                "inst_level_strict_acc",
                "inst_level_loose_acc",
            )
            available = [float(metrics[key]) for key in score_keys if metrics.get(key) is not None]
            row = {
                "label": label,
                "primary_score": sum(available) / len(available) if available else None,
                "prompt_strict": metrics.get("prompt_level_strict_acc"),
                "prompt_loose": metrics.get("prompt_level_loose_acc"),
                "instruction_strict": metrics.get("inst_level_strict_acc"),
                "instruction_loose": metrics.get("inst_level_loose_acc"),
                "total_prompts": metrics.get("total_prompts"),
                "total_instructions": metrics.get("total_instructions"),
            }
            for category, values in metrics.get("per_category", {}).items():
                categories.append({"label": label, "category": category, **values})
        else:
            path = root / "eval" / label / "commonsense" / "summary.json"
            metrics = json.loads(path.read_text()) if path.is_file() else {}
            score = metrics.get("macro_accuracy")
            row = {
                "label": label,
                "primary_score": score,
                "macro_accuracy": score,
                "micro_accuracy": metrics.get("micro_accuracy"),
                "correct": metrics.get("correct"),
                "total": metrics.get("total"),
            }
            for category, values in metrics.get("tasks", {}).items():
                categories.append({"label": label, "category": category, **values})
        rows.append(row)

    baseline = next((row["primary_score"] for row in rows if row["label"] == "lora"), None)
    for row in rows:
        row["gain_vs_lora"] = (
            float(row["primary_score"]) - float(baseline)
            if row["primary_score"] is not None and baseline is not None
            else None
        )
    write_tsv(root / "summary.tsv", rows)
    if categories:
        write_tsv(root / "category_breakdown.tsv", categories)
    payload = {
        "task": args.task,
        "primary_metric": {
            "metamath": "gsm8k_accuracy",
            "tulu": "mean_ifeval_four_metrics",
            "commonsense": "eight_task_macro_accuracy",
        }[args.task],
        "rows": rows,
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
