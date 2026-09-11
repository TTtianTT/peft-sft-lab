#!/usr/bin/env python3
"""Compare endpoint metrics across two numerical-gate generation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_forgetting_matrix import (
    read_rows,
    score_commonsense,
    score_ifeval,
    score_metamath,
)
from finetune.eval.eval_humaneval import normalize_humaneval_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run1", required=True)
    parser.add_argument("--run2", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def scored_rows(task: str, rows: list[dict]) -> tuple[list[dict], str]:
    if task == "metamath":
        scored, _ = score_metamath(rows)
        return scored, "correct_strict"
    if task == "tulu":
        scored, _ = score_ifeval(rows)
        return scored, "prompt_strict_passed"
    if task == "commonsense":
        scored, _ = score_commonsense(rows)
        return scored, "correct"
    for row in rows:
        row["normalized_completion"] = normalize_humaneval_completion(
            row["prediction_text"], row["problem_prompt"], row["entry_point"]
        )
    return rows, "normalized_completion"


def main() -> None:
    args = parse_args()
    roots = (Path(args.run1), Path(args.run2))
    result: dict = {"tasks": {}, "condition_differences": []}
    for task in ("magicoder", "metamath", "tulu", "commonsense"):
        labels1 = {path.name for path in (roots[0] / task).iterdir() if path.is_dir()}
        labels2 = {path.name for path in (roots[1] / task).iterdir() if path.is_dir()}
        per_task = []
        for label in sorted(labels1 & labels2):
            left, field = scored_rows(task, read_rows(roots[0] / task / label / "predictions.jsonl"))
            right, _ = scored_rows(task, read_rows(roots[1] / task / label / "predictions.jsonl"))
            if [row["id"] for row in left] != [row["id"] for row in right]:
                raise RuntimeError(f"ID mismatch: {task}/{label}")
            changed = sum(a[field] != b[field] for a, b in zip(left, right))
            row = {"task": task, "label": label, "samples": len(left), "metric_field": field,
                   "changed": changed, "fraction_stable": 1.0 - changed / len(left)}
            if field != "normalized_completion":
                row["run1_score"] = sum(bool(item[field]) for item in left) / len(left)
                row["run2_score"] = sum(bool(item[field]) for item in right) / len(right)
                row["score_delta"] = row["run2_score"] - row["run1_score"]
            per_task.append(row)
            if changed:
                result["condition_differences"].append(row)
        result["tasks"][task] = {
            "conditions": len(per_task),
            "conditions_with_metric_change": sum(row["changed"] > 0 for row in per_task),
            "changed_items": sum(row["changed"] for row in per_task),
            "total_items": sum(row["samples"] for row in per_task),
            "min_fraction_stable": min(row["fraction_stable"] for row in per_task),
        }
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
