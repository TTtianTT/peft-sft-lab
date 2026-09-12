#!/usr/bin/env python3
"""Check exact token reproducibility across deterministic diagnostic runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--adapter_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="append", default=["run1", "run2", "reverse"])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    root = Path(args.eval_root)
    run_names = args.run[-3:] if len(args.run) > 3 else args.run
    first_manifest = json.loads((root / run_names[0] / "evaluation_manifest.json").read_text())
    labels = [row["label"] for row in first_manifest["variants"]]
    adapter_manifest = json.loads(Path(args.adapter_manifest).read_text())
    comparisons = []
    passed = True
    for label in labels:
        baseline = read_jsonl(root / run_names[0] / label / "predictions.jsonl")
        baseline_by_id = {str(row["id"]): row for row in baseline}
        for run_name in run_names[1:]:
            other = read_jsonl(root / run_name / label / "predictions.jsonl")
            other_by_id = {str(row["id"]): row for row in other}
            if set(baseline_by_id) != set(other_by_id):
                raise RuntimeError(f"ID mismatch for {label}/{run_name}")
            ids = sorted(baseline_by_id)
            exact_tokens = sum(
                baseline_by_id[identity]["token_ids"] == other_by_id[identity]["token_ids"] for identity in ids
            )
            exact_text = sum(
                baseline_by_id[identity]["prediction_text"] == other_by_id[identity]["prediction_text"]
                for identity in ids
            )
            exact_strict = sum(
                baseline_by_id[identity]["correct_strict"] == other_by_id[identity]["correct_strict"]
                for identity in ids
            )
            row = {
                "label": label,
                "left_run": run_names[0],
                "right_run": run_name,
                "samples": len(ids),
                "exact_token_sequences": exact_tokens,
                "exact_text": exact_text,
                "exact_strict_status": exact_strict,
                "token_match_fraction": exact_tokens / len(ids),
            }
            comparisons.append(row)
            passed = passed and exact_tokens == len(ids)

    representation = []
    run1 = root / run_names[0]
    for left, right in (
        ("untouched_lora", "zero_rebuild"),
        ("original_factor_permodule", "common_per_module"),
        ("existing_hns", "common_hns"),
    ):
        left_rows = {str(row["id"]): row for row in read_jsonl(run1 / left / "predictions.jsonl")}
        right_rows = {str(row["id"]): row for row in read_jsonl(run1 / right / "predictions.jsonl")}
        ids = sorted(left_rows)
        representation.append({
            "left": left,
            "right": right,
            "samples": len(ids),
            "exact_token_sequences": sum(left_rows[i]["token_ids"] == right_rows[i]["token_ids"] for i in ids),
            "strict_status_changes": sum(
                left_rows[i]["correct_strict"] != right_rows[i]["correct_strict"] for i in ids
            ),
        })

    passed = passed and adapter_manifest["max_saved_update_relative_error"] <= 5e-4
    result = {
        "status": "pass" if passed else "fail",
        "criterion": "100% exact token IDs for every identical adapter across two same-order and one reverse-order fresh processes",
        "runs": run_names,
        "adapter_max_saved_update_relative_error": adapter_manifest["max_saved_update_relative_error"],
        "repeatability": comparisons,
        "representation_sensitivity": representation,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
