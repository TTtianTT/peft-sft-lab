#!/usr/bin/env python3
"""Require exact repeatability across two forgetting diagnostic processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("magicoder", "metamath", "tulu", "commonsense")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run1", required=True)
    parser.add_argument("--run2", required=True)
    parser.add_argument("--variant_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--warn_only",
        action="store_true",
        help="Record failures without returning a non-zero status.",
    )
    return parser.parse_args()


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    run1, run2 = Path(args.run1), Path(args.run2)
    manifest = json.loads(Path(args.variant_manifest).read_text())
    labels = ["base"] + [row["label"] for row in manifest["variants"]]
    checks = []
    failures = []
    for task in TASKS:
        for label in labels:
            left = rows(run1 / task / label / "predictions.jsonl")
            right = rows(run2 / task / label / "predictions.jsonl")
            left_ids = [str(row["id"]) for row in left]
            right_ids = [str(row["id"]) for row in right]
            exact = sum(
                a.get("token_ids") == b.get("token_ids")
                for a, b in zip(left, right)
            ) if left_ids == right_ids else 0
            check = {
                "task": task,
                "label": label,
                "samples": len(left),
                "exact_token_sequences": exact,
                "fraction": exact / len(left) if left else 0.0,
            }
            checks.append(check)
            if not left or left_ids != right_ids or exact != len(left):
                failures.append(check)
    result = {
        "status": "pass" if not failures else "fail",
        "criterion": "100% exact token IDs for all identical base/adapter conditions",
        "checks": checks,
        "failures": failures,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failures": len(failures)}, indent=2))
    if failures and not args.warn_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
