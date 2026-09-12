#!/usr/bin/env python3
"""Join greedy Base->LoRA->HNS transition classes to stochastic rollout rewards."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


SELECTED = {
    ("Qwen3-8B", "magicoder", "metamath"),
    ("Qwen3-8B", "metamath", "magicoder"),
    ("Llama-3.1-8B-Instruct", "magicoder", "metamath"),
    ("Llama-3.1-8B-Instruct", "metamath", "magicoder"),
}
CONDITIONS = ("base", "original_lora", "original_hns", "common_lora", "common_hns")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reward(task: str, row: dict) -> float:
    count = row["strict_count"] if task == "metamath" else row["correct_count"]
    return count / len(row["rollouts"])


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    with gzip.open(args.transitions, "rt", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["base"], row["train_task"], row["eval_task"])
            if key in SELECTED:
                grouped[key].append(row)
    run_root = Path(args.run_dir)
    output_rows = []
    for (base, train_task, eval_task), transitions in sorted(grouped.items()):
        values = {}
        for condition in CONDITIONS:
            rows = read_jsonl(run_root / base / eval_task / condition / "scored.jsonl")
            values[condition] = {str(row["id"]): reward(eval_task, row) for row in rows}
        strata: dict[str, list[str]] = defaultdict(list)
        for row in transitions:
            strata[row["transition"]].append(str(row["id"]))
        for transition, ids in sorted(strata.items()):
            means = {
                condition: sum(values[condition][identity] for identity in ids) / len(ids)
                for condition in CONDITIONS
            }
            output_rows.append({
                "base": base,
                "train_task": train_task,
                "eval_task": eval_task,
                "greedy_transition": transition,
                "samples": len(ids),
                **{f"{condition}_expected_reward": means[condition] for condition in CONDITIONS},
                "original_hns_minus_base": means["original_hns"] - means["base"],
                "original_hns_minus_lora": means["original_hns"] - means["original_lora"],
                "common_hns_minus_base": means["common_hns"] - means["base"],
                "common_hns_minus_lora": means["common_hns"] - means["common_lora"],
            })
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({"status": "complete", "rows": len(output_rows), "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
