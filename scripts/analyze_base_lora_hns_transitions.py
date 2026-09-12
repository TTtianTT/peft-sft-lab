#!/usr/bin/env python3
"""Paired Base -> LoRA -> HNS outcome-transition analysis."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path


TASKS = ("magicoder", "metamath", "tulu", "commonsense")
BASE_DIRS = {
    "Qwen3-8B": "Qwen3-8B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
}
FIELDS = {
    "magicoder": "correct",
    "metamath": "correct_strict",
    "tulu": "prompt_strict_passed",
    "commonsense": "correct",
}
LABELS = {
    (1, 0, 1): "recovered_forgetting",
    (1, 0, 0): "persistent_forgetting",
    (1, 1, 0): "hns_new_harm",
    (1, 1, 1): "preserved_base_success",
    (0, 1, 1): "preserved_positive_transfer",
    (0, 1, 0): "lost_positive_transfer",
    (0, 0, 1): "hns_novel_gain",
    (0, 0, 0): "persistent_failure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(path: Path) -> dict[str, dict]:
    rows = read_rows(path)
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate ids in {path}")
    return result


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(counts: Counter, total: int) -> dict:
    forgetting = counts["recovered_forgetting"] + counts["persistent_forgetting"]
    stable_base = counts["preserved_base_success"] + counts["hns_new_harm"]
    positive_transfer = counts["preserved_positive_transfer"] + counts["lost_positive_transfer"]
    return {
        "samples": total,
        **{label: counts[label] for label in LABELS.values()},
        "forgetting_opportunities": forgetting,
        "recovery_rate": safe_ratio(counts["recovered_forgetting"], forgetting),
        "base_success_harm_rate": safe_ratio(counts["hns_new_harm"], stable_base),
        "positive_transfer_opportunities": positive_transfer,
        "positive_transfer_preservation_rate": safe_ratio(
            counts["preserved_positive_transfer"], positive_transfer
        ),
        "net_hns_minus_lora": (
            counts["recovered_forgetting"] + counts["hns_novel_gain"]
            - counts["hns_new_harm"] - counts["lost_positive_transfer"]
        ) / total,
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.run_root) / "formal"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    edge_rows: list[dict] = []
    stratum_rows: list[dict] = []
    edge_signs = Counter()
    total_items = 0
    destination = output / "per_item_transitions.tsv.gz"
    columns = [
        "base", "train_task", "eval_task", "subtask", "id",
        "base_correct", "lora_correct", "hns_correct", "transition",
    ]
    with gzip.open(destination, "wt", encoding="utf-8", newline="", compresslevel=9) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for base, base_dir in BASE_DIRS.items():
            matrix = root / base_dir
            for train_task in TASKS:
                for eval_task in TASKS:
                    if eval_task == train_task:
                        continue
                    field = FIELDS[eval_task]
                    paths = {
                        "base": matrix / eval_task / "base" / "scored.jsonl",
                        "lora": matrix / eval_task / f"{train_task}__original_lora" / "scored.jsonl",
                        "hns": matrix / eval_task / f"{train_task}__original_hns" / "scored.jsonl",
                    }
                    data = {name: keyed(path) for name, path in paths.items()}
                    ids = list(data["base"])
                    if any(set(rows) != set(ids) for rows in data.values()):
                        raise RuntimeError(f"Pairing mismatch: {base}/{train_task}/{eval_task}")
                    counts: Counter = Counter()
                    strata: dict[str, Counter] = {}
                    scores = Counter()
                    for identity in ids:
                        source = data["base"][identity]
                        values = tuple(int(bool(data[name][identity][field])) for name in ("base", "lora", "hns"))
                        label = LABELS[values]
                        subtask = str(source.get("subtask", "all"))
                        counts[label] += 1
                        strata.setdefault(subtask, Counter())[label] += 1
                        scores.update(base=values[0], lora=values[1], hns=values[2])
                        writer.writerow({
                            "base": base, "train_task": train_task, "eval_task": eval_task,
                            "subtask": subtask, "id": identity,
                            "base_correct": values[0], "lora_correct": values[1],
                            "hns_correct": values[2], "transition": label,
                        })
                    summary = summarize(counts, len(ids))
                    edge_rows.append({
                        "base": base, "train_task": train_task, "eval_task": eval_task,
                        "base_score": scores["base"] / len(ids),
                        "lora_score": scores["lora"] / len(ids),
                        "hns_score": scores["hns"] / len(ids),
                        **summary,
                    })
                    x = scores["lora"] - scores["base"]
                    y = scores["hns"] - scores["lora"]
                    edge_signs[("forgotten" if x < 0 else "transfer", "improve" if y > 0 else "worsen" if y < 0 else "same")] += 1
                    total_items += len(ids)
                    for subtask, subcounts in sorted(strata.items()):
                        stratum_rows.append({
                            "base": base, "train_task": train_task, "eval_task": eval_task,
                            "subtask": subtask, **summarize(subcounts, sum(subcounts.values())),
                        })

    write_tsv(output / "edge_transition_summary.tsv", edge_rows)
    write_tsv(output / "stratum_transition_summary.tsv", stratum_rows)
    result = {
        "status": "complete",
        "edges": len(edge_rows),
        "paired_item_records": total_items,
        "edge_quadrants": {f"{left}_{right}": value for (left, right), value in edge_signs.items()},
        "per_item_file": destination.name,
    }
    (output / "transition_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
