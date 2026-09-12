#!/usr/bin/env python3
"""Paired analysis of the stochastic HNS-above-Base audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


COMPARISONS = (
    ("original_hns", "base", "original_hns_vs_base"),
    ("original_hns", "original_lora", "original_hns_vs_original_lora"),
    ("common_hns", "base", "common_hns_vs_base"),
    ("common_hns", "common_lora", "common_hns_vs_common_lora"),
    ("common_hns", "calibrated_scalar", "common_hns_vs_calibrated_scalar"),
    ("original_hns", "common_hns", "original_vs_common_hns"),
    ("original_lora", "common_lora", "original_vs_common_lora"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260911)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_tsv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    seen = set(fields)
    for row in rows[1:]:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_vectors(task: str, rows: list[dict]) -> dict[str, np.ndarray]:
    if task == "metamath":
        n = float(len(rows[0]["rollouts"]))
        return {
            "expected_reward": np.asarray([row["strict_count"] / n for row in rows]),
            "any_at_n": np.asarray([row["strict_count"] > 0 for row in rows], dtype=float),
            "majority": np.asarray([row["majority_correct_strict"] for row in rows], dtype=float),
        }
    n_int = len(rows[0]["rollouts"])
    return {
        "expected_reward": np.asarray([row["correct_count"] / n_int for row in rows]),
        "any_at_n": np.asarray([row["correct_count"] > 0 for row in rows], dtype=float),
        "pass_at_4": np.asarray([
            1.0 if n_int - row["correct_count"] < min(4, n_int)
            else 1.0 - math.comb(n_int - row["correct_count"], min(4, n_int))
            / math.comb(n_int, min(4, n_int))
            for row in rows
        ]),
    }


def paired_ci(left: np.ndarray, right: np.ndarray, draws: int, rng: np.random.Generator):
    delta = left - right
    n = len(delta)
    means = np.empty(draws, dtype=np.float64)
    block = 1000
    for start in range(0, draws, block):
        size = min(block, draws - start)
        indices = rng.integers(0, n, size=(size, n))
        means[start : start + size] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high)


def find_label(labels: list[str], prefix: str) -> str | None:
    exact = [label for label in labels if label == prefix]
    if exact:
        return exact[0]
    matches = [label for label in labels if label.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    return None


def main() -> None:
    args = parse_args()
    root = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for base_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for task in ("metamath", "magicoder"):
            task_dir = base_dir / task
            if not task_dir.is_dir():
                continue
            labels = sorted(path.name for path in task_dir.iterdir() if (path / "scored.jsonl").is_file())
            data: dict[str, dict[str, np.ndarray]] = {}
            ids: dict[str, list[str]] = {}
            for label in labels:
                rows = read_rows(task_dir / label / "scored.jsonl")
                ids[label] = [str(row["id"]) for row in rows]
                data[label] = metric_vectors(task, rows)
                metrics = json.loads((task_dir / label / "metrics.json").read_text(encoding="utf-8"))
                metric_rows.append({
                    "base": base_dir.name,
                    "eval_task": task,
                    "condition": label,
                    "samples": len(rows),
                    **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
                })
            for left_prefix, right_prefix, name in COMPARISONS:
                left_label = find_label(labels, left_prefix)
                right_label = find_label(labels, right_prefix)
                if left_label is None or right_label is None:
                    continue
                if ids[left_label] != ids[right_label]:
                    raise RuntimeError(f"Pairing mismatch: {base_dir.name}/{task}/{name}")
                for metric in data[left_label]:
                    if metric not in data[right_label]:
                        continue
                    delta, low, high = paired_ci(
                        data[left_label][metric], data[right_label][metric], args.bootstrap_draws, rng
                    )
                    comparison_rows.append({
                        "base": base_dir.name,
                        "eval_task": task,
                        "comparison": name,
                        "left": left_label,
                        "right": right_label,
                        "metric": metric,
                        "left_score": float(data[left_label][metric].mean()),
                        "right_score": float(data[right_label][metric].mean()),
                        "delta": delta,
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_draws": args.bootstrap_draws,
                    })
    write_tsv(output / "rollout_metrics.tsv", metric_rows)
    write_tsv(output / "rollout_comparisons.tsv", comparison_rows)
    manifest = {
        "status": "complete",
        "metric_rows": len(metric_rows),
        "comparison_rows": len(comparison_rows),
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
