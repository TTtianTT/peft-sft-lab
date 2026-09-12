#!/usr/bin/env python3
"""Summarize paired GSM8K evaluation of signed spectral standardization."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ENDPOINTS = ("correct_strict", "correct_numeric")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--adapter_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260911)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_mcnemar(repaired: int, broken: int) -> float:
    total = repaired + broken
    if total == 0:
        return 1.0
    lower = min(repaired, broken)
    return min(1.0, 2 * sum(math.comb(total, k) for k in range(lower + 1)) / (2**total))


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    labels = [
        "original_lora", "zero_rebuild", "global_0p40", "common_hns",
        "zscore_variance", "zscore_std",
    ]
    eval_dir = Path(args.eval_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    reference_ids = None
    for label in labels:
        rows = read_jsonl(eval_dir / label / "predictions.jsonl")
        ids = [str(row["id"]) for row in rows]
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError(f"ID mismatch for {label}")
        arrays[label] = {
            endpoint: np.asarray([bool(row[endpoint]) for row in rows]) for endpoint in ENDPOINTS
        }

    scores = [{
        "label": label,
        "samples": len(reference_ids or []),
        "strict_accuracy": float(arrays[label]["correct_strict"].mean()),
        "numeric_accuracy": float(arrays[label]["correct_numeric"].mean()),
    } for label in labels]
    write_tsv(output / "scores.tsv", scores)

    pairs = []
    for standardized in ("zscore_variance", "zscore_std"):
        for baseline in ("zero_rebuild", "global_0p40", "common_hns", "original_lora"):
            pairs.append((baseline, standardized))
    rng = np.random.default_rng(args.seed)
    contrasts = []
    for endpoint in ENDPOINTS:
        for left, right in pairs:
            delta = arrays[right][endpoint].astype(np.int8) - arrays[left][endpoint].astype(np.int8)
            indices = rng.integers(0, len(delta), size=(args.bootstrap, len(delta)))
            distribution = delta[indices].mean(axis=1)
            repaired = int(np.sum(delta == 1))
            broken = int(np.sum(delta == -1))
            contrasts.append({
                "contrast": f"{right}_minus_{left}_{endpoint}",
                "endpoint": endpoint,
                "left": left,
                "right": right,
                "delta": float(delta.mean()),
                "ci_low": float(np.quantile(distribution, 0.025)),
                "ci_high": float(np.quantile(distribution, 0.975)),
                "repaired": repaired,
                "broken": broken,
                "mcnemar_p": exact_mcnemar(repaired, broken),
            })
    write_tsv(output / "contrasts.tsv", contrasts)
    manifest = json.loads(Path(args.adapter_manifest).read_text())
    result = {
        "status": "complete",
        "samples": len(reference_ids or []),
        "scores": scores,
        "contrasts": contrasts,
        "spectral_norms": {
            "original_total_fro": manifest["original_total_fro"],
            "target_total_fro": manifest["target_total_fro"],
            "target_to_original_fro_ratio": manifest["target_to_original_fro_ratio"],
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
