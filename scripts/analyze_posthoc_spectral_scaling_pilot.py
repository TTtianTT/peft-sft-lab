#!/usr/bin/env python3
"""Paired analysis for the fixed post-hoc spectral-scaling pilot variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


LABELS = (
    "lora",
    "global_norm_matched",
    "per_module_spectral_scale",
    "shuffled_scale_1",
    "shuffled_scale_2",
    "shuffled_scale_3",
    "full_hns",
)
SHUFFLES = ("shuffled_scale_1", "shuffled_scale_2", "shuffled_scale_3")
ENDPOINTS = ("correct_strict", "correct_numeric")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "validation"), required=True)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--adapter_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--permutation", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260911)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def exact_mcnemar(repaired: int, broken: int) -> float:
    discordant = repaired + broken
    if discordant == 0:
        return 1.0
    lower = min(repaired, broken)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def bootstrap_interval(delta: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    distribution = delta[indices].mean(axis=1)
    return float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))


def paired_binary(
    name: str,
    left_label: str,
    right_label: str,
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict:
    delta = right.astype(np.int8) - left.astype(np.int8)
    low, high = bootstrap_interval(delta, rng, draws)
    repaired = int(np.sum(delta == 1))
    broken = int(np.sum(delta == -1))
    return {
        "contrast": name,
        "left": left_label,
        "right": right_label,
        "samples": len(delta),
        "left_accuracy": float(left.mean()),
        "right_accuracy": float(right.mean()),
        "delta": float(delta.mean()),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "repaired": repaired,
        "broken": broken,
        "paired_test": "exact_mcnemar",
        "paired_p": exact_mcnemar(repaired, broken),
    }


def sign_flip_p(delta: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    observed = abs(float(delta.mean()))
    exceed = 0
    completed = 0
    while completed < draws:
        chunk = min(5000, draws - completed)
        signs = rng.integers(0, 2, size=(chunk, len(delta)), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs * delta).mean(axis=1))
        exceed += int(np.sum(permuted >= observed - 1e-15))
        completed += chunk
    return (exceed + 1) / (draws + 1)


def load_values(eval_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    values: dict[str, dict[str, np.ndarray]] = {}
    reference_ids: list[str] | None = None
    for label in LABELS:
        rows = read_jsonl(eval_dir / label / "predictions.jsonl")
        ids = [str(row["id"]) for row in rows]
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError(f"question IDs/order differ for {label}")
        values[label] = {
            endpoint: np.asarray([bool(row[endpoint]) for row in rows], dtype=bool)
            for endpoint in ENDPOINTS
        }
    return values, reference_ids or []


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.adapter_manifest).read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("adapter build is not complete")
    values, ids = load_values(Path(args.eval_dir))
    rng = np.random.default_rng(args.seed)

    scores = []
    for label in LABELS:
        scores.append({
            "stage": args.stage,
            "label": label,
            "samples": len(ids),
            "strict_accuracy": float(values[label]["correct_strict"].mean()),
            "numeric_accuracy": float(values[label]["correct_numeric"].mean()),
        })
    write_tsv(output / f"{args.stage}_scores.tsv", scores)

    contrasts: list[dict] = []
    binary_pairs = [
        ("lora", "global_norm_matched"),
        ("lora", "per_module_spectral_scale"),
        ("lora", "shuffled_scale_1"),
        ("lora", "shuffled_scale_2"),
        ("lora", "shuffled_scale_3"),
        ("lora", "full_hns"),
        ("global_norm_matched", "per_module_spectral_scale"),
        ("shuffled_scale_1", "per_module_spectral_scale"),
        ("shuffled_scale_2", "per_module_spectral_scale"),
        ("shuffled_scale_3", "per_module_spectral_scale"),
        ("per_module_spectral_scale", "full_hns"),
    ]
    for endpoint in ENDPOINTS:
        for left, right in binary_pairs:
            row = paired_binary(
                f"{right}_minus_{left}_{endpoint}",
                left,
                right,
                values[left][endpoint],
                values[right][endpoint],
                rng,
                args.bootstrap,
            )
            row["endpoint"] = endpoint
            contrasts.append(row)

        shuffle_mean = np.stack([values[label][endpoint].astype(float) for label in SHUFFLES]).mean(axis=0)
        per_module = values["per_module_spectral_scale"][endpoint].astype(float)
        delta = per_module - shuffle_mean
        low, high = bootstrap_interval(delta, rng, args.bootstrap)
        contrasts.append({
            "contrast": f"per_module_spectral_scale_minus_shuffle_mean_{endpoint}",
            "left": "mean(shuffled_scale_1..3)",
            "right": "per_module_spectral_scale",
            "samples": len(delta),
            "left_accuracy": float(shuffle_mean.mean()),
            "right_accuracy": float(per_module.mean()),
            "delta": float(delta.mean()),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "repaired": "",
            "broken": "",
            "paired_test": "question_cluster_sign_flip",
            "paired_p": sign_flip_p(delta, rng, args.permutation),
            "endpoint": endpoint,
        })

    write_tsv(output / f"{args.stage}_contrasts.tsv", contrasts)
    result = {
        "status": "complete",
        "stage": args.stage,
        "samples": len(ids),
        "adapter_manifest": str(Path(args.adapter_manifest).resolve()),
        "scores": scores,
        "contrasts": contrasts,
    }
    (output / f"{args.stage}_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
