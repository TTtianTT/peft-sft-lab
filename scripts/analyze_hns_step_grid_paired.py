#!/usr/bin/env python3
"""Paired uncertainty and transition analysis for the HNS step-grid evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


BASES = ("Qwen3-8B", "Llama-3.1-8B-Instruct")
TASKS = ("magicoder", "metamath", "tulu", "commonsense")
CONFIGS = [(0, 0)] + [(fast, stable) for fast in (2, 4, 8) for stable in (0, 1, 2)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--bootstrap_draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def correctness(task: str, row: dict) -> bool:
    if task in ("magicoder", "commonsense"):
        return bool(row["correct"])
    if task == "metamath":
        return bool(row["correct_strict"])
    if task == "tulu":
        return bool(row["prompt_strict_passed"])
    raise ValueError(task)


def seed_for(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return seed + int.from_bytes(digest[:4], "little")


def bootstrap_delta(
    differences_by_group: list[np.ndarray], draws: int, rng: np.random.Generator
) -> np.ndarray:
    result = np.zeros(draws, dtype=np.float64)
    for differences in differences_by_group:
        counts = np.array([
            np.count_nonzero(differences == -1),
            np.count_nonzero(differences == 0),
            np.count_nonzero(differences == 1),
        ])
        samples = rng.multinomial(len(differences), counts / len(differences), size=draws)
        result += (samples[:, 2] - samples[:, 0]) / len(differences)
    return result / len(differences_by_group)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def exact_mcnemar_p(gains: int, losses: int) -> float:
    """Two-sided exact binomial p-value for paired binary outcomes."""
    total = gains + losses
    if total == 0:
        return 1.0
    edge = min(gains, losses)
    log_probabilities = [
        math.lgamma(total + 1)
        - math.lgamma(index + 1)
        - math.lgamma(total - index + 1)
        - total * math.log(2.0)
        for index in range(edge + 1)
    ]
    maximum = max(log_probabilities)
    tail = math.exp(maximum) * sum(math.exp(value - maximum) for value in log_probabilities)
    return min(1.0, 2.0 * tail)


def main() -> None:
    args = parse_args()
    root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    results = []

    for base in BASES:
        for task in TASKS:
            task_root = root / "eval" / base / task
            lora_path = task_root / f"{task}__original_lora" / "scored.jsonl"
            if not lora_path.is_file():
                print(f"[Missing] {lora_path}")
                continue
            lora_rows = {str(row["id"]): row for row in read_rows(lora_path)}
            task_results = []
            for fast, stable in CONFIGS:
                variant = f"{task}__hns_f{fast}_s{stable}"
                variant_path = task_root / variant / "scored.jsonl"
                if not variant_path.is_file():
                    print(f"[Missing] {variant_path}")
                    continue
                edited_rows = {str(row["id"]): row for row in read_rows(variant_path)}
                if edited_rows.keys() != lora_rows.keys():
                    raise RuntimeError(f"paired id mismatch: {base}/{task}/{variant}")
                differences = []
                groups: dict[str, list[int]] = {}
                gains = losses = 0
                for identity, baseline in lora_rows.items():
                    edited = edited_rows[identity]
                    before = int(correctness(task, baseline))
                    after = int(correctness(task, edited))
                    difference = after - before
                    differences.append(difference)
                    group = str(baseline.get("subtask", "all"))
                    groups.setdefault(group, []).append(difference)
                    gains += int(difference == 1)
                    losses += int(difference == -1)
                arrays = [np.asarray(values, dtype=np.int8) for values in groups.values()]
                point = float(np.mean([array.mean() for array in arrays]))
                rng = np.random.default_rng(seed_for(args.seed, base, task, variant))
                bootstrap = bootstrap_delta(arrays, args.bootstrap_draws, rng)
                discordant = gains + losses
                p_value = exact_mcnemar_p(gains, losses)
                task_results.append({
                    "base": base,
                    "task": task,
                    "configuration": f"{fast}+{stable}",
                    "fast_steps": fast,
                    "stable_steps": stable,
                    "samples": len(differences),
                    "groups": len(groups),
                    "gain_pp": 100 * point,
                    "ci95_low_pp": 100 * float(np.quantile(bootstrap, 0.025)),
                    "ci95_high_pp": 100 * float(np.quantile(bootstrap, 0.975)),
                    "wrong_to_right": gains,
                    "right_to_wrong": losses,
                    "mcnemar_exact_p": p_value,
                })
            adjusted = holm_adjust([row["mcnemar_exact_p"] for row in task_results])
            for row, p_adjusted in zip(task_results, adjusted):
                row["mcnemar_holm_p"] = p_adjusted
            results.extend(task_results)

    fields = [
        "base", "task", "configuration", "fast_steps", "stable_steps", "samples", "groups",
        "gain_pp", "ci95_low_pp", "ci95_high_pp", "wrong_to_right", "right_to_wrong",
        "mcnemar_exact_p", "mcnemar_holm_p",
    ]
    with (output / "paired_stats.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)
    print(f"[Done] wrote {len(results)} paired comparisons to {output / 'paired_stats.tsv'}")


if __name__ == "__main__":
    main()
