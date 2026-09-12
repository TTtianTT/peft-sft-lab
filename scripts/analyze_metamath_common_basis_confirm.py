#!/usr/bin/env python3
"""Select a global scalar and analyze common-basis HNS confirmation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


GAMMAS = (1.0, 0.85, 0.70, 0.60, 0.50, 0.40)
ENDPOINTS = ("correct_strict", "correct_numeric")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "validation"), required=True)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter_root", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_mcnemar(repaired: int, broken: int) -> float:
    n = repaired + broken
    if n == 0:
        return 1.0
    lower = min(repaired, broken)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(lower + 1)) / (2**n))


def compare(name: str, left_name: str, right_name: str, left: np.ndarray, right: np.ndarray,
            rng: np.random.Generator, draws: int) -> dict:
    delta = right.astype(np.int8) - left.astype(np.int8)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    dist = delta[indices].mean(axis=1)
    repaired = int(np.sum(delta == 1))
    broken = int(np.sum(delta == -1))
    return {
        "contrast": name,
        "left": left_name,
        "right": right_name,
        "samples": len(delta),
        "left_accuracy": float(left.mean()),
        "right_accuracy": float(right.mean()),
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(dist, 0.025)),
        "ci_high": float(np.quantile(dist, 0.975)),
        "repaired": repaired,
        "broken": broken,
        "mcnemar_p": exact_mcnemar(repaired, broken),
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load(eval_dir: Path, labels: list[str]) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    reference = None
    values = {}
    for label in labels:
        rows = read_jsonl(eval_dir / label / "predictions.jsonl")
        ids = [str(row["id"]) for row in rows]
        if reference is None:
            reference = ids
        elif ids != reference:
            raise RuntimeError(f"ID mismatch: {label}")
        values[label] = {
            endpoint: np.asarray([bool(row[endpoint]) for row in rows]) for endpoint in ENDPOINTS
        }
    return reference or [], values


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.stage == "calibration":
        labels = ["original_lora"] + [f"global_{tag(gamma)}" for gamma in GAMMAS]
        labels += ["common_per_module", "common_hns"]
    else:
        labels = ["original_lora", "zero_rebuild", "selected_global", "common_per_module", "common_hns", "existing_hns"]
    ids, values = load(Path(args.eval_dir), labels)
    scores = [{
        "stage": args.stage,
        "label": label,
        "samples": len(ids),
        "strict_accuracy": float(values[label]["correct_strict"].mean()),
        "numeric_accuracy": float(values[label]["correct_numeric"].mean()),
    } for label in labels]
    write_tsv(output / f"{args.stage}_scores.tsv", scores)

    if args.stage == "calibration":
        strict = {gamma: float(values[f"global_{tag(gamma)}"]["correct_strict"].mean()) for gamma in GAMMAS}
        best = max(strict.values())
        chosen = max(gamma for gamma, score in strict.items() if score == best)
        selection = {
            "status": "locked",
            "rule": "maximize strict calibration accuracy; ties choose gamma closest to 1",
            "candidate_gammas": list(GAMMAS),
            "strict_accuracy": {str(gamma): score for gamma, score in strict.items()},
            "chosen_gamma": chosen,
            "selected_path": str((Path(args.adapter_root) / f"common-global-{tag(chosen)}").resolve()),
        }
        (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2))
        return

    if not args.selection:
        raise ValueError("--selection required for validation")
    selection = json.loads(Path(args.selection).read_text())
    pairs = [
        ("common_per_module", "common_hns"),
        ("selected_global", "common_hns"),
        ("zero_rebuild", "common_hns"),
        ("zero_rebuild", "selected_global"),
        ("zero_rebuild", "common_per_module"),
        ("common_hns", "existing_hns"),
        ("original_lora", "zero_rebuild"),
    ]
    contrasts = []
    for endpoint in ENDPOINTS:
        for left, right in pairs:
            row = compare(
                f"{right}_minus_{left}_{endpoint}", left, right,
                values[left][endpoint], values[right][endpoint], rng, args.bootstrap,
            )
            row["endpoint"] = endpoint
            contrasts.append(row)
    write_tsv(output / "validation_contrasts.tsv", contrasts)
    result = {
        "status": "complete_same_question_numerical_recheck",
        "chosen_gamma": selection["chosen_gamma"],
        "samples": len(ids),
        "scores": scores,
        "contrasts": contrasts,
    }
    (output / "validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
