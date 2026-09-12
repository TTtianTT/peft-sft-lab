#!/usr/bin/env python3
"""Analyze locked PIQA direction-block interventions and A/B non-additivity."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def load(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return (
        [row["id"] for row in rows],
        np.asarray([bool(row["correct"]) for row in rows], dtype=np.int8),
        np.asarray([not row.get("prediction_letter") for row in rows], dtype=np.int8),
    )


def path(root: Path, label: str) -> Path:
    return root / label / "commonsense" / "piqa" / "predictions.jsonl"


def bootstrap(values: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    index = rng.integers(0, len(values), size=(draws, len(values)))
    estimates = values[index].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def exact_mcnemar_p(repaired: int, broken: int) -> float:
    discordant = repaired + broken
    if discordant == 0:
        return 1.0
    lower = min(repaired, broken)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    args = parse_args()
    root = Path(args.eval_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = (
        "full_hns",
        "suppress_block_A",
        "suppress_block_B",
        "suppress_block_AB",
        "predicted_retain_block",
        "scalar_global_fro_match",
    )
    ids, baseline, baseline_invalid = load(path(root, "lora"))
    values = {"lora": baseline}
    invalid = {"lora": baseline_invalid}
    rng = np.random.default_rng(args.seed)
    rows = []
    for label in labels:
        edit_ids, edited, edited_invalid = load(path(root, label))
        if edit_ids != ids:
            raise RuntimeError(f"ordered IDs differ for {label}")
        values[label] = edited
        invalid[label] = edited_invalid
        delta = edited - baseline
        repaired = int(np.sum((baseline == 0) & (edited == 1)))
        broken = int(np.sum((baseline == 1) & (edited == 0)))
        ci = bootstrap(delta.astype(float), args.bootstrap, rng)
        p = exact_mcnemar_p(repaired, broken)
        rows.append(
            {
                "label": label,
                "samples": len(ids),
                "baseline_accuracy": float(baseline.mean()),
                "accuracy": float(edited.mean()),
                "delta_accuracy": float(delta.mean()),
                "bootstrap_ci_low": ci[0],
                "bootstrap_ci_high": ci[1],
                "repaired": repaired,
                "broken": broken,
                "wrong_to_correct_fraction": repaired / len(ids),
                "correct_to_wrong_fraction": broken / len(ids),
                "invalid": int(edited_invalid.sum()),
                "mcnemar_exact_p": p,
            }
        )

    with (output / "block_interventions.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    interaction = (
        values["suppress_block_AB"]
        - values["suppress_block_A"]
        - values["suppress_block_B"]
        + baseline
    ).astype(float)
    interaction_ci = bootstrap(interaction, args.bootstrap, rng)
    contrasts = []
    for left, right, name in (
        ("suppress_block_AB", "predicted_retain_block", "signed_block_contrast"),
        ("suppress_block_AB", "scalar_global_fro_match", "shape_vs_scalar"),
        ("suppress_block_AB", "full_hns", "selected_vs_full_hns"),
    ):
        delta = (values[left] - values[right]).astype(float)
        ci = bootstrap(delta, args.bootstrap, rng)
        contrasts.append(
            {
                "contrast": name,
                "left": left,
                "right": right,
                "delta": float(delta.mean()),
                "bootstrap_ci_low": ci[0],
                "bootstrap_ci_high": ci[1],
            }
        )
    contrasts.append(
        {
            "contrast": "AB_nonadditivity",
            "left": "suppress_block_AB",
            "right": "A+B-baseline",
            "delta": float(interaction.mean()),
            "bootstrap_ci_low": interaction_ci[0],
            "bootstrap_ci_high": interaction_ci[1],
        }
    )
    with (output / "block_contrasts.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(contrasts)
    print(json.dumps({"rows": rows, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
