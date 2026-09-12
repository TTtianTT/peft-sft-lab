#!/usr/bin/env python3
"""Analyze paired PIQA finite interventions for matched opposite-sign spectral directions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", required=True, choices=("dose", "validation"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def bootstrap_mean(values: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    estimates = values[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def prediction_path(root: Path, label: str) -> Path:
    return root / label / "commonsense" / "piqa" / "predictions.jsonl"


def exact_mcnemar_p(repaired: int, broken: int) -> float:
    discordant = repaired + broken
    if discordant == 0:
        return 1.0
    lower = min(repaired, broken)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_permutation(
    predictor: list[float], outcome: list[float], *, draws: int, rng: np.random.Generator
) -> tuple[float, float]:
    x = rankdata(predictor)
    y = rankdata(outcome)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), 1.0
    observed = float(np.corrcoef(x, y)[0, 1])
    exceed = 0
    for _ in range(draws):
        permuted = rng.permutation(y)
        exceed += abs(float(np.corrcoef(x, permuted)[0, 1])) >= abs(observed) - 1e-15
    return observed, (exceed + 1) / (draws + 1)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    eval_root = Path(args.eval_root)
    variants = json.loads(Path(args.manifest).read_text())["variants"]
    selection = read_tsv(Path(args.selection))
    selection_by_index = {int(row["pair"]): row for row in selection if row["class"] == "predicted_suppress"}

    baseline_rows = read_jsonl(prediction_path(eval_root, "lora"))
    baseline = {row["id"]: row for row in baseline_rows}
    baseline_ids = list(baseline)
    rng = np.random.default_rng(args.seed)
    summary_rows = []
    deltas_by_label: dict[str, np.ndarray] = {}
    for variant in variants:
        label = variant["label"]
        edited_rows = read_jsonl(prediction_path(eval_root, label))
        edited = {row["id"]: row for row in edited_rows}
        if set(edited) != set(baseline):
            raise RuntimeError(f"example IDs differ for {label}")
        base_correct = np.array([bool(baseline[item]["correct"]) for item in baseline_ids], dtype=np.int8)
        edit_correct = np.array([bool(edited[item]["correct"]) for item in baseline_ids], dtype=np.int8)
        delta = edit_correct - base_correct
        deltas_by_label[label] = delta
        repaired = int(np.sum((base_correct == 0) & (edit_correct == 1)))
        broken = int(np.sum((base_correct == 1) & (edit_correct == 0)))
        ci = bootstrap_mean(delta.astype(float), args.bootstrap, rng)
        mcnemar = exact_mcnemar_p(repaired, broken)
        summary_rows.append(
            {
                **variant,
                "split": args.split,
                "samples": len(delta),
                "baseline_accuracy": float(base_correct.mean()),
                "accuracy": float(edit_correct.mean()),
                "delta_accuracy": float(delta.mean()),
                "bootstrap_ci_low": ci[0],
                "bootstrap_ci_high": ci[1],
                "wrong_to_correct_fraction": repaired / len(delta),
                "correct_to_wrong_fraction": broken / len(delta),
                "repaired": repaired,
                "broken": broken,
                "invalid": sum(not edited[item].get("prediction_letter") for item in baseline_ids),
                "mcnemar_exact_p": mcnemar,
            }
        )

    with (output / "direction_interventions.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    contrasts = []
    strengths = sorted({float(row["strength"]) for row in variants})
    for strength in strengths:
        pair_differences = []
        suppress_deltas = []
        retain_deltas = []
        predictions, observations = [], []
        for pair in sorted(selection_by_index):
            pair_variants = [
                row for row in variants if int(row["pair"]) == pair and float(row["strength"]) == strength
            ]
            suppress_variant = next(row for row in pair_variants if row["class"] == "predicted_suppress")
            retain_variant = next(row for row in pair_variants if row["class"] == "predicted_retain")
            suppress_delta = deltas_by_label[suppress_variant["label"]].astype(float)
            retain_delta = deltas_by_label[retain_variant["label"]].astype(float)
            suppress_deltas.append(suppress_delta)
            retain_deltas.append(retain_delta)
            pair_differences.append(suppress_delta - retain_delta)
            for variant, observed in (
                (suppress_variant, suppress_delta.mean()),
                (retain_variant, retain_delta.mean()),
            ):
                source = next(
                    row
                    for row in selection
                    if int(row["pair"]) == pair and row["class"] == variant["class"]
                )
                predictions.append(float(source["predicted_hns_margin_gain"]) * strength)
                observations.append(float(observed))

        suppress_matrix = np.stack(suppress_deltas)
        retain_matrix = np.stack(retain_deltas)
        difference_matrix = np.stack(pair_differences)
        boot = np.empty(args.bootstrap, dtype=np.float64)
        for draw in range(args.bootstrap):
            pair_index = rng.integers(0, difference_matrix.shape[0], size=difference_matrix.shape[0])
            example_index = rng.integers(0, difference_matrix.shape[1], size=difference_matrix.shape[1])
            boot[draw] = difference_matrix[pair_index][:, example_index].mean()
        rho, rho_p = spearman_permutation(predictions, observations, draws=args.bootstrap, rng=rng)
        contrasts.append(
            {
                "split": args.split,
                "strength": strength,
                "directions_per_class": suppress_matrix.shape[0],
                "samples": suppress_matrix.shape[1],
                "predicted_suppress_mean_delta_accuracy": float(suppress_matrix.mean()),
                "predicted_retain_mean_delta_accuracy": float(retain_matrix.mean()),
                "matched_contrast_suppress_minus_retain": float(difference_matrix.mean()),
                "matched_contrast_ci_low": float(np.quantile(boot, 0.025)),
                "matched_contrast_ci_high": float(np.quantile(boot, 0.975)),
                "spearman_predicted_margin_gain_vs_accuracy_delta": float(rho),
                "spearman_p": float(rho_p),
            }
        )

    with (output / "matched_signal_contrasts.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(contrasts)

    if args.split == "dose":
        # Predeclared low-dimensional rule: choose the dose with the largest
        # mean accuracy change over the predicted-suppress directions. Ties go
        # to the smaller dose. Direction membership is never tuned here.
        chosen = sorted(
            contrasts,
            key=lambda row: (-row["predicted_suppress_mean_delta_accuracy"], row["strength"]),
        )[0]
        decision = {
            "rule": "maximize mean dose-set accuracy gain over the preselected predicted-suppress directions; tie -> smaller strength",
            "chosen_strength": chosen["strength"],
            "dose_contrasts": contrasts,
        }
        (output / "dose_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    print(json.dumps({"split": args.split, "variants": len(summary_rows), "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
