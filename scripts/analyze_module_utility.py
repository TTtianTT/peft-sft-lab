#!/usr/bin/env python3
"""Join exact module utility with functional, gradient, structural, and p99 features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


CASES = {
    "qwen_magicoder": ("Qwen3-8B", "magicoder"),
    "qwen_commonsense": ("Qwen3-8B", "commonsense"),
    "llama_tulu": ("Llama-3.1-8B-Instruct", "tulu"),
}


def canonical(name: str) -> str:
    position = name.find("layers.")
    if position < 0:
        raise ValueError(name)
    return name[position:]


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    output = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        output[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return output


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    x, y = ranks(left), ranks(right)
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else float("nan")


def design(rows: list[dict], numeric: tuple[str, ...], module_types: list[str]) -> tuple[np.ndarray, list[str]]:
    columns, names = [], []
    for feature in numeric:
        values = np.asarray([float(row[feature]) for row in rows])
        if feature == "p99_contribution":
            values = np.log10(np.maximum(values, 1e-12))
        columns.append(values)
        names.append(feature)
    for module_type in module_types[1:]:
        columns.append(np.asarray([row["module_type"] == module_type for row in rows], dtype=float))
        names.append(f"module_type={module_type}")
    return np.stack(columns, axis=1), names


def ridge_predict(x_train, y_train, x_test, alpha: float = 1.0):
    mean, scale = x_train.mean(0), x_train.std(0)
    scale[scale < 1e-12] = 1.0
    train, test = (x_train - mean) / scale, (x_test - mean) / scale
    y_mean = y_train.mean()
    coef = np.linalg.solve(train.T @ train + alpha * np.eye(train.shape[1]), train.T @ (y_train - y_mean))
    return y_mean + test @ coef, coef


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - prediction) ** 2) / denominator) if denominator else float("nan")


def model_metrics(
    rows: list[dict], numeric: tuple[str, ...], rng: np.random.Generator,
    target_key: str = "utility",
) -> dict:
    module_types = sorted({row["module_type"] for row in rows})
    x, names = design(rows, numeric, module_types)
    y = np.asarray([float(row[target_key]) for row in rows])
    fitted, coefficients = ridge_predict(x, y, x)

    fold_id = rng.permutation(len(rows)) % 5
    cv_prediction = np.empty(len(rows))
    for fold in range(5):
        train, test = fold_id != fold, fold_id == fold
        cv_prediction[test], _ = ridge_predict(x[train], y[train], x[test])

    max_layer = max(int(row["layer"]) for row in rows)
    layer_fold = np.asarray([min(3, int(row["layer"]) * 4 // (max_layer + 1)) for row in rows])
    layer_prediction = np.empty(len(rows))
    for fold in range(4):
        train, test = layer_fold != fold, layer_fold == fold
        layer_prediction[test], _ = ridge_predict(x[train], y[train], x[test])
    return {
        "in_sample_r2": r2(y, fitted),
        "random_5fold_r2": r2(y, cv_prediction),
        "leave_layer_quartile_out_r2": r2(y, layer_prediction),
        "coefficients": dict(zip(names, coefficients)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate_root", required=True)
    parser.add_argument("--utility_root", required=True)
    parser.add_argument("--gradient_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    aggregate, utility_root = Path(args.aggregate_root), Path(args.utility_root)
    gradient_root, output = Path(args.gradient_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    direction = defaultdict(dict)
    with (aggregate / "direction_response.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["adapter"] == "lora" and int(row["direction"]) == 1:
                direction[(row["base_model"], row["task"])][canonical(row["module"])] = float(row["response_energy_share"])
    raw = defaultdict(dict)
    with (aggregate / "module_spectra.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw[(row["base_model"], row["task"])][canonical(row["module"])] = float(row["lora_top1_energy_share"])
    p99 = defaultdict(dict)
    with (aggregate / "module_response.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            p99[(row["base_model"], row["task"])][canonical(row["module"])] = float(row["lora_p99"])

    feature_rows = []
    for case, key in CASES.items():
        utilities = read_tsv(utility_root / case / "module_utility.tsv")
        gradient_meta = json.loads((gradient_root / case / "spectral_edit_meta.json").read_text())
        gradients = {canonical(name): values for name, values in gradient_meta["module_selection"].items()}
        for row in utilities:
            module = canonical(row["module"])
            feature_rows.append({
                "case": case, "base_model": key[0], "task": key[1], "module": module,
                "layer": int(row["layer"]), "module_type": row["module_type"],
                "utility": float(row["utility"]), "utility_ci_low": float(row["utility_ci_low"]),
                "utility_ci_high": float(row["utility_ci_high"]),
                "functional_concentration": direction[key][module],
                "raw_concentration": raw[key][module],
                "gradient_compatibility": float(gradients[module]["compatibility"]),
                "gradient_importance": float(gradients[module]["importance"]),
                "p99_contribution": p99[key][module],
            })
    for case in CASES:
        selected = [row for row in feature_rows if row["case"] == case]
        values = np.asarray([float(row["utility"]) for row in selected])
        scale = float(values.std()) or 1.0
        for row, value in zip(selected, values):
            row["utility_z_within_case"] = (float(value) - float(values.mean())) / scale
    write_tsv(output / "module_utility_features.tsv", feature_rows)

    correlation_rows, model_rows, coefficient_rows = [], [], []
    model_specs = {
        "structure": ("layer",),
        "structure_plus_F": ("layer", "functional_concentration"),
        "structure_plus_C": ("layer", "gradient_compatibility"),
        "structure_plus_p99": ("layer", "p99_contribution"),
        "full": ("layer", "functional_concentration", "gradient_compatibility", "p99_contribution"),
    }
    groups = [(case, [row for row in feature_rows if row["case"] == case]) for case in CASES]
    groups.append(("pooled", feature_rows))
    for case, rows in groups:
        target_key = "utility_z_within_case" if case == "pooled" else "utility"
        utility = np.asarray([float(row[target_key]) for row in rows])
        for feature in ("functional_concentration", "raw_concentration", "gradient_compatibility", "p99_contribution"):
            values = np.asarray([float(row[feature]) for row in rows])
            observed = spearman(values, utility)
            exceed = 0
            for _ in range(args.permutations):
                exceed += abs(spearman(values, rng.permutation(utility))) >= abs(observed) - 1e-15
            correlation_rows.append({
                "case": case, "feature": feature, "spearman_rho": observed,
                "permutation_p": (exceed + 1) / (args.permutations + 1), "modules": len(rows),
            })
        for label, numeric in model_specs.items():
            result = model_metrics(rows, numeric, rng, target_key=target_key)
            model_rows.append({
                "case": case, "model": label, "features": ",".join(numeric) + ",module_type",
                "in_sample_r2": result["in_sample_r2"],
                "random_5fold_r2": result["random_5fold_r2"],
                "leave_layer_quartile_out_r2": result["leave_layer_quartile_out_r2"],
                "modules": len(rows),
            })
            if label == "full":
                for name, value in result["coefficients"].items():
                    coefficient_rows.append({"case": case, "feature": name, "standardized_ridge_coefficient": value})
    write_tsv(output / "utility_correlations.tsv", correlation_rows)
    write_tsv(output / "utility_models.tsv", model_rows)
    write_tsv(output / "utility_full_coefficients.tsv", coefficient_rows)
    (output / "metadata.json").write_text(json.dumps({
        "utility_definition": "held-out paired example NLL improvement from an exact single-module observed-HNS intervention",
        "model": "ridge(alpha=1) with module type fixed effects; predictive R2 is out-of-fold",
        "permutations": args.permutations, "seed": args.seed,
    }, indent=2) + "\n")
    print(json.dumps({"module_rows": len(feature_rows), "cases": list(CASES)}, indent=2))


if __name__ == "__main__":
    main()
