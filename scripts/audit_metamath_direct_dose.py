#!/usr/bin/env python3
"""Post-hoc integrity and sensitivity audit for the locked MetaMath direct-dose experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DOSE_LABELS = ("lora", "head_0p25", "head_0p50", "head_1p00")
ENDPOINTS = ("correct_strict", "correct_numeric")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_root", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_report", required=True)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260911)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arrays(root: Path, split: str, label: str) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = read_jsonl(root / "eval" / split / label / "predictions.jsonl")
    return rows, {endpoint: np.asarray([row[endpoint] for row in rows], dtype=bool) for endpoint in ENDPOINTS}


def paired(left: np.ndarray, right: np.ndarray, rng: np.random.Generator, draws: int) -> dict:
    delta = right.astype(np.int8) - left.astype(np.int8)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    dist = delta[indices].mean(axis=1)
    return {
        "left_accuracy": float(left.mean()),
        "right_accuracy": float(right.mean()),
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(dist, 0.025)),
        "ci_high": float(np.quantile(dist, 0.975)),
        "repaired": int(np.sum(delta == 1)),
        "broken": int(np.sum(delta == -1)),
    }


def percent(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    rng = np.random.default_rng(args.seed)
    split_manifest = json.loads((root / "splits" / "split_manifest.json").read_text())
    adapter_manifest = json.loads((root / "adapters" / "manifest.json").read_text())
    selection = json.loads((root / "analysis" / "calibration" / "selection.json").read_text())

    calibration_labels = (
        "lora", "zero_rebuild", "head_0p25", "scalar_0p25",
        "head_0p50", "scalar_0p50", "head_1p00", "scalar_1p00", "full_hns",
    )
    calibration: dict[str, dict[str, np.ndarray]] = {}
    calibration_rows: dict[str, list[dict]] = {}
    ids = None
    for label in calibration_labels:
        rows, values = arrays(root, "calibration", label)
        current_ids = [row["id"] for row in rows]
        if ids is None:
            ids = current_ids
        elif current_ids != ids:
            raise RuntimeError(f"calibration IDs differ for {label}")
        calibration[label] = values
        calibration_rows[label] = rows

    validation_labels = ("lora", "full_hns", "selected_head", "matched_scalar")
    validation: dict[str, dict[str, np.ndarray]] = {}
    validation_ids = None
    for label in validation_labels:
        rows, values = arrays(root, "validation", label)
        current_ids = [row["id"] for row in rows]
        if validation_ids is None:
            validation_ids = current_ids
        elif current_ids != validation_ids:
            raise RuntimeError(f"validation IDs differ for {label}")
        validation[label] = values

    zero_contrasts = {
        endpoint: paired(calibration["lora"][endpoint], calibration["zero_rebuild"][endpoint], rng, args.bootstrap)
        for endpoint in ENDPOINTS
    }
    zero_flips = []
    for left, right in zip(calibration_rows["lora"], calibration_rows["zero_rebuild"]):
        if left["correct_strict"] != right["correct_strict"]:
            zero_flips.append({
                "id": left["id"],
                "gold": left["gold"],
                "lora_prediction": left["prediction_extracted"],
                "rebuild_prediction": right["prediction_extracted"],
                "delta": int(right["correct_strict"]) - int(left["correct_strict"]),
            })

    rebuild_matched: list[dict] = []
    for label in calibration_labels:
        if not (label.startswith("head_") or label.startswith("scalar_")):
            continue
        for endpoint in ENDPOINTS:
            rebuild_matched.append({
                "contrast": f"{label}_minus_zero_rebuild_{endpoint}",
                "endpoint": endpoint,
                **paired(calibration["zero_rebuild"][endpoint], calibration[label][endpoint], rng, args.bootstrap),
            })

    # Reproduce the locked strict selection rule under question bootstrap and
    # leave-one-question-out perturbations. This is sensitivity analysis, not
    # a new dose-selection rule.
    dose_matrix = np.stack([calibration[label]["correct_strict"] for label in DOSE_LABELS]).astype(float)
    indices = rng.integers(0, dose_matrix.shape[1], size=(args.bootstrap, dose_matrix.shape[1]))
    bootstrap_scores = dose_matrix[:, indices].mean(axis=2)
    bootstrap_winners = np.argmax(bootstrap_scores, axis=0)
    bootstrap_probability = {
        label: float(np.mean(bootstrap_winners == index)) for index, label in enumerate(DOSE_LABELS)
    }
    loo_winners = []
    for held_out in range(dose_matrix.shape[1]):
        loo_winners.append(int(np.argmax(np.delete(dose_matrix, held_out, axis=1).mean(axis=1))))
    loo_counts = {
        label: int(np.sum(np.asarray(loo_winners) == index)) for index, label in enumerate(DOSE_LABELS)
    }

    validation_contrasts: list[dict] = []
    for left_label, right_label in (
        ("lora", "full_hns"),
        ("lora", "selected_head"),
        ("lora", "matched_scalar"),
        ("matched_scalar", "selected_head"),
        ("selected_head", "full_hns"),
        ("matched_scalar", "full_hns"),
    ):
        for endpoint in ENDPOINTS:
            validation_contrasts.append({
                "contrast": f"{right_label}_minus_{left_label}_{endpoint}",
                "left": left_label,
                "right": right_label,
                "endpoint": endpoint,
                **paired(validation[left_label][endpoint], validation[right_label][endpoint], rng, args.bootstrap),
            })

    # Preserve the precomputed, locked intervals for contrasts produced by the
    # primary analyzer.  The audit adds new contrasts, but must not replace a
    # locked Monte Carlo interval with a second bootstrap draw.
    locked_path = root / "analysis" / "validation" / "validation_contrasts.tsv"
    with locked_path.open(encoding="utf-8", newline="") as handle:
        locked_rows = {row["contrast"]: row for row in csv.DictReader(handle, delimiter="\t")}
    for row in validation_contrasts:
        locked = locked_rows.get(row["contrast"])
        if locked is None:
            continue
        for key in ("left_accuracy", "right_accuracy", "delta", "bootstrap_ci_low", "bootstrap_ci_high"):
            target = {"bootstrap_ci_low": "ci_low", "bootstrap_ci_high": "ci_high"}.get(key, key)
            row[target] = float(locked[key])
        row["repaired"] = int(locked["repaired"])
        row["broken"] = int(locked["broken"])

    audit = {
        "status": "complete",
        "scope": "post-hoc CPU integrity/sensitivity audit; locked selection and validation remain unchanged",
        "run_root": str(root.resolve()),
        "split_integrity": {
            "calibration_samples": len(ids or []),
            "validation_samples": len(validation_ids or []),
            "calibration_validation_disjoint": not bool(set(ids or []) & set(validation_ids or [])),
            "excluded_reward_gradient_questions": split_manifest["excluded_reward_gradient_questions"],
            "train_overlap": split_manifest["exact_normalized_train_overlap"],
        },
        "adapter_integrity": {
            "modules": adapter_manifest["modules"],
            "max_hns_basis_error": adapter_manifest["max_hns_basis_error"],
            "max_zero_dose_reconstruction_relative_error": adapter_manifest[
                "max_zero_dose_reconstruction_relative_error"
            ],
        },
        "chosen_dose": selection["chosen_dose"],
        "selection_sensitivity": {
            "bootstrap_probability": bootstrap_probability,
            "leave_one_question_out_counts": loo_counts,
            "note": "post-hoc sensitivity only; does not change the locked alpha=1 selection",
        },
        "zero_rebuild": {"contrasts": zero_contrasts, "strict_flip_details": zero_flips},
        "calibration_vs_zero_rebuild": rebuild_matched,
        "validation_contrasts": validation_contrasts,
    }
    Path(args.output_json).write_text(json.dumps(audit, indent=2) + "\n")

    lookup = {row["contrast"]: row for row in validation_contrasts}
    strict_head = lookup["selected_head_minus_lora_correct_strict"]
    strict_scalar = lookup["matched_scalar_minus_lora_correct_strict"]
    strict_shape = lookup["selected_head_minus_matched_scalar_correct_strict"]
    numeric_head = lookup["selected_head_minus_lora_correct_numeric"]
    numeric_scalar = lookup["matched_scalar_minus_lora_correct_numeric"]
    numeric_shape = lookup["selected_head_minus_matched_scalar_correct_numeric"]
    full_vs_head_strict = lookup["full_hns_minus_selected_head_correct_strict"]
    full_vs_head_numeric = lookup["full_hns_minus_selected_head_correct_numeric"]

    lines = [
        "# Qwen MetaMath direct HeadOnly-dose result",
        "",
        "This experiment was specified separately from the closed reward-gradient pilot. Slurm 766 completed",
        "successfully with one B300. The locked primary endpoint is strict greedy GSM8K accuracy; decimal-numeric",
        "equivalence is secondary.",
        "",
        "## Selection and locked result",
        "",
        "Calibration selected the full HeadOnly dose (alpha=1): strict accuracies for alpha 0/0.25/0.5/1 were",
        "86.72%, 86.33%, 86.72%, and 88.28%. Alpha=1 remains selected in all 256 leave-one-question-out",
        f"datasets, but only {100 * bootstrap_probability['head_1p00']:.1f}% of question bootstraps; dose selection is",
        "directionally stable to single examples but still sampling-uncertain.",
        "",
        "| Locked contrast | Strict delta [95% CI] | Numeric delta [95% CI] |",
        "|---|---:|---:|",
        f"| HeadOnly(alpha=1) - LoRA | {percent(strict_head['delta'])} [{percent(strict_head['ci_low'])}, {percent(strict_head['ci_high'])}] | {percent(numeric_head['delta'])} [{percent(numeric_head['ci_low'])}, {percent(numeric_head['ci_high'])}] |",
        f"| Matched Scalar - LoRA | {percent(strict_scalar['delta'])} [{percent(strict_scalar['ci_low'])}, {percent(strict_scalar['ci_high'])}] | {percent(numeric_scalar['delta'])} [{percent(numeric_scalar['ci_low'])}, {percent(numeric_scalar['ci_high'])}] |",
        f"| HeadOnly - Matched Scalar | {percent(strict_shape['delta'])} [{percent(strict_shape['ci_low'])}, {percent(strict_shape['ci_high'])}] | {percent(numeric_shape['delta'])} [{percent(numeric_shape['ci_low'])}, {percent(numeric_shape['ci_high'])}] |",
        f"| Full HNS - HeadOnly | {percent(full_vs_head_strict['delta'])} [{percent(full_vs_head_strict['ci_low'])}, {percent(full_vs_head_strict['ci_high'])}] | {percent(full_vs_head_numeric['delta'])} [{percent(full_vs_head_numeric['ci_low'])}, {percent(full_vs_head_numeric['ci_high'])}] |",
        "",
        "The full-dose HeadOnly point estimate is positive on the independent validation split, and Full HNS is",
        "essentially tied with it. However, the matched scalar captures most of the point-estimate gain (75% strict,",
        "73% numeric), while the direct HeadOnly-minus-scalar intervals cross zero. This supports suppression/scale",
        "as a useful finite-dose regime but does not resolve a spectral-shape benefit beyond overall magnitude reduction.",
        "",
        "## Reconstruction warning",
        "",
        f"The alpha=0 rebuilt adapter has maximum relative module reconstruction error {adapter_manifest['max_zero_dose_reconstruction_relative_error']:.2e},",
        f"yet changes six calibration decisions and has strict delta {percent(zero_contrasts['correct_strict']['delta'])}",
        f"[{percent(zero_contrasts['correct_strict']['ci_low'])}, {percent(zero_contrasts['correct_strict']['ci_high'])}] versus",
        "the untouched LoRA. Autoregressive outputs are therefore sensitive to tiny factor-reconstruction changes.",
        "Comparisons against untouched LoRA carry this implementation perturbation; HeadOnly versus its matched scalar",
        "is cleaner because both adapters use the same SVD reconstruction path.",
        "",
        "## Mechanism decision",
        "",
        "The experiment improves whether/how-strongly calibration: the predeclared procedure chose alpha=1 and its",
        "locked point estimate was positive. It does not establish HeadOnly > ScalarShrink, so it is premature to infer",
        "that dominant-mode shape suppression rather than LoRA magnitude reduction causes the gain. Per protocol, stop",
        "here and do not proceed to module/direction localization from this result alone.",
        "",
    ]
    Path(args.output_report).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "chosen_dose": selection["chosen_dose"],
        "bootstrap_select_alpha_1": bootstrap_probability["head_1p00"],
        "strict_head_gain": strict_head["delta"],
        "strict_head_minus_scalar": strict_shape["delta"],
        "numeric_head_gain": numeric_head["delta"],
        "numeric_head_minus_scalar": numeric_shape["delta"],
    }, indent=2))


if __name__ == "__main__":
    main()
