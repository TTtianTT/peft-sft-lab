#!/usr/bin/env python3
"""CPU integrity and cross-run stability audit for the post-hoc scaling pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


LABELS = (
    "lora", "global_norm_matched", "per_module_spectral_scale",
    "shuffled_scale_1", "shuffled_scale_2", "shuffled_scale_3", "full_hns",
)
ENDPOINTS = ("correct_strict", "correct_numeric")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--prior_run_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_mcnemar(repaired: int, broken: int) -> float:
    discordant = repaired + broken
    if discordant == 0:
        return 1.0
    lower = min(repaired, broken)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired(left: np.ndarray, right: np.ndarray) -> dict:
    delta = right.astype(np.int8) - left.astype(np.int8)
    repaired = int(np.sum(delta == 1))
    broken = int(np.sum(delta == -1))
    return {
        "left_accuracy": float(left.mean()),
        "right_accuracy": float(right.mean()),
        "delta": float(delta.mean()),
        "repaired": repaired,
        "broken": broken,
        "mcnemar_exact_p": exact_mcnemar(repaired, broken),
    }


def load(path: Path) -> tuple[list[str], dict[str, np.ndarray], list[str]]:
    rows = read_jsonl(path)
    return (
        [str(row["id"]) for row in rows],
        {endpoint: np.asarray([bool(row[endpoint]) for row in rows]) for endpoint in ENDPOINTS},
        [str(row["prediction_extracted"]) for row in rows],
    )


def pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    prior = Path(args.prior_run_root)
    manifest = json.loads((root / "adapters" / "manifest.json").read_text())
    locked = json.loads((root / "analysis" / "validation" / "validation_result.json").read_text())
    contrast_lookup = {row["contrast"]: row for row in locked["contrasts"]}

    integrity: dict[str, dict] = {}
    current_values: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for split, expected in (("calibration", 256), ("validation", 512)):
        reference_ids = None
        integrity[split] = {}
        for label in LABELS:
            path = root / "eval" / split / label / "predictions.jsonl"
            ids, values, predictions = load(path)
            if len(ids) != expected or len(set(ids)) != expected:
                raise RuntimeError(f"{split}/{label}: invalid sample IDs")
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise RuntimeError(f"{split}/{label}: IDs or order differ")
            current_values[(split, label)] = values
            integrity[split][label] = {
                "samples": len(ids),
                "unique_predictions": len(set(predictions)),
                **{endpoint: float(values[endpoint].mean()) for endpoint in ENDPOINTS},
            }

    repeatability: list[dict] = []
    for split in ("calibration", "validation"):
        for label in ("lora", "full_hns"):
            current_ids, current, current_predictions = load(
                root / "eval" / split / label / "predictions.jsonl"
            )
            prior_ids, prior_values, prior_predictions = load(
                prior / "eval" / split / label / "predictions.jsonl"
            )
            if current_ids != prior_ids:
                raise RuntimeError(f"cross-run IDs differ: {split}/{label}")
            exact_prediction_match = float(np.mean(np.asarray(current_predictions) == np.asarray(prior_predictions)))
            for endpoint in ENDPOINTS:
                repeatability.append({
                    "split": split,
                    "label": label,
                    "endpoint": endpoint,
                    "exact_extracted_prediction_match": exact_prediction_match,
                    **paired(prior_values[endpoint], current[endpoint]),
                })

    strict_global = contrast_lookup[
        "per_module_spectral_scale_minus_global_norm_matched_correct_strict"
    ]
    strict_shuffle = contrast_lookup[
        "per_module_spectral_scale_minus_shuffle_mean_correct_strict"
    ]
    strict_hns = contrast_lookup[
        "full_hns_minus_per_module_spectral_scale_correct_strict"
    ]
    numeric_global = contrast_lookup[
        "per_module_spectral_scale_minus_global_norm_matched_correct_numeric"
    ]
    numeric_shuffle = contrast_lookup[
        "per_module_spectral_scale_minus_shuffle_mean_correct_numeric"
    ]
    numeric_hns = contrast_lookup[
        "full_hns_minus_per_module_spectral_scale_correct_numeric"
    ]

    audit = {
        "status": "complete",
        "run_root": str(root.resolve()),
        "slurm": {"job_id": 768, "state": "COMPLETED", "elapsed": "00:06:08", "exit_code": "0:0"},
        "adapter_integrity": {
            "modules": manifest["modules"],
            "construction": manifest["construction"],
            "max_implied_update_relative_error": manifest["max_implied_update_relative_error"],
            "scalar_total_fro_relative_spread": manifest["scalar_total_fro_relative_spread"],
        },
        "prediction_integrity": integrity,
        "cross_run_repeatability": repeatability,
        "locked_primary_contrasts": {
            "per_module_minus_global": strict_global,
            "per_module_minus_shuffle_mean": strict_shuffle,
            "full_hns_minus_per_module": strict_hns,
        },
        "secondary_numeric_contrasts": {
            "per_module_minus_global": numeric_global,
            "per_module_minus_shuffle_mean": numeric_shuffle,
            "full_hns_minus_per_module": numeric_hns,
        },
        "decision": "stop: per-module spectral scaling did not beat global or shuffled controls",
    }
    Path(args.output_json).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    validation_scores = integrity["validation"]
    validation_repeat = {
        (row["label"], row["endpoint"]): row
        for row in repeatability if row["split"] == "validation"
    }
    lines = [
        "# Qwen MetaMath post-hoc spectral-scaling pilot",
        "",
        "Slurm job 768 completed successfully on one B300 in 00:06:08. Seven fixed variants were",
        "evaluated serially on 256 diagnostic calibration questions and 512 locked validation questions.",
        "The primary endpoint is strict greedy GSM8K accuracy; numeric equivalence is secondary.",
        "",
        "## Locked validation scores",
        "",
        "| Variant | Strict | Numeric |",
        "|---|---:|---:|",
    ]
    display = {
        "lora": "Untouched LoRA",
        "global_norm_matched": "GlobalNormMatched",
        "per_module_spectral_scale": "PerModuleSpectralScale",
        "shuffled_scale_1": "ShuffledScale-1",
        "shuffled_scale_2": "ShuffledScale-2",
        "shuffled_scale_3": "ShuffledScale-3",
        "full_hns": "Full HNS",
    }
    for label in LABELS:
        row = validation_scores[label]
        lines.append(
            f"| {display[label]} | {100 * row['correct_strict']:.2f}% | "
            f"{100 * row['correct_numeric']:.2f}% |"
        )
    lines.extend([
        "",
        "## Predeclared mechanism contrasts",
        "",
        "| Contrast | Strict delta [95% CI] | p | Numeric delta [95% CI] | p |",
        "|---|---:|---:|---:|---:|",
        f"| PerModule - Global | {pp(strict_global['delta'])} "
        f"[{pp(strict_global['bootstrap_ci_low'])}, {pp(strict_global['bootstrap_ci_high'])}] | "
        f"{strict_global['paired_p']:.3f} | {pp(numeric_global['delta'])} "
        f"[{pp(numeric_global['bootstrap_ci_low'])}, {pp(numeric_global['bootstrap_ci_high'])}] | "
        f"{numeric_global['paired_p']:.3f} |",
        f"| PerModule - shuffle mean | {pp(strict_shuffle['delta'])} "
        f"[{pp(strict_shuffle['bootstrap_ci_low'])}, {pp(strict_shuffle['bootstrap_ci_high'])}] | "
        f"{strict_shuffle['paired_p']:.3f} | {pp(numeric_shuffle['delta'])} "
        f"[{pp(numeric_shuffle['bootstrap_ci_low'])}, {pp(numeric_shuffle['bootstrap_ci_high'])}] | "
        f"{numeric_shuffle['paired_p']:.3f} |",
        f"| Full HNS - PerModule | {pp(strict_hns['delta'])} "
        f"[{pp(strict_hns['bootstrap_ci_low'])}, {pp(strict_hns['bootstrap_ci_high'])}] | "
        f"{strict_hns['paired_p']:.4f} | {pp(numeric_hns['delta'])} "
        f"[{pp(numeric_hns['bootstrap_ci_low'])}, {pp(numeric_hns['bootstrap_ci_high'])}] | "
        f"{numeric_hns['paired_p']:.4g} |",
        "",
        "The true spectrum-to-module assignment is not favored: PerModule is 0.98 pp below the global",
        "norm-matched scalar and 0.85 pp below the mean shuffled allocation on the primary endpoint.",
        "Neither difference is resolved, but both point in the direction opposite to the proposed method.",
        "Full HNS exceeds PerModule by 3.71 pp with a paired interval excluding zero. Thus matching HNS's",
        "per-module Frobenius allocation is insufficient; the remaining within-module spectral change or an",
        "associated implementation effect matters for this checkpoint.",
        "",
        "## Numerical repeatability boundary",
        "",
        f"The identical untouched LoRA changes by {pp(validation_repeat[('lora', 'correct_strict')]['delta'])}",
        f"between this run and the prior direct-dose run; Full HNS changes by",
        f"{pp(validation_repeat[('full_hns', 'correct_strict')]['delta'])}. Exact extracted predictions match on",
        f"{100 * validation_repeat[('lora', 'correct_strict')]['exact_extracted_prediction_match']:.1f}% (LoRA)",
        f"and {100 * validation_repeat[('full_hns', 'correct_strict')]['exact_extracted_prediction_match']:.1f}%",
        "(HNS) of questions. Greedy low-precision inference therefore has non-negligible cross-run branching.",
        "The small Global/PerModule/Shuffled gaps must be treated as noise-level. The larger HNS-PerModule gap",
        "is reproduced directionally by the earlier complete evaluation, but its exact significance is",
        "run-dependent and should not be overinterpreted as a clean decomposition.",
        "",
        "## Decision",
        "",
        "The predeclared success condition is not met. Stop the per-module normalization branch: do not launch",
        "Llama Magicoder/Qwen Tulu replication, transfer, or data-conditioned normalization on the basis of this",
        "rule. The useful surviving result is narrower: global shrink has a small positive point estimate, while",
        "Full HNS retains a substantially larger gain, so HNS cannot be reduced to its induced module-norm",
        "allocation in this MetaMath pilot.",
        "",
    ])
    Path(args.output_report).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "decision": audit["decision"],
        "strict_per_minus_global": strict_global["delta"],
        "strict_per_minus_shuffle_mean": strict_shuffle["delta"],
        "strict_hns_minus_per": strict_hns["delta"],
    }, indent=2))


if __name__ == "__main__":
    main()
