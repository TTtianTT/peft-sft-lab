#!/usr/bin/env python3
"""Analyze calibration selection or locked validation for the direct MetaMath dose study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DOSES = (0.0, 0.25, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("calibration", "validation"), required=True)
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--adapter_manifest", required=True)
    p.add_argument("--lora_path", required=True)
    p.add_argument("--hns_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--selection", help="Required for validation")
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260911)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def exact_mcnemar(repaired: int, broken: int) -> float:
    discordant = repaired + broken
    if discordant == 0:
        return 1.0
    lower = min(repaired, broken)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def compare(
    label: str,
    left_label: str,
    right_label: str,
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict:
    delta = right.astype(np.int8) - left.astype(np.int8)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    repaired = int(np.sum(delta == 1))
    broken = int(np.sum(delta == -1))
    return {
        "contrast": label,
        "left": left_label,
        "right": right_label,
        "samples": len(delta),
        "left_accuracy": float(left.mean()),
        "right_accuracy": float(right.mean()),
        "delta": float(delta.mean()),
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "repaired": repaired,
        "broken": broken,
        "mcnemar_exact_p": exact_mcnemar(repaired, broken),
    }


def load_endpoint(eval_dir: Path, label: str, endpoint: str) -> tuple[list[str], np.ndarray]:
    rows = read_jsonl(eval_dir / label / "predictions.jsonl")
    ids = [str(row["id"]) for row in rows]
    return ids, np.asarray([bool(row[endpoint]) for row in rows], dtype=bool)


def dose_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter_manifest = json.loads(Path(args.adapter_manifest).read_text())
    adapter_paths = {row["label"]: row["path"] for row in adapter_manifest["variants"]}
    rng = np.random.default_rng(args.seed)

    if args.stage == "calibration":
        labels = ["lora", "zero_rebuild"]
        for dose in DOSES[1:]:
            tag = dose_tag(dose)
            labels.extend([f"head_{tag}", f"scalar_{tag}"])
        labels.append("full_hns")
    else:
        if not args.selection:
            raise ValueError("--selection is required for validation")
        labels = ["lora", "full_hns", "selected_head", "matched_scalar"]

    endpoints: dict[str, dict[str, np.ndarray]] = {}
    reference_ids: list[str] | None = None
    for label in labels:
        endpoints[label] = {}
        for endpoint in ("correct_strict", "correct_numeric"):
            ids, values = load_endpoint(eval_dir, label, endpoint)
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise RuntimeError(f"question IDs/order differ for {label}/{endpoint}")
            endpoints[label][endpoint] = values

    score_rows: list[dict] = []
    for label in labels:
        score_rows.append(
            {
                "stage": args.stage,
                "label": label,
                "samples": len(reference_ids or []),
                "strict_accuracy": float(endpoints[label]["correct_strict"].mean()),
                "numeric_accuracy": float(endpoints[label]["correct_numeric"].mean()),
                "numeric_only_corrections": int(np.sum(
                    endpoints[label]["correct_numeric"] & ~endpoints[label]["correct_strict"]
                )),
            }
        )
    write_tsv(output / f"{args.stage}_scores.tsv", score_rows)

    contrasts: list[dict] = []
    for endpoint in ("correct_strict", "correct_numeric"):
        for label in labels:
            if label == "lora":
                continue
            row = compare(
                f"{label}_minus_lora_{endpoint}",
                "lora",
                label,
                endpoints["lora"][endpoint],
                endpoints[label][endpoint],
                rng,
                args.bootstrap,
            )
            row["endpoint"] = endpoint
            contrasts.append(row)
        if args.stage == "calibration":
            for dose in DOSES[1:]:
                tag = dose_tag(dose)
                row = compare(
                    f"head_minus_scalar_{tag}_{endpoint}",
                    f"scalar_{tag}",
                    f"head_{tag}",
                    endpoints[f"scalar_{tag}"][endpoint],
                    endpoints[f"head_{tag}"][endpoint],
                    rng,
                    args.bootstrap,
                )
                row["endpoint"] = endpoint
                contrasts.append(row)
        else:
            row = compare(
                f"selected_head_minus_matched_scalar_{endpoint}",
                "matched_scalar",
                "selected_head",
                endpoints["matched_scalar"][endpoint],
                endpoints["selected_head"][endpoint],
                rng,
                args.bootstrap,
            )
            row["endpoint"] = endpoint
            contrasts.append(row)
    write_tsv(output / f"{args.stage}_contrasts.tsv", contrasts)

    if args.stage == "calibration":
        strict_scores = {0.0: float(endpoints["lora"]["correct_strict"].mean())}
        for dose in DOSES[1:]:
            strict_scores[dose] = float(endpoints[f"head_{dose_tag(dose)}"]["correct_strict"].mean())
        best = max(strict_scores.values())
        chosen = min(dose for dose, value in strict_scores.items() if value == best)
        if chosen == 0:
            selected_head_path = str(Path(args.lora_path).resolve())
            scalar_path = str(Path(args.lora_path).resolve())
        else:
            selected_head_path = adapter_paths[f"head_{dose_tag(chosen)}"]
            scalar_path = adapter_paths[f"scalar_{dose_tag(chosen)}"]
        selection = {
            "status": "locked",
            "rule": "maximize strict greedy calibration accuracy over HeadOnly doses; exact tie selects smaller dose",
            "candidate_doses": list(DOSES),
            "strict_calibration_accuracy": {str(key): value for key, value in strict_scores.items()},
            "chosen_dose": chosen,
            "selected_head_path": selected_head_path,
            "matched_scalar_path": scalar_path,
            "lora_path": str(Path(args.lora_path).resolve()),
            "full_hns_path": str(Path(args.hns_path).resolve()),
            "validation_variants": [
                {"label": "lora", "path": str(Path(args.lora_path).resolve())},
                {"label": "full_hns", "path": str(Path(args.hns_path).resolve())},
                {"label": "selected_head", "path": selected_head_path},
                {"label": "matched_scalar", "path": scalar_path},
            ],
        }
        (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2))
    else:
        selection = json.loads(Path(args.selection).read_text())
        result = {
            "status": "complete",
            "chosen_dose": selection["chosen_dose"],
            "scores": score_rows,
            "contrasts": contrasts,
        }
        (output / "validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
