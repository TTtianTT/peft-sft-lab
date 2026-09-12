#!/usr/bin/env python3
"""Select matched PIQA spectral directions with opposite signed reward sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--max_direction", type=int, default=8)
    parser.add_argument("--layer_bins", type=int, default=4)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def numeric(row: dict) -> dict:
    converted = dict(row)
    for key in (
        "layer",
        "direction",
        "sigma_lora",
        "sigma_hns",
        "hns_gate",
        "hns_suppression_fraction",
        "lora_trajectory_functional_energy",
        "lora_trajectory_functional_share",
        "gate_reward_gradient",
        "gate_reward_gradient_ci_low",
        "gate_reward_gradient_ci_high",
        "predicted_hns_margin_gain",
        "predicted_hns_margin_gain_ci_low",
        "predicted_hns_margin_gain_ci_high",
        "old_module_sft_compatibility",
        "old_module_sft_importance",
    ):
        converted[key] = float(row[key])
    converted["layer"] = int(converted["layer"])
    converted["direction"] = int(converted["direction"])
    converted["hns_suppressed"] = row["hns_suppressed"].lower() == "true"
    converted["key"] = f"{row['module']}::sv{converted['direction']}"
    return converted


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [numeric(row) for row in read_tsv(Path(args.signals))]
    max_layer = max(row["layer"] for row in rows)
    for row in rows:
        row["layer_bin"] = min(args.layer_bins - 1, row["layer"] * args.layer_bins // (max_layer + 1))
        row["edit_energy"] = (
            row["lora_trajectory_functional_energy"] * row["hns_suppression_fraction"] ** 2
        )
        width = row["predicted_hns_margin_gain_ci_high"] - row["predicted_hns_margin_gain_ci_low"]
        row["signal_to_ci_width"] = abs(row["predicted_hns_margin_gain"]) / max(width, 1e-30)

    candidates = [
        row
        for row in rows
        if row["hns_suppressed"] and row["direction"] <= args.max_direction
    ]
    suppress = [row for row in candidates if row["predicted_hns_margin_gain_ci_low"] > 0]
    retain = [row for row in candidates if row["predicted_hns_margin_gain_ci_high"] < 0]
    if len(suppress) < args.pairs or len(retain) < args.pairs:
        raise RuntimeError(
            "Insufficient CI-resolved opposite-sign directions; refusing a forced median split: "
            f"suppress={len(suppress)} retain={len(retain)} requested={args.pairs}"
        )

    features = np.array(
        [
            [
                math.log(max(row["lora_trajectory_functional_energy"], 1e-30)),
                math.log(max(row["edit_energy"], 1e-30)),
                row["hns_suppression_fraction"],
            ]
            for row in candidates
        ]
    )
    scale = np.std(features, axis=0)
    scale[scale < 1e-12] = 1.0

    pair_options = []
    for positive in suppress:
        for negative in retain:
            if (positive["module_type"], positive["layer_bin"]) != (
                negative["module_type"],
                negative["layer_bin"],
            ):
                continue
            a = np.array(
                [
                    math.log(max(positive["lora_trajectory_functional_energy"], 1e-30)),
                    math.log(max(positive["edit_energy"], 1e-30)),
                    positive["hns_suppression_fraction"],
                ]
            )
            b = np.array(
                [
                    math.log(max(negative["lora_trajectory_functional_energy"], 1e-30)),
                    math.log(max(negative["edit_energy"], 1e-30)),
                    negative["hns_suppression_fraction"],
                ]
            )
            distance = float(np.linalg.norm((a - b) / scale))
            confidence_bonus = positive["signal_to_ci_width"] + negative["signal_to_ci_width"]
            pair_options.append((distance - 0.05 * confidence_bonus, distance, positive, negative))
    pair_options.sort(key=lambda value: (value[0], value[2]["key"], value[3]["key"]))

    selected_pairs = []
    used_positive, used_negative = set(), set()
    for _, distance, positive, negative in pair_options:
        if positive["key"] in used_positive or negative["key"] in used_negative:
            continue
        selected_pairs.append((positive, negative, distance))
        used_positive.add(positive["key"])
        used_negative.add(negative["key"])
        if len(selected_pairs) == args.pairs:
            break
    if len(selected_pairs) != args.pairs:
        raise RuntimeError(
            f"Only {len(selected_pairs)} exact layer-bin/type matched pairs available; requested {args.pairs}"
        )

    selected_rows = []
    for pair_index, (positive, negative, distance) in enumerate(selected_pairs):
        block = "A" if pair_index % 2 == 0 else "B"
        for class_name, row in (("predicted_suppress", positive), ("predicted_retain", negative)):
            selected_rows.append(
                {
                    "pair": pair_index + 1,
                    "class": class_name,
                    "suppress_block": block if class_name == "predicted_suppress" else "",
                    "match_distance": distance,
                    **{key: value for key, value in row.items() if key != "key"},
                }
            )

    with (output / "selected_directions.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(selected_rows)
    payload = {
        "definition": {
            "predicted_suppress": "95% bootstrap CI of first-order HNS-path margin gain is above zero",
            "predicted_retain": "95% bootstrap CI of first-order HNS-path margin gain is below zero",
        },
        "candidate_filter": {
            "hns_suppressed": True,
            "max_direction": args.max_direction,
            "candidate_count": len(candidates),
            "ci_resolved_suppress_count": len(suppress),
            "ci_resolved_retain_count": len(retain),
        },
        "matching": "exact module type and layer quartile; nearest functional energy, HNS edit energy, and suppression fraction",
        "pairs": args.pairs,
        "directions": selected_rows,
    }
    (output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"pairs": args.pairs, **payload["candidate_filter"]}, indent=2))


if __name__ == "__main__":
    main()
