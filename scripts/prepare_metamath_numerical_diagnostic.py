#!/usr/bin/env python3
"""Create a fixed diagnostic set from known unstable and stable GSM8K questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current_eval", required=True)
    parser.add_argument("--prior_eval", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stable_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def predictions(root: Path, label: str) -> dict[str, str]:
    rows = read_jsonl(root / label / "predictions.jsonl")
    return {str(row["id"]): str(row["prediction_extracted"]) for row in rows}


def main() -> None:
    args = parse_args()
    current = Path(args.current_eval)
    prior = Path(args.prior_eval)
    split_rows = read_jsonl(Path(args.split))
    by_id = {str(row["id"]): row for row in split_rows}
    current_lora = predictions(current, "lora")
    current_hns = predictions(current, "full_hns")
    prior_lora = predictions(prior, "lora")
    prior_hns = predictions(prior, "full_hns")
    if not (set(by_id) == set(current_lora) == set(current_hns) == set(prior_lora) == set(prior_hns)):
        raise RuntimeError("diagnostic source IDs differ")
    unstable = sorted(
        identity for identity in by_id
        if current_lora[identity] != prior_lora[identity] or current_hns[identity] != prior_hns[identity]
    )
    stable = sorted(set(by_id) - set(unstable))
    rng = np.random.default_rng(args.seed)
    chosen_stable = sorted(rng.choice(stable, size=args.stable_samples, replace=False).tolist())
    chosen = unstable + chosen_stable
    output_rows = []
    for identity in chosen:
        row = dict(by_id[identity])
        row["diagnostic_group"] = "known_cross_run_unstable" if identity in unstable else "fixed_random_stable"
        output_rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "complete",
        "seed": args.seed,
        "known_unstable": len(unstable),
        "fixed_random_stable": len(chosen_stable),
        "samples": len(output_rows),
        "ids": chosen,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
