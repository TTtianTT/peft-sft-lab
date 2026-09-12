#!/usr/bin/env python3
"""Prepare local Tulu, GSM8K, and IFEval assets for mechanism experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--gsm8k_predictions", required=True)
    parser.add_argument("--ifeval_outputs", required=True)
    parser.add_argument(
        "--tulu_dataset",
        default="allenai/tulu-3-sft-personas-instruction-following",
    )
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    gsm8k_path = root / "gsm8k-test.jsonl"
    if not gsm8k_path.is_file():
        rows = []
        for line in Path(args.gsm8k_predictions).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows.append({"question": row["question"], "answer": f"#### {row['gold']}"})
        write_jsonl(gsm8k_path, rows)
        print(f"[Save] {len(rows)} GSM8K examples -> {gsm8k_path}")

    ifeval_path = root / "ifeval-train.jsonl"
    if not ifeval_path.is_file():
        rows = []
        for line in Path(args.ifeval_outputs).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            instructions = list(row["inst_results"])
            rows.append({
                "key": row["key"],
                "prompt": row["prompt"],
                "instruction_id_list": [item["instruction_id"] for item in instructions],
                "kwargs": [item.get("kwargs", {}) for item in instructions],
            })
        write_jsonl(ifeval_path, rows)
        print(f"[Save] {len(rows)} IFEval examples -> {ifeval_path}")

    tulu_path = root / "tulu-3-sft-personas-instruction-following-train.parquet"
    if not tulu_path.is_file():
        from datasets import load_dataset

        dataset = load_dataset(args.tulu_dataset, split="train")
        dataset.to_parquet(str(tulu_path))
        print(
            f"[Save] {len(dataset)} Tulu examples columns={dataset.column_names} -> {tulu_path}"
        )

    manifest = {
        "tulu_dataset": args.tulu_dataset,
        "tulu_train": str(tulu_path),
        "gsm8k_test": str(gsm8k_path),
        "ifeval": str(ifeval_path),
        "gsm8k_source_predictions": str(Path(args.gsm8k_predictions).resolve()),
        "ifeval_source_outputs": str(Path(args.ifeval_outputs).resolve()),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[Done] {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
