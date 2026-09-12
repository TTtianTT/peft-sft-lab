#!/usr/bin/env python3
"""Lock disjoint GSM8K calibration/validation splits for direct HNS dose selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--metamath_train", required=True)
    p.add_argument("--exclude_manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--calibration", type=int, default=256)
    p.add_argument("--validation", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260911)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(rows: list[dict]) -> str:
    payload = "\n".join(f"{row['id']}\t{row['question']}\t{row['answer']}" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "split_manifest.json"
    if manifest_path.is_file():
        print(f"[Skip] locked split exists: {manifest_path}")
        return

    source = read_jsonl(Path(args.gsm8k))
    excluded_payload = json.loads(Path(args.exclude_manifest).read_text())
    excluded = {int(index) for index in excluded_payload["selected_source_indices"]}
    train = pq.read_table(args.metamath_train, columns=["query", "original_question"]).to_pydict()
    train_questions = {
        normalize(value)
        for column in ("query", "original_question")
        for value in train[column]
        if value is not None and str(value).strip()
    }
    overlap = {index for index, row in enumerate(source) if normalize(row["question"]) in train_questions}
    eligible = sorted(set(range(len(source))) - excluded - overlap)
    requested = args.calibration + args.validation
    if len(eligible) < requested:
        raise RuntimeError(f"only {len(eligible)} eligible questions for {requested} requested")
    selected = np.random.default_rng(args.seed).permutation(eligible)[:requested].tolist()
    split_indices = {
        "calibration": selected[: args.calibration],
        "validation": selected[args.calibration :],
    }
    split_rows: dict[str, list[dict]] = {}
    for split, indices in split_indices.items():
        rows = [
            {
                "id": f"gsm8k-test-{index:04d}",
                "source_index": index,
                "question": str(source[index]["question"]),
                "answer": str(source[index]["answer"]),
            }
            for index in indices
        ]
        split_rows[split] = rows
        write_jsonl(output / f"{split}.jsonl", rows)

    if set(split_indices["calibration"]) & set(split_indices["validation"]):
        raise RuntimeError("calibration and validation splits overlap")
    manifest = {
        "status": "locked",
        "source": str(Path(args.gsm8k).resolve()),
        "exclude_manifest": str(Path(args.exclude_manifest).resolve()),
        "metamath_train": str(Path(args.metamath_train).resolve()),
        "seed": args.seed,
        "source_questions": len(source),
        "excluded_reward_gradient_questions": len(excluded),
        "exact_normalized_train_overlap": len(overlap),
        "eligible_questions": len(eligible),
        "splits": {
            split: {
                "samples": len(rows),
                "source_indices": split_indices[split],
                "ids": [row["id"] for row in rows],
                "sha256": digest(rows),
            }
            for split, rows in split_rows.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "calibration": len(split_rows["calibration"]),
        "validation": len(split_rows["validation"]),
        "excluded": len(excluded),
        "train_overlap": len(overlap),
    }, indent=2))


if __name__ == "__main__":
    main()
