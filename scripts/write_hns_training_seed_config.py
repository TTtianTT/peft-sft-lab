#!/usr/bin/env python3
"""Write the three-task evaluation config for one replicated training seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASES = {
    "Qwen3-8B": "/dataset1/zailong/models/Qwen3-8B",
    "Llama-3.1-8B-Instruct": "/dataset1/zailong/models/Llama-3.1-8B-Instruct",
}

EVALUATION_TASKS = {
    "magicoder": {
        "benchmark": "HumanEval",
        "dataset": "/dataset1/zailong/data/peft-sft-lab/humaneval-test.parquet",
        "samples": 164,
        "primary_metric": "pass@1",
        "max_new_tokens": 512,
    },
    "metamath": {
        "benchmark": "GSM8K",
        "dataset": "/dataset1/zailong/data/peft-sft-lab/cross-task-mechanism/gsm8k-test.jsonl",
        "samples": 1319,
        "primary_metric": "strict_accuracy",
        "secondary_metric": "numeric_accuracy",
        "max_new_tokens": 512,
    },
    "tulu": {
        "benchmark": "IFEval",
        "dataset": "/dataset1/zailong/data/peft-sft-lab/cross-task-mechanism/ifeval-train.jsonl",
        "samples": 541,
        "primary_metric": "prompt_level_strict_accuracy",
        "max_new_tokens": 2048,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, choices=sorted(BASES))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_root = Path(args.run_root).resolve() / args.base / f"seed{args.seed}"
    checkpoints = [
        {
            "base": args.base,
            "train_task": task,
            "lora": str(seed_root / "lora" / task),
        }
        for task in ("magicoder", "metamath", "tulu")
    ]
    missing = [row["lora"] for row in checkpoints if not (Path(row["lora"]) / "adapter_config.json").is_file()]
    if missing:
        raise FileNotFoundError(f"missing completed LoRA adapters: {missing}")

    result = {
        "experiment": "hns_training_seed_replication_3task",
        "date": "2026-09-12",
        "training_seed": args.seed,
        "dataset_seed": 42,
        "bases": {args.base: BASES[args.base]},
        "checkpoints": checkpoints,
        "evaluation_tasks": EVALUATION_TASKS,
        "inference": {
            "greedy": True,
            "batch_invariant": True,
            "seed": 42,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
