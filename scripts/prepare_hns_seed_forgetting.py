#!/usr/bin/env python3
"""Validate seed-grid inputs and add the existing Commonsense retention benchmark."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seed_config", "reference_config", "variant_manifest", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.seed_config).read_text())
    reference = json.loads(Path(args.reference_config).read_text())
    manifest = json.loads(Path(args.variant_manifest).read_text())
    variants = manifest["variants"]
    assert len(variants) == 33, len(variants)
    assert len({v["label"] for v in variants}) == 33
    assert {v["train_task"] for v in variants} == {"magicoder", "metamath", "tulu"}
    for variant in variants:
        path = Path(variant["path"]) / "adapter_model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg["evaluation_tasks"]["commonsense"] = reference["evaluation_tasks"]["commonsense"]
    for task, spec in cfg["evaluation_tasks"].items():
        if task == "commonsense":
            for subtask in spec["tasks"]:
                path = Path(spec["reference_predictions_root"]) / subtask / "predictions.jsonl"
                if not path.is_file():
                    raise FileNotFoundError(path)
        elif not Path(spec["dataset"]).is_file():
            raise FileNotFoundError(spec["dataset"])
    cfg["experiment"] = "hns_training_seed_forgetting_3train_4eval"
    cfg["retention_protocol"] = {
        "train_tasks": ["magicoder", "metamath", "tulu"],
        "evaluation_tasks": ["magicoder", "metamath", "tulu", "commonsense"],
        "off_task_aggregation": "equal weight over the other three benchmark families",
        "forgetting_gap": "max(base_score - adapter_score, 0), in percentage points",
        "controls": ["base", "original_lora", "svd_0plus0"],
        "scalar_controls_included": False,
        "cells_per_seed": 136,
        "fresh_diagonal_evaluation": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[Validated] 33 adapters, four benchmark families: {output}")


if __name__ == "__main__":
    main()
