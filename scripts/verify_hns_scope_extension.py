#!/usr/bin/env python3
"""Verify that an all-module HNS adapter is a scope-only extension of a partial one."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key


def factors(path: str) -> dict[str, dict[str, torch.Tensor]]:
    state, _ = load_lora_state_dict(path)
    result: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, value in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is not None:
            prefix, which, _ = parsed
            result[prefix][which] = value.float()
    return {prefix: pair for prefix, pair in result.items() if {"A", "B"} <= pair.keys()}


def lowrank_norm_sq(pair: dict[str, torch.Tensor]) -> float:
    gram_b = pair["B"].T @ pair["B"]
    gram_a = pair["A"] @ pair["A"].T
    return float(torch.sum(gram_b * gram_a.T))


def lowrank_inner(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    cross_b = left["B"].T @ right["B"]
    cross_a = right["A"] @ left["A"].T
    return float(torch.sum(cross_b * cross_a.T))


def relative_delta(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    left_sq = lowrank_norm_sq(left)
    right_sq = lowrank_norm_sq(right)
    diff_sq = max(0.0, left_sq + right_sq - 2.0 * lowrank_inner(left, right))
    return math.sqrt(diff_sq) / max(math.sqrt(left_sq), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--partial_hns_path", required=True)
    parser.add_argument("--allmodule_hns_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--changed_threshold", type=float, default=1e-3)
    args = parser.parse_args()

    lora = factors(args.lora_path)
    partial = factors(args.partial_hns_path)
    allmodule = factors(args.allmodule_hns_path)
    if not (set(lora) == set(partial) == set(allmodule)):
        raise RuntimeError("LoRA, partial HNS, and all-module HNS module sets differ")

    partial_delta = {name: relative_delta(lora[name], partial[name]) for name in lora}
    allmodule_delta = {name: relative_delta(lora[name], allmodule[name]) for name in lora}
    partial_edited = [name for name, value in partial_delta.items() if value > args.changed_threshold]
    partial_untouched = [name for name in lora if name not in set(partial_edited)]
    overlap_delta = [relative_delta(partial[name], allmodule[name]) for name in partial_edited]
    payload = {
        "lora_path": str(Path(args.lora_path).resolve()),
        "partial_hns_path": str(Path(args.partial_hns_path).resolve()),
        "allmodule_hns_path": str(Path(args.allmodule_hns_path).resolve()),
        "total_modules": len(lora),
        "partial_edited_modules": len(partial_edited),
        "partial_edited_module_types": sorted({name.rsplit(".", 1)[-1] for name in partial_edited}),
        "allmodule_edited_modules": sum(value > args.changed_threshold for value in allmodule_delta.values()),
        "max_partial_untouched_vs_lora_relative_delta": max(partial_delta[name] for name in partial_untouched),
        "mean_partial_vs_allmodule_overlap_relative_delta": sum(overlap_delta) / len(overlap_delta),
        "max_partial_vs_allmodule_overlap_relative_delta": max(overlap_delta),
        "changed_threshold": args.changed_threshold,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
