#!/usr/bin/env python3
"""Verify that each dose adapter changes only its declared singular direction."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key
from finetune.spectral_edit.mechanism import align_edited_spectrum_to_reference
from finetune.spectral_edit.svd import lowrank_svd_from_ba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtol", type=float, default=5e-3)
    return parser.parse_args()


def canonical(prefix: str) -> str:
    position = prefix.find("layers.")
    if position < 0:
        raise ValueError(prefix)
    return prefix[position:]


def collect_pairs(state: dict[str, torch.Tensor]):
    pairs = defaultdict(dict)
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is None:
            continue
        prefix, which, _ = parsed
        pairs[canonical(prefix)][which] = (key, tensor)
    return {name: pair for name, pair in pairs.items() if {"A", "B"} <= pair.keys()}


def main() -> None:
    args = parse_args()
    root = Path(args.adapter_root)
    manifest = json.loads((root / "manifest.json").read_text())
    lora_state, _ = load_lora_state_dict(manifest["lora_path"])
    lora_pairs = collect_pairs(lora_state)
    decompositions = {}
    for module, pair in lora_pairs.items():
        U, sigma, Vh, _ = lowrank_svd_from_ba(pair["B"][1], pair["A"][1])
        decompositions[module] = (U, sigma, Vh)

    results = []
    for variant in manifest["variants"]:
        state, _ = load_lora_state_dict(str(root / variant["label"]))
        pairs = collect_pairs(state)
        target = variant["module"]
        unchanged_mismatches = 0
        for module in lora_pairs:
            if module == target:
                continue
            for which in ("A", "B"):
                if not torch.equal(pairs[module][which][1], lora_pairs[module][which][1]):
                    unchanged_mismatches += 1
        U, sigma, Vh = decompositions[target]
        aligned, basis = align_edited_spectrum_to_reference(
            U, Vh, pairs[target]["B"][1], pairs[target]["A"][1]
        )
        expected = sigma.clone()
        expected[int(variant["direction"]) - 1] = float(variant["sigma_edited"])
        relative_error = float(
            (torch.linalg.vector_norm(aligned - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-12)).item()
        )
        if unchanged_mismatches or relative_error > args.rtol:
            raise RuntimeError(
                f"verification failed for {variant['label']}: unchanged={unchanged_mismatches} "
                f"relative_spectrum_error={relative_error}"
            )
        results.append(
            {
                "label": variant["label"],
                "unchanged_tensor_mismatches": unchanged_mismatches,
                "relative_spectrum_error": relative_error,
                **basis,
            }
        )
    payload = {
        "status": "PASS",
        "variants": len(results),
        "max_relative_spectrum_error": max(row["relative_spectrum_error"] for row in results),
        "max_basis_offdiag_fraction": max(row["basis_offdiag_fraction"] for row in results),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
