#!/usr/bin/env python3
"""Build single-direction LoRA adapters interpolated toward observed all-module HNS."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key, save_lora_state_dict
from finetune.spectral_edit.mechanism import align_edited_spectrum_to_reference
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strengths", nargs="+", type=float, default=(0.1, 1.0))
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


def read_selection(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError("empty direction selection")
    return rows


def main() -> None:
    args = parse_args()
    for strength in args.strengths:
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"strength must be in (0,1], got {strength}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selection = read_selection(Path(args.selection))
    lora_state, weight_format = load_lora_state_dict(args.lora_path)
    hns_state, _ = load_lora_state_dict(args.hns_path)
    lora_pairs, hns_pairs = collect_pairs(lora_state), collect_pairs(hns_state)
    if set(lora_pairs) != set(hns_pairs):
        raise RuntimeError("LoRA/HNS module sets differ")

    requested_modules = sorted({row["module"] for row in selection})
    decompositions = {}
    for module in requested_modules:
        pair = lora_pairs[module]
        U, sigma, Vh, _ = lowrank_svd_from_ba(pair["B"][1], pair["A"][1])
        sigma_hns, basis_stats = align_edited_spectrum_to_reference(
            U,
            Vh,
            hns_pairs[module]["B"][1],
            hns_pairs[module]["A"][1],
        )
        decompositions[module] = (U, sigma, Vh, sigma_hns, basis_stats)

    variants = []
    for selection_index, row in enumerate(selection, start=1):
        module = row["module"]
        direction = int(row["direction"]) - 1
        U, sigma, Vh, sigma_hns, basis_stats = decompositions[module]
        if sigma_hns[direction] >= sigma[direction]:
            raise RuntimeError(f"selected direction is not HNS-suppressed: {module}:{direction + 1}")
        for strength in args.strengths:
            dose = int(round(100 * strength))
            class_short = "sup" if row["class"] == "predicted_suppress" else "ret"
            label = f"dir{selection_index:02d}_{class_short}_dose{dose:03d}"
            adapter_dir = output / label
            if adapter_dir.exists():
                raise FileExistsError(f"refusing to overwrite {adapter_dir}")
            shutil.copytree(args.lora_path, adapter_dir)
            edited_state = dict(lora_state)
            sigma_new = sigma.clone()
            sigma_new[direction] = sigma[direction] + strength * (
                sigma_hns[direction] - sigma[direction]
            )
            b_new, a_new = rebuild_ba_from_uv_sigma(U, Vh, sigma_new)
            key_a, old_a = lora_pairs[module]["A"]
            key_b, old_b = lora_pairs[module]["B"]
            edited_state[key_a] = a_new.to(dtype=old_a.dtype)
            edited_state[key_b] = b_new.to(dtype=old_b.dtype)
            save_lora_state_dict(str(adapter_dir), edited_state, weight_format)
            metadata = {
                "label": label,
                "selection_index": selection_index,
                "pair": int(row["pair"]),
                "class": row["class"],
                "suppress_block": row["suppress_block"],
                "module": module,
                "direction": direction + 1,
                "strength": strength,
                "sigma_lora": float(sigma[direction]),
                "sigma_hns": float(sigma_hns[direction]),
                "sigma_edited": float(sigma_new[direction]),
                "basis_stats": basis_stats,
                "source_lora": str(Path(args.lora_path).resolve()),
                "source_hns": str(Path(args.hns_path).resolve()),
            }
            (adapter_dir / "direction_dose_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
            variants.append(metadata)
            print(f"[Build] {label}: {module} sv{direction + 1} strength={strength:g}", flush=True)

    manifest = {
        "method": "single_spectral_direction_interpolation_to_observed_hns",
        "lora_path": str(Path(args.lora_path).resolve()),
        "hns_path": str(Path(args.hns_path).resolve()),
        "selection_path": str(Path(args.selection).resolve()),
        "variants": variants,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"directions": len(selection), "variants": len(variants)}, indent=2))


if __name__ == "__main__":
    main()
