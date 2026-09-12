#!/usr/bin/env python3
"""Build predeclared PIQA direction blocks and a global Frobenius-matched scalar control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from finetune.spectral_edit.io import (
    get_scaling_for_module,
    load_adapter_config,
    load_lora_state_dict,
    parse_lora_ab_key,
    save_lora_state_dict,
)
from finetune.spectral_edit.mechanism import align_edited_spectrum_to_reference
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--dose_decision", required=True)
    parser.add_argument("--output_dir", required=True)
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
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.selection).open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle, delimiter="\t"))
    strength = float(json.loads(Path(args.dose_decision).read_text())["chosen_strength"])
    lora_state, weight_format = load_lora_state_dict(args.lora_path)
    hns_state, _ = load_lora_state_dict(args.hns_path)
    adapter_config = load_adapter_config(args.lora_path)
    lora_pairs, hns_pairs = collect_pairs(lora_state), collect_pairs(hns_state)
    if set(lora_pairs) != set(hns_pairs):
        raise RuntimeError("LoRA/HNS module sets differ")

    decompositions = {}
    for module, pair in lora_pairs.items():
        U, sigma, Vh, _ = lowrank_svd_from_ba(pair["B"][1], pair["A"][1])
        sigma_hns, basis_stats = align_edited_spectrum_to_reference(
            U,
            Vh,
            hns_pairs[module]["B"][1],
            hns_pairs[module]["A"][1],
        )
        decompositions[module] = (U, sigma, Vh, sigma_hns, basis_stats)

    groups = {
        "suppress_block_A": [row for row in selected if row["class"] == "predicted_suppress" and row["suppress_block"] == "A"],
        "suppress_block_B": [row for row in selected if row["class"] == "predicted_suppress" and row["suppress_block"] == "B"],
        "suppress_block_AB": [row for row in selected if row["class"] == "predicted_suppress"],
        "predicted_retain_block": [row for row in selected if row["class"] == "predicted_retain"],
    }
    variants = []
    edited_sigmas_by_group = {}
    for label, rows in groups.items():
        by_module = defaultdict(list)
        for row in rows:
            by_module[row["module"]].append(int(row["direction"]) - 1)
        edited_state = dict(lora_state)
        sigma_map = {module: values[1].clone() for module, values in decompositions.items()}
        for module, directions in by_module.items():
            U, sigma, Vh, sigma_hns, _ = decompositions[module]
            sigma_new = sigma.clone()
            for direction in directions:
                sigma_new[direction] = sigma[direction] + strength * (
                    sigma_hns[direction] - sigma[direction]
                )
            b_new, a_new = rebuild_ba_from_uv_sigma(U, Vh, sigma_new)
            key_a, old_a = lora_pairs[module]["A"]
            key_b, old_b = lora_pairs[module]["B"]
            edited_state[key_a] = a_new.to(dtype=old_a.dtype)
            edited_state[key_b] = b_new.to(dtype=old_b.dtype)
            sigma_map[module] = sigma_new
        adapter_dir = output / label
        if adapter_dir.exists():
            raise FileExistsError(f"refusing to overwrite {adapter_dir}")
        shutil.copytree(args.lora_path, adapter_dir)
        save_lora_state_dict(str(adapter_dir), edited_state, weight_format)
        edited_sigmas_by_group[label] = sigma_map
        meta = {
            "label": label,
            "strength": strength,
            "directions": len(rows),
            "selected": [{"module": row["module"], "direction": int(row["direction"])} for row in rows],
        }
        (adapter_dir / "direction_block_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        variants.append(meta)
        print(f"[Build] {label}: directions={len(rows)} strength={strength:g}", flush=True)

    # Match the Frobenius norm of the entire effective LoRA update collection.
    original_norm2 = 0.0
    target_norm2 = 0.0
    target_sigmas = edited_sigmas_by_group["suppress_block_AB"]
    for module, (_, sigma, _, _, _) in decompositions.items():
        scaling = get_scaling_for_module(adapter_config, module)
        original_norm2 += float((sigma.double().square().sum() * scaling**2).item())
        target_norm2 += float((target_sigmas[module].double().square().sum() * scaling**2).item())
    gamma = math.sqrt(target_norm2 / original_norm2)
    scalar_state = dict(lora_state)
    for module, (U, sigma, Vh, _, _) in decompositions.items():
        b_new, a_new = rebuild_ba_from_uv_sigma(U, Vh, sigma * gamma)
        key_a, old_a = lora_pairs[module]["A"]
        key_b, old_b = lora_pairs[module]["B"]
        scalar_state[key_a] = a_new.to(dtype=old_a.dtype)
        scalar_state[key_b] = b_new.to(dtype=old_b.dtype)
    scalar_dir = output / "scalar_global_fro_match"
    if scalar_dir.exists():
        raise FileExistsError(f"refusing to overwrite {scalar_dir}")
    shutil.copytree(args.lora_path, scalar_dir)
    save_lora_state_dict(str(scalar_dir), scalar_state, weight_format)
    scalar_meta = {
        "label": "scalar_global_fro_match",
        "strength": strength,
        "directions": 0,
        "global_scale": gamma,
        "target": "suppress_block_AB total effective LoRA Frobenius norm",
    }
    (scalar_dir / "direction_block_meta.json").write_text(json.dumps(scalar_meta, indent=2) + "\n")
    variants.append(scalar_meta)

    manifest = {
        "method": "predeclared matched direction blocks",
        "chosen_strength": strength,
        "scalar_control_scope": "global effective LoRA Frobenius norm across all modules",
        "variants": variants,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"variants": len(variants), "chosen_strength": strength, "scalar_gamma": gamma}, indent=2))


if __name__ == "__main__":
    main()
