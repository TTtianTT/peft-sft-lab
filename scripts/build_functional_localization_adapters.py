#!/usr/bin/env python3
"""Build observed-HNS module-subset adapters for functional localization.

The edited tensors are copied from the corresponding all-module HNS adapter.
Only module selection changes; the intervention itself is therefore identical
to the observed full HNS edit for every selected module.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch

from finetune.spectral_edit.io import (
    load_lora_state_dict,
    parse_lora_ab_key,
    save_lora_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--module_spectra", required=True)
    parser.add_argument("--direction_response", required=True)
    parser.add_argument("--base_model_label", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--scopes", nargs="+", type=float, default=(0.25, 0.50))
    parser.add_argument("--random_seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--num_layer_bins", type=int, default=4)
    return parser.parse_args()


def canonical_module(name: str) -> str:
    marker = "layers."
    position = name.find(marker)
    if position < 0:
        raise ValueError(f"Cannot canonicalize module without {marker!r}: {name}")
    return name[position:]


def collect_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is None:
            continue
        prefix, which, _ = parsed
        pairs.setdefault(canonical_module(prefix), {})[which] = (key, tensor)
    complete = {name: pair for name, pair in pairs.items() if {"A", "B"} <= pair.keys()}
    if not complete:
        raise RuntimeError("No complete LoRA A/B pairs found")
    return complete


def load_raw_scores(path: Path, base_model: str, task: str) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["base_model"] == base_model and row["task"] == task
        ]
    return {
        canonical_module(row["module"]): float(row["lora_top1_energy_share"])
        for row in rows
    }


def load_functional_scores(path: Path, base_model: str, task: str) -> dict[str, float]:
    energies: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["base_model"] == base_model
                and row["task"] == task
                and row["adapter"] == "lora"
            ):
                energies[canonical_module(row["module"])].append(
                    (int(row["direction"]), float(row["response_energy_share"]))
                )
    result = {}
    for module, values in energies.items():
        ordered = sorted(values)
        if not ordered or ordered[0][0] != 1:
            raise RuntimeError(f"Missing first functional singular direction for {module}")
        result[module] = ordered[0][1]
    return result


def module_parts(module: str) -> tuple[int, str]:
    pieces = module.split(".")
    layer = int(pieces[1])
    return layer, pieces[-1]


def matched_random(
    universe: list[str], reference: set[str], seed: int, num_layer_bins: int
) -> set[str]:
    max_layer = max(module_parts(module)[0] for module in universe)

    def stratum(module: str) -> tuple[int, str]:
        layer, module_type = module_parts(module)
        layer_bin = min(num_layer_bins - 1, layer * num_layer_bins // (max_layer + 1))
        return layer_bin, module_type

    candidates: dict[tuple[int, str], list[str]] = defaultdict(list)
    for module in universe:
        candidates[stratum(module)].append(module)
    wanted = Counter(stratum(module) for module in reference)
    rng = random.Random(seed)
    selected: set[str] = set()
    for key, count in sorted(wanted.items()):
        pool = sorted(candidates[key])
        if count > len(pool):
            raise RuntimeError(f"Matched stratum {key} requests {count}/{len(pool)} modules")
        selected.update(rng.sample(pool, count))
    if len(selected) != len(reference):
        raise RuntimeError("Matched random selection did not preserve scope")
    return selected


def copy_selected(
    lora_state: dict[str, torch.Tensor],
    lora_pairs: dict[str, dict[str, tuple[str, torch.Tensor]]],
    hns_pairs: dict[str, dict[str, tuple[str, torch.Tensor]]],
    selected: set[str],
) -> dict[str, torch.Tensor]:
    output = dict(lora_state)
    for module in selected:
        for which in ("A", "B"):
            lora_key, lora_tensor = lora_pairs[module][which]
            _, hns_tensor = hns_pairs[module][which]
            if lora_tensor.shape != hns_tensor.shape:
                raise RuntimeError(f"Shape mismatch for {module}/{which}")
            output[lora_key] = hns_tensor.to(dtype=lora_tensor.dtype, device="cpu")
    return output


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    lora_state, weight_format = load_lora_state_dict(args.lora_path)
    hns_state, _ = load_lora_state_dict(args.hns_path)
    lora_pairs, hns_pairs = collect_pairs(lora_state), collect_pairs(hns_state)
    universe = sorted(lora_pairs)
    if set(universe) != set(hns_pairs):
        raise RuntimeError("LoRA/HNS module sets differ")

    raw = load_raw_scores(Path(args.module_spectra), args.base_model_label, args.task)
    functional = load_functional_scores(
        Path(args.direction_response), args.base_model_label, args.task
    )
    if set(universe) != set(raw) or set(universe) != set(functional):
        raise RuntimeError(
            f"Score/module mismatch: adapters={len(universe)} raw={len(raw)} functional={len(functional)}"
        )

    variants: list[tuple[str, set[str], dict]] = []
    for scope in args.scopes:
        if not 0 < scope <= 1:
            raise ValueError(f"Invalid scope: {scope}")
        count = max(1, math.floor(len(universe) * scope + 0.5))
        pct = int(round(scope * 100))
        functional_set = set(sorted(universe, key=lambda name: (-functional[name], name))[:count])
        raw_set = set(sorted(universe, key=lambda name: (-raw[name], name))[:count])
        variants.extend([
            (f"functional_top{pct}", functional_set, {"selection": "functional_top1_share"}),
            (f"raw_top{pct}", raw_set, {"selection": "raw_top1_energy_share"}),
        ])
        for seed in args.random_seeds:
            rng = random.Random(seed)
            variants.append((
                f"random_top{pct}_seed{seed}", set(rng.sample(universe, count)),
                {"selection": "uniform_random", "seed": seed},
            ))
            variants.append((
                f"matched_random_top{pct}_seed{seed}",
                matched_random(universe, functional_set, seed, args.num_layer_bins),
                {
                    "selection": "functional_matched_random",
                    "seed": seed,
                    "matching": f"module_type_x_{args.num_layer_bins}_equal_width_layer_bins",
                },
            ))

    manifest = {
        "method": "observed_full_hns_functional_localization",
        "base_model_label": args.base_model_label,
        "task": args.task,
        "lora_path": str(Path(args.lora_path).resolve()),
        "hns_path": str(Path(args.hns_path).resolve()),
        "total_modules": len(universe),
        "functional_score": "LoRA E_1 / sum_i E_i on fixed pretrained-base hidden states",
        "raw_score": "LoRA sigma_1^2 / sum_i sigma_i^2",
        "variants": [],
    }
    for label, selected, selection_meta in variants:
        destination = out_root / label
        metadata_path = destination / "functional_localization_meta.json"
        if metadata_path.is_file():
            print(f"[Resume] {label}")
            manifest["variants"].append(json.loads(metadata_path.read_text(encoding="utf-8")))
            continue
        if destination.exists():
            raise FileExistsError(f"Refusing incomplete existing output: {destination}")
        shutil.copytree(args.lora_path, destination)
        state = copy_selected(lora_state, lora_pairs, hns_pairs, selected)
        save_lora_state_dict(str(destination), state, weight_format)
        type_counts = Counter(module_parts(module)[1] for module in selected)
        layer_counts = Counter(module_parts(module)[0] for module in selected)
        metadata = {
            "label": label,
            **selection_meta,
            "scope_fraction_requested": len(selected) / len(universe),
            "selected_count": len(selected),
            "selected_modules": sorted(selected),
            "selected_module_type_counts": dict(sorted(type_counts.items())),
            "selected_layer_counts": {str(key): value for key, value in sorted(layer_counts.items())},
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        manifest["variants"].append(metadata)
        print(f"[Save] {label}: {len(selected)}/{len(universe)} modules")

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[Done] {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
