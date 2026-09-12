#!/usr/bin/env python3
"""Build structure-matched HNS subset adapters from functional/compatibility corners."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key, save_lora_state_dict


def canonical(name: str) -> str:
    position = name.find("layers.")
    if position < 0:
        raise ValueError(name)
    return name[position:]


def collect_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is None:
            continue
        prefix, which, _ = parsed
        pairs.setdefault(canonical(prefix), {})[which] = (key, tensor)
    return {name: pair for name, pair in pairs.items() if {"A", "B"} <= pair.keys()}


def load_functional(path: Path, base_model: str, task: str) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["base_model"] == base_model
                and row["task"] == task
                and row["adapter"] == "lora"
                and int(row["direction"]) == 1
            ):
                values[canonical(row["module"])] = float(row["response_energy_share"])
    return values


def load_compatibility(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {
        canonical(name): float(row["compatibility"])
        for name, row in payload["module_selection"].items()
    }


def module_parts(module: str) -> tuple[int, str]:
    pieces = module.split(".")
    return int(pieces[1]), pieces[-1]


def centered_ranks(modules: list[str], values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(modules, key=lambda name: (values[name], name))
    if len(ordered) == 1:
        return {ordered[0]: 0.0}
    return {name: -1.0 + 2.0 * index / (len(ordered) - 1) for index, name in enumerate(ordered)}


def quotas(strata: dict[tuple[int, str], list[str]], scope: float) -> dict[tuple[int, str], int]:
    total = sum(len(values) for values in strata.values())
    target = int(math.floor(total * scope + 0.5))
    result = {key: int(math.floor(len(values) * scope)) for key, values in strata.items()}
    remaining = target - sum(result.values())
    order = sorted(
        strata,
        key=lambda key: (-(len(strata[key]) * scope - result[key]), key),
    )
    for key in order[:remaining]:
        result[key] += 1
    if sum(result.values()) != target:
        raise RuntimeError("Failed to construct exact structure-matched quota")
    return result


def copy_selected(
    state: dict[str, torch.Tensor],
    lora_pairs: dict[str, dict[str, tuple[str, torch.Tensor]]],
    hns_pairs: dict[str, dict[str, tuple[str, torch.Tensor]]],
    selected: set[str],
) -> dict[str, torch.Tensor]:
    output = dict(state)
    for module in selected:
        for which in ("A", "B"):
            lora_key, lora_tensor = lora_pairs[module][which]
            _, hns_tensor = hns_pairs[module][which]
            if lora_tensor.shape != hns_tensor.shape:
                raise RuntimeError(f"shape mismatch for {module}/{which}")
            output[lora_key] = hns_tensor.to(dtype=lora_tensor.dtype, device="cpu")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--direction_response", required=True)
    parser.add_argument("--gradient_meta", required=True)
    parser.add_argument("--base_model_label", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--scopes", nargs="+", type=float, default=(0.25, 0.50))
    parser.add_argument("--num_layer_bins", type=int, default=4)
    args = parser.parse_args()

    output = Path(args.out_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lora_state, weight_format = load_lora_state_dict(args.lora_path)
    hns_state, _ = load_lora_state_dict(args.hns_path)
    lora_pairs, hns_pairs = collect_pairs(lora_state), collect_pairs(hns_state)
    modules = sorted(lora_pairs)
    if set(modules) != set(hns_pairs):
        raise RuntimeError("LoRA/HNS module sets differ")

    functional = load_functional(Path(args.direction_response), args.base_model_label, args.task)
    compatibility = load_compatibility(Path(args.gradient_meta))
    if set(modules) != set(functional) or set(modules) != set(compatibility):
        raise RuntimeError(
            f"module mismatch: adapters={len(modules)} F={len(functional)} C={len(compatibility)}"
        )

    max_layer = max(module_parts(module)[0] for module in modules)
    strata: dict[tuple[int, str], list[str]] = defaultdict(list)
    for module in modules:
        layer, module_type = module_parts(module)
        layer_bin = min(args.num_layer_bins - 1, layer * args.num_layer_bins // (max_layer + 1))
        strata[(layer_bin, module_type)].append(module)

    corner_signs = {
        "high_f_high_c": (1.0, 1.0),
        "high_f_low_c": (1.0, -1.0),
        "low_f_high_c": (-1.0, 1.0),
        "low_f_low_c": (-1.0, -1.0),
        "compatibility": (0.0, 1.0),
    }
    variants = []
    for scope in args.scopes:
        if not 0.0 < scope <= 1.0:
            raise ValueError(scope)
        quota = quotas(strata, scope)
        pct = int(round(100 * scope))
        for selection, (f_sign, c_sign) in corner_signs.items():
            selected: set[str] = set()
            module_scores: dict[str, float] = {}
            for key, members in sorted(strata.items()):
                f_rank = centered_ranks(members, functional)
                c_rank = centered_ranks(members, compatibility)
                scores = {
                    module: f_sign * f_rank[module] + c_sign * c_rank[module]
                    for module in members
                }
                module_scores.update(scores)
                selected.update(sorted(members, key=lambda name: (-scores[name], name))[: quota[key]])
            expected = int(math.floor(len(modules) * scope + 0.5))
            if len(selected) != expected:
                raise RuntimeError(f"{selection}/{scope}: selected {len(selected)} != {expected}")
            label = f"{selection}_top{pct}"
            adapter_dir = output / label
            if adapter_dir.exists() and any(adapter_dir.iterdir()):
                raise FileExistsError(f"Refusing to overwrite {adapter_dir}")
            shutil.copytree(args.lora_path, adapter_dir)
            edited = copy_selected(lora_state, lora_pairs, hns_pairs, selected)
            save_lora_state_dict(str(adapter_dir), edited, weight_format)
            rows = [
                {
                    "module": module,
                    "layer": module_parts(module)[0],
                    "module_type": module_parts(module)[1],
                    "functional_concentration": functional[module],
                    "gradient_compatibility": compatibility[module],
                    "corner_score": module_scores[module],
                    "selected": module in selected,
                }
                for module in modules
            ]
            metadata = {
                "method": "structure_matched_functional_compatibility_corner",
                "selection": selection,
                "scope_fraction_requested": scope,
                "selected_count": len(selected),
                "total_modules": len(modules),
                "num_layer_bins": args.num_layer_bins,
                "selected_modules": sorted(selected),
                "selected_mean_functional": sum(functional[m] for m in selected) / len(selected),
                "selected_mean_compatibility": sum(compatibility[m] for m in selected) / len(selected),
                "selected_positive_compatibility_fraction": sum(compatibility[m] > 0 for m in selected) / len(selected),
                "source_lora": str(Path(args.lora_path).resolve()),
                "source_hns": str(Path(args.hns_path).resolve()),
                "rows": rows,
            }
            (adapter_dir / "selection_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
            variants.append({"label": label, **{k: v for k, v in metadata.items() if k != "rows"}})
            print(
                f"[Build] {label}: n={len(selected)} "
                f"meanF={metadata['selected_mean_functional']:.4g} "
                f"meanC={metadata['selected_mean_compatibility']:.4g}",
                flush=True,
            )

    manifest = {
        "base_model": args.base_model_label,
        "task": args.task,
        "design": "within module-type x layer-quartile rank corners; exact matched quotas",
        "variants": variants,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"variants": len(variants), "modules": len(modules)}, indent=2))


if __name__ == "__main__":
    main()
