#!/usr/bin/env python3
"""Build global, spectrum-assigned, and shuffled LoRA scalars without SVD reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key, save_lora_state_dict
from finetune.spectral_edit.mechanism import frobenius_norm_from_factors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--module_scaling", required=True)
    parser.add_argument("--shuffle_plan", required=True)
    parser.add_argument("--base_model_label", default="Qwen3-8B")
    parser.add_argument("--task", default="metamath")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_relative_error", type=float, default=5e-4)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is None:
            continue
        prefix, which, _ = parsed
        pairs.setdefault(prefix, {})[which] = (key, tensor)
    return {prefix: pair for prefix, pair in pairs.items() if {"A", "B"} <= pair.keys()}


def difference_frobenius(
    b_left: torch.Tensor,
    a_left: torch.Tensor,
    b_right: torch.Tensor,
    a_right: torch.Tensor,
) -> torch.Tensor:
    b_join = torch.cat([b_left, b_right], dim=1)
    a_join = torch.cat([a_left, -a_right], dim=0)
    return frobenius_norm_from_factors(b_join, a_join)


def main() -> None:
    args = parse_args()
    source_path = Path(args.lora_path).resolve()
    output = Path(args.output_dir).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        print(f"[Skip] scalar adapters already built: {manifest_path}")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty output without manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)

    scaling_rows = [
        row for row in read_csv(Path(args.module_scaling))
        if row["base_model"] == args.base_model_label and row["task"] == args.task
    ]
    if not scaling_rows:
        raise RuntimeError("no matching module scaling rows")
    assigned = {row["module"]: float(row["gamma_hns_matched"]) for row in scaling_rows}
    global_values = {float(row["adapter_global_gamma"]) for row in scaling_rows}
    if len(global_values) != 1:
        raise RuntimeError(f"expected one global gamma, found {global_values}")
    global_gamma = global_values.pop()

    shuffle_rows = [
        row for row in read_csv(Path(args.shuffle_plan))
        if row["base_model"] == args.base_model_label and row["task"] == args.task
    ]
    shuffle_gammas: dict[int, dict[str, float]] = {}
    for row in shuffle_rows:
        shuffle_gammas.setdefault(int(row["seed"]), {})[row["target_module"]] = float(row["final_gamma"])
    if not shuffle_gammas:
        raise RuntimeError("no matching shuffle plans")

    state, weight_format = load_lora_state_dict(str(source_path))
    pairs = collect_pairs(state)
    if set(pairs) != set(assigned):
        missing = sorted(set(pairs) - set(assigned))
        extra = sorted(set(assigned) - set(pairs))
        raise RuntimeError(f"module mismatch: missing scaling={missing[:4]}, extra scaling={extra[:4]}")
    for seed, mapping in shuffle_gammas.items():
        if set(mapping) != set(pairs):
            raise RuntimeError(f"shuffle seed {seed} does not cover the LoRA module set")

    variant_gammas: dict[str, dict[str, float]] = {
        "global_norm_matched": {prefix: global_gamma for prefix in pairs},
        "per_module_spectral_scale": assigned,
    }
    for index, seed in enumerate(sorted(shuffle_gammas), start=1):
        variant_gammas[f"shuffled_scale_{index}"] = shuffle_gammas[seed]

    variants: list[dict] = []
    max_error = 0.0
    for label, gamma_by_module in variant_gammas.items():
        variant_state = dict(state)
        original_total_sq = 0.0
        achieved_total_sq = 0.0
        intended_total_sq = 0.0
        variant_max_error = 0.0
        with torch.inference_mode():
            for prefix, pair in pairs.items():
                key_a, a_cpu = pair["A"]
                key_b, b_cpu = pair["B"]
                gamma = gamma_by_module[prefix]
                a = a_cpu.float()
                b = b_cpu.float()
                b_intended = gamma * b
                b_saved = b_intended.to(dtype=b_cpu.dtype)
                variant_state[key_a] = a_cpu
                variant_state[key_b] = b_saved
                original_fro = frobenius_norm_from_factors(b, a)
                intended_fro = frobenius_norm_from_factors(b_intended, a)
                achieved_fro = frobenius_norm_from_factors(b_saved.float(), a)
                error = difference_frobenius(
                    b_saved.float(), a, b_intended, a
                ) / intended_fro.clamp_min(1e-12)
                variant_max_error = max(variant_max_error, float(error.item()))
                original_total_sq += float(original_fro.square().item())
                intended_total_sq += float(intended_fro.square().item())
                achieved_total_sq += float(achieved_fro.square().item())
        if variant_max_error > args.max_relative_error:
            raise RuntimeError(f"{label}: relative update error {variant_max_error} exceeds threshold")
        max_error = max(max_error, variant_max_error)
        destination = output / label.replace("_", "-")
        shutil.copytree(source_path, destination)
        save_lora_state_dict(str(destination), variant_state, weight_format)
        metadata = {
            "method": "posthoc_spectral_scaling",
            "variant": label,
            "source_lora": str(source_path),
            "construction": "directly multiply each stored LoRA B factor; A is bit-identical",
            "global_gamma": global_gamma if label == "global_norm_matched" else None,
            "max_implied_update_relative_error": variant_max_error,
            "original_total_factor_fro": original_total_sq**0.5,
            "intended_total_factor_fro": intended_total_sq**0.5,
            "achieved_total_factor_fro": achieved_total_sq**0.5,
        }
        (destination / "posthoc_scaling_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        variants.append({"label": label, "path": str(destination), **metadata})
        print(
            f"[Build] {label}: total_fro={achieved_total_sq**0.5:.8f}, "
            f"max_error={variant_max_error:.3e}",
            flush=True,
        )

    target_totals = [row["achieved_total_factor_fro"] for row in variants]
    total_spread = (max(target_totals) - min(target_totals)) / max(target_totals)
    if total_spread > args.max_relative_error:
        raise RuntimeError(f"scalar variant total norms differ by {total_spread}")
    manifest = {
        "status": "complete",
        "method": "posthoc_spectral_scaling",
        "source_lora": str(source_path),
        "base_model_label": args.base_model_label,
        "task": args.task,
        "modules": len(pairs),
        "global_gamma": global_gamma,
        "shuffle_seeds": sorted(shuffle_gammas),
        "construction": "direct B-factor scaling without SVD reconstruction",
        "max_implied_update_relative_error": max_error,
        "scalar_total_fro_relative_spread": total_spread,
        "variants": variants,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "modules": manifest["modules"],
        "global_gamma": manifest["global_gamma"],
        "max_implied_update_relative_error": manifest["max_implied_update_relative_error"],
        "scalar_total_fro_relative_spread": manifest["scalar_total_fro_relative_spread"],
    }, indent=2))


if __name__ == "__main__":
    main()
