#!/usr/bin/env python3
"""Build a controlled bank of post-hoc spectral LoRA ablations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from finetune.spectral_edit.ablations import transform_singular_values
from finetune.spectral_edit.io import (
    layer_idx_from_module_prefix,
    load_lora_state_dict,
    parse_lora_ab_key,
    save_lora_state_dict,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_modules", nargs="+", default=["all"])
    return parser.parse_args()


def collect_pairs(state_dict: dict[str, torch.Tensor], targets: list[str]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    parsed: list[tuple[str, str, str, torch.Tensor]] = []
    suffixes: set[str] = set()
    for key, tensor in state_dict.items():
        match = parse_lora_ab_key(key)
        if match is None:
            continue
        prefix, which, _ = match
        suffixes.add(prefix.rsplit(".", 1)[-1])
        parsed.append((prefix, which, key, tensor))
    selected = suffixes if {item.lower() for item in targets} & {"all", "all_modules"} else set(targets)
    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for prefix, which, key, tensor in parsed:
        if prefix.rsplit(".", 1)[-1] in selected:
            pairs.setdefault(prefix, {})[which] = (key, tensor)
    complete = {prefix: pair for prefix, pair in pairs.items() if {"A", "B"} <= pair.keys()}
    if not complete:
        raise RuntimeError(f"No complete LoRA A/B pairs found; available suffixes: {sorted(suffixes)}")
    return complete


def temperature_label(value: float) -> str:
    return f"temperature-tau-{value:.2f}".replace(".", "p")


def main() -> None:
    args = parse_args()
    source = Path(args.lora_path).resolve()
    out_root = Path(args.out_root).resolve()
    if not (source / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter config: {source / 'adapter_config.json'}")
    out_root.mkdir(parents=True, exist_ok=True)

    specs: list[tuple[str, str, float | None]] = [
        ("scalar-shrink-fro-match-flat-nuclear", "scalar_shrink", None),
        ("exact-flat-nuclear", "exact_flat_nuclear", None),
        ("top-shrink-to-mean", "top_shrink", None),
        ("tail-lift-to-mean", "tail_lift", None),
    ]
    specs.extend((temperature_label(tau), "temperature", tau) for tau in args.temperatures)
    if len({name for name, _, _ in specs}) != len(specs):
        raise ValueError("Temperatures produce duplicate output labels")

    state_dict, weight_format = load_lora_state_dict(str(source))
    pairs = collect_pairs(state_dict, args.target_modules)
    variant_states = {name: dict(state_dict) for name, _, _ in specs}
    variant_stats: dict[str, dict[str, dict]] = {name: {} for name, _, _ in specs}

    print(f"[Build] source={source} modules={len(pairs)} variants={len(specs)} device={args.device}")
    with torch.inference_mode():
        for index, (prefix, pair) in enumerate(sorted(pairs.items()), start=1):
            key_a, a_old = pair["A"]
            key_b, b_old = pair["B"]
            u, singular_values, vh, _ = lowrank_svd_from_ba(
                b_old.to(args.device), a_old.to(args.device)
            )
            for name, mode, temperature in specs:
                edited, stats = transform_singular_values(
                    singular_values, mode=mode, temperature=temperature
                )
                b_new, a_new = rebuild_ba_from_uv_sigma(u, vh, edited)
                variant_states[name][key_a] = a_new.to(dtype=a_old.dtype, device="cpu")
                variant_states[name][key_b] = b_new.to(dtype=b_old.dtype, device="cpu")
                stats.update(
                    {
                        "module_suffix": prefix.rsplit(".", 1)[-1],
                        "layer_index": layer_idx_from_module_prefix(prefix),
                    }
                )
                variant_stats[name][prefix] = stats
            if index % 32 == 0 or index == len(pairs):
                print(f"[Build] decomposed {index}/{len(pairs)} modules", flush=True)

    manifest: dict[str, object] = {
        "source_adapter": str(source),
        "num_modules": len(pairs),
        "definitions": {
            "scalar_shrink": "Scale the whole spectrum so its Frobenius norm matches ExactFlat-Nuclear.",
            "exact_flat_nuclear": "Set every singular value to the original nuclear norm divided by rank.",
            "top_shrink": "Clip only singular values above the original per-module mean down to that mean.",
            "tail_lift": "Lift only singular values below the original per-module mean up to that mean.",
            "temperature": "sigma_tau = ||sigma||_1 * sigma^tau / sum(sigma^tau).",
        },
        "variants": [],
    }
    for name, mode, temperature in specs:
        destination = out_root / name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
        shutil.copytree(source, destination)
        save_lora_state_dict(str(destination), variant_states[name], weight_format)
        module_stats = variant_stats[name]
        summary = {
            "mean_effective_rank_before": sum(x["effective_rank_before"] for x in module_stats.values()) / len(module_stats),
            "mean_effective_rank_after": sum(x["effective_rank_after"] for x in module_stats.values()) / len(module_stats),
            "mean_nuclear_ratio": sum(x["nuclear_ratio"] for x in module_stats.values()) / len(module_stats),
            "mean_fro_ratio": sum(x["fro_ratio"] for x in module_stats.values()) / len(module_stats),
        }
        metadata = {
            "method": "controlled_spectral_ablation",
            "mode": mode,
            "temperature": temperature,
            "source_adapter": str(source),
            "target_modules": args.target_modules,
            "summary": summary,
            "module_stats": module_stats,
        }
        (destination / "spectral_ablation_meta.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        manifest["variants"].append(
            {"label": name, "path": str(destination), "mode": mode, "temperature": temperature, **summary}
        )
        print(f"[Save] {destination}")

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[Done] {manifest_path}")


if __name__ == "__main__":
    main()
