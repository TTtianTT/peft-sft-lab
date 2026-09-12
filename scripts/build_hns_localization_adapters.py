#!/usr/bin/env python3
"""Build coarse module- and layer-localized HNS4+1 LoRA adapters."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from finetune.spectral_edit.io import (
    layer_idx_from_module_prefix,
    load_lora_state_dict,
    parse_lora_ab_key,
    save_lora_state_dict,
)
from finetune.spectral_edit.posthoc_hns import HNSEditConfig, apply_hns_to_svd
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fast_steps", type=int, default=4)
    parser.add_argument("--stable_steps", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=36)
    return parser.parse_args()


def collect_pairs(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state_dict.items():
        match = parse_lora_ab_key(key)
        if match is None:
            continue
        prefix, which, _ = match
        pairs.setdefault(prefix, {})[which] = (key, tensor)
    complete = {prefix: pair for prefix, pair in pairs.items() if {"A", "B"} <= pair.keys()}
    if not complete:
        raise RuntimeError("No complete LoRA A/B pairs found")
    return complete


def selected(label: str, prefix: str, layer: int | None, num_layers: int) -> bool:
    suffix = prefix.rsplit(".", 1)[-1]
    if label == "hns-attention":
        return suffix in {"q_proj", "k_proj", "v_proj", "o_proj"}
    if label == "hns-mlp":
        return suffix in {"gate_proj", "up_proj", "down_proj"}
    if label == "hns-qkv":
        return suffix in {"q_proj", "k_proj", "v_proj"}
    if label == "hns-o-proj":
        return suffix == "o_proj"
    if label == "hns-gate-up":
        return suffix in {"gate_proj", "up_proj"}
    if label == "hns-down-proj":
        return suffix == "down_proj"
    third = num_layers // 3
    if label == "hns-layers-early":
        return layer is not None and 0 <= layer < third
    if label == "hns-layers-middle":
        return layer is not None and third <= layer < 2 * third
    if label == "hns-layers-late":
        return layer is not None and 2 * third <= layer < num_layers
    raise ValueError(f"Unknown localization label: {label}")


def main() -> None:
    args = parse_args()
    source = Path(args.lora_path).resolve()
    out_root = Path(args.out_root).resolve()
    labels = [
        "hns-attention",
        "hns-mlp",
        "hns-qkv",
        "hns-o-proj",
        "hns-gate-up",
        "hns-down-proj",
        "hns-layers-early",
        "hns-layers-middle",
        "hns-layers-late",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    state_dict, weight_format = load_lora_state_dict(str(source))
    pairs = collect_pairs(state_dict)
    variant_states = {label: dict(state_dict) for label in labels}
    variant_stats: dict[str, dict[str, dict]] = {label: {} for label in labels}
    config = HNSEditConfig(
        fast_steps=args.fast_steps,
        stable_steps=args.stable_steps,
        preserve_nuclear_norm=True,
    )

    print(f"[Build] source={source} modules={len(pairs)} variants={len(labels)} device={args.device}")
    with torch.inference_mode():
        for index, (prefix, pair) in enumerate(sorted(pairs.items()), start=1):
            key_a, a_old = pair["A"]
            key_b, b_old = pair["B"]
            layer = layer_idx_from_module_prefix(prefix)
            u, sigma, vh, _ = lowrank_svd_from_ba(b_old.to(args.device), a_old.to(args.device))
            _, _, sigma_hns, stats = apply_hns_to_svd(u, vh, sigma, config=config)
            b_hns, a_hns = rebuild_ba_from_uv_sigma(u, vh, sigma_hns)
            for label in labels:
                if not selected(label, prefix, layer, args.num_layers):
                    continue
                variant_states[label][key_a] = a_hns.to(dtype=a_old.dtype, device="cpu")
                variant_states[label][key_b] = b_hns.to(dtype=b_old.dtype, device="cpu")
                variant_stats[label][prefix] = {
                    **stats,
                    "module_suffix": prefix.rsplit(".", 1)[-1],
                    "layer_index": layer,
                }
            if index % 32 == 0 or index == len(pairs):
                print(f"[Build] decomposed {index}/{len(pairs)} modules", flush=True)

    manifest: dict[str, object] = {
        "method": "hns4p1_localization",
        "source_adapter": str(source),
        "fast_steps": args.fast_steps,
        "stable_steps": args.stable_steps,
        "preserve_nuclear_norm": True,
        "num_layers": args.num_layers,
        "variants": [],
    }
    for label in labels:
        destination = out_root / label
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
        shutil.copytree(source, destination)
        save_lora_state_dict(str(destination), variant_states[label], weight_format)
        stats = variant_stats[label]
        metadata = {
            "method": "hns4p1_localization",
            "label": label,
            "source_adapter": str(source),
            "fast_steps": args.fast_steps,
            "stable_steps": args.stable_steps,
            "preserve_nuclear_norm": True,
            "num_edited_modules": len(stats),
            "module_stats": stats,
        }
        (destination / "spectral_edit_meta.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        manifest["variants"].append(
            {"label": label, "path": str(destination), "num_edited_modules": len(stats)}
        )
        print(f"[Save] {label}: {len(stats)} modules -> {destination}")

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[Done] {manifest_path}")


if __name__ == "__main__":
    main()
