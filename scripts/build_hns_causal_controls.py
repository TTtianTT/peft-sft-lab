#!/usr/bin/env python3
"""Build ScalarShrink/ShapeOnly/HeadOnly/TailOnly from an observed LoRA/HNS pair."""

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
from finetune.spectral_edit.mechanism import (
    CONTROL_NAMES,
    align_edited_spectrum_to_reference,
    build_causal_control_spectra,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_basis_error", type=float, default=5e-3)
    return parser.parse_args()


def collect_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is not None:
            prefix, which, _ = parsed
            pairs.setdefault(prefix, {})[which] = (key, tensor)
    return {prefix: pair for prefix, pair in pairs.items() if {"A", "B"} <= pair.keys()}


def main() -> None:
    args = parse_args()
    lora_path, hns_path = Path(args.lora_path).resolve(), Path(args.hns_path).resolve()
    out_root = Path(args.out_root).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    lora_state, weight_format = load_lora_state_dict(str(lora_path))
    hns_state, _ = load_lora_state_dict(str(hns_path))
    lora_pairs, hns_pairs = collect_pairs(lora_state), collect_pairs(hns_state)
    if set(lora_pairs) != set(hns_pairs):
        raise RuntimeError("LoRA/HNS module sets differ")
    variant_states = {name: dict(lora_state) for name in CONTROL_NAMES}
    module_stats: dict[str, dict] = {}

    with torch.inference_mode():
        for index, prefix in enumerate(sorted(lora_pairs), start=1):
            lora_pair, hns_pair = lora_pairs[prefix], hns_pairs[prefix]
            key_a, a_lora_cpu = lora_pair["A"]
            key_b, b_lora_cpu = lora_pair["B"]
            _, a_hns_cpu = hns_pair["A"]
            _, b_hns_cpu = hns_pair["B"]
            a_lora, b_lora = a_lora_cpu.to(args.device), b_lora_cpu.to(args.device)
            a_hns, b_hns = a_hns_cpu.to(args.device), b_hns_cpu.to(args.device)
            u, sigma_lora, vh, _ = lowrank_svd_from_ba(b_lora, a_lora)
            sigma_hns, alignment = align_edited_spectrum_to_reference(u, vh, b_hns, a_hns)
            basis_error = max(alignment["basis_offdiag_fraction"], alignment["basis_projection_residual_fraction"])
            if basis_error > args.max_basis_error:
                raise RuntimeError(
                    f"{prefix}: HNS does not preserve the LoRA singular basis "
                    f"(error={basis_error:.6g} > {args.max_basis_error})"
                )
            controls, stats = build_causal_control_spectra(sigma_lora, sigma_hns)
            stats.update(alignment)
            stats.update({
                "layer_index": layer_idx_from_module_prefix(prefix),
                "module_suffix": prefix.rsplit(".", 1)[-1],
            })
            module_stats[prefix] = stats
            for name, spectrum in controls.items():
                b_new, a_new = rebuild_ba_from_uv_sigma(u, vh, spectrum)
                variant_states[name][key_a] = a_new.to(dtype=a_lora_cpu.dtype, device="cpu")
                variant_states[name][key_b] = b_new.to(dtype=b_lora_cpu.dtype, device="cpu")
            if index % 32 == 0 or index == len(lora_pairs):
                print(f"[Build] {index}/{len(lora_pairs)} modules", flush=True)

    definitions = {
        "scalar_shrink": "Original LoRA spectrum times one per-module scalar; Frobenius norm matches HNS.",
        "shape_only": "HNS relative spectrum rescaled to restore the original LoRA Frobenius norm.",
        "head_only": "Use HNS values only where HNS suppresses the original singular value.",
        "tail_only": "Use HNS values only where HNS amplifies the original singular value.",
    }
    manifest = {
        "method": "observed_hns_causal_controls",
        "source_lora": str(lora_path),
        "source_hns": str(hns_path),
        "num_modules": len(module_stats),
        "definitions": definitions,
        "max_basis_error": max(
            max(row["basis_offdiag_fraction"], row["basis_projection_residual_fraction"])
            for row in module_stats.values()
        ),
        "variants": [],
    }
    for name in CONTROL_NAMES:
        destination = out_root / name.replace("_", "-")
        shutil.copytree(lora_path, destination)
        save_lora_state_dict(str(destination), variant_states[name], weight_format)
        metadata = {
            "method": "observed_hns_causal_control",
            "control": name,
            "definition": definitions[name],
            "source_lora": str(lora_path),
            "source_hns": str(hns_path),
            "module_stats": module_stats,
        }
        (destination / "spectral_control_meta.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        manifest["variants"].append({"label": name, "path": str(destination)})
        print(f"[Save] {name}: {destination}")
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[Done] {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
