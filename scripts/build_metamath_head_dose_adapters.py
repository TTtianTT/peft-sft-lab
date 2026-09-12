#!/usr/bin/env python3
"""Build HeadOnly dose adapters and per-module Frobenius-matched scalar controls."""

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
    align_edited_spectrum_to_reference,
    frobenius_norm_from_factors,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


DOSES = (0.0, 0.25, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lora_path", required=True)
    p.add_argument("--hns_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_basis_error", type=float, default=5e-3)
    return p.parse_args()


def dose_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def collect_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is not None:
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
    lora_path = Path(args.lora_path).resolve()
    hns_path = Path(args.hns_path).resolve()
    output = Path(args.output_dir).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        print(f"[Skip] dose adapters already built: {manifest_path}")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty output without manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)

    lora_state, weight_format = load_lora_state_dict(str(lora_path))
    hns_state, _ = load_lora_state_dict(str(hns_path))
    lora_pairs = collect_pairs(lora_state)
    hns_pairs = collect_pairs(hns_state)
    if set(lora_pairs) != set(hns_pairs):
        raise RuntimeError("LoRA/HNS module sets differ")

    variant_names = ["head_0p00_rebuild"]
    for dose in DOSES[1:]:
        variant_names.extend([f"head_{dose_tag(dose)}", f"scalar_{dose_tag(dose)}"])
    variant_states = {name: dict(lora_state) for name in variant_names}
    module_stats: dict[str, dict] = {}
    max_basis_error = 0.0
    max_zero_rebuild_error = 0.0

    with torch.inference_mode():
        for index, prefix in enumerate(sorted(lora_pairs), start=1):
            lora_pair = lora_pairs[prefix]
            hns_pair = hns_pairs[prefix]
            key_a, a_cpu = lora_pair["A"]
            key_b, b_cpu = lora_pair["B"]
            a = a_cpu.to(args.device)
            b = b_cpu.to(args.device)
            U, sigma, Vh, _ = lowrank_svd_from_ba(b, a)
            hns_sigma, alignment = align_edited_spectrum_to_reference(
                U,
                Vh,
                hns_pair["B"][1].to(args.device),
                hns_pair["A"][1].to(args.device),
            )
            basis_error = max(alignment["basis_offdiag_fraction"], alignment["basis_projection_residual_fraction"])
            max_basis_error = max(max_basis_error, basis_error)
            if basis_error > args.max_basis_error:
                raise RuntimeError(f"{prefix}: basis error {basis_error} exceeds {args.max_basis_error}")
            head_target = torch.minimum(sigma, hns_sigma)
            original_fro = torch.linalg.vector_norm(sigma)
            per_dose: dict[str, dict] = {}
            for dose in DOSES:
                head_sigma = sigma + dose * (head_target - sigma)
                head_fro = torch.linalg.vector_norm(head_sigma)
                gamma = head_fro / original_fro.clamp_min(1e-12)
                scalar_sigma = gamma * sigma
                tag = dose_tag(dose)
                per_dose[tag] = {
                    "dose": dose,
                    "head_fro": float(head_fro.item()),
                    "scalar_fro": float(torch.linalg.vector_norm(scalar_sigma).item()),
                    "scalar_gamma": float(gamma.item()),
                    "head_path_norm": float(torch.linalg.vector_norm(head_sigma - sigma).item()),
                }
                if dose == 0:
                    names_and_spectra = [("head_0p00_rebuild", head_sigma)]
                else:
                    names_and_spectra = [
                        (f"head_{tag}", head_sigma),
                        (f"scalar_{tag}", scalar_sigma),
                    ]
                for name, spectrum in names_and_spectra:
                    b_new, a_new = rebuild_ba_from_uv_sigma(U, Vh, spectrum)
                    a_saved = a_new.to(dtype=a_cpu.dtype)
                    b_saved = b_new.to(dtype=b_cpu.dtype)
                    variant_states[name][key_a] = a_saved.cpu()
                    variant_states[name][key_b] = b_saved.cpu()
                    if dose == 0:
                        relative = difference_frobenius(
                            b_saved.to(args.device),
                            a_saved.to(args.device),
                            b,
                            a,
                        ) / frobenius_norm_from_factors(b, a).clamp_min(1e-12)
                        max_zero_rebuild_error = max(max_zero_rebuild_error, float(relative.item()))
            module_stats[prefix] = {
                "layer": layer_idx_from_module_prefix(prefix),
                "module_type": prefix.rsplit(".", 1)[-1],
                "rank": len(sigma),
                "lora_fro": float(original_fro.item()),
                "head_target_fro": float(torch.linalg.vector_norm(head_target).item()),
                "suppressed_directions": int((head_target < sigma - 1e-8).sum().item()),
                "basis": alignment,
                "doses": per_dose,
            }
            if index % 32 == 0 or index == len(lora_pairs):
                print(f"[Build] {index}/{len(lora_pairs)} modules", flush=True)

    variants: list[dict] = []
    for name in variant_names:
        destination = output / name.replace("_", "-")
        shutil.copytree(lora_path, destination)
        save_lora_state_dict(str(destination), variant_states[name], weight_format)
        metadata = {
            "method": "metamath_direct_head_dose",
            "variant": name,
            "source_lora": str(lora_path),
            "source_hns": str(hns_path),
            "definition": (
                "HeadOnly interpolation in the aligned LoRA basis"
                if name.startswith("head_")
                else "Per-module scalar shrink matching the corresponding HeadOnly-dose Frobenius norm"
            ),
        }
        (destination / "spectral_dose_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        variants.append({"label": name, "path": str(destination)})
        print(f"[Save] {name}: {destination}", flush=True)

    manifest = {
        "status": "complete",
        "method": "direct_head_dose_with_matched_scalar",
        "source_lora": str(lora_path),
        "source_hns": str(hns_path),
        "doses": list(DOSES),
        "scalar_definition": "per-module Frobenius norm matched to HeadOnly at the same dose",
        "modules": len(module_stats),
        "max_hns_basis_error": max_basis_error,
        "max_zero_dose_reconstruction_relative_error": max_zero_rebuild_error,
        "variants": variants,
        "module_stats": module_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "status", "modules", "max_hns_basis_error", "max_zero_dose_reconstruction_relative_error"
    )}, indent=2))


if __name__ == "__main__":
    main()
