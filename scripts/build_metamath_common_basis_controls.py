#!/usr/bin/env python3
"""Build HNS, matched scalar, and global scalar controls in one shared U,V factorization."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key, save_lora_state_dict
from finetune.spectral_edit.mechanism import align_edited_spectrum_to_reference, frobenius_norm_from_factors
from finetune.spectral_edit.svd import lowrank_svd_from_ba


GLOBAL_GAMMAS = (1.0, 0.85, 0.70, 0.60, 0.50, 0.40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max_basis_error", type=float, default=5e-3)
    return parser.parse_args()


def gamma_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def collect_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is not None:
            prefix, which, _ = parsed
            pairs.setdefault(prefix, {})[which] = (key, tensor)
    return {prefix: pair for prefix, pair in pairs.items() if {"A", "B"} <= pair.keys()}


def common_factors(U: torch.Tensor, sigma: torch.Tensor, Vh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return U * sigma.unsqueeze(0), Vh


def difference_frobenius(
    b_left: torch.Tensor,
    a_left: torch.Tensor,
    b_right: torch.Tensor,
    a_right: torch.Tensor,
) -> torch.Tensor:
    return frobenius_norm_from_factors(
        torch.cat([b_left, b_right], dim=1), torch.cat([a_left, -a_right], dim=0)
    )


def main() -> None:
    args = parse_args()
    lora_path = Path(args.lora_path).resolve()
    hns_path = Path(args.hns_path).resolve()
    output = Path(args.output_dir).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        print(f"[Skip] common-basis adapters already built: {manifest_path}")
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

    labels = ["zero_rebuild", "common_per_module", "common_hns"]
    labels.extend(f"common_global_{gamma_tag(gamma)}" for gamma in GLOBAL_GAMMAS)
    states = {label: dict(lora_state) for label in labels}
    module_stats: dict[str, dict] = {}
    max_basis_error = 0.0
    max_saved_error = 0.0
    total_lora_sq = 0.0
    total_hns_sq = 0.0

    with torch.inference_mode():
        cached: dict[str, tuple] = {}
        for prefix in sorted(lora_pairs):
            lora_pair = lora_pairs[prefix]
            hns_pair = hns_pairs[prefix]
            key_a, a_cpu = lora_pair["A"]
            key_b, b_cpu = lora_pair["B"]
            a = a_cpu.to(args.device).float()
            b = b_cpu.to(args.device).float()
            U, sigma, Vh, _ = lowrank_svd_from_ba(b, a)
            hns_sigma, alignment = align_edited_spectrum_to_reference(
                U,
                Vh,
                hns_pair["B"][1].to(args.device).float(),
                hns_pair["A"][1].to(args.device).float(),
            )
            basis_error = max(alignment["basis_offdiag_fraction"], alignment["basis_projection_residual_fraction"])
            if basis_error > args.max_basis_error:
                raise RuntimeError(f"{prefix}: HNS basis error {basis_error} exceeds threshold")
            max_basis_error = max(max_basis_error, basis_error)
            lora_fro = torch.linalg.vector_norm(sigma)
            hns_fro = torch.linalg.vector_norm(hns_sigma)
            total_lora_sq += float(lora_fro.square().item())
            total_hns_sq += float(hns_fro.square().item())
            cached[prefix] = (key_a, key_b, a_cpu, b_cpu, U, sigma, Vh, hns_sigma, alignment)

        global_norm_gamma = (total_hns_sq / total_lora_sq) ** 0.5
        for index, (prefix, values) in enumerate(cached.items(), start=1):
            key_a, key_b, a_cpu, b_cpu, U, sigma, Vh, hns_sigma, alignment = values
            per_gamma = torch.linalg.vector_norm(hns_sigma) / torch.linalg.vector_norm(sigma).clamp_min(1e-12)
            spectra = {
                "zero_rebuild": sigma,
                "common_per_module": per_gamma * sigma,
                "common_hns": hns_sigma,
            }
            spectra.update({
                f"common_global_{gamma_tag(gamma)}": gamma * sigma for gamma in GLOBAL_GAMMAS
            })
            saved_errors: dict[str, float] = {}
            for label, target_sigma in spectra.items():
                b_new, a_new = common_factors(U, target_sigma, Vh)
                a_saved = a_new.to(dtype=a_cpu.dtype)
                b_saved = b_new.to(dtype=b_cpu.dtype)
                states[label][key_a] = a_saved.cpu()
                states[label][key_b] = b_saved.cpu()
                target_fro = torch.linalg.vector_norm(target_sigma).clamp_min(1e-12)
                error = difference_frobenius(
                    b_saved.float(), a_saved.float(), b_new, a_new
                ) / target_fro
                saved_errors[label] = float(error.item())
                max_saved_error = max(max_saved_error, saved_errors[label])
            module_stats[prefix] = {
                "lora_fro": float(torch.linalg.vector_norm(sigma).item()),
                "hns_fro": float(torch.linalg.vector_norm(hns_sigma).item()),
                "per_module_gamma": float(per_gamma.item()),
                "basis": alignment,
                "saved_relative_errors": saved_errors,
            }
            if index % 32 == 0 or index == len(cached):
                print(f"[Factor] {index}/{len(cached)} modules", flush=True)

    variants = []
    for label in labels:
        destination = output / label.replace("_", "-")
        shutil.copytree(lora_path, destination)
        save_lora_state_dict(str(destination), states[label], weight_format)
        metadata = {
            "method": "common_uv_spectral_control",
            "variant": label,
            "source_lora": str(lora_path),
            "source_hns": str(hns_path),
            "factorization": "A=Vh, B=U*diag(sigma); common U,V within each module",
        }
        (destination / "common_basis_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        variants.append({"label": label, "path": str(destination)})
        print(f"[Save] {label}: {destination}", flush=True)

    manifest = {
        "status": "complete",
        "method": "common_uv_spectral_control",
        "source_lora": str(lora_path),
        "source_hns": str(hns_path),
        "global_gamma_candidates": list(GLOBAL_GAMMAS),
        "hns_norm_matched_global_gamma": global_norm_gamma,
        "modules": len(module_stats),
        "max_hns_basis_error": max_basis_error,
        "max_saved_update_relative_error": max_saved_error,
        "variants": variants,
        "module_stats": module_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "status", "modules", "hns_norm_matched_global_gamma", "max_hns_basis_error",
        "max_saved_update_relative_error",
    )}, indent=2))


if __name__ == "__main__":
    main()
