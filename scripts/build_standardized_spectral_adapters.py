#!/usr/bin/env python3
"""Build signed, centered spectral-coefficient adapters in one shared U,V basis."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key, save_lora_state_dict
from finetune.spectral_edit.mechanism import align_edited_spectrum_to_reference, frobenius_norm_from_factors
from finetune.spectral_edit.svd import lowrank_svd_from_ba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--global_gamma", type=float, default=0.4)
    parser.add_argument("--eps", type=float, default=1e-12)
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


def common_factors(
    u: torch.Tensor, coefficients: torch.Tensor, vh: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return u * coefficients.unsqueeze(0), vh


def update_difference(
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
        print(f"[Skip] standardized adapters already built: {manifest_path}")
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

    labels = ("zero_rebuild", "zscore_variance", "zscore_std", "global_0p40", "common_hns")
    states = {label: dict(lora_state) for label in labels}
    module_stats: dict[str, dict] = {}
    total_original_sq = {label: 0.0 for label in labels}
    total_target_sq = {label: 0.0 for label in labels}
    max_saved_error = 0.0
    max_basis_error = 0.0

    with torch.inference_mode():
        for index, prefix in enumerate(sorted(lora_pairs), start=1):
            lora_pair = lora_pairs[prefix]
            hns_pair = hns_pairs[prefix]
            key_a, a_cpu = lora_pair["A"]
            key_b, b_cpu = lora_pair["B"]
            a = a_cpu.to(args.device).float()
            b = b_cpu.to(args.device).float()
            u, sigma, vh, _ = lowrank_svd_from_ba(b, a)
            hns_sigma, alignment = align_edited_spectrum_to_reference(
                u,
                vh,
                hns_pair["B"][1].to(args.device).float(),
                hns_pair["A"][1].to(args.device).float(),
            )
            basis_error = max(
                alignment["basis_offdiag_fraction"], alignment["basis_projection_residual_fraction"]
            )
            if basis_error > args.max_basis_error:
                raise RuntimeError(f"{prefix}: HNS basis error {basis_error} exceeds threshold")
            max_basis_error = max(max_basis_error, basis_error)

            mean = sigma.mean()
            variance = sigma.var(correction=0)
            std = variance.sqrt()
            if variance <= args.eps or std <= args.eps:
                raise RuntimeError(f"{prefix}: degenerate spectrum cannot be standardized")
            spectra = {
                "zero_rebuild": sigma,
                # This is the user's literal proposal. The result is a signed coefficient vector,
                # not a valid nonnegative singular-value vector.
                "zscore_variance": (sigma - mean) / variance,
                # Conventional z-score, included because 'variance' is often used colloquially
                # when standard deviation is intended.
                "zscore_std": (sigma - mean) / std,
                "global_0p40": args.global_gamma * sigma,
                "common_hns": hns_sigma,
            }
            saved_errors: dict[str, float] = {}
            original_norm = torch.linalg.vector_norm(sigma)
            for label, coefficients in spectra.items():
                b_new, a_new = common_factors(u, coefficients, vh)
                a_saved = a_new.to(dtype=a_cpu.dtype)
                b_saved = b_new.to(dtype=b_cpu.dtype)
                states[label][key_a] = a_saved.cpu()
                states[label][key_b] = b_saved.cpu()
                target_norm = torch.linalg.vector_norm(coefficients).clamp_min(args.eps)
                error = update_difference(b_saved.float(), a_saved.float(), b_new, a_new) / target_norm
                saved_errors[label] = float(error.item())
                max_saved_error = max(max_saved_error, saved_errors[label])
                total_original_sq[label] += float(original_norm.square().item())
                total_target_sq[label] += float(target_norm.square().item())
            module_stats[prefix] = {
                "rank": int(sigma.numel()),
                "mean": float(mean.item()),
                "variance_population": float(variance.item()),
                "standard_deviation_population": float(std.item()),
                "original_fro": float(original_norm.item()),
                "coefficient_min": {label: float(value.min().item()) for label, value in spectra.items()},
                "coefficient_max": {label: float(value.max().item()) for label, value in spectra.items()},
                "coefficient_fro": {
                    label: float(torch.linalg.vector_norm(value).item()) for label, value in spectra.items()
                },
                "negative_coefficients": {label: int((value < 0).sum().item()) for label, value in spectra.items()},
                "hns_basis_alignment": alignment,
                "saved_relative_errors": saved_errors,
            }
            if index % 32 == 0 or index == len(lora_pairs):
                print(f"[Factor] {index}/{len(lora_pairs)} modules", flush=True)

    variants = []
    for label in labels:
        destination = output / label.replace("_", "-")
        shutil.copytree(lora_path, destination)
        save_lora_state_dict(str(destination), states[label], weight_format)
        metadata = {
            "method": "signed_spectral_standardization",
            "variant": label,
            "source_lora": str(lora_path),
            "formula": {
                "zscore_variance": "(sigma - population_mean(sigma)) / population_variance(sigma)",
                "zscore_std": "(sigma - population_mean(sigma)) / population_std(sigma)",
            }.get(label),
            "factorization": "A=Vh, B=U*coefficients; common U,V within each module",
            "signed_coefficients": label.startswith("zscore_"),
        }
        (destination / "standardization_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        variants.append({"label": label, "path": str(destination), **metadata})
        print(f"[Save] {label}: {destination}", flush=True)

    total_original = sum(total_original_sq.values()) ** 0.5 / len(labels) ** 0.5
    total_target = {label: value**0.5 for label, value in total_target_sq.items()}
    manifest = {
        "status": "complete",
        "method": "signed_spectral_standardization",
        "source_lora": str(lora_path),
        "source_hns": str(hns_path),
        "population_statistics": True,
        "epsilon": args.eps,
        "modules": len(module_stats),
        "original_total_fro": total_original,
        "target_total_fro": total_target,
        "target_to_original_fro_ratio": {
            label: value / total_original for label, value in total_target.items()
        },
        "max_hns_basis_error": max_basis_error,
        "max_saved_update_relative_error": max_saved_error,
        "variants": variants,
        "module_stats": module_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "status", "modules", "original_total_fro", "target_total_fro",
        "target_to_original_fro_ratio", "max_hns_basis_error", "max_saved_update_relative_error",
    )}, indent=2))


if __name__ == "__main__":
    main()
