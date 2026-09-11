#!/usr/bin/env python3
"""Build common-basis LoRA/HNS/scalar controls for the forgetting study."""

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
    parser.add_argument("--gammas", nargs="+", type=float, default=(0.25, 0.40, 0.55, 0.70, 0.85, 1.0))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max_basis_error", type=float, default=5e-3)
    return parser.parse_args()


def tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def collect_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is None:
            continue
        prefix, which, _ = parsed
        pairs.setdefault(prefix, {})[which] = (key, tensor)
    return {prefix: pair for prefix, pair in pairs.items() if {"A", "B"} <= pair.keys()}


def update_difference_fro(
    b_left: torch.Tensor,
    a_left: torch.Tensor,
    b_right: torch.Tensor,
    a_right: torch.Tensor,
) -> torch.Tensor:
    return frobenius_norm_from_factors(
        torch.cat([b_left, b_right], dim=1),
        torch.cat([a_left, -a_right], dim=0),
    )


def copy_skeleton(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in {"adapter_model.safetensors", "adapter_model.bin", ".cache"}}
        ignored.update(name for name in names if name.startswith("eval"))
        return ignored

    shutil.copytree(source, destination, ignore=ignore)


def main() -> None:
    args = parse_args()
    lora_path = Path(args.lora_path).resolve()
    hns_path = Path(args.hns_path).resolve()
    output = Path(args.output_dir).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        print(f"[Skip] {manifest_path}")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output without manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)

    gammas = tuple(dict.fromkeys(float(value) for value in args.gammas))
    if not gammas or any(not 0 < value <= 1 for value in gammas) or 1.0 not in gammas:
        raise ValueError("Gammas must be unique values in (0,1] and include 1.0")

    lora_state, weight_format = load_lora_state_dict(str(lora_path))
    hns_state, _ = load_lora_state_dict(str(hns_path))
    lora_pairs = collect_pairs(lora_state)
    hns_pairs = collect_pairs(hns_state)
    if not lora_pairs or set(lora_pairs) != set(hns_pairs):
        raise RuntimeError("LoRA/HNS module sets are empty or differ")

    labels = ["common_hns", "common_per_module"]
    labels.extend(f"common_global_{tag(gamma)}" for gamma in gammas)
    states = {label: dict(lora_state) for label in labels}
    module_stats: dict[str, dict] = {}
    max_basis_error = 0.0
    max_saved_target_error = 0.0
    max_common_lora_source_error = 0.0
    total_lora_sq = 0.0
    total_hns_sq = 0.0

    with torch.inference_mode():
        for index, prefix in enumerate(sorted(lora_pairs), start=1):
            key_a, a_cpu = lora_pairs[prefix]["A"]
            key_b, b_cpu = lora_pairs[prefix]["B"]
            a = a_cpu.to(args.device).float()
            b = b_cpu.to(args.device).float()
            U, sigma, Vh, _ = lowrank_svd_from_ba(b, a)
            hns_sigma, alignment = align_edited_spectrum_to_reference(
                U,
                Vh,
                hns_pairs[prefix]["B"][1].to(args.device).float(),
                hns_pairs[prefix]["A"][1].to(args.device).float(),
            )
            basis_error = max(
                alignment["basis_offdiag_fraction"],
                alignment["basis_projection_residual_fraction"],
            )
            if basis_error > args.max_basis_error:
                raise RuntimeError(f"{prefix}: basis error {basis_error:.6g} exceeds threshold")
            max_basis_error = max(max_basis_error, basis_error)

            lora_fro = torch.linalg.vector_norm(sigma)
            hns_fro = torch.linalg.vector_norm(hns_sigma)
            total_lora_sq += float(lora_fro.square().item())
            total_hns_sq += float(hns_fro.square().item())
            per_gamma = hns_fro / lora_fro.clamp_min(1e-12)
            spectra = {
                "common_hns": hns_sigma,
                "common_per_module": per_gamma * sigma,
            }
            spectra.update({f"common_global_{tag(gamma)}": gamma * sigma for gamma in gammas})

            saved_errors: dict[str, float] = {}
            for label, target_sigma in spectra.items():
                b_target = U * target_sigma.unsqueeze(0)
                a_target = Vh
                a_saved = a_target.to(dtype=a_cpu.dtype)
                b_saved = b_target.to(dtype=b_cpu.dtype)
                states[label][key_a] = a_saved.cpu()
                states[label][key_b] = b_saved.cpu()
                target_fro = torch.linalg.vector_norm(target_sigma).clamp_min(1e-12)
                error = update_difference_fro(
                    b_saved.float(), a_saved.float(), b_target, a_target
                ) / target_fro
                saved_errors[label] = float(error.item())
                max_saved_target_error = max(max_saved_target_error, saved_errors[label])

            common_lora_b = states["common_global_1p00"][key_b].to(args.device).float()
            common_lora_a = states["common_global_1p00"][key_a].to(args.device).float()
            source_error = update_difference_fro(common_lora_b, common_lora_a, b, a) / lora_fro.clamp_min(1e-12)
            max_common_lora_source_error = max(max_common_lora_source_error, float(source_error.item()))
            module_stats[prefix] = {
                "lora_fro": float(lora_fro.item()),
                "hns_fro": float(hns_fro.item()),
                "per_module_gamma": float(per_gamma.item()),
                "basis": alignment,
                "common_lora_source_relative_error": float(source_error.item()),
                "saved_target_relative_errors": saved_errors,
            }
            if index % 32 == 0 or index == len(lora_pairs):
                print(f"[Factor] {index}/{len(lora_pairs)}", flush=True)

    variants = []
    for label in labels:
        destination = output / label.replace("_", "-")
        copy_skeleton(lora_path, destination)
        save_lora_state_dict(str(destination), states[label], weight_format)
        metadata = {
            "method": "forgetting_common_uv_control",
            "variant": label,
            "source_lora": str(lora_path),
            "source_hns": str(hns_path),
            "factorization": "A=Vh, B=U*diag(d); shared U,V within every module",
        }
        (destination / "common_basis_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        variants.append({"label": label, "path": str(destination)})
        print(f"[Save] {label}: {destination}", flush=True)

    manifest = {
        "status": "complete",
        "source_lora": str(lora_path),
        "source_hns": str(hns_path),
        "global_gamma_candidates": list(gammas),
        "hns_norm_matched_global_gamma": (total_hns_sq / total_lora_sq) ** 0.5,
        "modules": len(module_stats),
        "max_hns_basis_error": max_basis_error,
        "max_saved_target_relative_error": max_saved_target_error,
        "max_common_lora_source_relative_error": max_common_lora_source_error,
        "variants": variants,
        "module_stats": module_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "status", "modules", "hns_norm_matched_global_gamma", "max_hns_basis_error",
        "max_saved_target_relative_error", "max_common_lora_source_relative_error",
    )}, indent=2))


if __name__ == "__main__":
    main()
