#!/usr/bin/env python3
"""Build HeadOnly while restoring each LoRA module's original Frobenius norm."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key, save_lora_state_dict
from finetune.spectral_edit.mechanism import align_edited_spectrum_to_reference
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    result: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, value in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is not None:
            prefix, which, _ = parsed
            result.setdefault(prefix, {})[which] = (key, value)
    return {prefix: item for prefix, item in result.items() if {"A", "B"} <= item.keys()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_basis_error", type=float, default=5e-3)
    args = parser.parse_args()

    source, target = Path(args.lora_path).resolve(), Path(args.hns_path).resolve()
    output = Path(args.out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)

    state, weight_format = load_lora_state_dict(str(source))
    edited_state, _ = load_lora_state_dict(str(target))
    source_pairs, edited_pairs = pairs(state), pairs(edited_state)
    if set(source_pairs) != set(edited_pairs):
        raise RuntimeError("LoRA/HNS module sets differ")

    module_stats: dict[str, dict] = {}
    with torch.inference_mode():
        for index, prefix in enumerate(sorted(source_pairs), start=1):
            key_a, a_cpu = source_pairs[prefix]["A"]
            key_b, b_cpu = source_pairs[prefix]["B"]
            _, a_hns_cpu = edited_pairs[prefix]["A"]
            _, b_hns_cpu = edited_pairs[prefix]["B"]
            a, b = a_cpu.to(args.device), b_cpu.to(args.device)
            u, sigma, vh, _ = lowrank_svd_from_ba(b, a)
            sigma_hns, alignment = align_edited_spectrum_to_reference(
                u, vh, b_hns_cpu.to(args.device), a_hns_cpu.to(args.device)
            )
            basis_error = max(alignment["basis_offdiag_fraction"], alignment["basis_projection_residual_fraction"])
            if basis_error > args.max_basis_error:
                raise RuntimeError(f"{prefix}: basis error {basis_error:.6g} exceeds {args.max_basis_error}")
            head_only = torch.minimum(sigma, sigma_hns)
            restore_scale = torch.linalg.vector_norm(sigma) / torch.linalg.vector_norm(head_only).clamp_min(1e-12)
            restored = head_only * restore_scale
            b_new, a_new = rebuild_ba_from_uv_sigma(u, vh, restored)
            state[key_a] = a_new.to(dtype=a_cpu.dtype, device="cpu")
            state[key_b] = b_new.to(dtype=b_cpu.dtype, device="cpu")
            module_stats[prefix] = {
                **alignment,
                "restore_scale": float(restore_scale),
                "lora_fro": float(torch.linalg.vector_norm(sigma)),
                "head_only_fro_before_restore": float(torch.linalg.vector_norm(head_only)),
                "restored_fro": float(torch.linalg.vector_norm(restored)),
                "num_suppressed": int((sigma_hns < sigma - 1e-12).sum()),
            }
            if index % 32 == 0 or index == len(source_pairs):
                print(f"[Build] {index}/{len(source_pairs)} modules", flush=True)

    save_lora_state_dict(str(output), state, weight_format)
    metadata = {
        "method": "head_only_frobenius_restore",
        "definition": "Apply only HNS suppressions, then rescale each module spectrum to its original LoRA Frobenius norm.",
        "source_lora": str(source),
        "source_hns": str(target),
        "num_modules": len(module_stats),
        "max_basis_error": max(
            max(row["basis_offdiag_fraction"], row["basis_projection_residual_fraction"])
            for row in module_stats.values()
        ),
        "mean_restore_scale": sum(row["restore_scale"] for row in module_stats.values()) / len(module_stats),
        "module_stats": module_stats,
    }
    (output / "spectral_control_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[Done] {output}")


if __name__ == "__main__":
    main()
