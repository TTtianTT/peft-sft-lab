#!/usr/bin/env python3
"""Build norm-restored signed spectral-standardization adapters for one 2x4 base."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key, save_lora_state_dict
from finetune.spectral_edit.mechanism import frobenius_norm_from_factors
from finetune.spectral_edit.svd import lowrank_svd_from_ba


MODES = (
    "restore_lora_fro",
    "restore_lora_nuclear",
    "restore_hns_fro",
    "restore_hns_nuclear",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--max_norm_error", type=float, default=1e-6)
    parser.add_argument("--max_denominator_equivalence_error", type=float, default=2e-6)
    return parser.parse_args()


def collect_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is not None:
            prefix, which, _ = parsed
            pairs.setdefault(prefix, {})[which] = (key, tensor)
    return {prefix: pair for prefix, pair in pairs.items() if {"A", "B"} <= pair.keys()}


def update_difference(
    b_left: torch.Tensor,
    a_left: torch.Tensor,
    b_right: torch.Tensor,
    a_right: torch.Tensor,
) -> torch.Tensor:
    return frobenius_norm_from_factors(
        torch.cat([b_left, b_right], dim=1), torch.cat([a_left, -a_right], dim=0)
    )


def restore(values: torch.Tensor, target: torch.Tensor, norm: str, eps: float) -> torch.Tensor:
    if norm == "fro":
        current = torch.linalg.vector_norm(values)
    elif norm == "nuclear":
        current = values.abs().sum()
    else:
        raise ValueError(norm)
    return values * (target / current.clamp_min(eps))


def build_checkpoint(row: dict, output: Path, args: argparse.Namespace) -> dict:
    task = row["train_task"]
    source = Path(row["lora"]).resolve()
    hns_source = Path(row["hns"]).resolve()
    task_root = output / task
    manifest_path = task_root / "manifest.json"
    if manifest_path.is_file():
        print(f"[Skip] {row['base']}/{task}: {manifest_path}", flush=True)
        return json.loads(manifest_path.read_text())
    if task_root.exists() and any(task_root.iterdir()):
        raise FileExistsError(f"refusing nonempty output without manifest: {task_root}")
    task_root.mkdir(parents=True, exist_ok=True)

    state, weight_format = load_lora_state_dict(str(source))
    hns_state, _ = load_lora_state_dict(str(hns_source))
    pairs = collect_pairs(state)
    hns_pairs = collect_pairs(hns_state)
    if set(pairs) != set(hns_pairs):
        raise RuntimeError(f"LoRA/HNS module mismatch for {row['base']}/{task}")
    states = {mode: dict(state) for mode in MODES}
    module_stats = {}
    max_saved_error = 0.0
    max_norm_error = 0.0
    max_denominator_equivalence_error = 0.0

    with torch.inference_mode():
        for index, prefix in enumerate(sorted(pairs), start=1):
            key_a, a_cpu = pairs[prefix]["A"]
            key_b, b_cpu = pairs[prefix]["B"]
            a = a_cpu.to(args.device).float()
            b = b_cpu.to(args.device).float()
            hns_a = hns_pairs[prefix]["A"][1].to(args.device).float()
            hns_b = hns_pairs[prefix]["B"][1].to(args.device).float()
            u, sigma, vh, _ = lowrank_svd_from_ba(b, a)
            _, hns_sigma, _, _ = lowrank_svd_from_ba(hns_b, hns_a)

            mean = sigma.mean()
            variance = sigma.var(correction=0)
            std = variance.sqrt()
            if variance <= args.eps:
                raise RuntimeError(f"{prefix}: degenerate spectrum")
            standardized = (sigma - mean) / std
            standardized_by_variance = (sigma - mean) / variance
            targets = {
                "restore_lora_fro": torch.linalg.vector_norm(sigma),
                "restore_lora_nuclear": sigma.sum(),
                "restore_hns_fro": torch.linalg.vector_norm(hns_sigma),
                "restore_hns_nuclear": hns_sigma.sum(),
            }
            norm_by_mode = {
                "restore_lora_fro": "fro",
                "restore_lora_nuclear": "nuclear",
                "restore_hns_fro": "fro",
                "restore_hns_nuclear": "nuclear",
            }
            coefficients = {
                mode: restore(standardized, targets[mode], norm_by_mode[mode], args.eps)
                for mode in MODES
            }
            denominator_errors = {}
            achieved_norm_errors = {}
            saved_errors = {}
            for mode, values in coefficients.items():
                variance_values = restore(
                    standardized_by_variance, targets[mode], norm_by_mode[mode], args.eps
                )
                denominator_error = (
                    torch.linalg.vector_norm(values - variance_values)
                    / torch.linalg.vector_norm(values).clamp_min(args.eps)
                )
                denominator_errors[mode] = float(denominator_error.item())
                max_denominator_equivalence_error = max(
                    max_denominator_equivalence_error, denominator_errors[mode]
                )

                achieved = (
                    torch.linalg.vector_norm(values)
                    if norm_by_mode[mode] == "fro"
                    else values.abs().sum()
                )
                norm_error = (achieved - targets[mode]).abs() / targets[mode].clamp_min(args.eps)
                achieved_norm_errors[mode] = float(norm_error.item())
                max_norm_error = max(max_norm_error, achieved_norm_errors[mode])

                b_new = u * values.unsqueeze(0)
                a_new = vh
                b_saved = b_new.to(dtype=b_cpu.dtype)
                a_saved = a_new.to(dtype=a_cpu.dtype)
                states[mode][key_a] = a_saved.cpu()
                states[mode][key_b] = b_saved.cpu()
                scale = torch.linalg.vector_norm(values).clamp_min(args.eps)
                saved_error = update_difference(
                    b_saved.float(), a_saved.float(), b_new, a_new
                ) / scale
                saved_errors[mode] = float(saved_error.item())
                max_saved_error = max(max_saved_error, saved_errors[mode])

            module_stats[prefix] = {
                "rank": int(sigma.numel()),
                "mean": float(mean.item()),
                "population_variance": float(variance.item()),
                "population_std": float(std.item()),
                "lora_fro": float(torch.linalg.vector_norm(sigma).item()),
                "lora_nuclear": float(sigma.sum().item()),
                "hns_fro": float(torch.linalg.vector_norm(hns_sigma).item()),
                "hns_nuclear": float(hns_sigma.sum().item()),
                "negative_coefficients": {
                    mode: int((values < 0).sum().item()) for mode, values in coefficients.items()
                },
                "denominator_equivalence_relative_error": denominator_errors,
                "target_norm_relative_error": achieved_norm_errors,
                "saved_update_relative_error": saved_errors,
            }
            if index % 32 == 0 or index == len(pairs):
                print(f"[Factor] {row['base']}/{task}: {index}/{len(pairs)}", flush=True)

    if max_norm_error > args.max_norm_error:
        raise RuntimeError(f"target norm relative error {max_norm_error} exceeds threshold")
    if max_denominator_equivalence_error > args.max_denominator_equivalence_error:
        raise RuntimeError(
            "variance/std restored transforms differ by "
            f"{max_denominator_equivalence_error}, exceeding threshold"
        )

    variants = []
    for mode in MODES:
        destination = task_root / mode.replace("_", "-")
        shutil.copytree(source, destination)
        save_lora_state_dict(str(destination), states[mode], weight_format)
        target_source = "lora" if "_lora_" in mode else "hns"
        target_norm = "fro" if mode.endswith("_fro") else "nuclear"
        metadata = {
            "method": mode,
            "formula": "z=(sigma-population_mean(sigma))/population_std(sigma); z rescaled to target norm",
            "target_source": target_source,
            "target_norm": target_norm,
            "signed_spectral_coefficients": True,
            "source_lora": str(source),
            "source_hns": str(hns_source),
            "factorization": "A=Vh, B=U*coefficients; U,V are from original LoRA",
        }
        (destination / "restored_standardization_meta.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        variants.append({"method": mode, "path": str(destination)})

    manifest = {
        "status": "complete",
        "base": row["base"],
        "task": task,
        "source_lora": str(source),
        "source_hns": str(hns_source),
        "modules": len(pairs),
        "population_statistics": True,
        "denominator_equivalence": (
            "After norm restoration, dividing by variance or standard deviation is scale-equivalent."
        ),
        "max_denominator_equivalence_relative_error": max_denominator_equivalence_error,
        "max_target_norm_relative_error": max_norm_error,
        "max_saved_update_relative_error": max_saved_error,
        "variants": variants,
        "module_stats": module_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.base not in cfg["bases"]:
        raise ValueError(f"unknown base {args.base}")
    output = Path(args.output_dir).resolve() / args.base
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = [row for row in cfg["checkpoints"] if row["base"] == args.base]
    if len(checkpoints) != 4:
        raise RuntimeError(f"expected four checkpoints for {args.base}, got {len(checkpoints)}")
    variants = []
    checkpoint_manifests = []
    for row in checkpoints:
        checkpoint = build_checkpoint(row, output, args)
        checkpoint_manifests.append(str(output / row["train_task"] / "manifest.json"))
        task = row["train_task"]
        additions = [
            ("original_lora", str(Path(row["lora"]).resolve())),
            ("hns", str(Path(row["hns"]).resolve())),
        ]
        additions.extend((item["method"], item["path"]) for item in checkpoint["variants"])
        for method, path in additions:
            variants.append({
                "label": f"{task}__{method}",
                "path": path,
                "train_task": task,
                "method": method,
            })
    manifest = {
        "status": "complete",
        "base": args.base,
        "base_model": str(Path(cfg["bases"][args.base]).resolve()),
        "checkpoint_manifests": checkpoint_manifests,
        "variants": variants,
    }
    destination = output / "variant_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[Manifest] {destination} ({len(variants)} variants)")


if __name__ == "__main__":
    main()
