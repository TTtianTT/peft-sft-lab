#!/usr/bin/env python3
"""Build per-module signed spectral-standardization adapters for one 2x4 base."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key, save_lora_state_dict
from finetune.spectral_edit.mechanism import frobenius_norm_from_factors
from finetune.spectral_edit.svd import lowrank_svd_from_ba


MODES = ("zscore_variance", "zscore_std")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--eps", type=float, default=1e-12)
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


def build_checkpoint(row: dict, output: Path, device: str, eps: float) -> dict:
    task = row["train_task"]
    source = Path(row["lora"]).resolve()
    task_root = output / task
    manifest_path = task_root / "manifest.json"
    if manifest_path.is_file():
        print(f"[Skip] {row['base']}/{task}: {manifest_path}", flush=True)
        return json.loads(manifest_path.read_text())
    if task_root.exists() and any(task_root.iterdir()):
        raise FileExistsError(f"refusing nonempty output without manifest: {task_root}")
    task_root.mkdir(parents=True, exist_ok=True)

    state, weight_format = load_lora_state_dict(str(source))
    pairs = collect_pairs(state)
    states = {mode: dict(state) for mode in MODES}
    module_stats = {}
    original_total_sq = 0.0
    target_total_sq = {mode: 0.0 for mode in MODES}
    max_saved_error = 0.0
    with torch.inference_mode():
        for index, prefix in enumerate(sorted(pairs), start=1):
            key_a, a_cpu = pairs[prefix]["A"]
            key_b, b_cpu = pairs[prefix]["B"]
            a = a_cpu.to(device).float()
            b = b_cpu.to(device).float()
            u, sigma, vh, _ = lowrank_svd_from_ba(b, a)
            mean = sigma.mean()
            variance = sigma.var(correction=0)
            std = variance.sqrt()
            if variance <= eps:
                raise RuntimeError(f"{prefix}: degenerate spectrum")
            coefficients = {
                "zscore_variance": (sigma - mean) / variance,
                "zscore_std": (sigma - mean) / std,
            }
            original_total_sq += float(torch.linalg.vector_norm(sigma).square().item())
            saved_errors = {}
            for mode, values in coefficients.items():
                b_new = u * values.unsqueeze(0)
                a_new = vh
                b_saved = b_new.to(dtype=b_cpu.dtype)
                a_saved = a_new.to(dtype=a_cpu.dtype)
                states[mode][key_a] = a_saved.cpu()
                states[mode][key_b] = b_saved.cpu()
                target_norm = torch.linalg.vector_norm(values)
                target_total_sq[mode] += float(target_norm.square().item())
                error = update_difference(b_saved.float(), a_saved.float(), b_new, a_new) / target_norm
                saved_errors[mode] = float(error.item())
                max_saved_error = max(max_saved_error, saved_errors[mode])
            module_stats[prefix] = {
                "rank": int(sigma.numel()),
                "mean": float(mean.item()),
                "population_variance": float(variance.item()),
                "population_std": float(std.item()),
                "original_fro": float(torch.linalg.vector_norm(sigma).item()),
                "target_fro": {
                    mode: float(torch.linalg.vector_norm(values).item())
                    for mode, values in coefficients.items()
                },
                "negative_coefficients": {
                    mode: int((values < 0).sum().item()) for mode, values in coefficients.items()
                },
                "saved_relative_error": saved_errors,
            }
            if index % 32 == 0 or index == len(pairs):
                print(f"[Factor] {row['base']}/{task}: {index}/{len(pairs)}", flush=True)

    original_total = original_total_sq**0.5
    variants = []
    for mode in MODES:
        destination = task_root / mode.replace("_", "-")
        shutil.copytree(source, destination)
        save_lora_state_dict(str(destination), states[mode], weight_format)
        metadata = {
            "method": mode,
            "formula": (
                "(sigma - population_mean(sigma)) / population_variance(sigma)"
                if mode == "zscore_variance"
                else "(sigma - population_mean(sigma)) / population_std(sigma)"
            ),
            "signed_spectral_coefficients": True,
            "source_lora": str(source),
            "factorization": "A=Vh, B=U*coefficients",
        }
        (destination / "spectral_standardization_meta.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        variants.append({"method": mode, "path": str(destination)})
    manifest = {
        "status": "complete",
        "base": row["base"],
        "task": task,
        "source_lora": str(source),
        "source_hns": str(Path(row["hns"]).resolve()),
        "modules": len(pairs),
        "population_statistics": True,
        "original_total_fro": original_total,
        "target_total_fro": {mode: value**0.5 for mode, value in target_total_sq.items()},
        "target_to_original_fro_ratio": {
            mode: value**0.5 / original_total for mode, value in target_total_sq.items()
        },
        "max_saved_update_relative_error": max_saved_error,
        "variants": variants,
        "module_stats": module_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def safe(value: str) -> str:
    return value.replace("/", "-").replace(".", "p")


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
        checkpoint = build_checkpoint(row, output, args.device, args.eps)
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
