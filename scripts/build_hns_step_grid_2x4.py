#!/usr/bin/env python3
"""Build a common-implementation HNS step grid for one base model's four LoRAs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key, save_lora_state_dict
from finetune.spectral_edit.mechanism import frobenius_norm_from_factors
from finetune.spectral_edit.posthoc_hns import HNSEditConfig, apply_hns_to_svd
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_config", required=True)
    parser.add_argument("--grid_config", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def collect_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor]]]:
    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is None:
            continue
        prefix, which, _ = parsed
        pairs.setdefault(prefix, {})[which] = (key, tensor)
    return {prefix: pair for prefix, pair in pairs.items() if {"A", "B"} <= pair.keys()}


def factor_error(
    b_saved: torch.Tensor,
    a_saved: torch.Tensor,
    b_target: torch.Tensor,
    a_target: torch.Tensor,
) -> float:
    denominator = frobenius_norm_from_factors(b_target, a_target).clamp_min(1e-12)
    numerator = frobenius_norm_from_factors(
        torch.cat([b_saved, b_target], dim=1),
        torch.cat([a_saved, -a_target], dim=0),
    )
    return float((numerator / denominator).item())


def step_pairs(grid: dict) -> list[tuple[int, int]]:
    pairs = [(0, 0)] if grid.get("include_svd_reconstruction_control", True) else []
    pairs.extend(
        (int(fast), int(stable))
        for fast in grid["fast_steps"]
        for stable in grid["stable_steps"]
    )
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"duplicate HNS step configurations: {pairs}")
    return pairs


def variant_name(fast: int, stable: int) -> str:
    return f"hns_f{fast}_s{stable}"


def build_checkpoint(
    row: dict,
    base_output: Path,
    grid: dict,
    configs: list[tuple[int, int]],
    device: str,
) -> dict:
    task = row["train_task"]
    source = Path(row["lora"]).resolve()
    task_root = base_output / task
    manifest_path = task_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        observed = [(item["fast_steps"], item["stable_steps"]) for item in manifest["variants"]]
        if observed != configs:
            raise RuntimeError(f"existing grid mismatch in {manifest_path}: {observed} != {configs}")
        print(f"[Skip] {row['base']}/{task}: {manifest_path}", flush=True)
        return manifest
    if task_root.exists() and any(task_root.iterdir()):
        raise FileExistsError(f"refusing nonempty output without manifest: {task_root}")
    task_root.mkdir(parents=True, exist_ok=True)

    state, weight_format = load_lora_state_dict(str(source))
    pairs = collect_pairs(state)
    if not pairs:
        raise RuntimeError(f"no complete LoRA A/B pairs in {source}")

    decompositions = {}
    with torch.inference_mode():
        for index, prefix in enumerate(sorted(pairs), start=1):
            key_a, a_cpu = pairs[prefix]["A"]
            key_b, b_cpu = pairs[prefix]["B"]
            u, sigma, vh, _ = lowrank_svd_from_ba(
                b_cpu.to(device=device, dtype=torch.float32),
                a_cpu.to(device=device, dtype=torch.float32),
            )
            decompositions[prefix] = (key_a, key_b, a_cpu, b_cpu, u, sigma, vh)
            if index % 32 == 0 or index == len(pairs):
                print(f"[SVD] {row['base']}/{task}: {index}/{len(pairs)}", flush=True)

    variant_rows = []
    for fast, stable in configs:
        name = variant_name(fast, stable)
        destination = task_root / name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite incomplete variant: {destination}")
        shutil.copytree(source, destination)
        edited_state = dict(state)
        module_stats = {}
        max_saved_error = 0.0
        before_fro_sq = 0.0
        after_fro_sq = 0.0
        before_effective_rank = []
        after_effective_rank = []
        hns_config = HNSEditConfig(
            fast_steps=fast,
            stable_steps=stable,
            preserve_nuclear_norm=bool(grid["preserve_nuclear_norm"]),
            hns_strength=float(grid["hns_strength"]),
        )
        with torch.inference_mode():
            for prefix in sorted(decompositions):
                key_a, key_b, a_cpu, b_cpu, u, sigma, vh = decompositions[prefix]
                u_new, vh_new, sigma_new, stats = apply_hns_to_svd(
                    u, vh, sigma, config=hns_config
                )
                b_new, a_new = rebuild_ba_from_uv_sigma(u_new, vh_new, sigma_new)
                a_saved = a_new.to(dtype=a_cpu.dtype)
                b_saved = b_new.to(dtype=b_cpu.dtype)
                edited_state[key_a] = a_saved.cpu()
                edited_state[key_b] = b_saved.cpu()
                saved_error = factor_error(
                    b_saved.float(), a_saved.float(), b_new.float(), a_new.float()
                )
                max_saved_error = max(max_saved_error, saved_error)
                stats["saved_update_relative_error"] = saved_error
                module_stats[prefix] = stats
                before_fro_sq += stats["fro_norm_before_full"] ** 2
                after_fro_sq += stats["fro_norm_after"] ** 2
                before_effective_rank.append(stats["effective_rank_before_full"])
                after_effective_rank.append(stats["effective_rank_after"])
        save_lora_state_dict(str(destination), edited_state, weight_format)
        summary = {
            "num_modules": len(module_stats),
            "mean_effective_rank_before": sum(before_effective_rank) / len(before_effective_rank),
            "mean_effective_rank_after": sum(after_effective_rank) / len(after_effective_rank),
            "adapter_fro_before": before_fro_sq**0.5,
            "adapter_fro_after": after_fro_sq**0.5,
            "adapter_fro_ratio": (after_fro_sq / before_fro_sq) ** 0.5,
            "max_saved_update_relative_error": max_saved_error,
        }
        metadata = {
            "meta": {
                "method": "posthoc_hns",
                "source_lora": str(source),
                "scope": "all_modules",
                "fast_steps": fast,
                "stable_steps": stable,
                "preserve_nuclear_norm": hns_config.preserve_nuclear_norm,
                "hns_strength": hns_config.hns_strength,
                "is_svd_reconstruction_control": fast == 0 and stable == 0,
            },
            "summary": summary,
            "module_stats": module_stats,
        }
        (destination / "spectral_edit_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        variant_rows.append({
            "method": name,
            "path": str(destination),
            "fast_steps": fast,
            "stable_steps": stable,
            **summary,
        })
        print(
            f"[Build] {row['base']}/{task}/{name}: "
            f"r_eff={summary['mean_effective_rank_after']:.4f} "
            f"fro_ratio={summary['adapter_fro_ratio']:.4f}",
            flush=True,
        )

    manifest = {
        "status": "complete",
        "base": row["base"],
        "task": task,
        "source_lora": str(source),
        "scope": "all_modules",
        "variants": variant_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    del decompositions, state
    if device == "cuda":
        torch.cuda.empty_cache()
    return manifest


def main() -> None:
    args = parse_args()
    task_cfg = json.loads(Path(args.task_config).read_text())
    grid = json.loads(Path(args.grid_config).read_text())
    if args.base not in task_cfg["bases"]:
        raise ValueError(f"unknown base: {args.base}")
    configs = step_pairs(grid)
    output = Path(args.output_dir).resolve() / args.base
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = [row for row in task_cfg["checkpoints"] if row["base"] == args.base]
    if len(checkpoints) != 4:
        raise RuntimeError(f"expected four checkpoints for {args.base}, got {len(checkpoints)}")

    variants = []
    task_manifests = []
    for row in checkpoints:
        manifest = build_checkpoint(row, output, grid, configs, args.device)
        task = row["train_task"]
        task_manifests.append(str(output / task / "manifest.json"))
        variants.append({
            "label": f"{task}__original_lora",
            "path": str(Path(row["lora"]).resolve()),
            "train_task": task,
            "method": "original_lora",
        })
        for item in manifest["variants"]:
            variants.append({
                "label": f"{task}__{item['method']}",
                "path": item["path"],
                "train_task": task,
                "method": item["method"],
                "fast_steps": item["fast_steps"],
                "stable_steps": item["stable_steps"],
            })

    result = {
        "status": "complete",
        "base": args.base,
        "base_model": str(Path(task_cfg["bases"][args.base]).resolve()),
        "task_config": str(Path(args.task_config).resolve()),
        "grid_config": str(Path(args.grid_config).resolve()),
        "step_configurations": configs,
        "task_manifests": task_manifests,
        "variants": variants,
    }
    destination = output / "variant_manifest.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[Manifest] {destination} ({len(variants)} variants)", flush=True)


if __name__ == "__main__":
    main()
