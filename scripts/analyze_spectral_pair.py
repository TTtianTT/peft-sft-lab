#!/usr/bin/env python3
"""Write parameter-space spectrum statistics for one observed LoRA/HNS pair."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from finetune.spectral_edit.io import layer_idx_from_module_prefix, load_lora_state_dict, parse_lora_ab_key
from finetune.spectral_edit.mechanism import align_edited_spectrum_to_reference, spectrum_dominance
from finetune.spectral_edit.svd import lowrank_svd_from_ba


def collect(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is not None:
            prefix, which, _ = parsed
            result[prefix][which] = tensor
    return {prefix: pair for prefix, pair in result.items() if {"A", "B"} <= pair.keys()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lora_state, _ = load_lora_state_dict(args.lora_path)
    hns_state, _ = load_lora_state_dict(args.hns_path)
    lora, hns = collect(lora_state), collect(hns_state)
    if set(lora) != set(hns):
        raise RuntimeError("LoRA/HNS module sets differ")
    rows = []
    normalized: dict[str, list[np.ndarray]] = {"lora": [], "hns": []}
    stats: dict[str, list[dict[str, float]]] = {"lora": [], "hns": []}
    alignments = []
    for prefix in sorted(lora):
        u, sigma_lora, vh, _ = lowrank_svd_from_ba(
            lora[prefix]["B"].to(args.device), lora[prefix]["A"].to(args.device)
        )
        sigma_hns, alignment = align_edited_spectrum_to_reference(
            u,
            vh,
            hns[prefix]["B"].to(args.device),
            hns[prefix]["A"].to(args.device),
        )
        spectra = {"lora": sigma_lora, "hns": sigma_hns}
        row = {
            "module": prefix,
            "layer": layer_idx_from_module_prefix(prefix),
            "module_type": prefix.rsplit(".", 1)[-1],
            **alignment,
        }
        alignments.append(alignment)
        for adapter, sigma in spectra.items():
            share = sigma / sigma.sum().clamp_min(1e-12)
            normalized[adapter].append(share.cpu().numpy())
            current = spectrum_dominance(sigma)
            stats[adapter].append(current)
            row.update({f"{adapter}_{key}": value for key, value in current.items()})
            for index, (value, fraction) in enumerate(zip(sigma.tolist(), share.tolist()), start=1):
                row[f"{adapter}_s{index}"] = float(value)
                row[f"{adapter}_s{index}_share"] = float(fraction)
        rows.append(row)
    with (out / "module_spectra.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "lora_path": str(Path(args.lora_path).resolve()),
        "hns_path": str(Path(args.hns_path).resolve()),
        "modules": len(rows),
        "mean_basis_offdiag_fraction": float(np.mean([x["basis_offdiag_fraction"] for x in alignments])),
        "max_basis_offdiag_fraction": float(np.max([x["basis_offdiag_fraction"] for x in alignments])),
        "mean_basis_projection_residual_fraction": float(np.mean([x["basis_projection_residual_fraction"] for x in alignments])),
    }
    for adapter in ("lora", "hns"):
        summary[adapter] = {
            key: float(np.mean([row[key] for row in stats[adapter]]))
            for key in stats[adapter][0]
        }
        summary[adapter]["mean_normalized_singular_values"] = np.mean(
            np.stack(normalized[adapter]), axis=0
        ).tolist()
    (out / "spectrum_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
