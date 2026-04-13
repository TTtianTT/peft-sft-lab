#!/usr/bin/env python3
"""Global MP-style spectrum scan over all selected LoRA modules."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    HAVE_MATPLOTLIB = True
except Exception:
    HAVE_MATPLOTLIB = False
    np = None


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key, layer_idx_from_module_prefix
from finetune.spectral_edit.rmt import estimate_mp_summary
from finetune.spectral_edit.svd import lowrank_svd_from_ba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a global MP-style scan over all edited LoRA modules.")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["down_proj", "o_proj"],
    )
    parser.add_argument("--layer_min", type=int, default=0)
    parser.add_argument("--layer_max", type=int, default=10**9)
    parser.add_argument("--rmt_tail_count", type=int, default=0)
    parser.add_argument("--rmt_edge_margin", type=float, default=0.10)
    parser.add_argument("--save_plot", action="store_true")
    return parser.parse_args()


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_csv(path: Path, rows: List[dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def short_module_name(prefix: str) -> str:
    return re.sub(r"^.*?layers\.(\d+)\.", r"L\1.", prefix)


def select_pairs(
    state_dict: Dict[str, Any],
    target_modules: List[str],
    layer_min: int,
    layer_max: int,
) -> Dict[str, dict[str, Any]]:
    pairs: Dict[str, dict[str, Any]] = {}
    target_set = set(target_modules)
    for key, tensor in state_dict.items():
        parsed = parse_lora_ab_key(key)
        if not parsed:
            continue
        prefix, which, _adapter = parsed
        suffix = prefix.split(".")[-1]
        if suffix not in target_set:
            continue
        layer_idx = layer_idx_from_module_prefix(prefix)
        if layer_idx is not None and not (layer_min <= layer_idx <= layer_max):
            continue
        pairs.setdefault(prefix, {})
        pairs[prefix][which] = tensor
    return {prefix: bundle for prefix, bundle in pairs.items() if "A" in bundle and "B" in bundle}


def representative_modules(sorted_rows: List[dict[str, Any]]) -> dict[str, Any]:
    n = len(sorted_rows)
    if n == 0:
        return {"high_noise": [], "middle_noise": [], "low_noise": []}

    high = sorted_rows[:2]
    low = sorted_rows[-2:] if n >= 2 else sorted_rows[-1:]
    mid_idx = n // 2
    mid = [sorted_rows[mid_idx]]
    selected = []
    seen = set()
    for bucket_name, bucket_rows in [("high_noise", high), ("middle_noise", mid), ("low_noise", low)]:
        cleaned = []
        for row in bucket_rows:
            if row["module_prefix"] in seen:
                continue
            seen.add(row["module_prefix"])
            cleaned.append(
                {
                    "module_prefix": row["module_prefix"],
                    "module_short": row["module_short"],
                    "module_suffix": row["module_suffix"],
                    "layer_idx": row["layer_idx"],
                    "noise_ratio": row["noise_ratio"],
                }
            )
        selected.append((bucket_name, cleaned))
    return {name: rows for name, rows in selected}


def maybe_plot_heatmap(path: Path, rows: List[dict[str, Any]]) -> None:
    if not HAVE_MATPLOTLIB or not rows:
        return

    suffixes = sorted({row["module_suffix"] for row in rows})
    layers = sorted({int(row["layer_idx"]) for row in rows if row["layer_idx"] is not None})
    if not suffixes or not layers:
        return

    suffix_to_idx = {suffix: idx for idx, suffix in enumerate(suffixes)}
    layer_to_idx = {layer: idx for idx, layer in enumerate(layers)}
    mat = np.full((len(suffixes), len(layers)), np.nan, dtype=np.float64)
    for row in rows:
        layer_idx = row["layer_idx"]
        if layer_idx is None:
            continue
        mat[suffix_to_idx[row["module_suffix"]], layer_to_idx[int(layer_idx)]] = float(row["noise_ratio"])

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.35), 2.8 + len(suffixes) * 0.6))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90)
    ax.set_yticks(range(len(suffixes)))
    ax.set_yticklabels(suffixes)
    ax.set_xlabel("Layer")
    ax.set_title("RMT bulk/noise ratio by module")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Noise ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict, _fmt = load_lora_state_dict(args.adapter_dir)
    pairs = select_pairs(
        state_dict,
        target_modules=list(args.target_modules),
        layer_min=args.layer_min,
        layer_max=args.layer_max,
    )
    if not pairs:
        raise RuntimeError("No matching LoRA modules found for the requested target modules / layer range.")

    per_module_rows: List[dict[str, Any]] = []
    per_component_rows: List[dict[str, Any]] = []

    for prefix in sorted(pairs):
        a_cpu = pairs[prefix]["A"]
        b_cpu = pairs[prefix]["B"]
        u, sigma, vh, _v = lowrank_svd_from_ba(b_cpu, a_cpu)
        del u, vh
        sigma_list = [float(v) for v in sigma.detach().cpu().tolist()]

        out_dim = int(b_cpu.shape[0])
        in_dim = int(a_cpu.shape[1])
        summary = estimate_mp_summary(
            sigma_list,
            out_dim=out_dim,
            in_dim=in_dim,
            tail_count=args.rmt_tail_count,
            edge_margin=args.rmt_edge_margin,
        )
        suffix = prefix.split(".")[-1]
        module_row = {
            "module_prefix": prefix,
            "module_short": short_module_name(prefix),
            "layer_idx": layer_idx_from_module_prefix(prefix),
            "module_suffix": suffix,
            "rank": summary["rank"],
            "out_dim": out_dim,
            "in_dim": in_dim,
            "theoretical_sigma_plus": summary["theoretical_sigma_plus"],
            "conservative_sigma_plus": summary["conservative_sigma_plus"],
            "signal_count": summary["label_counts"]["likely_signal"],
            "near_edge_count": summary["label_counts"]["near_edge"],
            "bulk_noise_count": summary["label_counts"]["likely_bulk_noise"],
            "noise_ratio": summary["noise_ratio"],
            "singular_values": json.dumps([round(v, 8) for v in sigma_list]),
        }
        per_module_rows.append(module_row)

        for comp in summary["components"]:
            per_component_rows.append(
                {
                    "module_prefix": prefix,
                    "module_short": short_module_name(prefix),
                    "layer_idx": layer_idx_from_module_prefix(prefix),
                    "module_suffix": suffix,
                    "component_index": int(comp["component_index"]),
                    "singular_value": float(comp["singular_value"]),
                    "rmt_label": comp["rmt_label"],
                    "above_theoretical_edge": bool(comp["above_theoretical_edge"]),
                    "above_conservative_edge": bool(comp["above_conservative_edge"]),
                    "theoretical_sigma_plus": summary["theoretical_sigma_plus"],
                    "conservative_sigma_plus": summary["conservative_sigma_plus"],
                    "noise_ratio": summary["noise_ratio"],
                }
            )

    sorted_by_noise = sorted(
        per_module_rows,
        key=lambda row: (float(row["noise_ratio"]), int(row["layer_idx"]) if row["layer_idx"] is not None else -1),
        reverse=True,
    )

    by_suffix: Dict[str, List[float]] = defaultdict(list)
    by_layer: Dict[int, List[float]] = defaultdict(list)
    for row in per_module_rows:
        by_suffix[row["module_suffix"]].append(float(row["noise_ratio"]))
        if row["layer_idx"] is not None:
            by_layer[int(row["layer_idx"])].append(float(row["noise_ratio"]))

    suffix_summary = [
        {
            "module_suffix": suffix,
            "n_modules": len(values),
            "mean_noise_ratio": sum(values) / len(values),
            "min_noise_ratio": min(values),
            "max_noise_ratio": max(values),
        }
        for suffix, values in sorted(by_suffix.items())
    ]
    layer_summary = [
        {
            "layer_idx": layer,
            "n_modules": len(values),
            "mean_noise_ratio": sum(values) / len(values),
            "min_noise_ratio": min(values),
            "max_noise_ratio": max(values),
        }
        for layer, values in sorted(by_layer.items())
    ]

    representative = representative_modules(sorted_by_noise)
    summary = {
        "adapter_dir": str(Path(args.adapter_dir).resolve()),
        "target_modules": list(args.target_modules),
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "module_count": len(per_module_rows),
        "highest_noise_modules": sorted_by_noise[:10],
        "lowest_noise_modules": list(reversed(sorted_by_noise[-10:])),
        "suffix_summary": suffix_summary,
        "layer_summary": layer_summary,
        "representative_modules": representative,
    }

    save_json(output_dir / "global_rmt_summary.json", summary)
    save_json(output_dir / "per_module_rmt.json", per_module_rows)
    save_json(output_dir / "per_component_rmt.json", per_component_rows)
    save_csv(
        output_dir / "per_module_rmt.csv",
        per_module_rows,
        [
            "module_prefix",
            "module_short",
            "layer_idx",
            "module_suffix",
            "rank",
            "out_dim",
            "in_dim",
            "theoretical_sigma_plus",
            "conservative_sigma_plus",
            "signal_count",
            "near_edge_count",
            "bulk_noise_count",
            "noise_ratio",
            "singular_values",
        ],
    )
    save_csv(
        output_dir / "per_component_rmt.csv",
        per_component_rows,
        [
            "module_prefix",
            "module_short",
            "layer_idx",
            "module_suffix",
            "component_index",
            "singular_value",
            "rmt_label",
            "above_theoretical_edge",
            "above_conservative_edge",
            "theoretical_sigma_plus",
            "conservative_sigma_plus",
            "noise_ratio",
        ],
    )
    save_csv(
        output_dir / "suffix_summary.csv",
        suffix_summary,
        ["module_suffix", "n_modules", "mean_noise_ratio", "min_noise_ratio", "max_noise_ratio"],
    )
    save_csv(
        output_dir / "layer_summary.csv",
        layer_summary,
        ["layer_idx", "n_modules", "mean_noise_ratio", "min_noise_ratio", "max_noise_ratio"],
    )

    if args.save_plot:
        maybe_plot_heatmap(output_dir / "noise_ratio_heatmap.png", per_module_rows)

    print(f"[GlobalRMT] Scanned {len(per_module_rows)} modules")
    if sorted_by_noise:
        print(
            f"[GlobalRMT] Highest noise ratio: {sorted_by_noise[0]['module_short']} "
            f"({sorted_by_noise[0]['noise_ratio']:.3f})"
        )
        print(
            f"[GlobalRMT] Lowest noise ratio: {sorted_by_noise[-1]['module_short']} "
            f"({sorted_by_noise[-1]['noise_ratio']:.3f})"
        )
    print(f"[GlobalRMT] Outputs: {output_dir}")


if __name__ == "__main__":
    main()
