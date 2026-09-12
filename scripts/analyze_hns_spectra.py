#!/usr/bin/env python3
"""Compare all 16 singular values for a bank of observed LoRA/HNS pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key
from finetune.spectral_edit.mechanism import align_edited_spectrum_to_reference, spectrum_dominance
from finetune.spectral_edit.svd import lowrank_svd_from_ba


def pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is not None:
            prefix, which, _ = parsed
            result[prefix][which] = tensor
    return {prefix: item for prefix, item in result.items() if {"A", "B"} <= item.keys()}


def mean_dict(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(np.mean([row[key] for row in rows])) for key in rows[0]}


def rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def write_key_spectra_svg(path: Path, rows: list[dict]) -> None:
    """Write a dependency-free two-panel plot for a success and failure case."""
    wanted = ("qwen3_magicoder_hns4p1", "llama31_commonsense_allmods_hns8p2_failure")
    selected = [next(row for row in rows if row["case"] == label) for label in wanted]
    width, height = 1080, 430
    panel_w, left, top, plot_h = 470, 65, 65, 285
    colors = {"LoRA": "#dc2626", "HNS": "#2563eb"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:16px;font-weight:600}.tick{font-size:11px}.label{font-size:13px}</style>',
    ]
    for panel, row in enumerate(selected):
        x0 = left + panel * 535
        ymax = max(row["mean_normalized_spectrum_lora"]) * 1.08
        for tick in range(5):
            value = ymax * tick / 4
            y = top + plot_h * (1 - tick / 4)
            svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            svg.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')
        for name, key in (("LoRA", "mean_normalized_spectrum_lora"), ("HNS", "mean_normalized_spectrum_hns")):
            values = row[key]
            points = []
            for index, value in enumerate(values):
                x = x0 + panel_w * index / max(1, len(values) - 1)
                y = top + plot_h * (1 - value / ymax)
                points.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[name]}" stroke-width="3"/>')
            for point in points:
                x, y = point.split(",")
                svg.append(f'<circle cx="{x}" cy="{y}" r="2.8" fill="{colors[name]}"/>')
        gain = 100 * float(row["performance_gain"])
        svg.append(f'<text x="{x0+panel_w/2:.1f}" y="28" text-anchor="middle" class="title">{row["case"]}</text>')
        svg.append(f'<text x="{x0+panel_w/2:.1f}" y="48" text-anchor="middle" class="label">performance gain {gain:+.2f} pp</text>')
        for index in (1, 4, 8, 12, 16):
            x = x0 + panel_w * (index - 1) / 15
            svg.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle" class="tick">{index}</text>')
        svg.append(f'<text x="{x0+panel_w/2:.1f}" y="{height-22}" text-anchor="middle" class="label">singular-value index</text>')
    svg.extend([
        '<line x1="420" y1="405" x2="450" y2="405" stroke="#dc2626" stroke-width="3"/><text x="458" y="410" class="label">LoRA</text>',
        '<line x1="550" y1="405" x2="580" y2="405" stroke="#2563eb" stroke-width="3"/><text x="588" y="410" class="label">HNS</text>',
        '</svg>',
    ])
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hns_mechanism_cases.json")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_root = Path(config["model_root"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    module_rows, case_rows = [], []

    for case in config["cases"]:
        label = case["label"]
        lora_state, _ = load_lora_state_dict(str(model_root / case["lora"]))
        hns_state, _ = load_lora_state_dict(str(model_root / case["hns"]))
        lp, hp = pairs(lora_state), pairs(hns_state)
        if set(lp) != set(hp):
            raise RuntimeError(f"{label}: module set mismatch")
        lora_stats, hns_stats, basis_stats = [], [], []
        normalized_lora, normalized_hns = [], []
        for prefix in sorted(lp):
            u, sigma_lora, vh, _ = lowrank_svd_from_ba(
                lp[prefix]["B"].to(args.device), lp[prefix]["A"].to(args.device)
            )
            sigma_hns, alignment = align_edited_spectrum_to_reference(
                u, vh, hp[prefix]["B"].to(args.device), hp[prefix]["A"].to(args.device)
            )
            ls, hs = spectrum_dominance(sigma_lora), spectrum_dominance(sigma_hns)
            lora_stats.append(ls)
            hns_stats.append(hs)
            basis_stats.append(alignment)
            lnorm = sigma_lora / sigma_lora.sum().clamp_min(1e-12)
            hnorm = sigma_hns / sigma_hns.sum().clamp_min(1e-12)
            normalized_lora.append(lnorm.cpu().numpy())
            normalized_hns.append(hnorm.cpu().numpy())
            row = {
                "case": label,
                "module": prefix,
                "layer": next((int(part) for i, part in enumerate(prefix.split(".")) if i and prefix.split(".")[i-1] == "layers"), -1),
                "module_type": prefix.rsplit(".", 1)[-1],
                **{f"lora_{key}": value for key, value in ls.items()},
                **{f"hns_{key}": value for key, value in hs.items()},
                **alignment,
            }
            for i, value in enumerate(sigma_lora.tolist(), 1):
                row[f"lora_s{i}"] = float(value)
                row[f"lora_s{i}_share"] = float(lnorm[i - 1])
            for i, value in enumerate(sigma_hns.tolist(), 1):
                row[f"hns_s{i}"] = float(value)
                row[f"hns_s{i}_share"] = float(hnorm[i - 1])
            module_rows.append(row)

        mean_l = np.mean(np.stack(normalized_lora), axis=0)
        mean_h = np.mean(np.stack(normalized_hns), axis=0)
        metrics = case.get("metrics")
        lora_score = float(np.mean(metrics["lora"])) if metrics else None
        hns_score = float(np.mean(metrics["hns"])) if metrics else None
        summary = {
            "case": label,
            "lora_path": str(model_root / case["lora"]),
            "hns_path": str(model_root / case["hns"]),
            "modules": len(lp),
            "lora_score": lora_score,
            "hns_score": hns_score,
            "performance_gain": hns_score - lora_score if metrics else None,
            **mean_dict(lora_stats, "lora"),
            **mean_dict(hns_stats, "hns"),
            **mean_dict(basis_stats, "basis"),
            "mean_normalized_spectrum_lora": mean_l.tolist(),
            "mean_normalized_spectrum_hns": mean_h.tolist(),
        }
        case_rows.append(summary)
        print(
            f"[{label}] modules={len(lp)} top1_energy {summary['lora_top1_energy_share']:.3f}"
            f"->{summary['hns_top1_energy_share']:.3f} erank {summary['lora_effective_rank']:.2f}"
            f"->{summary['hns_effective_rank']:.2f} gain={summary['performance_gain']}"
        )

    with (out / "module_spectra.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(module_rows[0]))
        writer.writeheader()
        writer.writerows(module_rows)
    with (out / "case_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [{k: v for k, v in row.items() if not isinstance(v, list)} for row in case_rows]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    scored = [row for row in case_rows if row["performance_gain"] is not None]
    correlations = {}
    for feature in ("lora_top1_energy_share", "lora_top4_energy_share", "lora_effective_rank", "hns_top1_energy_share"):
        x = rankdata([float(row[feature]) for row in scored])
        y = rankdata([float(row["performance_gain"]) for row in scored])
        correlations[feature] = float(np.corrcoef(x, y)[0, 1])
    result = {"cases": case_rows, "exploratory_spearman": correlations}
    (out / "spectrum_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_key_spectra_svg(out / "key_spectra.svg", case_rows)

    try:
        import matplotlib.pyplot as plt

        cols = 3
        rows = (len(case_rows) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3.4 * rows), squeeze=False)
        for ax, row in zip(axes.flat, case_rows):
            x = np.arange(1, len(row["mean_normalized_spectrum_lora"]) + 1)
            ax.plot(x, row["mean_normalized_spectrum_lora"], marker="o", ms=3, label="LoRA")
            ax.plot(x, row["mean_normalized_spectrum_hns"], marker="o", ms=3, label="HNS")
            gain = row["performance_gain"]
            suffix = "unscored" if gain is None else f"gain={100 * gain:+.2f} pp"
            ax.set_title(f"{row['case']}\n{suffix}", fontsize=9)
            ax.set_xticks([1, 4, 8, 12, 16])
            ax.set_xlabel("singular-value index")
            ax.set_ylabel("mean nuclear share")
            ax.grid(alpha=0.25)
        for ax in axes.flat[len(case_rows):]:
            ax.axis("off")
        axes.flat[0].legend()
        fig.tight_layout()
        fig.savefig(out / "mean_spectra.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    print(f"[Done] {out}")


if __name__ == "__main__":
    main()
