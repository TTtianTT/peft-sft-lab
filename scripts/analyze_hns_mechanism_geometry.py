#!/usr/bin/env python3
"""CPU-only geometric analysis of cached HNS spectra and frozen-trajectory energies.

This performs no model inference, dose selection, downstream evaluation, or utility fitting.
Module summaries are descriptive; modules are not independent checkpoint replications.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="Also export figures; requires matplotlib")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    energies: dict[tuple, np.ndarray] = {}
    with (args.source / "direction_response.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["adapter"] not in ("lora", "hns"):
                continue
            key = (row["base_model"], row["task"], row["module"], row["adapter"])
            vector = energies.setdefault(key, np.full(16, np.nan))
            index = int(row["direction"]) - 1
            if np.isfinite(vector[index]):
                raise ValueError(f"Duplicate energy: {key}, {index}")
            vector[index] = float(row["response_energy"])

    results = []
    spectra_by_case = defaultdict(list)
    with (args.source / "module_spectra.csv").open() as handle:
        for row in csv.DictReader(handle):
            module = "layers." + row["module"].split(".layers.", 1)[1]
            case = (row["base_model"], row["task"])
            key = (*case, module)
            sigma = np.array([float(row[f"lora_s{i}"]) for i in range(1, 17)])
            edited = np.array([float(row[f"hns_s{i}"]) for i in range(1, 17)])
            energy = energies[(*key, "lora")]
            hns_energy = energies[(*key, "hns")]
            if not all(np.all(np.isfinite(v)) for v in (sigma, edited, energy, hns_energy)):
                raise ValueError(f"Nonfinite/missing values: {key}")
            if np.any(sigma <= 0) or energy.sum() <= 0 or hns_energy.sum() <= 0:
                raise ValueError(f"Degenerate spectrum or response: {key}")
            beta = edited / sigma  # Keep the original direction order, including for HNS.
            weights = energy / energy.sum()
            fit_gamma = float(weights @ beta)
            energy_ratio = float(weights @ np.square(beta))
            residual = 1 - fit_gamma**2 / energy_ratio
            if residual < -1e-10 or residual > 1 + 1e-10:
                raise ValueError(f"Invalid projection residual: {key}, {residual}")
            predicted_energy = energy * beta**2
            consistency = float(np.linalg.norm(hns_energy - predicted_energy) / np.linalg.norm(hns_energy))
            if consistency > 1e-5:
                raise ValueError(f"Energy/spectrum alignment failed: {key}, {consistency}")
            p = sigma / sigma.sum()
            p_hns = edited / edited.sum()
            majorized = bool(np.all(
                np.cumsum(np.sort(p_hns)[::-1])[:-1]
                <= np.cumsum(np.sort(p)[::-1])[:-1] + 1e-6
            ))
            results.append({
                "base_model": case[0], "task": case[1], "module": module,
                "gamma_fro": float(np.linalg.norm(edited) / np.linalg.norm(sigma)),
                "gamma_energy": float(np.sqrt(energy_ratio)),
                "gamma_fit": fit_gamma,
                "shape_residual_fraction": float(np.clip(residual, 0, 1)),
                "suppressed_original_energy_share": float(weights[beta < 1].sum()),
                "relative_distance_to_equal_spectrum": float(np.linalg.norm(edited - edited.mean()) / np.linalg.norm(edited)),
                "peak_gain_ratio": float(edited.max() / sigma.max()),
                "nuclear_relative_error": float(abs(edited.sum() / sigma.sum() - 1)),
                "participation_rank_sigma_probabilities": float(1 / (p @ p)),
                "hns_majorized_by_lora_tolerance_1e_6": majorized,
                "cached_energy_consistency_relative_error": consistency,
            })
            spectra_by_case[case].append((p, p_hns))

    def write_tsv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    write_tsv(args.output / "module_geometry.tsv", results)
    order = [(base, task) for base in ("Qwen3-8B", "Llama-3.1-8B-Instruct")
             for task in ("magicoder", "metamath", "tulu", "commonsense")]
    metrics = ("gamma_fro", "gamma_energy", "gamma_fit", "shape_residual_fraction",
               "suppressed_original_energy_share", "relative_distance_to_equal_spectrum", "peak_gain_ratio")
    summary = []
    for base, task in order:
        selected = [row for row in results if (row["base_model"], row["task"]) == (base, task)]
        if not selected:
            raise ValueError(f"Missing checkpoint: {base}/{task}")
        item = {"base_model": base, "task": task, "modules": len(selected)}
        for metric in metrics:
            values = np.array([row[metric] for row in selected])
            for name, quantile in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
                item[f"{metric}_{name}"] = float(np.quantile(values, quantile))
        item["majorization_fraction"] = float(np.mean([row["hns_majorized_by_lora_tolerance_1e_6"] for row in selected]))
        summary.append(item)
    write_tsv(args.output / "checkpoint_geometry.tsv", summary)
    audit = {
        "source": str(args.source.resolve()), "modules": len(results), "checkpoints": len(summary),
        "trajectory": "Frozen pretrained-base hidden states; cached token-energy sums, not new inference",
        "aggregation": "Unweighted module medians/IQR within checkpoint; not confidence intervals",
        "max_cached_energy_consistency_relative_error": max(row["cached_energy_consistency_relative_error"] for row in results),
        "max_nuclear_relative_error": max(row["nuclear_relative_error"] for row in results),
        "majorized_modules": sum(row["hns_majorized_by_lora_tolerance_1e_6"] for row in results),
        "fit_objective": "minimize sum_h ||HNS(h) - gamma * LoRA(h)||^2 for each module on fixed cached states",
        "limitation": "The fitted gamma and residual measure output geometry, not downstream utility or a validated selection rule.",
        "summary": summary,
    }
    (args.output / "geometry_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    print(json.dumps({key: audit[key] for key in ("modules", "checkpoints", "majorized_modules",
                                                "max_cached_energy_consistency_relative_error")}, indent=2))
    for row in summary:
        print(row["base_model"], row["task"], " ".join(f"{metric}={row[f'{metric}_median']:.4f}"
              for metric in ("gamma_fro", "gamma_energy", "gamma_fit", "shape_residual_fraction")))
    if not args.plot:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hns-mechanism-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    profiles = spectra_by_case[("Qwen3-8B", "metamath")]
    for index, color, label in ((0, "#4267AC", "LoRA"), (1, "#D56A36", "HNS")):
        values = np.stack([pair[index] for pair in profiles])
        x = np.arange(1, 17)
        axes[0].plot(x, np.median(values, axis=0), color=color, marker="o", ms=3, label=label)
        axes[0].fill_between(x, *np.quantile(values, [0.25, 0.75], axis=0), color=color, alpha=0.13)
    axes[0].axhline(1 / 16, color="#666666", ls="--", lw=1, label="Equal spectrum")
    axes[0].set(xlabel="Original singular-direction index", ylabel="Singular value / nuclear norm",
                title="A  Spectral balancing: Qwen MetaMath", yscale="log", xticks=[1, 4, 8, 12, 16])
    axes[0].legend(frameon=False, fontsize=8)
    labels = [("Qwen" if row["base_model"] == "Qwen3-8B" else "Llama") + " / " +
              {"magicoder": "Code", "metamath": "Math", "tulu": "Tulu", "commonsense": "CS"}[row["task"]]
              for row in summary]
    y = np.arange(len(summary))
    for metric, shift, color, marker, label in (
        ("gamma_fro", -0.2, "#4267AC", "o", "Frobenius matched"),
        ("gamma_energy", 0, "#D56A36", "s", "Response energy matched"),
        ("gamma_fit", 0.2, "#399B7A", "^", "Best response fit"),
    ):
        axes[1].scatter([row[f"{metric}_median"] for row in summary], y + shift, color=color,
                        marker=marker, s=30, label=label)
    axes[1].set(yticks=y, yticklabels=labels, xlabel="Module-median scalar coefficient", xlim=(0, 1),
                title="B  Parameter scale differs from functional scale")
    axes[1].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    residual = np.array([row["shape_residual_fraction_median"] for row in summary]) * 100
    bars = axes[2].barh(y, residual, color="#7465A0", alpha=0.85)
    axes[2].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    axes[2].set(yticks=y, yticklabels=labels, xlim=(0, 65), xlabel="Residual / HNS response energy (%)",
                title="C  Residual after best per-module scalar fit")
    axes[2].invert_yaxis()
    fig.suptitle("Cached frozen-base trajectories; module summaries describe geometry, not task utility", fontsize=11)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"spectral_functional_geometry.{suffix}", dpi=180)
    plt.close(fig)
if __name__ == "__main__":
    main()
