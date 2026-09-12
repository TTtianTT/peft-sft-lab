#!/usr/bin/env python3
"""Audit HNS-derived per-module LoRA scaling as a post-hoc method candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


SHUFFLE_SEEDS = (20260911, 20260912, 20260913)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module_spectra", required=True)
    parser.add_argument("--cross_task_summary", required=True)
    parser.add_argument("--direct_dose_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_rows(path: Path, rows: list[dict], delimiter: str = ",") -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def stable_rng(seed: int, *labels: str) -> np.random.Generator:
    digest = hashlib.sha256("\0".join(labels).encode()).digest()
    offset = int.from_bytes(digest[:8], "little")
    return np.random.default_rng((seed + offset) % (2**63 - 1))


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pp(value: float) -> str:
    return f"{100 * value:+.2f}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra = read_csv(Path(args.module_spectra))
    performance = read_csv(Path(args.cross_task_summary), delimiter="\t")
    performance_by_key = {(row["base_model"], row["task"]): row for row in performance}

    module_rows: list[dict] = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for source in spectra:
        lora_fro = float(source["lora_frobenius_norm"])
        hns_fro = float(source["hns_frobenius_norm"])
        nuclear = float(source["lora_nuclear_norm"])
        rank = sum(float(source[f"lora_s{i}"]) > 0 for i in range(1, 17))
        gamma = hns_fro / lora_fro
        hhi = (lora_fro / nuclear) ** 2
        ideal_flat_gamma = nuclear / (math.sqrt(rank) * lora_fro)
        row = {
            "base_model": source["base_model"],
            "task": source["task"],
            "module": source["module"],
            "layer": int(source["layer"]),
            "module_type": source["module_type"],
            "rank": rank,
            "lora_frobenius_norm": lora_fro,
            "hns_frobenius_norm": hns_fro,
            "gamma_hns_matched": gamma,
            "lora_nuclear_norm": nuclear,
            "lora_spectral_hhi": hhi,
            "ideal_flat_gamma": ideal_flat_gamma,
            "gamma_to_ideal_flat_ratio": gamma / ideal_flat_gamma,
            "lora_top1_nuclear_share": float(source["lora_top1_nuclear_share"]),
            "lora_top1_energy_share": float(source["lora_top1_energy_share"]),
            "lora_top4_energy_share": float(source["lora_top4_energy_share"]),
            "lora_effective_rank": float(source["lora_effective_rank"]),
        }
        module_rows.append(row)
        grouped[(row["base_model"], row["task"])].append(row)

    checkpoint_rows: list[dict] = []
    module_type_rows: list[dict] = []
    shuffle_rows: list[dict] = []
    shuffle_norm_relative_errors: list[float] = []
    for (base_model, task), rows in sorted(grouped.items()):
        gammas = np.asarray([row["gamma_hns_matched"] for row in rows])
        lora_fro = np.asarray([row["lora_frobenius_norm"] for row in rows])
        hns_fro = np.asarray([row["hns_frobenius_norm"] for row in rows])
        weights = lora_fro**2
        global_gamma = float(math.sqrt(np.sum(hns_fro**2) / np.sum(lora_fro**2)))
        weighted_rmse = float(math.sqrt(np.sum(weights * (gammas - global_gamma) ** 2) / np.sum(weights)))
        for row in rows:
            row["adapter_global_gamma"] = global_gamma
            row["gamma_minus_global"] = row["gamma_hns_matched"] - global_gamma

        perf = performance_by_key[(base_model, task)]
        checkpoint_rows.append({
            "base_model": base_model,
            "task": task,
            "modules": len(rows),
            "adapter_global_gamma": global_gamma,
            "gamma_mean": float(np.mean(gammas)),
            "gamma_p05": quantile(gammas, 0.05),
            "gamma_p25": quantile(gammas, 0.25),
            "gamma_median": quantile(gammas, 0.50),
            "gamma_p75": quantile(gammas, 0.75),
            "gamma_p95": quantile(gammas, 0.95),
            "gamma_min": float(np.min(gammas)),
            "gamma_max": float(np.max(gammas)),
            "gamma_iqr": quantile(gammas, 0.75) - quantile(gammas, 0.25),
            "fro_weighted_gamma_rmse_from_global": weighted_rmse,
            "spearman_gamma_vs_top1_nuclear": spearman(
                gammas, np.asarray([row["lora_top1_nuclear_share"] for row in rows])
            ),
            "spearman_gamma_vs_top1_energy": spearman(
                gammas, np.asarray([row["lora_top1_energy_share"] for row in rows])
            ),
            "spearman_gamma_vs_effective_rank": spearman(
                gammas, np.asarray([row["lora_effective_rank"] for row in rows])
            ),
            "spearman_gamma_vs_spectral_hhi": spearman(
                gammas, np.asarray([row["lora_spectral_hhi"] for row in rows])
            ),
            "median_gamma_to_ideal_flat_ratio": quantile(
                np.asarray([row["gamma_to_ideal_flat_ratio"] for row in rows]), 0.5
            ),
            "scalar_shrink_gain": float(perf["scalar_shrink_gain"]),
            "hns_gain": float(perf["hns_gain"]),
            "head_only_gain": float(perf["head_only_gain"]),
            "shape_only_gain": float(perf["shape_only_gain"]),
        })

        by_type: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_type[row["module_type"]].append(row)
        for module_type, type_rows in sorted(by_type.items()):
            values = np.asarray([row["gamma_hns_matched"] for row in type_rows])
            module_type_rows.append({
                "base_model": base_model,
                "task": task,
                "module_type": module_type,
                "modules": len(type_rows),
                "gamma_mean": float(np.mean(values)),
                "gamma_median": quantile(values, 0.5),
                "gamma_p05": quantile(values, 0.05),
                "gamma_p95": quantile(values, 0.95),
            })

        # Predeclare three within-type shuffled allocation controls.  After
        # permutation, one checkpoint-wide correction preserves the exact HNS
        # target Frobenius norm.
        for seed in SHUFFLE_SEEDS:
            pending: list[dict] = []
            for module_type, type_rows in sorted(by_type.items()):
                ordered = sorted(type_rows, key=lambda row: row["module"])
                permutation = stable_rng(seed, base_model, task, module_type).permutation(len(ordered))
                for target, source_index in zip(ordered, permutation):
                    source = ordered[int(source_index)]
                    pending.append({
                        "base_model": base_model,
                        "task": task,
                        "seed": seed,
                        "module_type": module_type,
                        "target_module": target["module"],
                        "source_module": source["module"],
                        "lora_frobenius_norm": target["lora_frobenius_norm"],
                        "raw_permuted_gamma": source["gamma_hns_matched"],
                    })
            raw_norm_sq = sum(
                (row["raw_permuted_gamma"] * row["lora_frobenius_norm"]) ** 2 for row in pending
            )
            target_norm_sq = float(np.sum(hns_fro**2))
            correction = math.sqrt(target_norm_sq / raw_norm_sq)
            for row in pending:
                row["norm_correction"] = correction
                row["final_gamma"] = correction * row["raw_permuted_gamma"]
                shuffle_rows.append(row)
            achieved_norm_sq = sum(
                (row["final_gamma"] * row["lora_frobenius_norm"]) ** 2 for row in pending
            )
            relative_error = abs(math.sqrt(achieved_norm_sq / target_norm_sq) - 1.0)
            if relative_error > 1e-12:
                raise RuntimeError(f"shuffled total-norm mismatch: {base_model}/{task}/{seed}: {relative_error}")
            shuffle_norm_relative_errors.append(relative_error)

    # The newer direct-dose experiment matches HeadOnly rather than Full HNS.
    direct_manifest = json.loads(Path(args.direct_dose_manifest).read_text())
    head_rows: list[dict] = []
    for module, stats in direct_manifest["module_stats"].items():
        for dose_label, values in stats["doses"].items():
            head_rows.append({
                "base_model": "Qwen3-8B",
                "task": "metamath",
                "module": module,
                "layer": stats["layer"],
                "module_type": stats["module_type"],
                "dose_label": dose_label,
                "dose": values["dose"],
                "lora_fro": stats["lora_fro"],
                "head_fro": values["head_fro"],
                "scalar_gamma_head_matched": values["scalar_gamma"],
                "suppressed_directions": stats["suppressed_directions"],
            })

    head_dose_summary: list[dict] = []
    head_by_dose: dict[str, list[dict]] = defaultdict(list)
    for row in head_rows:
        head_by_dose[row["dose_label"]].append(row)
    for dose_label, rows in sorted(head_by_dose.items(), key=lambda item: float(item[1][0]["dose"])):
        gamma = np.asarray([row["scalar_gamma_head_matched"] for row in rows])
        lora_fro = np.asarray([row["lora_fro"] for row in rows])
        head_fro = np.asarray([row["head_fro"] for row in rows])
        head_dose_summary.append({
            "dose_label": dose_label,
            "dose": rows[0]["dose"],
            "modules": len(rows),
            "adapter_global_gamma": float(math.sqrt(np.sum(head_fro**2) / np.sum(lora_fro**2))),
            "gamma_mean": float(np.mean(gamma)),
            "gamma_p05": quantile(gamma, 0.05),
            "gamma_median": quantile(gamma, 0.5),
            "gamma_p95": quantile(gamma, 0.95),
            "gamma_min": float(np.min(gamma)),
            "gamma_max": float(np.max(gamma)),
        })

    write_rows(output_dir / "module_scaling.csv", module_rows)
    write_rows(output_dir / "checkpoint_summary.tsv", checkpoint_rows, delimiter="\t")
    write_rows(output_dir / "module_type_summary.tsv", module_type_rows, delimiter="\t")
    write_rows(output_dir / "shuffle_scaling_plan.csv", shuffle_rows)
    write_rows(output_dir / "headonly_direct_dose_scaling.csv", head_rows)
    write_rows(output_dir / "headonly_direct_dose_summary.tsv", head_dose_summary, delimiter="\t")

    scalar_gains = np.asarray([row["scalar_shrink_gain"] for row in checkpoint_rows])
    hns_gains = np.asarray([row["hns_gain"] for row in checkpoint_rows])
    positive_scalar = int(np.sum(scalar_gains > 0))
    gamma_all = np.asarray([row["gamma_hns_matched"] for row in module_rows])
    top1_all = np.asarray([row["lora_top1_energy_share"] for row in module_rows])
    reff_all = np.asarray([row["lora_effective_rank"] for row in module_rows])
    hhi_all = np.asarray([row["lora_spectral_hhi"] for row in module_rows])

    report = [
        "# Post-hoc LoRA normalization: CPU evidence audit",
        "",
        "This audit reinterprets the existing Full-HNS Frobenius-matched `ScalarShrink` as a candidate",
        "post-hoc method. It does not add downstream evaluations and does not claim that a global scalar",
        "or shuffled module allocation has already been tested.",
        "",
        "## What the existing scalar actually is",
        "",
        "For every module, the 2x4 control keeps the original LoRA singular-value shape and applies",
        "`gamma_m = ||Delta W_HNS,m||_F / ||Delta W_LoRA,m||_F`. Thus it is a checkpoint-derived",
        "per-module allocation rule, not one global LoRA scaling coefficient. The newer MetaMath",
        "direct-dose control is separate: it matches each HeadOnly dose rather than Full HNS.",
        "",
        "## Checkpoint-level scaling and observed performance",
        "",
        "| Base | Task | global norm-matched gamma | module gamma p05 / median / p95 | scalar gain (pp) | HNS gain (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in checkpoint_rows:
        report.append(
            f"| {row['base_model']} | {row['task']} | {fmt(row['adapter_global_gamma'])} | "
            f"{fmt(row['gamma_p05'])} / {fmt(row['gamma_median'])} / {fmt(row['gamma_p95'])} | "
            f"{pp(row['scalar_shrink_gain'])} | {pp(row['hns_gain'])} |"
        )
    report.extend([
        "",
        f"The per-module scalar point estimate is positive on {positive_scalar}/8 checkpoints. Across only eight",
        f"checkpoints, its gain has Spearman rho={spearman(scalar_gains, hns_gains):.3f} with Full-HNS gain;",
        "this is descriptive and is not a causal decomposition.",
        "",
        "## Is the allocation actually spectrum-derived?",
        "",
        f"Across {len(module_rows)} modules, gamma has Spearman rho={spearman(gamma_all, top1_all):.3f} with",
        f"top-1 spectral energy, rho={spearman(gamma_all, reff_all):.3f} with effective rank, and",
        f"rho={spearman(gamma_all, hhi_all):.3f} with spectral HHI. These strong relationships are largely",
        "structural: nuclear-norm-preserving flattening necessarily couples concentration to Frobenius shrinkage.",
        "They show that the rule is spectrum-conditioned; they do not show that the module-to-gamma assignment",
        "is task-useful.",
        "",
        "The module gamma distribution is broad within every checkpoint (see `checkpoint_summary.tsv` and",
        "`module_type_summary.tsv`). Therefore a single global coefficient is not algebraically equivalent to",
        "the existing scalar control. Whether this heterogeneity improves behavior remains untested.",
        "",
        "For the newer Qwen MetaMath HeadOnly-matched control, the full-dose module gamma p05/median/p95 is",
        f"{head_dose_summary[-1]['gamma_p05']:.3f}/{head_dose_summary[-1]['gamma_median']:.3f}/"
        f"{head_dose_summary[-1]['gamma_p95']:.3f}, with adapter-global gamma "
        f"{head_dose_summary[-1]['adapter_global_gamma']:.3f}. This is materially stronger shrinkage than the",
        "Full-HNS-matched rule because HeadOnly lowers dominant values without the compensating HNS tail lift.",
        "The two observed scalar results must therefore remain separate pieces of evidence.",
        "",
        "## Evidence boundary and next decision",
        "",
        "Existing results justify treating per-module HNS-matched scaling as a candidate method, but not yet as",
        "a contribution. The decisive next comparison is untouched LoRA vs one global norm-matched scalar vs",
        "the HNS-derived per-module scalar vs within-module-type shuffled allocations vs Full HNS. All scalar",
        "variants must match the same whole-adapter Frobenius norm, and scalar adapters should be constructed by",
        "directly scaling the original LoRA factors so that the zero-edit path is bit-identical to the original.",
        "",
        "Three fixed shuffled allocation plans are materialized in `shuffle_scaling_plan.csv`; each is permuted",
        "within module type and then corrected by one checkpoint-wide factor to match the Full-HNS total norm.",
        "Do not proceed to transfer or data-conditioned normalization unless the per-module rule beats both the",
        "global and shuffled controls on an independent split.",
        "",
    ])
    (output_dir / "mechanism_report.md").write_text("\n".join(report), encoding="utf-8")

    summary = {
        "status": "complete_cpu_only",
        "modules": len(module_rows),
        "checkpoints": len(checkpoint_rows),
        "positive_scalar_checkpoints": positive_scalar,
        "spearman_scalar_gain_vs_hns_gain": spearman(scalar_gains, hns_gains),
        "spearman_module_gamma_vs_top1_energy": spearman(gamma_all, top1_all),
        "spearman_module_gamma_vs_effective_rank": spearman(gamma_all, reff_all),
        "spearman_module_gamma_vs_spectral_hhi": spearman(gamma_all, hhi_all),
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "max_shuffle_total_norm_relative_error": max(shuffle_norm_relative_errors),
        "warning": "global and shuffled downstream behavior has not yet been evaluated",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
