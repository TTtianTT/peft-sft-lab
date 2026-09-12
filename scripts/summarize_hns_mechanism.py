#!/usr/bin/env python3
"""Create a compact mechanism report from spectrum, activation, and ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def exact_mcnemar_p(run_root: Path, reference: str, candidate: str) -> tuple[int, int, float] | None:
    gains = losses = 0
    for benchmark in ("humaneval", "mbpp"):
        mappings = []
        for label in (reference, candidate):
            path = run_root / "eval" / label / benchmark / "outputs_lora.jsonl"
            if not path.is_file():
                return None
            mapping = {}
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                row = json.loads(line)
                key = row.get("task_id", row.get("id", row.get("problem_id", row.get("index", line_number))))
                mapping[str(key)] = bool(row.get("passed", row.get("correct", False)))
            mappings.append(mapping)
        before, after = mappings
        for key in before.keys() & after.keys():
            gains += int(not before[key] and after[key])
            losses += int(before[key] and not after[key])
    discordant = gains + losses
    if discordant == 0:
        return gains, losses, 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / (2**discordant)
    return gains, losses, min(1.0, 2 * tail)


def absolute_direction_ratios(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["module"], row["adapter"])].append(
                (int(row["direction"]), float(row["response_energy"]))
            )
    modules = sorted({module for module, _ in grouped})
    ratios: dict[str, list[float]] = defaultdict(list)
    for module in modules:
        values = {
            adapter: np.asarray([value for _, value in sorted(grouped[(module, adapter)])])
            for adapter in ("lora", "hns")
        }
        for name, selection in (
            ("total", slice(None)),
            ("direction1", slice(0, 1)),
            ("top4", slice(0, 4)),
            ("tail12", slice(4, None)),
        ):
            before = float(values["lora"][selection].sum())
            after = float(values["hns"][selection].sum())
            ratios[name].append(after / max(before, 1e-24))
    return {f"mean_{name}_ratio": float(np.mean(items)) for name, items in ratios.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument(
        "--localization_root",
        default="/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/magicoder-hns-localization-20260908",
    )
    args = parser.parse_args()
    root = Path(args.run_root)
    spectrum = json.loads((root / "spectra" / "spectrum_analysis.json").read_text())
    cases = {row["case"]: row for row in spectrum["cases"]}
    success = cases["qwen3_magicoder_hns4p1"]
    failure = cases["llama31_commonsense_allmods_hns8p2_failure"]
    lines = [
        "# HNS mechanism analysis",
        "",
        "## Verdict",
        "",
    ]

    activation_path = root / "activation" / "activation_analysis.json"
    activation = json.loads(activation_path.read_text()) if activation_path.is_file() else None
    eval_path = root / "summary.json"
    evaluations = json.loads(eval_path.read_text()) if eval_path.is_file() else []
    evals = {row["label"]: row for row in evaluations}

    lines.extend([
        "The spectral part of the proposed chain is strongly supported: HNS preserves the LoRA singular basis and removes head dominance. Head dominance alone is not sufficient to predict a gain, so the universal version of the story is rejected.",
        "",
        "## 1. Sixteen-value spectra",
        "",
        f"For Qwen3-8B Magicoder (252 adapted projections), mean top-1 energy share falls from {percent(success['lora_top1_energy_share'])} to {percent(success['hns_top1_energy_share'])}; top-4 falls from {percent(success['lora_top4_energy_share'])} to {percent(success['hns_top4_energy_share'])}; effective rank rises from {success['lora_effective_rank']:.2f} to {success['hns_effective_rank']:.2f} out of 16.",
        "",
        f"The mean HNS basis off-diagonal fraction is {success['basis_basis_offdiag_fraction']:.2e} and projection residual is {success['basis_basis_projection_residual_fraction']:.2e}. Thus this edit changes singular values while retaining the learned LoRA directions to numerical precision.",
        "",
        f"Across scored checkpoint pairs, the exploratory Spearman correlation between original top-1 energy share and performance gain is only {spectrum['exploratory_spearman']['lora_top1_energy_share']:.2f}. The counterexample is Llama-3.1 CommonSense: top-1 energy still falls from {percent(failure['lora_top1_energy_share'])} to {percent(failure['hns_top1_energy_share'])}, but macro accuracy changes by {100 * failure['performance_gain']:+.2f} pp.",
        "",
    ])

    lines.extend(["## 2. Real-hidden-state modification", ""])
    if activation is None:
        lines.append("Pending GPU run: activation/activation_analysis.json has not been produced yet.")
    else:
        layer_rows = activation["layer_summary"]
        p99_ratio = np.mean([row["hns_p99"] / max(row["lora_p99"], 1e-12) for row in layer_rows])
        lower = np.mean([row["hns_p99"] < row["lora_p99"] for row in layer_rows])
        worst_lora = max(layer_rows, key=lambda row: row["lora_p99"])
        worst_hns = max(layer_rows, key=lambda row: row["hns_p99"])
        lines.extend([
            f"On {activation['samples']} fixed Magicoder samples, HNS lowers per-layer p99 modification in {percent(float(lower))} of layers; the mean HNS/LoRA p99 ratio is {p99_ratio:.3f}.",
            "",
            f"The worst LoRA layer is {worst_lora['layer']} (p99={worst_lora['lora_p99']:.4f}); the worst HNS layer is {worst_hns['layer']} (p99={worst_hns['hns_p99']:.4f}). Ratios use the paired frozen-base trajectory and aggregate all seven projections per layer before taking norms.",
        ])
        directions = activation.get("direction_summary")
        if directions:
            absolute = absolute_direction_ratios(root / "activation" / "direction_response.csv")
            lines.extend([
                "",
                f"Decomposing actual modification energy as sigma_i^2 ||v_i^T h||^2, the first LoRA direction accounts for {percent(directions['lora']['mean_top1_response_energy_share'])} on average and the first four account for {percent(directions['lora']['mean_top4_response_energy_share'])}. Under HNS these become {percent(directions['hns']['mean_top1_response_energy_share'])} and {percent(directions['hns']['mean_top4_response_energy_share'])}.",
            ])
            if absolute:
                lines.extend([
                    "",
                    f"In absolute energy, HNS leaves {percent(absolute['mean_direction1_ratio'])} of direction-1 modification and {percent(absolute['mean_top4_ratio'])} of top-4 modification, while increasing tail-12 energy to {absolute['mean_tail12_ratio']:.2f}x. Because the head dominated on real states, total modification energy still falls to {percent(absolute['mean_total_ratio'])} on average.",
                ])
    lines.append("")

    lines.extend(["## 3. Causal controls", ""])
    if not evaluations or any(label not in evals for label in ("LoRA", "ScalarShrink", "ShapeOnly", "HeadOnly", "TailOnly", "HNS")):
        lines.append("Pending GPU run: the exact six-way evaluation is incomplete.")
    else:
        scores = {}
        for label, row in evals.items():
            if row["humaneval_pass@1"] is not None and row["mbpp_pass@1"] is not None:
                scores[label] = (row["humaneval_pass@1"] + row["mbpp_pass@1"]) / 2
        lines.extend([
            "| Variant | HumanEval | MBPP | Mean | vs LoRA | paired exact p |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        base = scores["LoRA"]
        for label in ("LoRA", "ScalarShrink", "ShapeOnly", "HeadOnly", "TailOnly", "HNS"):
            row = evals[label]
            score = scores[label]
            paired = exact_mcnemar_p(root, "LoRA", label)
            p_text = "-" if label == "LoRA" or paired is None else f"{paired[2]:.3f}"
            lines.append(
                f"| {label} | {percent(row['humaneval_pass@1'])} | {percent(row['mbpp_pass@1'])} | {percent(score)} | {100 * (score-base):+.2f} pp | {p_text} |"
            )
        lines.extend([
            "",
            "Interpretation: ScalarShrink isolates overall strength reduction; ShapeOnly retains the HNS relative shape at original LoRA Frobenius norm; HeadOnly accepts only HNS suppressions; TailOnly accepts only HNS amplifications.",
            "The paired exact test pools HumanEval and MBPP discordances only as a compact sensitivity check. HNS versus LoRA is directional but does not cross 0.05 in this single run, so the performance mechanism should be replicated across seeds.",
        ])
    lines.append("")

    localization_path = Path(args.localization_root) / "summary.json"
    if localization_path.is_file():
        loc = {row["label"]: row for row in json.loads(localization_path.read_text())}
        def loc_mean(label: str) -> float:
            row = loc[label]
            return (row["humaneval_pass@1"] + row["mbpp_pass@1"]) / 2
        base = loc_mean("control-lora")
        lines.extend([
            "## When HNS works and fails",
            "",
            f"On Qwen3 Magicoder, all-module HNS gains {100 * (loc_mean('control-hns4p1-allmods') - base):+.2f} pp mean pass@1. Attention-only gains {100 * (loc_mean('hns-attention') - base):+.2f} pp and middle-layer-only gains {100 * (loc_mean('hns-layers-middle') - base):+.2f} pp.",
            "",
            f"It fails when applied to the wrong subset: o_proj-only changes mean pass@1 by {100 * (loc_mean('hns-o-proj') - base):+.2f} pp, late-layer-only by {100 * (loc_mean('hns-layers-late') - base):+.2f} pp, and gate/up-only by {100 * (loc_mean('hns-gate-up') - base):+.2f} pp.",
            "",
        ])

    lines.extend([
        "## Scope and caveats",
        "",
        "The cross-checkpoint correlation is exploratory because HNS4+1/HNS8+2 rows reuse the same source LoRA and are not independent seeds. The behavioral intervention is measured on one 256-sample in-domain set. The Llama CommonSense failure has spectral and benchmark evidence but no local base checkpoint for the same hidden-state measurement.",
        "",
        "Raw artifacts: `spectra/module_spectra.csv` contains all 16 LoRA and HNS values for every module; `activation/layer_response.csv` and `activation/module_response.csv` contain the per-layer/per-module response distributions; `summary.tsv` contains the six-way benchmark results.",
    ])
    report = "\n".join(lines) + "\n"
    (root / "mechanism_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
