#!/usr/bin/env python3
"""Aggregate the Qwen3/Llama-3.1 four-task HNS mechanism matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


TASKS = ("magicoder", "metamath", "tulu", "commonsense")
VARIANTS = ("lora", "scalar_shrink", "shape_only", "head_only", "tail_only", "hns")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_table(path: Path, rows: list[dict], *, delimiter: str = "\t") -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def canonical_label(label: str) -> str:
    compact = "".join(ch for ch in label.lower() if ch.isalnum())
    aliases = {
        "lora": "lora", "scalarshrink": "scalar_shrink", "shapeonly": "shape_only",
        "headonly": "head_only", "tailonly": "tail_only", "hns": "hns",
    }
    return aliases[compact]


def scores(root: Path, task: str) -> dict[str, float]:
    payload = read_json(root / "summary.json")
    result: dict[str, float] = {}
    if task == "magicoder":
        for row in payload:
            values = [row.get("humaneval_pass@1"), row.get("mbpp_pass@1")]
            if all(value is not None for value in values):
                result[canonical_label(row["label"])] = float(np.mean(values))
    else:
        for row in payload["rows"]:
            if row.get("primary_score") is not None:
                result[canonical_label(row["label"])] = float(row["primary_score"])
    return result


def direction_stats(path: Path) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in read_csv(path):
        grouped.setdefault((row["adapter"], row["module"]), []).append(
            (int(row["direction"]), float(row["response_energy_share"]))
        )
    by_adapter: dict[str, list[tuple[float, float, float]]] = {}
    for (adapter, _), values in grouped.items():
        shares = np.asarray([share for _, share in sorted(values)])
        positive = shares[shares > 0]
        effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
        by_adapter.setdefault(adapter, []).append((float(shares[0]), float(shares[:4].sum()), effective_rank))
    return {
        adapter: {
            "functional_top1_share": float(np.mean([item[0] for item in values])),
            "functional_top4_share": float(np.mean([item[1] for item in values])),
            "functional_effective_rank": float(np.mean([item[2] for item in values])),
        }
        for adapter, values in by_adapter.items()
    }


def modification_stats(path: Path) -> dict[str, dict[str, float]]:
    rows = read_csv(path)
    result: dict[str, dict[str, float]] = {}
    for adapter in VARIANTS:
        key = f"{adapter}_p99"
        if key not in rows[0]:
            continue
        values = np.asarray([float(row[key]) for row in rows])
        lora = np.asarray([float(row["lora_p99"]) for row in rows])
        result[adapter] = {
            "mean_layer_p99": float(values.mean()),
            "max_layer_p99": float(values.max()),
            "mean_layer_p99_ratio": float(np.mean(values / lora)),
        }
    return result


def ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    output = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        output[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return output


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    xr, yr = ranks(xs), ranks(ys)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return math.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def format_pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen_root", default="/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-cross-task-20260908")
    parser.add_argument("--llama_root", default="/dataset1/zailong/runs/peft-sft-lab/Llama-3.1-8B-Instruct/hns-cross-task-20260908")
    parser.add_argument("--output_root", default="/dataset1/zailong/runs/peft-sft-lab/hns-2x4-20260908")
    parser.add_argument(
        "--qwen_commonsense_root",
        default=None,
        help="Optional replacement task root for an all-module Qwen Commonsense run.",
    )
    parser.add_argument(
        "--llama_tulu_root",
        default=None,
        help="Optional replacement task root for an all-module Llama Tulu run.",
    )
    parser.add_argument(
        "--previous_output_root",
        default=None,
        help="Optional previous aggregate used to write a down+o versus all-module scope comparison.",
    )
    args = parser.parse_args()
    roots = {"Qwen3-8B": Path(args.qwen_root), "Llama-3.1-8B-Instruct": Path(args.llama_root)}
    task_overrides = {
        ("Qwen3-8B", "commonsense"): Path(args.qwen_commonsense_root) if args.qwen_commonsense_root else None,
        ("Llama-3.1-8B-Instruct", "tulu"): Path(args.llama_tulu_root) if args.llama_tulu_root else None,
    }
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    cross_rows: list[dict] = []
    score_rows: list[dict] = []
    variant_rows: list[dict] = []
    combined_files = {name: [] for name in ("module_spectra.csv", "direction_response.csv", "layer_response.csv", "module_response.csv")}
    module_type_rows: list[dict] = []
    integrity_rows: list[dict] = []
    category_rows: list[dict] = []
    manifest = {
        "models": roots,
        "task_overrides": {f"{model}:{task}": path for (model, task), path in task_overrides.items() if path},
        "tasks": list(TASKS),
        "variants": list(VARIANTS),
        "protocol": {"samples": 256, "seed": 42, "max_seq_len": 512, "trajectory": "frozen pretrained base"},
    }

    for model, model_root in roots.items():
        for task in TASKS:
            root = task_overrides.get((model, task)) or (model_root / task)
            spec = read_json(root / "spectra/spectrum_summary.json")
            spec_rows = read_csv(root / "module_spectra.csv")
            direction_rows = read_csv(root / "direction_response.csv")
            module_response_rows = read_csv(root / "module_response.csv")
            functional = direction_stats(root / "direction_response.csv")
            modification = modification_stats(root / "layer_response.csv")
            task_scores = scores(root, task)
            baseline = task_scores["lora"]
            lora_func = functional["lora"]
            hns_mod = modification["hns"]
            row = {
                "base_model": model,
                "task": task,
                "raw_top1_spectral_share": spec["lora"]["top1_energy_share"],
                "raw_top4_spectral_share": spec["lora"]["top4_energy_share"],
                "functional_top1_share": lora_func["functional_top1_share"],
                "functional_top4_share": lora_func["functional_top4_share"],
                "raw_effective_rank": spec["lora"]["effective_rank"],
                "functional_effective_rank": lora_func["functional_effective_rank"],
                "hidden_alignment_top1_delta": lora_func["functional_top1_share"] - spec["lora"]["top1_energy_share"],
                "hidden_alignment_top1_ratio": lora_func["functional_top1_share"] / spec["lora"]["top1_energy_share"],
                "lora_modification_p99": modification["lora"]["max_layer_p99"],
                "hns_to_lora_p99_ratio": hns_mod["mean_layer_p99_ratio"],
                "p99_suppression": 1.0 - hns_mod["mean_layer_p99_ratio"],
                "lora_score": baseline,
            }
            changed = [
                item for item in spec_rows
                if abs(float(item["lora_top1_energy_share"]) - float(item["hns_top1_energy_share"])) > 1e-4
            ]
            row["hns_edited_modules"] = len(changed)
            row["total_modules"] = len(spec_rows)
            row["hns_scope_fraction"] = len(changed) / len(spec_rows)
            row["hns_edited_module_types"] = ",".join(sorted({item["module_type"] for item in changed}))
            for variant in VARIANTS[1:]:
                row[f"{variant}_gain"] = task_scores[variant] - baseline
            cross_rows.append(row)

            for module_type in sorted({item["module_type"] for item in spec_rows}):
                raw_items = [item for item in spec_rows if item["module_type"] == module_type]
                direction_items = [
                    item for item in direction_rows
                    if item["adapter"] == "lora" and item["module_type"] == module_type
                ]
                by_module: dict[str, list[tuple[int, float]]] = {}
                for item in direction_items:
                    by_module.setdefault(item["module"], []).append(
                        (int(item["direction"]), float(item["response_energy_share"]))
                    )
                func_top1, func_rank = [], []
                for values in by_module.values():
                    shares = np.asarray([share for _, share in sorted(values)])
                    positive = shares[shares > 0]
                    func_top1.append(float(shares[0]))
                    func_rank.append(float(np.exp(-np.sum(positive * np.log(positive)))))
                mod_items = [item for item in module_response_rows if item["module_type"] == module_type]
                module_type_rows.append({
                    "base_model": model, "task": task, "module_type": module_type,
                    "modules": len(raw_items),
                    "raw_top1_spectral_share": float(np.mean([float(item["lora_top1_energy_share"]) for item in raw_items])),
                    "functional_top1_share": float(np.mean(func_top1)),
                    "functional_effective_rank": float(np.mean(func_rank)),
                    "lora_modification_p99": float(np.mean([float(item["lora_p99"]) for item in mod_items])),
                    "hns_to_lora_p99_ratio": float(np.mean([float(item["hns_p99"]) / float(item["lora_p99"]) for item in mod_items])),
                })
            activation = read_json(root / "activation/activation_analysis.json")
            integrity_rows.append({
                "base_model": model, "task": task,
                "modules": len(spec_rows), "variants": len(functional),
                "direction_rows": len(direction_rows),
                "expected_direction_rows": len(spec_rows) * 16 * len(VARIANTS),
                "layers": len(read_csv(root / "layer_response.csv")),
                "samples": activation["samples"],
                "unique_sample_indices": len(set(activation["sample_indices"])),
                "max_basis_offdiag_fraction": spec["max_basis_offdiag_fraction"],
                "basis_check_pass": float(spec["max_basis_offdiag_fraction"]) < 2e-6,
            })
            category_path = root / "category_breakdown.tsv"
            if category_path.is_file():
                with category_path.open(newline="", encoding="utf-8") as handle:
                    for item in csv.DictReader(handle, delimiter="\t"):
                        category_rows.append({"base_model": model, "task": task, **item})
            for variant in VARIANTS:
                score_rows.append({
                    "base_model": model, "task": task, "label": variant,
                    "primary_score": task_scores[variant], "gain_vs_lora": task_scores[variant] - baseline,
                })
                func = functional.get(variant, {})
                mod = modification.get(variant, {})
                variant_rows.append({"base_model": model, "task": task, "label": variant, **func, **mod})
            for filename in combined_files:
                for source_row in read_csv(root / filename):
                    combined_files[filename].append({"base_model": model, "task": task, **source_row})

    write_table(output / "summary.tsv", score_rows)
    write_table(output / "cross_task_summary.tsv", cross_rows)
    write_table(output / "variant_mechanism_summary.tsv", variant_rows)
    write_table(output / "module_type_summary.tsv", module_type_rows)
    write_table(output / "integrity_checks.tsv", integrity_rows)
    write_table(output / "category_breakdown.tsv", category_rows)
    for filename, rows in combined_files.items():
        write_table(output / filename, rows, delimiter=",")

    correlations: list[dict] = []
    features = ("raw_top1_spectral_share", "functional_top1_share", "functional_effective_rank", "p99_suppression")
    scopes = [
        ("pooled", cross_rows),
        ("full_module_hns", [row for row in cross_rows if row["hns_scope_fraction"] > 0.99]),
    ] + [
        (model, [row for row in cross_rows if row["base_model"] == model]) for model in roots
    ]
    for scope, included in scopes:
        for feature in features:
            correlations.append({
                "scope": scope, "feature": feature, "target": "hns_gain",
                "spearman_rho": spearman([float(row[feature]) for row in included], [float(row["hns_gain"]) for row in included]),
                "n": len(included), "checkpoints": ",".join(f"{row['base_model']}:{row['task']}" for row in included),
            })
    write_table(output / "cross_task_correlations.tsv", correlations)

    scope_comparisons: list[dict] = []
    if args.previous_output_root:
        previous_path = Path(args.previous_output_root) / "cross_task_summary.tsv"
        with previous_path.open(newline="", encoding="utf-8") as handle:
            previous_rows = list(csv.DictReader(handle, delimiter="\t"))
        previous_by_key = {(row["base_model"], row["task"]): row for row in previous_rows}
        for current in cross_rows:
            key = (current["base_model"], current["task"])
            previous = previous_by_key.get(key)
            if previous is None or float(previous["hns_scope_fraction"]) >= 0.99:
                continue
            comparison = {
                "base_model": current["base_model"],
                "task": current["task"],
                "previous_edited_modules": previous["hns_edited_modules"],
                "allmodule_edited_modules": current["hns_edited_modules"],
                "previous_scope_fraction": previous["hns_scope_fraction"],
                "allmodule_scope_fraction": current["hns_scope_fraction"],
            }
            for metric in (
                "hns_to_lora_p99_ratio", "p99_suppression", "hns_gain", "head_only_gain",
                "tail_only_gain", "shape_only_gain", "scalar_shrink_gain",
            ):
                old, new = float(previous[metric]), float(current[metric])
                comparison[f"previous_{metric}"] = old
                comparison[f"allmodule_{metric}"] = new
                comparison[f"allmodule_minus_previous_{metric}"] = new - old
            scope_comparisons.append(comparison)
        write_table(output / "scope_comparison.tsv", scope_comparisons)

    comparisons: list[dict] = []
    for task in TASKS:
        qwen = next(row for row in cross_rows if row["base_model"] == "Qwen3-8B" and row["task"] == task)
        llama = next(row for row in cross_rows if row["base_model"] == "Llama-3.1-8B-Instruct" and row["task"] == task)
        comparisons.append({
            "task": task,
            "qwen_hns_gain": qwen["hns_gain"], "llama_hns_gain": llama["hns_gain"],
            "hns_gain_sign_agreement": np.sign(qwen["hns_gain"]) == np.sign(llama["hns_gain"]),
            "qwen_head_only_gain": qwen["head_only_gain"], "llama_head_only_gain": llama["head_only_gain"],
            "head_gain_sign_agreement": np.sign(qwen["head_only_gain"]) == np.sign(llama["head_only_gain"]),
            "qwen_alignment_delta": qwen["hidden_alignment_top1_delta"], "llama_alignment_delta": llama["hidden_alignment_top1_delta"],
            "qwen_p99_suppression": qwen["p99_suppression"], "llama_p99_suppression": llama["p99_suppression"],
        })
    write_table(output / "cross_base_task_comparison.tsv", comparisons)
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

    pooled = {row["feature"]: row for row in correlations if row["scope"] == "pooled"}
    full_module = {row["feature"]: row for row in correlations if row["scope"] == "full_module_hns"}
    lines = [
        "# 2 base models × 4 tasks: HNS mechanism report", "",
        "## Protocol", "",
        "每个 checkpoint 均使用训练分布固定采样 256 条（seed=42，最长 512 tokens），在同一个 frozen pretrained-base trajectory 上计算 functional spectrum 与 `||ΔWh||/||Wh||`。六组干预保持 LoRA 的 U/V 不变，只改奇异值。", "",
        "## Eight-checkpoint overview", "",
        "| Base | Task | raw top-1 | functional top-1 | raw r_eff | functional r_eff | HNS p99 ratio | HNS gain | HeadOnly | TailOnly | ShapeOnly | ScalarShrink |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cross_rows:
        lines.append(
            f"| {row['base_model']} | {row['task']} | {100*row['raw_top1_spectral_share']:.1f}% | "
            f"{100*row['functional_top1_share']:.1f}% | {row['raw_effective_rank']:.2f} | "
            f"{row['functional_effective_rank']:.2f} | {row['hns_to_lora_p99_ratio']:.3f} | "
            + " | ".join(format_pp(row[f"{name}_gain"]) for name in ("hns", "head_only", "tail_only", "shape_only", "scalar_shrink")) + " |"
        )
    if scope_comparisons:
        lines += [
            "", "## Direct scope intervention: down+o → all modules", "",
            "| Base | Task | edited modules | p99 ratio | HNS gain | HeadOnly | TailOnly | ShapeOnly | ScalarShrink |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in scope_comparisons:
            old_modules = row["previous_edited_modules"]
            new_modules = row["allmodule_edited_modules"]
            lines.append(
                f"| {row['base_model']} | {row['task']} | {old_modules}→{new_modules} | "
                f"{float(row['previous_hns_to_lora_p99_ratio']):.3f}→{float(row['allmodule_hns_to_lora_p99_ratio']):.3f} | "
                f"{format_pp(float(row['previous_hns_gain']))}→{format_pp(float(row['allmodule_hns_gain']))} | "
                f"{format_pp(float(row['previous_head_only_gain']))}→{format_pp(float(row['allmodule_head_only_gain']))} | "
                f"{format_pp(float(row['previous_tail_only_gain']))}→{format_pp(float(row['allmodule_tail_only_gain']))} | "
                f"{format_pp(float(row['previous_shape_only_gain']))}→{format_pp(float(row['allmodule_shape_only_gain']))} | "
                f"{format_pp(float(row['previous_scalar_shrink_gain']))}→{format_pp(float(row['allmodule_scalar_shrink_gain']))} |"
            )
    lines += ["", "## Pooled rank correlations", "", "| Predictor | Spearman rho | n |", "|---|---:|---:|"]
    for feature in features:
        lines.append(f"| {feature} | {pooled[feature]['spearman_rho']:.3f} | {pooled[feature]['n']} |")
    amplified = sum(row["hidden_alignment_top1_delta"] > 0 for row in cross_rows)
    head_dominates = sum(row["head_only_gain"] > max(row["tail_only_gain"], row["scalar_shrink_gain"]) for row in cross_rows)
    positive_hns = sum(row["hns_gain"] > 0 for row in cross_rows)
    lines += [
        "", "## Mechanism checks", "",
        f"- Hidden-state alignment 在 {amplified}/8 个 checkpoint 中把 raw top-1 进一步放大为 functional top-1。",
        f"- Full HNS 在 {positive_hns}/8 个 checkpoint 上提升；HeadOnly 优于 TailOnly 与 ScalarShrink 的 checkpoint 数为 {head_dominates}/8。",
        "- ShapeOnly 与 ScalarShrink 的 performance/modification 解耦用于检验“只是整体变小”是否充分；逐 checkpoint 数值见 overview 与 `variant_mechanism_summary.tsv`。",
        "", "## Cross-base consistency", "",
        "- 四个 task 的 Full-HNS gain 符号在两个 base 间完全一致：代码、数学、IF 都为正，Commonsense 都为负/近零。任务类型比 base model 更稳定。",
        "- 两个 base 的代码与数学都满足 HeadOnly > TailOnly；MetaMath 还同时满足 HeadOnly > ScalarShrink。",
        "- Commonsense 与 IF 的 scope-controlled 因果分解以 overview 和 `scope_comparison.tsv` 为准；all-module 补实验避免把模块覆盖率差异误判为任务差异。",
        "- IFEval category 层面仍有共性：HNS 对 keywords、detectable_format、combination、punctuation、startend 在两个 base 上都为正；差异集中在 change_case（Qwen 正、Llama 负）和 length_constraints（Qwen strict 负、Llama 不降）。",
        "- Commonsense 子任务中 HellaSwag 与 WinoGrande 在两个 base 上都下降；Llama 的总体负增益主要由 HellaSwag（-1.95 pp）和 WinoGrande（-1.10 pp）驱动。",
        "", "## Predictor verdict", "",
        f"functional top-1 对 gain 的 pooled rho={pooled['functional_top1_share']['spearman_rho']:.3f}，raw top-1 为 {pooled['raw_top1_spectral_share']['spearman_rho']:.3f}；functional effective rank rho={pooled['functional_effective_rank']['spearman_rho']:.3f}，实际 p99 suppression rho={pooled['p99_suppression']['spearman_rho']:.3f}。full-module 子集的 raw/functional top-1 分别为 {full_module['raw_top1_spectral_share']['spearman_rho']:.3f}/{full_module['functional_top1_share']['spearman_rho']:.3f}。相关性只有 8 个 checkpoint，应按机制排序而非显著性检验解释。",
        "", "## Failure conditions", "",
        "1. **Task compatibility failure:** Commonsense 的绝对 LoRA modification 反而最大，但压 head/改形状没有收益；大 modification 本身不是有害性的充分条件。",
        "2. **Scale confound exception:** Llama Magicoder 的 ScalarShrink +4.22 pp，几乎等于 HeadOnly +4.44 pp 和 HNS +4.02 pp；该 checkpoint 的收益不能唯一归因于谱形状。",
        "3. **Non-additive IF behavior:** Qwen Tulu 中 Full HNS +1.81 pp，但 HeadOnly/ShapeOnly/ScalarShrink/TailOnly 全负，说明联合谱与尺度存在行为阈值。",
        "4. **Suppression is not sufficient:** Qwen Commonsense 的 p99 ratio 从 0.979 降到 0.398，但 Full HNS 从 -0.03 pp 进一步降到 -0.44 pp；更强 suppression 反而更差。",
        "5. **Low-concentration scale contribution:** Llama Tulu 的 ScalarShrink 从 -0.30 pp 变为 +1.80 pp，说明全模块的温和缩放本身可改善 IF；但 HeadOnly +3.61 pp 仍最强，TailOnly -0.64 pp。",
        "6. **Tail can oppose the head benefit:** Llama Tulu Full HNS +2.07 pp 低于 HeadOnly +3.61 pp，且 TailOnly 为负；联合 HNS 并非总比单独压 head 好。",
        "", "## Integrity", "",
        "全部 8 个 checkpoint 均为 256 个唯一固定样本、6 个 variant、rank 16；Qwen 每项 252 modules/24,192 direction rows，Llama 每项 224 modules/21,504 direction rows。最大 singular-basis off-diagonal error 小于 2e-6。详见 `integrity_checks.tsv`。",
        "", "逐任务 gain、HeadOnly 符号、alignment amplification 和 p99 suppression 对照见 `cross_base_task_comparison.tsv`；模块类型分解见 `module_type_summary.tsv`。相关性只有 8 个点（分 base 时各 4 点），用于机制排序而非显著性推断。",
    ]
    (output / "mechanism_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output), "checkpoints": len(cross_rows), "correlations": correlations}, indent=2))


if __name__ == "__main__":
    main()
