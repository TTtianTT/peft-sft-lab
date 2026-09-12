#!/usr/bin/env python3
"""Create the four-task HNS mechanism table, correlations, and concise report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


MAGIC_ROOT = Path("/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B/hns-mechanism-20260908")
SPECTRUM_ALL = MAGIC_ROOT / "spectra/spectrum_analysis.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_tsv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def ensure_response_schema(path: Path) -> None:
    """Backfill explicit p50 and p99-ratio fields into earlier completed runs."""
    rows = read_csv(path)
    for row in rows:
        for key, value in list(row.items()):
            if key.endswith("_median"):
                row.setdefault(key.removesuffix("_median") + "_p50", value)
        lora_p99 = float(row["lora_p99"])
        for key, value in list(row.items()):
            if key.endswith("_p99") and key != "lora_p99":
                adapter = key.removesuffix("_p99")
                row.setdefault(f"{adapter}_to_lora_p99_ratio", float(value) / lora_p99)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def experiment_scores(path: Path, magicoder: bool = False) -> dict[str, float]:
    payload = read_json(path)
    scores = {}
    if magicoder:
        aliases = {
            "lora": "lora",
            "scalarshrink": "scalar_shrink",
            "shapeonly": "shape_only",
            "headonly": "head_only",
            "tailonly": "tail_only",
            "hns": "hns",
        }
        for row in payload:
            scores[aliases[row["label"].lower()]] = np.mean(
                [float(row["humaneval_pass@1"]), float(row["mbpp_pass@1"])]
            ).item()
    else:
        for row in payload["rows"]:
            if row["primary_score"] is not None:
                scores[row["label"].lower()] = float(row["primary_score"])
    return scores


def direction_stats(path: Path, adapter: str = "lora") -> tuple[float, float, float]:
    rows = [row for row in read_csv(path) if row["adapter"] == adapter]
    by_module: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        by_module.setdefault(row["module"], []).append(
            (int(row["direction"]), float(row["response_energy_share"]))
        )
    top1, top4, ranks = [], [], []
    for values in by_module.values():
        shares = np.asarray([share for _, share in sorted(values)])
        top1.append(shares[0])
        top4.append(shares[:4].sum())
        positive = shares[shares > 0]
        ranks.append(np.exp(-np.sum(positive * np.log(positive))))
    return float(np.mean(top1)), float(np.mean(top4)), float(np.mean(ranks))


def modification_stats(path: Path) -> tuple[float, float, float]:
    rows = read_csv(path)
    lora_p99 = max(float(row["lora_p99"]) for row in rows)
    ratios = [float(row["hns_p99"]) / float(row["lora_p99"]) for row in rows]
    ratio = float(np.mean(ratios))
    return lora_p99, ratio, 1.0 - ratio


def gains(scores: dict[str, float]) -> dict[str, float | None]:
    base = scores["lora"]
    return {
        f"{name}_gain": scores.get(name, math.nan) - base if name in scores else None
        for name in ("hns", "head_only", "tail_only", "shape_only", "scalar_shrink")
    }


def average_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    xr, yr = average_ranks(xs), average_ranks(ys)
    if len(xs) < 2 or np.std(xr) == 0 or np.std(yr) == 0:
        return math.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    spectrum_cases = {item["case"]: item for item in read_json(SPECTRUM_ALL)["cases"]}
    task_roots = {"MetaMath": root / "metamath", "Tulu": root / "tulu"}
    rows: list[dict] = []

    magic_spec = spectrum_cases["qwen3_magicoder_hns4p1"]
    magic_func = direction_stats(MAGIC_ROOT / "activation/direction_response.csv")
    magic_mod = modification_stats(MAGIC_ROOT / "activation/layer_response.csv")
    magic_scores = experiment_scores(MAGIC_ROOT / "summary.json", magicoder=True)
    rows.append({
        "task": "Magicoder",
        "checkpoint_family": "Qwen3-8B",
        "raw_top1_spectral_share": magic_spec["lora_top1_energy_share"],
        "raw_top4_spectral_share": magic_spec["lora_top4_energy_share"],
        "functional_top1_share": magic_func[0],
        "functional_top4_share": magic_func[1],
        "raw_effective_rank": magic_spec["lora_effective_rank"],
        "functional_effective_rank": magic_func[2],
        "lora_modification_p99": magic_mod[0],
        "hns_to_lora_p99_ratio": magic_mod[1],
        "p99_suppression": magic_mod[2],
        **gains(magic_scores),
        "functional_available": True,
        "notes": "positive control; HumanEval/MBPP mean",
    })

    for task, case_name in (("MetaMath", "qwen3_metamath_hns4p1"), ("Tulu", "qwen3_instruction_hns4p1")):
        task_root = task_roots[task]
        spec = read_json(task_root / "spectra/spectrum_summary.json")
        func = direction_stats(task_root / "activation/direction_response.csv")
        mod = modification_stats(task_root / "activation/layer_response.csv")
        scores = experiment_scores(task_root / "summary.json")
        rows.append({
            "task": task,
            "checkpoint_family": "Qwen3-8B",
            "raw_top1_spectral_share": spec["lora"]["top1_energy_share"],
            "raw_top4_spectral_share": spec["lora"]["top4_energy_share"],
            "functional_top1_share": func[0],
            "functional_top4_share": func[1],
            "raw_effective_rank": spec["lora"]["effective_rank"],
            "functional_effective_rank": func[2],
            "lora_modification_p99": mod[0],
            "hns_to_lora_p99_ratio": mod[1],
            "p99_suppression": mod[2],
            **gains(scores),
            "functional_available": True,
            "notes": "GSM8K" if task == "MetaMath" else "IFEval four-metric mean",
        })

    common = spectrum_cases["llama31_commonsense_allmods_hns8p2_failure"]
    rows.append({
        "task": "Commonsense",
        "checkpoint_family": "Llama-3.1-8B-Instruct",
        "raw_top1_spectral_share": common["lora_top1_energy_share"],
        "raw_top4_spectral_share": common["lora_top4_energy_share"],
        "functional_top1_share": None,
        "functional_top4_share": None,
        "raw_effective_rank": common["lora_effective_rank"],
        "functional_effective_rank": None,
        "lora_modification_p99": None,
        "hns_to_lora_p99_ratio": None,
        "p99_suppression": None,
        "hns_gain": common["performance_gain"],
        "head_only_gain": None,
        "tail_only_gain": None,
        "shape_only_gain": None,
        "scalar_shrink_gain": None,
        "functional_available": False,
        "notes": "negative control reused from prior study; compatible pretrained base unavailable locally",
    })
    write_tsv(root / "cross_task_summary.tsv", rows)

    correlation_specs = (
        ("raw_top1_spectral_share", 4),
        ("functional_top1_share", 3),
        ("functional_effective_rank", 3),
        ("p99_suppression", 3),
    )
    correlations = []
    for feature, _ in correlation_specs:
        included = [row for row in rows if row[feature] is not None and row["hns_gain"] is not None]
        correlations.append({
            "feature": feature,
            "target": "hns_gain",
            "spearman_rho": spearman(
                [float(row[feature]) for row in included],
                [float(row["hns_gain"]) for row in included],
            ),
            "n": len(included),
            "tasks": ",".join(row["task"] for row in included),
        })
    write_tsv(root / "cross_task_correlations.tsv", correlations)

    combined = []
    for task, task_root in task_roots.items():
        ensure_response_schema(task_root / "activation/layer_response.csv")
        ensure_response_schema(task_root / "activation/module_response.csv")
        shutil.copy2(task_root / "activation/layer_response.csv", task_root / "layer_response.csv")
        shutil.copy2(task_root / "activation/module_response.csv", task_root / "module_response.csv")
        for row in read_json(task_root / "summary.json")["rows"]:
            combined.append({"task": task, **row})
    write_tsv(root / "summary.tsv", combined)

    by_task = {row["task"]: row for row in rows}
    corr = {row["feature"]: row for row in correlations}
    layer_controls = {}
    for task in ("MetaMath", "Tulu"):
        layer_rows = read_csv(task_roots[task] / "activation/layer_response.csv")
        layer_controls[task] = {
            adapter: max(float(row[f"{adapter}_p99"]) for row in layer_rows)
            for adapter in ("lora", "scalar_shrink", "shape_only", "head_only", "tail_only", "hns")
        }
    lines = [
        "# HNS 跨任务机制报告",
        "",
        "## 协议",
        "",
        "MetaMath 和 Tulu 各固定采样 256 条训练分布样本；functional spectrum 与 modification 全部使用同一个 frozen pretrained-base trajectory，避免前层轨迹变化。六组干预沿用 Magicoder 实现，保持 rank-16 LoRA 的 U/V 不变，只修改奇异值。",
        "",
        "## 六组下游结果（相对 LoRA）",
        "",
        "| Task | HNS gain | HeadOnly | TailOnly | ShapeOnly | ScalarShrink |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in ("Magicoder", "MetaMath", "Tulu"):
        row = by_task[task]
        lines.append("| " + task + " | " + " | ".join(
            f"{100 * float(row[key]):+.2f} pp" for key in (
                "hns_gain", "head_only_gain", "tail_only_gain", "shape_only_gain", "scalar_shrink_gain"
            )
        ) + " |")
    lines += [
        "",
        "## 谱集中与真实 hidden-state modification",
        "",
        "| Task | raw top-1 energy | functional top-1 | raw effective rank | functional effective rank | max layer LoRA p99 | mean HNS/LoRA layer-p99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in ("Magicoder", "MetaMath", "Tulu"):
        row = by_task[task]
        lines.append(
            f"| {task} | {100*row['raw_top1_spectral_share']:.2f}% | {100*row['functional_top1_share']:.2f}% | "
            f"{row['raw_effective_rank']:.2f} | {row['functional_effective_rank']:.2f} | "
            f"{row['lora_modification_p99']:.4f} | {row['hns_to_lora_p99_ratio']:.3f} |"
        )
    lines += [
        "",
        "MetaMath 与 Tulu 中，hidden-state alignment 都进一步放大了头部：raw→functional top-1 分别为 66.30%→79.54% 和 53.31%→62.36%；Magicoder 为 75.74%→93.37%。",
        "",
        "### 新任务各控制组的最大 layer p99",
        "",
        "| Task | LoRA | ScalarShrink | ShapeOnly | HeadOnly | TailOnly | HNS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in ("MetaMath", "Tulu"):
        values = layer_controls[task]
        lines.append("| " + task + " | " + " | ".join(
            f"{values[name]:.4f}" for name in ("lora", "scalar_shrink", "shape_only", "head_only", "tail_only", "hns")
        ) + " |")
    lines += [
        "",
        "ShapeOnly 在恢复原始 LoRA Frobenius norm 后仍明显降低 extreme modification（MetaMath 0.1115→0.0366；Tulu 0.0829→0.0402），所以该现象不是单纯由整体缩放造成。ScalarShrink 也降低 modification，但性能只在 MetaMath 小幅 +0.99 pp，在 Tulu 反而 −5.36 pp。",
        "",
        "## 跨 checkpoint 相关性",
        "",
        "| Predictor | Spearman rho with HNS gain | n |",
        "|---|---:|---:|",
    ]
    for feature in ("raw_top1_spectral_share", "functional_top1_share", "functional_effective_rank", "p99_suppression"):
        item = corr[feature]
        lines.append(f"| {feature} | {item['spearman_rho']:.3f} | {item['n']} |")
    lines += [
        "",
        "当前结果不支持‘functional concentration 比 raw spectrum 更能预测 HNS gain’这一强结论：raw top-1 的 rho=0.80（n=4），functional top-1 的 rho=0.50（n=3），functional effective rank 的 rho=−0.50（n=3）。样本点过少，而且 Commonsense 缺少 functional 统计，因此这些数值只能看方向，不能视为稳定相关性估计。",
        "",
        "## 假设判定",
        "",
        "1. **参数集中 × hidden-state alignment：支持。** 三个有 activation 数据的 Qwen3 checkpoint 中，functional top-1 均高于 raw top-1，且 functional effective rank 仅 1.37–3.36，说明真实数据进一步集中 LoRA 响应。",
        "2. **收益主要来自 HeadOnly：仅对 Magicoder/MetaMath 成立，不能跨任务泛化。** 两者 HeadOnly 分别 +3.13/+3.11 pp，TailOnly −0.11/+1.06 pp，ScalarShrink +0.42/+0.99 pp；但 Tulu 的 HeadOnly 为 −1.98 pp，而 Full HNS 为 +1.81 pp。",
        "3. **完全谱平坦不是充分条件：强支持。** Tulu ShapeOnly 已把 functional top-1 降到与 HNS 相同的 17.98%，却使 IFEval 下降 3.02 pp；必须同时匹配谱形状和正确尺度/质量分配。",
        "4. **正负例结构：部分支持。** Magicoder 与 MetaMath 是清晰的 dominant-suppression 正例；Commonsense 仍是 −0.19 pp 的负例；Tulu 是‘Full HNS 有效但任何单组件均失效’的交互型正例。",
        "",
        "## Tulu category 诊断与新 failure condition",
        "",
        "Full HNS 的严格 category 增益主要来自 keywords（+4.91 pp）、change_case（+4.49 pp）、punctuation（+3.03 pp）和 startend（+2.99 pp），但 length_constraints 为 −2.80 pp。更关键的是四个单组件在 language 类别均从 LoRA 的 100% 降到 0%，而 Full HNS 保持 100%。这表明 instruction-following 对谱干预存在明显的非加性阈值/兼容性：只压 head、只抬 tail、只改形状或只改尺度都可能跨过行为边界，精确的联合重分配才有效。",
        "",
        "## 限制",
        "",
        "Commonsense 负例复用先前 Llama-3.1-8B HNS8+2 结果，只参与 raw-spectrum correlation；本地没有其兼容 pretrained base，所以未补 functional spectrum、modification 和六组对照。Tulu 使用当前已有且已验证的 IFEval（541 prompts/834 instructions），未额外引入 IFBench。完整证据以 per-layer、per-module、per-direction 和 IFEval category 表为准。",
        "",
    ]
    (root / "mechanism_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": rows, "correlations": correlations}, indent=2))


if __name__ == "__main__":
    main()
