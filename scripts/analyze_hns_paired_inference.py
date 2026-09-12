#!/usr/bin/env python3
"""Paired uncertainty tests for the 2x4 all-module HNS mechanism matrix."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np


VARIANTS = ("scalar_shrink", "shape_only", "head_only", "tail_only", "hns")


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def add_multiple_testing_adjustments(rows: list[dict], pvalue_key: str, prefix: str) -> None:
    """Add global Holm FWER and Benjamini-Hochberg FDR adjustments in-place."""
    if not rows:
        return
    pvalues = np.asarray([float(row[pvalue_key]) for row in rows], dtype=float)
    order = np.argsort(pvalues, kind="mergesort")
    total = len(rows)

    holm_sorted = np.empty(total, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * pvalues[index])
        holm_sorted[rank] = min(1.0, running)

    bh_sorted = np.empty(total, dtype=float)
    running = 1.0
    for reverse_rank in range(total - 1, -1, -1):
        index = order[reverse_rank]
        running = min(running, pvalues[index] * total / (reverse_rank + 1))
        bh_sorted[reverse_rank] = min(1.0, running)

    for rank, index in enumerate(order):
        rows[index][f"{prefix}_holm"] = float(holm_sorted[rank])
        rows[index][f"{prefix}_bh_fdr"] = float(bh_sorted[rank])


def canonical(label: str) -> str:
    compact = "".join(char for char in label.lower() if char.isalnum())
    return {
        "lora": "lora", "scalarshrink": "scalar_shrink", "shapeonly": "shape_only",
        "headonly": "head_only", "tailonly": "tail_only", "hns": "hns",
        "headfrorestore": "head_fro_restore",
    }[compact]


def load_binary_categories(task: str, root: Path) -> dict[str, dict[str, bool]]:
    if task == "magicoder":
        he_path = next((root / "humaneval").glob("samples_*.jsonl_results.jsonl"))
        mbpp_path = next((root / "mbpp").glob("outputs_*.jsonl"))
        return {
            "humaneval": {str(row["task_id"]): bool(row["passed"]) for row in jsonl(he_path)},
            "mbpp": {str(row["task_id"]): bool(row["passed"]) for row in jsonl(mbpp_path)},
        }
    if task == "metamath":
        rows = jsonl(root / "gsm8k" / "predictions.jsonl")
        return {"gsm8k": {str(index): bool(row["correct"]) for index, row in enumerate(rows)}}
    if task == "commonsense":
        suite = root / "commonsense"
        return {
            task_dir.name: {
                str(row["id"]): bool(row["correct"])
                for row in jsonl(task_dir / "predictions.jsonl")
            }
            for task_dir in sorted(suite.iterdir())
            if (task_dir / "predictions.jsonl").is_file()
        }
    raise ValueError(task)


def load_ifeval(root: Path) -> list[dict]:
    rows = jsonl(root / "ifeval" / "outputs.jsonl")
    return sorted(rows, key=lambda row: str(row["key"]))


def exact_mcnemar(left: np.ndarray, right: np.ndarray) -> tuple[int, int, float]:
    left, right = left.astype(bool), right.astype(bool)
    lora_only = int(np.sum(left & ~right))
    variant_only = int(np.sum(~left & right))
    discordant = lora_only + variant_only
    if discordant:
        smaller = min(lora_only, variant_only)
        lower_tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2 ** discordant)
        pvalue = min(1.0, 2.0 * lower_tail)
    else:
        pvalue = 1.0
    return lora_only, variant_only, pvalue


def average_ranks(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values_array[order[stop]] == values_array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def spearman_rho(left: list[float], right: list[float]) -> float:
    left_ranks, right_ranks = average_ranks(left), average_ranks(right)
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def exact_spearman_permutation_p(left: list[float], right: list[float], observed: float) -> float:
    if not math.isfinite(observed):
        return float("nan")
    exceed, total = 0, 0
    for permuted in itertools.permutations(right):
        total += 1
        exceed += abs(spearman_rho(left, list(permuted))) >= abs(observed) - 1e-15
    return exceed / total


def bootstrap_binary_macro(
    pairs: list[tuple[np.ndarray, np.ndarray]], samples: int, rng: np.random.Generator
) -> np.ndarray:
    distributions = []
    for left, right in pairs:
        delta = right.astype(np.int8) - left.astype(np.int8)
        minus, zero, plus = (int(np.sum(delta == value)) for value in (-1, 0, 1))
        counts = rng.multinomial(len(delta), [minus / len(delta), zero / len(delta), plus / len(delta)], size=samples)
        distributions.append((counts[:, 2] - counts[:, 0]) / len(delta))
    return np.mean(np.stack(distributions), axis=0)


def permutation_binary_macro(
    pairs: list[tuple[np.ndarray, np.ndarray]], samples: int, rng: np.random.Generator
) -> np.ndarray:
    distributions = []
    for left, right in pairs:
        discordant = int(np.sum(left != right))
        plus = rng.binomial(discordant, 0.5, size=samples)
        distributions.append((2 * plus - discordant) / len(left))
    return np.mean(np.stack(distributions), axis=0)


def ifeval_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "prompt_strict": np.asarray([row["prompt_strict_passed"] for row in rows], dtype=float),
        "prompt_loose": np.asarray([row["prompt_loose_passed"] for row in rows], dtype=float),
        "inst_strict_sum": np.asarray([sum(item["strict_passed"] for item in row["inst_results"]) for row in rows], dtype=float),
        "inst_loose_sum": np.asarray([sum(item["loose_passed"] for item in row["inst_results"]) for row in rows], dtype=float),
        "inst_count": np.asarray([len(row["inst_results"]) for row in rows], dtype=float),
    }


def ifeval_score(values: dict[str, np.ndarray]) -> float:
    return float(np.mean([
        values["prompt_strict"].mean(), values["prompt_loose"].mean(),
        values["inst_strict_sum"].sum() / values["inst_count"].sum(),
        values["inst_loose_sum"].sum() / values["inst_count"].sum(),
    ]))


def bootstrap_ifeval(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], samples: int, rng: np.random.Generator
) -> np.ndarray:
    n = len(left["prompt_strict"])
    result = np.empty(samples)
    for start in range(0, samples, 500):
        stop = min(samples, start + 500)
        indices = rng.integers(0, n, size=(stop - start, n))
        metrics = []
        for key in ("prompt_strict", "prompt_loose"):
            metrics.append(right[key][indices].mean(1) - left[key][indices].mean(1))
        denominator = left["inst_count"][indices].sum(1)
        for key in ("inst_strict_sum", "inst_loose_sum"):
            metrics.append((right[key][indices].sum(1) - left[key][indices].sum(1)) / denominator)
        result[start:stop] = np.mean(np.stack(metrics), axis=0)
    return result


def permutation_ifeval(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], samples: int, rng: np.random.Generator
) -> np.ndarray:
    n = len(left["prompt_strict"])
    denominator = left["inst_count"].sum()
    coefficients = (
        (right["prompt_strict"] - left["prompt_strict"]) / n,
        (right["prompt_loose"] - left["prompt_loose"]) / n,
        (right["inst_strict_sum"] - left["inst_strict_sum"]) / denominator,
        (right["inst_loose_sum"] - left["inst_loose_sum"]) / denominator,
    )
    result = np.empty(samples)
    for start in range(0, samples, 1000):
        stop = min(samples, start + 1000)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(stop - start, n))
        result[start:stop] = np.mean(np.stack([signs @ values for values in coefficients]), axis=0)
    return result


def bootstrap_ifeval_instruction_endpoint(
    left_rows: list[dict], right_rows: list[dict], endpoint: str,
    samples: int, rng: np.random.Generator, category: str | None = None,
) -> np.ndarray:
    """Prompt-cluster bootstrap for instruction-level endpoints/categories."""
    differences, counts = [], []
    for left_row, right_row in zip(left_rows, right_rows):
        delta_sum, count = 0.0, 0
        for left_item, right_item in zip(left_row["inst_results"], right_row["inst_results"]):
            item_category = left_item["instruction_id"].split(":", 1)[0]
            if category is not None and item_category != category:
                continue
            delta_sum += float(right_item[endpoint]) - float(left_item[endpoint])
            count += 1
        differences.append(delta_sum)
        counts.append(count)
    differences_array = np.asarray(differences)
    counts_array = np.asarray(counts)
    n = len(left_rows)
    result = np.empty(samples)
    for start in range(0, samples, 500):
        stop = min(samples, start + 500)
        indices = rng.integers(0, n, size=(stop - start, n))
        denominator = counts_array[indices].sum(1)
        result[start:stop] = differences_array[indices].sum(1) / np.maximum(denominator, 1)
    return result


def two_sided_permutation_p(null: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-15)) / (len(null) + 1))


def variant_roots(first: Path) -> dict[tuple[str, str], tuple[str, dict[str, Path]]]:
    qwen = Path("/dataset1/zailong/runs/peft-sft-lab/Qwen3-8B")
    llama = Path("/dataset1/zailong/runs/peft-sft-lab/Llama-3.1-8B-Instruct")
    allscope = Path("/dataset1/zailong/runs/peft-sft-lab/hns-allmodules-scope-20260909")
    definitions: dict[tuple[str, str], tuple[str, Path, dict[str, str]]] = {
        ("Qwen3-8B", "magicoder"): ("magicoder", qwen / "hns-mechanism-20260908/eval", {
            "lora": "LoRA", "scalar_shrink": "ScalarShrink", "shape_only": "ShapeOnly",
            "head_only": "HeadOnly", "tail_only": "TailOnly", "hns": "HNS"}),
        ("Qwen3-8B", "metamath"): ("metamath", qwen / "hns-cross-task-20260908/metamath/eval", {}),
        ("Qwen3-8B", "tulu"): ("tulu", qwen / "hns-cross-task-20260908/tulu/eval", {}),
        ("Qwen3-8B", "commonsense"): ("commonsense", allscope / "Qwen3-8B/commonsense/eval", {}),
        ("Llama-3.1-8B-Instruct", "magicoder"): ("magicoder", llama / "hns-cross-task-20260908/magicoder/eval", {}),
        ("Llama-3.1-8B-Instruct", "metamath"): ("metamath", llama / "hns-cross-task-20260908/metamath/eval", {}),
        ("Llama-3.1-8B-Instruct", "tulu"): ("tulu", allscope / "Llama-3.1-8B-Instruct/tulu/eval", {}),
        ("Llama-3.1-8B-Instruct", "commonsense"): ("commonsense", llama / "hns-cross-task-20260908/commonsense/eval", {}),
    }
    result = {}
    for key, (kind, root, aliases) in definitions.items():
        paths = {name: root / aliases.get(name, name) for name in ("lora", *VARIANTS)}
        if key == ("Llama-3.1-8B-Instruct", "commonsense"):
            rerun = first / "paired_assets/llama_commonsense"
            paths["lora"], paths["hns"] = rerun / "lora", rerun / "hns"
        head_fro_names = {
            ("Qwen3-8B", "magicoder"): "qwen_magicoder",
            ("Qwen3-8B", "metamath"): "qwen_metamath",
            ("Qwen3-8B", "tulu"): "qwen_tulu",
            ("Qwen3-8B", "commonsense"): "qwen_commonsense",
            ("Llama-3.1-8B-Instruct", "magicoder"): "llama_magicoder",
            ("Llama-3.1-8B-Instruct", "metamath"): "llama_metamath",
            ("Llama-3.1-8B-Instruct", "tulu"): "llama_tulu",
            ("Llama-3.1-8B-Instruct", "commonsense"): "llama_commonsense",
        }
        candidate = first / "headonly_fro_restore" / head_fro_names[key] / "eval"
        if candidate.is_dir():
            paths["head_fro_restore"] = candidate
        result[key] = (kind, paths)
    return result


def binary_inference(
    model: str, task: str, data: dict[str, dict[str, dict[str, bool]]],
    bootstrap_samples: int, permutation_samples: int, rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    primary, categories = [], []
    baseline = data["lora"]
    for variant, current in data.items():
        if variant == "lora":
            continue
        pairs = []
        for category in sorted(baseline):
            keys = sorted(baseline[category])
            if set(keys) != set(current[category]):
                raise RuntimeError(f"{model}/{task}/{variant}/{category}: paired keys differ")
            left = np.asarray([baseline[category][key] for key in keys], dtype=bool)
            right = np.asarray([current[category][key] for key in keys], dtype=bool)
            pairs.append((left, right))
            boot = bootstrap_binary_macro([(left, right)], bootstrap_samples, rng)
            lora_only, variant_only, pvalue = exact_mcnemar(left, right)
            categories.append({
                "base_model": model, "task": task, "variant": variant, "category": category,
                "endpoint": "accuracy", "n": len(left), "lora_score": left.mean(),
                "variant_score": right.mean(), "delta": right.mean() - left.mean(),
                "bootstrap_ci_low": np.quantile(boot, 0.025), "bootstrap_ci_high": np.quantile(boot, 0.975),
                "lora_only_correct": lora_only, "variant_only_correct": variant_only,
                "mcnemar_exact_p": pvalue,
            })
        boot = bootstrap_binary_macro(pairs, bootstrap_samples, rng)
        null = permutation_binary_macro(pairs, permutation_samples, rng)
        left_score = float(np.mean([left.mean() for left, _ in pairs]))
        right_score = float(np.mean([right.mean() for _, right in pairs]))
        delta = right_score - left_score
        primary.append({
            "base_model": model, "task": task, "variant": variant,
            "primary_metric": "macro_accuracy", "categories": len(pairs),
            "paired_units": sum(len(left) for left, _ in pairs), "lora_score": left_score,
            "variant_score": right_score, "delta": delta,
            "bootstrap_ci_low": np.quantile(boot, 0.025), "bootstrap_ci_high": np.quantile(boot, 0.975),
            "paired_permutation_p": two_sided_permutation_p(null, delta),
            "ci_excludes_zero": bool(np.quantile(boot, 0.025) > 0 or np.quantile(boot, 0.975) < 0),
        })
    return primary, categories


def ifeval_inference(
    model: str, task: str, data: dict[str, list[dict]], bootstrap_samples: int,
    permutation_samples: int, rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    primary, categories = [], []
    base_rows = data["lora"]
    base_keys = [str(row["key"]) for row in base_rows]
    left_values = ifeval_arrays(base_rows)
    for variant, rows in data.items():
        if variant == "lora":
            continue
        if [str(row["key"]) for row in rows] != base_keys:
            raise RuntimeError(f"{model}/{task}/{variant}: IFEval prompt keys differ")
        right_values = ifeval_arrays(rows)
        boot = bootstrap_ifeval(left_values, right_values, bootstrap_samples, rng)
        null = permutation_ifeval(left_values, right_values, permutation_samples, rng)
        left_score, right_score = ifeval_score(left_values), ifeval_score(right_values)
        delta = right_score - left_score
        primary.append({
            "base_model": model, "task": task, "variant": variant,
            "primary_metric": "mean_ifeval_four_metrics", "categories": 4,
            "paired_units": len(rows), "lora_score": left_score, "variant_score": right_score,
            "delta": delta, "bootstrap_ci_low": np.quantile(boot, 0.025),
            "bootstrap_ci_high": np.quantile(boot, 0.975),
            "paired_permutation_p": two_sided_permutation_p(null, delta),
            "ci_excludes_zero": bool(np.quantile(boot, 0.025) > 0 or np.quantile(boot, 0.975) < 0),
        })
        endpoint_pairs = {
            "prompt_strict": (left_values["prompt_strict"], right_values["prompt_strict"]),
            "prompt_loose": (left_values["prompt_loose"], right_values["prompt_loose"]),
        }
        left_inst, right_inst = [], []
        for left_row, right_row in zip(base_rows, rows):
            if len(left_row["inst_results"]) != len(right_row["inst_results"]):
                raise RuntimeError("IFEval instruction counts differ")
            for left_item, right_item in zip(left_row["inst_results"], right_row["inst_results"]):
                if left_item["instruction_id"] != right_item["instruction_id"]:
                    raise RuntimeError("IFEval instruction IDs differ")
                left_inst.append(left_item)
                right_inst.append(right_item)
        endpoint_pairs.update({
            "instruction_strict": (
                np.asarray([item["strict_passed"] for item in left_inst]),
                np.asarray([item["strict_passed"] for item in right_inst]),
            ),
            "instruction_loose": (
                np.asarray([item["loose_passed"] for item in left_inst]),
                np.asarray([item["loose_passed"] for item in right_inst]),
            ),
        })
        for endpoint, (left, right) in endpoint_pairs.items():
            if endpoint.startswith("instruction_"):
                item_endpoint = "strict_passed" if endpoint.endswith("strict") else "loose_passed"
                boot_endpoint = bootstrap_ifeval_instruction_endpoint(
                    base_rows, rows, item_endpoint, bootstrap_samples, rng
                )
            else:
                boot_endpoint = bootstrap_binary_macro([(left, right)], bootstrap_samples, rng)
            b, c, pvalue = exact_mcnemar(left, right)
            categories.append({
                "base_model": model, "task": task, "variant": variant,
                "category": "all", "endpoint": endpoint, "n": len(left),
                "lora_score": left.mean(), "variant_score": right.mean(), "delta": right.mean() - left.mean(),
                "bootstrap_ci_low": np.quantile(boot_endpoint, 0.025),
                "bootstrap_ci_high": np.quantile(boot_endpoint, 0.975),
                "lora_only_correct": b, "variant_only_correct": c, "mcnemar_exact_p": pvalue,
            })
        for category in sorted({item["instruction_id"].split(":", 1)[0] for item in left_inst}):
            selected = [index for index, item in enumerate(left_inst) if item["instruction_id"].split(":", 1)[0] == category]
            for endpoint in ("strict_passed", "loose_passed"):
                left = np.asarray([left_inst[index][endpoint] for index in selected])
                right = np.asarray([right_inst[index][endpoint] for index in selected])
                boot_category = bootstrap_ifeval_instruction_endpoint(
                    base_rows, rows, endpoint, bootstrap_samples, rng, category=category
                )
                b, c, pvalue = exact_mcnemar(left, right)
                categories.append({
                    "base_model": model, "task": task, "variant": variant,
                    "category": category, "endpoint": endpoint.replace("_passed", ""), "n": len(left),
                    "lora_score": left.mean(), "variant_score": right.mean(), "delta": right.mean() - left.mean(),
                    "bootstrap_ci_low": np.quantile(boot_category, 0.025),
                    "bootstrap_ci_high": np.quantile(boot_category, 0.975),
                    "lora_only_correct": b, "variant_only_correct": c, "mcnemar_exact_p": pvalue,
                })
    return primary, categories


def correlation_robustness(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    features = (
        "raw_top1_spectral_share", "functional_top1_share", "functional_effective_rank",
        "hidden_alignment_top1_delta", "hidden_alignment_top1_ratio", "p99_suppression",
    )
    exclusions = [("none", "none", rows)]
    for task in sorted({row["task"] for row in rows}):
        exclusions.append(("task", task, [row for row in rows if row["task"] != task]))
    for model in sorted({row["base_model"] for row in rows}):
        exclusions.append(("base", model, [row for row in rows if row["base_model"] != model]))
    output = []
    for kind, excluded, included in exclusions:
        for feature in features:
            left = [float(row[feature]) for row in included]
            right = [float(row["hns_gain"]) for row in included]
            rho = spearman_rho(left, right)
            output.append({
                "exclusion_kind": kind, "excluded": excluded, "feature": feature,
                "spearman_rho": rho,
                "exact_permutation_p": exact_spearman_permutation_p(left, right, rho),
                "n": len(included),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate_root", required=True)
    parser.add_argument("--first_batch_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--permutation_samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    first, output = Path(args.first_batch_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    primary_rows, category_rows = [], []
    for (model, task), (kind, paths) in variant_roots(first).items():
        if kind == "tulu":
            data = {label: load_ifeval(path) for label, path in paths.items()}
            primary, categories = ifeval_inference(
                model, task, data, args.bootstrap_samples, args.permutation_samples, rng
            )
        else:
            data = {label: load_binary_categories(kind, path) for label, path in paths.items()}
            primary, categories = binary_inference(
                model, task, data, args.bootstrap_samples, args.permutation_samples, rng
            )
        primary_rows.extend(primary)
        category_rows.extend(categories)
    correlations = correlation_robustness(Path(args.aggregate_root) / "cross_task_summary.tsv")
    add_multiple_testing_adjustments(primary_rows, "paired_permutation_p", "paired_permutation_p")
    add_multiple_testing_adjustments(category_rows, "mcnemar_exact_p", "mcnemar_exact_p")
    write_tsv(output / "paired_inference.tsv", primary_rows)
    write_tsv(output / "category_inference.tsv", category_rows)
    write_tsv(output / "correlation_robustness.tsv", correlations)
    metadata = {
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "seed": args.seed,
        "paired_comparisons": len(primary_rows),
        "category_comparisons": len(category_rows),
        "ci_excludes_zero": sum(bool(row["ci_excludes_zero"]) for row in primary_rows),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
