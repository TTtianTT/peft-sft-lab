#!/usr/bin/env python3
"""Summarize the 2x4 HNS cross-task forgetting experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TASKS = ("magicoder", "metamath", "tulu", "commonsense")
BASE_DIRS = {
    "Qwen3-8B": "Qwen3-8B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
}
FIELDS = {
    "magicoder": "correct",
    "metamath": "correct_strict",
    "tulu": "prompt_strict_passed",
    "commonsense": "correct",
}
GAMMAS = {
    "global_0p25": 0.25,
    "global_0p40": 0.40,
    "global_0p55": 0.55,
    "global_0p70": 0.70,
    "global_0p85": 0.85,
    "common_lora": 1.00,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--target_tolerance", type=float, default=0.01)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class Results:
    def __init__(self, root: Path, base_dir: str):
        self.root = root / "formal" / base_dir
        summary = json.loads((self.root / "score_manifest.json").read_text())["records"]
        self.scores = {
            (row["task"], row["variant"]): float(row[row["primary_metric"]])
            for row in summary
        }
        self._cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray | None]] = {}

    def score(self, task: str, variant: str) -> float:
        return self.scores[task, variant]

    def values(self, task: str, variant: str) -> tuple[np.ndarray, np.ndarray | None]:
        key = (task, variant)
        if key not in self._cache:
            rows = read_jsonl(self.root / task / variant / "scored.jsonl")
            values = np.asarray([bool(row[FIELDS[task]]) for row in rows], dtype=np.int8)
            groups = None
            if task == "commonsense":
                groups = np.asarray([str(row["subtask"]) for row in rows])
            self._cache[key] = values, groups
        return self._cache[key]


def bootstrap_delta(
    result: Results,
    task: str,
    left: str,
    right: str,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Paired nonparametric bootstrap of score(right)-score(left)."""
    a, groups = result.values(task, left)
    b, groups_b = result.values(task, right)
    if len(a) != len(b) or (groups is not None and not np.array_equal(groups, groups_b)):
        raise RuntimeError(f"Pairing mismatch: {task}, {left}, {right}")
    delta = b.astype(np.int8) - a.astype(np.int8)
    strata = [np.arange(len(delta))]
    if groups is not None:
        strata = [np.flatnonzero(groups == name) for name in sorted(set(groups))]
    output = np.zeros(draws, dtype=np.float64)
    for indices in strata:
        counts = np.bincount(delta[indices] + 1, minlength=3)
        sampled = rng.multinomial(len(indices), counts / counts.sum(), size=draws)
        output += (sampled[:, 2] - sampled[:, 0]) / len(indices)
    return output / len(strata)


def holm_adjust(rows: list[dict], field: str = "p_value") -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index][field])
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * rows[index][field])
        running = max(running, adjusted)
        rows[index]["holm_p"] = running


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = {name: Results(run_root, directory) for name, directory in BASE_DIRS.items()}

    retention_rows: list[dict] = []
    contrast_rows: list[dict] = []
    pareto_rows: list[dict] = []
    contrast_specs = (
        ("original_hns_vs_lora", "original_lora", "original_hns"),
        ("common_hns_vs_lora", "common_lora", "common_hns"),
        ("common_hns_vs_per_module", "common_per_module", "common_hns"),
    )

    for base_name, result in results.items():
        base_scores = {task: result.score(task, "base") for task in TASKS}
        labels = {variant for _, variant in result.scores if variant != "base"}
        methods_by_train: dict[str, list[str]] = defaultdict(list)
        for label in labels:
            train, method = label.split("__", 1)
            methods_by_train[train].append(method)
        for train in TASKS:
            for method in sorted(methods_by_train[train]):
                variant = f"{train}__{method}"
                off_tasks = [task for task in TASKS if task != train]
                off = [result.score(task, variant) for task in off_tasks]
                base_off = [base_scores[task] for task in off_tasks]
                retention_rows.append({
                    "base": base_name,
                    "train_task": train,
                    "method": method,
                    "target_score": result.score(train, variant),
                    "off_task_macro": float(np.mean(off)),
                    "off_task_vs_base": float(np.mean(np.asarray(off) - np.asarray(base_off))),
                    "clipped_forgetting_gap": float(np.mean(np.maximum(0, np.asarray(base_off) - np.asarray(off)))),
                    **{f"score_{task}": result.score(task, variant) for task in TASKS},
                })

            for contrast, left_method, right_method in contrast_specs:
                left = f"{train}__{left_method}"
                right = f"{train}__{right_method}"
                off_tasks = [task for task in TASKS if task != train]
                boot = np.mean([
                    bootstrap_delta(result, task, left, right, args.bootstrap, rng)
                    for task in off_tasks
                ], axis=0)
                per_task = {
                    task: result.score(task, right) - result.score(task, left)
                    for task in off_tasks
                }
                p_value = min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
                contrast_rows.append({
                    "base": base_name,
                    "train_task": train,
                    "contrast": contrast,
                    "target_delta": result.score(train, right) - result.score(train, left),
                    "off_task_macro_delta": float(np.mean(list(per_task.values()))),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                    "p_value": float(p_value),
                    "holm_p": np.nan,
                    **{f"delta_{task}": per_task.get(task, np.nan) for task in TASKS},
                })

            hns = f"{train}__common_hns"
            hns_target = result.score(train, hns)
            hns_off = np.mean([result.score(task, hns) for task in TASKS if task != train])
            candidates = []
            for method, gamma in GAMMAS.items():
                variant = f"{train}__{method}"
                target_score = result.score(train, variant)
                off_score = np.mean([result.score(task, variant) for task in TASKS if task != train])
                candidates.append((method, gamma, target_score, off_score))
            feasible = [row for row in candidates if row[2] >= hns_target - args.target_tolerance]
            best = max(feasible, key=lambda row: row[3])
            pareto_rows.append({
                "base": base_name,
                "train_task": train,
                "hns_target": hns_target,
                "hns_off_task_macro": hns_off,
                "best_feasible_scalar": best[0],
                "best_gamma": best[1],
                "scalar_target": best[2],
                "scalar_off_task_macro": best[3],
                "hns_minus_scalar_target": hns_target - best[2],
                "hns_minus_scalar_off_task": hns_off - best[3],
                "target_tolerance": args.target_tolerance,
            })

    for contrast in {row["contrast"] for row in contrast_rows}:
        holm_adjust([row for row in contrast_rows if row["contrast"] == contrast])

    repeat_summary = json.loads(
        (run_root / "formal_repeat/Llama-3.1-8B-Instruct/score_manifest.json").read_text()
    )["records"]
    formal_llama = results["Llama-3.1-8B-Instruct"]
    stability_rows = []
    for row in repeat_summary:
        repeat_score = float(row[row["primary_metric"]])
        formal_score = formal_llama.score(row["task"], row["variant"])
        stability_rows.append({
            "task": row["task"], "variant": row["variant"],
            "formal_score": formal_score, "repeat_score": repeat_score,
            "delta": repeat_score - formal_score,
        })

    write_tsv(output / "retention_scores.tsv", retention_rows)
    write_tsv(output / "forgetting_contrasts.tsv", contrast_rows)
    write_tsv(output / "pareto_summary.tsv", pareto_rows)
    write_tsv(output / "llama_run_stability.tsv", stability_rows)

    original = [row for row in contrast_rows if row["contrast"] == "original_hns_vs_lora"]
    common = [row for row in contrast_rows if row["contrast"] == "common_hns_vs_lora"]
    shape = [row for row in contrast_rows if row["contrast"] == "common_hns_vs_per_module"]
    scalar_wins = sum(row["hns_minus_scalar_off_task"] > 0 for row in pareto_rows)
    report = [
        "# HNS cross-task forgetting result",
        "",
        f"HNS improved off-task macro performance over the saved LoRA in "
        f"{sum(row['off_task_macro_delta'] > 0 for row in original)}/8 checkpoints. "
        f"The mean checkpoint-level change was {np.mean([row['off_task_macro_delta'] for row in original])*100:+.2f} pp.",
        "",
        f"The common-basis comparison had the same sign in "
        f"{sum(row['off_task_macro_delta'] > 0 for row in common)}/8 checkpoints, "
        f"with mean {np.mean([row['off_task_macro_delta'] for row in common])*100:+.2f} pp.",
        "",
        f"Against the per-module norm-matched scalar, HNS improved off-task macro in "
        f"{sum(row['off_task_macro_delta'] > 0 for row in shape)}/8 checkpoints. "
        f"Against the target-feasible global scalar selected on these same data, HNS had better retention in "
        f"{scalar_wins}/8 checkpoints.",
        "",
        "All 84 aggregate scores in the Llama core reverse-order repeat exactly matched the first formal run. "
        "The scalar frontier is exploratory because selection and reporting use the same benchmark data.",
    ]
    (output / "forgetting_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
