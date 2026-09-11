#!/usr/bin/env python3
"""Score generated forgetting-matrix outputs with the existing benchmark logic."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from finetune.eval.eval_gsm8k import _extract_answer, _norm  # noqa: E402
from finetune.eval.eval_humaneval import (  # noqa: E402
    jsonl_read,
    jsonl_write,
    normalize_humaneval_completion,
    parse_results_jsonl,
    run_humaneval_evaluate_functional_correctness,
)
from finetune.eval.eval_ifeval import (  # noqa: E402
    _norm_inst_id,
    check_instruction_loose,
    check_instruction_strict,
)


def load_commonsense_module():
    path = REPO_ROOT / "scripts" / "eval_commonsense_8tasks.py"
    spec = importlib.util.spec_from_file_location("score_forgetting_commonsense", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix_dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=3.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def numeric_equal(left: str, right: str) -> bool:
    left_norm, right_norm = _norm(left), _norm(right)
    try:
        return Decimal(left_norm) == Decimal(right_norm)
    except InvalidOperation:
        return left_norm == right_norm


def score_metamath(rows: list[dict]) -> tuple[list[dict], dict]:
    scored = []
    for row in rows:
        prediction = _norm(_extract_answer(row["prediction_text"]))
        gold = _norm(str(row["gold"]))
        item = dict(row)
        item.update({
            "prediction_extracted": prediction,
            "correct_strict": prediction == gold,
            "correct_numeric": numeric_equal(prediction, gold),
        })
        scored.append(item)
    n = len(scored)
    return scored, {
        "samples": n,
        "primary_metric": "strict_accuracy",
        "strict_accuracy": sum(row["correct_strict"] for row in scored) / n,
        "numeric_accuracy": sum(row["correct_numeric"] for row in scored) / n,
    }


def score_commonsense(rows: list[dict]) -> tuple[list[dict], dict]:
    cs = load_commonsense_module()
    scored = []
    per_task: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        choices = tuple(str(value) for value in row["choices"])
        prediction = cs._extract_prediction(row["prediction_text"], choices)
        gold = cs.LETTERS[int(row["gold_index"])]
        correct = prediction == gold
        item = dict(row)
        item.update({"prediction_letter": prediction, "gold": gold, "correct": correct})
        scored.append(item)
        per_task[row["subtask"]].update(total=1, correct=int(correct), invalid=int(not prediction))
    task_metrics = {}
    for task, counts in sorted(per_task.items()):
        task_metrics[task] = {
            **dict(counts),
            "accuracy": counts["correct"] / counts["total"],
        }
    macro = sum(row["accuracy"] for row in task_metrics.values()) / len(task_metrics)
    return scored, {
        "samples": len(scored),
        "primary_metric": "macro_accuracy",
        "macro_accuracy": macro,
        "per_task": task_metrics,
    }


def score_ifeval(rows: list[dict]) -> tuple[list[dict], dict]:
    scored = []
    prompt_strict = prompt_loose = inst_total = inst_strict = inst_loose = 0
    categories: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        strict_ok = True
        loose_ok = True
        inst_results = []
        for inst_id, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
            kwargs = dict(kwargs or {})
            strict = check_instruction_strict(
                inst_id=inst_id,
                resp=row["prediction_text"],
                prompt=row["prompt"],
                kwargs=kwargs,
            )
            loose, loose_detail = check_instruction_loose(
                inst_id=inst_id,
                resp=row["prediction_text"],
                prompt=row["prompt"],
                kwargs=kwargs,
            )
            strict_ok = strict_ok and bool(strict.passed)
            loose_ok = loose_ok and bool(loose)
            inst_total += 1
            inst_strict += int(strict.passed)
            inst_loose += int(loose)
            category, _ = _norm_inst_id(inst_id)
            categories[category].update(total=1, strict=int(strict.passed), loose=int(loose))
            inst_results.append({
                "instruction_id": inst_id,
                "strict_passed": bool(strict.passed),
                "strict_detail": strict.detail,
                "loose_passed": bool(loose),
                "loose_detail": loose_detail,
            })
        prompt_strict += int(strict_ok)
        prompt_loose += int(loose_ok)
        item = dict(row)
        item.update({
            "prompt_strict_passed": strict_ok,
            "prompt_loose_passed": loose_ok,
            "inst_results": inst_results,
        })
        scored.append(item)
    n = len(scored)
    category_metrics = {
        key: {
            "total": value["total"],
            "strict_accuracy": value["strict"] / value["total"],
            "loose_accuracy": value["loose"] / value["total"],
        }
        for key, value in sorted(categories.items())
    }
    return scored, {
        "samples": n,
        "instructions": inst_total,
        "primary_metric": "prompt_level_strict_accuracy",
        "prompt_level_strict_accuracy": prompt_strict / n,
        "prompt_level_loose_accuracy": prompt_loose / n,
        "instruction_level_strict_accuracy": inst_strict / inst_total,
        "instruction_level_loose_accuracy": inst_loose / inst_total,
        "per_category": category_metrics,
    }


def score_humaneval(
    rows: list[dict],
    variant_dir: Path,
    problems_path: Path,
    workers: int,
    timeout: float,
) -> tuple[list[dict], dict]:
    samples = []
    normalized: dict[str, str] = {}
    for row in rows:
        completion = normalize_humaneval_completion(
            row["prediction_text"], row["problem_prompt"], row["entry_point"]
        )
        normalized[str(row["id"])] = completion
        samples.append({"task_id": row["id"], "completion": completion})
    samples_path = variant_dir / "samples.jsonl"
    jsonl_write(str(samples_path), samples)
    _, results_path = run_humaneval_evaluate_functional_correctness(
        samples_path=str(samples_path),
        problem_file=str(problems_path),
        k=[1],
        n_workers=workers,
        timeout=timeout,
        ignore_incomplete=False,
    )
    if not results_path or not Path(results_path).is_file():
        raise RuntimeError(f"HumanEval did not produce results for {variant_dir}")
    by_id = {}
    for result in jsonl_read(results_path):
        by_id.setdefault(str(result["task_id"]), result)
    scored = []
    for row in rows:
        identity = str(row["id"])
        result = by_id[identity]
        item = dict(row)
        item.update({
            "completion": normalized[identity],
            "correct": bool(result.get("passed", False)),
            "result": result.get("result"),
        })
        scored.append(item)
    correct = sum(row["correct"] for row in scored)
    return scored, {
        "samples": len(scored),
        "primary_metric": "pass_at_1",
        "pass_at_1": correct / len(scored),
        "correct": correct,
        "results_path": str(results_path),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.matrix_dir).resolve()
    summary = []
    for task in ("magicoder", "metamath", "tulu", "commonsense"):
        task_root = root / task
        if not task_root.is_dir():
            continue
        for variant_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
            predictions = variant_dir / "predictions.jsonl"
            if not predictions.is_file():
                continue
            metrics_path = variant_dir / "metrics.json"
            if metrics_path.is_file() and (variant_dir / "scored.jsonl").is_file():
                metrics = json.loads(metrics_path.read_text())
                summary.append({"task": task, "variant": variant_dir.name, **metrics})
                print(f"[Skip score] {task}/{variant_dir.name}", flush=True)
                continue
            rows = read_rows(predictions)
            if task == "metamath":
                scored, metrics = score_metamath(rows)
            elif task == "commonsense":
                scored, metrics = score_commonsense(rows)
            elif task == "tulu":
                scored, metrics = score_ifeval(rows)
            else:
                scored, metrics = score_humaneval(
                    rows, variant_dir, task_root / "problems.jsonl", args.workers, args.timeout
                )
            write_rows(variant_dir / "scored.jsonl", scored)
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
            summary.append({"task": task, "variant": variant_dir.name, **metrics})
            print(f"[Score] {task}/{variant_dir.name}: {metrics[metrics['primary_metric']]:.6f}", flush=True)
    (root / "score_manifest.json").write_text(json.dumps({
        "status": "complete", "records": summary
    }, indent=2) + "\n")
    print(f"[Done] scored {len(summary)} task/variant cells", flush=True)


if __name__ == "__main__":
    main()
