#!/usr/bin/env python3
"""Score stochastic GSM8K/HumanEval rollouts from the HNS-above-Base audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from finetune.eval.eval_gsm8k import _extract_answer, _norm  # noqa: E402
from finetune.eval.eval_humaneval import (  # noqa: E402
    jsonl_read,
    jsonl_write,
    normalize_humaneval_completion,
    run_humaneval_evaluate_functional_correctness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--workers", type=int, default=32)
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
    try:
        return Decimal(_norm(left)) == Decimal(_norm(right))
    except InvalidOperation:
        return _norm(left) == _norm(right)


def score_gsm8k(rows: list[dict]) -> tuple[list[dict], dict]:
    scored: list[dict] = []
    strict_total = numeric_total = any_strict = any_numeric = majority_strict = 0
    invalid = tokens = rollouts_total = 0
    for row in rows:
        gold = _norm(str(row["gold"]))
        rollout_scores = []
        extracted_values = []
        for rollout in row["rollouts"]:
            extracted = _norm(_extract_answer(rollout["text"]))
            strict = extracted == gold
            numeric = numeric_equal(extracted, gold)
            rollout_scores.append({
                **rollout,
                "extracted": extracted,
                "correct_strict": strict,
                "correct_numeric": numeric,
            })
            extracted_values.append(extracted)
            strict_total += int(strict)
            numeric_total += int(numeric)
            invalid += int(not extracted)
            tokens += len(rollout.get("token_ids", []))
            rollouts_total += 1
        counts = Counter(value for value in extracted_values if value)
        majority = counts.most_common(1)[0][0] if counts else ""
        strict_count = sum(int(item["correct_strict"]) for item in rollout_scores)
        numeric_count = sum(int(item["correct_numeric"]) for item in rollout_scores)
        any_strict += int(strict_count > 0)
        any_numeric += int(numeric_count > 0)
        majority_strict += int(majority == gold)
        scored.append({
            **{key: value for key, value in row.items() if key != "rollouts"},
            "rollouts": rollout_scores,
            "strict_count": strict_count,
            "numeric_count": numeric_count,
            "majority_extracted": majority,
            "majority_correct_strict": majority == gold,
        })
    n = len(rows)
    return scored, {
        "samples": n,
        "rollouts_per_sample": rollouts_total // n,
        "expected_strict_accuracy": strict_total / rollouts_total,
        "expected_numeric_accuracy": numeric_total / rollouts_total,
        "any_at_n_strict": any_strict / n,
        "any_at_n_numeric": any_numeric / n,
        "majority_strict_accuracy": majority_strict / n,
        "invalid_extraction_rate": invalid / rollouts_total,
        "mean_output_tokens": tokens / rollouts_total,
    }


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def score_humaneval(rows: list[dict], variant_dir: Path, problems: Path, workers: int, timeout: float):
    samples = []
    raw_meta: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        identity = str(row["id"])
        for rollout_id, rollout in enumerate(row["rollouts"]):
            completion = normalize_humaneval_completion(
                rollout["text"], row["problem_prompt"], row["entry_point"]
            )
            samples.append({"task_id": identity, "completion": completion})
            raw_meta[identity].append({
                **rollout,
                "rollout_id": rollout_id,
                "completion": completion,
                "had_code_fence": "```" in rollout["text"],
            })
    samples_path = variant_dir / "samples.jsonl"
    jsonl_write(str(samples_path), samples)
    n_rollouts = len(rows[0]["rollouts"])
    ks = sorted(set([1, min(4, n_rollouts), n_rollouts]))
    _, results_path = run_humaneval_evaluate_functional_correctness(
        samples_path=str(samples_path),
        problem_file=str(problems),
        k=ks,
        n_workers=workers,
        timeout=timeout,
        ignore_incomplete=False,
    )
    results_by_id: dict[str, list[dict]] = defaultdict(list)
    for result in jsonl_read(str(results_path)):
        results_by_id[str(result["task_id"])].append(result)
    scored = []
    total_passed = total_rollouts = total_tokens = fences = 0
    pass_sums = {k: 0.0 for k in ks}
    for row in rows:
        identity = str(row["id"])
        results = results_by_id[identity]
        if len(results) != n_rollouts:
            raise RuntimeError(f"Expected {n_rollouts} HumanEval results for {identity}, got {len(results)}")
        # HumanEval emits completion_id; sorting makes result-to-rollout alignment explicit.
        results.sort(key=lambda item: int(item.get("completion_id", 0)))
        detailed = []
        for meta, result in zip(raw_meta[identity], results):
            passed = bool(result.get("passed", False))
            detailed.append({**meta, "passed": passed, "result": result.get("result")})
            total_passed += int(passed)
            total_rollouts += 1
            total_tokens += len(meta.get("token_ids", []))
            fences += int(meta["had_code_fence"])
        correct_count = sum(int(item["passed"]) for item in detailed)
        for k in ks:
            pass_sums[k] += pass_at_k(n_rollouts, correct_count, k)
        scored.append({
            **{key: value for key, value in row.items() if key != "rollouts"},
            "rollouts": detailed,
            "correct_count": correct_count,
        })
    task_count = len(rows)
    metrics = {
        "samples": task_count,
        "rollouts_per_sample": n_rollouts,
        "expected_pass_rate": total_passed / total_rollouts,
        "mean_output_tokens": total_tokens / total_rollouts,
        "code_fence_rate": fences / total_rollouts,
        "results_path": str(results_path),
    }
    metrics.update({f"pass_at_{k}": pass_sums[k] / task_count for k in ks})
    return scored, metrics


def main() -> None:
    args = parse_args()
    root = Path(args.run_dir).resolve()
    summary = []
    for base_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for task in ("magicoder", "metamath"):
            task_dir = base_dir / task
            if not task_dir.is_dir():
                continue
            for variant_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
                predictions = variant_dir / "predictions.jsonl"
                if not predictions.is_file():
                    continue
                metrics_path = variant_dir / "metrics.json"
                if metrics_path.is_file() and (variant_dir / "scored.jsonl").is_file():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    rows = read_rows(predictions)
                    if task == "metamath":
                        scored, metrics = score_gsm8k(rows)
                    else:
                        scored, metrics = score_humaneval(
                            rows, variant_dir, task_dir / "problems.jsonl", args.workers, args.timeout
                        )
                    write_rows(variant_dir / "scored.jsonl", scored)
                    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                summary.append({"base": base_dir.name, "task": task, "variant": variant_dir.name, **metrics})
                print(f"[Score] {base_dir.name}/{task}/{variant_dir.name}", flush=True)
    (root / "score_manifest.json").write_text(
        json.dumps({"status": "complete", "records": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[Done] {len(summary)} cells", flush=True)


if __name__ == "__main__":
    main()
