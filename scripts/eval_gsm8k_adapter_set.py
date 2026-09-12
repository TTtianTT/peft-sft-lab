#!/usr/bin/env python3
"""Evaluate a small adapter set on one fixed GSM8K JSONL split with one vLLM engine."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from finetune.eval.eval_gsm8k import _build_gsm8k_user_instruction, _extract_answer, _norm  # noqa: E402
from finetune.eval.generation import load_eval_tokenizer, render_chat_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--variant", action="append", required=True, help="label=adapter_path")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_variants(values: list[str]) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"variant must be label=path: {value}")
        label, path = value.split("=", 1)
        if not label or not Path(path).is_dir():
            raise ValueError(f"invalid variant: {value}")
        variants.append((label, str(Path(path).resolve())))
    labels = [label for label, _ in variants]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate variant labels: {labels}")
    return variants


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def numerically_equal(left: str, right: str) -> bool:
    left_norm, right_norm = _norm(left), _norm(right)
    if not (_DECIMAL_RE.fullmatch(left_norm) and _DECIMAL_RE.fullmatch(right_norm)):
        return left_norm == right_norm
    try:
        return Decimal(left_norm) == Decimal(right_norm)
    except InvalidOperation:
        return left_norm == right_norm


def main() -> None:
    args = parse_args()
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    variants = parse_variants(args.variant)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.dataset_path))
    if not rows:
        raise RuntimeError("empty evaluation split")
    ids = [str(row.get("id", index)) for index, row in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate evaluation IDs")

    tokenizer = load_eval_tokenizer(base_model=args.base_model, adapter_dir=None)
    prompts = [
        render_chat_prompt(
            tokenizer=tokenizer,
            base_model=args.base_model,
            user_content=_build_gsm8k_user_instruction(str(row["question"])),
            chat_template_mode="non_thinking",
        )
        for row in rows
    ]

    missing = [
        (label, path)
        for label, path in variants
        if not (output / label / "metrics.json").is_file()
    ]
    if not missing:
        print(f"[Skip] all {len(variants)} variants already evaluated under {output}")
        return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.base_model,
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=256,
        max_loras=1,
        max_cpu_loras=max(16, len(variants)),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend="FLASH_ATTN",
    )
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    manifest_rows: list[dict] = []
    for adapter_id, (label, path) in enumerate(variants, start=1):
        variant_output = output / label
        metrics_path = variant_output / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            manifest_rows.append({"label": label, "adapter_path": path, **metrics})
            print(f"[Skip] {label}: {metrics_path}", flush=True)
            continue
        variant_output.mkdir(parents=True, exist_ok=True)
        generations = llm.generate(
            prompts,
            params,
            lora_request=LoRARequest(label, adapter_id, path),
        )
        if len(generations) != len(rows):
            raise RuntimeError(f"{label}: {len(generations)} outputs for {len(rows)} prompts")
        prediction_rows: list[dict] = []
        for identity, source, request in zip(ids, rows, generations):
            text = request.outputs[0].text.strip() if request.outputs else ""
            prediction = _norm(_extract_answer(text))
            gold = _norm(_extract_answer(str(source["answer"])))
            strict = prediction == gold
            numeric = numerically_equal(prediction, gold)
            prediction_rows.append(
                {
                    "id": identity,
                    "source_index": source.get("source_index"),
                    "question": str(source["question"]),
                    "gold": gold,
                    "prediction_text": text,
                    "prediction_extracted": prediction,
                    "correct": strict,
                    "correct_strict": strict,
                    "correct_numeric": numeric,
                }
            )
        with (variant_output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = {
            "samples": len(prediction_rows),
            "strict_correct": sum(row["correct_strict"] for row in prediction_rows),
            "strict_accuracy": sum(row["correct_strict"] for row in prediction_rows) / len(prediction_rows),
            "numeric_correct": sum(row["correct_numeric"] for row in prediction_rows),
            "numeric_accuracy": sum(row["correct_numeric"] for row in prediction_rows) / len(prediction_rows),
            "numeric_only_corrections": sum(
                row["correct_numeric"] and not row["correct_strict"] for row in prediction_rows
            ),
            "invalid_extractions": sum(not row["prediction_extracted"] for row in prediction_rows),
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "chat_template_mode": "non_thinking",
            "dataset_path": str(Path(args.dataset_path).resolve()),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        manifest_rows.append({"label": label, "adapter_path": path, **metrics})
        print(f"[Eval] {label}: strict={metrics['strict_correct']}/{metrics['samples']} "
              f"numeric={metrics['numeric_correct']}/{metrics['samples']}", flush=True)
    (output / "evaluation_manifest.json").write_text(
        json.dumps({"status": "complete", "variants": manifest_rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
