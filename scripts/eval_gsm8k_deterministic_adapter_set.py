#!/usr/bin/env python3
"""Evaluate LoRA adapters with explicit deterministic scheduling and token-ID logging."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant", action="append", required=True, help="label=adapter_path")
    parser.add_argument("--reverse_variants", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--max_samples", type=int, help="Evaluate only the first N rows (for smoke tests).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_variants(values: list[str], reverse: bool) -> list[tuple[str, str]]:
    variants = []
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not Path(path).is_dir():
            raise ValueError(f"invalid variant: {value}")
        variants.append((label, str(Path(path).resolve())))
    if len({label for label, _ in variants}) != len(variants):
        raise ValueError("duplicate variant labels")
    return list(reversed(variants)) if reverse else variants


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
    variants = parse_variants(args.variant, args.reverse_variants)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.dataset_path))
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive")
        rows = rows[: args.max_samples]
    ids = [str(row.get("id", index)) for index, row in enumerate(rows)]
    if not rows or len(set(ids)) != len(ids):
        raise RuntimeError("empty dataset or duplicate IDs")

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
        async_scheduling=False,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
    )
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, seed=args.seed)
    manifest_rows = []
    for adapter_id, (label, path) in enumerate(variants, start=1):
        destination = output / label
        destination.mkdir(parents=True, exist_ok=True)
        generations = llm.generate(prompts, params, lora_request=LoRARequest(label, adapter_id, path))
        prediction_rows = []
        for identity, source, request in zip(ids, rows, generations):
            sample = request.outputs[0]
            text = sample.text.strip()
            prediction = _norm(_extract_answer(text))
            gold = _norm(_extract_answer(str(source["answer"])))
            prediction_rows.append({
                "id": identity,
                "source_index": source.get("source_index"),
                "diagnostic_group": source.get("diagnostic_group"),
                "gold": gold,
                "prediction_text": text,
                "token_ids": list(sample.token_ids),
                "prediction_extracted": prediction,
                "correct_strict": prediction == gold,
                "correct_numeric": numerically_equal(prediction, gold),
            })
        with (destination / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = {
            "samples": len(prediction_rows),
            "strict_accuracy": sum(row["correct_strict"] for row in prediction_rows) / len(prediction_rows),
            "numeric_accuracy": sum(row["correct_numeric"] for row in prediction_rows) / len(prediction_rows),
            "adapter_id": adapter_id,
        }
        (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        manifest_rows.append({"label": label, "adapter_path": path, **metrics})
        print(f"[Eval] {label}: strict={metrics['strict_accuracy']:.6f}", flush=True)
    manifest = {
        "status": "complete",
        "dataset": str(Path(args.dataset_path).resolve()),
        "variant_order": [label for label, _ in variants],
        "configuration": {
            "vllm_batch_invariant": os.getenv("VLLM_BATCH_INVARIANT"),
            "async_scheduling": False,
            "enable_prefix_caching": False,
            "disable_custom_all_reduce": True,
            "max_num_seqs": args.max_num_seqs,
            "seed": args.seed,
            "temperature": 0.0,
        },
        "variants": manifest_rows,
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
