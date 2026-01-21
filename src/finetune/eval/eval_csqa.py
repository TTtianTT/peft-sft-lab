from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from finetune.data.base import format_instruction_response
from finetune.eval.generation import (
    generate_greedy,
    generate_greedy_vllm_batch,
    load_transformers_model,
    save_json,
)
from finetune.utils import seed_everything


def _choices_to_map(choices: Any) -> dict[str, str]:
    # Keep identical behavior to finetune task plugin (strict + no forced upper)
    if isinstance(choices, dict):
        labels = choices.get("label")
        texts = choices.get("text")
        if isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts):
            return {str(l): str(t) for l, t in zip(labels, texts)}
    if isinstance(choices, list):
        out: dict[str, str] = {}
        for item in choices:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            text = item.get("text")
            if label is None or text is None:
                continue
            out[str(label)] = str(text)
        if out:
            return out
    raise ValueError(f"Unrecognized choices format: {type(choices)}")


def _extract_letter(text: str) -> str:
    text = text.strip().upper()
    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1)
    for ch in text:
        if ch in {"A", "B", "C", "D", "E"}:
            return ch
    return ""


def _build_csqa_instruction(example: dict[str, Any]) -> tuple[str, str, str]:
    """
    Build (question, gold_letter, instruction) using the *same* formatting as finetune format_example().
    """
    question = str(example.get("question", "")).strip()
    gold = str(example.get("answerKey", "")).strip().upper()
    if not question or not gold:
        raise ValueError(
            f"CSQA example missing required fields. Keys: {sorted(example.keys())}. "
            "Expected (question, choices, answerKey)."
        )
    if gold not in {"A", "B", "C", "D", "E"}:
        raise ValueError(f"CSQA answerKey must be one of A/B/C/D/E, got {gold!r}.")

    choices_map = _choices_to_map(example.get("choices"))
    lines: list[str] = []
    for label in ["A", "B", "C", "D", "E"]:
        if label in choices_map:
            lines.append(f"{label}. {choices_map[label]}")
    if len(lines) < 2:
        raise ValueError(f"CSQA choices malformed. Keys: {sorted(example.keys())}.")

    instruction = (
        f"Question:\n{question}\n\nChoices:\n" + "\n".join(lines) + "\n\n"
        "Answer with a single letter: A, B, C, D, or E."
    )
    return question, gold, instruction


def _build_csqa_prompt(instruction: str) -> str:
    # Align evaluation prompt with training template: same wrapper, empty response as generation slot.
    return format_instruction_response(instruction=instruction, response="")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate CommonsenseQA accuracy (single-letter A/B/C/D/E).")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(f"datasets is required: {exc}") from exc

    try:
        ds = load_dataset("tau/commonsense_qa", split=args.split)
    except Exception as exc:
        raise RuntimeError(f"Failed to load commonsense_qa: {exc}") from exc

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    loaded = None
    if not args.use_vllm:
        loaded = load_transformers_model(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            dtype=args.dtype,
            device_map="auto",
        )

    correct = 0
    total = 0

    with preds_path.open("w", encoding="utf-8") as f:
        if args.use_vllm:
            examples = list(ds)

            prompts: list[str] = []
            records: list[dict[str, Any]] = []
            for ex in examples:
                q, gold, instruction = _build_csqa_instruction(ex)
                prompt = _build_csqa_prompt(instruction)

                prompts.append(prompt)
                records.append({"question": q, "gold": gold, "instruction": instruction})

            generations = generate_greedy_vllm_batch(
                base_model=args.base_model,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                adapter_dir=args.adapter_dir,
                tensor_parallel_size=args.tensor_parallel_size,
            )

            for rec, gen in zip(records, generations):
                pred_letter = _extract_letter(gen)
                is_correct = int(pred_letter == rec["gold"])
                correct += is_correct
                total += 1

                f.write(
                    json.dumps(
                        {
                            "question": rec["question"],
                            "gold": rec["gold"],
                            "instruction": rec["instruction"],
                            "prediction_text": gen,
                            "prediction_letter": pred_letter,
                            "correct": bool(is_correct),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        else:
            assert loaded is not None
            for ex in ds:
                q, gold, instruction = _build_csqa_instruction(ex)
                prompt = _build_csqa_prompt(instruction)

                gen = generate_greedy(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )

                pred_letter = _extract_letter(gen)
                is_correct = int(pred_letter == gold)
                correct += is_correct
                total += 1

                f.write(
                    json.dumps(
                        {
                            "question": q,
                            "gold": gold,
                            "instruction": instruction,
                            "prediction_text": gen,
                            "prediction_letter": pred_letter,
                            "correct": bool(is_correct),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    metrics = {"accuracy": (correct / total if total else 0.0), "correct": correct, "total": total}
    save_json(out_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
