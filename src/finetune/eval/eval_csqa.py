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
    t = (text or "").strip().upper()

    # 1) Prefer explicit "ANSWER: C" / "FINAL ANSWER: C"
    m = re.search(r"(?:^|\n)\s*(?:FINAL\s+ANSWER|ANSWER)\s*[:\-]?\s*([A-E])\b", t)
    if m:
        return m.group(1)

    # 2) Otherwise take the last standalone letter (avoid picking "A" from "A, B, C, D, E")
    hits = re.findall(r"\b([A-E])\b", t)
    return hits[-1] if hits else ""


def _build_csqa_instruction(example: dict[str, Any]) -> tuple[str, str, str]:
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
    return format_instruction_response(instruction=instruction, response="")


def _resolve_split(requested: str) -> tuple[str, str]:
    """
    Force using validation as the evaluation split (treat as 'test').
    Returns: (split_to_load, note)
    """
    req = (requested or "").strip().lower()
    # Anything that looks like test will be mapped to validation.
    if req in {"test", "testing", "test_rand_split", "test_rand_split_no_answers", "test_no_answers"}:
        return "validation", f"Requested split={requested!r} mapped to 'validation' (CSQA test split typically has no answerKey)."
    if req in {"validation", "val", "dev"}:
        return "validation", "Using 'validation' as evaluation split."
    if req in {"train"}:
        # Still allow train explicitly, but we keep the “validation as test” policy unless user insists.
        # Here: respect train if they really want it.
        return "train", "Using 'train' as evaluation split (you explicitly requested train)."
    # Default fallback
    return "validation", f"Unrecognized split={requested!r}; fallback to 'validation'."


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate CommonsenseQA accuracy (single-letter A/B/C/D/E).")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)

    # User can still pass --split, but we map test->validation and default to validation anyway
    p.add_argument("--split", type=str, default="validation", help="train/validation/test (note: test will be mapped to validation).")

    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--log_every", type=int, default=100)
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

    split_to_load, split_note = _resolve_split(args.split)
    print(f"[CSQA] {split_note}")

    try:
        ds = load_dataset("tau/commonsense_qa", split=split_to_load)
    except Exception as exc:
        raise RuntimeError(f"Failed to load tau/commonsense_qa split={split_to_load!r}: {exc}") from exc

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
    pred_hist = {k: 0 for k in ["A", "B", "C", "D", "E", ""]}

    with preds_path.open("w", encoding="utf-8") as f:
        if args.use_vllm:
            examples = list(ds)

            prompts: list[str] = []
            records: list[dict[str, Any]] = []
            for ex in examples:
                q, gold, instruction = _build_csqa_instruction(ex)
                prompt = _build_csqa_prompt(instruction)
                prompts.append(prompt)
                records.append(
                    {
                        "id": ex.get("id", None),
                        "question": q,
                        "gold": gold,
                        "instruction": instruction,
                    }
                )

            generations = generate_greedy_vllm_batch(
                base_model=args.base_model,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                adapter_dir=args.adapter_dir,
                tensor_parallel_size=args.tensor_parallel_size,
            )

            if len(generations) != len(records):
                raise RuntimeError(f"vLLM returned {len(generations)} generations for {len(records)} prompts.")

            for i, (rec, gen) in enumerate(zip(records, generations), start=1):
                pred_letter = _extract_letter(gen)
                pred_hist[pred_letter] = pred_hist.get(pred_letter, 0) + 1

                is_correct = int(pred_letter == rec["gold"])
                correct += is_correct
                total += 1

                f.write(
                    json.dumps(
                        {
                            "id": rec.get("id"),
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

                if args.log_every > 0 and (i % args.log_every == 0 or i == len(records)):
                    print(f"[CSQA] {i}/{len(records)} done | acc={correct/total:.4f}")

        else:
            assert loaded is not None
            for i, ex in enumerate(ds, start=1):
                q, gold, instruction = _build_csqa_instruction(ex)
                prompt = _build_csqa_prompt(instruction)

                gen = generate_greedy(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )

                pred_letter = _extract_letter(gen)
                pred_hist[pred_letter] = pred_hist.get(pred_letter, 0) + 1

                is_correct = int(pred_letter == gold)
                correct += is_correct
                total += 1

                f.write(
                    json.dumps(
                        {
                            "id": ex.get("id", None),
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

                if args.log_every > 0 and (i % args.log_every == 0 or i == len(ds)):
                    print(f"[CSQA] {i}/{len(ds)} done | acc={correct/total:.4f}")

    metrics = {
        "dataset": "tau/commonsense_qa",
        "requested_split": args.split,
        "loaded_split": split_to_load,
        "split_note": split_note,
        "accuracy": (correct / total if total else 0.0),
        "correct": correct,
        "total": total,
        "pred_histogram": pred_hist,
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "use_vllm": bool(args.use_vllm),
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dtype": args.dtype,
    }
    save_json(out_dir / "metrics.json", metrics)
    print(f"[CSQA] Done. accuracy={metrics['accuracy']:.4f} ({correct}/{total})")
    print(f"[CSQA] Wrote: {preds_path}")
    print(f"[CSQA] Wrote: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
