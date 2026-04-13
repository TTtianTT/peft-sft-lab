from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from finetune.eval.generation import (
    generate_greedy,
    generate_greedy_vllm_batch,
    load_transformers_model,
    save_json,
)
from finetune.eval.fewshot import (
    PromptStatsAccumulator,
    load_dataset_split,
    load_tokenizer_for_prompt_stats,
    save_exemplars_jsonl,
    select_fixed_exemplars,
)
from finetune.utils import seed_everything


# MetaMathQA "Model Usage" prompting template
_METAMATH_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response: Let's think step by step."
)

_NEXT_PROMPT_MARKERS = [
    "\n\nBelow is an instruction",
    "\n### Instruction:",
]


def _extract_answer(text: str) -> str:
    # Preferred GSM8K-style marker
    if "####" in text:
        tail = text.split("####")[-1].strip()
        if not tail:
            return ""
        lines = tail.splitlines()
        return lines[0].strip() if lines else ""

    # Common alternative markers
    m = re.search(r"(?:The answer is|Answer is|Final answer)\s*[:：]\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: last number-ish token
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if matches:
        return matches[-1].strip()

    # Final fallback: last non-empty line
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _norm(s: str) -> str:
    return s.strip().replace(",", "")


def _build_prompt_gsm8k_metamath_style(question: str) -> str:
    """
    Build a GSM8K prompt that *uses* the MetaMathQA Model Usage template, while still
    instructing the model to output '#### <answer>' for strict-match evaluation.
    """
    instruction = (
        "Solve the following math word problem.\n"
        "Put your final numeric answer on the last line exactly as:\n"
        "#### <answer>\n\n"
        f"{question.strip()}"
    )
    return _METAMATH_PROMPT_TEMPLATE.format(instruction=instruction) + "\n"


def _build_fewshot_example_text(ex: dict[str, Any]) -> dict[str, Any]:
    question = str(ex.get("question", "")).strip()
    answer = str(ex.get("answer", "")).strip()
    return {
        "orig_index": ex.get("__fewshot_orig_idx__"),
        "question": question,
        "answer": answer,
        "prompt": _build_prompt_gsm8k_metamath_style(question),
    }


def _build_fewshot_prefix(exemplars: list[dict[str, Any]]) -> str:
    if not exemplars:
        return ""
    blocks = []
    for ex in exemplars:
        blocks.append(ex["prompt"] + ex["answer"] + "\n")
    return "\n".join(blocks).strip() + "\n\n"


def _truncate_at_next_prompt(text: str) -> tuple[str, str | None]:
    cut_idx: int | None = None
    cut_marker: str | None = None
    for marker in _NEXT_PROMPT_MARKERS:
        idx = text.find(marker)
        if idx == -1:
            continue
        if cut_idx is None or idx < cut_idx:
            cut_idx = idx
            cut_marker = marker
    if cut_idx is None:
        return text, None
    return text[:cut_idx].rstrip(), cut_marker


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate GSM8K strict-match accuracy (MetaMath-style prompt).")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--fewshot_k", type=int, default=0)
    p.add_argument("--fewshot_seed", type=int, default=42)
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
        ds = load_dataset("gsm8k", "main", split=args.split)
    except Exception as exc:
        raise RuntimeError(f"Failed to load gsm8k: {exc}") from exc

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tokenizer_for_stats = load_tokenizer_for_prompt_stats(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
    )
    prompt_stats = PromptStatsAccumulator(tokenizer_for_stats)

    fewshot_exemplars: list[dict[str, Any]] = []
    fewshot_prefix = ""
    if args.fewshot_k > 0:
        source_ds = load_dataset_split("gsm8k", dataset_config="main", split="train")
        raw_exemplars = select_fixed_exemplars(source_ds, k=args.fewshot_k, seed=args.fewshot_seed)
        fewshot_exemplars = [_build_fewshot_example_text(ex) for ex in raw_exemplars]
        fewshot_prefix = _build_fewshot_prefix(fewshot_exemplars)
        save_exemplars_jsonl(out_dir / "fewshot_exemplars.jsonl", fewshot_exemplars)

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
    next_prompt_truncation_count = 0
    stop_strings = _NEXT_PROMPT_MARKERS if args.fewshot_k > 0 else None

    with preds_path.open("w", encoding="utf-8") as f:
        if args.use_vllm:
            examples = list(ds)
            prompts: list[str] = []
            records: list[tuple[str, str]] = []

            for ex in examples:
                q = str(ex.get("question", "")).strip()
                gold_raw = str(ex.get("answer", "")).strip()
                gold = _norm(_extract_answer(gold_raw))

                base_prompt = _build_prompt_gsm8k_metamath_style(q)
                prompt = fewshot_prefix + base_prompt
                base_prompt_tokens, prompt_tokens = prompt_stats.add(base_prompt=base_prompt, prompt=prompt)
                prompts.append(prompt)
                records.append((q, gold, base_prompt_tokens, prompt_tokens))

            generations = generate_greedy_vllm_batch(
                base_model=args.base_model,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                adapter_dir=args.adapter_dir,
                tensor_parallel_size=args.tensor_parallel_size,
                stop_strings=stop_strings,
            )

            for (q, gold, base_prompt_tokens, prompt_tokens), raw_gen in zip(records, generations):
                gen, truncation_marker = _truncate_at_next_prompt(raw_gen)
                next_prompt_truncation_count += int(truncation_marker is not None)
                pred = _norm(_extract_answer(gen))
                is_correct = int(pred == gold)
                correct += is_correct
                total += 1

                rec: dict[str, Any] = {
                    "question": q,
                    "gold": gold,
                    "prompt_style": "metamath_model_usage",
                    "fewshot_k": args.fewshot_k,
                    "base_prompt_tokens": base_prompt_tokens,
                    "prompt_tokens": prompt_tokens,
                    "extra_prompt_tokens": prompt_tokens - base_prompt_tokens,
                    "prediction_text_raw": raw_gen,
                    "prediction_text": gen,
                    "next_prompt_truncation_marker": truncation_marker,
                    "next_prompt_truncated": bool(truncation_marker is not None),
                    "prediction_extracted": pred,
                    "correct": bool(is_correct),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            for ex in ds:
                q = str(ex.get("question", "")).strip()
                gold_raw = str(ex.get("answer", "")).strip()
                gold = _norm(_extract_answer(gold_raw))

                base_prompt = _build_prompt_gsm8k_metamath_style(q)
                prompt = fewshot_prefix + base_prompt
                base_prompt_tokens, prompt_tokens = prompt_stats.add(base_prompt=base_prompt, prompt=prompt)

                gen = generate_greedy(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    stop_strings=stop_strings,
                )
                gen, truncation_marker = _truncate_at_next_prompt(gen)
                next_prompt_truncation_count += int(truncation_marker is not None)

                pred = _norm(_extract_answer(gen))
                is_correct = int(pred == gold)
                correct += is_correct
                total += 1

                rec: dict[str, Any] = {
                    "question": q,
                    "gold": gold,
                    "prompt_style": "metamath_model_usage",
                    "fewshot_k": args.fewshot_k,
                    "base_prompt_tokens": base_prompt_tokens,
                    "prompt_tokens": prompt_tokens,
                    "extra_prompt_tokens": prompt_tokens - base_prompt_tokens,
                    "prediction_text": gen,
                    "next_prompt_truncation_marker": truncation_marker,
                    "next_prompt_truncated": bool(truncation_marker is not None),
                    "prediction_extracted": pred,
                    "correct": bool(is_correct),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    prompt_summary = prompt_stats.summary()
    metrics = {
        "accuracy_strict": (correct / total if total else 0.0),
        "correct": correct,
        "total": total,
        "prompt_style": "metamath_model_usage",
        "fewshot_k": args.fewshot_k,
        "fewshot_seed": args.fewshot_seed,
        "fewshot_source_dataset": "gsm8k" if args.fewshot_k > 0 else None,
        "fewshot_source_config": "main" if args.fewshot_k > 0 else None,
        "fewshot_source_split": "train" if args.fewshot_k > 0 else None,
        "fewshot_exemplar_count": len(fewshot_exemplars),
        "fewshot_exemplar_orig_indices": [ex.get("orig_index") for ex in fewshot_exemplars],
        "fewshot_prefix_tokens": (
            len(tokenizer_for_stats(fewshot_prefix, add_special_tokens=False).input_ids)
            if fewshot_prefix
            else 0
        ),
        "next_prompt_stop_markers": stop_strings,
        "next_prompt_truncation_count": next_prompt_truncation_count,
        **prompt_summary,
    }
    save_json(out_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
