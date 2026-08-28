#!/usr/bin/env python3
"""Evaluate one base/PEFT model on the Commonsense170K eight-task suite.

All prompts are rendered with the model tokenizer's chat template.  The default
splits follow the common LLM-Adapters/DoRA protocol: validation for BoolQ, PIQA,
SocialIQA, HellaSwag and WinoGrande; test for ARC-Easy, ARC-Challenge and
OpenBookQA.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.eval.generation import (  # noqa: E402
    generate_greedy,
    generate_greedy_vllm_batch,
    load_eval_tokenizer,
    load_transformers_model,
    render_chat_prompt,
    save_json,
)
from finetune.utils import seed_everything  # noqa: E402


LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    question: str
    choices: tuple[str, ...]
    gold_index: int


@dataclass(frozen=True)
class TaskSpec:
    dataset_id: str
    config: str | None
    split: str
    formatter: Callable[[dict[str, Any], int], EvalItem]


def _required_text(example: dict[str, Any], key: str) -> str:
    value = str(example.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing non-empty field {key!r}; keys={sorted(example)}")
    return value


def _item_id(example: dict[str, Any], index: int) -> str:
    for key in ("id", "idx", "question_id"):
        value = example.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(index)


def _zero_based_label(value: Any, num_choices: int, *, task: str) -> int:
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{task}: invalid numeric label {value!r}") from exc
    if not 0 <= label < num_choices:
        raise ValueError(f"{task}: label {label} outside [0, {num_choices})")
    return label


def _format_boolq(example: dict[str, Any], index: int) -> EvalItem:
    passage = _required_text(example, "passage")
    question = _required_text(example, "question")
    answer = example.get("answer")
    if not isinstance(answer, bool):
        raise ValueError(f"BoolQ: expected boolean answer, got {answer!r}")
    return EvalItem(
        _item_id(example, index),
        f"Passage:\n{passage}\n\nQuestion:\n{question}",
        ("Yes", "No"),
        0 if answer else 1,
    )


def _format_piqa(example: dict[str, Any], index: int) -> EvalItem:
    choices = (_required_text(example, "sol1"), _required_text(example, "sol2"))
    return EvalItem(
        _item_id(example, index),
        _required_text(example, "goal"),
        choices,
        _zero_based_label(example.get("label"), len(choices), task="PIQA"),
    )


def _format_siqa(example: dict[str, Any], index: int) -> EvalItem:
    context = _required_text(example, "context")
    question = _required_text(example, "question")
    choices = tuple(_required_text(example, key) for key in ("answerA", "answerB", "answerC"))
    # The datasets package exposes SocialIQA's ClassLabel as zero-based 0/1/2.
    gold = _zero_based_label(example.get("label"), len(choices), task="SocialIQA")
    return EvalItem(
        _item_id(example, index),
        f"Context:\n{context}\n\nQuestion:\n{question}",
        choices,
        gold,
    )


def _format_hellaswag(example: dict[str, Any], index: int) -> EvalItem:
    context = str(example.get("ctx", "")).strip()
    if not context:
        context = f"{example.get('ctx_a', '')} {example.get('ctx_b', '')}".strip()
    if not context:
        raise ValueError("HellaSwag: missing context")
    endings = example.get("endings")
    if not isinstance(endings, list) or len(endings) < 2:
        raise ValueError(f"HellaSwag: malformed endings {endings!r}")
    choices = tuple(str(choice).strip() for choice in endings)
    return EvalItem(
        _item_id(example, index),
        f"Choose the most plausible continuation.\n\nContext:\n{context}",
        choices,
        _zero_based_label(example.get("label"), len(choices), task="HellaSwag"),
    )


def _format_winogrande(example: dict[str, Any], index: int) -> EvalItem:
    choices = (_required_text(example, "option1"), _required_text(example, "option2"))
    try:
        gold = int(example.get("answer")) - 1
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WinoGrande: invalid answer {example.get('answer')!r}") from exc
    if not 0 <= gold < len(choices):
        raise ValueError(f"WinoGrande: answer outside 1..{len(choices)}")
    return EvalItem(
        _item_id(example, index),
        f"Fill in the blank in the sentence.\n\n{_required_text(example, 'sentence')}",
        choices,
        gold,
    )


def _choice_fields(example: dict[str, Any]) -> tuple[list[str], list[str]]:
    raw = example.get("choices")
    if not isinstance(raw, dict):
        raise ValueError(f"Malformed choices: {raw!r}")
    labels = raw.get("label")
    texts = raw.get("text")
    if not isinstance(labels, list) or not isinstance(texts, list) or len(labels) != len(texts):
        raise ValueError(f"Malformed choice labels/text: {raw!r}")
    return [str(x).strip() for x in labels], [str(x).strip() for x in texts]


def _format_arc(example: dict[str, Any], index: int) -> EvalItem:
    labels, texts = _choice_fields(example)
    answer = _required_text(example, "answerKey")
    try:
        gold = labels.index(answer)
    except ValueError as exc:
        raise ValueError(f"ARC: answerKey={answer!r} not in labels={labels!r}") from exc
    return EvalItem(_item_id(example, index), _required_text(example, "question"), tuple(texts), gold)


def _format_obqa(example: dict[str, Any], index: int) -> EvalItem:
    labels, texts = _choice_fields(example)
    answer = _required_text(example, "answerKey")
    try:
        gold = labels.index(answer)
    except ValueError as exc:
        raise ValueError(f"OpenBookQA: answerKey={answer!r} not in labels={labels!r}") from exc
    return EvalItem(
        _item_id(example, index),
        _required_text(example, "question_stem"),
        tuple(texts),
        gold,
    )


TASKS: dict[str, TaskSpec] = {
    "boolq": TaskSpec("google/boolq", None, "validation", _format_boolq),
    "piqa": TaskSpec("ybisk/piqa", None, "validation", _format_piqa),
    "siqa": TaskSpec("allenai/social_i_qa", None, "validation", _format_siqa),
    "hellaswag": TaskSpec("Rowan/hellaswag", None, "validation", _format_hellaswag),
    "winogrande": TaskSpec("allenai/winogrande", "winogrande_xl", "validation", _format_winogrande),
    "arc_easy": TaskSpec("allenai/ai2_arc", "ARC-Easy", "test", _format_arc),
    "arc_challenge": TaskSpec("allenai/ai2_arc", "ARC-Challenge", "test", _format_arc),
    "openbookqa": TaskSpec("allenai/openbookqa", "main", "test", _format_obqa),
}


def _instruction(item: EvalItem) -> str:
    if not 2 <= len(item.choices) <= len(LETTERS):
        raise ValueError(f"Unsupported number of choices: {len(item.choices)}")
    choice_lines = [f"{LETTERS[i]}. {choice}" for i, choice in enumerate(item.choices)]
    valid = ", ".join(LETTERS[: len(item.choices)])
    return (
        f"{item.question}\n\nChoices:\n"
        + "\n".join(choice_lines)
        + f"\n\nAnswer with only one letter: {valid}."
    )


def _extract_prediction(text: str, choices: tuple[str, ...]) -> str:
    valid = "".join(LETTERS[: len(choices)])
    upper = (text or "").strip().upper()
    patterns = (
        rf"(?:FINAL\s+ANSWER|ANSWER)\s*[:\-]?\s*\(?([{valid}])\)?\b",
        rf"^\s*\(?([{valid}])\)?(?:\s|[.:'\-]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    hits = re.findall(rf"\b([{valid}])\b", upper)
    if hits:
        return hits[-1]

    normalized = re.sub(r"\s+", " ", (text or "").strip()).casefold().rstrip(".!")
    for index, choice in enumerate(choices):
        normalized_choice = re.sub(r"\s+", " ", choice.strip()).casefold().rstrip(".!")
        if normalized == normalized_choice:
            return LETTERS[index]
    return ""


def _parse_tasks(value: str) -> list[str]:
    aliases = {
        "all": list(TASKS),
        "arc-e": ["arc_easy"],
        "arce": ["arc_easy"],
        "arc-c": ["arc_challenge"],
        "arcc": ["arc_challenge"],
        "obqa": ["openbookqa"],
    }
    selected: list[str] = []
    for token in (part.strip().lower() for part in value.split(",")):
        names = aliases.get(token, [token])
        for name in names:
            if name not in TASKS:
                raise ValueError(f"Unknown task {token!r}; choices={','.join(TASKS)}")
            if name not in selected:
                selected.append(name)
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat-template evaluation on eight commonsense tasks.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", default="all", help="Comma-separated task names or 'all'.")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--chat_template_mode", choices=("auto", "thinking", "non_thinking"), default="auto")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional per-task smoke-test limit.")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--request_batch_size", type=int, default=256)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_model_len", type=int, default=2048)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--vllm_attention_backend", default="FLASH_ATTN")
    parser.add_argument(
        "--enable_flashinfer_sampler",
        action="store_true",
        help="FlashInfer sampling is disabled by default for wider GPU compatibility.",
    )
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=500)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    selected_tasks = _parse_tasks(args.tasks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(f"datasets is required: {exc}") from exc

    tokenizer = load_eval_tokenizer(base_model=args.base_model, adapter_dir=args.adapter_dir)
    all_prompts: list[str] = []
    all_records: list[tuple[str, EvalItem, str]] = []

    for task_name in selected_tasks:
        spec = TASKS[task_name]
        print(f"[{task_name}] Loading {spec.dataset_id} {spec.config or ''} split={spec.split}")
        load_kwargs: dict[str, Any] = {"split": spec.split}
        if spec.config is not None:
            load_kwargs["name"] = spec.config
        dataset = load_dataset(spec.dataset_id, **load_kwargs)
        if args.max_samples is not None:
            if args.max_samples <= 0:
                raise ValueError("--max_samples must be > 0")
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        for index, example in enumerate(dataset):
            item = spec.formatter(example, index)
            instruction = _instruction(item)
            prompt = render_chat_prompt(
                tokenizer=tokenizer,
                base_model=args.base_model,
                user_content=instruction,
                chat_template_mode=args.chat_template_mode,
            )
            all_prompts.append(prompt)
            all_records.append((task_name, item, instruction))
        print(f"[{task_name}] Prepared {len(dataset)} prompts")

    if args.backend == "vllm":
        generations = generate_greedy_vllm_batch(
            base_model=args.base_model,
            prompts=all_prompts,
            max_new_tokens=args.max_new_tokens,
            adapter_dir=args.adapter_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            attention_backend=args.vllm_attention_backend or None,
            disable_flashinfer_sampler=not args.enable_flashinfer_sampler,
            request_batch_size=args.request_batch_size,
        )
    else:
        loaded = load_transformers_model(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            dtype=args.dtype,
            device_map="auto",
        )
        generations = []
        for index, prompt in enumerate(all_prompts, start=1):
            generations.append(
                generate_greedy(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            )
            if args.log_every > 0 and index % args.log_every == 0:
                print(f"[transformers] Generated {index}/{len(all_prompts)}")

    if len(generations) != len(all_records):
        raise RuntimeError(f"Generated {len(generations)} outputs for {len(all_records)} examples")

    counters = {name: {"correct": 0, "total": 0, "invalid": 0} for name in selected_tasks}
    handles: dict[str, Any] = {}
    try:
        for name in selected_tasks:
            task_dir = output_dir / name
            task_dir.mkdir(parents=True, exist_ok=True)
            handles[name] = (task_dir / "predictions.jsonl").open("w", encoding="utf-8")

        for index, ((task_name, item, instruction), generation) in enumerate(
            zip(all_records, generations), start=1
        ):
            prediction = _extract_prediction(generation, item.choices)
            gold = LETTERS[item.gold_index]
            correct = prediction == gold
            counters[task_name]["total"] += 1
            counters[task_name]["correct"] += int(correct)
            counters[task_name]["invalid"] += int(not prediction)
            handles[task_name].write(
                json.dumps(
                    {
                        "id": item.item_id,
                        "question": item.question,
                        "choices": list(item.choices),
                        "gold": gold,
                        "instruction": instruction,
                        "prompt_style": "tokenizer_chat_template",
                        "prediction_text": generation,
                        "prediction_letter": prediction,
                        "correct": correct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if args.log_every > 0 and index % args.log_every == 0:
                print(f"[score] {index}/{len(all_records)}")
    finally:
        for handle in handles.values():
            handle.close()

    task_results: dict[str, dict[str, Any]] = {}
    for name in selected_tasks:
        spec = TASKS[name]
        count = counters[name]
        accuracy = count["correct"] / count["total"] if count["total"] else 0.0
        metrics = {
            "task": name,
            "dataset": spec.dataset_id,
            "dataset_config": spec.config,
            "split": spec.split,
            "accuracy": accuracy,
            **count,
        }
        task_results[name] = metrics
        save_json(output_dir / name / "metrics.json", metrics)

    macro_accuracy = sum(result["accuracy"] for result in task_results.values()) / len(task_results)
    total_correct = sum(result["correct"] for result in task_results.values())
    total_examples = sum(result["total"] for result in task_results.values())
    summary = {
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "backend": args.backend,
        "prompt_style": "tokenizer_chat_template",
        "chat_template_mode": args.chat_template_mode,
        "macro_accuracy": macro_accuracy,
        "micro_accuracy": total_correct / total_examples if total_examples else 0.0,
        "correct": total_correct,
        "total": total_examples,
        "tasks": task_results,
        "seed": args.seed,
    }
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("task", "accuracy", "correct", "total", "invalid", "split"))
        for name, result in task_results.items():
            writer.writerow(
                (name, result["accuracy"], result["correct"], result["total"], result["invalid"], result["split"])
            )
        writer.writerow(("macro_average", macro_accuracy, "", "", "", ""))

    print("\nTask                 Accuracy     Correct/Total  Invalid")
    print("-" * 62)
    for name, result in task_results.items():
        print(
            f"{name:<20} {result['accuracy']:>9.6f}  "
            f"{result['correct']:>7}/{result['total']:<7}  {result['invalid']:>7}"
        )
    print("-" * 62)
    print(f"{'macro_average':<20} {macro_accuracy:>9.6f}")
    print(f"Wrote: {output_dir / 'summary.json'}")
    print(f"Wrote: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
