#!/usr/bin/env python3
"""Generate fixed Qwen MetaMath/GSM8K rollouts for spectral-path reward gradients."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from finetune.eval.eval_gsm8k import _build_gsm8k_user_instruction, _extract_answer, _norm  # noqa: E402
from finetune.eval.generation import load_eval_tokenizer, render_chat_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", required=True)
    p.add_argument("--lora_path", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--metamath_train", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--questions", type=int, default=128)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return p.parse_args()


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise RuntimeError(f"empty dataset: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    # B300 nodes in this cluster do not expose nvcc under /usr/local/cuda.
    # Native vLLM sampling avoids FlashInfer's runtime JIT dependency and is
    # also the established setting used by the existing evaluation scripts.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rollout_path = output / "rollouts.jsonl"
    manifest_path = output / "manifest.json"
    if rollout_path.is_file() and manifest_path.is_file():
        print(f"[Skip] existing rollouts: {rollout_path}")
        return

    source = load_jsonl(Path(args.dataset_path))
    train = pq.read_table(args.metamath_train, columns=["query", "original_question"]).to_pydict()
    train_questions = {
        normalize(value)
        for column in ("query", "original_question")
        for value in train[column]
        if value is not None and str(value).strip()
    }
    eligible = [
        index for index, row in enumerate(source)
        if normalize(row["question"]) not in train_questions
    ]
    if len(eligible) < args.questions:
        raise RuntimeError(f"only {len(eligible)} non-overlap questions for requested {args.questions}")
    chosen = np.random.default_rng(args.seed).permutation(eligible)[: args.questions].tolist()

    tokenizer = load_eval_tokenizer(base_model=args.base_model, adapter_dir=args.lora_path)
    prompts: list[str] = []
    question_rows: list[dict] = []
    for order, source_index in enumerate(chosen):
        row = source[source_index]
        prompt = render_chat_prompt(
            tokenizer=tokenizer,
            base_model=args.base_model,
            user_content=_build_gsm8k_user_instruction(str(row["question"])),
            chat_template_mode="non_thinking",
        )
        prompts.append(prompt)
        question_rows.append(
            {
                "question_id": f"gsm8k-test-{source_index:04d}",
                "source_index": source_index,
                "order": order,
                "half": "A" if order < args.questions // 2 else "B",
                "question": str(row["question"]),
                "gold_raw": str(row["answer"]),
                "gold": _norm(_extract_answer(str(row["answer"]))),
                "prompt": prompt,
            }
        )

    # Keep the import local so manifest/data checks remain usable without vLLM.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.base_model,
        # V1 defaults to raw_logprobs (before temperature). Record the
        # distribution actually used to sample so replay comparisons are valid.
        logprobs_mode="processed_logprobs",
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=256,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend="FLASH_ATTN",
    )
    sampling_distribution = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    params = SamplingParams(
        n=args.rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        logprobs=0,
    )
    outputs = llm.generate(prompts, params, lora_request=LoRARequest("metamath-lora", 1, args.lora_path))
    if len(outputs) != len(question_rows):
        raise RuntimeError(f"vLLM returned {len(outputs)} requests for {len(question_rows)} prompts")

    records: list[dict] = []
    for qrow, request in zip(question_rows, outputs):
        samples = list(request.outputs)
        if len(samples) != args.rollouts:
            raise RuntimeError(
                f"{qrow['question_id']} returned {len(samples)} rollouts, expected {args.rollouts}"
            )
        prompt_ids = list(request.prompt_token_ids)
        for rollout_index, sample in enumerate(samples):
            text = sample.text.strip()
            prediction = _norm(_extract_answer(text))
            records.append(
                {
                    **qrow,
                    "rollout_index": rollout_index,
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": list(sample.token_ids),
                    "response_text": text,
                    "prediction_extracted": prediction,
                    "reward": int(prediction == qrow["gold"]),
                    "finish_reason": sample.finish_reason,
                    "stop_reason": sample.stop_reason,
                    "response_tokens": len(sample.token_ids),
                    "cumulative_logprob_vllm": sample.cumulative_logprob,
                    "vllm_logprobs_mode": "processed_logprobs",
                    "sampling_distribution": sampling_distribution,
                }
            )
    write_jsonl(rollout_path, records)

    by_question: dict[str, list[int]] = {}
    for record in records:
        by_question.setdefault(record["question_id"], []).append(record["reward"])
    mixed = sum(0 < sum(values) < len(values) for values in by_question.values())
    truncated = sum(record["finish_reason"] == "length" for record in records)
    empty = sum(not record["response_token_ids"] for record in records)
    manifest = {
        "status": "complete",
        "base_model": str(Path(args.base_model).resolve()),
        "lora_path": str(Path(args.lora_path).resolve()),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "metamath_train": str(Path(args.metamath_train).resolve()),
        "seed": args.seed,
        "vllm_logprobs_mode": "processed_logprobs",
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "rollouts_per_question": args.rollouts,
        },
        "source_questions": len(source),
        "exact_normalized_train_overlap": len(source) - len(eligible),
        "selected_questions": len(question_rows),
        "selected_source_indices": chosen,
        "halves": {
            "A": [row["question_id"] for row in question_rows if row["half"] == "A"],
            "B": [row["question_id"] for row in question_rows if row["half"] == "B"],
        },
        "rollout_count": len(records),
        "reward_mean": float(np.mean([record["reward"] for record in records])),
        "mixed_outcome_questions": mixed,
        "all_correct_questions": sum(sum(v) == len(v) for v in by_question.values()),
        "all_wrong_questions": sum(sum(v) == 0 for v in by_question.values()),
        "truncated_rollouts": truncated,
        "empty_rollouts": empty,
        "response_tokens": {
            "median": float(np.median([record["response_tokens"] for record in records])),
            "p95": float(np.quantile([record["response_tokens"] for record in records], 0.95)),
            "max": max(record["response_tokens"] for record in records),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
