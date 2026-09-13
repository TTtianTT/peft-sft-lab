#!/usr/bin/env python3
"""Generate a full task-by-adapter forgetting matrix with one persistent vLLM engine."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from finetune.eval.eval_gsm8k import (  # noqa: E402
    _build_gsm8k_user_instruction,
    _extract_answer,
    _norm,
    load_gsm8k_split,
)
from finetune.eval.eval_humaneval import (  # noqa: E402
    build_humaneval_chat_user_prompt,
    jsonl_write,
    load_humaneval_problems,
)
from finetune.eval.eval_ifeval import _load_ifeval_dataset  # noqa: E402
from finetune.eval.generation import load_eval_tokenizer, render_chat_prompt  # noqa: E402


def load_commonsense_module():
    path = REPO_ROOT / "scripts" / "eval_commonsense_8tasks.py"
    spec = importlib.util.spec_from_file_location("forgetting_commonsense", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--variant_manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=("magicoder", "metamath", "tulu", "commonsense"))
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--max_num_seqs", type=int, default=128)
    parser.add_argument("--max_num_batched_tokens", type=int, default=65536)
    parser.add_argument("--adapter_block_size", type=int, default=8)
    parser.add_argument("--max_lora_rank", type=int, default=256)
    parser.add_argument("--prompt_chunk_short", type=int, default=1024)
    parser.add_argument("--prompt_chunk_long", type=int, default=64)
    parser.add_argument("--diagnostic_max_samples", type=int)
    parser.add_argument("--reverse_variants", action="store_true")
    parser.add_argument(
        "--diagonal_only",
        action="store_true",
        help="For each task, evaluate only variants whose train_task matches that task.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_task(task: str, cfg: dict, tokenizer, base_model: str, limit: int | None):
    spec = cfg["evaluation_tasks"][task]
    records: list[dict[str, Any]] = []
    prompts: list[str] = []
    if task == "metamath":
        ds = load_gsm8k_split(split="test", dataset_path=spec["dataset"], dataset_config="main")
        examples = list(ds)
        if limit is not None:
            examples = examples[:limit]
        for index, row in enumerate(examples):
            question = str(row.get("question", "")).strip()
            gold = _norm(_extract_answer(str(row.get("answer", ""))))
            prompts.append(render_chat_prompt(
                tokenizer=tokenizer,
                base_model=base_model,
                user_content=_build_gsm8k_user_instruction(question),
                chat_template_mode="non_thinking",
            ))
            records.append({"id": f"gsm8k-{index}", "question": question, "gold": gold})
        return prompts, records, int(spec["max_new_tokens"]), False

    if task == "tulu":
        ds = list(_load_ifeval_dataset("train", spec["dataset"]))
        if limit is not None:
            ds = ds[:limit]
        for index, row in enumerate(ds):
            raw_prompt = str(row["prompt"])
            inst_ids = list(row.get("instruction_id_list") or [])
            kwargs = list(row.get("kwargs") or [])
            if len(inst_ids) != len(kwargs):
                raise RuntimeError(f"IFEval instruction/kwargs mismatch at {index}")
            prompts.append(render_chat_prompt(
                tokenizer=tokenizer,
                base_model=base_model,
                user_content=raw_prompt,
                chat_template_mode="non_thinking",
            ))
            records.append({
                "id": str(row.get("key", index)),
                "prompt": raw_prompt,
                "instruction_id_list": inst_ids,
                "kwargs": kwargs,
            })
        return prompts, records, int(spec["max_new_tokens"]), False

    if task == "magicoder":
        problems, source = load_humaneval_problems(split="test", dataset_path=spec["dataset"])
        task_ids = sorted(problems)
        if limit is not None:
            task_ids = task_ids[:limit]
        for task_id in task_ids:
            row = problems[task_id]
            prompts.append(render_chat_prompt(
                tokenizer=tokenizer,
                base_model=base_model,
                user_content=build_humaneval_chat_user_prompt(row["prompt"], style="strict_continuation"),
                system_content="",
                chat_template_mode="non_thinking",
            ))
            records.append({
                "id": task_id,
                "problem_prompt": row["prompt"],
                "entry_point": row["entry_point"],
                "dataset_source": source,
            })
        task_root = Path(cfg["_output_dir"]) / task
        task_root.mkdir(parents=True, exist_ok=True)
        jsonl_write(str(task_root / "problems.jsonl"), [problems[task_id] for task_id in task_ids])
        return prompts, records, int(spec["max_new_tokens"]), False

    if task == "commonsense":
        selected = list(spec["tasks"])
        reference_root = Path(spec["reference_predictions_root"])
        for task_name in selected:
            source_path = reference_root / task_name / "predictions.jsonl"
            with source_path.open(encoding="utf-8") as handle:
                source_rows = [json.loads(line) for line in handle if line.strip()]
            if limit is not None:
                source_rows = source_rows[:limit]
            for index, item in enumerate(source_rows):
                instruction = str(item["instruction"])
                choices = [str(value) for value in item["choices"]]
                gold = str(item["gold"])
                prompts.append(render_chat_prompt(
                    tokenizer=tokenizer,
                    base_model=base_model,
                    user_content=instruction,
                    chat_template_mode="non_thinking",
                ))
                records.append({
                    "id": f"{task_name}:{item.get('id', index)}",
                    "subtask": task_name,
                    "question": str(item["question"]),
                    "choices": choices,
                    "gold_index": "ABCDEFGH".index(gold),
                    "source_reference": str(source_path),
                })
        return prompts, records, int(spec["max_new_tokens"]), True
    raise ValueError(f"Unknown task: {task}")


def complete_path(output: Path, task: str, label: str) -> Path:
    return output / task / label / "COMPLETE"


def generate_one(
    *,
    llm,
    prompts: list[str],
    records: list[dict[str, Any]],
    params,
    label: str,
    request,
    output: Path,
    task: str,
    chunk_size: int,
) -> None:
    marker = complete_path(output, task, label)
    if marker.is_file():
        print(f"[Skip] {task}/{label}", flush=True)
        return
    destination = marker.parent
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / "predictions.jsonl.partial"
    partial.unlink(missing_ok=True)
    with partial.open("w", encoding="utf-8") as handle:
        for start in range(0, len(prompts), chunk_size):
            outputs = llm.generate(
                prompts[start:start + chunk_size],
                params,
                lora_request=request,
                use_tqdm=False,
            )
            for record, generated in zip(records[start:start + chunk_size], outputs):
                sample = generated.outputs[0]
                row = dict(record)
                row.update({
                    "prediction_text": sample.text,
                    "token_ids": list(sample.token_ids),
                    "finish_reason": sample.finish_reason,
                })
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[Generate] {task}/{label} {min(start + chunk_size, len(prompts))}/{len(prompts)}", flush=True)
    partial.replace(destination / "predictions.jsonl")
    marker.write_text("complete\n")


def generate_blocks(
    *,
    llm,
    prompts: list[str],
    records: list[dict[str, Any]],
    params,
    variants: list[dict],
    output: Path,
    task: str,
    chunk_size: int,
    block_size: int,
) -> None:
    from vllm.lora.request import LoRARequest

    pending = [row for row in variants if not complete_path(output, task, row["label"]).is_file()]
    for block_start in range(0, len(pending), block_size):
        block = pending[block_start:block_start + block_size]
        if len(block) == 1:
            row = block[0]
            generate_one(
                llm=llm,
                prompts=prompts,
                records=records,
                params=params,
                label=row["label"],
                request=LoRARequest(row["label"], int(row["adapter_id"]), row["path"]),
                output=output,
                task=task,
                chunk_size=chunk_size,
            )
            continue
        handles: dict[str, Any] = {}
        partials: dict[str, Path] = {}
        try:
            for row in block:
                destination = output / task / row["label"]
                destination.mkdir(parents=True, exist_ok=True)
                partial = destination / "predictions.jsonl.partial"
                partial.unlink(missing_ok=True)
                partials[row["label"]] = partial
                handles[row["label"]] = partial.open("w", encoding="utf-8")
            for start in range(0, len(prompts), chunk_size):
                prompt_chunk = prompts[start:start + chunk_size]
                record_chunk = records[start:start + chunk_size]
                flat_prompts: list[str] = []
                requests = []
                ownership: list[tuple[str, dict[str, Any]]] = []
                for row in block:
                    request = LoRARequest(row["label"], int(row["adapter_id"]), row["path"])
                    flat_prompts.extend(prompt_chunk)
                    requests.extend([request] * len(prompt_chunk))
                    ownership.extend((row["label"], record) for record in record_chunk)
                outputs = llm.generate(flat_prompts, params, lora_request=requests, use_tqdm=False)
                for (label, record), generated in zip(ownership, outputs):
                    sample = generated.outputs[0]
                    item = dict(record)
                    item.update({
                        "prediction_text": sample.text,
                        "token_ids": list(sample.token_ids),
                        "finish_reason": sample.finish_reason,
                    })
                    handles[label].write(json.dumps(item, ensure_ascii=False) + "\n")
                print(
                    f"[Generate] {task} block={block_start // block_size + 1} "
                    f"variants={len(block)} prompts={min(start + chunk_size, len(prompts))}/{len(prompts)}",
                    flush=True,
                )
        finally:
            for handle in handles.values():
                handle.close()
        for row in block:
            destination = output / task / row["label"]
            partials[row["label"]].replace(destination / "predictions.jsonl")
            (destination / "COMPLETE").write_text("complete\n")


def main() -> None:
    args = parse_args()
    cfg = read_json(Path(args.config))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg["_output_dir"] = str(output)
    manifest = read_json(Path(args.variant_manifest))
    variants = list(manifest["variants"])
    if not variants or len({row["label"] for row in variants}) != len(variants):
        raise RuntimeError("Empty or duplicate variant labels")
    # The integer id is part of vLLM's LoRA dispatch state.  Keep it tied to
    # the manifest entry so the repeatability diagnostic changes only loading
    # and batching order, not adapter identity.
    for index, row in enumerate(variants, start=1):
        row["adapter_id"] = index
        if not (Path(row["path"]) / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(row["path"])
    if args.reverse_variants:
        variants.reverse()

    tokenizer = load_eval_tokenizer(base_model=args.base_model, adapter_dir=None)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.base_model,
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=min(args.adapter_block_size, len(variants)),
        max_cpu_loras=max(args.adapter_block_size, len(variants)),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend="FLASH_ATTN",
        async_scheduling=False,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        seed=args.seed,
    )
    task_manifest = []
    for task in args.tasks:
        prompts, records, max_tokens, is_short = prepare_task(
            task, cfg, tokenizer, args.base_model, args.diagnostic_max_samples
        )
        if len(prompts) != len(records) or not prompts:
            raise RuntimeError(f"Invalid prepared task {task}")
        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=args.seed)
        chunk_size = args.prompt_chunk_short if is_short else args.prompt_chunk_long
        generate_one(
            llm=llm,
            prompts=prompts,
            records=records,
            params=params,
            label="base",
            request=None,
            output=output,
            task=task,
            chunk_size=chunk_size,
        )
        task_variants = (
            [row for row in variants if row.get("train_task") == task]
            if args.diagonal_only
            else variants
        )
        if not task_variants:
            raise RuntimeError(f"No variants selected for task {task}")
        generate_blocks(
            llm=llm,
            prompts=prompts,
            records=records,
            params=params,
            variants=task_variants,
            output=output,
            task=task,
            chunk_size=chunk_size,
            block_size=args.adapter_block_size,
        )
        task_manifest.append({"task": task, "samples": len(prompts), "max_tokens": max_tokens})
    result = {
        "status": "generation_complete",
        "base_model": str(Path(args.base_model).resolve()),
        "variant_manifest": str(Path(args.variant_manifest).resolve()),
        "variant_order": [row["label"] for row in variants],
        "tasks": task_manifest,
        "configuration": {
            "vllm_batch_invariant": os.getenv("VLLM_BATCH_INVARIANT"),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "adapter_block_size": args.adapter_block_size,
            "seed": args.seed,
        },
    }
    (output / "generation_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
