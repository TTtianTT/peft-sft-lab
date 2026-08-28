#!/usr/bin/env python3
"""Evaluate MBPP code generation with an optional chat template and LoRA adapter.

The default chat protocol matches LlamaFactory ``template: llama3`` training on
Magicoder: one user instruction followed by one assistant generation, without a
system message. Generated code is executed with the MBPP tests in a temporary
directory. This is not a security sandbox; only run it in an isolated machine.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from finetune.data.base import first_present, get_writable_datasets_cache_dir, load_local_dataset
from finetune.eval.eval_humaneval import ensure_lora_has_config
from finetune.eval.generation import (
    extract_first_python_code_block,
    find_parseable_python_segment,
    generate_greedy,
    generate_greedy_vllm_batch,
    load_eval_tokenizer,
    load_transformers_model,
    render_chat_prompt,
    save_json,
    strip_outer_blank_lines,
    strip_code_fences,
)
from finetune.utils import seed_everything


HF_DATASET_ID = "google-research-datasets/mbpp"
DEFAULT_SPLIT = "test"
DEFAULT_PROMPT_STYLE = "chat"


def _as_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return [stripped]
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if str(item).strip()]
        return [str(parsed)]
    raise ValueError(f"MBPP field {field_name!r} must be a string or list, got {type(value).__name__}.")


def _normalize_mbpp_problem(example: dict[str, Any]) -> dict[str, Any]:
    task_id = first_present(example, ["task_id", "id", "problem_id"])
    text = first_present(example, ["text", "prompt", "instruction"])
    if task_id is None or text is None:
        raise ValueError(
            "MBPP example missing required fields. "
            f"Keys: {sorted(example.keys())}. Expected at least task_id and text/prompt."
        )

    tests = _as_string_list(example.get("test_list", example.get("tests")), field_name="test_list")
    if not tests:
        raise ValueError(f"MBPP task {task_id!r} has no test_list/tests entries.")

    setup = first_present(example, ["test_setup_code", "test_imports", "setup_code"]) or ""
    if isinstance(example.get("test_imports"), (list, tuple)):
        setup = "\n".join(str(item) for item in example["test_imports"] if str(item).strip())

    return {
        "task_id": str(task_id),
        "text": str(text),
        "test_list": tests,
        "test_setup_code": str(setup),
        "reference_code": first_present(example, ["code", "canonical_solution", "solution"]) or "",
    }


def load_mbpp_problems(*, split: str, dataset_path: str | None) -> tuple[list[dict[str, Any]], str]:
    if dataset_path is not None:
        dataset = load_local_dataset(
            dataset_path,
            task_name="MBPP",
            expected_fields_hint="task_id, text/prompt, test_list, test_setup_code",
        )
        source = f"local://{dataset_path}"
    else:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(f"datasets is required to load MBPP: {exc}") from exc

        try:
            dataset = load_dataset(
                HF_DATASET_ID,
                "sanitized",
                split=split,
                cache_dir=get_writable_datasets_cache_dir(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {HF_DATASET_ID!r} config='sanitized' split={split!r}: {exc}. "
                "Download the sanitized MBPP file locally and pass --dataset_path if this host cannot reach Hugging Face."
            ) from exc
        source = f"hf://{HF_DATASET_ID}/sanitized/{split}"

    problems = [_normalize_mbpp_problem(dict(row)) for row in dataset]
    if not problems:
        raise RuntimeError(f"Loaded an empty MBPP split from {source}.")
    return problems, source


def build_mbpp_chat_user_prompt(problem_text: str, public_test: str) -> str:
    return (
        "Write a correct Python solution for the following programming problem. "
        "Return only executable Python code. Do not add explanations or Markdown code fences.\n\n"
        f"Problem:\n{problem_text}\n\n"
        "Your solution must satisfy this public test:\n"
        f"{public_test}"
    )


def normalize_mbpp_completion(raw_text: str) -> str:
    text = strip_outer_blank_lines(raw_text)
    fenced_code = extract_first_python_code_block(text)
    if fenced_code is not None:
        text = fenced_code
    else:
        text = strip_code_fences(text).strip()
    try:
        import re

        # Recover code when an instruct model emits one prose line before an
        # otherwise valid un-fenced Python answer.
        code_start = re.search(r"(?m)^(?:from\s+\S+\s+import\s+|import\s+|(?:async\s+)?def\s+|class\s+|@)", text)
        if code_start and code_start.start() > 0:
            text = text[code_start.start() :]
    except Exception:
        pass
    parseable = find_parseable_python_segment(text)
    return parseable if parseable is not None else text


def _mbpp_program(problem: dict[str, Any], completion: str) -> str:
    parts = [problem["test_setup_code"].strip(), completion.strip(), "\n".join(problem["test_list"])]
    return "\n\n".join(part for part in parts if part) + "\n"


def _run_mbpp_problem(problem: dict[str, Any], completion: str, timeout_s: float) -> dict[str, Any]:
    program = _mbpp_program(problem, completion)
    with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as tmpdir:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": tmpdir,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(timeout_s),
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "result": "timeout", "error": f"Timed out after {timeout_s}s."}

    if proc.returncode == 0:
        return {"passed": True, "result": "passed", "error": None}
    error = (proc.stderr or proc.stdout or f"Python exited with code {proc.returncode}.").strip()
    return {"passed": False, "result": "failed", "error": error[-4000:]}


def evaluate_mbpp(
    problems: list[dict[str, Any]],
    completions: list[str],
    *,
    timeout_s: float,
    n_workers: int,
) -> list[dict[str, Any]]:
    if len(problems) != len(completions):
        raise ValueError("MBPP problems and completions must have equal length.")

    def run(index: int) -> tuple[int, dict[str, Any]]:
        return index, _run_mbpp_problem(problems[index], completions[index], timeout_s)

    results: list[dict[str, Any]] = [{} for _ in problems]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(n_workers))) as pool:
        futures = [pool.submit(run, index) for index in range(len(problems))]
        for future in concurrent.futures.as_completed(futures):
            index, result = future.result()
            results[index] = result
    return results


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MBPP greedy pass@1 with optional LlamaFactory-compatible chat prompting.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--config_src", default=None, help="Original LoRA dir used only when an edited adapter lacks adapter_config.json.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_path", default=None, help="Optional local MBPP parquet/json/jsonl file or directory.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--prompt_style", choices=["chat", "raw"], default=DEFAULT_PROMPT_STYLE)
    parser.add_argument(
        "--chat_template_mode",
        default="auto",
        choices=["auto", "thinking", "non_thinking"],
        help=(
            "Thinking mode passed to tokenizer.apply_chat_template for chat prompts. "
            "Use non_thinking when evaluating Qwen3 adapters trained with enable_thinking=False."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--timeout_s", type=float, default=3.0)
    parser.add_argument("--eval_n_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_attention_backend", default=None)
    parser.add_argument("--vllm_disable_flashinfer_sampler", action="store_true")
    parser.add_argument("--vllm_request_batch_size", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.adapter_dir:
        ensure_lora_has_config(args.adapter_dir, args.config_src)

    problems, dataset_source = load_mbpp_problems(split=args.split, dataset_path=args.dataset_path)
    if args.max_samples is not None and args.max_samples > 0:
        problems = problems[: args.max_samples]
    print(f"[Data] Loaded {len(problems)} MBPP tasks from {dataset_source}.")

    # MBPP task text commonly omits the required function name. Its standard
    # protocol exposes one test case in the prompt so the target entry point is
    # known to the model; the remaining tests still determine correctness.
    user_prompts = [
        build_mbpp_chat_user_prompt(problem["text"], problem["test_list"][0])
        for problem in problems
    ]
    if args.prompt_style == "chat":
        tokenizer = load_eval_tokenizer(base_model=args.base_model, adapter_dir=args.adapter_dir)
        model_inputs = [
            render_chat_prompt(
                tokenizer=tokenizer,
                base_model=args.base_model,
                user_content=user_prompt,
                system_content=None,
                chat_template_mode=args.chat_template_mode,
            )
            for user_prompt in user_prompts
        ]
    else:
        model_inputs = user_prompts

    if args.use_vllm:
        raw_completions = generate_greedy_vllm_batch(
            base_model=args.base_model,
            prompts=model_inputs,
            adapter_dir=args.adapter_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.vllm_max_model_len,
            attention_backend=args.vllm_attention_backend,
            disable_flashinfer_sampler=args.vllm_disable_flashinfer_sampler,
            request_batch_size=args.vllm_request_batch_size,
        )
    else:
        loaded = load_transformers_model(base_model=args.base_model, adapter_dir=args.adapter_dir, dtype=args.dtype, device_map="auto")
        raw_completions = [
            generate_greedy(model=loaded.model, tokenizer=loaded.tokenizer, prompt=model_input, max_new_tokens=args.max_new_tokens)
            for model_input in model_inputs
        ]

    completions = [normalize_mbpp_completion(text) for text in raw_completions]
    results = evaluate_mbpp(
        problems,
        completions,
        timeout_s=args.timeout_s,
        n_workers=args.eval_n_workers,
    )

    output_rows = [
        {
            "task_id": problem["task_id"],
            "passed": result["passed"],
            "result": result["result"],
            "error": result["error"],
            "prompt": problem["text"],
            "raw_completion": raw_completion,
            "completion": completion,
            "model_input_prompt": model_input,
        }
        for problem, raw_completion, completion, model_input, result in zip(
            problems,
            raw_completions,
            completions,
            model_inputs,
            results,
        )
    ]
    adapter_tag = "lora" if args.adapter_dir else "base"
    outputs_path = out_dir / f"outputs_{adapter_tag}.jsonl"
    _jsonl_write(outputs_path, output_rows)

    correct = sum(1 for result in results if result["passed"])
    total = len(results)
    metrics = {
        "pass@1": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "dataset_source": dataset_source,
        "prompt_style": args.prompt_style,
        "chat_template_mode": args.chat_template_mode if args.prompt_style == "chat" else None,
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "use_vllm": bool(args.use_vllm),
        "vllm_max_model_len": args.vllm_max_model_len,
        "vllm_attention_backend": args.vllm_attention_backend,
        "vllm_disable_flashinfer_sampler": bool(args.vllm_disable_flashinfer_sampler),
        "vllm_request_batch_size": args.vllm_request_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "timeout_s": args.timeout_s,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "eval_config.json", {"task": "mbpp", **metrics})
    print(f"[Done] pass@1={metrics['pass@1']}, correct={correct}, total={total}")
    print(f"[Done] outputs: {outputs_path}")


if __name__ == "__main__":
    main()
