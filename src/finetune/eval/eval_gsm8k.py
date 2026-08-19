from __future__ import annotations

import argparse
import json
import re
from glob import glob
from pathlib import Path
from typing import Any

from finetune.eval.generation import (
    generate_greedy,
    generate_greedy_vllm_batch,
    load_eval_tokenizer,
    load_transformers_model,
    render_chat_prompt,
    save_json,
)
from finetune.utils import seed_everything


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


def _build_gsm8k_user_instruction(question: str) -> str:
    return (
        "Solve the following math word problem.\n"
        "Put your final numeric answer on the last line exactly as:\n"
        "#### <answer>\n\n"
        f"{question.strip()}"
    )


def _resolve_local_gsm8k_data_files(
    dataset_path: str,
    split: str,
    dataset_config: str,
) -> tuple[str, list[str]]:
    path = Path(dataset_path)
    if not path.exists():
        raise RuntimeError(f"Local GSM8K dataset path not found: {dataset_path}")

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return "parquet", [str(path)]
        if suffix == ".json":
            return "json", [str(path)]
        if suffix == ".jsonl":
            return "json", [str(path)]
        raise RuntimeError(
            f"Unsupported local GSM8K file extension: {path.suffix!r}. Use .parquet, .json, or .jsonl."
        )

    candidates = [
        str(path / dataset_config / f"{split}-*.parquet"),
        str(path / f"{split}-*.parquet"),
    ]
    matches: list[str] = []
    for pattern in candidates:
        matched = sorted(glob(pattern))
        if matched:
            matches = matched
            break

    if matches:
        return "parquet", matches

    raise RuntimeError(
        f"Could not find local GSM8K files for split={split!r}, dataset_config={dataset_config!r} under {dataset_path}. "
        f"Tried patterns: {candidates}"
    )


def load_gsm8k_split(
    *,
    split: str,
    dataset_path: str | None,
    dataset_config: str,
):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(f"datasets is required: {exc}") from exc

    if dataset_path is None:
        try:
            return load_dataset("gsm8k", dataset_config, split=split)
        except Exception as exc:
            raise RuntimeError(f"Failed to load gsm8k: {exc}") from exc

    loader_name, data_files = _resolve_local_gsm8k_data_files(
        dataset_path=dataset_path,
        split=split,
        dataset_config=dataset_config,
    )
    try:
        return load_dataset(loader_name, data_files=data_files, split="train")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load local GSM8K data from {dataset_path}: {exc}\n"
            f"Resolved loader={loader_name!r}, files={data_files!r}"
        ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate GSM8K strict-match accuracy using tokenizer chat templates.")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional local GSM8K dataset root or file. Supports a snapshot directory with main/test-*.parquet.",
    )
    p.add_argument(
        "--dataset_config",
        type=str,
        default="main",
        help="GSM8K config/subset name. Defaults to 'main'.",
    )
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Optional vLLM max_model_len override. Useful on 24GB GPUs where the base model's full context window is too large.",
    )
    p.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=None,
        help="Optional vLLM GPU memory utilization target, e.g. 0.9 or 0.95.",
    )
    p.add_argument(
        "--vllm_attention_backend",
        type=str,
        default=None,
        help="Optional vLLM attention backend override, e.g. FLASH_ATTN or FLASHINFER.",
    )
    p.add_argument(
        "--vllm_disable_flashinfer_sampler",
        action="store_true",
        help="Disable FlashInfer top-k/top-p sampler inside vLLM and fall back to the native sampler.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"

    ds = load_gsm8k_split(
        split=args.split,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
    )

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    eval_tokenizer = load_eval_tokenizer(base_model=args.base_model, adapter_dir=args.adapter_dir)

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
            records: list[tuple[str, str]] = []

            for ex in examples:
                q = str(ex.get("question", "")).strip()
                gold_raw = str(ex.get("answer", "")).strip()
                gold = _norm(_extract_answer(gold_raw))

                prompt = render_chat_prompt(
                    tokenizer=eval_tokenizer,
                    base_model=args.base_model,
                    user_content=_build_gsm8k_user_instruction(q),
                )
                prompts.append(prompt)
                records.append((q, gold))

            generations = generate_greedy_vllm_batch(
                base_model=args.base_model,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                adapter_dir=args.adapter_dir,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.vllm_max_model_len,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                attention_backend=args.vllm_attention_backend,
                disable_flashinfer_sampler=args.vllm_disable_flashinfer_sampler,
            )

            for (q, gold), gen in zip(records, generations):
                pred = _norm(_extract_answer(gen))
                is_correct = int(pred == gold)
                correct += is_correct
                total += 1

                rec: dict[str, Any] = {
                    "question": q,
                    "gold": gold,
                    "prompt_style": "tokenizer_chat_template",
                    "prediction_text": gen,
                    "prediction_extracted": pred,
                    "correct": bool(is_correct),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            for ex in ds:
                q = str(ex.get("question", "")).strip()
                gold_raw = str(ex.get("answer", "")).strip()
                gold = _norm(_extract_answer(gold_raw))

                prompt = render_chat_prompt(
                    tokenizer=eval_tokenizer,
                    base_model=args.base_model,
                    user_content=_build_gsm8k_user_instruction(q),
                )

                gen = generate_greedy(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )

                pred = _norm(_extract_answer(gen))
                is_correct = int(pred == gold)
                correct += is_correct
                total += 1

                rec: dict[str, Any] = {
                    "question": q,
                    "gold": gold,
                    "prompt_style": "tokenizer_chat_template",
                    "prediction_text": gen,
                    "prediction_extracted": pred,
                    "correct": bool(is_correct),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = {
        "accuracy_strict": (correct / total if total else 0.0),
        "correct": correct,
        "total": total,
        "prompt_style": "tokenizer_chat_template",
        "dataset_source": args.dataset_path or "hf://gsm8k",
        "dataset_config": args.dataset_config,
        "split": args.split,
        "use_vllm": bool(args.use_vllm),
        "vllm_attention_backend": args.vllm_attention_backend,
        "vllm_disable_flashinfer_sampler": bool(args.vllm_disable_flashinfer_sampler),
        "vllm_max_model_len": args.vllm_max_model_len,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
    }
    save_json(out_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
