from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
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
from finetune.data.base import get_writable_datasets_cache_dir
from finetune.utils import seed_everything

HF_DATASET_ID = "allenai/IFBench_test"


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_local_ifbench_data_files(dataset_path: str, split: str) -> tuple[str, list[str]]:
    path = Path(dataset_path)
    if not path.exists():
        raise RuntimeError(f"Local IFBench dataset path not found: {dataset_path}")

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return "parquet", [str(path)]
        if suffix == ".json":
            return "json", [str(path)]
        if suffix == ".jsonl":
            return "json", [str(path)]
        raise RuntimeError(
            f"Unsupported local IFBench file extension: {path.suffix!r}. Use .parquet, .json, or .jsonl."
        )

    candidates = [
        str(path / "data" / f"{split}-*.parquet"),
        str(path / f"{split}-*.parquet"),
        str(path / "data" / f"{split}.jsonl"),
        str(path / f"{split}.jsonl"),
        str(path / "data" / f"{split}.json"),
        str(path / f"{split}.json"),
    ]
    for pattern in candidates:
        matches = sorted(glob(pattern))
        if matches:
            loader_name = "parquet" if matches[0].endswith(".parquet") else "json"
            return loader_name, matches

    raise RuntimeError(
        f"Could not find local IFBench files for split={split!r} under {dataset_path}. "
        f"Tried patterns: {candidates}"
    )


def load_ifbench_split(*, split: str, dataset_path: str | None):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(f"datasets is required: {exc}") from exc

    if dataset_path is None:
        try:
            return load_dataset(HF_DATASET_ID, split=split)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {HF_DATASET_ID}: {exc}") from exc

    loader_name, data_files = _resolve_local_ifbench_data_files(dataset_path=dataset_path, split=split)
    try:
        return load_dataset(
            loader_name,
            data_files=data_files,
            split="train",
            cache_dir=get_writable_datasets_cache_dir(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load local IFBench data from {dataset_path}: {exc}\n"
            f"Resolved loader={loader_name!r}, files={data_files!r}"
        ) from exc


def _resolve_official_ifbench_root(official_eval_root: str) -> Path:
    root = Path(official_eval_root).expanduser().resolve()
    run_eval = root / "run_eval.py"
    evaluation_lib = root / "evaluation_lib.py"
    if run_eval.is_file() and evaluation_lib.is_file():
        return root
    raise RuntimeError(
        f"Invalid --official_eval_root={official_eval_root}. "
        "Expected the root of the allenai/IFBench repository containing run_eval.py."
    )


def run_official_ifbench(
    *,
    out_dir: Path,
    input_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    official_eval_root: str,
) -> Path:
    official_root = _resolve_official_ifbench_root(official_eval_root)
    official_out = out_dir / "official_ifbench"
    official_out.mkdir(parents=True, exist_ok=True)

    input_data_path = official_out / "input_data.jsonl"
    input_response_data_path = official_out / "input_response_data.jsonl"
    _dump_jsonl(input_data_path, input_rows)
    _dump_jsonl(input_response_data_path, response_rows)

    cmd = [
        sys.executable,
        str(official_root / "run_eval.py"),
        f"--input_data={input_data_path}",
        f"--input_response_data={input_response_data_path}",
        f"--output_dir={official_out}",
    ]
    subprocess.run(cmd, check=True, cwd=str(official_root))
    return official_out


def _load_official_outputs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Official IFBench output file is empty: {path}")
    return rows


def _summarize_official_outputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_total = len(rows)
    prompt_correct = sum(1 for row in rows if bool(row.get("follow_all_instructions")))
    instruction_total = 0
    instruction_correct = 0
    by_family_total: dict[str, int] = defaultdict(int)
    by_family_correct: dict[str, int] = defaultdict(int)
    by_instruction_total: dict[str, int] = defaultdict(int)
    by_instruction_correct: dict[str, int] = defaultdict(int)

    for row in rows:
        inst_ids = list(row.get("instruction_id_list") or [])
        follow_list = [bool(v) for v in row.get("follow_instruction_list") or []]
        instruction_total += len(inst_ids)
        instruction_correct += sum(follow_list)
        for inst_id, followed in zip(inst_ids, follow_list):
            family = str(inst_id).split(":", 1)[0]
            by_family_total[family] += 1
            by_family_correct[family] += int(followed)
            by_instruction_total[str(inst_id)] += 1
            by_instruction_correct[str(inst_id)] += int(followed)

    return {
        "prompt_accuracy": (prompt_correct / prompt_total if prompt_total else 0.0),
        "instruction_accuracy": (instruction_correct / instruction_total if instruction_total else 0.0),
        "prompt_correct": prompt_correct,
        "prompt_total": prompt_total,
        "instruction_correct": instruction_correct,
        "instruction_total": instruction_total,
        "by_family": {
            key: (by_family_correct[key] / by_family_total[key] if by_family_total[key] else 0.0)
            for key in sorted(by_family_total)
        },
        "by_instruction": {
            key: (by_instruction_correct[key] / by_instruction_total[key] if by_instruction_total[key] else 0.0)
            for key in sorted(by_instruction_total)
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate IFBench using the official allenai/IFBench scorer.")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--official_eval_root",
        type=str,
        required=True,
        help="Path to a local checkout of https://github.com/allenai/IFBench.",
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional local IFBench dataset root or file. Supports a repo snapshot or a parquet/json/jsonl file.",
    )
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Optional vLLM max_model_len override.",
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
    responses_path = out_dir / "responses.jsonl"

    ds = load_ifbench_split(split=args.split, dataset_path=args.dataset_path)
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

    examples = list(ds)
    prompts: list[str] = []
    input_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    prompt_metadata: list[dict[str, Any]] = []

    for ex in examples:
        prompt = str(ex.get("prompt", "")).strip()
        if not prompt:
            raise RuntimeError(f"IFBench example missing prompt. Keys: {sorted(ex.keys())}")

        instruction_id_list = list(ex.get("instruction_id_list") or [])
        kwargs = list(ex.get("kwargs") or [])
        if len(instruction_id_list) != len(kwargs):
            raise RuntimeError(
                f"Mismatch: {len(instruction_id_list)} instruction_ids vs {len(kwargs)} kwargs for key={ex.get('key')}"
            )

        prompts.append(
            render_chat_prompt(
                tokenizer=eval_tokenizer,
                base_model=args.base_model,
                user_content=prompt,
            )
        )
        input_rows.append(
            {
                "key": ex.get("key"),
                "prompt": prompt,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
            }
        )
        prompt_metadata.append(
            {
                "key": ex.get("key"),
                "prompt": prompt,
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
            }
        )

    if args.use_vllm:
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
    else:
        generations = [
            generate_greedy(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                prompt=rendered_prompt,
                max_new_tokens=args.max_new_tokens,
            )
            for rendered_prompt in prompts
        ]

    for meta, generation in zip(prompt_metadata, generations):
        response_rows.append(
            {
                "key": meta["key"],
                "prompt": meta["prompt"],
                "response": generation,
            }
        )

    _dump_jsonl(responses_path, response_rows)
    official_out = run_official_ifbench(
        out_dir=out_dir,
        input_rows=input_rows,
        response_rows=response_rows,
        official_eval_root=args.official_eval_root,
    )

    strict_rows = _load_official_outputs(official_out / "input_response_data-eval_results_strict.jsonl")
    loose_rows = _load_official_outputs(official_out / "input_response_data-eval_results_loose.jsonl")
    strict_summary = _summarize_official_outputs(strict_rows)
    loose_summary = _summarize_official_outputs(loose_rows)

    metrics = {
        "ifbench_prompt_loose_accuracy": loose_summary["prompt_accuracy"],
        "ifbench_prompt_strict_accuracy": strict_summary["prompt_accuracy"],
        "ifbench_instruction_loose_accuracy": loose_summary["instruction_accuracy"],
        "ifbench_instruction_strict_accuracy": strict_summary["instruction_accuracy"],
        "primary_metric": "ifbench_prompt_loose_accuracy",
        "dataset_source": args.dataset_path or f"hf://{HF_DATASET_ID}",
        "split": args.split,
        "use_vllm": bool(args.use_vllm),
        "official_eval_root": str(Path(args.official_eval_root).expanduser()),
        "official_output_dir": str(official_out),
        "max_new_tokens": args.max_new_tokens,
        "num_examples": len(input_rows),
        "strict": strict_summary,
        "loose": loose_summary,
        "vllm_attention_backend": args.vllm_attention_backend,
        "vllm_disable_flashinfer_sampler": bool(args.vllm_disable_flashinfer_sampler),
        "vllm_max_model_len": args.vllm_max_model_len,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
    }
    save_json(out_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
