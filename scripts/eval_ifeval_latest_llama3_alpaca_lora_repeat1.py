#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run IFEval for the Llama-3.1-8B alpaca LoRA baseline and edited adapters (repeat 1).
Outputs land under the eval_outputs run root, with optional suffixing to avoid overwrites.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_RESULTS = Path("eval_results/latest_llama3_spectral_edit/results.jsonl")
DEFAULT_RUN_ID = "meta-llama-Llama-3.1-8B_alpaca_lora_paper_alpaca_3ep_16_42"
DEFAULT_EVAL_ROOT = Path(
    "eval_results/latest_llama3_spectral_edit/eval_outputs/meta-llama-Llama-3.1-8B/alpaca/"
    + DEFAULT_RUN_ID
)

METHOD_ORDER = ["baseline", "abs_select", "grad_direction", "random_index", "smooth_abs"]


def load_results(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def pick_entries(
    results: Iterable[Dict[str, object]],
    run_id: str,
    repeat_idx: int,
    repeat_seed: int | None,
) -> Dict[str, Dict[str, object]]:
    selected: Dict[str, Dict[str, object]] = {}
    for entry in results:
        if entry.get("run_id") != run_id:
            continue
        if entry.get("repeat_idx") != repeat_idx:
            continue
        if repeat_seed is not None and entry.get("repeat_seed") != repeat_seed:
            continue
        method = entry.get("edit_method")
        if isinstance(method, str) and method in METHOD_ORDER and method not in selected:
            selected[method] = entry
    return selected


def output_dir_for(
    eval_root: Path,
    method: str,
    repeat_tag: str,
    output_suffix: str | None,
    overwrite: bool,
) -> Path:
    method_root = eval_root / method
    method_root.mkdir(parents=True, exist_ok=True)
    name = repeat_tag if not output_suffix else f"{repeat_tag}_{output_suffix}"
    out_dir = method_root / name
    if out_dir.exists() and not overwrite:
        out_dir = method_root / f"{name}_rerun"
    return out_dir


def build_command(
    base_model: str,
    adapter_dir: str | None,
    output_dir: Path,
    seed: int,
    max_new_tokens: int,
    max_samples: int | None,
    dtype: str,
    use_vllm: bool,
    tensor_parallel_size: int,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "finetune.eval.eval_ifeval",
        "--base_model",
        base_model,
        "--output_dir",
        str(output_dir),
        "--max_new_tokens",
        str(max_new_tokens),
        "--seed",
        str(seed),
        "--dtype",
        dtype,
    ]
    if adapter_dir:
        cmd += ["--adapter_dir", adapter_dir]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]
    if use_vllm:
        cmd += ["--use_vllm", "--tensor_parallel_size", str(tensor_parallel_size)]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run IFEval for baseline + edited adapters (repeat 1) for Llama-3.1-8B alpaca LoRA."
    )
    parser.add_argument("--results_jsonl", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run_id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument("--eval_root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--repeat_idx", type=int, default=1)
    parser.add_argument("--repeat_seed", type=int, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--output_suffix", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not args.results_jsonl.exists():
        raise FileNotFoundError(f"results.jsonl not found: {args.results_jsonl}")

    entries = pick_entries(
        load_results(args.results_jsonl),
        run_id=args.run_id,
        repeat_idx=args.repeat_idx,
        repeat_seed=args.repeat_seed,
    )
    if not entries:
        raise RuntimeError(
            f"No matching entries for run_id={args.run_id} repeat_idx={args.repeat_idx} "
            f"in {args.results_jsonl}"
        )

    base_model = args.base_model
    if base_model is None:
        base_model = str(next(iter(entries.values())).get("base_model_id"))

    for method in METHOD_ORDER:
        entry = entries.get(method)
        if not entry:
            continue
        repeat_tag = str(entry.get("repeat_tag") or f"repeat_{args.repeat_idx:02d}")
        seed = int(entry.get("repeat_seed") or args.repeat_idx)
        if method == "baseline":
            adapter_dir = entry.get("adapter_dir")
        else:
            adapter_dir = entry.get("edited_adapter_dir")
        if not adapter_dir:
            raise RuntimeError(f"Missing adapter_dir for method={method}")
        output_dir = output_dir_for(
            args.eval_root,
            method=method,
            repeat_tag=repeat_tag,
            output_suffix=args.output_suffix,
            overwrite=args.overwrite,
        )
        cmd = build_command(
            base_model=base_model,
            adapter_dir=str(adapter_dir),
            output_dir=output_dir,
            seed=seed,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            dtype=args.dtype,
            use_vllm=args.use_vllm,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
