#!/usr/bin/env python3
"""Stochastic rollout audit for off-task cases where HNS exceeded the base model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_forgetting_matrix_vllm import prepare_task  # noqa: E402
from finetune.eval.generation import load_eval_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_config", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--max_num_batched_tokens", type=int, default=65536)
    parser.add_argument("--prompt_chunk", type=int, default=64)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_condition(
    *,
    llm,
    prompts: list[str],
    records: list[dict[str, Any]],
    params,
    request,
    destination: Path,
    chunk_size: int,
) -> None:
    marker = destination / "COMPLETE"
    if marker.is_file():
        print(f"[Skip] {destination}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / "predictions.jsonl.partial"
    partial.unlink(missing_ok=True)
    with partial.open("w", encoding="utf-8") as handle:
        for start in range(0, len(prompts), chunk_size):
            generated_rows = llm.generate(
                prompts[start : start + chunk_size],
                params,
                lora_request=request,
                use_tqdm=False,
            )
            for record, generated in zip(records[start : start + chunk_size], generated_rows):
                row = dict(record)
                row["rollouts"] = [
                    {
                        "text": sample.text,
                        "token_ids": list(sample.token_ids),
                        "finish_reason": sample.finish_reason,
                    }
                    for sample in generated.outputs
                ]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[Generate] {destination.parent.name}/{destination.name} "
                f"{min(start + chunk_size, len(prompts))}/{len(prompts)}",
                flush=True,
            )
    partial.replace(destination / "predictions.jsonl")
    marker.write_text("complete\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = read_json(Path(args.experiment_config))
    base_cfg = cfg["bases"][args.base]
    base_model = str(Path(base_cfg["base_model"]).resolve())
    output = Path(args.output_dir).resolve() / args.base
    output.mkdir(parents=True, exist_ok=True)

    forgetting_cfg = read_json(Path(cfg["forgetting_config"]))
    forgetting_cfg["_output_dir"] = str(output)
    tokenizer = load_eval_tokenizer(base_model=base_model, adapter_dir=None)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    conditions = [condition for edge in base_cfg["edges"] for condition in edge["conditions"]]
    max_rank = max(int(condition.get("max_rank", 256)) for condition in conditions)
    llm = LLM(
        model=base_model,
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=max_rank,
        max_loras=1,
        max_cpu_loras=len(conditions),
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

    run_manifest: list[dict[str, Any]] = []
    adapter_id = 1
    for edge in base_cfg["edges"]:
        task = edge["eval_task"]
        prompts, records, max_tokens, _ = prepare_task(
            task, forgetting_cfg, tokenizer, base_model, edge.get("max_samples")
        )
        params = SamplingParams(
            n=args.num_rollouts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=max_tokens,
            seed=args.seed,
        )
        generate_condition(
            llm=llm,
            prompts=prompts,
            records=records,
            params=params,
            request=None,
            destination=output / task / "base",
            chunk_size=args.prompt_chunk,
        )
        for condition in edge["conditions"]:
            adapter_path = Path(condition["path"])
            if not (adapter_path / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(adapter_path)
            request = LoRARequest(condition["label"], adapter_id, str(adapter_path.resolve()))
            generate_condition(
                llm=llm,
                prompts=prompts,
                records=records,
                params=params,
                request=request,
                destination=output / task / condition["label"],
                chunk_size=args.prompt_chunk,
            )
            adapter_id += 1
        run_manifest.append({
            "eval_task": task,
            "train_task": edge["train_task"],
            "samples": len(prompts),
            "conditions": ["base"] + [row["label"] for row in edge["conditions"]],
        })

    manifest = {
        "status": "generation_complete",
        "base": args.base,
        "base_model": base_model,
        "edges": run_manifest,
        "sampling": {
            "n": args.num_rollouts,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "vllm_batch_invariant": os.getenv("VLLM_BATCH_INVARIANT"),
        },
    }
    (output / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
