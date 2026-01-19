from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

from finetune.eval.generation import (
    generate_greedy,
    generate_greedy_vllm,
    load_transformers_model,
    save_json,
    strip_code_fences,
)
from finetune.utils import seed_everything


def _run_program(program: str, timeout_s: float) -> tuple[bool, str]:
    """
    Minimal functional-correctness check: exec(prompt + completion + tests) in a subprocess.
    WARNING: This executes generated code. Use in a controlled environment.
    """

    def worker(code: str, q: mp.Queue):
        try:
            glb: dict[str, Any] = {}
            exec(code, glb, glb)
            q.put({"passed": True, "error": ""})
        except BaseException as exc:  # noqa: BLE001
            q.put({"passed": False, "error": repr(exc)})

    q: mp.Queue = mp.Queue()
    p = mp.Process(target=worker, args=(program, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(1)
        return False, "timeout"
    try:
        res = q.get_nowait()
    except Exception:
        return False, "no_result"
    return bool(res.get("passed", False)), str(res.get("error", ""))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate HumanEval pass@1 (minimal, greedy).")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout_s", type=float, default=3.0)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = out_dir / "outputs.jsonl"

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(f"datasets is required: {exc}") from exc

    try:
        ds = load_dataset("openai_humaneval", split="test")
    except Exception as exc:
        raise RuntimeError(f"Failed to load openai_humaneval: {exc}") from exc

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

    passed = 0
    total = 0
    with outputs_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            task_id = str(ex.get("task_id", ""))
            prompt = str(ex.get("prompt", ""))
            test = str(ex.get("test", ""))
            entry_point = str(ex.get("entry_point", "")).strip()
            if not prompt or not test or not entry_point:
                continue

            if args.use_vllm:
                completion_raw = generate_greedy_vllm(
                    base_model=args.base_model,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    adapter_dir=args.adapter_dir,
                    tensor_parallel_size=args.tensor_parallel_size,
                )
            else:
                completion_raw = generate_greedy(
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )

            completion = strip_code_fences(completion_raw)
            program = f"{prompt}{completion}\n\n{test}\n\ncheck({entry_point})\n"

            ok, error = _run_program(program, timeout_s=args.timeout_s)
            passed += int(ok)
            total += 1

            f.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "passed": bool(ok),
                        "error": error,
                        "prompt": prompt,
                        "completion": completion,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {"pass@1": (passed / total if total else 0.0), "passed": passed, "total": total}
    save_json(out_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()


