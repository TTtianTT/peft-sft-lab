#!/usr/bin/env python3
"""
Rerun only the math few-shot operating points after the next-prompt stop/truncation fix.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


BASE_MODEL_DIR_TO_ID = {
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
}


@dataclass
class AdapterInfo:
    base_model_dir: str
    base_model_id: str
    adapter_dir: Path


@dataclass
class EvalRecord:
    timestamp: str
    base_model_dir: str
    base_model_id: str
    task: str
    method: str
    fewshot_k: int
    fewshot_seed: int
    eval_seed: int
    adapter_dir: str
    eval_output_dir: str
    metric_name: str
    metric_value: Optional[float]
    runtime_seconds: Optional[float]
    avg_prompt_tokens: Optional[float]
    avg_extra_prompt_tokens: Optional[float]
    total_extra_prompt_tokens: Optional[float]
    max_prompt_tokens: Optional[int]
    all_metrics: Optional[dict[str, Any]]
    fix_tag: str
    error: Optional[str] = None


class ResultsWriter:
    def __init__(self, out_root: Path):
        self.out_root = out_root
        self.jsonl_path = out_root / "results.jsonl"
        self.csv_path = out_root / "results.csv"
        self.out_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.existing_keys: set[str] = set()
        self._load_existing()

    def _key(self, base_model_dir: str, k: int, eval_seed: int) -> str:
        return f"{base_model_dir}|math|{k}|{eval_seed}"

    def _load_existing(self) -> None:
        if not self.jsonl_path.exists():
            return
        with self.jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.existing_keys.add(
                    self._key(
                        rec["base_model_dir"],
                        int(rec["fewshot_k"]),
                        int(rec["eval_seed"]),
                    )
                )

    def is_done(self, base_model_dir: str, k: int, eval_seed: int) -> bool:
        return self._key(base_model_dir, k, eval_seed) in self.existing_keys

    def write(self, record: EvalRecord) -> None:
        with self._lock:
            key = self._key(record.base_model_dir, record.fewshot_k, record.eval_seed)
            if key in self.existing_keys:
                return
            self.existing_keys.add(key)
            data = asdict(record)
            csv_row = dict(data)
            if csv_row["all_metrics"] is not None:
                csv_row["all_metrics"] = json.dumps(csv_row["all_metrics"], ensure_ascii=False)
            with self.jsonl_path.open("a") as f:
                f.write(json.dumps(data) + "\n")
            exists = self.csv_path.exists()
            with self.csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
                if not exists:
                    writer.writeheader()
                writer.writerow(csv_row)


class GPUPool:
    def __init__(self, gpu_ids: list[int]):
        self._q: Queue[int] = Queue()
        for gpu_id in gpu_ids:
            self._q.put(gpu_id)

    def acquire(self) -> int:
        return self._q.get()

    def release(self, gpu_id: int) -> None:
        self._q.put(gpu_id)


def discover_math_adapters(runs_root: Path) -> list[AdapterInfo]:
    adapters: list[AdapterInfo] = []
    for base_model_dir, base_model_id in BASE_MODEL_DIR_TO_ID.items():
        adapter_dir = runs_root / base_model_dir / "math" / "lora"
        if not adapter_dir.exists():
            continue
        for root, _, files in os.walk(adapter_dir):
            root_path = Path(root)
            if "checkpoint-" in str(root_path):
                continue
            if "adapter_config.json" not in files:
                continue
            if "adapter_model.safetensors" not in files and "adapter_model.bin" not in files:
                continue
            if root_path.name != "seed42":
                continue
            adapters.append(
                AdapterInfo(
                    base_model_dir=base_model_dir,
                    base_model_id=base_model_id,
                    adapter_dir=root_path,
                )
            )
    adapters.sort(key=lambda x: x.base_model_dir)
    return adapters


def run_single_eval(
    *,
    adapter: AdapterInfo,
    k: int,
    fewshot_seed: int,
    eval_seed: int,
    output_dir: Path,
    gpu_id: int,
) -> tuple[Optional[dict[str, Any]], Optional[float], Optional[str]]:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            return metrics, None, None
        except Exception:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "finetune.eval.eval_gsm8k",
        "--base_model",
        adapter.base_model_id,
        "--adapter_dir",
        str(adapter.adapter_dir),
        "--output_dir",
        str(output_dir),
        "--seed",
        str(eval_seed),
        "--fewshot_k",
        str(k),
        "--fewshot_seed",
        str(fewshot_seed),
        "--max_new_tokens",
        "256",
        "--use_vllm",
        "--tensor_parallel_size",
        "1",
    ]

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "PYTHONPATH": str(SRC_DIR),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=4 * 3600,
        )
    except subprocess.TimeoutExpired:
        return None, None, "Eval timed out after 4 hours"
    except Exception as exc:
        return None, None, str(exc)

    runtime = time.time() - t0
    (output_dir / "stdout.txt").write_text(result.stdout or "")
    (output_dir / "stderr.txt").write_text(result.stderr or "")
    (output_dir / "cmd.txt").write_text(" ".join(cmd) + "\n")

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "Unknown error")[-2000:]
        return None, runtime, f"Eval failed (code {result.returncode}): {err}"

    if not metrics_path.exists():
        return None, runtime, "Eval completed but no metrics.json found"

    metrics = json.loads(metrics_path.read_text())
    return metrics, runtime, None


def process_job(
    *,
    adapter: AdapterInfo,
    k: int,
    fewshot_seed: int,
    eval_seed: int,
    out_root: Path,
    writer: ResultsWriter,
    gpu_pool: GPUPool,
    idx: int,
    total: int,
) -> str:
    if writer.is_done(adapter.base_model_dir, k, eval_seed):
        return f"[{idx}/{total}] SKIP {adapter.base_model_dir}/math/k={k}"

    method = f"fewshot_k{k}"
    output_dir = out_root / "eval_outputs" / f"{adapter.base_model_dir}_math_{method}_s{eval_seed}"
    gpu_id = gpu_pool.acquire()
    try:
        print(f"[{idx}/{total}] GPU{gpu_id} {adapter.base_model_dir}/math/k={k}")
        metrics, runtime, error = run_single_eval(
            adapter=adapter,
            k=k,
            fewshot_seed=fewshot_seed,
            eval_seed=eval_seed,
            output_dir=output_dir,
            gpu_id=gpu_id,
        )
        metric_value = None if not metrics else metrics.get("accuracy_strict")
        writer.write(
            EvalRecord(
                timestamp=datetime.now().isoformat(),
                base_model_dir=adapter.base_model_dir,
                base_model_id=adapter.base_model_id,
                task="math",
                method=method,
                fewshot_k=k,
                fewshot_seed=fewshot_seed,
                eval_seed=eval_seed,
                adapter_dir=str(adapter.adapter_dir),
                eval_output_dir=str(output_dir),
                metric_name="accuracy_strict",
                metric_value=None if metric_value is None else float(metric_value),
                runtime_seconds=runtime,
                avg_prompt_tokens=(metrics or {}).get("avg_prompt_tokens"),
                avg_extra_prompt_tokens=(metrics or {}).get("avg_extra_prompt_tokens"),
                total_extra_prompt_tokens=(metrics or {}).get("total_extra_prompt_tokens"),
                max_prompt_tokens=(metrics or {}).get("max_prompt_tokens"),
                all_metrics=metrics,
                fix_tag="math_next_prompt_stop_truncation_v1",
                error=error,
            )
        )
        if error:
            return f"[{idx}/{total}] GPU{gpu_id} FAIL {adapter.base_model_dir}/math/k={k}: {error[:160]}"
        return (
            f"[{idx}/{total}] GPU{gpu_id} OK {adapter.base_model_dir}/math/k={k}: "
            f"accuracy_strict={float(metric_value):.4f}, runtime={runtime:.0f}s"
        )
    finally:
        gpu_pool.release(gpu_id)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun corrected math few-shot settings only.")
    parser.add_argument("--runs_root", type=str, default="runs_refactor_data_20260121")
    parser.add_argument("--out_root", type=str, default="outputs/rebuttal_exp/raw/fewshot_eval_mathfix")
    parser.add_argument("--k_values", nargs="+", type=int, default=[0, 1, 3, 5, 32])
    parser.add_argument("--fewshot_seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[4, 5, 6, 7])
    args = parser.parse_args()

    runs_root = REPO_ROOT / args.runs_root
    out_root = REPO_ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    adapters = discover_math_adapters(runs_root)
    write_json(
        out_root / "task_selection.json",
        {
            "included_tasks": ["math"],
            "excluded_tasks": {
                "csqa": "Reused from the existing few-shot study; no rerun requested.",
                "alpaca": "Excluded because the current harness evaluates those adapters on IFEval.",
            },
        },
    )
    write_json(
        out_root / "fix_metadata.json",
        {
            "fix_tag": "math_next_prompt_stop_truncation_v1",
            "description": (
                "Math few-shot rerun with vLLM stop strings for next-prompt markers and "
                "defensive truncation before answer extraction."
            ),
            "next_prompt_markers": [
                "\\n\\nBelow is an instruction",
                "\\n### Instruction:",
            ],
            "gpu_ids": args.gpu_ids,
            "fewshot_seed": args.fewshot_seed,
            "eval_seed": args.eval_seed,
        },
    )

    print(f"[Discovery] Found {len(adapters)} math seed42 adapters")
    for adapter in adapters:
        print(f"  - {adapter.base_model_dir}: {adapter.adapter_dir}")

    jobs = [(adapter, k) for adapter in adapters for k in args.k_values]
    writer = ResultsWriter(out_root)
    gpu_pool = GPUPool(args.gpu_ids)

    print(f"[Launch] {len(jobs)} corrected math jobs on GPUs {args.gpu_ids}")
    messages: list[str] = []
    with ThreadPoolExecutor(max_workers=len(args.gpu_ids)) as executor:
        future_to_job = {
            executor.submit(
                process_job,
                adapter=adapter,
                k=k,
                fewshot_seed=args.fewshot_seed,
                eval_seed=args.eval_seed,
                out_root=out_root,
                writer=writer,
                gpu_pool=gpu_pool,
                idx=idx,
                total=len(jobs),
            ): (adapter, k)
            for idx, (adapter, k) in enumerate(jobs, start=1)
        }
        for future in as_completed(future_to_job):
            msg = future.result()
            messages.append(msg)
            print(msg)

    (out_root / "run_log.txt").write_text("\n".join(messages) + "\n")
    print(f"[Done] Results: {writer.jsonl_path}")


if __name__ == "__main__":
    main()
