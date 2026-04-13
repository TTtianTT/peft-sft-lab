#!/usr/bin/env python3
"""
Run the rebuttal few-shot operating-point comparison.

This script:
1. Discovers the seed42 rebuttal adapters for selected tasks.
2. Runs a prompt-length/context sanity check for fixed-k few-shot prompting.
3. Computes one-time spectral-edit cost proxies from existing random_index metadata.
4. Launches resumable few-shot eval jobs on the unedited adapters.

The spectral baselines themselves are reused from existing rebuttal outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.eval.eval_csqa import (  # noqa: E402
    _build_csqa_instruction,
    _build_csqa_prompt,
    _build_fewshot_example_text as build_csqa_fewshot_example_text,
    _build_fewshot_prefix as build_csqa_fewshot_prefix,
)
from finetune.eval.eval_gsm8k import (  # noqa: E402
    _build_fewshot_example_text as build_math_fewshot_example_text,
    _build_fewshot_prefix as build_math_fewshot_prefix,
    _build_prompt_gsm8k_metamath_style,
)
from finetune.eval.fewshot import (  # noqa: E402
    PromptStatsAccumulator,
    load_dataset_split,
    load_tokenizer_for_prompt_stats,
    select_fixed_exemplars,
)
from finetune.spectral_edit.calib import (  # noqa: E402
    build_calib_formatter,
    sample_calibration_examples,
)


BASE_MODEL_DIR_TO_ID = {
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
}

TASK_TO_EVAL_MODULE = {
    "math": "finetune.eval.eval_gsm8k",
    "csqa": "finetune.eval.eval_csqa",
}

TASK_TO_METRIC = {
    "math": "accuracy_strict",
    "csqa": "accuracy",
}

TASK_TO_MAX_NEW_TOKENS = {
    "math": 256,
    "csqa": 8,
}

EXCLUDED_TASKS = {
    "alpaca": (
        "Excluded: the current alpaca adapter is evaluated on IFEval, and prepending fixed "
        "few-shot exemplars would change the benchmark prompt format in a non-standard way."
    ),
}


@dataclass
class AdapterInfo:
    base_model_dir: str
    base_model_id: str
    task: str
    adapter_dir: Path
    profile: str
    rank: str
    seed: str


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

    def _key(self, base_model_dir: str, task: str, k: int, eval_seed: int) -> str:
        return f"{base_model_dir}|{task}|{k}|{eval_seed}"

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
                        rec["task"],
                        int(rec["fewshot_k"]),
                        int(rec["eval_seed"]),
                    )
                )

    def is_done(self, base_model_dir: str, task: str, k: int, eval_seed: int) -> bool:
        return self._key(base_model_dir, task, k, eval_seed) in self.existing_keys

    def write(self, record: EvalRecord) -> None:
        with self._lock:
            key = self._key(record.base_model_dir, record.task, record.fewshot_k, record.eval_seed)
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


def discover_adapters(runs_root: Path, tasks: list[str]) -> list[AdapterInfo]:
    adapters: list[AdapterInfo] = []
    for base_model_dir, base_model_id in BASE_MODEL_DIR_TO_ID.items():
        for task in tasks:
            task_root = runs_root / base_model_dir / task / "lora"
            if not task_root.exists():
                continue
            for root, _, files in os.walk(task_root):
                root_path = Path(root)
                if "checkpoint-" in str(root_path):
                    continue
                if "adapter_config.json" not in files:
                    continue
                if "adapter_model.safetensors" not in files and "adapter_model.bin" not in files:
                    continue
                try:
                    rel = root_path.relative_to(task_root)
                except ValueError:
                    continue
                parts = rel.parts
                if len(parts) < 3:
                    continue
                profile = parts[0].replace("profile-", "") if parts[0].startswith("profile-") else parts[0]
                rank = parts[1].replace("rank-", "") if parts[1].startswith("rank-") else parts[1]
                seed = parts[2].replace("seed", "") if parts[2].startswith("seed") else parts[2]
                if seed != "42":
                    continue
                adapters.append(
                    AdapterInfo(
                        base_model_dir=base_model_dir,
                        base_model_id=base_model_id,
                        task=task,
                        adapter_dir=root_path,
                        profile=profile,
                        rank=rank,
                        seed=seed,
                    )
                )
    adapters.sort(key=lambda x: (x.base_model_dir, x.task))
    return adapters


def _context_limit(tokenizer) -> Optional[int]:
    max_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_len, int) and 0 < max_len < 10**9:
        return max_len
    return None


def _math_eval_prompts_for_k(k: int, fewshot_seed: int):
    eval_ds = load_dataset_split("gsm8k", dataset_config="main", split="test")
    source_ds = load_dataset_split("gsm8k", dataset_config="main", split="train")
    fewshot_examples = select_fixed_exemplars(source_ds, k=k, seed=fewshot_seed)
    exemplars = [build_math_fewshot_example_text(ex) for ex in fewshot_examples]
    prefix = build_math_fewshot_prefix(exemplars)
    for ex in eval_ds:
        base_prompt = _build_prompt_gsm8k_metamath_style(str(ex.get("question", "")).strip())
        yield base_prompt, prefix + base_prompt


def _csqa_eval_prompts_for_k(k: int, fewshot_seed: int):
    eval_ds = load_dataset_split("tau/commonsense_qa", split="validation")
    source_ds = load_dataset_split("tau/commonsense_qa", split="train")
    fewshot_examples = select_fixed_exemplars(source_ds, k=k, seed=fewshot_seed)
    exemplars = [build_csqa_fewshot_example_text(ex) for ex in fewshot_examples]
    prefix = build_csqa_fewshot_prefix(exemplars)
    for ex in eval_ds:
        _, _, instruction = _build_csqa_instruction(ex)
        base_prompt = _build_csqa_prompt(instruction)
        yield base_prompt, prefix + base_prompt


def compute_context_summary(
    *,
    base_model_id: str,
    task: str,
    k_values: list[int],
    fewshot_seed: int,
) -> dict[str, Any]:
    tokenizer = load_tokenizer_for_prompt_stats(base_model=base_model_id, adapter_dir=None)
    limit = _context_limit(tokenizer)
    per_k: dict[str, Any] = {}

    for k in k_values:
        acc = PromptStatsAccumulator(tokenizer)
        prompt_iter = _math_eval_prompts_for_k(k, fewshot_seed) if task == "math" else _csqa_eval_prompts_for_k(k, fewshot_seed)
        for base_prompt, prompt in prompt_iter:
            acc.add(base_prompt=base_prompt, prompt=prompt)
        summary = acc.summary()
        max_prompt_tokens = summary["max_prompt_tokens"] or 0
        per_k[str(k)] = {
            **summary,
            "context_limit_tokens": limit,
            "feasible": (limit is None) or (max_prompt_tokens < limit),
            "large_context_stress_test": (k == 32),
        }

    return {
        "base_model_id": base_model_id,
        "task": task,
        "fewshot_seed": fewshot_seed,
        "per_k": per_k,
    }


def compute_spectral_cost_summary(
    *,
    base_model_id: str,
    spectral_meta_path: Path,
) -> dict[str, Any]:
    payload = json.loads(spectral_meta_path.read_text())
    meta = payload["meta"]
    sigma_stats = payload["sigma_stats"]

    tokenizer = load_tokenizer_for_prompt_stats(base_model=base_model_id, adapter_dir=str(spectral_meta_path.parent))
    formatter, _ = build_calib_formatter(meta["calib_dataset"], meta["calib_text_fields"])
    ds = load_dataset_split(
        meta["calib_dataset"],
        dataset_config=meta["calib_config"],
        split=meta["calib_split"],
    )
    examples = sample_calibration_examples(
        ds,
        int(meta["calib_samples"]),
        bool(meta["calib_shuffle"]),
        int(meta["calib_seed"]),
        int(meta["calib_start"]),
    )

    total_tokens = 0
    for ex in examples:
        prompt, answer = formatter(ex)
        prompt = "" if prompt is None else str(prompt)
        answer = "" if answer is None else str(answer)
        full = prompt
        if answer:
            if prompt and (not prompt[-1].isspace()):
                full = prompt + " " + answer
            else:
                full = prompt + answer
        if tokenizer.eos_token:
            full = full + tokenizer.eos_token
        total_tokens += len(tokenizer(full, add_special_tokens=False).input_ids)

    ranks = sorted({int(stats["r"]) for stats in sigma_stats.values()})
    rank = ranks[0] if ranks else None
    module_count = len(sigma_stats)
    batch_size = int(meta["calib_batch_size"])
    steps = math.ceil(len(examples) / max(1, batch_size))

    return {
        "spectral_meta_path": str(spectral_meta_path),
        "calib_dataset": meta["calib_dataset"],
        "calib_config": meta["calib_config"],
        "calib_split": meta["calib_split"],
        "calib_samples": int(meta["calib_samples"]),
        "calib_samples_used": int(meta["calib_samples_used"]),
        "calib_batch_size": batch_size,
        "calibration_total_tokens": total_tokens,
        "calibration_avg_tokens_per_example": (total_tokens / len(examples)) if examples else None,
        "forward_passes": steps,
        "backward_passes": steps,
        "forward_backward_token_passes": total_tokens * 2,
        "edited_module_count": module_count,
        "rank": rank,
        "edited_scalars": (module_count * rank) if rank is not None else None,
        "per_query_extra_prompt_tokens": 0,
    }


def parse_metric(metrics: dict[str, Any], task: str) -> Optional[float]:
    key = TASK_TO_METRIC[task]
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


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
        TASK_TO_EVAL_MODULE[adapter.task],
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
        str(TASK_TO_MAX_NEW_TOKENS[adapter.task]),
        "--use_vllm",
        "--tensor_parallel_size",
        "1",
    ]
    if adapter.task == "csqa":
        cmd.extend(["--split", "validation"])

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
    method = f"fewshot_k{k}"
    if writer.is_done(adapter.base_model_dir, adapter.task, k, eval_seed):
        return f"[{idx}/{total}] SKIP {adapter.base_model_dir}/{adapter.task}/{method}"

    output_dir = out_root / "eval_outputs" / f"{adapter.base_model_dir}_{adapter.task}_{method}_s{eval_seed}"
    gpu_id = gpu_pool.acquire()
    try:
        print(f"[{idx}/{total}] GPU{gpu_id} {adapter.base_model_dir}/{adapter.task}/{method}")
        metrics, runtime, error = run_single_eval(
            adapter=adapter,
            k=k,
            fewshot_seed=fewshot_seed,
            eval_seed=eval_seed,
            output_dir=output_dir,
            gpu_id=gpu_id,
        )

        metric_value = parse_metric(metrics, adapter.task) if metrics else None
        writer.write(
            EvalRecord(
                timestamp=datetime.now().isoformat(),
                base_model_dir=adapter.base_model_dir,
                base_model_id=adapter.base_model_id,
                task=adapter.task,
                method=method,
                fewshot_k=k,
                fewshot_seed=fewshot_seed,
                eval_seed=eval_seed,
                adapter_dir=str(adapter.adapter_dir),
                eval_output_dir=str(output_dir),
                metric_name=TASK_TO_METRIC[adapter.task],
                metric_value=metric_value,
                runtime_seconds=runtime,
                avg_prompt_tokens=(metrics or {}).get("avg_prompt_tokens"),
                avg_extra_prompt_tokens=(metrics or {}).get("avg_extra_prompt_tokens"),
                total_extra_prompt_tokens=(metrics or {}).get("total_extra_prompt_tokens"),
                max_prompt_tokens=(metrics or {}).get("max_prompt_tokens"),
                all_metrics=metrics,
                error=error,
            )
        )

        if error:
            return f"[{idx}/{total}] GPU{gpu_id} FAIL {adapter.base_model_dir}/{adapter.task}/{method}: {error[:160]}"
        val_str = "N/A" if metric_value is None else f"{metric_value:.4f}"
        extra_str = (metrics or {}).get("avg_extra_prompt_tokens")
        extra_desc = "N/A" if extra_str is None else f"{extra_str:.1f}"
        rt_str = "N/A" if runtime is None else f"{runtime:.0f}s"
        return (
            f"[{idx}/{total}] GPU{gpu_id} OK {adapter.base_model_dir}/{adapter.task}/{method}: "
            f"{TASK_TO_METRIC[adapter.task]}={val_str}, extra_toks={extra_desc}, runtime={rt_str}"
        )
    finally:
        gpu_pool.release(gpu_id)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run few-shot operating-point evals for the rebuttal.")
    parser.add_argument("--runs_root", type=str, default="runs_refactor_data_20260121")
    parser.add_argument("--out_root", type=str, default="outputs/rebuttal_exp/raw/fewshot_eval")
    parser.add_argument("--tasks", nargs="+", default=["math", "csqa", "alpaca"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 3, 5, 32])
    parser.add_argument("--fewshot_seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[4, 5, 6, 7])
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selected_tasks = [task for task in args.tasks if task not in EXCLUDED_TASKS]
    task_selection = {
        "included_tasks": selected_tasks,
        "excluded_tasks": {task: EXCLUDED_TASKS[task] for task in args.tasks if task in EXCLUDED_TASKS},
    }
    write_json(out_root / "task_selection.json", task_selection)

    adapters = discover_adapters(runs_root, selected_tasks)
    print(f"[Discovery] Found {len(adapters)} seed42 adapters")
    for adapter in adapters:
        print(f"  - {adapter.base_model_dir}/{adapter.task}: {adapter.adapter_dir}")

    context_summary: dict[str, Any] = {"fewshot_seed": args.fewshot_seed, "settings": {}}
    spectral_costs: dict[str, Any] = {}

    for adapter in adapters:
        key = f"{adapter.base_model_dir}/{adapter.task}"
        if key not in context_summary["settings"]:
            context_summary["settings"][key] = compute_context_summary(
                base_model_id=adapter.base_model_id,
                task=adapter.task,
                k_values=[0] + sorted(set(args.k_values)),
                fewshot_seed=args.fewshot_seed,
            )

        meta_path = (
            REPO_ROOT
            / "outputs/rebuttal_exp/raw/multiseed/edited_adapters"
            / f"{adapter.base_model_dir}_{adapter.task}_{adapter.profile}_{adapter.rank}_{adapter.seed}"
            / "random_index"
            / "repeat_00_seed42"
            / "spectral_edit_meta.json"
        )
        if meta_path.exists() and key not in spectral_costs:
            spectral_costs[key] = compute_spectral_cost_summary(
                base_model_id=adapter.base_model_id,
                spectral_meta_path=meta_path,
            )

    write_json(out_root / "context_summary.json", context_summary)
    write_json(out_root / "spectral_costs.json", spectral_costs)

    jobs: list[tuple[AdapterInfo, int]] = []
    for adapter in adapters:
        key = f"{adapter.base_model_dir}/{adapter.task}"
        per_k = context_summary["settings"][key]["per_k"]
        for k in args.k_values:
            if not per_k[str(k)]["feasible"]:
                print(f"[Skip] {key} k={k} exceeds context limit; not launching.")
                continue
            jobs.append((adapter, k))

    writer = ResultsWriter(out_root)
    gpu_pool = GPUPool(args.gpu_ids)

    print(f"[Launch] {len(jobs)} few-shot eval jobs on GPUs {args.gpu_ids}")
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
    print(f"[Done] Context: {out_root / 'context_summary.json'}")
    print(f"[Done] Spectral costs: {out_root / 'spectral_costs.json'}")


if __name__ == "__main__":
    main()
