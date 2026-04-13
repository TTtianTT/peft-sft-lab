#!/usr/bin/env python3
"""
P0-1: Evaluate control-edited adapters via lm-eval-harness.

Finds all completed control edits, groups by model/task, and runs evaluation
in parallel across available GPUs. Writes results to eval_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTROL_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "p0_1_random_mechanism" / "edited"
OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "p0_1_random_mechanism"
PYTHON = sys.executable

TASK_DIR_TO_LM_EVAL = {"math": "gsm8k", "alpaca": "ifeval", "csqa": "commonsense_qa"}
BASE_MODELS = {
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
}
TASK_CONFIGS = {
    "math": {"num_fewshot": 5, "gen_kwargs": "temperature=0,top_p=1", "gpu_mem": 0.95},
    "alpaca": {"num_fewshot": None, "gen_kwargs": "max_gen_toks=2048,temperature=0,top_p=1", "gpu_mem": 0.95},
    "csqa": {"num_fewshot": 0, "gen_kwargs": None, "gpu_mem": 0.85},
}


def find_pending_evals() -> list:
    """Find all control edits that need evaluation."""
    pending = []
    for meta_path in sorted(CONTROL_ROOT.rglob("spectral_edit_meta.json")):
        # Check if eval already done
        eval_dir = meta_path.parent / "eval"
        result_file = meta_path.parent / "eval_result.json"
        if result_file.exists():
            continue

        # Parse path
        parts = meta_path.relative_to(CONTROL_ROOT).parts
        if len(parts) < 4:
            continue
        model_dir, task, method, seed_dir = parts[0], parts[1], parts[2], parts[3]
        seed = int(seed_dir.replace("seed", ""))

        base_model_id = BASE_MODELS.get(model_dir)
        if not base_model_id:
            continue

        pending.append({
            "model_dir": model_dir,
            "task": task,
            "method": method,
            "seed": seed,
            "base_model_id": base_model_id,
            "adapter_dir": str(meta_path.parent),
        })

    return pending


def run_single_eval(item: dict, gpu_id: int) -> dict:
    """Evaluate a single edited adapter."""
    task = item["task"]
    base_model_id = item["base_model_id"]
    adapter_dir = Path(item["adapter_dir"])
    eval_dir = adapter_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = eval_dir / "merged_model"

    lm_task = TASK_DIR_TO_LM_EVAL[task]
    tcfg = TASK_CONFIGS[task]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    # Step 1: Merge adapter
    merge_script = f"""
import torch, shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "{base_model_id}"
adapter_dir = "{str(adapter_dir)}"
merged_dir = "{str(merged_dir)}"

tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()
model.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
del model
torch.cuda.empty_cache()
"""
    merge_result = subprocess.run(
        [PYTHON, "-c", merge_script],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if merge_result.returncode != 0:
        return {**item, "error": f"merge failed: {merge_result.stderr[-300:]}", "metric_value": None}

    # Step 2: Run lm-eval
    cmd = [
        PYTHON, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", f"pretrained={merged_dir},tensor_parallel_size=1,gpu_memory_utilization={tcfg['gpu_mem']},dtype=float16",
        "--tasks", lm_task,
        "--batch_size", "auto",
        "--output_path", str(eval_dir / "lm_eval_results"),
    ]
    if tcfg["num_fewshot"] is not None:
        cmd += ["--num_fewshot", str(tcfg["num_fewshot"])]
    if tcfg["gen_kwargs"]:
        cmd += ["--gen_kwargs", tcfg["gen_kwargs"]]

    eval_result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=1800,
    )

    # Clean up merged model
    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)

    if eval_result.returncode != 0:
        return {**item, "error": f"lm-eval failed: {eval_result.stderr[-300:]}", "metric_value": None}

    # Parse result
    metric_keys = {"gsm8k": "exact_match,strict-match", "ifeval": "prompt_level_strict_acc,none", "commonsense_qa": "acc,none"}
    primary_key = metric_keys.get(lm_task, "acc,none")
    metric_value = None

    for root, dirs, files in os.walk(eval_dir / "lm_eval_results"):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                try:
                    with open(os.path.join(root, f)) as fp:
                        data = json.load(fp)
                    results = data.get("results", {})
                    task_results = results.get(lm_task, {})
                    if primary_key in task_results:
                        metric_value = float(task_results[primary_key])
                    else:
                        for k, v in task_results.items():
                            if "acc" in k or "exact_match" in k:
                                metric_value = float(v)
                                break
                except Exception:
                    continue

    result = {**item, "metric_value": metric_value, "error": None}

    # Save per-adapter result
    with open(adapter_dir / "eval_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--max_parallel", type=int, default=4,
                        help="Max parallel evals (each uses one GPU)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--filter_task", type=str, default=None,
                        help="Only eval this task (math/alpaca/csqa)")
    parser.add_argument("--filter_method", type=str, default=None,
                        help="Only eval this control method")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    pending = find_pending_evals()

    if args.filter_task:
        pending = [p for p in pending if p["task"] == args.filter_task]
    if args.filter_method:
        pending = [p for p in pending if p["method"] == args.filter_method]

    print(f"Found {len(pending)} pending evaluations")
    print(f"Using GPUs: {gpu_ids}")
    print(f"Max parallel: {min(args.max_parallel, len(gpu_ids))}")

    if args.dry_run:
        for p in pending:
            print(f"  {p['model_dir']}/{p['task']}/{p['method']}/seed{p['seed']}")
        return

    if not pending:
        print("Nothing to evaluate!")
        return

    # Process pool: assign GPUs round-robin
    max_workers = min(args.max_parallel, len(gpu_ids), len(pending))
    results_path = OUT_ROOT / "control_eval_results.jsonl"

    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, item in enumerate(pending):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            future = executor.submit(run_single_eval, item, gpu_id)
            futures[future] = item

        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                with open(results_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

                if result.get("metric_value") is not None:
                    completed += 1
                    print(f"  [OK] {item['model_dir']}/{item['task']}/{item['method']}/seed{item['seed']}: "
                          f"{result['metric_value']:.4f}")
                else:
                    failed += 1
                    print(f"  [FAIL] {item['model_dir']}/{item['task']}/{item['method']}/seed{item['seed']}: "
                          f"{result.get('error', 'unknown')[:100]}")
            except Exception as e:
                failed += 1
                print(f"  [ERROR] {item['model_dir']}/{item['task']}/{item['method']}/seed{item['seed']}: {e}")

    print(f"\nDone: {completed} succeeded, {failed} failed out of {len(pending)} total")


if __name__ == "__main__":
    main()
