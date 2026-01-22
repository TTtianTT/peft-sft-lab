#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from finetune.eval.generation import (
    generate_greedy,
    load_transformers_model,
    save_json,
    strip_code_fences,
)
from finetune.utils import seed_everything

# ---------------------------
# Optional deps
# ---------------------------

HAVE_VLLM = False
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAVE_VLLM = True
except Exception:
    LLM = None
    SamplingParams = None
    LoRARequest = None

HAVE_HUMAN_EVAL = False
try:
    from human_eval.data import read_problems, write_jsonl
    HAVE_HUMAN_EVAL = True
except Exception:
    read_problems = None
    write_jsonl = None


# ---------------------------
# IO utils
# ---------------------------

def jsonl_read(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def jsonl_write(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_adapter_config(lora_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in: {lora_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_lora_has_config(lora_dir: str, config_src: Optional[str]) -> None:
    """If adapter_config.json is missing in lora_dir, copy it from config_src."""
    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    if os.path.exists(cfg_path):
        return
    if not config_src:
        raise FileNotFoundError(
            f"{cfg_path} missing.\n"
            f"Your edited dir likely only has adapter_model.*.\n"
            f"Provide --config_src <original_lora_dir> to copy adapter_config.json."
        )
    src_cfg = os.path.join(config_src, "adapter_config.json")
    if not os.path.exists(src_cfg):
        raise FileNotFoundError(f"--config_src does not contain adapter_config.json: {src_cfg}")
    os.makedirs(lora_dir, exist_ok=True)
    shutil.copy2(src_cfg, cfg_path)


def infer_max_lora_rank(adapter_cfg: Dict[str, Any]) -> int:
    """
    vLLM needs max_lora_rank at engine init.
    Try cfg["r"] or max(cfg["rank_pattern"].values()).
    """
    r = int(adapter_cfg.get("r", 0) or adapter_cfg.get("rank", 0) or 0)
    if r <= 0 and isinstance(adapter_cfg.get("rank_pattern", None), dict):
        r = max(int(v) for v in adapter_cfg["rank_pattern"].values())
    if r <= 0:
        raise ValueError("Cannot infer LoRA rank from adapter_config.json (no r / rank_pattern).")
    return r


# ---------------------------
# HumanEval evaluation
# ---------------------------

def run_humaneval_evaluate_functional_correctness(
    samples_path: str,
    k: List[int],
    n_workers: int,
    timeout: float,
    ignore_incomplete: bool,
) -> Tuple[dict, Optional[str]]:
    """
    Run HumanEval evaluation. Prefer Python API; fallback to CLI.
    """
    try:
        from human_eval.evaluation import evaluate_functional_correctness

        sig = inspect.signature(evaluate_functional_correctness)
        kwargs: Dict[str, Any] = {}
        if "k" in sig.parameters:
            kwargs["k"] = k
        if "n_workers" in sig.parameters:
            kwargs["n_workers"] = int(n_workers)
        if "timeout" in sig.parameters:
            kwargs["timeout"] = float(timeout)
        if "ignore_incomplete" in sig.parameters:
            kwargs["ignore_incomplete"] = bool(ignore_incomplete)

        res = evaluate_functional_correctness(samples_path, **kwargs)

        # human-eval commonly writes: <samples>_results.jsonl
        results_path = samples_path.replace(".jsonl", "_results.jsonl")
        if not os.path.exists(results_path):
            alt = samples_path + "_results.jsonl"
            results_path = alt if os.path.exists(alt) else results_path

        return res, results_path if os.path.exists(results_path) else None

    except Exception as e:
        print(f"[Eval][Warn] Python API failed ({type(e).__name__}: {e}). Fallback to CLI...")

    exe = shutil.which("evaluate_functional_correctness")
    if exe is None:
        raise RuntimeError(
            "Cannot find `evaluate_functional_correctness` on PATH.\n"
            "Install HumanEval:\n"
            "  git clone https://github.com/openai/human-eval\n"
            "  pip install -e human-eval\n"
        )

    cmd = [
        exe,
        samples_path,
        f"--k={','.join(map(str, k))}",
        f"--n_workers={int(n_workers)}",
        f"--timeout={float(timeout)}",
    ]
    print("[Eval][CLI] " + " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate_functional_correctness failed with code {proc.returncode}")

    results_path = samples_path.replace(".jsonl", "_results.jsonl")
    if not os.path.exists(results_path):
        alt = samples_path + "_results.jsonl"
        results_path = alt if os.path.exists(alt) else results_path

    return {}, results_path if os.path.exists(results_path) else None


def parse_results_jsonl(results_path: str) -> Dict[str, List[bool]]:
    """
    Parse results JSONL and return per-task pass/fail lists.
    """
    rows = jsonl_read(results_path)
    task_results: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        tid = r.get("task_id")
        if tid is None:
            continue
        task_results[str(tid)].append(bool(r.get("passed", False)))
    return dict(task_results)


def compute_pass_at_1_single_sample(task_results: Dict[str, List[bool]]) -> Dict[str, Any]:
    """
    For greedy mode (n=1 per task), pass@1 = (#tasks with any pass) / (#tasks).
    """
    total = len(task_results)
    correct = sum(1 for v in task_results.values() if any(v))
    return {
        "pass@1": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "num_tasks": total,
    }


def write_eval_config(
    out_dir: str,
    task: str,
    n_generations: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    total_tasks: int,
    metric_name: str,
    score: float,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    config = {
        "task": task,
        "split": "test",
        "num_fewshot": 0,
        "decoding": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n_generations": n_generations,
            "strategy": "greedy" if temperature == 0.0 else "sampling",
        },
        "total_tasks": total_tasks,
        "total_samples": total_tasks * n_generations,
        "metric": {"name": metric_name, "score": score},
        "timestamp": datetime.now().isoformat(),
    }
    if extra_meta:
        config["meta"] = extra_meta

    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "eval_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config_path


# ---------------------------
# Generation
# ---------------------------

def generate_greedy_vllm(
    base_model: str,
    prompts: List[str],
    adapter_dir: Optional[str],
    tensor_parallel_size: int,
    max_new_tokens: int,
    max_model_len: int,
    seed: int,
    stop_words: List[str],
    config_src: Optional[str],
) -> List[str]:
    if not HAVE_VLLM:
        raise RuntimeError("vLLM not available. Please `pip install vllm`.")

    enable_lora = adapter_dir is not None
    max_r = 16
    if adapter_dir:
        ensure_lora_has_config(adapter_dir, config_src)
        cfg = load_adapter_config(adapter_dir)
        max_r = max(max_r, infer_max_lora_rank(cfg))

    llm = LLM(
        model=base_model,
        max_model_len=int(max_model_len),
        tensor_parallel_size=int(tensor_parallel_size),
        enable_lora=enable_lora,
        max_lora_rank=int(max_r) if enable_lora else 16,
        seed=int(seed),
    )

    sp = SamplinqgParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(max_new_tokens),
        stop=stop_words if stop_words else None,
    )

    if adapter_dir:
        lora_req = LoRARequest("lora", 1, adapter_dir)
        outs = llm.generate(prompts, sp, lora_request=lora_req)
    else:
        outs = llm.generate(prompts, sp)

    completions: List[str] = []
    for out in outs:
        text = out.outputs[0].text if out.outputs else ""
        completions.append(strip_code_fences(text))
    return completions


# ---------------------------
# Main
# ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate HumanEval pass@1 with temperature=0 (greedy).")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_samples", type=int, default=None, help="Max HumanEval tasks (None = all 164).")
    p.add_argument("--max_new_tokens", type=int, default=256)

    # Transformers path (optional)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)

    # Evaluation runtime
    p.add_argument("--timeout_s", type=float, default=3.0)
    p.add_argument("--eval_n_workers", type=int, default=4)

    # vLLM
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument(
        "--stop_words",
        type=str,
        nargs="*",
        default=["\ndef ", "\nclass ", "\nif __name__", "\n#", "\nprint"],
        help="Stop sequences for code generation.",
    )

    # LoRA config补齐（给 edited dir 只有 adapter_model.* 的场景）
    p.add_argument("--config_src", type=str, default=None)

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    if not HAVE_HUMAN_EVAL:
        raise RuntimeError(
            "human_eval not available. Install:\n"
            "  git clone https://github.com/openai/human-eval\n"
            "  pip install -e human-eval\n"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load problems
    problems = read_problems()
    task_ids = sorted(problems.keys())
    if args.max_samples is not None and int(args.max_samples) > 0:
        task_ids = task_ids[: int(args.max_samples)]

    prompts = [problems[tid]["prompt"] for tid in task_ids]
    print(f"[Data] Loaded {len(prompts)} HumanEval tasks.")

    # 2) Generate (temperature=0, n=1)
    if args.use_vllm:
        completions = generate_greedy_vllm(
            base_model=args.base_model,
            prompts=prompts,
            adapter_dir=args.adapter_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            seed=args.seed,
            stop_words=args.stop_words,
            config_src=args.config_src,
        )
    else:
        loaded = load_transformers_model(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            dtype=args.dtype,
            device_map="auto",
        )
        completions = []
        for prompt in prompts:
            comp_raw = generate_greedy(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )
            completions.append(strip_code_fences(comp_raw))

    # 3) Write samples JSONL (HumanEval format)
    adapter_tag = "lora" if args.adapter_dir else "base"
    samples_path = str(out_dir / f"samples_{adapter_tag}.jsonl")
    samples = [{"task_id": tid, "completion": c} for tid, c in zip(task_ids, completions)]

    try:
        write_jsonl(samples_path, samples)
    except Exception:
        jsonl_write(samples_path, samples)
    print(f"[Gen] Wrote {len(samples)} samples to: {samples_path}")

    # (Optional) store prompts+completions for auditing
    gens_path = str(out_dir / f"generations_{adapter_tag}.jsonl")
    jsonl_write(
        gens_path,
        [{"task_id": tid, "prompt": problems[tid]["prompt"], "completion": c}
         for tid, c in zip(task_ids, completions)],
    )

    # 4) Run evaluate_functional_correctness (temperature=0 => greedy pass@1)
    raw_res, results_path = run_humaneval_evaluate_functional_correctness(
        samples_path=samples_path,
        k=[1],
        n_workers=args.eval_n_workers,
        timeout=args.timeout_s,
        ignore_incomplete=bool(args.max_samples is not None and int(args.max_samples) > 0),
    )

    # 5) Parse results and compute pass@1
    metrics: Dict[str, Any]
    if results_path and os.path.exists(results_path):
        task_results = parse_results_jsonl(results_path)
        metrics = compute_pass_at_1_single_sample(task_results)
    else:
        # Fallback: if API returned dict with pass@1
        if isinstance(raw_res, dict) and "pass@1" in raw_res:
            metrics = {"pass@1": float(raw_res["pass@1"]), "correct": None, "total": len(task_ids), "num_tasks": len(task_ids)}
        else:
            metrics = {"pass@1": None, "correct": None, "total": len(task_ids), "num_tasks": len(task_ids)}

    metrics.update(
        {
            "samples_path": samples_path,
            "results_path": results_path,
            "generations_path": gens_path,
            "n_generations": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "timeout_s": args.timeout_s,
            "eval_n_workers": args.eval_n_workers,
            "base_model": args.base_model,
            "adapter_dir": args.adapter_dir,
            "use_vllm": bool(args.use_vllm),
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "stop_words": args.stop_words,
            "seed": args.seed,
        }
    )

    # 6) Write outputs.jsonl (merge pass/fail if possible)
    outputs_path = str(out_dir / f"outputs_{adapter_tag}.jsonl")
    merged_rows: List[dict] = []

    passed_map: Dict[str, dict] = {}
    if results_path and os.path.exists(results_path):
        for r in jsonl_read(results_path):
            tid = str(r.get("task_id", ""))
            # greedy mode: one row per task_id; keep first
            if tid and tid not in passed_map:
                passed_map[tid] = r

    for tid, comp in zip(task_ids, completions):
        rr = passed_map.get(tid, {})
        merged_rows.append(
            {
                "task_id": tid,
                "passed": bool(rr.get("passed", False)) if rr else None,
                "prompt": problems[tid]["prompt"],
                "completion": comp,
                "result": rr.get("result", None),
                "error": rr.get("error", None),
            }
        )
    jsonl_write(outputs_path, merged_rows)

    # 7) Save metrics + eval_config
    save_json(out_dir / "metrics.json", metrics)

    config_path = write_eval_config(
        out_dir=str(out_dir),
        task="humaneval",
        n_generations=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(args.max_new_tokens),
        total_tasks=len(task_ids),
        metric_name="pass@1",
        score=float(metrics["pass@1"] or 0.0),
        extra_meta={
            "base_model": args.base_model,
            "adapter_dir": args.adapter_dir,
            "use_vllm": bool(args.use_vllm),
        },
    )

    print(f"[Done] pass@1={metrics.get('pass@1')}, correct={metrics.get('correct')}, total={metrics.get('total')}")
    print(f"[Done] metrics: {out_dir / 'metrics.json'}")
    print(f"[Done] outputs: {outputs_path}")
    print(f"[Done] eval_config: {config_path}")


if __name__ == "__main__":
    main()
