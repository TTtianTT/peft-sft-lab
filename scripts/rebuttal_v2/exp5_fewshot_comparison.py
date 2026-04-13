#!/usr/bin/env python3
"""
Experiment 5: Few-Shot Baseline Comparison

Compare spectral-edited 0-shot performance against k-shot prompting baselines
to support the "solidified few-shot" narrative.

For each (model, task):
  For k in {0, 1, 3, 5}:
    Run lm-eval with k-shot on the BASELINE adapter (no spectral edit)

Compare: k-shot baseline vs spectral-edited 0-shot
Note: GSM8K already uses 5-shot, CSQA uses 0-shot, IFEval uses 0-shot.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "exp5_fewshot"
PYTHON = sys.executable

SETTINGS = [
    {
        "model_dir": "meta-llama-Llama-3.1-8B",
        "base_model_id": "meta-llama/Llama-3.1-8B",
        "task": "csqa",
        "lora_path": str(REPO_ROOT / "runs/meta-llama-Llama-3.1-8B/csqa/lora/profile-paper_csqa_3ep/rank-16/seed42"),
        "lm_task": "commonsense_qa",
        "default_fewshot": 0,
        "gpu_mem": 0.85,
    },
    {
        "model_dir": "meta-llama-Llama-3.1-8B",
        "base_model_id": "meta-llama/Llama-3.1-8B",
        "task": "math",
        "lora_path": str(REPO_ROOT / "runs/meta-llama-Llama-3.1-8B/metamath/lora/profile-paper_math_ift_3ep/rank-16/seed42"),
        "lm_task": "gsm8k",
        "default_fewshot": 5,
        "gen_kwargs": "temperature=0,top_p=1",
        "gpu_mem": 0.95,
    },
    {
        "model_dir": "Qwen-Qwen3-8B",
        "base_model_id": "Qwen/Qwen3-8B",
        "task": "math",
        "lora_path": str(REPO_ROOT / "runs/Qwen-Qwen3-8B/metamath/lora/profile-paper_math_ift_3ep/rank-16/seed42"),
        "lm_task": "gsm8k",
        "default_fewshot": 5,
        "gen_kwargs": "temperature=0,top_p=1",
        "gpu_mem": 0.95,
    },
    {
        "model_dir": "Qwen-Qwen3-8B",
        "base_model_id": "Qwen/Qwen3-8B",
        "task": "alpaca",
        "lora_path": str(REPO_ROOT / "runs/Qwen-Qwen3-8B/alpaca/lora/profile-paper_alpaca_3ep/rank-16/seed42"),
        "lm_task": "ifeval",
        "default_fewshot": None,
        "gen_kwargs": "max_gen_toks=2048,temperature=0,top_p=1",
        "gpu_mem": 0.95,
    },
]

FEWSHOT_VALUES = [0, 1, 3, 5]


def run_single_eval(item: dict, gpu_id: int) -> dict:
    """Run lm-eval with specified fewshot for a merged adapter."""
    base_model_id = item["base_model_id"]
    lora_path = item["lora_path"]
    lm_task = item["lm_task"]
    k_shot = item["k_shot"]
    gpu_mem = item.get("gpu_mem", 0.90)
    gen_kwargs = item.get("gen_kwargs")

    eval_dir = OUT_ROOT / "evals" / item["model_dir"] / item["task"] / f"fewshot_{k_shot}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = eval_dir / "merged_model"
    result_file = eval_dir / "eval_result.json"

    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    # Merge adapter
    merge_script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("{base_model_id}", use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("{base_model_id}", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, "{lora_path}")
model = model.merge_and_unload()
model.save_pretrained("{merged_dir}")
tok.save_pretrained("{merged_dir}")
del model; torch.cuda.empty_cache()
"""
    r = subprocess.run([PYTHON, "-c", merge_script], capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        return {**item, "error": f"merge failed: {r.stderr[-300:]}", "metric_value": None}

    cmd = [
        PYTHON, "-m", "lm_eval", "--model", "vllm",
        "--model_args", f"pretrained={merged_dir},tensor_parallel_size=1,gpu_memory_utilization={gpu_mem},dtype=float16",
        "--tasks", lm_task, "--batch_size", "auto",
        "--output_path", str(eval_dir / "lm_eval_results"),
    ]
    if k_shot is not None:
        cmd += ["--num_fewshot", str(k_shot)]
    if gen_kwargs:
        cmd += ["--gen_kwargs", gen_kwargs]

    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    import shutil
    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)

    if r.returncode != 0:
        return {**item, "error": f"lm-eval failed: {r.stderr[-300:]}", "metric_value": None}

    metric_keys = {"gsm8k": "exact_match,strict-match", "ifeval": "prompt_level_strict_acc,none",
                   "commonsense_qa": "acc,none"}
    primary_key = metric_keys.get(lm_task, "acc,none")
    metric_value = None
    for root, dirs, files in os.walk(eval_dir / "lm_eval_results"):
        for fname in files:
            if fname.startswith("results_") and fname.endswith(".json"):
                try:
                    with open(os.path.join(root, fname)) as fp:
                        data = json.load(fp)
                    task_results = data.get("results", {}).get(lm_task, {})
                    if primary_key in task_results:
                        metric_value = float(task_results[primary_key])
                    else:
                        for mk, mv in task_results.items():
                            if "acc" in mk or "exact_match" in mk:
                                metric_value = float(mv)
                                break
                except Exception:
                    continue

    result = {**item, "metric_value": metric_value, "error": None}
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    return result


def phase1_evaluate(gpus: list[int], max_parallel: int):
    """Run all fewshot evaluations."""
    print("=" * 80)
    print("Evaluating few-shot baselines")
    print("=" * 80)

    pending = []
    for setting in SETTINGS:
        for k in FEWSHOT_VALUES:
            # Skip if IFEval (doesn't support fewshot well) and k > 0
            if setting["lm_task"] == "ifeval" and k > 0:
                continue
            result_file = OUT_ROOT / "evals" / setting["model_dir"] / setting["task"] / f"fewshot_{k}" / "eval_result.json"
            if result_file.exists():
                continue
            pending.append({
                **setting,
                "k_shot": k,
            })

    print(f"Found {len(pending)} pending evaluations")
    if not pending:
        return

    max_workers = min(max_parallel, len(gpus), len(pending))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, item in enumerate(pending):
            gpu_id = gpus[i % len(gpus)]
            futures[executor.submit(run_single_eval, item, gpu_id)] = item

        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                status = "OK" if result.get("metric_value") is not None else "FAIL"
                val = f"{result['metric_value']:.4f}" if result.get("metric_value") is not None else result.get("error", "")[:60]
                print(f"  [{status}] {item['model_dir']}/{item['task']}/k={item['k_shot']}: {val}")
            except Exception as e:
                print(f"  [ERROR] {item['model_dir']}/{item['task']}/k={item['k_shot']}: {e}")


def phase2_analyze():
    """Analyze few-shot comparison results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv

    print("=" * 80)
    print("Analyzing few-shot comparison")
    print("=" * 80)

    # Load fewshot results
    fewshot_data = defaultdict(dict)
    for setting in SETTINGS:
        for k in FEWSHOT_VALUES:
            result_file = OUT_ROOT / "evals" / setting["model_dir"] / setting["task"] / f"fewshot_{k}" / "eval_result.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                if data.get("metric_value") is not None:
                    skey = f"{setting['model_dir']}/{setting['task']}"
                    fewshot_data[skey][k] = data["metric_value"]

    # Load spectral editing results from prior data
    baseline_path = REPO_ROOT / "outputs" / "rebuttal_exp" / "raw" / "multiseed_eval" / "eval_results.csv"
    prior_eval = defaultdict(lambda: defaultdict(list))
    if baseline_path.exists():
        with open(baseline_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("metric_value"):
                    skey = f"{row.get('base_model_dir', '')}/{row.get('task', '')}"
                    prior_eval[skey][row.get("method", "")].append(float(row["metric_value"]))

    plot_dir = OUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = OUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print(f"\n{'Setting':<40} {'k-shot':>8} {'Metric':>10} {'vs spectral':>12}")
    print("-" * 75)

    for skey, kdata in sorted(fewshot_data.items()):
        spectral_methods = {}
        for method in ["random_index", "smooth_abs", "grad_direction"]:
            vals = prior_eval.get(skey, {}).get(method, [])
            if vals:
                spectral_methods[method] = np.mean(vals)

        best_spectral = max(spectral_methods.values()) if spectral_methods else None
        best_spectral_name = max(spectral_methods, key=spectral_methods.get) if spectral_methods else "N/A"

        for k in sorted(kdata.keys()):
            val = kdata[k]
            diff = ""
            if best_spectral is not None:
                diff = f"{(val - best_spectral) / max(best_spectral, 1e-8) * 100:+.2f}%"
            print(f"{skey:<40} {k:>8} {val:>10.4f} {diff:>12}")
            all_results.append({
                "setting": skey, "k_shot": k, "metric": val,
                "best_spectral": best_spectral, "best_spectral_method": best_spectral_name,
            })

        if best_spectral is not None:
            print(f"{skey:<40} {'spectral':>8} {best_spectral:>10.4f} {'(best)':>12} [{best_spectral_name}]")
        print()

    # Plot: bar chart for each setting
    for skey, kdata in sorted(fewshot_data.items()):
        spectral_methods = {}
        for method in ["random_index", "smooth_abs", "grad_direction"]:
            vals = prior_eval.get(skey, {}).get(method, [])
            if vals:
                spectral_methods[method] = np.mean(vals)

        baseline_val = np.mean(prior_eval.get(skey, {}).get("baseline", [])) if prior_eval.get(skey, {}).get("baseline") else None

        fig, ax = plt.subplots(figsize=(8, 4))
        labels = []
        vals = []
        colors = []

        for k in sorted(kdata.keys()):
            labels.append(f"{k}-shot")
            vals.append(kdata[k])
            colors.append("#1f77b4")

        for method, val in spectral_methods.items():
            labels.append(f"spectral\n({method})")
            vals.append(val)
            colors.append("#2ca02c")

        bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("Metric", fontsize=11)
        model_short = skey.split("-")[-1].split("/")[0]
        task_short = skey.split("/")[1]
        ax.set_title(f"{model_short}/{task_short} — Few-Shot vs Spectral Editing", fontsize=11)
        if baseline_val:
            ax.axhline(baseline_val, color="red", ls="--", label=f"Baseline ({baseline_val:.3f})")
            ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        safe_name = skey.replace("/", "_")
        fig.savefig(plot_dir / f"exp5_fewshot_{safe_name}.pdf", dpi=150)
        plt.close(fig)

    with open(analysis_dir / "fewshot_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[Saved] Analysis and plots")


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Few-Shot Baseline Comparison")
    parser.add_argument("--phase", type=str, default="all", choices=["eval", "analyze", "all"])
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_parallel", type=int, default=6)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    if args.phase in ("eval", "all"):
        phase1_evaluate(gpus, args.max_parallel)
    if args.phase in ("analyze", "all"):
        phase2_analyze()


if __name__ == "__main__":
    main()
