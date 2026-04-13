#!/usr/bin/env python3
"""
Experiment 6: Calibration Set Scaling

Show how spectral editing behaves as calibration data increases.
For 2 representative settings:
  For N_cal in {16, 32, 64, 128, 256, 512}:
    Run random_index, smooth_abs, grad_direction, continued_ft (3 seeds each)

Analysis:
  - At what N_cal does continued FT stop being harmful?
  - At what N_cal does guided start consistently beating random?
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "exp6_calib_scaling"
PYTHON = sys.executable

# 2 representative settings: one "aligned" (math), one "misaligned" (alpaca/IFEval)
SETTINGS = [
    {
        "model_dir": "Qwen-Qwen3-8B",
        "base_model_id": "Qwen/Qwen3-8B",
        "task": "math",
        "lora_path": str(REPO_ROOT / "runs/Qwen-Qwen3-8B/metamath/lora/profile-paper_math_ift_3ep/rank-16/seed42"),
        "calib_dataset": "gsm8k",
        "calib_config": "main",
        "lm_task": "gsm8k",
        "num_fewshot": 5,
        "gen_kwargs": "temperature=0,top_p=1",
        "gpu_mem": 0.95,
    },
    {
        "model_dir": "Qwen-Qwen3-8B",
        "base_model_id": "Qwen/Qwen3-8B",
        "task": "alpaca",
        "lora_path": str(REPO_ROOT / "runs/Qwen-Qwen3-8B/alpaca/lora/profile-paper_alpaca_3ep/rank-16/seed42"),
        "calib_dataset": "tatsu-lab/alpaca",
        "calib_config": None,
        "lm_task": "ifeval",
        "num_fewshot": None,
        "gen_kwargs": "max_gen_toks=2048,temperature=0,top_p=1",
        "gpu_mem": 0.95,
    },
]

N_CAL_VALUES = [16, 32, 64, 128, 256, 512]
METHODS = ["random_index", "smooth_abs", "grad_direction"]
SEEDS = [42, 43, 44]
TARGET_MODULES = ["down_proj", "o_proj"]


def phase1_create_and_eval(gpus: list[int], max_parallel: int):
    """Create edited adapters for all (N_cal, method, seed) combos and evaluate."""
    print("=" * 80)
    print("Creating and evaluating calibration scaling experiments")
    print("=" * 80)

    pending = []

    for setting in SETTINGS:
        model_dir = setting["model_dir"]
        task = setting["task"]
        base_model_id = setting["base_model_id"]
        lora_path = setting["lora_path"]
        calib_dataset = setting["calib_dataset"]
        calib_config = setting["calib_config"]

        for n_cal in N_CAL_VALUES:
            for method in METHODS:
                for seed in SEEDS:
                    out_dir = OUT_ROOT / "edited" / model_dir / task / f"ncal{n_cal}" / method / f"seed{seed}"
                    result_file = out_dir / "eval_result.json"
                    if result_file.exists():
                        continue

                    pending.append({
                        "model_dir": model_dir,
                        "task": task,
                        "base_model_id": base_model_id,
                        "lora_path": lora_path,
                        "calib_dataset": calib_dataset,
                        "calib_config": calib_config,
                        "n_cal": n_cal,
                        "method": method,
                        "seed": seed,
                        "out_dir": str(out_dir),
                        "lm_task": setting["lm_task"],
                        "num_fewshot": setting["num_fewshot"],
                        "gen_kwargs": setting.get("gen_kwargs"),
                        "gpu_mem": setting.get("gpu_mem", 0.90),
                    })

    print(f"Found {len(pending)} pending jobs")
    if not pending:
        return

    max_workers = min(max_parallel, len(gpus), len(pending))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, item in enumerate(pending):
            gpu_id = gpus[i % len(gpus)]
            futures[executor.submit(run_single_job, item, gpu_id)] = item

        results_path = OUT_ROOT / "scaling_results.jsonl"
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                with open(results_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
                status = "OK" if result.get("metric_value") is not None else "FAIL"
                val = f"{result['metric_value']:.4f}" if result.get("metric_value") is not None else result.get("error", "")[:60]
                print(f"  [{status}] {item['model_dir']}/{item['task']}/ncal{item['n_cal']}/{item['method']}/seed{item['seed']}: {val}")
            except Exception as e:
                print(f"  [ERROR] {item['model_dir']}/{item['task']}/ncal{item['n_cal']}/{item['method']}/seed{item['seed']}: {e}")


def run_single_job(item: dict, gpu_id: int) -> dict:
    """Run spectral edit + evaluation for a single (N_cal, method, seed) combo."""
    out_dir = Path(item["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    # Step 1: Run spectral edit with specified N_cal
    calib_config_arg = f'--calib_config {item["calib_config"]}' if item["calib_config"] else ""
    edit_cmd = [
        PYTHON, "-m", "finetune.spectral_edit.cli", "edit",
        "--base_model", item["base_model_id"],
        "--lora_path", item["lora_path"],
        "--out_dir", str(out_dir),
        "--target_modules", "down_proj", "o_proj",
        "--mode", item["method"],
        "--calib_dataset", item["calib_dataset"],
        "--calib_samples", str(item["n_cal"]),
        "--calib_batch_size", "2",
        "--calib_shuffle",
        "--seed", str(item["seed"]),
        "--preserve_energy", "l1",
        "--grad_norm", "mean_abs",
    ]
    if item["calib_config"]:
        edit_cmd += ["--calib_config", item["calib_config"]]

    edit_result = subprocess.run(edit_cmd, capture_output=True, text=True, env=env, timeout=600)
    if edit_result.returncode != 0:
        return {**item, "error": f"edit failed: {edit_result.stderr[-300:]}", "metric_value": None}

    # Step 2: Merge and evaluate
    merged_dir = out_dir / "eval" / "merged_model"
    merged_dir.parent.mkdir(parents=True, exist_ok=True)

    merge_script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("{item['base_model_id']}", use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("{item['base_model_id']}", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, "{str(out_dir)}")
model = model.merge_and_unload()
model.save_pretrained("{str(merged_dir)}")
tok.save_pretrained("{str(merged_dir)}")
del model; torch.cuda.empty_cache()
"""
    r = subprocess.run([PYTHON, "-c", merge_script], capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        return {**item, "error": f"merge failed: {r.stderr[-300:]}", "metric_value": None}

    eval_dir = out_dir / "eval" / "lm_eval_results"
    cmd = [
        PYTHON, "-m", "lm_eval", "--model", "vllm",
        "--model_args", f"pretrained={merged_dir},tensor_parallel_size=1,gpu_memory_utilization={item['gpu_mem']},dtype=float16",
        "--tasks", item["lm_task"], "--batch_size", "auto",
        "--output_path", str(eval_dir),
    ]
    if item["num_fewshot"] is not None:
        cmd += ["--num_fewshot", str(item["num_fewshot"])]
    if item.get("gen_kwargs"):
        cmd += ["--gen_kwargs", item["gen_kwargs"]]

    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)

    if r.returncode != 0:
        return {**item, "error": f"lm-eval failed: {r.stderr[-300:]}", "metric_value": None}

    # Parse result
    metric_keys = {"gsm8k": "exact_match,strict-match", "ifeval": "prompt_level_strict_acc,none",
                   "commonsense_qa": "acc,none"}
    primary_key = metric_keys.get(item["lm_task"], "acc,none")
    metric_value = None
    for root, dirs, files in os.walk(eval_dir):
        for fname in files:
            if fname.startswith("results_") and fname.endswith(".json"):
                try:
                    with open(os.path.join(root, fname)) as fp:
                        data = json.load(fp)
                    task_results = data.get("results", {}).get(item["lm_task"], {})
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
    with open(out_dir / "eval_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def phase2_analyze():
    """Analyze calibration scaling results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv

    print("=" * 80)
    print("Analyzing calibration scaling")
    print("=" * 80)

    results_path = OUT_ROOT / "scaling_results.jsonl"
    if not results_path.exists():
        print("No scaling results found!")
        return

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("metric_value") is None:
                continue
            skey = f"{rec['model_dir']}/{rec['task']}"
            data[skey][rec["method"]][rec["n_cal"]].append(rec["metric_value"])

    # Load baselines
    baseline_path = REPO_ROOT / "outputs" / "rebuttal_exp" / "raw" / "multiseed_eval" / "eval_results.csv"
    baselines = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("method") == "baseline" and row.get("metric_value"):
                    skey = f"{row.get('base_model_dir', '')}/{row.get('task', '')}"
                    baselines.setdefault(skey, []).append(float(row["metric_value"]))
        baselines = {k: np.mean(v) for k, v in baselines.items()}

    plot_dir = OUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = OUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for skey in sorted(data.keys()):
        baseline = baselines.get(skey, None)
        print(f"\n--- {skey} (baseline={baseline:.4f if baseline else 'N/A'}) ---")

        fig, ax = plt.subplots(figsize=(8, 5))
        method_colors = {"random_index": "#1f77b4", "smooth_abs": "#ff7f0e", "grad_direction": "#2ca02c"}

        for method in METHODS:
            if method not in data[skey]:
                continue

            ncals = sorted(data[skey][method].keys())
            means = [np.mean(data[skey][method][n]) for n in ncals]
            stds = [np.std(data[skey][method][n]) for n in ncals]
            ns = [len(data[skey][method][n]) for n in ncals]

            # Delta vs baseline
            if baseline:
                deltas = [(m - baseline) / max(baseline, 1e-8) * 100 for m in means]
                delta_stds = [s / max(baseline, 1e-8) * 100 for s in stds]
            else:
                deltas = means
                delta_stds = stds

            print(f"  {method}:")
            for n, m, s, d in zip(ncals, means, stds, deltas):
                print(f"    N_cal={n:4d}: {m:.4f} ± {s:.4f} (Δ={d:+.2f}%)")

            color = method_colors.get(method, "gray")
            ax.errorbar(ncals, deltas, yerr=delta_stds, label=method,
                        marker="o", capsize=3, color=color, linewidth=1.5)

            for n, m, d, s, count in zip(ncals, means, deltas, stds, ns):
                all_results.append({
                    "setting": skey, "method": method, "n_cal": n,
                    "metric_mean": float(m), "metric_std": float(s),
                    "delta_pct": float(d), "n_seeds": count,
                    "baseline": baseline,
                })

        ax.axhline(0, color="black", ls="--", linewidth=0.8, label="Baseline")
        ax.set_xlabel("N_cal (calibration samples)", fontsize=12)
        ax.set_ylabel("Δ% vs baseline", fontsize=12)
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_CAL_VALUES)
        ax.set_xticklabels([str(n) for n in N_CAL_VALUES])
        model_short = skey.split("-")[-1].split("/")[0]
        task_short = skey.split("/")[1]
        ax.set_title(f"{model_short}/{task_short} — Calibration Scaling", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_name = skey.replace("/", "_")
        fig.savefig(plot_dir / f"exp6_scaling_{safe_name}.pdf", dpi=150)
        plt.close(fig)

    with open(analysis_dir / "scaling_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[Saved] Analysis and plots")


def main():
    parser = argparse.ArgumentParser(description="Experiment 6: Calibration Set Scaling")
    parser.add_argument("--phase", type=str, default="all", choices=["eval", "analyze", "all"])
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_parallel", type=int, default=6)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    if args.phase in ("eval", "all"):
        phase1_create_and_eval(gpus, args.max_parallel)
    if args.phase in ("analyze", "all"):
        phase2_analyze()


if __name__ == "__main__":
    main()
