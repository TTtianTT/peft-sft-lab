#!/usr/bin/env python3
"""
Experiment 2: Oracle Spectrum Editing

Using leave-one-out results from Experiment 1, establish the UPPER BOUND
of spectrum-only editing by:

Strategy A: "Zero harmful components" — zero components where removal helped
Strategy B: "Greedy best subset" — progressively zero most harmful components
Strategy C: "Amplify helpful, suppress harmful" — scale components based on LOO
            importance, with L1 normalization

Compares oracle performance against: baseline, random_index, smooth_abs, grad_direction
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
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from finetune.spectral_edit.io import (
    load_adapter_config, load_lora_state_dict, parse_lora_ab_key,
    save_lora_state_dict, layer_idx_from_module_prefix,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma

EXP1_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "exp1_leave_one_out"
OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "exp2_oracle_editing"
PYTHON = sys.executable

SETTINGS = [
    {
        "model_dir": "meta-llama-Llama-3.1-8B",
        "base_model_id": "meta-llama/Llama-3.1-8B",
        "task": "csqa",
        "lora_path": str(REPO_ROOT / "runs/meta-llama-Llama-3.1-8B/csqa/lora/profile-paper_csqa_3ep/rank-16/seed42"),
    },
    {
        "model_dir": "meta-llama-Llama-3.1-8B",
        "base_model_id": "meta-llama/Llama-3.1-8B",
        "task": "math",
        "lora_path": str(REPO_ROOT / "runs/meta-llama-Llama-3.1-8B/metamath/lora/profile-paper_math_ift_3ep/rank-16/seed42"),
    },
    {
        "model_dir": "Qwen-Qwen3-8B",
        "base_model_id": "Qwen/Qwen3-8B",
        "task": "math",
        "lora_path": str(REPO_ROOT / "runs/Qwen-Qwen3-8B/metamath/lora/profile-paper_math_ift_3ep/rank-16/seed42"),
    },
    {
        "model_dir": "Qwen-Qwen3-8B",
        "base_model_id": "Qwen/Qwen3-8B",
        "task": "alpaca",
        "lora_path": str(REPO_ROOT / "runs/Qwen-Qwen3-8B/alpaca/lora/profile-paper_alpaca_3ep/rank-16/seed42"),
    },
]

TARGET_MODULES = ["down_proj", "o_proj"]
TASK_DIR_TO_LM_EVAL = {"math": "gsm8k", "alpaca": "ifeval", "csqa": "commonsense_qa"}
TASK_CONFIGS = {
    "math": {"num_fewshot": 5, "gen_kwargs": "temperature=0,top_p=1", "gpu_mem": 0.95},
    "alpaca": {"num_fewshot": None, "gen_kwargs": "max_gen_toks=2048,temperature=0,top_p=1", "gpu_mem": 0.95},
    "csqa": {"num_fewshot": 0, "gen_kwargs": None, "gpu_mem": 0.85},
}


def load_loo_results():
    """Load leave-one-out results from Experiment 1."""
    analysis_path = EXP1_ROOT / "analysis" / "loo_analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Exp 1 analysis not found: {analysis_path}")
    with open(analysis_path) as f:
        return json.load(f)


def phase1_create_oracle_adapters():
    """Create oracle-edited adapters based on LOO results."""
    print("=" * 80)
    print("PHASE 1: Creating oracle-edited adapters")
    print("=" * 80)

    loo_results = load_loo_results()
    loo_by_setting = {f"{r['model_dir']}/{r['task']}": r for r in loo_results}

    for setting in SETTINGS:
        model_dir = setting["model_dir"]
        task = setting["task"]
        lora_path = setting["lora_path"]
        skey = f"{model_dir}/{task}"

        if skey not in loo_by_setting:
            print(f"  [SKIP] {skey} — no LOO results")
            continue

        loo = loo_by_setting[skey]
        baseline = loo["baseline"]
        loo_delta = {int(k): v for k, v in loo["loo_delta"].items()}
        rank = len(loo_delta)

        print(f"\n--- {skey} (baseline={baseline:.4f}) ---")

        # Load adapter
        adapter_cfg = load_adapter_config(lora_path)
        sd, fmt = load_lora_state_dict(lora_path)

        pairs = {}
        target_set = set(TARGET_MODULES)
        for k, t in sd.items():
            parsed = parse_lora_ab_key(k)
            if not parsed:
                continue
            prefix, which, adapter = parsed
            suffix = prefix.split(".")[-1]
            if suffix not in target_set:
                continue
            pairs.setdefault(prefix, {})
            pairs[prefix][which] = (k, t, adapter)

        selected = [p for p in pairs if "A" in pairs[p] and "B" in pairs[p]]

        # Compute SVD
        module_svds = {}
        for prefix in selected:
            keyA, A_cpu, _ = pairs[prefix]["A"]
            keyB, B_cpu, _ = pairs[prefix]["B"]
            U, S, Vh, V = lowrank_svd_from_ba(B_cpu, A_cpu)
            module_svds[prefix] = {
                "U": U, "S": S, "Vh": Vh,
                "keyA": keyA, "keyB": keyB,
                "A_dtype": A_cpu.dtype, "B_dtype": B_cpu.dtype,
            }

        # Strategy A: Zero harmful components (where removal helped, i.e., loo_delta > 0)
        harmful_components = [k for k in range(rank) if loo_delta.get(k, 0) > 0]
        strategies = {}

        strategies["oracle_zero_harmful"] = {
            "description": f"Zero {len(harmful_components)} components where removal helped",
            "zero_components": harmful_components,
            "scale_components": None,
        }

        # Strategy B: Greedy — zero components ordered by how much removal helped (most positive Δ first)
        sorted_by_benefit = sorted(range(rank), key=lambda k: loo_delta.get(k, 0), reverse=True)
        # Try zeroing top-1, top-2, ..., top-N harmful
        for n_zero in [1, 2, 4]:
            to_zero = [k for k in sorted_by_benefit[:n_zero] if loo_delta.get(k, 0) > 0]
            if not to_zero:
                continue
            strategies[f"oracle_greedy_top{n_zero}"] = {
                "description": f"Zero top-{n_zero} most harmful components",
                "zero_components": to_zero,
                "scale_components": None,
            }

        # Strategy C: Amplify helpful (removal hurts) by 1.2x, suppress harmful (removal helps) by 0.5x
        scale_map = {}
        for k in range(rank):
            d = loo_delta.get(k, 0)
            if d > 0:  # removal helped → this component is harmful
                scale_map[k] = 0.5
            elif d < -0.001:  # removal hurt → this component is helpful
                scale_map[k] = 1.2
            else:
                scale_map[k] = 1.0
        strategies["oracle_amplify_suppress"] = {
            "description": "Amplify helpful (1.2x), suppress harmful (0.5x), L1-preserve",
            "zero_components": None,
            "scale_components": scale_map,
        }

        # Create edited adapters for each strategy
        for strategy_name, strategy in strategies.items():
            out_dir = OUT_ROOT / "edited" / model_dir / task / strategy_name
            if (out_dir / "adapter_model.safetensors").exists() or (out_dir / "adapter_model.bin").exists():
                print(f"    [SKIP] {strategy_name} — already exists")
                continue

            if out_dir.exists():
                shutil.rmtree(out_dir)
            shutil.copytree(lora_path, str(out_dir))

            new_sd = dict(sd)
            for prefix in selected:
                svd_info = module_svds[prefix]
                sigma = svd_info["S"].clone()

                if strategy["zero_components"]:
                    for k_idx in strategy["zero_components"]:
                        sigma[k_idx] = 0.0

                if strategy["scale_components"]:
                    for k_idx, scale in strategy["scale_components"].items():
                        sigma[int(k_idx)] *= scale
                    # L1 preserve
                    s0 = svd_info["S"].sum().clamp_min(1e-8)
                    s1 = sigma.sum().clamp_min(1e-8)
                    sigma = sigma * (s0 / s1)

                B_new, A_new = rebuild_ba_from_uv_sigma(svd_info["U"], svd_info["Vh"], sigma)
                new_sd[svd_info["keyA"]] = A_new.to(dtype=svd_info["A_dtype"])
                new_sd[svd_info["keyB"]] = B_new.to(dtype=svd_info["B_dtype"])

            save_lora_state_dict(str(out_dir), new_sd, fmt)

            meta = {
                "model_dir": model_dir, "task": task,
                "strategy": strategy_name, **strategy,
            }
            with open(out_dir / "oracle_meta.json", "w") as f:
                json.dump(meta, f, indent=2, default=str)

            print(f"    Created: {strategy_name}")


def run_single_eval(item: dict, gpu_id: int) -> dict:
    """Evaluate a single oracle-edited adapter."""
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

    merge_script = f"""
import torch, shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "{base_model_id}"
adapter_dir = "{str(adapter_dir)}"
merged_dir = "{str(merged_dir)}"

tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()
model.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
del model; torch.cuda.empty_cache()
"""
    merge_result = subprocess.run([PYTHON, "-c", merge_script],
                                  capture_output=True, text=True, env=env, timeout=600)
    if merge_result.returncode != 0:
        return {**item, "error": f"merge failed: {merge_result.stderr[-500:]}", "metric_value": None}

    cmd = [
        PYTHON, "-m", "lm_eval", "--model", "vllm",
        "--model_args", f"pretrained={merged_dir},tensor_parallel_size=1,gpu_memory_utilization={tcfg['gpu_mem']},dtype=float16",
        "--tasks", lm_task, "--batch_size", "auto",
        "--output_path", str(eval_dir / "lm_eval_results"),
    ]
    if tcfg["num_fewshot"] is not None:
        cmd += ["--num_fewshot", str(tcfg["num_fewshot"])]
    if tcfg["gen_kwargs"]:
        cmd += ["--gen_kwargs", tcfg["gen_kwargs"]]

    eval_result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)
    if eval_result.returncode != 0:
        return {**item, "error": f"lm-eval failed: {eval_result.stderr[-500:]}", "metric_value": None}

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
    with open(adapter_dir / "eval_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def phase2_evaluate(gpus: list[int], max_parallel: int):
    """Evaluate all oracle-edited adapters."""
    print("=" * 80)
    print("PHASE 2: Evaluating oracle-edited adapters")
    print("=" * 80)

    pending = []
    for setting in SETTINGS:
        model_dir = setting["model_dir"]
        task = setting["task"]
        edit_root = OUT_ROOT / "edited" / model_dir / task
        if not edit_root.exists():
            continue
        for strat_dir in sorted(edit_root.iterdir()):
            if not strat_dir.is_dir():
                continue
            if (strat_dir / "eval_result.json").exists():
                continue
            pending.append({
                "model_dir": model_dir, "task": task,
                "base_model_id": setting["base_model_id"],
                "adapter_dir": str(strat_dir),
                "strategy": strat_dir.name,
            })

    print(f"Found {len(pending)} pending evaluations")
    if not pending:
        return

    results_path = OUT_ROOT / "oracle_eval_results.jsonl"
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
                with open(results_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
                status = "OK" if result.get("metric_value") is not None else "FAIL"
                val = f"{result['metric_value']:.4f}" if result.get("metric_value") is not None else result.get("error", "")[:80]
                print(f"  [{status}] {item['model_dir']}/{item['task']}/{item['strategy']}: {val}")
            except Exception as e:
                print(f"  [ERROR] {item['model_dir']}/{item['task']}/{item['strategy']}: {e}")


def phase3_analyze():
    """Analyze oracle editing results and compare against baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 80)
    print("PHASE 3: Analyzing oracle editing results")
    print("=" * 80)

    # Load oracle results
    results_path = OUT_ROOT / "oracle_eval_results.jsonl"
    if not results_path.exists():
        print("No oracle eval results found!")
        return

    oracle_data = defaultdict(dict)
    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("metric_value") is None:
                continue
            skey = f"{rec['model_dir']}/{rec['task']}"
            oracle_data[skey][rec["strategy"]] = rec["metric_value"]

    # Load prior eval baselines
    import csv
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

    all_results = []

    print(f"\n{'Setting':<35} {'Strategy':<28} {'Metric':>8} {'Δ%':>8}")
    print("-" * 85)

    for setting in SETTINGS:
        model_dir = setting["model_dir"]
        task = setting["task"]
        skey = f"{model_dir}/{task}"

        if skey not in oracle_data:
            continue

        bl_vals = prior_eval.get(skey, {}).get("baseline", [])
        baseline = np.mean(bl_vals) if bl_vals else None

        # Comparison methods
        comparisons = {}
        for method in ["baseline", "random_index", "smooth_abs", "grad_direction"]:
            vals = prior_eval.get(skey, {}).get(method, [])
            if vals:
                comparisons[method] = np.mean(vals)

        for strategy, metric in sorted(oracle_data[skey].items()):
            delta = (metric - baseline) / max(baseline, 1e-8) * 100 if baseline else 0
            print(f"{skey:<35} {strategy:<28} {metric:>8.4f} {delta:>+8.2f}%")
            all_results.append({
                "model_dir": model_dir, "task": task, "strategy": strategy,
                "metric": metric, "delta_pct": delta, "baseline": baseline,
            })

        # Add comparison methods
        for method, val in comparisons.items():
            delta = (val - baseline) / max(baseline, 1e-8) * 100 if baseline else 0
            print(f"{skey:<35} {method:<28} {val:>8.4f} {delta:>+8.2f}%")

        print()

    # Bar chart per setting
    for setting in SETTINGS:
        model_dir = setting["model_dir"]
        task = setting["task"]
        skey = f"{model_dir}/{task}"
        if skey not in oracle_data:
            continue

        bl_vals = prior_eval.get(skey, {}).get("baseline", [])
        baseline = np.mean(bl_vals) if bl_vals else 1.0

        methods_and_vals = {}
        for method in ["random_index", "smooth_abs", "grad_direction"]:
            vals = prior_eval.get(skey, {}).get(method, [])
            if vals:
                methods_and_vals[method] = (np.mean(vals) - baseline) / max(baseline, 1e-8) * 100

        for strategy, metric in oracle_data[skey].items():
            methods_and_vals[strategy] = (metric - baseline) / max(baseline, 1e-8) * 100

        if not methods_and_vals:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        names = list(methods_and_vals.keys())
        vals = [methods_and_vals[n] for n in names]
        colors = ["#2ca02c" if "oracle" in n else "#1f77b4" for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Δ% vs baseline", fontsize=11)
        ax.set_title(f"{skey} — Oracle vs Heuristic Editing", fontsize=11)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:+.2f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

        fig.tight_layout()
        fig.savefig(plot_dir / f"exp2_oracle_{model_dir}_{task}.pdf", dpi=150)
        plt.close(fig)

    # Save analysis
    analysis_dir = OUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis_dir / "oracle_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[Saved] Analysis and plots")


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Oracle Spectrum Editing")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["create", "eval", "analyze", "all"])
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_parallel", type=int, default=6)
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    if args.phase in ("create", "all"):
        phase1_create_oracle_adapters()
    if args.phase in ("eval", "all"):
        phase2_evaluate(gpus, args.max_parallel)
    if args.phase in ("analyze", "all"):
        phase3_analyze()


if __name__ == "__main__":
    main()
