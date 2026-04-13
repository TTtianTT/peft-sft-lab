#!/usr/bin/env python3
"""
Experiment 1: Component-wise Leave-One-Out

For each (model, task) pair:
  For each singular component k in {0, ..., 15}:
    1. Load LoRA adapter, compute SVD for each edited module (o_proj, down_proj)
    2. Zero out component k across ALL layers simultaneously
    3. Reconstruct LoRA B, A from modified sigma
    4. Save edited adapter
    5. Evaluate downstream via lm-eval-harness

Also computes aggregated sensitivity scores per component and correlates
with leave-one-out importance.

Two phases:
  Phase 1: Create all 64 edited adapters (no GPU needed for SVD, but needs adapter files)
  Phase 2: Evaluate all 64 adapters (GPU needed, parallelized)
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
    load_adapter_config,
    load_lora_state_dict,
    parse_lora_ab_key,
    save_lora_state_dict,
    layer_idx_from_module_prefix,
    get_scaling_for_module,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma

OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "exp1_leave_one_out"
PYTHON = sys.executable

# Settings: (model_dir, base_model_id, task, lora_path, calib_dataset, calib_config)
SETTINGS = [
    {
        "model_dir": "meta-llama-Llama-3.1-8B",
        "base_model_id": "meta-llama/Llama-3.1-8B",
        "task": "csqa",
        "lora_path": str(REPO_ROOT / "runs/meta-llama-Llama-3.1-8B/csqa/lora/profile-paper_csqa_3ep/rank-16/seed42"),
        "calib_dataset": "tau/commonsense_qa",
        "calib_config": None,
    },
    {
        "model_dir": "meta-llama-Llama-3.1-8B",
        "base_model_id": "meta-llama/Llama-3.1-8B",
        "task": "math",
        "lora_path": str(REPO_ROOT / "runs/meta-llama-Llama-3.1-8B/metamath/lora/profile-paper_math_ift_3ep/rank-16/seed42"),
        "calib_dataset": "gsm8k",
        "calib_config": "main",
    },
    {
        "model_dir": "Qwen-Qwen3-8B",
        "base_model_id": "Qwen/Qwen3-8B",
        "task": "math",
        "lora_path": str(REPO_ROOT / "runs/Qwen-Qwen3-8B/metamath/lora/profile-paper_math_ift_3ep/rank-16/seed42"),
        "calib_dataset": "gsm8k",
        "calib_config": "main",
    },
    {
        "model_dir": "Qwen-Qwen3-8B",
        "base_model_id": "Qwen/Qwen3-8B",
        "task": "alpaca",
        "lora_path": str(REPO_ROOT / "runs/Qwen-Qwen3-8B/alpaca/lora/profile-paper_alpaca_3ep/rank-16/seed42"),
        "calib_dataset": "tatsu-lab/alpaca",
        "calib_config": None,
    },
]

TARGET_MODULES = ["down_proj", "o_proj"]

TASK_DIR_TO_LM_EVAL = {"math": "gsm8k", "alpaca": "ifeval", "csqa": "commonsense_qa"}
TASK_CONFIGS = {
    "math": {"num_fewshot": 5, "gen_kwargs": "temperature=0,top_p=1", "gpu_mem": 0.95},
    "alpaca": {"num_fewshot": None, "gen_kwargs": "max_gen_toks=2048,temperature=0,top_p=1", "gpu_mem": 0.95},
    "csqa": {"num_fewshot": 0, "gen_kwargs": None, "gpu_mem": 0.85},
}


def phase1_create_edited_adapters():
    """Create all leave-one-out edited adapters (CPU only, no GPU needed)."""
    print("=" * 80)
    print("PHASE 1: Creating leave-one-out edited adapters")
    print("=" * 80)

    for setting in SETTINGS:
        model_dir = setting["model_dir"]
        task = setting["task"]
        lora_path = setting["lora_path"]

        print(f"\n--- {model_dir}/{task} ---")
        print(f"    LoRA path: {lora_path}")

        # Load adapter
        adapter_cfg = load_adapter_config(lora_path)
        sd, fmt = load_lora_state_dict(lora_path)

        # Find target modules
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
        print(f"    Found {len(selected)} target modules")

        # Compute SVD for all modules, store per-component sigma
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

        rank = module_svds[selected[0]]["S"].numel()
        print(f"    Rank: {rank}")

        # For each component k, create edited adapter with sigma_k = 0
        for k in range(rank):
            out_dir = OUT_ROOT / "edited" / model_dir / task / f"drop_k{k}"
            if (out_dir / "adapter_model.safetensors").exists() or (out_dir / "adapter_model.bin").exists():
                continue  # Already created

            # Copy original adapter
            if out_dir.exists():
                shutil.rmtree(out_dir)
            shutil.copytree(lora_path, str(out_dir))

            # Modify sigma: zero out component k across all modules
            new_sd = dict(sd)
            meta_modules = {}

            for prefix in selected:
                svd_info = module_svds[prefix]
                sigma = svd_info["S"].clone()
                sigma_orig_k = float(sigma[k].item())
                sigma[k] = 0.0

                B_new, A_new = rebuild_ba_from_uv_sigma(svd_info["U"], svd_info["Vh"], sigma)
                new_sd[svd_info["keyA"]] = A_new.to(dtype=svd_info["A_dtype"])
                new_sd[svd_info["keyB"]] = B_new.to(dtype=svd_info["B_dtype"])

                li = layer_idx_from_module_prefix(prefix)
                meta_modules[prefix] = {
                    "layer": li,
                    "suffix": prefix.split(".")[-1],
                    "sigma_k_original": sigma_orig_k,
                    "sigma_total": float(svd_info["S"].sum().item()),
                    "sigma_k_fraction": sigma_orig_k / max(float(svd_info["S"].sum().item()), 1e-8),
                }

            save_lora_state_dict(str(out_dir), new_sd, fmt)

            # Save metadata
            meta = {
                "model_dir": model_dir,
                "task": task,
                "dropped_component": k,
                "rank": rank,
                "n_modules": len(selected),
                "modules": meta_modules,
            }
            with open(out_dir / "loo_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"    Created: drop_k{k}")

        print(f"    Done: {rank} edited adapters for {model_dir}/{task}")


def phase2_evaluate(gpus: list[int], max_parallel: int, filter_setting: str = None):
    """Evaluate all leave-one-out edited adapters."""
    print("=" * 80)
    print("PHASE 2: Evaluating leave-one-out edited adapters")
    print("=" * 80)

    # Discover pending evals
    pending = []
    for setting in SETTINGS:
        model_dir = setting["model_dir"]
        task = setting["task"]

        if filter_setting and f"{model_dir}/{task}" != filter_setting:
            continue

        edit_root = OUT_ROOT / "edited" / model_dir / task
        if not edit_root.exists():
            continue

        for drop_dir in sorted(edit_root.iterdir()):
            if not drop_dir.is_dir() or not drop_dir.name.startswith("drop_k"):
                continue

            result_file = drop_dir / "eval_result.json"
            if result_file.exists():
                continue

            k = int(drop_dir.name.replace("drop_k", ""))
            pending.append({
                "model_dir": model_dir,
                "task": task,
                "base_model_id": setting["base_model_id"],
                "adapter_dir": str(drop_dir),
                "component_k": k,
            })

    print(f"Found {len(pending)} pending evaluations")
    if not pending:
        print("Nothing to evaluate!")
        return

    max_workers = min(max_parallel, len(gpus), len(pending))
    results_path = OUT_ROOT / "loo_eval_results.jsonl"

    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, item in enumerate(pending):
            gpu_id = gpus[i % len(gpus)]
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
                    print(f"  [OK] {item['model_dir']}/{item['task']}/drop_k{item['component_k']}: "
                          f"{result['metric_value']:.4f}")
                else:
                    failed += 1
                    print(f"  [FAIL] {item['model_dir']}/{item['task']}/drop_k{item['component_k']}: "
                          f"{result.get('error', 'unknown')[:100]}")
            except Exception as e:
                failed += 1
                print(f"  [ERROR] {item['model_dir']}/{item['task']}/drop_k{item['component_k']}: {e}")

    print(f"\nDone: {completed} succeeded, {failed} failed out of {len(pending)} total")


def run_single_eval(item: dict, gpu_id: int) -> dict:
    """Evaluate a single leave-one-out edited adapter."""
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
        return {**item, "error": f"merge failed: {merge_result.stderr[-500:]}", "metric_value": None}

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
        return {**item, "error": f"lm-eval failed: {eval_result.stderr[-500:]}", "metric_value": None}

    # Parse result
    metric_keys = {
        "gsm8k": "exact_match,strict-match",
        "ifeval": "prompt_level_strict_acc,none",
        "commonsense_qa": "acc,none",
    }
    primary_key = metric_keys.get(lm_task, "acc,none")
    metric_value = None

    for root, dirs, files in os.walk(eval_dir / "lm_eval_results"):
        for fname in files:
            if fname.startswith("results_") and fname.endswith(".json"):
                try:
                    with open(os.path.join(root, fname)) as fp:
                        data = json.load(fp)
                    results = data.get("results", {})
                    task_results = results.get(lm_task, {})
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


def phase3_sensitivity(gpus: list[int]):
    """Compute sensitivity scores for each setting (requires GPU for gradient computation)."""
    print("=" * 80)
    print("PHASE 3: Computing per-component sensitivity scores")
    print("=" * 80)

    for i, setting in enumerate(SETTINGS):
        model_dir = setting["model_dir"]
        task = setting["task"]
        base_model_id = setting["base_model_id"]
        lora_path = setting["lora_path"]
        calib_dataset = setting["calib_dataset"]
        calib_config = setting["calib_config"]

        out_file = OUT_ROOT / "sensitivity" / f"{model_dir}_{task}_sensitivity.json"
        if out_file.exists():
            print(f"  [SKIP] {model_dir}/{task} — already computed")
            continue

        out_file.parent.mkdir(parents=True, exist_ok=True)
        gpu_id = gpus[i % len(gpus)]

        print(f"\n  Computing sensitivity for {model_dir}/{task} on GPU {gpu_id}...")

        # Use the existing CLI to run a dummy edit that captures g_sigma
        # We'll use a special script that loads the model, runs calibration,
        # and saves per-component sensitivity scores
        sensitivity_script = f"""
import json, sys, torch
sys.path.insert(0, "{REPO_ROOT / 'src'}")
from finetune.spectral_edit.io import (
    load_adapter_config, load_lora_state_dict, parse_lora_ab_key,
    layer_idx_from_module_prefix, get_scaling_for_module,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba
from finetune.spectral_edit.hooks import HOOK_CTX, ModuleSpec, register_sigma_hooks, remove_hooks
from finetune.spectral_edit.calib import (
    build_calib_formatter, load_calibration_split,
    make_calib_batch, sample_calibration_examples,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import math

device = "cuda"
lora_dir = "{lora_path}"
base_model_id = "{base_model_id}"
calib_dataset = "{calib_dataset}"
calib_config = {repr(calib_config)}

adapter_cfg = load_adapter_config(lora_dir)
sd, fmt = load_lora_state_dict(lora_dir)

target_modules = {TARGET_MODULES}
pairs = {{}}
for k, t in sd.items():
    parsed = parse_lora_ab_key(k)
    if not parsed:
        continue
    prefix, which, adapter = parsed
    suffix = prefix.split(".")[-1]
    if suffix not in set(target_modules):
        continue
    pairs.setdefault(prefix, {{}})
    pairs[prefix][which] = (k, t, adapter)

selected = [p for p in pairs if "A" in pairs[p] and "B" in pairs[p]]
print(f"[Info] {{len(selected)}} target modules")

tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device)
model = PeftModel.from_pretrained(base, lora_dir, is_trainable=True).to(device)
model.eval()
model.config.use_cache = False

for n, p in model.named_parameters():
    p.requires_grad_("lora_" in n)

name_to_module = dict(model.named_modules())
specs = {{}}
for prefix in selected:
    keyA, A_cpu, adapterA = pairs[prefix]["A"]
    keyB, B_cpu, adapterB = pairs[prefix]["B"]
    adapter_name = adapterA or adapterB
    A = A_cpu.to(device)
    B = B_cpu.to(device)
    U, S, Vh, V = lowrank_svd_from_ba(B, A)
    scaling = get_scaling_for_module(adapter_cfg, prefix)

    if prefix not in name_to_module:
        candidates = [nm for nm in name_to_module if nm.endswith(prefix)]
        module_name = candidates[0] if candidates else prefix
    else:
        module_name = prefix
    mod = name_to_module[module_name]

    specs[prefix] = ModuleSpec(
        module_prefix=prefix, module=mod,
        U=U.detach(), V=V.detach(), Vh=Vh.detach(),
        sigma0=S.detach().cpu(), scaling=scaling, adapter=adapter_name,
    )

handles = register_sigma_hooks(specs)

formatter, _ = build_calib_formatter(calib_dataset, None)
ds_split = load_calibration_split(calib_dataset, calib_config, "train")
calib_examples = sample_calibration_examples(ds_split, 128, True, 42, 0)

HOOK_CTX.reset()
bs = 2
for i in range(0, len(calib_examples), bs):
    batch_ex = calib_examples[i:i+bs]
    input_ids, attn_mask, labels = make_calib_batch(tok, batch_ex, formatter, add_eos=True)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    labels = labels.to(device)
    HOOK_CTX.attn_mask = attn_mask
    out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    loss = out.loss
    model.zero_grad(set_to_none=True)
    loss.backward()
    model.zero_grad(set_to_none=True)

remove_hooks(handles)

# Collect per-module, per-component sensitivity
results = {{}}
for prefix, spec in specs.items():
    g = HOOK_CTX.gsum.get(prefix, None)
    if g is None:
        continue
    g_abs = g.abs().cpu().tolist()
    sigma0 = spec.sigma0.cpu().tolist()
    li = layer_idx_from_module_prefix(prefix)
    results[prefix] = {{
        "layer": li,
        "suffix": prefix.split(".")[-1],
        "g_abs": g_abs,
        "sigma0": sigma0,
    }}

with open("{str(out_file)}", "w") as f:
    json.dump(results, f, indent=2)

print(f"[Done] Saved sensitivity for {{len(results)}} modules to {str(out_file)}")
del model, base
torch.cuda.empty_cache()
"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

        result = subprocess.run(
            [PYTHON, "-c", sensitivity_script],
            capture_output=True, text=True, env=env, timeout=600,
        )
        if result.returncode != 0:
            print(f"  [FAIL] {model_dir}/{task}: {result.stderr[-300:]}")
        else:
            print(f"  [OK] {model_dir}/{task}")


def phase4_analyze():
    """Analyze leave-one-out results and compute correlations with sensitivity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    print("=" * 80)
    print("PHASE 4: Analyzing leave-one-out results")
    print("=" * 80)

    results_path = OUT_ROOT / "loo_eval_results.jsonl"
    if not results_path.exists():
        print("No eval results found!")
        return

    # Load eval results
    eval_data = defaultdict(dict)
    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("metric_value") is None:
                continue
            key = f"{rec['model_dir']}/{rec['task']}"
            eval_data[key][rec["component_k"]] = rec["metric_value"]

    # Load baselines from prior eval data
    baseline_path = REPO_ROOT / "outputs" / "rebuttal_exp" / "raw" / "multiseed_eval" / "eval_results.csv"
    baselines = {}
    if baseline_path.exists():
        import csv
        with open(baseline_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("method") == "baseline" and row.get("metric_value"):
                    key = f"{row.get('base_model_dir', '')}/{row.get('task', '')}"
                    baselines.setdefault(key, []).append(float(row["metric_value"]))
        baselines = {k: np.mean(v) for k, v in baselines.items()}

    plot_dir = OUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = OUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for setting in SETTINGS:
        model_dir = setting["model_dir"]
        task = setting["task"]
        skey = f"{model_dir}/{task}"

        if skey not in eval_data:
            print(f"  [SKIP] {skey} — no eval data")
            continue

        comp_metrics = eval_data[skey]
        baseline = baselines.get(skey, None)

        if baseline is None:
            # Try to infer from component metrics
            print(f"  [WARN] {skey} — no baseline found, using mean of all components")
            baseline = np.mean(list(comp_metrics.values()))

        # Load sensitivity
        sens_path = OUT_ROOT / "sensitivity" / f"{model_dir}_{task}_sensitivity.json"
        per_component_sensitivity = None
        if sens_path.exists():
            with open(sens_path) as f:
                sens_data = json.load(f)
            # Aggregate: mean |g_k| across all modules for each component k
            rank = 16
            agg_g = np.zeros(rank)
            n_modules = 0
            for prefix, mdata in sens_data.items():
                g_abs = np.array(mdata["g_abs"])
                agg_g += g_abs
                n_modules += 1
            if n_modules > 0:
                per_component_sensitivity = agg_g / n_modules

        # Compute leave-one-out importance
        rank = max(comp_metrics.keys()) + 1
        loo_importance = np.zeros(rank)  # I_k = baseline - metric_without_k (positive = important)
        loo_delta = np.zeros(rank)       # Δ = metric_without_k - baseline (negative = important)

        for k in range(rank):
            if k in comp_metrics:
                loo_delta[k] = comp_metrics[k] - baseline
                loo_importance[k] = baseline - comp_metrics[k]

        print(f"\n  === {skey} (baseline={baseline:.4f}) ===")
        print(f"  {'k':>3} | {'Δ_metric':>10} | {'|g_k| mean':>12} | {'σ_k':>10}")
        print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

        for k in range(rank):
            g_str = f"{per_component_sensitivity[k]:.6f}" if per_component_sensitivity is not None else "N/A"
            print(f"  {k:3d} | {loo_delta[k]:+10.4f} | {g_str:>12} | {'':>10}")

        # Correlation analysis
        corr_result = None
        if per_component_sensitivity is not None:
            S_k = per_component_sensitivity
            I_k = loo_importance
            rho, pval = stats.spearmanr(S_k, I_k)
            pearson_r, pearson_p = stats.pearsonr(S_k, I_k)
            print(f"\n  Spearman correlation: ρ = {rho:.4f}, p = {pval:.4f}")
            print(f"  Pearson correlation:  r = {pearson_r:.4f}, p = {pearson_p:.4f}")
            corr_result = {
                "spearman_rho": float(rho), "spearman_p": float(pval),
                "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            }

            # Scatter plot: sensitivity vs importance
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.scatter(S_k, I_k, s=80, edgecolors="black", linewidths=0.5, zorder=3)
            for k_idx in range(rank):
                ax.annotate(str(k_idx), (S_k[k_idx], I_k[k_idx]),
                            fontsize=8, ha="center", va="bottom")
            ax.set_xlabel("Aggregated Sensitivity |g_k|", fontsize=12)
            ax.set_ylabel("Leave-One-Out Importance (I_k)", fontsize=12)
            ax.set_title(f"{skey}\nSpearman ρ={rho:.3f} (p={pval:.3f})", fontsize=11)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / f"exp1_loo_scatter_{model_dir}_{task}.pdf", dpi=150)
            plt.close(fig)

        # Bar chart: per-component Δ_metric
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        colors = []
        if per_component_sensitivity is not None:
            quartiles = np.percentile(per_component_sensitivity, [25, 50, 75])
            for k_idx in range(rank):
                if per_component_sensitivity[k_idx] >= quartiles[2]:
                    colors.append("#d62728")  # Q4 (most sensitive) - red
                elif per_component_sensitivity[k_idx] >= quartiles[1]:
                    colors.append("#ff7f0e")  # Q3 - orange
                elif per_component_sensitivity[k_idx] >= quartiles[0]:
                    colors.append("#2ca02c")  # Q2 - green
                else:
                    colors.append("#1f77b4")  # Q1 (least sensitive) - blue
        else:
            colors = ["#1f77b4"] * rank

        ax.bar(range(rank), loo_delta * 100, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Component Index k", fontsize=12)
        ax.set_ylabel("Δ metric (%) when k zeroed", fontsize=12)
        ax.set_title(f"{skey} — Leave-One-Out Impact", fontsize=11)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(rank))
        ax.grid(True, axis="y", alpha=0.3)

        # Legend for sensitivity quartiles
        if per_component_sensitivity is not None:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#d62728", label="Q4 (highest |g_k|)"),
                Patch(facecolor="#ff7f0e", label="Q3"),
                Patch(facecolor="#2ca02c", label="Q2"),
                Patch(facecolor="#1f77b4", label="Q1 (lowest |g_k|)"),
            ]
            ax.legend(handles=legend_elements, loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(plot_dir / f"exp1_loo_bar_{model_dir}_{task}.pdf", dpi=150)
        plt.close(fig)

        setting_result = {
            "model_dir": model_dir,
            "task": task,
            "baseline": float(baseline),
            "loo_delta": {int(k): float(v) for k, v in enumerate(loo_delta)},
            "loo_importance": {int(k): float(v) for k, v in enumerate(loo_importance)},
            "sensitivity": {int(k): float(v) for k, v in enumerate(per_component_sensitivity)} if per_component_sensitivity is not None else None,
            "correlation": corr_result,
        }
        all_results.append(setting_result)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Sensitivity vs Leave-One-Out Correlation")
    print("=" * 80)
    print(f"{'Setting':<40} {'Spearman ρ':>12} {'p-value':>10} {'Interpretation':<25}")
    print("-" * 90)
    for r in all_results:
        skey = f"{r['model_dir']}/{r['task']}"
        if r["correlation"]:
            rho = r["correlation"]["spearman_rho"]
            pval = r["correlation"]["spearman_p"]
            if pval < 0.05 and rho > 0.5:
                interp = "SIGNIFICANT positive"
            elif pval < 0.05 and rho < -0.5:
                interp = "SIGNIFICANT negative"
            elif abs(rho) > 0.3:
                interp = f"Weak trend (p={pval:.3f})"
            else:
                interp = "No correlation"
            print(f"{skey:<40} {rho:>12.4f} {pval:>10.4f} {interp:<25}")
        else:
            print(f"{skey:<40} {'N/A':>12} {'N/A':>10} {'No sensitivity data':<25}")

    # Combined scatter plot
    if any(r["correlation"] is not None for r in all_results):
        fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4.5), squeeze=False)
        for idx, r in enumerate(all_results):
            ax = axes[0, idx]
            skey = f"{r['model_dir'].split('-')[-1]}/{r['task']}"
            if r["sensitivity"] is not None and r["correlation"] is not None:
                S = np.array([r["sensitivity"][k] for k in sorted(r["sensitivity"])])
                I = np.array([r["loo_importance"][k] for k in sorted(r["loo_importance"])])
                rho = r["correlation"]["spearman_rho"]
                pval = r["correlation"]["spearman_p"]
                ax.scatter(S, I, s=60, edgecolors="black", linewidths=0.5)
                for k_idx in range(len(S)):
                    ax.annotate(str(k_idx), (S[k_idx], I[k_idx]), fontsize=7, ha="center", va="bottom")
                ax.set_title(f"{skey}\nρ={rho:.3f} (p={pval:.3f})", fontsize=10)
            else:
                ax.set_title(f"{skey}\nNo data", fontsize=10)
            ax.set_xlabel("|g_k|", fontsize=10)
            if idx == 0:
                ax.set_ylabel("LOO Importance", fontsize=10)
            ax.axhline(0, color="gray", ls="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "exp1_loo_combined_scatter.pdf", dpi=150)
        plt.close(fig)

    # Save analysis
    with open(analysis_dir / "loo_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[Saved] Analysis to {analysis_dir / 'loo_analysis.json'}")
    print(f"[Saved] Plots to {plot_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Component-wise Leave-One-Out")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["create", "eval", "sensitivity", "analyze", "all"],
                        help="Which phase to run")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--max_parallel", type=int, default=6,
                        help="Max parallel evals")
    parser.add_argument("--filter_setting", type=str, default=None,
                        help="Only process this setting (e.g., Qwen-Qwen3-8B/math)")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    if args.phase == "create" or args.phase == "all":
        phase1_create_edited_adapters()

    if args.phase == "sensitivity" or args.phase == "all":
        phase3_sensitivity(gpus)

    if args.phase == "eval" or args.phase == "all":
        phase2_evaluate(gpus, args.max_parallel, args.filter_setting)

    if args.phase == "analyze" or args.phase == "all":
        phase4_analyze()


if __name__ == "__main__":
    main()
