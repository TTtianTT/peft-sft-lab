#!/usr/bin/env python3
"""
Experiment 3: Sensitivity Signal Stability

Test whether the sensitivity ranking |g_k| is STABLE across different
calibration subsets. For each (model, task):
  1. Draw 5 disjoint/random subsets of 128 calibration examples
  2. Compute g_k for each subset
  3. Compute pairwise Spearman rank correlation of |g_k| rankings
  4. Report mean pairwise correlation ± std
  5. Compute per-component CV across subsets

Requires GPU for gradient computation. One setting per GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "rebuttal_v2" / "exp3_signal_stability"
PYTHON = sys.executable

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
N_SUBSETS = 5
CALIB_SAMPLES_PER_SUBSET = 128


def run_sensitivity_for_subset(setting: dict, subset_idx: int, gpu_id: int) -> dict:
    """Compute sensitivity scores for one calibration subset."""
    model_dir = setting["model_dir"]
    task = setting["task"]
    base_model_id = setting["base_model_id"]
    lora_path = setting["lora_path"]
    calib_dataset = setting["calib_dataset"]
    calib_config = setting["calib_config"]

    out_file = OUT_ROOT / "raw" / f"{model_dir}_{task}_subset{subset_idx}.json"
    if out_file.exists():
        return {"status": "skip", "path": str(out_file)}

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Each subset uses a different calib_start offset (disjoint subsets)
    calib_start = subset_idx * CALIB_SAMPLES_PER_SUBSET

    script = f"""
import json, sys, torch, math
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
    A = A_cpu.to(device); B = B_cpu.to(device)
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
# Use shuffled dataset with different seed per subset to get diverse subsets
calib_examples = sample_calibration_examples(
    ds_split, {CALIB_SAMPLES_PER_SUBSET}, True, {subset_idx * 1000 + 42}, 0
)

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

results = {{}}
for prefix, spec in specs.items():
    g = HOOK_CTX.gsum.get(prefix, None)
    if g is None:
        continue
    g_abs = g.abs().cpu().tolist()
    li = layer_idx_from_module_prefix(prefix)
    results[prefix] = {{
        "layer": li,
        "suffix": prefix.split(".")[-1],
        "g_abs": g_abs,
    }}

with open("{str(out_file)}", "w") as f:
    json.dump(results, f, indent=2)
print(f"[Done] Saved subset {subset_idx} sensitivity to {str(out_file)}")
del model, base; torch.cuda.empty_cache()
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run([PYTHON, "-c", script],
                            capture_output=True, text=True, env=env, timeout=600)
    if result.returncode != 0:
        return {"status": "fail", "error": result.stderr[-300:]}
    return {"status": "ok", "path": str(out_file)}


def phase1_compute(gpus: list[int]):
    """Compute sensitivity for all settings × subsets."""
    print("=" * 80)
    print("PHASE 1: Computing sensitivity across calibration subsets")
    print("=" * 80)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Run sequentially per setting (each setting needs a full model load)
    # but we can parallelize subsets across GPUs
    for setting in SETTINGS:
        model_dir = setting["model_dir"]
        task = setting["task"]
        print(f"\n--- {model_dir}/{task} ---")

        for subset_idx in range(N_SUBSETS):
            gpu_id = gpus[subset_idx % len(gpus)]
            print(f"  Subset {subset_idx} on GPU {gpu_id}...", end=" ", flush=True)
            result = run_sensitivity_for_subset(setting, subset_idx, gpu_id)
            print(result["status"])


def phase2_analyze():
    """Analyze pairwise rank correlations across subsets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    print("=" * 80)
    print("PHASE 2: Analyzing signal stability")
    print("=" * 80)

    plot_dir = OUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = OUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for setting in SETTINGS:
        model_dir = setting["model_dir"]
        task = setting["task"]
        skey = f"{model_dir}/{task}"

        # Load all subsets
        subsets = {}
        for subset_idx in range(N_SUBSETS):
            path = OUT_ROOT / "raw" / f"{model_dir}_{task}_subset{subset_idx}.json"
            if path.exists():
                with open(path) as f:
                    subsets[subset_idx] = json.load(f)

        if len(subsets) < 2:
            print(f"  [SKIP] {skey} — need at least 2 subsets, have {len(subsets)}")
            continue

        print(f"\n--- {skey} ({len(subsets)} subsets) ---")

        # Get common modules
        all_modules = set.intersection(*[set(s.keys()) for s in subsets.values()])
        rank = len(list(subsets.values())[0][list(all_modules)[0]]["g_abs"])

        # Per-module pairwise Spearman correlations
        module_pairwise_rhos = defaultdict(list)
        module_per_component_g = defaultdict(lambda: np.zeros((N_SUBSETS, rank)))

        for module in sorted(all_modules):
            for subset_idx, sdata in subsets.items():
                g_abs = np.array(sdata[module]["g_abs"])
                module_per_component_g[module][subset_idx] = g_abs

            for i, j in combinations(sorted(subsets.keys()), 2):
                g_i = np.array(subsets[i][module]["g_abs"])
                g_j = np.array(subsets[j][module]["g_abs"])
                rho, _ = stats.spearmanr(g_i, g_j)
                module_pairwise_rhos[module].append(rho)

        # Aggregate: mean pairwise Spearman across all modules
        all_rhos = []
        for module, rhos in module_pairwise_rhos.items():
            all_rhos.extend(rhos)

        mean_rho = np.mean(all_rhos)
        std_rho = np.std(all_rhos)

        print(f"  Mean pairwise Spearman ρ = {mean_rho:.4f} ± {std_rho:.4f} (N_pairs={len(all_rhos)})")

        # Per-component CV across subsets (aggregated across modules)
        agg_g_per_subset = np.zeros((len(subsets), rank))
        for module in all_modules:
            for idx, subset_idx in enumerate(sorted(subsets.keys())):
                g_abs = np.array(subsets[subset_idx][module]["g_abs"])
                agg_g_per_subset[idx] += g_abs
        agg_g_per_subset /= len(all_modules)

        per_component_cv = np.std(agg_g_per_subset, axis=0) / np.maximum(np.mean(agg_g_per_subset, axis=0), 1e-8)

        print(f"  Per-component CV: min={per_component_cv.min():.4f}, max={per_component_cv.max():.4f}, mean={per_component_cv.mean():.4f}")

        # Heatmap: layer × component, colored by per-module per-component CV
        layers = sorted(set(
            subsets[list(subsets.keys())[0]][m].get("layer", 0)
            for m in all_modules if subsets[list(subsets.keys())[0]][m].get("layer") is not None
        ))
        suffixes = sorted(set(
            subsets[list(subsets.keys())[0]][m]["suffix"] for m in all_modules
        ))

        for suffix in suffixes:
            suffix_modules = [m for m in all_modules
                              if subsets[list(subsets.keys())[0]][m]["suffix"] == suffix]
            if not suffix_modules:
                continue

            layer_to_cv = {}
            for m in suffix_modules:
                layer = subsets[list(subsets.keys())[0]][m].get("layer", 0)
                g_matrix = module_per_component_g[m][:len(subsets)]
                cv = np.std(g_matrix, axis=0) / np.maximum(np.mean(g_matrix, axis=0), 1e-8)
                layer_to_cv[layer] = cv

            if not layer_to_cv:
                continue

            sorted_layers = sorted(layer_to_cv.keys())
            heatmap_data = np.array([layer_to_cv[l] for l in sorted_layers])

            fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_layers) * 0.2)))
            im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", vmin=0)
            ax.set_xlabel("Component k", fontsize=11)
            ax.set_ylabel("Layer", fontsize=11)
            ax.set_title(f"{skey} / {suffix}\nCV of |g_k| across {len(subsets)} subsets", fontsize=10)
            ax.set_xticks(range(rank))
            ax.set_yticks(range(len(sorted_layers)))
            ax.set_yticklabels(sorted_layers, fontsize=7)
            fig.colorbar(im, ax=ax, label="CV")
            fig.tight_layout()
            fig.savefig(plot_dir / f"exp3_cv_heatmap_{model_dir}_{task}_{suffix}.pdf", dpi=150)
            plt.close(fig)

        setting_result = {
            "model_dir": model_dir,
            "task": task,
            "n_subsets": len(subsets),
            "n_modules": len(all_modules),
            "rank": rank,
            "mean_pairwise_spearman": float(mean_rho),
            "std_pairwise_spearman": float(std_rho),
            "n_pairs": len(all_rhos),
            "per_component_cv_mean": float(per_component_cv.mean()),
            "per_component_cv_min": float(per_component_cv.min()),
            "per_component_cv_max": float(per_component_cv.max()),
            "interpretation": (
                "Highly stable" if mean_rho > 0.8 else
                "Moderately stable" if mean_rho > 0.5 else
                "Unstable"
            ),
        }
        all_results.append(setting_result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Signal Stability")
    print("=" * 80)
    print(f"{'Setting':<40} {'Mean ρ':>10} {'Std ρ':>10} {'Interpretation':<20}")
    print("-" * 85)
    for r in all_results:
        skey = f"{r['model_dir']}/{r['task']}"
        print(f"{skey:<40} {r['mean_pairwise_spearman']:>10.4f} {r['std_pairwise_spearman']:>10.4f} {r['interpretation']:<20}")

    with open(analysis_dir / "stability_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[Saved] Analysis and plots")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Sensitivity Signal Stability")
    parser.add_argument("--phase", type=str, required=True, choices=["compute", "analyze", "all"])
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    if args.phase in ("compute", "all"):
        phase1_compute(gpus)
    if args.phase in ("analyze", "all"):
        phase2_analyze()


if __name__ == "__main__":
    main()
