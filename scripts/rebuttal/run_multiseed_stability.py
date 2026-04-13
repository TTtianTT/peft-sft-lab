#!/usr/bin/env python3
"""
P0-C: Multi-Seed Stability Experiment.

Runs all methods (spectral edit heuristics, continued FT, sigma-only)
across multiple seeds for all (model, task) pairs. Aggregates results
with mean/std/win-rate statistics.

Methods:
  - baseline: Original adapter (no edit)
  - random_index, smooth_abs, abs_select, grad_direction: Spectral heuristics
  - continued_ft: Continued LoRA fine-tuning on calibration data
  - sigma_only: Sigma-only optimization

Usage:
    python scripts/rebuttal/run_multiseed_stability.py \
        --runs_root /path/to/runs_refactor_data_20260121 \
        --out_root outputs/rebuttal_exp/raw/multiseed \
        --n_repeats 5 \
        --calib_samples 128 \
        --tasks math code alpaca csqa
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# Constants
# ============================================================================

BASE_MODEL_DIR_TO_ID = {
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
}

TASK_TO_CALIB_DATASET = {
    "math": ("gsm8k", "main"),
    "code": ("ise-uiuc/Magicoder-Evol-Instruct-110K", None),
    "alpaca": ("tatsu-lab/alpaca", None),
    "csqa": ("tau/commonsense_qa", None),
}

TASK_ALIASES = {
    "metamath": "math",
    "magicoder": "code",
    "general": "alpaca",
    "commonsense_qa": "csqa",
}

# Spectral edit methods
SPECTRAL_METHODS = ["random_index", "smooth_abs", "abs_select", "grad_direction"]
METHOD_TO_MODE = {
    "random_index": "random_index",
    "smooth_abs": "smooth_abs",
    "abs_select": "abs_select",
    "grad_direction": "gd",
}

# All methods including baselines
ALL_METHODS = ["baseline"] + SPECTRAL_METHODS + ["continued_ft", "sigma_only"]

# lm_eval task mapping
TASK_DIR_TO_LM_EVAL = {
    "math": "gsm8k",
    "code": "humaneval",
    "alpaca": "ifeval",
    "csqa": "commonsense_qa",
}

TASK_CONFIGS = {
    "math": {"num_fewshot": 5, "gen_kwargs": "temperature=0,top_p=1", "gpu_mem": 0.95},
    "code": {"num_fewshot": 0, "gen_kwargs": "temperature=0,top_p=1", "gpu_mem": 0.90},
    "alpaca": {"num_fewshot": None, "gen_kwargs": "max_gen_toks=2048,temperature=0,top_p=1", "gpu_mem": 0.95},
    "csqa": {"num_fewshot": 0, "gen_kwargs": None, "gpu_mem": 0.85},
}


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class AdapterInfo:
    adapter_dir: Path
    task: str
    base_model_dir: str
    base_model_id: str
    profile: str
    rank: str
    seed: str

    @property
    def run_id(self) -> str:
        return f"{self.base_model_dir}_{self.task}_{self.profile}_{self.rank}_{self.seed}"


@dataclass
class RunResult:
    timestamp: str
    base_model_id: str
    base_model_dir: str
    task: str
    adapter_dir: str
    method: str
    repeat_seed: int
    repeat_idx: int
    calib_samples: int
    metric_name: str
    metric_value: Optional[float]
    edited_adapter_dir: Optional[str]
    eval_output_dir: str
    error: Optional[str] = None
    # Continued FT / sigma-only specific
    steps: Optional[int] = None
    lr: Optional[float] = None
    loss_initial: Optional[float] = None
    loss_final: Optional[float] = None


# ============================================================================
# Adapter discovery
# ============================================================================

def discover_adapters(
    runs_root: Path,
    tasks: Optional[List[str]] = None,
) -> List[AdapterInfo]:
    """Discover final LoRA adapters (no checkpoints, lora only)."""
    adapters = []
    for root, dirs, files in os.walk(runs_root):
        root_path = Path(root)
        if "checkpoint-" in str(root_path):
            dirs.clear()
            continue
        has_weights = "adapter_model.safetensors" in files or "adapter_model.bin" in files
        has_config = "adapter_config.json" in files
        if not (has_weights and has_config):
            continue

        # Parse path: {runs_root}/{base_model_dir}/{task}/lora/profile-{}/rank-{}/seed{}/
        try:
            rel = root_path.relative_to(runs_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 6:
            continue
        base_model_dir = parts[0]
        task = parts[1]
        peft_method = parts[2]
        if peft_method != "lora":
            continue

        # Normalize task
        task = TASK_ALIASES.get(task, task)
        if tasks and task not in tasks:
            continue

        base_model_id = BASE_MODEL_DIR_TO_ID.get(base_model_dir)
        if not base_model_id:
            continue

        profile = parts[3].replace("profile-", "") if parts[3].startswith("profile-") else parts[3]
        rank = parts[4].replace("rank-", "") if parts[4].startswith("rank-") else parts[4]
        seed = parts[5].replace("seed", "") if parts[5].startswith("seed") else parts[5]

        adapters.append(AdapterInfo(
            adapter_dir=root_path,
            task=task,
            base_model_dir=base_model_dir,
            base_model_id=base_model_id,
            profile=profile,
            rank=rank,
            seed=seed,
        ))

    return adapters


# ============================================================================
# Results writer (with resume support)
# ============================================================================

class ResultsWriter:
    def __init__(self, out_root: Path):
        self.out_root = out_root
        self.jsonl_path = out_root / "results.jsonl"
        self.csv_path = out_root / "results.csv"
        out_root.mkdir(parents=True, exist_ok=True)
        self.existing_keys: set = set()
        self._load_existing()

    def _key(self, adapter_dir: str, method: str, repeat_seed: int) -> str:
        return f"{adapter_dir}|{method}|{repeat_seed}"

    def _load_existing(self):
        if self.jsonl_path.exists():
            try:
                with open(self.jsonl_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        key = self._key(rec["adapter_dir"], rec["method"], rec["repeat_seed"])
                        self.existing_keys.add(key)
            except Exception as e:
                print(f"[WARN] Failed to load existing: {e}")

    def is_done(self, adapter_dir: str, method: str, repeat_seed: int) -> bool:
        return self._key(adapter_dir, method, repeat_seed) in self.existing_keys

    def write(self, result: RunResult):
        key = self._key(result.adapter_dir, result.method, result.repeat_seed)
        if key in self.existing_keys:
            return
        self.existing_keys.add(key)
        d = asdict(result)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(d) + "\n")
        exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(d.keys()))
            if not exists:
                w.writeheader()
            w.writerow(d)


# ============================================================================
# Spectral edit (existing CLI)
# ============================================================================

def run_spectral_edit(
    adapter_dir: Path,
    out_dir: Path,
    method: str,
    base_model_id: str,
    calib_samples: int,
    calib_batch_size: int,
    calib_dataset: str,
    calib_config: Optional[str],
    seed: int,
    dry_run: bool = False,
    gradient_checkpointing: bool = False,
) -> Tuple[bool, Optional[str]]:
    mode = METHOD_TO_MODE[method]
    cmd = [
        sys.executable, "-m", "finetune.spectral_edit.cli", "edit",
        "--base_model", base_model_id,
        "--lora_path", str(adapter_dir),
        "--out_dir", str(out_dir),
        "--mode", mode,
        "--target_modules", "down_proj", "o_proj",
        "--calib_samples", str(calib_samples),
        "--calib_batch_size", str(calib_batch_size),
        "--calib_dataset", calib_dataset,
        "--seed", str(seed),
        "--calib_seed", str(seed),
        "--calib_shuffle",
        "--grad_norm", "mean_abs",
        "--preserve_energy", "l1",
    ]
    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if calib_config:
        cmd.extend(["--calib_config", calib_config])

    # Method-specific args
    if method in ["random_index", "abs_select"]:
        cmd.extend(["--core_frac", "0.2", "--noise_frac", "0.2",
                     "--amp_factor", "1.25", "--sup_factor", "0.80"])
    elif method == "smooth_abs":
        cmd.extend(["--smooth_temperature", "0.35", "--smooth_center_q", "0.5",
                     "--amp_factor", "1.25", "--sup_factor", "0.80"])
    elif method == "grad_direction":
        cmd.extend(["--update_mode", "multiplicative", "--asymmetric_update",
                     "--eta_suppress", "2.0", "--eta_enhance", "0.2"])

    if dry_run:
        print(f"  [DRY] {' '.join(cmd[:10])}...")
        return True, None

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}:{env.get('PYTHONPATH', '')}"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                                env=env, cwd=REPO_ROOT)
        if result.returncode != 0:
            return False, result.stderr[-2000:] if result.stderr else "Unknown"
        return True, None
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


# ============================================================================
# Continued FT / Sigma-only (new scripts)
# ============================================================================

def run_continued_ft(
    adapter_dir: Path,
    out_dir: Path,
    base_model_id: str,
    task: str,
    calib_samples: int,
    calib_batch_size: int,
    steps: int,
    lr: float,
    seed: int,
    dry_run: bool = False,
    gradient_checkpointing: bool = False,
    proxy_samples: int = 128,
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_continued_lora_ft.py"),
        "--base_model", base_model_id,
        "--adapter_dir", str(adapter_dir),
        "--out_dir", str(out_dir),
        "--task", task,
        "--calib_samples", str(calib_samples),
        "--calib_batch_size", str(calib_batch_size),
        "--steps", str(steps),
        "--lr", str(lr),
        "--seed", str(seed),
        "--calib_seed", str(seed),
        "--calib_shuffle",
        "--proxy_samples", str(proxy_samples),
    ]
    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")

    if dry_run:
        print(f"  [DRY] continued_ft: {' '.join(cmd[:8])}...")
        return True, None, None

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}:{env.get('PYTHONPATH', '')}"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                                env=env, cwd=REPO_ROOT)
        if result.returncode != 0:
            return False, result.stderr[-2000:] if result.stderr else "Unknown", None

        meta_path = out_dir / "continued_ft_meta.json"
        meta = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        return True, None, meta
    except subprocess.TimeoutExpired:
        return False, "Timeout", None
    except Exception as e:
        return False, str(e), None


def run_sigma_only(
    adapter_dir: Path,
    out_dir: Path,
    base_model_id: str,
    task: str,
    calib_samples: int,
    calib_batch_size: int,
    steps: int,
    lr: float,
    seed: int,
    preserve_energy: str = "none",
    dry_run: bool = False,
    gradient_checkpointing: bool = False,
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_sigma_only_train.py"),
        "--base_model", base_model_id,
        "--adapter_dir", str(adapter_dir),
        "--out_dir", str(out_dir),
        "--task", task,
        "--calib_samples", str(calib_samples),
        "--calib_batch_size", str(calib_batch_size),
        "--steps", str(steps),
        "--lr", str(lr),
        "--seed", str(seed),
        "--calib_seed", str(seed),
        "--calib_shuffle",
        "--preserve_energy", preserve_energy,
    ]
    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")

    if dry_run:
        print(f"  [DRY] sigma_only: {' '.join(cmd[:8])}...")
        return True, None, None

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}:{env.get('PYTHONPATH', '')}"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                                env=env, cwd=REPO_ROOT)
        if result.returncode != 0:
            return False, result.stderr[-2000:] if result.stderr else "Unknown", None

        meta_path = out_dir / "sigma_only_train_meta.json"
        meta = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        return True, None, meta
    except subprocess.TimeoutExpired:
        return False, "Timeout", None
    except Exception as e:
        return False, str(e), None


# ============================================================================
# Evaluation via lm_eval harness (merge + vLLM)
# ============================================================================

def merge_adapter(base_model_id: str, adapter_dir: Path, merged_dir: Path,
                  cache_dir: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Merge adapter into base model for vLLM evaluation."""
    import shutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        if merged_dir.exists():
            # Check if already merged
            if (merged_dir / "config.json").exists():
                return True, None
            shutil.rmtree(merged_dir)

        print(f"  [Merge] {adapter_dir.name} -> {merged_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map="cpu", cache_dir=cache_dir,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
        model.save_pretrained(str(merged_dir))

        tok = AutoTokenizer.from_pretrained(base_model_id, cache_dir=cache_dir)
        tok.save_pretrained(str(merged_dir))

        del model
        import gc
        gc.collect()
        return True, None
    except Exception as e:
        return False, str(e)


def run_lm_eval(
    model_path: str,
    task: str,
    output_dir: Path,
    tensor_parallel_size: int = 1,
    dry_run: bool = False,
) -> Tuple[Optional[float], Optional[str]]:
    """Run lm_eval on a merged model."""
    lm_eval_task = TASK_DIR_TO_LM_EVAL.get(task)
    if not lm_eval_task:
        return None, f"Unknown task: {task}"

    cfg = TASK_CONFIGS.get(task, {})
    gpu_mem = cfg.get("gpu_mem", 0.90)

    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", f"pretrained={model_path},tensor_parallel_size={tensor_parallel_size},dtype=auto,gpu_memory_utilization={gpu_mem}",
        "--tasks", lm_eval_task,
        "--batch_size", "auto",
        "--output_path", str(output_dir),
    ]

    if cfg.get("num_fewshot") is not None:
        cmd.extend(["--num_fewshot", str(cfg["num_fewshot"])])
    if cfg.get("gen_kwargs"):
        cmd.extend(["--gen_kwargs", cfg["gen_kwargs"]])
    if task == "code":
        cmd.append("--confirm_unsafe_code")

    if dry_run:
        print(f"  [DRY] lm_eval: {' '.join(cmd[:8])}...")
        return 0.0, None

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,
            start_new_session=True,
        )
        if result.returncode != 0:
            return None, result.stderr[-2000:] if result.stderr else "Unknown"

        # Parse result from output directory
        metric = _parse_lm_eval_result(output_dir, task)
        return metric, None
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


def _parse_lm_eval_result(output_dir: Path, task: str) -> Optional[float]:
    """Parse lm_eval results JSON to extract the primary metric."""
    lm_eval_task = TASK_DIR_TO_LM_EVAL[task]
    # lm_eval writes results to output_dir/<model_name>/results.json
    for json_path in output_dir.rglob("results.json"):
        try:
            with open(json_path) as f:
                data = json.load(f)
            results = data.get("results", {})
            task_results = results.get(lm_eval_task, {})
            # Try common metric keys
            for key in ["acc,none", "exact_match,strict-match", "prompt_level_strict_acc,none",
                        "pass@1,none", "acc_norm,none"]:
                if key in task_results:
                    return float(task_results[key])
            # Fallback: first numeric value
            for v in task_results.values():
                if isinstance(v, (int, float)):
                    return float(v)
        except Exception:
            continue
    return None


# ============================================================================
# Aggregation
# ============================================================================

def aggregate_results(results: List[Dict]) -> Dict[str, Dict]:
    """Group results by (model, task, method) and compute statistics."""
    groups: Dict[str, List[float]] = {}
    for r in results:
        if r.get("error") or r.get("metric_value") is None:
            continue
        key = f"{r['base_model_dir']}|{r['task']}|{r['method']}"
        groups.setdefault(key, [])
        groups[key].append(r["metric_value"])

    agg = {}
    for key, values in groups.items():
        parts = key.split("|")
        n = len(values)
        agg[key] = {
            "base_model_dir": parts[0],
            "task": parts[1],
            "method": parts[2],
            "n": n,
            "mean": statistics.mean(values) if n > 0 else None,
            "std": statistics.stdev(values) if n > 1 else 0.0,
            "median": statistics.median(values) if n > 0 else None,
            "min": min(values) if n > 0 else None,
            "max": max(values) if n > 0 else None,
            "values": values,
        }

    return agg


def compute_win_rates(agg: Dict, baseline_method: str = "baseline") -> Dict[str, Dict]:
    """Compute win rates: P(method > baseline) for each (model, task)."""
    win_rates = {}
    for key, stats in agg.items():
        model, task, method = stats["base_model_dir"], stats["task"], stats["method"]
        if method == baseline_method:
            continue
        bl_key = f"{model}|{task}|{baseline_method}"
        if bl_key not in agg:
            continue
        bl_values = agg[bl_key]["values"]
        method_values = stats["values"]
        if not bl_values or not method_values:
            continue
        bl_mean = statistics.mean(bl_values)
        wins = sum(1 for v in method_values if v > bl_mean)
        win_rates[key] = {
            **stats,
            "baseline_mean": bl_mean,
            "win_rate": wins / len(method_values),
            "n_wins": wins,
            "n_total": len(method_values),
        }
    return win_rates


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Stability (P0-C)")
    parser.add_argument("--runs_root", type=str,
                        default="/mnt/data/zailongtian/workspace/peft-sft-lab/runs_refactor_data_20260121")
    parser.add_argument("--out_root", type=str,
                        default="outputs/rebuttal_exp/raw/multiseed")
    parser.add_argument("--tasks", type=str, nargs="+", default=["math", "code", "alpaca", "csqa"])
    parser.add_argument("--methods", type=str, nargs="+", default=ALL_METHODS)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--calib_batch_size", type=int, default=2)

    # Continued FT hyperparams
    parser.add_argument("--ft_steps", type=int, default=50)
    parser.add_argument("--ft_lr", type=float, default=5e-5)

    # Sigma-only hyperparams
    parser.add_argument("--sigma_steps", type=int, default=50)
    parser.add_argument("--sigma_lr", type=float, default=5e-3)
    parser.add_argument("--sigma_preserve_energy", type=str, default="none")

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--skip_eval", action="store_true", help="Only edit, skip lm_eval")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)

    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    out_root = Path(args.out_root)

    # Resolve seeds
    if args.seeds:
        seeds = args.seeds
        assert len(seeds) == args.n_repeats
    else:
        seeds = [args.base_seed + i for i in range(args.n_repeats)]

    print(f"[Config] runs_root={runs_root}")
    print(f"[Config] out_root={out_root}")
    print(f"[Config] tasks={args.tasks}, methods={args.methods}")
    print(f"[Config] seeds={seeds}, calib_samples={args.calib_samples}")

    # Discover adapters
    adapters = discover_adapters(runs_root, args.tasks)
    print(f"[Discovery] Found {len(adapters)} LoRA adapters")
    for a in adapters:
        print(f"  {a.base_model_dir}/{a.task} rank={a.rank} seed={a.seed}")

    if not adapters:
        print("[Error] No adapters found!")
        return

    writer = ResultsWriter(out_root)

    total_jobs = len(adapters) * len(args.methods) * len(seeds)
    done = 0

    for adapter in adapters:
        for method in args.methods:
            for repeat_idx, repeat_seed in enumerate(seeds):
                done += 1

                # Check resume
                if writer.is_done(str(adapter.adapter_dir), method, repeat_seed):
                    print(f"[{done}/{total_jobs}] SKIP {adapter.base_model_dir}/{adapter.task} "
                          f"{method} seed={repeat_seed} (done)")
                    continue

                print(f"\n[{done}/{total_jobs}] {adapter.base_model_dir}/{adapter.task} "
                      f"{method} seed={repeat_seed}")

                tag = f"repeat_{repeat_idx:02d}_seed{repeat_seed}"
                edited_dir = out_root / "edited_adapters" / adapter.run_id / method / tag
                eval_dir = out_root / "eval_outputs" / adapter.run_id / method / tag

                error = None
                ft_meta = None

                if method == "baseline":
                    # No editing needed - use original adapter
                    edited_dir = adapter.adapter_dir
                elif method in SPECTRAL_METHODS:
                    ok, err = run_spectral_edit(
                        adapter_dir=adapter.adapter_dir,
                        out_dir=edited_dir,
                        method=method,
                        base_model_id=adapter.base_model_id,
                        calib_samples=args.calib_samples,
                        calib_batch_size=args.calib_batch_size,
                        calib_dataset=TASK_TO_CALIB_DATASET[adapter.task][0],
                        calib_config=TASK_TO_CALIB_DATASET[adapter.task][1],
                        seed=repeat_seed,
                        dry_run=args.dry_run,
                    )
                    if not ok:
                        error = err
                        print(f"  [ERROR] Spectral edit failed: {err}")
                elif method == "continued_ft":
                    ok, err, ft_meta = run_continued_ft(
                        adapter_dir=adapter.adapter_dir,
                        out_dir=edited_dir,
                        base_model_id=adapter.base_model_id,
                        task=adapter.task,
                        calib_samples=args.calib_samples,
                        calib_batch_size=args.calib_batch_size,
                        steps=args.ft_steps,
                        lr=args.ft_lr,
                        seed=repeat_seed,
                        dry_run=args.dry_run,
                    )
                    if not ok:
                        error = err
                        print(f"  [ERROR] Continued FT failed: {err}")
                elif method == "sigma_only":
                    ok, err, ft_meta = run_sigma_only(
                        adapter_dir=adapter.adapter_dir,
                        out_dir=edited_dir,
                        base_model_id=adapter.base_model_id,
                        task=adapter.task,
                        calib_samples=args.calib_samples,
                        calib_batch_size=args.calib_batch_size,
                        steps=args.sigma_steps,
                        lr=args.sigma_lr,
                        seed=repeat_seed,
                        preserve_energy=args.sigma_preserve_energy,
                        dry_run=args.dry_run,
                    )
                    if not ok:
                        error = err
                        print(f"  [ERROR] Sigma-only failed: {err}")

                # Evaluate
                metric_value = None
                metric_name = TASK_DIR_TO_LM_EVAL.get(adapter.task, adapter.task)

                if error is None and not args.skip_eval:
                    # Merge and evaluate
                    merged_dir = out_root / "merged_models" / adapter.run_id / method / tag
                    ok, err = merge_adapter(
                        adapter.base_model_id, edited_dir, merged_dir,
                        cache_dir=args.cache_dir,
                    )
                    if ok:
                        metric_value, err = run_lm_eval(
                            model_path=str(merged_dir),
                            task=adapter.task,
                            output_dir=eval_dir,
                            tensor_parallel_size=args.tensor_parallel_size,
                            dry_run=args.dry_run,
                        )
                        if err:
                            error = f"eval: {err}"
                    else:
                        error = f"merge: {err}"

                    # Cleanup merged model to save disk
                    import shutil
                    if merged_dir.exists() and method != "baseline":
                        shutil.rmtree(merged_dir, ignore_errors=True)

                result = RunResult(
                    timestamp=datetime.now().isoformat(),
                    base_model_id=adapter.base_model_id,
                    base_model_dir=adapter.base_model_dir,
                    task=adapter.task,
                    adapter_dir=str(adapter.adapter_dir),
                    method=method,
                    repeat_seed=repeat_seed,
                    repeat_idx=repeat_idx,
                    calib_samples=args.calib_samples,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    edited_adapter_dir=str(edited_dir) if method != "baseline" else None,
                    eval_output_dir=str(eval_dir),
                    error=error,
                    steps=args.ft_steps if method == "continued_ft" else (
                        args.sigma_steps if method == "sigma_only" else None),
                    lr=args.ft_lr if method == "continued_ft" else (
                        args.sigma_lr if method == "sigma_only" else None),
                    loss_initial=ft_meta.get("loss_initial") if ft_meta else None,
                    loss_final=ft_meta.get("loss_final") if ft_meta else None,
                )
                writer.write(result)

                if metric_value is not None:
                    print(f"  [Result] {method} seed={repeat_seed}: {metric_value:.4f}")
                elif error:
                    print(f"  [Error] {error[:100]}")

    # ---- Aggregate ----
    print("\n" + "=" * 60)
    print("Aggregating results...")
    all_results = []
    if writer.jsonl_path.exists():
        with open(writer.jsonl_path) as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    agg = aggregate_results(all_results)
    win_rates = compute_win_rates(agg)

    # Save aggregated results
    agg_path = out_root / "summary_aggregated.json"
    with open(agg_path, "w") as f:
        json.dump({"aggregated": agg, "win_rates": win_rates}, f, indent=2)

    # Print summary table
    print(f"\n{'Model':<30} {'Task':<8} {'Method':<20} {'Mean':>8} {'Std':>8} {'N':>4}")
    print("-" * 80)
    for key in sorted(agg.keys()):
        s = agg[key]
        mean_str = f"{s['mean']:.4f}" if s['mean'] is not None else "N/A"
        std_str = f"{s['std']:.4f}" if s['std'] is not None else "N/A"
        print(f"{s['base_model_dir']:<30} {s['task']:<8} {s['method']:<20} "
              f"{mean_str:>8} {std_str:>8} {s['n']:>4}")

    if win_rates:
        print(f"\n{'Model':<30} {'Task':<8} {'Method':<20} {'WinRate':>8} {'Wins':>6}")
        print("-" * 75)
        for key in sorted(win_rates.keys()):
            w = win_rates[key]
            print(f"{w['base_model_dir']:<30} {w['task']:<8} {w['method']:<20} "
                  f"{w['win_rate']:>8.2%} {w['n_wins']:>3}/{w['n_total']:<3}")

    print(f"\n[Done] Results: {writer.jsonl_path}")
    print(f"[Done] Summary: {agg_path}")


if __name__ == "__main__":
    main()
