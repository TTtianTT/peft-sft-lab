#!/usr/bin/env python3
"""
Driver script for running repeated spectral editing experiments on LoRA/LoRA+ adapters.

This script:
1. Discovers LoRA/LoRA+ adapters under one or more runs roots
2. Applies spectral editing methods from finetune.spectral_edit (repeated seeds)
3. Evaluates on task-specific benchmarks using peft-sft-lab evaluators
4. Saves per-repeat results incrementally and aggregates mean/std across repeats

Methods:
  - baseline: Original adapter (no edit)
  - random_index: Randomly select singular values to scale
  - smooth_abs: Select by |gradient| with smooth sigmoid scaling
  - abs_select: Select by |gradient| with hard/non-smooth scaling
  - grad_direction: Edit based on gradient direction/sign (gd mode)

Tasks:
  - metamath → GSM8K (5-shot, greedy, strict-match accuracy)
  - magicoder → HumanEval (0-shot, pass@1)
  - alpaca → IFEval (strict accuracy)
  - csqa → CommonsenseQA (single-letter accuracy)

Usage:
    python scripts/run_refactor_lora_spectral_edit_repeat.py \
        --out_root /path/to/output \
        --use_vllm \
        --tensor_parallel_size 8 \
        --n_repeats 5
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add peft-sft-lab src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PEFT_SFT_LAB_ROOT = SCRIPT_DIR.parent
SRC_DIR = PEFT_SFT_LAB_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# Constants
# ============================================================================

DEFAULT_RUNS_ROOTS = [
    Path("/home/zailongtian/workspace/peft-sft-lab/runs_refactor_data_20260121/meta-llama-Llama-3.1-8B"),
    Path("/home/zailongtian/workspace/peft-sft-lab/runs_refactor_data_20260121/Qwen-Qwen3-8B"),
]

EDIT_METHODS = ["baseline", "random_index", "smooth_abs", "abs_select", "grad_direction"]

# Map method names to spectral edit modes
METHOD_TO_MODE = {
    "random_index": "random_index",
    "smooth_abs": "smooth_abs",
    "abs_select": "abs_select",
    "grad_direction": "gd",
}

TASK_TO_EVAL_SCRIPT = {
    "metamath": "finetune.eval.eval_gsm8k",
    "magicoder": "finetune.eval.eval_humaneval",
    "alpaca": "finetune.eval.eval_ifeval",
    "csqa": "finetune.eval.eval_csqa",
}

TASK_TO_METRIC = {
    "metamath": "accuracy_strict",
    "magicoder": "pass@1",
    "alpaca": "ifeval_strict_accuracy",
    "csqa": "accuracy",
}

ALLOWED_PEFT_METHODS = {"lora", "loraplus"}

# Base model directory to HuggingFace ID mapping
BASE_MODEL_DIR_TO_ID = {
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "meta-llama-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama-Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "mistralai-Mistral-7B-v0.3": "mistralai/Mistral-7B-v0.3",
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
}

# Task name aliases
TASK_ALIASES = {
    "math": "metamath",
    "code": "magicoder",
    "general": "alpaca",
    "commonsenseqa": "csqa",
}


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class AdapterInfo:
    """Information about a discovered adapter."""
    adapter_dir: Path
    task: str
    method: str  # peft method: lora, loraplus
    profile: str
    rank: str
    seed: str
    base_model_dir: str
    base_model_id: str

    @property
    def run_id(self) -> str:
        """Generate a unique run ID for this adapter."""
        return sanitize_tag(
            f"{self.base_model_dir}_{self.task}_{self.method}_{self.profile}_{self.rank}_{self.seed}"
        )


@dataclass
class EvalResult:
    """Result from a single evaluation repeat."""
    timestamp: str
    base_model_id: str
    base_model_dir: str
    task: str
    adapter_dir: str
    edited_adapter_dir: Optional[str]
    eval_output_dir: str
    edit_method: str
    peft_method: str
    profile: str
    rank: str
    seed: str
    run_id: str
    repeat_idx: int
    repeat_seed: int
    repeat_tag: str
    metric_name: str
    metric_value: float
    num_examples: int
    backend: str
    tensor_parallel_size: int
    git_commit: Optional[str]
    error: Optional[str] = None


# ============================================================================
# Utility functions
# ============================================================================

def sanitize_tag(text: str) -> str:
    text = text.strip().replace(os.sep, "-")
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in text).strip("-")


def get_git_commit() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=PEFT_SFT_LAB_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def normalize_task(task: str) -> str:
    """Normalize task name to canonical form."""
    task_lower = task.lower().strip()
    return TASK_ALIASES.get(task_lower, task_lower)


def parse_adapter_path(adapter_dir: Path, runs_root: Path) -> Optional[AdapterInfo]:
    """
    Parse adapter directory path to extract metadata.

    Handles two structures:
        1. {runs_root}/{task}/{peft_method}/profile-{profile}/rank-{rank}/seed{seed}/
           (when runs_root already includes base_model_dir)
        2. {runs_root}/{base_model_dir}/{task}/{peft_method}/profile-{profile}/rank-{rank}/seed{seed}/
    """
    try:
        rel_path = adapter_dir.relative_to(runs_root)
        parts = rel_path.parts
    except ValueError:
        return None

    base_model_dir = None
    task = None
    peft_method = None
    profile = None
    rank = None
    seed = None

    runs_root_name = runs_root.name
    if runs_root_name in BASE_MODEL_DIR_TO_ID or runs_root_name.startswith("meta-llama") or \
       runs_root_name.startswith("mistralai") or runs_root_name.startswith("Qwen"):
        base_model_dir = runs_root_name
        for i, part in enumerate(parts):
            part_lower = part.lower()
            if i == 0:
                task = normalize_task(part)
            elif i == 1:
                peft_method = part_lower
            elif part_lower.startswith("profile-"):
                profile = part[8:]
            elif part_lower.startswith("rank-"):
                rank = part[5:]
            elif part_lower.startswith("seed"):
                seed = part[4:]
    else:
        for i, part in enumerate(parts):
            part_lower = part.lower()
            if i == 0:
                base_model_dir = part
            elif i == 1:
                task = normalize_task(part)
            elif i == 2:
                peft_method = part_lower
            elif part_lower.startswith("profile-"):
                profile = part[8:]
            elif part_lower.startswith("rank-"):
                rank = part[5:]
            elif part_lower.startswith("seed"):
                seed = part[4:]

    base_model_id = None
    if base_model_dir:
        base_model_id = BASE_MODEL_DIR_TO_ID.get(base_model_dir)

    # Try to load from adapter_config.json if not found
    if not base_model_id:
        config_path = adapter_dir / "adapter_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                base_model_id = cfg.get("base_model_name_or_path")
            except Exception:
                pass

    # Fallback: try run_args.json
    run_args_path = adapter_dir / "run_args.json"
    if run_args_path.exists():
        try:
            with open(run_args_path) as f:
                run_args = json.load(f)
            if not base_model_id:
                base_model_id = run_args.get("base_model")
            if not task:
                task = normalize_task(run_args.get("task", ""))
            if not peft_method:
                peft_method = run_args.get("peft_method")
            if not profile:
                profile = run_args.get("train_profile", "unknown")
            if not rank:
                rank = str(run_args.get("r", "unknown"))
            if not seed:
                seed = str(run_args.get("seed", "unknown"))
        except Exception:
            pass

    if not task:
        return None

    return AdapterInfo(
        adapter_dir=adapter_dir,
        task=task,
        method=(peft_method or "lora").lower(),
        profile=profile or "unknown",
        rank=rank or "unknown",
        seed=seed or "unknown",
        base_model_dir=base_model_dir or "unknown",
        base_model_id=base_model_id or f"unknown/{base_model_dir or 'model'}",
    )


def discover_adapters(
    runs_root: Path,
    tasks: Optional[List[str]] = None,
    peft_methods: Optional[set[str]] = None,
) -> Tuple[List[AdapterInfo], int]:
    """
    Recursively discover adapter directories.

    An adapter directory must contain BOTH:
      - adapter_model.safetensors (or adapter_model.bin)
      - adapter_config.json

    Excludes any path containing "/checkpoint-" anywhere.
    """
    adapters = []
    checkpoint_count = 0

    for root, dirs, files in os.walk(runs_root):
        root_path = Path(root)

        # Skip checkpoint directories
        if "/checkpoint-" in str(root_path) or root_path.name.startswith("checkpoint-"):
            checkpoint_count += 1
            dirs.clear()
            continue

        has_safetensors = "adapter_model.safetensors" in files
        has_bin = "adapter_model.bin" in files
        has_config = "adapter_config.json" in files

        if (has_safetensors or has_bin) and has_config:
            info = parse_adapter_path(root_path, runs_root)
            if not info:
                continue
            if tasks and info.task not in tasks:
                continue
            if peft_methods and info.method not in peft_methods:
                continue
            adapters.append(info)

    return adapters, checkpoint_count


def has_adapter_weights(adapter_dir: Path) -> bool:
    return (
        (adapter_dir / "adapter_model.safetensors").exists()
        or (adapter_dir / "adapter_model.bin").exists()
    )


def build_repeat_tag(repeat_idx: int, repeat_seed: int) -> str:
    return f"repeat_{repeat_idx:02d}_seed{repeat_seed}"


def resolve_repeat_seeds(base_seed: int, n_repeats: int, seeds: Optional[List[int]]) -> List[int]:
    if n_repeats < 1:
        raise ValueError("--n_repeats must be >= 1")
    if seeds:
        if len(seeds) != n_repeats:
            raise ValueError("--seeds length must match --n_repeats")
        if len(set(seeds)) != len(seeds):
            raise ValueError("--seeds must be distinct")
        return seeds
    return [base_seed + i for i in range(n_repeats)]


def load_metrics_if_exists(eval_output_dir: Path) -> Optional[Dict[str, Any]]:
    metrics_path = eval_output_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return None


# ============================================================================
# Spectral editing
# ============================================================================

def run_spectral_edit(
    adapter_dir: Path,
    out_dir: Path,
    edit_method: str,
    base_model_id: str,
    seed: int = 42,
    calib_samples: int = 32,
    calib_batch_size: int = 2,
    target_modules: List[str] = None,
    calib_dataset: Optional[str] = None,
    calib_config: Optional[str] = None,
    calib_split: Optional[str] = None,
    calib_text_fields: Optional[List[str]] = None,
    calib_shuffle: bool = False,
    calib_seed: Optional[int] = None,
    calib_start: int = 0,
    dry_run: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Run spectral editing on an adapter using finetune.spectral_edit.
    """
    if target_modules is None:
        target_modules = ["down_proj", "o_proj"]

    mode = METHOD_TO_MODE.get(edit_method)
    if not mode:
        return False, f"Unknown edit method: {edit_method}"

    cmd = [
        sys.executable, "-m", "finetune.spectral_edit.cli", "edit",
        "--base_model", base_model_id,
        "--lora_path", str(adapter_dir),
        "--out_dir", str(out_dir),
        "--mode", mode,
        "--target_modules", *target_modules,
        "--calib_samples", str(calib_samples),
        "--calib_batch_size", str(calib_batch_size),
        "--seed", str(seed),
        "--grad_norm", "mean_abs",
        "--preserve_energy", "l1",
    ]

    if calib_dataset:
        cmd.extend(["--calib_dataset", calib_dataset])
    if calib_config is not None:
        cmd.extend(["--calib_config", calib_config])
    if calib_split:
        cmd.extend(["--calib_split", calib_split])
    if calib_text_fields:
        cmd.extend(["--calib_text_fields", *calib_text_fields])
    if calib_shuffle:
        cmd.append("--calib_shuffle")
    if calib_seed is not None:
        cmd.extend(["--calib_seed", str(calib_seed)])
    if calib_start:
        cmd.extend(["--calib_start", str(calib_start)])

    if edit_method in ["random_index", "abs_select"]:
        cmd.extend([
            "--core_frac", "0.2",
            "--noise_frac", "0.2",
            "--amp_factor", "1.25",
            "--sup_factor", "0.80",
        ])
    elif edit_method == "smooth_abs":
        cmd.extend([
            "--smooth_temperature", "0.35",
            "--smooth_center_q", "0.5",
            "--amp_factor", "1.25",
            "--sup_factor", "0.80",
        ])
    elif edit_method == "grad_direction":
        cmd.extend([
            "--update_mode", "multiplicative",
            "--asymmetric_update",
            "--eta_suppress", "2.0",
            "--eta_enhance", "0.2",
        ])

    if dry_run:
        print(f"  [DRY-RUN] Would run: {' '.join(cmd[:10])}...")
        return True, None

    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{SRC_DIR}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(SRC_DIR)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
            cwd=PEFT_SFT_LAB_ROOT,
        )

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "Unknown error"
            return False, f"Edit failed (code {result.returncode}): {error_msg}"

        if not has_adapter_weights(out_dir):
            return False, "Edit completed but no adapter weights found in output"

        return True, None

    except subprocess.TimeoutExpired:
        return False, "Edit timed out after 30 minutes"
    except Exception as e:
        return False, f"Edit failed with exception: {e}"


# ============================================================================
# Evaluation
# ============================================================================

def run_evaluation(
    base_model_id: str,
    adapter_dir: Optional[Path],
    task: str,
    output_dir: Path,
    use_vllm: bool = True,
    tensor_parallel_size: int = 8,
    max_samples: Optional[int] = None,
    seed: int = 42,
    dry_run: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Run evaluation on a task using peft-sft-lab evaluators.
    """
    eval_module = TASK_TO_EVAL_SCRIPT.get(task)
    if not eval_module:
        return None, f"Unknown task: {task}"

    cmd = [
        sys.executable, "-m", eval_module,
        "--base_model", base_model_id,
        "--output_dir", str(output_dir),
        "--seed", str(seed),
    ]

    if adapter_dir:
        cmd.extend(["--adapter_dir", str(adapter_dir)])

    if use_vllm:
        cmd.append("--use_vllm")
        cmd.extend(["--tensor_parallel_size", str(tensor_parallel_size)])

    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])

    if task == "metamath":
        cmd.extend(["--max_new_tokens", "256"])
    elif task == "magicoder":
        cmd.extend(["--max_new_tokens", "256", "--timeout_s", "3.0"])
    elif task == "alpaca":
        cmd.extend(["--max_new_tokens", "256", "--split", "train"])
    elif task == "csqa":
        cmd.extend(["--max_new_tokens", "8"])

    if dry_run:
        print(f"  [DRY-RUN] Would run: {' '.join(cmd[:8])}...")
        return {"metric": 0.0, "dry_run": True}, None

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=PEFT_SFT_LAB_ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        )

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "Unknown error"
            return None, f"Eval failed (code {result.returncode}): {error_msg}"

        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            return None, "Eval completed but no metrics.json found"

        with open(metrics_path) as f:
            metrics = json.load(f)

        return metrics, None

    except subprocess.TimeoutExpired:
        return None, "Eval timed out after 2 hours"
    except Exception as e:
        return None, f"Eval failed with exception: {e}"


# ============================================================================
# Results saving and aggregation
# ============================================================================

class ResultsWriter:
    """Handles incremental writing of per-repeat results to JSONL and CSV."""

    def __init__(self, out_root: Path):
        self.out_root = out_root
        self.jsonl_path = out_root / "results.jsonl"
        self.csv_path = out_root / "results.csv"

        out_root.mkdir(parents=True, exist_ok=True)
        self.existing_keys: set = set()
        self._load_existing()

    def _result_key(self, adapter_dir: str, edit_method: str, task: str, repeat_seed: int) -> str:
        return f"{adapter_dir}|{edit_method}|{task}|{repeat_seed}"

    def _load_existing(self):
        if self.jsonl_path.exists():
            try:
                with open(self.jsonl_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        repeat_seed = rec.get("repeat_seed", "unknown")
                        key = self._result_key(
                            rec.get("adapter_dir", ""),
                            rec.get("edit_method", ""),
                            rec.get("task", ""),
                            repeat_seed,
                        )
                        self.existing_keys.add(key)
            except Exception as e:
                print(f"[WARN] Failed to load existing results: {e}")

    def is_completed(self, adapter_dir: str, edit_method: str, task: str, repeat_seed: int) -> bool:
        key = self._result_key(adapter_dir, edit_method, task, repeat_seed)
        return key in self.existing_keys

    def write_result(self, result: EvalResult):
        key = self._result_key(result.adapter_dir, result.edit_method, result.task, result.repeat_seed)
        if key in self.existing_keys:
            return

        self.existing_keys.add(key)
        result_dict = asdict(result)

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(result_dict) + "\n")

        csv_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result_dict.keys()))
            if not csv_exists:
                writer.writeheader()
            writer.writerow(result_dict)


def load_results_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
    except Exception as e:
        print(f"[WARN] Failed to load results for aggregation: {e}")
    return records


def compute_mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mean_val = statistics.mean(values)
    std_val = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean_val, std_val


def aggregate_results(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, Any]]] = {}
    for rec in records:
        key = (
            rec.get("base_model_id", ""),
            rec.get("base_model_dir", ""),
            rec.get("task", ""),
            rec.get("peft_method", ""),
            rec.get("edit_method", ""),
            rec.get("adapter_dir", ""),
        )
        groups.setdefault(key, []).append(rec)

    summaries: List[Dict[str, Any]] = []
    for key, recs in groups.items():
        recs_sorted = sorted(
            recs,
            key=lambda r: (r.get("repeat_idx", 0), r.get("repeat_seed", 0)),
        )

        metric_name = recs_sorted[0].get("metric_name", "metric")
        repeat_entries = []
        valid_scores = []

        for rec in recs_sorted:
            metric_value = rec.get("metric_value")
            error = rec.get("error")
            repeat_entries.append({
                "repeat_idx": rec.get("repeat_idx"),
                "repeat_seed": rec.get("repeat_seed"),
                "metric_value": metric_value,
                "num_examples": rec.get("num_examples"),
                "error": error,
            })
            if error is None and isinstance(metric_value, (int, float)) and metric_value >= 0:
                valid_scores.append(metric_value)

        mean_val, std_val = compute_mean_std(valid_scores)
        summary = {
            "base_model_id": recs_sorted[0].get("base_model_id"),
            "base_model_dir": recs_sorted[0].get("base_model_dir"),
            "task": recs_sorted[0].get("task"),
            "adapter_type": recs_sorted[0].get("peft_method"),
            "edit_method": recs_sorted[0].get("edit_method"),
            "adapter_dir": recs_sorted[0].get("adapter_dir"),
            "run_id": recs_sorted[0].get("run_id"),
            "profile": recs_sorted[0].get("profile"),
            "rank": recs_sorted[0].get("rank"),
            "seed": recs_sorted[0].get("seed"),
            "metric_name": metric_name,
            "mean": mean_val,
            "std": std_val,
            "num_repeats": len(recs_sorted),
            "num_successful": len(valid_scores),
            "num_failed": len(recs_sorted) - len(valid_scores),
            "repeats": repeat_entries,
        }
        summaries.append(summary)

    return summaries


def write_aggregate_outputs(out_root: Path, summaries: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    summary_root = out_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    for summary in summaries:
        base_model_dir = summary.get("base_model_dir") or "unknown"
        task = summary.get("task") or "unknown"
        adapter_type = summary.get("adapter_type") or "unknown"
        edit_method = summary.get("edit_method") or "unknown"
        run_id = summary.get("run_id") or "unknown"

        summary_dir = summary_root / base_model_dir / task / adapter_type / edit_method
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"{sanitize_tag(run_id)}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    summary_all_jsonl = out_root / "summary_all.jsonl"
    with open(summary_all_jsonl, "w") as f:
        for summary in summaries:
            f.write(json.dumps(summary) + "\n")

    summary_all_csv = out_root / "summary_all.csv"
    rows = []
    for summary in summaries:
        repeat_seeds = [r.get("repeat_seed") for r in summary.get("repeats", [])]
        repeat_scores = [r.get("metric_value") for r in summary.get("repeats", [])]
        repeat_errors = [r.get("error") for r in summary.get("repeats", [])]

        flat = {k: summary.get(k) for k in [
            "base_model_id",
            "base_model_dir",
            "task",
            "adapter_type",
            "edit_method",
            "adapter_dir",
            "run_id",
            "profile",
            "rank",
            "seed",
            "metric_name",
            "mean",
            "std",
            "num_repeats",
            "num_successful",
            "num_failed",
        ]}
        flat["repeat_seeds"] = json.dumps(repeat_seeds)
        flat["repeat_scores"] = json.dumps(repeat_scores)
        flat["repeat_errors"] = json.dumps(repeat_errors)
        rows.append(flat)

    if rows:
        with open(summary_all_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return summary_all_jsonl, summary_all_csv


# ============================================================================
# Main driver
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run repeated spectral editing experiments on LoRA/LoRA+ adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--runs_root",
        type=Path,
        nargs="+",
        default=DEFAULT_RUNS_ROOTS,
        help="Root directories containing adapter runs (default: Llama3.1-8B and Qwen3-8B)",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Output root for edited adapters and results",
    )

    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["metamath", "magicoder", "alpaca", "csqa"],
        help="Tasks to evaluate (default: all four)",
    )

    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=EDIT_METHODS,
        choices=EDIT_METHODS,
        help="Edit methods to apply (default: all)",
    )

    p.add_argument("--use_vllm", action="store_true", help="Use vLLM for evaluation")
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="Tensor parallel size for vLLM (default: 8)",
    )

    p.add_argument("--calib_samples", type=int, default=32, help="Calibration samples for spectral edit")
    p.add_argument("--calib_batch_size", type=int, default=2, help="Calibration batch size")
    p.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["down_proj", "o_proj"],
        help="Target modules for spectral edit",
    )
    p.add_argument(
        "--calib_dataset",
        type=str,
        default=None,
        help="Calibration dataset (default: gsm8k)",
    )
    p.add_argument(
        "--calib_config",
        type=str,
        default=None,
        help="Calibration dataset config (default: main for gsm8k)",
    )
    p.add_argument(
        "--calib_split",
        type=str,
        default=None,
        help="Calibration split (default: train)",
    )
    p.add_argument(
        "--calib_text_fields",
        type=str,
        nargs="+",
        default=None,
        help="Calibration text fields for prompt/answer",
    )
    p.add_argument("--calib_shuffle", action="store_true", help="Shuffle calibration dataset")
    p.add_argument("--calib_seed", type=int, default=None, help="Seed for calibration shuffle")
    p.add_argument("--calib_start", type=int, default=0, help="Start offset into calibration dataset")

    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="Number of repeats per adapter/method (default: 5)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of repeat seeds (length must match --n_repeats)",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples for evaluation (smoke test mode)",
    )

    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run: discover adapters and print planned jobs without executing",
    )
    p.add_argument(
        "--skip_edit",
        action="store_true",
        help="Skip editing phase, only run evaluation on existing edited adapters",
    )
    p.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation phase, only run editing",
    )
    p.add_argument(
        "--adapter_filter",
        type=str,
        default=None,
        help="Only process adapters matching this substring (for testing)",
    )

    return p


def main():
    args = build_arg_parser().parse_args()

    runs_roots = list(args.runs_root)
    for runs_root in runs_roots:
        if not runs_root.exists():
            print(f"[ERROR] Runs root does not exist: {runs_root}")
            sys.exit(1)

    try:
        repeat_seeds = resolve_repeat_seeds(args.seed, args.n_repeats, args.seeds)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    tasks = [normalize_task(t) for t in args.tasks]
    for t in tasks:
        if t not in TASK_TO_EVAL_SCRIPT:
            print(f"[ERROR] Unknown task: {t}")
            sys.exit(1)

    git_commit = get_git_commit()

    print("=" * 70)
    print("LoRA/LoRA+ Spectral Edit Repeat Driver")
    print("=" * 70)
    print(f"Runs roots: {', '.join(str(p) for p in runs_roots)}")
    print(f"Output root: {args.out_root}")
    print(f"Tasks: {tasks}")
    print(f"Methods: {args.methods}")
    print(f"Repeats: {args.n_repeats} (seeds={repeat_seeds})")
    print(f"Use vLLM: {args.use_vllm}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Git commit: {git_commit or 'unknown'}")
    print("=" * 70)

    print("\n[1/4] Discovering adapters...")
    adapters: List[AdapterInfo] = []
    checkpoint_skipped = 0
    for runs_root in runs_roots:
        found, skipped = discover_adapters(runs_root, tasks, ALLOWED_PEFT_METHODS)
        adapters.extend(found)
        checkpoint_skipped += skipped
        print(f"  {runs_root}: {len(found)} adapters")

    print(f"\n  Total adapters: {len(adapters)}")
    print(f"  Skipped {checkpoint_skipped} checkpoint directories")

    if args.adapter_filter:
        adapters = [a for a in adapters if args.adapter_filter in str(a.adapter_dir)]
        print(f"  After filter '{args.adapter_filter}': {len(adapters)} adapters")

    if not adapters:
        print("[ERROR] No adapters found!")
        sys.exit(1)

    adapters.sort(key=lambda a: (a.base_model_dir, a.task, a.run_id, str(a.adapter_dir)))

    adapters_by_task: Dict[str, List[AdapterInfo]] = {}
    adapters_by_type: Dict[str, int] = {}
    for adapter in adapters:
        adapters_by_task.setdefault(adapter.task, []).append(adapter)
        adapters_by_type[adapter.method] = adapters_by_type.get(adapter.method, 0) + 1

    print("\n  Adapters by task:")
    for task, task_adapters in sorted(adapters_by_task.items()):
        print(f"    {task}: {len(task_adapters)}")

    print("\n  Adapters by type:")
    for method_name, count in sorted(adapters_by_type.items()):
        print(f"    {method_name}: {count}")

    num_edit_methods = len([m for m in args.methods if m != "baseline"])
    total_edits = len(adapters) * num_edit_methods * len(repeat_seeds)
    total_evals = len(adapters) * len(args.methods) * len(repeat_seeds)

    print(f"\n  Planned jobs:")
    print(f"    Edits: {total_edits}")
    print(f"    Evaluations: {total_evals}")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("[DRY-RUN] Planned execution:")
        print("=" * 70)

        for adapter in adapters[:3]:
            print(f"\n  Adapter: {adapter.run_id}")
            print(f"    Path: {adapter.adapter_dir}")
            print(f"    Task: {adapter.task}, Adapter type: {adapter.method}")
            print(f"    Profile: {adapter.profile}, Rank: {adapter.rank}, Seed: {adapter.seed}")
            print(f"    Base model: {adapter.base_model_id}")

            for method in args.methods:
                for repeat_idx, repeat_seed in enumerate(repeat_seeds[:2], start=1):
                    repeat_tag = build_repeat_tag(repeat_idx, repeat_seed)
                    if method == "baseline":
                        print(f"    [baseline] Eval repeat {repeat_tag}")
                    else:
                        print(f"    [{method}] Edit → Eval repeat {repeat_tag}")

        if len(adapters) > 3:
            print(f"\n  ... and {len(adapters) - 3} more adapters")

        print("\n[DRY-RUN] No changes made.")
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    results_writer = ResultsWriter(args.out_root)

    print("\n" + "=" * 70)
    print("[2/4] Processing adapters...")
    print("=" * 70)

    completed = 0
    failed = 0
    skipped = 0

    for i, adapter in enumerate(adapters, 1):
        print(f"\n[{i}/{len(adapters)}] Processing: {adapter.run_id}")
        print(f"  Path: {adapter.adapter_dir}")
        print(f"  Base model: {adapter.base_model_id}")

        for method in args.methods:
            print(f"\n  Method: {method}")

            for repeat_idx, repeat_seed in enumerate(repeat_seeds, start=1):
                repeat_tag = build_repeat_tag(repeat_idx, repeat_seed)
                print(f"    Repeat {repeat_tag}")

                if results_writer.is_completed(str(adapter.adapter_dir), method, adapter.task, repeat_seed):
                    print("      [SKIP] Already completed")
                    skipped += 1
                    continue

                if method == "baseline":
                    edited_adapter_dir = None
                    eval_adapter_dir = adapter.adapter_dir
                else:
                    edited_adapter_dir = (
                        args.out_root / "edited_adapters" / adapter.base_model_dir /
                        adapter.task / adapter.run_id / method / repeat_tag
                    )
                    eval_adapter_dir = edited_adapter_dir

                eval_output_dir = (
                    args.out_root / "eval_outputs" / adapter.base_model_dir /
                    adapter.task / adapter.run_id / method / repeat_tag
                )

                if method != "baseline" and not args.skip_edit:
                    if edited_adapter_dir.exists():
                        if has_adapter_weights(edited_adapter_dir):
                            print("      [SKIP EDIT] Edited adapter already exists")
                        else:
                            shutil.rmtree(edited_adapter_dir)

                    if not edited_adapter_dir.exists():
                        print(f"      Running spectral edit ({method})...")
                        success, error = run_spectral_edit(
                            adapter_dir=adapter.adapter_dir,
                            out_dir=edited_adapter_dir,
                            edit_method=method,
                            base_model_id=adapter.base_model_id,
                            seed=repeat_seed,
                            calib_samples=args.calib_samples,
                            calib_batch_size=args.calib_batch_size,
                            target_modules=args.target_modules,
                            calib_dataset=args.calib_dataset,
                            calib_config=args.calib_config,
                            calib_split=args.calib_split,
                            calib_text_fields=args.calib_text_fields,
                            calib_shuffle=args.calib_shuffle,
                            calib_seed=args.calib_seed,
                            calib_start=args.calib_start,
                        )

                        if not success:
                            print(f"      [EDIT FAILED] {error}")
                            result = EvalResult(
                                timestamp=datetime.now().isoformat(),
                                base_model_id=adapter.base_model_id,
                                base_model_dir=adapter.base_model_dir,
                                task=adapter.task,
                                adapter_dir=str(adapter.adapter_dir),
                                edited_adapter_dir=str(edited_adapter_dir),
                                eval_output_dir=str(eval_output_dir),
                                edit_method=method,
                                peft_method=adapter.method,
                                profile=adapter.profile,
                                rank=adapter.rank,
                                seed=adapter.seed,
                                run_id=adapter.run_id,
                                repeat_idx=repeat_idx,
                                repeat_seed=repeat_seed,
                                repeat_tag=repeat_tag,
                                metric_name=TASK_TO_METRIC[adapter.task],
                                metric_value=-1.0,
                                num_examples=0,
                                backend="vllm" if args.use_vllm else "transformers",
                                tensor_parallel_size=args.tensor_parallel_size,
                                git_commit=git_commit,
                                error=error,
                            )
                            results_writer.write_result(result)
                            failed += 1
                            continue

                        gc.collect()
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                if method != "baseline" and args.skip_edit and edited_adapter_dir is not None:
                    if not has_adapter_weights(edited_adapter_dir):
                        if not args.skip_eval:
                            error = "Edited adapter missing; re-run without --skip_edit"
                            print(f"      [SKIP EVAL] {error}")
                            result = EvalResult(
                                timestamp=datetime.now().isoformat(),
                                base_model_id=adapter.base_model_id,
                                base_model_dir=adapter.base_model_dir,
                                task=adapter.task,
                                adapter_dir=str(adapter.adapter_dir),
                                edited_adapter_dir=str(edited_adapter_dir),
                                eval_output_dir=str(eval_output_dir),
                                edit_method=method,
                                peft_method=adapter.method,
                                profile=adapter.profile,
                                rank=adapter.rank,
                                seed=adapter.seed,
                                run_id=adapter.run_id,
                                repeat_idx=repeat_idx,
                                repeat_seed=repeat_seed,
                                repeat_tag=repeat_tag,
                                metric_name=TASK_TO_METRIC[adapter.task],
                                metric_value=-1.0,
                                num_examples=0,
                                backend="vllm" if args.use_vllm else "transformers",
                                tensor_parallel_size=args.tensor_parallel_size,
                                git_commit=git_commit,
                                error=error,
                            )
                            results_writer.write_result(result)
                            failed += 1
                        continue

                if args.skip_eval:
                    print("      [SKIP EVAL] --skip_eval set")
                    continue

                metrics = load_metrics_if_exists(eval_output_dir)
                error = None
                if metrics is None:
                    print(f"      Running evaluation ({adapter.task})...")
                    metrics, error = run_evaluation(
                        base_model_id=adapter.base_model_id,
                        adapter_dir=eval_adapter_dir,
                        task=adapter.task,
                        output_dir=eval_output_dir,
                        use_vllm=args.use_vllm,
                        tensor_parallel_size=args.tensor_parallel_size,
                        max_samples=args.max_samples,
                        seed=repeat_seed,
                    )
                else:
                    print("      [SKIP EVAL] Using existing metrics.json")

                if error:
                    print(f"      [EVAL FAILED] {error}")
                    metric_value = -1.0
                    num_examples = 0
                else:
                    metric_name = TASK_TO_METRIC[adapter.task]
                    metric_value = metrics.get(metric_name, metrics.get("accuracy", -1.0))
                    num_examples = metrics.get("total", metrics.get("num_examples", 0))
                    print(f"      [SUCCESS] {metric_name}={metric_value:.4f} (n={num_examples})")

                result = EvalResult(
                    timestamp=datetime.now().isoformat(),
                    base_model_id=adapter.base_model_id,
                    base_model_dir=adapter.base_model_dir,
                    task=adapter.task,
                    adapter_dir=str(adapter.adapter_dir),
                    edited_adapter_dir=str(edited_adapter_dir) if edited_adapter_dir else None,
                    eval_output_dir=str(eval_output_dir),
                    edit_method=method,
                    peft_method=adapter.method,
                    profile=adapter.profile,
                    rank=adapter.rank,
                    seed=adapter.seed,
                    run_id=adapter.run_id,
                    repeat_idx=repeat_idx,
                    repeat_seed=repeat_seed,
                    repeat_tag=repeat_tag,
                    metric_name=TASK_TO_METRIC[adapter.task],
                    metric_value=metric_value,
                    num_examples=num_examples,
                    backend="vllm" if args.use_vllm else "transformers",
                    tensor_parallel_size=args.tensor_parallel_size,
                    git_commit=git_commit,
                    error=error,
                )
                results_writer.write_result(result)

                if error:
                    failed += 1
                else:
                    completed += 1

                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    print("\n" + "=" * 70)
    print("[3/4] Aggregating results...")
    print("=" * 70)

    summaries = aggregate_results(load_results_jsonl(results_writer.jsonl_path))
    summary_jsonl, summary_csv = write_aggregate_outputs(args.out_root, summaries)

    print("\n" + "=" * 70)
    print("[4/4] Summary")
    print("=" * 70)
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"\n  Results saved to:")
    print(f"    {results_writer.jsonl_path}")
    print(f"    {results_writer.csv_path}")
    print(f"    {summary_jsonl}")
    print(f"    {summary_csv}")


if __name__ == "__main__":
    main()
