#!/usr/bin/env python3
"""
Driver script for running spectral editing experiments on LoRA adapters.

This script:
1. Discovers LoRA adapters under a runs root directory
2. Applies spectral editing methods from finetune.spectral_edit
3. Evaluates on task-specific benchmarks using peft-sft-lab evaluators
4. Saves results incrementally to JSONL and CSV files

Methods:
  - baseline: Original LoRA adapter (no edit)
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
    python scripts/run_llama31_lora_spectral_edit_all.py \
        --runs_root /path/to/runs/meta-llama-Llama-3.1-8B \
        --tasks metamath magicoder alpaca csqa \
        --out_root /path/to/output \
        --use_vllm \
        --tensor_parallel_size 8 \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import time
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
    method: str  # peft method: lora, pissa, adalora, loraplus
    profile: str
    rank: str
    seed: str
    base_model_dir: str
    base_model_id: str

    @property
    def run_id(self) -> str:
        """Generate a unique run ID for this adapter."""
        return f"{self.task}_{self.method}_{self.profile}_{self.rank}_{self.seed}"


@dataclass
class EvalResult:
    """Result from a single evaluation."""
    timestamp: str
    base_model_id: str
    base_model_dir: str
    task: str
    adapter_dir: str
    edited_adapter_dir: Optional[str]
    edit_method: str
    peft_method: str
    profile: str
    rank: str
    seed: str
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

    Returns AdapterInfo or None if parsing fails.
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

    # Check if runs_root name looks like a base model dir
    runs_root_name = runs_root.name
    if runs_root_name in BASE_MODEL_DIR_TO_ID or runs_root_name.startswith("meta-llama") or \
       runs_root_name.startswith("mistralai") or runs_root_name.startswith("Qwen"):
        # runs_root already includes base_model_dir
        base_model_dir = runs_root_name
        # parts[0] = task, parts[1] = peft_method, etc.
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
        # Standard structure: parts[0] = base_model_dir
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

    # Determine base model ID from directory name
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

    # Validate minimum required fields
    if not task:
        return None

    return AdapterInfo(
        adapter_dir=adapter_dir,
        task=task,
        method=peft_method or "lora",
        profile=profile or "unknown",
        rank=rank or "unknown",
        seed=seed or "unknown",
        base_model_dir=base_model_dir or "unknown",
        base_model_id=base_model_id or f"unknown/{base_model_dir or 'model'}",
    )


def discover_adapters(
    runs_root: Path,
    tasks: Optional[List[str]] = None,
) -> Tuple[List[AdapterInfo], int]:
    """
    Recursively discover LoRA adapter directories.

    An adapter directory must contain BOTH:
      - adapter_model.safetensors (or adapter_model.bin)
      - adapter_config.json

    Excludes any path containing "/checkpoint-" anywhere.

    Returns:
        Tuple of (list of AdapterInfo, count of skipped checkpoint dirs)
    """
    adapters = []
    checkpoint_count = 0

    for root, dirs, files in os.walk(runs_root):
        root_path = Path(root)

        # Skip checkpoint directories
        if "/checkpoint-" in str(root_path) or root_path.name.startswith("checkpoint-"):
            checkpoint_count += 1
            # Don't descend into checkpoint dirs
            dirs.clear()
            continue

        # Check for adapter files
        has_safetensors = "adapter_model.safetensors" in files
        has_bin = "adapter_model.bin" in files
        has_config = "adapter_config.json" in files

        if (has_safetensors or has_bin) and has_config:
            info = parse_adapter_path(root_path, runs_root)
            if info:
                # Filter by task if specified
                if tasks is None or info.task in tasks:
                    adapters.append(info)

    return adapters, checkpoint_count


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

    Args:
        adapter_dir: Path to original LoRA adapter
        out_dir: Path to save edited adapter
        edit_method: One of random_index, smooth_abs, abs_select, grad_direction
        base_model_id: HuggingFace model ID
        seed: Random seed
        calib_samples: Number of calibration samples
        calib_batch_size: Batch size for calibration
        target_modules: List of modules to edit (default: down_proj, o_proj)
        dry_run: If True, only print command without executing

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    if target_modules is None:
        target_modules = ["down_proj", "o_proj"]

    mode = METHOD_TO_MODE.get(edit_method)
    if not mode:
        return False, f"Unknown edit method: {edit_method}"

    # Build command
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

    # Add method-specific parameters
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

    # Set up environment
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{SRC_DIR}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(SRC_DIR)

    # Run command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            env=env,
            cwd=PEFT_SFT_LAB_ROOT,
        )

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "Unknown error"
            return False, f"Edit failed (code {result.returncode}): {error_msg}"

        # Verify output
        if not (out_dir / "adapter_model.safetensors").exists() and not (out_dir / "adapter_model.bin").exists():
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

    Returns:
        Tuple of (metrics_dict, error_message)
    """
    eval_module = TASK_TO_EVAL_SCRIPT.get(task)
    if not eval_module:
        return None, f"Unknown task: {task}"

    # Build command
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

    # Task-specific settings
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

    # Run evaluation
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
            cwd=PEFT_SFT_LAB_ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        )

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "Unknown error"
            return None, f"Eval failed (code {result.returncode}): {error_msg}"

        # Read metrics
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
# Results saving
# ============================================================================

class ResultsWriter:
    """Handles incremental writing of results to JSONL and CSV."""

    def __init__(self, out_root: Path):
        self.out_root = out_root
        self.jsonl_path = out_root / "results.jsonl"
        self.csv_path = out_root / "results.csv"

        # Ensure output directory exists
        out_root.mkdir(parents=True, exist_ok=True)

        # Load existing results for deduplication
        self.existing_keys: set = set()
        self._load_existing()

    def _result_key(self, result: EvalResult) -> str:
        """Generate a unique key for deduplication."""
        return f"{result.adapter_dir}|{result.edit_method}|{result.task}"

    def _load_existing(self):
        """Load existing results from JSONL for deduplication."""
        if self.jsonl_path.exists():
            try:
                with open(self.jsonl_path) as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            key = f"{rec.get('adapter_dir', '')}|{rec.get('edit_method', '')}|{rec.get('task', '')}"
                            self.existing_keys.add(key)
            except Exception as e:
                print(f"[WARN] Failed to load existing results: {e}")

    def is_completed(self, adapter_dir: str, edit_method: str, task: str) -> bool:
        """Check if this evaluation has already been completed."""
        key = f"{adapter_dir}|{edit_method}|{task}"
        return key in self.existing_keys

    def write_result(self, result: EvalResult):
        """Write a single result incrementally."""
        key = self._result_key(result)
        if key in self.existing_keys:
            return  # Skip duplicate

        self.existing_keys.add(key)
        result_dict = asdict(result)

        # Write to JSONL
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

        # Write to CSV
        csv_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result_dict.keys()))
            if not csv_exists:
                writer.writeheader()
            writer.writerow(result_dict)


# ============================================================================
# Main driver
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run spectral editing experiments on LoRA adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required paths
    p.add_argument(
        "--runs_root",
        type=Path,
        required=True,
        help="Root directory containing LoRA adapter runs (e.g., runs/meta-llama-Llama-3.1-8B)",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Output root for edited adapters and results",
    )

    # Task selection
    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["metamath", "magicoder", "alpaca", "csqa"],
        help="Tasks to evaluate (default: all four)",
    )

    # Edit methods
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=EDIT_METHODS,
        choices=EDIT_METHODS,
        help="Edit methods to apply (default: all)",
    )

    # Evaluation settings
    p.add_argument("--use_vllm", action="store_true", help="Use vLLM for evaluation")
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="Tensor parallel size for vLLM (default: 8)",
    )

    # Spectral edit settings
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

    # General settings
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples for evaluation (smoke test mode)",
    )

    # Modes
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

    # Filtering
    p.add_argument(
        "--adapter_filter",
        type=str,
        default=None,
        help="Only process adapters matching this substring (for testing)",
    )

    return p


def main():
    args = build_arg_parser().parse_args()

    # Validate paths
    if not args.runs_root.exists():
        print(f"[ERROR] Runs root does not exist: {args.runs_root}")
        sys.exit(1)

    # Normalize tasks
    tasks = [normalize_task(t) for t in args.tasks]
    for t in tasks:
        if t not in TASK_TO_EVAL_SCRIPT:
            print(f"[ERROR] Unknown task: {t}")
            sys.exit(1)

    # Get git commit
    git_commit = get_git_commit()

    print("=" * 70)
    print("LoRA Spectral Edit Experiment Driver")
    print("=" * 70)
    print(f"Runs root: {args.runs_root}")
    print(f"Output root: {args.out_root}")
    print(f"Tasks: {tasks}")
    print(f"Methods: {args.methods}")
    print(f"Use vLLM: {args.use_vllm}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Seed: {args.seed}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Git commit: {git_commit or 'unknown'}")
    print("=" * 70)

    # Discover adapters
    print("\n[1/4] Discovering adapters...")
    adapters, checkpoint_skipped = discover_adapters(args.runs_root, tasks)

    print(f"  Found {len(adapters)} adapters")
    print(f"  Skipped {checkpoint_skipped} checkpoint directories")

    # Apply adapter filter if specified
    if args.adapter_filter:
        adapters = [a for a in adapters if args.adapter_filter in str(a.adapter_dir)]
        print(f"  After filter '{args.adapter_filter}': {len(adapters)} adapters")

    if not adapters:
        print("[ERROR] No adapters found!")
        sys.exit(1)

    # Group by task
    adapters_by_task: Dict[str, List[AdapterInfo]] = {}
    for adapter in adapters:
        adapters_by_task.setdefault(adapter.task, []).append(adapter)

    print("\n  Adapters by task:")
    for task, task_adapters in sorted(adapters_by_task.items()):
        print(f"    {task}: {len(task_adapters)} adapters")

    # Calculate total jobs
    num_edit_methods = len([m for m in args.methods if m != "baseline"])
    total_edits = len(adapters) * num_edit_methods
    total_evals = len(adapters) * len(args.methods)  # baseline + edited

    print(f"\n  Planned jobs:")
    print(f"    Edits: {total_edits}")
    print(f"    Evaluations: {total_evals}")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("[DRY-RUN] Planned execution:")
        print("=" * 70)

        for adapter in adapters[:5]:  # Show first 5
            print(f"\n  Adapter: {adapter.adapter_dir.name}")
            print(f"    Task: {adapter.task}, Method: {adapter.method}")
            print(f"    Profile: {adapter.profile}, Rank: {adapter.rank}, Seed: {adapter.seed}")
            print(f"    Base model: {adapter.base_model_id}")

            for method in args.methods:
                if method == "baseline":
                    print(f"    [baseline] Evaluate original adapter")
                else:
                    print(f"    [{method}] Edit → Evaluate")

        if len(adapters) > 5:
            print(f"\n  ... and {len(adapters) - 5} more adapters")

        print("\n[DRY-RUN] No changes made.")
        return

    # Initialize results writer
    args.out_root.mkdir(parents=True, exist_ok=True)
    results_writer = ResultsWriter(args.out_root)

    # Process adapters
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

            # Determine adapter to evaluate
            if method == "baseline":
                eval_adapter_dir = adapter.adapter_dir
                edited_adapter_dir = None
            else:
                # Create output directory for edited adapter
                edited_adapter_dir = (
                    args.out_root / "edited_adapters" / adapter.task /
                    adapter.run_id / method
                )

                # Check if already completed
                if results_writer.is_completed(str(adapter.adapter_dir), method, adapter.task):
                    print(f"    [SKIP] Already completed")
                    skipped += 1
                    continue

                # Run spectral edit
                if not args.skip_edit:
                    if edited_adapter_dir.exists():
                        # Check if edit output is valid
                        has_weights = (
                            (edited_adapter_dir / "adapter_model.safetensors").exists() or
                            (edited_adapter_dir / "adapter_model.bin").exists()
                        )
                        if has_weights:
                            print(f"    [SKIP EDIT] Edited adapter already exists")
                        else:
                            shutil.rmtree(edited_adapter_dir)

                    if not edited_adapter_dir.exists():
                        print(f"    Running spectral edit ({method})...")
                        success, error = run_spectral_edit(
                            adapter_dir=adapter.adapter_dir,
                            out_dir=edited_adapter_dir,
                            edit_method=method,
                            base_model_id=adapter.base_model_id,
                            seed=args.seed,
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
                            print(f"    [EDIT FAILED] {error}")
                            result = EvalResult(
                                timestamp=datetime.now().isoformat(),
                                base_model_id=adapter.base_model_id,
                                base_model_dir=adapter.base_model_dir,
                                task=adapter.task,
                                adapter_dir=str(adapter.adapter_dir),
                                edited_adapter_dir=str(edited_adapter_dir),
                                edit_method=method,
                                peft_method=adapter.method,
                                profile=adapter.profile,
                                rank=adapter.rank,
                                seed=adapter.seed,
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

                        # Force GPU cleanup after edit
                        gc.collect()
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                eval_adapter_dir = edited_adapter_dir

            # Run evaluation
            if not args.skip_eval:
                # Check if already completed (for baseline)
                if method == "baseline" and results_writer.is_completed(
                    str(adapter.adapter_dir), method, adapter.task
                ):
                    print(f"    [SKIP] Already completed")
                    skipped += 1
                    continue

                eval_output_dir = (
                    args.out_root / "eval_outputs" / adapter.task /
                    adapter.run_id / method
                )

                print(f"    Running evaluation ({adapter.task})...")
                metrics, error = run_evaluation(
                    base_model_id=adapter.base_model_id,
                    adapter_dir=eval_adapter_dir,
                    task=adapter.task,
                    output_dir=eval_output_dir,
                    use_vllm=args.use_vllm,
                    tensor_parallel_size=args.tensor_parallel_size,
                    max_samples=args.max_samples,
                    seed=args.seed,
                )

                if error:
                    print(f"    [EVAL FAILED] {error}")
                    metric_value = -1.0
                    num_examples = 0
                else:
                    metric_name = TASK_TO_METRIC[adapter.task]
                    metric_value = metrics.get(metric_name, metrics.get("accuracy", -1.0))
                    num_examples = metrics.get("total", metrics.get("num_examples", 0))
                    print(f"    [SUCCESS] {metric_name}={metric_value:.4f} (n={num_examples})")

                # Record result
                result = EvalResult(
                    timestamp=datetime.now().isoformat(),
                    base_model_id=adapter.base_model_id,
                    base_model_dir=adapter.base_model_dir,
                    task=adapter.task,
                    adapter_dir=str(adapter.adapter_dir),
                    edited_adapter_dir=str(edited_adapter_dir) if edited_adapter_dir else None,
                    edit_method=method,
                    peft_method=adapter.method,
                    profile=adapter.profile,
                    rank=adapter.rank,
                    seed=adapter.seed,
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

                # Force GPU cleanup after eval
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    # Summary
    print("\n" + "=" * 70)
    print("[4/4] Summary")
    print("=" * 70)
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"\n  Results saved to:")
    print(f"    {results_writer.jsonl_path}")
    print(f"    {results_writer.csv_path}")


if __name__ == "__main__":
    main()
