#!/usr/bin/env python3
"""
Driver script for LoRA spectral editing experiments on Llama-3.1-8B.

This script:
1. Discovers all LoRA adapters under runs/meta-llama-Llama-3.1-8B
2. Applies various spectral editing policies (random_index, smooth_abs, abs_select, gd)
3. Evaluates baseline and edited adapters on task-specific benchmarks
4. Saves results incrementally to JSONL/CSV files

Usage:
    python scripts/run_lora_spectral_edit_eval_llama31.py \
        --runs_root /path/to/runs/meta-llama-Llama-3.1-8B \
        --lora_spectral_edit_root /path/to/lora-spectral-edit \
        --tasks metamath magicoder alpaca csqa \
        --use_vllm

Task-Evaluation Mapping:
    - metamath -> GSM8K (5-shot, greedy, strict-match accuracy)
    - magicoder -> HumanEval (0-shot, pass@1)
    - alpaca -> IFEval (instruction-following compliance)
    - csqa -> CommonsenseQA (validation split, letter accuracy)
"""

import os
import sys
import json
import csv
import re
import gc
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Set tokenizers parallelism BEFORE importing anything that uses HF
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

# Add src to path for vllm_utils import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import robust vLLM cleanup utilities
try:
    from finetune.eval.vllm_utils import shutdown_vllm_engine
    HAVE_VLLM_UTILS = True
except ImportError:
    HAVE_VLLM_UTILS = False
    shutdown_vllm_engine = None


# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL = "meta-llama/Llama-3.1-8B"

# Editing policies to apply
EDIT_POLICIES = {
    "random_index": {
        "mode": "random_index",
        "core_frac": 0.2,
        "noise_frac": 0.2,
        "amp_factor": 1.25,
        "sup_factor": 0.80,
    },
    "smooth_abs": {
        "mode": "smooth_abs",
        "core_frac": 0.2,
        "noise_frac": 0.2,
        "amp_factor": 1.25,
        "sup_factor": 0.80,
        "smooth_temperature": 0.35,
    },
    "abs_select": {
        "mode": "abs_select",
        "core_frac": 0.2,
        "noise_frac": 0.2,
        "amp_factor": 1.25,
        "sup_factor": 0.80,
    },
    "grad_direction": {
        "mode": "gd",
        "eta": 0.2,
        "update_mode": "multiplicative",
        "asymmetric_update": True,
        "eta_suppress": 2.0,
        "eta_enhance": 0.2,
    },
}

# Task to evaluation mapping
TASK_EVAL_MAP = {
    "metamath": "gsm8k",
    "magicoder": "humaneval",
    "alpaca": "ifeval",
    "csqa": "csqa",
}


@dataclass
class AdapterInfo:
    """Information about a discovered adapter."""
    task: str
    peft_method: str
    profile: str
    rank: str
    seed: str
    adapter_dir: str
    base_model: str = BASE_MODEL

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    """Evaluation result for a single adapter+policy combination."""
    timestamp: str
    task: str
    peft_method: str
    rank: str
    seed: str
    policy: str
    adapter_path: str
    eval_task: str
    metric_name: str
    metric_value: float
    eval_command: str
    git_commit_hash: str
    status: str = "ok"  # "ok" or "failed"
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_csv_row(self) -> Dict[str, Any]:
        """Flatten for CSV output."""
        d = {
            "timestamp": self.timestamp,
            "task": self.task,
            "peft_method": self.peft_method,
            "rank": self.rank,
            "seed": self.seed,
            "policy": self.policy,
            "adapter_path": self.adapter_path,
            "eval_task": self.eval_task,
            "status": self.status,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "eval_command": self.eval_command,
            "git_commit_hash": self.git_commit_hash,
        }
        for k, v in self.extra_metrics.items():
            d[f"extra_{k}"] = v
        return d

    def is_success(self) -> bool:
        return self.status == "ok"


@dataclass
class ErrorRecord:
    """Detailed error record for failed evaluations."""
    timestamp: str
    task: str
    peft_method: str
    rank: str
    seed: str
    policy: str
    adapter_path: str
    error_type: str  # "eval_failed", "edit_failed", "vllm_init_failed", etc.
    error_message: str
    full_traceback: str
    eval_command: str
    runtime_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Utility Functions
# =============================================================================

def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_vllm_runtime_info() -> Dict[str, Any]:
    """Capture vLLM and GPU runtime information for error diagnosis."""
    info = {
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }

    # vLLM version
    try:
        import vllm
        info["vllm_version"] = vllm.__version__
    except Exception:
        info["vllm_version"] = "not_installed"

    # PyTorch and CUDA
    try:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name()
            # Memory info
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_stats(i) if torch.cuda.is_available() else {}
                info[f"gpu{i}_allocated_mb"] = torch.cuda.memory_allocated(i) / 1024 / 1024
                info[f"gpu{i}_reserved_mb"] = torch.cuda.memory_reserved(i) / 1024 / 1024
    except Exception as e:
        info["torch_error"] = str(e)

    return info


def is_transient_vllm_error(error_msg: str) -> bool:
    """Check if a vLLM error is likely transient and retryable."""
    transient_patterns = [
        "CUDA out of memory",
        "NCCL error",
        "connection reset",
        "timeout",
        "Engine initialization",
        "EngineCore",
        "ZMQ",
        "socket",
        "process died",
        "worker died",
    ]
    error_lower = error_msg.lower()
    return any(p.lower() in error_lower for p in transient_patterns)


def append_error_jsonl(error: ErrorRecord, output_path: Path):
    """Append error record to errors JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(error.to_dict()) + "\n")


def log_error(
    task: str,
    peft_method: str,
    rank: str,
    seed: str,
    policy: str,
    adapter_path: str,
    error_type: str,
    error_message: str,
    full_traceback: str,
    eval_command: str,
    errors_jsonl: Path,
) -> ErrorRecord:
    """Create and log an error record."""
    runtime_info = get_vllm_runtime_info()

    error = ErrorRecord(
        timestamp=datetime.now().isoformat(),
        task=task,
        peft_method=peft_method,
        rank=rank,
        seed=seed,
        policy=policy,
        adapter_path=adapter_path,
        error_type=error_type,
        error_message=error_message,
        full_traceback=full_traceback,
        eval_command=eval_command,
        runtime_info=runtime_info,
    )

    append_error_jsonl(error, errors_jsonl)
    return error


def discover_adapters(runs_root: Path, tasks: Optional[List[str]] = None) -> List[AdapterInfo]:
    """
    Discover all LoRA adapters under runs_root.

    Args:
        runs_root: Root directory containing task subdirectories
        tasks: Optional filter for specific tasks

    Returns:
        List of AdapterInfo objects
    """
    adapters = []

    for adapter_file in runs_root.rglob("adapter_model.safetensors"):
        adapter_dir = adapter_file.parent

        # Skip checkpoint directories
        if "/checkpoint-" in str(adapter_dir):
            continue

        # Parse path structure: task/peft_method/profile-xxx/rank-xxx/seedxxx
        try:
            rel_path = adapter_dir.relative_to(runs_root)
            parts = rel_path.parts

            if len(parts) < 5:
                continue

            task = parts[0]
            peft_method = parts[1]
            profile = parts[2]
            rank = parts[3]
            seed = parts[4]

            # Filter by tasks if specified
            if tasks and task not in tasks:
                continue

            adapters.append(AdapterInfo(
                task=task,
                peft_method=peft_method,
                profile=profile,
                rank=rank,
                seed=seed,
                adapter_dir=str(adapter_dir),
            ))
        except Exception as e:
            print(f"[Warn] Could not parse adapter path: {adapter_dir}: {e}")
            continue

    return adapters


def save_manifest(adapters: List[AdapterInfo], output_path: Path):
    """Save adapter manifest as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([a.to_dict() for a in adapters], f, indent=2)
    print(f"[Manifest] Saved {len(adapters)} adapters to {output_path}")


def append_result_jsonl(result: EvalResult, output_path: Path):
    """Append result to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


def update_results_csv(results_jsonl: Path, output_csv: Path, include_failures: bool = False):
    """
    Update CSV from JSONL file.

    Args:
        results_jsonl: Path to results JSONL file
        output_csv: Path to output CSV file
        include_failures: If True, include failed results; if False, only successful results
    """
    if not results_jsonl.exists():
        return

    rows = []
    failed_count = 0
    with open(results_jsonl) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                status = data.get("status", "ok")

                # Skip failures unless explicitly requested
                if status != "ok" and not include_failures:
                    failed_count += 1
                    continue

                row = {
                    "timestamp": data.get("timestamp", ""),
                    "task": data.get("task", ""),
                    "peft_method": data.get("peft_method", ""),
                    "rank": data.get("rank", ""),
                    "seed": data.get("seed", ""),
                    "policy": data.get("policy", ""),
                    "adapter_path": data.get("adapter_path", ""),
                    "eval_task": data.get("eval_task", ""),
                    "status": status,
                    "metric_name": data.get("metric_name", ""),
                    "metric_value": data.get("metric_value", ""),
                    "eval_command": data.get("eval_command", ""),
                    "git_commit_hash": data.get("git_commit_hash", ""),
                }
                for k, v in data.get("extra_metrics", {}).items():
                    row[f"extra_{k}"] = v
                rows.append(row)

    if failed_count > 0:
        print(f"[CSV] Excluded {failed_count} failed results (see errors.jsonl for details)")

    if not rows:
        return

    # Get all columns
    all_cols = set()
    for row in rows:
        all_cols.update(row.keys())

    # Define column order - status comes after policy
    base_cols = [
        "timestamp", "task", "peft_method", "rank", "seed", "policy", "status",
        "eval_task", "metric_name", "metric_value", "adapter_path",
        "eval_command", "git_commit_hash"
    ]
    extra_cols = sorted([c for c in all_cols if c not in base_cols])
    cols = base_cols + extra_cols

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def generate_summary_markdown(results_jsonl: Path, output_md: Path):
    """Generate markdown summary table from results."""
    if not results_jsonl.exists():
        return

    # Load all results
    results_by_task = {}
    with open(results_jsonl) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                task = data.get("task", "unknown")
                if task not in results_by_task:
                    results_by_task[task] = {}

                key = (data.get("peft_method", ""), data.get("rank", ""), data.get("seed", ""))
                if key not in results_by_task[task]:
                    results_by_task[task][key] = {}

                policy = data.get("policy", "unknown")
                metric_value = data.get("metric_value", 0)
                results_by_task[task][key][policy] = metric_value

    # Generate markdown
    lines = [
        "# LoRA Spectral Edit Evaluation Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]

    policies = ["baseline", "random_index", "smooth_abs", "abs_select", "grad_direction"]

    for task, task_results in sorted(results_by_task.items()):
        lines.append(f"## Task: {task}")
        lines.append("")

        eval_task = TASK_EVAL_MAP.get(task, task)
        lines.append(f"Evaluation: {eval_task}")
        lines.append("")

        # Table header
        header = "| PEFT Method | Rank | Seed | " + " | ".join(policies) + " |"
        separator = "|" + "|".join(["---"] * (len(policies) + 3)) + "|"
        lines.append(header)
        lines.append(separator)

        for (peft_method, rank, seed), policy_results in sorted(task_results.items()):
            row = [peft_method, rank, seed]
            for policy in policies:
                val = policy_results.get(policy, "")
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(str(val) if val else "-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w") as f:
        f.write("\n".join(lines))

    print(f"[Summary] Generated {output_md}")


# =============================================================================
# Editing Functions
# =============================================================================

def run_spectral_edit(
    base_model: str,
    lora_path: str,
    out_dir: str,
    policy_name: str,
    policy_config: Dict[str, Any],
    lora_spectral_edit_root: str,
    seed: int = 42,
    calib_samples: int = 32,
    target_modules: List[str] = None,
) -> Tuple[bool, str]:
    """
    Run spectral editing using lora-spectral-edit CLI.

    Returns:
        Tuple of (success, edited_adapter_path or error message)
    """
    if target_modules is None:
        target_modules = ["down_proj", "o_proj"]

    # Build command
    cmd = [
        sys.executable, "-m", "lora_spectral_edit", "edit",
        "--base_model", base_model,
        "--lora_path", lora_path,
        "--out_dir", out_dir,
        "--seed", str(seed),
        "--calib_samples", str(calib_samples),
        "--target_modules", *target_modules,
    ]

    # Add policy-specific args
    mode = policy_config.get("mode", "abs_select")
    cmd.extend(["--mode", mode])

    for key, value in policy_config.items():
        if key == "mode":
            continue
        arg_name = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(arg_name)
        else:
            cmd.extend([arg_name, str(value)])

    # Set PYTHONPATH to include lora-spectral-edit
    env = os.environ.copy()
    src_path = os.path.join(lora_spectral_edit_root, "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path

    cmd_str = " ".join(cmd)
    print(f"[Edit] Running: {cmd_str[:200]}...")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
        )

        if result.returncode != 0:
            print(f"[Edit] FAILED: {result.stderr[:500]}")
            return False, result.stderr

        # Verify output
        edited_adapter = os.path.join(out_dir, "adapter_model.safetensors")
        if not os.path.exists(edited_adapter):
            edited_adapter = os.path.join(out_dir, "adapter_model.bin")

        if not os.path.exists(edited_adapter):
            return False, "Edited adapter file not found"

        # Sanity check: verify no NaN/Inf
        try:
            from safetensors.torch import load_file
            state_dict = load_file(edited_adapter)
            for name, tensor in state_dict.items():
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    return False, f"NaN/Inf found in tensor: {name}"
        except Exception as e:
            print(f"[Warn] Could not verify tensor values: {e}")

        return True, out_dir

    except subprocess.TimeoutExpired:
        return False, "Timeout expired"
    except Exception as e:
        return False, str(e)


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_gsm8k(
    base_model: str,
    lora_dir: str,
    lora_spectral_edit_root: str,
    max_samples: int = -1,
    seed: int = 0,
    use_vllm: bool = True,
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Evaluate on GSM8K using lora-spectral-edit's evaluator.

    Returns:
        Tuple of (success, metrics_dict, command_str)
    """
    env = os.environ.copy()
    src_path = os.path.join(lora_spectral_edit_root, "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path

    cmd = [
        sys.executable, "-m", "lora_spectral_edit", "eval",
        "--base_model", base_model,
        "--lora_dir", lora_dir,
        "--eval_profile", "paper_math",
        "--seed", str(seed),
    ]

    if max_samples > 0:
        cmd.extend(["--max_samples", str(max_samples)])

    cmd_str = " ".join(cmd)
    print(f"[Eval GSM8K] Running: {cmd_str[:200]}...")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        # Parse accuracy from output
        output = result.stdout + result.stderr
        acc_match = re.search(r"acc=([0-9.]+)", output)
        correct_match = re.search(r"\((\d+)/(\d+)\)", output)

        if acc_match:
            acc = float(acc_match.group(1))
            metrics = {"acc": acc, "metric_name": "gsm8k_accuracy"}
            if correct_match:
                metrics["correct"] = int(correct_match.group(1))
                metrics["total"] = int(correct_match.group(2))
            return True, metrics, cmd_str

        # Try reading from eval_config.json
        config_path = os.path.join(lora_dir, "eval_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            if "metric" in config:
                return True, {
                    "acc": config["metric"].get("score", 0),
                    "correct": config["metric"].get("correct", 0),
                    "total": config["metric"].get("total", 0),
                    "metric_name": "gsm8k_accuracy",
                }, cmd_str

        return False, {"error": "Could not parse accuracy"}, cmd_str

    except Exception as e:
        return False, {"error": str(e)}, cmd_str


def evaluate_humaneval(
    base_model: str,
    lora_dir: str,
    lora_spectral_edit_root: str,
    max_samples: int = -1,
    seed: int = 0,
    use_vllm: bool = True,
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Evaluate on HumanEval using lora-spectral-edit's evaluator.
    """
    env = os.environ.copy()
    src_path = os.path.join(lora_spectral_edit_root, "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path

    # Use greedy_code for faster evaluation (paper_code_main requires 50 samples per problem)
    cmd = [
        sys.executable, "-m", "lora_spectral_edit", "eval-humaneval",
        "--base_model", base_model,
        "--lora_dir", lora_dir,
        "--eval_profile", "greedy_code",
        "--seed", str(seed),
    ]

    if max_samples > 0:
        cmd.extend(["--max_samples", str(max_samples)])

    cmd_str = " ".join(cmd)
    print(f"[Eval HumanEval] Running: {cmd_str[:200]}...")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout for code execution
        )

        output = result.stdout + result.stderr

        # Parse pass@1 from output
        pass1_match = re.search(r"pass@1=([0-9.]+)", output)

        if pass1_match:
            pass1 = float(pass1_match.group(1))
            return True, {"pass@1": pass1, "metric_name": "humaneval_pass@1"}, cmd_str

        return False, {"error": "Could not parse pass@1"}, cmd_str

    except Exception as e:
        return False, {"error": str(e)}, cmd_str


def evaluate_csqa(
    base_model: str,
    lora_dir: str,
    output_dir: str,
    max_samples: int = -1,
    seed: int = 0,
    use_vllm: bool = True,
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Evaluate on CommonsenseQA.

    Simple implementation: generate answer letter and compare to gold.
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError as e:
        return False, {"error": f"Missing dependency: {e}"}, ""

    # Try vLLM first
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            HAVE_VLLM = True
        except ImportError:
            HAVE_VLLM = False
            use_vllm = False

    cmd_str = f"evaluate_csqa(base_model={base_model}, lora_dir={lora_dir}, use_vllm={use_vllm})"
    print(f"[Eval CSQA] {cmd_str[:100]}...")

    try:
        # Load dataset
        ds = load_dataset("tau/commonsense_qa", split="validation")
        if max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))

        # Build prompts
        prompts = []
        gold_answers = []

        for ex in ds:
            question = ex["question"]
            choices = ex["choices"]
            answer_key = ex["answerKey"]

            # Build choices text
            labels = choices["label"]
            texts = choices["text"]
            choices_text = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])

            prompt = (
                f"Question: {question}\n\n"
                f"Choices:\n{choices_text}\n\n"
                f"Answer with a single letter (A, B, C, D, or E):"
            )
            prompts.append(prompt)
            gold_answers.append(answer_key.upper())

        if use_vllm and HAVE_VLLM:
            # Load adapter config to get rank
            config_path = os.path.join(lora_dir, "adapter_config.json")
            with open(config_path) as f:
                cfg = json.load(f)
            r = cfg.get("r", 16)

            llm = None
            predictions = []
            try:
                pid = os.getpid()
                print(f"[vLLM CSQA] Initializing engine from PID {pid}...")

                llm = LLM(
                    model=base_model,
                    dtype="float16",
                    max_model_len=4096,
                    enable_lora=True,
                    max_lora_rank=r,
                    seed=seed,
                )

                sp = SamplingParams(
                    temperature=0.0,
                    max_tokens=8,
                )

                lora_req = LoRARequest("adapter", 1, lora_dir)
                outputs = llm.generate(prompts, sp, lora_request=lora_req)

                for out in outputs:
                    gen = out.outputs[0].text.strip().upper()
                    # Extract first letter A-E
                    match = re.search(r"[ABCDE]", gen)
                    pred = match.group(0) if match else ""
                    predictions.append(pred)

            finally:
                # Robust cleanup
                print(f"[vLLM CSQA] Starting engine cleanup...")
                if llm is not None:
                    if HAVE_VLLM_UTILS and shutdown_vllm_engine:
                        shutdown_vllm_engine(llm, verbose=True)
                    else:
                        # Fallback: try internal shutdown
                        try:
                            engine = getattr(llm, 'llm_engine', None)
                            if engine:
                                engine_core = getattr(engine, 'engine_core', None)
                                if engine_core:
                                    shutdown_fn = getattr(engine_core, 'shutdown', None)
                                    if shutdown_fn:
                                        shutdown_fn()
                        except Exception as e:
                            print(f"[vLLM CSQA] Fallback shutdown failed: {e}")
                        del llm
                gc.collect()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print(f"[vLLM CSQA] Engine cleanup complete")
        else:
            # Use transformers
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(model, lora_dir)
            model.eval()

            predictions = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=8,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                gen = gen.strip().upper()
                match = re.search(r"[ABCDE]", gen)
                pred = match.group(0) if match else ""
                predictions.append(pred)

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Calculate accuracy
        correct = sum(1 for p, g in zip(predictions, gold_answers) if p == g)
        total = len(gold_answers)
        acc = correct / total if total > 0 else 0.0

        # Save predictions
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "csqa_predictions.jsonl"), "w") as f:
            for i, (pred, gold) in enumerate(zip(predictions, gold_answers)):
                f.write(json.dumps({"idx": i, "pred": pred, "gold": gold, "correct": pred == gold}) + "\n")

        return True, {
            "acc": acc,
            "correct": correct,
            "total": total,
            "metric_name": "csqa_accuracy",
        }, cmd_str

    except Exception as e:
        import traceback
        return False, {"error": str(e), "traceback": traceback.format_exc()}, cmd_str


def evaluate_ifeval(
    base_model: str,
    lora_dir: str,
    output_dir: str,
    max_samples: int = -1,
    seed: int = 0,
    use_vllm: bool = True,
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Evaluate on IFEval (instruction following evaluation).

    Simplified implementation: measures basic instruction compliance.
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError as e:
        return False, {"error": f"Missing dependency: {e}"}, ""

    # Try vLLM first
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            HAVE_VLLM = True
        except ImportError:
            HAVE_VLLM = False
            use_vllm = False

    cmd_str = f"evaluate_ifeval(base_model={base_model}, lora_dir={lora_dir}, use_vllm={use_vllm})"
    print(f"[Eval IFEval] {cmd_str[:100]}...")

    try:
        # Load IFEval dataset
        ds = load_dataset("google/IFEval", split="train")  # IFEval only has train split
        if max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))

        prompts = []
        instructions = []

        for ex in ds:
            prompt = ex["prompt"]
            prompts.append(prompt)
            instructions.append(ex.get("instruction_id_list", []))

        # Load adapter config
        config_path = os.path.join(lora_dir, "adapter_config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        r = cfg.get("r", 16)

        if use_vllm and HAVE_VLLM:
            llm = None
            generations = []
            try:
                pid = os.getpid()
                print(f"[vLLM IFEval] Initializing engine from PID {pid}...")

                llm = LLM(
                    model=base_model,
                    dtype="float16",
                    max_model_len=4096,
                    enable_lora=True,
                    max_lora_rank=r,
                    seed=seed,
                )

                sp = SamplingParams(
                    temperature=0.0,
                    max_tokens=512,
                )

                lora_req = LoRARequest("adapter", 1, lora_dir)
                outputs = llm.generate(prompts, sp, lora_request=lora_req)

                generations = [out.outputs[0].text for out in outputs]

            finally:
                # Robust cleanup
                print(f"[vLLM IFEval] Starting engine cleanup...")
                if llm is not None:
                    if HAVE_VLLM_UTILS and shutdown_vllm_engine:
                        shutdown_vllm_engine(llm, verbose=True)
                    else:
                        try:
                            engine = getattr(llm, 'llm_engine', None)
                            if engine:
                                engine_core = getattr(engine, 'engine_core', None)
                                if engine_core:
                                    shutdown_fn = getattr(engine_core, 'shutdown', None)
                                    if shutdown_fn:
                                        shutdown_fn()
                        except Exception as e:
                            print(f"[vLLM IFEval] Fallback shutdown failed: {e}")
                        del llm
                gc.collect()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print(f"[vLLM IFEval] Engine cleanup complete")
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(model, lora_dir)
            model.eval()

            generations = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                generations.append(gen)

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Simple compliance check (simplified - full IFEval requires instruction-specific checks)
        # For now, measure response length and non-empty responses as basic compliance
        compliant = 0
        total = len(generations)

        for gen in generations:
            # Basic compliance: non-empty response with reasonable length
            if len(gen.strip()) > 10:
                compliant += 1

        compliance_rate = compliant / total if total > 0 else 0.0

        # Save generations
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "ifeval_generations.jsonl"), "w") as f:
            for i, (prompt, gen) in enumerate(zip(prompts, generations)):
                f.write(json.dumps({
                    "idx": i,
                    "prompt": prompt[:200],
                    "generation": gen[:500],
                    "gen_length": len(gen),
                }) + "\n")

        return True, {
            "compliance_rate": compliance_rate,
            "compliant": compliant,
            "total": total,
            "metric_name": "ifeval_compliance",
        }, cmd_str

    except Exception as e:
        import traceback
        return False, {"error": str(e), "traceback": traceback.format_exc()}, cmd_str


def run_evaluation(
    task: str,
    base_model: str,
    lora_dir: str,
    output_dir: str,
    lora_spectral_edit_root: str,
    max_samples: int = -1,
    seed: int = 0,
    use_vllm: bool = True,
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Run appropriate evaluation based on task.
    """
    eval_task = TASK_EVAL_MAP.get(task, task)

    if eval_task == "gsm8k":
        return evaluate_gsm8k(
            base_model, lora_dir, lora_spectral_edit_root,
            max_samples, seed, use_vllm
        )
    elif eval_task == "humaneval":
        return evaluate_humaneval(
            base_model, lora_dir, lora_spectral_edit_root,
            max_samples, seed, use_vllm
        )
    elif eval_task == "csqa":
        return evaluate_csqa(
            base_model, lora_dir, output_dir,
            max_samples, seed, use_vllm
        )
    elif eval_task == "ifeval":
        return evaluate_ifeval(
            base_model, lora_dir, output_dir,
            max_samples, seed, use_vllm
        )
    else:
        return False, {"error": f"Unknown eval task: {eval_task}"}, ""


# =============================================================================
# Main Pipeline
# =============================================================================

def process_adapter(
    adapter: AdapterInfo,
    output_root: Path,
    lora_spectral_edit_root: str,
    results_jsonl: Path,
    results_csv: Path,
    max_eval_samples: int = -1,
    seed: int = 42,
    use_vllm: bool = True,
    skip_baseline: bool = False,
    policies_to_run: Optional[List[str]] = None,
) -> List[EvalResult]:
    """
    Process a single adapter: create edits and evaluate.
    """
    results = []
    git_hash = get_git_commit_hash()

    if policies_to_run is None:
        policies_to_run = list(EDIT_POLICIES.keys())

    # Evaluate baseline
    if not skip_baseline:
        print(f"\n[Baseline] Evaluating {adapter.task}/{adapter.peft_method}/{adapter.seed}")

        eval_output_dir = output_root / adapter.task / adapter.peft_method / adapter.seed / "baseline" / "eval"

        success, metrics, cmd_str = run_evaluation(
            adapter.task,
            adapter.base_model,
            adapter.adapter_dir,
            str(eval_output_dir),
            lora_spectral_edit_root,
            max_eval_samples,
            seed,
            use_vllm,
        )

        metric_name = metrics.get("metric_name", "unknown")
        metric_value = metrics.get("acc", metrics.get("pass@1", metrics.get("compliance_rate", 0.0)))

        result = EvalResult(
            timestamp=datetime.now().isoformat(),
            task=adapter.task,
            peft_method=adapter.peft_method,
            rank=adapter.rank,
            seed=adapter.seed,
            policy="baseline",
            adapter_path=adapter.adapter_dir,
            eval_task=TASK_EVAL_MAP.get(adapter.task, adapter.task),
            metric_name=metric_name,
            metric_value=metric_value if success else 0.0,
            eval_command=cmd_str,
            git_commit_hash=git_hash,
            extra_metrics=metrics if success else {"error": metrics.get("error", "unknown")},
        )

        results.append(result)
        append_result_jsonl(result, results_jsonl)
        update_results_csv(results_jsonl, results_csv)

        if success:
            print(f"[Baseline] {metric_name}={metric_value:.4f}")
        else:
            print(f"[Baseline] FAILED: {metrics.get('error', 'unknown')}")

    # Process each editing policy
    for policy_name in policies_to_run:
        if policy_name not in EDIT_POLICIES:
            continue

        policy_config = EDIT_POLICIES[policy_name]

        print(f"\n[Policy: {policy_name}] Processing {adapter.task}/{adapter.peft_method}/{adapter.seed}")

        # Create edited adapter
        edited_dir = output_root / adapter.task / adapter.peft_method / adapter.seed / "svd_edit" / policy_name

        if not edited_dir.exists():
            success, result_or_error = run_spectral_edit(
                adapter.base_model,
                adapter.adapter_dir,
                str(edited_dir),
                policy_name,
                policy_config,
                lora_spectral_edit_root,
                seed,
            )

            if not success:
                print(f"[Edit] FAILED for {policy_name}: {result_or_error[:200]}")

                result = EvalResult(
                    timestamp=datetime.now().isoformat(),
                    task=adapter.task,
                    peft_method=adapter.peft_method,
                    rank=adapter.rank,
                    seed=adapter.seed,
                    policy=policy_name,
                    adapter_path=str(edited_dir),
                    eval_task=TASK_EVAL_MAP.get(adapter.task, adapter.task),
                    metric_name="edit_failed",
                    metric_value=0.0,
                    eval_command="",
                    git_commit_hash=git_hash,
                    extra_metrics={"error": result_or_error[:500]},
                )
                results.append(result)
                append_result_jsonl(result, results_jsonl)
                update_results_csv(results_jsonl, results_csv)
                continue
        else:
            print(f"[Edit] Using existing edited adapter at {edited_dir}")

        # Evaluate edited adapter
        eval_output_dir = output_root / adapter.task / adapter.peft_method / adapter.seed / "svd_edit" / policy_name / "eval"

        success, metrics, cmd_str = run_evaluation(
            adapter.task,
            adapter.base_model,
            str(edited_dir),
            str(eval_output_dir),
            lora_spectral_edit_root,
            max_eval_samples,
            seed,
            use_vllm,
        )

        metric_name = metrics.get("metric_name", "unknown")
        metric_value = metrics.get("acc", metrics.get("pass@1", metrics.get("compliance_rate", 0.0)))

        result = EvalResult(
            timestamp=datetime.now().isoformat(),
            task=adapter.task,
            peft_method=adapter.peft_method,
            rank=adapter.rank,
            seed=adapter.seed,
            policy=policy_name,
            adapter_path=str(edited_dir),
            eval_task=TASK_EVAL_MAP.get(adapter.task, adapter.task),
            metric_name=metric_name,
            metric_value=metric_value if success else 0.0,
            eval_command=cmd_str,
            git_commit_hash=git_hash,
            extra_metrics=metrics if success else {"error": metrics.get("error", "unknown")},
        )

        results.append(result)
        append_result_jsonl(result, results_jsonl)
        update_results_csv(results_jsonl, results_csv)

        if success:
            print(f"[{policy_name}] {metric_name}={metric_value:.4f}")
        else:
            print(f"[{policy_name}] FAILED: {metrics.get('error', 'unknown')}")

        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run LoRA spectral editing experiments on Llama-3.1-8B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--runs_root",
        type=str,
        default="/home/zailongtian/workspace/peft-sft-lab/runs/meta-llama-Llama-3.1-8B",
        help="Root directory containing LoRA adapters",
    )
    parser.add_argument(
        "--lora_spectral_edit_root",
        type=str,
        default="/home/zailongtian/workspace/lora-spectral-edit",
        help="Root directory of lora-spectral-edit repository",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/zailongtian/workspace/peft-sft-lab/outputs/spectral_edit_eval/llama31",
        help="Output directory for results",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["metamath", "magicoder", "alpaca", "csqa"],
        help="Tasks to process",
    )
    parser.add_argument(
        "--peft_methods",
        nargs="+",
        default=None,
        help="PEFT methods to filter (default: all)",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help="Editing policies to run (default: all)",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        default=None,
        help="Ranks to filter (e.g., rank-16)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Seeds to filter (e.g., seed42)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=-1,
        help="Maximum samples for evaluation (-1 for all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for faster inference",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline evaluation",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only discover adapters and save manifest",
    )
    parser.add_argument(
        "--single_adapter",
        type=str,
        default=None,
        help="Path to a single adapter to process (for testing)",
    )

    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_root = Path(args.output_root)

    # Setup output paths
    results_jsonl = output_root / "results.jsonl"
    results_csv = output_root / "results.csv"
    manifest_path = output_root / "adapter_manifest.json"
    summary_path = output_root / "summary.md"

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LoRA Spectral Edit Evaluation Pipeline")
    print("=" * 70)
    print(f"Runs root: {runs_root}")
    print(f"Output root: {output_root}")
    print(f"Tasks: {args.tasks}")
    print(f"Use vLLM: {args.use_vllm}")
    print("=" * 70)

    # Discover adapters
    if args.single_adapter:
        # Process single adapter for testing
        adapter_dir = Path(args.single_adapter)
        rel_path = adapter_dir.relative_to(runs_root)
        parts = rel_path.parts

        adapters = [AdapterInfo(
            task=parts[0],
            peft_method=parts[1],
            profile=parts[2],
            rank=parts[3],
            seed=parts[4],
            adapter_dir=str(adapter_dir),
        )]
    else:
        adapters = discover_adapters(runs_root, args.tasks)

    # Apply filters
    if args.peft_methods:
        adapters = [a for a in adapters if a.peft_method in args.peft_methods]
    if args.ranks:
        adapters = [a for a in adapters if a.rank in args.ranks]
    if args.seeds:
        adapters = [a for a in adapters if a.seed in args.seeds]

    print(f"\nDiscovered {len(adapters)} adapters")

    # Save manifest
    save_manifest(adapters, manifest_path)

    if args.dry_run:
        print("\n[Dry run] Exiting without processing")
        return

    # Process each adapter
    all_results = []
    for i, adapter in enumerate(adapters):
        print(f"\n{'=' * 70}")
        print(f"Processing adapter {i+1}/{len(adapters)}")
        print(f"Task: {adapter.task}, Method: {adapter.peft_method}, Seed: {adapter.seed}")
        print(f"Path: {adapter.adapter_dir}")
        print("=" * 70)

        try:
            results = process_adapter(
                adapter,
                output_root,
                args.lora_spectral_edit_root,
                results_jsonl,
                results_csv,
                args.max_eval_samples,
                args.seed,
                args.use_vllm,
                args.skip_baseline,
                args.policies,
            )
            all_results.extend(results)
        except Exception as e:
            import traceback
            print(f"[Error] Failed to process adapter: {e}")
            traceback.print_exc()
            continue

    # Generate summary
    generate_summary_markdown(results_jsonl, summary_path)

    print("\n" + "=" * 70)
    print("Pipeline Complete")
    print("=" * 70)
    print(f"Results JSONL: {results_jsonl}")
    print(f"Results CSV: {results_csv}")
    print(f"Summary: {summary_path}")
    print(f"Total results: {len(all_results)}")


if __name__ == "__main__":
    main()
