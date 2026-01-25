#!/usr/bin/env python3
"""
Run spectral edits on LoRA/LoRA+ adapters and evaluate with lm_eval (vLLM).

This script:
  - Discovers final adapters under one or more runs roots (skipping checkpoints).
  - Applies spectral edits using the CLI
  - Evaluates baseline (no adapter), unedited adapter, and edited adapters
    with lm_eval harness using the vLLM backend.

IMPORTANT (modified behavior):
  - vLLM LoRA loading is DISABLED. Any adapter evaluation is done via FORCE MERGE:
      adapter -> merged full model dir -> lm_eval(vLLM) on merged dir.
  - After each run_lm_eval call, we aggressively kill leftover vLLM processes
    (process group) and clear CUDA cache to avoid zombie GPU memory usage.

Outputs are stored under:
  {out_root}/{base_model_tag}/{task}/{adapter_type}/{profile}/{rank}/{seed}/{variant}/

Edited adapters are stored under:
  {out_root}/edited_adapters/{base_model_tag}/{task}/{adapter_type}/{profile}/{rank}/{seed}/{policy}/

Usage:
  python scripts/run_lm_eval_harness_spectral_edits.py \
    --runs_roots /path/to/meta-llama-Llama-3.1-8B /path/to/Qwen-Qwen3-8B \
    --out_root /path/to/lm_eval_outputs \
    --policies abs_select smooth_abs random_index grad_direction \
    --merge_device cpu
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import signal
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# Constants
# ============================================================================

EDIT_POLICIES = ["random_index", "smooth_abs", "abs_select", "grad_direction"]

METHOD_TO_MODE = {
    "random_index": "random_index",
    "smooth_abs": "smooth_abs",
    "abs_select": "abs_select",
    "grad_direction": "gd",
}

BASE_MODEL_TAG_TO_ID = {
    "meta-llama-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Qwen-Qwen3-8B": "Qwen/Qwen3-8B",
}

TASK_DIR_TO_LM_EVAL = {
    "math": "gsm8k",
    "code": "humaneval",
    "alpaca": "ifeval",
    "csqa": "commonsense_qa",
}

TASK_CONFIGS = {
    "math": {
        "num_fewshot": 5,
        "gen_kwargs": "temperature=0,top_p=1",
        "gpu_memory_utilization": 0.95,
        "confirm_unsafe_code": False,
    },
    "code": {
        "num_fewshot": 0,
        "gen_kwargs": "temperature=0,top_p=1",
        "gpu_memory_utilization": 0.90,
        "confirm_unsafe_code": True,
    },
    "alpaca": {
        "num_fewshot": None,
        "gen_kwargs": "max_gen_toks=2048,temperature=0,top_p=1",
        "gpu_memory_utilization": 0.95,
        "confirm_unsafe_code": False,
    },
    "csqa": {
        "num_fewshot": 0,
        "gen_kwargs": None,
        "gpu_memory_utilization": 0.85,
        "confirm_unsafe_code": False,
    },
}

TASK_METRIC_KEYS = {
    "math": ["acc", "exact_match", "acc_norm", "exact_match_norm"],
    "code": ["pass@1", "pass@1,normalized"],
    "alpaca": [
        "prompt_level_strict_accuracy",
        "strict_accuracy",
        "inst_level_strict_accuracy",
        "acc",
    ],
    "csqa": ["acc", "acc_norm"],
}


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class AdapterInfo:
    adapter_dir: Path
    base_model_tag: str
    base_model_id: str
    task: str
    adapter_type: str
    profile: str
    rank: str
    seed: str

    @property
    def run_id(self) -> str:
        return f"{self.task}_{self.adapter_type}_{self.profile}_{self.rank}_{self.seed}"


@dataclass
class EvalRecord:
    timestamp: str
    base_model_tag: str
    base_model_id: str
    task: str
    lm_eval_task: str
    adapter_type: str
    profile: str
    rank: str
    seed: str
    variant: str
    adapter_dir: Optional[str]
    edited_adapter_dir: Optional[str]
    output_dir: str
    used_vllm_lora: bool
    used_fallback_merge: bool
    metric_key: Optional[str]
    metric_value: Optional[float]
    metrics: Optional[Dict[str, Any]]
    num_examples: Optional[int]
    error: Optional[str] = None


# ============================================================================
# Utilities
# ============================================================================

def is_checkpoint_path(path: Path) -> bool:
    """Return True if any path segment is a checkpoint directory."""
    return any(part.startswith("checkpoint-") for part in path.parts)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def has_adapter_weights(adapter_dir: Path) -> bool:
    return (adapter_dir / "adapter_model.safetensors").exists() or (
        adapter_dir / "adapter_model.bin"
    ).exists()


def parse_rank_value(rank: Optional[str]) -> Optional[int]:
    if not rank:
        return None
    match = re.search(r"\d+", rank)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def read_lora_rank(adapter_dir: Path, rank_hint: Optional[str]) -> Optional[int]:
    cfg = read_json(adapter_dir / "adapter_config.json")
    if cfg and "r" in cfg:
        try:
            return int(cfg["r"])
        except Exception:
            pass
    return parse_rank_value(rank_hint)


def parse_adapter_path(adapter_dir: Path, runs_root: Path) -> Optional[AdapterInfo]:
    try:
        rel_parts = adapter_dir.relative_to(runs_root).parts
    except ValueError:
        return None

    runs_root_name = runs_root.name
    if runs_root_name in BASE_MODEL_TAG_TO_ID or runs_root_name.startswith("meta-llama") or \
       runs_root_name.startswith("Qwen"):
        base_model_tag = runs_root_name
        offset = 0
    else:
        if not rel_parts:
            return None
        base_model_tag = rel_parts[0]
        offset = 1

    if len(rel_parts) < offset + 2:
        return None

    task = rel_parts[offset]
    adapter_type = rel_parts[offset + 1].lower()
    profile = None
    rank = None
    seed = None

    for part in rel_parts[offset + 2 :]:
        part_lower = part.lower()
        if part_lower.startswith("profile-"):
            profile = part[len("profile-") :]
        elif part_lower.startswith("rank-"):
            rank = part[len("rank-") :]
        elif part_lower.startswith("seed"):
            seed = part[len("seed") :]

    if task not in TASK_DIR_TO_LM_EVAL:
        return None
    if adapter_type not in {"lora", "loraplus"}:
        return None
    if not profile or not rank or not seed:
        return None

    base_model_id = BASE_MODEL_TAG_TO_ID.get(base_model_tag)
    if not base_model_id:
        cfg = read_json(adapter_dir / "adapter_config.json")
        if cfg:
            base_model_id = cfg.get("base_model_name_or_path")
    if not base_model_id:
        return None

    return AdapterInfo(
        adapter_dir=adapter_dir,
        base_model_tag=base_model_tag,
        base_model_id=base_model_id,
        task=task,
        adapter_type=adapter_type,
        profile=profile,
        rank=rank,
        seed=seed,
    )


def discover_adapters(runs_roots: Iterable[Path], tasks: Optional[List[str]]) -> Tuple[List[AdapterInfo], int]:
    adapters: List[AdapterInfo] = []
    skipped = 0
    seen: set[str] = set()

    for runs_root in runs_roots:
        for root, dirs, files in os.walk(runs_root):
            root_path = Path(root)

            if is_checkpoint_path(root_path):
                skipped += 1
                dirs.clear()
                continue

            dirs[:] = [d for d in dirs if not d.startswith("checkpoint-")]

            if "adapter_config.json" not in files:
                continue
            if "adapter_model.safetensors" not in files and "adapter_model.bin" not in files:
                continue

            info = parse_adapter_path(root_path, runs_root)
            if not info:
                continue
            if tasks and info.task not in tasks:
                continue
            key = str(info.adapter_dir)
            if key in seen:
                continue
            seen.add(key)
            adapters.append(info)

    return adapters, skipped


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def format_cmd(cmd: List[str], env_prefix: Optional[Dict[str, str]] = None) -> str:
    cmd_str = shlex.join(cmd)
    if not env_prefix:
        return cmd_str
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_prefix.items())
    return f"{prefix} {cmd_str}"


def ensure_error_logs(output_dir: Path, message: str) -> None:
    cmd_path = output_dir / "cmd.txt"
    if not cmd_path.exists():
        write_text(cmd_path, f"# skipped: {message}\n")
    stdout_path = output_dir / "stdout.txt"
    if not stdout_path.exists():
        write_text(stdout_path, "")
    stderr_path = output_dir / "stderr.txt"
    if not stderr_path.exists():
        write_text(stderr_path, message + "\n")


def results_json_path(output_dir: Path) -> Path:
    return output_dir / "results.json"


def results_json_tmp_path(output_dir: Path) -> Path:
    return output_dir / "results.json.tmp"


def write_results_json_atomic(output_dir: Path, data: Dict[str, Any]) -> None:
    tmp_path = results_json_tmp_path(output_dir)
    final_path = results_json_path(output_dir)
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, final_path)


def is_valid_results_json(data: Dict[str, Any], lm_task: str) -> bool:
    if not isinstance(data, dict):
        return False
    results = data.get("results")
    if not isinstance(results, dict):
        return False
    if lm_task not in results:
        return False
    return True


def read_tail(path: Path, max_bytes: int = 20000) -> str:
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    for path in src_dir.glob("tokenizer*"):
        if path.is_file():
            dest = dst_dir / path.name
            if not dest.exists():
                shutil.copy2(path, dest)
    for name in ("special_tokens_map.json", "added_tokens.json"):
        src = src_dir / name
        if src.exists() and not (dst_dir / name).exists():
            shutil.copy2(src, dst_dir / name)


def has_tokenizer_files(model_dir: Path) -> bool:
    return (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer.model").exists()


def validate_merged_model(model_dir: Path) -> Optional[str]:
    if not model_dir.exists():
        return "Merged model directory does not exist."

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        if not index or "weight_map" not in index:
            return f"Invalid index JSON: {index_path}"
        weight_map = index.get("weight_map", {})
        missing = [p for p in set(weight_map.values()) if not (model_dir / p).exists()]
        if missing:
            return f"Missing shard(s): {missing}"
    else:
        shards = list(model_dir.glob("model-*-of-*.safetensors"))
        if shards:
            max_total = None
            present = set()
            for shard in shards:
                match = re.search(r"model-(\d+)-of-(\d+)\.safetensors", shard.name)
                if not match:
                    continue
                idx = int(match.group(1))
                total = int(match.group(2))
                present.add(idx)
                max_total = total if max_total is None else max(max_total, total)
            if max_total:
                missing = [i for i in range(1, max_total + 1) if i not in present]
                if missing:
                    return f"Missing shard indices: {missing} of {max_total}"
        else:
            if not (model_dir / "model.safetensors").exists() and not (model_dir / "pytorch_model.bin").exists():
                return "No model weights found in merged model directory."

    try:
        from safetensors import safe_open
    except Exception:
        return None

    for st_file in model_dir.glob("*.safetensors"):
        try:
            with safe_open(st_file, framework="pt") as f:
                _ = list(f.keys())
        except Exception as exc:
            return f"Safetensors validation failed for {st_file.name}: {exc}"

    return None


def ensure_tokenizer_assets(
    model_dir: Path,
    adapter_dir: Path,
    base_model_id: str,
) -> Optional[str]:
    copy_tokenizer_files(adapter_dir, model_dir)
    if has_tokenizer_files(model_dir):
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        return f"Failed to import AutoTokenizer: {exc}"
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
        tokenizer.save_pretrained(model_dir)
    except Exception as exc:
        return f"Tokenizer save failed: {exc}"
    if not has_tokenizer_files(model_dir):
        return "Tokenizer assets missing after save."
    return None


def parse_lm_eval_results(raw: Dict[str, Any], lm_task: str) -> Dict[str, Any]:
    if "results" in raw and isinstance(raw["results"], dict):
        return raw["results"].get(lm_task, {})
    if lm_task in raw and isinstance(raw[lm_task], dict):
        return raw[lm_task]
    return {}


def select_metric(task: str, metrics: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    desired = TASK_METRIC_KEYS.get(task, [])
    if not desired:
        return None, None
    for base in desired:
        if base in metrics:
            try:
                return base, float(metrics[base])
            except Exception:
                return base, None
        for key, value in metrics.items():
            if key.startswith(base + ","):
                try:
                    return key, float(value)
                except Exception:
                    return key, None
    return None, None


def extract_num_examples(raw: Dict[str, Any], lm_task: str) -> Optional[int]:
    for key in ("num_examples", "total", "n_samples", "n-samples"):
        val = raw.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, dict):
            task_val = val.get(lm_task)
            if isinstance(task_val, int):
                return task_val
            if isinstance(task_val, dict):
                for subkey in ("effective", "original", "n"):
                    subval = task_val.get(subkey)
                    if isinstance(subval, int):
                        return subval
    n_map = raw.get("n")
    if isinstance(n_map, dict):
        val = n_map.get(lm_task)
        if isinstance(val, int):
            return val
    return None


def load_existing_results(
    output_dir: Path,
    task: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[int], Optional[str], Optional[Path]]:
    result_path = results_json_path(output_dir)
    if not result_path.exists():
        return None, None, None, None, None
    raw = read_json(result_path)
    if raw is None:
        return None, None, None, f"Failed to read results JSON: {result_path}", result_path
    lm_task = TASK_DIR_TO_LM_EVAL[task]
    if not is_valid_results_json(raw, lm_task):
        return None, None, None, f"Invalid results JSON: {result_path}", result_path
    metrics = parse_lm_eval_results(raw, lm_task)
    num_examples = extract_num_examples(raw, lm_task)
    return raw, metrics, num_examples, None, result_path


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
    preserve_energy: str = "l1",
    dry_run: bool = False,
) -> Tuple[bool, Optional[str]]:
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
        "--preserve_energy", preserve_energy,
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
        print(f"[DRY-RUN] Would run: {shlex.join(cmd)}")
        return True, None

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}:{env.get('PYTHONPATH', '')}".strip(":")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "Unknown error"
            return False, f"Edit failed (code {result.returncode}): {error_msg}"
        if not has_adapter_weights(out_dir):
            return False, "Edit completed but no adapter weights found in output"
        return True, None
    except subprocess.TimeoutExpired:
        return False, "Edit timed out after 30 minutes"
    except Exception as exc:
        return False, f"Edit failed with exception: {exc}"


# ============================================================================
# lm_eval execution (vLLM)
# ============================================================================

def _cleanup_vllm_and_cuda(proc: Optional[subprocess.Popen]) -> None:
    """
    Best-effort cleanup:
      - Kill the entire process group started by lm_eval (vLLM workers should be in it).
      - Release Python-side memory and empty CUDA cache.
    Intentionally aggressive to prevent zombie vLLM workers holding GPU memory.
    """
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            pass

        # brief grace, then hard kill
        time.sleep(1.0)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass

    gc.collect()

    # Clear CUDA cache in *this* process; vLLM workers must be killed to actually release VRAM.
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def build_lm_eval_command(
    base_model: str,
    task: str,
    tensor_parallel_size: int,
    output_path: Optional[Path],
    gpu_memory_utilization: Optional[float] = None,
    max_num_seqs: Optional[int] = None,
) -> Tuple[List[str], Dict[str, str]]:
    lm_task = TASK_DIR_TO_LM_EVAL[task]
    task_cfg = TASK_CONFIGS[task]
    mem_util = gpu_memory_utilization if gpu_memory_utilization is not None else task_cfg["gpu_memory_utilization"]
    model_args = (
        f"pretrained={base_model},"
        f"tensor_parallel_size={tensor_parallel_size},"
        f"dtype=auto,"
        f"gpu_memory_utilization={mem_util}"
    )
    if max_num_seqs is not None:
        model_args += f",max_num_seqs={max_num_seqs}"

    cmd = [
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        lm_task,
        "--batch_size",
        "auto",
    ]

    if task_cfg["num_fewshot"] is not None:
        cmd.extend(["--num_fewshot", str(task_cfg["num_fewshot"])])
    if task_cfg["gen_kwargs"]:
        cmd.extend(["--gen_kwargs", task_cfg["gen_kwargs"]])
    if task_cfg["confirm_unsafe_code"]:
        cmd.append("--confirm_run_unsafe_code")
    if output_path:
        cmd.extend(["--output_path", str(output_path)])

    env = {}
    if task_cfg["confirm_unsafe_code"]:
        env["HF_ALLOW_CODE_EVAL"] = "1"

    return cmd, env


def run_lm_eval(
    base_model: str,
    task: str,
    output_dir: Path,
    tensor_parallel_size: int,
    log_suffix: Optional[str] = None,
    timeout_s: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[int], Optional[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "lm_eval_out"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_json_path(output_dir)
    results_tmp_path = results_json_tmp_path(output_dir)
    if results_tmp_path.exists():
        results_tmp_path.unlink()
    if results_path.exists():
        results_path.unlink()

    for path in list(output_dir.glob("results*.json")):
        if path != results_path:
            path.unlink(missing_ok=True)
    for path in results_dir.glob("results*.json"):
        path.unlink(missing_ok=True)

    base_mem_util = TASK_CONFIGS[task]["gpu_memory_utilization"]
    oom_mem_util = max(0.5, round(base_mem_util - 0.1, 2))
    log_prefix = f"_{log_suffix}" if log_suffix else ""
    log_path = output_dir / (f"eval{log_prefix}.log")

    for attempt in range(2):
        max_num_seqs = None if attempt == 0 else 128
        mem_override = None if attempt == 0 else oom_mem_util
        cmd_suffix = f"{log_prefix}" if attempt == 0 else f"{log_prefix}_oom_retry"

        try:
            cmd, extra_env = build_lm_eval_command(
                base_model=base_model,
                task=task,
                tensor_parallel_size=tensor_parallel_size,
                output_path=results_dir,
                gpu_memory_utilization=mem_override,
                max_num_seqs=max_num_seqs,
            )
        except Exception as exc:
            write_text(output_dir / f"cmd{cmd_suffix}.txt", f"# failed to build lm_eval command: {exc}\n")
            return None, None, None, f"lm_eval command build failed: {exc}"

        env = os.environ.copy()
        env.update(extra_env)

        write_text(output_dir / f"cmd{cmd_suffix}.txt", format_cmd(cmd, extra_env if extra_env else None))

        start_time = time.time()
        log_mode = "w" if attempt == 0 else "a"
        proc: Optional[subprocess.Popen] = None

        try:
            with open(log_path, log_mode) as log_file:
                if attempt > 0:
                    log_file.write(
                        f"\n# retry: oom fallback (gpu_memory_utilization={mem_override}, max_num_seqs={max_num_seqs})\n"
                    )
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=output_dir,
                        env=env,
                        start_new_session=True,  # gives us a new process group to kill
                    )
                except Exception as exc:
                    return None, None, None, f"lm_eval failed to start: {exc}"

                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    return None, None, None, f"lm_eval timed out after {timeout_s} seconds"
        finally:
            # Always kill leftovers + clear cache even on success.
            _cleanup_vllm_and_cuda(proc)

        if log_suffix:
            try:
                shutil.copy2(log_path, output_dir / "eval.log")
            except Exception:
                pass

        if proc is None:
            return None, None, None, "lm_eval did not start"

        if proc.returncode != 0:
            log_tail = read_tail(log_path)
            if attempt == 0 and is_vllm_oom(log_tail):
                continue
            error_msg = log_tail[-2000:] if log_tail else "Unknown error"
            return None, None, None, f"lm_eval failed (code {proc.returncode}): {error_msg}"

        recent_json = find_recent_result_json(results_dir, start_time - 1)
        if recent_json is None:
            recent_json = find_recent_result_json(output_dir, start_time - 1)
        if recent_json is None:
            return None, None, None, "lm_eval completed but no results JSON found"

        raw = read_json(recent_json)
        if raw is None:
            return None, None, None, f"Failed to read results JSON: {recent_json}"

        lm_task = TASK_DIR_TO_LM_EVAL[task]
        if not is_valid_results_json(raw, lm_task):
            return None, None, None, f"Invalid results JSON: {recent_json}"

        write_results_json_atomic(output_dir, raw)
        metrics = parse_lm_eval_results(raw, lm_task)
        num_examples = extract_num_examples(raw, lm_task)
        return raw, metrics, num_examples, None

    return None, None, None, "lm_eval failed after OOM retry"


def extract_json_from_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


def find_recent_result_json(output_dir: Path, min_mtime: float) -> Optional[Path]:
    candidates = []
    for path in output_dir.rglob("*.json"):
        try:
            name = path.name.lower()
            if not (name.startswith("results") or name.startswith("result")):
                continue
            if path.is_file() and path.stat().st_mtime >= min_mtime:
                candidates.append(path)
        except Exception:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def is_vllm_oom(stderr: str) -> bool:
    lower = (stderr or "").lower()
    return "out of memory" in lower or "cuda oom" in lower


def merge_adapter(
    base_model_id: str,
    adapter_dir: Path,
    output_dir: Path,
    device: str,
) -> Tuple[Optional[Path], Optional[str]]:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as exc:
        return None, f"Failed to import merge dependencies: {exc}"

    if output_dir.exists() and (output_dir / "config.json").exists():
        validation_error = validate_merged_model(output_dir)
        if validation_error:
            shutil.rmtree(output_dir)
        else:
            tok_error = ensure_tokenizer_assets(output_dir, adapter_dir, base_model_id)
            if tok_error:
                return output_dir, tok_error
            return output_dir, None

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="auto",
            device_map=device,
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(model, adapter_dir)
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
        tokenizer.save_pretrained(output_dir)
        tok_error = ensure_tokenizer_assets(output_dir, adapter_dir, base_model_id)
        if tok_error:
            return output_dir, tok_error
        validation_error = validate_merged_model(output_dir)
        if validation_error:
            return output_dir, validation_error
    except Exception as exc:
        return None, f"Merge failed: {exc}"
    finally:
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return output_dir, None


# ============================================================================
# Main
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run spectral edits and evaluate with lm_eval (vLLM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--runs_roots",
        type=Path,
        nargs="+",
        required=True,
        help="One or more roots containing adapters (base-model directories or parent runs root).",
    )
    p.add_argument(
        "--out_root",
        type=Path,
        required=True,
        help="Output root for edited adapters and lm_eval outputs.",
    )
    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["math", "code", "alpaca", "csqa"],
        choices=list(TASK_DIR_TO_LM_EVAL.keys()),
        help="Tasks to include (default: all).",
    )
    p.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=EDIT_POLICIES,
        choices=EDIT_POLICIES,
        help="Spectral edit policies to run.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for spectral edit.",
    )
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="vLLM tensor parallel size.",
    )
    p.add_argument(
        "--merge_device",
        type=str,
        default="cpu",
        help="Device for merge_and_unload (e.g., cpu, cuda, auto).",
    )
    p.add_argument(
        "--eval_timeout_s",
        type=int,
        default=None,
        help="Optional timeout (seconds) per lm_eval run.",
    )
    p.add_argument(
        "--adapter_filter",
        type=str,
        default=None,
        help="Only process adapters matching this substring (debug).",
    )
    p.add_argument(
        "--adapter_types",
        type=str,
        nargs="+",
        choices=["lora", "loraplus"],
        default=None,
        help="Limit processing to specific adapter types.",
    )
    p.add_argument(
        "--edited_out_dir",
        type=Path,
        default=None,
        help="Root directory for edited adapters when --keep_edited_adapter is set.",
    )
    p.add_argument(
        "--keep_edited_adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep edited adapters on disk (default: use temporary directories).",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing lm_eval results in the output directory when available.",
    )
    p.add_argument(
        "--reuse_results",
        action="store_true",
        help="Deprecated alias for --resume.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Discover adapters and print planned actions without running.",
    )

    p.add_argument("--calib_samples", type=int, default=32, help="Calibration samples for spectral edit")
    p.add_argument("--calib_batch_size", type=int, default=2, help="Calibration batch size")
    p.add_argument(
        "--preserve_energy",
        type=str,
        default="l1",
        help="Spectral edit preserve_energy setting (e.g., l1, none).",
    )
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

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.reuse_results:
        args.resume = True

    runs_roots = [r.resolve() for r in args.runs_roots]
    for root in runs_roots:
        if not root.exists():
            print(f"[ERROR] Runs root does not exist: {root}")
            sys.exit(1)

    tasks = args.tasks
    out_root = args.out_root.resolve()
    edited_root = None
    if args.keep_edited_adapter:
        edited_root = args.edited_out_dir.resolve() if args.edited_out_dir else (out_root / "edited_adapters")
        edited_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Spectral Edit + lm_eval Harness Driver")
    print("=" * 70)
    print(f"Runs roots: {', '.join(str(r) for r in runs_roots)}")
    print(f"Output root: {out_root}")
    print(f"Tasks: {tasks}")
    print(f"Policies: {args.policies}")
    print("Adapter eval mode: FORCE MERGE (no vLLM LoRA loading)")
    print("=" * 70)

    print("\n[1/4] Discovering adapters...")
    adapters, skipped = discover_adapters(runs_roots, tasks)
    if args.adapter_filter:
        adapters = [a for a in adapters if args.adapter_filter in str(a.adapter_dir)]
        print(f"  After filter '{args.adapter_filter}': {len(adapters)} adapters")
    if args.adapter_types:
        allowed_types = set(args.adapter_types)
        adapters = [a for a in adapters if a.adapter_type in allowed_types]
        print(f"  After adapter_types {sorted(allowed_types)}: {len(adapters)} adapters")
    print(f"  Found {len(adapters)} adapters")
    print(f"  Skipped {skipped} checkpoint directories")

    if not adapters:
        print("[ERROR] No adapters found.")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY-RUN] Planned actions:")
        for adapter in adapters[:5]:
            print(f"  Adapter: {adapter.adapter_dir}")
            print(f"    Task: {adapter.task}, Type: {adapter.adapter_type}")
            print(f"    Profile: {adapter.profile}, Rank: {adapter.rank}, Seed: {adapter.seed}")
            print(f"    Base model: {adapter.base_model_id}")
            for policy in args.policies:
                print(f"    Edit policy: {policy}")
        if len(adapters) > 5:
            print(f"  ... and {len(adapters) - 5} more adapters")
        print("\n[DRY-RUN] No changes made.")
        return

    summary_records: List[EvalRecord] = []

    if args.keep_edited_adapter:
        print("\n[2/4] Editing adapters...")
        for i, adapter in enumerate(adapters, 1):
            print(f"\n[{i}/{len(adapters)}] {adapter.run_id}")
            print(f"  Adapter: {adapter.adapter_dir}")

            for policy in args.policies:
                edited_dir = (
                    edited_root
                    / adapter.base_model_tag
                    / adapter.task
                    / adapter.adapter_type
                    / f"profile-{adapter.profile}"
                    / f"rank-{adapter.rank}"
                    / f"seed{adapter.seed}"
                    / policy
                )

                if edited_dir.exists() and has_adapter_weights(edited_dir):
                    print(f"  [SKIP EDIT] {policy} already exists")
                    continue
                if edited_dir.exists():
                    shutil.rmtree(edited_dir)

                print(f"  [EDIT] {policy}")
                success, error = run_spectral_edit(
                    adapter_dir=adapter.adapter_dir,
                    out_dir=edited_dir,
                    edit_method=policy,
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
                    preserve_energy=args.preserve_energy,
                )
                if not success:
                    print(f"  [EDIT FAILED] {policy}: {error}")
                gc.collect()
    else:
        print("\n[2/4] Skipping persistent edits (temporary edited adapters will be used).")

    print("\n[3/4] Evaluating with lm_eval...")
    for i, adapter in enumerate(adapters, 1):
        print(f"\n[{i}/{len(adapters)}] {adapter.run_id}")

        variants: List[Tuple[str, Optional[Path], Optional[str], bool]] = [
            ("baseline", None, None, False),
            ("unedited", adapter.adapter_dir, None, False),
        ]
        for policy in args.policies:
            variants.append((f"edited/{policy}", None, policy, True))

        def evaluate_variant_with_merge(adapter_path: Optional[Path]) -> Tuple[
            Optional[Dict[str, Any]],
            Optional[Dict[str, Any]],
            Optional[int],
            Optional[str],
            bool,
            bool,
        ]:
            used_lora = False
            used_merge = False
            lm_error: Optional[str] = None
            raw = None
            metrics = None
            num_examples = None

            if adapter_path is not None:
                if not adapter_path.exists():
                    return None, None, None, f"Adapter path missing: {adapter_path}", False, False
                if not has_adapter_weights(adapter_path):
                    return None, None, None, f"Adapter weights missing: {adapter_path}", False, False

            try:
                if adapter_path is None:
                    # Baseline: directly eval base model with vLLM.
                    raw, metrics, num_examples, lm_error = run_lm_eval(
                        base_model=adapter.base_model_id,
                        task=adapter.task,
                        output_dir=output_dir,
                        tensor_parallel_size=args.tensor_parallel_size,
                        timeout_s=args.eval_timeout_s,
                    )
                    return raw, metrics, num_examples, lm_error, used_lora, False

                # Adapter: FORCE MERGE -> eval merged model with vLLM.
                used_merge = True
                with tempfile.TemporaryDirectory(prefix="merged_model_") as tmp_merge:
                    merge_dir = Path(tmp_merge)
                    merged_path, merge_error = merge_adapter(
                        base_model_id=adapter.base_model_id,
                        adapter_dir=adapter_path,
                        output_dir=merge_dir,
                        device=args.merge_device,
                    )
                    if merge_error:
                        lm_error = merge_error
                        write_text(output_dir / "merge_error.txt", merge_error)
                        return None, None, None, lm_error, used_lora, True

                    raw, metrics, num_examples, lm_error = run_lm_eval(
                        base_model=str(merged_path),
                        task=adapter.task,
                        output_dir=output_dir,
                        tensor_parallel_size=args.tensor_parallel_size,
                        timeout_s=args.eval_timeout_s,
                    )
                    return raw, metrics, num_examples, lm_error, used_lora, True
            except Exception as exc:
                return None, None, None, f"lm_eval execution failed: {exc}", used_lora, used_merge

        for variant, adapter_path, policy, is_edited in variants:
            output_dir = (
                out_root
                / adapter.base_model_tag
                / adapter.task
                / adapter.adapter_type
                / f"profile-{adapter.profile}"
                / f"rank-{adapter.rank}"
                / f"seed{adapter.seed}"
                / Path(variant)
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            used_lora = False
            used_merge = False
            lm_error = None
            raw = None
            metrics = None
            num_examples = None
            edited_adapter_dir = None
            reused = False

            if args.resume:
                raw_existing, metrics_existing, num_existing, _, result_path = load_existing_results(
                    output_dir,
                    adapter.task,
                )
                if result_path and raw_existing is not None and metrics_existing:
                    raw = raw_existing
                    metrics = metrics_existing
                    num_examples = num_existing
                    used_lora = False
                    used_merge = (variant != "baseline")  # in FORCE MERGE mode, any non-baseline implies merge
                    edited_adapter_dir = str(adapter_path) if adapter_path else None
                    print(f"  [REUSE] {variant}: {result_path.name}")
                    reused = True

            if not reused:
                if is_edited:
                    if args.keep_edited_adapter:
                        edited_dir = (
                            edited_root
                            / adapter.base_model_tag
                            / adapter.task
                            / adapter.adapter_type
                            / f"profile-{adapter.profile}"
                            / f"rank-{adapter.rank}"
                            / f"seed{adapter.seed}"
                            / policy
                        )
                        if not (edited_dir.exists() and has_adapter_weights(edited_dir)):
                            if edited_dir.exists():
                                shutil.rmtree(edited_dir)
                            success, error = run_spectral_edit(
                                adapter_dir=adapter.adapter_dir,
                                out_dir=edited_dir,
                                edit_method=policy,
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
                                preserve_energy=args.preserve_energy,
                            )
                            if not success:
                                lm_error = error
                        adapter_path = edited_dir if lm_error is None else None
                        edited_adapter_dir = str(edited_dir) if args.keep_edited_adapter else None
                        if lm_error is None:
                            raw, metrics, num_examples, lm_error, used_lora, used_merge = evaluate_variant_with_merge(
                                adapter_path
                            )
                    else:
                        with tempfile.TemporaryDirectory(prefix="edited_adapter_") as tmpdir:
                            edited_dir = Path(tmpdir)
                            success, error = run_spectral_edit(
                                adapter_dir=adapter.adapter_dir,
                                out_dir=edited_dir,
                                edit_method=policy,
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
                                preserve_energy=args.preserve_energy,
                            )
                            if not success:
                                lm_error = error
                            else:
                                raw, metrics, num_examples, lm_error, used_lora, used_merge = evaluate_variant_with_merge(
                                    edited_dir
                                )
                else:
                    raw, metrics, num_examples, lm_error, used_lora, used_merge = evaluate_variant_with_merge(
                        adapter_path
                    )

            metric_key, metric_value = (None, None)
            if metrics:
                metric_key, metric_value = select_metric(adapter.task, metrics)

            if lm_error:
                ensure_error_logs(output_dir, lm_error)

            record = EvalRecord(
                timestamp=datetime.now().isoformat(),
                base_model_tag=adapter.base_model_tag,
                base_model_id=adapter.base_model_id,
                task=adapter.task,
                lm_eval_task=TASK_DIR_TO_LM_EVAL[adapter.task],
                adapter_type=adapter.adapter_type,
                profile=adapter.profile,
                rank=adapter.rank,
                seed=adapter.seed,
                variant=variant,
                adapter_dir=str(adapter.adapter_dir) if adapter.adapter_dir else None,
                edited_adapter_dir=edited_adapter_dir,
                output_dir=str(output_dir),
                used_vllm_lora=False,               # forced off
                used_fallback_merge=bool(used_merge),  # repurposed: True means we merged
                metric_key=metric_key,
                metric_value=metric_value,
                metrics=metrics,
                num_examples=num_examples,
                error=lm_error,
            )
            summary_records.append(record)

            if lm_error:
                print(f"  [EVAL FAILED] {variant}: {lm_error}")
            else:
                metric_display = f"{metric_key}={metric_value}" if metric_key else "metric=unknown"
                print(f"  [EVAL OK] {variant}: {metric_display}")

    print("\n[4/4] Writing summary outputs...")
    summary_json = out_root / "summary.json"
    summary_csv = out_root / "summary.csv"
    summary_json.write_text(json.dumps([asdict(r) for r in summary_records], indent=2))

    if summary_records:
        fieldnames = list(asdict(summary_records[0]).keys())
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in summary_records:
                row = asdict(record)
                if row.get("metrics") is not None:
                    row["metrics"] = json.dumps(row["metrics"])
                writer.writerow(row)

    print("  Done.")
    print(f"  Summary JSON: {summary_json}")
    print(f"  Summary CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
