#!/usr/bin/env python3
"""
Run top-k singular value edits on LoRA adapters and evaluate with lm_eval (vLLM).

This script:
  - Discovers final adapters under one or more runs roots (skipping checkpoints).
  - Applies magnitude-only top-k keep edits on singular values (no gradients).
  - Evaluates baseline (no adapter), unedited adapter, and edited adapters
    with lm_eval harness using the vLLM backend.

Edits:
  - topk_keep_20: keep top 20% singular values per module (L1-preserve)
  - topk_keep_80: keep top 80% singular values per module (L1-preserve)

Outputs are stored under:
  {out_root}/{base_model_tag}/{task}/{adapter_type}/{profile}/{rank}/{seed}/{variant}/

Edited adapters are stored under (when --save_edited_adapter):
  {out_root}/edited_adapters/{base_model_tag}/{task}/{adapter_type}/{profile}/{rank}/{seed}/{policy}/
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
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

try:
    import torch
except Exception as exc:
    raise RuntimeError(f"Failed to import torch: {exc}")

from finetune.spectral_edit.io import load_lora_state_dict, save_lora_state_dict, parse_lora_ab_key
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TOPK_RATIOS = [0.2, 0.8]

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


@dataclass
class ModuleEditInfo:
    module_prefix: str
    adapter_name: Optional[str]
    a_key: str
    b_key: str
    r: int
    scaling: float
    U: torch.Tensor
    Vh: torch.Tensor
    sigma: torch.Tensor
    dtype_a: torch.dtype
    dtype_b: torch.dtype


@dataclass
class AdapterEditState:
    adapter_dir: Path
    adapter_cfg: Dict[str, Any]
    state_dict: Dict[str, torch.Tensor]
    fmt: str
    module_infos: List[ModuleEditInfo]


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
    if cfg:
        r = cfg.get("r") or cfg.get("rank")
        rank_pattern = cfg.get("rank_pattern")
        if isinstance(rank_pattern, dict) and rank_pattern:
            try:
                r = max([int(r or 0)] + [int(v) for v in rank_pattern.values()])
            except Exception:
                pass
        if r:
            try:
                return int(r)
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
# Top-k singular value editing
# ============================================================================


def ratio_to_policy(ratio: float) -> str:
    pct = ratio * 100.0
    if abs(pct - round(pct)) < 1e-6:
        return f"topk_keep_{int(round(pct))}"
    tag = f"{pct:.3f}".rstrip("0").rstrip(".")
    return f"topk_keep_{tag.replace('.', 'p')}"


def validate_topk_ratios(ratios: List[float]) -> None:
    for ratio in ratios:
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"Invalid top-k ratio {ratio}; must be in (0, 1].")


def copy_adapter_assets(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.name in {"adapter_model.safetensors", "adapter_model.bin"}:
            continue
        dest = dst_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        elif item.is_file():
            shutil.copy2(item, dest)


def compute_scaling(adapter_cfg: Dict[str, Any], module_prefix: str, r_actual: int) -> float:
    r_cfg = adapter_cfg.get("r") or adapter_cfg.get("rank")
    alpha_cfg = adapter_cfg.get("lora_alpha") or adapter_cfg.get("alpha")
    rank_pattern = adapter_cfg.get("rank_pattern")
    alpha_pattern = adapter_cfg.get("alpha_pattern")

    suffix = module_prefix.split(".")[-1]
    if isinstance(rank_pattern, dict) and suffix in rank_pattern:
        r_cfg = rank_pattern[suffix]
    if isinstance(alpha_pattern, dict) and suffix in alpha_pattern:
        alpha_cfg = alpha_pattern[suffix]

    if r_cfg is None:
        r_cfg = r_actual
    if alpha_cfg is None:
        raise ValueError(f"Missing lora_alpha for module {module_prefix}")

    try:
        r_cfg = int(r_cfg)
    except Exception as exc:
        raise ValueError(f"Invalid rank r={r_cfg} for module {module_prefix}: {exc}")

    if r_actual != r_cfg:
        print(
            f"  [WARN] Rank mismatch for {module_prefix}: config r={r_cfg} vs weight r={r_actual}; using weight r"
        )
        r_cfg = r_actual

    if r_cfg <= 0:
        raise ValueError(f"Invalid rank r={r_cfg} for module {module_prefix}")

    alpha = float(alpha_cfg)
    use_rslora = bool(adapter_cfg.get("use_rslora", False))
    if use_rslora:
        return alpha / math.sqrt(float(r_cfg))
    return alpha / float(r_cfg)


def compute_module_svd(
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float,
    module_prefix: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[str]]:
    with torch.no_grad():
        A_f = A.detach().cpu().float()
        B_f = B.detach().cpu().float()
        try:
            U, S_unscaled, Vh, _ = lowrank_svd_from_ba(B_f, A_f)
            sigma = S_unscaled * float(scaling)
            return U, sigma, Vh, None
        except Exception as exc:
            warn = (
                f"[WARN] lowrank_svd failed for {module_prefix}: {exc}; "
                "falling back to torch.linalg.svd on CPU float32"
            )
            delta_w = (B_f @ A_f) * float(scaling)
            try:
                U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
            except Exception as exc2:
                delta_w = delta_w.detach().cpu().float()
                try:
                    U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
                except Exception as exc3:
                    raise RuntimeError(f"SVD failed for {module_prefix}: {exc3}") from exc3
            return U, S, Vh, warn


def prepare_topk_edits(
    adapter_dir: Path,
    target_modules: List[str],
) -> Tuple[Optional[AdapterEditState], Optional[str]]:
    cfg = read_json(adapter_dir / "adapter_config.json")
    if not cfg:
        return None, f"Missing or invalid adapter_config.json in {adapter_dir}"

    try:
        state_dict, fmt = load_lora_state_dict(str(adapter_dir))
    except Exception as exc:
        return None, f"Failed to load adapter weights: {exc}"

    groups: Dict[Tuple[str, Optional[str]], Dict[str, str]] = {}
    for key in state_dict.keys():
        parsed = parse_lora_ab_key(key)
        if not parsed:
            continue
        module_prefix, which, adapter_name = parsed
        module_name = module_prefix.split(".")[-1]
        if target_modules and module_name not in target_modules:
            continue
        group_key = (module_prefix, adapter_name)
        groups.setdefault(group_key, {})[which] = key

    module_infos: List[ModuleEditInfo] = []
    if not groups:
        print("  [WARN] No target modules matched in adapter weights.")

    for (module_prefix, adapter_name), keys in groups.items():
        if "A" not in keys or "B" not in keys:
            print(f"  [WARN] Missing LoRA A/B for module {module_prefix}")
            continue

        a_key = keys["A"]
        b_key = keys["B"]
        A = state_dict[a_key]
        B = state_dict[b_key]

        if A.ndim != 2 or B.ndim != 2:
            print(f"  [WARN] Expected 2D LoRA weights for {module_prefix}; skipping")
            continue
        if A.shape[0] != B.shape[1]:
            print(
                f"  [WARN] Rank mismatch for {module_prefix}: A{tuple(A.shape)} vs B{tuple(B.shape)}; skipping"
            )
            continue

        r = int(A.shape[0])
        try:
            scaling = compute_scaling(cfg, module_prefix, r)
        except Exception as exc:
            return None, f"Failed to compute scaling for {module_prefix}: {exc}"

        if scaling <= 0.0:
            return None, f"Invalid scaling ({scaling}) for {module_prefix}"

        try:
            U, sigma, Vh, warn = compute_module_svd(A, B, scaling, module_prefix)
        except Exception as exc:
            return None, str(exc)

        if warn:
            print(f"  {warn}")

        module_infos.append(
            ModuleEditInfo(
                module_prefix=module_prefix,
                adapter_name=adapter_name,
                a_key=a_key,
                b_key=b_key,
                r=r,
                scaling=float(scaling),
                U=U,
                Vh=Vh,
                sigma=sigma,
                dtype_a=A.dtype,
                dtype_b=B.dtype,
            )
        )

    return AdapterEditState(
        adapter_dir=adapter_dir,
        adapter_cfg=cfg,
        state_dict=state_dict,
        fmt=fmt,
        module_infos=module_infos,
    ), None


def apply_topk_edit(
    edit_state: AdapterEditState,
    ratio: float,
    preserve_energy: str,
    out_dir: Path,
) -> Tuple[bool, Optional[str]]:
    if preserve_energy.lower() != "l1":
        return False, f"Unsupported preserve_energy='{preserve_energy}' (only 'l1' supported)."

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_adapter_assets(edit_state.adapter_dir, out_dir)

    new_state = dict(edit_state.state_dict)

    if not edit_state.module_infos:
        print("  [WARN] No target modules edited; saving adapter unchanged.")
        save_lora_state_dict(str(out_dir), new_state, edit_state.fmt)
        return True, None

    for info in edit_state.module_infos:
        sigma = info.sigma
        r = int(info.r)
        k = max(1, int(round(ratio * r)))
        k = min(k, r)

        sigma_keep = torch.zeros_like(sigma)
        sigma_keep[:k] = sigma[:k]

        sum_orig = float(sigma.sum().item())
        sum_keep = float(sigma_keep.sum().item())
        if sum_keep > 0.0 and sum_orig > 0.0:
            rescale = sum_orig / sum_keep
            sigma_new = sigma_keep * rescale
        else:
            rescale = 0.0
            sigma_new = sigma_keep

        sigma_new = sigma_new.clamp_min(0.0)
        sum_new = float(sigma_new.sum().item())
        delta = sum_new - sum_orig
        rel_err = abs(delta) / (abs(sum_orig) + 1e-8)

        module_label = info.module_prefix
        if info.adapter_name:
            module_label = f"{module_label} (adapter={info.adapter_name})"

        print(
            f"    [EDIT] module={module_label} ratio={ratio:.3f} k={k}/{r} "
            f"sum_sigma={sum_orig:.6f} sum_new={sum_new:.6f} "
            f"delta={delta:.3e} rel_err={rel_err:.3e}"
        )

        if sum_keep == 0.0 and sum_orig > 0.0:
            print(f"    [WARN] sum_keep=0 for {module_label}; L1 preservation skipped")

        sigma_unscaled = sigma_new / float(info.scaling)
        sigma_unscaled = sigma_unscaled.clamp_min(0.0)

        with torch.no_grad():
            B_new, A_new = rebuild_ba_from_uv_sigma(info.U, info.Vh, sigma_unscaled)

        new_state[info.b_key] = B_new.to(dtype=info.dtype_b)
        new_state[info.a_key] = A_new.to(dtype=info.dtype_a)

    save_lora_state_dict(str(out_dir), new_state, edit_state.fmt)
    return True, None


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

        time.sleep(1.0)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass

    gc.collect()

    try:
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
    lora_path: Optional[Path] = None,
    max_lora_rank: Optional[int] = None,
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
    if lora_path is not None:
        model_args += f",lora_local_path={lora_path}"
        if max_lora_rank is not None:
            model_args += f",max_lora_rank={max_lora_rank}"

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
    lora_path: Optional[Path] = None,
    max_lora_rank: Optional[int] = None,
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
                lora_path=lora_path,
                max_lora_rank=max_lora_rank,
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
                        start_new_session=True,
                    )
                except Exception as exc:
                    return None, None, None, f"lm_eval failed to start: {exc}"

                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    return None, None, None, f"lm_eval timed out after {timeout_s} seconds"
        finally:
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
        description="Run top-k singular value edits and evaluate with lm_eval (vLLM).",
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
        "--topk_ratios",
        type=float,
        nargs="+",
        default=DEFAULT_TOPK_RATIOS,
        help="Top-k ratios to keep (e.g., 0.2 0.8).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (unused; kept for compatibility).",
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
        "--adapter_type",
        type=str,
        choices=["lora", "loraplus"],
        default=None,
        help="Alias for --adapter_types (single value).",
    )
    p.add_argument(
        "--edited_out_dir",
        type=Path,
        default=None,
        help="Root directory for edited adapters when --save_edited_adapter is set.",
    )
    p.add_argument(
        "--save_edited_adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep edited adapters on disk (default: True).",
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

    p.add_argument(
        "--preserve_energy",
        type=str,
        default="l1",
        help="Preserve energy mode (fixed to l1).",
    )
    p.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["o_proj", "down_proj"],
        help="Target modules for top-k edits.",
    )
    p.add_argument(
        "--use_vllm_lora",
        action="store_true",
        default=False,
        help="Attempt vLLM LoRA adapter loading instead of force-merge.",
    )
    p.add_argument(
        "--fallback_merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to merge if vLLM LoRA eval fails.",
    )

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.reuse_results:
        args.resume = True
    if args.adapter_type:
        if args.adapter_types and args.adapter_type not in args.adapter_types:
            args.adapter_types.append(args.adapter_type)
        elif not args.adapter_types:
            args.adapter_types = [args.adapter_type]

    try:
        validate_topk_ratios(args.topk_ratios)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if args.preserve_energy.lower() != "l1":
        print("[ERROR] --preserve_energy must be 'l1' for this script.")
        sys.exit(1)

    runs_roots = [r.resolve() for r in args.runs_roots]
    for root in runs_roots:
        if not root.exists():
            print(f"[ERROR] Runs root does not exist: {root}")
            sys.exit(1)

    tasks = args.tasks
    out_root = args.out_root.resolve()
    edited_root = None
    if args.save_edited_adapter:
        edited_root = args.edited_out_dir.resolve() if args.edited_out_dir else (out_root / "edited_adapters")
        edited_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    policies = [ratio_to_policy(r) for r in args.topk_ratios]

    print("=" * 70)
    print("Top-k Sigma Edit + lm_eval Harness Driver")
    print("=" * 70)
    print(f"Runs roots: {', '.join(str(r) for r in runs_roots)}")
    print(f"Output root: {out_root}")
    print(f"Tasks: {tasks}")
    print(f"Top-k ratios: {args.topk_ratios}")
    print(f"Target modules: {args.target_modules}")
    print(f"Preserve energy: {args.preserve_energy}")
    if args.use_vllm_lora:
        print("Adapter eval mode: vLLM LoRA (with fallback merge)")
    else:
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
            for ratio, policy in zip(args.topk_ratios, policies):
                print(f"    Edit policy: {policy} (ratio={ratio})")
        if len(adapters) > 5:
            print(f"  ... and {len(adapters) - 5} more adapters")
        print("\n[DRY-RUN] No changes made.")
        return

    summary_records: List[EvalRecord] = []

    if args.save_edited_adapter:
        print("\n[2/4] Editing adapters...")
        for i, adapter in enumerate(adapters, 1):
            print(f"\n[{i}/{len(adapters)}] {adapter.run_id}")
            print(f"  Adapter: {adapter.adapter_dir}")

            edit_state, error = prepare_topk_edits(adapter.adapter_dir, args.target_modules)
            if error:
                print(f"  [EDIT FAILED] {error}")
                continue

            for ratio, policy in zip(args.topk_ratios, policies):
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
                success, error = apply_topk_edit(
                    edit_state=edit_state,
                    ratio=ratio,
                    preserve_energy=args.preserve_energy,
                    out_dir=edited_dir,
                )
                if not success:
                    print(f"  [EDIT FAILED] {policy}: {error}")
                gc.collect()
    else:
        print("\n[2/4] Skipping persistent edits (temporary edited adapters will be used).")

    print("\n[3/4] Evaluating with lm_eval...")
    for i, adapter in enumerate(adapters, 1):
        print(f"\n[{i}/{len(adapters)}] {adapter.run_id}")

        variants: List[Tuple[str, Optional[Path], Optional[float], Optional[str], bool]] = [
            ("baseline", None, None, None, False),
            ("unedited", adapter.adapter_dir, None, None, False),
        ]
        for ratio, policy in zip(args.topk_ratios, policies):
            variants.append((f"edited/{policy}", None, ratio, policy, True))

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
                    raw, metrics, num_examples, lm_error = run_lm_eval(
                        base_model=adapter.base_model_id,
                        task=adapter.task,
                        output_dir=output_dir,
                        tensor_parallel_size=args.tensor_parallel_size,
                        timeout_s=args.eval_timeout_s,
                    )
                    return raw, metrics, num_examples, lm_error, used_lora, False

                if args.use_vllm_lora:
                    max_lora_rank = read_lora_rank(adapter_path, adapter.rank)
                    raw, metrics, num_examples, lm_error = run_lm_eval(
                        base_model=adapter.base_model_id,
                        task=adapter.task,
                        output_dir=output_dir,
                        tensor_parallel_size=args.tensor_parallel_size,
                        timeout_s=args.eval_timeout_s,
                        lora_path=adapter_path,
                        max_lora_rank=max_lora_rank,
                    )
                    if lm_error is None:
                        used_lora = True
                        return raw, metrics, num_examples, lm_error, used_lora, False
                    if not args.fallback_merge:
                        return None, None, None, lm_error, True, False

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

        edit_state_cache: Optional[AdapterEditState] = None

        for variant, adapter_path, ratio, policy, is_edited in variants:
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
                    used_lora = args.use_vllm_lora and variant != "baseline"
                    used_merge = variant != "baseline" and not args.use_vllm_lora
                    edited_adapter_dir = str(adapter_path) if adapter_path else None
                    print(f"  [REUSE] {variant}: {result_path.name}")
                    reused = True

            if not reused:
                if is_edited:
                    if args.save_edited_adapter:
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
                            if edit_state_cache is None:
                                edit_state_cache, error = prepare_topk_edits(
                                    adapter.adapter_dir,
                                    args.target_modules,
                                )
                                if error:
                                    lm_error = error
                            if lm_error is None and edit_state_cache is not None:
                                success, error = apply_topk_edit(
                                    edit_state=edit_state_cache,
                                    ratio=float(ratio),
                                    preserve_energy=args.preserve_energy,
                                    out_dir=edited_dir,
                                )
                                if not success:
                                    lm_error = error
                        adapter_path = edited_dir if lm_error is None else None
                        edited_adapter_dir = str(edited_dir) if args.save_edited_adapter else None
                        if lm_error is None:
                            raw, metrics, num_examples, lm_error, used_lora, used_merge = evaluate_variant_with_merge(
                                adapter_path
                            )
                    else:
                        if edit_state_cache is None:
                            edit_state_cache, error = prepare_topk_edits(
                                adapter.adapter_dir,
                                args.target_modules,
                            )
                            if error:
                                lm_error = error
                        if lm_error is None and edit_state_cache is not None:
                            with tempfile.TemporaryDirectory(prefix="edited_adapter_") as tmpdir:
                                edited_dir = Path(tmpdir)
                                success, error = apply_topk_edit(
                                    edit_state=edit_state_cache,
                                    ratio=float(ratio),
                                    preserve_energy=args.preserve_energy,
                                    out_dir=edited_dir,
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
                used_vllm_lora=bool(used_lora),
                used_fallback_merge=bool(used_merge),
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
