#!/usr/bin/env python3
"""Analyze LoRA singular components with MP-style diagnostics and ablations."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except Exception:
    HAVE_MATPLOTLIB = False


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.csqa_prompt import resolve_csqa_prompt_style
from finetune.spectral_edit.direct_search import (
    _active_adapter_names,
    _assert_adapter_matches_base_model,
    _build_specs,
    _load_csqa_subset,
    _load_math_subset,
    _set_module_lora_weights,
    apply_state_dict_to_model,
    evaluate_csqa_subset,
    evaluate_math_subset,
)
from finetune.spectral_edit.io import (
    ensure_local_lora_dir,
    layer_idx_from_module_prefix,
    load_adapter_config,
    load_lora_state_dict,
)
from finetune.spectral_edit.rmt import estimate_mp_summary
from finetune.spectral_edit.svd import rebuild_ba_from_uv_sigma
from finetune.utils import seed_everything


DEFAULT_MAX_NEW_TOKENS = {
    "csqa": 8,
    "math": 256,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze LoRA singular components with a conservative MP-style threshold "
            "and leave-one-out / sequential singular ablation."
        )
    )
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="csqa", choices=["csqa", "math"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--subset_start", type=int, default=0)
    parser.add_argument(
        "--subset_size",
        type=int,
        default=256,
        help="Use <=0 to evaluate the full split.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="auto",
        choices=["auto", "task_native", "alpaca_legacy"],
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["down_proj", "o_proj"],
        help="Module suffixes to consider before ranking/selecting explicit module prefixes.",
    )
    parser.add_argument("--layer_min", type=int, default=0)
    parser.add_argument("--layer_max", type=int, default=10**9)
    parser.add_argument(
        "--module_prefixes",
        type=str,
        nargs="*",
        default=None,
        help="Explicit full module prefixes. If omitted, top modules are selected automatically.",
    )
    parser.add_argument("--max_modules", type=int, default=1)
    parser.add_argument(
        "--module_selection_metric",
        type=str,
        default="fro_norm",
        choices=["fro_norm", "spectral_norm", "l1_sum", "top1_ratio"],
    )
    parser.add_argument(
        "--rmt_tail_count",
        type=int,
        default=0,
        help="Number of smallest singular values treated as the bulk candidate set. 0 means rank//2.",
    )
    parser.add_argument(
        "--rmt_edge_margin",
        type=float,
        default=0.10,
        help="Relative band around the conservative edge used for near-edge classification.",
    )
    parser.add_argument(
        "--sequential_order",
        type=str,
        default="rmt_noise_first",
        choices=["rmt_noise_first", "ascending_sigma", "descending_sigma"],
    )
    parser.add_argument("--skip_sequential", action="store_true")
    parser.add_argument(
        "--signal_drop_threshold_frac",
        type=float,
        default=0.005,
        help="Ablation hurt threshold as a fraction of the evaluated set size.",
    )
    parser.add_argument("--signal_drop_min_examples", type=int, default=2)
    parser.add_argument("--save_plots", action="store_true")
    return parser.parse_args()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _to_builtin(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_builtin(data), handle, indent=2, sort_keys=False)


def save_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _to_builtin(row.get(key)) for key in fieldnames})


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def short_module_name(prefix: str) -> str:
    return re.sub(r"^.*?layers\.(\d+)\.", r"L\1.", prefix)


def resolve_torch_dtype(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return None


def resolve_subset_size(subset_size: int) -> Optional[int]:
    return None if subset_size <= 0 else int(subset_size)


def classify_loo(delta_correct: int, total: int, signal_drop_examples: int) -> str:
    if delta_correct > 0:
        return "likely_harmful_or_noise"
    if delta_correct == 0:
        return "likely_noise"
    if delta_correct <= -signal_drop_examples:
        return "likely_signal"
    return "uncertain_small_hurt"


def agreement_label(rmt_label: str, loo_label: str) -> str:
    if rmt_label == "likely_signal" and loo_label == "likely_signal":
        return "agree_signal"
    if rmt_label == "likely_bulk_noise" and loo_label in {"likely_noise", "likely_harmful_or_noise"}:
        return "agree_noise"
    if rmt_label == "near_edge" or loo_label == "uncertain_small_hurt":
        return "uncertain"
    return "disagree"


def build_module_ranking(specs: Dict[str, Any]) -> List[dict[str, Any]]:
    ranking: List[dict[str, Any]] = []
    for prefix, spec in specs.items():
        sigma = spec.sigma0.detach().float().cpu()
        l1_sum = float(sigma.sum().item())
        ranking.append(
            {
                "module_prefix": prefix,
                "module_short": short_module_name(prefix),
                "layer_idx": layer_idx_from_module_prefix(prefix),
                "module_suffix": prefix.split(".")[-1],
                "rank": int(sigma.numel()),
                "out_dim": int(spec.U.shape[0]),
                "in_dim": int(spec.Vh.shape[1]),
                "fro_norm": float(torch.linalg.vector_norm(sigma).item()),
                "spectral_norm": float(sigma[0].item()),
                "l1_sum": l1_sum,
                "top1_ratio": float(sigma[0].item() / max(l1_sum, 1e-12)),
                "singular_values": [float(v) for v in sigma.tolist()],
            }
        )
    return ranking


def select_module_prefixes(
    ranking: Sequence[dict[str, Any]],
    explicit_prefixes: Optional[Sequence[str]],
    metric: str,
    max_modules: int,
) -> List[str]:
    available = {row["module_prefix"] for row in ranking}
    if explicit_prefixes:
        missing = [prefix for prefix in explicit_prefixes if prefix not in available]
        if missing:
            raise ValueError(f"Requested module_prefixes not found: {missing}")
        return list(explicit_prefixes)

    sorted_rows = sorted(
        ranking,
        key=lambda row: (float(row[metric]), row["module_prefix"]),
        reverse=True,
    )
    if not sorted_rows:
        raise RuntimeError("No editable modules found for the requested target_modules / layer range.")
    return [row["module_prefix"] for row in sorted_rows[: max(1, int(max_modules))]]


def build_sequential_order(
    singular_values: Sequence[float],
    rmt_labels: Sequence[str],
    mode: str,
) -> List[int]:
    sigma = list(float(v) for v in singular_values)
    if mode == "ascending_sigma":
        return sorted(range(len(sigma)), key=lambda idx: (sigma[idx], idx))
    if mode == "descending_sigma":
        return sorted(range(len(sigma)), key=lambda idx: (-sigma[idx], idx))
    if mode == "rmt_noise_first":
        priority = {"likely_bulk_noise": 0, "near_edge": 1, "likely_signal": 2}
        return sorted(
            range(len(sigma)),
            key=lambda idx: (priority.get(rmt_labels[idx], 1), sigma[idx], idx),
        )
    raise ValueError(f"Unknown sequential order: {mode}")


def maybe_plot_spectrum(
    out_path: Path,
    module_label: str,
    singular_values: Sequence[float],
    theoretical_sigma_plus: float,
    conservative_sigma_plus: float,
    rmt_labels: Sequence[str],
) -> None:
    if not HAVE_MATPLOTLIB:
        return

    colors = {
        "likely_signal": "#1b9e77",
        "near_edge": "#d95f02",
        "likely_bulk_noise": "#7570b3",
    }
    xs = list(range(len(singular_values)))
    ys = [float(v) for v in singular_values]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, color="#333333", linewidth=1.5, marker="o", markersize=4)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        ax.scatter([x], [y], color=colors.get(rmt_labels[idx], "#333333"), s=30, zorder=3)
    ax.axhline(theoretical_sigma_plus, color="#666666", linestyle="--", linewidth=1.2, label="MP edge")
    ax.axhline(conservative_sigma_plus, color="#111111", linestyle="-.", linewidth=1.2, label="Conservative edge")
    ax.set_xlabel("Component k")
    ax.set_ylabel("Singular value")
    ax.set_title(f"{module_label}: singular spectrum")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def maybe_plot_loo(
    out_path: Path,
    module_label: str,
    loo_rows: Sequence[dict[str, Any]],
) -> None:
    if not HAVE_MATPLOTLIB or not loo_rows:
        return

    colors = {
        "likely_signal": "#1b9e77",
        "uncertain_small_hurt": "#e6ab02",
        "likely_noise": "#7570b3",
        "likely_harmful_or_noise": "#d95f02",
    }
    xs = [int(row["component_index"]) for row in loo_rows]
    ys = [100.0 * float(row["delta_accuracy"]) for row in loo_rows]
    bar_colors = [colors.get(str(row["loo_label"]), "#666666") for row in loo_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xs, ys, color=bar_colors)
    ax.axhline(0.0, color="#111111", linewidth=1.0)
    ax.set_xlabel("Component k")
    ax.set_ylabel("Accuracy delta (percentage points)")
    ax.set_title(f"{module_label}: leave-one-out ablation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def maybe_plot_sequential(
    out_path: Path,
    module_label: str,
    baseline_accuracy: float,
    sequential_rows: Sequence[dict[str, Any]],
) -> None:
    if not HAVE_MATPLOTLIB or not sequential_rows:
        return

    xs = [int(row["step_index"]) for row in sequential_rows]
    ys = [100.0 * float(row["ablated_accuracy"]) for row in sequential_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, color="#1f78b4", marker="o")
    ax.axhline(100.0 * baseline_accuracy, color="#111111", linestyle="--", linewidth=1.0, label="Baseline")
    ax.set_xlabel("Sequential removals")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{module_label}: sequential ablation")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def load_subset(
    args: argparse.Namespace,
    adapter_dir: str,
) -> tuple[dict[str, Any], Callable[..., dict[str, Any]], dict[str, Any]]:
    subset_size = resolve_subset_size(args.subset_size)
    if args.task == "csqa":
        prompt_resolution = resolve_csqa_prompt_style(args.prompt_style, adapter_dir)
        subset = _load_csqa_subset(
            split=args.split,
            start=args.subset_start,
            max_samples=subset_size,
            prompt_style=prompt_resolution.resolved,
        )
        return (
            subset,
            evaluate_csqa_subset,
            {
                "prompt_style_requested": args.prompt_style,
                "prompt_style_resolved": prompt_resolution.resolved,
                "prompt_style_reason": prompt_resolution.reason,
            },
        )

    subset = _load_math_subset(
        split=args.split,
        start=args.subset_start,
        max_samples=subset_size,
    )
    return (
        subset,
        evaluate_math_subset,
        {
            "prompt_style_requested": "metamath_model_usage",
            "prompt_style_resolved": "metamath_model_usage",
            "prompt_style_reason": "fixed math prompt template",
        },
    )


def run_eval(
    *,
    label: str,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    golds: Sequence[str],
    max_new_tokens: int,
    eval_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    metrics = eval_fn(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        golds=golds,
        max_new_tokens=max_new_tokens,
        return_predictions=False,
    )
    print(
        "[Eval] "
        f"{label}: accuracy={metrics['accuracy']:.4f} "
        f"({metrics['correct']}/{metrics['total']})"
    , flush=True)
    return metrics


def build_summary_answers(
    baseline_metrics: dict[str, Any],
    rmt_rows: Sequence[dict[str, Any]],
    loo_rows: Sequence[dict[str, Any]],
    selected_prefixes: Sequence[str],
) -> dict[str, Any]:
    agreement_counter = Counter(row["agreement_label"] for row in loo_rows)
    removable_bulk = [
        row
        for row in loo_rows
        if row["rmt_label"] == "likely_bulk_noise" and row["delta_correct"] >= 0
    ]
    improved = [row for row in loo_rows if row["delta_correct"] > 0]
    hurtful_signal = [row for row in loo_rows if row["loo_label"] == "likely_signal"]
    total_rows = max(1, len(loo_rows))
    agree_count = agreement_counter.get("agree_signal", 0) + agreement_counter.get("agree_noise", 0)
    agreement_fraction = agree_count / total_rows

    if agreement_fraction >= 0.60:
        q1 = (
            "Yes. The MP-style threshold and leave-one-out ablation agree on a majority "
            f"of analyzed components ({agree_count}/{len(loo_rows)})."
        )
    elif agree_count > 0:
        q1 = (
            "Partially. Some components are highlighted by both diagnostics, but the agreement "
            f"is mixed ({agree_count}/{len(loo_rows)})."
        )
    else:
        q1 = "No clear agreement in this first controlled run."

    if removable_bulk:
        q2 = (
            "Yes. At least one component classified as theoretical bulk/noise can be removed "
            "with no loss on the evaluated validation subset."
        )
    else:
        q2 = "Not yet. This run did not find a bulk/noise component removable at zero measured cost."

    if improved:
        q3 = (
            "Yes. Some removals improve validation accuracy, which is consistent with at least "
            "part of the learned spectrum being harmful or redundant."
        )
    else:
        q3 = "No improvement from removal was observed in this controlled run."

    if removable_bulk or improved:
        q4 = (
            "Cautiously yes. The result is consistent with a noisy or suboptimal tail in the learned "
            "spectrum, but this should be verified on more modules or the full validation split."
        )
    elif hurtful_signal:
        q4 = (
            "Partially. The run isolates useful components, but it is weaker evidence for a noisy tail "
            "because removal rarely helped."
        )
    else:
        q4 = "Inconclusive. The current run is too weak to support the noisy-spectrum claim."

    return {
        "baseline_accuracy": float(baseline_metrics["accuracy"]),
        "baseline_correct": int(baseline_metrics["correct"]),
        "baseline_total": int(baseline_metrics["total"]),
        "selected_modules": list(selected_prefixes),
        "agreement_counts": dict(agreement_counter),
        "agreement_fraction": agreement_fraction,
        "removable_bulk_count": len(removable_bulk),
        "improved_removal_count": len(improved),
        "signal_component_count": len(hurtful_signal),
        "answers": {
            "q1_do_rmt_and_loo_highlight_similar_components": q1,
            "q2_bulk_noise_components_removable": q2,
            "q3_any_removals_improve_performance": q3,
            "q4_support_for_noisy_or_suboptimal_spectrum": q4,
        },
    }


def write_summary_markdown(
    out_path: Path,
    *,
    args: argparse.Namespace,
    baseline_metrics: dict[str, Any],
    ranking_rows: Sequence[dict[str, Any]],
    rmt_rows: Sequence[dict[str, Any]],
    loo_rows: Sequence[dict[str, Any]],
    sequential_rows: Sequence[dict[str, Any]],
    summary_answers: dict[str, Any],
    signal_drop_examples: int,
) -> None:
    lines: List[str] = []
    lines.append("# LoRA signal/noise analysis")
    lines.append("")
    lines.append("## Setup")
    lines.append(
        f"- Task: `{args.task}`"
    )
    lines.append(f"- Base model: `{args.base_model}`")
    lines.append(f"- Adapter: `{args.adapter_dir}`")
    lines.append(
        f"- Evaluated subset: start={args.subset_start}, size={baseline_metrics['total']} on split `{args.split}`"
    )
    lines.append(
        f"- Module selection metric: `{args.module_selection_metric}`; target suffixes={list(args.target_modules)}"
    )
    lines.append(
        f"- Signal-hurt threshold: at least {signal_drop_examples} fewer correct predictions than baseline"
    )
    lines.append("")
    lines.append("## Module Ranking")
    for row in ranking_rows[: max(5, len(summary_answers["selected_modules"]))]:
        marker = " [selected]" if row["module_prefix"] in summary_answers["selected_modules"] else ""
        lines.append(
            "- "
            f"`{row['module_short']}` "
            f"(fro={row['fro_norm']:.4f}, top1={row['spectral_norm']:.4f}, top1_ratio={row['top1_ratio']:.3f})"
            f"{marker}"
        )
    lines.append("")
    lines.append("## RMT Summary")
    for prefix in summary_answers["selected_modules"]:
        rows = [row for row in rmt_rows if row["module_prefix"] == prefix]
        if not rows:
            continue
        module_edge = rows[0]["conservative_sigma_plus"]
        module_theoretical = rows[0]["theoretical_sigma_plus"]
        signal_components = [str(row["component_index"]) for row in rows if row["rmt_label"] == "likely_signal"]
        noise_components = [str(row["component_index"]) for row in rows if row["rmt_label"] == "likely_bulk_noise"]
        lines.append(
            "- "
            f"`{rows[0]['module_short']}`: MP edge={module_theoretical:.4f}, "
            f"conservative edge={module_edge:.4f}, "
            f"signal={signal_components or ['none']}, "
            f"bulk/noise={noise_components or ['none']}"
        )
    lines.append("")
    lines.append("## Leave-One-Out Summary")
    lines.append(
        f"- Baseline accuracy: {baseline_metrics['accuracy']:.4f} "
        f"({baseline_metrics['correct']}/{baseline_metrics['total']})"
    )
    best_hurts = sorted(loo_rows, key=lambda row: (row["delta_correct"], row["delta_accuracy"]))[:5]
    best_improves = sorted(loo_rows, key=lambda row: (row["delta_correct"], row["delta_accuracy"]), reverse=True)[:5]
    lines.append("- Largest performance drops:")
    for row in best_hurts:
        lines.append(
            f"  - `{row['module_short']}` k={row['component_index']}: "
            f"{row['original_accuracy']:.4f} -> {row['ablated_accuracy']:.4f} "
            f"(delta={row['delta_accuracy']:+.4f}, {row['loo_label']})"
        )
    lines.append("- Largest performance gains:")
    for row in best_improves:
        lines.append(
            f"  - `{row['module_short']}` k={row['component_index']}: "
            f"{row['original_accuracy']:.4f} -> {row['ablated_accuracy']:.4f} "
            f"(delta={row['delta_accuracy']:+.4f}, {row['loo_label']})"
        )
    if sequential_rows:
        lines.append("")
        lines.append("## Sequential Ablation")
        for prefix in summary_answers["selected_modules"]:
            rows = [row for row in sequential_rows if row["module_prefix"] == prefix]
            if not rows:
                continue
            final_row = rows[-1]
            order = [str(row["component_index"]) for row in rows]
            lines.append(
                "- "
                f"`{rows[0]['module_short']}` order={order}; "
                f"final accuracy={final_row['ablated_accuracy']:.4f} "
                f"(delta={final_row['delta_accuracy']:+.4f})"
            )
    lines.append("")
    lines.append("## Agreement Analysis")
    for question_key, text in summary_answers["answers"].items():
        lines.append(f"- {question_key.replace('_', ' ')}: {text}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This analysis requires a CUDA GPU.")

    max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS[args.task]
    output_dir = Path(args.output_dir).resolve()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = ensure_local_lora_dir(args.adapter_dir, cache_dir=args.cache_dir)
    adapter_cfg = load_adapter_config(adapter_dir)
    _assert_adapter_matches_base_model(adapter_cfg, args.base_model, adapter_dir)
    state_dict, _ = load_lora_state_dict(adapter_dir)

    subset, eval_fn, prompt_meta = load_subset(args, adapter_dir)
    prompts = subset["prompts"]
    golds = subset["golds"]
    save_json(
        output_dir / "validation_subset.json",
        {
            "task": args.task,
            "dataset": subset["dataset"],
            "split": subset["split"],
            "split_note": subset["split_note"],
            "subset_start": args.subset_start,
            "subset_size": len(subset["records"]),
            "records": subset["records"],
            **prompt_meta,
        },
    )

    torch_dtype = resolve_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Info] Loading base model {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        cache_dir=args.cache_dir,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    if not getattr(model, "peft_config", None):
        raise RuntimeError(f"Adapter load failed: no PEFT config found for {adapter_dir}")
    if not _active_adapter_names(model):
        raise RuntimeError(f"Adapter load failed: no active adapter after loading {adapter_dir}")
    model.eval()

    specs, _ = _build_specs(
        model=model,
        state_dict=state_dict,
        target_modules=args.target_modules,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
    )
    ranking_rows = build_module_ranking(specs)
    selected_prefixes = select_module_prefixes(
        ranking_rows,
        args.module_prefixes,
        args.module_selection_metric,
        args.max_modules,
    )
    save_json(
        output_dir / "run_config.json",
        {
            "args": vars(args),
            "resolved": {
                "device": device,
                "adapter_dir": adapter_dir,
                "max_new_tokens": max_new_tokens,
                "selected_prefixes": selected_prefixes,
                "prompt_meta": prompt_meta,
                "evaluated_examples": len(prompts),
            },
        },
    )

    ranking_rows_sorted = sorted(
        ranking_rows,
        key=lambda row: (float(row[args.module_selection_metric]), row["module_prefix"]),
        reverse=True,
    )
    save_json(output_dir / "module_ranking.json", ranking_rows_sorted)
    save_csv(
        output_dir / "module_ranking.csv",
        ranking_rows_sorted,
        [
            "module_prefix",
            "module_short",
            "layer_idx",
            "module_suffix",
            "rank",
            "out_dim",
            "in_dim",
            "fro_norm",
            "spectral_norm",
            "l1_sum",
            "top1_ratio",
        ],
    )

    print("[Info] Selected modules:")
    for prefix in selected_prefixes:
        print(f"  - {prefix}", flush=True)

    baseline_metrics = run_eval(
        label="baseline_original",
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        golds=golds,
        max_new_tokens=max_new_tokens,
        eval_fn=eval_fn,
    )
    save_json(output_dir / "baseline_metrics.json", baseline_metrics)

    total_examples = int(baseline_metrics["total"])
    signal_drop_examples = max(
        int(args.signal_drop_min_examples),
        int(math.ceil(float(args.signal_drop_threshold_frac) * max(1, total_examples))),
    )

    rmt_rows: List[dict[str, Any]] = []
    rmt_by_module: Dict[str, dict[str, Any]] = {}
    for prefix in selected_prefixes:
        row = next(item for item in ranking_rows if item["module_prefix"] == prefix)
        rmt_summary = estimate_mp_summary(
            singular_values=row["singular_values"],
            out_dim=int(row["out_dim"]),
            in_dim=int(row["in_dim"]),
            tail_count=args.rmt_tail_count,
            edge_margin=args.rmt_edge_margin,
        )
        rmt_by_module[prefix] = rmt_summary
        for comp in rmt_summary["components"]:
            rmt_rows.append(
                {
                    "module_prefix": prefix,
                    "module_short": row["module_short"],
                    "layer_idx": row["layer_idx"],
                    "module_suffix": row["module_suffix"],
                    "out_dim": row["out_dim"],
                    "in_dim": row["in_dim"],
                    "rank": row["rank"],
                    "component_index": comp["component_index"],
                    "singular_value": comp["singular_value"],
                    "rmt_label": comp["rmt_label"],
                    "above_theoretical_edge": comp["above_theoretical_edge"],
                    "above_conservative_edge": comp["above_conservative_edge"],
                    "aspect_ratio_beta": rmt_summary["aspect_ratio_beta"],
                    "tail_count": rmt_summary["tail_count"],
                    "tail_max_singular_value": rmt_summary["tail_max_singular_value"],
                    "theoretical_lambda_plus": rmt_summary["theoretical_lambda_plus"],
                    "theoretical_sigma_plus": rmt_summary["theoretical_sigma_plus"],
                    "conservative_sigma_plus": rmt_summary["conservative_sigma_plus"],
                }
            )
        if args.save_plots:
            maybe_plot_spectrum(
                plots_dir / f"{slugify(short_module_name(prefix))}_spectrum.png",
                short_module_name(prefix),
                row["singular_values"],
                rmt_summary["theoretical_sigma_plus"],
                rmt_summary["conservative_sigma_plus"],
                [comp["rmt_label"] for comp in rmt_summary["components"]],
            )

    save_json(output_dir / "rmt_summary.json", rmt_rows)
    save_csv(
        output_dir / "rmt_summary.csv",
        rmt_rows,
        [
            "module_prefix",
            "module_short",
            "layer_idx",
            "module_suffix",
            "out_dim",
            "in_dim",
            "rank",
            "component_index",
            "singular_value",
            "rmt_label",
            "above_theoretical_edge",
            "above_conservative_edge",
            "aspect_ratio_beta",
            "tail_count",
            "tail_max_singular_value",
            "theoretical_lambda_plus",
            "theoretical_sigma_plus",
            "conservative_sigma_plus",
        ],
    )

    loo_rows: List[dict[str, Any]] = []
    for prefix in selected_prefixes:
        spec = specs[prefix]
        module_label = short_module_name(prefix)
        sigma_base = spec.sigma0.clone()
        component_to_rmt = {
            int(comp["component_index"]): str(comp["rmt_label"])
            for comp in rmt_by_module[prefix]["components"]
        }

        for component_index in range(int(sigma_base.numel())):
            apply_state_dict_to_model(specs=specs, state_dict=state_dict)
            sigma_new = sigma_base.clone()
            sigma_new[component_index] = 0.0
            b_new, a_new = rebuild_ba_from_uv_sigma(spec.U, spec.Vh, sigma_new)
            _set_module_lora_weights(spec, a_new, b_new)
            metrics = run_eval(
                label=f"loo::{module_label}::k{component_index}",
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                golds=golds,
                max_new_tokens=max_new_tokens,
                eval_fn=eval_fn,
            )
            delta_accuracy = float(metrics["accuracy"] - baseline_metrics["accuracy"])
            delta_correct = int(metrics["correct"] - baseline_metrics["correct"])
            loo_label = classify_loo(delta_correct, total_examples, signal_drop_examples)
            row = {
                "module_prefix": prefix,
                "module_short": module_label,
                "layer_idx": layer_idx_from_module_prefix(prefix),
                "module_suffix": prefix.split(".")[-1],
                "component_index": int(component_index),
                "original_accuracy": float(baseline_metrics["accuracy"]),
                "original_correct": int(baseline_metrics["correct"]),
                "original_total": total_examples,
                "ablated_accuracy": float(metrics["accuracy"]),
                "ablated_correct": int(metrics["correct"]),
                "ablated_total": int(metrics["total"]),
                "delta_accuracy": delta_accuracy,
                "delta_correct": delta_correct,
                "loo_label": loo_label,
                "rmt_label": component_to_rmt[component_index],
                "agreement_label": agreement_label(component_to_rmt[component_index], loo_label),
            }
            loo_rows.append(row)

        if args.save_plots:
            maybe_plot_loo(
                plots_dir / f"{slugify(module_label)}_loo.png",
                module_label,
                [row for row in loo_rows if row["module_prefix"] == prefix],
            )

        save_json(output_dir / "leave_one_out.json", loo_rows)
        save_csv(
            output_dir / "leave_one_out.csv",
            loo_rows,
            [
                "module_prefix",
                "module_short",
                "layer_idx",
                "module_suffix",
                "component_index",
                "original_accuracy",
                "ablated_accuracy",
                "delta_accuracy",
                "original_correct",
                "ablated_correct",
                "delta_correct",
                "loo_label",
                "rmt_label",
                "agreement_label",
            ],
        )

    apply_state_dict_to_model(specs=specs, state_dict=state_dict)

    sequential_rows: List[dict[str, Any]] = []
    if not args.skip_sequential:
        for prefix in selected_prefixes:
            spec = specs[prefix]
            module_label = short_module_name(prefix)
            sigma_base_cpu = [float(v) for v in spec.sigma0.detach().float().cpu().tolist()]
            rmt_labels = [str(comp["rmt_label"]) for comp in rmt_by_module[prefix]["components"]]
            order = build_sequential_order(sigma_base_cpu, rmt_labels, args.sequential_order)
            sigma_current = spec.sigma0.clone()
            removed: List[int] = []
            for step_index, component_index in enumerate(order, start=1):
                sigma_current[component_index] = 0.0
                removed.append(int(component_index))
                apply_state_dict_to_model(specs=specs, state_dict=state_dict)
                b_new, a_new = rebuild_ba_from_uv_sigma(spec.U, spec.Vh, sigma_current)
                _set_module_lora_weights(spec, a_new, b_new)
                metrics = run_eval(
                    label=f"seq::{module_label}::step{step_index}",
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    golds=golds,
                    max_new_tokens=max_new_tokens,
                    eval_fn=eval_fn,
                )
                sequential_rows.append(
                    {
                        "module_prefix": prefix,
                        "module_short": module_label,
                        "layer_idx": layer_idx_from_module_prefix(prefix),
                        "module_suffix": prefix.split(".")[-1],
                        "sequential_order_mode": args.sequential_order,
                        "step_index": int(step_index),
                        "component_index": int(component_index),
                        "removed_components": list(removed),
                        "ablated_accuracy": float(metrics["accuracy"]),
                        "ablated_correct": int(metrics["correct"]),
                        "delta_accuracy": float(metrics["accuracy"] - baseline_metrics["accuracy"]),
                        "delta_correct": int(metrics["correct"] - baseline_metrics["correct"]),
                    }
                )

            if args.save_plots:
                maybe_plot_sequential(
                    plots_dir / f"{slugify(module_label)}_sequential.png",
                    module_label,
                    float(baseline_metrics["accuracy"]),
                    [row for row in sequential_rows if row["module_prefix"] == prefix],
                )

            save_json(output_dir / "sequential_ablation.json", sequential_rows)
            save_csv(
                output_dir / "sequential_ablation.csv",
                sequential_rows,
                [
                    "module_prefix",
                    "module_short",
                    "layer_idx",
                    "module_suffix",
                    "sequential_order_mode",
                    "step_index",
                    "component_index",
                    "ablated_accuracy",
                    "ablated_correct",
                    "delta_accuracy",
                    "delta_correct",
                    "removed_components",
                ],
            )

    apply_state_dict_to_model(specs=specs, state_dict=state_dict)

    summary_answers = build_summary_answers(
        baseline_metrics=baseline_metrics,
        rmt_rows=rmt_rows,
        loo_rows=loo_rows,
        selected_prefixes=selected_prefixes,
    )
    save_json(output_dir / "agreement_summary.json", summary_answers)
    write_summary_markdown(
        output_dir / "summary.md",
        args=args,
        baseline_metrics=baseline_metrics,
        ranking_rows=ranking_rows_sorted,
        rmt_rows=rmt_rows,
        loo_rows=loo_rows,
        sequential_rows=sequential_rows,
        summary_answers=summary_answers,
        signal_drop_examples=signal_drop_examples,
    )

    try:
        model.to("cpu")
        base_model.to("cpu")
    except Exception:
        pass
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
