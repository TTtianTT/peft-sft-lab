from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from finetune.data.registry import get_task_plugin
from finetune.train_profiles import (
    DEFAULT_PROFILE,
    TASK_PROFILE_MAP,
    TRAIN_PROFILES,
    pick_profile,
    profile_overrides,
)
from finetune.train_sft_peft import build_arg_parser as build_train_arg_parser
from finetune.utils import parse_csv_list


METHOD_ORDER = ["lora", "loraplus", "adalora", "pissa"]

TASK_FORMATTING = {
    "MetaMathQATask": (
        "Uses first present of query/original_question/question/instruction/prompt as instruction and "
        "response/answer/output/solution as response; formats with "
        "'### Instruction' + '### Response'."
    ),
    "MagicoderTask": (
        "Uses first present of instruction/prompt/query/problem as instruction and "
        "response/output/answer/completion as response; formats with "
        "'### Instruction' + '### Response'."
    ),
    "AlpacaTask": (
        "If text is present, uses text as-is; otherwise uses instruction + optional input + output, "
        "formatted with '### Instruction' + '### Response'."
    ),
    "CommonsenseQATask": (
        "Formats question + choices (A-E) and expects a single-letter answer; "
        "wraps with '### Instruction' + '### Response'."
    ),
}


def _unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _defaults_from_parser() -> dict[str, Any]:
    parser = build_train_arg_parser()
    defaults: dict[str, Any] = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        if action.default is not argparse.SUPPRESS:
            defaults[action.dest] = action.default
    return defaults


def _adalora_schedule(max_steps: int) -> dict[str, int]:
    total_step = max_steps
    tinit = max(1, int(0.10 * total_step))
    tfinal = max(1, int(0.80 * total_step))
    delta_t = max(1, int(0.01 * total_step))
    return {
        "total_step": total_step,
        "tinit": tinit,
        "tfinal": tfinal,
        "deltaT": delta_t,
    }


def _resolve_batch_settings(
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    global_train_batch_size: int | None,
    num_gpus: int,
) -> dict[str, Any]:
    per_device_batch_size = max(1, int(per_device_batch_size))
    num_gpus = max(1, int(num_gpus))
    denom = per_device_batch_size * num_gpus

    if global_train_batch_size is not None:
        grad_accum = max(1, math.ceil(global_train_batch_size / denom))
        grad_formula = f"ceil({global_train_batch_size} / ({per_device_batch_size} * {num_gpus}))"
        source = "global_train_batch_size"
    else:
        grad_accum = max(1, int(gradient_accumulation_steps))
        grad_formula = "config.gradient_accumulation_steps"
        source = "config"

    effective_gbs = grad_accum * denom
    effective_formula = f"{grad_accum} * {per_device_batch_size} * {num_gpus}"

    return {
        "assumed_num_gpus": num_gpus,
        "gradient_accumulation_steps": grad_accum,
        "gradient_accumulation_source": source,
        "gradient_accumulation_formula": grad_formula,
        "effective_global_batch_size": effective_gbs,
        "effective_global_batch_size_formula": effective_formula,
    }


def resolve_config(cfg: dict[str, Any], defaults: dict[str, Any], num_gpus: int) -> dict[str, Any]:
    resolved = defaults.copy()
    resolved.update(cfg)

    for key in ["bf16", "fp16", "gradient_checkpointing", "use_qlora"]:
        resolved[key] = bool(resolved.get(key, False))

    task = str(resolved.get("task", "")).strip()
    profile = pick_profile(task, resolved.get("train_profile"))
    if profile:
        resolved.update(profile_overrides(profile))
        resolved["train_profile"] = profile.name

    if not resolved.get("lr_scheduler_type"):
        resolved["lr_scheduler_type"] = "linear"
    if resolved.get("min_lr_ratio") is not None and resolved["lr_scheduler_type"] == "linear":
        resolved["lr_scheduler_type"] = "cosine_with_min_lr"

    per_device_bs = int(resolved.get("per_device_train_batch_size") or 1)
    grad_accum = int(resolved.get("gradient_accumulation_steps") or 1)
    gbs_target = resolved.get("global_train_batch_size")
    batch_info = _resolve_batch_settings(per_device_bs, grad_accum, gbs_target, num_gpus)

    peft_method = str(resolved.get("peft_method", "")).lower()
    max_steps = int(resolved.get("max_steps") or 0)
    adalora = _adalora_schedule(max_steps) if peft_method == "adalora" else None

    target_modules = parse_csv_list(str(resolved.get("target_modules") or ""))
    plugin = get_task_plugin(task)
    dataset_id = getattr(plugin, "dataset_id", None)
    formatting = TASK_FORMATTING.get(plugin.__class__.__name__, "Unknown formatting.")

    init_lora_weights = None
    if peft_method == "pissa":
        init_lora_weights = resolved.get("pissa_init_mode")

    lr = float(resolved.get("lr") or 0.0)
    loraplus_lr_ratio = None
    loraplus_lr_embedding = None
    loraplus_lr_assignment = None
    if peft_method == "loraplus":
        loraplus_lr_ratio = resolved.get("loraplus_lr_ratio")
        loraplus_lr_embedding = resolved.get("loraplus_lr_embedding")
        if loraplus_lr_embedding is None:
            emb_note = "lr"
        else:
            emb_note = f"{float(loraplus_lr_embedding)}"
        loraplus_lr_assignment = (
            f"lora_A=lr({lr}), lora_B=lr*ratio({lr}*{loraplus_lr_ratio}), "
            f"lora_embedding_A={emb_note}, lora_embedding_B={emb_note}*ratio, other=lr"
        )

    optimizer_type = "AdamW (transformers default)"
    if peft_method == "loraplus":
        optimizer_type = "AdamW (torch.optim) with LoRA+ param groups"

    resolved_summary = {
        "base_model": resolved.get("base_model"),
        "tokenizer_model": resolved.get("base_model"),
        "task": task,
        "dataset_id": dataset_id,
        "dataset_split": "train",
        "train_profile": resolved.get("train_profile"),
        "max_seq_len": int(resolved.get("max_seq_len") or 0),
        "max_steps": max_steps,
        "num_train_epochs": None,
        "training_schedule": "max_steps",
        "per_device_train_batch_size": per_device_bs,
        "gradient_accumulation_steps": batch_info["gradient_accumulation_steps"],
        "gradient_accumulation_source": batch_info["gradient_accumulation_source"],
        "gradient_accumulation_formula": batch_info["gradient_accumulation_formula"],
        "global_train_batch_size_target": gbs_target,
        "assumed_num_gpus": batch_info["assumed_num_gpus"],
        "effective_global_batch_size": batch_info["effective_global_batch_size"],
        "effective_global_batch_size_formula": batch_info["effective_global_batch_size_formula"],
        "optimizer_type": optimizer_type,
        "optimizer_betas": [resolved.get("adam_beta1"), resolved.get("adam_beta2")],
        "weight_decay": resolved.get("weight_decay"),
        "scheduler_type": resolved.get("lr_scheduler_type"),
        "warmup_ratio": resolved.get("warmup_ratio"),
        "min_lr_ratio": resolved.get("min_lr_ratio"),
        "precision_bf16": resolved.get("bf16"),
        "precision_fp16": resolved.get("fp16"),
        "gradient_checkpointing": resolved.get("gradient_checkpointing"),
        "grad_clip": resolved.get("grad_clip"),
        "target_modules": target_modules,
        "lora_r": resolved.get("r"),
        "lora_alpha": resolved.get("lora_alpha"),
        "lora_dropout": resolved.get("lora_dropout"),
        "init_lora_weights": init_lora_weights,
        "pissa_init_mode": resolved.get("pissa_init_mode"),
        "loraplus_lr_ratio": loraplus_lr_ratio,
        "loraplus_lr_embedding": loraplus_lr_embedding,
        "loraplus_lr_assignment": loraplus_lr_assignment,
        "adalora_init_r": resolved.get("init_r") if peft_method == "adalora" else None,
        "adalora_target_r": resolved.get("target_r") if peft_method == "adalora" else None,
        "adalora_total_step": adalora["total_step"] if adalora else None,
        "adalora_tinit": adalora["tinit"] if adalora else None,
        "adalora_tfinal": adalora["tfinal"] if adalora else None,
        "adalora_deltaT": adalora["deltaT"] if adalora else None,
        "peft_method": peft_method,
        "lr": lr,
        "seed": resolved.get("seed"),
        "output_dir": resolved.get("output_dir"),
        "task_formatting": formatting,
    }
    return resolved_summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(line) for line in lines]


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_bool(value: Any) -> str:
    return "true" if value else "false"


def _format_list(values: list[str]) -> str:
    return ", ".join(values)


def _format_optional(value: Any) -> str:
    return "-" if value is None else str(value)


def build_markdown_report(
    resolved_runs: list[dict[str, Any]],
    out_path: Path,
    num_gpus: int,
) -> None:
    tasks = _unique_in_order([run["task"] for run in resolved_runs])
    base_models = _unique_in_order([str(run["base_model"]) for run in resolved_runs])
    peft_methods = _unique_in_order([run["peft_method"] for run in resolved_runs])

    lines: list[str] = []
    lines.append("# Sweep configuration summary")
    lines.append("")
    lines.append("## Sources and defaults")
    lines.append("- Training defaults: src/finetune/train_sft_peft.py (build_arg_parser)")
    lines.append("- Sweep grid: src/finetune/sweep/make_grid.py")
    lines.append("- Train profiles: src/finetune/train_profiles.py")
    lines.append("- PEFT method specifics: src/finetune/peft_builders.py")
    lines.append("- Task datasets/formatting: src/finetune/data/*.py")
    lines.append("- Base-model specific defaults: none found in code; tokenizer uses base_model")
    lines.append("")
    lines.append("## Assumptions")
    lines.append(f"- assumed_num_gpus: {num_gpus}")
    lines.append("- gradient_accumulation_steps formula:")
    lines.append("  - if global_train_batch_size is set: ceil(target / (per_device_bs * num_gpus))")
    lines.append("  - otherwise: config.gradient_accumulation_steps")
    lines.append("- effective_global_batch_size formula: grad_accum * per_device_bs * num_gpus")
    lines.append("")
    lines.append("## Train profiles")
    profile_headers = [
        "profile",
        "max_seq_len",
        "warmup_ratio",
        "min_lr_ratio",
        "weight_decay",
        "grad_clip",
        "bf16",
        "fp16",
        "adam_beta1",
        "adam_beta2",
        "global_train_batch_size",
        "lr_scheduler_type",
    ]
    profile_rows: list[list[str]] = []
    for name in sorted(TRAIN_PROFILES):
        prof = TRAIN_PROFILES[name]
        profile_rows.append(
            [
                prof.name,
                str(prof.max_seq_len),
                str(prof.warmup_ratio),
                str(prof.min_lr_ratio),
                str(prof.weight_decay),
                str(prof.grad_clip),
                _format_bool(prof.bf16),
                _format_bool(prof.fp16),
                str(prof.adam_beta1),
                str(prof.adam_beta2),
                _format_optional(prof.global_train_batch_size),
                prof.lr_scheduler_type,
            ]
        )
    lines.append(_md_table(profile_headers, profile_rows))
    lines.append("")
    lines.append("### Task -> profile (auto)")
    lines.append(f"- default profile when task not mapped: {DEFAULT_PROFILE}")
    for task_name in sorted(TASK_PROFILE_MAP):
        lines.append(f"- {task_name} -> {TASK_PROFILE_MAP[task_name]}")
    lines.append("")
    lines.append("## Task plugins")
    task_headers = ["task", "dataset_id", "split", "prompt_formatting"]
    task_rows: list[list[str]] = []
    for task in tasks:
        plugin = get_task_plugin(task)
        dataset_id = getattr(plugin, "dataset_id", None)
        formatting = TASK_FORMATTING.get(plugin.__class__.__name__, "Unknown formatting.")
        task_rows.append([task, str(dataset_id), "train", formatting])
    lines.append(_md_table(task_headers, task_rows))
    lines.append("")
    lines.append("## PEFT method specifics")
    lines.append("- lora: LoraConfig without init_lora_weights (peft default).")
    lines.append("- pissa: LoraConfig with init_lora_weights=pissa_init_mode.")
    lines.append("- adalora: AdaLoraConfig with total_step=max_steps, tinit=0.10, tfinal=0.80, deltaT=0.01.")
    lines.append("- loraplus: LoraConfig like lora; optimizer uses LoRA+ param grouping.")
    lines.append("")
    lines.append("## Resolved runs")
    lines.append(f"- base_models: {', '.join(base_models)}")
    lines.append(f"- tasks: {', '.join(tasks)}")
    lines.append(f"- peft_methods: {', '.join(peft_methods)}")
    lines.append("")

    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in resolved_runs:
        task_groups[run["task"]].append(run)

    for task in tasks:
        runs = task_groups[task]
        if not runs:
            continue
        lines.append(f"### Task: {task}")
        dataset_id = runs[0].get("dataset_id")
        lines.append(f"- dataset_id: {dataset_id}")
        lines.append("- dataset_split: train")
        lines.append("")

        core_headers = [
            "base_model",
            "tokenizer_model",
            "peft_method",
            "train_profile",
            "max_seq_len",
            "max_steps",
            "per_device_bs",
            "grad_accum",
            "global_bs_target",
            "assumed_num_gpus",
            "effective_global_bs",
            "grad_accum_formula",
            "lr",
            "warmup_ratio",
            "min_lr_ratio",
            "weight_decay",
            "adam_betas",
            "optimizer",
            "lr_scheduler_type",
            "precision",
            "grad_clip",
            "grad_checkpointing",
        ]
        core_rows: list[list[str]] = []

        peft_headers = [
            "base_model",
            "peft_method",
            "target_modules",
            "r",
            "lora_alpha",
            "lora_dropout",
            "init_lora_weights",
            "loraplus_lr_ratio",
            "loraplus_lr_embedding",
            "loraplus_lr_assignment",
            "adalora_init_r",
            "adalora_target_r",
            "adalora_total_step",
            "adalora_tinit",
            "adalora_tfinal",
            "adalora_deltaT",
        ]
        peft_rows: list[list[str]] = []

        runs_sorted = sorted(
            runs,
            key=lambda r: (
                r["base_model"],
                METHOD_ORDER.index(r["peft_method"]) if r["peft_method"] in METHOD_ORDER else 999,
            ),
        )
        for run in runs_sorted:
            precision = "bf16" if run["precision_bf16"] else "fp16" if run["precision_fp16"] else "fp32"
            core_rows.append(
                [
                    str(run["base_model"]),
                    str(run["tokenizer_model"]),
                    str(run["peft_method"]),
                    str(run["train_profile"]),
                    str(run["max_seq_len"]),
                    str(run["max_steps"]),
                    str(run["per_device_train_batch_size"]),
                    str(run["gradient_accumulation_steps"]),
                    _format_optional(run["global_train_batch_size_target"]),
                    str(run["assumed_num_gpus"]),
                    str(run["effective_global_batch_size"]),
                    str(run["gradient_accumulation_formula"]),
                    str(run["lr"]),
                    str(run["warmup_ratio"]),
                    _format_optional(run["min_lr_ratio"]),
                    str(run["weight_decay"]),
                    f"{run['optimizer_betas'][0]},{run['optimizer_betas'][1]}",
                    str(run["optimizer_type"]),
                    str(run["scheduler_type"]),
                    precision,
                    _format_optional(run["grad_clip"]),
                    _format_bool(run["gradient_checkpointing"]),
                ]
            )

            init_lora_weights = (
                run["init_lora_weights"]
                if run["init_lora_weights"] is not None
                else "not set"
            )
            peft_rows.append(
                [
                    str(run["base_model"]),
                    str(run["peft_method"]),
                    _format_list(run["target_modules"]),
                    str(run["lora_r"]),
                    str(run["lora_alpha"]),
                    str(run["lora_dropout"]),
                    str(init_lora_weights),
                    _format_optional(run["loraplus_lr_ratio"]),
                    _format_optional(run["loraplus_lr_embedding"]),
                    _format_optional(run["loraplus_lr_assignment"]),
                    _format_optional(run["adalora_init_r"]),
                    _format_optional(run["adalora_target_r"]),
                    _format_optional(run["adalora_total_step"]),
                    _format_optional(run["adalora_tinit"]),
                    _format_optional(run["adalora_tfinal"]),
                    _format_optional(run["adalora_deltaT"]),
                ]
            )

        lines.append(_md_table(core_headers, core_rows))
        lines.append("")
        lines.append(_md_table(peft_headers, peft_rows))
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_json_report(resolved_runs: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(resolved_runs, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize sweep configs with resolved hyperparameters.")
    parser.add_argument("--configs", type=str, required=True, help="Path to configs JSONL file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for report outputs.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Assumed number of GPUs for resolving global batch size.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg_path = Path(args.configs)
    if not cfg_path.exists():
        raise FileNotFoundError(str(cfg_path))

    configs = _load_jsonl(cfg_path)
    defaults = _defaults_from_parser()
    resolved = [resolve_config(cfg, defaults, args.num_gpus) for cfg in configs]

    out_dir = Path(args.out_dir)
    build_markdown_report(resolved, out_dir / "sweep_config_summary.md", args.num_gpus)
    build_json_report(resolved, out_dir / "sweep_config_summary.json")


if __name__ == "__main__":
    main()
