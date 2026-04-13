"""Analysis-only black-box search over LoRA singular values."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.csqa_prompt import resolve_csqa_prompt_style
from finetune.eval.eval_csqa import (
    _build_csqa_instruction,
    _build_csqa_prompt,
    _extract_letter,
    _resolve_split,
)
from finetune.eval.eval_gsm8k import (
    _build_prompt_gsm8k_metamath_style,
    _extract_answer as _extract_math_answer,
    _norm as _norm_math_answer,
)
from finetune.eval.generation import generate_greedy, save_json
from finetune.spectral_edit.io import (
    ensure_local_lora_dir,
    layer_idx_from_module_prefix,
    load_adapter_config,
    load_lora_state_dict,
    parse_lora_ab_key,
    save_lora_state_dict,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma
from finetune.utils import seed_everything


def _normalize_model_ref(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    ref = str(value).strip().rstrip("/")
    if not ref:
        return None
    if os.path.isdir(ref):
        return os.path.abspath(ref)
    return ref


def _assert_adapter_matches_base_model(adapter_cfg: dict, base_model: str, lora_dir: str) -> None:
    cfg_base = _normalize_model_ref(adapter_cfg.get("base_model_name_or_path"))
    requested = _normalize_model_ref(base_model)
    if cfg_base is None or requested is None:
        return
    if cfg_base == requested:
        return
    if os.path.basename(cfg_base) == os.path.basename(requested):
        return
    raise ValueError(
        "Adapter/base-model mismatch: "
        f"adapter_config.json expects {cfg_base!r} but --base_model={requested!r} for {lora_dir}."
    )


def _active_adapter_names(model: Any) -> List[str]:
    try:
        active = model.active_adapters
        if callable(active):
            active = active()
    except Exception:
        active = None

    if active is None:
        active = getattr(model, "active_adapter", None)
    if active is None:
        return []
    if isinstance(active, str):
        return [active]
    return [str(name) for name in active]


@dataclass
class EditableModuleSpec:
    module_prefix: str
    module: Any
    key_a: str
    key_b: str
    adapter_name: Optional[str]
    layer_idx: Optional[int]
    U: torch.Tensor
    Vh: torch.Tensor
    sigma0: torch.Tensor
    a_dtype: torch.dtype
    b_dtype: torch.dtype


@dataclass
class GroupMapping:
    mode: str
    names: List[str]
    module_group_ids: Dict[str, torch.Tensor]


def _container_keys(container: Any) -> List[str]:
    if hasattr(container, "keys"):
        try:
            return [str(k) for k in container.keys()]
        except Exception:
            pass
    modules = getattr(container, "_modules", None)
    if isinstance(modules, dict):
        return [str(k) for k in modules.keys()]
    return []


def _resolve_adapter_entry(container: Any, adapter_name: Optional[str]) -> Any:
    keys = _container_keys(container)
    if not keys:
        raise RuntimeError("LoRA container has no adapter entries.")

    if adapter_name is not None and adapter_name in keys:
        return container[adapter_name]
    if "default" in keys:
        return container["default"]
    if len(keys) == 1:
        return container[keys[0]]
    raise RuntimeError(
        f"Cannot resolve active adapter entry. Requested={adapter_name!r}, available={keys!r}"
    )


def _set_module_lora_weights(spec: EditableModuleSpec, a_new: torch.Tensor, b_new: torch.Tensor) -> None:
    lora_a = getattr(spec.module, "lora_A", None)
    lora_b = getattr(spec.module, "lora_B", None)
    if lora_a is None or lora_b is None:
        raise RuntimeError(f"Module {spec.module_prefix} does not expose lora_A/lora_B.")

    a_proj = _resolve_adapter_entry(lora_a, spec.adapter_name)
    b_proj = _resolve_adapter_entry(lora_b, spec.adapter_name)
    a_weight = getattr(a_proj, "weight", None)
    b_weight = getattr(b_proj, "weight", None)
    if a_weight is None or b_weight is None:
        raise RuntimeError(f"Module {spec.module_prefix} adapter entry is missing a weight tensor.")

    with torch.no_grad():
        a_weight.copy_(a_new.to(device=a_weight.device, dtype=a_weight.dtype))
        b_weight.copy_(b_new.to(device=b_weight.device, dtype=b_weight.dtype))


def _selected_pairs(
    state_dict: Dict[str, torch.Tensor],
    target_modules: Sequence[str],
    layer_min: int,
    layer_max: int,
) -> Dict[str, dict[str, tuple[str, torch.Tensor, Optional[str]]]]:
    pairs: Dict[str, dict[str, tuple[str, torch.Tensor, Optional[str]]]] = {}
    target_modules_set = set(target_modules)

    for key, tensor in state_dict.items():
        parsed = parse_lora_ab_key(key)
        if not parsed:
            continue
        prefix, which, adapter_name = parsed
        suffix = prefix.split(".")[-1]
        if suffix not in target_modules_set:
            continue
        layer_idx = layer_idx_from_module_prefix(prefix)
        if layer_idx is not None and not (layer_min <= layer_idx <= layer_max):
            continue
        pairs.setdefault(prefix, {})
        pairs[prefix][which] = (key, tensor, adapter_name)

    return {
        prefix: bundle
        for prefix, bundle in pairs.items()
        if "A" in bundle and "B" in bundle
    }


def _resolve_model_module(name_to_module: Dict[str, Any], prefix: str) -> Any:
    if prefix in name_to_module:
        return name_to_module[prefix]
    candidates = [name for name in name_to_module if name.endswith(prefix)]
    if not candidates:
        raise RuntimeError(f"Cannot find module {prefix!r} in loaded model.")
    return name_to_module[candidates[0]]


def _build_specs(
    *,
    model: Any,
    state_dict: Dict[str, torch.Tensor],
    target_modules: Sequence[str],
    layer_min: int,
    layer_max: int,
) -> tuple[Dict[str, EditableModuleSpec], Dict[str, dict[str, tuple[str, torch.Tensor, Optional[str]]]]]:
    pairs = _selected_pairs(state_dict, target_modules, layer_min, layer_max)
    if not pairs:
        raise RuntimeError("No matching LoRA (A,B) pairs found for the requested modules/layer range.")

    name_to_module = dict(model.named_modules())
    specs: Dict[str, EditableModuleSpec] = {}

    for prefix, bundle in pairs.items():
        key_a, a_cpu, adapter_a = bundle["A"]
        key_b, b_cpu, adapter_b = bundle["B"]
        adapter_name = adapter_a if adapter_a is not None else adapter_b
        module = _resolve_model_module(name_to_module, prefix)
        try:
            module_device = next(module.parameters()).device
        except StopIteration:
            module_device = torch.device("cpu")

        a_dev = a_cpu.to(module_device)
        b_dev = b_cpu.to(module_device)
        U, S, Vh, _ = lowrank_svd_from_ba(b_dev, a_dev)
        specs[prefix] = EditableModuleSpec(
            module_prefix=prefix,
            module=module,
            key_a=key_a,
            key_b=key_b,
            adapter_name=adapter_name,
            layer_idx=layer_idx_from_module_prefix(prefix),
            U=U.detach(),
            Vh=Vh.detach(),
            sigma0=S.detach(),
            a_dtype=a_cpu.dtype,
            b_dtype=b_cpu.dtype,
        )

    return specs, pairs


def _load_csqa_subset(
    *,
    split: str,
    start: int,
    max_samples: Optional[int],
    prompt_style: str,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(f"datasets is required: {exc}") from exc

    split_to_load, split_note = _resolve_split(split)
    dataset = load_dataset("tau/commonsense_qa", split=split_to_load)

    total = len(dataset)
    if start < 0:
        raise ValueError("--subset_start must be non-negative.")
    if start >= total:
        raise ValueError(f"--subset_start={start} is out of range for split size {total}.")

    end = total if max_samples is None else min(total, start + max_samples)
    indices = list(range(start, end))
    subset = dataset.select(indices)

    prompts: List[str] = []
    golds: List[str] = []
    records: List[dict[str, Any]] = []
    for orig_idx, example in zip(indices, subset):
        question, gold, instruction = _build_csqa_instruction(example)
        prompt = _build_csqa_prompt(instruction, prompt_style=prompt_style)
        prompts.append(prompt)
        golds.append(gold)
        records.append(
            {
                "orig_index": int(orig_idx),
                "id": example.get("id"),
                "question": question,
                "gold": gold,
                "instruction": instruction,
                "prompt": prompt,
            }
        )

    return {
        "dataset": "tau/commonsense_qa",
        "split": split_to_load,
        "split_note": split_note,
        "prompt_style": prompt_style,
        "prompts": prompts,
        "golds": golds,
        "records": records,
        "subset_indices": indices,
        "metric_name": "accuracy",
    }


def _load_math_subset(
    *,
    split: str,
    start: int,
    max_samples: Optional[int],
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(f"datasets is required: {exc}") from exc

    dataset = load_dataset("gsm8k", "main", split=split)
    total = len(dataset)
    if start < 0:
        raise ValueError("--subset_start must be non-negative.")
    if start >= total:
        raise ValueError(f"--subset_start={start} is out of range for split size {total}.")

    end = total if max_samples is None else min(total, start + max_samples)
    indices = list(range(start, end))
    subset = dataset.select(indices)

    prompts: List[str] = []
    golds: List[str] = []
    records: List[dict[str, Any]] = []
    for orig_idx, example in zip(indices, subset):
        question = str(example.get("question", "")).strip()
        gold_raw = str(example.get("answer", "")).strip()
        gold = _norm_math_answer(_extract_math_answer(gold_raw))
        prompt = _build_prompt_gsm8k_metamath_style(question)
        prompts.append(prompt)
        golds.append(gold)
        records.append(
            {
                "orig_index": int(orig_idx),
                "question": question,
                "gold_raw": gold_raw,
                "gold": gold,
                "prompt": prompt,
            }
        )

    return {
        "dataset": "gsm8k/main",
        "split": split,
        "split_note": f"Using split={split!r}.",
        "prompt_style": "metamath_model_usage",
        "prompts": prompts,
        "golds": golds,
        "records": records,
        "subset_indices": indices,
        "metric_name": "accuracy_strict",
    }


def evaluate_csqa_subset(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    golds: Sequence[str],
    max_new_tokens: int,
    return_predictions: bool = False,
) -> dict[str, Any]:
    correct = 0
    total = 0
    predictions: List[dict[str, Any]] = []

    for prompt, gold in zip(prompts, golds):
        text = generate_greedy(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        pred = _extract_letter(text)
        is_correct = int(pred == gold)
        correct += is_correct
        total += 1
        if return_predictions:
            predictions.append(
                {
                    "raw_generation": text,
                    "prediction": pred,
                    "gold": gold,
                    "correct": bool(is_correct),
                }
            )

    result: dict[str, Any] = {
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
    }
    if return_predictions:
        result["predictions"] = predictions
    return result


def evaluate_math_subset(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    golds: Sequence[str],
    max_new_tokens: int,
    return_predictions: bool = False,
) -> dict[str, Any]:
    correct = 0
    total = 0
    predictions: List[dict[str, Any]] = []

    for prompt, gold in zip(prompts, golds):
        text = generate_greedy(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        pred = _norm_math_answer(_extract_math_answer(text))
        is_correct = int(pred == gold)
        correct += is_correct
        total += 1
        if return_predictions:
            predictions.append(
                {
                    "raw_generation": text,
                    "prediction": pred,
                    "gold": gold,
                    "correct": bool(is_correct),
                }
            )

    result: dict[str, Any] = {
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
    }
    if return_predictions:
        result["predictions"] = predictions
    return result


def _extract_sigma_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    specs: Dict[str, EditableModuleSpec],
) -> Dict[str, torch.Tensor]:
    sigma_by_prefix: Dict[str, torch.Tensor] = {}
    for prefix, spec in specs.items():
        a_cpu = state_dict[spec.key_a]
        b_cpu = state_dict[spec.key_b]
        device = spec.sigma0.device
        _, sigma, _, _ = lowrank_svd_from_ba(b_cpu.to(device), a_cpu.to(device))
        sigma_by_prefix[prefix] = sigma.detach()
    return sigma_by_prefix


def _group_mapping(specs: Dict[str, EditableModuleSpec], mode: str) -> GroupMapping:
    names: List[str] = []
    module_group_ids: Dict[str, torch.Tensor] = {}

    if mode == "global":
        names = ["global"]
        for prefix, spec in specs.items():
            module_group_ids[prefix] = torch.zeros(spec.sigma0.numel(), dtype=torch.long)
        return GroupMapping(mode=mode, names=names, module_group_ids=module_group_ids)

    if mode == "per_layer":
        layer_to_gid: Dict[str, int] = {}
        for prefix, spec in specs.items():
            key = f"layer_{spec.layer_idx}" if spec.layer_idx is not None else prefix
            if key not in layer_to_gid:
                layer_to_gid[key] = len(names)
                names.append(key)
            gid = layer_to_gid[key]
            module_group_ids[prefix] = torch.full((spec.sigma0.numel(),), gid, dtype=torch.long)
        return GroupMapping(mode=mode, names=names, module_group_ids=module_group_ids)

    if mode == "per_module":
        for prefix, spec in specs.items():
            gid = len(names)
            names.append(prefix)
            module_group_ids[prefix] = torch.full((spec.sigma0.numel(),), gid, dtype=torch.long)
        return GroupMapping(mode=mode, names=names, module_group_ids=module_group_ids)

    if mode == "per_singular_value":
        for prefix, spec in specs.items():
            group_ids = []
            for idx in range(spec.sigma0.numel()):
                gid = len(names)
                names.append(f"{prefix}#sv{idx:02d}")
                group_ids.append(gid)
            module_group_ids[prefix] = torch.tensor(group_ids, dtype=torch.long)
        return GroupMapping(mode=mode, names=names, module_group_ids=module_group_ids)

    raise ValueError(f"Unknown parameterization mode: {mode}")


def _sigma_from_theta(
    *,
    spec: EditableModuleSpec,
    base_sigma: torch.Tensor,
    group_ids: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    theta_local = theta[group_ids].to(device=base_sigma.device, dtype=base_sigma.dtype)
    return base_sigma * torch.exp(theta_local)


def apply_candidate_to_model(
    *,
    specs: Dict[str, EditableModuleSpec],
    base_sigma_by_prefix: Dict[str, torch.Tensor],
    mapping: GroupMapping,
    theta: torch.Tensor,
) -> None:
    for prefix, spec in specs.items():
        sigma = _sigma_from_theta(
            spec=spec,
            base_sigma=base_sigma_by_prefix[prefix],
            group_ids=mapping.module_group_ids[prefix],
            theta=theta,
        )
        b_new, a_new = rebuild_ba_from_uv_sigma(spec.U, spec.Vh, sigma)
        _set_module_lora_weights(spec, a_new, b_new)


def apply_state_dict_to_model(
    *,
    specs: Dict[str, EditableModuleSpec],
    state_dict: Dict[str, torch.Tensor],
) -> None:
    for spec in specs.values():
        a_new = state_dict[spec.key_a]
        b_new = state_dict[spec.key_b]
        _set_module_lora_weights(spec, a_new, b_new)


def _save_candidate_adapter(
    *,
    out_dir: str,
    source_lora_dir: str,
    state_dict: Dict[str, torch.Tensor],
    fmt: str,
    specs: Dict[str, EditableModuleSpec],
    base_sigma_by_prefix: Dict[str, torch.Tensor],
    mapping: GroupMapping,
    theta: torch.Tensor,
    meta: dict,
) -> None:
    out_path = Path(out_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    shutil.copytree(source_lora_dir, out_path)

    new_state_dict = dict(state_dict)
    for prefix, spec in specs.items():
        sigma = _sigma_from_theta(
            spec=spec,
            base_sigma=base_sigma_by_prefix[prefix],
            group_ids=mapping.module_group_ids[prefix],
            theta=theta,
        )
        b_new, a_new = rebuild_ba_from_uv_sigma(spec.U, spec.Vh, sigma)
        new_state_dict[spec.key_a] = a_new.to(dtype=spec.a_dtype).detach().cpu()
        new_state_dict[spec.key_b] = b_new.to(dtype=spec.b_dtype).detach().cpu()

    save_lora_state_dict(str(out_path), new_state_dict, fmt)
    save_json(out_path / "direct_search_meta.json", meta)


def _sample_theta(
    *,
    generator: torch.Generator,
    center: torch.Tensor,
    std: float,
    clip: float,
) -> torch.Tensor:
    proposal = center + torch.randn(center.shape[0], generator=generator) * float(std)
    return proposal.clamp(min=-float(clip), max=float(clip))


def _candidate_summary(theta: torch.Tensor, score: float, label: str, source: str) -> dict:
    return {
        "label": label,
        "source": source,
        "score": float(score),
        "theta_min": float(theta.min().item()) if theta.numel() else 0.0,
        "theta_max": float(theta.max().item()) if theta.numel() else 0.0,
        "theta_l2": float(torch.linalg.norm(theta).item()) if theta.numel() else 0.0,
    }


def run_search(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Direct black-box spectrum search requires CUDA.")

    lora_dir = ensure_local_lora_dir(args.adapter_dir, cache_dir=args.cache_dir)
    adapter_cfg = load_adapter_config(lora_dir)
    _assert_adapter_matches_base_model(adapter_cfg, args.base_model, lora_dir)
    state_dict, fmt = load_lora_state_dict(lora_dir)

    gradient_state_dict = None
    gradient_dir = None
    if args.gradient_edit_dir:
        gradient_dir = ensure_local_lora_dir(args.gradient_edit_dir, cache_dir=args.cache_dir)
        gradient_state_dict, _ = load_lora_state_dict(gradient_dir)

    if args.task == "csqa":
        prompt_resolution = resolve_csqa_prompt_style(args.prompt_style, gradient_dir or lora_dir)
        prompt_style_resolved = prompt_resolution.resolved
        prompt_style_reason = prompt_resolution.reason
    elif args.task == "math":
        prompt_style_resolved = "metamath_model_usage"
        prompt_style_reason = "fixed prompt template for GSM8K/MetaMath evaluation"
    else:
        raise ValueError(f"Unsupported task for direct search: {args.task}")

    print(f"[DirectSearch] task={args.task}")
    print(f"[DirectSearch] base_model={args.base_model}")
    print(f"[DirectSearch] adapter_dir={lora_dir}")
    print(f"[DirectSearch] gradient_edit_dir={gradient_dir}")
    print(
        "[DirectSearch] "
        f"prompt_style={prompt_style_resolved} "
        f"(requested={args.prompt_style}; reason={prompt_style_reason})"
    )

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = None

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        cache_dir=args.cache_dir,
    )
    model = PeftModel.from_pretrained(base, lora_dir).to(device)
    if not getattr(model, "peft_config", None):
        raise RuntimeError(f"Adapter load failed: no PEFT config found for {lora_dir}")
    if not _active_adapter_names(model):
        raise RuntimeError(f"Adapter load failed: no active adapter after loading {lora_dir}")
    model.eval()

    specs, _ = _build_specs(
        model=model,
        state_dict=state_dict,
        target_modules=args.target_modules,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
    )
    print(f"[DirectSearch] editable_modules={len(specs)}")

    if args.task == "csqa":
        subset = _load_csqa_subset(
            split=args.split,
            start=args.subset_start,
            max_samples=args.subset_size,
            prompt_style=prompt_style_resolved,
        )
        eval_fn = evaluate_csqa_subset
    else:
        subset = _load_math_subset(
            split=args.split,
            start=args.subset_start,
            max_samples=args.subset_size,
        )
        eval_fn = evaluate_math_subset
    print(
        "[DirectSearch] "
        f"dataset={subset['dataset']} split={subset['split']} subset_start={args.subset_start} "
        f"subset_size={len(subset['records'])}"
    )

    mapping = _group_mapping(specs, args.parameterization)
    print(
        "[DirectSearch] "
        f"parameterization={args.parameterization} search_dims={len(mapping.names)} "
        f"search_method={args.search_method}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(
        Path(args.output_dir) / "validation_subset.json",
        {
            "task": args.task,
            "dataset": subset["dataset"],
            "split": subset["split"],
            "split_note": subset["split_note"],
            "prompt_style": prompt_style_resolved,
            "subset_start": args.subset_start,
            "subset_size": len(subset["records"]),
            "records": subset["records"],
        },
    )

    prompts = subset["prompts"]
    golds = subset["golds"]

    def run_eval(
        *,
        label: str,
        return_predictions: bool = False,
    ) -> dict[str, Any]:
        metrics = eval_fn(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            golds=golds,
            max_new_tokens=args.max_new_tokens,
            return_predictions=return_predictions,
        )
        print(
            "[Eval] "
            f"{label}: accuracy={metrics['accuracy']:.4f} "
            f"({metrics['correct']}/{metrics['total']})"
        )
        return metrics

    apply_state_dict_to_model(specs=specs, state_dict=state_dict)
    original_metrics = run_eval(label="original", return_predictions=True)
    save_json(Path(args.output_dir) / "original_eval.json", original_metrics)

    gradient_metrics = None
    if gradient_state_dict is not None:
        apply_state_dict_to_model(specs=specs, state_dict=gradient_state_dict)
        gradient_metrics = run_eval(label="gradient_edit", return_predictions=True)
        save_json(Path(args.output_dir) / "gradient_edit_eval.json", gradient_metrics)
        apply_state_dict_to_model(specs=specs, state_dict=state_dict)

    base_sigma_by_prefix = (
        _extract_sigma_from_state_dict(gradient_state_dict, specs)
        if gradient_state_dict is not None and args.init_from == "gradient_edit"
        else {prefix: spec.sigma0.clone() for prefix, spec in specs.items()}
    )
    init_label = "gradient_edit" if gradient_state_dict is not None and args.init_from == "gradient_edit" else "original"
    init_metrics = gradient_metrics if init_label == "gradient_edit" else original_metrics
    init_theta = torch.zeros(len(mapping.names), dtype=torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    search_trace: List[dict[str, Any]] = []

    def evaluate_theta(theta: torch.Tensor, label: str, source: str) -> tuple[float, dict[str, Any]]:
        apply_candidate_to_model(
            specs=specs,
            base_sigma_by_prefix=base_sigma_by_prefix,
            mapping=mapping,
            theta=theta,
        )
        metrics = run_eval(label=label, return_predictions=False)
        search_trace.append(_candidate_summary(theta, metrics["accuracy"], label, source))
        return float(metrics["accuracy"]), metrics

    best_random_theta = init_theta.clone()
    best_random_score = float("-inf")
    for idx in range(args.random_trials):
        theta = _sample_theta(
            generator=generator,
            center=init_theta,
            std=args.step_std,
            clip=args.log_scale_clip,
        )
        score, _ = evaluate_theta(theta, label=f"random_{idx:03d}", source="random")
        if score > best_random_score:
            best_random_score = score
            best_random_theta = theta.clone()

    apply_candidate_to_model(
        specs=specs,
        base_sigma_by_prefix=base_sigma_by_prefix,
        mapping=mapping,
        theta=best_random_theta,
    )
    random_metrics = run_eval(label="best_random", return_predictions=True)
    save_json(Path(args.output_dir) / "random_eval.json", random_metrics)

    current_theta = init_theta.clone()
    current_score = float(init_metrics["accuracy"])
    best_theta = current_theta.clone()
    best_score = current_score
    step_std = float(args.step_std)

    if args.search_method == "random":
        current_theta = best_random_theta.clone()
        current_score = float(random_metrics["accuracy"])
        best_theta = current_theta.clone()
        best_score = current_score
    else:
        search_trace.append(_candidate_summary(current_theta, current_score, "search_init", "init"))
        for iteration in range(args.search_iterations):
            population_best_theta = current_theta.clone()
            population_best_score = float("-inf")
            for cand_idx in range(args.population_size):
                theta = _sample_theta(
                    generator=generator,
                    center=current_theta,
                    std=step_std,
                    clip=args.log_scale_clip,
                )
                score, _ = evaluate_theta(
                    theta,
                    label=f"search_iter{iteration:02d}_cand{cand_idx:02d}",
                    source="local_es",
                )
                if score > population_best_score:
                    population_best_score = score
                    population_best_theta = theta.clone()

            improved = population_best_score > current_score + 1e-12
            if improved:
                current_theta = population_best_theta
                current_score = population_best_score
                step_std = min(step_std * args.step_expand, args.step_std_max)
            else:
                step_std = max(step_std * args.step_shrink, args.step_std_min)

            print(
                "[Search] "
                f"iter={iteration + 1}/{args.search_iterations} "
                f"current_score={current_score:.4f} step_std={step_std:.4f}"
            )
            if current_score > best_score + 1e-12:
                best_theta = current_theta.clone()
                best_score = current_score

    apply_candidate_to_model(
        specs=specs,
        base_sigma_by_prefix=base_sigma_by_prefix,
        mapping=mapping,
        theta=best_theta,
    )
    direct_metrics = run_eval(label="best_direct", return_predictions=True)
    save_json(Path(args.output_dir) / "direct_eval.json", direct_metrics)

    common_meta = {
        "task": args.task,
        "base_model": args.base_model,
        "adapter_dir": lora_dir,
        "gradient_edit_dir": gradient_dir,
        "search_method": args.search_method,
        "parameterization": args.parameterization,
        "search_dims": len(mapping.names),
        "group_names": mapping.names,
        "target_modules": list(args.target_modules),
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "dataset": subset["dataset"],
        "split": subset["split"],
        "split_note": subset["split_note"],
        "subset_start": args.subset_start,
        "subset_size": len(subset["records"]),
        "metric_name": subset["metric_name"],
        "prompt_style_requested": args.prompt_style,
        "prompt_style_resolved": prompt_style_resolved,
        "prompt_style_reason": prompt_style_reason,
        "eval_batch_size": args.eval_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "seed": args.seed,
        "init_from": args.init_from,
        "random_trials": args.random_trials,
        "search_iterations": args.search_iterations,
        "population_size": args.population_size,
        "step_std_init": args.step_std,
        "step_std_final": step_std,
        "log_scale_clip": args.log_scale_clip,
        "results": {
            "original": {k: v for k, v in original_metrics.items() if k != "predictions"},
            "gradient_edit": None if gradient_metrics is None else {k: v for k, v in gradient_metrics.items() if k != "predictions"},
            "random": {k: v for k, v in random_metrics.items() if k != "predictions"},
            "direct": {k: v for k, v in direct_metrics.items() if k != "predictions"},
        },
    }

    _save_candidate_adapter(
        out_dir=str(Path(args.output_dir) / "best_random_adapter"),
        source_lora_dir=lora_dir,
        state_dict=state_dict,
        fmt=fmt,
        specs=specs,
        base_sigma_by_prefix=base_sigma_by_prefix,
        mapping=mapping,
        theta=best_random_theta,
        meta={
            **common_meta,
            "candidate_type": "random",
            "theta": best_random_theta.tolist(),
            "score": float(random_metrics["accuracy"]),
        },
    )

    _save_candidate_adapter(
        out_dir=str(Path(args.output_dir) / "best_direct_adapter"),
        source_lora_dir=lora_dir,
        state_dict=state_dict,
        fmt=fmt,
        specs=specs,
        base_sigma_by_prefix=base_sigma_by_prefix,
        mapping=mapping,
        theta=best_theta,
        meta={
            **common_meta,
            "candidate_type": "direct",
            "theta": best_theta.tolist(),
            "score": float(direct_metrics["accuracy"]),
        },
    )

    save_json(
        Path(args.output_dir) / "search_summary.json",
        {
            **common_meta,
            "best_random_theta": best_random_theta.tolist(),
            "best_direct_theta": best_theta.tolist(),
            "trace": search_trace,
        },
    )

    apply_state_dict_to_model(specs=specs, state_dict=state_dict)

    try:
        model.to("cpu")
        base.to("cpu")
    except Exception:
        pass
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analysis-only direct black-box search over LoRA singular values."
    )
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--gradient_edit_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="csqa", choices=["csqa", "math"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--subset_start", type=int, default=0)
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Reserved for future batched evaluators. The CSQA analysis path currently scores one prompt at a time to match eval_csqa exactly.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=8)
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
    )
    parser.add_argument("--layer_min", type=int, default=0)
    parser.add_argument("--layer_max", type=int, default=10**9)

    parser.add_argument(
        "--parameterization",
        type=str,
        default="per_layer",
        choices=["global", "per_layer", "per_module", "per_singular_value"],
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default="gradient_edit",
        choices=["original", "gradient_edit"],
        help="Where the direct-search center starts. gradient_edit requires --gradient_edit_dir.",
    )
    parser.add_argument(
        "--search_method",
        type=str,
        default="local_es",
        choices=["local_es", "random"],
    )
    parser.add_argument("--random_trials", type=int, default=8)
    parser.add_argument("--search_iterations", type=int, default=6)
    parser.add_argument("--population_size", type=int, default=4)
    parser.add_argument("--step_std", type=float, default=0.12)
    parser.add_argument("--step_std_min", type=float, default=0.02)
    parser.add_argument("--step_std_max", type=float, default=0.35)
    parser.add_argument("--step_expand", type=float, default=1.1)
    parser.add_argument("--step_shrink", type=float, default=0.7)
    parser.add_argument(
        "--log_scale_clip",
        type=float,
        default=0.5,
        help="Candidate log-scale clamp. 0.5 means multiplicative scales stay in [exp(-0.5), exp(0.5)].",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.init_from == "gradient_edit" and not args.gradient_edit_dir:
        raise ValueError("--init_from gradient_edit requires --gradient_edit_dir.")
    run_search(args)


if __name__ == "__main__":
    main()
