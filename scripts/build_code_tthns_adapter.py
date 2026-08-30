#!/usr/bin/env python3
"""Build a label-free test-time HNS adapter for HumanEval-style code tasks.

Stage 1 is read from a ``sensitivity-hns`` adapter calibrated on supervised
code SFT data. Stage 2 never reads HumanEval solutions or tests: it uses only
problem prompts, ranks the fixed HNS module proposals with next-token entropy,
and validates the exact candidate with prompt-view consistency plus a KL trust
region against the original LoRA adapter.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finetune.eval.eval_humaneval import (  # noqa: E402
    CHAT_USER_PROMPT_STYLES,
    build_humaneval_chat_user_prompt,
    load_humaneval_problems,
)
from finetune.eval.generation import load_eval_tokenizer, render_chat_prompt  # noqa: E402
from finetune.eval.test_time_hns import select_candidate  # noqa: E402
from finetune.spectral_edit.cli import _collect_lora_pairs  # noqa: E402
from finetune.spectral_edit.hooks import (  # noqa: E402
    HOOK_CTX,
    ModuleSpec,
    register_sigma_hooks,
    remove_hooks,
)
from finetune.spectral_edit.io import (  # noqa: E402
    get_scaling_for_module,
    load_adapter_config,
    load_lora_state_dict,
    save_lora_state_dict,
)
from finetune.spectral_edit.svd import lowrank_svd_from_ba  # noqa: E402
from finetune.spectral_edit.runtime import saved_tensor_offload_context  # noqa: E402
from finetune.utils import seed_everything  # noqa: E402


def _code_tt_objective(
    probabilities: torch.Tensor,
    *,
    js_weight: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return normalized entropy + prompt-view JS for ``[N, V, vocab]``."""
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [examples, views, vocabulary]")
    if probabilities.shape[-1] < 2:
        raise ValueError("the vocabulary dimension must contain at least two tokens")
    if probabilities.shape[1] < 1:
        raise ValueError("at least one prompt view is required")
    if js_weight < 0:
        raise ValueError("js_weight must be >= 0")

    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    mean_probabilities = probabilities.mean(dim=1)
    entropy_mean = -(
        mean_probabilities * mean_probabilities.clamp_min(eps).log()
    ).sum(dim=-1)
    entropy_views = -(
        probabilities * probabilities.clamp_min(eps).log()
    ).sum(dim=-1).mean(dim=1)
    normalizer = math.log(probabilities.shape[-1])
    entropy = entropy_mean / normalizer
    prompt_view_js = (entropy_mean - entropy_views) / normalizer
    return (entropy + js_weight * prompt_view_js).mean()


def _select_problem_prompts(
    problems: dict[str, dict[str, str]],
    *,
    selection_samples: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    if selection_samples < 1:
        raise ValueError("--selection_samples must be >= 1")
    task_ids = sorted(problems)
    random.Random(seed).shuffle(task_ids)
    task_ids = task_ids[: min(selection_samples, len(task_ids))]
    return task_ids, [problems[task_id]["prompt"] for task_id in task_ids]


def _render_prompt_views(
    *,
    problem_prompts: Sequence[str],
    tokenizer: Any,
    base_model: str,
    prompt_style: str,
    chat_user_prompt_styles: Sequence[str],
    chat_template_mode: str,
    system_prompt: str | None,
) -> list[tuple[str, ...]]:
    if prompt_style == "raw":
        return [(prompt,) for prompt in problem_prompts]

    styles = tuple(dict.fromkeys(style.strip() for style in chat_user_prompt_styles if style.strip()))
    if not styles:
        raise ValueError("chat prompt selection requires at least one --chat_user_prompt_styles value")
    unknown = sorted(set(styles) - set(CHAT_USER_PROMPT_STYLES))
    if unknown:
        raise ValueError(f"Unknown HumanEval chat prompt style: {unknown[0]}")

    return [
        tuple(
            render_chat_prompt(
                tokenizer=tokenizer,
                base_model=base_model,
                user_content=build_humaneval_chat_user_prompt(prompt, style=style),
                system_content=system_prompt,
                chat_template_mode=chat_template_mode,
            )
            for style in styles
        )
        for prompt in problem_prompts
    ]


def _score_prompt_views(
    *,
    model: Any,
    tokenizer: Any,
    adapter_name: str,
    prompt_views: Sequence[Sequence[str]],
    batch_size: int,
    max_seq_len: int,
    js_weight: float,
    compute_gradients: bool,
    cpu_activation_offload: bool,
) -> torch.Tensor:
    if not prompt_views:
        raise ValueError("test-time prompt selection produced zero examples")
    num_views = len(prompt_views[0])
    if num_views < 1 or any(len(row) != num_views for row in prompt_views):
        raise ValueError("every test-time example must have the same non-zero number of views")

    model.set_adapter(adapter_name)
    device = next(model.parameters()).device
    probability_batches: list[torch.Tensor] = []

    if compute_gradients:
        model.zero_grad(set_to_none=True)
        HOOK_CTX.gsum = {}

    total_examples = len(prompt_views)
    for start in range(0, total_examples, batch_size):
        stop = min(start + batch_size, total_examples)
        rows = prompt_views[start:stop]
        flat_prompts = [prompt for row in rows for prompt in row]
        encoded = tokenizer(
            flat_prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        HOOK_CTX.attn_mask = encoded.get("attention_mask")

        grad_context = torch.enable_grad() if compute_gradients else torch.no_grad()
        with grad_context:
            with saved_tensor_offload_context(compute_gradients and cpu_activation_offload):
                next_token_logits = model(**encoded).logits[:, -1, :].float()
                probabilities = next_token_logits.softmax(dim=-1).reshape(
                    stop - start,
                    num_views,
                    next_token_logits.shape[-1],
                )
                if compute_gradients:
                    weight = (stop - start) / total_examples
                    (_code_tt_objective(probabilities, js_weight=js_weight) * weight).backward()
        probability_batches.append(probabilities.detach().cpu())

    return torch.cat(probability_batches, dim=0)


def _copy_adapter_tree(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {destination}. Pass --overwrite to replace it."
            )
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", required=True, type=Path)
    parser.add_argument(
        "--calibration_hns_path",
        required=True,
        type=Path,
        help="Code adapter emitted by `finetune.spectral_edit.cli sensitivity-hns`.",
    )
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--selection_samples", type=int, default=64)
    parser.add_argument("--selection_batch_size", type=int, default=2)
    parser.add_argument("--max_test_modules", type=int, default=None)
    parser.add_argument("--min_test_utility", type=float, default=0.0)
    parser.add_argument("--js_weight", type=float, default=1.0)
    parser.add_argument("--reference_kl_weight", type=float, default=0.25)
    parser.add_argument("--max_reference_kl", type=float, default=0.10)
    parser.add_argument("--min_improvement", type=float, default=0.0)
    parser.add_argument("--prompt_style", choices=("chat", "raw"), default="chat")
    parser.add_argument(
        "--chat_user_prompt_styles",
        nargs="+",
        choices=CHAT_USER_PROMPT_STYLES,
        default=list(CHAT_USER_PROMPT_STYLES),
        help="Equivalent unlabeled prompt views used for test-time consistency.",
    )
    parser.add_argument(
        "--chat_template_mode",
        choices=("auto", "thinking", "non_thinking"),
        default="auto",
    )
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--cpu_activation_offload",
        action="store_true",
        help="Offload autograd-saved tensors to CPU during test-time gradient scoring.",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Code test-time HNS adapter construction requires CUDA")
    if args.selection_batch_size < 1:
        raise ValueError("--selection_batch_size must be >= 1")
    if args.max_test_modules is not None and args.max_test_modules < 1:
        raise ValueError("--max_test_modules must be >= 1")
    if args.max_seq_len < 1:
        raise ValueError("--max_seq_len must be >= 1")
    seed_everything(args.seed)

    meta_path = args.calibration_hns_path / "spectral_edit_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing sensitivity-HNS metadata: {meta_path}")
    calibration_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if calibration_meta.get("meta", {}).get("method") != "calibration_sensitivity_hns":
        raise ValueError(f"Not a sensitivity-hns adapter: {args.calibration_hns_path}")
    calibration_modules = list(calibration_meta.get("summary", {}).get("selected_modules", []))
    if not calibration_modules:
        raise RuntimeError("Calibration sensitivity-HNS selected zero modules")

    original_sd, original_format = load_lora_state_dict(str(args.lora_path))
    hns_sd, _ = load_lora_state_dict(str(args.calibration_hns_path))
    adapter_config = load_adapter_config(str(args.lora_path))
    original_pairs, available_prefixes, _ = _collect_lora_pairs(
        sd=original_sd,
        target_modules=["all_modules"],
        layer_min=0,
        layer_max=10**9,
    )
    hns_pairs, _, _ = _collect_lora_pairs(
        sd=hns_sd,
        target_modules=["all_modules"],
        layer_min=0,
        layer_max=10**9,
    )
    missing = sorted(set(calibration_modules) - set(available_prefixes))
    if missing:
        raise RuntimeError(f"Calibration metadata refers to missing LoRA module: {missing[0]}")

    problems, dataset_source = load_humaneval_problems(
        split=args.split,
        dataset_path=args.dataset_path,
    )
    selected_task_ids, problem_prompts = _select_problem_prompts(
        problems,
        selection_samples=args.selection_samples,
        seed=args.seed,
    )
    tokenizer = load_eval_tokenizer(base_model=args.base_model, adapter_dir=str(args.lora_path))
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    prompt_views = _render_prompt_views(
        problem_prompts=problem_prompts,
        tokenizer=tokenizer,
        base_model=args.base_model,
        prompt_style=args.prompt_style,
        chat_user_prompt_styles=args.chat_user_prompt_styles,
        chat_template_mode=args.chat_template_mode,
        system_prompt=args.system_prompt,
    )

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        cache_dir=args.cache_dir,
    ).to("cuda")
    reference_adapter_name = "lora_reference"
    model = PeftModel.from_pretrained(
        base,
        str(args.lora_path),
        adapter_name=reference_adapter_name,
        is_trainable=True,
    ).to("cuda")
    model.eval()
    model.config.use_cache = False
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name and reference_adapter_name in name)

    name_to_module = dict(model.named_modules())
    specs: dict[str, ModuleSpec] = {}
    hns_deltas: dict[str, torch.Tensor] = {}
    for prefix in calibration_modules:
        _, A_original, adapter_a = original_pairs[prefix]["A"]
        _, B_original, adapter_b = original_pairs[prefix]["B"]
        module_name = prefix
        if module_name not in name_to_module:
            matches = [name for name in name_to_module if name.endswith(prefix)]
            if not matches:
                raise RuntimeError(f"Cannot find module {prefix!r} in model")
            module_name = matches[0]

        U, sigma, Vh, V = lowrank_svd_from_ba(B_original.cuda(), A_original.cuda())
        module_hns_stats = calibration_meta.get("module_stats", {}).get(prefix)
        if module_hns_stats is None:
            raise RuntimeError(f"Missing HNS spectrum metadata for {prefix}")
        sigma_before = torch.tensor(module_hns_stats["sigma_before"], dtype=torch.float32)
        sigma_after = torch.tensor(module_hns_stats["sigma_after"], dtype=torch.float32)
        if sigma_before.shape != sigma.shape or sigma_after.shape != sigma.shape:
            raise RuntimeError(
                f"HNS spectrum metadata rank mismatch for {prefix}: "
                f"original={sigma.numel()} before={sigma_before.numel()} after={sigma_after.numel()}"
            )
        specs[prefix] = ModuleSpec(
            module_prefix=prefix,
            module=name_to_module[module_name],
            U=U.detach(),
            V=V.detach(),
            Vh=Vh.detach(),
            sigma0=sigma.detach().cpu(),
            scaling=get_scaling_for_module(adapter_config, prefix),
            adapter=adapter_a if adapter_a is not None else adapter_b,
        )
        hns_deltas[prefix] = (sigma_after - sigma_before).detach().cpu()

    handles = register_sigma_hooks(specs)
    try:
        print(
            f"[HumanEval] Scoring {len(prompt_views)} unlabeled prompts "
            f"with {len(prompt_views[0])} view(s) each"
        )
        HOOK_CTX.reset()
        reference_probabilities = _score_prompt_views(
            model=model,
            tokenizer=tokenizer,
            adapter_name=reference_adapter_name,
            prompt_views=prompt_views,
            batch_size=args.selection_batch_size,
            max_seq_len=args.max_seq_len,
            js_weight=args.js_weight,
            compute_gradients=True,
            cpu_activation_offload=args.cpu_activation_offload,
        )
        missing_gradients = [prefix for prefix in calibration_modules if prefix not in HOOK_CTX.gsum]
        if missing_gradients:
            raise RuntimeError(f"No test-time gradient for module {missing_gradients[0]}")

        utilities = {
            prefix: float(
                -(HOOK_CTX.gsum[prefix].double() * hns_deltas[prefix].double()).sum().item()
            )
            for prefix in calibration_modules
        }
        ranked = sorted(utilities, key=lambda prefix: (-utilities[prefix], prefix))
        selected = [prefix for prefix in ranked if utilities[prefix] > args.min_test_utility]
        if args.max_test_modules is not None:
            selected = selected[: args.max_test_modules]

        _copy_adapter_tree(args.lora_path, args.out_dir, overwrite=args.overwrite)
        candidate_sd = dict(original_sd)
        for prefix in selected:
            original_key_a, _, _ = original_pairs[prefix]["A"]
            original_key_b, _, _ = original_pairs[prefix]["B"]
            hns_key_a, hns_a, _ = hns_pairs[prefix]["A"]
            hns_key_b, hns_b, _ = hns_pairs[prefix]["B"]
            if original_key_a != hns_key_a or original_key_b != hns_key_b:
                raise RuntimeError(f"Adapter key mismatch for {prefix}")
            candidate_sd[original_key_a] = hns_a
            candidate_sd[original_key_b] = hns_b
        save_lora_state_dict(str(args.out_dir), candidate_sd, original_format)

        candidate_adapter_name = "tthns_code"
        model.load_adapter(
            str(args.out_dir),
            adapter_name=candidate_adapter_name,
            is_trainable=False,
        )
        candidate_probabilities = _score_prompt_views(
            model=model,
            tokenizer=tokenizer,
            adapter_name=candidate_adapter_name,
            prompt_views=prompt_views,
            batch_size=args.selection_batch_size,
            max_seq_len=args.max_seq_len,
            js_weight=args.js_weight,
            compute_gradients=False,
            cpu_activation_offload=False,
        )
        decision = select_candidate(
            {
                "lora": reference_probabilities,
                "test_time_hns": candidate_probabilities,
            },
            reference_name="lora",
            js_weight=args.js_weight,
            reference_kl_weight=args.reference_kl_weight,
            max_reference_kl=args.max_reference_kl,
            min_improvement=args.min_improvement,
        )
        accepted = decision.selected_name == "test_time_hns"
        if not accepted:
            save_lora_state_dict(str(args.out_dir), original_sd, original_format)

        result = {
            "method": "code_calibration_importance_plus_test_time_hns",
            "base_model": args.base_model,
            "lora_path": str(args.lora_path),
            "calibration_hns_path": str(args.calibration_hns_path),
            "dataset_source": dataset_source,
            "split": args.split,
            "selection_task_ids": selected_task_ids,
            "selection_examples": len(prompt_views),
            "prompt_style": args.prompt_style,
            "chat_user_prompt_styles": (
                list(args.chat_user_prompt_styles) if args.prompt_style == "chat" else []
            ),
            "chat_template_mode": args.chat_template_mode,
            "cpu_activation_offload": args.cpu_activation_offload,
            "calibration_candidate_modules": calibration_modules,
            "test_time_module_utilities": utilities,
            "proposed_modules": selected,
            "accepted_modules": selected if accepted else [],
            "selected_adapter": decision.selected_name,
            "candidate_scores": {
                name: score.to_dict() for name, score in decision.scores.items()
            },
            "adapter_dir": str(args.out_dir),
            "seed": args.seed,
        }
        meta_out = args.out_dir / "test_time_hns_meta.json"
        meta_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[HumanEval] proposed={len(selected)} accepted={accepted} "
            f"selected={decision.selected_name} adapter={args.out_dir}"
        )
        print(f"Wrote: {meta_out}")
    finally:
        remove_hooks(handles)
        HOOK_CTX.reset()
        try:
            model.to("cpu")
            base.to("cpu")
        except Exception:
            pass
        del model, base
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
