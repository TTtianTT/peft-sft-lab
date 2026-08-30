#!/usr/bin/env python3
"""Build task-specific test-time HNS adapters for the Commonsense 8-task suite.

Stage 1 is read from a ``sensitivity-hns`` adapter: calibration CE has already
identified task-important, HNS-compatible modules. Stage 2 uses only unlabeled
option probabilities on each test task. It ranks the fixed module edits by
``-<dL_TT/dsigma, sigma_hns-sigma>`` and exactly validates the resulting adapter
with entropy, option-permutation consistency, and a KL trust region.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_commonsense_8tasks import (  # noqa: E402
    LETTERS,
    TASKS,
    _load_task_dataset,
    _parse_tasks,
)
from finetune.eval.generation import load_eval_tokenizer, render_chat_prompt  # noqa: E402
from finetune.eval.test_time_hns import (  # noqa: E402
    make_choice_permutations,
    select_candidate_grouped,
)
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
from finetune.utils import seed_everything  # noqa: E402


@dataclass(frozen=True)
class ViewGroup:
    num_choices: int
    prompts: tuple[tuple[str, ...], ...]
    permutations: tuple[tuple[tuple[int, ...], ...], ...]

    @property
    def num_examples(self) -> int:
        return len(self.prompts)


def _instruction(question: str, displayed_choices: tuple[str, ...]) -> str:
    lines = [f"{LETTERS[index]}. {choice}" for index, choice in enumerate(displayed_choices)]
    valid = ", ".join(LETTERS[: len(displayed_choices)])
    return (
        f"{question}\n\nChoices:\n"
        + "\n".join(lines)
        + f"\n\nAnswer with only one letter: {valid}."
    )


def _stable_task_offset(task_name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(task_name))


def _prepare_view_groups(
    *,
    task_name: str,
    tokenizer: Any,
    base_model: str,
    chat_template_mode: str,
    selection_samples: int,
    num_permutations: int,
    seed: int,
) -> list[ViewGroup]:
    spec = TASKS[task_name]
    dataset = _load_task_dataset(task_name, spec)
    if selection_samples < 1:
        raise ValueError("--selection_samples must be >= 1")
    count = min(selection_samples, len(dataset))
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed + _stable_task_offset(task_name))
    if hasattr(dataset, "select"):
        dataset = dataset.select(range(count))

    grouped: dict[int, list[tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]]] = {}
    for index, example in enumerate(dataset):
        item = spec.formatter(example, index)
        choice_count = len(item.choices)
        permutations = make_choice_permutations(
            choice_count,
            num_permutations,
            seed=seed + _stable_task_offset(task_name) + index,
        )
        prompts = []
        for display_to_original in permutations:
            displayed = tuple(item.choices[original] for original in display_to_original)
            prompts.append(
                render_chat_prompt(
                    tokenizer=tokenizer,
                    base_model=base_model,
                    user_content=_instruction(item.question, displayed),
                    chat_template_mode=chat_template_mode,
                )
            )
        grouped.setdefault(choice_count, []).append((tuple(prompts), tuple(permutations)))

    return [
        ViewGroup(
            num_choices=choice_count,
            prompts=tuple(row[0] for row in rows),
            permutations=tuple(row[1] for row in rows),
        )
        for choice_count, rows in sorted(grouped.items())
    ]


def _choice_token_ids(tokenizer: Any, prompt: str, num_choices: int) -> list[int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    token_ids: list[int] = []
    for letter in LETTERS[:num_choices]:
        full_ids = tokenizer(prompt + letter, add_special_tokens=False).input_ids
        if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) != len(prompt_ids) + 1:
            direct = tokenizer(letter, add_special_tokens=False).input_ids
            if len(direct) != 1:
                raise RuntimeError(
                    f"Choice {letter!r} is not a single token for this tokenizer; "
                    "the current test-time scorer requires single-token answer letters."
                )
            token_ids.append(int(direct[0]))
        else:
            token_ids.append(int(full_ids[-1]))
    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError(f"Choice letters map to duplicate token IDs: {token_ids}")
    return token_ids


def _restore_probabilities(
    displayed: torch.Tensor,
    permutations: tuple[tuple[tuple[int, ...], ...], ...],
) -> torch.Tensor:
    """Differentiably map ``[examples, views, displayed choices]`` to originals."""
    restored_rows = []
    for example_index, example_permutations in enumerate(permutations):
        views = []
        for view_index, display_to_original in enumerate(example_permutations):
            inverse = [0] * len(display_to_original)
            for display_index, original_index in enumerate(display_to_original):
                inverse[original_index] = display_index
            views.append(displayed[example_index, view_index, inverse])
        restored_rows.append(torch.stack(views, dim=0))
    return torch.stack(restored_rows, dim=0)


def _tt_objective(probabilities: torch.Tensor, js_weight: float, eps: float = 1e-12) -> torch.Tensor:
    mean_probabilities = probabilities.mean(dim=1)
    entropy_mean = -(
        mean_probabilities * mean_probabilities.clamp_min(eps).log()
    ).sum(dim=-1)
    entropy_views = -(
        probabilities * probabilities.clamp_min(eps).log()
    ).sum(dim=-1).mean(dim=1)
    normalizer = math.log(probabilities.shape[-1])
    entropy = entropy_mean / normalizer
    permutation_js = (entropy_mean - entropy_views) / normalizer
    return (entropy + js_weight * permutation_js).mean()


def _score_groups(
    *,
    model: Any,
    tokenizer: Any,
    adapter_name: str,
    groups: list[ViewGroup],
    batch_size: int,
    max_seq_len: int,
    js_weight: float,
    compute_gradients: bool,
) -> list[torch.Tensor]:
    model.set_adapter(adapter_name)
    device = next(model.parameters()).device
    total_examples = sum(group.num_examples for group in groups)
    output_groups: list[torch.Tensor] = []

    if compute_gradients:
        model.zero_grad(set_to_none=True)
        HOOK_CTX.gsum = {}

    for group in groups:
        group_probabilities = []
        if group.num_examples == 0:
            continue
        choice_ids = _choice_token_ids(tokenizer, group.prompts[0][0], group.num_choices)
        choice_ids_tensor = torch.tensor(choice_ids, dtype=torch.long, device=device)

        for start in range(0, group.num_examples, batch_size):
            stop = min(start + batch_size, group.num_examples)
            prompt_rows = group.prompts[start:stop]
            permutation_rows = group.permutations[start:stop]
            flat_prompts = [prompt for row in prompt_rows for prompt in row]
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

            context = torch.enable_grad() if compute_gradients else torch.no_grad()
            with context:
                logits = model(**encoded).logits[:, -1, :]
                choice_logits = logits.index_select(-1, choice_ids_tensor)
                displayed = choice_logits.softmax(dim=-1).reshape(
                    stop - start,
                    len(prompt_rows[0]),
                    group.num_choices,
                )
                restored = _restore_probabilities(displayed, permutation_rows)
                if compute_gradients:
                    weight = (stop - start) / total_examples
                    (_tt_objective(restored, js_weight) * weight).backward()
            group_probabilities.append(restored.detach().cpu())

        output_groups.append(torch.cat(group_probabilities, dim=0))

    return output_groups


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
        help="Adapter emitted by `finetune.spectral_edit.cli sensitivity-hns`.",
    )
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--selection_samples", type=int, default=64)
    parser.add_argument("--num_permutations", type=int, default=4)
    parser.add_argument("--selection_batch_size", type=int, default=4)
    parser.add_argument("--max_test_modules", type=int, default=None)
    parser.add_argument("--min_test_utility", type=float, default=0.0)
    parser.add_argument("--js_weight", type=float, default=1.0)
    parser.add_argument("--reference_kl_weight", type=float, default=0.25)
    parser.add_argument("--max_reference_kl", type=float, default=0.10)
    parser.add_argument("--min_improvement", type=float, default=0.0)
    parser.add_argument(
        "--chat_template_mode",
        choices=("auto", "thinking", "non_thinking"),
        default="auto",
    )
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Test-time HNS adapter construction requires CUDA")
    if args.selection_batch_size < 1:
        raise ValueError("--selection_batch_size must be >= 1")
    if args.max_test_modules is not None and args.max_test_modules < 1:
        raise ValueError("--max_test_modules must be >= 1")
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

    tokenizer = load_eval_tokenizer(base_model=args.base_model, adapter_dir=str(args.lora_path))
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
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

    selected_tasks = _parse_tasks(args.tasks)
    handles = register_sigma_hooks(specs)
    all_results: dict[str, Any] = {}
    args.out_root.mkdir(parents=True, exist_ok=True)
    try:
        for task_name in selected_tasks:
            print(f"\n[{task_name}] Preparing unlabeled test-time views")
            groups = _prepare_view_groups(
                task_name=task_name,
                tokenizer=tokenizer,
                base_model=args.base_model,
                chat_template_mode=args.chat_template_mode,
                selection_samples=args.selection_samples,
                num_permutations=args.num_permutations,
                seed=args.seed,
            )
            HOOK_CTX.reset()
            reference_probabilities = _score_groups(
                model=model,
                tokenizer=tokenizer,
                adapter_name=reference_adapter_name,
                groups=groups,
                batch_size=args.selection_batch_size,
                max_seq_len=args.max_seq_len,
                js_weight=args.js_weight,
                compute_gradients=True,
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

            task_dir = args.out_root / task_name
            _copy_adapter_tree(args.lora_path, task_dir, overwrite=args.overwrite)
            task_sd = dict(original_sd)
            for prefix in selected:
                original_key_a, _, _ = original_pairs[prefix]["A"]
                original_key_b, _, _ = original_pairs[prefix]["B"]
                hns_key_a, hns_a, _ = hns_pairs[prefix]["A"]
                hns_key_b, hns_b, _ = hns_pairs[prefix]["B"]
                if original_key_a != hns_key_a or original_key_b != hns_key_b:
                    raise RuntimeError(f"Adapter key mismatch for {prefix}")
                task_sd[original_key_a] = hns_a
                task_sd[original_key_b] = hns_b
            save_lora_state_dict(str(task_dir), task_sd, original_format)

            candidate_adapter_name = f"tthns_{task_name}"
            model.load_adapter(
                str(task_dir),
                adapter_name=candidate_adapter_name,
                is_trainable=False,
            )
            candidate_probabilities = _score_groups(
                model=model,
                tokenizer=tokenizer,
                adapter_name=candidate_adapter_name,
                groups=groups,
                batch_size=args.selection_batch_size,
                max_seq_len=args.max_seq_len,
                js_weight=args.js_weight,
                compute_gradients=False,
            )
            decision = select_candidate_grouped(
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
                save_lora_state_dict(str(task_dir), original_sd, original_format)

            result = {
                "task": task_name,
                "selection_examples": sum(group.num_examples for group in groups),
                "num_permutations": args.num_permutations,
                "calibration_candidate_modules": calibration_modules,
                "test_time_module_utilities": utilities,
                "proposed_modules": selected,
                "accepted_modules": selected if accepted else [],
                "selected_adapter": decision.selected_name,
                "candidate_scores": {
                    name: score.to_dict() for name, score in decision.scores.items()
                },
                "adapter_dir": str(task_dir),
            }
            (task_dir / "test_time_hns_meta.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            all_results[task_name] = result
            print(
                f"[{task_name}] proposed={len(selected)} accepted={accepted} "
                f"adapter={task_dir}"
            )

            model.set_adapter(reference_adapter_name)
            if hasattr(model, "delete_adapter"):
                model.delete_adapter(candidate_adapter_name)
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        remove_hooks(handles)
        HOOK_CTX.reset()

    summary = {
        "method": "calibration_importance_plus_test_time_hns",
        "base_model": args.base_model,
        "lora_path": str(args.lora_path),
        "calibration_hns_path": str(args.calibration_hns_path),
        "chat_template_mode": args.chat_template_mode,
        "selection_samples": args.selection_samples,
        "num_permutations": args.num_permutations,
        "tasks": all_results,
    }
    summary_path = args.out_root / "test_time_hns_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote: {summary_path}")


if __name__ == "__main__":
    main()
