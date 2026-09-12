#!/usr/bin/env python3
"""Measure PIQA reward-margin sensitivity and LoRA-trajectory energy per LoRA SVD gate.

The signed signal is the derivative of the correct-vs-incorrect first-token
margin with respect to a multiplicative singular-value gate s_j at s=1.
No adapter is edited in this script; finite edits are evaluated separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_commonsense_8tasks import _instruction  # noqa: E402
from finetune.eval.generation import render_chat_prompt  # noqa: E402
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
    parse_lora_ab_key,
)
from finetune.spectral_edit.mechanism import align_edited_spectrum_to_reference  # noqa: E402
from finetune.spectral_edit.svd import lowrank_svd_from_ba  # noqa: E402


LETTERS = ("A", "B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--piqa_predictions", required=True)
    parser.add_argument("--sft_gradient_meta", required=True)
    parser.add_argument("--commonsense_train", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--signal_samples", type=int, default=128)
    parser.add_argument("--dose_samples", type=int, default=128)
    parser.add_argument("--validation_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def canonical(prefix: str) -> str:
    position = prefix.find("layers.")
    if position < 0:
        raise ValueError(prefix)
    return prefix[position:]


def collect_pairs(path: str) -> tuple[dict[str, dict[str, tuple[str, torch.Tensor, str | None]]], dict, str]:
    state, weight_format = load_lora_state_dict(path)
    config = load_adapter_config(path)
    pairs: dict[str, dict[str, tuple[str, torch.Tensor, str | None]]] = defaultdict(dict)
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is None:
            continue
        prefix, which, adapter = parsed
        pairs[canonical(prefix)][which] = (key, tensor, adapter)
    complete = {name: pair for name, pair in pairs.items() if {"A", "B"} <= pair.keys()}
    return complete, config, weight_format


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(len(row.get("choices", [])) != 2 for row in rows):
        raise RuntimeError("PIQA predictions are empty or non-binary")
    return rows


def split_records(rows: list[dict], args: argparse.Namespace) -> dict[str, list[int]]:
    total = args.signal_samples + args.dose_samples + args.validation_samples
    if total > len(rows):
        raise ValueError(f"requested {total} PIQA rows, only {len(rows)} available")
    permutation = np.random.default_rng(args.seed).permutation(len(rows))[:total]
    a = args.signal_samples
    b = a + args.dose_samples
    return {
        "signal": permutation[:a].tolist(),
        "dose": permutation[a:b].tolist(),
        "validation": permutation[b:].tolist(),
    }


def bootstrap_interval(values: np.ndarray, *, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_records(Path(args.piqa_predictions))
    splits = split_records(rows, args)

    # The evaluation IDs are unique, and the official validation questions
    # should not occur in Commonsense170K train. Check exact normalized text as
    # a concrete guard without claiming fuzzy semantic deduplication.
    require_ids = [rows[index]["id"] for values in splits.values() for index in values]
    if len(set(require_ids)) != len(require_ids):
        raise RuntimeError("duplicate PIQA IDs across splits")
    train = pq.read_table(args.commonsense_train, columns=["instruction"]).column("instruction").to_pylist()
    train_text = {normalize_text(str(value)) for value in train}
    exact_overlap = [
        rows[index]["id"]
        for values in splits.values()
        for index in values
        if normalize_text(rows[index]["question"]) in train_text
    ]
    if exact_overlap:
        raise RuntimeError(f"exact normalized PIQA/train overlap: {exact_overlap[:5]}")

    lora_pairs, adapter_config, _ = collect_pairs(args.lora_path)
    hns_pairs, _, _ = collect_pairs(args.hns_path)
    if set(lora_pairs) != set(hns_pairs):
        raise RuntimeError("LoRA/HNS module sets differ")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{letter!r} is not one token: {ids}")
        letter_ids.append(ids[0])

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(base, args.lora_path, is_trainable=True).to("cuda")
    model.eval()
    model.config.use_cache = False
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name)

    module_lookup = dict(model.named_modules())
    specs: dict[str, ModuleSpec] = {}
    hns_sigmas: dict[str, torch.Tensor] = {}
    for module_name, pair in lora_pairs.items():
        candidates = [name for name in module_lookup if name.endswith(module_name)]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one loaded module for {module_name}, got {candidates}")
        a = pair["A"][1].to("cuda")
        b = pair["B"][1].to("cuda")
        U, sigma, Vh, V = lowrank_svd_from_ba(b, a)
        hns_sigma, _ = align_edited_spectrum_to_reference(
            U,
            Vh,
            hns_pairs[module_name]["B"][1].to("cuda"),
            hns_pairs[module_name]["A"][1].to("cuda"),
        )
        adapter = pair["A"][2] or pair["B"][2]
        specs[module_name] = ModuleSpec(
            module_prefix=module_name,
            module=module_lookup[candidates[0]],
            U=U.detach(),
            V=V.detach(),
            Vh=Vh.detach(),
            sigma0=sigma.detach().cpu(),
            scaling=get_scaling_for_module(adapter_config, module_name),
            adapter=adapter,
        )
        hns_sigmas[module_name] = hns_sigma.detach().cpu()

    signal_rows = [rows[index] for index in splits["signal"]]
    prompts = [
        render_chat_prompt(
            tokenizer=tokenizer,
            base_model=args.base_model,
            user_content=row["instruction"],
            chat_template_mode="non_thinking",
        )
        for row in signal_rows
    ]
    gradient_batches: dict[str, list[torch.Tensor]] = {name: [] for name in specs}
    energy_sum = {name: torch.zeros_like(spec.sigma0, dtype=torch.float64) for name, spec in specs.items()}
    energy_count = {name: 0 for name in specs}
    agreement_rows = []
    handles = register_sigma_hooks(specs, capture_energy=True)
    try:
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            batch_rows = signal_rows[start : start + args.batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to("cuda")
            attention_mask = encoded["attention_mask"].to("cuda")
            HOOK_CTX.reset()
            HOOK_CTX.attn_mask = attention_mask
            model.zero_grad(set_to_none=True)
            result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            last_position = attention_mask.sum(dim=1) - 1
            last_logits = result.logits[torch.arange(len(batch_rows), device="cuda"), last_position]
            choice_logits = last_logits[:, letter_ids].float()
            gold = torch.tensor([LETTERS.index(row["gold"]) for row in batch_rows], device="cuda")
            wrong = 1 - gold
            margin = choice_logits.gather(1, gold[:, None]).squeeze(1) - choice_logits.gather(
                1, wrong[:, None]
            ).squeeze(1)
            margin.mean().backward()

            missing = set(specs) - set(HOOK_CTX.gsum)
            if missing:
                raise RuntimeError(f"missing gradients for {len(missing)} modules; first={sorted(missing)[0]}")
            for name in specs:
                gradient_batches[name].append(HOOK_CTX.gsum[name].clone())
                energy_sum[name] += HOOK_CTX.energy_sum[name].to(torch.float64)
                energy_count[name] += HOOK_CTX.energy_count[name]

            restricted = choice_logits.argmax(dim=1).cpu().tolist()
            global_ids = last_logits.argmax(dim=1).cpu().tolist()
            for row, value, restricted_index, global_id in zip(
                batch_rows, margin.detach().cpu().tolist(), restricted, global_ids
            ):
                original = row.get("prediction_letter", "")
                agreement_rows.append(
                    {
                        "id": row["id"],
                        "gold": row["gold"],
                        "original_parser_prediction": original,
                        "restricted_first_token_prediction": LETTERS[restricted_index],
                        "restricted_agrees_with_parser": original == LETTERS[restricted_index],
                        "global_first_token_id": global_id,
                        "global_first_token_text": tokenizer.decode([global_id]),
                        "global_first_token_is_choice": global_id in letter_ids,
                        "correct_minus_incorrect_margin": value,
                    }
                )
            print(f"[Signal] {min(start + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)
    finally:
        remove_hooks(handles)
        HOOK_CTX.reset()

    old_sft = json.loads(Path(args.sft_gradient_meta).read_text())["module_selection"]
    old_sft = {canonical(name): value for name, value in old_sft.items()}
    rng = np.random.default_rng(args.seed)
    direction_rows = []
    batch_rows_out = []
    for name, spec in specs.items():
        batches = torch.stack(gradient_batches[name]).double().numpy()
        sigma = spec.sigma0.double().numpy()
        sigma_hns = hns_sigmas[name].double().numpy()
        projection_energy = energy_sum[name].numpy() / max(energy_count[name], 1)
        functional_energy = projection_energy * (sigma * spec.scaling) ** 2
        shares = functional_energy / max(functional_energy.sum(), 1e-30)
        for direction in range(len(sigma)):
            gate_gradient_batches = batches[:, direction] * sigma[direction]
            delta_sigma = sigma_hns[direction] - sigma[direction]
            hns_gain_batches = batches[:, direction] * delta_sigma
            gate_ci = bootstrap_interval(gate_gradient_batches, draws=args.bootstrap, rng=rng)
            gain_ci = bootstrap_interval(hns_gain_batches, draws=args.bootstrap, rng=rng)
            suppressed = bool(delta_sigma < -max(1e-8, 1e-6 * sigma[direction]))
            direction_rows.append(
                {
                    "module": name,
                    "layer": int(name.split(".")[1]),
                    "module_type": name.rsplit(".", 1)[-1],
                    "direction": direction + 1,
                    "sigma_lora": sigma[direction],
                    "sigma_hns": sigma_hns[direction],
                    "hns_gate": sigma_hns[direction] / max(sigma[direction], 1e-30),
                    "hns_suppression_fraction": 1.0 - sigma_hns[direction] / max(sigma[direction], 1e-30),
                    "hns_suppressed": suppressed,
                    "lora_trajectory_functional_energy": functional_energy[direction],
                    "lora_trajectory_functional_share": shares[direction],
                    "gate_reward_gradient": gate_gradient_batches.mean(),
                    "gate_reward_gradient_ci_low": gate_ci[0],
                    "gate_reward_gradient_ci_high": gate_ci[1],
                    "predicted_hns_margin_gain": hns_gain_batches.mean(),
                    "predicted_hns_margin_gain_ci_low": gain_ci[0],
                    "predicted_hns_margin_gain_ci_high": gain_ci[1],
                    "old_module_sft_compatibility": old_sft[name]["compatibility"],
                    "old_module_sft_importance": old_sft[name]["importance"],
                }
            )
            for batch_index, (gate_value, gain_value) in enumerate(
                zip(gate_gradient_batches, hns_gain_batches)
            ):
                batch_rows_out.append(
                    {
                        "module": name,
                        "direction": direction + 1,
                        "batch": batch_index,
                        "gate_reward_gradient": gate_value,
                        "predicted_hns_margin_gain": gain_value,
                    }
                )

    def write_tsv(path: Path, values: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(values)

    write_tsv(output / "direction_signals.tsv", direction_rows)
    write_tsv(output / "direction_signal_batches.tsv", batch_rows_out)
    write_tsv(output / "first_token_agreement.tsv", agreement_rows)
    split_payload = {
        "seed": args.seed,
        "source": str(Path(args.piqa_predictions).resolve()),
        "total_source_rows": len(rows),
        "exact_normalized_train_question_overlap": len(exact_overlap),
        "splits": {
            name: {
                "positions": positions,
                "ids": [rows[index]["id"] for index in positions],
            }
            for name, positions in splits.items()
        },
    }
    (output / "split_manifest.json").write_text(json.dumps(split_payload, indent=2) + "\n")
    metadata = {
        "definition": "N_j = d(correct-vs-incorrect first-token margin)/d multiplicative singular gate s_j at s=1",
        "finite_path_prediction": "predicted_hns_margin_gain = dJ/dsigma_j * (sigma_hns_j-sigma_j)",
        "functional_energy": "(LoRA scaling*sigma_j)^2 E[(v_j^T h)^2] on actual LoRA PIQA prompt trajectories",
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "hns_path": args.hns_path,
        "signal_samples": len(signal_rows),
        "batch_size": args.batch_size,
        "letter_token_ids": dict(zip(LETTERS, letter_ids)),
        "restricted_first_token_parser_agreement": float(
            np.mean([row["restricted_agrees_with_parser"] for row in agreement_rows])
        ),
        "global_first_token_choice_fraction": float(
            np.mean([row["global_first_token_is_choice"] for row in agreement_rows])
        ),
        "modules": len(specs),
        "directions": len(direction_rows),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
