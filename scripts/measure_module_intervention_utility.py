#!/usr/bin/env python3
"""Measure exact held-out NLL utility of editing one LoRA module at a time."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.data.chat_sft import ensure_chat_template
from finetune.spectral_edit.calib import build_calib_formatter, make_chat_calib_batch
from finetune.spectral_edit.io import load_lora_state_dict, parse_lora_ab_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample_start", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--chat_template_mode", default="auto")
    return parser.parse_args()


def canonical_module(name: str) -> str:
    position = name.find("layers.")
    if position < 0:
        raise ValueError(name)
    return name[position:]


def collect_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict[str, tuple[str, torch.Tensor, str | None]]]:
    pairs: dict[str, dict[str, tuple[str, torch.Tensor, str | None]]] = {}
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is None:
            continue
        prefix, which, adapter = parsed
        pairs.setdefault(canonical_module(prefix), {})[which] = (key, tensor, adapter)
    return {name: pair for name, pair in pairs.items() if {"A", "B"} <= pair.keys()}


def prepare_examples(args: argparse.Namespace, tokenizer) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
    table = pq.read_table(args.dataset_path)
    permutation = np.random.default_rng(args.seed).permutation(table.num_rows)
    indices = permutation[args.sample_start : args.sample_start + args.samples].tolist()
    examples = table.take(indices).to_pylist()
    formatter, _ = build_calib_formatter(args.dataset_name, None)
    features = []
    kept_indices = []
    for source_index, example in zip(indices, examples):
        try:
            input_ids, attention_mask, labels = make_chat_calib_batch(
                tokenizer, [example], formatter,
                chat_template_mode=args.chat_template_mode,
                max_seq_len=args.max_seq_len,
            )
        except ValueError:
            continue
        features.append({
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "labels": labels[0],
        })
        kept_indices.append(source_index)
    if not features:
        raise RuntimeError("No held-out examples retain supervised tokens")
    return features, kept_indices


def make_batches(features: list[dict[str, torch.Tensor]], batch_size: int, pad_token_id: int):
    batches = []
    for start in range(0, len(features), batch_size):
        group = features[start : start + batch_size]
        batches.append({
            "input_ids": pad_sequence(
                [row["input_ids"] for row in group], batch_first=True, padding_value=pad_token_id
            ),
            "attention_mask": pad_sequence(
                [row["attention_mask"] for row in group], batch_first=True, padding_value=0
            ),
            "labels": pad_sequence(
                [row["labels"] for row in group], batch_first=True, padding_value=-100
            ),
        })
    return batches


def per_example_nll(model, batches: list[dict[str, torch.Tensor]]) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for batch in batches:
            inputs = {key: value.to("cuda") for key, value in batch.items()}
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).logits
            labels = inputs["labels"][:, 1:]
            token_loss = F.cross_entropy(
                logits[:, :-1].float().transpose(1, 2), labels,
                ignore_index=-100, reduction="none",
            )
            valid = labels.ne(-100)
            sample_loss = (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
            values.extend(sample_loss.detach().cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    lora_state, _ = load_lora_state_dict(args.lora_path)
    hns_state, _ = load_lora_state_dict(args.hns_path)
    lora_pairs, hns_pairs = collect_pairs(lora_state), collect_pairs(hns_state)
    if set(lora_pairs) != set(hns_pairs):
        raise RuntimeError("LoRA/HNS module sets differ")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_chat_template(tokenizer, args.base_model)
    features, source_indices = prepare_examples(args, tokenizer)
    batches = make_batches(features, args.batch_size, tokenizer.pad_token_id)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).to("cuda")
    model = PeftModel.from_pretrained(base, args.lora_path, is_trainable=False).to("cuda")
    model.eval()
    model.config.use_cache = False
    module_lookup = dict(model.named_modules())
    targets = {}
    for canonical, pair in lora_pairs.items():
        candidates = [name for name in module_lookup if name.endswith(canonical)]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one loaded module for {canonical}, got {candidates}")
        target = module_lookup[candidates[0]]
        adapter = pair["A"][2] or pair["B"][2] or "default"
        if adapter not in target.lora_A:
            adapter = next(iter(target.lora_A))
        targets[canonical] = (target, adapter)

    baseline = per_example_nll(model, batches)
    rows, example_rows = [], []
    rng = np.random.default_rng(args.seed)
    for index, canonical in enumerate(sorted(lora_pairs), start=1):
        target, adapter = targets[canonical]
        lora_a, lora_b = lora_pairs[canonical]["A"][1], lora_pairs[canonical]["B"][1]
        hns_a, hns_b = hns_pairs[canonical]["A"][1], hns_pairs[canonical]["B"][1]
        with torch.no_grad():
            target.lora_A[adapter].weight.copy_(hns_a.to(target.lora_A[adapter].weight))
            target.lora_B[adapter].weight.copy_(hns_b.to(target.lora_B[adapter].weight))
        edited = per_example_nll(model, batches)
        with torch.no_grad():
            target.lora_A[adapter].weight.copy_(lora_a.to(target.lora_A[adapter].weight))
            target.lora_B[adapter].weight.copy_(lora_b.to(target.lora_B[adapter].weight))

        utility = baseline - edited
        boot_indices = rng.integers(0, len(utility), size=(5000, len(utility)))
        boot = utility[boot_indices].mean(1)
        layer = int(canonical.split(".")[1])
        rows.append({
            "task": args.task, "module": canonical, "layer": layer,
            "module_type": canonical.rsplit(".", 1)[-1], "samples": len(utility),
            "baseline_nll": float(baseline.mean()), "edited_nll": float(edited.mean()),
            "utility": float(utility.mean()),
            "utility_ci_low": float(np.quantile(boot, 0.025)),
            "utility_ci_high": float(np.quantile(boot, 0.975)),
            "fraction_examples_benefit": float(np.mean(utility > 0)),
        })
        for sample_index, value in zip(source_indices, utility):
            example_rows.append({"task": args.task, "module": canonical, "sample_index": sample_index, "utility": value})
        if index % 16 == 0 or index == len(lora_pairs):
            print(f"[Utility] {index}/{len(lora_pairs)} modules", flush=True)

    for filename, values in (("module_utility.tsv", rows), ("module_utility_examples.tsv", example_rows)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(values)
    metadata = {
        "definition": "U_m = mean_example_NLL(LoRA) - mean_example_NLL(only module m copied from observed HNS)",
        "base_model": args.base_model, "lora_path": args.lora_path, "hns_path": args.hns_path,
        "dataset_path": args.dataset_path, "dataset_name": args.dataset_name,
        "sample_start": args.sample_start, "requested_samples": args.samples,
        "used_samples": len(features), "source_indices": source_indices,
        "max_seq_len": args.max_seq_len, "seed": args.seed,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"modules": len(rows), "samples": len(features), "baseline_nll": float(baseline.mean())}, indent=2))


if __name__ == "__main__":
    main()
