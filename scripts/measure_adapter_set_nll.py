#!/usr/bin/env python3
"""Measure paired held-out SFT NLL for a small set of complete LoRA adapters."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.data.chat_sft import ensure_chat_template
from finetune.spectral_edit.io import load_lora_state_dict
from measure_module_intervention_utility import (
    collect_pairs,
    make_batches,
    per_example_nll,
    prepare_examples,
)


def parse_adapter(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("adapter must be LABEL=PATH")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter", action="append", type=parse_adapter, default=[])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample_start", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--chat_template_mode", default="auto")
    args = parser.parse_args()
    if not args.adapter:
        parser.error("at least one --adapter is required")
    labels = [label for label, _ in args.adapter]
    if len(set(labels)) != len(labels) or "lora" in labels:
        parser.error("adapter labels must be unique and cannot be 'lora'")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    lora_state, _ = load_lora_state_dict(args.lora_path)
    lora_pairs = collect_pairs(lora_state)
    adapter_pairs = {}
    for label, path in args.adapter:
        state, _ = load_lora_state_dict(path)
        pairs = collect_pairs(state)
        if set(pairs) != set(lora_pairs):
            raise RuntimeError(f"module mismatch for {label}")
        adapter_pairs[label] = pairs

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
    for module, pair in lora_pairs.items():
        candidates = [name for name in module_lookup if name.endswith(module)]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one loaded module for {module}, got {candidates}")
        target = module_lookup[candidates[0]]
        adapter = pair["A"][2] or pair["B"][2] or "default"
        if adapter not in target.lora_A:
            adapter = next(iter(target.lora_A))
        targets[module] = (target, adapter)

    baseline = per_example_nll(model, batches)
    summary_rows = [{
        "label": "lora", "samples": len(baseline), "mean_nll": float(baseline.mean()),
        "utility": 0.0, "utility_ci_low": 0.0, "utility_ci_high": 0.0,
        "fraction_examples_benefit": 0.0,
    }]
    example_rows = [
        {"label": "lora", "sample_index": sample, "nll": nll, "utility": 0.0}
        for sample, nll in zip(source_indices, baseline)
    ]
    rng = np.random.default_rng(args.seed)
    for label, _ in args.adapter:
        pairs = adapter_pairs[label]
        with torch.no_grad():
            for module, pair in pairs.items():
                target, adapter = targets[module]
                target.lora_A[adapter].weight.copy_(pair["A"][1].to(target.lora_A[adapter].weight))
                target.lora_B[adapter].weight.copy_(pair["B"][1].to(target.lora_B[adapter].weight))
        edited = per_example_nll(model, batches)
        utility = baseline - edited
        boot_indices = rng.integers(0, len(utility), size=(10000, len(utility)))
        boot = utility[boot_indices].mean(1)
        summary_rows.append({
            "label": label, "samples": len(utility), "mean_nll": float(edited.mean()),
            "utility": float(utility.mean()),
            "utility_ci_low": float(np.quantile(boot, 0.025)),
            "utility_ci_high": float(np.quantile(boot, 0.975)),
            "fraction_examples_benefit": float(np.mean(utility > 0)),
        })
        example_rows.extend(
            {"label": label, "sample_index": sample, "nll": nll, "utility": gain}
            for sample, nll, gain in zip(source_indices, edited, utility)
        )
        print(f"[Adapter NLL] {label}: U={utility.mean():.8g}", flush=True)

    for filename, rows in (("adapter_nll.tsv", summary_rows), ("adapter_nll_examples.tsv", example_rows)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    (output / "metadata.json").write_text(json.dumps({
        "definition": "utility = per-example NLL(LoRA) - NLL(adapter)",
        "base_model": args.base_model,
        "lora_path": str(Path(args.lora_path).resolve()),
        "adapters": dict(args.adapter),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "dataset_name": args.dataset_name,
        "sample_start": args.sample_start,
        "requested_samples": args.samples,
        "used_samples": len(features),
        "source_indices": source_indices,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
