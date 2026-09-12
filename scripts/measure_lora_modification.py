#!/usr/bin/env python3
"""Measure per-layer ||Delta W h|| / ||W h|| on real calibration examples."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.spectral_edit.io import (
    get_scaling_for_module,
    load_adapter_config,
    load_lora_state_dict,
    parse_lora_ab_key,
)
from finetune.spectral_edit.mechanism import align_edited_spectrum_to_reference
from finetune.spectral_edit.svd import lowrank_svd_from_ba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_kind", choices=("magicoder", "metamath", "commonsense", "tulu"), required=True)
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional singular-spectrum control adapter; may be repeated.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def collect_factors(path: str) -> tuple[dict[str, dict[str, torch.Tensor]], dict]:
    state, _ = load_lora_state_dict(path)
    config = load_adapter_config(path)
    result: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in state.items():
        parsed = parse_lora_ab_key(key)
        if parsed is not None:
            prefix, which, _ = parsed
            result[prefix][which] = tensor
    complete = {prefix: item for prefix, item in result.items() if {"A", "B"} <= item.keys()}
    return complete, config


def canonical_name(prefix: str) -> str:
    marker = ".model.layers."
    if marker in prefix:
        return "layers." + prefix.split(marker, 1)[1]
    marker = "model.layers."
    if marker in prefix:
        return "layers." + prefix.split(marker, 1)[1]
    raise ValueError(f"Cannot map adapter module to transformer backbone: {prefix}")


def messages(example: dict, kind: str) -> list[dict[str, str]]:
    if kind == "magicoder":
        user, assistant = example["instruction"], example["response"]
    elif kind == "metamath":
        user, assistant = example["query"], example["response"]
    elif kind == "commonsense":
        user = str(example.get("instruction", ""))
        extra = str(example.get("input", "") or "")
        if extra.strip():
            user += "\n\n" + extra
        assistant = str(example.get("output", example.get("answer", "")))
    else:
        raw_messages = example.get("messages")
        if not raw_messages:
            raise ValueError(f"Tulu example is missing messages: keys={sorted(example)}")
        return [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in raw_messages
        ]
    return [{"role": "user", "content": str(user)}, {"role": "assistant", "content": str(assistant)}]


def normalize_token_ids(value) -> list[int]:
    """Normalize tokenizer list/Encoding/BatchEncoding outputs."""
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "ids"):
        value = value.ids
    if value and hasattr(value[0], "ids"):
        if len(value) != 1:
            raise ValueError(f"Expected one encoded sequence, got {len(value)}")
        value = value[0].ids
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"Expected one token-id row, got {len(value)}")
        value = value[0]
    return [int(token_id) for token_id in value]


def render(tokenizer, example: dict, kind: str, max_seq_len: int) -> list[int]:
    kwargs = dict(tokenize=True, add_generation_prompt=False)
    try:
        ids = tokenizer.apply_chat_template(messages(example, kind), enable_thinking=False, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages(example, kind), **kwargs)
    # Newer tokenizers may expose ``Encoding``/``BatchEncoding`` here instead
    # of a bare list. Normalize all supported return types before truncation.
    return normalize_token_ids(ids)[:max_seq_len]


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "fraction_gt_0p10": float(np.mean(values > 0.10)),
        "fraction_gt_0p25": float(np.mean(values > 0.25)),
        "fraction_gt_0p50": float(np.mean(values > 0.50)),
        "fraction_gt_1p00": float(np.mean(values > 1.00)),
    }


def write_layer_svg(path: Path, rows: list[dict]) -> None:
    width, height = 900, 470
    left, right, top, bottom = 75, 30, 45, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(max(float(row["lora_p95"]), float(row["hns_p95"])) for row in rows) * 1.08
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:18px;font-weight:600}.tick{font-size:11px}.label{font-size:13px}</style>',
        f'<text x="{width/2}" y="25" text-anchor="middle" class="title">Per-layer p95 of ||Delta W h|| / ||W h|| (256 samples)</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_h * (1 - tick / 5)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.3f}</text>')
    for label, key, color in (("LoRA", "lora_p95", "#dc2626"), ("HNS", "hns_p95", "#2563eb")):
        points = []
        for index, row in enumerate(rows):
            x = left + plot_w * index / max(1, len(rows) - 1)
            y = top + plot_h * (1 - float(row[key]) / maximum)
            points.append(f"{x:.1f},{y:.1f}")
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
    for index, row in enumerate(rows):
        if index % 4 == 0 or index == len(rows) - 1:
            x = left + plot_w * index / max(1, len(rows) - 1)
            svg.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle" class="tick">{row["layer"]}</text>')
    svg.extend([
        f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" class="label">transformer layer</text>',
        f'<line x1="{left+15}" y1="{top+17}" x2="{left+45}" y2="{top+17}" stroke="#dc2626" stroke-width="3"/><text x="{left+52}" y="{top+22}" class="label">LoRA</text>',
        f'<line x1="{left+120}" y1="{top+17}" x2="{left+150}" y2="{top+17}" stroke="#2563eb" stroke-width="3"/><text x="{left+157}" y="{top+22}" class="label">HNS</text>',
        '</svg>',
    ])
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    lora, lora_cfg = collect_factors(args.lora_path)
    hns, hns_cfg = collect_factors(args.hns_path)
    if set(lora) != set(hns):
        raise RuntimeError("LoRA/HNS module sets differ")
    adapter_sources: dict[str, tuple[dict[str, dict[str, torch.Tensor]], dict]] = {
        "lora": (lora, lora_cfg),
        "hns": (hns, hns_cfg),
    }
    for spec in args.control:
        if "=" not in spec:
            raise ValueError(f"--control must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip().lower().replace("-", "_")
        if not name or name in adapter_sources:
            raise ValueError(f"Invalid or duplicate control name: {name!r}")
        source, cfg = collect_factors(path)
        if set(source) != set(lora):
            raise RuntimeError(f"{name}: module set differs from LoRA")
        adapter_sources[name] = (source, cfg)
    adapter_names = tuple(adapter_sources)

    table = pq.read_table(args.dataset_path)
    rng = np.random.default_rng(args.seed)
    selected = rng.permutation(table.num_rows)[: args.samples].tolist()
    examples = table.take(selected).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [render(tokenizer, ex, args.dataset_kind, args.max_seq_len) for ex in examples]

    print(f"[Model] loading {args.base_model} dtype={args.dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    backbone = model.model
    modules = dict(backbone.named_modules())

    factors: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor, float]]] = {}
    spectral: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}
    prefix_by_name = {}
    for prefix in sorted(lora):
        name = canonical_name(prefix)
        if name not in modules:
            raise KeyError(f"Base module {name!r} (from {prefix!r}) not found")
        prefix_by_name[name] = prefix
        factors[name] = {}
        for adapter, (source, cfg) in adapter_sources.items():
            factors[name][adapter] = (
                source[prefix]["A"].to(device="cuda", dtype=dtype),
                source[prefix]["B"].to(device="cuda", dtype=dtype),
                get_scaling_for_module(cfg, prefix),
            )
        u, sigma_lora, vh, _ = lowrank_svd_from_ba(
            lora[prefix]["B"].to("cuda"), lora[prefix]["A"].to("cuda")
        )
        aligned_spectra = {"lora": sigma_lora.to(dtype=torch.float32)}
        for adapter, (source, _) in adapter_sources.items():
            if adapter == "lora":
                continue
            sigma_aligned, _ = align_edited_spectrum_to_reference(
                u,
                vh,
                source[prefix]["B"].to("cuda"),
                source[prefix]["A"].to("cuda"),
            )
            aligned_spectra[adapter] = sigma_aligned.to(dtype=torch.float32)
        spectral[name] = (
            vh.to(dtype=dtype),
            aligned_spectra,
        )

    # Each entry stores per-sample squared numerator/denominator. Layer-level
    # ratios can therefore aggregate modules before taking the norm ratio.
    records: dict[str, dict[str, dict[str, list[float]]]] = {
        adapter: {name: {"num2": [], "den2": []} for name in factors}
        for adapter in adapter_names
    }
    direction_energy: dict[str, dict[str, torch.Tensor]] = {
        adapter: {
            name: torch.zeros(spectral[name][1][adapter].numel(), dtype=torch.float64)
            for name in factors
        }
        for adapter in adapter_names
    }
    current_mask: torch.Tensor | None = None

    def make_hook(name: str):
        def hook(_module, inputs, output):
            assert current_mask is not None
            hidden = inputs[0]
            base_output = output[0] if isinstance(output, tuple) else output
            mask = current_mask.to(device=hidden.device, dtype=torch.float32).unsqueeze(-1)
            den2 = (base_output.float().square() * mask).sum(dim=(1, 2)).clamp_min(1e-24)
            vh, spectra = spectral[name]
            coordinate_energy = (
                F.linear(hidden.to(dtype=vh.dtype), vh).float().square() * mask
            ).sum(dim=(0, 1))
            for adapter in adapter_names:
                a, b, scale = factors[name][adapter]
                delta = F.linear(F.linear(hidden.to(dtype=a.dtype), a), b) * scale
                num2 = (delta.float().square() * mask).sum(dim=(1, 2))
                records[adapter][name]["num2"].extend(num2.detach().cpu().tolist())
                records[adapter][name]["den2"].extend(den2.detach().cpu().tolist())
                scaled_sigma = spectra[adapter] * factors[name][adapter][2]
                direction_energy[adapter][name] += (
                    coordinate_energy * scaled_sigma.square()
                ).detach().to(device="cpu", dtype=torch.float64)
        return hook

    handles = [modules[name].register_forward_hook(make_hook(name)) for name in factors]
    try:
        with torch.inference_mode():
            for start in range(0, len(encoded), args.batch_size):
                batch = encoded[start : start + args.batch_size]
                width = max(len(ids) for ids in batch)
                input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
                attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, ids in enumerate(batch):
                    input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attention_mask[row, : len(ids)] = 1
                current_mask = attention_mask.to("cuda")
                backbone(
                    input_ids=input_ids.to("cuda"),
                    attention_mask=current_mask,
                    use_cache=False,
                    return_dict=True,
                )
                done = min(start + args.batch_size, len(encoded))
                if done % 32 == 0 or done == len(encoded):
                    print(f"[Forward] {done}/{len(encoded)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    module_rows, layer_rows, direction_rows = [], [], []
    layer_members: dict[int, list[str]] = defaultdict(list)
    for name in factors:
        layer = int(name.split(".")[1])
        layer_members[layer].append(name)
        arrays = {}
        for adapter in adapter_names:
            num2 = np.asarray(records[adapter][name]["num2"], dtype=np.float64)
            den2 = np.asarray(records[adapter][name]["den2"], dtype=np.float64)
            arrays[adapter] = np.sqrt(num2 / den2)
        row = {
            "module": name,
            "layer": layer,
            "module_type": name.rsplit(".", 1)[-1],
            **{
                f"{adapter}_{key}": value
                for adapter in adapter_names
                for key, value in summary(arrays[adapter]).items()
            },
            "paired_mean_delta": float(np.mean(arrays["hns"] - arrays["lora"])),
            "paired_fraction_hns_lower": float(np.mean(arrays["hns"] < arrays["lora"])),
        }
        for adapter in adapter_names:
            if adapter != "lora":
                row[f"{adapter}_to_lora_p99_ratio"] = float(
                    np.quantile(arrays[adapter], 0.99) / np.quantile(arrays["lora"], 0.99)
                )
        module_rows.append(row)
        for adapter in adapter_names:
            energies = direction_energy[adapter][name].numpy()
            shares = energies / max(float(energies.sum()), 1e-24)
            for index, (energy, share) in enumerate(zip(energies, shares), start=1):
                direction_rows.append({
                    "module": name,
                    "layer": layer,
                    "module_type": name.rsplit(".", 1)[-1],
                    "adapter": adapter,
                    "direction": index,
                    "response_energy": float(energy),
                    "response_energy_share": float(share),
                })

    for layer, names in sorted(layer_members.items()):
        arrays = {}
        for adapter in adapter_names:
            num2 = sum(np.asarray(records[adapter][name]["num2"], dtype=np.float64) for name in names)
            den2 = sum(np.asarray(records[adapter][name]["den2"], dtype=np.float64) for name in names)
            arrays[adapter] = np.sqrt(num2 / den2)
        row = {
            "layer": layer,
            "modules": len(names),
            **{
                f"{adapter}_{key}": value
                for adapter in adapter_names
                for key, value in summary(arrays[adapter]).items()
            },
            "paired_mean_delta": float(np.mean(arrays["hns"] - arrays["lora"])),
            "paired_fraction_hns_lower": float(np.mean(arrays["hns"] < arrays["lora"])),
        }
        for adapter in adapter_names:
            if adapter != "lora":
                row[f"{adapter}_to_lora_p99_ratio"] = float(
                    np.quantile(arrays[adapter], 0.99) / np.quantile(arrays["lora"], 0.99)
                )
        layer_rows.append(row)

    for filename, rows in (
        ("module_response.csv", module_rows),
        ("layer_response.csv", layer_rows),
        ("direction_response.csv", direction_rows),
    ):
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    direction_summary = {}
    for adapter in adapter_names:
        shares_by_module = []
        for name in factors:
            energies = direction_energy[adapter][name].numpy()
            shares_by_module.append(energies / max(float(energies.sum()), 1e-24))
        shares = np.stack(shares_by_module)
        safe = np.where(shares > 0.0, shares, 1.0)
        effective_ranks = np.exp(-np.sum(np.where(shares > 0.0, shares * np.log(safe), 0.0), axis=1))
        direction_summary[adapter] = {
            "mean_top1_response_energy_share": float(np.mean(shares[:, 0])),
            "mean_top4_response_energy_share": float(np.mean(np.sum(shares[:, :4], axis=1))),
            "median_top1_response_energy_share": float(np.median(shares[:, 0])),
            "median_top4_response_energy_share": float(np.median(np.sum(shares[:, :4], axis=1))),
            "mean_functional_effective_rank": float(np.mean(effective_ranks)),
            "median_functional_effective_rank": float(np.median(effective_ranks)),
            "mean_direction_response_energy_share": np.mean(shares, axis=0).tolist(),
        }
    result = {
        "definition": "Per sample and module, Frobenius norm over non-padding tokens: ||Delta W h||_F / ||W h||_F. Layer rows concatenate all adapted projections in that transformer layer before taking the ratio.",
        "trajectory": "Paired frozen-base trajectory: LoRA and HNS deltas are evaluated on exactly the same real hidden states from the pretrained base model.",
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "hns_path": args.hns_path,
        "dataset_path": args.dataset_path,
        "dataset_kind": args.dataset_kind,
        "samples": len(encoded),
        "sample_indices": selected,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "adapters": list(adapter_names),
        "layer_summary": layer_rows,
        "direction_summary": direction_summary,
    }
    (out / "activation_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_layer_svg(out / "layer_p95.svg", layer_rows)
    print(f"[Done] {out / 'activation_analysis.json'}")


if __name__ == "__main__":
    main()
