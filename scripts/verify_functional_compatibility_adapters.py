#!/usr/bin/env python3
"""Verify every F x C adapter is exactly LoRA/HNS according to its selection metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from build_functional_compatibility_adapters import collect_pairs
from finetune.spectral_edit.io import load_lora_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--hns_path", required=True)
    parser.add_argument("--adapter_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lora_state, _ = load_lora_state_dict(args.lora_path)
    hns_state, _ = load_lora_state_dict(args.hns_path)
    lora, hns = collect_pairs(lora_state), collect_pairs(hns_state)
    if set(lora) != set(hns):
        raise RuntimeError("LoRA/HNS module mismatch")
    manifest = json.loads((Path(args.adapter_root) / "manifest.json").read_text())
    results = []
    for variant in manifest["variants"]:
        label = variant["label"]
        state, _ = load_lora_state_dict(str(Path(args.adapter_root) / label))
        current = collect_pairs(state)
        selected = set(variant["selected_modules"])
        if set(current) != set(lora):
            raise RuntimeError(f"module mismatch for {label}")
        mismatches = []
        for module in lora:
            reference = hns if module in selected else lora
            for which in ("A", "B"):
                if not torch.equal(current[module][which][1], reference[module][which][1]):
                    mismatches.append(f"{module}/{which}")
        if mismatches:
            raise RuntimeError(f"{label}: {len(mismatches)} tensor mismatches; first={mismatches[:3]}")
        results.append({"label": label, "selected_modules": len(selected), "tensor_mismatches": 0})
        print(f"[Verify] {label}: {len(selected)} selected, exact", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "PASS",
        "variants": len(results),
        "modules": len(lora),
        "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
