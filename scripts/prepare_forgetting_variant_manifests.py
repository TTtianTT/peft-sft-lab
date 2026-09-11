#!/usr/bin/env python3
"""Assemble per-base adapter manifests after common-basis controls are built."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter_root", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def safe(value: str) -> str:
    return value.replace("/", "-").replace(".", "p")


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    adapter_root = Path(args.adapter_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_gammas = [float(x) for x in cfg["common_basis_variants"]["global_scalar_gammas"]]
    for base, base_path in cfg["bases"].items():
        variants = []
        checkpoints = [row for row in cfg["checkpoints"] if row["base"] == base]
        for row in checkpoints:
            task = row["train_task"]
            common_root = adapter_root / base / task
            manifest_path = common_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            got_gammas = [float(x) for x in manifest["global_gamma_candidates"]]
            if got_gammas != expected_gammas:
                raise RuntimeError(f"Gamma mismatch for {base}/{task}: {got_gammas}")
            path_by_label = {item["label"]: item["path"] for item in manifest["variants"]}
            additions = [
                ("original_lora", row["lora"], None),
                ("original_hns", row["hns"], None),
                ("common_hns", path_by_label["common_hns"], None),
                ("common_per_module", path_by_label["common_per_module"], None),
            ]
            for method, path, gamma in additions:
                variants.append({
                    "label": f"{task}__{method}",
                    "path": str(Path(path).resolve()),
                    "train_task": task,
                    "method": method,
                    "gamma": gamma,
                })
            for gamma in expected_gammas:
                source_label = f"common_global_{gamma:.2f}".replace(".", "p")
                method = "common_lora" if gamma == 1.0 else f"global_{gamma:.2f}".replace(".", "p")
                variants.append({
                    "label": f"{task}__{method}",
                    "path": str(Path(path_by_label[source_label]).resolve()),
                    "train_task": task,
                    "method": method,
                    "gamma": gamma,
                })
        if len(variants) != 40 or len({row["label"] for row in variants}) != len(variants):
            raise RuntimeError(f"Expected 40 unique variants for {base}, got {len(variants)}")
        for index, row in enumerate(variants, start=1):
            row["adapter_id"] = index
            if not (Path(row["path"]) / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(row["path"])
        result = {
            "status": "complete",
            "base": base,
            "base_model": str(Path(base_path).resolve()),
            "variants": variants,
        }
        destination = output / f"{safe(base)}.json"
        destination.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[Manifest] {base}: {destination} ({len(variants)} variants)")


if __name__ == "__main__":
    main()
