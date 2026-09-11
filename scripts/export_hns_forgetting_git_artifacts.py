#!/usr/bin/env python3
"""Create a compact, Git-safe audit export of the HNS forgetting run."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path


MATRICES = {
    "formal_qwen": "formal/Qwen3-8B",
    "formal_llama": "formal/Llama-3.1-8B-Instruct",
    "repeat_llama_core": "formal_repeat/Llama-3.1-8B-Instruct",
}
TASK_FIELDS = {
    "magicoder": ("correct", "result"),
    "metamath": ("correct_strict", "correct_numeric", "prediction_extracted", "gold"),
    "tulu": ("prompt_strict_passed", "prompt_loose_passed", "instruction_id_list", "inst_results"),
    "commonsense": ("correct", "prediction_letter", "gold", "subtask"),
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "run_metadata"
    metadata.mkdir(exist_ok=True)

    for name in ("experiment_config.json", "protocol.md"):
        shutil.copy2(run_root / name, metadata / name)
    for path in sorted((run_root / "variant_manifests").glob("*.json")):
        shutil.copy2(path, metadata / f"variant_manifest__{path.name}")
    for path in sorted((run_root / "adapters").glob("*/*/manifest.json")):
        relative = path.relative_to(run_root / "adapters")
        shutil.copy2(path, metadata / f"adapter_build__{'__'.join(relative.parts[:-1])}.json")
    for path in sorted((run_root / "diagnostic").glob("*/gate.json")):
        shutil.copy2(path, metadata / f"numerical_gate__{path.parent.name}.json")

    inventory = []
    for matrix_name, relative in MATRICES.items():
        matrix = run_root / relative
        shutil.copy2(matrix / "score_manifest.json", metadata / f"score_manifest__{matrix_name}.json")
        shutil.copy2(matrix / "generation_manifest.json", metadata / f"generation_manifest__{matrix_name}.json")
        destination = output / f"per_item_outcomes__{matrix_name}.tsv.gz"
        row_count = 0
        with gzip.open(destination, "wt", encoding="utf-8", newline="", compresslevel=9) as handle:
            columns = ["task", "variant", "id", "primary", "secondary", "prediction", "gold", "subtask", "details_json"]
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            for task, fields in TASK_FIELDS.items():
                task_root = matrix / task
                for variant_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
                    scored = variant_dir / "scored.jsonl"
                    if not scored.is_file():
                        continue
                    for row in read_jsonl(scored):
                        if task == "magicoder":
                            values = (row.get("correct"), None, None, None, None, row.get("result"))
                        elif task == "metamath":
                            values = (row.get("correct_strict"), row.get("correct_numeric"), row.get("prediction_extracted"), row.get("gold"), None, None)
                        elif task == "tulu":
                            compact = [
                                {
                                    "instruction_id": item.get("instruction_id"),
                                    "strict_passed": item.get("strict_passed"),
                                    "loose_passed": item.get("loose_passed"),
                                }
                                for item in row.get("inst_results", [])
                            ]
                            values = (row.get("prompt_strict_passed"), row.get("prompt_loose_passed"), None, None, None, compact)
                        else:
                            values = (row.get("correct"), None, row.get("prediction_letter"), row.get("gold"), row.get("subtask"), None)
                        writer.writerow({
                            "task": task,
                            "variant": variant_dir.name,
                            "id": row.get("id"),
                            "primary": values[0],
                            "secondary": values[1],
                            "prediction": values[2],
                            "gold": values[3],
                            "subtask": values[4],
                            "details_json": json.dumps(values[5], separators=(",", ":")) if values[5] is not None else "",
                        })
                        row_count += 1
        inventory.append({
            "matrix": matrix_name,
            "source": str(matrix),
            "compact_file": destination.name,
            "rows": row_count,
            "compressed_bytes": destination.stat().st_size,
        })

    run_bytes = sum(path.stat().st_size for path in run_root.rglob("*") if path.is_file())
    manifest = {
        "source_run_root": str(run_root),
        "source_total_bytes": run_bytes,
        "matrices": inventory,
        "included": [
            "complete per-item primary outcomes required for paired statistics",
            "secondary outcomes and compact task-specific evaluator details",
            "score and generation manifests",
            "numerical gates, adapter build manifests, variant manifests, config, and protocol",
        ],
        "not_in_git": [
            "derived adapter_model.safetensors files (reconstructible from source checkpoints and scripts)",
            "full generated response text and token IDs (retained at source_run_root)",
            "HumanEval temporary execution files",
        ],
        "reason": "The complete source run is about 16 GB and Git LFS is unavailable in this checkout.",
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Git-safe forgetting-run export\n\n"
        "This directory contains every scored per-item endpoint needed to reproduce the reported paired "
        "statistics, plus all run and adapter manifests. Large reconstructible adapter weights and verbose "
        "generation traces remain in the source run directory recorded in `artifact_manifest.json`.\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
