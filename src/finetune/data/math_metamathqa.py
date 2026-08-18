from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finetune.data.base import TaskPlugin, first_present, make_single_turn_example


class MetaMathQATask(TaskPlugin):
    name = "math"
    dataset_id = "meta-math/MetaMathQA"

    def load(self, split: str, dataset_path: str | None = None):
        try:
            from datasets import Dataset, load_dataset
        except Exception as exc:
            raise RuntimeError(f"datasets is required: {exc}") from exc

        if dataset_path:
            dataset_file = Path(dataset_path)
            if not dataset_file.exists():
                raise RuntimeError(f"Local MetaMathQA file not found: {dataset_path}")

            try:
                suffix = dataset_file.suffix.lower()
                if suffix == ".json":
                    with dataset_file.open("r", encoding="utf-8") as handle:
                        records = json.load(handle)
                elif suffix == ".jsonl":
                    with dataset_file.open("r", encoding="utf-8") as handle:
                        records = [json.loads(line) for line in handle if line.strip()]
                else:
                    raise RuntimeError(
                        f"Unsupported local MetaMathQA file extension: {dataset_file.suffix!r}. "
                        "Use .json or .jsonl."
                    )
                if not isinstance(records, list):
                    raise RuntimeError(
                        f"Expected local MetaMathQA file to contain a list of records, got {type(records).__name__}."
                    )
                return Dataset.from_list(records)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load local MetaMathQA json from {dataset_path}: {exc}\n"
                    "Expected a JSON/JSONL file with fields like (query,response)."
                ) from exc

        try:
            return load_dataset(self.dataset_id, split=split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.dataset_id} split={split}: {exc}\n"
                "Try: pip install -U datasets\n"
                "Or verify the dataset id on Hugging Face."
            ) from exc

    def format_example(self, example: dict[str, Any]):
        instruction = first_present(
            example,
            ["query", "original_question", "question", "instruction", "prompt"],
        )
        response = first_present(example, ["response", "answer", "output", "solution"])
        if instruction is None or response is None:
            raise ValueError(
                f"MetaMathQA example missing required fields. Keys: {sorted(example.keys())}. "
                "Expected something like (query,response)."
            )
        return make_single_turn_example(user_content=instruction, assistant_content=response)
