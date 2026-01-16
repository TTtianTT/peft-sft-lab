from __future__ import annotations

from typing import Any

from finetune.data.base import TaskPlugin, first_present, format_instruction_response


class MetaMathQATask(TaskPlugin):
    name = "math"
    dataset_id = "meta-math/MetaMathQA"

    def load(self, split: str):
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(f"datasets is required: {exc}") from exc

        try:
            return load_dataset(self.dataset_id, split=split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.dataset_id} split={split}: {exc}\n"
                "Try: pip install -U datasets\n"
                "Or verify the dataset id on Hugging Face."
            ) from exc

    def format_example(self, example: dict[str, Any]) -> str:
        instruction = first_present(example, ["query", "question", "instruction", "prompt"])
        response = first_present(example, ["response", "answer", "output", "solution"])
        if instruction is None or response is None:
            raise ValueError(
                f"MetaMathQA example missing required fields. Keys: {sorted(example.keys())}. "
                "Expected something like (query,response)."
            )
        return format_instruction_response(instruction=instruction, response=response)

