from __future__ import annotations

from typing import Any

from finetune.data.base import TaskPlugin, first_present, format_instruction_response


class MagicoderTask(TaskPlugin):
    name = "code"
    dataset_id = "ise-uiuc/Magicoder-Evol-Instruct-110K"

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
                "Verify the dataset id on Hugging Face."
            ) from exc

    def format_example(self, example: dict[str, Any]) -> str:
        instruction = first_present(example, ["instruction", "prompt", "query", "problem"])
        response = first_present(example, ["response", "output", "answer", "completion"])
        if instruction is None or response is None:
            raise ValueError(
                f"Magicoder example missing required fields. Keys: {sorted(example.keys())}. "
                "Expected something like (instruction,response) or (prompt,output)."
            )
        return format_instruction_response(instruction=instruction, response=response)

