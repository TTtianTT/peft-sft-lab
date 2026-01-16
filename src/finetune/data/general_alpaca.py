from __future__ import annotations

from typing import Any

from finetune.data.base import TaskPlugin, format_instruction_response


class AlpacaTask(TaskPlugin):
    name = "alpaca"
    dataset_id = "tatsu-lab/alpaca"

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
        instruction = str(example.get("instruction", "")).strip()
        input_text = str(example.get("input", "")).strip()
        output = str(example.get("output", "")).strip()
        if not instruction or not output:
            raise ValueError(
                f"Alpaca example missing required fields. Keys: {sorted(example.keys())}. "
                "Expected (instruction, output) with optional (input)."
            )
        if input_text:
            instruction = f"{instruction}\n\nInput:\n{input_text}"
        return format_instruction_response(instruction=instruction, response=output)

