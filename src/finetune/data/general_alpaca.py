from __future__ import annotations

from typing import Any

from finetune.data.base import TaskPlugin, format_instruction_response


class AlpacaTask(TaskPlugin):
    name = "alpaca"
    dataset_id = "tatsu-lab/alpaca"

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

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
        text = example.get("text")
        if isinstance(text, str):
            text = text.strip()
            if text:
                return text
        instruction = self._normalize_text(example.get("instruction"))
        input_text = self._normalize_text(example.get("input"))
        output = self._normalize_text(example.get("output"))
        if not instruction or not output:
            raise ValueError(
                f"Alpaca example missing required fields. Keys: {sorted(example.keys())}. "
                "Expected non-empty text or (instruction, output) with optional (input)."
            )
        if input_text:
            instruction = f"{instruction}\n\nInput:\n{input_text}"
        return format_instruction_response(instruction=instruction, response=output)
