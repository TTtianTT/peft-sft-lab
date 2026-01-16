from __future__ import annotations

from typing import Any

from finetune.data.base import TaskPlugin, format_instruction_response


def _choices_to_map(choices: Any) -> dict[str, str]:
    if isinstance(choices, dict):
        labels = choices.get("label")
        texts = choices.get("text")
        if isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts):
            return {str(l): str(t) for l, t in zip(labels, texts)}
    if isinstance(choices, list):
        out: dict[str, str] = {}
        for item in choices:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            text = item.get("text")
            if label is None or text is None:
                continue
            out[str(label)] = str(text)
        if out:
            return out
    raise ValueError(f"Unrecognized choices format: {type(choices)}")


class CommonsenseQATask(TaskPlugin):
    name = "csqa"
    dataset_id = "tau/commonsense_qa"

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
        question = str(example.get("question", "")).strip()
        answer = str(example.get("answerKey", "")).strip().upper()
        if not question or not answer:
            raise ValueError(
                f"CSQA example missing required fields. Keys: {sorted(example.keys())}. "
                "Expected (question, choices, answerKey)."
            )
        if answer not in {"A", "B", "C", "D", "E"}:
            raise ValueError(f"CSQA answerKey must be one of A/B/C/D/E, got {answer!r}.")

        choices_map = _choices_to_map(example.get("choices"))
        lines = []
        for label in ["A", "B", "C", "D", "E"]:
            if label in choices_map:
                lines.append(f"{label}. {choices_map[label]}")
        if len(lines) < 2:
            raise ValueError(f"CSQA choices malformed. Keys: {sorted(example.keys())}.")

        instruction = (
            f"Question:\n{question}\n\nChoices:\n" + "\n".join(lines) + "\n\n"
            "Answer with a single letter: A, B, C, D, or E."
        )
        return format_instruction_response(instruction=instruction, response=answer)

