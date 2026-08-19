from __future__ import annotations

from typing import Any

from finetune.data.base import (
    TaskPlugin,
    coerce_chat_messages,
    load_local_dataset,
    make_single_turn_example,
    make_sft_example,
    split_chat_messages_for_sft,
)


class TuluInstructionFollowingTask(TaskPlugin):
    name = "instruction_following"
    dataset_id = "allenai/tulu-3-sft-personas-instruction-following"

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def load(self, split: str, dataset_path: str | None = None):
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(f"datasets is required: {exc}") from exc

        if dataset_path:
            return load_local_dataset(
                dataset_path,
                task_name="instruction-following",
                expected_fields_hint="(messages) or (prompt,completion) or (prompt,response)",
            )

        try:
            return load_dataset(self.dataset_id, split=split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.dataset_id} split={split}: {exc}\n"
                "Verify the dataset id on Hugging Face."
            ) from exc

    def format_example(self, example: dict[str, Any]):
        prompt_messages = example.get("prompt")
        completion_messages = example.get("completion")
        if prompt_messages is not None and completion_messages is not None:
            return make_sft_example(
                prompt=coerce_chat_messages(prompt_messages, field_name="prompt"),
                completion=coerce_chat_messages(completion_messages, field_name="completion"),
            )

        messages = example.get("messages")
        if messages is not None:
            return split_chat_messages_for_sft(messages, field_name="messages")

        prompt_text = self._normalize_text(example.get("prompt"))
        response_text = self._normalize_text(
            example.get("response") or example.get("output") or example.get("answer")
        )
        if prompt_text and response_text:
            return make_single_turn_example(user_content=prompt_text, assistant_content=response_text)

        raise ValueError(
            "Instruction-following example missing supported fields. "
            f"Keys: {sorted(example.keys())}. "
            "Expected either (messages), (prompt/completion), or (prompt,response)."
        )
