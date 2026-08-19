from __future__ import annotations

from typing import Any

from finetune.data.base import (
    TaskPlugin,
    coerce_chat_messages,
    first_present,
    load_local_dataset,
    make_sft_example,
    make_single_turn_example,
    split_chat_messages_for_sft,
)


class MagicoderTask(TaskPlugin):
    name = "code"
    dataset_id = "ise-uiuc/Magicoder-Evol-Instruct-110K"

    def load(self, split: str, dataset_path: str | None = None):
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(f"datasets is required: {exc}") from exc

        if dataset_path:
            return load_local_dataset(
                dataset_path,
                task_name="Magicoder",
                expected_fields_hint="(instruction,response) or (messages) or (prompt,completion)",
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
        if isinstance(prompt_messages, list) and isinstance(completion_messages, list):
            return make_sft_example(
                prompt=coerce_chat_messages(prompt_messages, field_name="prompt"),
                completion=coerce_chat_messages(completion_messages, field_name="completion"),
            )

        messages = example.get("messages")
        if messages is None:
            messages = example.get("conversations")
        if messages is not None:
            field_name = "messages" if example.get("messages") is not None else "conversations"
            return split_chat_messages_for_sft(messages, field_name=field_name)

        instruction = first_present(example, ["instruction", "prompt", "query", "problem"])
        response = first_present(example, ["response", "output", "answer", "completion", "solution"])
        if instruction is None or response is None:
            raise ValueError(
                "Magicoder example missing supported fields. "
                f"Keys: {sorted(example.keys())}. "
                "Expected either (instruction,response), (messages)/(conversations), or (prompt,completion)."
            )
        return make_single_turn_example(user_content=instruction, assistant_content=response)
