from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class SFTExample(TypedDict):
    prompt: list[ChatMessage]
    completion: list[ChatMessage]


def first_present(example: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        val = example.get(key)
        if val is None:
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        return str(val)
    return None


def make_chat_message(*, role: str, content: str) -> ChatMessage:
    normalized_role = str(role).strip()
    normalized_content = str(content)
    if not normalized_role:
        raise ValueError("Chat message role must be non-empty.")
    if not normalized_content.strip():
        raise ValueError(f"Chat message content for role={normalized_role!r} must be non-empty.")
    return {
        "role": normalized_role,
        "content": normalized_content,
    }


def make_sft_example(
    *,
    prompt: Sequence[ChatMessage],
    completion: Sequence[ChatMessage],
) -> SFTExample:
    prompt_messages = [make_chat_message(role=message["role"], content=message["content"]) for message in prompt]
    completion_messages = [
        make_chat_message(role=message["role"], content=message["content"]) for message in completion
    ]
    if not prompt_messages:
        raise ValueError("SFT example prompt must contain at least one message.")
    if not completion_messages:
        raise ValueError("SFT example completion must contain at least one message.")
    return {
        "prompt": prompt_messages,
        "completion": completion_messages,
    }


def make_single_turn_example(*, user_content: str, assistant_content: str) -> SFTExample:
    return make_sft_example(
        prompt=[make_chat_message(role="user", content=user_content)],
        completion=[make_chat_message(role="assistant", content=assistant_content)],
    )


class TaskPlugin(ABC):
    name: str

    @abstractmethod
    def load(self, split: str, dataset_path: str | None = None):
        """Return a `datasets.Dataset` for the given split (usually `train`)."""

    @abstractmethod
    def format_example(self, example: dict[str, Any]) -> SFTExample:
        """Return one structured SFT example with prompt/completion messages."""
