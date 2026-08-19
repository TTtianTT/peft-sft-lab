from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from glob import glob
import json
import os
from pathlib import Path
import tempfile
from typing import Any, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class SFTExample(TypedDict):
    prompt: list[ChatMessage]
    completion: list[ChatMessage]


def get_writable_datasets_cache_dir() -> str:
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return cache_dir

    fallback = Path(tempfile.gettempdir()) / "hf_datasets_cache"
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)


def load_local_dataset(
    dataset_path: str,
    *,
    task_name: str,
    expected_fields_hint: str,
):
    try:
        from datasets import Dataset, load_dataset
    except Exception as exc:
        raise RuntimeError(f"datasets is required: {exc}") from exc

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise RuntimeError(f"Local {task_name} file not found: {dataset_path}")

    try:
        if dataset_file.is_dir():
            parquet_matches = sorted(glob(str(dataset_file / "data" / "*.parquet"))) or sorted(
                glob(str(dataset_file / "*.parquet"))
            )
            if parquet_matches:
                return load_dataset(
                    "parquet",
                    data_files=parquet_matches,
                    split="train",
                    cache_dir=get_writable_datasets_cache_dir(),
                )

            json_matches = sorted(glob(str(dataset_file / "data" / "*.json"))) or sorted(
                glob(str(dataset_file / "*.json"))
            )
            jsonl_matches = sorted(glob(str(dataset_file / "data" / "*.jsonl"))) or sorted(
                glob(str(dataset_file / "*.jsonl"))
            )
            if json_matches:
                return load_dataset(
                    "json",
                    data_files=json_matches,
                    split="train",
                    cache_dir=get_writable_datasets_cache_dir(),
                )
            if jsonl_matches:
                return load_dataset(
                    "json",
                    data_files=jsonl_matches,
                    split="train",
                    cache_dir=get_writable_datasets_cache_dir(),
                )

            raise RuntimeError(
                f"Unsupported local {task_name} directory layout: {dataset_path}. "
                "Expected parquet/json/jsonl files either at the root or under a data/ subdirectory."
            )

        suffix = dataset_file.suffix.lower()
        if suffix == ".json":
            with dataset_file.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
        elif suffix == ".jsonl":
            with dataset_file.open("r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        elif suffix == ".parquet":
            return load_dataset(
                "parquet",
                data_files=[str(dataset_file)],
                split="train",
                cache_dir=get_writable_datasets_cache_dir(),
            )
        else:
            raise RuntimeError(
                f"Unsupported local {task_name} file extension: {dataset_file.suffix!r}. "
                "Use .parquet, .json, or .jsonl."
            )
        if not isinstance(records, list):
            raise RuntimeError(
                f"Expected local {task_name} file to contain a list of records, got {type(records).__name__}."
            )
        return Dataset.from_list(records)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load local {task_name} json from {dataset_path}: {exc}\n"
            f"Expected local parquet/json/jsonl data with fields like {expected_fields_hint}."
        ) from exc


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


def coerce_chat_messages(messages: Any, *, field_name: str = "messages") -> list[ChatMessage]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of chat messages.")

    normalized: list[ChatMessage] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{field_name}[{idx}] must be a dict with role/content keys.")
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            raise ValueError(f"{field_name}[{idx}] must contain non-empty role and content values.")
        normalized.append(make_chat_message(role=str(role), content=str(content)))
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one chat message.")
    return normalized


def split_chat_messages_for_sft(messages: Any, *, field_name: str = "messages") -> SFTExample:
    normalized = coerce_chat_messages(messages, field_name=field_name)
    if len(normalized) < 2:
        raise ValueError(f"{field_name} must contain at least one prompt message and one assistant completion.")

    if normalized[-1]["role"].strip().lower() != "assistant":
        raise ValueError(f"{field_name} must end with an assistant message.")

    return make_sft_example(prompt=normalized[:-1], completion=[normalized[-1]])


class TaskPlugin(ABC):
    name: str

    @abstractmethod
    def load(self, split: str, dataset_path: str | None = None):
        """Return a `datasets.Dataset` for the given split (usually `train`)."""

    @abstractmethod
    def format_example(self, example: dict[str, Any]) -> SFTExample:
        """Return one structured SFT example with prompt/completion messages."""
