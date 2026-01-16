from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def format_instruction_response(*, instruction: str, response: str) -> str:
    instruction = instruction.strip()
    response = response.strip()
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n"


def first_present(example: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        val = example.get(key)
        if val is None:
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        return str(val)
    return None


class TaskPlugin(ABC):
    name: str

    @abstractmethod
    def load(self, split: str):
        """Return a `datasets.Dataset` for the given split (usually `train`)."""

    @abstractmethod
    def format_example(self, example: dict[str, Any]) -> str:
        """Return a single SFT training string (prompt + answer) for one example."""

