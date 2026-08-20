from __future__ import annotations

from typing import Any

from finetune.data.base import SFTExample, make_sft_example
from finetune.data.chat_sft import IGNORE_INDEX


def _tokenize_rendered_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise RuntimeError("Tokenizer output is missing input_ids.")
    return list(input_ids)


def _render_plain_messages(messages: list[dict[str, str]]) -> str:
    if not messages:
        return ""

    if len(messages) == 1 and messages[0]["role"].strip().lower() in {"user", "assistant"}:
        return str(messages[0]["content"])

    rendered_parts: list[str] = []
    for message in messages:
        role = str(message["role"]).strip().lower()
        content = str(message["content"])
        if role == "system":
            rendered_parts.append(content)
        elif role == "user":
            rendered_parts.append(content)
        elif role == "assistant":
            rendered_parts.append(content)
        else:
            rendered_parts.append(f"{role}: {content}")
    return "\n\n".join(part for part in rendered_parts if part.strip())


def _join_prompt_and_completion(prompt_text: str, completion_text: str) -> tuple[str, str]:
    if not prompt_text:
        return "", completion_text
    if not completion_text:
        return prompt_text, ""
    if prompt_text.endswith((" ", "\n", "\t")):
        return prompt_text, completion_text
    return prompt_text + "\n\n", completion_text


def preprocess_plain_example(
    *,
    tokenizer: Any,
    example: SFTExample,
    max_seq_len: int | None,
    append_eos: bool = True,
) -> dict[str, Any]:
    structured = make_sft_example(
        prompt=example["prompt"],
        completion=example["completion"],
    )
    prompt_text = _render_plain_messages(structured["prompt"])
    completion_text = _render_plain_messages(structured["completion"])
    prompt_text, completion_text = _join_prompt_and_completion(prompt_text, completion_text)
    if append_eos and getattr(tokenizer, "eos_token", None):
        completion_text = completion_text + tokenizer.eos_token

    prompt_ids = _tokenize_rendered_text(tokenizer, prompt_text)
    completion_ids = _tokenize_rendered_text(tokenizer, completion_text)
    full_ids = prompt_ids + completion_ids

    input_ids = full_ids if max_seq_len is None else full_ids[:max_seq_len]
    labels = ([-100] * len(prompt_ids) + completion_ids)[: len(input_ids)]
    attention_mask = [1] * len(input_ids)
    supervised_token_count = sum(1 for value in labels if value != IGNORE_INDEX)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_length": min(len(prompt_ids), len(input_ids)),
        "supervised_token_count": supervised_token_count,
    }
