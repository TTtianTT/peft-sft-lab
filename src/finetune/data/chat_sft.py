from __future__ import annotations

from typing import Any

from finetune.data.base import SFTExample, make_sft_example

IGNORE_INDEX = -100
CHAT_TEMPLATE_MODES = ("auto", "thinking", "non_thinking")


def ensure_chat_template(tokenizer: Any, model_name: str) -> None:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            f"Tokenizer for {model_name!r} does not expose apply_chat_template(). "
            "Use an instruct/chat tokenizer with a built-in chat template."
        )
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            f"Tokenizer for {model_name!r} does not define chat_template. "
            "Refusing to fall back to a hard-coded prompt format."
        )


def chat_template_kwargs(mode: str) -> dict[str, bool]:
    normalized = mode.strip().lower()
    if normalized not in CHAT_TEMPLATE_MODES:
        known = ", ".join(CHAT_TEMPLATE_MODES)
        raise ValueError(f"Unknown chat template mode {mode!r}. Expected one of: {known}.")
    if normalized == "auto":
        return {}
    return {"enable_thinking": normalized == "thinking"}


def _tokenize_rendered_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise RuntimeError("Tokenizer output is missing input_ids.")
    return list(input_ids)


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    matched = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        matched += 1
    return matched


def preprocess_chat_example(
    *,
    tokenizer: Any,
    example: SFTExample,
    max_seq_len: int | None,
    chat_template_mode: str = "auto",
) -> dict[str, Any]:
    structured = make_sft_example(
        prompt=example["prompt"],
        completion=example["completion"],
    )
    prompt_messages = structured["prompt"]
    completion_messages = structured["completion"]
    template_kwargs = chat_template_kwargs(chat_template_mode)

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )

    prompt_ids = _tokenize_rendered_text(tokenizer, prompt_text)
    full_ids = _tokenize_rendered_text(tokenizer, full_text)
    prompt_len = len(prompt_ids)
    prefix_len = _common_prefix_length(prompt_ids, full_ids)
    if prefix_len != prompt_len:
        raise ValueError(
            "Chat template is not prefix-preserving for prompt/completion rendering. "
            f"Matched {prefix_len} of {prompt_len} prompt tokens."
        )

    input_ids = full_ids if max_seq_len is None else full_ids[:max_seq_len]
    labels = ([-100] * prompt_len + full_ids[prompt_len:])[: len(input_ids)]
    attention_mask = [1] * len(input_ids)
    supervised_token_count = sum(1 for value in labels if value != IGNORE_INDEX)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_length": min(prompt_len, len(input_ids)),
        "supervised_token_count": supervised_token_count,
    }


def decode_supervised_labels(tokenizer: Any, labels: list[int]) -> str:
    supervised_ids = [token_id for token_id in labels if token_id != IGNORE_INDEX]
    if not supervised_ids:
        return ""
    return tokenizer.decode(supervised_ids, skip_special_tokens=False)


def format_sft_debug_sample(tokenizer: Any, sample: dict[str, Any]) -> str:
    full_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
    supervised = decode_supervised_labels(tokenizer, sample["labels"])
    return (
        "===== SFT SAMPLE =====\n\n"
        f"Input:\n{full_input}\n\n"
        f"Supervised labels:\n{supervised}\n\n"
        "======================"
    )


class CompletionOnlyDataCollator:
    def __init__(self, tokenizer: Any, label_pad_token_id: int = IGNORE_INDEX) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = self.tokenizer.pad(
            [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
                for feature in features
            ],
            padding=True,
            return_tensors="pt",
        )
        max_length = int(batch["input_ids"].shape[1])
        labels = [
            feature["labels"] + [self.label_pad_token_id] * (max_length - len(feature["labels"]))
            for feature in features
        ]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
