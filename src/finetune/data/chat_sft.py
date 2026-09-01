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


def _tokenize_rendered_text_with_offsets(
    tokenizer: Any,
    text: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError) as exc:
        raise RuntimeError(
            "Chat SFT preprocessing requires a fast tokenizer with offset mapping support."
        ) from exc

    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise RuntimeError("Tokenizer output is missing input_ids.")
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError(
            "Tokenizer output is missing offset_mapping; use a fast tokenizer for chat SFT."
        )

    normalized_ids = list(input_ids)
    normalized_offsets = [(int(start), int(end)) for start, end in offsets]
    if len(normalized_ids) != len(normalized_offsets):
        raise RuntimeError(
            "Tokenizer returned different lengths for input_ids and offset_mapping."
        )
    return normalized_ids, normalized_offsets


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

    if not full_text.startswith(prompt_text):
        raise ValueError(
            "Chat template is not text-prefix-preserving for prompt/completion rendering. "
            "The rendered generation prompt must be an exact character prefix of the "
            "rendered conversation."
        )

    full_ids, offsets = _tokenize_rendered_text_with_offsets(tokenizer, full_text)
    prompt_char_len = len(prompt_text)
    labels = [
        token_id if start >= prompt_char_len and end > start else IGNORE_INDEX
        for token_id, (start, end) in zip(full_ids, offsets)
    ]
    prompt_len = next(
        (index for index, label in enumerate(labels) if label != IGNORE_INDEX),
        len(labels),
    )

    input_ids = full_ids if max_seq_len is None else full_ids[:max_seq_len]
    labels = labels[: len(input_ids)]
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
