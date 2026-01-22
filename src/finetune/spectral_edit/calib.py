"""Calibration dataset helpers for spectral editing."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence


def _normalize_text_fields(raw_fields: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not raw_fields:
        return None
    if len(raw_fields) == 1 and "," in raw_fields[0]:
        parts = [p.strip() for p in raw_fields[0].split(",") if p.strip()]
        return parts or None
    return list(raw_fields)


def _format_gsm8k_example(ex: dict) -> Tuple[str, str]:
    q = ex["question"]
    a = ex["answer"]
    prompt = f"Question: {q}\nAnswer:"
    return prompt, a


def _format_generic_example(ex: dict, fields: Sequence[str]) -> Tuple[str, str]:
    if len(fields) == 1:
        return "", str(ex.get(fields[0], ""))
    if len(fields) == 2:
        return str(ex.get(fields[0], "")), str(ex.get(fields[1], ""))
    raise ValueError(f"calib_text_fields must have 1 or 2 entries, got {len(fields)}")


def build_calib_formatter(
    calib_dataset: str,
    calib_text_fields: Optional[Sequence[str]],
) -> Tuple[Callable[[dict], Tuple[str, str]], Optional[List[str]]]:
    """
    Return a formatter and normalized text fields.

    If calib_text_fields is provided, uses a generic formatter. Otherwise, uses
    the GSM8K prompt template when calib_dataset == "gsm8k".
    """
    fields = _normalize_text_fields(calib_text_fields)
    if fields:
        return lambda ex: _format_generic_example(ex, fields), fields
    if calib_dataset == "gsm8k":
        return _format_gsm8k_example, None
    raise ValueError(
        "calib_text_fields must be provided for non-gsm8k datasets."
    )


def load_calibration_split(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    cache_dir: Optional[str] = None,
):
    """Load a dataset split for calibration."""
    cfg = dataset_config if dataset_config not in (None, "", "none", "null") else None
    if cfg:
        return load_dataset(dataset_name, cfg, split=split, cache_dir=cache_dir)
    return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


def sample_calibration_examples(
    dataset,
    calib_samples: int,
    calib_shuffle: bool,
    calib_seed: int,
    calib_start: int,
) -> List[dict]:
    """Sample calibration examples with optional shuffling and start offset."""
    total = len(dataset)
    start = max(0, calib_start)
    if start >= total:
        return []

    if calib_shuffle and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=calib_seed)

    n = min(max(0, calib_samples), total - start)
    if hasattr(dataset, "select"):
        subset = dataset.select(range(start, start + n))
        return [subset[i] for i in range(len(subset))]
    return [dataset[i] for i in range(start, start + n)]


def make_calib_batch(
    tokenizer,
    examples: Iterable[dict],
    formatter: Callable[[dict], Tuple[str, str]],
    add_eos: bool = True,
):
    """
    Build teacher-forcing inputs for calibration.

    Returns input_ids, attention_mask, labels tensors.
    Labels have -100 for prompt tokens (not included in loss).
    """
    input_ids_list = []
    labels_list = []

    for ex in examples:
        prompt, answer = formatter(ex)
        prompt = "" if prompt is None else str(prompt)
        answer = "" if answer is None else str(answer)

        if answer:
            if prompt and not prompt.endswith(" "):
                full = prompt + " " + answer
            else:
                full = prompt + answer
        else:
            full = prompt

        if add_eos and tokenizer.eos_token:
            full = full + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids if prompt else []
        full_ids = tokenizer(full, add_special_tokens=False).input_ids

        mask_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * mask_len + full_ids[mask_len:]

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attn_mask = (input_ids != tokenizer.pad_token_id).to(torch.long)
    return input_ids, attn_mask, labels
