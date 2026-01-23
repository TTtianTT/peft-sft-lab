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

"""Calibration dataset helpers for spectral editing."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence


# ----------------------------
# Field helpers
# ----------------------------

def _normalize_text_fields(raw_fields: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not raw_fields:
        return None
    if len(raw_fields) == 1 and "," in raw_fields[0]:
        parts = [p.strip() for p in raw_fields[0].split(",") if p.strip()]
        return parts or None
    return list(raw_fields)


# ----------------------------
# Task / dataset formatters
# ----------------------------

def _format_gsm8k_example(ex: dict) -> Tuple[str, str]:
    q = ex["question"]
    a = ex["answer"]
    prompt = f"Question: {q}\nAnswer:"
    return prompt, a


def _format_metamath_example(ex: dict) -> Tuple[str, str]:
    # meta-math/MetaMathQA commonly uses fields: query / response
    q = ex.get("query", "")
    a = ex.get("response", "")
    prompt = f"Question: {q}\nAnswer:"
    return prompt, a


def _format_magicoder_example(ex: dict) -> Tuple[str, str]:
    # ise-uiuc/Magicoder-Evol-Instruct-110K commonly uses: instruction / response
    inst = ex.get("instruction", "")
    resp = ex.get("response", "")
    prompt = f"### Instruction:\n{inst}\n\n### Response:"
    return prompt, resp


def _format_alpaca_example(ex: dict) -> Tuple[str, str]:
    # tatsu-lab/alpaca uses: instruction / input / output
    inst = str(ex.get("instruction", "") or "")
    inp = str(ex.get("input", "") or "")
    out = str(ex.get("output", "") or "")

    # Keep a stable, explicit prompt boundary to make teacher-forcing sensible.
    # (Avoid trailing whitespace weirdness; make_calib_batch handles whitespace robustly.)
    if inp.strip():
        prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:"
    else:
        prompt = f"### Instruction:\n{inst}\n\n### Response:"
    return prompt, out


def _choices_to_map(choices) -> dict[str, str]:
    """
    Robustly parse CSQA-style 'choices' into a {label: text} mapping.

    Handles:
      - dict with keys: label (list), text (list)
      - list[dict] items with keys: label, text
    """
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
            l = item.get("label")
            t = item.get("text")
            if l is None or t is None:
                continue
            out[str(l)] = str(t)
        if out:
            return out
    raise ValueError(f"Unrecognized choices format: {type(choices)}")


def _format_csqa_example(ex: dict) -> Tuple[str, str]:
    # tau/commonsense_qa fields: question, choices, answerKey
    q = str(ex.get("question", "") or "")
    choices = ex.get("choices")
    answer_key = str(ex.get("answerKey", "") or "")

    m = _choices_to_map(choices)
    # Keep deterministic order A-E when possible; otherwise sorted by label.
    order = ["A", "B", "C", "D", "E"]
    labels = [l for l in order if l in m] + [l for l in sorted(m.keys()) if l not in order]

    choices_lines = "\n".join([f"{l}. {m[l]}" for l in labels])
    prompt = (
        f"Question: {q}\n\n"
        f"Choices:\n{choices_lines}\n\n"
        f"Answer with a single letter: A, B, C, D, or E.\n"
        f"Answer:"
    )
    return prompt, answer_key


def _format_generic_example(ex: dict, fields: Sequence[str]) -> Tuple[str, str]:
    """
    Generic formatter for datasets that already contain prompt/answer fields.

    Supported:
      - 1 field: answer only (prompt="")
      - 2 fields: (prompt, answer)
      - 3 fields: (part1, part2, answer) where prompt = part1 + "\n\n" + part2 (if part2 non-empty)
    """
    if len(fields) == 1:
        return "", str(ex.get(fields[0], ""))
    if len(fields) == 2:
        return str(ex.get(fields[0], "")), str(ex.get(fields[1], ""))
    if len(fields) == 3:
        p1 = str(ex.get(fields[0], "") or "")
        p2 = str(ex.get(fields[1], "") or "")
        ans = str(ex.get(fields[2], "") or "")
        prompt = p1 if not p2.strip() else (p1 + "\n\n" + p2)
        return prompt, ans
    raise ValueError(f"calib_text_fields must have 1, 2, or 3 entries, got {len(fields)}")


def build_calib_formatter(
    calib_dataset: str,
    calib_text_fields: Optional[Sequence[str]],
) -> Tuple[Callable[[dict], Tuple[str, str]], Optional[List[str]]]:
    """
    Return a formatter and normalized text fields.

    Priority:
      1) If calib_text_fields is provided: use the generic formatter.
      2) Otherwise, use built-in dataset-specific defaults for common datasets:
         - gsm8k
         - meta-math/MetaMathQA
         - ise-uiuc/Magicoder-Evol-Instruct-110K
         - tatsu-lab/alpaca
         - tau/commonsense_qa

    For other datasets, calib_text_fields must be provided.
    """
    fields = _normalize_text_fields(calib_text_fields)
    if fields:
        return lambda ex: _format_generic_example(ex, fields), fields

    ds = (calib_dataset or "").strip().lower()

    # Built-in defaults
    if ds in {"gsm8k"}:
        return _format_gsm8k_example, None
    if ds in {"meta-math/metamathqa", "meta-math/metamathqa"}:
        return _format_metamath_example, None
    if ds in {"ise-uiuc/magicoder-evol-instruct-110k"}:
        return _format_magicoder_example, None
    if ds in {"tatsu-lab/alpaca", "tatsu-lab/alpaca-cleaned"}:
        return _format_alpaca_example, None
    if ds in {"tau/commonsense_qa", "tau/commonsenseqa"}:
        return _format_csqa_example, None

    raise ValueError(
        "calib_text_fields must be provided for non-default datasets. "
        f"Got calib_dataset={calib_dataset!r} without calib_text_fields."
    )


# ----------------------------
# Dataset loading / sampling
# ----------------------------

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


# ----------------------------
# Batch builder
# ----------------------------

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
            # Treat any trailing whitespace as a valid separator (space/newline/etc.)
            if prompt and (not prompt[-1].isspace()):
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
