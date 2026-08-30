"""Calibration dataset helpers for spectral editing."""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

from finetune.data.base import make_single_turn_example
from finetune.data.chat_sft import preprocess_chat_example


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

    if inp.strip():
        prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:"
    else:
        prompt = f"### Instruction:\n{inst}\n\n### Response:"
    return prompt, out


def _first_nonempty(ex: dict, names: Sequence[str]) -> str:
    for name in names:
        value = ex.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _format_commonsense170k_example(ex: dict) -> Tuple[str, str]:
    """Accept the common query/response and instruction/output mirrors."""
    prompt = _first_nonempty(ex, ("query", "question", "instruction", "prompt"))
    answer = _first_nonempty(ex, ("response", "answer", "output", "solution"))
    if not prompt or not answer:
        raise ValueError(
            "Commonsense170K example must contain a prompt and response; "
            f"got keys={sorted(ex)}"
        )
    return prompt, answer


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
         - commonsense170k (local mirrors with query/response or instruction/output)

    For other datasets, calib_text_fields must be provided.
    """
    fields = _normalize_text_fields(calib_text_fields)
    if fields:
        return lambda ex: _format_generic_example(ex, fields), fields

    ds = (calib_dataset or "").strip().lower()

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
    if ds in {
        "commonsense170k",
        "commonsense_170k",
        "zwhe99/commonsense_170k",
    }:
        return _format_commonsense170k_example, None

    raise ValueError(
        "calib_text_fields must be provided for non-default datasets. "
        f"Got calib_dataset={calib_dataset!r} without calib_text_fields."
    )


# ----------------------------
# Dataset loading / sampling
# ----------------------------

def _resolve_local_calibration_data_files(
    dataset_path: str,
    split: str,
    dataset_config: Optional[str],
) -> tuple[str, list[str]]:
    """Resolve a local calibration dataset path into a datasets loader + file list."""
    path = Path(dataset_path)
    if not path.exists():
        raise RuntimeError(f"Local calibration dataset path not found: {dataset_path}")

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return "parquet", [str(path)]
        if suffix == ".json":
            return "json", [str(path)]
        if suffix == ".jsonl":
            return "json", [str(path)]
        raise RuntimeError(
            f"Unsupported local calibration file extension: {path.suffix!r}. Use .parquet, .json, or .jsonl."
        )

    candidates = []
    cfg = dataset_config if dataset_config not in (None, "", "none", "null") else None
    if cfg:
        candidates.extend(
            [
                str(path / cfg / f"{split}-*.parquet"),
                str(path / cfg / f"{split}.parquet"),
                str(path / cfg / f"{split}-*.json"),
                str(path / cfg / f"{split}.json"),
                str(path / cfg / f"{split}-*.jsonl"),
                str(path / cfg / f"{split}.jsonl"),
            ]
        )
    candidates.extend(
        [
            str(path / f"{split}-*.parquet"),
            str(path / f"{split}.parquet"),
            str(path / f"{split}-*.json"),
            str(path / f"{split}.json"),
            str(path / f"{split}-*.jsonl"),
            str(path / f"{split}.jsonl"),
        ]
    )

    matches: list[str] = []
    loader_name = ""
    for pattern in candidates:
        matched = sorted(glob(pattern))
        if matched:
            matches = matched
            sample_suffix = Path(matched[0]).suffix.lower()
            loader_name = "parquet" if sample_suffix == ".parquet" else "json"
            break

    if matches:
        return loader_name, matches

    raise RuntimeError(
        f"Could not find local calibration files for split={split!r}, dataset_config={cfg!r} under {dataset_path}. "
        f"Tried patterns: {candidates}"
    )


def load_calibration_split(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    cache_dir: Optional[str] = None,
    dataset_path: Optional[str] = None,
):
    """Load a dataset split for calibration."""
    if dataset_path is not None:
        loader_name, data_files = _resolve_local_calibration_data_files(
            dataset_path=dataset_path,
            split=split,
            dataset_config=dataset_config,
        )
        return load_dataset(loader_name, data_files=data_files, split="train", cache_dir=cache_dir)

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


def make_chat_calib_batch(
    tokenizer,
    examples: Iterable[dict],
    formatter: Callable[[dict], Tuple[str, str]],
    *,
    chat_template_mode: str = "auto",
    max_seq_len: int | None = 2048,
):
    """Build a response-only calibration batch with the model chat template.

    This mirrors chat SFT preprocessing, including Qwen thinking/non-thinking
    template kwargs, so calibration sensitivity is measured under the same
    representation used during training and evaluation.
    """
    features = []
    for raw_example in examples:
        prompt, answer = formatter(raw_example)
        feature = preprocess_chat_example(
            tokenizer=tokenizer,
            example=make_single_turn_example(
                user_content=str(prompt or ""),
                assistant_content=str(answer or ""),
            ),
            max_seq_len=max_seq_len,
            chat_template_mode=chat_template_mode,
        )
        if feature["supervised_token_count"] > 0:
            features.append(feature)

    if not features:
        raise ValueError(
            "Calibration batch has no supervised assistant tokens after chat-template rendering/truncation"
        )

    input_ids = pad_sequence(
        [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    labels = pad_sequence(
        [torch.tensor(feature["labels"], dtype=torch.long) for feature in features],
        batch_first=True,
        padding_value=-100,
    )
    attn_mask = pad_sequence(
        [torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features],
        batch_first=True,
        padding_value=0,
    )
    return input_ids, attn_mask, labels
