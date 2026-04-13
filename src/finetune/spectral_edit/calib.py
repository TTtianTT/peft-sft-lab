"""Calibration dataset helpers for spectral editing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

from finetune.csqa_prompt import build_csqa_prompt


_METAMATH_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response: Let's think step by step."
)


@dataclass(frozen=True)
class CalibrationSpec:
    task: Optional[str]
    dataset: str
    config: Optional[str]
    split: str
    text_fields: Optional[List[str]]
    selection_mode: str


TASK_ALIASES = {
    "math": "math",
    "metamath": "math",
    "metamathqa": "math",
    "gsm8k": "math",
    "code": "code",
    "magicoder": "code",
    "humaneval": "code",
    "alpaca": "alpaca",
    "general": "alpaca",
    "ifeval": "alpaca",
    "csqa": "csqa",
    "commonsenseqa": "csqa",
    "commonsense_qa": "csqa",
}

TASK_TO_DEFAULT_CALIBRATION = {
    "math": CalibrationSpec(
        task="math",
        dataset="gsm8k",
        config="main",
        split="train",
        text_fields=None,
        selection_mode="per_task",
    ),
    "code": CalibrationSpec(
        task="code",
        dataset="ise-uiuc/Magicoder-Evol-Instruct-110K",
        config=None,
        split="train",
        text_fields=None,
        selection_mode="per_task",
    ),
    "alpaca": CalibrationSpec(
        task="alpaca",
        dataset="tatsu-lab/alpaca",
        config=None,
        split="train",
        text_fields=None,
        selection_mode="per_task",
    ),
    "csqa": CalibrationSpec(
        task="csqa",
        dataset="tau/commonsense_qa",
        config=None,
        split="train",
        text_fields=None,
        selection_mode="per_task",
    ),
}


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


def normalize_task_name(task: Optional[str]) -> Optional[str]:
    if task is None:
        return None
    key = str(task).strip().lower()
    if not key:
        return None
    return TASK_ALIASES.get(key, key)


def resolve_calibration(
    *,
    task: Optional[str],
    calib_dataset: Optional[str],
    calib_config: Optional[str],
    calib_split: Optional[str],
    calib_text_fields: Optional[Sequence[str]],
    selection_mode: str = "explicit",
) -> CalibrationSpec:
    normalized_fields = _normalize_text_fields(calib_text_fields)
    normalized_task = normalize_task_name(task)

    if calib_dataset:
        dataset = str(calib_dataset).strip()
        config = calib_config
        split = calib_split or "train"

        if normalized_task in TASK_TO_DEFAULT_CALIBRATION:
            default = TASK_TO_DEFAULT_CALIBRATION[normalized_task]
            if dataset == default.dataset:
                config = default.config if calib_config is None else calib_config
                split = calib_split or default.split
        elif dataset.strip().lower() == "gsm8k" and calib_config is None:
            config = "main"

        return CalibrationSpec(
            task=normalized_task,
            dataset=dataset,
            config=config,
            split=split,
            text_fields=normalized_fields,
            selection_mode=selection_mode,
        )

    if normalized_task is None:
        raise ValueError(
            "Calibration is ambiguous. Pass --task to use the task default calibration set, "
            "or pass --calib_dataset explicitly."
        )

    if normalized_task not in TASK_TO_DEFAULT_CALIBRATION:
        known = ", ".join(sorted(TASK_TO_DEFAULT_CALIBRATION))
        raise ValueError(
            f"Unknown task {task!r}. Known task defaults: {known}. "
            "Pass --calib_dataset explicitly if you want a custom calibration set."
        )

    default = TASK_TO_DEFAULT_CALIBRATION[normalized_task]
    return CalibrationSpec(
        task=default.task,
        dataset=default.dataset,
        config=default.config if calib_config is None else calib_config,
        split=calib_split or default.split,
        text_fields=normalized_fields,
        selection_mode=selection_mode,
    )


# ----------------------------
# Task / dataset formatters
# ----------------------------

def _format_gsm8k_example(ex: dict) -> Tuple[str, str]:
    q = str(ex.get("question", "") or "").strip()
    a = str(ex.get("answer", "") or "")
    instruction = (
        "Solve the following math word problem.\n"
        "Put your final numeric answer on the last line exactly as:\n"
        "#### <answer>\n\n"
        f"{q}"
    )
    prompt = _METAMATH_PROMPT_TEMPLATE.format(instruction=instruction) + "\n"
    return prompt, a


def _format_metamath_example(ex: dict) -> Tuple[str, str]:
    q = str(ex.get("query", "") or ex.get("question", "") or "").strip()
    a = str(ex.get("response", "") or ex.get("answer", "") or "")
    prompt = _METAMATH_PROMPT_TEMPLATE.format(instruction=q) + "\n"
    return prompt, a


def _format_magicoder_example(ex: dict) -> Tuple[str, str]:
    inst = str(ex.get("instruction", "") or ex.get("prompt", "") or "").strip()
    resp = str(ex.get("response", "") or ex.get("output", "") or "")
    prompt = inst + "\n"
    return prompt, resp


def _build_csqa_instruction(ex: dict) -> Tuple[str, str]:
    q = str(ex.get("question", "") or "").strip()
    answer_key = str(ex.get("answerKey", "") or "").strip().upper()
    choices = ex.get("choices")
    if answer_key not in {"A", "B", "C", "D", "E"}:
        raise ValueError(f"CSQA answerKey must be one of A/B/C/D/E, got {answer_key!r}.")

    m = _choices_to_map(choices)
    order = ["A", "B", "C", "D", "E"]
    labels = [l for l in order if l in m] + [l for l in sorted(m.keys()) if l not in order]
    choices_lines = "\n".join([f"{l}. {m[l]}" for l in labels])
    instruction = (
        f"Question:\n{q}\n\n"
        f"Choices:\n{choices_lines}\n\n"
        "Answer with a single letter: A, B, C, D, or E."
    )
    return instruction, answer_key


def _format_csqa_example(ex: dict, prompt_style: str = "task_native") -> Tuple[str, str]:
    instruction, answer_key = _build_csqa_instruction(ex)
    return build_csqa_prompt(
        instruction=instruction,
        prompt_style=prompt_style,
        response=None,
    ), answer_key


def _format_alpaca_example(ex: dict) -> Tuple[str, str]:
    inst = str(ex.get("instruction", "") or "").strip()
    inp = str(ex.get("input", "") or "")
    out = str(ex.get("output", "") or "")

    if inp.strip():
        inst = f"{inst}\n\nInput:\n{inp.strip()}"
    prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
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
    csqa_prompt_style: str = "task_native",
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

    if ds in {"gsm8k"}:
        return _format_gsm8k_example, None
    if ds in {"meta-math/metamathqa", "meta-math/metamathqa"}:
        return _format_metamath_example, None
    if ds in {"ise-uiuc/magicoder-evol-instruct-110k"}:
        return _format_magicoder_example, None
    if ds in {"tatsu-lab/alpaca", "tatsu-lab/alpaca-cleaned"}:
        return _format_alpaca_example, None
    if ds in {"tau/commonsense_qa", "tau/commonsenseqa"}:
        return lambda ex: _format_csqa_example(ex, prompt_style=csqa_prompt_style), None

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
    max_seq_len: Optional[int] = None,
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

        if max_seq_len is not None and len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            labels = labels[:max_seq_len]

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attn_mask = (input_ids != tokenizer.pad_token_id).to(torch.long)
    return input_ids, attn_mask, labels
