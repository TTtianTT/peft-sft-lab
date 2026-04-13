from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _pick_tokenizer_source(base_model: str, adapter_dir: str | None) -> str:
    if adapter_dir is None:
        return base_model
    p = Path(adapter_dir)
    if (p / "tokenizer.json").exists() or (p / "tokenizer.model").exists():
        return adapter_dir
    return base_model


def load_tokenizer_for_prompt_stats(
    *,
    base_model: str,
    adapter_dir: str | None,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        _pick_tokenizer_source(base_model, adapter_dir),
        use_fast=True,
        local_files_only=True,
    )
    return tokenizer


def load_dataset_split(
    dataset_name: str,
    *,
    split: str,
    dataset_config: str | None = None,
):
    from datasets import load_dataset

    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split)
    return load_dataset(dataset_name, split=split)


def select_fixed_exemplars(
    dataset,
    *,
    k: int,
    seed: int,
) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    n = min(k, len(dataset))
    if n <= 0:
        return []

    if "__fewshot_orig_idx__" not in getattr(dataset, "column_names", []):
        dataset = dataset.add_column("__fewshot_orig_idx__", list(range(len(dataset))))

    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)
    if hasattr(dataset, "select"):
        dataset = dataset.select(range(n))
    return [dataset[i] for i in range(len(dataset))]


def save_exemplars_jsonl(
    path: str | Path,
    exemplars: list[dict[str, Any]],
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in exemplars:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


class PromptStatsAccumulator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.total_base_prompt_tokens = 0
        self.total_prompt_tokens = 0
        self.max_base_prompt_tokens = 0
        self.max_prompt_tokens = 0
        self.count = 0

    def add(self, *, base_prompt: str, prompt: str) -> tuple[int, int]:
        base_tokens = len(self.tokenizer(base_prompt, add_special_tokens=False).input_ids)
        prompt_tokens = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
        self.total_base_prompt_tokens += base_tokens
        self.total_prompt_tokens += prompt_tokens
        self.max_base_prompt_tokens = max(self.max_base_prompt_tokens, base_tokens)
        self.max_prompt_tokens = max(self.max_prompt_tokens, prompt_tokens)
        self.count += 1
        return base_tokens, prompt_tokens

    def summary(self) -> dict[str, Optional[float]]:
        if self.count == 0:
            return {
                "avg_base_prompt_tokens": None,
                "avg_prompt_tokens": None,
                "avg_extra_prompt_tokens": None,
                "total_base_prompt_tokens": 0,
                "total_prompt_tokens": 0,
                "total_extra_prompt_tokens": 0,
                "max_base_prompt_tokens": 0,
                "max_prompt_tokens": 0,
            }
        total_extra = self.total_prompt_tokens - self.total_base_prompt_tokens
        return {
            "avg_base_prompt_tokens": self.total_base_prompt_tokens / self.count,
            "avg_prompt_tokens": self.total_prompt_tokens / self.count,
            "avg_extra_prompt_tokens": total_extra / self.count,
            "total_base_prompt_tokens": self.total_base_prompt_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_extra_prompt_tokens": total_extra,
            "max_base_prompt_tokens": self.max_base_prompt_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
        }
