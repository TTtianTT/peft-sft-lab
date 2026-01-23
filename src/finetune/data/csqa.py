from __future__ import annotations

from typing import Any

from finetune.data.base import TaskPlugin, format_instruction_response

_LABEL_ORDER = ["A", "B", "C", "D", "E"]


def _choices_to_map(choices: Any) -> dict[str, str]:
    """
    HF tau/commonsense_qa canonical format:
      choices = {"label": ["A","B","C","D","E"], "text": ["...", ...]}
    But keep compatibility with list-of-dict variants.
    """
    # Canonical: dict with list fields
    if isinstance(choices, dict):
        labels = choices.get("label", choices.get("labels"))
        texts = choices.get("text", choices.get("texts"))

        if isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts):
            out: dict[str, str] = {}
            for l, t in zip(labels, texts):
                if l is None or t is None:
                    continue
                key = str(l).strip().upper()
                val = str(t).strip()
                if key:
                    out[key] = val
            if out:
                return out

    # Compatibility: list of {"label": "...", "text": "..."}
    if isinstance(choices, list):
        out: dict[str, str] = {}
        for item in choices:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            text = item.get("text")
            if label is None or text is None:
                continue
            key = str(label).strip().upper()
            val = str(text).strip()
            if key:
                out[key] = val
        if out:
            return out

    raise ValueError(f"Unrecognized choices format: {type(choices)}")


def _get_question_text(q: Any) -> str:
    """
    HF field is a string, but keep a safe fallback if upstream changes.
    """
    if isinstance(q, str):
        return q.strip()
    if isinstance(q, dict):
        # common alt format: {"stem": "..."}
        stem = q.get("stem")
        if isinstance(stem, str):
            return stem.strip()
    return str(q or "").strip()


class CommonsenseQATask(TaskPlugin):
    name = "csqa"
    dataset_id = "tau/commonsense_qa"

    def load(self, split: str):
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(f"datasets is required: {exc}") from exc

        try:
            return load_dataset(self.dataset_id, split=split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.dataset_id} split={split}: {exc}\n"
                "Verify the dataset id on Hugging Face."
            ) from exc

    def format_example(self, example: dict[str, Any]) -> str:
        ex_id = example.get("id", None)
        question = _get_question_text(example.get("question", ""))
        answer = str(example.get("answerKey", "") or "").strip().upper()

        # For SFT we require gold labels (A-E)
        if not question:
            raise ValueError(
                f"CSQA example missing question. id={ex_id!r}. Keys: {sorted(example.keys())}."
            )
        if answer not in set(_LABEL_ORDER):
            raise ValueError(
                f"CSQA example missing/invalid answerKey. id={ex_id!r}, answerKey={answer!r}. "
                "Expected one of A/B/C/D/E (do not use unlabeled splits for SFT)."
            )

        choices_map = _choices_to_map(example.get("choices"))
        lines: list[str] = []
        for label in _LABEL_ORDER:
            if label in choices_map:
                lines.append(f"{label}. {choices_map[label]}")
        if len(lines) != len(_LABEL_ORDER):
            # Still allow if some are missing, but keep it strict by default
            raise ValueError(
                f"CSQA choices malformed/incomplete. id={ex_id!r}. "
                f"Have={sorted(choices_map.keys())}, expected={_LABEL_ORDER}."
            )


        instruction = (
                f"Question:\n{question}\n\nChoices:\n" + "\n".join(lines) + "\n\n"
                "Answer with a single letter: A, B, C, D, or E."
        )
        return f"{instruction}\n{answer}"
