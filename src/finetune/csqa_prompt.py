from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from finetune.data.base import format_instruction_response

CSQA_PROMPT_STYLE_CHOICES = ("auto", "task_native", "alpaca_legacy")

_PROMPT_SWITCH_TIMESTAMP = datetime(2026, 1, 21, 18, 30, 49)
_TRAIN_LOG_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


@dataclass(frozen=True)
class CSQAPromptStyleResolution:
    requested: str
    resolved: str
    reason: str


def _parse_train_log_timestamp(adapter_dir: Path) -> Optional[datetime]:
    log_path = adapter_dir / "train.log"
    if not log_path.exists():
        return None

    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = _TRAIN_LOG_TIMESTAMP_RE.match(line.strip())
                if match:
                    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
    return None


def _parse_manifest_timestamp(adapter_dir: Path) -> Optional[datetime]:
    manifest_path = adapter_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None

    created_at = payload.get("created_at")
    if not created_at:
        return None
    try:
        return datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def resolve_csqa_prompt_style(
    requested: str | None,
    adapter_dir: str | Path | None,
) -> CSQAPromptStyleResolution:
    style = (requested or "auto").strip().lower()
    if style not in CSQA_PROMPT_STYLE_CHOICES:
        known = ", ".join(CSQA_PROMPT_STYLE_CHOICES)
        raise ValueError(f"Unknown CSQA prompt style {requested!r}. Expected one of: {known}.")

    if style != "auto":
        return CSQAPromptStyleResolution(
            requested=style,
            resolved=style,
            reason=f"explicit style={style}",
        )

    if adapter_dir is None:
        return CSQAPromptStyleResolution(
            requested=style,
            resolved="task_native",
            reason="auto: no adapter metadata available; defaulting to current task-native prompt",
        )

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        return CSQAPromptStyleResolution(
            requested=style,
            resolved="task_native",
            reason="auto: adapter path not found locally; defaulting to current task-native prompt",
        )

    train_started_at = _parse_train_log_timestamp(adapter_path)
    if train_started_at is not None and train_started_at < _PROMPT_SWITCH_TIMESTAMP:
        return CSQAPromptStyleResolution(
            requested=style,
            resolved="alpaca_legacy",
            reason=(
                "auto: train.log predates the 2026-01-21 CSQA prompt-format change; "
                "using legacy Alpaca-style prompt"
            ),
        )

    manifest_created_at = _parse_manifest_timestamp(adapter_path)
    if manifest_created_at is not None and manifest_created_at < _PROMPT_SWITCH_TIMESTAMP:
        return CSQAPromptStyleResolution(
            requested=style,
            resolved="alpaca_legacy",
            reason=(
                "auto: run_manifest.json predates the 2026-01-21 CSQA prompt-format change; "
                "using legacy Alpaca-style prompt"
            ),
        )

    return CSQAPromptStyleResolution(
        requested=style,
        resolved="task_native",
        reason="auto: adapter metadata is current or unavailable; using current task-native prompt",
    )


def build_csqa_prompt(
    *,
    instruction: str,
    prompt_style: str,
    response: str | None = None,
) -> str:
    instruction = instruction.strip()
    style = prompt_style.strip().lower()

    if style == "task_native":
        if response is None:
            return instruction + "\n"
        response_text = str(response).strip()
        if not response_text:
            return instruction + "\n"
        return instruction + "\n" + response_text

    if style == "alpaca_legacy":
        return format_instruction_response(instruction=instruction, response=response or "")

    raise ValueError(f"CSQA prompt style must be resolved before use, got {prompt_style!r}.")
