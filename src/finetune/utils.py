from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def is_rank0() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        from transformers import set_seed

        set_seed(seed)
    except Exception:
        pass


def parse_csv_list(value: str) -> list[str]:
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def setup_logging(output_dir: str) -> logging.Logger:
    logger = logging.getLogger("peft_sft_lab")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if is_rank0():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(output_dir) / "train.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_jsonable)


def best_effort_pip_freeze(output_dir: str) -> None:
    if not is_rank0():
        return
    out_path = Path(output_dir) / "requirements-freeze.txt"
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.STDOUT, text=True
        )
        out_path.write_text(freeze, encoding="utf-8")
    except Exception as exc:
        out_path.write_text(f"pip freeze failed: {exc}\n", encoding="utf-8")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_oom_hint() -> str:
    return (
        "CUDA OOM. Suggestions:\n"
        "- reduce --per_device_train_batch_size\n"
        "- increase --gradient_accumulation_steps\n"
        "- lower --max_seq_len\n"
        "- enable --gradient_checkpointing\n"
        "- try --use_qlora (4-bit)\n"
    )

