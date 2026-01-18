from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainProfile:
    name: str
    max_seq_len: int
    warmup_ratio: float
    min_lr_ratio: float
    weight_decay: float
    grad_clip: float
    bf16: bool
    fp16: bool
    adam_beta1: float
    adam_beta2: float
    global_train_batch_size: int | None
    lr_scheduler_type: str
    num_train_epochs: float | None = None


TRAIN_PROFILES: dict[str, TrainProfile] = {
    "paper_code_ift": TrainProfile(
        name="paper_code_ift",
        max_seq_len=4096,
        warmup_ratio=0.1,
        min_lr_ratio=0.01,
        weight_decay=0.0,
        grad_clip=1.0,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        global_train_batch_size=192,
        lr_scheduler_type="cosine_with_min_lr",
    ),
    "paper_code_ift_3ep": TrainProfile(
        name="paper_code_ift_3ep",
        max_seq_len=4096,
        warmup_ratio=0.1,
        min_lr_ratio=0.01,
        weight_decay=0.0,
        grad_clip=1.0,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        global_train_batch_size=192,
        lr_scheduler_type="cosine_with_min_lr",
        num_train_epochs=3.0,
    ),
    "paper_math_ift": TrainProfile(
        name="paper_math_ift",
        max_seq_len=1024,
        warmup_ratio=0.1,
        min_lr_ratio=0.01,
        weight_decay=0.0,
        grad_clip=1.0,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        global_train_batch_size=768,
        lr_scheduler_type="cosine_with_min_lr",
    ),
    "paper_math_ift_3ep": TrainProfile(
        name="paper_math_ift_3ep",
        max_seq_len=1024,
        warmup_ratio=0.1,
        min_lr_ratio=0.01,
        weight_decay=0.0,
        grad_clip=1.0,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        global_train_batch_size=768,
        lr_scheduler_type="cosine_with_min_lr",
        num_train_epochs=3.0,
    ),
    "paper_default_ift": TrainProfile(
        name="paper_default_ift",
        max_seq_len=2048,
        warmup_ratio=0.1,
        min_lr_ratio=0.01,
        weight_decay=0.0,
        grad_clip=1.0,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        global_train_batch_size=256,
        lr_scheduler_type="cosine_with_min_lr",
    ),
    "paper_alpaca_3ep": TrainProfile(
        name="paper_alpaca_3ep",
        max_seq_len=2048,
        warmup_ratio=0.1,
        min_lr_ratio=0.01,
        weight_decay=0.0,
        grad_clip=1.0,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        global_train_batch_size=256,
        lr_scheduler_type="cosine_with_min_lr",
        num_train_epochs=3.0,
    ),
    "paper_csqa_3ep": TrainProfile(
        name="paper_csqa_3ep",
        max_seq_len=2048,
        warmup_ratio=0.1,
        min_lr_ratio=0.01,
        weight_decay=0.0,
        grad_clip=1.0,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        global_train_batch_size=256,
        lr_scheduler_type="cosine_with_min_lr",
        num_train_epochs=3.0,
    ),
}


TASK_PROFILE_MAP: dict[str, str] = {
    "magicoder": "paper_code_ift_3ep",
    "code": "paper_code_ift_3ep",
    "metamath": "paper_math_ift_3ep",
    "metamathqa": "paper_math_ift_3ep",
    "math": "paper_math_ift_3ep",
    "alpaca": "paper_alpaca_3ep",
    "csqa": "paper_csqa_3ep",
    "commonsenseqa": "paper_csqa_3ep",
}

DEFAULT_PROFILE = "paper_default_ift"


def resolve_profile(name: str) -> TrainProfile:
    key = name.strip().lower()
    if key not in TRAIN_PROFILES:
        known = ", ".join(sorted(TRAIN_PROFILES))
        raise ValueError(f"Unknown train_profile={name!r}. Known profiles: {known}")
    return TRAIN_PROFILES[key]


def pick_profile(task: str, requested: str | None) -> TrainProfile | None:
    if not requested:
        return None
    key = requested.strip().lower()
    if key in {"none", "off", "disable", "disabled"}:
        return None
    if key == "auto":
        profile_key = TASK_PROFILE_MAP.get(task.strip().lower(), DEFAULT_PROFILE)
        return resolve_profile(profile_key)
    return resolve_profile(key)


def profile_overrides(profile: TrainProfile) -> dict[str, Any]:
    return {
        "max_seq_len": profile.max_seq_len,
        "warmup_ratio": profile.warmup_ratio,
        "min_lr_ratio": profile.min_lr_ratio,
        "weight_decay": profile.weight_decay,
        "grad_clip": profile.grad_clip,
        "bf16": profile.bf16,
        "fp16": profile.fp16,
        "adam_beta1": profile.adam_beta1,
        "adam_beta2": profile.adam_beta2,
        "global_train_batch_size": profile.global_train_batch_size,
        "lr_scheduler_type": profile.lr_scheduler_type,
        "num_train_epochs": profile.num_train_epochs,
    }
