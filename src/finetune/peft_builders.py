from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any

from packaging import version


def _get_peft_version() -> str | None:
    try:
        import peft

        return getattr(peft, "__version__", None)
    except Exception:
        return None


def _fail_with_install_hint(msg: str, pip_hint: str) -> RuntimeError:
    return RuntimeError(f"{msg}\n\nInstall/upgrade guidance:\n  {pip_hint}\n")


def build_peft_config(
    *,
    peft_method: str,
    target_modules: list[str],
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    pissa_init_mode: str,
    init_r: int,
    target_r: int,
    max_steps: int,
) -> Any:
    peft_method = peft_method.lower()
    if peft_method not in {"lora", "pissa", "adalora", "loraplus"}:
        raise ValueError(f"Unsupported --peft_method: {peft_method}")

    try:
        from peft import AdaLoraConfig, LoraConfig, TaskType
    except Exception as exc:
        raise _fail_with_install_hint(
            f"Failed to import peft ({exc}).",
            "pip install -U peft",
        ) from exc

    if peft_method in {"lora", "loraplus"}:
        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )

    if peft_method == "pissa":
        sig = inspect.signature(LoraConfig)
        if "init_lora_weights" not in sig.parameters:
            raise _fail_with_install_hint(
                "Your `peft` version does not support PiSSA (missing LoraConfig(init_lora_weights=...)).",
                "pip install -U 'peft>=0.11.0'",
            )
        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            init_lora_weights=pissa_init_mode,
        )

    if peft_method == "adalora":
        if max_steps <= 0:
            raise ValueError("--max_steps must be > 0 for AdaLoRA.")
        total_step = max_steps
        tinit = max(1, int(0.10 * total_step))
        tfinal = max(tinit + 1, int(0.80 * total_step))
        delta_t = max(1, int(0.01 * total_step))

        # AdaLoRA config uses step counts (not ratios).
        return AdaLoraConfig(
            init_r=init_r,
            target_r=target_r,
            beta1=0.85,
            beta2=0.85,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=delta_t,
            total_step=total_step,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )

    raise ValueError(f"Unhandled peft_method: {peft_method}")


@dataclass(frozen=True)
class LoraPlusGrouping:
    param_groups: list[dict[str, Any]]
    summary: dict[str, int]


_RE_LORA_A = re.compile(r"(?:^|\.)lora_A(?:\.|$)")
_RE_LORA_B = re.compile(r"(?:^|\.)lora_B(?:\.|$)")
_RE_LORA_EMB_A = re.compile(r"(?:^|\.)lora_embedding_A(?:\.|$)")
_RE_LORA_EMB_B = re.compile(r"(?:^|\.)lora_embedding_B(?:\.|$)")


def build_loraplus_param_groups(
    *,
    model: Any,
    lr: float,
    lr_ratio: float,
    weight_decay: float,
    lr_embedding: float | None,
) -> LoraPlusGrouping:
    if lr_ratio <= 0:
        raise ValueError("--loraplus_lr_ratio must be > 0.")

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required for LoRA+: {exc}") from exc

    lora_a: list[torch.nn.Parameter] = []
    lora_b: list[torch.nn.Parameter] = []
    lora_emb_a: list[torch.nn.Parameter] = []
    lora_emb_b: list[torch.nn.Parameter] = []
    other: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        if _RE_LORA_EMB_A.search(name):
            lora_emb_a.append(param)
        elif _RE_LORA_EMB_B.search(name):
            lora_emb_b.append(param)
        elif _RE_LORA_A.search(name):
            lora_a.append(param)
        elif _RE_LORA_B.search(name):
            lora_b.append(param)
        else:
            other.append(param)

    emb_lr = lr if lr_embedding is None else lr_embedding
    param_groups: list[dict[str, Any]] = []
    if lora_a:
        param_groups.append({"params": lora_a, "lr": lr, "weight_decay": weight_decay})
    if lora_b:
        param_groups.append({"params": lora_b, "lr": lr * lr_ratio, "weight_decay": weight_decay})
    if lora_emb_a:
        param_groups.append({"params": lora_emb_a, "lr": emb_lr, "weight_decay": weight_decay})
    if lora_emb_b:
        param_groups.append(
            {"params": lora_emb_b, "lr": emb_lr * lr_ratio, "weight_decay": weight_decay}
        )
    if other:
        param_groups.append({"params": other, "lr": lr, "weight_decay": weight_decay})

    summary = {
        "lora_A": len(lora_a),
        "lora_B": len(lora_b),
        "lora_embedding_A": len(lora_emb_a),
        "lora_embedding_B": len(lora_emb_b),
        "other_trainable": len(other),
    }
    return LoraPlusGrouping(param_groups=param_groups, summary=summary)


def check_peft_method_support(peft_method: str) -> None:
    peft_method = peft_method.lower()
    if peft_method != "pissa":
        return

    peft_ver = _get_peft_version()
    if peft_ver is None:
        return
    if version.parse(peft_ver) < version.parse("0.11.0"):
        raise _fail_with_install_hint(
            f"PiSSA is not supported in peft=={peft_ver}.",
            "pip install -U 'peft>=0.11.0'",
        )

