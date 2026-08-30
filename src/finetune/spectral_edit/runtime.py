"""Runtime helpers for memory-constrained spectral-edit calibration."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ContextManager

import torch


def enable_deterministic_gradient_checkpointing(model: Any) -> None:
    """Checkpoint decoder blocks without enabling stochastic dropout.

    Transformers only applies gradient checkpointing while the model is in
    training mode. Spectral scoring needs deterministic eval semantics, so the
    model is switched to training mode and every dropout layer is immediately
    switched back to evaluation mode.
    """
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise RuntimeError("Model does not support gradient checkpointing")

    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    try:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        enable()

    enable_inputs = getattr(model, "enable_input_require_grads", None)
    if callable(enable_inputs):
        enable_inputs()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def saved_tensor_offload_context(enabled: bool) -> ContextManager[None]:
    """Offload autograd-saved tensors to CPU while preserving eval semantics.

    Gradient checkpointing in Transformers is normally active only in training
    mode. Spectral calibration deliberately runs in evaluation mode so LoRA
    dropout stays disabled. PyTorch's saved-tensor offload reduces the same
    activation peak without changing the model's training/evaluation state.
    """
    if not enabled:
        return nullcontext()
    save_on_cpu = getattr(torch.autograd.graph, "save_on_cpu", None)
    if save_on_cpu is None:
        raise RuntimeError(
            "--cpu_activation_offload requires torch.autograd.graph.save_on_cpu; "
            "upgrade PyTorch or disable the option."
        )
    return save_on_cpu(pin_memory=False)
