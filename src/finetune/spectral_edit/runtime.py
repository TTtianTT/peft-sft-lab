"""Runtime helpers for memory-constrained spectral-edit calibration."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


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
