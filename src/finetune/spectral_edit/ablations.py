"""Deterministic singular-value ablations for post-hoc LoRA analysis."""

from __future__ import annotations

import math
from typing import Any

import torch

from .posthoc_hns import effective_rank_from_sigma


ABLATION_MODES = (
    "scalar_shrink",
    "exact_flat_nuclear",
    "top_shrink",
    "tail_lift",
    "temperature",
)


def _spectrum_stats(before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
    before64 = before.detach().to(dtype=torch.float64).clamp_min(0.0)
    after64 = after.detach().to(dtype=torch.float64).clamp_min(0.0)
    return {
        "rank": int(before64.numel()),
        "nuclear_norm_before": float(before64.sum().item()),
        "nuclear_norm_after": float(after64.sum().item()),
        "fro_norm_before": float(torch.linalg.vector_norm(before64).item()),
        "fro_norm_after": float(torch.linalg.vector_norm(after64).item()),
        "effective_rank_before": effective_rank_from_sigma(before64),
        "effective_rank_after": effective_rank_from_sigma(after64),
        "largest_before": float(before64.max().item()) if before64.numel() else 0.0,
        "largest_after": float(after64.max().item()) if after64.numel() else 0.0,
        "smallest_before": float(before64.min().item()) if before64.numel() else 0.0,
        "smallest_after": float(after64.min().item()) if after64.numel() else 0.0,
        "sigma_before": [float(value) for value in before64.tolist()],
        "sigma_after": [float(value) for value in after64.tolist()],
    }


def transform_singular_values(
    sigma: torch.Tensor,
    *,
    mode: str,
    temperature: float | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply one controlled spectrum ablation while keeping singular vectors fixed.

    Definitions for a non-negative spectrum ``s`` with rank ``r`` and mean ``m``:

    * scalar_shrink: ``c*s``, where its Frobenius norm equals that of the
      nuclear-norm-preserving flat spectrum.
    * exact_flat_nuclear: ``sum(s)/r`` in every direction.
    * top_shrink: replace only entries above ``m`` by ``m``.
    * tail_lift: replace only entries below ``m`` by ``m``.
    * temperature: ``sum(s) * s**tau / sum(s**tau)``.
    """
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode {mode!r}; choose from {ABLATION_MODES}")

    original_dtype = sigma.dtype
    original_device = sigma.device
    values = sigma.detach().to(dtype=torch.float64).clamp_min(0.0)
    rank = int(values.numel())
    if rank == 0:
        stats = _spectrum_stats(values, values)
        stats.update({"mode": mode, "temperature": temperature})
        return sigma.clone(), stats

    nuclear = values.sum()
    mean = nuclear / rank

    if mode == "exact_flat_nuclear":
        edited = torch.full_like(values, float(mean.item()))
    elif mode == "scalar_shrink":
        fro = torch.linalg.vector_norm(values)
        target_fro = nuclear / math.sqrt(rank)
        scale = target_fro / fro if float(fro.item()) > eps else torch.zeros_like(fro)
        edited = values * scale
    elif mode == "top_shrink":
        edited = torch.minimum(values, mean)
    elif mode == "tail_lift":
        edited = torch.maximum(values, mean)
    else:
        if temperature is None or not math.isfinite(float(temperature)) or float(temperature) < 0.0:
            raise ValueError("temperature mode requires a finite --temperature >= 0")
        tau = float(temperature)
        if tau == 0.0:
            edited = torch.full_like(values, float(mean.item()))
        elif float(nuclear.item()) <= eps:
            edited = torch.zeros_like(values)
        else:
            powered = torch.where(values > 0.0, values.pow(tau), torch.zeros_like(values))
            powered_sum = powered.sum()
            edited = powered * (nuclear / powered_sum) if float(powered_sum.item()) > eps else torch.zeros_like(values)

    stats = _spectrum_stats(values, edited)
    stats.update(
        {
            "mode": mode,
            "temperature": float(temperature) if temperature is not None else None,
            "original_mean": float(mean.item()),
            "nuclear_ratio": (
                stats["nuclear_norm_after"] / stats["nuclear_norm_before"]
                if stats["nuclear_norm_before"] > eps
                else 0.0
            ),
            "fro_ratio": (
                stats["fro_norm_after"] / stats["fro_norm_before"]
                if stats["fro_norm_before"] > eps
                else 0.0
            ),
        }
    )
    return edited.to(dtype=original_dtype, device=original_device), stats
