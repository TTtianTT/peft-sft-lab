"""Post-Optimization Adaptive Rank Allocation (PARA) primitives.

This implements epsilon-PARA from arXiv:2604.27796: one global threshold is
chosen from the pooled singular values of every LoRA update in an adapter, and
all values at or above that threshold are retained without rescaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class ParaThreshold:
    epsilon: float
    threshold: float
    requested_energy_ratio: float
    retained_energy_ratio: float
    retained_components: int
    total_components: int


def epsilon_para_masks(
    singular_values: Mapping[str, torch.Tensor],
    *,
    epsilon: float,
    scaling: Mapping[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], ParaThreshold]:
    """Return per-module masks selected by one global effective-SV threshold.

    ``scaling`` contains the LoRA multiplier applied to each ``B @ A``.  PARA's
    global comparison is performed on the singular values of the effective
    updates, ``abs(scale) * sigma``.  The original, unscaled ``sigma`` values
    are retained verbatim during reconstruction.
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError(f"epsilon must be in (0, 1], got {epsilon}")
    if not singular_values:
        raise ValueError("at least one module spectrum is required")

    effective: dict[str, torch.Tensor] = {}
    flat = []
    for name, raw in singular_values.items():
        values = raw.detach().to(dtype=torch.float64, device="cpu")
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError(f"{name}: spectrum must be a nonempty vector")
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name}: spectrum must be finite and nonnegative")
        multiplier = 1.0 if scaling is None else float(scaling[name])
        if not torch.isfinite(torch.tensor(multiplier)) or multiplier == 0:
            raise ValueError(f"{name}: invalid LoRA scaling {multiplier}")
        current = values * abs(multiplier)
        effective[name] = current
        flat.append(current)

    pooled = torch.cat(flat)
    energy = pooled.square()
    total = energy.sum()
    if total <= 0:
        raise ValueError("pooled spectrum has zero energy")
    order = torch.argsort(pooled, descending=True, stable=True)
    cumulative = torch.cumsum(energy[order], dim=0)
    target = float(epsilon) * total
    first = int(torch.searchsorted(cumulative, target, right=False).item())
    threshold = pooled[order[first]]
    masks = {name: values >= threshold for name, values in effective.items()}
    retained = sum(int(mask.sum().item()) for mask in masks.values())
    retained_energy = sum(
        float(effective[name][mask].square().sum().item()) for name, mask in masks.items()
    )
    stats = ParaThreshold(
        epsilon=float(epsilon),
        threshold=float(threshold.item()),
        requested_energy_ratio=float(epsilon),
        retained_energy_ratio=retained_energy / float(total.item()),
        retained_components=retained,
        total_components=int(pooled.numel()),
    )
    return masks, stats
