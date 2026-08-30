"""Module-level sensitivity scoring and selection for calibration-guided HNS."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping

import torch


@dataclass(frozen=True)
class ModuleSensitivityScore:
    """Calibration statistics for one LoRA update matrix.

    ``importance`` is the AdaLoRA/SNIP-inspired first-order saliency
    ``mean_b sum_i |sigma_i * grad_i| / rank``. ``compatibility`` is the
    signed first-order benefit predicted for the fixed HNS intervention,
    ``mean_b -sum_i (sigma_hns_i - sigma_i) * grad_i``.
    """

    importance: float
    compatibility: float
    hns_risk: float
    rank: int
    selected_by_importance: bool = False
    selected: bool = False
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        return asdict(self)


def score_module_gradient_batches(
    sigma: torch.Tensor,
    sigma_hns: torch.Tensor,
    gradient_batches: torch.Tensor,
) -> ModuleSensitivityScore:
    """Score one module from per-calibration-batch singular-value gradients.

    Args:
        sigma: Original singular values, shape ``[rank]``.
        sigma_hns: Singular values after the fixed full HNS edit, shape ``[rank]``.
        gradient_batches: One ``dL/dsigma`` vector per calibration batch,
            shape ``[num_batches, rank]``. Absolute values are taken before
            averaging so opposing batches cannot cancel in the importance and
            risk estimates.
    """
    sigma = sigma.detach().to(dtype=torch.float64, device="cpu")
    sigma_hns = sigma_hns.detach().to(dtype=torch.float64, device="cpu")
    gradients = gradient_batches.detach().to(dtype=torch.float64, device="cpu")

    if sigma.ndim != 1 or sigma_hns.ndim != 1:
        raise ValueError("sigma and sigma_hns must be one-dimensional")
    if sigma.shape != sigma_hns.shape:
        raise ValueError("sigma and sigma_hns shapes differ")
    if gradients.ndim != 2 or gradients.shape[1:] != sigma.shape:
        raise ValueError(
            "gradient_batches must have shape [num_batches, rank]; "
            f"got {tuple(gradients.shape)} for rank={sigma.numel()}"
        )
    if gradients.shape[0] == 0:
        raise ValueError("at least one calibration gradient batch is required")

    delta = sigma_hns - sigma
    importance_per_batch = (gradients * sigma.unsqueeze(0)).abs().mean(dim=1)
    compatibility_per_batch = -(gradients * delta.unsqueeze(0)).sum(dim=1)
    risk_per_batch = (gradients * delta.unsqueeze(0)).abs().sum(dim=1)

    return ModuleSensitivityScore(
        importance=float(importance_per_batch.mean().item()),
        compatibility=float(compatibility_per_batch.mean().item()),
        hns_risk=float(risk_per_batch.mean().item()),
        rank=int(sigma.numel()),
    )


def select_important_modules(
    scores: Mapping[str, ModuleSensitivityScore],
    *,
    module_budget: int,
    require_positive_compatibility: bool = True,
    min_compatibility: float = 0.0,
) -> tuple[list[str], dict[str, ModuleSensitivityScore]]:
    """Select a fixed-budget high-importance shortlist, then validate HNS direction.

    The importance budget is applied before the compatibility gate. This makes
    the selected set exactly match the intended two-stage rule: a module must
    first be task-important and must then predict a beneficial fixed HNS edit.
    Consequently the final set may contain fewer than ``module_budget`` modules.
    """
    if module_budget < 1:
        raise ValueError("module_budget must be >= 1")
    if not scores:
        return [], {}

    ordered = sorted(scores, key=lambda name: (-scores[name].importance, name))
    shortlist = set(ordered[: min(module_budget, len(ordered))])
    selected: list[str] = []
    annotated: dict[str, ModuleSensitivityScore] = {}

    for name, score in scores.items():
        selected_by_importance = name in shortlist
        rejection_reason: str | None = None
        keep = selected_by_importance
        if not selected_by_importance:
            rejection_reason = "outside_importance_budget"
        elif require_positive_compatibility and score.compatibility <= min_compatibility:
            keep = False
            rejection_reason = "non_positive_hns_compatibility"

        annotated[name] = replace(
            score,
            selected_by_importance=selected_by_importance,
            selected=keep,
            rejection_reason=rejection_reason,
        )
        if keep:
            selected.append(name)

    selected.sort(key=lambda name: (-scores[name].importance, name))
    return selected, annotated
