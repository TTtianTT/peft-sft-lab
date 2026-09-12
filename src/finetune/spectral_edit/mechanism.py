"""Utilities for controlled LoRA/HNS mechanism experiments."""

from __future__ import annotations

import math
from typing import Any

import torch

from .posthoc_hns import effective_rank_from_sigma


CONTROL_NAMES = ("scalar_shrink", "shape_only", "head_only", "tail_only")


def frobenius_norm_from_factors(B: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Return ``||B @ A||_F`` without materialising the dense product."""
    b64 = B.detach().to(dtype=torch.float64)
    a64 = A.detach().to(dtype=torch.float64)
    squared = torch.sum((b64.T @ b64) * (a64 @ a64.T).T).clamp_min(0.0)
    return torch.sqrt(squared)


def align_edited_spectrum_to_reference(
    U: torch.Tensor,
    Vh: torch.Tensor,
    B_edited: torch.Tensor,
    A_edited: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Express an edited low-rank matrix in a reference SVD basis.

    HNS is supposed to preserve the LoRA singular vectors.  The compact matrix
    ``U.T @ (B_edit @ A_edit) @ V`` both recovers the aligned singular values
    from its diagonal and quantifies any violation through its off-diagonal
    energy.  This remains stable when a nearly flat HNS spectrum makes an
    independent SVD rotate within a degenerate subspace.
    """
    u64 = U.detach().to(dtype=torch.float64)
    vh64 = Vh.detach().to(dtype=torch.float64)
    b64 = B_edited.detach().to(dtype=torch.float64)
    a64 = A_edited.detach().to(dtype=torch.float64)
    compact = (u64.T @ b64) @ (a64 @ vh64.T)
    diagonal = torch.diagonal(compact)
    off_diagonal = compact - torch.diag(diagonal)
    full_fro = frobenius_norm_from_factors(b64, a64)
    compact_fro = torch.linalg.vector_norm(compact)
    stats = {
        "basis_offdiag_fraction": float(
            (torch.linalg.vector_norm(off_diagonal) / compact_fro.clamp_min(eps)).item()
        ),
        "basis_projection_residual_fraction": float(
            torch.sqrt((1.0 - (compact_fro / full_fro.clamp_min(eps)).square()).clamp_min(0.0)).item()
        ),
        "negative_aligned_fraction": float((diagonal < -eps).to(torch.float64).mean().item()),
    }
    # Tiny negative values can arise from bf16 roundoff after factor rebuilding.
    return diagonal.clamp_min(0.0).to(dtype=U.dtype, device=U.device), stats


def build_causal_control_spectra(
    sigma_lora: torch.Tensor,
    sigma_hns: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build four controls that decompose an observed HNS spectrum.

    ``scalar_shrink`` preserves the LoRA shape but matches HNS Frobenius norm.
    ``shape_only`` preserves the HNS relative shape but restores LoRA Frobenius
    norm. ``head_only`` accepts only HNS suppressions, while ``tail_only``
    accepts only HNS amplifications.  The latter two combine exactly to HNS on
    disjoint singular directions.
    """
    dtype, device = sigma_lora.dtype, sigma_lora.device
    before = sigma_lora.detach().to(dtype=torch.float64).clamp_min(0.0)
    target = sigma_hns.detach().to(dtype=torch.float64).clamp_min(0.0)
    if before.shape != target.shape:
        raise ValueError(f"Spectrum shape mismatch: {tuple(before.shape)} != {tuple(target.shape)}")

    before_fro = torch.linalg.vector_norm(before)
    target_fro = torch.linalg.vector_norm(target)
    scalar = target_fro / before_fro.clamp_min(eps)
    shape_scale = before_fro / target_fro.clamp_min(eps)
    controls64 = {
        "scalar_shrink": before * scalar,
        "shape_only": target * shape_scale,
        "head_only": torch.minimum(before, target),
        "tail_only": torch.maximum(before, target),
    }
    stats: dict[str, Any] = {
        "rank": int(before.numel()),
        "lora_fro": float(before_fro.item()),
        "hns_fro": float(target_fro.item()),
        "hns_to_lora_fro_ratio": float((target_fro / before_fro.clamp_min(eps)).item()),
        "lora_nuclear": float(before.sum().item()),
        "hns_nuclear": float(target.sum().item()),
        "lora_effective_rank": effective_rank_from_sigma(before),
        "hns_effective_rank": effective_rank_from_sigma(target),
        "num_suppressed": int((target < before - eps).sum().item()),
        "num_amplified": int((target > before + eps).sum().item()),
        "sigma_lora": [float(x) for x in before.tolist()],
        "sigma_hns_aligned": [float(x) for x in target.tolist()],
        "controls": {},
    }
    for name, spectrum in controls64.items():
        stats["controls"][name] = {
            "fro": float(torch.linalg.vector_norm(spectrum).item()),
            "nuclear": float(spectrum.sum().item()),
            "effective_rank": effective_rank_from_sigma(spectrum),
            "sigma": [float(x) for x in spectrum.tolist()],
        }
    controls = {name: value.to(dtype=dtype, device=device) for name, value in controls64.items()}
    return controls, stats


def spectrum_dominance(sigma: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    """Return scale-free head-dominance statistics for a singular spectrum."""
    values = sigma.detach().to(dtype=torch.float64).clamp_min(0.0)
    nuclear = values.sum().clamp_min(eps)
    energy = values.square().sum().clamp_min(eps)
    rank = int(values.numel())
    return {
        "nuclear_norm": float(nuclear.item()),
        "frobenius_norm": float(torch.sqrt(energy).item()),
        "top1_nuclear_share": float((values[:1].sum() / nuclear).item()),
        "top4_nuclear_share": float((values[: min(4, rank)].sum() / nuclear).item()),
        "top1_energy_share": float((values[:1].square().sum() / energy).item()),
        "top4_energy_share": float((values[: min(4, rank)].square().sum() / energy).item()),
        "effective_rank": effective_rank_from_sigma(values),
        "condition_number": float((values[0] / values[-1].clamp_min(eps)).item()) if rank else math.nan,
    }
