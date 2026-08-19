"""Post-hoc Hybrid Newton-Schulz editing for LoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch


FAST_HNS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
STABLE_HNS_COEFFICIENTS = (2.0, -1.5, 0.5)


@dataclass(frozen=True)
class HNSEditConfig:
    """Configuration for post-hoc Hybrid Newton-Schulz editing."""

    fast_steps: int = 8
    stable_steps: int = 2
    fast_coefficients: tuple[float, float, float] = FAST_HNS_COEFFICIENTS
    stable_coefficients: tuple[float, float, float] = STABLE_HNS_COEFFICIENTS
    preserve_nuclear_norm: bool = True
    output_rank: int | None = None
    eps: float = 1e-7


def effective_rank_from_sigma(sigma: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute effective rank from a non-negative singular value vector."""
    s = sigma.detach().to(dtype=torch.float64).clamp_min(0.0)
    total = s.sum()
    if float(total.item()) <= eps:
        return 0.0
    probs = s / total
    probs = probs[probs > eps]
    if probs.numel() == 0:
        return 0.0
    entropy = -(probs * probs.log()).sum()
    return float(torch.exp(entropy).item())


def _ns_polynomial_step(
    sigma: torch.Tensor,
    coefficients: Tuple[float, float, float],
) -> torch.Tensor:
    """Apply one Newton-Schulz polynomial step to singular values."""
    a, b, c = coefficients
    sigma_sq = sigma * sigma
    return sigma * (a + b * sigma_sq + c * sigma_sq * sigma_sq)


def hybrid_newton_schulz_singular_values(
    sigma: torch.Tensor,
    config: HNSEditConfig | None = None,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply the Hybrid Newton-Schulz iteration to a singular value spectrum.

    For X = U diag(sigma) V^T, the Muon / DeepSeek-style NS update
    X <- a X + b (X X^T) X + c (X X^T)^2 X keeps U and V fixed and only
    changes singular values elementwise. This lets us edit LoRA updates
    without materializing the dense delta-W matrix.
    """
    if config is None:
        config = HNSEditConfig()

    sigma0 = sigma.detach().to(dtype=torch.float64).clamp_min(0.0)
    rank_full = int(sigma0.numel())
    rank_out = rank_full if config.output_rank is None else min(max(1, int(config.output_rank)), rank_full)
    sigma_ref = sigma0[:rank_out].clone()

    if rank_full == 0:
        return sigma.clone(), {
            "rank_before_full": 0,
            "rank_after": 0,
            "effective_rank_before_full": 0.0,
            "effective_rank_before": 0.0,
            "effective_rank_after": 0.0,
            "nuclear_norm_before_full": 0.0,
            "nuclear_norm_before": 0.0,
            "nuclear_norm_after": 0.0,
            "fro_norm_before_full": 0.0,
            "fro_norm_after": 0.0,
            "preserve_nuclear_norm": bool(config.preserve_nuclear_norm),
            "fast_steps": int(config.fast_steps),
            "stable_steps": int(config.stable_steps),
            "fast_coefficients": list(config.fast_coefficients),
            "stable_coefficients": list(config.stable_coefficients),
            "sigma_before": [],
            "sigma_after": [],
        }

    fro_norm = torch.linalg.vector_norm(sigma0).clamp_min(config.eps)
    sigma_ns = sigma0 / fro_norm

    for _ in range(int(config.fast_steps)):
        sigma_ns = _ns_polynomial_step(sigma_ns, config.fast_coefficients)
    for _ in range(int(config.stable_steps)):
        sigma_ns = _ns_polynomial_step(sigma_ns, config.stable_coefficients)

    sigma_ns = sigma_ns[:rank_out].clamp_min(0.0)

    if config.preserve_nuclear_norm:
        target_sum = sigma_ref.sum()
        current_sum = sigma_ns.sum()
        if float(current_sum.item()) <= config.eps:
            fill_value = float(target_sum.item()) / max(1, rank_out)
            sigma_ns = torch.full_like(sigma_ns, fill_value)
        else:
            sigma_ns = sigma_ns * (target_sum / current_sum)

    stats: Dict[str, Any] = {
        "rank_before_full": rank_full,
        "rank_after": rank_out,
        "effective_rank_before_full": effective_rank_from_sigma(sigma0),
        "effective_rank_before": effective_rank_from_sigma(sigma_ref),
        "effective_rank_after": effective_rank_from_sigma(sigma_ns),
        "nuclear_norm_before_full": float(sigma0.sum().item()),
        "nuclear_norm_before": float(sigma_ref.sum().item()),
        "nuclear_norm_after": float(sigma_ns.sum().item()),
        "fro_norm_before_full": float(fro_norm.item()),
        "fro_norm_after": float(torch.linalg.vector_norm(sigma_ns).item()),
        "preserve_nuclear_norm": bool(config.preserve_nuclear_norm),
        "fast_steps": int(config.fast_steps),
        "stable_steps": int(config.stable_steps),
        "fast_coefficients": list(config.fast_coefficients),
        "stable_coefficients": list(config.stable_coefficients),
        "sigma_before": [float(x) for x in sigma_ref.tolist()],
        "sigma_after": [float(x) for x in sigma_ns.tolist()],
    }
    return sigma_ns.to(dtype=sigma.dtype, device=sigma.device), stats


def apply_hns_to_svd(
    U: torch.Tensor,
    Vh: torch.Tensor,
    sigma: torch.Tensor,
    config: HNSEditConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Apply post-hoc HNS to an SVD and optionally truncate the output rank."""
    sigma_new, stats = hybrid_newton_schulz_singular_values(sigma, config=config)
    rank_out = int(sigma_new.numel())
    return (
        U[:, :rank_out].contiguous(),
        Vh[:rank_out, :].contiguous(),
        sigma_new,
        stats,
    )
