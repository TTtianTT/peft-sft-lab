"""Conservative RMT-style spectrum diagnostics for LoRA singular values."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, Sequence

import numpy as np
import torch


def effective_tail_count(rank: int, requested_tail_count: int) -> int:
    """Resolve how many smallest singular values to treat as the bulk candidate set."""
    if rank <= 1:
        return 1
    if requested_tail_count > 0:
        return max(1, min(rank - 1, int(requested_tail_count)))
    return max(1, min(rank - 1, rank // 2))


@lru_cache(maxsize=None)
def mp_median_eigenvalue(beta: float, grid_size: int = 20000) -> float:
    """Numerically approximate the median eigenvalue of the unit-variance MP law."""
    beta = float(beta)
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    sqrt_beta = math.sqrt(beta)
    edge_low = (1.0 - sqrt_beta) ** 2
    edge_high = (1.0 + sqrt_beta) ** 2

    thetas = np.linspace(0.0, math.pi, int(grid_size), dtype=np.float64)
    xs = 1.0 + beta - (2.0 * sqrt_beta * np.cos(thetas))
    # Parameterizing lambda by theta removes the integrable edge singularity at beta=1.
    d_cdf_d_theta = (2.0 * np.sin(thetas) ** 2) / (math.pi * np.clip(xs, 1e-12, None))
    d_theta = np.diff(thetas)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d_cdf_d_theta[:-1] + d_cdf_d_theta[1:]) * d_theta)])
    total = float(cdf[-1])
    if total <= 0.0:
        return float((edge_low + edge_high) / 2.0)
    cdf /= total

    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = max(0, min(idx, xs.size - 1))
    return float(xs[idx])


def estimate_mp_summary(
    singular_values: Sequence[float],
    out_dim: int,
    in_dim: int,
    tail_count: int = 0,
    edge_margin: float = 0.10,
) -> Dict[str, Any]:
    """Estimate a conservative MP-style bulk edge and classify singular components."""
    sigma = np.asarray(list(float(v) for v in singular_values), dtype=np.float64)
    if sigma.ndim != 1 or sigma.size == 0:
        raise ValueError("singular_values must be a non-empty 1D sequence")

    rank = int(sigma.size)
    resolved_tail_count = effective_tail_count(rank, tail_count)
    bulk_tail = sigma[-resolved_tail_count:]
    beta = float(min(out_dim, in_dim) / max(out_dim, in_dim))

    bulk_lambda = bulk_tail ** 2
    median_lambda = float(np.median(bulk_lambda))
    mp_median = float(mp_median_eigenvalue(round(beta, 8)))
    noise_variance = (median_lambda / mp_median) if mp_median > 0.0 else 0.0

    theoretical_lambda_plus = float(noise_variance * (1.0 + math.sqrt(beta)) ** 2)
    theoretical_sigma_plus = float(math.sqrt(max(theoretical_lambda_plus, 0.0)))
    tail_max = float(np.max(bulk_tail))
    conservative_sigma_plus = max(theoretical_sigma_plus, tail_max)

    near_low = max(0.0, (1.0 - edge_margin) * conservative_sigma_plus)
    near_high = (1.0 + edge_margin) * conservative_sigma_plus

    components = []
    label_counts = {"likely_signal": 0, "near_edge": 0, "likely_bulk_noise": 0}
    for idx, value in enumerate(sigma):
        if value > near_high:
            label = "likely_signal"
        elif value < near_low:
            label = "likely_bulk_noise"
        else:
            label = "near_edge"
        label_counts[label] += 1
        components.append(
            {
                "component_index": int(idx),
                "singular_value": float(value),
                "above_theoretical_edge": bool(value > theoretical_sigma_plus),
                "above_conservative_edge": bool(value > conservative_sigma_plus),
                "rmt_label": label,
            }
        )

    return {
        "rank": rank,
        "out_dim": int(out_dim),
        "in_dim": int(in_dim),
        "aspect_ratio_beta": beta,
        "tail_count": int(resolved_tail_count),
        "tail_indices": [int(i) for i in range(rank - resolved_tail_count, rank)],
        "tail_singular_values": [float(v) for v in bulk_tail.tolist()],
        "tail_max_singular_value": tail_max,
        "median_tail_lambda": median_lambda,
        "mp_median_eigenvalue": mp_median,
        "noise_variance_hat": noise_variance,
        "theoretical_lambda_plus": theoretical_lambda_plus,
        "theoretical_sigma_plus": theoretical_sigma_plus,
        "conservative_sigma_plus": conservative_sigma_plus,
        "edge_margin": float(edge_margin),
        "label_counts": label_counts,
        "noise_ratio": float(label_counts["likely_bulk_noise"] / max(1, rank)),
        "components": components,
    }


def bulk_noise_mask_from_summary(summary: Dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    """Return a boolean mask for the components classified as likely bulk/noise."""
    mask = [comp["rmt_label"] == "likely_bulk_noise" for comp in summary["components"]]
    return torch.tensor(mask, dtype=torch.bool, device=device)
