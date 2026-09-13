"""Flat spectra with the original Frobenius or nuclear budget."""
import math
import torch


def flat_spectrum(sigma: torch.Tensor, method: str) -> torch.Tensor:
    if sigma.ndim != 1 or not sigma.numel() or not torch.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError('expected a finite nonnegative compact spectrum')
    if method == 'flat_fro':
        level = torch.linalg.vector_norm(sigma) / math.sqrt(sigma.numel())
    elif method == 'flat_nuclear':
        level = sigma.sum() / sigma.numel()
    else:
        raise ValueError(method)
    return torch.ones_like(sigma) * level
