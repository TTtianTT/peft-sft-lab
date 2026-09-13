"""Original unknown-noise Gavish–Donoho SVHT, including the full zero spectrum.

Paper https://arxiv.org/abs/1305.5870, equations (11), (26) and III-E.
MP density normalization follows the authors' optimal_SVHT_coef.m supplement.
No active-spectrum median or polynomial approximation is used.
"""
from functools import lru_cache
import math
import numpy as np
import torch


def lambda_star(beta: float) -> float:
    if not 0 < beta <= 1:
        raise ValueError('aspect ratio must be in (0, 1]')
    return math.sqrt(2 * (beta + 1) + 8 * beta / (beta + 1 + math.sqrt(beta**2 + 14 * beta + 1)))


@lru_cache(maxsize=32)
def mp_median(beta: float) -> float:
    lambda_star(beta)  # Validate beta.
    root_beta = math.sqrt(beta)
    previous = None
    for order in (64, 128, 256, 512, 1024):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        def cdf(theta):
            if beta == 1:
                return (theta + math.sin(theta)) / math.pi
            angles = theta * (nodes + 1) / 2
            x = (1 - root_beta)**2 + 4 * root_beta * np.sin(angles / 2)**2
            # x=1+beta-2 sqrt(beta) cos(theta) removes endpoint square roots
            # from sqrt((b-x)(x-a))/(2*pi*beta*x) dx.
            density = 2 * np.sin(angles)**2 / (math.pi * x)
            return float(theta / 2 * np.dot(weights, density))
        lo, hi = 0., math.pi
        for _ in range(64):
            middle = (lo + hi) / 2
            if cdf(middle) < 0.5:
                lo = middle
            else:
                hi = middle
        theta = (lo + hi) / 2
        median = (1 - root_beta)**2 + 4 * root_beta * math.sin(theta / 2)**2
        if previous is not None and abs(median - previous) < 1e-11:
            return median
        previous = median
    raise RuntimeError(f'MP median quadrature did not converge: beta={beta}')


def dg_hard_spectrum(sigma: torch.Tensor, m: int, n: int):
    p = min(m, n)
    if m <= 0 or n <= 0 or sigma.ndim != 1 or sigma.numel() > p or not torch.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError('invalid dimensions or compact spectrum')
    beta = p / max(m, n)
    full_spectrum = np.zeros(p, dtype=np.float64)
    full_spectrum[:sigma.numel()] = sigma.detach().double().cpu().numpy()
    median = float(np.median(full_spectrum))  # Even-size median averages central values.
    mu = mp_median(beta)
    coefficient = lambda_star(beta) / math.sqrt(mu)
    threshold = coefficient * median
    # Compare in float64 so a nonzero threshold is not rounded to sigma's dtype.
    keep = sigma.double() > threshold
    target = sigma * keep.to(sigma.dtype)
    return target, {'matrix_shape': [m, n], 'aspect_ratio': beta, 'full_spectrum_size': p,
        'implicit_zero_count': p - sigma.numel(), 'full_spectrum_median': median,
        'mp_median': mu, 'lambda_star': lambda_star(beta), 'omega': coefficient,
        'threshold': threshold, 'retained_rank': int(keep.sum()),
        'unchanged': torch.equal(target, sigma), 'identity_due_to_zero_median': median == 0 and torch.equal(target, sigma),
        'fro_before': float(sigma.norm()), 'fro_retained': float(target.norm()),
        'nuclear_before': float(sigma.sum()), 'nuclear_retained': float(target.sum()),
        'singular_values_before': sigma.tolist(), 'singular_values_retained': target.tolist()}
