from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune.spectral_edit.posthoc_hns import HNSEditConfig, apply_hns_to_svd, effective_rank_from_sigma
from finetune.spectral_edit.svd import lowrank_svd_from_ba


def _explicit_hns_matrix(
    matrix: torch.Tensor,
    config: HNSEditConfig,
) -> torch.Tensor:
    x = matrix.to(dtype=torch.float64)
    x = x / (torch.linalg.matrix_norm(x) + config.eps)
    for _ in range(config.fast_steps):
        gram = x @ x.transpose(-2, -1)
        x = (
            config.fast_coefficients[0] * x
            + config.fast_coefficients[1] * (gram @ x)
            + config.fast_coefficients[2] * ((gram @ gram) @ x)
        )
    for _ in range(config.stable_steps):
        gram = x @ x.transpose(-2, -1)
        x = (
            config.stable_coefficients[0] * x
            + config.stable_coefficients[1] * (gram @ x)
            + config.stable_coefficients[2] * ((gram @ gram) @ x)
        )
    if config.preserve_nuclear_norm:
        target = torch.linalg.svdvals(matrix.to(dtype=torch.float64)).sum()
        current = torch.linalg.svdvals(x).sum().clamp_min(config.eps)
        x = x * (target / current)
    return x


class PosthocHNSTests(unittest.TestCase):
    def test_effective_rank_increases_after_hns(self):
        sigma = torch.tensor([8.0, 1.0, 0.2, 0.02], dtype=torch.float32)
        config = HNSEditConfig()

        _, _, sigma_new, stats = apply_hns_to_svd(
            U=torch.eye(4, dtype=torch.float32),
            Vh=torch.eye(4, dtype=torch.float32),
            sigma=sigma,
            config=config,
        )

        self.assertGreater(stats["effective_rank_after"], stats["effective_rank_before"])
        self.assertAlmostEqual(float(sigma.sum().item()), float(sigma_new.sum().item()), places=5)

    def test_singular_value_only_update_matches_explicit_matrix_iteration(self):
        torch.manual_seed(0)
        B = torch.randn(6, 3, dtype=torch.float32)
        A = torch.randn(3, 5, dtype=torch.float32)
        matrix = B @ A
        U, S, Vh, _ = lowrank_svd_from_ba(B, A)

        config = HNSEditConfig(fast_steps=3, stable_steps=1)
        U_new, Vh_new, sigma_new, _ = apply_hns_to_svd(U, Vh, S, config=config)
        matrix_from_sigma = (U_new.to(torch.float64) * sigma_new.to(torch.float64).unsqueeze(0)) @ Vh_new.to(torch.float64)
        matrix_explicit = _explicit_hns_matrix(matrix, config=config)

        self.assertTrue(torch.allclose(matrix_from_sigma, matrix_explicit, atol=1e-5, rtol=1e-5))

    def test_effective_rank_helper_handles_zero_spectrum(self):
        self.assertEqual(effective_rank_from_sigma(torch.zeros(4)), 0.0)

    def test_hns_strength_zero_is_identity_and_half_is_interpolation(self):
        sigma = torch.tensor([8.0, 1.0, 0.2, 0.02], dtype=torch.float32)
        eye = torch.eye(4, dtype=torch.float32)

        _, _, sigma_full, _ = apply_hns_to_svd(
            eye, eye, sigma, HNSEditConfig(hns_strength=1.0)
        )
        _, _, sigma_zero, _ = apply_hns_to_svd(
            eye, eye, sigma, HNSEditConfig(hns_strength=0.0)
        )
        _, _, sigma_half, stats = apply_hns_to_svd(
            eye, eye, sigma, HNSEditConfig(hns_strength=0.5)
        )

        self.assertTrue(torch.allclose(sigma_zero, sigma))
        self.assertTrue(torch.allclose(sigma_half, (sigma + sigma_full) / 2, atol=1e-6))
        self.assertAlmostEqual(float(sigma.sum()), float(sigma_half.sum()), places=5)
        self.assertEqual(stats["hns_strength"], 0.5)

    def test_hns_strength_must_be_in_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "hns_strength"):
            HNSEditConfig(hns_strength=1.01)


if __name__ == "__main__":
    unittest.main()
