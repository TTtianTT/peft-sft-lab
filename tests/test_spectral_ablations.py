from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune.spectral_edit.ablations import transform_singular_values


class SpectralAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sigma = torch.tensor([8.0, 1.0, 0.2, 0.02])

    def test_exact_flat_preserves_nuclear_norm(self):
        edited, stats = transform_singular_values(self.sigma, mode="exact_flat_nuclear")
        self.assertTrue(torch.allclose(edited, torch.full_like(edited, self.sigma.mean())))
        self.assertAlmostEqual(float(edited.sum()), float(self.sigma.sum()), places=5)
        self.assertAlmostEqual(stats["effective_rank_after"], 4.0, places=6)

    def test_scalar_shrink_preserves_shape_and_matches_flat_frobenius(self):
        edited, _ = transform_singular_values(self.sigma, mode="scalar_shrink")
        flat, _ = transform_singular_values(self.sigma, mode="exact_flat_nuclear")
        self.assertTrue(torch.allclose(edited / edited[0], self.sigma / self.sigma[0]))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(edited)), float(torch.linalg.vector_norm(flat)), places=5)

    def test_top_shrink_only_changes_values_above_mean(self):
        edited, _ = transform_singular_values(self.sigma, mode="top_shrink")
        mean = self.sigma.mean()
        self.assertTrue(torch.equal(edited[self.sigma <= mean], self.sigma[self.sigma <= mean]))
        self.assertTrue(torch.equal(edited[self.sigma > mean], torch.full_like(edited[self.sigma > mean], mean)))

    def test_tail_lift_only_changes_values_below_mean(self):
        edited, _ = transform_singular_values(self.sigma, mode="tail_lift")
        mean = self.sigma.mean()
        self.assertTrue(torch.equal(edited[self.sigma >= mean], self.sigma[self.sigma >= mean]))
        self.assertTrue(torch.equal(edited[self.sigma < mean], torch.full_like(edited[self.sigma < mean], mean)))

    def test_temperature_endpoints_and_nuclear_norm(self):
        tau_one, _ = transform_singular_values(self.sigma, mode="temperature", temperature=1.0)
        tau_zero, _ = transform_singular_values(self.sigma, mode="temperature", temperature=0.0)
        flat, _ = transform_singular_values(self.sigma, mode="exact_flat_nuclear")
        tau_half, _ = transform_singular_values(self.sigma, mode="temperature", temperature=0.5)
        self.assertTrue(torch.allclose(tau_one, self.sigma))
        self.assertTrue(torch.allclose(tau_zero, flat))
        self.assertAlmostEqual(float(tau_half.sum()), float(self.sigma.sum()), places=5)

    def test_invalid_temperature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            transform_singular_values(self.sigma, mode="temperature", temperature=-0.1)


if __name__ == "__main__":
    unittest.main()
