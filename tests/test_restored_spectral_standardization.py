from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_2x4_restored_standardization import restore  # noqa: E402


class RestoredSpectralStandardizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sigma = torch.tensor([8.0, 4.0, 2.0, 1.0], dtype=torch.float64)
        centered = self.sigma - self.sigma.mean()
        self.by_std = centered / self.sigma.std(correction=0)
        self.by_variance = centered / self.sigma.var(correction=0)

    def test_frobenius_restoration_hits_target(self):
        target = torch.tensor(3.5, dtype=torch.float64)
        result = restore(self.by_std, target, "fro", 1e-12)
        torch.testing.assert_close(torch.linalg.vector_norm(result), target)

    def test_nuclear_restoration_hits_target(self):
        target = torch.tensor(5.25, dtype=torch.float64)
        result = restore(self.by_std, target, "nuclear", 1e-12)
        torch.testing.assert_close(result.abs().sum(), target)

    def test_variance_and_std_denominators_cancel_after_restoration(self):
        for norm in ("fro", "nuclear"):
            target = torch.tensor(7.0, dtype=torch.float64)
            left = restore(self.by_std, target, norm, 1e-12)
            right = restore(self.by_variance, target, norm, 1e-12)
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
