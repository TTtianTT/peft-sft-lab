from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune.spectral_edit.module_selection import (  # noqa: E402
    ModuleSensitivityScore,
    score_module_gradient_batches,
    select_important_modules,
)


class SensitivityHNSSelectionTests(unittest.TestCase):
    def test_scores_average_absolute_importance_before_batches_can_cancel(self):
        sigma = torch.tensor([2.0, 1.0])
        sigma_hns = torch.tensor([1.5, 1.5])
        gradients = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])

        score = score_module_gradient_batches(sigma, sigma_hns, gradients)

        # Each batch has mean(|sigma * grad|) = mean([2, 2]) = 2.
        self.assertAlmostEqual(score.importance, 2.0)
        # Signed HNS compatibility cancels, while intervention risk does not.
        self.assertAlmostEqual(score.compatibility, 0.0)
        self.assertAlmostEqual(score.hns_risk, 1.5)

    def test_importance_budget_is_applied_before_compatibility_gate(self):
        scores = {
            "important_bad": ModuleSensitivityScore(10.0, -1.0, 1.0, 16),
            "important_good": ModuleSensitivityScore(9.0, 0.5, 1.0, 16),
            "outside_good": ModuleSensitivityScore(8.0, 5.0, 1.0, 16),
        }

        selected, annotated = select_important_modules(
            scores,
            module_budget=2,
            require_positive_compatibility=True,
        )

        self.assertEqual(selected, ["important_good"])
        self.assertEqual(
            annotated["important_bad"].rejection_reason,
            "non_positive_hns_compatibility",
        )
        self.assertEqual(
            annotated["outside_good"].rejection_reason,
            "outside_importance_budget",
        )

    def test_importance_only_mode_keeps_full_shortlist(self):
        scores = {
            "a": ModuleSensitivityScore(2.0, -3.0, 1.0, 16),
            "b": ModuleSensitivityScore(1.0, 1.0, 1.0, 16),
        }
        selected, _ = select_important_modules(
            scores,
            module_budget=1,
            require_positive_compatibility=False,
        )
        self.assertEqual(selected, ["a"])


if __name__ == "__main__":
    unittest.main()
