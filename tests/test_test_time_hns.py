from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune.eval.test_time_hns import (
    make_choice_permutations,
    restore_original_choice_order,
    select_candidate,
    select_candidate_grouped,
)


class TestTimeHNSTests(unittest.TestCase):
    def test_permutations_are_unique_and_start_with_identity(self):
        values = make_choice_permutations(4, 8, seed=42)
        self.assertEqual(values[0], (0, 1, 2, 3))
        self.assertEqual(len(values), 8)
        self.assertEqual(len(set(values)), 8)

    def test_probability_mapping_restores_original_choice_order(self):
        displayed = torch.tensor([0.1, 0.6, 0.2, 0.1])
        permutation = (2, 0, 3, 1)
        restored = restore_original_choice_order(displayed, permutation)
        self.assertTrue(torch.allclose(restored, torch.tensor([0.6, 0.1, 0.1, 0.2])))

    def test_selector_accepts_more_confident_consistent_candidate(self):
        reference = torch.tensor(
            [[[0.60, 0.40], [0.55, 0.45]], [[0.45, 0.55], [0.40, 0.60]]]
        )
        improved = torch.tensor(
            [[[0.72, 0.28], [0.71, 0.29]], [[0.28, 0.72], [0.29, 0.71]]]
        )
        result = select_candidate(
            {"lora": reference, "hns": improved},
            reference_name="lora",
            max_reference_kl=1.0,
        )
        self.assertEqual(result.selected_name, "hns")
        self.assertTrue(result.scores["hns"].eligible)

    def test_selector_falls_back_when_candidate_leaves_kl_trust_region(self):
        reference = torch.tensor([[[0.95, 0.05], [0.95, 0.05]]])
        flipped = torch.tensor([[[0.01, 0.99], [0.01, 0.99]]])
        result = select_candidate(
            {"lora": reference, "hns": flipped},
            reference_name="lora",
            max_reference_kl=0.01,
        )
        self.assertEqual(result.selected_name, "lora")
        self.assertFalse(result.scores["hns"].eligible)
        self.assertEqual(result.scores["hns"].rejection_reason, "reference_kl_exceeds_limit")

    def test_grouped_selector_supports_variable_choice_counts(self):
        reference = [
            torch.tensor([[[0.60, 0.40], [0.55, 0.45]]]),
            torch.tensor([[[0.50, 0.30, 0.20], [0.45, 0.35, 0.20]]]),
        ]
        improved = [
            torch.tensor([[[0.75, 0.25], [0.74, 0.26]]]),
            torch.tensor([[[0.70, 0.20, 0.10], [0.68, 0.22, 0.10]]]),
        ]
        result = select_candidate_grouped(
            {"lora": reference, "hns": improved},
            reference_name="lora",
            max_reference_kl=1.0,
        )
        self.assertEqual(result.selected_name, "hns")


if __name__ == "__main__":
    unittest.main()
