from __future__ import annotations

import unittest

import torch

from scripts.build_commonsense_tthns_adapters import _restore_probabilities, _tt_objective


class CommonsenseTTHNSBuilderTests(unittest.TestCase):
    def test_restore_probabilities_is_differentiable(self):
        displayed = torch.tensor([[[0.1, 0.6, 0.3]]], requires_grad=True)
        restored = _restore_probabilities(displayed, (((2, 0, 1),),))
        self.assertTrue(torch.allclose(restored, torch.tensor([[[0.6, 0.3, 0.1]]])))
        restored.sum().backward()
        self.assertIsNotNone(displayed.grad)

    def test_tt_objective_prefers_confident_consistent_views(self):
        uncertain = torch.tensor([[[0.50, 0.50], [0.50, 0.50]]])
        consistent = torch.tensor([[[0.90, 0.10], [0.88, 0.12]]])
        self.assertLess(
            float(_tt_objective(consistent, js_weight=1.0)),
            float(_tt_objective(uncertain, js_weight=1.0)),
        )


if __name__ == "__main__":
    unittest.main()
