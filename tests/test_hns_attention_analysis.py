from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_hns_attention_activations import reduce_attention, select_flip_tasks


class HNSAttentionAnalysisTests(unittest.TestCase):
    def test_select_flip_tasks_balances_categories(self):
        ref = {"a": False, "b": True, "c": True, "d": False}
        hns = {"a": True, "b": False, "c": True, "d": False}
        self.assertEqual(
            select_flip_tasks(ref, hns, max_flip=2, max_stable=2),
            {"a": "gain", "b": "loss", "c": "stable_pass", "d": "stable_fail"},
        )

    def test_reduce_attention_returns_per_head_values(self):
        attention = torch.zeros(1, 2, 4, 4)
        for query in range(4):
            attention[:, :, query, : query + 1] = 1.0 / (query + 1)
        entropy, prompt_mass, top1 = reduce_attention(attention, prefix_len=2)
        self.assertEqual(entropy.shape, (2,))
        self.assertTrue(torch.all((entropy >= 0) & (entropy <= 1)))
        self.assertTrue(torch.all((prompt_mass >= 0) & (prompt_mass <= 1)))
        self.assertTrue(torch.all((top1 >= 0) & (top1 <= 1)))


if __name__ == "__main__":
    unittest.main()
