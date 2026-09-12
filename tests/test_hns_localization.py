from __future__ import annotations

import sys
import unittest
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_hns_localization_adapters import selected
from build_functional_compatibility_adapters import centered_ranks, quotas


class HNSLocalizationSelectionTests(unittest.TestCase):
    def test_attention_and_mlp_are_disjoint(self):
        attention = "base.model.layers.7.self_attn.q_proj"
        mlp = "base.model.layers.7.mlp.down_proj"
        self.assertTrue(selected("hns-attention", attention, 7, 36))
        self.assertFalse(selected("hns-mlp", attention, 7, 36))
        self.assertTrue(selected("hns-mlp", mlp, 7, 36))
        self.assertFalse(selected("hns-attention", mlp, 7, 36))

    def test_fine_module_groups_partition_coarse_groups(self):
        self.assertTrue(selected("hns-qkv", "x.q_proj", 0, 36))
        self.assertFalse(selected("hns-qkv", "x.o_proj", 0, 36))
        self.assertTrue(selected("hns-o-proj", "x.o_proj", 0, 36))
        self.assertTrue(selected("hns-gate-up", "x.gate_proj", 0, 36))
        self.assertFalse(selected("hns-gate-up", "x.down_proj", 0, 36))
        self.assertTrue(selected("hns-down-proj", "x.down_proj", 0, 36))

    def test_layer_thirds_cover_36_layers_without_overlap(self):
        labels = ("hns-layers-early", "hns-layers-middle", "hns-layers-late")
        for layer in range(36):
            matches = [selected(label, "x.q_proj", layer, 36) for label in labels]
            self.assertEqual(sum(matches), 1)
        self.assertTrue(selected("hns-layers-early", "x.q_proj", 11, 36))
        self.assertTrue(selected("hns-layers-middle", "x.q_proj", 12, 36))
        self.assertTrue(selected("hns-layers-late", "x.q_proj", 24, 36))

    def test_fc_quota_is_exact_and_structure_matched(self):
        strata = {
            (0, "q_proj"): [f"q{index}" for index in range(9)],
            (0, "o_proj"): [f"o{index}" for index in range(9)],
            (1, "q_proj"): [f"m{index}" for index in range(8)],
        }
        quarter = quotas(strata, 0.25)
        half = quotas(strata, 0.50)
        self.assertEqual(sum(quarter.values()), math.floor(26 * 0.25 + 0.5))
        self.assertEqual(sum(half.values()), math.floor(26 * 0.50 + 0.5))
        self.assertTrue(all(0 <= quarter[key] <= len(strata[key]) for key in strata))
        self.assertTrue(all(0 <= half[key] <= len(strata[key]) for key in strata))

    def test_fc_centered_ranks_have_fixed_endpoints(self):
        modules = ["a", "b", "c"]
        ranked = centered_ranks(modules, {"a": 2.0, "b": -1.0, "c": 0.0})
        self.assertEqual(ranked["b"], -1.0)
        self.assertEqual(ranked["c"], 0.0)
        self.assertEqual(ranked["a"], 1.0)


if __name__ == "__main__":
    unittest.main()
