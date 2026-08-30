from __future__ import annotations

import unittest

import torch

from scripts.build_code_tthns_adapter import (
    _code_tt_objective,
    _select_problem_prompts,
)


class CodeTTHNSBuilderTests(unittest.TestCase):
    def test_objective_prefers_confident_consistent_prompt_views(self):
        uncertain = torch.tensor([[[0.50, 0.50], [0.50, 0.50]]])
        consistent = torch.tensor([[[0.95, 0.05], [0.90, 0.10]]])
        self.assertLess(
            float(_code_tt_objective(consistent, js_weight=1.0)),
            float(_code_tt_objective(uncertain, js_weight=1.0)),
        )

    def test_objective_penalizes_disagreement_between_prompt_views(self):
        consistent = torch.tensor([[[0.90, 0.10], [0.90, 0.10]]])
        disagreeing = torch.tensor([[[0.90, 0.10], [0.10, 0.90]]])
        self.assertLess(
            float(_code_tt_objective(consistent, js_weight=1.0)),
            float(_code_tt_objective(disagreeing, js_weight=1.0)),
        )

    def test_problem_selection_is_deterministic_and_prompt_only(self):
        problems = {
            f"HumanEval/{index}": {
                "prompt": f"def f{index}():\n",
                "canonical_solution": f"    return {index}\n",
                "test": f"assert f{index}() == {index}",
                "entry_point": f"f{index}",
                "task_id": f"HumanEval/{index}",
            }
            for index in range(10)
        }
        first_ids, first_prompts = _select_problem_prompts(
            problems,
            selection_samples=4,
            seed=42,
        )
        second_ids, second_prompts = _select_problem_prompts(
            problems,
            selection_samples=4,
            seed=42,
        )
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first_prompts, second_prompts)
        self.assertTrue(all(prompt.startswith("def ") for prompt in first_prompts))
        self.assertTrue(all("return" not in prompt and "assert" not in prompt for prompt in first_prompts))


if __name__ == "__main__":
    unittest.main()
