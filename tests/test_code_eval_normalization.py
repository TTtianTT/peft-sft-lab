from __future__ import annotations

import ast
import unittest

from finetune.eval.eval_humaneval import normalize_humaneval_completion
from finetune.eval.eval_mbpp import normalize_mbpp_completion


HUMANEVAL_PROMPT = '''def increment(value: int) -> int:
    """Return value plus one."""
'''


class HumanEvalNormalizationTests(unittest.TestCase):
    def assert_program_parses(self, completion: str) -> None:
        ast.parse(HUMANEVAL_PROMPT + completion)

    def test_preserves_first_line_indentation(self) -> None:
        completion = normalize_humaneval_completion(
            "\n    return value + 1\n",
            HUMANEVAL_PROMPT,
            "increment",
        )
        self.assertEqual(completion, "    return value + 1")
        self.assert_program_parses(completion)

    def test_extracts_fenced_continuation_without_losing_indentation(self) -> None:
        completion = normalize_humaneval_completion(
            "Here is the solution:\n```python\n    return value + 1\n```\nThis increments it.",
            HUMANEVAL_PROMPT,
            "increment",
        )
        self.assertEqual(completion, "    return value + 1")
        self.assert_program_parses(completion)

    def test_recovers_unfenced_leading_and_trailing_prose(self) -> None:
        completion = normalize_humaneval_completion(
            "Here is the continuation:\n    return value + 1\nThis solves the task.",
            HUMANEVAL_PROMPT,
            "increment",
        )
        self.assertEqual(completion, "    return value + 1")
        self.assert_program_parses(completion)

    def test_accepts_py_fence(self) -> None:
        completion = normalize_humaneval_completion(
            "```py\n    return value + 1\n```",
            HUMANEVAL_PROMPT,
            "increment",
        )
        self.assertEqual(completion, "    return value + 1")


class MbppNormalizationTests(unittest.TestCase):
    def test_extracts_fenced_code_and_discards_explanation(self) -> None:
        completion = normalize_mbpp_completion(
            "Explanation first.\n```python\ndef increment(value):\n    return value + 1\n```\nMore prose."
        )
        self.assertEqual(completion, "def increment(value):\n    return value + 1")
        ast.parse(completion)

    def test_recovers_unfenced_code_between_prose(self) -> None:
        completion = normalize_mbpp_completion(
            "Here is the code:\ndef increment(value):\n    return value + 1\nThis solves the task."
        )
        self.assertEqual(completion, "def increment(value):\n    return value + 1")
        ast.parse(completion)


if __name__ == "__main__":
    unittest.main()
