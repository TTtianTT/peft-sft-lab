from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune.data.base import make_single_turn_example
from finetune.data.base import split_chat_messages_for_sft
from finetune.data.chat_sft import (
    decode_supervised_labels,
    ensure_chat_template,
    format_sft_debug_sample,
    preprocess_chat_example,
)
from finetune.eval.eval_gsm8k import (
    _build_gsm8k_user_instruction,
    _resolve_local_gsm8k_data_files,
)
from finetune.eval.eval_humaneval import (
    _normalize_humaneval_problem,
    _resolve_local_humaneval_data_files,
    build_arg_parser as build_humaneval_arg_parser,
    build_humaneval_chat_user_prompt,
    normalize_humaneval_completion,
)
from finetune.eval.eval_ifeval import build_arg_parser as build_ifeval_arg_parser
from finetune.eval.eval_mbpp import (
    _normalize_mbpp_problem,
    _run_mbpp_problem,
    build_arg_parser as build_mbpp_arg_parser,
    build_mbpp_chat_user_prompt,
    normalize_mbpp_completion,
)
from finetune.eval.eval_ifbench import _resolve_local_ifbench_data_files
from finetune.eval.generation import render_chat_prompt
from finetune.data.code_magicoder import MagicoderTask
from finetune.data.instruction_following_tulu import TuluInstructionFollowingTask
from finetune.data.math_metamathqa import MetaMathQATask


class DummyChatTokenizer:
    chat_template = "dummy"
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=None,
    ):
        if tokenize:
            raise NotImplementedError("Dummy tokenizer only supports tokenize=False in these tests.")
        parts = ["<bos>"]
        for message in messages:
            parts.append(f"<|{message['role']}|>\n")
            if message["role"] == "assistant" and enable_thinking is False:
                parts.append("<think>\n\n</think>\n\n")
            parts.append(f"{message['content']}<eot>")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
            if enable_thinking is False:
                parts.append("<think>\n\n</think>\n\n")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in ids)


class MissingTemplateTokenizer:
    chat_template = None

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("apply_chat_template should not be called when chat_template is missing.")


class MetaMathTaskTests(unittest.TestCase):
    def test_metamath_field_mapping_stays_semantic(self):
        task = MetaMathQATask()
        example = task.format_example(
            {
                "query": "What is 2 + 3?",
                "response": "We add 2 and 3 to get 5.",
            }
        )

        self.assertEqual(example["prompt"][0]["role"], "user")
        self.assertEqual(example["prompt"][0]["content"], "What is 2 + 3?")
        self.assertEqual(example["completion"][0]["role"], "assistant")
        self.assertEqual(example["completion"][0]["content"], "We add 2 and 3 to get 5.")

        flattened = str(example)
        self.assertNotIn("Below is an instruction", flattened)
        self.assertNotIn("### Instruction:", flattened)
        self.assertNotIn("### Response:", flattened)
        self.assertNotIn("Let's think step by step.", flattened)

    def test_metamath_can_load_local_json_file(self):
        try:
            import datasets  # noqa: F401
        except Exception as exc:
            self.skipTest(f"datasets unavailable: {exc}")

        payload = [
            {
                "query": "What is 2 + 3?",
                "response": "We add 2 and 3 to get 5.",
                "original_question": "What is 2 + 3?",
                "type": "MATH_AnsAug",
            }
        ]

        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "metamath.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")

            task = MetaMathQATask()
            dataset = task.load("train", dataset_path=str(dataset_path))

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["query"], payload[0]["query"])
            self.assertEqual(dataset[0]["response"], payload[0]["response"])


class MagicoderTaskTests(unittest.TestCase):
    def test_magicoder_field_mapping_stays_semantic(self):
        task = MagicoderTask()
        example = task.format_example(
            {
                "instruction": "Write a Python function that returns 42.",
                "response": "def answer():\n    return 42",
            }
        )

        self.assertEqual(example["prompt"][0]["role"], "user")
        self.assertEqual(example["prompt"][0]["content"], "Write a Python function that returns 42.")
        self.assertEqual(example["completion"][0]["role"], "assistant")
        self.assertEqual(example["completion"][0]["content"], "def answer():\n    return 42")

        flattened = str(example)
        self.assertNotIn("### Instruction:", flattened)
        self.assertNotIn("### Response:", flattened)
        self.assertNotIn("Below is an instruction", flattened)

    def test_magicoder_repo_native_chat_format_round_trips(self):
        task = MagicoderTask()
        example = task.format_example(
            {
                "prompt": [{"role": "user", "content": "Implement fibonacci(n)."}],
                "completion": [{"role": "assistant", "content": "def fibonacci(n):\n    ..."}],
            }
        )

        self.assertEqual(example["prompt"][0]["content"], "Implement fibonacci(n).")
        self.assertEqual(example["completion"][0]["content"], "def fibonacci(n):\n    ...")

    def test_magicoder_can_load_local_json_file(self):
        try:
            import datasets  # noqa: F401
        except Exception as exc:
            self.skipTest(f"datasets unavailable: {exc}")

        payload = [
            {
                "instruction": "Return the larger integer.",
                "response": "def larger(a, b):\n    return a if a > b else b",
            }
        ]

        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "magicoder.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")

            task = MagicoderTask()
            dataset = task.load("train", dataset_path=str(dataset_path))

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["instruction"], payload[0]["instruction"])
            self.assertEqual(dataset[0]["response"], payload[0]["response"])


class InstructionFollowingTaskTests(unittest.TestCase):
    def test_instruction_following_tulu_messages_are_split_semantically(self):
        task = TuluInstructionFollowingTask()
        example = task.format_example(
            {
                "id": "personas_IF_demo",
                "prompt": "Name two rivers in Europe.",
                "messages": [
                    {"role": "user", "content": "Name two rivers in Europe."},
                    {"role": "assistant", "content": "* Danube\n* Rhine"},
                ],
                "constraints": ["format:number of bullet lists"],
            }
        )

        self.assertEqual(example["prompt"], [{"role": "user", "content": "Name two rivers in Europe."}])
        self.assertEqual(example["completion"], [{"role": "assistant", "content": "* Danube\n* Rhine"}])

    def test_instruction_following_repo_native_prompt_completion_format(self):
        task = TuluInstructionFollowingTask()
        example = task.format_example(
            {
                "prompt": [{"role": "user", "content": "Write one sentence about bees."}],
                "completion": [{"role": "assistant", "content": "Bees are vital pollinators."}],
            }
        )

        self.assertEqual(example["prompt"][0]["role"], "user")
        self.assertEqual(example["completion"][0]["role"], "assistant")
        self.assertEqual(example["completion"][0]["content"], "Bees are vital pollinators.")

    def test_instruction_following_can_load_local_json_file(self):
        try:
            import datasets  # noqa: F401
        except Exception as exc:
            self.skipTest(f"datasets unavailable: {exc}")

        payload = [
            {
                "prompt": "List two planets.",
                "response": "Earth and Mars.",
            }
        ]

        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "if.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")

            task = TuluInstructionFollowingTask()
            dataset = task.load("train", dataset_path=str(dataset_path))

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["prompt"], payload[0]["prompt"])
            self.assertEqual(dataset[0]["response"], payload[0]["response"])

    def test_instruction_following_can_load_local_parquet_file(self):
        try:
            from datasets import Dataset
        except Exception as exc:
            self.skipTest(f"datasets unavailable: {exc}")

        payload = [
            {
                "id": "personas_IF_demo",
                "prompt": "Name two planets.",
                "messages": [
                    {"role": "user", "content": "Name two planets."},
                    {"role": "assistant", "content": "Earth and Mars."},
                ],
            }
        ]

        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "if.parquet"
            Dataset.from_list(payload).to_parquet(str(dataset_path))

            task = TuluInstructionFollowingTask()
            dataset = task.load("train", dataset_path=str(dataset_path))

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["prompt"], payload[0]["prompt"])
            self.assertEqual(dataset[0]["messages"][1]["content"], "Earth and Mars.")


class ChatPreprocessingTests(unittest.TestCase):
    def test_missing_chat_template_raises_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "chat_template"):
            ensure_chat_template(MissingTemplateTokenizer(), "dummy-model")

    def test_response_only_labels_mask_prompt_region(self):
        tokenizer = DummyChatTokenizer()
        example = make_single_turn_example(
            user_content="What is 2 + 3?",
            assistant_content="We add 2 and 3 to get 5.",
        )

        sample = preprocess_chat_example(tokenizer=tokenizer, example=example, max_seq_len=4096)

        self.assertGreater(sample["prompt_length"], 0)
        self.assertTrue(all(label == -100 for label in sample["labels"][: sample["prompt_length"]]))
        self.assertTrue(any(label != -100 for label in sample["labels"][sample["prompt_length"] :]))

        full_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
        supervised_text = decode_supervised_labels(tokenizer, sample["labels"])

        self.assertIn("<|user|>", full_input)
        self.assertIn("<|assistant|>", full_input)
        self.assertIn("What is 2 + 3?", full_input)
        self.assertIn("We add 2 and 3 to get 5.", full_input)

        self.assertNotIn("What is 2 + 3?", supervised_text)
        self.assertIn("We add 2 and 3 to get 5.", supervised_text)

        debug_text = format_sft_debug_sample(tokenizer, sample)
        self.assertIn("===== SFT SAMPLE =====", debug_text)
        self.assertIn("Supervised labels:", debug_text)

    def test_truncation_can_drop_all_supervised_tokens(self):
        tokenizer = DummyChatTokenizer()
        example = make_single_turn_example(
            user_content="x" * 200,
            assistant_content="answer",
        )

        sample = preprocess_chat_example(tokenizer=tokenizer, example=example, max_seq_len=20)

        self.assertEqual(sample["supervised_token_count"], 0)
        self.assertTrue(all(label == -100 for label in sample["labels"]))

    def test_qwen_non_thinking_marker_is_part_of_masked_prompt(self):
        tokenizer = DummyChatTokenizer()
        example = make_single_turn_example(
            user_content="What is 2 + 3?",
            assistant_content="5",
        )

        sample = preprocess_chat_example(
            tokenizer=tokenizer,
            example=example,
            max_seq_len=4096,
            chat_template_mode="non_thinking",
        )
        prompt_text = tokenizer.decode(
            sample["input_ids"][: sample["prompt_length"]],
            skip_special_tokens=False,
        )
        supervised_text = decode_supervised_labels(tokenizer, sample["labels"])

        self.assertIn("<think>\n\n</think>", prompt_text)
        self.assertNotIn("<think>", supervised_text)
        self.assertIn("5", supervised_text)

    def test_split_chat_messages_requires_final_assistant_turn(self):
        with self.assertRaisesRegex(ValueError, "end with an assistant message"):
            split_chat_messages_for_sft(
                [
                    {"role": "user", "content": "Say hello."},
                    {"role": "user", "content": "Actually say hi."},
                ]
            )


class GSM8KEvalPromptTests(unittest.TestCase):
    def test_gsm8k_instruction_keeps_answer_format_contract(self):
        instruction = _build_gsm8k_user_instruction("What is 2 + 3?")
        self.assertIn("Solve the following math word problem.", instruction)
        self.assertIn("#### <answer>", instruction)
        self.assertIn("What is 2 + 3?", instruction)
        self.assertNotIn("### Instruction:", instruction)
        self.assertNotIn("Let's think step by step.", instruction)

    def test_gsm8k_eval_prompt_uses_chat_template_not_metamath_string_template(self):
        prompt = render_chat_prompt(
            tokenizer=DummyChatTokenizer(),
            base_model="dummy-model",
            user_content=_build_gsm8k_user_instruction("What is 2 + 3?"),
        )

        self.assertIn("<|user|>", prompt)
        self.assertIn("<|assistant|>", prompt)
        self.assertIn("#### <answer>", prompt)
        self.assertNotIn("### Instruction:", prompt)
        self.assertNotIn("### Response:", prompt)
        self.assertNotIn("Let's think step by step.", prompt)

    def test_gsm8k_eval_can_explicitly_disable_qwen_thinking(self):
        prompt = render_chat_prompt(
            tokenizer=DummyChatTokenizer(),
            base_model="Qwen/Qwen3-8B",
            user_content=_build_gsm8k_user_instruction("What is 2 + 3?"),
            chat_template_mode="non_thinking",
        )

        self.assertTrue(prompt.endswith("<|assistant|>\n<think>\n\n</think>\n\n"))

    def test_resolve_local_gsm8k_snapshot_directory(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "main"
            split_dir.mkdir(parents=True, exist_ok=True)
            target = split_dir / "test-00000-of-00001.parquet"
            target.write_text("placeholder", encoding="utf-8")

            loader_name, files = _resolve_local_gsm8k_data_files(
                dataset_path=str(root),
                split="test",
                dataset_config="main",
            )

            self.assertEqual(loader_name, "parquet")
            self.assertEqual(files, [str(target)])


class IFBenchEvalDataTests(unittest.TestCase):
    def test_resolve_local_ifbench_snapshot_directory(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            target = data_dir / "train-00000-of-00001.parquet"
            target.write_text("placeholder", encoding="utf-8")

            loader_name, files = _resolve_local_ifbench_data_files(
                dataset_path=str(root),
                split="train",
            )

            self.assertEqual(loader_name, "parquet")
            self.assertEqual(files, [str(target)])

    def test_resolve_local_ifbench_single_jsonl_file(self):
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "ifbench-train.jsonl"
            target.write_text('{"key":"0","prompt":"hello","instruction_id_list":[],"kwargs":[]}\n', encoding="utf-8")

            loader_name, files = _resolve_local_ifbench_data_files(
                dataset_path=str(target),
                split="train",
            )

            self.assertEqual(loader_name, "json")
            self.assertEqual(files, [str(target)])

    def test_resolve_local_gsm8k_single_parquet_file(self):
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "gsm8k-test.parquet"
            target.write_text("placeholder", encoding="utf-8")

            loader_name, files = _resolve_local_gsm8k_data_files(
                dataset_path=str(target),
                split="test",
                dataset_config="main",
            )

            self.assertEqual(loader_name, "parquet")
            self.assertEqual(files, [str(target)])


class HumanEvalDataTests(unittest.TestCase):
    def test_resolve_local_humaneval_snapshot_directory(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "openai_humaneval"
            split_dir.mkdir(parents=True, exist_ok=True)
            target = split_dir / "test-00000-of-00001.parquet"
            target.write_text("placeholder", encoding="utf-8")

            loader_name, files = _resolve_local_humaneval_data_files(
                dataset_path=str(root),
                split="test",
            )

            self.assertEqual(loader_name, "parquet")
            self.assertEqual(files, [str(target)])

    def test_normalize_humaneval_problem_accepts_hf_fields(self):
        normalized = _normalize_humaneval_problem(
            {
                "task_id": "HumanEval/0",
                "prompt": "def return1():\n",
                "canonical_solution": "    return 1",
                "test": "def check(candidate):\n    assert candidate() == 1",
                "entry_point": "return1",
            }
        )

        self.assertEqual(normalized["task_id"], "HumanEval/0")
        self.assertEqual(normalized["entry_point"], "return1")
        self.assertIn("def return1():", normalized["prompt"])

class OptionalLlamaTokenizerTests(unittest.TestCase):
    def test_llama31_chat_template_when_available(self):
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            self.skipTest(f"transformers unavailable: {exc}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                use_fast=True,
                local_files_only=True,
            )
        except Exception as exc:
            self.skipTest(f"Llama-3.1 tokenizer unavailable locally: {exc}")

        ensure_chat_template(tokenizer, "meta-llama/Llama-3.1-8B-Instruct")
        example = make_single_turn_example(
            user_content="What is 2 + 3?",
            assistant_content="We add 2 and 3 to get 5.",
        )

        sample = preprocess_chat_example(tokenizer=tokenizer, example=example, max_seq_len=4096)
        full_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
        supervised_text = decode_supervised_labels(tokenizer, sample["labels"])

        self.assertIn("user", full_input)
        self.assertIn("assistant", full_input)
        self.assertIn("We add 2 and 3 to get 5.", supervised_text)
        self.assertNotIn("What is 2 + 3?", supervised_text)
        self.assertNotIn("### Instruction:", full_input)


class IFEvalCliTests(unittest.TestCase):
    def test_ifeval_cli_accepts_qwen_non_thinking_mode(self):
        args = build_ifeval_arg_parser().parse_args(
            [
                "--base_model",
                "Qwen/Qwen3-8B",
                "--output_dir",
                "eval/ifeval-qwen3",
                "--chat_template_mode",
                "non_thinking",
            ]
        )

        self.assertEqual(args.chat_template_mode, "non_thinking")


class HumanEvalChatPromptTests(unittest.TestCase):
    def test_humaneval_opencompass_user_prompt_matches_requested_template(self):
        problem_prompt = "def answer():\n    pass\n"

        rendered = build_humaneval_chat_user_prompt(
            problem_prompt,
            style="opencompass",
        )

        self.assertEqual(
            rendered,
            "Complete the following python code:\ndef answer():\n    pass\n",
        )
        self.assertNotIn("Return only the missing continuation", rendered)
        self.assertNotIn("Do not repeat the prefix", rendered)

    def test_humaneval_cli_accepts_opencompass_user_prompt_style(self):
        args = build_humaneval_arg_parser().parse_args(
            [
                "--base_model",
                "Qwen/Qwen3-8B",
                "--output_dir",
                "eval/humaneval-qwen3-opencompass",
                "--chat_user_prompt_style",
                "opencompass",
            ]
        )

        self.assertEqual(args.chat_user_prompt_style, "opencompass")

    def test_humaneval_cli_accepts_qwen_non_thinking_mode(self):
        args = build_humaneval_arg_parser().parse_args(
            [
                "--base_model",
                "Qwen/Qwen3-8B",
                "--output_dir",
                "eval/humaneval-qwen3",
                "--chat_template_mode",
                "non_thinking",
            ]
        )

        self.assertEqual(args.prompt_style, "chat")
        self.assertEqual(args.chat_template_mode, "non_thinking")

    def test_humaneval_qwen_prompt_can_disable_thinking(self):
        rendered = render_chat_prompt(
            tokenizer=DummyChatTokenizer(),
            base_model="Qwen/Qwen3-8B",
            user_content=build_humaneval_chat_user_prompt("def answer():\n    pass\n"),
            chat_template_mode="non_thinking",
        )

        self.assertTrue(rendered.endswith("<|assistant|>\n<think>\n\n</think>\n\n"))

    def test_llamafactory_llama3_prompt_is_single_user_turn(self):
        problem_prompt = "def answer():\n    \"\"\"Return 42.\"\"\"\n"
        user_content = build_humaneval_chat_user_prompt(problem_prompt)
        rendered = render_chat_prompt(
            tokenizer=DummyChatTokenizer(),
            base_model="dummy-llama3",
            user_content=user_content,
            system_content=None,
        )

        self.assertIn("<|user|>", rendered)
        self.assertIn("<|assistant|>", rendered)
        self.assertNotIn("<|system|>", rendered)
        self.assertIn(problem_prompt, rendered)

    def test_humaneval_normalizer_removes_repeated_function_signature(self):
        problem_prompt = "def answer():\n    \"\"\"Return 42.\"\"\"\n"
        completion = normalize_humaneval_completion(
            "```python\ndef answer():\n    return 42\n```",
            problem_prompt,
            "answer",
        )

        self.assertEqual(completion, "    return 42")


class MBPPEvaluationTests(unittest.TestCase):
    def test_mbpp_cli_accepts_qwen_non_thinking_mode(self):
        args = build_mbpp_arg_parser().parse_args(
            [
                "--base_model",
                "Qwen/Qwen3-8B",
                "--output_dir",
                "eval/mbpp-qwen3",
                "--chat_template_mode",
                "non_thinking",
            ]
        )

        self.assertEqual(args.prompt_style, "chat")
        self.assertEqual(args.chat_template_mode, "non_thinking")

    def test_mbpp_qwen_prompt_can_disable_thinking(self):
        rendered = render_chat_prompt(
            tokenizer=DummyChatTokenizer(),
            base_model="Qwen/Qwen3-8B",
            user_content=build_mbpp_chat_user_prompt(
                "Write answer().",
                "assert answer() == 42",
            ),
            chat_template_mode="non_thinking",
        )

        self.assertTrue(rendered.endswith("<|assistant|>\n<think>\n\n</think>\n\n"))

    def test_normalize_mbpp_problem_accepts_standard_fields(self):
        problem = _normalize_mbpp_problem(
            {
                "task_id": 7,
                "text": "Write a function that returns 42.",
                "test_list": ["assert answer() == 42"],
                "test_imports": ["import math"],
            }
        )

        self.assertEqual(problem["task_id"], "7")
        self.assertEqual(problem["test_setup_code"], "import math")
        self.assertEqual(problem["test_list"], ["assert answer() == 42"])

    def test_mbpp_chat_prompt_is_single_user_turn(self):
        rendered = render_chat_prompt(
            tokenizer=DummyChatTokenizer(),
            base_model="dummy-llama3",
            user_content=build_mbpp_chat_user_prompt("Write answer().", "assert answer() == 42"),
            system_content=None,
        )

        self.assertIn("<|user|>", rendered)
        self.assertIn("<|assistant|>", rendered)
        self.assertNotIn("<|system|>", rendered)
        self.assertIn("assert answer() == 42", rendered)

    def test_mbpp_normalizes_fenced_code_and_executes_tests(self):
        completion = normalize_mbpp_completion("```python\ndef answer():\n    return 42\n```")
        result = _run_mbpp_problem(
            {"test_setup_code": "", "test_list": ["assert answer() == 42"]},
            completion,
            timeout_s=1.0,
        )

        self.assertEqual(completion, "def answer():\n    return 42")
        self.assertTrue(result["passed"])

    def test_mbpp_normalizes_unfenced_prose_prefix(self):
        completion = normalize_mbpp_completion("Here is the solution:\n\ndef answer():\n    return 42")

        self.assertEqual(completion, "def answer():\n    return 42")


if __name__ == "__main__":
    unittest.main()
