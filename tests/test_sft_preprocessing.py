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
from finetune.eval.eval_ifbench import _resolve_local_ifbench_data_files
from finetune.eval.generation import render_chat_prompt
from finetune.data.instruction_following_tulu import TuluInstructionFollowingTask
from finetune.data.math_metamathqa import MetaMathQATask


class DummyChatTokenizer:
    chat_template = "dummy"
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if tokenize:
            raise NotImplementedError("Dummy tokenizer only supports tokenize=False in these tests.")
        parts = ["<bos>"]
        for message in messages:
            parts.append(f"<|{message['role']}|>\n{message['content']}<eot>")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
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


if __name__ == "__main__":
    unittest.main()
