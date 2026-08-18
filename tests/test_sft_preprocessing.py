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
from finetune.data.chat_sft import (
    decode_supervised_labels,
    ensure_chat_template,
    format_sft_debug_sample,
    preprocess_chat_example,
)
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
