from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune.spectral_edit.calib import (
    _resolve_local_calibration_data_files,
    build_calib_formatter,
    make_chat_calib_batch,
)


class _Encoding(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class _FakeChatTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        del tokenize, kwargs
        user = messages[0]["content"]
        prefix = f"U:{user}|A:"
        if add_generation_prompt:
            return prefix
        return prefix + messages[-1]["content"]

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        encoded = _Encoding(input_ids=[ord(char) + 1 for char in text])
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded


class SpectralEditCalibrationTests(unittest.TestCase):
    def test_resolve_local_calibration_snapshot_directory(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "main"
            split_dir.mkdir(parents=True, exist_ok=True)
            target = split_dir / "test-00000-of-00001.parquet"
            target.write_text("placeholder", encoding="utf-8")

            loader_name, files = _resolve_local_calibration_data_files(
                dataset_path=str(root),
                split="test",
                dataset_config="main",
            )

            self.assertEqual(loader_name, "parquet")
            self.assertEqual(files, [str(target)])

    def test_chat_calibration_batch_masks_user_prompt(self):
        tokenizer = _FakeChatTokenizer()
        input_ids, attention_mask, labels = make_chat_calib_batch(
            tokenizer,
            [{"question": "Q", "answer": "A"}],
            lambda example: (example["question"], example["answer"]),
            chat_template_mode="auto",
            max_seq_len=32,
        )
        self.assertEqual(input_ids.shape, labels.shape)
        self.assertEqual(attention_mask.shape, labels.shape)
        supervised = labels[labels != -100]
        self.assertGreater(supervised.numel(), 0)

    def test_commonsense170k_formatter_accepts_training_field_aliases(self):
        formatter, fields = build_calib_formatter("commonsense170k", None)
        self.assertIsNone(fields)
        self.assertEqual(
            formatter({"instruction": "question", "output": "answer"}),
            ("question", "answer"),
        )


if __name__ == "__main__":
    unittest.main()
