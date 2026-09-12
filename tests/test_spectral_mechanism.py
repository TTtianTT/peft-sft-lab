from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune.spectral_edit.mechanism import (
    align_edited_spectrum_to_reference,
    build_causal_control_spectra,
    frobenius_norm_from_factors,
)


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _Tokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return _Encoding([1, 2, 3, 4])


class _BatchEncoding(dict):
    def __getitem__(self, key):
        if isinstance(key, int):
            return _Encoding(super().__getitem__("input_ids"))
        return super().__getitem__(key)


class SpectralMechanismTests(unittest.TestCase):
    def test_activation_render_accepts_tokenizers_encoding(self):
        import importlib.util

        path = ROOT / "scripts" / "measure_lora_modification.py"
        spec = importlib.util.spec_from_file_location("measure_lora_modification", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rendered = module.render(
            _Tokenizer(),
            {"instruction": "question", "response": "answer"},
            "magicoder",
            max_seq_len=3,
        )
        self.assertEqual(rendered, [1, 2, 3])

    def test_activation_normalizer_accepts_batch_encoding(self):
        import importlib.util

        path = ROOT / "scripts" / "measure_lora_modification.py"
        spec = importlib.util.spec_from_file_location("measure_lora_modification", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        encoded = _BatchEncoding(input_ids=[5, 6, 7])
        self.assertEqual(module.normalize_token_ids(encoded), [5, 6, 7])

    def test_factor_frobenius_matches_dense(self):
        torch.manual_seed(0)
        b = torch.randn(9, 3)
        a = torch.randn(3, 7)
        self.assertAlmostEqual(
            float(frobenius_norm_from_factors(b, a)),
            float(torch.linalg.matrix_norm(b @ a)),
            places=5,
        )

    def test_alignment_recovers_reference_basis_spectrum(self):
        torch.manual_seed(1)
        u, _ = torch.linalg.qr(torch.randn(10, 4))
        v, _ = torch.linalg.qr(torch.randn(8, 4))
        sigma = torch.tensor([3.0, 2.0, 1.0, 0.5])
        root = sigma.sqrt()
        b = u * root
        a = root[:, None] * v.T
        aligned, stats = align_edited_spectrum_to_reference(u, v.T, b, a)
        self.assertTrue(torch.allclose(aligned, sigma, atol=1e-5))
        self.assertLess(stats["basis_offdiag_fraction"], 1e-6)
        self.assertLess(stats["basis_projection_residual_fraction"], 1e-6)

    def test_causal_controls_have_intended_invariants(self):
        lora = torch.tensor([8.0, 2.0, 0.5, 0.1])
        hns = torch.tensor([3.0, 2.5, 2.0, 1.5])
        controls, _ = build_causal_control_spectra(lora, hns)
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(controls["scalar_shrink"]), torch.linalg.vector_norm(hns)))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(controls["shape_only"]), torch.linalg.vector_norm(lora)))
        self.assertTrue(torch.equal(controls["head_only"], torch.minimum(lora, hns)))
        self.assertTrue(torch.equal(controls["tail_only"], torch.maximum(lora, hns)))


if __name__ == "__main__":
    unittest.main()
