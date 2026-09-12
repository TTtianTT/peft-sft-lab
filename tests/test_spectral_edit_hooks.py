from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune.spectral_edit.hooks import (  # noqa: E402
    HOOK_CTX,
    ModuleSpec,
    register_sigma_hooks,
    remove_hooks,
)


class SpectralEditHookTests(unittest.TestCase):
    def test_projected_activation_cache_matches_sigma_gradient_formula(self):
        torch.manual_seed(7)
        module = torch.nn.Linear(3, 4, bias=False)
        U, _ = torch.linalg.qr(torch.randn(4, 2), mode="reduced")
        V, _ = torch.linalg.qr(torch.randn(3, 2), mode="reduced")
        scaling = 0.5
        spec = ModuleSpec(
            module_prefix="linear",
            module=module,
            U=U,
            V=V,
            Vh=V.t(),
            sigma0=torch.ones(2),
            scaling=scaling,
            adapter=None,
        )
        x = torch.randn(2, 3, requires_grad=True)
        output_weight = torch.randn(2, 4)
        expected = scaling * ((output_weight @ U) * (x.detach() @ V)).sum(dim=0)

        HOOK_CTX.reset()
        handles = register_sigma_hooks({"linear": spec})
        try:
            loss = (module(x) * output_weight).sum()
            loss.backward()
        finally:
            remove_hooks(handles)

        self.assertTrue(torch.allclose(HOOK_CTX.gsum["linear"], expected, atol=1e-6))
        self.assertIsNone(getattr(module, "__xv_cache", None))

    def test_optional_energy_capture_respects_attention_mask(self):
        torch.manual_seed(11)
        module = torch.nn.Linear(3, 4, bias=False)
        U, _ = torch.linalg.qr(torch.randn(4, 2), mode="reduced")
        V, _ = torch.linalg.qr(torch.randn(3, 2), mode="reduced")
        spec = ModuleSpec(
            module_prefix="linear",
            module=module,
            U=U,
            V=V,
            Vh=V.t(),
            sigma0=torch.tensor([2.0, 0.5]),
            scaling=1.0,
            adapter=None,
        )
        x = torch.randn(2, 3, 3, requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        projected = (x.detach() * mask.unsqueeze(-1)) @ V
        expected = projected.square().sum(dim=(0, 1))

        HOOK_CTX.reset()
        HOOK_CTX.attn_mask = mask
        handles = register_sigma_hooks({"linear": spec}, capture_energy=True)
        try:
            module(x).sum().backward()
        finally:
            remove_hooks(handles)

        self.assertEqual(HOOK_CTX.energy_count["linear"], 3)
        self.assertTrue(torch.allclose(HOOK_CTX.energy_sum["linear"], expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
