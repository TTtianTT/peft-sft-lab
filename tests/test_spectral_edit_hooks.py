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


if __name__ == "__main__":
    unittest.main()
