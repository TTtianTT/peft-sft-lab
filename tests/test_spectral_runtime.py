from __future__ import annotations

import unittest

import torch

from finetune.spectral_edit.runtime import saved_tensor_offload_context


class SpectralRuntimeTests(unittest.TestCase):
    def test_disabled_context_preserves_gradients(self):
        value = torch.tensor([2.0], requires_grad=True)
        with saved_tensor_offload_context(False):
            loss = value.square().sum()
        loss.backward()
        self.assertTrue(torch.equal(value.grad, torch.tensor([4.0])))

    def test_cpu_offload_context_preserves_gradients(self):
        value = torch.tensor([3.0], requires_grad=True)
        with saved_tensor_offload_context(True):
            loss = value.square().sum()
        loss.backward()
        self.assertTrue(torch.equal(value.grad, torch.tensor([6.0])))


if __name__ == "__main__":
    unittest.main()
