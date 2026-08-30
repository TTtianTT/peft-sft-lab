from __future__ import annotations

import types
import unittest

import torch

from finetune.spectral_edit.runtime import (
    enable_deterministic_gradient_checkpointing,
    saved_tensor_offload_context,
)


class _CheckpointableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.config = types.SimpleNamespace(use_cache=True)
        self.checkpointing_kwargs = None
        self.input_grads_enabled = False

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs=None):
        self.checkpointing_kwargs = gradient_checkpointing_kwargs

    def enable_input_require_grads(self):
        self.input_grads_enabled = True


class SpectralRuntimeTests(unittest.TestCase):
    def test_checkpointing_keeps_dropout_disabled(self):
        model = _CheckpointableModel()
        model.eval()

        enable_deterministic_gradient_checkpointing(model)

        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assertEqual(model.checkpointing_kwargs, {"use_reentrant": False})
        self.assertTrue(model.input_grads_enabled)
        self.assertFalse(model.config.use_cache)

    def test_unsupported_checkpointing_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            enable_deterministic_gradient_checkpointing(torch.nn.Linear(2, 2))

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
