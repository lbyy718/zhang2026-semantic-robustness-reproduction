from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from semantic_robustness.semantic_evaluator import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    SEMANTIC_EVALUATOR_ARCHITECTURE,
    build_cifar10_resnet18,
    load_frozen_cifar10_resnet18,
)


class SemanticEvaluatorTests(unittest.TestCase):
    def test_cifar_resnet18_architecture_and_internal_normalization(self) -> None:
        model = build_cifar10_resnet18()
        self.assertEqual(model.backbone.conv1.kernel_size, (3, 3))
        self.assertEqual(model.backbone.conv1.stride, (1, 1))
        self.assertIsInstance(model.backbone.maxpool, nn.Identity)
        torch.testing.assert_close(
            model.input_mean.flatten(), torch.tensor(CIFAR10_MEAN)
        )
        torch.testing.assert_close(
            model.input_std.flatten(), torch.tensor(CIFAR10_STD)
        )
        model.eval()
        self.assertEqual(model(torch.rand(2, 3, 32, 32)).shape, (2, 10))

    def test_frozen_checkpoint_preserves_input_gradient(self) -> None:
        torch.manual_seed(7)
        source = build_cifar10_resnet18()
        payload = {
            "architecture": SEMANTIC_EVALUATOR_ARCHITECTURE,
            "num_classes": 10,
            "model_state": source.state_dict(),
            "qualified": True,
            "test_accuracy": 0.91,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint_best.pt"
            torch.save(payload, path)
            model, metadata = load_frozen_cifar10_resnet18(
                path, require_qualified=True
            )

        self.assertTrue(metadata["qualified"])
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        images = torch.rand(1, 3, 32, 32, requires_grad=True)
        model(images).square().mean().backward()
        self.assertIsNotNone(images.grad)
        self.assertTrue(bool(torch.isfinite(images.grad).all()))
        self.assertGreater(float(images.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_quality_gate_can_be_required(self) -> None:
        model = build_cifar10_resnet18()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unqualified.pt"
            torch.save(
                {
                    "architecture": SEMANTIC_EVALUATOR_ARCHITECTURE,
                    "num_classes": 10,
                    "model_state": model.state_dict(),
                    "qualified": False,
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "accuracy gate"):
                load_frozen_cifar10_resnet18(path, require_qualified=True)

    def test_boolean_qualification_cannot_override_low_numeric_accuracy(self) -> None:
        model = build_cifar10_resnet18()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inconsistent.pt"
            torch.save(
                {
                    "architecture": SEMANTIC_EVALUATOR_ARCHITECTURE,
                    "num_classes": 10,
                    "model_state": model.state_dict(),
                    "qualified": True,
                    "test_accuracy": 0.899,
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_frozen_cifar10_resnet18(path, require_qualified=True)


if __name__ == "__main__":
    unittest.main()
