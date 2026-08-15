"""Frozen CIFAR-10 semantic evaluator used by cross-task experiments.

The evaluator owns its input normalization so callers can pass images in the
same ``[0, 1]`` convention used by the reconstruction models.  Freezing model
parameters does not disable autograd with respect to the input image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
SEMANTIC_EVALUATOR_ARCHITECTURE = "cifar10_resnet18"
MINIMUM_QUALIFIED_TEST_ACCURACY = 0.90


class CIFAR10SemanticEvaluator(nn.Module):
    """CIFAR-style ResNet-18 with differentiable internal normalization."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18
        except ImportError as exc:  # pragma: no cover - dependency error is explicit
            raise ImportError("The semantic evaluator requires torchvision.") from exc

        backbone = resnet18(weights=None, num_classes=num_classes)
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        self.backbone = backbone
        self.register_buffer(
            "input_mean", torch.tensor(CIFAR10_MEAN).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "input_std", torch.tensor(CIFAR10_STD).reshape(1, 3, 1, 1)
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "CIFAR10SemanticEvaluator expects images shaped [batch, 3, H, W]."
            )
        normalized = (images - self.input_mean) / self.input_std
        return self.backbone(normalized)


def build_cifar10_resnet18(*, num_classes: int = 10) -> CIFAR10SemanticEvaluator:
    """Build an unfrozen evaluator for training."""
    return CIFAR10SemanticEvaluator(num_classes=num_classes)


def freeze_semantic_evaluator(model: nn.Module) -> nn.Module:
    """Freeze parameters while preserving gradients with respect to inputs."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def _torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Semantic evaluator checkpoint must contain a mapping.")
    return checkpoint


def load_frozen_cifar10_resnet18(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    require_qualified: bool = False,
) -> tuple[CIFAR10SemanticEvaluator, dict[str, Any]]:
    """Load and freeze a trained evaluator and return its recorded metadata.

    ``require_qualified=True`` rejects a checkpoint that did not pass the
    original-image CIFAR-10 test-accuracy gate recorded by the training script.
    """
    resolved_device = torch.device(device)
    checkpoint = _torch_load(checkpoint_path, resolved_device)
    architecture = checkpoint.get("architecture")
    if architecture != SEMANTIC_EVALUATOR_ARCHITECTURE:
        raise ValueError(
            f"Expected architecture {SEMANTIC_EVALUATOR_ARCHITECTURE!r}, "
            f"found {architecture!r}."
        )
    if "model_state" not in checkpoint:
        raise ValueError("Semantic evaluator checkpoint has no model_state.")
    num_classes = int(checkpoint.get("num_classes", 10))
    if num_classes != 10:
        raise ValueError(f"CIFAR-10 evaluator requires 10 classes, found {num_classes}.")
    if require_qualified:
        if not bool(checkpoint.get("qualified", False)):
            raise ValueError("Semantic evaluator did not pass its accuracy gate.")
        try:
            test_accuracy = float(checkpoint["test_accuracy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Qualified evaluator must record a numeric test_accuracy."
            ) from exc
        if not test_accuracy >= MINIMUM_QUALIFIED_TEST_ACCURACY:
            raise ValueError(
                "Semantic evaluator accuracy gate is inconsistent: "
                f"test_accuracy={test_accuracy:.6f} < "
                f"{MINIMUM_QUALIFIED_TEST_ACCURACY:.2f}."
            )

    model = build_cifar10_resnet18(num_classes=num_classes).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    freeze_semantic_evaluator(model)
    return model, checkpoint
