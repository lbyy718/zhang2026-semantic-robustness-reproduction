"""Image reconstruction metrics used by the CIFAR-10 experiments."""

from __future__ import annotations

import torch
from torch import Tensor


def mse_per_sample(target: Tensor, reconstruction: Tensor) -> Tensor:
    if target.shape != reconstruction.shape:
        raise ValueError("target and reconstruction must have identical shapes.")
    return (target - reconstruction).square().flatten(start_dim=1).mean(dim=1)


def psnr(target: Tensor, reconstruction: Tensor, peak: float = 1.0) -> Tensor:
    mse = mse_per_sample(target, reconstruction).clamp_min(
        torch.finfo(target.dtype).eps
    )
    return 10.0 * torch.log10((peak**2) / mse)


def classification_margin(labels: Tensor, logits: Tensor) -> Tensor:
    """True-class logit minus the strongest competing logit."""
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("Expected logits [batch, classes] and labels [batch].")
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], float("-inf"))
    return true_logits - competitors.max(dim=1).values


def classification_failure_score(labels: Tensor, logits: Tensor) -> Tensor:
    """Non-negative values indicate misclassification (attack success)."""
    return -classification_margin(labels, logits)


def accuracy_per_sample(labels: Tensor, logits: Tensor) -> Tensor:
    return logits.argmax(dim=1).eq(labels).to(logits.dtype)


def distortion_per_sample(task: str, target: Tensor, reconstruction: Tensor) -> Tensor:
    if task != "image":
        raise ValueError("Only the CIFAR-10 image task is supported.")
    return mse_per_sample(target, reconstruction)


def target_distortion(task: str, target_quality_db: float) -> float:
    """Convert a PSNR failure threshold into its equivalent MSE threshold."""
    if task != "image":
        raise ValueError("Only the CIFAR-10 image task is supported.")
    return 10.0 ** (-target_quality_db / 10.0)


def quality_name(task: str) -> str:
    if task != "image":
        raise ValueError("Only the CIFAR-10 image task is supported.")
    return "psnr_db"
