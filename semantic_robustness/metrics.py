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
