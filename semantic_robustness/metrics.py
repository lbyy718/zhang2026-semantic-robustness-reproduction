"""Distortion metrics used by the image and CSI experiments."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def mse_per_sample(target: Tensor, reconstruction: Tensor) -> Tensor:
    if target.shape != reconstruction.shape:
        raise ValueError("target and reconstruction must have identical shapes.")
    return (target - reconstruction).square().flatten(start_dim=1).mean(dim=1)


def squared_error_per_sample(target: Tensor, reconstruction: Tensor) -> Tensor:
    if target.shape != reconstruction.shape:
        raise ValueError("target and reconstruction must have identical shapes.")
    return (target - reconstruction).square().flatten(start_dim=1).sum(dim=1)


def psnr(target: Tensor, reconstruction: Tensor, peak: float = 1.0) -> Tensor:
    mse = mse_per_sample(target, reconstruction).clamp_min(torch.finfo(target.dtype).eps)
    return 10.0 * torch.log10((peak**2) / mse)


def nmse_ratio(target: Tensor, reconstruction: Tensor, eps: float = 1e-12) -> Tensor:
    numerator = squared_error_per_sample(target, reconstruction)
    denominator = target.square().flatten(start_dim=1).sum(dim=1).clamp_min(eps)
    return numerator / denominator


def nmse_db(target: Tensor, reconstruction: Tensor, eps: float = 1e-12) -> Tensor:
    return 10.0 * torch.log10(nmse_ratio(target, reconstruction).clamp_min(eps))


def distortion_per_sample(task: str, target: Tensor, reconstruction: Tensor) -> Tensor:
    if task == "image":
        return mse_per_sample(target, reconstruction)
    if task == "csi":
        return nmse_ratio(target, reconstruction)
    raise ValueError(f"Unsupported task: {task!r}")


def target_distortion(task: str, target_quality_db: float) -> float:
    """Convert image PSNR or CSI NMSE dB threshold to a linear distortion."""
    if task == "image":
        return 10.0 ** (-target_quality_db / 10.0)
    if task == "csi":
        return 10.0 ** (target_quality_db / 10.0)
    raise ValueError(f"Unsupported task: {task!r}")


def quality_name(task: str) -> str:
    return "psnr_db" if task == "image" else "nmse_db"


def target_quality_default(task: str) -> float:
    return 15.0 if task == "image" else -16.0


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan
