"""CIFAR-10 image reproduction of Zhang et al. (2026)."""

from .channel import AWGNChannel
from .metrics import distortion_per_sample, mse_per_sample, psnr
from .model import DeepJSCC

__all__ = [
    "AWGNChannel",
    "DeepJSCC",
    "distortion_per_sample",
    "mse_per_sample",
    "psnr",
]

__version__ = "0.1.0"
