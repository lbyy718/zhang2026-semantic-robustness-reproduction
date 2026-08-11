"""Executable semantic-system bounds from Section IV-A."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def lemma1_attack_power_lower_bound(
    attacked_distortion: float,
    clean_distortion: float,
    decoder_lipschitz: float,
) -> float:
    """Eq. (10): ((sqrt(D1)-sqrt(D0))/G)^2."""
    if attacked_distortion < clean_distortion:
        raise ValueError("Lemma 1 assumes attacked distortion >= clean distortion.")
    if decoder_lipschitz <= 0:
        raise ValueError("decoder_lipschitz must be positive.")
    return ((math.sqrt(attacked_distortion) - math.sqrt(clean_distortion)) / decoder_lipschitz) ** 2


def theorem3_attack_power_lower_bound(
    target_distortion: float,
    decoder_lipschitz: float,
    channel_uses: int,
    noise_variance: float,
) -> float:
    """Eq. (12), reproduced literally from the paper."""
    if target_distortion < 0 or decoder_lipschitz <= 0:
        raise ValueError("Distortion must be nonnegative and Lipschitz constant positive.")
    if channel_uses <= 0 or noise_variance < 0:
        raise ValueError("Invalid channel parameters.")
    return (
        math.sqrt(target_distortion) / decoder_lipschitz
        - math.sqrt(channel_uses * noise_variance)
    ) ** 2


def clean_distortion_upper_bound(
    decoder_lipschitz: float, channel_uses: int, noise_variance: float
) -> float:
    """Rightmost inequality in Eq. (11): N sigma_w^2 G^2."""
    if decoder_lipschitz <= 0 or channel_uses <= 0 or noise_variance < 0:
        raise ValueError("Invalid parameters for Eq. (11).")
    return channel_uses * noise_variance * decoder_lipschitz**2


def estimate_local_lipschitz(
    decoder: nn.Module,
    received: Tensor,
    *,
    power_iterations: int = 20,
    eps: float = 1e-12,
) -> Tensor:
    """Estimate the decoder Jacobian spectral norm at each received sample.

    This is a *local* Jacobian estimate, not the global G assumed by Eq. (4).
    It is included as a diagnostic and must not be reported as a certified bound.
    """
    if power_iterations < 1:
        raise ValueError("power_iterations must be positive.")
    decoder.eval()
    estimates: list[Tensor] = []
    for sample in received:
        point = sample.unsqueeze(0).detach()
        vector = torch.randn_like(point)
        vector = vector / vector.norm().clamp_min(eps)
        for _ in range(power_iterations):
            _, jacobian_vector = torch.autograd.functional.jvp(
                decoder, point, vector, create_graph=False, strict=False
            )
            _, transpose_product = torch.autograd.functional.vjp(
                decoder,
                point,
                v=jacobian_vector,
                create_graph=False,
                strict=False,
            )
            vector = transpose_product / transpose_product.norm().clamp_min(eps)
        _, jacobian_vector = torch.autograd.functional.jvp(
            decoder, point, vector, create_graph=False, strict=False
        )
        estimates.append(jacobian_vector.norm() / vector.norm().clamp_min(eps))
    return torch.stack(estimates).detach()
