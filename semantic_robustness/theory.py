"""Executable semantic-system bounds from Section IV-A."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FailureMarginDiagnostics:
    """Per-sample first-order diagnostics for a scalar failure score.

    ``margin`` is positive on the non-failure side when the supplied score is
    negative there. ``linearized_distance`` is a signed first-order distance;
    callers should normally interpret it only for samples with positive margin.
    """

    margin: Tensor
    gradient_l2: Tensor
    linearized_distance: Tensor


@dataclass(frozen=True)
class LocalLipschitzDiagnostics:
    """Convergence diagnostics for implicit Jacobian spectral-norm estimates."""

    estimate: Tensor
    estimate_at_20: Tensor
    estimate_at_30: Tensor
    relative_change_20_30: Tensor
    iterations: Tensor
    converged: Tensor


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


def diagnose_failure_margin(
    failure_score: Callable[[Tensor], Tensor],
    received: Tensor,
) -> FailureMarginDiagnostics:
    """Measure margin and local sensitivity for one scalar score per sample.

    ``failure_score(received)`` must return one scalar per batch element and
    follow the convention ``score >= 0`` means failure. The returned quantities
    are

    ``margin = -score``, ``gradient_l2 = ||grad_received score||_2``, and
    ``linearized_distance = margin / gradient_l2``.

    The function differentiates each sample's score with respect to its own
    received representation. It therefore supports a decoder alone as well as
    a composed differentiable system such as decoder + frozen classifier,
    without materializing a Jacobian matrix.
    """
    if received.ndim < 1 or received.shape[0] == 0:
        raise ValueError("received must contain at least one sample.")

    point = received.detach().requires_grad_(True)
    scores = failure_score(point)
    if scores.numel() != point.shape[0]:
        raise ValueError("failure_score must return exactly one scalar per sample.")
    scores = scores.reshape(point.shape[0])
    margin = -scores

    if not scores.requires_grad:
        gradient_l2 = torch.zeros_like(scores)
    else:
        gradient_norms: list[Tensor] = []
        for index, score in enumerate(scores):
            gradient = torch.autograd.grad(
                score,
                point,
                retain_graph=index + 1 < scores.numel(),
                create_graph=False,
                allow_unused=True,
            )[0]
            if gradient is None:
                gradient_norms.append(torch.zeros_like(score))
            else:
                gradient_norms.append(gradient[index].norm())
        gradient_l2 = torch.stack(gradient_norms)

    return FailureMarginDiagnostics(
        margin=margin.detach(),
        gradient_l2=gradient_l2.detach(),
        linearized_distance=(margin / gradient_l2).detach(),
    )


def estimate_local_lipschitz(
    decoder: nn.Module,
    received: Tensor,
    *,
    power_iterations: int = 20,
    eps: float = 1e-12,
    return_diagnostics: bool = False,
    adaptive: bool = False,
    convergence_rtol: float = 0.05,
    max_power_iterations: int = 60,
) -> Tensor | LocalLipschitzDiagnostics:
    """Estimate the decoder Jacobian spectral norm at each received sample.

    This is a *local* Jacobian estimate, not the global G assumed by Eq. (4).
    It is included as a diagnostic and must not be reported as a certified bound.

    The default path preserves the original behavior and returns a tensor after
    ``power_iterations``. With ``return_diagnostics=True``, the estimator also
    records iterations 20 and 30. With ``adaptive=True``, a sample whose relative
    change from iteration 20 to 30 exceeds ``convergence_rtol`` is extended to
    ``max_power_iterations`` (60 by default); otherwise it stops at iteration 30.
    JVP/VJP products are used throughout, so no full Jacobian is constructed.
    """
    if power_iterations < 1:
        raise ValueError("power_iterations must be positive.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if convergence_rtol < 0:
        raise ValueError("convergence_rtol must be nonnegative.")
    if max_power_iterations < 30:
        raise ValueError("max_power_iterations must be at least 30.")
    if received.ndim < 1 or received.shape[0] == 0:
        raise ValueError("received must contain at least one sample.")

    decoder.eval()
    estimates: list[Tensor] = []
    estimates_at_20: list[Tensor] = []
    estimates_at_30: list[Tensor] = []
    relative_changes: list[Tensor] = []
    iterations_used: list[int] = []
    convergence_flags: list[bool] = []

    for sample in received:
        point = sample.unsqueeze(0).detach()
        vector = torch.randn_like(point)
        vector = vector / vector.norm().clamp_min(eps)
        diagnostic_mode = return_diagnostics or adaptive
        planned_iterations = 30 if adaptive else (
            max(power_iterations, 30) if return_diagnostics else power_iterations
        )
        checkpoint_20: Tensor | None = None
        checkpoint_30: Tensor | None = None

        for iteration in range(1, planned_iterations + 1):
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
            if diagnostic_mode and iteration in (20, 30):
                _, checkpoint_product = torch.autograd.functional.jvp(
                    decoder, point, vector, create_graph=False, strict=False
                )
                checkpoint = checkpoint_product.norm() / vector.norm().clamp_min(eps)
                if iteration == 20:
                    checkpoint_20 = checkpoint
                else:
                    checkpoint_30 = checkpoint

        if diagnostic_mode:
            assert checkpoint_20 is not None and checkpoint_30 is not None
            relative_change = (
                (checkpoint_30 - checkpoint_20).abs()
                / checkpoint_30.abs().clamp_min(eps)
            )
            converged = bool(relative_change <= convergence_rtol)
            actual_iterations = planned_iterations
            if adaptive and not converged:
                for _ in range(31, max_power_iterations + 1):
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
                actual_iterations = max_power_iterations

        _, jacobian_vector = torch.autograd.functional.jvp(
            decoder, point, vector, create_graph=False, strict=False
        )
        estimates.append(jacobian_vector.norm() / vector.norm().clamp_min(eps))
        if diagnostic_mode:
            estimates_at_20.append(checkpoint_20)
            estimates_at_30.append(checkpoint_30)
            relative_changes.append(relative_change)
            iterations_used.append(actual_iterations)
            convergence_flags.append(converged)

    final_estimate = torch.stack(estimates).detach()
    if not return_diagnostics:
        return final_estimate
    return LocalLipschitzDiagnostics(
        estimate=final_estimate,
        estimate_at_20=torch.stack(estimates_at_20).detach(),
        estimate_at_30=torch.stack(estimates_at_30).detach(),
        relative_change_20_30=torch.stack(relative_changes).detach(),
        iterations=torch.tensor(iterations_used, device=received.device, dtype=torch.int64),
        converged=torch.tensor(convergence_flags, device=received.device, dtype=torch.bool),
    )
