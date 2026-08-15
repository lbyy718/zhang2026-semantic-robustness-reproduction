"""Shared utilities for the factorial mechanism diagnostics.

This module deliberately keeps the system-level effect (including clean
margin) and the local first-order diagnostics side by side.  It never
materializes a full Jacobian: spectral norms are estimated with JVP/VJP power
iterations and failure-margin sensitivity needs only one scalar gradient per
sample.
"""

from __future__ import annotations

from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, Subset

from .theory import (
    FailureMarginDiagnostics,
    LocalLipschitzDiagnostics,
)


MECHANISM_SCHEMA_VERSION = "factorial_mechanism_v1"
DEFAULT_NOISE_SEEDS = (102026, 102027, 102028)
DEFAULT_SNR_DB = 10.0
DEFAULT_RECONSTRUCTION_FAILURE_PSNR_DB = 15.0
DEFAULT_SELECTION_SEED = 42026


class ReconstructionSemanticEndpoint(nn.Module):
    """Compose a reconstruction decoder with a frozen image classifier."""

    def __init__(self, decoder: nn.Module, evaluator: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.evaluator = evaluator

    def forward(self, received: Tensor) -> Tensor:
        return self.evaluator(self.decoder(received))


def _dataset_labels(dataset: Dataset[Any]) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return [int(value) for value in targets]
    labels: list[int] = []
    for index in range(len(dataset)):
        item = dataset[index]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise TypeError("Balanced sampling requires dataset labels.")
        label = item[1]
        labels.append(int(label.item()) if isinstance(label, Tensor) else int(label))
    return labels


def balanced_class_indices(
    dataset: Dataset[Any],
    max_samples: int,
    *,
    num_classes: int = 10,
    selection_seed: int = DEFAULT_SELECTION_SEED,
) -> list[int]:
    """Select a fixed pseudorandom, exactly class-balanced sample.

    The returned indices are sorted into original test-set order.  Consequently
    selection is deterministic for ``selection_seed`` and independent of
    DataLoader worker scheduling.  Sampling is performed independently within
    every class, avoiding a hidden dependence on CIFAR-10's source ordering.
    """
    if max_samples <= 0 or max_samples > len(dataset):
        raise ValueError("max_samples must be in [1, len(dataset)].")
    if num_classes < 2 or max_samples % num_classes:
        raise ValueError("max_samples must be divisible by num_classes.")
    per_class = max_samples // num_classes
    labels = _dataset_labels(dataset)
    candidates_by_class: list[list[int]] = [[] for _ in range(num_classes)]
    for index, label in enumerate(labels):
        if not 0 <= label < num_classes:
            raise ValueError(f"Dataset label {label} is outside [0, {num_classes}).")
        candidates_by_class[label].append(index)
    missing = {
        label: per_class - len(values)
        for label, values in enumerate(candidates_by_class)
        if len(values) < per_class
    }
    if missing:
        raise ValueError(f"Dataset cannot provide a balanced sample: {missing}.")
    generator = torch.Generator(device="cpu").manual_seed(int(selection_seed))
    selected_by_class: list[list[int]] = []
    for candidates in candidates_by_class:
        permutation = torch.randperm(len(candidates), generator=generator)
        selected_by_class.append(
            [candidates[int(position)] for position in permutation[:per_class]]
        )
    return sorted(index for values in selected_by_class for index in values)


def balanced_subset(
    dataset: Dataset[Any],
    max_samples: int,
    *,
    num_classes: int = 10,
    selection_seed: int = DEFAULT_SELECTION_SEED,
) -> tuple[Subset[Any], list[int]]:
    indices = balanced_class_indices(
        dataset,
        max_samples,
        num_classes=num_classes,
        selection_seed=selection_seed,
    )
    return Subset(dataset, indices), indices


def indices_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(array("I", (int(index) for index in indices)).tobytes())
    return digest.hexdigest()


def shared_standard_normal(
    sample_count: int,
    channel_uses: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate the canonical CPU standard-normal matrix for one repeat.

    Generation on CPU makes the exact tensor independent of the selected CUDA
    device.  Every cell slices this same matrix in the same balanced sample
    order before scaling it by the channel standard deviation.
    """
    if sample_count <= 0 or channel_uses <= 0:
        raise ValueError("sample_count and channel_uses must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(
        sample_count, channel_uses, dtype=dtype, device="cpu", generator=generator
    )


def tensor_sha256(tensor: Tensor) -> str:
    canonical = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(canonical.numpy().tobytes(order="C")).hexdigest()


def diagnose_independent_failure_margin(
    failure_score: Callable[[Tensor], Tensor],
    received: Tensor,
) -> FailureMarginDiagnostics:
    """Fast per-sample margin diagnostics for sample-independent modules.

    For networks in evaluation mode, outputs do not mix examples.  The gradient
    of the sum of per-sample scores therefore contains exactly each sample's own
    score gradient, avoiding the O(B^2) loop needed by a fully general routine.
    """
    if received.ndim < 2 or received.shape[0] == 0:
        raise ValueError("received must contain a non-empty batch.")
    point = received.detach().requires_grad_(True)
    scores = failure_score(point)
    if scores.numel() != point.shape[0]:
        raise ValueError("failure_score must return exactly one scalar per sample.")
    scores = scores.reshape(point.shape[0])
    if scores.requires_grad:
        gradient = torch.autograd.grad(
            scores.sum(), point, create_graph=False, allow_unused=True
        )[0]
    else:
        gradient = None
    gradient_l2 = (
        torch.zeros_like(scores)
        if gradient is None
        else gradient.flatten(start_dim=1).norm(p=2, dim=1)
    )
    margin = -scores
    distance = torch.zeros_like(margin)
    nonzero_gradient = gradient_l2 > 0
    distance[nonzero_gradient] = (
        margin[nonzero_gradient] / gradient_l2[nonzero_gradient]
    )
    positive_zero_gradient = ~nonzero_gradient & (margin > 0)
    distance[positive_zero_gradient] = float("inf")
    return FailureMarginDiagnostics(
        margin=margin.detach(),
        gradient_l2=gradient_l2.detach(),
        linearized_distance=distance.detach(),
    )


def linearized_distance_status(
    margin: Tensor, gradient_l2: Tensor
) -> list[str]:
    """Explain finite/degenerate first-order distances without emitting NaN."""
    if margin.shape != gradient_l2.shape:
        raise ValueError("margin and gradient_l2 must have identical shapes.")
    statuses: list[str] = []
    for sample_margin, sample_gradient in zip(
        margin.detach().cpu(), gradient_l2.detach().cpu(), strict=True
    ):
        if float(sample_gradient) > 0:
            statuses.append("finite_gradient")
        elif float(sample_margin) > 0:
            statuses.append("positive_margin_zero_gradient")
        elif float(sample_margin) < 0:
            statuses.append("already_failed_zero_gradient")
        else:
            statuses.append("boundary_zero_gradient")
    return statuses


def estimate_adaptive_spectral_norm(
    module: nn.Module,
    received: Tensor,
    sample_seeds: Sequence[int],
    *,
    convergence_rtol: float = 0.05,
    max_power_iterations: int = 60,
) -> LocalLipschitzDiagnostics:
    """Estimate implicit per-sample Jacobian norms with batched JVP/VJP.

    Each sample is seeded independently so the semantic endpoint receives the
    same initial power-iteration direction across R0/R1/C0/C1.  A 20-to-30
    relative change above ``convergence_rtol`` extends that sample to iteration
    60.  ``converged`` records whether the 20/30 criterion passed; an extended
    estimate is still returned for non-converged samples.
    """
    if len(sample_seeds) != received.shape[0]:
        raise ValueError("sample_seeds must have one value per received sample.")
    if received.ndim < 2 or received.shape[0] == 0:
        raise ValueError("received must contain a non-empty batch.")
    if max_power_iterations < 30:
        raise ValueError("max_power_iterations must be at least 30.")
    eps = 1e-12
    module.eval()
    point = received.detach()
    initial_vectors: list[Tensor] = []
    for sample, sample_seed in zip(point, sample_seeds, strict=True):
        generator = torch.Generator(device="cpu").manual_seed(int(sample_seed))
        vector = torch.randn(
            tuple(sample.shape),
            generator=generator,
            device="cpu",
            dtype=sample.dtype,
        )
        initial_vectors.append(vector)
    vector = torch.stack(initial_vectors).to(point.device)

    def normalize_per_sample(values: Tensor) -> Tensor:
        flat_norm = values.flatten(start_dim=1).norm(p=2, dim=1)
        return values / flat_norm.clamp_min(eps).reshape(
            -1, *([1] * (values.ndim - 1))
        )

    def power_step(local_point: Tensor, local_vector: Tensor) -> Tensor:
        _, jacobian_vector = torch.autograd.functional.jvp(
            module, local_point, local_vector, create_graph=False, strict=False
        )
        _, transpose_product = torch.autograd.functional.vjp(
            module,
            local_point,
            v=jacobian_vector,
            create_graph=False,
            strict=False,
        )
        return normalize_per_sample(transpose_product)

    def estimates(local_point: Tensor, local_vector: Tensor) -> Tensor:
        _, product = torch.autograd.functional.jvp(
            module, local_point, local_vector, create_graph=False, strict=False
        )
        numerator = product.flatten(start_dim=1).norm(p=2, dim=1)
        denominator = local_vector.flatten(start_dim=1).norm(p=2, dim=1)
        return numerator / denominator.clamp_min(eps)

    vector = normalize_per_sample(vector)
    checkpoint_20: Tensor | None = None
    checkpoint_30: Tensor | None = None
    for iteration in range(1, 31):
        vector = power_step(point, vector)
        if iteration == 20:
            checkpoint_20 = estimates(point, vector)
        elif iteration == 30:
            checkpoint_30 = estimates(point, vector)
    assert checkpoint_20 is not None and checkpoint_30 is not None
    relative_change = (
        (checkpoint_30 - checkpoint_20).abs()
        / checkpoint_30.abs().clamp_min(eps)
    )
    converged = relative_change <= convergence_rtol
    final_estimate = checkpoint_30.clone()
    iterations = torch.full(
        (point.shape[0],), 30, dtype=torch.int64, device=point.device
    )
    unconverged = ~converged
    if bool(unconverged.any()):
        extended_point = point[unconverged]
        extended_vector = vector[unconverged]
        for _ in range(31, max_power_iterations + 1):
            extended_vector = power_step(extended_point, extended_vector)
        final_estimate[unconverged] = estimates(extended_point, extended_vector)
        iterations[unconverged] = max_power_iterations
    return LocalLipschitzDiagnostics(
        estimate=final_estimate.detach(),
        estimate_at_20=checkpoint_20.detach(),
        estimate_at_30=checkpoint_30.detach(),
        relative_change_20_30=relative_change.detach(),
        iterations=iterations,
        converged=converged.detach(),
    )


def power_per_use_db(power: float) -> float | None:
    """Return finite dB power, using missing for an exact zero perturbation."""
    if power < 0:
        raise ValueError("Power cannot be negative.")
    return None if power == 0 else 10.0 * math.log10(power)


@dataclass(frozen=True)
class MechanismJob:
    cell: str
    training_seed: int
    repeat_index: int
    channel_seed: int

    def __post_init__(self) -> None:
        normalized = self.cell.lower()
        if normalized not in {"r0", "r1", "c0", "c1"}:
            raise ValueError(f"Unknown factorial cell: {self.cell!r}.")
        if self.repeat_index < 0:
            raise ValueError("repeat_index cannot be negative.")
        object.__setattr__(self, "cell", normalized)

    @property
    def objective(self) -> str:
        return "reconstruction" if self.cell.startswith("r") else "classification"

    @property
    def identifier(self) -> str:
        return (
            f"{self.cell}_seed{self.training_seed}_repeat{self.repeat_index}_"
            f"noise{self.channel_seed}"
        )
