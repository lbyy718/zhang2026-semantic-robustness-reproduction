"""White-box attacks for the DeepJSCC receiver.

PGA follows Eqs. (35)-(36).  The C&W objective printed in Eq. (38) has a sign
that would reward *smaller* distortion when minimized.  ``CWRegressionAttack``
therefore uses the constraint-correct hinge ``relu(D* + kappa - D)`` and makes
that correction explicit in every result file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

DistortionFunction = Callable[[Tensor, Tensor], Tensor]


@dataclass
class AttackResult:
    adversarial_received: Tensor
    perturbation: Tensor
    reconstruction: Tensor
    distortion: Tensor
    success: Tensor
    steps: Tensor
    total_power: Tensor
    power_per_channel_use: Tensor
    objective_variant: str

    def detached(self) -> "AttackResult":
        return AttackResult(
            adversarial_received=self.adversarial_received.detach(),
            perturbation=self.perturbation.detach(),
            reconstruction=self.reconstruction.detach(),
            distortion=self.distortion.detach(),
            success=self.success.detach(),
            steps=self.steps.detach(),
            total_power=self.total_power.detach(),
            power_per_channel_use=self.power_per_channel_use.detach(),
            objective_variant=self.objective_variant,
        )


def _finish_result(
    decoder: nn.Module,
    target: Tensor,
    received: Tensor,
    adversarial_received: Tensor,
    distortion_fn: DistortionFunction,
    target_distortion: float,
    steps: Tensor,
    objective_variant: str,
) -> AttackResult:
    with torch.no_grad():
        reconstruction = decoder(adversarial_received)
        distortion = distortion_fn(target, reconstruction)
        success = distortion >= target_distortion
        perturbation = adversarial_received - received
        flat = perturbation.flatten(start_dim=1)
        total_power = flat.square().sum(dim=1)
        normalized_power = flat.square().mean(dim=1)
    return AttackResult(
        adversarial_received=adversarial_received,
        perturbation=perturbation,
        reconstruction=reconstruction,
        distortion=distortion,
        success=success,
        steps=steps,
        total_power=total_power,
        power_per_channel_use=normalized_power,
        objective_variant=objective_variant,
    ).detached()


class ProgressiveGradientAscent:
    """Progressive gradient ascent stopped at the first target crossing."""

    def __init__(
        self,
        step_size: float = 0.1,
        max_steps: int = 2000,
        eps: float = 1e-8,
        refine_steps: int = 0,
    ) -> None:
        if step_size <= 0 or max_steps < 1 or eps <= 0 or refine_steps < 0:
            raise ValueError("Invalid PGA hyperparameters.")
        self.step_size = step_size
        self.max_steps = max_steps
        self.eps = eps
        self.refine_steps = refine_steps

    def __call__(
        self,
        decoder: nn.Module,
        target: Tensor,
        received: Tensor,
        distortion_fn: DistortionFunction,
        target_distortion: float,
    ) -> AttackResult:
        decoder.eval()
        start = received.detach()
        adversarial = start.clone()
        batch_size = start.shape[0]
        steps = torch.full(
            (batch_size,), self.max_steps, dtype=torch.long, device=start.device
        )
        lower = adversarial.clone()
        upper = adversarial.clone()

        with torch.no_grad():
            initial = distortion_fn(target, decoder(adversarial))
        done = initial >= target_distortion
        steps[done] = 0

        for step in range(self.max_steps):
            active = ~done
            if not bool(active.any()):
                break

            current = adversarial.detach().requires_grad_(True)
            reconstruction = decoder(current)
            distortion = distortion_fn(target, reconstruction)
            active_loss = (distortion * active.to(distortion.dtype)).sum()
            gradient = torch.autograd.grad(active_loss, current)[0]
            flat_gradient = gradient.flatten(start_dim=1)
            gradient_norm = flat_gradient.norm(p=2, dim=1, keepdim=True)
            direction = flat_gradient / (gradient_norm + self.eps)
            direction = direction.reshape_as(current)

            previous = adversarial.clone()
            adversarial = (
                adversarial + self.step_size * direction * active.reshape(-1, *([1] * (received.ndim - 1)))
            ).detach()

            with torch.no_grad():
                updated_distortion = distortion_fn(target, decoder(adversarial))
            newly_done = active & (updated_distortion >= target_distortion)
            if bool(newly_done.any()):
                lower[newly_done] = previous[newly_done]
                upper[newly_done] = adversarial[newly_done]
                steps[newly_done] = step + 1
            done |= newly_done

        if self.refine_steps and bool(done.any()):
            # Optional final-step bisection.  Disabled by paper-faithful configs
            # because the paper records the first alpha=0.1 threshold crossing.
            refine_mask = done & (steps > 0)
            for _ in range(self.refine_steps):
                middle = 0.5 * (lower + upper)
                with torch.no_grad():
                    middle_distortion = distortion_fn(target, decoder(middle))
                successful_middle = refine_mask & (middle_distortion >= target_distortion)
                unsuccessful_middle = refine_mask & ~successful_middle
                upper[successful_middle] = middle[successful_middle]
                lower[unsuccessful_middle] = middle[unsuccessful_middle]
            adversarial[refine_mask] = upper[refine_mask]

        return _finish_result(
            decoder,
            target,
            start,
            adversarial,
            distortion_fn,
            target_distortion,
            steps,
            objective_variant="paper_pga_eq36",
        )


class CWRegressionAttack:
    """Minimum-norm C&W-style attack for a reconstruction threshold.

    The optimization is performed independently for each sample so that the
    binary search over c and the first-success stopping rule remain unambiguous.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        initial_c: float = 1.0,
        c_min: float = 1e-6,
        c_max: float = 100.0,
        binary_search_steps: int = 9,
        max_steps: int = 2000,
        kappa: float = 0.0,
        early_stop_on_success: bool = True,
    ) -> None:
        if not (0 < c_min <= initial_c <= c_max):
            raise ValueError("Require 0 < c_min <= initial_c <= c_max.")
        if learning_rate <= 0 or binary_search_steps < 1 or max_steps < 1:
            raise ValueError("Invalid C&W hyperparameters.")
        self.learning_rate = learning_rate
        self.initial_c = initial_c
        self.c_min = c_min
        self.c_max = c_max
        self.binary_search_steps = binary_search_steps
        self.max_steps = max_steps
        self.kappa = kappa
        self.early_stop_on_success = early_stop_on_success

    def _attack_one(
        self,
        decoder: nn.Module,
        target: Tensor,
        received: Tensor,
        distortion_fn: DistortionFunction,
        target_distortion: float,
    ) -> tuple[Tensor, int]:
        with torch.no_grad():
            if bool((distortion_fn(target, decoder(received)) >= target_distortion).item()):
                return received.clone(), 0

        best = received.clone()
        best_power = torch.tensor(float("inf"), device=received.device)
        best_steps = self.max_steps
        lower_c = self.c_min
        upper_c = self.c_max
        current_c = self.initial_c

        for _ in range(self.binary_search_steps):
            delta = nn.Parameter(torch.zeros_like(received))
            optimizer = torch.optim.Adam([delta], lr=self.learning_rate)
            trial_succeeded = False

            for step in range(self.max_steps):
                optimizer.zero_grad(set_to_none=True)
                reconstruction = decoder(received + delta)
                distortion = distortion_fn(target, reconstruction)[0]
                power = delta.flatten().square().sum()
                # Correct constrained-minimization hinge.  The sign printed in
                # Eq. (38) would minimize distortion and cannot produce attacks.
                constraint = torch.relu(
                    torch.as_tensor(
                        target_distortion + self.kappa,
                        dtype=distortion.dtype,
                        device=distortion.device,
                    )
                    - distortion
                )
                objective = power + current_c * constraint
                objective.backward()
                optimizer.step()

                with torch.no_grad():
                    candidate_distortion = distortion_fn(target, decoder(received + delta))[0]
                    if bool(candidate_distortion >= target_distortion):
                        trial_succeeded = True
                        candidate_power = delta.flatten().square().sum()
                        if candidate_power < best_power:
                            best_power = candidate_power.clone()
                            best = (received + delta).detach().clone()
                            best_steps = step + 1
                        if self.early_stop_on_success:
                            break

            if trial_succeeded:
                upper_c = min(upper_c, current_c)
            else:
                lower_c = max(lower_c, current_c)
            current_c = 0.5 * (lower_c + upper_c)

        return best, best_steps

    def __call__(
        self,
        decoder: nn.Module,
        target: Tensor,
        received: Tensor,
        distortion_fn: DistortionFunction,
        target_distortion: float,
    ) -> AttackResult:
        decoder.eval()
        adversarial_samples: list[Tensor] = []
        steps: list[int] = []
        for index in range(received.shape[0]):
            adversarial, sample_steps = self._attack_one(
                decoder,
                target[index : index + 1],
                received[index : index + 1].detach(),
                distortion_fn,
                target_distortion,
            )
            adversarial_samples.append(adversarial)
            steps.append(sample_steps)
        adversarial_received = torch.cat(adversarial_samples, dim=0)
        step_tensor = torch.tensor(steps, dtype=torch.long, device=received.device)
        return _finish_result(
            decoder,
            target,
            received.detach(),
            adversarial_received,
            distortion_fn,
            target_distortion,
            step_tensor,
            objective_variant="corrected_cw_constraint_hinge",
        )
