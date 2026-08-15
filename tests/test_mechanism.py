from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import Dataset

from semantic_robustness.attacks import ProgressiveGradientAscent
from semantic_robustness.mechanism import (
    MechanismJob,
    ReconstructionSemanticEndpoint,
    balanced_class_indices,
    diagnose_independent_failure_margin,
    estimate_adaptive_spectral_norm,
    linearized_distance_status,
    shared_standard_normal,
    tensor_sha256,
)
from semantic_robustness.metrics import (
    classification_failure_score,
    classification_margin,
)


class LabelDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self) -> None:
        self.targets = [label for _ in range(4) for label in range(10)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.tensor([float(index)]), self.targets[index]


class ReshapeDecoder(nn.Module):
    def forward(self, received: torch.Tensor) -> torch.Tensor:
        return received.reshape(received.shape[0], 1, 1, 2)


class FlatEvaluator(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat = images.flatten(start_dim=1)
        return torch.stack((flat[:, 0] - flat[:, 1], flat[:, 1]), dim=1)


class BinarySemanticEndpoint(nn.Module):
    def forward(self, received: torch.Tensor) -> torch.Tensor:
        value = received[:, 0]
        return torch.stack((value, -value), dim=1)


class MechanismTests(unittest.TestCase):
    def test_balanced_selection_is_deterministic_and_in_source_order(self) -> None:
        dataset = LabelDataset()
        first = balanced_class_indices(dataset, 20)
        second = balanced_class_indices(dataset, 20)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertNotEqual(first, balanced_class_indices(dataset, 20, selection_seed=7))
        self.assertEqual(
            {label: sum(dataset.targets[index] == label for index in first) for label in range(10)},
            {label: 2 for label in range(10)},
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            balanced_class_indices(dataset, 21)

    def test_shared_standard_noise_is_repeatable_and_seed_specific(self) -> None:
        first = shared_standard_normal(20, 8, 102026)
        same = shared_standard_normal(20, 8, 102026)
        different = shared_standard_normal(20, 8, 102027)
        torch.testing.assert_close(first, same, rtol=0, atol=0)
        self.assertEqual(tensor_sha256(first), tensor_sha256(same))
        self.assertNotEqual(tensor_sha256(first), tensor_sha256(different))

    def test_fast_margin_diagnostic_matches_analytic_distance(self) -> None:
        received = torch.tensor([[0.25, 2.0], [-0.5, 3.0]])
        diagnostics = diagnose_independent_failure_margin(
            lambda point: point[:, 0] - 1.0, received
        )
        torch.testing.assert_close(diagnostics.margin, torch.tensor([0.75, 1.5]))
        torch.testing.assert_close(diagnostics.gradient_l2, torch.ones(2))
        torch.testing.assert_close(
            diagnostics.linearized_distance, torch.tensor([0.75, 1.5])
        )

    def test_composed_semantic_endpoint_preserves_latent_gradient(self) -> None:
        endpoint = ReconstructionSemanticEndpoint(ReshapeDecoder(), FlatEvaluator())
        received = torch.tensor([[0.5, 0.1]], requires_grad=True)
        logits = endpoint(received)
        logits[0, 0].backward()
        self.assertIsNotNone(received.grad)
        torch.testing.assert_close(received.grad, torch.tensor([[1.0, -1.0]]))

    def test_implicit_spectral_norm_matches_exact_linear_svd(self) -> None:
        layer = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.diag(torch.tensor([3.0, 1.0])))
        received = torch.zeros(2, 2)
        diagnostics = estimate_adaptive_spectral_norm(
            layer, received, [7, 8], convergence_rtol=0.05
        )
        torch.testing.assert_close(
            diagnostics.estimate,
            torch.full((2,), 3.0),
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertTrue(bool(diagnostics.converged.all()))
        self.assertTrue(bool((diagnostics.iterations == 30).all()))

    def test_batched_spectral_matches_individual_fixed_seed_estimates(self) -> None:
        torch.manual_seed(11)
        layer = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
        received = torch.randn(3, 3)
        seeds = [101, 102, 103]
        batched = estimate_adaptive_spectral_norm(layer, received, seeds)
        individual = [
            estimate_adaptive_spectral_norm(
                layer, received[index : index + 1], [seeds[index]]
            )
            for index in range(3)
        ]
        torch.testing.assert_close(
            batched.estimate,
            torch.cat([result.estimate for result in individual]),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            batched.estimate_at_30,
            torch.cat([result.estimate_at_30 for result in individual]),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_zero_gradient_distance_is_explicit_and_never_nan(self) -> None:
        received = torch.zeros(2, 2)
        diagnostics = diagnose_independent_failure_margin(
            lambda point: torch.tensor([-1.0, 0.0], device=point.device), received
        )
        self.assertTrue(torch.isinf(diagnostics.linearized_distance[0]))
        self.assertEqual(float(diagnostics.linearized_distance[1]), 0.0)
        self.assertEqual(
            linearized_distance_status(
                diagnostics.margin, diagnostics.gradient_l2
            ),
            ["positive_margin_zero_gradient", "boundary_zero_gradient"],
        )

    def test_current_pga_reaches_common_semantic_failure_boundary(self) -> None:
        endpoint = BinarySemanticEndpoint()
        labels = torch.tensor([0])
        received = torch.tensor([[0.25, 0.0]])
        attack = ProgressiveGradientAscent(
            step_size=0.1, max_steps=10, eps=1e-8, refine_steps=0
        )
        result = attack(
            endpoint,
            labels,
            received,
            lambda expected, logits: classification_failure_score(expected, logits),
            0.0,
        )
        self.assertTrue(bool(result.success.item()))
        self.assertGreaterEqual(float(result.distortion.item()), 0.0)
        self.assertLessEqual(
            float(classification_margin(labels, result.reconstruction).item()), 0.0
        )
        self.assertEqual(result.objective_variant, "paper_pga_eq36")

    def test_job_identity_and_objective(self) -> None:
        reconstruction = MechanismJob("R1", 2026, 2, 102028)
        classification = MechanismJob("c0", 2027, 0, 102026)
        self.assertEqual(reconstruction.cell, "r1")
        self.assertEqual(reconstruction.objective, "reconstruction")
        self.assertEqual(classification.objective, "classification")
        self.assertIn("repeat2", reconstruction.identifier)


if __name__ == "__main__":
    unittest.main()
