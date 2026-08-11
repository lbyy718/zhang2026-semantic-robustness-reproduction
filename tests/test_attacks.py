from __future__ import annotations

import unittest

import torch
from torch import nn

from semantic_robustness.attacks import CWRegressionAttack, ProgressiveGradientAscent
from semantic_robustness.metrics import mse_per_sample


class FlatDecoder(nn.Module):
    def forward(self, received: torch.Tensor) -> torch.Tensor:
        return received.reshape(received.shape[0], 1, 1, 2)


class AttackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = FlatDecoder()
        self.target = torch.zeros(2, 1, 1, 2)
        self.received = torch.full((2, 2), 0.1)

    def test_pga_reaches_reconstruction_threshold(self) -> None:
        attack = ProgressiveGradientAscent(step_size=0.1, max_steps=20, eps=1e-8)
        result = attack(self.decoder, self.target, self.received, mse_per_sample, 0.05)
        self.assertTrue(bool(result.success.all()))
        self.assertTrue(bool((result.distortion >= 0.05).all()))
        self.assertTrue(bool((result.total_power > 0).all()))
        self.assertEqual(result.objective_variant, "paper_pga_eq36")

    def test_corrected_cw_reaches_reconstruction_threshold(self) -> None:
        attack = CWRegressionAttack(
            learning_rate=0.05,
            initial_c=1.0,
            c_min=1e-3,
            c_max=10.0,
            binary_search_steps=4,
            max_steps=100,
            early_stop_on_success=True,
        )
        result = attack(
            self.decoder,
            self.target[:1],
            self.received[:1],
            mse_per_sample,
            0.05,
        )
        self.assertTrue(bool(result.success.item()))
        self.assertGreaterEqual(float(result.distortion.item()), 0.05)
        self.assertEqual(result.objective_variant, "corrected_cw_constraint_hinge")


if __name__ == "__main__":
    unittest.main()
