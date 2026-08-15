from __future__ import annotations

import math
import unittest

import torch

from semantic_robustness.channel import AWGNChannel, NoiselessChannel
from semantic_robustness.metrics import classification_margin, psnr, target_distortion
from semantic_robustness.model import DeepJSCC, DeepJSCCClassifier
from semantic_robustness.theory import (
    clean_distortion_upper_bound,
    diagnose_failure_margin,
    estimate_local_lipschitz,
    lemma1_attack_power_lower_bound,
    theorem3_attack_power_lower_bound,
)


class ModelTests(unittest.TestCase):
    def test_image_shape_bandwidth_and_power(self) -> None:
        torch.manual_seed(1)
        model = DeepJSCC()
        inputs = torch.rand(2, 3, 32, 32)
        symbols = model.encode(inputs)

        self.assertEqual(symbols.shape, (2, 768))
        self.assertAlmostEqual(model.bandwidth_ratio, 0.25)
        torch.testing.assert_close(
            symbols.square().sum(dim=1),
            torch.full((2,), 768.0),
            rtol=1e-5,
            atol=1e-4,
        )

        outputs = model.decode(symbols)
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertTrue(bool(((outputs >= 0) & (outputs <= 1)).all()))

    def test_classification_uses_same_codeword_and_matched_head_size(self) -> None:
        reconstruction = DeepJSCC()
        classification = DeepJSCCClassifier()
        inputs = torch.rand(2, 3, 32, 32)
        self.assertEqual(classification.encode(inputs).shape, (2, 768))
        self.assertEqual(classification.decode(classification.encode(inputs)).shape, (2, 10))
        reconstruction_head = sum(p.numel() for p in reconstruction.decoder.parameters())
        classification_head = sum(p.numel() for p in classification.decoder.parameters())
        self.assertLess(abs(classification_head - reconstruction_head) / reconstruction_head, 0.02)


class ChannelMetricTests(unittest.TestCase):
    def test_explicit_awgn(self) -> None:
        channel = AWGNChannel(fading_gain=2.0)
        symbols = torch.ones(2, 4)
        noise = torch.full_like(symbols, 0.25)
        received = channel(symbols, 10.0, noise=noise)
        torch.testing.assert_close(received, torch.full_like(symbols, 2.25))
        self.assertAlmostEqual(float(channel.noise_variance(10.0, symbols)), 0.4)

    def test_noiseless_channel_is_exact_identity(self) -> None:
        symbols = torch.randn(2, 8)
        torch.testing.assert_close(NoiselessChannel()(symbols, 10.0), symbols)

    def test_classification_margin(self) -> None:
        logits = torch.tensor([[4.0, 1.0, 3.0], [0.0, 2.0, 5.0]])
        labels = torch.tensor([0, 1])
        torch.testing.assert_close(classification_margin(labels, logits), torch.tensor([1.0, -3.0]))

    def test_psnr_threshold_conversion(self) -> None:
        self.assertAlmostEqual(target_distortion("image", 15.0), 10**-1.5)
        target = torch.ones(1, 1, 2, 2)
        reconstruction = target - math.sqrt(0.01)
        self.assertAlmostEqual(float(psnr(target, reconstruction)), 20.0, places=4)


class TheoryTests(unittest.TestCase):
    def test_semantic_bounds(self) -> None:
        self.assertAlmostEqual(lemma1_attack_power_lower_bound(0.09, 0.01, 2.0), 0.01)
        expected = (math.sqrt(0.09) / 2.0 - math.sqrt(4 * 0.01)) ** 2
        self.assertAlmostEqual(
            theorem3_attack_power_lower_bound(0.09, 2.0, 4, 0.01), expected
        )
        self.assertAlmostEqual(clean_distortion_upper_bound(2.0, 4, 0.01), 0.16)

    def test_local_lipschitz_estimator(self) -> None:
        decoder = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            decoder.weight.copy_(torch.diag(torch.tensor([2.0, 1.0])))
        estimate = estimate_local_lipschitz(
            decoder, torch.tensor([[0.2, -0.1]]), power_iterations=20
        )
        self.assertAlmostEqual(float(estimate.item()), 2.0, places=4)

    def test_implicit_spectral_norm_matches_exact_linear_svd(self) -> None:
        torch.manual_seed(7)
        decoder = torch.nn.Linear(3, 4, bias=False, dtype=torch.float64)
        with torch.no_grad():
            decoder.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0, 0.0],
                        [0.0, 1.5, 0.0],
                        [0.0, 0.0, 0.5],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=torch.float64,
                )
            )
        diagnostics = estimate_local_lipschitz(
            decoder,
            torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64),
            return_diagnostics=True,
            adaptive=True,
        )
        exact = torch.linalg.svdvals(decoder.weight.detach())[0]
        relative_error = (diagnostics.estimate[0] - exact).abs() / exact

        self.assertLessEqual(float(relative_error), 0.05)
        self.assertEqual(diagnostics.estimate_at_20.shape, (1,))
        self.assertEqual(diagnostics.estimate_at_30.shape, (1,))
        self.assertIn(int(diagnostics.iterations[0]), (30, 60))

    def test_failure_margin_gradient_matches_analytic_result(self) -> None:
        received = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        coefficient = torch.tensor([3.0, 4.0])
        diagnostics = diagnose_failure_margin(
            lambda point: (point * coefficient).sum(dim=1) - 12.0,
            received,
        )

        torch.testing.assert_close(diagnostics.margin, torch.tensor([1.0, 13.0]))
        torch.testing.assert_close(diagnostics.gradient_l2, torch.tensor([5.0, 5.0]))
        torch.testing.assert_close(
            diagnostics.linearized_distance,
            torch.tensor([0.2, 2.6]),
        )

    def test_linearized_distance_is_invariant_to_positive_score_scaling(self) -> None:
        received = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        coefficient = torch.tensor([3.0, 4.0])

        def score(point: torch.Tensor) -> torch.Tensor:
            return (point * coefficient).sum(dim=1) - 12.0

        baseline = diagnose_failure_margin(score, received)
        scaled = diagnose_failure_margin(lambda point: 7.0 * score(point), received)

        torch.testing.assert_close(
            scaled.linearized_distance,
            baseline.linearized_distance,
        )


if __name__ == "__main__":
    unittest.main()
