from __future__ import annotations

import math
import unittest

import torch

from semantic_robustness.channel import AWGNChannel
from semantic_robustness.metrics import psnr, target_distortion
from semantic_robustness.model import DeepJSCC
from semantic_robustness.theory import (
    clean_distortion_upper_bound,
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


class ChannelMetricTests(unittest.TestCase):
    def test_explicit_awgn(self) -> None:
        channel = AWGNChannel(fading_gain=2.0)
        symbols = torch.ones(2, 4)
        noise = torch.full_like(symbols, 0.25)
        received = channel(symbols, 10.0, noise=noise)
        torch.testing.assert_close(received, torch.full_like(symbols, 2.25))
        self.assertAlmostEqual(float(channel.noise_variance(10.0, symbols)), 0.4)

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


if __name__ == "__main__":
    unittest.main()
