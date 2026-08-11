from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from semantic_robustness.channel import AWGNChannel
from semantic_robustness.data import prepare_cost2100_array
from semantic_robustness.metrics import nmse_db, psnr, target_distortion
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
        model = DeepJSCC(3, 6)
        inputs = torch.rand(2, 3, 32, 32)
        symbols = model.encode(inputs)
        self.assertEqual(symbols.shape, (2, 768))
        self.assertAlmostEqual(model.bandwidth_ratio, 0.25)
        energy = symbols.square().sum(dim=1)
        torch.testing.assert_close(energy, torch.full_like(energy, 768.0), rtol=1e-5, atol=1e-4)
        outputs = model.decode(symbols)
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertTrue(bool(((outputs >= 0) & (outputs <= 1)).all()))

    def test_csi_shape_bandwidth_and_power(self) -> None:
        model = DeepJSCC(1, 2)
        inputs = torch.rand(2, 1, 32, 32)
        symbols = model.encode(inputs)
        self.assertEqual(symbols.shape, (2, 256))
        self.assertAlmostEqual(model.bandwidth_ratio, 0.25)
        self.assertEqual(model.decode(symbols).shape, inputs.shape)

    def test_intermediate_sigmoid_ablation_preserves_final_output_bound(self) -> None:
        default_model = DeepJSCC(2, 2)
        ablated_model = DeepJSCC(2, 2, intermediate_sigmoid=False)
        default_sigmoids = sum(
            isinstance(module, torch.nn.Sigmoid) for module in default_model.decoder.modules()
        )
        ablated_sigmoids = sum(
            isinstance(module, torch.nn.Sigmoid) for module in ablated_model.decoder.modules()
        )
        self.assertEqual(default_sigmoids, 2)
        self.assertEqual(ablated_sigmoids, 1)
        outputs = ablated_model.decode(ablated_model.encode(torch.rand(2, 2, 32, 32)))
        self.assertEqual(outputs.shape, (2, 2, 32, 32))
        self.assertTrue(bool(((outputs >= 0) & (outputs <= 1)).all()))

    def test_zero_mean_symbol_ablation_preserves_power(self) -> None:
        model = DeepJSCC(2, 2, zero_mean_symbols=True)
        symbols = model.encode(torch.rand(3, 2, 32, 32))
        torch.testing.assert_close(
            symbols.mean(dim=1), torch.zeros(3), rtol=0.0, atol=1e-6
        )
        torch.testing.assert_close(
            symbols.square().sum(dim=1),
            torch.full((3,), 256.0),
            rtol=1e-5,
            atol=1e-4,
        )

    def test_global_mixing_preserves_shape_bandwidth_and_power(self) -> None:
        model = DeepJSCC(1, 2, global_mixing=True)
        inputs = torch.rand(3, 1, 32, 32)
        symbols = model.encode(inputs)
        self.assertEqual(symbols.shape, (3, 256))
        self.assertAlmostEqual(model.bandwidth_ratio, 0.25)
        torch.testing.assert_close(
            symbols.square().sum(dim=1),
            torch.full((3,), 256.0),
            rtol=1e-5,
            atol=1e-4,
        )
        self.assertEqual(model.decode(symbols).shape, inputs.shape)

    def test_complex_channel_use_count_and_power(self) -> None:
        model = DeepJSCC(1, 4, complex_symbols=True)
        symbols = model.encode(torch.rand(3, 1, 32, 32))
        self.assertEqual(symbols.shape, (3, 512))
        self.assertEqual(model.real_channel_dimensions, 512)
        self.assertEqual(model.channel_uses, 256)
        self.assertAlmostEqual(model.bandwidth_ratio, 0.25)
        torch.testing.assert_close(
            symbols.square().sum(dim=1),
            torch.full((3,), 256.0),
            rtol=1e-5,
            atol=1e-4,
        )

    def test_global_mixing_uses_configured_spatial_size(self) -> None:
        model = DeepJSCC(1, 2, spatial_size=(16, 16), global_mixing=True)
        inputs = torch.rand(2, 1, 16, 16)
        symbols = model.encode(inputs)
        self.assertEqual(symbols.shape, (2, 64))
        self.assertEqual(model.decode(symbols).shape, inputs.shape)


class ChannelMetricTests(unittest.TestCase):
    def test_explicit_awgn(self) -> None:
        channel = AWGNChannel(fading_gain=2.0)
        symbols = torch.ones(2, 4)
        noise = torch.full_like(symbols, 0.25)
        received = channel(symbols, 10.0, noise=noise)
        torch.testing.assert_close(received, torch.full_like(symbols, 2.25))
        self.assertAlmostEqual(float(channel.noise_variance(10.0, symbols)), 0.4)

    def test_threshold_conversions_and_metrics(self) -> None:
        self.assertAlmostEqual(target_distortion("image", 15.0), 10 ** -1.5)
        self.assertAlmostEqual(target_distortion("csi", -16.0), 10 ** -1.6)
        target = torch.ones(1, 1, 2, 2)
        reconstruction = target - math.sqrt(0.01)
        self.assertAlmostEqual(float(psnr(target, reconstruction)), 20.0, places=4)
        self.assertAlmostEqual(float(nmse_db(target, reconstruction)), -20.0, places=4)


class DataTheoryTests(unittest.TestCase):
    def test_cost2100_preparation(self) -> None:
        raw = np.ones((2, 64, 32), dtype=np.complex64)
        prepared = prepare_cost2100_array(raw, representation="magnitude")
        self.assertEqual(prepared.shape, (2, 32, 32))
        self.assertEqual(prepared.dtype, np.float32)

    def test_flattened_csinet_outdoor_preparation(self) -> None:
        raw = np.full((2, 2048), 0.5, dtype=np.float64)
        raw[:, :1024] += 0.3
        raw[:, 1024:] += 0.4
        prepared = prepare_cost2100_array(
            raw, representation="magnitude", already_angular_delay=True
        )
        self.assertEqual(prepared.shape, (2, 32, 32))
        self.assertEqual(prepared.dtype, np.float32)
        np.testing.assert_allclose(prepared, 0.5, rtol=1e-6, atol=1e-6)

    def test_flattened_csinet_real_imag_preparation(self) -> None:
        raw = np.full((2, 2048), 0.5, dtype=np.float64)
        raw[:, :1024] += 0.3
        raw[:, 1024:] += 0.4
        prepared = prepare_cost2100_array(
            raw, representation="real_imag", already_angular_delay=True
        )
        self.assertEqual(prepared.shape, (2, 2, 32, 32))
        self.assertEqual(prepared.dtype, np.float32)
        np.testing.assert_allclose(prepared[:, 0], 0.8, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(prepared[:, 1], 0.9, rtol=1e-6, atol=1e-6)

    def test_semantic_bounds(self) -> None:
        self.assertAlmostEqual(lemma1_attack_power_lower_bound(0.09, 0.01, 2.0), 0.01)
        expected = (math.sqrt(0.09) / 2.0 - math.sqrt(4 * 0.01)) ** 2
        self.assertAlmostEqual(theorem3_attack_power_lower_bound(0.09, 2.0, 4, 0.01), expected)
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
