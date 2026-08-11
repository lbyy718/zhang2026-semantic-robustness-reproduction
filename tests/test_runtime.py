from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from semantic_robustness.config import load_config
from semantic_robustness.runtime import (
    _loss_per_sample,
    evaluate_attacks,
    evaluate_clean,
    train,
)


class RuntimeIntegrationTests(unittest.TestCase):
    def test_centered_csi_nmse_loss(self) -> None:
        import torch

        config = {
            "task": "csi",
            "data": {"metric_center": 0.5},
            "training": {"loss": "nmse"},
        }
        target = torch.tensor([[[[0.7, 0.3]]]])
        reconstruction = torch.tensor([[[[0.6, 0.4]]]])
        loss = _loss_per_sample(config, target, reconstruction)
        self.assertAlmostEqual(float(loss), 0.25, places=6)

    def test_tiny_csi_train_clean_and_pga_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(2026)
            np.savez(
                root / "csi.npz",
                train=rng.random((4, 32, 32), dtype=np.float32),
                validation=rng.random((2, 32, 32), dtype=np.float32),
                test=rng.random((2, 32, 32), dtype=np.float32),
            )
            config_data = {
                "task": "csi",
                "seed": 2026,
                "deterministic": True,
                "output_dir": str(root / "output"),
                "data": {
                    "path": str(root / "csi.npz"),
                    "train_key": "train",
                    "validation_key": "validation",
                    "test_key": "test",
                    "representation": "magnitude",
                    "already_angular_delay": True,
                    "bandwidth_ratio": 0.25,
                    "batch_size": 2,
                    "evaluation_batch_size": 2,
                    "validation_samples": 2,
                    "num_workers": 0,
                },
                "model": {
                    "in_channels": 1,
                    "spatial_size": [32, 32],
                    "channel_multiplier": 2,
                    "kernel_size": 3,
                    "residual_kernel_size": 3,
                },
                "channel": {"fading_gain": 1.0},
                "training": {
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.01,
                    "snr_db": 10.0,
                    "validation_snr_db": 10.0,
                    "checkpoint_interval": 0,
                },
                "evaluation": {"snr_db": [10], "max_samples": 2, "channel_repeats": 1},
                "attacks": {
                    "target_quality_db": -16.0,
                    "max_samples": 2,
                    "pga": {
                        "step_size": 0.1,
                        "eps": 1e-8,
                        "max_steps": 2,
                        "refine_steps": 0,
                        "batch_size": 2,
                    },
                    "cw": {
                        "learning_rate": 0.01,
                        "initial_c": 1.0,
                        "c_min": 1e-6,
                        "c_max": 100.0,
                        "binary_search_steps": 1,
                        "max_steps": 2,
                    },
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config_data), encoding="utf-8")
            config = load_config(config_path)
            output = train(config, device_name="cpu")
            checkpoint = output / "checkpoint_best.pt"
            self.assertTrue(checkpoint.exists())
            clean_path = evaluate_clean(config, checkpoint, device_name="cpu")
            samples_path, summary_path = evaluate_attacks(
                config, checkpoint, ["pga"], device_name="cpu"
            )
            self.assertTrue(clean_path.exists())
            self.assertTrue(samples_path.exists())
            self.assertTrue(summary_path.exists())
            with summary_path.open("r", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["attack"], "pga")


if __name__ == "__main__":
    unittest.main()
