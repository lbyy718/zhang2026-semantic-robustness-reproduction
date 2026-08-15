from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from semantic_robustness.channel import AWGNChannel, NoiselessChannel
from semantic_robustness.config import load_config, validate_config
from semantic_robustness.cstrong import (
    centered_logit_rms,
    latent_failure_jacobian_penalty,
    validate_cstrong_training_config,
)
from semantic_robustness.model import DeepJSCCResNetClassifier
from semantic_robustness.runtime import build_channel, build_model, load_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "cstrong"


class CStrongTests(unittest.TestCase):
    def test_resnet_bottleneck_is_768_dimensional_and_unit_power(self) -> None:
        torch.manual_seed(2026)
        model = DeepJSCCResNetClassifier()
        model.eval()
        images = torch.rand(2, 3, 32, 32)
        latent = model.encode(images)
        self.assertEqual(tuple(latent.shape), (2, 768))
        torch.testing.assert_close(
            latent.square().sum(dim=1),
            torch.full((2,), 768.0),
            rtol=1e-5,
            atol=1e-4,
        )
        self.assertAlmostEqual(model.bandwidth_ratio, 0.25)
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 10_000_000)

    def test_channel_preserves_latent_gradient(self) -> None:
        latent = torch.randn(2, 768, requires_grad=True)
        noise = torch.zeros_like(latent)
        received = AWGNChannel()(latent, 10.0, noise=noise)
        received.square().mean().backward()
        self.assertIsNotNone(latent.grad)
        self.assertTrue(torch.isfinite(latent.grad).all())

    def test_failure_jacobian_penalty_matches_linear_solution(self) -> None:
        latent = torch.tensor([[1.0, -2.0], [0.5, 1.0]], requires_grad=True)
        weight = torch.tensor([[1.0, 0.0], [3.0, 4.0]])
        logits = latent @ weight.T
        labels = torch.zeros(2, dtype=torch.long)
        penalty, per_sample = latent_failure_jacobian_penalty(
            logits, latent, labels, create_graph=True
        )
        # grad_z(logit_1-logit_0)=[2,4], so ||grad||^2/N=20/2=10.
        torch.testing.assert_close(per_sample, torch.full((2,), 10.0))
        torch.testing.assert_close(penalty, torch.tensor(10.0))

    def test_centered_logit_normalization_removes_scale(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 4.0]])
        rms = centered_logit_rms(logits)
        torch.testing.assert_close(centered_logit_rms(7.0 * logits), 7.0 * rms)

    def test_all_locked_configs_build_and_csj_has_no_awgn(self) -> None:
        expected = {
            "cs0_resnet18_noiseless_seed2026.json": ("CS0", False, "none"),
            "cs1_resnet18_awgn10_seed2026.json": ("CS1", True, "none"),
            "csj_resnet18_jacobian_seed2026.json": (
                "CSJ",
                False,
                "latent_failure_jacobian",
            ),
        }
        for name, (arm, noisy, regularizer) in expected.items():
            config = load_config(CONFIG_ROOT / name)
            validate_cstrong_training_config(config)
            self.assertEqual(config["experiment_cell"], arm)
            self.assertEqual(config["training"]["channel_noise"], noisy)
            self.assertEqual(config["training"]["regularizer"], regularizer)
            self.assertIsInstance(
                build_channel(config, training=True),
                AWGNChannel if noisy else NoiselessChannel,
            )

    def test_config_rejects_jacobian_plus_awgn(self) -> None:
        config = load_config(CONFIG_ROOT / "csj_resnet18_jacobian_seed2026.json")
        config["training"]["channel_noise"] = True
        with self.assertRaisesRegex(ValueError, "CSJ control"):
            validate_config(config)

    def test_trainability_diagnostic_locks_lower_lr_and_no_early_stop(self) -> None:
        config = load_config(
            CONFIG_ROOT / "cs0_resnet18_noiseless_lr001_seed2026.json"
        )
        validate_cstrong_training_config(config)
        self.assertEqual(config["experiment_phase"], "trainability_diagnostic")
        self.assertEqual(config["training"]["learning_rate"], 0.01)
        self.assertEqual(config["training"]["early_stopping_patience"], 200)
        config["training"]["learning_rate"] = 0.02
        with self.assertRaisesRegex(ValueError, "locked protocol mismatch"):
            validate_cstrong_training_config(config)

    def test_csj_trainability_diagnostic_uses_lower_lr_without_awgn(self) -> None:
        config = load_config(
            CONFIG_ROOT / "csj_resnet18_jacobian_lr001_seed2026.json"
        )
        validate_cstrong_training_config(config)
        self.assertEqual(config["experiment_phase"], "trainability_diagnostic")
        self.assertEqual(config["experiment_cell"], "CSJ")
        self.assertEqual(config["training"]["learning_rate"], 0.01)
        self.assertEqual(config["training"]["regularizer"], "latent_failure_jacobian")
        self.assertFalse(config["training"]["channel_noise"])

    def test_formal_low_lr_configs_are_locked_and_comparable(self) -> None:
        for name, arm, regularizer in (
            ("cs0_resnet18_noiseless_formal_lr001.json", "CS0", "none"),
            (
                "csj_resnet18_jacobian_formal_lr001.json",
                "CSJ",
                "latent_failure_jacobian",
            ),
        ):
            config = load_config(CONFIG_ROOT / name)
            validate_cstrong_training_config(config)
            self.assertEqual(config["experiment_phase"], "formal_low_lr")
            self.assertEqual(config["experiment_cell"], arm)
            self.assertEqual(config["training"]["regularizer"], regularizer)
            self.assertEqual(config["training"]["learning_rate"], 0.01)
            self.assertEqual(config["training"]["early_stopping_patience"], 200)

    def test_checkpoint_loader_supports_strong_architecture(self) -> None:
        config = load_config(CONFIG_ROOT / "cs0_resnet18_noiseless_seed2026.json")
        model = build_model(config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save({"model_state": model.state_dict()}, path)
            restored, metadata = load_checkpoint(config, path, torch.device("cpu"))
        self.assertIsInstance(restored, DeepJSCCResNetClassifier)
        self.assertIn("model_state", metadata)

    def test_pilot_dry_run_contains_exactly_three_seed2026_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dry"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_cstrong_pilot.py"),
                    "--dry-run",
                    "--output-root",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "registry.csv").open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            manifest = json.loads((output / "batch_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["job"] for row in rows], ["cs0_seed2026", "cs1_seed2026", "csj_seed2026"])
        self.assertEqual(manifest["job_count"], 3)
        self.assertFalse(manifest["low_polling_policy"]["automatic_retry"])

    def test_formal_low_lr_dry_run_contains_six_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dry"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_cstrong_pilot.py"),
                    "--protocol",
                    "formal_low_lr_cs0_csj",
                    "--seeds",
                    "2026",
                    "2027",
                    "2028",
                    "--dry-run",
                    "--output-root",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "registry.csv").open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            manifest = json.loads((output / "batch_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [row["job"] for row in rows],
            [
                "cs0_seed2026",
                "csj_seed2026",
                "cs0_seed2027",
                "csj_seed2027",
                "cs0_seed2028",
                "csj_seed2028",
            ],
        )
        self.assertEqual(manifest["protocol"], "formal_low_lr_cs0_csj")


if __name__ == "__main__":
    unittest.main()
