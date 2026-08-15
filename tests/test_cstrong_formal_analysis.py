from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_cstrong_formal import (  # noqa: E402
    aggregate_seed_effects,
    paired_seed_effects,
)


def diagnostic_row(multiplier: float) -> dict[str, str]:
    return {
        "clean_correct": "1",
        "clean_cross_entropy": "0.2",
        "failure_margin": str(2.0 * multiplier),
        "failure_gradient_l2": str(4.0 * multiplier),
        "linearized_distance": str(0.5 * multiplier),
        "spectral_norm": str(3.0 * multiplier),
        "normalized_spectral_norm": str(1.5 * multiplier),
        "centered_logit_rms": str(2.5 * multiplier),
        "pga_success": "1",
        "pga_l2": str(0.8 * multiplier),
        "pga_power_per_channel_use": "0.01",
        "spectral_converged": "1",
    }


class CStrongFormalAnalysisTests(unittest.TestCase):
    def test_two_seed_effects_aggregate_repeats_before_seeds(self) -> None:
        records: dict[tuple[str, int, int], list[dict[str, str]]] = {}
        for seed, multiplier in ((2026, 2.0), (2027, 4.0)):
            for repeat in range(3):
                records[("CS0", seed, repeat)] = [diagnostic_row(1.0)]
                records[("CSJ", seed, repeat)] = [diagnostic_row(multiplier)]
        repeats, seeds = paired_seed_effects(records, [2026, 2027])
        aggregate = {row["metric"]: row for row in aggregate_seed_effects(seeds)}
        self.assertEqual(len(repeats), 6)
        self.assertAlmostEqual(seeds[0]["failure_margin_ratio"], 2.0)
        self.assertAlmostEqual(seeds[1]["failure_margin_ratio"], 4.0)
        self.assertAlmostEqual(
            aggregate["failure_margin"]["geometric_mean_ratio_csj_over_cs0"],
            8.0**0.5,
        )
        self.assertTrue(aggregate["failure_margin"]["direction_consistent"])

    def test_multi_seed_two_arm_diagnostic_dry_run_has_twelve_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "diagnostics"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_cstrong_diagnostics.py"),
                    "--root",
                    str(root),
                    "--seeds",
                    "2026",
                    "2027",
                    "--arms",
                    "CS0",
                    "CSJ",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "diagnostics_registry.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            manifest = json.loads(
                (root / "diagnostics_batch_manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual(len(rows), 12)
        self.assertEqual(manifest["seeds"], [2026, 2027])
        self.assertEqual(manifest["arms"], ["CS0", "CSJ"])
        self.assertEqual(rows[0]["job"], "cs0_seed2026_repeat0")
        self.assertEqual(rows[-1]["job"], "csj_seed2027_repeat2")


if __name__ == "__main__":
    unittest.main()
