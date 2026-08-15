from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import analyze_mechanism as analysis


def diagnostic_row(
    *,
    cell: str = "R0",
    seed: int = 2026,
    repeat: int = 0,
    channel_seed: int = 102026,
    dataset_index: int = 0,
    class_label: int = 0,
    correct: bool = True,
    success: bool = True,
    power: float = 0.01,
    margin: float = 2.0,
    gradient: float = 4.0,
    distance: float = 0.5,
    spectral: float = 3.0,
) -> dict[str, object]:
    return {
        "cell": cell,
        "training_seed": seed,
        "repeat_index": repeat,
        "channel_seed": channel_seed,
        "dataset_index": dataset_index,
        "class_label": class_label,
        "clean_semantic_correct": int(correct),
        "semantic_failure_margin": margin,
        "semantic_margin_gradient_l2": gradient,
        "semantic_linearized_distance_l2": distance,
        "semantic_spectral_norm": spectral,
        "semantic_spectral_converged_20_30": 1,
        "pga_success": int(success),
        "pga_attack_power_total_l2_sq": power * 768 if success else "",
        "pga_attack_power_per_channel_use": power if success else "",
    }


class MechanismAnalysisTests(unittest.TestCase):
    def test_standard_cell_validation_is_order_independent(self) -> None:
        self.assertEqual(set(sorted(analysis.CELLS)), set(analysis.CELLS))

    def test_failure_power_encodes_clean_error_and_censoring(self) -> None:
        clean_error = diagnostic_row(correct=False, success=True, power=5.0)
        attacked = diagnostic_row(correct=True, success=True, power=0.1)
        censored = diagnostic_row(correct=True, success=False)
        self.assertEqual(analysis.failure_power_per_use(clean_error), 0.0)
        self.assertEqual(analysis.failure_power_per_use(attacked), 0.1)
        self.assertTrue(math.isinf(analysis.failure_power_per_use(censored)))
        self.assertEqual(
            analysis.failure_power_per_use(censored, unresolved_policy="lower"), 0.0
        )
        curve = analysis.robust_accuracy(
            [clean_error, attacked, censored], np.asarray([0.01, 0.1])
        )
        np.testing.assert_allclose(curve, [2 / 3, 1 / 3])
        lower, upper = analysis.pga_empirical_robust_accuracy_bounds(
            [clean_error, attacked, censored], np.asarray([0.01, 0.1])
        )
        np.testing.assert_allclose(lower, [1 / 3, 0.0])
        np.testing.assert_allclose(upper, [2 / 3, 1 / 3])

    def test_small_synthetic_csv_passes_hash_and_identity_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "job"
            output.mkdir()
            row = diagnostic_row()
            csv_path = output / "diagnostics.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "rows": 1,
                        "skip_pga": False,
                        "skip_spectral": False,
                        "diagnostics_csv_sha256": digest,
                    }
                ),
                encoding="utf-8",
            )
            registry_fields = [
                "cell",
                "training_seed",
                "repeat_index",
                "channel_seed",
                "status",
                "output_dir",
            ]
            with (root / "registry.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=registry_fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "cell": "R0",
                        "training_seed": 2026,
                        "repeat_index": 0,
                        "channel_seed": 102026,
                        "status": "completed",
                        "output_dir": str(output),
                    }
                )
            rows, metadata = analysis.load_validated_diagnostics(
                root, expected_jobs=1, enforce_standard_design=False
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(metadata["jobs"], 1)

    def test_channel_repeats_are_averaged_inside_one_training_seed(self) -> None:
        rows: list[dict[str, object]] = []
        for cell in analysis.CELLS:
            rows.append(
                diagnostic_row(
                    cell=cell,
                    repeat=0,
                    channel_seed=102026,
                    power=1e-4,
                )
            )
            rows.append(
                diagnostic_row(
                    cell=cell,
                    repeat=1,
                    channel_seed=102027,
                    success=False,
                )
            )
        curves, summaries = analysis.build_robust_curve_outputs(
            rows, np.asarray([-40.0, 0.0])
        )
        self.assertEqual(len(summaries), 4)
        self.assertTrue(all(row["channel_repeats_averaged"] == 2 for row in summaries))
        r0_seed = [
            row
            for row in curves
            if row["cell"] == "R0"
            and row["sample_scope"] == "all"
            and row["aggregation_level"] == "training_seed"
        ]
        self.assertEqual(len(r0_seed), 2)
        self.assertTrue(
            all(
                float(row["pga_empirical_robust_accuracy_upper_bound"]) == 0.5
                for row in r0_seed
            )
        )
        self.assertTrue(
            all(
                float(row["pga_empirical_robust_accuracy_lower_bound"]) == 0.0
                for row in r0_seed
            )
        )
        r0_across = [
            row
            for row in curves
            if row["cell"] == "R0"
            and row["sample_scope"] == "all"
            and row["aggregation_level"] == "across_training_seeds"
        ]
        self.assertEqual(len(r0_across), 2)
        self.assertTrue(
            all(row["upper_bound_sd_across_training_seeds"] == "" for row in r0_across)
        )

    def test_pairwise_common_correct_is_not_four_cell_intersection(self) -> None:
        rows: list[dict[str, object]] = []
        for cell in analysis.CELLS:
            rows.append(diagnostic_row(cell=cell, dataset_index=0, correct=True))
            rows.append(
                diagnostic_row(
                    cell=cell,
                    dataset_index=1,
                    correct=cell in {"R0", "R1"},
                )
            )
        _curves, summaries = analysis.build_robust_curve_outputs(
            rows, np.asarray([-40.0, 0.0])
        )
        r0 = next(row for row in summaries if row["cell"] == "R0")
        c0 = next(row for row in summaries if row["cell"] == "C0")
        self.assertEqual(r0["pairwise_common_clean_correct_samples_per_repeat_mean"], 2)
        self.assertEqual(r0["four_cell_common_clean_correct_samples_per_repeat_mean"], 1)
        self.assertEqual(c0["pairwise_common_clean_correct_samples_per_repeat_mean"], 1)

    def test_log_decomposition_identity(self) -> None:
        off = diagnostic_row(margin=2.0, gradient=4.0, distance=0.5, power=0.25)
        on = diagnostic_row(margin=8.0, gradient=2.0, distance=4.0, power=1.0)
        result = analysis.paired_log_identity_diagnostics(off, on)
        self.assertAlmostEqual(float(result["identity_residual"]), 0.0, places=12)
        self.assertAlmostEqual(
            float(result["delta_ln_linearized_distance"]), math.log(8.0), places=12
        )
        self.assertAlmostEqual(
            float(result["complete_case_delta_ln_pga_attack_l2"]), math.log(2.0)
        )

    def test_paired_effects_report_all_four_pga_success_patterns(self) -> None:
        rows: list[dict[str, object]] = []
        r0_success = [True, True, False, False]
        r1_success = [True, False, True, False]
        for cell in analysis.CELLS:
            for dataset_index in range(4):
                if cell == "R0":
                    success = r0_success[dataset_index]
                elif cell == "R1":
                    success = r1_success[dataset_index]
                else:
                    success = True
                rows.append(
                    diagnostic_row(
                        cell=cell,
                        dataset_index=dataset_index,
                        success=success,
                    )
                )
        _curves, seed_summaries = analysis.build_robust_curve_outputs(
            rows, np.asarray([-40.0, 0.0])
        )
        effects = analysis.build_paired_effects(rows, seed_summaries)
        reconstruction = next(
            row for row in effects if row["comparison"] == "R1_minus_R0"
        )
        for mode in (
            "both_success",
            "off_only_success",
            "on_only_success",
            "neither_success",
        ):
            self.assertEqual(reconstruction[f"pga_{mode}_sample_repeat_pairs"], 1)
        self.assertNotIn("successful_pga_sample_repeat_pairs", reconstruction)

    def test_margin_matching_qualification_uses_pair_count_and_smd(self) -> None:
        matched: list[dict[str, object]] = []
        source: list[dict[str, object]] = []
        for comparison, off_cell, on_cell, _objective in analysis.COMPARISONS:
            index = 0
            for seed in (2026, 2027, 2028):
                for repeat in (0, 1, 2):
                    for local_index in range(24):
                        base = index * 0.001
                        shift = 1e-5 if index % 2 else -1e-5
                        matched.append(
                            {
                                "comparison": comparison,
                                "training_seed": seed,
                                "repeat_index": repeat,
                                "off_ln_margin": base,
                                "on_ln_margin": base + shift,
                                "delta_ln_gradient": 0.1,
                                "delta_ln_linearized_distance": -0.1,
                                "complete_case_delta_ln_pga_attack_l2": 0.2,
                                "delta_ln_semantic_spectral_norm": 0.05,
                            }
                        )
                        source.append(
                            diagnostic_row(
                                cell=off_cell,
                                seed=seed,
                                repeat=repeat,
                                channel_seed=102026 + repeat,
                                dataset_index=index,
                                class_label=local_index % 10,
                                margin=math.exp(base),
                            )
                        )
                        source.append(
                            diagnostic_row(
                                cell=on_cell,
                                seed=seed,
                                repeat=repeat,
                                channel_seed=102026 + repeat,
                                dataset_index=index,
                                class_label=local_index % 10,
                                margin=math.exp(base + shift),
                            )
                        )
                        index += 1
        summary = analysis.summarize_margin_matching(matched, source_rows=source)
        self.assertTrue(summary["R1_minus_R0"]["qualified"])
        self.assertTrue(summary["C1_minus_C0"]["qualified"])
        self.assertLess(abs(summary["R1_minus_R0"]["matched_log_margin_smd"]), 0.1)
        self.assertEqual(
            summary["R1_minus_R0"]["per_training_seed_audit"]["2026"]["covered_repeats"],
            [0, 1, 2],
        )
        incomplete = [row for row in matched if int(row["repeat_index"]) == 0]
        incomplete_summary = analysis.summarize_margin_matching(
            incomplete, source_rows=source, minimum_pairs=1
        )
        self.assertFalse(incomplete_summary["R1_minus_R0"]["qualified"])

    def test_fisher_z_is_used_for_repeat_correlation_aggregation(self) -> None:
        correlations = [0.1, 0.8]
        expected = math.tanh((math.atanh(0.1) + math.atanh(0.8)) / 2)
        self.assertAlmostEqual(
            analysis.fisher_z_mean_correlations(correlations), expected, places=12
        )


if __name__ == "__main__":
    unittest.main()
