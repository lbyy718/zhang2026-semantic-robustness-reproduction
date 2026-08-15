"""Aggregate the 2x2 training and clean-evaluation results and draw figures."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FACTORIAL = REPO_ROOT / "outputs" / "factorial"
ANALYSIS = FACTORIAL / "factorial_analysis"
CELLS = ("R0", "R1", "C0", "C1")
SEEDS = (2026, 2027, 2028)
T_CRITICAL_DF2_95 = 4.30265272975


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_CRITICAL_DF2_95 * sd / math.sqrt(len(values))
    return mean, sd, half


def paired_standardized_mean_difference(values: list[float]) -> float:
    """Return paired Cohen's dz; with n=3 this is descriptive, not stable."""
    sd = statistics.stdev(values)
    return statistics.mean(values) / sd if sd > 0 else math.inf


def sign_test_two_sided_nonzero(differences: list[float]) -> float:
    nonzero = [value for value in differences if value != 0]
    n = len(nonzero)
    if not n:
        return 1.0
    positives = sum(value > 0 for value in nonzero)
    tail = sum(math.comb(n, k) for k in range(0, min(positives, n - positives) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    training_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    histories: dict[tuple[str, int], list[dict[str, str]]] = {}
    for cell in CELLS:
        for seed in SEEDS:
            output = FACTORIAL / f"{cell.lower()}_seed{seed}"
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            history = read_csv(output / "training_log.csv")
            histories[cell, seed] = history
            best = min(history, key=lambda row: float(row["validation_loss"]))
            last = history[-1]
            metric = "mse" if cell.startswith("R") else "accuracy"
            training_rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "objective": manifest["objective"],
                    "channel_noise": manifest["training_channel_noise"],
                    "epochs": len(history),
                    "best_epoch": int(best["epoch"]),
                    "best_validation_loss": float(best["validation_loss"]),
                    "best_validation_metric_name": metric,
                    "best_validation_metric": float(best[f"validation_{metric}"]),
                    "last_validation_loss": float(last["validation_loss"]),
                    "last_validation_metric": float(last[f"validation_{metric}"]),
                    "elapsed_hours": float(last["elapsed_seconds"]) / 3600.0,
                }
            )
            for row in read_csv(output / "clean_metrics.csv"):
                clean_rows.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        **row,
                    }
                )
    write_csv(ANALYSIS / "training_summary.csv", training_rows)
    write_csv(ANALYSIS / "clean_all_runs.csv", clean_rows)

    clean_summary: list[dict[str, Any]] = []
    for cell in CELLS:
        snrs = sorted(
            {float(row["snr_db"]) for row in clean_rows if row["cell"] == cell}
        )
        objective = "reconstruction" if cell.startswith("R") else "classification"
        primary = "mean_psnr_db" if objective == "reconstruction" else "mean_accuracy"
        secondary = "mean_mse" if objective == "reconstruction" else "mean_cross_entropy"
        for snr in snrs:
            selected = [
                row
                for row in clean_rows
                if row["cell"] == cell and float(row["snr_db"]) == snr
            ]
            primary_values = [float(row[primary]) for row in selected]
            secondary_values = [float(row[secondary]) for row in selected]
            mean, sd, ci = mean_sd_ci(primary_values)
            second_mean, second_sd, second_ci = mean_sd_ci(secondary_values)
            clean_summary.append(
                {
                    "cell": cell,
                    "objective": objective,
                    "snr_db": snr,
                    "training_seeds": len(selected),
                    "channel_repeats_per_seed": int(selected[0]["channel_repeats"]),
                    "primary_metric": primary,
                    "primary_mean": mean,
                    "primary_sd_across_seeds": sd,
                    "primary_ci95_halfwidth_t_df2": ci,
                    "secondary_metric": secondary,
                    "secondary_mean": second_mean,
                    "secondary_sd_across_seeds": second_sd,
                    "secondary_ci95_halfwidth_t_df2": second_ci,
                }
            )
    write_csv(ANALYSIS / "clean_summary.csv", clean_summary)

    by_clean = {
        (row["cell"], int(row["seed"]), float(row["snr_db"])): row
        for row in clean_rows
    }
    paired_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for snr in sorted({float(row["snr_db"]) for row in clean_rows}):
            r0, r1 = by_clean["R0", seed, snr], by_clean["R1", seed, snr]
            c0, c1 = by_clean["C0", seed, snr], by_clean["C1", seed, snr]
            paired_rows.append(
                {
                    "seed": seed,
                    "snr_db": snr,
                    "r_psnr_gain_db_r1_minus_r0": float(r1["mean_psnr_db"])
                    - float(r0["mean_psnr_db"]),
                    "r_mse_ratio_r1_over_r0": float(r1["mean_mse"])
                    / float(r0["mean_mse"]),
                    "c_accuracy_gain_pp_c1_minus_c0": 100.0
                    * (float(c1["mean_accuracy"]) - float(c0["mean_accuracy"])),
                    "c_cross_entropy_change_c1_minus_c0": float(c1["mean_cross_entropy"])
                    - float(c0["mean_cross_entropy"]),
                }
            )
    write_csv(ANALYSIS / "paired_clean_effects.csv", paired_rows)

    # Figure 1: full training trajectories (mean and min-max across seeds).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for cell, color in (("R0", "tab:orange"), ("R1", "tab:blue")):
        matrix = np.array(
            [[float(row["validation_mse"]) for row in histories[cell, seed]] for seed in SEEDS]
        )
        x = np.arange(1, matrix.shape[1] + 1)
        axes[0].plot(x, matrix.mean(axis=0), label=cell, color=color)
        axes[0].fill_between(x, matrix.min(axis=0), matrix.max(axis=0), color=color, alpha=0.18)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation MSE at 10 dB")
    axes[0].set_title("Reconstruction training")
    for cell, color in (("C0", "tab:orange"), ("C1", "tab:blue")):
        matrix = np.array(
            [[float(row["validation_accuracy"]) for row in histories[cell, seed]] for seed in SEEDS]
        )
        x = np.arange(1, matrix.shape[1] + 1)
        axes[1].plot(x, 100 * matrix.mean(axis=0), label=cell, color=color)
        axes[1].fill_between(x, 100 * matrix.min(axis=0), 100 * matrix.max(axis=0), color=color, alpha=0.18)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy at 10 dB (%)")
    axes[1].set_title("Classification training")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS / "training_validation_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: clean channel curves across three training seeds.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for cell, color in (("R0", "tab:orange"), ("R1", "tab:blue")):
        selected = [row for row in clean_summary if row["cell"] == cell]
        x = np.array([float(row["snr_db"]) for row in selected])
        y = np.array([float(row["primary_mean"]) for row in selected])
        sd = np.array([float(row["primary_sd_across_seeds"]) for row in selected])
        axes[0].plot(x, y, "o-", label=cell, color=color)
        axes[0].fill_between(x, y - sd, y + sd, color=color, alpha=0.18)
    axes[0].set_xlabel("Test SNR (dB)")
    axes[0].set_ylabel("Mean PSNR (dB)")
    axes[0].set_title("Reconstruction clean performance")
    for cell, color in (("C0", "tab:orange"), ("C1", "tab:blue")):
        selected = [row for row in clean_summary if row["cell"] == cell]
        x = np.array([float(row["snr_db"]) for row in selected])
        y = 100 * np.array([float(row["primary_mean"]) for row in selected])
        sd = 100 * np.array([float(row["primary_sd_across_seeds"]) for row in selected])
        axes[1].plot(x, y, "o-", label=cell, color=color)
        axes[1].fill_between(x, y - sd, y + sd, color=color, alpha=0.18)
    axes[1].set_xlabel("Test SNR (dB)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Classification clean performance")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS / "clean_snr_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: paired noise-training effects by SNR.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    snrs = sorted({float(row["snr_db"]) for row in paired_rows})
    for metric, axis, label in (
        ("r_psnr_gain_db_r1_minus_r0", axes[0], "R1 - R0 PSNR (dB)"),
        ("c_accuracy_gain_pp_c1_minus_c0", axes[1], "C1 - C0 accuracy (pp)"),
    ):
        means, sds = [], []
        for snr in snrs:
            values = [float(row[metric]) for row in paired_rows if float(row["snr_db"]) == snr]
            means.append(statistics.mean(values))
            sds.append(statistics.stdev(values))
        means_array, sd_array = np.array(means), np.array(sds)
        axis.axhline(0, color="black", linewidth=1)
        axis.plot(snrs, means_array, "o-", color="tab:green")
        axis.fill_between(snrs, means_array - sd_array, means_array + sd_array, color="tab:green", alpha=0.2)
        axis.set_xlabel("Test SNR (dB)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[0].set_title("Noise-training gain: reconstruction")
    axes[1].set_title("Noise-training gain: classification")
    fig.tight_layout()
    fig.savefig(ANALYSIS / "noise_training_gain.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    target_snr = 10.0
    target = [row for row in paired_rows if float(row["snr_db"]) == target_snr]
    r_diff = [float(row["r_psnr_gain_db_r1_minus_r0"]) for row in target]
    c_diff = [float(row["c_accuracy_gain_pp_c1_minus_c0"]) for row in target]
    r_mean, r_sd, r_ci = mean_sd_ci(r_diff)
    c_mean, c_sd, c_ci = mean_sd_ci(c_diff)
    summary = {
        "training_runs": 12,
        "training_seeds_per_cell": 3,
        "clean_test_images_per_snr_per_seed": 10000,
        "channel_repeats_per_snr_per_seed": 3,
        "snr_grid_db": sorted({float(row["snr_db"]) for row in clean_rows}),
        "at_10db": {
            "reconstruction_psnr_gain_r1_minus_r0_db": {
                "values": r_diff,
                "mean": r_mean,
                "sd": r_sd,
                "ci95_t_df2": [r_mean - r_ci, r_mean + r_ci],
                "paired_standardized_mean_difference_dz": paired_standardized_mean_difference(r_diff),
                "paired_sign_test_two_sided_p": sign_test_two_sided_nonzero(r_diff),
            },
            "classification_accuracy_gain_c1_minus_c0_pp": {
                "values": c_diff,
                "mean": c_mean,
                "sd": c_sd,
                "ci95_t_df2": [c_mean - c_ci, c_mean + c_ci],
                "paired_standardized_mean_difference_dz": paired_standardized_mean_difference(c_diff),
                "paired_sign_test_two_sided_p": sign_test_two_sided_nonzero(c_diff),
            },
        },
        "inference_note": "n=3 seeds; exact two-sided sign test cannot be below 0.25 when all three directions agree.",
    }
    (ANALYSIS / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    attack_root = ANALYSIS / "pga_10db_128"
    if attack_root.is_dir():
        attack_rows: list[dict[str, Any]] = []
        attack_seed_rows: list[dict[str, Any]] = []
        for cell in CELLS:
            for seed in SEEDS:
                samples = read_csv(attack_root / f"{cell.lower()}_seed{seed}" / "attack_samples.csv")
                eligible = (
                    [row for row in samples if int(row["clean_correct"]) == 1]
                    if cell.startswith("C")
                    else samples
                )
                powers = np.array(
                    [float(row["attack_power_total_l2_sq"]) for row in eligible]
                )
                steps = np.array([int(row["steps"]) for row in eligible])
                margins = np.array(
                    [
                        float(row["clean_logit_margin"])
                        if cell.startswith("C")
                        else float(row["target_distortion"])
                        - float(row["clean_distortion"])
                        for row in eligible
                    ]
                )
                for row in eligible:
                    attack_rows.append(
                        {
                            "cell": cell,
                            "seed": seed,
                            "sample_index": int(row["sample_index"]),
                            "clean_margin_to_failure": (
                                float(row["clean_logit_margin"])
                                if cell.startswith("C")
                                else float(row["target_distortion"])
                                - float(row["clean_distortion"])
                            ),
                            "attack_power_total_l2_sq": float(
                                row["attack_power_total_l2_sq"]
                            ),
                            "attack_power_per_channel_use": float(
                                row["attack_power_per_channel_use"]
                            ),
                            "steps": int(row["steps"]),
                            "success": int(row["success"]),
                        }
                    )
                correlation = (
                    float(np.corrcoef(margins, powers)[0, 1])
                    if len(margins) > 1 and margins.std() > 0 and powers.std() > 0
                    else None
                )
                attack_seed_rows.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "total_test_samples": len(samples),
                        "eligible_samples": len(eligible),
                        "success_rate": float(
                            np.mean([int(row["success"]) for row in eligible])
                        ),
                        "initial_failure_rate_steps0": float(np.mean(steps == 0)),
                        "mean_attack_power_total_l2_sq": float(powers.mean()),
                        "median_attack_power_total_l2_sq": float(np.median(powers)),
                        "q25_attack_power_total_l2_sq": float(np.quantile(powers, 0.25)),
                        "q75_attack_power_total_l2_sq": float(np.quantile(powers, 0.75)),
                        "mean_attack_power_per_channel_use": float(powers.mean() / 768.0),
                        "mean_clean_margin_to_failure": float(margins.mean()),
                        "pearson_clean_margin_vs_attack_power": correlation,
                        "mean_steps": float(steps.mean()),
                        "max_steps_observed": int(steps.max()),
                    }
                )
        write_csv(ANALYSIS / "pga_10db_all_eligible_samples.csv", attack_rows)
        write_csv(ANALYSIS / "pga_10db_seed_summary.csv", attack_seed_rows)

        pga_effects: list[dict[str, Any]] = []
        by_attack = {
            (row["cell"], int(row["seed"])): row for row in attack_seed_rows
        }
        for seed in SEEDS:
            for objective, off, on in (
                ("reconstruction", "R0", "R1"),
                ("classification", "C0", "C1"),
            ):
                off_row, on_row = by_attack[off, seed], by_attack[on, seed]
                pga_effects.append(
                    {
                        "objective": objective,
                        "seed": seed,
                        "off_cell": off,
                        "on_cell": on,
                        "mean_power_off": off_row["mean_attack_power_total_l2_sq"],
                        "mean_power_on": on_row["mean_attack_power_total_l2_sq"],
                        "mean_power_ratio_on_over_off": float(
                            on_row["mean_attack_power_total_l2_sq"]
                        )
                        / float(off_row["mean_attack_power_total_l2_sq"]),
                        "median_power_ratio_on_over_off": float(
                            on_row["median_attack_power_total_l2_sq"]
                        )
                        / float(off_row["median_attack_power_total_l2_sq"]),
                    }
                )
        write_csv(ANALYSIS / "pga_10db_paired_effects.csv", pga_effects)

        # Fair classification comparison: keep only images that both C0 and C1
        # classify correctly before attack within the same seed.
        common_correct_rows: list[dict[str, Any]] = []
        common_correct_effects: list[dict[str, Any]] = []
        for seed in SEEDS:
            cell_maps: dict[str, dict[int, dict[str, str]]] = {}
            for cell in ("C0", "C1"):
                samples = read_csv(
                    attack_root / f"{cell.lower()}_seed{seed}" / "attack_samples.csv"
                )
                cell_maps[cell] = {
                    int(row["sample_index"]): row
                    for row in samples
                    if int(row["clean_correct"]) == 1
                }
            common = sorted(set(cell_maps["C0"]) & set(cell_maps["C1"]))
            per_cell_power: dict[str, list[float]] = {}
            for cell in ("C0", "C1"):
                powers = [
                    float(cell_maps[cell][index]["attack_power_total_l2_sq"])
                    for index in common
                ]
                margins = [
                    float(cell_maps[cell][index]["clean_logit_margin"])
                    for index in common
                ]
                per_cell_power[cell] = powers
                common_correct_rows.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "common_clean_correct_samples": len(common),
                        "mean_attack_power_total_l2_sq": statistics.mean(powers),
                        "median_attack_power_total_l2_sq": statistics.median(powers),
                        "mean_clean_logit_margin": statistics.mean(margins),
                    }
                )
            per_sample_ratios = [
                per_cell_power["C1"][index] / per_cell_power["C0"][index]
                for index in range(len(common))
                if per_cell_power["C0"][index] > 0
            ]
            common_correct_effects.append(
                {
                    "seed": seed,
                    "common_clean_correct_samples": len(common),
                    "c0_mean_power": statistics.mean(per_cell_power["C0"]),
                    "c1_mean_power": statistics.mean(per_cell_power["C1"]),
                    "ratio_of_mean_power_c1_over_c0": statistics.mean(
                        per_cell_power["C1"]
                    )
                    / statistics.mean(per_cell_power["C0"]),
                    "median_per_sample_power_ratio_c1_over_c0": statistics.median(
                        per_sample_ratios
                    ),
                }
            )
        write_csv(
            ANALYSIS / "pga_10db_classification_common_correct_summary.csv",
            common_correct_rows,
        )
        write_csv(
            ANALYSIS / "pga_10db_classification_common_correct_effects.csv",
            common_correct_effects,
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for axis, cells, title in (
            (axes[0], ("R0", "R1"), "Reconstruction PGA at 10 dB"),
            (axes[1], ("C0", "C1"), "Classification PGA at 10 dB"),
        ):
            values = [
                [
                    float(by_attack[cell, seed]["mean_attack_power_total_l2_sq"])
                    for seed in SEEDS
                ]
                for cell in cells
            ]
            x = np.arange(len(cells))
            means = np.array([statistics.mean(value) for value in values])
            sds = np.array([statistics.stdev(value) for value in values])
            axis.bar(x, means, yerr=sds, capsize=5, color=["tab:orange", "tab:blue"], alpha=0.75)
            for seed_index, seed in enumerate(SEEDS):
                axis.plot(x, [values[0][seed_index], values[1][seed_index]], "o-", color="black", alpha=0.55, linewidth=1)
            axis.set_xticks(x, cells)
            axis.set_yscale("log")
            axis.set_ylabel("Mean total squared L2 attack power")
            axis.set_title(title)
            axis.grid(True, axis="y", which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(ANALYSIS / "pga_10db_attack_power.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        r_ratios = [
            float(row["mean_power_ratio_on_over_off"])
            for row in pga_effects
            if row["objective"] == "reconstruction"
        ]
        c_ratios = [
            float(row["mean_power_ratio_on_over_off"])
            for row in pga_effects
            if row["objective"] == "classification"
        ]
        summary["exploratory_pga_10db_128"] = {
            "status": "all 12 cells completed; all eligible samples attacked successfully",
            "reconstruction_mean_power_ratio_r1_over_r0": {
                "values": r_ratios,
                "geometric_mean": float(np.exp(np.mean(np.log(r_ratios)))),
                "paired_log_ratio_dz": paired_standardized_mean_difference(
                    [float(value) for value in np.log(r_ratios)]
                ),
            },
            "classification_mean_power_ratio_c1_over_c0": {
                "values": c_ratios,
                "geometric_mean": float(np.exp(np.mean(np.log(c_ratios)))),
                "paired_log_ratio_dz": paired_standardized_mean_difference(
                    [float(value) for value in np.log(c_ratios)]
                ),
            },
            "classification_common_clean_correct_ratio_c1_over_c0": {
                "common_samples_by_seed": [
                    int(row["common_clean_correct_samples"])
                    for row in common_correct_effects
                ],
                "ratio_of_means_by_seed": [
                    float(row["ratio_of_mean_power_c1_over_c0"])
                    for row in common_correct_effects
                ],
                "geometric_mean_ratio": float(
                    np.exp(
                        np.mean(
                            np.log(
                                [
                                    float(row["ratio_of_mean_power_c1_over_c0"])
                                    for row in common_correct_effects
                                ]
                            )
                        )
                    )
                ),
                "paired_log_ratio_dz": paired_standardized_mean_difference(
                    [
                        math.log(float(row["ratio_of_mean_power_c1_over_c0"]))
                        for row in common_correct_effects
                    ]
                ),
            },
            "limitations": [
                "single deterministic PGA start",
                "128 test samples per model",
                "reconstruction uses fixed 15 dB threshold",
                "clean margin is not matched across cells",
                "no C&W or independent solver cross-check",
            ],
        }
        (ANALYSIS / "analysis_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
