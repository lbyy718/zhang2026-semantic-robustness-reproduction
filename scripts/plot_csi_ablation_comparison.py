"""Plot a reproducible clean-NMSE comparison from two evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8") as stream:
        return [
            {
                "snr_db": float(row["snr_db"]),
                "mean_nmse_db": float(row["mean_nmse_db"]),
                "repeat_std_nmse_db": float(row["repeat_std_nmse_db"]),
            }
            for row in csv.DictReader(stream)
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--improved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_rows(args.baseline)
    improved = read_rows(args.improved)
    if [row["snr_db"] for row in baseline] != [row["snr_db"] for row in improved]:
        raise ValueError("Baseline and improved CSV files must use the same SNR points.")

    rows = []
    for old, new in zip(baseline, improved, strict=True):
        rows.append(
            {
                "snr_db": old["snr_db"],
                "baseline_nmse_db": old["mean_nmse_db"],
                "improved_nmse_db": new["mean_nmse_db"],
                "nmse_gain_db": old["mean_nmse_db"] - new["mean_nmse_db"],
                "baseline_repeat_std_db": old["repeat_std_nmse_db"],
                "improved_repeat_std_db": new["repeat_std_nmse_db"],
            }
        )

    args.comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.comparison_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    snr = [row["snr_db"] for row in rows]
    baseline_nmse = [row["baseline_nmse_db"] for row in rows]
    improved_nmse = [row["improved_nmse_db"] for row in rows]
    baseline_std = [row["baseline_repeat_std_db"] for row in rows]
    improved_std = [row["improved_repeat_std_db"] for row in rows]
    gains = [row["nmse_gain_db"] for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].errorbar(
        snr,
        baseline_nmse,
        yerr=baseline_std,
        marker="o",
        linewidth=2,
        capsize=2,
        label="Original 3x3 baseline (200 epochs)",
    )
    axes[0].errorbar(
        snr,
        improved_nmse,
        yerr=improved_std,
        marker="s",
        linewidth=2,
        capsize=2,
        label="Complex-256 + global mixing (200 epochs)",
    )
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Mean NMSE (dB, lower is better)")
    axes[0].set_title("COST2100 clean reconstruction")
    axes[0].legend(frameon=True)
    axes[0].annotate(
        f"{improved_nmse[-1]:.3f} dB",
        (snr[-1], improved_nmse[-1]),
        xytext=(-70, 12),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )

    axes[1].plot(snr, gains, marker="o", linewidth=2)
    axes[1].axhline(3.0, linestyle="--", linewidth=1.5, label="3 dB threshold")
    axes[1].fill_between(snr, gains, 3.0, where=[gain >= 3.0 for gain in gains], alpha=0.18)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("NMSE gain over baseline (dB)")
    axes[1].set_title("Ablation improvement")
    axes[1].legend(frameon=True)
    axes[1].annotate(
        f"{gains[-1]:.3f} dB",
        (snr[-1], gains[-1]),
        xytext=(-72, -28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    axes[1].set_ylim(min(gains) - 0.15, max(gains) + 0.18)

    figure.suptitle("Five-repeat, 20,000-sample CSI ablation verification")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
