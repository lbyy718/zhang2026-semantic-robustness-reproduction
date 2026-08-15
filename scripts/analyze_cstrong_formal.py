"""Analyze formal CS0/CSJ diagnostics across training seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import stdev
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "cstrong_formal_lr001"
DEFAULT_RESULTS = REPO_ROOT / "results" / "cstrong_formal_lr001_two_seed"
DEFAULT_REPORT = REPO_ROOT / "docs" / "13_C-strong无噪声Jacobian正则化两种子结果.md"
ARMS = ("CS0", "CSJ")
REPEATS = (0, 1, 2)
RATIO_FIELDS = (
    "failure_margin",
    "failure_gradient_l2",
    "linearized_distance",
    "spectral_norm",
    "normalized_spectral_norm",
    "centered_logit_rms",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item) for item in value]
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else float("nan")


def geometric_ratio(pairs: Iterable[tuple[float, float]]) -> tuple[float, int]:
    logs = [
        math.log(on / off)
        for off, on in pairs
        if off > 0 and on > 0 and math.isfinite(off) and math.isfinite(on)
    ]
    return (math.exp(mean(logs)), len(logs)) if logs else (float("nan"), 0)


def load_training(root: Path, seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            directory = root / f"{arm.lower()}_seed{seed}"
            manifest_path = directory / "manifest.json"
            log_path = directory / "training_log.csv"
            checkpoint_path = directory / "checkpoint_best.pt"
            if not all(path.is_file() for path in (manifest_path, log_path, checkpoint_path)):
                raise FileNotFoundError(f"Incomplete training artifacts: {arm} seed {seed}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                raise RuntimeError(f"Training not completed: {arm} seed {seed}")
            if manifest.get("arm") != arm or int(manifest.get("seed", -1)) != seed:
                raise RuntimeError(f"Training identity mismatch: {arm} seed {seed}")
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "completed_epochs": int(manifest["completed_epochs"]),
                    "best_epoch": int(manifest["best_epoch"]),
                    "test_noiseless_accuracy": float(manifest["test_noiseless_accuracy"]),
                    "test_10db_accuracy": float(manifest["test_10db_accuracy"]),
                    "test_noiseless_cross_entropy": float(
                        manifest["test_noiseless_cross_entropy"]
                    ),
                    "test_10db_cross_entropy": float(manifest["test_10db_cross_entropy"]),
                    "quality_gate": bool(manifest["individual_training_quality_gate_passed"]),
                }
            )
    return rows


def load_diagnostics(
    root: Path, seeds: list[int]
) -> dict[tuple[str, int, int], list[dict[str, str]]]:
    loaded: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    reference_indices: dict[int, list[str]] = {}
    reference_noise: dict[int, str] = {}
    for seed in seeds:
        for arm in ARMS:
            for repeat in REPEATS:
                directory = (
                    root
                    / f"{arm.lower()}_seed{seed}"
                    / "diagnostics"
                    / f"repeat{repeat}"
                )
                manifest_path = directory / "manifest.json"
                csv_path = directory / "diagnostics.csv"
                if not manifest_path.is_file() or not csv_path.is_file():
                    raise FileNotFoundError(
                        f"Incomplete diagnostics: {arm} seed {seed} repeat {repeat}"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                signature = manifest.get("job_signature", {})
                if manifest.get("status") != "completed":
                    raise RuntimeError(
                        f"Diagnostics not completed: {arm} seed {seed} repeat {repeat}"
                    )
                if (
                    signature.get("arm") != arm
                    or int(signature.get("training_seed", -1)) != seed
                    or int(signature.get("repeat_index", -1)) != repeat
                ):
                    raise RuntimeError("Diagnostic identity mismatch.")
                if manifest.get("diagnostics_sha256") != sha256(csv_path):
                    raise RuntimeError("Diagnostic CSV hash mismatch.")
                rows = read_csv(csv_path)
                if len(rows) != int(manifest.get("rows", -1)):
                    raise RuntimeError("Diagnostic row count mismatch.")
                indices = [row["dataset_index"] for row in rows]
                noise_hash = str(signature["standard_noise_sha256"])
                if repeat in reference_indices and reference_indices[repeat] != indices:
                    raise RuntimeError(f"Shared sample order failed at repeat {repeat}.")
                if repeat in reference_noise and reference_noise[repeat] != noise_hash:
                    raise RuntimeError(f"Shared channel noise failed at repeat {repeat}.")
                reference_indices[repeat] = indices
                reference_noise[repeat] = noise_hash
                loaded[(arm, seed, repeat)] = rows
    return loaded


def arm_seed_summary(
    records: dict[tuple[str, int, int], list[dict[str, str]]], seeds: list[int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            repeat_rows: list[dict[str, float]] = []
            for repeat in REPEATS:
                data = records[(arm, seed, repeat)]
                correct = [row for row in data if int(row["clean_correct"])]
                successful = [row for row in correct if int(row["pga_success"])]
                repeat_rows.append(
                    {
                        "clean_accuracy": mean(int(row["clean_correct"]) for row in data),
                        "clean_cross_entropy": mean(
                            float(row["clean_cross_entropy"]) for row in data
                        ),
                        "failure_margin": mean(float(row["failure_margin"]) for row in correct),
                        "failure_gradient_l2": mean(
                            float(row["failure_gradient_l2"]) for row in correct
                        ),
                        "spectral_norm": mean(float(row["spectral_norm"]) for row in correct),
                        "normalized_spectral_norm": mean(
                            float(row["normalized_spectral_norm"]) for row in correct
                        ),
                        "linearized_distance": mean(
                            float(row["linearized_distance"]) for row in correct
                        ),
                        "pga_l2_successful": mean(float(row["pga_l2"]) for row in successful),
                        "pga_success_rate": len(successful) / len(correct) if correct else float("nan"),
                        "spectral_convergence_rate": mean(
                            int(row["spectral_converged"]) for row in data
                        ),
                        "clean_correct_samples": float(len(correct)),
                    }
                )
            result.append(
                {
                    "arm": arm,
                    "seed": seed,
                    **{
                        field: mean(row[field] for row in repeat_rows)
                        for field in repeat_rows[0]
                    },
                }
            )
    return result


def paired_seed_effects(
    records: dict[tuple[str, int, int], list[dict[str, str]]], seeds: list[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repeat_effects: list[dict[str, Any]] = []
    seed_effects: list[dict[str, Any]] = []
    for seed in seeds:
        per_repeat: list[dict[str, Any]] = []
        for repeat in REPEATS:
            off = records[("CS0", seed, repeat)]
            on = records[("CSJ", seed, repeat)]
            common = [
                (left, right)
                for left, right in zip(off, on, strict=True)
                if int(left["clean_correct"]) and int(right["clean_correct"])
            ]
            row: dict[str, Any] = {
                "seed": seed,
                "repeat": repeat,
                "pairwise_common_correct": len(common),
            }
            for field in RATIO_FIELDS:
                ratio, count = geometric_ratio(
                    (float(left[field]), float(right[field])) for left, right in common
                )
                row[f"{field}_ratio"] = ratio
                row[f"{field}_pairs"] = count
            both = [
                (left, right)
                for left, right in common
                if int(left["pga_success"]) and int(right["pga_success"])
            ]
            ratio, count = geometric_ratio(
                (float(left["pga_l2"]), float(right["pga_l2"])) for left, right in both
            )
            row["pga_l2_complete_case_ratio"] = ratio
            row["pga_l2_complete_case_pairs"] = count
            row["pga_success_both"] = len(both)
            row["pga_success_cs0_only"] = sum(
                int(left["pga_success"]) and not int(right["pga_success"])
                for left, right in common
            )
            row["pga_success_csj_only"] = sum(
                not int(left["pga_success"]) and int(right["pga_success"])
                for left, right in common
            )
            row["pga_success_neither"] = sum(
                not int(left["pga_success"]) and not int(right["pga_success"])
                for left, right in common
            )
            per_repeat.append(row)
            repeat_effects.append(row)
        aggregate: dict[str, Any] = {
            "seed": seed,
            "pairwise_common_correct_mean": mean(
                row["pairwise_common_correct"] for row in per_repeat
            ),
        }
        ratio_names = [f"{field}_ratio" for field in RATIO_FIELDS] + [
            "pga_l2_complete_case_ratio"
        ]
        for name in ratio_names:
            values = [float(row[name]) for row in per_repeat]
            aggregate[name] = math.exp(mean(math.log(value) for value in values))
        for name in (
            "pga_success_both",
            "pga_success_cs0_only",
            "pga_success_csj_only",
            "pga_success_neither",
        ):
            aggregate[f"{name}_mean"] = mean(float(row[name]) for row in per_repeat)
        seed_effects.append(aggregate)
    return repeat_effects, seed_effects


def aggregate_seed_effects(seed_effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    ratio_names = [f"{field}_ratio" for field in RATIO_FIELDS] + [
        "pga_l2_complete_case_ratio"
    ]
    for name in ratio_names:
        logs = [math.log(float(row[name])) for row in seed_effects]
        result.append(
            {
                "metric": name.removesuffix("_ratio"),
                "geometric_mean_ratio_csj_over_cs0": math.exp(mean(logs)),
                "log_ratio_sd_across_seeds": stdev(logs) if len(logs) >= 2 else float("nan"),
                "seed_ratios": ";".join(
                    f"{row['seed']}:{float(row[name]):.8g}" for row in seed_effects
                ),
                "direction_consistent": all(value > 0 for value in logs)
                or all(value < 0 for value in logs),
            }
        )
    return result


def robust_curves(
    records: dict[tuple[str, int, int], list[dict[str, str]]], seeds: list[int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for arm in ARMS:
        for budget_db in range(-40, 1):
            budget = 10.0 ** (budget_db / 10.0)
            seed_lower: list[float] = []
            seed_upper: list[float] = []
            for seed in seeds:
                repeat_lower: list[float] = []
                repeat_upper: list[float] = []
                for repeat in REPEATS:
                    data = records[(arm, seed, repeat)]
                    lower = upper = 0
                    for row in data:
                        if not int(row["clean_correct"]):
                            continue
                        if int(row["pga_success"]):
                            robust = float(row["pga_power_per_channel_use"]) > budget
                            lower += int(robust)
                            upper += int(robust)
                        else:
                            upper += 1
                    repeat_lower.append(lower / len(data))
                    repeat_upper.append(upper / len(data))
                seed_lower.append(mean(repeat_lower))
                seed_upper.append(mean(repeat_upper))
            result.append(
                {
                    "arm": arm,
                    "budget_db_per_channel_use": budget_db,
                    "pga_empirical_robust_accuracy_lower_mean": mean(seed_lower),
                    "pga_empirical_robust_accuracy_upper_mean": mean(seed_upper),
                    "pga_empirical_robust_accuracy_lower_by_seed": ";".join(
                        f"{seed}:{value:.8g}" for seed, value in zip(seeds, seed_lower, strict=True)
                    ),
                    "pga_empirical_robust_accuracy_upper_by_seed": ";".join(
                        f"{seed}:{value:.8g}" for seed, value in zip(seeds, seed_upper, strict=True)
                    ),
                }
            )
    return result


def clean_curves(root: Path, seeds: list[int]) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for seed in seeds:
        for arm in ARMS:
            rows = read_csv(root / f"{arm.lower()}_seed{seed}" / "test_clean_curve.csv")
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                grouped[row["snr_db"]].append(row)
            for snr, group in grouped.items():
                values[(arm, snr)].append(mean(float(row["accuracy"]) for row in group))
    order = ["noiseless", *[str(value) for value in range(0, 21, 2)]]
    return [
        {
            "arm": arm,
            "snr_db": snr,
            "accuracy_mean_across_seeds": mean(values[(arm, snr)]),
            "accuracy_by_seed": ";".join(
                f"{seed}:{value:.8g}"
                for seed, value in zip(seeds, values[(arm, snr)], strict=True)
            ),
        }
        for arm in ARMS
        for snr in order
    ]


def figures(
    result_root: Path,
    training: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    curves: list[dict[str, Any]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    seeds = sorted({int(row["seed"]) for row in training})
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(len(seeds))
    width = 0.35
    for offset, arm in ((-width / 2, "CS0"), (width / 2, "CSJ")):
        rows = [row for row in training if row["arm"] == arm]
        axis.bar(x + offset, [row["test_10db_accuracy"] for row in rows], width, label=arm)
    axis.set_xticks(x, [str(seed) for seed in seeds])
    axis.set(ylim=(0.85, 1.0), xlabel="Training seed", ylabel="10 dB test accuracy")
    axis.legend()
    fig.tight_layout()
    fig.savefig(result_root / "test_accuracy_by_seed.png", dpi=180)
    plt.close(fig)

    selected = {
        row["metric"]: row["geometric_mean_ratio_csj_over_cs0"] for row in aggregate
    }
    names = [
        "failure_margin",
        "failure_gradient_l2",
        "normalized_spectral_norm",
        "linearized_distance",
        "pga_l2_complete_case",
    ]
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(np.arange(len(names)), [selected[name] for name in names])
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(names)), names, rotation=20, ha="right")
    axis.set(ylabel="Geometric ratio: CSJ / CS0", title="Two-seed local and attack effects")
    fig.tight_layout()
    fig.savefig(result_root / "csj_over_cs0_effect_ratios.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for arm in ARMS:
        rows = [row for row in curves if row["arm"] == arm]
        axis.plot(
            [row["budget_db_per_channel_use"] for row in rows],
            [row["pga_empirical_robust_accuracy_upper_mean"] for row in rows],
            label=arm,
        )
    axis.set(
        xlabel="Attack power per channel use (dB)",
        ylabel="PGA-estimated robust accuracy",
        title="Empirical attack curves (not certified)",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(result_root / "pga_empirical_robust_curves.png", dpi=180)
    plt.close(fig)


def report_text(summary: dict[str, Any]) -> str:
    effects = {row["metric"]: row for row in summary["aggregate_effects"]}
    training = summary["training_summary"]
    by_arm = {
        arm: [row for row in training if row["arm"] == arm] for arm in ARMS
    }

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    def accuracy(arm: str) -> str:
        return pct(mean(row["test_10db_accuracy"] for row in by_arm[arm]))

    def ratio(field: str) -> str:
        return f"{effects[field]['geometric_mean_ratio_csj_over_cs0']:.3f}x"

    return f"""# C-strong 无噪声 Jacobian 正则化两种子结果

> 本报告由 `scripts/analyze_cstrong_formal.py` 从锁定训练与诊断产物生成。当前只有两个独立训练种子，只作描述性复盘，不进行显著性检验。

## 1. 实验范围

- 模型：同一 `resnet18_bottleneck` 架构。
- CS0：无 AWGN、无 Jacobian 正则化。
- CSJ：无 AWGN、加入 latent failure-score Jacobian penalty。
- 已完成种子：{', '.join(str(seed) for seed in summary['seeds'])}。
- 2028 两组按用户决定延期，因此本报告不是原计划的三种子终局分析。

## 2. Clean 性能

| 模型 | 两种子平均 10 dB test accuracy |
|---|---:|
| CS0 | {accuracy('CS0')} |
| CSJ | {accuracy('CSJ')} |

逐种子准确率、最佳 epoch 和交叉熵见 `training_summary.csv`。

## 3. 共同正确样本上的局部与攻击比值

比值均为 `CSJ / CS0`。每个训练种子内部先对三个共享信道重复聚合，再对训练种子的对数比取平均。

| 指标 | CSJ / CS0 |
|---|---:|
| clean failure margin | {ratio('failure_margin')} |
| failure gradient L2 | {ratio('failure_gradient_l2')} |
| raw Jacobian spectrum | {ratio('spectral_norm')} |
| normalized Jacobian spectrum | {ratio('normalized_spectral_norm')} |
| linearized failure distance | {ratio('linearized_distance')} |
| PGA L2（两侧均成功的 complete cases） | {ratio('pga_l2_complete_case')} |

`normalized spectrum = raw spectrum / centered-logit RMS`，用于降低单纯 logits 缩放造成的伪平滑影响。

## 4. 解释规则

{summary['interpretation']}

## 5. 结论边界

- 两个训练种子不足以支持可靠显著性检验或普适外推。
- pairwise-common-correct 是处理后的条件子集，可能存在幸存者/碰撞变量选择偏差。
- PGA 是单一固定攻击的经验估计，不是认证鲁棒性；complete-case 比值必须结合删失模式阅读。
- 本批次没有 CS1，不能直接比较 AWGN 训练与 Jacobian 正则化谁更优，也不能单独识别 AWGN 的机制。
- 即使 CSJ 改善局部谱和 PGA，也只能说明显式局部平滑在当前架构上足以产生相似现象。

## 6. 产物

- `results/cstrong_formal_lr001_two_seed/training_summary.csv`
- `results/cstrong_formal_lr001_two_seed/arm_seed_summary.csv`
- `results/cstrong_formal_lr001_two_seed/repeat_effects.csv`
- `results/cstrong_formal_lr001_two_seed/seed_effects.csv`
- `results/cstrong_formal_lr001_two_seed/aggregate_effects.csv`
- `results/cstrong_formal_lr001_two_seed/robust_curves.csv`
- `results/cstrong_formal_lr001_two_seed/clean_curves.csv`
- 三张 PNG 图和 `analysis_summary.json`
"""


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    command.add_argument("--result-root", type=Path, default=DEFAULT_RESULTS)
    command.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    command.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    return command


def main() -> int:
    args = parser().parse_args()
    seeds = list(args.seeds)
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain unique non-negative integers.")
    root = args.input_root.resolve()
    result_root = args.result_root.resolve()
    report = args.report.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    training = load_training(root, seeds)
    records = load_diagnostics(root, seeds)
    arm_summary = arm_seed_summary(records, seeds)
    repeat_effects, seed_effects = paired_seed_effects(records, seeds)
    aggregate = aggregate_seed_effects(seed_effects)
    curves = robust_curves(records, seeds)
    clean = clean_curves(root, seeds)
    effects = {row["metric"]: row for row in aggregate}
    smooth = effects["normalized_spectral_norm"][
        "geometric_mean_ratio_csj_over_cs0"
    ] < 1.0
    attack = effects["pga_l2_complete_case"][
        "geometric_mean_ratio_csj_over_cs0"
    ] > 1.0
    raw_only = (
        effects["spectral_norm"]["geometric_mean_ratio_csj_over_cs0"] < 1.0
        and not smooth
        and not attack
    )
    if smooth and attack:
        interpretation = (
            "CSJ 同时降低 normalized Jacobian spectrum 并提高 PGA L2；结果与“显式局部平滑足以复现经验鲁棒收益”相容。"
            "这不是因果机制证明，也不能替代缺失的 CS1/AWGN 对照。"
        )
    elif raw_only:
        interpretation = (
            "CSJ 只降低 raw spectrum，而 normalized spectrum 与 PGA 没有同步改善；应视为可能的 logits 缩放伪象。"
        )
    elif smooth and not attack:
        interpretation = (
            "CSJ 降低了 normalized spectrum，但没有转化为更大的 PGA 失败距离；局部平滑信号尚不足以称为经验鲁棒提升。"
        )
    elif attack and not smooth:
        interpretation = (
            "CSJ 提高了 PGA 距离，但 normalized spectrum 未下降；收益可能来自 margin 或非局部几何，不能归因于 Jacobian 平滑。"
        )
    else:
        interpretation = "CSJ 未稳定改善 normalized spectrum 或 PGA 距离，当前强架构下显式正则化收益不成立。"

    diagnostic_batch = json.loads(
        (root / "diagnostics_batch_manifest.json").read_text(encoding="utf-8")
    )
    summary = {
        "schema_version": "cstrong_formal_two_seed_analysis_v1",
        "seeds": seeds,
        "arms": list(ARMS),
        "inference_scope": "descriptive two-training-seed comparison",
        "training_summary": training,
        "arm_seed_summary": arm_summary,
        "seed_effects": seed_effects,
        "aggregate_effects": aggregate,
        "technical_gate": {
            "training_all_individual_gates_passed": all(row["quality_gate"] for row in training),
            "diagnostics_gate_passed": diagnostic_batch.get(
                "diagnostics_technical_gate_passed"
            ),
            "details": diagnostic_batch.get("technical_gate_details", {}),
        },
        "interpretation": interpretation,
        "cautions": [
            "Only two independent training seeds are available.",
            "Pairwise-common-correct is a post-treatment conditional subset.",
            "PGA results are attack-estimated, not certified robustness.",
            "PGA L2 ratios use complete cases when both attacks succeed.",
            "CS1/AWGN is absent from this stopped batch.",
        ],
    }
    write_csv(result_root / "training_summary.csv", training)
    write_csv(result_root / "arm_seed_summary.csv", arm_summary)
    write_csv(result_root / "repeat_effects.csv", repeat_effects)
    write_csv(result_root / "seed_effects.csv", seed_effects)
    write_csv(result_root / "aggregate_effects.csv", aggregate)
    write_csv(result_root / "robust_curves.csv", curves)
    write_csv(result_root / "clean_curves.csv", clean)
    figures(result_root, training, aggregate, curves)
    atomic_json(result_root / "analysis_summary.json", summary)
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(report_text(summary), encoding="utf-8")
    os.replace(temporary, report)
    print(
        json.dumps(
            {"status": "completed", "results": str(result_root), "report": str(report)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
