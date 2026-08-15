"""Analyze completed C-strong diagnostics and write figures plus a Chinese report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "cstrong_pilot"
DEFAULT_RESULTS = REPO_ROOT / "results" / "cstrong_pilot_seed2026"
DEFAULT_REPORT = REPO_ROOT / "docs" / "12_C-strong与Jacobian正则化实验结果.md"
ARMS = ("CS0", "CS1", "CSJ")
COMPARISONS = (("CS0", "CS1"), ("CS0", "CSJ"), ("CS1", "CSJ"))
REPEATS = (0, 1, 2)


def atomic_json(path: Path, payload: Any) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else float("nan")


def geometric_ratio(pairs: Iterable[tuple[float, float]]) -> tuple[float, int]:
    logs = [math.log(on / off) for off, on in pairs if off > 0 and on > 0 and math.isfinite(off) and math.isfinite(on)]
    return (math.exp(mean(logs)), len(logs)) if logs else (float("nan"), 0)


def validate_and_load(root: Path, seed: int) -> dict[tuple[str, int], list[dict[str, str]]]:
    loaded: dict[tuple[str, int], list[dict[str, str]]] = {}
    reference_indices: dict[int, list[str]] = {}
    reference_noise: dict[int, str] = {}
    for arm in ARMS:
        for repeat in REPEATS:
            directory = root / f"{arm.lower()}_seed{seed}" / "diagnostics" / f"repeat{repeat}"
            manifest_path = directory / "manifest.json"
            csv_path = directory / "diagnostics.csv"
            if not manifest_path.is_file() or not csv_path.is_file():
                raise FileNotFoundError(f"Incomplete diagnostic job: {arm} repeat {repeat}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                raise RuntimeError(f"Diagnostic job not completed: {arm} repeat {repeat}")
            rows = read_rows(csv_path)
            if len(rows) != int(manifest.get("rows", -1)):
                raise RuntimeError(f"Row count mismatch: {arm} repeat {repeat}")
            indices = [row["dataset_index"] for row in rows]
            noise_hash = manifest["job_signature"]["standard_noise_sha256"]
            if repeat in reference_indices and reference_indices[repeat] != indices:
                raise RuntimeError(f"Shared sample identity failed at repeat {repeat}.")
            if repeat in reference_noise and reference_noise[repeat] != noise_hash:
                raise RuntimeError(f"Shared noise identity failed at repeat {repeat}.")
            reference_indices[repeat] = indices
            reference_noise[repeat] = noise_hash
            loaded[(arm, repeat)] = rows
    return loaded


def arm_summaries(records: dict[tuple[str, int], list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        repeat_metrics: list[dict[str, float]] = []
        for repeat in REPEATS:
            data = records[(arm, repeat)]
            correct = [row for row in data if int(row["clean_correct"])]
            successful = [row for row in correct if int(row["pga_success"])]
            repeat_metrics.append(
                {
                    "clean_accuracy": mean(int(row["clean_correct"]) for row in data),
                    "clean_ce": mean(float(row["clean_cross_entropy"]) for row in data),
                    "clean_margin_correct": mean(float(row["failure_margin"]) for row in correct),
                    "gradient_correct": mean(float(row["failure_gradient_l2"]) for row in correct),
                    "spectral_correct": mean(float(row["spectral_norm"]) for row in correct),
                    "normalized_spectral_correct": mean(float(row["normalized_spectral_norm"]) for row in correct),
                    "linearized_distance_correct": mean(float(row["linearized_distance"]) for row in correct),
                    "pga_l2_successful_correct": mean(float(row["pga_l2"]) for row in successful),
                    "pga_success_rate_correct": len(successful) / len(correct) if correct else float("nan"),
                    "spectral_convergence_rate": mean(int(row["spectral_converged"]) for row in data),
                    "clean_correct_samples": float(len(correct)),
                }
            )
        aggregate = {name: mean(item[name] for item in repeat_metrics) for name in repeat_metrics[0]}
        rows.append({"arm": arm, **aggregate})
    return rows


def paired_effects(records: dict[tuple[str, int], list[dict[str, str]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    fields = (
        "failure_margin",
        "failure_gradient_l2",
        "linearized_distance",
        "spectral_norm",
        "normalized_spectral_norm",
        "centered_logit_rms",
    )
    for off, on in COMPARISONS:
        repeat_rows: list[dict[str, Any]] = []
        for repeat in REPEATS:
            left = records[(off, repeat)]
            right = records[(on, repeat)]
            common = [(a, b) for a, b in zip(left, right, strict=True) if int(a["clean_correct"]) and int(b["clean_correct"])]
            row: dict[str, Any] = {
                "comparison": f"{on}-{off}",
                "repeat": repeat,
                "pairwise_common_correct": len(common),
            }
            for field in fields:
                ratio, count = geometric_ratio((float(a[field]), float(b[field])) for a, b in common)
                row[f"{field}_geometric_ratio"] = ratio
                row[f"{field}_pairs"] = count
            successful_pairs = [
                (a, b)
                for a, b in common
                if int(a["pga_success"]) and int(b["pga_success"])
            ]
            pga_ratio, pga_count = geometric_ratio(
                (float(a["pga_l2"]), float(b["pga_l2"])) for a, b in successful_pairs
            )
            row["pga_l2_complete_case_geometric_ratio"] = pga_ratio
            row["pga_l2_complete_case_pairs"] = pga_count
            row["pga_success_both"] = len(successful_pairs)
            row["pga_success_off_only"] = sum(int(a["pga_success"]) and not int(b["pga_success"]) for a, b in common)
            row["pga_success_on_only"] = sum(not int(a["pga_success"]) and int(b["pga_success"]) for a, b in common)
            row["pga_success_neither"] = sum(not int(a["pga_success"]) and not int(b["pga_success"]) for a, b in common)
            repeat_rows.append(row)
            result.append(row)
        aggregate: dict[str, Any] = {"comparison": f"{on}-{off}", "repeat": "mean_of_repeats"}
        for key in repeat_rows[0]:
            if key not in {"comparison", "repeat"}:
                aggregate[key] = mean(float(row[key]) for row in repeat_rows)
        result.append(aggregate)
    return result


def robust_curves(records: dict[tuple[str, int], list[dict[str, str]]]) -> list[dict[str, Any]]:
    budgets = list(range(-40, 1))
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for budget_db in budgets:
            budget = 10.0 ** (budget_db / 10.0)
            repeat_lower: list[float] = []
            repeat_upper: list[float] = []
            for repeat in REPEATS:
                lower = upper = 0
                data = records[(arm, repeat)]
                for row in data:
                    if not int(row["clean_correct"]):
                        continue
                    if int(row["pga_success"]):
                        robust = float(row["pga_power_per_channel_use"]) > budget
                        lower += int(robust)
                        upper += int(robust)
                    else:
                        # Right-censored: pessimistic and optimistic envelopes.
                        upper += 1
                repeat_lower.append(lower / len(data))
                repeat_upper.append(upper / len(data))
            rows.append(
                {
                    "arm": arm,
                    "scope": "all_samples",
                    "budget_db_per_channel_use": budget_db,
                    "pga_empirical_robust_accuracy_lower": mean(repeat_lower),
                    "pga_empirical_robust_accuracy_upper": mean(repeat_upper),
                }
            )
    return rows


def clean_curve(root: Path, seed: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for arm in ARMS:
        rows = read_rows(root / f"{arm.lower()}_seed{seed}" / "test_clean_curve.csv")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["snr_db"]].append(row)
        order = ["noiseless", *[str(value) for value in range(0, 21, 2)]]
        for snr in order:
            values = grouped[snr]
            result.append(
                {
                    "arm": arm,
                    "snr_db": snr,
                    "accuracy": mean(float(row["accuracy"]) for row in values),
                    "cross_entropy": mean(float(row["cross_entropy"]) for row in values),
                    "mean_margin": mean(float(row["mean_margin"]) for row in values),
                    "mean_centered_logit_rms": mean(float(row["mean_centered_logit_rms"]) for row in values),
                }
            )
    return result


def make_figures(result_root: Path, arms: list[dict[str, Any]], effects: list[dict[str, Any]], curves: list[dict[str, Any]], clean: list[dict[str, Any]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for arm in ARMS:
        values = [row for row in clean if row["arm"] == arm and row["snr_db"] != "noiseless"]
        axis.plot([float(row["snr_db"]) for row in values], [row["accuracy"] for row in values], marker="o", label=arm)
    axis.set(xlabel="Test SNR (dB)", ylabel="Clean accuracy", title="C-strong clean accuracy across channel SNR")
    axis.legend()
    fig.tight_layout()
    fig.savefig(result_root / "clean_accuracy_curve.png", dpi=180)
    plt.close(fig)

    aggregate = [row for row in effects if row["repeat"] == "mean_of_repeats"]
    labels = [row["comparison"] for row in aggregate]
    x = np.arange(len(labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(x - width, [row["normalized_spectral_norm_geometric_ratio"] for row in aggregate], width, label="normalized spectrum")
    axis.bar(x, [row["linearized_distance_geometric_ratio"] for row in aggregate], width, label="linearized distance")
    axis.bar(x + width, [row["pga_l2_complete_case_geometric_ratio"] for row in aggregate], width, label="PGA L2 (complete case)")
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(x, labels)
    axis.set(ylabel="Geometric ratio (second / first)", title="Pairwise local and attack diagnostics")
    axis.legend()
    fig.tight_layout()
    fig.savefig(result_root / "paired_effect_ratios.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for arm in ARMS:
        values = [row for row in curves if row["arm"] == arm]
        axis.plot([row["budget_db_per_channel_use"] for row in values], [row["pga_empirical_robust_accuracy_upper"] for row in values], label=arm)
    axis.set(xlabel="Attack power per channel use (dB)", ylabel="PGA-estimated robust accuracy", title="Empirical attack curves (not certified)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(result_root / "pga_empirical_robust_curves.png", dpi=180)
    plt.close(fig)


def report_text(summary: dict[str, Any]) -> str:
    arms = {row["arm"]: row for row in summary["arm_summary"]}
    effects = {row["comparison"]: row for row in summary["paired_effects_mean"]}
    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"
    def ratio(comparison: str, field: str) -> str:
        return f"{effects[comparison][field]:.3f}x"
    gate = summary["technical_gate"]
    conclusion = summary["prelocked_interpretation"]
    return f"""# C-strong 与无噪声 Jacobian 正则化实验结果

> 本文档由 `scripts/analyze_cstrong.py` 从锁定产物自动生成。seed 2026 是描述性 pilot；不做显著性检验，也不外推为普适结论。

## 1. 技术验收

- 训练质量门：`{gate['training_quality_gate_passed']}`。
- 诊断质量门：`{gate['diagnostics_quality_gate_passed']}`。
- 联合门槛：`{gate['all_passed']}`。
- 谱估计收敛率：`{pct(gate['spectral_convergence_rate'])}`。
- pairwise-common-correct 数量见 `analysis_summary.json`；该条件集是处理后筛选，只用于机制描述。

## 2. 三臂系统表现

| 模型 | 10 dB clean accuracy | clean-correct margin | normalized spectrum | PGA success rate on clean-correct |
|---|---:|---:|---:|---:|
| CS0 | {pct(arms['CS0']['clean_accuracy'])} | {arms['CS0']['clean_margin_correct']:.4f} | {arms['CS0']['normalized_spectral_correct']:.4f} | {pct(arms['CS0']['pga_success_rate_correct'])} |
| CS1 | {pct(arms['CS1']['clean_accuracy'])} | {arms['CS1']['clean_margin_correct']:.4f} | {arms['CS1']['normalized_spectral_correct']:.4f} | {pct(arms['CS1']['pga_success_rate_correct'])} |
| CSJ | {pct(arms['CSJ']['clean_accuracy'])} | {arms['CSJ']['clean_margin_correct']:.4f} | {arms['CSJ']['normalized_spectral_correct']:.4f} | {pct(arms['CSJ']['pga_success_rate_correct'])} |

## 3. 同一样本、共同正确条件下的比值

以下为 3 个共享信道重复先分别计算，再等权平均。比值是“前者相对后者”；PGA L2 只在两侧攻击均成功的样本上计算，右删失样本未删除但不进入该 complete-case 比值。

| 比较 | margin | failure gradient | raw spectrum | normalized spectrum | linearized distance | PGA L2 |
|---|---:|---:|---:|---:|---:|---:|
| CS1 / CS0 | {ratio('CS1-CS0','failure_margin_geometric_ratio')} | {ratio('CS1-CS0','failure_gradient_l2_geometric_ratio')} | {ratio('CS1-CS0','spectral_norm_geometric_ratio')} | {ratio('CS1-CS0','normalized_spectral_norm_geometric_ratio')} | {ratio('CS1-CS0','linearized_distance_geometric_ratio')} | {ratio('CS1-CS0','pga_l2_complete_case_geometric_ratio')} |
| CSJ / CS0 | {ratio('CSJ-CS0','failure_margin_geometric_ratio')} | {ratio('CSJ-CS0','failure_gradient_l2_geometric_ratio')} | {ratio('CSJ-CS0','spectral_norm_geometric_ratio')} | {ratio('CSJ-CS0','normalized_spectral_norm_geometric_ratio')} | {ratio('CSJ-CS0','linearized_distance_geometric_ratio')} | {ratio('CSJ-CS0','pga_l2_complete_case_geometric_ratio')} |
| CSJ / CS1 | {ratio('CSJ-CS1','failure_margin_geometric_ratio')} | {ratio('CSJ-CS1','failure_gradient_l2_geometric_ratio')} | {ratio('CSJ-CS1','spectral_norm_geometric_ratio')} | {ratio('CSJ-CS1','normalized_spectral_norm_geometric_ratio')} | {ratio('CSJ-CS1','linearized_distance_geometric_ratio')} | {ratio('CSJ-CS1','pga_l2_complete_case_geometric_ratio')} |

`normalized spectrum = raw spectral norm / centered-logit RMS`。它对整体 logits 缩放近似不变，用来识别“只把 logits 变小”的伪平滑。

## 4. 预锁定规则下的解释

{conclusion}

## 5. 结论边界

- PGA 曲线只是单一攻击、固定步长下的经验结果，不是认证鲁棒性。
- seed 2026 只有一个独立训练种子；只有技术门槛通过后，才按计划无条件补跑 2027/2028。
- CSJ 与 CS1 即使同向，也只能说明显式局部平滑“足以复现该现象”，不能说明 AWGN 的唯一机制。
- C-strong 仍包含 768 维通信瓶颈，因此不能单凭本实验证明通信结构完全无关。

## 6. 产物

- `results/cstrong_pilot_seed2026/analysis_summary.json`
- `results/cstrong_pilot_seed2026/arm_summary.csv`
- `results/cstrong_pilot_seed2026/paired_effects.csv`
- `results/cstrong_pilot_seed2026/robust_curves.csv`
- `results/cstrong_pilot_seed2026/clean_curve.csv`
- 三张 PNG 图
"""


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    command.add_argument("--result-root", type=Path, default=DEFAULT_RESULTS)
    command.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    command.add_argument("--seed", type=int, default=2026)
    return command


def main() -> int:
    args = parser().parse_args()
    input_root = args.input_root.resolve()
    result_root = args.result_root.resolve()
    report = args.report.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    records = validate_and_load(input_root, args.seed)
    arms = arm_summaries(records)
    effects = paired_effects(records)
    aggregate_effects = [row for row in effects if row["repeat"] == "mean_of_repeats"]
    curves = robust_curves(records)
    clean = clean_curve(input_root, args.seed)
    diagnostics_batch = json.loads((input_root / "diagnostics_batch_manifest.json").read_text(encoding="utf-8"))
    training_batch = json.loads((input_root / "batch_manifest.json").read_text(encoding="utf-8"))
    training_gate = training_batch.get("seed2026_training_quality_gate_passed")
    diagnostics_gate = diagnostics_batch.get("diagnostics_technical_gate_passed")
    effects_by_name = {row["comparison"]: row for row in aggregate_effects}
    cs1_smooth = effects_by_name["CS1-CS0"]["normalized_spectral_norm_geometric_ratio"] < 1
    csj_smooth = effects_by_name["CSJ-CS0"]["normalized_spectral_norm_geometric_ratio"] < 1
    cs1_pga = effects_by_name["CS1-CS0"]["pga_l2_complete_case_geometric_ratio"] > 1
    csj_pga = effects_by_name["CSJ-CS0"]["pga_l2_complete_case_geometric_ratio"] > 1
    csj_raw_only = (
        effects_by_name["CSJ-CS0"]["spectral_norm_geometric_ratio"] < 1
        and not csj_smooth
        and not csj_pga
    )
    if cs1_smooth and csj_smooth and cs1_pga and csj_pga:
        interpretation = "CS1 与 CSJ 都降低 normalized spectrum，并提高 PGA L2；结果符合‘显式平滑足以复现经验鲁棒现象’。这削弱语义通信特异解释，但不证明 AWGN 只有这一条机制。"
    elif csj_raw_only:
        interpretation = "CSJ 只降低 raw spectrum，而 normalized spectrum 与 PGA 没有同步改善；按预锁定规则应视为 logit 缩放伪象，不能称为真实鲁棒性提升。"
    elif cs1_smooth and cs1_pga and not (csj_smooth and csj_pga):
        interpretation = "AWGN 臂出现局部平滑与 PGA 改善，但无噪声 Jacobian 正则化没有完整复现；当前结果更支持信道分布匹配或正则化形式差异，不能把来源归结为一般 Jacobian 平滑。"
    elif not cs1_pga and not csj_pga:
        interpretation = "强模型中两种处理的 PGA 收益都消失；旧 C-small 结果可能依赖弱架构、训练质量或可用 clean margin。"
    else:
        interpretation = "三项诊断呈混合模式，不满足任何单一预锁定结论。应查看 clean 曲线、normalized spectrum 与右删失模式，而不是事后选择最有利指标。"
    gate_details = diagnostics_batch.get("technical_gate_details", {})
    summary = {
        "schema_version": "cstrong_analysis_v1",
        "seed": args.seed,
        "inference_scope": "descriptive single-seed pilot",
        "arm_summary": arms,
        "paired_effects_mean": aggregate_effects,
        "technical_gate": {
            "training_quality_gate_passed": training_gate,
            "diagnostics_quality_gate_passed": diagnostics_gate,
            "all_passed": bool(training_gate and diagnostics_gate),
            "spectral_convergence_rate": gate_details.get("spectral_convergence_rate", float("nan")),
            "pairwise_common_correct_by_repeat": gate_details.get("pairwise_common_correct_by_repeat", {}),
        },
        "prelocked_interpretation": interpretation,
        "cautions": [
            "PGA-estimated empirical robustness is not certified robustness.",
            "Pairwise-common-correct is a post-treatment conditional subset.",
            "PGA L2 ratio is complete-case when both attacks succeed; censoring patterns remain in paired_effects.csv.",
            "One training seed cannot support significance or population inference.",
        ],
    }
    write_csv(result_root / "arm_summary.csv", arms)
    write_csv(result_root / "paired_effects.csv", effects)
    write_csv(result_root / "robust_curves.csv", curves)
    write_csv(result_root / "clean_curve.csv", clean)
    make_figures(result_root, arms, effects, curves, clean)
    atomic_json(result_root / "analysis_summary.json", summary)
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(report_text(summary), encoding="utf-8")
    os.replace(temporary, report)
    print(json.dumps({"status": "completed", "result_root": str(result_root), "report": str(report), "technical_gate": summary["technical_gate"]["all_passed"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
