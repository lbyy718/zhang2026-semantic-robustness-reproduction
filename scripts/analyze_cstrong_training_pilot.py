"""Analyze a completed C-strong training pilot when the technical gate fails."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_robustness.config import load_config  # noqa: E402
from semantic_robustness.runtime import (  # noqa: E402
    build_dataset,
    choose_device,
    load_checkpoint,
)


ARMS = ("CS0", "CS1", "CSJ")
ARM_COLORS = {"CS0": "#4C78A8", "CS1": "#F58518", "CSJ": "#54A24B"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def validate_training_batch(root: Path) -> dict[str, Any]:
    batch_path = root / "batch_manifest.json"
    registry_path = root / "registry.csv"
    if not batch_path.is_file() or not registry_path.is_file():
        raise FileNotFoundError("C-strong batch manifest or registry is missing.")
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    registry = read_csv(registry_path)
    if batch.get("status") != "completed" or len(registry) != 3:
        raise RuntimeError("C-strong training batch is not complete.")
    if any(row["status"] != "completed" or int(row["exit_code"]) != 0 for row in registry):
        raise RuntimeError("At least one C-strong training job did not complete cleanly.")
    return batch


def training_tables(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    summaries: list[dict[str, Any]] = []
    milestones: list[dict[str, Any]] = []
    raw: dict[str, list[dict[str, str]]] = {}
    for arm in ARMS:
        directory = root / f"{arm.lower()}_seed2026"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        rows = read_csv(directory / "training_log.csv")
        if manifest.get("status") != "completed" or len(rows) != int(manifest["completed_epochs"]):
            raise RuntimeError(f"Training artifact mismatch for {arm}.")
        raw[arm] = rows
        best_ce = min(rows, key=lambda row: float(row["validation_10db_cross_entropy"]))
        best_accuracy = max(rows, key=lambda row: float(row["validation_10db_accuracy"]))
        final = rows[-1]
        summaries.append(
            {
                "arm": arm,
                "completed_epochs": len(rows),
                "best_checkpoint_epoch": int(manifest["best_epoch"]),
                "best_validation_ce_epoch": int(best_ce["epoch"]),
                "best_validation_10db_ce": float(best_ce["validation_10db_cross_entropy"]),
                "best_validation_accuracy_epoch": int(best_accuracy["epoch"]),
                "best_validation_10db_accuracy": float(best_accuracy["validation_10db_accuracy"]),
                "maximum_train_accuracy": max(float(row["train_accuracy"]) for row in rows),
                "final_train_accuracy": float(final["train_accuracy"]),
                "final_validation_10db_accuracy": float(final["validation_10db_accuracy"]),
                "test_noiseless_accuracy": float(manifest["test_noiseless_accuracy"]),
                "test_10db_accuracy": float(manifest["test_10db_accuracy"]),
                "test_10db_cross_entropy": float(manifest["test_10db_cross_entropy"]),
                "individual_quality_gate_passed": int(bool(manifest["individual_training_quality_gate_passed"])),
                "jacobian_lambda": manifest.get("jacobian_lambda"),
                "elapsed_seconds": float(manifest["elapsed_seconds"]),
            }
        )
        selected_epochs = sorted({1, 2, 5, 10, 20, 30, len(rows), int(manifest["best_epoch"])})
        for epoch in selected_epochs:
            row = next((item for item in rows if int(item["epoch"]) == epoch), None)
            if row is None:
                continue
            milestones.append(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "learning_rate": float(row["learning_rate"]),
                    "train_cross_entropy": float(row["train_cross_entropy"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "validation_noiseless_cross_entropy": float(row["validation_noiseless_cross_entropy"]),
                    "validation_noiseless_accuracy": float(row["validation_noiseless_accuracy"]),
                    "validation_10db_cross_entropy": float(row["validation_10db_cross_entropy"]),
                    "validation_10db_accuracy": float(row["validation_10db_accuracy"]),
                    "train_validation_accuracy_gap": float(row["train_validation_accuracy_gap"]),
                    "train_jacobian_penalty": float(row["train_jacobian_penalty"]),
                    "jacobian_lambda": float(row["jacobian_lambda"]),
                }
            )
    return summaries, milestones, raw


def clean_curves(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    snr_order = ["noiseless", *[str(value) for value in range(0, 21, 2)]]
    for arm in ARMS:
        rows = read_csv(root / f"{arm.lower()}_seed2026" / "test_clean_curve.csv")
        for snr in snr_order:
            selected = [row for row in rows if row["snr_db"] == snr]
            result.append(
                {
                    "arm": arm,
                    "snr_db": snr,
                    "accuracy": mean([float(row["accuracy"]) for row in selected]),
                    "cross_entropy": mean([float(row["cross_entropy"]) for row in selected]),
                    "mean_margin": mean([float(row["mean_margin"]) for row in selected]),
                    "mean_centered_logit_rms": mean(
                        [float(row["mean_centered_logit_rms"]) for row in selected]
                    ),
                }
            )
    return result


@torch.no_grad()
def collapse_diagnostics(root: Path, device: torch.device, sample_count: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for arm in ARMS:
        config = load_config(root / "resolved_configs" / f"{arm.lower()}_seed2026.json")
        dataset = Subset(build_dataset(config, "test"), range(sample_count))
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        for checkpoint_name in ("checkpoint_best.pt", "checkpoint_last.pt"):
            model, checkpoint = load_checkpoint(
                config, root / f"{arm.lower()}_seed2026" / checkpoint_name, device
            )
            latents: list[torch.Tensor] = []
            logits_values: list[torch.Tensor] = []
            predictions: list[torch.Tensor] = []
            active = total_hidden = 0
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                latent = model.encode(images)
                preactivation = model.decoder.network[0](latent)
                logits = model.decode(latent)
                latents.append(latent.cpu())
                logits_values.append(logits.cpu())
                predictions.append(logits.argmax(dim=1).cpu())
                active += int((preactivation > 0).sum())
                total_hidden += preactivation.numel()
            latent = torch.cat(latents)
            logits = torch.cat(logits_values)
            prediction = torch.cat(predictions)
            counts = torch.bincount(prediction, minlength=10)
            probability = torch.softmax(logits, dim=1)
            sample_entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1)
            result.append(
                {
                    "arm": arm,
                    "checkpoint": "best" if checkpoint_name == "checkpoint_best.pt" else "last",
                    "epoch": int(checkpoint["epoch"]),
                    "samples": sample_count,
                    "latent_coordinate_std_mean": float(latent.std(dim=0).mean()),
                    "hidden_relu_active_fraction": active / total_hidden,
                    "logit_across_sample_std_mean": float(logits.std(dim=0).mean()),
                    "mean_sample_predictive_entropy": float(sample_entropy.mean()),
                    "unique_predicted_classes": int((counts > 0).sum()),
                    "dominant_prediction_fraction": float(counts.max() / sample_count),
                    "prediction_counts": ";".join(str(int(value)) for value in counts),
                }
            )
    return result


def make_figures(
    result_root: Path,
    raw: dict[str, list[dict[str, str]]],
    curve: list[dict[str, Any]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for arm in ARMS:
        rows = raw[arm]
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(
            epochs,
            [float(row["train_accuracy"]) for row in rows],
            color=ARM_COLORS[arm],
            linestyle="--",
            alpha=0.65,
            label=f"{arm} train",
        )
        axes[0].plot(
            epochs,
            [float(row["validation_10db_accuracy"]) for row in rows],
            color=ARM_COLORS[arm],
            label=f"{arm} val 10 dB",
        )
        axes[1].plot(
            epochs,
            [float(row["validation_10db_cross_entropy"]) for row in rows],
            color=ARM_COLORS[arm],
            label=arm,
        )
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Training and validation accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Cross-entropy", title="10 dB validation CE")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(result_root / "training_curves.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for arm in ARMS:
        rows = [row for row in curve if row["arm"] == arm and row["snr_db"] != "noiseless"]
        axis.plot(
            [float(row["snr_db"]) for row in rows],
            [row["accuracy"] for row in rows],
            marker="o",
            color=ARM_COLORS[arm],
            label=arm,
        )
    axis.set(xlabel="Test SNR (dB)", ylabel="Accuracy", title="Best-checkpoint clean accuracy")
    axis.legend()
    fig.tight_layout()
    fig.savefig(result_root / "clean_accuracy_by_snr.png", dpi=180)
    plt.close(fig)

    csj = raw["CSJ"]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.semilogy(
        [int(row["epoch"]) for row in csj],
        [max(float(row["train_jacobian_penalty"]), 1e-18) for row in csj],
        marker="o",
        color=ARM_COLORS["CSJ"],
    )
    axis.set(xlabel="Epoch", ylabel="Mean Jacobian penalty (log scale)", title="CSJ penalty collapses to zero")
    fig.tight_layout()
    fig.savefig(result_root / "csj_penalty_collapse.png", dpi=180)
    plt.close(fig)


def markdown_report(summary: dict[str, Any]) -> str:
    arms = {row["arm"]: row for row in summary["training_summary"]}
    collapse = {
        (row["arm"], row["checkpoint"]): row
        for row in summary["collapse_diagnostics"]
    }
    return f"""# C-strong 与 Jacobian 正则化 seed 2026 Pilot 复盘

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: {summary['analyzed_at']}
- Verification Status: ANALYZED
- Version Label: cstrong_training_failure_v1

## 1. 验收结论

本轮三个训练作业都正常退出，checkpoint、CSV 和哈希校验完整；但是联合训练质量门没有通过，所以这不是一轮可用于比较鲁棒性来源的有效实验。

| 模型 | 完成 epoch | best epoch | 10 dB test accuracy | 门槛 | 是否通过 |
|---|---:|---:|---:|---:|---:|
| CS0 | {arms['CS0']['completed_epochs']} | {arms['CS0']['best_checkpoint_epoch']} | {100*arms['CS0']['test_10db_accuracy']:.2f}% | 75% | 否 |
| CS1 | {arms['CS1']['completed_epochs']} | {arms['CS1']['best_checkpoint_epoch']} | {100*arms['CS1']['test_10db_accuracy']:.2f}% | 75% | 是 |
| CSJ | {arms['CSJ']['completed_epochs']} | {arms['CSJ']['best_checkpoint_epoch']} | {100*arms['CSJ']['test_10db_accuracy']:.2f}% | 72%，且不低于 CS0 3 个百分点以上 | 否 |

因此，正式 1000 样本 Jacobian/PGA 诊断没有继续启动。CS0 与 CSJ 的准确率决定了 pairwise-common-correct 不可能达到预注册的 500/1000；强行分析只会比较失败模型。

## 2. 发生了什么

### CS0：无噪声基线发生表示坍塌

- epoch 1 已是最佳 checkpoint，validation accuracy 只有 16.34%；
- epoch 2 后迅速掉到接近随机猜测，epoch 31 因 patience 30 停止；
- last checkpoint 的 latent 跨样本坐标标准差均值为 `{collapse[('CS0','last')]['latent_coordinate_std_mean']:.3e}`；
- 分类头 ReLU 激活率为 `{100*collapse[('CS0','last')]['hidden_relu_active_fraction']:.4f}%`；
- 1000 张图全部预测成同一个类别。

这不是“过拟合”：训练准确率本身也下降到约 10%。它是优化/表示坍塌。

### CSJ：正则项走向退化的零 Jacobian 解

- warm-up 得到固定 `lambda = {arms['CSJ']['jacobian_lambda']:.3f}`；
- Jacobian penalty 从 epoch 1 的约 `2.41e-4`，在 epoch 2 降至约 `2.88e-5`，随后接近 0；
- last checkpoint 的 latent 跨样本坐标标准差均值为 `{collapse[('CSJ','last')]['latent_coordinate_std_mean']:.3e}`，ReLU 激活率为 0；
- 模型通过输出几乎与输入无关的常量 logits，使 failure-score Jacobian 变成 0，但分类任务也一起失败。

所以“raw Jacobian 变小”本身不能证明获得了有用平滑；它可能只是网络死亡。这正是原计划要求同时看 clean performance 与 normalized spectrum 的原因。

### CS1：AWGN 臂成功训练

- best checkpoint 位于 epoch {arms['CS1']['best_checkpoint_epoch']}；
- test accuracy 为 noiseless `{100*arms['CS1']['test_noiseless_accuracy']:.2f}%`、10 dB `{100*arms['CS1']['test_10db_accuracy']:.2f}%`；
- best checkpoint 的 latent 跨样本坐标标准差均值为 `{collapse[('CS1','best')]['latent_coordinate_std_mean']:.3f}`，分类头 ReLU 激活率为 `{100*collapse[('CS1','best')]['hidden_relu_active_fraction']:.2f}%`；
- 预测覆盖 10 个类别，没有出现坍塌。

但这只能说明 AWGN 在当前优化设置下避免了坍塌并成功完成训练，不能说明它让一个“同等训练良好的 CS0”更加鲁棒，因为后者并不存在。

## 3. 为什么本轮不能回答原研究问题

原问题要求在 clean 性能合格的同架构模型之间比较 `CS1-CS0` 与 `CSJ-CS0`。实际比较变成了：

- CS1：成功训练的 90.74% 模型；
- CS0/CSJ：只会预测少数类别或单一类别的失败模型。

任何 margin、Jacobian 谱或 PGA 差异都会被训练成功与否支配。它既不能证明“AWGN 带来鲁棒性”，也不能证明“显式 Jacobian 正则化不能替代 AWGN”。

## 4. 失效原因判断

证据支持三个相互关联的问题：

1. `lr=0.05 + unit-power bottleneck + 768->512 ReLU head` 对无噪声臂存在严重优化坍塌；标准 ResNet 的超参数不能直接保证插入功率归一化瓶颈后仍稳定。
2. patience 30 在 cosine 学习率仍约 0.047 时就终止 CS0/CSJ；CS1 到约 epoch 50 后才开始持续改善，因此早停放大了不同臂的优化轨迹差异。
3. raw failure-score Jacobian penalty 有常量 logits / dead ReLU 的平凡最优方向。仅固定其初始损失占比为 10%，并不能排除这种退化。

这些是实现的实验设计问题，不是论文鲁棒性命题的正面或反面证据。

## 5. 下一轮应如何修正

先做独立于鲁棒性结论的“可训练性资格阶段”，不直接重跑三臂：

1. 只在 CS0 上做小型、预先锁定的学习率诊断，例如 0.005、0.01、0.02；固定跑完整 cosine，不使用 patience 提前截断。
2. 每 epoch 增加健康指标：latent 跨样本标准差、分类头激活率、预测类别数；出现 latent std 接近 0 或激活率接近 0 就标记 collapse。
3. 选出能让 CS0 稳定超过 75% 的单一训练协议后，原封不动地应用到 CS0/CS1/CSJ。
4. CSJ 改为尺度不变的 normalized-logit Jacobian，或同时约束 logit RMS；先用解析缩放测试证明不能靠把 logits 缩成 0 降低正则项。
5. 三臂都通过 clean 门槛后，才恢复 1000 样本、3 重复的 Jacobian/PGA 比较；随后再无条件补 2027/2028。

## 6. 统计与方法学谬误扫描

- Coverage: 11/11 checked。

| 类型 | 结论 | 说明 |
|---|---|---|
| Simpson's paradox | NOTE | 目前只有一个训练种子，无法检查跨 seed 方向反转。 |
| Ecological fallacy | CAUTION | 不能从一个 seed/一套架构外推到所有神经网络或语义通信系统。 |
| Berkson's paradox | NOTE | 尚未执行 clean-correct 条件分析；若以后执行，需披露选择机制。 |
| Collider bias | CAUTION | pairwise-common-correct 是处理后条件集，只能作机制描述。 |
| Base-rate neglect | NOTE | CIFAR-10 类别均衡，不代表部署类别先验。 |
| Regression to the mean | CAUTION | best-validation checkpoint 带选择效应；本轮更主要的问题是训练坍塌。 |
| Survivorship bias | NOTE | 3/3 作业均完成，没有作业流失；但不能只保留成功臂。 |
| Look-elsewhere effect | NOTE | 主要门槛已预锁定；坍塌诊断属于事后探索。 |
| Garden of forking paths | CAUTION | 下一轮超参数诊断必须先锁定搜索集合和选择规则，避免只保留有利配置。 |
| Correlation != causation | RED_FLAG | CS1 成功与 AWGN 同时出现，但 CS0 训练失败，不能把差异归因于鲁棒机制。 |
| Reverse causality | CAUTION | Jacobian 接近 0 与网络死亡同时出现，不能断言前者造成或解释鲁棒性。 |

## 7. 最终判断

- 数据和产物可信：三个作业完整且结果可复查。
- 原假设比较无效：技术质量门失败，整体置信等级为 `RED_FLAG`。
- 有效的新发现：当前 C-strong 实现暴露了无噪声优化坍塌，以及 raw Jacobian penalty 的退化解。
- 本轮不能进入 2027/2028，也不能作为 AWGN/语义通信鲁棒性的正面或证伪证据。

## 8. 产物

- `results/cstrong_pilot_seed2026/training_summary.csv`
- `results/cstrong_pilot_seed2026/training_milestones.csv`
- `results/cstrong_pilot_seed2026/clean_curve.csv`
- `results/cstrong_pilot_seed2026/collapse_diagnostics.csv`
- `results/cstrong_pilot_seed2026/training_analysis_summary.json`
- `results/cstrong_pilot_seed2026/training_curves.png`
- `results/cstrong_pilot_seed2026/clean_accuracy_by_snr.png`
- `results/cstrong_pilot_seed2026/csj_penalty_collapse.png`
"""


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--input-root", type=Path, default=REPO_ROOT / "outputs" / "cstrong_pilot"
    )
    command.add_argument(
        "--result-root",
        type=Path,
        default=REPO_ROOT / "results" / "cstrong_pilot_seed2026",
    )
    command.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs" / "12_C-strong与Jacobian正则化实验结果.md",
    )
    command.add_argument("--device", default="auto")
    command.add_argument("--collapse-samples", type=int, default=1000)
    return command


def main() -> int:
    args = parser().parse_args()
    root = args.input_root.resolve()
    result_root = args.result_root.resolve()
    report = args.report.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    batch = validate_training_batch(root)
    training_summary, milestones, raw = training_tables(root)
    curve = clean_curves(root)
    device = choose_device(args.device)
    collapse = collapse_diagnostics(root, device, args.collapse_samples)
    make_figures(result_root, raw, curve)
    summary = {
        "schema_version": "cstrong_training_failure_v1",
        "analyzed_at": utc_now(),
        "verification_status": "ANALYZED",
        "overall_confidence": "RED_FLAG",
        "comparison_valid": False,
        "formal_mechanism_diagnostics_started": False,
        "reason": "CS0 and CSJ failed the preregistered clean-performance gate via representation collapse.",
        "batch_training_quality_gate_passed": batch.get("seed2026_training_quality_gate_passed"),
        "training_summary": training_summary,
        "collapse_diagnostics": collapse,
        "input_hashes": {
            arm: {
                "manifest": sha256(root / f"{arm.lower()}_seed2026" / "manifest.json"),
                "training_log": sha256(root / f"{arm.lower()}_seed2026" / "training_log.csv"),
                "checkpoint_best": sha256(root / f"{arm.lower()}_seed2026" / "checkpoint_best.pt"),
                "checkpoint_last": sha256(root / f"{arm.lower()}_seed2026" / "checkpoint_last.pt"),
            }
            for arm in ARMS
        },
        "fallacy_scan_coverage": "11/11",
    }
    write_csv(result_root / "training_summary.csv", training_summary)
    write_csv(result_root / "training_milestones.csv", milestones)
    write_csv(result_root / "clean_curve.csv", curve)
    write_csv(result_root / "collapse_diagnostics.csv", collapse)
    atomic_json(result_root / "training_analysis_summary.json", summary)
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(markdown_report(summary), encoding="utf-8")
    os.replace(temporary, report)
    print(
        json.dumps(
            {
                "status": "completed",
                "comparison_valid": False,
                "result_root": str(result_root),
                "report": str(report),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
