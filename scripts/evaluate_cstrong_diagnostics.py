"""Evaluate one C-strong arm/repeat with shared-noise local diagnostics and PGA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_robustness.attacks import ProgressiveGradientAscent  # noqa: E402
from semantic_robustness.channel import AWGNChannel  # noqa: E402
from semantic_robustness.config import load_config, without_runtime_fields  # noqa: E402
from semantic_robustness.cstrong import (  # noqa: E402
    CSTRONG_SCHEMA_VERSION,
    centered_logit_rms,
    file_sha256,
)
from semantic_robustness.data import make_loader, unpack_batch  # noqa: E402
from semantic_robustness.mechanism import (  # noqa: E402
    DEFAULT_NOISE_SEEDS,
    DEFAULT_SELECTION_SEED,
    balanced_subset,
    diagnose_independent_failure_margin,
    estimate_adaptive_spectral_norm,
    indices_sha256,
    linearized_distance_status,
    power_per_use_db,
    shared_standard_normal,
    tensor_sha256,
)
from semantic_robustness.metrics import (  # noqa: E402
    classification_failure_score,
    classification_margin,
)
from semantic_robustness.model import DeepJSCCResNetClassifier  # noqa: E402
from semantic_robustness.runtime import (  # noqa: E402
    build_dataset,
    choose_device,
    load_checkpoint,
    set_seed,
)


DIAGNOSTIC_FIELDS = (
    "schema_version",
    "arm",
    "training_seed",
    "repeat_index",
    "channel_seed",
    "snr_db",
    "sample_position",
    "dataset_index",
    "class_label",
    "channel_uses",
    "clean_prediction",
    "clean_correct",
    "clean_cross_entropy",
    "clean_logit_margin",
    "centered_logit_rms",
    "failure_score",
    "failure_margin",
    "failure_gradient_l2",
    "linearized_distance",
    "linearized_distance_status",
    "spectral_norm",
    "normalized_spectral_norm",
    "spectral_norm_at_20",
    "spectral_norm_at_30",
    "spectral_relative_change_20_30",
    "spectral_iterations",
    "spectral_converged",
    "pga_success",
    "pga_right_censored",
    "pga_steps",
    "pga_l2",
    "pga_power_per_channel_use",
    "pga_power_per_channel_use_db",
    "pga_attacked_prediction",
    "pga_attacked_margin",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(DIAGNOSTIC_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def values(tensor: Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu()]


def integers(tensor: Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu()]


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--arm", choices=("CS0", "CS1", "CSJ"), required=True)
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--repeat-index", type=int, choices=(0, 1, 2), required=True)
    command.add_argument("--noise-seed", type=int, required=True)
    command.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    command.add_argument("--max-samples", type=int, default=1000)
    command.add_argument("--snr-db", type=float, default=10.0)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--workers", type=int, default=0)
    command.add_argument("--device", default="auto")
    command.add_argument("--skip-pga", action="store_true")
    command.add_argument("--skip-spectral", action="store_true")
    command.add_argument("--resume", action="store_true")
    return command


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples <= 0 or args.max_samples % 10:
        raise ValueError("--max-samples must be positive and divisible by ten.")
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("--batch-size must be positive and --workers non-negative.")
    if args.noise_seed != DEFAULT_NOISE_SEEDS[args.repeat_index]:
        raise ValueError("repeat-index/noise-seed must use the locked shared-noise mapping.")


def evaluate(args: argparse.Namespace) -> Path:
    validate_args(args)
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output / "diagnostics.csv"
    manifest_path = output / "manifest.json"
    if diagnostics_path.exists():
        raise FileExistsError("Completed diagnostics are never overwritten.")
    config = load_config(config_path)
    if str(config.get("experiment_cell", "")).upper() != args.arm:
        raise ValueError("Configuration arm does not match --arm.")
    if int(config.get("seed", -1)) != args.seed:
        raise ValueError("Configuration seed does not match --seed.")
    if str(config.get("model", {}).get("architecture", "")) != "resnet18_bottleneck":
        raise ValueError("Diagnostics require a resnet18_bottleneck checkpoint.")
    device = choose_device(args.device)
    set_seed(args.seed, True)
    model, checkpoint = load_checkpoint(config, checkpoint_path, device)
    if not isinstance(model, DeepJSCCResNetClassifier):
        raise TypeError("Loaded checkpoint is not C-strong.")
    if checkpoint.get("config") != without_runtime_fields(config):
        raise ValueError("Checkpoint embedded configuration differs from the config file.")
    model.eval()
    dataset = build_dataset(config, "test")
    subset, selected_indices = balanced_subset(
        dataset,
        args.max_samples,
        selection_seed=args.selection_seed,
    )
    loader = make_loader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        seed=args.noise_seed,
    )
    standard_noise = shared_standard_normal(
        args.max_samples, model.channel_uses, args.noise_seed
    )
    channel = AWGNChannel(float(config.get("channel", {}).get("fading_gain", 1.0))).to(device)
    pga_config = config["attacks"]["pga"]
    attack = ProgressiveGradientAscent(
        step_size=float(pga_config["step_size"]),
        max_steps=int(pga_config["max_steps"]),
        eps=float(pga_config.get("eps", 1e-8)),
        refine_steps=int(pga_config.get("refine_steps", 0)),
    )
    signature = {
        "arm": args.arm,
        "training_seed": args.seed,
        "repeat_index": args.repeat_index,
        "channel_seed": args.noise_seed,
        "snr_db": args.snr_db,
        "max_samples": args.max_samples,
        "selection_seed": args.selection_seed,
        "config_sha256": file_sha256(config_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "selected_indices_sha256": indices_sha256(selected_indices),
        "standard_noise_sha256": tensor_sha256(standard_noise),
        "skip_pga": bool(args.skip_pga),
        "skip_spectral": bool(args.skip_spectral),
    }
    previous: dict[str, Any] | None = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError("Partial diagnostics require --resume.")
        if previous.get("status") not in {"running", "interrupted"}:
            raise RuntimeError(f"Cannot resume diagnostics status={previous.get('status')!r}.")
        if previous.get("job_signature") != signature:
            raise RuntimeError("Partial diagnostics immutable signature changed.")
    manifest: dict[str, Any] = {
        "schema_version": CSTRONG_SCHEMA_VERSION,
        "scope": "one C-strong arm/repeat local diagnostic",
        "status": "running",
        "started_at": previous.get("started_at", utc_now()) if previous else utc_now(),
        "resumed_at": utc_now() if previous else None,
        "job_signature": signature,
        "completed_shards": previous.get("completed_shards", []) if previous else [],
    }
    atomic_json(manifest_path, manifest)
    rows: list[dict[str, Any]] = []
    shards = output / "shards"
    shards.mkdir(exist_ok=True)
    recorded = {int(item["batch_index"]): item for item in manifest["completed_shards"]}
    offset = 0
    start = time.time()
    try:
        for batch_index, batch in enumerate(loader):
            images, labels = unpack_batch(batch)
            if labels is None:
                raise ValueError("C-strong diagnostics require labels.")
            count = images.shape[0]
            batch_indices = selected_indices[offset : offset + count]
            shard = shards / f"batch_{batch_index:04d}.csv"
            if shard.exists():
                if batch_index not in recorded or file_sha256(shard) != recorded[batch_index]["sha256"]:
                    raise RuntimeError(f"Unregistered or changed shard {batch_index}.")
                shard_rows = read_csv(shard)
                if len(shard_rows) != count or [int(row["dataset_index"]) for row in shard_rows] != batch_indices:
                    raise RuntimeError(f"Shard identity mismatch at batch {batch_index}.")
                rows.extend(shard_rows)
                offset += count
                continue
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            canonical = standard_noise[offset : offset + count].to(device, non_blocking=True)
            with torch.no_grad():
                latent = model.encode(images)
                variance = channel.noise_variance(args.snr_db, latent)
                received = channel(
                    latent,
                    args.snr_db,
                    noise=(canonical.reshape_as(latent) * variance.sqrt()).to(latent.dtype),
                ).float()
                logits = model.decode(received)
            score_fn = lambda point: classification_failure_score(labels, model.decode(point))
            local = diagnose_independent_failure_margin(score_fn, received)
            spectrum = None
            if not args.skip_spectral:
                seeds = [args.noise_seed * 100_000 + index for index in batch_indices]
                spectrum = estimate_adaptive_spectral_norm(model.decoder, received, seeds)
            attack_result = None
            if not args.skip_pga:
                attack_result = attack(
                    model.decoder,
                    labels,
                    received,
                    lambda expected, output_logits: classification_failure_score(
                        expected.long(), output_logits
                    ),
                    0.0,
                )
            clean_prediction = logits.argmax(dim=1)
            clean_correct = clean_prediction.eq(labels)
            clean_ce = nn.functional.cross_entropy(logits, labels, reduction="none")
            clean_margin = classification_margin(labels, logits)
            logit_rms = centered_logit_rms(logits)
            distances_status = linearized_distance_status(local.margin, local.gradient_l2)
            if spectrum is None:
                spectral_estimate = torch.full((count,), float("nan"), device=device)
                spectral20 = spectral30 = relative = spectral_estimate
                iterations = torch.zeros(count, dtype=torch.long, device=device)
                converged = torch.zeros(count, dtype=torch.bool, device=device)
            else:
                spectral_estimate = spectrum.estimate
                spectral20 = spectrum.estimate_at_20
                spectral30 = spectrum.estimate_at_30
                relative = spectrum.relative_change_20_30
                iterations = spectrum.iterations
                converged = spectrum.converged
            normalized_spectrum = spectral_estimate / logit_rms.clamp_min(1e-12)
            if attack_result is None:
                pga_success = torch.zeros(count, dtype=torch.bool, device=device)
                pga_steps = torch.zeros(count, dtype=torch.long, device=device)
                pga_l2 = pga_power = torch.full((count,), float("nan"), device=device)
                attacked_prediction = torch.full((count,), -1, dtype=torch.long, device=device)
                attacked_margin = torch.full((count,), float("nan"), device=device)
            else:
                pga_success = attack_result.success
                pga_steps = attack_result.steps
                pga_l2 = attack_result.total_power.sqrt()
                pga_power = attack_result.power_per_channel_use
                attacked_prediction = attack_result.reconstruction.argmax(dim=1)
                attacked_margin = classification_margin(labels, attack_result.reconstruction)
            columns = {
                "label": integers(labels),
                "prediction": integers(clean_prediction),
                "correct": integers(clean_correct),
                "ce": values(clean_ce),
                "margin": values(clean_margin),
                "rms": values(logit_rms),
                "failure_margin": values(local.margin),
                "gradient": values(local.gradient_l2),
                "distance": values(local.linearized_distance),
                "spectral": values(spectral_estimate),
                "normalized_spectral": values(normalized_spectrum),
                "spectral20": values(spectral20),
                "spectral30": values(spectral30),
                "relative": values(relative),
                "iterations": integers(iterations),
                "converged": integers(converged),
                "pga_success": integers(pga_success),
                "pga_steps": integers(pga_steps),
                "pga_l2": values(pga_l2),
                "pga_power": values(pga_power),
                "attacked_prediction": integers(attacked_prediction),
                "attacked_margin": values(attacked_margin),
            }
            batch_rows: list[dict[str, Any]] = []
            for local_index in range(count):
                power = columns["pga_power"][local_index]
                power_db = None if not math.isfinite(power) else power_per_use_db(power)
                row = {
                    "schema_version": CSTRONG_SCHEMA_VERSION,
                    "arm": args.arm,
                    "training_seed": args.seed,
                    "repeat_index": args.repeat_index,
                    "channel_seed": args.noise_seed,
                    "snr_db": args.snr_db,
                    "sample_position": offset + local_index,
                    "dataset_index": batch_indices[local_index],
                    "class_label": columns["label"][local_index],
                    "channel_uses": model.channel_uses,
                    "clean_prediction": columns["prediction"][local_index],
                    "clean_correct": columns["correct"][local_index],
                    "clean_cross_entropy": columns["ce"][local_index],
                    "clean_logit_margin": columns["margin"][local_index],
                    "centered_logit_rms": columns["rms"][local_index],
                    "failure_score": -columns["failure_margin"][local_index],
                    "failure_margin": columns["failure_margin"][local_index],
                    "failure_gradient_l2": columns["gradient"][local_index],
                    "linearized_distance": columns["distance"][local_index],
                    "linearized_distance_status": distances_status[local_index],
                    "spectral_norm": columns["spectral"][local_index],
                    "normalized_spectral_norm": columns["normalized_spectral"][local_index],
                    "spectral_norm_at_20": columns["spectral20"][local_index],
                    "spectral_norm_at_30": columns["spectral30"][local_index],
                    "spectral_relative_change_20_30": columns["relative"][local_index],
                    "spectral_iterations": columns["iterations"][local_index],
                    "spectral_converged": columns["converged"][local_index],
                    "pga_success": columns["pga_success"][local_index],
                    "pga_right_censored": int(
                        columns["correct"][local_index]
                        and not columns["pga_success"][local_index]
                    ),
                    "pga_steps": columns["pga_steps"][local_index],
                    "pga_l2": columns["pga_l2"][local_index],
                    "pga_power_per_channel_use": power,
                    "pga_power_per_channel_use_db": power_db,
                    "pga_attacked_prediction": columns["attacked_prediction"][local_index],
                    "pga_attacked_margin": columns["attacked_margin"][local_index],
                }
                finite_required = (
                    row["clean_cross_entropy"],
                    row["clean_logit_margin"],
                    row["centered_logit_rms"],
                    row["failure_margin"],
                    row["failure_gradient_l2"],
                )
                if not all(math.isfinite(float(value)) for value in finite_required):
                    raise FloatingPointError(f"Non-finite core diagnostic: {row}")
                batch_rows.append(row)
            atomic_csv(shard, batch_rows)
            recorded[batch_index] = {
                "batch_index": batch_index,
                "sha256": file_sha256(shard),
                "rows": count,
            }
            manifest["completed_shards"] = [recorded[index] for index in sorted(recorded)]
            manifest["completed_samples"] = offset + count
            atomic_json(manifest_path, manifest)
            rows.extend(batch_rows)
            offset += count

        if len(rows) != args.max_samples:
            raise RuntimeError(f"Expected {args.max_samples} rows, got {len(rows)}.")
        atomic_csv(diagnostics_path, rows)
        convergence_values = [int(row["spectral_converged"]) for row in rows]
        convergence_rate = None if args.skip_spectral else sum(convergence_values) / len(rows)
        manifest.update(
            status="completed",
            finished_at=utc_now(),
            rows=len(rows),
            diagnostics_sha256=file_sha256(diagnostics_path),
            clean_accuracy=sum(int(row["clean_correct"]) for row in rows) / len(rows),
            clean_correct_samples=sum(int(row["clean_correct"]) for row in rows),
            spectral_convergence_rate=convergence_rate,
            pga_success_rate=(
                None
                if args.skip_pga
                else sum(int(row["pga_success"]) for row in rows) / len(rows)
            ),
            elapsed_seconds=time.time() - start,
        )
        atomic_json(manifest_path, manifest)
        print(json.dumps({"status": "completed", "arm": args.arm, "repeat": args.repeat_index, "rows": len(rows)}, ensure_ascii=False), flush=True)
        return diagnostics_path
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            finished_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.time() - start,
        )
        atomic_json(manifest_path, manifest)
        raise


def main() -> int:
    args = parser().parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
