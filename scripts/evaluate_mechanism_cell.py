"""Evaluate one cell/seed/channel-repeat mechanism-diagnostic job.

The job writes exactly one balanced CIFAR-10 row per selected test image.  Its
manifest records the selected-index and shared-standard-noise hashes so results
from different factorial cells can be paired and audited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
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
from semantic_robustness.data import make_loader, unpack_batch  # noqa: E402
from semantic_robustness.mechanism import (  # noqa: E402
    DEFAULT_RECONSTRUCTION_FAILURE_PSNR_DB,
    DEFAULT_SELECTION_SEED,
    DEFAULT_SNR_DB,
    MECHANISM_SCHEMA_VERSION,
    MechanismJob,
    ReconstructionSemanticEndpoint,
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
    mse_per_sample,
    psnr,
    target_distortion,
)
from semantic_robustness.runtime import (  # noqa: E402
    build_dataset,
    choose_device,
    load_checkpoint,
    set_seed,
)
from semantic_robustness.semantic_evaluator import (  # noqa: E402
    load_frozen_cifar10_resnet18,
)
from semantic_robustness.theory import LocalLipschitzDiagnostics  # noqa: E402


DIAGNOSTIC_FIELDS = (
    "schema_version",
    "cell",
    "training_seed",
    "repeat_index",
    "channel_seed",
    "snr_db",
    "objective",
    "sample_position",
    "dataset_index",
    "class_label",
    "channel_uses",
    "clean_native_prediction",
    "clean_native_correct",
    "clean_native_cross_entropy",
    "clean_native_logit_margin",
    "clean_native_mse",
    "clean_native_psnr_db",
    "clean_semantic_prediction",
    "clean_semantic_correct",
    "clean_semantic_cross_entropy",
    "clean_semantic_logit_margin",
    "native_failure_score",
    "native_failure_margin",
    "native_margin_gradient_l2",
    "native_linearized_distance_l2",
    "native_linearized_distance_status",
    "semantic_failure_score",
    "semantic_failure_margin",
    "semantic_margin_gradient_l2",
    "semantic_linearized_distance_l2",
    "semantic_linearized_distance_status",
    "semantic_spectral_norm",
    "semantic_spectral_estimate_at_20",
    "semantic_spectral_estimate_at_30",
    "semantic_spectral_relative_change_20_30",
    "semantic_spectral_iterations",
    "semantic_spectral_converged_20_30",
    "native_decoder_spectral_norm",
    "native_decoder_spectral_estimate_at_20",
    "native_decoder_spectral_estimate_at_30",
    "native_decoder_spectral_relative_change_20_30",
    "native_decoder_spectral_iterations",
    "native_decoder_spectral_converged_20_30",
    "pga_objective_variant",
    "pga_success",
    "pga_steps",
    "pga_attacked_prediction",
    "pga_attacked_logit_margin",
    "pga_failure_score",
    "pga_attack_power_total_l2_sq",
    "pga_attack_power_per_channel_use",
    "pga_attack_power_per_channel_use_db",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty diagnostic CSV.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_source_snapshot(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify the immutable code snapshot and every source file it binds."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != expected_sha256:
        raise RuntimeError("Source snapshot JSON hash changed during the batch.")
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    sources = snapshot.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Source snapshot has no source list.")
    for source in sources:
        source_path = Path(source["path"]).resolve()
        if not source_path.is_file() or sha256(source_path) != source.get("sha256"):
            raise RuntimeError(f"Frozen source changed or disappeared: {source_path}")
    return snapshot


def _values(tensor: Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu()]


def _integers(tensor: Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu()]


def _booleans(tensor: Tensor) -> list[int]:
    return [int(bool(value)) for value in tensor.detach().cpu()]


def _empty_spectral(count: int) -> dict[str, list[Any]]:
    return {
        "estimate": [None] * count,
        "estimate_at_20": [None] * count,
        "estimate_at_30": [None] * count,
        "relative_change": [None] * count,
        "iterations": [None] * count,
        "converged": [None] * count,
    }


def _spectral_columns(
    diagnostics: LocalLipschitzDiagnostics | None, count: int
) -> dict[str, list[Any]]:
    if diagnostics is None:
        return _empty_spectral(count)
    return {
        "estimate": _values(diagnostics.estimate),
        "estimate_at_20": _values(diagnostics.estimate_at_20),
        "estimate_at_30": _values(diagnostics.estimate_at_30),
        "relative_change": _values(diagnostics.relative_change_20_30),
        "iterations": _integers(diagnostics.iterations),
        "converged": _booleans(diagnostics.converged),
    }


def _validate_job_config(
    config: dict[str, Any],
    job: MechanismJob,
    config_path: Path,
    checkpoint_path: Path,
) -> None:
    objective = str(config.get("objective", "reconstruction")).lower()
    if objective != job.objective:
        raise ValueError(
            f"Cell {job.cell.upper()} requires objective {job.objective!r}, "
            f"found {objective!r}."
        )
    config_seed = int(config.get("seed", job.training_seed))
    if config_seed != job.training_seed:
        raise ValueError(
            f"Configuration seed {config_seed} does not match job seed "
            f"{job.training_seed}."
        )
    configured_cell = str(config.get("experiment_cell", "")).lower()
    if configured_cell != job.cell:
        raise ValueError(
            f"Configuration cell {configured_cell!r} does not match {job.cell!r}."
        )
    expected_noise = job.cell.endswith("1")
    configured_noise = bool(config.get("training", {}).get("channel_noise", True))
    if configured_noise != expected_noise:
        raise ValueError(
            f"Cell {job.cell.upper()} requires training.channel_noise={expected_noise}, "
            f"found {configured_noise}."
        )
    expected_directory = f"{job.cell}_seed{job.training_seed}"
    if config_path.name != "resolved_config.json" or config_path.parent.name != expected_directory:
        raise ValueError("Configuration path does not follow the factorial job directory rule.")
    if checkpoint_path.name != "checkpoint_best.pt" or checkpoint_path.parent != config_path.parent:
        raise ValueError("Checkpoint path does not match the factorial configuration directory.")


def _validate_checkpoint_metadata(
    checkpoint: dict[str, Any], config: dict[str, Any], job: MechanismJob
) -> None:
    if str(checkpoint.get("objective", "")).lower() != job.objective:
        raise ValueError("Checkpoint objective does not match the requested job.")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint has no embedded configuration.")
    if checkpoint_config != without_runtime_fields(config):
        raise ValueError("Checkpoint embedded configuration differs from resolved_config.json.")


def _spectral_sample_seeds(
    channel_seed: int, dataset_indices: list[int], *, native: bool = False
) -> list[int]:
    offset = 50_000 if native else 0
    return [channel_seed * 100_000 + offset + index for index in dataset_indices]


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--evaluator-checkpoint", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--cell", choices=("r0", "r1", "c0", "c1"), required=True)
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--repeat-index", type=int, required=True)
    command.add_argument("--noise-seed", type=int, required=True)
    command.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    command.add_argument("--max-samples", type=int, default=1000)
    command.add_argument("--snr-db", type=float, default=DEFAULT_SNR_DB)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--workers", type=int, default=0)
    command.add_argument("--device", default="auto")
    command.add_argument("--source-snapshot", type=Path, required=True)
    command.add_argument("--source-snapshot-sha256", required=True)
    command.add_argument("--skip-pga", action="store_true")
    command.add_argument("--skip-spectral", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    return command


def validate_arguments(args: argparse.Namespace) -> None:
    if args.max_samples <= 0 or args.max_samples > 10_000:
        raise ValueError("--max-samples must be in [1, 10000].")
    if args.max_samples % 10:
        raise ValueError("--max-samples must be divisible by 10 for class balance.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative.")
    if args.repeat_index < 0:
        raise ValueError("--repeat-index cannot be negative.")


def evaluate(args: argparse.Namespace) -> Path:
    validate_arguments(args)
    job = MechanismJob(args.cell, args.seed, args.repeat_index, args.noise_seed)
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    evaluator_path = args.evaluator_checkpoint.resolve()
    snapshot_path = args.source_snapshot.resolve()
    output = args.output.resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "job": job.identifier,
                    "samples": args.max_samples,
                    "snr_db": args.snr_db,
                    "selection_seed": args.selection_seed,
                    "skip_pga": args.skip_pga,
                    "skip_spectral": args.skip_spectral,
                    "source_snapshot": str(snapshot_path),
                    "source_snapshot_sha256": args.source_snapshot_sha256,
                    "output": str(output),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return output / "diagnostics.csv"
    for source in (config_path, checkpoint_path, evaluator_path, snapshot_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    verify_source_snapshot(snapshot_path, args.source_snapshot_sha256)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output / "diagnostics.csv"
    manifest_path = output / "manifest.json"
    recover_existing_final = diagnostics_path.exists()

    started_at = utc_now()
    start_time = time.time()
    job_signature = {
        "job": job.identifier,
        "requested_samples": args.max_samples,
        "snr_db": args.snr_db,
        "selection_seed": args.selection_seed,
        "skip_pga": bool(args.skip_pga),
        "skip_spectral": bool(args.skip_spectral),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "evaluator_checkpoint_sha256": sha256(evaluator_path),
        "source_snapshot_sha256": args.source_snapshot_sha256,
    }
    previous_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("job_signature") != job_signature:
            raise RuntimeError("Existing partial job has a different immutable signature.")
        if previous_manifest.get("status") == "completed":
            if recover_existing_final:
                raise FileExistsError(
                    "Completed diagnostics already exist and are never overwritten."
                )
            raise RuntimeError("Completed manifest exists without final diagnostics CSV.")
    elif recover_existing_final:
        raise RuntimeError("Final diagnostics exist without a recoverable manifest.")
    manifest: dict[str, Any] = {
        "schema_version": MECHANISM_SCHEMA_VERSION,
        "scope": "one factorial cell/seed/repeat mechanism diagnostic",
        "status": "running",
        "started_at": (
            previous_manifest.get("started_at", started_at)
            if previous_manifest is not None
            else started_at
        ),
        "resumed_at": started_at if previous_manifest is not None else None,
        "job": {
            "identifier": job.identifier,
            "cell": job.cell.upper(),
            "training_seed": job.training_seed,
            "repeat_index": job.repeat_index,
            "channel_seed": job.channel_seed,
        },
        "requested_samples": args.max_samples,
        "snr_db": args.snr_db,
        "selection_seed": args.selection_seed,
        "skip_pga": bool(args.skip_pga),
        "skip_spectral": bool(args.skip_spectral),
        "job_signature": job_signature,
        "source_snapshot": {
            "path": str(snapshot_path),
            "sha256": args.source_snapshot_sha256,
        },
        "completed_shards": (
            previous_manifest.get("completed_shards", [])
            if previous_manifest is not None
            else []
        ),
        "command": sys.argv,
    }
    write_json(manifest_path, manifest)

    try:
        config = load_config(config_path)
        _validate_job_config(config, job, config_path, checkpoint_path)
        set_seed(job.training_seed, bool(config.get("deterministic", True)))
        device = choose_device(args.device)
        model, model_metadata = load_checkpoint(config, checkpoint_path, device)
        _validate_checkpoint_metadata(model_metadata, config, job)
        evaluator, evaluator_metadata = load_frozen_cifar10_resnet18(
            evaluator_path, device=device, require_qualified=True
        )
        objective = job.objective
        semantic_endpoint: nn.Module
        if objective == "reconstruction":
            semantic_endpoint = ReconstructionSemanticEndpoint(
                model.decoder, evaluator
            ).to(device)
        else:
            semantic_endpoint = model.decoder
        semantic_endpoint.eval()

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
            seed=job.channel_seed,
        )
        channel_uses = int(model.channel_uses)
        standard_noise = shared_standard_normal(
            args.max_samples, channel_uses, job.channel_seed
        )
        channel = AWGNChannel(
            float(config.get("channel", {}).get("fading_gain", 1.0))
        ).to(device)
        reconstruction_threshold = target_distortion(
            "image", DEFAULT_RECONSTRUCTION_FAILURE_PSNR_DB
        )
        attack = ProgressiveGradientAscent(
            step_size=0.1, max_steps=2000, eps=1e-8, refine_steps=0
        )

        rows: list[dict[str, Any]] = []
        shards_directory = output / "shards"
        shards_directory.mkdir(exist_ok=True)
        recorded_shards = {
            int(item["batch_index"]): item
            for item in manifest.get("completed_shards", [])
        }
        offset = 0
        for batch_index, batch in enumerate(loader):
            images, labels = unpack_batch(batch)
            if labels is None:
                raise ValueError("CIFAR-10 mechanism evaluation requires labels.")
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            count = images.shape[0]
            batch_indices = selected_indices[offset : offset + count]
            shard_path = shards_directory / f"batch_{batch_index:04d}.csv"
            recorded_shard = recorded_shards.get(batch_index)
            if shard_path.exists() or recorded_shard is not None:
                if not shard_path.is_file():
                    raise RuntimeError(
                        f"Partial shard state is inconsistent for batch {batch_index}."
                    )
                if (
                    recorded_shard is not None
                    and sha256(shard_path) != recorded_shard.get("sha256")
                ):
                    raise RuntimeError(f"Shard hash mismatch for batch {batch_index}.")
                shard_rows = read_csv(shard_path)
                positions = [int(row["sample_position"]) for row in shard_rows]
                indices = [int(row["dataset_index"]) for row in shard_rows]
                if (
                    len(shard_rows) != count
                    or positions != list(range(offset, offset + count))
                    or indices != batch_indices
                    or list(shard_rows[0]) != list(DIAGNOSTIC_FIELDS)
                ):
                    raise RuntimeError(f"Shard identity mismatch for batch {batch_index}.")
                if recorded_shard is None:
                    recorded_shard = {
                        "batch_index": batch_index,
                        "path": str(shard_path),
                        "sha256": sha256(shard_path),
                        "rows": len(shard_rows),
                        "first_sample_position": offset,
                        "last_sample_position": offset + count - 1,
                        "dataset_indices_sha256": indices_sha256(batch_indices),
                        "recovered_orphan": True,
                    }
                    recorded_shards[batch_index] = recorded_shard
                    manifest["completed_shards"] = [
                        recorded_shards[index] for index in sorted(recorded_shards)
                    ]
                    manifest["completed_samples"] = offset + count
                    write_json(manifest_path, manifest)
                rows.extend(shard_rows)
                offset += count
                print(
                    json.dumps(
                        {
                            "job": job.identifier,
                            "batch": batch_index + 1,
                            "completed_samples": offset,
                            "total_samples": args.max_samples,
                            "resumed_shard": True,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            canonical_noise = standard_noise[offset : offset + count].to(
                device=device, non_blocking=True
            )
            with torch.no_grad():
                symbols = model.encode(images)
                variance = channel.noise_variance(args.snr_db, symbols)
                actual_noise = canonical_noise.reshape_as(symbols) * variance.sqrt()
                received = channel(symbols, args.snr_db, noise=actual_noise)
                native_output = model.decode(received)
                semantic_logits = (
                    evaluator(native_output)
                    if objective == "reconstruction"
                    else native_output
                )

            if objective == "reconstruction":
                native_score_fn = lambda point: (
                    mse_per_sample(images, model.decoder(point))
                    - reconstruction_threshold
                )
                native_mse = mse_per_sample(images, native_output)
                native_psnr = psnr(images, native_output)
                native_prediction = [None] * count
                native_correct = [None] * count
                native_cross_entropy = [None] * count
                native_logit_margin = [None] * count
            else:
                native_score_fn = lambda point: classification_failure_score(
                    labels, model.decoder(point)
                )
                native_mse = None
                native_psnr = None
                native_prediction = _integers(native_output.argmax(dim=1))
                native_correct = _booleans(native_output.argmax(dim=1).eq(labels))
                native_cross_entropy = _values(
                    nn.functional.cross_entropy(
                        native_output, labels, reduction="none"
                    )
                )
                native_logit_margin = _values(
                    classification_margin(labels, native_output)
                )

            semantic_score_fn = lambda point: classification_failure_score(
                labels, semantic_endpoint(point)
            )
            native_diagnostics = diagnose_independent_failure_margin(
                native_score_fn, received
            )
            if objective == "classification":
                semantic_diagnostics = native_diagnostics
            else:
                semantic_diagnostics = diagnose_independent_failure_margin(
                    semantic_score_fn, received
                )

            semantic_spectral: LocalLipschitzDiagnostics | None = None
            native_spectral: LocalLipschitzDiagnostics | None = None
            if not args.skip_spectral:
                semantic_spectral = estimate_adaptive_spectral_norm(
                    semantic_endpoint,
                    received,
                    _spectral_sample_seeds(job.channel_seed, batch_indices),
                )
                if objective == "reconstruction":
                    native_spectral = estimate_adaptive_spectral_norm(
                        model.decoder,
                        received,
                        _spectral_sample_seeds(
                            job.channel_seed, batch_indices, native=True
                        ),
                    )
            semantic_spectral_values = _spectral_columns(semantic_spectral, count)
            native_spectral_values = _spectral_columns(native_spectral, count)

            attack_result = None
            attacked_predictions: list[Any] = [None] * count
            attacked_margins: list[Any] = [None] * count
            attacked_power_db: list[Any] = [None] * count
            if not args.skip_pga:
                attack_result = attack(
                    semantic_endpoint,
                    labels,
                    received,
                    lambda expected, logits: classification_failure_score(
                        expected.long(), logits
                    ),
                    0.0,
                )
                attacked_predictions = _integers(
                    attack_result.reconstruction.argmax(dim=1)
                )
                attacked_margins = _values(
                    classification_margin(labels, attack_result.reconstruction)
                )
                attacked_power_db = [
                    power_per_use_db(value)
                    for value in _values(attack_result.power_per_channel_use)
                ]

            semantic_predictions = _integers(semantic_logits.argmax(dim=1))
            semantic_correct = _booleans(semantic_logits.argmax(dim=1).eq(labels))
            semantic_cross_entropy = _values(
                nn.functional.cross_entropy(
                    semantic_logits, labels, reduction="none"
                )
            )
            semantic_margins = _values(classification_margin(labels, semantic_logits))
            native_scores = _values(-native_diagnostics.margin)
            semantic_scores = _values(-semantic_diagnostics.margin)
            native_margins = _values(native_diagnostics.margin)
            semantic_failure_margins = _values(semantic_diagnostics.margin)
            native_gradient = _values(native_diagnostics.gradient_l2)
            semantic_gradient = _values(semantic_diagnostics.gradient_l2)
            native_distance = _values(native_diagnostics.linearized_distance)
            semantic_distance = _values(semantic_diagnostics.linearized_distance)
            native_distance_status = linearized_distance_status(
                native_diagnostics.margin, native_diagnostics.gradient_l2
            )
            semantic_distance_status = linearized_distance_status(
                semantic_diagnostics.margin, semantic_diagnostics.gradient_l2
            )
            labels_cpu = _integers(labels)

            batch_rows: list[dict[str, Any]] = []
            for local_index in range(count):
                row: dict[str, Any] = {
                    "schema_version": MECHANISM_SCHEMA_VERSION,
                    "cell": job.cell.upper(),
                    "training_seed": job.training_seed,
                    "repeat_index": job.repeat_index,
                    "channel_seed": job.channel_seed,
                    "snr_db": args.snr_db,
                    "objective": objective,
                    "sample_position": offset + local_index,
                    "dataset_index": batch_indices[local_index],
                    "class_label": labels_cpu[local_index],
                    "channel_uses": channel_uses,
                    "clean_native_prediction": native_prediction[local_index],
                    "clean_native_correct": native_correct[local_index],
                    "clean_native_cross_entropy": native_cross_entropy[local_index],
                    "clean_native_logit_margin": native_logit_margin[local_index],
                    "clean_native_mse": (
                        None
                        if native_mse is None
                        else float(native_mse[local_index])
                    ),
                    "clean_native_psnr_db": (
                        None
                        if native_psnr is None
                        else float(native_psnr[local_index])
                    ),
                    "clean_semantic_prediction": semantic_predictions[local_index],
                    "clean_semantic_correct": semantic_correct[local_index],
                    "clean_semantic_cross_entropy": semantic_cross_entropy[local_index],
                    "clean_semantic_logit_margin": semantic_margins[local_index],
                    "native_failure_score": native_scores[local_index],
                    "native_failure_margin": native_margins[local_index],
                    "native_margin_gradient_l2": native_gradient[local_index],
                    "native_linearized_distance_l2": native_distance[local_index],
                    "native_linearized_distance_status": native_distance_status[local_index],
                    "semantic_failure_score": semantic_scores[local_index],
                    "semantic_failure_margin": semantic_failure_margins[local_index],
                    "semantic_margin_gradient_l2": semantic_gradient[local_index],
                    "semantic_linearized_distance_l2": semantic_distance[local_index],
                    "semantic_linearized_distance_status": semantic_distance_status[local_index],
                    "semantic_spectral_norm": semantic_spectral_values["estimate"][local_index],
                    "semantic_spectral_estimate_at_20": semantic_spectral_values["estimate_at_20"][local_index],
                    "semantic_spectral_estimate_at_30": semantic_spectral_values["estimate_at_30"][local_index],
                    "semantic_spectral_relative_change_20_30": semantic_spectral_values["relative_change"][local_index],
                    "semantic_spectral_iterations": semantic_spectral_values["iterations"][local_index],
                    "semantic_spectral_converged_20_30": semantic_spectral_values["converged"][local_index],
                    "native_decoder_spectral_norm": native_spectral_values["estimate"][local_index],
                    "native_decoder_spectral_estimate_at_20": native_spectral_values["estimate_at_20"][local_index],
                    "native_decoder_spectral_estimate_at_30": native_spectral_values["estimate_at_30"][local_index],
                    "native_decoder_spectral_relative_change_20_30": native_spectral_values["relative_change"][local_index],
                    "native_decoder_spectral_iterations": native_spectral_values["iterations"][local_index],
                    "native_decoder_spectral_converged_20_30": native_spectral_values["converged"][local_index],
                    "pga_objective_variant": (
                        None if attack_result is None else attack_result.objective_variant
                    ),
                    "pga_success": (
                        None
                        if attack_result is None
                        else int(bool(attack_result.success[local_index]))
                    ),
                    "pga_steps": (
                        None
                        if attack_result is None
                        else int(attack_result.steps[local_index])
                    ),
                    "pga_attacked_prediction": attacked_predictions[local_index],
                    "pga_attacked_logit_margin": attacked_margins[local_index],
                    "pga_failure_score": (
                        None
                        if attack_result is None
                        else float(attack_result.distortion[local_index])
                    ),
                    "pga_attack_power_total_l2_sq": (
                        None
                        if attack_result is None
                        else float(attack_result.total_power[local_index])
                    ),
                    "pga_attack_power_per_channel_use": (
                        None
                        if attack_result is None
                        else float(attack_result.power_per_channel_use[local_index])
                    ),
                    "pga_attack_power_per_channel_use_db": attacked_power_db[local_index],
                }
                batch_rows.append(row)
            write_csv(shard_path, batch_rows)
            shard_record = {
                "batch_index": batch_index,
                "path": str(shard_path),
                "sha256": sha256(shard_path),
                "rows": len(batch_rows),
                "first_sample_position": offset,
                "last_sample_position": offset + count - 1,
                "dataset_indices_sha256": indices_sha256(batch_indices),
            }
            recorded_shards[batch_index] = shard_record
            manifest["completed_shards"] = [
                recorded_shards[index] for index in sorted(recorded_shards)
            ]
            manifest["completed_samples"] = offset + count
            write_json(manifest_path, manifest)
            rows.extend(batch_rows)
            offset += count
            print(
                json.dumps(
                    {
                        "job": job.identifier,
                        "batch": batch_index + 1,
                        "completed_samples": offset,
                        "total_samples": args.max_samples,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if offset != args.max_samples or len(rows) != args.max_samples:
            raise RuntimeError(
                f"Expected {args.max_samples} diagnostics, produced {len(rows)}."
            )
        observed_indices = [int(row["dataset_index"]) for row in rows]
        if observed_indices != selected_indices:
            raise RuntimeError("DataLoader order did not match selected test indices.")
        class_counts = Counter(int(row["class_label"]) for row in rows)
        if set(class_counts.values()) != {args.max_samples // 10}:
            raise RuntimeError(f"Balanced sample invariant failed: {class_counts}.")

        verify_source_snapshot(snapshot_path, args.source_snapshot_sha256)
        diagnostics_path = output / "diagnostics.csv"
        if recover_existing_final:
            existing_rows = read_csv(diagnostics_path)
            canonical_rows = [
                {
                    field: "" if row.get(field) is None else str(row.get(field))
                    for field in DIAGNOSTIC_FIELDS
                }
                for row in rows
            ]
            if existing_rows != canonical_rows:
                raise RuntimeError(
                    "Existing final CSV differs from the validated batch shards."
                )
        else:
            write_csv(diagnostics_path, rows)
        manifest.update(
            status="completed",
            finished_at=utc_now(),
            elapsed_seconds=time.time() - start_time,
            device=str(device),
            rows=len(rows),
            class_counts={str(key): class_counts[key] for key in sorted(class_counts)},
            selection={
                "method": "fixed pseudorandom sampling independently within class",
                "seed": args.selection_seed,
                "sha256": indices_sha256(selected_indices),
            },
            selected_indices_sha256=indices_sha256(selected_indices),
            standard_noise={
                "distribution": "torch CPU float32 standard normal",
                "shape": list(standard_noise.shape),
                "seed": job.channel_seed,
                "sha256": tensor_sha256(standard_noise),
                "scaling": "sqrt(|h|^2 / 10^(snr_db/10))",
            },
            channel={
                "model": "real-valued AWGN",
                "fading_gain": channel.fading_gain,
                "snr_db": args.snr_db,
                "noise_variance": channel.fading_gain**2
                / (10.0 ** (args.snr_db / 10.0)),
                "channel_uses": channel_uses,
            },
            source={
                "config": str(config_path),
                "config_sha256": sha256(config_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "checkpoint_epoch": model_metadata.get("epoch"),
                "evaluator_checkpoint": str(evaluator_path),
                "evaluator_checkpoint_sha256": sha256(evaluator_path),
                "evaluator_test_accuracy": evaluator_metadata.get("test_accuracy"),
                "evaluator_qualified": bool(evaluator_metadata.get("qualified", False)),
                "source_snapshot": str(snapshot_path),
                "source_snapshot_sha256": args.source_snapshot_sha256,
            },
            diagnostics={
                "clean_margin_policy": "retained in system result; also decomposed locally",
                "native_reconstruction_failure_psnr_db": (
                    DEFAULT_RECONSTRUCTION_FAILURE_PSNR_DB
                    if objective == "reconstruction"
                    else None
                ),
                "semantic_failure_rule": "max_other_logit - true_logit >= 0",
                "pga": {
                    "skipped": bool(args.skip_pga),
                    "step_size": 0.1,
                    "max_steps": 2000,
                    "refine_steps": 0,
                    "random_restarts": 0,
                },
                "implicit_spectral_norm": {
                    "skipped": bool(args.skip_spectral),
                    "method": "JVP/VJP power iteration; no full Jacobian",
                    "checkpoints": [20, 30],
                    "extend_to": 60,
                    "relative_tolerance": 0.05,
                    "converged_field": "20-to-30 relative change <= 0.05",
                },
            },
            software={
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
            diagnostics_csv=str(diagnostics_path),
            diagnostics_csv_sha256=sha256(diagnostics_path),
        )
        write_json(manifest_path, manifest)
        return diagnostics_path
    except Exception as exc:
        manifest.update(
            status="failed",
            finished_at=utc_now(),
            elapsed_seconds=time.time() - start_time,
            error=f"{type(exc).__name__}: {exc}",
        )
        write_json(manifest_path, manifest)
        raise


def main() -> int:
    args = parser().parse_args()
    path = evaluate(args)
    print(f"diagnostics: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
