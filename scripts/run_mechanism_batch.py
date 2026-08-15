"""Run the 36 factorial mechanism jobs serially and without auto-retry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_robustness.mechanism import (  # noqa: E402
    DEFAULT_NOISE_SEEDS,
    DEFAULT_SELECTION_SEED,
    DEFAULT_SNR_DB,
    MECHANISM_SCHEMA_VERSION,
    MechanismJob,
)


CELLS = ("r0", "r1", "c0", "c1")
SEEDS = (2026, 2027, 2028)
DEFAULT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "factorial"
    / "factorial_mechanism_v1"
    / "diagnostics"
)
DEFAULT_EVALUATOR = (
    REPO_ROOT
    / "outputs"
    / "factorial"
    / "factorial_mechanism_v1"
    / "frozen_resnet18"
    / "checkpoint_best.pt"
)
REGISTRY_FIELDS = (
    "cell",
    "training_seed",
    "repeat_index",
    "channel_seed",
    "status",
    "started_at",
    "finished_at",
    "exit_code",
    "output_dir",
    "note",
)
SOURCE_SNAPSHOT_RELATIVE_PATHS = (
    "scripts/evaluate_mechanism_cell.py",
    "scripts/run_mechanism_batch.py",
    "semantic_robustness/mechanism.py",
    "semantic_robustness/theory.py",
    "semantic_robustness/attacks.py",
    "semantic_robustness/metrics.py",
    "semantic_robustness/channel.py",
    "semantic_robustness/runtime.py",
    "semantic_robustness/model.py",
    "semantic_robustness/semantic_evaluator.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_source_snapshot() -> dict[str, Any]:
    sources: list[dict[str, str]] = []
    for relative in SOURCE_SNAPSHOT_RELATIVE_PATHS:
        path = (REPO_ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(
            {"relative_path": relative, "path": str(path), "sha256": sha256(path)}
        )
    return {
        "schema_version": MECHANISM_SCHEMA_VERSION,
        "kind": "immutable mechanism source snapshot",
        "created_at": utc_now(),
        "sources": sources,
    }


def verify_source_snapshot(snapshot: dict[str, Any]) -> None:
    for source in snapshot.get("sources", []):
        path = Path(source["path"])
        if not path.is_file() or sha256(path) != source.get("sha256"):
            raise RuntimeError(f"Frozen source changed or disappeared: {path}")


def pid_is_alive(pid: int) -> bool:
    if platform.system() == "Windows":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path: Path, signature: str) -> tuple[bool, dict[str, Any] | None]:
    """Acquire an atomic PID lock, removing only a demonstrably stale lock."""
    stale: dict[str, Any] | None = None
    if path.exists():
        try:
            stale = json.loads(path.read_text(encoding="utf-8"))
            owner_pid = int(stale["pid"])
            alive = pid_is_alive(owner_pid)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot prove existing batch lock is stale: {path}") from exc
        if alive:
            raise RuntimeError(f"Another mechanism runner is active with PID {owner_pid}.")
        path.unlink()
    payload = {"pid": os.getpid(), "created_at": utc_now(), "batch_signature": signature}
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
    except FileExistsError as exc:
        raise RuntimeError("Mechanism batch lock was acquired concurrently.") from exc
    return True, stale


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _indices_hash(indices: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(array("I", indices).tobytes())
    return digest.hexdigest()


def diagnostic_complete(
    output: Path,
    job: MechanismJob,
    *,
    expected_samples: int,
    snr_db: float,
    skip_pga: bool,
    skip_spectral: bool,
    evaluator_checkpoint: Path,
    config_path: Path,
    checkpoint_path: Path,
    selection_seed: int,
    source_snapshot_path: Path,
    source_snapshot_sha256: str,
) -> tuple[bool, str]:
    csv_path = output / "diagnostics.csv"
    manifest_path = output / "manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        return False, "diagnostics CSV or manifest missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"cannot parse final artifacts: {exc}"
    if manifest.get("schema_version") != MECHANISM_SCHEMA_VERSION:
        return False, "manifest schema version mismatch"
    if manifest.get("status") != "completed":
        return False, f"manifest status is {manifest.get('status')!r}"
    recorded_job = manifest.get("job", {})
    expected_job = {
        "cell": job.cell.upper(),
        "training_seed": job.training_seed,
        "repeat_index": job.repeat_index,
        "channel_seed": job.channel_seed,
    }
    if any(recorded_job.get(key) != value for key, value in expected_job.items()):
        return False, "manifest job identity mismatch"
    if int(manifest.get("rows", -1)) != expected_samples or len(rows) != expected_samples:
        return False, "diagnostic row count mismatch"
    if not math_isclose(float(manifest.get("snr_db", float("nan"))), snr_db):
        return False, "SNR mismatch"
    if bool(manifest.get("skip_pga")) != skip_pga:
        return False, "PGA policy mismatch"
    if bool(manifest.get("skip_spectral")) != skip_spectral:
        return False, "spectral policy mismatch"
    selection = manifest.get("selection", {})
    if selection.get("seed") != selection_seed:
        return False, "selection seed mismatch"
    recorded_snapshot = manifest.get("source_snapshot", {})
    if (
        Path(recorded_snapshot.get("path", "")).resolve()
        != source_snapshot_path.resolve()
        or recorded_snapshot.get("sha256") != source_snapshot_sha256
        or not source_snapshot_path.is_file()
        or sha256(source_snapshot_path) != source_snapshot_sha256
    ):
        return False, "source snapshot mismatch"
    source = manifest.get("source", {})
    try:
        recorded_evaluator = Path(source["evaluator_checkpoint"]).resolve()
    except (KeyError, TypeError):
        return False, "evaluator checkpoint absent from manifest"
    if recorded_evaluator != evaluator_checkpoint.resolve():
        return False, "evaluator checkpoint path mismatch"
    if not evaluator_checkpoint.is_file():
        return False, "evaluator checkpoint is now missing"
    if source.get("evaluator_checkpoint_sha256") != sha256(evaluator_checkpoint):
        return False, "evaluator checkpoint hash mismatch"
    if not bool(source.get("evaluator_qualified", False)):
        return False, "semantic evaluator is not qualified"
    try:
        evaluator_accuracy = float(source["evaluator_test_accuracy"])
    except (KeyError, TypeError, ValueError):
        return False, "semantic evaluator test accuracy is invalid"
    if not evaluator_accuracy >= 0.90:
        return False, "semantic evaluator is below the 90% gate"
    expected_sources = {
        "config": config_path,
        "checkpoint": checkpoint_path,
    }
    for name, expected_path in expected_sources.items():
        try:
            recorded_path = Path(source[name]).resolve()
        except (KeyError, TypeError):
            return False, f"{name} path absent from manifest"
        if recorded_path != expected_path.resolve():
            return False, f"{name} path mismatch"
        if not expected_path.is_file():
            return False, f"{name} is now missing"
        if source.get(f"{name}_sha256") != sha256(expected_path):
            return False, f"{name} hash mismatch"
    if manifest.get("diagnostics_csv_sha256") != sha256(csv_path):
        return False, "diagnostics CSV hash mismatch"

    try:
        positions = [int(row["sample_position"]) for row in rows]
        indices = [int(row["dataset_index"]) for row in rows]
        classes = [int(row["class_label"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False, "required diagnostic identity columns are invalid"
    if positions != list(range(expected_samples)) or len(set(indices)) != expected_samples:
        return False, "sample positions or dataset indices are not unique"
    if _indices_hash(indices) != manifest.get("selected_indices_sha256"):
        return False, "selected-index hash mismatch"
    expected_per_class = expected_samples // 10
    if Counter(classes) != Counter({label: expected_per_class for label in range(10)}):
        return False, "class balance mismatch"
    if any(
        row.get("cell") != job.cell.upper()
        or int(row.get("training_seed", -1)) != job.training_seed
        or int(row.get("repeat_index", -1)) != job.repeat_index
        or int(row.get("channel_seed", -1)) != job.channel_seed
        for row in rows
    ):
        return False, "CSV job identity mismatch"
    numeric_error = validate_numeric_rows(
        rows,
        job=job,
        skip_pga=skip_pga,
        skip_spectral=skip_spectral,
    )
    if numeric_error is not None:
        return False, numeric_error
    return True, "validated complete"


def _number(row: dict[str, str], field: str, *, allow_inf: bool = False) -> float:
    raw = row.get(field, "")
    if raw == "":
        raise ValueError(f"{field} is missing")
    value = float(raw)
    if math.isnan(value) or (math.isinf(value) and not allow_inf):
        raise ValueError(f"{field} is not an allowed finite value")
    return value


def validate_numeric_rows(
    rows: list[dict[str, str]],
    *,
    job: MechanismJob,
    skip_pga: bool,
    skip_spectral: bool,
) -> str | None:
    """Reject silent NaN, missing diagnostics, and impossible powers."""
    try:
        for row_index, row in enumerate(rows):
            for field in (
                "training_seed",
                "repeat_index",
                "channel_seed",
                "sample_position",
                "dataset_index",
                "class_label",
                "channel_uses",
                "clean_semantic_prediction",
                "clean_semantic_correct",
            ):
                int(row[field])
            for field in (
                "snr_db",
                "clean_semantic_cross_entropy",
                "clean_semantic_logit_margin",
                "native_failure_score",
                "semantic_failure_score",
            ):
                _number(row, field)
            if job.objective == "reconstruction":
                _number(row, "clean_native_mse")
                _number(row, "clean_native_psnr_db")
            else:
                int(row["clean_native_prediction"])
                int(row["clean_native_correct"])
                _number(row, "clean_native_cross_entropy")
                _number(row, "clean_native_logit_margin")
            for prefix in ("native", "semantic"):
                margin = _number(row, f"{prefix}_failure_margin")
                gradient = _number(row, f"{prefix}_margin_gradient_l2")
                if gradient < 0:
                    raise ValueError(f"{prefix} gradient norm is negative")
                status = row.get(f"{prefix}_linearized_distance_status", "")
                distance = _number(
                    row,
                    f"{prefix}_linearized_distance_l2",
                    allow_inf=status == "positive_margin_zero_gradient",
                )
                if status == "positive_margin_zero_gradient":
                    if not (margin > 0 and gradient == 0 and math.isinf(distance)):
                        raise ValueError(f"{prefix} zero-gradient status is inconsistent")
                elif status == "finite_gradient" and gradient <= 0:
                    raise ValueError(f"{prefix} finite-gradient status is inconsistent")
                elif status == "already_failed_zero_gradient" and not (
                    margin < 0 and gradient == 0 and distance == 0
                ):
                    raise ValueError(f"{prefix} failed zero-gradient status is inconsistent")
                elif status == "boundary_zero_gradient" and not (
                    margin == 0 and gradient == 0 and distance == 0
                ):
                    raise ValueError(f"{prefix} boundary zero-gradient status is inconsistent")
                elif status not in {
                    "finite_gradient",
                    "already_failed_zero_gradient",
                    "boundary_zero_gradient",
                }:
                    raise ValueError(f"{prefix} distance status is invalid")
            if not skip_spectral:
                spectral_prefixes = ["semantic"]
                if job.objective == "reconstruction":
                    spectral_prefixes.append("native_decoder")
                for prefix in spectral_prefixes:
                    if _number(row, f"{prefix}_spectral_norm") < 0:
                        raise ValueError(f"{prefix} spectral norm is negative")
                    _number(row, f"{prefix}_spectral_estimate_at_20")
                    _number(row, f"{prefix}_spectral_estimate_at_30")
                    _number(row, f"{prefix}_spectral_relative_change_20_30")
                    iterations = int(row[f"{prefix}_spectral_iterations"])
                    if iterations not in {30, 60}:
                        raise ValueError(f"{prefix} spectral iterations are invalid")
                    if row[f"{prefix}_spectral_converged_20_30"] not in {"0", "1"}:
                        raise ValueError(f"{prefix} convergence flag is invalid")
                    if (row[f"{prefix}_spectral_converged_20_30"] == "1") != (
                        iterations == 30
                    ):
                        raise ValueError(f"{prefix} convergence/iteration mismatch")
            if not skip_pga:
                for field in (
                    "pga_objective_variant",
                    "pga_success",
                    "pga_steps",
                    "pga_attacked_prediction",
                    "pga_attacked_logit_margin",
                    "pga_failure_score",
                    "pga_attack_power_total_l2_sq",
                    "pga_attack_power_per_channel_use",
                ):
                    if row.get(field, "") == "":
                        raise ValueError(f"{field} is missing")
                if int(row["pga_success"]) not in {0, 1}:
                    raise ValueError("PGA success flag is invalid")
                if int(row["pga_steps"]) < 0:
                    raise ValueError("PGA step count is negative")
                int(row["pga_attacked_prediction"])
                total = _number(row, "pga_attack_power_total_l2_sq")
                per_use = _number(row, "pga_attack_power_per_channel_use")
                _number(row, "pga_attacked_logit_margin")
                _number(row, "pga_failure_score")
                if total < 0 or per_use < 0:
                    raise ValueError("PGA power is negative")
                channel_uses = int(row["channel_uses"])
                if not math.isclose(total / channel_uses, per_use, rel_tol=2e-5, abs_tol=1e-9):
                    raise ValueError("PGA total and per-use powers are inconsistent")
                power_db = row.get("pga_attack_power_per_channel_use_db", "")
                if per_use == 0:
                    if power_db != "":
                        raise ValueError("zero PGA power must use a blank dB field")
                else:
                    _number(row, "pga_attack_power_per_channel_use_db")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return f"numeric validation failed at row {row_index}: {exc}"
    return None


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12 * max(1.0, abs(left), abs(right))


def validate_cross_job_invariants(output_root: Path, manifest_path: Path) -> tuple[bool, str]:
    """Ensure paired cells really used identical samples and standard noise."""
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_job = current["job"]
    current_selection = (
        current.get("selection", {}).get("seed"),
        current.get("selected_indices_sha256"),
    )
    current_noise = current.get("standard_noise", {})
    for other_path in output_root.glob("*/manifest.json"):
        if other_path.resolve() == manifest_path.resolve():
            continue
        other = json.loads(other_path.read_text(encoding="utf-8"))
        if other.get("status") != "completed":
            continue
        other_job = other.get("job", {})
        if other.get("snr_db") != current.get("snr_db"):
            continue
        other_selection = (
            other.get("selection", {}).get("seed"),
            other.get("selected_indices_sha256"),
        )
        if other_selection != current_selection:
            return False, f"sample selection differs from {other_job.get('identifier')}"
        if other_job.get("channel_seed") == current_job.get("channel_seed"):
            other_noise = other.get("standard_noise", {})
            comparable = ("seed", "shape", "sha256")
            if any(other_noise.get(key) != current_noise.get(key) for key in comparable):
                return False, f"shared noise differs from {other_job.get('identifier')}"
            current_channel = current.get("channel", {})
            other_channel = other.get("channel", {})
            for key in ("fading_gain", "noise_variance", "channel_uses"):
                if other_channel.get(key) != current_channel.get(key):
                    return False, f"channel {key} differs from {other_job.get('identifier')}"
    return True, "paired sample/noise invariants validated"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    command.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    command.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Use the first N fixed channel seeds (maximum 3).",
    )
    command.add_argument("--max-samples", type=int, default=1000)
    command.add_argument("--snr-db", type=float, default=DEFAULT_SNR_DB)
    command.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--workers", type=int, default=0)
    command.add_argument("--device", default="auto")
    command.add_argument("--skip-pga", action="store_true")
    command.add_argument("--skip-spectral", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--evaluator-checkpoint", type=Path, default=DEFAULT_EVALUATOR)
    command.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return command


def validate_arguments(args: argparse.Namespace) -> None:
    if not 1 <= args.repeats <= len(DEFAULT_NOISE_SEEDS):
        raise ValueError(f"--repeats must be in [1, {len(DEFAULT_NOISE_SEEDS)}].")
    if args.max_samples <= 0 or args.max_samples > 10_000 or args.max_samples % 10:
        raise ValueError("--max-samples must be a multiple of 10 in [10, 10000].")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative.")
    if len(set(args.cells)) != len(args.cells) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Cells and seeds must be unique.")


def main() -> int:
    args = parser().parse_args()
    validate_arguments(args)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evaluator = args.evaluator_checkpoint.resolve()
    jobs = [
        MechanismJob(cell, seed, repeat_index, DEFAULT_NOISE_SEEDS[repeat_index])
        for seed in args.seeds
        for cell in args.cells
        for repeat_index in range(args.repeats)
    ]
    job_sources: dict[str, tuple[Path, Path]] = {}
    artifact_records: list[dict[str, str]] = []
    for job in jobs:
        training_output = (
            REPO_ROOT / "outputs" / "factorial" / f"{job.cell}_seed{job.training_seed}"
        )
        config = training_output / "resolved_config.json"
        checkpoint = training_output / "checkpoint_best.pt"
        job_sources[job.identifier] = (config, checkpoint)
        for kind, path in (("config", config), ("checkpoint", checkpoint)):
            artifact_records.append(
                {
                    "job": job.identifier,
                    "kind": kind,
                    "path": str(path.resolve()),
                    "sha256": sha256(path) if path.is_file() else "missing",
                }
            )
    artifact_records.append(
        {
            "job": "shared",
            "kind": "evaluator",
            "path": str(evaluator),
            "sha256": sha256(evaluator) if evaluator.is_file() else "missing",
        }
    )

    snapshot_path = output_root / (
        "dry_run_source_snapshot.json" if args.dry_run else "source_snapshot.json"
    )
    if snapshot_path.exists() and not args.dry_run:
        source_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        verify_source_snapshot(source_snapshot)
    else:
        source_snapshot = build_source_snapshot()
        write_json(snapshot_path, source_snapshot)
    snapshot_hash = sha256(snapshot_path)
    signature_payload = {
        "schema_version": MECHANISM_SCHEMA_VERSION,
        "cells": args.cells,
        "training_seeds": args.seeds,
        "channel_seeds": list(DEFAULT_NOISE_SEEDS[: args.repeats]),
        "samples_per_job": args.max_samples,
        "snr_db": args.snr_db,
        "selection_seed": args.selection_seed,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "skip_pga": bool(args.skip_pga),
        "skip_spectral": bool(args.skip_spectral),
        "source_snapshot_sha256": snapshot_hash,
        "artifacts": artifact_records,
    }
    batch_signature = canonical_sha256(signature_payload)
    registry_path = output_root / (
        "dry_run_registry.csv" if args.dry_run else "registry.csv"
    )
    batch_manifest_path = output_root / (
        "dry_run_manifest.json" if args.dry_run else "batch_manifest.json"
    )
    previous_batch: dict[str, Any] | None = None
    if batch_manifest_path.exists() and not args.dry_run:
        previous_batch = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
        if previous_batch.get("batch_signature") != batch_signature:
            raise RuntimeError(
                "Existing formal batch manifest has a different immutable signature; "
                "use a new output root."
            )

    lock_path = output_root / ".run.lock"
    stale_lock: dict[str, Any] | None = None
    if not args.dry_run:
        _, stale_lock = acquire_lock(lock_path, batch_signature)

    rows = [
        {
            "cell": job.cell.upper(),
            "training_seed": job.training_seed,
            "repeat_index": job.repeat_index,
            "channel_seed": job.channel_seed,
            "status": "pending",
            "started_at": "",
            "finished_at": "",
            "exit_code": "",
            "output_dir": str((output_root / job.identifier).resolve()),
            "note": "",
        }
        for job in jobs
    ]
    write_registry(registry_path, rows)
    now = utc_now()
    batch_manifest: dict[str, Any] = {
        "schema_version": MECHANISM_SCHEMA_VERSION,
        "experiment": "common semantic endpoint and local sensitivity diagnostics",
        "status": "dry_run" if args.dry_run else "running",
        "created_at": previous_batch.get("created_at", now) if previous_batch else now,
        "resumed_at": now if previous_batch else None,
        "stale_lock_recovered": stale_lock,
        "batch_signature": batch_signature,
        "signature_payload": signature_payload,
        "source_snapshot": str(snapshot_path),
        "source_snapshot_sha256": snapshot_hash,
        "cells": args.cells,
        "training_seeds": args.seeds,
        "channel_seeds": list(DEFAULT_NOISE_SEEDS[: args.repeats]),
        "selection_seed": args.selection_seed,
        "jobs": len(jobs),
        "samples_per_job": args.max_samples,
        "expected_rows": len(jobs) * args.max_samples,
        "snr_db": args.snr_db,
        "device": args.device,
        "workers": args.workers,
        "skip_pga": bool(args.skip_pga),
        "skip_spectral": bool(args.skip_spectral),
        "evaluator_checkpoint": str(evaluator),
        "registry": str(registry_path),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "policy": (
            "serial; validated batch shards resume only on a new runner invocation; "
            "stop on first failure; never retry within the same invocation"
        ),
    }
    write_json(batch_manifest_path, batch_manifest)

    try:
        for index, job in enumerate(jobs):
            verify_source_snapshot(source_snapshot)
            row = rows[index]
            config, checkpoint = job_sources[job.identifier]
            output = Path(row["output_dir"])
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_mechanism_cell.py"),
                "--config", str(config),
                "--checkpoint", str(checkpoint),
                "--evaluator-checkpoint", str(evaluator),
                "--output", str(output),
                "--cell", job.cell,
                "--seed", str(job.training_seed),
                "--repeat-index", str(job.repeat_index),
                "--noise-seed", str(job.channel_seed),
                "--selection-seed", str(args.selection_seed),
                "--max-samples", str(args.max_samples),
                "--snr-db", str(args.snr_db),
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
                "--device", args.device,
                "--source-snapshot", str(snapshot_path),
                "--source-snapshot-sha256", snapshot_hash,
            ]
            if args.skip_pga:
                command.append("--skip-pga")
            if args.skip_spectral:
                command.append("--skip-spectral")
            if args.dry_run:
                row.update(status="dry_run", note=subprocess.list2cmdline(command))
                write_registry(registry_path, rows)
                continue

            missing = [str(path) for path in (config, checkpoint, evaluator) if not path.is_file()]
            if missing:
                row.update(
                    status="blocked_missing_source",
                    finished_at=utc_now(),
                    note="missing: " + "; ".join(missing),
                )
                write_registry(registry_path, rows)
                batch_manifest.update(
                    status="blocked", finished_at=utc_now(), blocked_job=job.identifier
                )
                write_json(batch_manifest_path, batch_manifest)
                return 2

            complete, reason = diagnostic_complete(
                output,
                job,
                expected_samples=args.max_samples,
                snr_db=args.snr_db,
                skip_pga=args.skip_pga,
                skip_spectral=args.skip_spectral,
                evaluator_checkpoint=evaluator,
                config_path=config,
                checkpoint_path=checkpoint,
                selection_seed=args.selection_seed,
                source_snapshot_path=snapshot_path,
                source_snapshot_sha256=snapshot_hash,
            )
            if complete:
                paired, paired_reason = validate_cross_job_invariants(
                    output_root, output / "manifest.json"
                )
                if not paired:
                    reason = paired_reason
                else:
                    row.update(
                        status="skipped_complete",
                        finished_at=utc_now(),
                        note=f"{reason}; {paired_reason}",
                    )
                    write_registry(registry_path, rows)
                    print(
                        f"[{index + 1}/{len(jobs)}] {job.identifier}: skipped (validated)",
                        flush=True,
                    )
                    continue

            resumable = False
            if output.exists() and any(output.iterdir()):
                partial_manifest = output / "manifest.json"
                if partial_manifest.is_file():
                    try:
                        partial = json.loads(partial_manifest.read_text(encoding="utf-8"))
                        resumable = partial.get("status") in {"running", "failed"}
                    except (OSError, json.JSONDecodeError):
                        resumable = False
                if not resumable:
                    row.update(
                        status="blocked_incomplete",
                        finished_at=utc_now(),
                        note=reason,
                    )
                    write_registry(registry_path, rows)
                    batch_manifest.update(
                        status="blocked", finished_at=utc_now(), blocked_job=job.identifier
                    )
                    write_json(batch_manifest_path, batch_manifest)
                    return 3

            output.mkdir(parents=True, exist_ok=True)
            command_line = subprocess.list2cmdline(command) + "\n"
            command_path = output / "command.txt"
            if command_path.exists() and command_path.read_text(encoding="utf-8") != command_line:
                row.update(status="blocked_incomplete", note="resume command differs")
                write_registry(registry_path, rows)
                return 3
            command_path.write_text(command_line, encoding="utf-8")
            row.update(
                status="resuming" if resumable else "running",
                started_at=utc_now(),
                note="validated shard resume" if resumable else "",
            )
            write_registry(registry_path, rows)
            print(
                f"[{index + 1}/{len(jobs)}] "
                f"{'resuming' if resumable else 'starting'} {job.identifier}",
                flush=True,
            )
            log_mode = "a" if resumable else "w"
            with (output / "diagnostics.stdout.log").open(
                log_mode, encoding="utf-8"
            ) as stdout, (output / "diagnostics.stderr.log").open(
                log_mode, encoding="utf-8"
            ) as stderr:
                if resumable:
                    stdout.write(f"\n--- resumed {utc_now()} ---\n")
                    stderr.write(f"\n--- resumed {utc_now()} ---\n")
                result = subprocess.run(
                    command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False
                )
            row.update(exit_code=result.returncode, finished_at=utc_now())
            if result.returncode != 0:
                row.update(status="failed", note="subprocess failed; not retried")
                write_registry(registry_path, rows)
                batch_manifest.update(
                    status="failed", finished_at=utc_now(), failed_job=job.identifier
                )
                write_json(batch_manifest_path, batch_manifest)
                return result.returncode or 1
            verify_source_snapshot(source_snapshot)
            complete, reason = diagnostic_complete(
                output,
                job,
                expected_samples=args.max_samples,
                snr_db=args.snr_db,
                skip_pga=args.skip_pga,
                skip_spectral=args.skip_spectral,
                evaluator_checkpoint=evaluator,
                config_path=config,
                checkpoint_path=checkpoint,
                selection_seed=args.selection_seed,
                source_snapshot_path=snapshot_path,
                source_snapshot_sha256=snapshot_hash,
            )
            if complete:
                complete, paired_reason = validate_cross_job_invariants(
                    output_root, output / "manifest.json"
                )
                reason = f"{reason}; {paired_reason}"
            if not complete:
                row.update(status="failed_validation", note=reason)
                write_registry(registry_path, rows)
                batch_manifest.update(
                    status="failed_validation",
                    finished_at=utc_now(),
                    failed_job=job.identifier,
                )
                write_json(batch_manifest_path, batch_manifest)
                return 4
            row.update(status="completed", note=reason)
            write_registry(registry_path, rows)

        if args.dry_run:
            print(f"dry-run registry: {registry_path}", flush=True)
            return 0
        batch_manifest.update(status="completed", finished_at=utc_now())
        write_json(batch_manifest_path, batch_manifest)
        print(f"all {len(jobs)} jobs completed; registry: {registry_path}", flush=True)
        return 0
    finally:
        if not args.dry_run and lock_path.exists():
            try:
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                if int(lock.get("pid", -1)) == os.getpid():
                    lock_path.unlink()
            except (OSError, ValueError, json.JSONDecodeError):
                pass


if __name__ == "__main__":
    raise SystemExit(main())
