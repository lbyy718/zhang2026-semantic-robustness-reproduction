"""Serial, resumable and low-noise runner for CS0/CS1/CSJ training jobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "cstrong"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "cstrong_pilot"
DEFAULT_FORMAL_LOW_LR_OUTPUT = REPO_ROOT / "outputs" / "cstrong_formal_lr001"
PILOT_ARM_CONFIGS = {
    "CS0": CONFIG_ROOT / "cs0_resnet18_noiseless_seed2026.json",
    "CS1": CONFIG_ROOT / "cs1_resnet18_awgn10_seed2026.json",
    "CSJ": CONFIG_ROOT / "csj_resnet18_jacobian_seed2026.json",
}
FORMAL_LOW_LR_ARM_CONFIGS = {
    "CS0": CONFIG_ROOT / "cs0_resnet18_noiseless_formal_lr001.json",
    "CSJ": CONFIG_ROOT / "csj_resnet18_jacobian_formal_lr001.json",
}
SOURCE_FILES = (
    REPO_ROOT / "semantic_robustness" / "model.py",
    REPO_ROOT / "semantic_robustness" / "config.py",
    REPO_ROOT / "semantic_robustness" / "channel.py",
    REPO_ROOT / "semantic_robustness" / "metrics.py",
    REPO_ROOT / "semantic_robustness" / "runtime.py",
    REPO_ROOT / "semantic_robustness" / "cstrong.py",
    REPO_ROOT / "scripts" / "train_cstrong.py",
    Path(__file__).resolve(),
    *PILOT_ARM_CONFIGS.values(),
    *FORMAL_LOW_LR_ARM_CONFIGS.values(),
)


@dataclass(frozen=True)
class Job:
    arm: str
    seed: int
    config: Path
    output: Path

    @property
    def identifier(self) -> str:
        return f"{self.arm.lower()}_seed{self.seed}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Registry cannot be empty.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def source_snapshot() -> dict[str, str]:
    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"C-strong source snapshot is incomplete: {missing}")
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in SOURCE_FILES}


def verify_source_snapshot(expected: dict[str, str]) -> None:
    actual = source_snapshot()
    if actual != expected:
        changed = sorted(set(actual) | set(expected))
        changed = [name for name in changed if actual.get(name) != expected.get(name)]
        raise RuntimeError(f"C-strong sources changed during the batch: {changed}")


def acquire_lock(path: Path) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(path, flags)
    except FileExistsError as exc:
        try:
            lock = json.loads(path.read_text(encoding="utf-8"))
            pid = int(lock.get("pid", -1))
            os.kill(pid, 0)
        except (OSError, ValueError, json.JSONDecodeError):
            raise RuntimeError(
                f"Stale lock exists at {path}; inspect it and remove it explicitly."
            ) from exc
        raise RuntimeError(f"C-strong runner is already active with PID {pid}.") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": utc_now()}, stream)


def release_lock(path: Path) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("pid", -1)) == os.getpid():
            path.unlink()
    except (OSError, ValueError, json.JSONDecodeError):
        pass


def resolved_config(job: Job, directory: Path) -> Path:
    payload = json.loads(job.config.read_text(encoding="utf-8"))
    payload["seed"] = job.seed
    payload["output_dir"] = str(job.output)
    data_root = (job.config.parent / payload["data"]["root"]).resolve()
    payload["data"]["root"] = str(data_root)
    path = directory / f"{job.identifier}.json"
    atomic_json(path, payload)
    return path


def complete_job(output: Path, expected_arm: str, expected_seed: int) -> tuple[bool, str]:
    required = (
        output / "manifest.json",
        output / "checkpoint_best.pt",
        output / "checkpoint_last.pt",
        output / "training_log.csv",
        output / "test_clean_curve.csv",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return False, f"missing {missing}"
    try:
        manifest = json.loads(required[0].read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            return False, f"manifest status={manifest.get('status')!r}"
        if manifest.get("arm") != expected_arm or int(manifest.get("seed", -1)) != expected_seed:
            return False, "manifest identity mismatch"
        hashes = {
            "checkpoint_best_sha256": sha256(required[1]),
            "training_log_sha256": sha256(required[3]),
            "test_clean_curve_sha256": sha256(required[4]),
        }
        for name, value in hashes.items():
            if manifest.get(name) != value:
                return False, f"hash mismatch: {name}"
        with required[3].open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != int(manifest.get("completed_epochs", -1)) or not rows:
            return False, "training-log row count mismatch"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, f"validation error: {type(exc).__name__}: {exc}"
    return True, "validated complete"


def jobs_for(
    seeds: list[int], output_root: Path, arm_configs: dict[str, Path]
) -> list[Job]:
    return [
        Job(arm, seed, path, output_root / f"{arm.lower()}_seed{seed}")
        for seed in seeds
        for arm, path in arm_configs.items()
    ]


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--seeds", type=int, nargs="+", default=[2026])
    command.add_argument(
        "--protocol",
        choices=("pilot_three_arm", "formal_low_lr_cs0_csj"),
        default="pilot_three_arm",
    )
    command.add_argument("--output-root", type=Path)
    command.add_argument("--device", default="cuda")
    command.add_argument("--workers", type=int, default=0)
    command.add_argument("--epochs-override", type=int)
    command.add_argument("--test-max-samples", type=int)
    command.add_argument("--dry-run", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must contain unique non-negative integers.")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative.")
    arm_configs = (
        PILOT_ARM_CONFIGS
        if args.protocol == "pilot_three_arm"
        else FORMAL_LOW_LR_ARM_CONFIGS
    )
    default_output = (
        DEFAULT_OUTPUT
        if args.protocol == "pilot_three_arm"
        else DEFAULT_FORMAL_LOW_LR_OUTPUT
    )
    output_root = (args.output_root or default_output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_directory = output_root / "resolved_configs"
    config_directory.mkdir(exist_ok=True)
    jobs = jobs_for(args.seeds, output_root, arm_configs)
    if args.protocol == "pilot_three_arm" and args.seeds == [2026] and len(jobs) != 3:
        raise RuntimeError("The seed-2026 pilot must contain exactly three jobs.")
    if args.protocol == "formal_low_lr_cs0_csj" and len(jobs) != 2 * len(args.seeds):
        raise RuntimeError("The formal low-LR batch must contain CS0 and CSJ for every seed.")
    registry_path = output_root / "registry.csv"
    manifest_path = output_root / "batch_manifest.json"
    lock_path = output_root / "runner.lock"
    snapshot = source_snapshot()
    snapshot_path = output_root / "source_snapshot.json"
    rows = [
        {
            "job": job.identifier,
            "arm": job.arm,
            "seed": job.seed,
            "status": "pending",
            "started_at": "",
            "finished_at": "",
            "exit_code": "",
            "note": "",
            "output": str(job.output),
        }
        for job in jobs
    ]
    for row, job in zip(rows, jobs, strict=True):
        complete, reason = complete_job(job.output, job.arm, job.seed)
        if complete:
            row.update(status="completed", note=reason)
        elif job.output.exists() and any(job.output.iterdir()):
            arm_manifest = job.output / "manifest.json"
            if arm_manifest.is_file():
                status = json.loads(arm_manifest.read_text(encoding="utf-8")).get("status")
                if status in {"running", "interrupted"}:
                    row.update(status="resumable", note=f"partial status={status}")
                else:
                    row.update(status="blocked", note=f"invalid existing output: {reason}")
            else:
                row.update(status="blocked", note=f"invalid existing output: {reason}")
    if snapshot_path.exists():
        previous_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if previous_snapshot.get("files") != snapshot:
            raise RuntimeError(
                "Existing batch was created from different source files; refusing mixed-code resume."
            )
    else:
        atomic_json(snapshot_path, {"created_at": utc_now(), "files": snapshot})
    write_registry(registry_path, rows)
    batch_manifest: dict[str, Any] = {
        "schema_version": "cstrong_batch_v1",
        "scope": (
            "serial C-strong three-arm training"
            if args.protocol == "pilot_three_arm"
            else "serial C-strong CS0/CSJ formal low-learning-rate training"
        ),
        "status": "dry_run" if args.dry_run else "running",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "seeds": args.seeds,
        "protocol": args.protocol,
        "job_count": len(jobs),
        "job_order": [job.identifier for job in jobs],
        "device": args.device,
        "workers": args.workers,
        "epochs_override": args.epochs_override,
        "test_max_samples": args.test_max_samples,
        "source_snapshot": str(snapshot_path),
        "source_snapshot_sha256": sha256(snapshot_path),
        "registry": str(registry_path),
        "low_polling_policy": {
            "training_stdout_interval_epochs": 10,
            "automatic_retry": False,
            "runner_mode": "blocking subprocess completion events",
        },
        "command": sys.argv,
    }
    atomic_json(manifest_path, batch_manifest)
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "registry": str(registry_path)}, ensure_ascii=False), flush=True)
        return 0

    acquire_lock(lock_path)
    try:
        for index, (job, row) in enumerate(zip(jobs, rows, strict=True), start=1):
            verify_source_snapshot(snapshot)
            if row["status"] == "completed":
                continue
            if row["status"] == "blocked":
                batch_manifest.update(status="blocked", failed_job=job.identifier, finished_at=utc_now())
                atomic_json(manifest_path, batch_manifest)
                return 3
            resume = row["status"] == "resumable"
            config_path = resolved_config(job, config_directory)
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "train_cstrong.py"),
                "--config",
                str(config_path),
                "--output",
                str(job.output),
                "--device",
                args.device,
                "--workers",
                str(args.workers),
            ]
            if args.epochs_override is not None:
                command.extend(["--epochs-override", str(args.epochs_override)])
            if args.test_max_samples is not None:
                command.extend(["--test-max-samples", str(args.test_max_samples)])
            if resume:
                command.append("--resume")
            row.update(status="running", started_at=utc_now(), note="resuming" if resume else "starting")
            write_registry(registry_path, rows)
            print(f"[{index}/{len(jobs)}] {'resuming' if resume else 'starting'} {job.identifier}", flush=True)
            job.output.mkdir(parents=True, exist_ok=True)
            stdout_path = job.output / "formal_train.stdout.log"
            stderr_path = job.output / "formal_train.stderr.log"
            mode = "a" if resume else "w"
            with stdout_path.open(mode, encoding="utf-8") as stdout, stderr_path.open(mode, encoding="utf-8") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert process.stdout is not None
                for line in process.stdout:
                    stdout.write(line)
                    stdout.flush()
                    print(line.rstrip(), flush=True)
                exit_code = process.wait()
            row.update(exit_code=exit_code, finished_at=utc_now())
            if exit_code != 0:
                row.update(status="failed", note="subprocess failed; not retried")
                write_registry(registry_path, rows)
                batch_manifest.update(status="failed", failed_job=job.identifier, finished_at=utc_now())
                atomic_json(manifest_path, batch_manifest)
                return exit_code or 1
            verify_source_snapshot(snapshot)
            complete, reason = complete_job(job.output, job.arm, job.seed)
            if not complete:
                row.update(status="failed_validation", note=reason)
                write_registry(registry_path, rows)
                batch_manifest.update(status="failed_validation", failed_job=job.identifier, finished_at=utc_now())
                atomic_json(manifest_path, batch_manifest)
                return 4
            row.update(status="completed", note=reason)
            write_registry(registry_path, rows)

        manifests = {
            (job.arm, job.seed): json.loads(
                (job.output / "manifest.json").read_text(encoding="utf-8")
            )
            for job in jobs
        }
        training_gate: bool | None = None
        seed2026 = {
            arm: manifest
            for (arm, seed), manifest in manifests.items()
            if seed == 2026
        }
        if (
            args.protocol == "pilot_three_arm"
            and set(seed2026) == {"CS0", "CS1", "CSJ"}
            and args.test_max_samples is None
        ):
            training_gate = bool(
                seed2026["CS0"]["test_10db_accuracy"] >= 0.75
                and seed2026["CS1"]["test_10db_accuracy"] >= 0.75
                and seed2026["CSJ"]["test_10db_accuracy"] >= 0.72
                and seed2026["CSJ"]["test_10db_accuracy"]
                >= seed2026["CS0"]["test_10db_accuracy"] - 0.03
            )
        elif args.protocol == "formal_low_lr_cs0_csj" and args.test_max_samples is None:
            training_gate = all(
                manifests[("CS0", seed)]["test_10db_accuracy"] >= 0.75
                and manifests[("CSJ", seed)]["test_10db_accuracy"] >= 0.72
                and manifests[("CSJ", seed)]["test_10db_accuracy"]
                >= manifests[("CS0", seed)]["test_10db_accuracy"] - 0.03
                for seed in args.seeds
            )
        batch_manifest.update(
            status="completed",
            finished_at=utc_now(),
            completed_jobs=len(jobs),
            seed2026_training_quality_gate_passed=training_gate,
        )
        atomic_json(manifest_path, batch_manifest)
        print(json.dumps({"status": "completed", "jobs": len(jobs), "training_gate": training_gate}, ensure_ascii=False), flush=True)
        return 0
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
