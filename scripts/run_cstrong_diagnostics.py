"""Serial low-noise runner for all C-strong arm/repeat diagnostics."""

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
DEFAULT_ROOT = REPO_ROOT / "outputs" / "cstrong_pilot"
NOISE_SEEDS = (102026, 102027, 102028)
ARMS = ("CS0", "CS1", "CSJ")
SOURCE_FILES = (
    REPO_ROOT / "semantic_robustness" / "model.py",
    REPO_ROOT / "semantic_robustness" / "channel.py",
    REPO_ROOT / "semantic_robustness" / "metrics.py",
    REPO_ROOT / "semantic_robustness" / "theory.py",
    REPO_ROOT / "semantic_robustness" / "mechanism.py",
    REPO_ROOT / "semantic_robustness" / "attacks.py",
    REPO_ROOT / "semantic_robustness" / "cstrong.py",
    REPO_ROOT / "scripts" / "evaluate_cstrong_diagnostics.py",
    Path(__file__).resolve(),
)


@dataclass(frozen=True)
class DiagnosticJob:
    arm: str
    seed: int
    repeat: int
    noise_seed: int
    root: Path

    @property
    def identifier(self) -> str:
        return f"{self.arm.lower()}_seed{self.seed}_repeat{self.repeat}"

    @property
    def training_output(self) -> Path:
        return self.root / f"{self.arm.lower()}_seed{self.seed}"

    @property
    def output(self) -> Path:
        return self.training_output / "diagnostics" / f"repeat{self.repeat}"

    @property
    def config(self) -> Path:
        return self.root / "resolved_configs" / f"{self.arm.lower()}_seed{self.seed}.json"


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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def snapshot() -> dict[str, str]:
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in SOURCE_FILES}


def validate_snapshot(expected: dict[str, str]) -> None:
    if snapshot() != expected:
        raise RuntimeError("Diagnostic source files changed during the batch.")


def acquire_lock(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Diagnostic runner lock already exists: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": utc_now()}, stream)


def release_lock(path: Path) -> None:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("pid", -1)) == os.getpid():
                path.unlink()
        except (OSError, ValueError, json.JSONDecodeError):
            pass


def diagnostic_complete(job: DiagnosticJob, samples: int) -> tuple[bool, str]:
    manifest_path = job.output / "manifest.json"
    csv_path = job.output / "diagnostics.csv"
    if not manifest_path.is_file() or not csv_path.is_file():
        return False, "manifest or diagnostics missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature = manifest["job_signature"]
        if manifest.get("status") != "completed" or int(manifest.get("rows", -1)) != samples:
            return False, "manifest incomplete"
        if signature["arm"] != job.arm or int(signature["training_seed"]) != job.seed:
            return False, "identity mismatch"
        if int(signature["repeat_index"]) != job.repeat or int(signature["channel_seed"]) != job.noise_seed:
            return False, "repeat identity mismatch"
        if manifest.get("diagnostics_sha256") != sha256(csv_path):
            return False, "diagnostics hash mismatch"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, f"validation error: {exc}"
    return True, "validated complete"


def technical_gate(jobs: list[DiagnosticJob], samples: int) -> tuple[bool, dict[str, Any]]:
    records: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    convergence: list[int] = []
    for job in jobs:
        with (job.output / "diagnostics.csv").open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        records[(job.arm, job.seed, job.repeat)] = rows
        convergence.extend(int(row["spectral_converged"]) for row in rows)
    pair_counts: dict[str, list[int]] = {}
    arms = tuple(dict.fromkeys(job.arm for job in jobs))
    seeds = tuple(dict.fromkeys(job.seed for job in jobs))
    comparisons = tuple(
        (arms[left], arms[right])
        for left in range(len(arms))
        for right in range(left + 1, len(arms))
    )
    for seed in seeds:
        for off, on in comparisons:
            name = f"seed{seed}:{off}-{on}"
            pair_counts[name] = []
            for repeat in range(3):
                left = records[(off, seed, repeat)]
                right = records[(on, seed, repeat)]
                if [row["dataset_index"] for row in left] != [row["dataset_index"] for row in right]:
                    raise RuntimeError(
                        f"Shared sample order failed for {name} repeat {repeat}."
                    )
                pair_counts[name].append(
                    sum(
                        int(a["clean_correct"]) and int(b["clean_correct"])
                        for a, b in zip(left, right, strict=True)
                    )
                )
    convergence_rate = sum(convergence) / len(convergence)
    required_common = min(500, samples)
    passed = convergence_rate >= 0.95 and all(
        count >= required_common for counts in pair_counts.values() for count in counts
    )
    return passed, {
        "spectral_convergence_rate": convergence_rate,
        "required_pairwise_common_correct": required_common,
        "pairwise_common_correct_by_repeat": pair_counts,
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    command.add_argument("--seed", type=int, default=2026)
    command.add_argument("--seeds", type=int, nargs="+")
    command.add_argument("--arms", nargs="+", choices=ARMS)
    command.add_argument("--device", default="cuda")
    command.add_argument("--workers", type=int, default=0)
    command.add_argument("--max-samples", type=int, default=1000)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--skip-pga", action="store_true")
    command.add_argument("--skip-spectral", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    if args.max_samples <= 0 or args.max_samples % 10:
        raise ValueError("--max-samples must be positive and divisible by ten.")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds or [args.seed]
    arms = tuple(args.arms or ARMS)
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain unique non-negative integers.")
    if len(set(arms)) != len(arms):
        raise ValueError("--arms must not contain duplicates.")
    jobs = [
        DiagnosticJob(arm, seed, repeat, noise_seed, root)
        for seed in seeds
        for arm in arms
        for repeat, noise_seed in enumerate(NOISE_SEEDS)
    ]
    rows = [
        {
            "job": job.identifier,
            "arm": job.arm,
            "seed": job.seed,
            "repeat": job.repeat,
            "noise_seed": job.noise_seed,
            "status": "pending",
            "started_at": "",
            "finished_at": "",
            "exit_code": "",
            "note": "",
        }
        for job in jobs
    ]
    for row, job in zip(rows, jobs, strict=True):
        complete, reason = diagnostic_complete(job, args.max_samples)
        if complete:
            row.update(status="completed", note=reason)
        elif job.output.exists() and any(job.output.iterdir()):
            manifest_path = job.output / "manifest.json"
            status = None
            if manifest_path.is_file():
                status = json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
            row.update(
                status="resumable" if status in {"running", "interrupted"} else "blocked",
                note=f"existing partial status={status}",
            )
    registry_path = root / "diagnostics_registry.csv"
    manifest_path = root / "diagnostics_batch_manifest.json"
    snapshot_path = root / "diagnostics_source_snapshot.json"
    source = snapshot()
    if snapshot_path.exists():
        previous_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if previous_snapshot.get("files") != source:
            raise RuntimeError(
                "Existing diagnostic batch uses different source files; refusing mixed-code resume."
            )
    else:
        atomic_json(snapshot_path, {"created_at": utc_now(), "files": source})
    write_registry(registry_path, rows)
    batch: dict[str, Any] = {
        "schema_version": "cstrong_diagnostics_batch_v1",
        "status": "dry_run" if args.dry_run else "running",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "seed": args.seed if len(seeds) == 1 else None,
        "seeds": seeds,
        "arms": arms,
        "jobs": [job.identifier for job in jobs],
        "max_samples": args.max_samples,
        "source_snapshot_sha256": sha256(snapshot_path),
        "automatic_retry": False,
        "command": sys.argv,
    }
    atomic_json(manifest_path, batch)
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "registry": str(registry_path)}, ensure_ascii=False), flush=True)
        return 0
    for job in jobs:
        if not job.config.is_file() or not (job.training_output / "checkpoint_best.pt").is_file():
            raise FileNotFoundError(f"Training artifacts missing for {job.identifier}.")
    lock_path = root / "diagnostics_runner.lock"
    acquire_lock(lock_path)
    try:
        for index, (job, row) in enumerate(zip(jobs, rows, strict=True), start=1):
            validate_snapshot(source)
            if row["status"] == "completed":
                continue
            if row["status"] == "blocked":
                batch.update(status="blocked", failed_job=job.identifier, finished_at=utc_now())
                atomic_json(manifest_path, batch)
                return 3
            resume = row["status"] == "resumable"
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_cstrong_diagnostics.py"),
                "--config", str(job.config),
                "--checkpoint", str(job.training_output / "checkpoint_best.pt"),
                "--output", str(job.output),
                "--arm", job.arm,
                "--seed", str(job.seed),
                "--repeat-index", str(job.repeat),
                "--noise-seed", str(job.noise_seed),
                "--max-samples", str(args.max_samples),
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
                "--device", args.device,
            ]
            if args.skip_pga:
                command.append("--skip-pga")
            if args.skip_spectral:
                command.append("--skip-spectral")
            if resume:
                command.append("--resume")
            row.update(status="running", started_at=utc_now(), note="resuming" if resume else "starting")
            write_registry(registry_path, rows)
            print(f"[{index}/{len(jobs)}] {'resuming' if resume else 'starting'} {job.identifier}", flush=True)
            job.output.mkdir(parents=True, exist_ok=True)
            with (job.output / "diagnostics.stdout.log").open("a" if resume else "w", encoding="utf-8") as stdout, (job.output / "diagnostics.stderr.log").open("a" if resume else "w", encoding="utf-8") as stderr:
                result = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False)
            row.update(exit_code=result.returncode, finished_at=utc_now())
            if result.returncode != 0:
                row.update(status="failed", note="subprocess failed; not retried")
                write_registry(registry_path, rows)
                batch.update(status="failed", failed_job=job.identifier, finished_at=utc_now())
                atomic_json(manifest_path, batch)
                return result.returncode or 1
            complete, reason = diagnostic_complete(job, args.max_samples)
            if not complete:
                row.update(status="failed_validation", note=reason)
                write_registry(registry_path, rows)
                batch.update(status="failed_validation", failed_job=job.identifier, finished_at=utc_now())
                atomic_json(manifest_path, batch)
                return 4
            row.update(status="completed", note=reason)
            write_registry(registry_path, rows)
        gate = None
        gate_details: dict[str, Any] = {}
        if not args.skip_spectral:
            gate, gate_details = technical_gate(jobs, args.max_samples)
        batch.update(
            status="completed",
            finished_at=utc_now(),
            completed_jobs=len(jobs),
            diagnostics_technical_gate_passed=gate,
            technical_gate_details=gate_details,
        )
        atomic_json(manifest_path, batch)
        print(json.dumps({"status": "completed", "jobs": len(jobs), "diagnostics_gate": gate}, ensure_ascii=False), flush=True)
        return 0
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
