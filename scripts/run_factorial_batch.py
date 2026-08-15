"""Run the R0/R1/C0/C1 three-seed training matrix serially and resumably.

The runner never overwrites an incomplete experiment and never retries a failed
experiment. A completed cell is skipped only when both its run manifest and its
1000-row training log pass validation.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIGS = {
    "r0": REPO_ROOT / "configs" / "factorial" / "r0_reconstruction_noiseless_seed2026.json",
    "r1": REPO_ROOT / "configs" / "factorial" / "r1_reconstruction_awgn10_seed2026.json",
    "c0": REPO_ROOT / "configs" / "factorial" / "c0_classification_noiseless_seed2026.json",
    "c1": REPO_ROOT / "configs" / "factorial" / "c1_classification_awgn10_seed2026.json",
}
REGISTRY_FIELDS = [
    "cell",
    "seed",
    "status",
    "started_at",
    "finished_at",
    "pid",
    "exit_code",
    "output_dir",
    "config_path",
    "note",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, ctypes.c_ulong(pid)
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_process(pid: int, poll_seconds: int) -> None:
    if not process_alive(pid):
        print(f"wait PID {pid}: already finished", flush=True)
        return
    print(f"waiting for existing training PID {pid}", flush=True)
    while process_alive(pid):
        print(f"wait PID {pid}: still running at {utc_now()}", flush=True)
        time.sleep(poll_seconds)
    print(f"wait PID {pid}: finished at {utc_now()}", flush=True)


def resolved_config(cell: str, seed: int, output_dir: Path) -> dict[str, Any]:
    base_path = BASE_CONFIGS[cell]
    config = json.loads(base_path.read_text(encoding="utf-8"))
    data_root = Path(config["data"]["root"])
    if not data_root.is_absolute():
        data_root = (base_path.parent / data_root).resolve()
    config["data"]["root"] = str(data_root)
    config["seed"] = seed
    config["output_dir"] = str(output_dir.resolve())
    config["batch_runner"] = {
        "base_config": str(base_path.relative_to(REPO_ROOT)),
        "base_config_sha256": sha256(base_path),
        "generated_at": utc_now(),
    }
    return config


def completed(output_dir: Path, expected_epochs: int) -> tuple[bool, str]:
    manifest_path = output_dir / "run_manifest.json"
    log_path = output_dir / "training_log.csv"
    best_path = output_dir / "checkpoint_best.pt"
    last_path = output_dir / "checkpoint_last.pt"
    required = [manifest_path, log_path, best_path, last_path]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False, "required final artifacts are missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with log_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"final artifacts cannot be parsed: {exc}"
    if int(manifest.get("epochs", -1)) != expected_epochs:
        return False, "manifest epoch count does not match configuration"
    if len(rows) != expected_epochs or int(rows[-1].get("epoch", -1)) != expected_epochs:
        return False, "training log is incomplete"
    return True, "validated complete"


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--cells",
        nargs="+",
        choices=sorted(BASE_CONFIGS),
        default=["r0", "r1", "c0", "c1"],
    )
    command.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    command.add_argument("--device", default="cuda")
    command.add_argument("--wait-pid", type=int)
    command.add_argument("--poll-seconds", type=int, default=30)
    command.add_argument(
        "--batch-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "factorial" / "factorial_2x2_three_seed_batch",
    )
    command.add_argument("--dry-run", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    if args.poll_seconds < 5:
        raise ValueError("--poll-seconds must be at least 5.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")

    batch_dir = args.batch_dir.resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    registry_path = batch_dir / "registry.csv"
    rows: list[dict[str, Any]] = []
    # Keep the four conditions adjacent within each seed. This makes partial
    # batches interpretable and reaches a complete 2x2 block before moving on.
    jobs = [(cell, seed) for seed in args.seeds for cell in args.cells]
    for cell, seed in jobs:
        output_dir = REPO_ROOT / "outputs" / "factorial" / f"{cell}_seed{seed}"
        rows.append(
            {
                "cell": cell.upper(),
                "seed": seed,
                "status": "pending",
                "started_at": "",
                "finished_at": "",
                "pid": "",
                "exit_code": "",
                "output_dir": str(output_dir.resolve()),
                "config_path": str((output_dir / "resolved_config.json").resolve()),
                "note": "",
            }
        )
    write_registry(registry_path, rows)
    write_json(
        batch_dir / "batch_manifest.json",
        {
            "experiment": "R0/R1/C0/C1 three-seed factorial training",
            "created_at": utc_now(),
            "status": "dry_run" if args.dry_run else "running",
            "cells": args.cells,
            "seeds": args.seeds,
            "jobs": len(jobs),
            "device": args.device,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_commit_at_launch": git_commit(),
            "wait_pid": args.wait_pid,
            "registry": str(registry_path),
            "batch_stdout": str(batch_dir / "batch.stdout.log"),
            "batch_stderr": str(batch_dir / "batch.stderr.log"),
            "policy": "serial; skip validated-complete; stop on first failure; never auto-retry",
        },
    )

    if args.dry_run:
        print(json.dumps({"jobs": jobs, "registry": str(registry_path)}, indent=2), flush=True)
        return 0
    if args.wait_pid:
        wait_for_process(args.wait_pid, args.poll_seconds)

    for index, (cell, seed) in enumerate(jobs):
        row = rows[index]
        output_dir = Path(row["output_dir"])
        expected_epochs = int(
            json.loads(BASE_CONFIGS[cell].read_text(encoding="utf-8"))["training"]["epochs"]
        )
        is_complete, reason = completed(output_dir, expected_epochs)
        if is_complete:
            row.update(status="skipped_complete", finished_at=utc_now(), note=reason)
            write_registry(registry_path, rows)
            print(f"[{index + 1}/{len(jobs)}] {cell.upper()} seed {seed}: skipped (complete)", flush=True)
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            row.update(status="blocked_incomplete", finished_at=utc_now(), note=reason)
            write_registry(registry_path, rows)
            print(
                f"[{index + 1}/{len(jobs)}] {cell.upper()} seed {seed}: "
                f"existing incomplete output; stopping without overwrite",
                file=sys.stderr,
                flush=True,
            )
            return 2

        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "resolved_config.json"
        write_json(config_path, resolved_config(cell, seed, output_dir))
        command = [
            sys.executable,
            str(REPO_ROOT / "run_semantic.py"),
            "train",
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--device",
            args.device,
        ]
        (output_dir / "command.txt").write_text(
            subprocess.list2cmdline(command) + "\n", encoding="utf-8"
        )
        row.update(status="running", started_at=utc_now(), note="")
        write_registry(registry_path, rows)
        print(
            f"[{index + 1}/{len(jobs)}] starting {cell.upper()} seed {seed}: "
            f"{subprocess.list2cmdline(command)}",
            flush=True,
        )
        with (output_dir / "formal_train.stdout.log").open("w", encoding="utf-8") as stdout, (
            output_dir / "formal_train.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr)
            row["pid"] = process.pid
            write_registry(registry_path, rows)
            exit_code = process.wait()
        row["exit_code"] = exit_code
        row["finished_at"] = utc_now()
        if exit_code != 0:
            row.update(status="failed", note="subprocess failed; not retried")
            write_registry(registry_path, rows)
            print(
                f"{cell.upper()} seed {seed} failed with exit code {exit_code}; batch stopped",
                file=sys.stderr,
                flush=True,
            )
            return exit_code or 1
        is_complete, reason = completed(output_dir, expected_epochs)
        if not is_complete:
            row.update(status="failed_validation", note=reason)
            write_registry(registry_path, rows)
            print(f"{cell.upper()} seed {seed} failed artifact validation: {reason}", file=sys.stderr, flush=True)
            return 3
        row.update(status="completed", note=reason)
        write_registry(registry_path, rows)
        print(f"[{index + 1}/{len(jobs)}] {cell.upper()} seed {seed}: completed", flush=True)

    manifest_path = batch_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(status="completed", finished_at=utc_now())
    write_json(manifest_path, manifest)
    print(f"all {len(jobs)} jobs completed; registry: {registry_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
