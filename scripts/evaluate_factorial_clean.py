"""Evaluate every completed 2x2 factorial checkpoint over the common SNR grid."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CELLS = ("r0", "r1", "c0", "c1")
SEEDS = (2026, 2027, 2028)
FIELDS = (
    "cell",
    "seed",
    "status",
    "started_at",
    "finished_at",
    "exit_code",
    "output_dir",
    "note",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def clean_complete(output: Path, expected_snr: list[float]) -> tuple[bool, str]:
    csv_path = output / "clean_metrics.csv"
    manifest_path = output / "clean_manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        return False, "clean CSV or manifest missing"
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"cannot parse clean artifacts: {exc}"
    actual_snr = [float(row["snr_db"]) for row in rows]
    if actual_snr != expected_snr:
        return False, "clean SNR grid is incomplete"
    if len(manifest.get("rows", [])) != len(expected_snr):
        return False, "clean manifest row count is incomplete"
    if any(int(row["channel_repeats"]) != 3 for row in rows):
        return False, "expected three channel repeats"
    return True, "validated complete"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--device", default="cuda")
    command.add_argument("--dry-run", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    analysis = REPO_ROOT / "outputs" / "factorial" / "factorial_analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    registry_path = analysis / "clean_evaluation_registry.csv"
    jobs = [(cell, seed) for seed in SEEDS for cell in CELLS]
    rows = [
        {
            "cell": cell.upper(),
            "seed": seed,
            "status": "pending",
            "started_at": "",
            "finished_at": "",
            "exit_code": "",
            "output_dir": str(
                (REPO_ROOT / "outputs" / "factorial" / f"{cell}_seed{seed}").resolve()
            ),
            "note": "",
        }
        for cell, seed in jobs
    ]
    write_csv(registry_path, rows)
    for index, (cell, seed) in enumerate(jobs):
        row = rows[index]
        output = Path(row["output_dir"])
        config_path = output / "resolved_config.json"
        checkpoint_path = output / "checkpoint_best.pt"
        if not config_path.is_file() or not checkpoint_path.is_file():
            row.update(status="blocked", finished_at=now(), note="training artifacts missing")
            write_csv(registry_path, rows)
            return 2
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_snr = [float(value) for value in config["evaluation"]["snr_db"]]
        complete, reason = clean_complete(output, expected_snr)
        if complete:
            row.update(status="skipped_complete", finished_at=now(), note=reason)
            write_csv(registry_path, rows)
            continue
        command = [
            sys.executable,
            str(REPO_ROOT / "run_semantic.py"),
            "clean",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output),
            "--device",
            args.device,
        ]
        if args.dry_run:
            row.update(status="dry_run", note=subprocess.list2cmdline(command))
            write_csv(registry_path, rows)
            continue
        row.update(status="running", started_at=now(), note="")
        write_csv(registry_path, rows)
        print(
            f"[{index + 1}/{len(jobs)}] evaluating {cell.upper()} seed {seed}",
            flush=True,
        )
        with (output / "clean_evaluation.stdout.log").open(
            "w", encoding="utf-8"
        ) as stdout, (output / "clean_evaluation.stderr.log").open(
            "w", encoding="utf-8"
        ) as stderr:
            result = subprocess.run(
                command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False
            )
        row.update(exit_code=result.returncode, finished_at=now())
        if result.returncode != 0:
            row.update(status="failed", note="subprocess failed; not retried")
            write_csv(registry_path, rows)
            return result.returncode or 1
        complete, reason = clean_complete(output, expected_snr)
        if not complete:
            row.update(status="failed_validation", note=reason)
            write_csv(registry_path, rows)
            return 3
        row.update(status="completed", note=reason)
        write_csv(registry_path, rows)
    print(f"clean evaluation registry: {registry_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
