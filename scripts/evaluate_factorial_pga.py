"""Run a bounded, exploratory 10 dB PGA audit for all factorial checkpoints."""

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
FACTORIAL = REPO_ROOT / "outputs" / "factorial"
ANALYSIS = FACTORIAL / "factorial_analysis"
ATTACK_ROOT = ANALYSIS / "pga_10db_128"
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


def complete(output: Path) -> tuple[bool, str]:
    sample_path = output / "attack_samples.csv"
    summary_path = output / "attack_summary.csv"
    manifest_path = output / "attack_manifest.json"
    if not all(path.is_file() for path in (sample_path, summary_path, manifest_path)):
        return False, "attack artifacts missing"
    try:
        with sample_path.open("r", encoding="utf-8", newline="") as stream:
            samples = list(csv.DictReader(stream))
        with summary_path.open("r", encoding="utf-8", newline="") as stream:
            summary = list(csv.DictReader(stream))
    except (OSError, ValueError) as exc:
        return False, f"cannot parse attack artifacts: {exc}"
    if len(samples) != 128 or len(summary) != 1:
        return False, "expected 128 samples and one 10 dB summary row"
    if summary[0]["attack"] != "pga" or float(summary[0]["snr_db"]) != 10.0:
        return False, "unexpected attack or SNR"
    return True, "validated exploratory audit"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--device", default="cuda")
    command.add_argument("--dry-run", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    ATTACK_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [(cell, seed) for seed in SEEDS for cell in CELLS]
    rows = [
        {
            "cell": cell.upper(),
            "seed": seed,
            "status": "pending",
            "started_at": "",
            "finished_at": "",
            "exit_code": "",
            "output_dir": str((ATTACK_ROOT / f"{cell}_seed{seed}").resolve()),
            "note": "",
        }
        for cell, seed in jobs
    ]
    registry = ATTACK_ROOT / "registry.csv"
    write_csv(registry, rows)
    for index, (cell, seed) in enumerate(jobs):
        row = rows[index]
        training_output = FACTORIAL / f"{cell}_seed{seed}"
        checkpoint = training_output / "checkpoint_best.pt"
        source_config = training_output / "resolved_config.json"
        output = Path(row["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        config = json.loads(source_config.read_text(encoding="utf-8"))
        config["evaluation"]["snr_db"] = [10]
        config["attacks"]["max_samples"] = 128
        config["attacks"]["pga"]["max_steps"] = 2000
        config["attacks"]["pga"]["refine_steps"] = 0
        config_path = output / "audit_config.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        done, reason = complete(output)
        if done:
            row.update(status="skipped_complete", finished_at=now(), note=reason)
            write_csv(registry, rows)
            continue
        command = [
            sys.executable,
            str(REPO_ROOT / "run_semantic.py"),
            "attack",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--attacks",
            "pga",
            "--output",
            str(output),
            "--device",
            args.device,
        ]
        if args.dry_run:
            row.update(status="dry_run", note=subprocess.list2cmdline(command))
            write_csv(registry, rows)
            continue
        row.update(status="running", started_at=now(), note="")
        write_csv(registry, rows)
        print(f"[{index + 1}/12] PGA {cell.upper()} seed {seed}", flush=True)
        with (output / "pga.stdout.log").open("w", encoding="utf-8") as stdout, (
            output / "pga.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False
            )
        row.update(exit_code=result.returncode, finished_at=now())
        if result.returncode != 0:
            row.update(status="failed", note="subprocess failed; not retried")
            write_csv(registry, rows)
            return result.returncode or 1
        done, reason = complete(output)
        if not done:
            row.update(status="failed_validation", note=reason)
            write_csv(registry, rows)
            return 3
        row.update(status="completed", note=reason)
        write_csv(registry, rows)
    print(f"PGA registry: {registry}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
