"""Train one CS0/CS1/CSJ C-strong arm from a locked JSON config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_robustness.config import load_config  # noqa: E402
from semantic_robustness.cstrong import train_cstrong  # noqa: E402


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--device", default="auto")
    command.add_argument("--workers", type=int)
    command.add_argument("--epochs-override", type=int)
    command.add_argument("--test-max-samples", type=int)
    command.add_argument("--resume", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    config = load_config(args.config.resolve())
    train_cstrong(
        config,
        output=args.output,
        device_name=args.device,
        workers_override=args.workers,
        epochs_override=args.epochs_override,
        test_max_samples=args.test_max_samples,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
