"""Train the frozen CIFAR-10 ResNet-18 semantic evaluator.

The CIFAR-10 test set is evaluated only after validation-based checkpoint
selection.  The default 90% original-image test-accuracy gate must pass before
the resulting checkpoint is considered qualified for mechanism experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_robustness.semantic_evaluator import (  # noqa: E402
    CIFAR10_MEAN,
    CIFAR10_STD,
    SEMANTIC_EVALUATOR_ARCHITECTURE,
    build_cifar10_resnet18,
    load_frozen_cifar10_resnet18,
)


SPLIT_SEED = 2026
TRAIN_SAMPLES = 45_000
VALIDATION_SAMPLES = 5_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty training log.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def split_indices(length: int) -> tuple[list[int], list[int]]:
    if length != TRAIN_SAMPLES + VALIDATION_SAMPLES:
        raise ValueError(f"Expected 50,000 CIFAR-10 train images, found {length}.")
    permutation = torch.randperm(
        length, generator=torch.Generator().manual_seed(SPLIT_SEED)
    ).tolist()
    return permutation[:TRAIN_SAMPLES], permutation[TRAIN_SAMPLES:]


def indices_sha256(train_indices: list[int], validation_indices: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(array("I", train_indices).tobytes())
    digest.update(array("I", validation_indices).tobytes())
    return digest.hexdigest()


def build_datasets(
    root: Path, *, download: bool
) -> tuple[Dataset[Any], Dataset[Any], Dataset[Any], str]:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError("Training the semantic evaluator requires torchvision.") from exc

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    evaluation_transform = transforms.ToTensor()
    augmented = datasets.CIFAR10(
        root=str(root), train=True, transform=train_transform, download=download
    )
    unaugmented = datasets.CIFAR10(
        root=str(root), train=True, transform=evaluation_transform, download=download
    )
    test = datasets.CIFAR10(
        root=str(root), train=False, transform=evaluation_transform, download=download
    )
    train_indices, validation_indices = split_indices(len(augmented))
    return (
        Subset(augmented, train_indices),
        Subset(unaugmented, validation_indices),
        test,
        indices_sha256(train_indices, validation_indices),
    )


def make_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    amp_enabled: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            loss_sum += float(loss.detach()) * images.shape[0]
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            count += images.shape[0]
    if count == 0:
        raise RuntimeError("Evaluation received no samples.")
    return loss_sum / count, correct / count


def checkpoint_payload(
    model: nn.Module,
    *,
    epoch: int,
    validation_accuracy: float,
    validation_loss: float,
    seed: int,
    split_hash: str,
) -> dict[str, Any]:
    return {
        "scope": "frozen CIFAR-10 semantic evaluator",
        "architecture": SEMANTIC_EVALUATOR_ARCHITECTURE,
        "num_classes": 10,
        "normalization": {"mean": CIFAR10_MEAN, "std": CIFAR10_STD},
        "epoch": epoch,
        "best_validation_accuracy": validation_accuracy,
        "validation_loss_at_best": validation_loss,
        "seed": seed,
        "split_seed": SPLIT_SEED,
        "split_sha256": split_hash,
        "model_state": model.state_dict(),
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "cifar10")
    command.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "factorial"
            / "factorial_mechanism_v1"
            / "frozen_resnet18"
        ),
    )
    command.add_argument("--epochs", type=int, default=200)
    command.add_argument("--patience", type=int, default=30)
    command.add_argument("--batch-size", type=int, default=128)
    command.add_argument("--evaluation-batch-size", type=int, default=256)
    command.add_argument("--workers", type=int, default=4)
    command.add_argument("--device", default="auto")
    command.add_argument("--seed", type=int, default=2026)
    command.add_argument("--learning-rate", type=float, default=0.1)
    command.add_argument("--momentum", type=float, default=0.9)
    command.add_argument("--weight-decay", type=float, default=5e-4)
    command.add_argument("--minimum-test-accuracy", type=float, default=0.90)
    command.add_argument("--download", action="store_true")
    command.add_argument("--no-amp", action="store_false", dest="amp")
    command.add_argument("--overwrite", action="store_true")
    command.set_defaults(amp=True)
    return command


def validate_arguments(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative.")
    if not 0.0 <= args.minimum_test_accuracy <= 1.0:
        raise ValueError("--minimum-test-accuracy must be a fraction in [0, 1].")


def train(args: argparse.Namespace) -> bool:
    validate_arguments(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = [
        output / "checkpoint_best.pt",
        output / "training_log.csv",
        output / "manifest.json",
    ]
    existing = [path.name for path in artifacts if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already contains {existing}; pass --overwrite to replace them."
        )

    set_seed(args.seed)
    device = choose_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    data_root = args.data_root.resolve()
    train_data, validation_data, test_data, split_hash = build_datasets(
        data_root, download=args.download
    )
    train_loader = make_loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = make_loader(
        validation_data,
        batch_size=args.evaluation_batch_size,
        shuffle=False,
        workers=args.workers,
        seed=args.seed + 1,
        pin_memory=device.type == "cuda",
    )
    test_loader = make_loader(
        test_data,
        batch_size=args.evaluation_batch_size,
        shuffle=False,
        workers=args.workers,
        seed=args.seed + 2,
        pin_memory=device.type == "cuda",
    )

    model = build_cifar10_resnet18().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old PyTorch fallback
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "qualified": False,
        "started_at": utc_now(),
        "scope": "frozen CIFAR-10 semantic evaluator",
        "architecture": SEMANTIC_EVALUATOR_ARCHITECTURE,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "seed": args.seed,
        "data_root": str(data_root),
        "download": bool(args.download),
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "test_samples": len(test_data),
        "split_seed": SPLIT_SEED,
        "split_sha256": split_hash,
        "selection_metric": "validation_accuracy",
        "minimum_test_accuracy": args.minimum_test_accuracy,
        "optimizer": {
            "name": "SGD",
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "schedule": "cosine",
        },
        "requested_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "workers": args.workers,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "command": sys.argv,
    }
    write_json(manifest_path, manifest)

    rows: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()
    checkpoint_path = output / "checkpoint_best.pt"
    try:
        for epoch in range(1, args.epochs + 1):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_loss, train_accuracy = run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, device
            )
            improved = validation_accuracy > best_accuracy
            if improved:
                best_accuracy = validation_accuracy
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_path,
                    checkpoint_payload(
                        model,
                        epoch=epoch,
                        validation_accuracy=validation_accuracy,
                        validation_loss=validation_loss,
                        seed=args.seed,
                        split_hash=split_hash,
                    ),
                )
            else:
                epochs_without_improvement += 1
            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "is_best": int(improved),
                "epochs_without_improvement": epochs_without_improvement,
                "elapsed_seconds": time.time() - start_time,
            }
            rows.append(row)
            write_csv(output / "training_log.csv", rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            scheduler.step()
            if epochs_without_improvement >= args.patience:
                break

        evaluator, checkpoint = load_frozen_cifar10_resnet18(
            checkpoint_path, device=device
        )
        test_loss, test_accuracy = run_epoch(evaluator, test_loader, device)
        qualified = test_accuracy >= args.minimum_test_accuracy
        checkpoint.update(
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            minimum_test_accuracy=args.minimum_test_accuracy,
            qualified=qualified,
        )
        save_checkpoint(checkpoint_path, checkpoint)
        manifest.update(
            status="completed" if qualified else "failed_quality_gate",
            qualified=qualified,
            finished_at=utc_now(),
            completed_epochs=len(rows),
            stopped_early=len(rows) < args.epochs,
            best_epoch=best_epoch,
            best_validation_accuracy=best_accuracy,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            elapsed_seconds=time.time() - start_time,
        )
        write_json(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
        return qualified
    except Exception as exc:
        manifest.update(
            status="failed",
            qualified=False,
            finished_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.time() - start_time,
        )
        write_json(manifest_path, manifest)
        raise


def main() -> int:
    args = parser().parse_args()
    return 0 if train(args) else 2


if __name__ == "__main__":
    raise SystemExit(main())
