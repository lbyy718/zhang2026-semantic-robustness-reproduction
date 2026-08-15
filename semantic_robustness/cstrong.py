"""Training utilities for the C-strong three-arm control experiment.

This module is intentionally separate from the historical C-small trainer.
It preserves the old artifacts while enforcing the locked CS0/CS1/CSJ design:
the same ResNet-18 bottleneck architecture, data split and optimizer, with only
training AWGN or the latent failure-score Jacobian penalty changed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from .channel import AWGNChannel, NoiselessChannel
from .config import resolve_relative, without_runtime_fields
from .metrics import classification_failure_score, classification_margin
from .model import DeepJSCCResNetClassifier
from .runtime import build_model, choose_device, set_seed


CSTRONG_SCHEMA_VERSION = "cstrong_v1"
EXPECTED_TRAIN_SAMPLES = 45_000
EXPECTED_VALIDATION_SAMPLES = 5_000
TEST_NOISE_SEEDS = (122026, 122027, 122028)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        without_runtime_fields(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def centered_logit_rms(logits: Tensor) -> Tensor:
    """Per-sample centered-logit RMS, invariant to a shared logit offset."""
    centered = logits - logits.mean(dim=1, keepdim=True)
    return centered.square().mean(dim=1).sqrt()


def latent_failure_jacobian_penalty(
    logits: Tensor,
    latent: Tensor,
    labels: Tensor,
    *,
    create_graph: bool,
) -> tuple[Tensor, Tensor]:
    """Return mean ||grad_z s||_2^2 / N and its per-sample values.

    ``s=max_{k!=y} f_k-f_y`` is the same failure score used by the PGA
    diagnostics.  The warm-up epoch requests ``create_graph=False`` only to
    calibrate lambda; regularized epochs request ``True`` so the penalty trains
    the classifier parameters.
    """
    if latent.ndim != 2 or logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected latent/logits/labels shaped [B,N]/[B,K]/[B].")
    if latent.shape[0] != logits.shape[0] or labels.shape[0] != logits.shape[0]:
        raise ValueError("latent, logits and labels must have the same batch size.")
    score = classification_failure_score(labels.long(), logits)
    gradient = torch.autograd.grad(
        score.sum(),
        latent,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]
    per_sample = gradient.flatten(start_dim=1).square().sum(dim=1) / latent.shape[1]
    return per_sample.mean(), per_sample


def _split_indices(length: int, validation_samples: int, split_seed: int) -> tuple[list[int], list[int]]:
    if not 0 < validation_samples < length:
        raise ValueError("validation_samples must lie strictly inside the train set.")
    permutation = torch.randperm(
        length, generator=torch.Generator(device="cpu").manual_seed(split_seed)
    ).tolist()
    train_count = length - validation_samples
    return permutation[:train_count], permutation[train_count:]


def _indices_sha256(train_indices: Iterable[int], validation_indices: Iterable[int]) -> str:
    digest = hashlib.sha256()
    digest.update(array("I", (int(value) for value in train_indices)).tobytes())
    digest.update(array("I", (int(value) for value in validation_indices)).tobytes())
    return digest.hexdigest()


def build_cstrong_datasets(
    config: dict[str, Any],
) -> tuple[Dataset[Any], Dataset[Any], Dataset[Any], str]:
    """Build augmented train and unaugmented validation/test CIFAR-10 views."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover
        raise ImportError("C-strong training requires torchvision.") from exc

    data = config["data"]
    if str(data.get("validation_source", "train_holdout")) != "train_holdout":
        raise ValueError("C-strong requires a train_holdout validation split.")
    if str(data.get("augmentation", "cifar_standard")) != "cifar_standard":
        raise ValueError("C-strong requires data.augmentation='cifar_standard'.")
    root = resolve_relative(config, data.get("root", "../../data/cifar10"))
    download = bool(data.get("download", False))
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
    validation_samples = int(data.get("validation_samples", EXPECTED_VALIDATION_SAMPLES))
    split_seed = int(data.get("split_seed", 2026))
    train_indices, validation_indices = _split_indices(
        len(augmented), validation_samples, split_seed
    )
    if len(train_indices) != EXPECTED_TRAIN_SAMPLES or len(validation_indices) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError(
            "The locked C-strong protocol requires a 45,000/5,000 split."
        )
    return (
        Subset(augmented, train_indices),
        Subset(unaugmented, validation_indices),
        test,
        _indices_sha256(train_indices, validation_indices),
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_cstrong_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    generator: torch.Generator,
    pin_memory: bool,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _fixed_awgn(
    channel: AWGNChannel,
    latent: Tensor,
    snr_db: float,
    generator: torch.Generator,
) -> Tensor:
    standard = torch.randn(
        latent.shape, dtype=torch.float32, device="cpu", generator=generator
    ).to(latent.device, non_blocking=True)
    variance = channel.noise_variance(snr_db, latent.float())
    return channel(
        latent,
        snr_db,
        noise=(standard.reshape_as(latent) * variance.sqrt()).to(latent.dtype),
    )


@torch.no_grad()
def evaluate_classifier(
    model: DeepJSCCResNetClassifier,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    noisy: bool,
    snr_db: float = 10.0,
    noise_seed: int = 0,
    amp_enabled: bool = True,
) -> dict[str, float]:
    model.eval()
    channel = AWGNChannel().to(device)
    generator = torch.Generator(device="cpu").manual_seed(int(noise_seed))
    loss_sum = accuracy_sum = margin_sum = rms_sum = 0.0
    latent_sum: Tensor | None = None
    latent_square_sum: Tensor | None = None
    hidden_active = 0
    hidden_total = 0
    prediction_counts = torch.zeros(10, dtype=torch.long, device=device)
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            latent = model.encode(images)
        received = _fixed_awgn(channel, latent, snr_db, generator) if noisy else latent
        logits = model.decode(received.float())
        preactivation = model.decoder.network[0](received.float())
        losses = nn.functional.cross_entropy(logits, labels, reduction="none")
        margins = classification_margin(labels, logits)
        predictions = logits.argmax(dim=1)
        latent_float = latent.float()
        batch_sum = latent_float.sum(dim=0)
        batch_square_sum = latent_float.square().sum(dim=0)
        latent_sum = batch_sum if latent_sum is None else latent_sum + batch_sum
        latent_square_sum = (
            batch_square_sum
            if latent_square_sum is None
            else latent_square_sum + batch_square_sum
        )
        hidden_active += int((preactivation > 0).sum())
        hidden_total += preactivation.numel()
        prediction_counts += torch.bincount(predictions, minlength=10)
        loss_sum += float(losses.sum())
        accuracy_sum += float(predictions.eq(labels).sum())
        margin_sum += float(margins.sum())
        rms_sum += float(centered_logit_rms(logits).sum())
        count += images.shape[0]
    if count == 0:
        raise RuntimeError("Classifier evaluation received no samples.")
    assert latent_sum is not None and latent_square_sum is not None
    latent_mean = latent_sum / count
    latent_variance = (latent_square_sum / count - latent_mean.square()).clamp_min(0)
    return {
        "cross_entropy": loss_sum / count,
        "accuracy": accuracy_sum / count,
        "mean_margin": margin_sum / count,
        "mean_centered_logit_rms": rms_sum / count,
        "latent_coordinate_std_mean": float(latent_variance.sqrt().mean()),
        "hidden_relu_active_fraction": hidden_active / hidden_total,
        "unique_predicted_classes": float((prediction_counts > 0).sum()),
        "dominant_prediction_fraction": float(prediction_counts.max() / count),
        "samples": float(count),
    }


def _read_training_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(stream):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"epoch", "is_best", "epochs_without_improvement"}:
                    parsed[key] = int(value)
                elif key in {"regularizer", "arm"}:
                    parsed[key] = value
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
        return rows


def _rng_state(train_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader_generator": train_generator.get_state(),
    }


def _restore_rng_state(state: dict[str, Any], train_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    train_generator.set_state(state["train_loader_generator"])


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    epoch: int,
    best_epoch: int,
    best_validation_ce: float,
    epochs_without_improvement: int,
    jacobian_lambda: float | None,
    config: dict[str, Any],
    split_hash: str,
    train_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema_version": CSTRONG_SCHEMA_VERSION,
        "scope": "C-strong CIFAR-10 semantic bottleneck classifier",
        "architecture": "resnet18_bottleneck",
        "objective": "classification",
        "experiment_cell": str(config.get("experiment_cell", "")).upper(),
        "seed": int(config["seed"]),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_10db_ce": best_validation_ce,
        "epochs_without_improvement": epochs_without_improvement,
        "jacobian_lambda": jacobian_lambda,
        "split_sha256": split_hash,
        "config_sha256": canonical_config_sha256(config),
        "config": without_runtime_fields(config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": _rng_state(train_generator),
    }


def _load_torch(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - old torch compatibility
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return payload


def _clean_curve(
    model: DeepJSCCResNetClassifier,
    loader: DataLoader[Any],
    device: torch.device,
    amp_enabled: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    noiseless = evaluate_classifier(
        model, loader, device, noisy=False, amp_enabled=amp_enabled
    )
    rows.append({"snr_db": "noiseless", "repeat": -1, **noiseless})
    for snr_db in range(0, 21, 2):
        for repeat, noise_seed in enumerate(TEST_NOISE_SEEDS):
            metrics = evaluate_classifier(
                model,
                loader,
                device,
                noisy=True,
                snr_db=float(snr_db),
                noise_seed=noise_seed,
                amp_enabled=amp_enabled,
            )
            rows.append({"snr_db": snr_db, "repeat": repeat, **metrics})
    return rows


def validate_cstrong_training_config(config: dict[str, Any]) -> None:
    model = config["model"]
    training = config["training"]
    data = config["data"]
    arm = str(config.get("experiment_cell", "")).upper()
    if arm not in {"CS0", "CS1", "CSJ"}:
        raise ValueError("experiment_cell must be CS0, CS1 or CSJ.")
    if str(config.get("objective", "")).lower() != "classification":
        raise ValueError("C-strong is classification-only.")
    if str(model.get("architecture", "")).lower() != "resnet18_bottleneck":
        raise ValueError("C-strong requires model.architecture=resnet18_bottleneck.")
    phase = str(config.get("experiment_phase", "formal_three_arm")).lower()
    diagnostic = phase == "trainability_diagnostic"
    low_lr_protocol = phase in {"trainability_diagnostic", "formal_low_lr"}
    if phase not in {"formal_three_arm", "trainability_diagnostic", "formal_low_lr"}:
        raise ValueError(
            "experiment_phase must be 'formal_three_arm', "
            "'trainability_diagnostic' or 'formal_low_lr'."
        )
    if low_lr_protocol and arm not in {"CS0", "CSJ"}:
        raise ValueError("The low-learning-rate protocol is restricted to CS0 or CSJ.")
    locked = {
        "batch_size": (int(data.get("batch_size", 0)), 64),
        "validation_samples": (int(data.get("validation_samples", 0)), 5000),
        "epochs": (int(training.get("epochs", 0)), 200),
        "momentum": (float(training.get("momentum", 0)), 0.9),
        "weight_decay": (float(training.get("weight_decay", 0)), 5e-4),
    }
    locked["learning_rate"] = (
        float(training.get("learning_rate", 0)),
        0.01 if low_lr_protocol else 0.05,
    )
    locked["patience"] = (
        int(training.get("early_stopping_patience", 0)),
        200 if low_lr_protocol else 30,
    )
    mismatched = {name: values for name, values in locked.items() if values[0] != values[1]}
    if mismatched:
        raise ValueError(f"C-strong locked protocol mismatch: {mismatched}")
    if str(training.get("optimizer", "")).lower() != "sgd":
        raise ValueError("C-strong requires SGD.")
    if str(training.get("scheduler", "")).lower() != "cosine":
        raise ValueError("C-strong requires cosine scheduling.")
    regularizer = str(training.get("regularizer", "none")).lower()
    expected = {
        "CS0": (False, "none"),
        "CS1": (True, "none"),
        "CSJ": (False, "latent_failure_jacobian"),
    }[arm]
    actual = (bool(training.get("channel_noise", False)), regularizer)
    if actual != expected:
        raise ValueError(f"{arm} requires channel_noise/regularizer={expected}, got {actual}.")
    if float(training.get("snr_db", 10.0)) != 10.0:
        raise ValueError("C-strong training SNR is locked to 10 dB.")
    if low_lr_protocol and float(training.get("early_stopping_min_delta", 0.0)) != 0.0:
        raise ValueError("The low-learning-rate protocol disables early stopping min_delta.")


def train_cstrong(
    config: dict[str, Any],
    *,
    output: Path,
    device_name: str = "auto",
    workers_override: int | None = None,
    epochs_override: int | None = None,
    test_max_samples: int | None = None,
    resume: bool = False,
) -> Path:
    """Train one locked C-strong arm and atomically write all artifacts."""
    validate_cstrong_training_config(config)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    checkpoint_last_path = output / "checkpoint_last.pt"
    checkpoint_best_path = output / "checkpoint_best.pt"
    log_path = output / "training_log.csv"
    existing = [p.name for p in (manifest_path, checkpoint_last_path, checkpoint_best_path, log_path) if p.exists()]
    if existing and not resume:
        raise FileExistsError(
            f"C-strong output contains {existing}; use --resume only for an interrupted job."
        )

    seed = int(config.get("seed", 2026))
    set_seed(seed, bool(config.get("deterministic", True)))
    device = choose_device(device_name)
    training = config["training"]
    data = config["data"]
    workers = int(data.get("num_workers", 0) if workers_override is None else workers_override)
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    train_data, validation_data, test_data, split_hash = build_cstrong_datasets(config)
    full_test_samples = len(test_data)
    if test_max_samples is not None:
        if not 0 < test_max_samples <= full_test_samples:
            raise ValueError("test_max_samples must be in [1, len(test_data)].")
        test_data = Subset(test_data, range(test_max_samples))
    train_generator = torch.Generator(device="cpu").manual_seed(seed)
    validation_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    test_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    train_loader = make_cstrong_loader(
        train_data,
        batch_size=int(data["batch_size"]),
        shuffle=True,
        workers=workers,
        generator=train_generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = make_cstrong_loader(
        validation_data,
        batch_size=int(data.get("evaluation_batch_size", 256)),
        shuffle=False,
        workers=workers,
        generator=validation_generator,
        pin_memory=device.type == "cuda",
    )
    test_loader = make_cstrong_loader(
        test_data,
        batch_size=int(data.get("evaluation_batch_size", 256)),
        shuffle=False,
        workers=workers,
        generator=test_generator,
        pin_memory=device.type == "cuda",
    )

    built = build_model(config)
    if not isinstance(built, DeepJSCCResNetClassifier):
        raise TypeError("C-strong config did not build DeepJSCCResNetClassifier.")
    model = built.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["learning_rate"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
    )
    requested_epochs = int(training["epochs"] if epochs_override is None else epochs_override)
    if requested_epochs <= 0 or requested_epochs > int(training["epochs"]):
        raise ValueError("epochs_override must be in [1, configured epochs].")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=requested_epochs
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    arm = str(config["experiment_cell"]).upper()
    regularizer = str(training.get("regularizer", "none")).lower()
    use_train_noise = bool(training.get("channel_noise", False))
    training_channel: nn.Module = AWGNChannel().to(device) if use_train_noise else NoiselessChannel().to(device)
    validation_snr = float(training.get("validation_snr_db", 10.0))
    validation_noise_seed = int(training.get("validation_noise_seed", 72026))
    patience = int(training["early_stopping_patience"])
    min_delta = float(training.get("early_stopping_min_delta", 0.0))
    target_ratio = float(training.get("jacobian_target_ratio", 0.1))
    config_hash = canonical_config_sha256(config)
    start_epoch = 0
    best_epoch = 0
    best_validation_ce = float("inf")
    epochs_without_improvement = 0
    jacobian_lambda: float | None = None
    rows: list[dict[str, Any]] = []

    if resume:
        if not checkpoint_last_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Interrupted C-strong job lacks checkpoint_last.pt or manifest.json.")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("status") not in {"running", "interrupted"}:
            raise RuntimeError(
                f"Only running/interrupted jobs may resume, found {previous_manifest.get('status')!r}."
            )
        checkpoint = _load_torch(checkpoint_last_path, device)
        if checkpoint.get("config_sha256") != config_hash or checkpoint.get("split_sha256") != split_hash:
            raise RuntimeError("Resume checkpoint does not match config or data split.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_ce = float(checkpoint["best_validation_10db_ce"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        jacobian_lambda = checkpoint.get("jacobian_lambda")
        _restore_rng_state(checkpoint["rng_state"], train_generator)
        rows = _read_training_log(log_path)
        if len(rows) != start_epoch:
            raise RuntimeError("training_log.csv and checkpoint_last.pt disagree.")

    manifest: dict[str, Any] = {
        "schema_version": CSTRONG_SCHEMA_VERSION,
        "scope": "C-strong classifier training",
        "status": "running",
        "started_at": (
            json.loads(manifest_path.read_text(encoding="utf-8")).get("started_at", utc_now())
            if resume and manifest_path.exists()
            else utc_now()
        ),
        "resumed_at": utc_now() if resume else None,
        "pid": os.getpid(),
        "arm": arm,
        "experiment_phase": str(
            config.get("experiment_phase", "formal_three_arm")
        ),
        "seed": seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp_enabled": amp_enabled,
        "config_sha256": config_hash,
        "split_seed": int(data.get("split_seed", 2026)),
        "split_sha256": split_hash,
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "test_samples": full_test_samples,
        "evaluated_test_samples": len(test_data),
        "selection_metric": "validation_10db_cross_entropy",
        "architecture": without_runtime_fields(config)["model"],
        "augmentation": data.get("augmentation"),
        "optimizer": {
            "name": "SGD",
            "learning_rate": float(training["learning_rate"]),
            "momentum": float(training["momentum"]),
            "weight_decay": float(training["weight_decay"]),
        },
        "scheduler": {"name": "cosine", "t_max": requested_epochs},
        "early_stopping": {"patience": patience, "min_delta": min_delta},
        "training_channel_noise": use_train_noise,
        "training_snr_db": float(training.get("snr_db", 10.0)),
        "regularizer": regularizer,
        "jacobian_target_ratio": target_ratio if regularizer != "none" else None,
        "jacobian_lambda": jacobian_lambda,
        "requested_epochs": requested_epochs,
        "configured_epochs": int(training["epochs"]),
        "checkpoint_policy": "best plus atomic recovery checkpoint every epoch",
        "workers": workers,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "command": sys.argv,
        "config": without_runtime_fields(config),
    }
    atomic_json(manifest_path, manifest)
    start_time = time.time()

    try:
        for epoch in range(start_epoch + 1, requested_epochs + 1):
            model.train()
            train_ce_sum = train_correct = train_margin_sum = 0.0
            train_penalty_sum = train_count = 0
            learning_rate = float(optimizer.param_groups[0]["lr"])
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    latent = model.encode(images)
                    received = training_channel(
                        latent, float(training.get("snr_db", 10.0))
                    )
                received = received.float()
                logits = model.decode(received)
                ce = nn.functional.cross_entropy(logits, labels)
                penalty = torch.zeros((), device=device)
                if regularizer == "latent_failure_jacobian":
                    penalty, _ = latent_failure_jacobian_penalty(
                        logits,
                        received,
                        labels,
                        create_graph=epoch >= 2,
                    )
                total_loss = ce
                if epoch >= 2 and regularizer == "latent_failure_jacobian":
                    if jacobian_lambda is None:
                        raise RuntimeError("CSJ lambda was not calibrated after warm-up.")
                    total_loss = ce + jacobian_lambda * penalty
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_count = images.shape[0]
                train_ce_sum += float(ce.detach()) * batch_count
                train_penalty_sum += float(penalty.detach()) * batch_count
                train_correct += int(logits.detach().argmax(dim=1).eq(labels).sum())
                train_margin_sum += float(classification_margin(labels, logits.detach()).sum())
                train_count += batch_count

            train_ce = train_ce_sum / train_count
            train_penalty = train_penalty_sum / train_count
            if epoch == 1 and regularizer == "latent_failure_jacobian":
                if not math.isfinite(train_penalty) or train_penalty <= 0:
                    raise RuntimeError(f"Cannot calibrate Jacobian lambda from penalty={train_penalty}.")
                jacobian_lambda = target_ratio * train_ce / train_penalty
                if not math.isfinite(jacobian_lambda) or jacobian_lambda <= 0:
                    raise RuntimeError(f"Invalid calibrated Jacobian lambda={jacobian_lambda}.")

            validation_noiseless = evaluate_classifier(
                model,
                validation_loader,
                device,
                noisy=False,
                amp_enabled=amp_enabled,
            )
            validation_10db = evaluate_classifier(
                model,
                validation_loader,
                device,
                noisy=True,
                snr_db=validation_snr,
                noise_seed=validation_noise_seed,
                amp_enabled=amp_enabled,
            )
            candidate_ce = validation_10db["cross_entropy"]
            is_best = candidate_ce < best_validation_ce
            meaningful_improvement = candidate_ce < best_validation_ce - min_delta
            if is_best:
                best_validation_ce = candidate_ce
                best_epoch = epoch
            if meaningful_improvement:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            row: dict[str, Any] = {
                "arm": arm,
                "regularizer": regularizer,
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_cross_entropy": train_ce,
                "train_accuracy": train_correct / train_count,
                "train_mean_margin": train_margin_sum / train_count,
                "train_jacobian_penalty": train_penalty,
                "jacobian_lambda": jacobian_lambda or 0.0,
                "validation_noiseless_cross_entropy": validation_noiseless["cross_entropy"],
                "validation_noiseless_accuracy": validation_noiseless["accuracy"],
                "validation_noiseless_mean_margin": validation_noiseless["mean_margin"],
                "validation_noiseless_centered_logit_rms": validation_noiseless["mean_centered_logit_rms"],
                "validation_noiseless_latent_coordinate_std_mean": validation_noiseless["latent_coordinate_std_mean"],
                "validation_noiseless_hidden_relu_active_fraction": validation_noiseless["hidden_relu_active_fraction"],
                "validation_noiseless_unique_predicted_classes": validation_noiseless["unique_predicted_classes"],
                "validation_noiseless_dominant_prediction_fraction": validation_noiseless["dominant_prediction_fraction"],
                "validation_10db_cross_entropy": validation_10db["cross_entropy"],
                "validation_10db_accuracy": validation_10db["accuracy"],
                "validation_10db_mean_margin": validation_10db["mean_margin"],
                "validation_10db_centered_logit_rms": validation_10db["mean_centered_logit_rms"],
                "train_validation_accuracy_gap": train_correct / train_count - validation_10db["accuracy"],
                "is_best": int(is_best),
                "epochs_without_improvement": epochs_without_improvement,
                "elapsed_seconds": time.time() - start_time,
            }
            numeric = [value for value in row.values() if isinstance(value, float)]
            if not all(math.isfinite(value) for value in numeric):
                raise FloatingPointError(f"Non-finite metric at epoch {epoch}: {row}")
            rows.append(row)
            atomic_csv(log_path, rows)
            scheduler.step()
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_ce=best_validation_ce,
                epochs_without_improvement=epochs_without_improvement,
                jacobian_lambda=jacobian_lambda,
                config=config,
                split_hash=split_hash,
                train_generator=train_generator,
            )
            atomic_torch_save(checkpoint_last_path, payload)
            if is_best:
                atomic_torch_save(checkpoint_best_path, payload)
            manifest.update(
                completed_epochs=epoch,
                best_epoch=best_epoch,
                best_validation_10db_ce=best_validation_ce,
                epochs_without_improvement=epochs_without_improvement,
                jacobian_lambda=jacobian_lambda,
                elapsed_seconds=time.time() - start_time,
            )
            atomic_json(manifest_path, manifest)
            if epoch == 1 or epoch % 10 == 0 or epoch == requested_epochs:
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "epoch": epoch,
                            "train_acc": round(row["train_accuracy"], 4),
                            "val10_acc": round(row["validation_10db_accuracy"], 4),
                            "val10_ce": round(row["validation_10db_cross_entropy"], 5),
                            "best_epoch": best_epoch,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if epochs_without_improvement >= patience:
                break

        best = _load_torch(checkpoint_best_path, device)
        model.load_state_dict(best["model_state"])
        model.eval()
        curve_rows = _clean_curve(model, test_loader, device, amp_enabled)
        atomic_csv(output / "test_clean_curve.csv", curve_rows)
        ten_db_rows = [row for row in curve_rows if row["snr_db"] == 10]
        noiseless_row = next(row for row in curve_rows if row["snr_db"] == "noiseless")
        test_10db_accuracy = sum(float(row["accuracy"]) for row in ten_db_rows) / len(ten_db_rows)
        test_10db_ce = sum(float(row["cross_entropy"]) for row in ten_db_rows) / len(ten_db_rows)
        minimum_accuracy = 0.72 if arm == "CSJ" else 0.75
        training_gate = (
            test_10db_accuracy >= minimum_accuracy
            if len(test_data) == full_test_samples
            else None
        )
        manifest.update(
            status="completed",
            finished_at=utc_now(),
            completed_epochs=len(rows),
            stopped_early=len(rows) < requested_epochs,
            best_epoch=best_epoch,
            best_validation_10db_ce=best_validation_ce,
            jacobian_lambda=jacobian_lambda,
            test_noiseless_accuracy=float(noiseless_row["accuracy"]),
            test_noiseless_cross_entropy=float(noiseless_row["cross_entropy"]),
            test_10db_accuracy=test_10db_accuracy,
            test_10db_cross_entropy=test_10db_ce,
            minimum_test_10db_accuracy=minimum_accuracy,
            individual_training_quality_gate_passed=training_gate,
            checkpoint_best_sha256=file_sha256(checkpoint_best_path),
            training_log_sha256=file_sha256(log_path),
            test_clean_curve_sha256=file_sha256(output / "test_clean_curve.csv"),
            elapsed_seconds=time.time() - start_time,
        )
        atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "status": "completed",
                    "epochs": len(rows),
                    "best_epoch": best_epoch,
                    "test10_accuracy": round(test_10db_accuracy, 4),
                    "training_gate": training_gate,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return output
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            finished_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.time() - start_time,
        )
        atomic_json(manifest_path, manifest)
        raise
