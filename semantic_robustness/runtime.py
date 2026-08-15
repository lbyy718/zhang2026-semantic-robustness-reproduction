"""Training, clean evaluation, and adversarial evaluation runners."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, random_split

from .attacks import CWRegressionAttack, ProgressiveGradientAscent
from .channel import AWGNChannel, NoiselessChannel
from .config import resolve_relative, without_runtime_fields
from .data import (
    cifar10_datasets,
    limited_dataset,
    make_loader,
    unpack_batch,
)
from .metrics import (
    accuracy_per_sample,
    classification_failure_score,
    classification_margin,
    distortion_per_sample,
    mse_per_sample,
    psnr,
    quality_name,
    target_distortion,
)
from .model import DeepJSCC, DeepJSCCClassifier, DeepJSCCResNetClassifier

SemanticModel = DeepJSCC | DeepJSCCClassifier | DeepJSCCResNetClassifier


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _objective(config: dict[str, Any]) -> str:
    return str(config.get("objective", "reconstruction")).lower()


def build_model(config: dict[str, Any]) -> SemanticModel:
    model = config["model"]
    architecture = str(model.get("architecture", "small")).lower()
    if architecture == "resnet18_bottleneck":
        return DeepJSCCResNetClassifier(
            in_channels=int(model["in_channels"]),
            spatial_size=tuple(model.get("spatial_size", [32, 32])),
            latent_dim=int(model.get("latent_dim", 768)),
            feature_dim=int(model.get("feature_dim", 512)),
            classifier_hidden=int(model.get("classifier_hidden", 512)),
            num_classes=int(model.get("num_classes", 10)),
        )
    common = dict(
        in_channels=int(model["in_channels"]),
        channel_multiplier=int(model["channel_multiplier"]),
        spatial_size=tuple(model.get("spatial_size", [32, 32])),
        kernel_size=int(model.get("kernel_size", 3)),
        residual_kernel_size=int(model.get("residual_kernel_size", 3)),
    )
    if _objective(config) == "classification":
        return DeepJSCCClassifier(
            **common,
            classifier_hidden=int(model.get("classifier_hidden", 53)),
            num_classes=int(model.get("num_classes", 10)),
        )
    return DeepJSCC(**common)


def build_channel(config: dict[str, Any], *, training: bool) -> nn.Module:
    use_noise = True
    if training:
        use_noise = bool(config.get("training", {}).get("channel_noise", True))
    if not use_noise:
        return NoiselessChannel()
    return AWGNChannel(float(config.get("channel", {}).get("fading_gain", 1.0)))


def build_dataset(config: dict[str, Any], split: str) -> Dataset[Any]:
    if split not in {"train", "test"}:
        raise ValueError("CIFAR-10 split must be 'train' or 'test'.")
    data = config["data"]
    root = resolve_relative(config, data.get("root", "../data/cifar10"))
    train_data, test_data = cifar10_datasets(
        root, download=bool(data.get("download", False))
    )
    return train_data if split == "train" else test_data


def build_training_datasets(config: dict[str, Any]) -> tuple[Dataset[Any], Dataset[Any]]:
    """Return train/validation sets, preserving legacy test validation on request."""
    data = config["data"]
    source = str(data.get("validation_source", "test"))
    if source == "test":
        return build_dataset(config, "train"), build_dataset(config, "test")
    full_train = build_dataset(config, "train")
    validation_samples = int(data.get("validation_samples", 5000))
    if not 0 < validation_samples < len(full_train):
        raise ValueError("train_holdout requires 0 < validation_samples < train size.")
    split_seed = int(data.get("split_seed", 2026))
    return tuple(
        random_split(
            full_train,
            [len(full_train) - validation_samples, validation_samples],
            generator=torch.Generator().manual_seed(split_seed),
        )
    )  # type: ignore[return-value]


def _sample_train_snr(specification: Any, device: torch.device) -> float | Tensor:
    if isinstance(specification, (int, float)):
        return float(specification)
    if isinstance(specification, list) and len(specification) == 2:
        low, high = map(float, specification)
        return torch.empty((), device=device).uniform_(low, high)
    if isinstance(specification, dict):
        low = float(specification["min_db"])
        high = float(specification["max_db"])
        return torch.empty((), device=device).uniform_(low, high)
    raise ValueError("training.snr_db must be a number, [min,max], or min/max mapping.")


def _output_dir(config: dict[str, Any], override: str | Path | None = None) -> Path:
    if override is not None:
        path = Path(override).resolve()
    else:
        path = resolve_relative(config, config.get("output_dir", "../outputs/default"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_payload(
    model: SemanticModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "paper": "Zhang et al. 2026, Unanticipated Adversarial Robustness of Semantic Communication",
        "scope": "cifar10-image-only reproduction",
        "objective": _objective(config),
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": without_runtime_fields(config),
    }


def load_checkpoint(
    config: dict[str, Any], checkpoint_path: str | Path, device: torch.device
) -> tuple[SemanticModel, dict[str, Any]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def train(
    config: dict[str, Any],
    *,
    output_override: str | Path | None = None,
    device_name: str = "auto",
) -> Path:
    seed = int(config.get("seed", 2026))
    set_seed(seed, bool(config.get("deterministic", True)))
    device = choose_device(device_name)
    output = _output_dir(config, output_override)
    train_data, validation_data = build_training_datasets(config)

    data_config = config["data"]
    train_loader = make_loader(
        train_data,
        batch_size=int(data_config.get("batch_size", 512)),
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 0)),
        seed=seed,
    )
    if str(data_config.get("validation_source", "test")) == "test":
        validation_data = limited_dataset(
            validation_data, int(data_config.get("validation_samples", 0)) or None
        )
    validation_loader = make_loader(
        validation_data,
        batch_size=int(data_config.get("evaluation_batch_size", 256)),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        seed=seed,
    )

    model = build_model(config).to(device)
    training_channel = build_channel(config, training=True).to(device)
    validation_channel = build_channel(config, training=False).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    start_epoch = 0
    resume = training.get("resume")
    if resume:
        resume_path = resolve_relative(config, resume)
        try:
            state = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"])

    epochs = int(training.get("epochs", 1000))
    loss_name = _loss_name(config)
    validation_snr = float(training.get("validation_snr_db", 10.0))
    log_rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_metric_sum = 0.0
        train_count = 0
        for batch in train_loader:
            inputs, labels = unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)
            snr_db = _sample_train_snr(training.get("snr_db", 10.0), device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs, training_channel, snr_db)
            losses = _loss_per_sample(config, inputs, labels, predictions)
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * inputs.shape[0]
            train_metric_sum += float(
                _primary_metric_per_sample(config, inputs, labels, predictions)
                .detach()
                .sum()
            )
            train_count += inputs.shape[0]

        model.eval()
        validation_loss_sum = 0.0
        validation_metric_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                inputs, labels = unpack_batch(batch)
                inputs = inputs.to(device, non_blocking=True)
                if labels is not None:
                    labels = labels.to(device, non_blocking=True)
                predictions = model(inputs, validation_channel, validation_snr)
                losses = _loss_per_sample(config, inputs, labels, predictions)
                validation_loss_sum += float(losses.sum())
                validation_metric_sum += float(
                    _primary_metric_per_sample(config, inputs, labels, predictions).sum()
                )
                validation_count += inputs.shape[0]
        train_loss = train_loss_sum / max(train_count, 1)
        validation_loss = validation_loss_sum / max(validation_count, 1)
        train_metric = train_metric_sum / max(train_count, 1)
        validation_metric = validation_metric_sum / max(validation_count, 1)
        row = {
            "epoch": epoch,
            "loss_name": loss_name,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_snr_db": validation_snr,
            "elapsed_seconds": time.time() - start_time,
        }
        metric_name = _primary_metric_name(config)
        row[f"train_{metric_name}"] = train_metric
        row[f"validation_{metric_name}"] = validation_metric
        log_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        payload = _checkpoint_payload(model, optimizer, epoch, config)
        torch.save(payload, output / "checkpoint_last.pt")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(payload, output / "checkpoint_best.pt")
        checkpoint_interval = int(training.get("checkpoint_interval", 100))
        if checkpoint_interval and epoch % checkpoint_interval == 0:
            torch.save(payload, output / f"checkpoint_epoch_{epoch:04d}.pt")
        _write_csv(output / "training_log.csv", log_rows)

    _write_json(
        output / "run_manifest.json",
        {
            "scope": "cifar10-image-only training",
            "objective": _objective(config),
            "device": str(device),
            "seed": seed,
            "epochs": epochs,
            "loss_name": loss_name,
            "selection_metric": f"validation_{loss_name}",
            "best_validation_loss": best_loss,
            f"best_validation_{loss_name}": best_loss,
            "training_channel_noise": bool(training.get("channel_noise", True)),
            "validation_source": str(data_config.get("validation_source", "test")),
            "config": without_runtime_fields(config),
        },
    )
    return output


def _require_labels(labels: Tensor | None) -> Tensor:
    if labels is None:
        raise ValueError("Classification objective requires class labels in every batch.")
    return labels


def _distortion(config: dict[str, Any], target: Tensor, predictions: Tensor) -> Tensor:
    if _objective(config) == "classification":
        return classification_failure_score(target.long(), predictions)
    return distortion_per_sample(config["task"], target, predictions)


def _quality(config: dict[str, Any], target: Tensor, predictions: Tensor) -> Tensor:
    if _objective(config) == "classification":
        return accuracy_per_sample(target.long(), predictions)
    return psnr(target, predictions)


def _loss_name(config: dict[str, Any]) -> str:
    default = "cross_entropy" if _objective(config) == "classification" else "mse"
    name = str(config.get("training", {}).get("loss", default)).lower()
    expected = {"classification": "cross_entropy", "reconstruction": "mse"}[
        _objective(config)
    ]
    if name != expected:
        raise ValueError(f"{_objective(config)} requires training.loss='{expected}'.")
    return name


def _loss_per_sample(
    config: dict[str, Any], inputs: Tensor, labels: Tensor | None, predictions: Tensor
) -> Tensor:
    _loss_name(config)
    if _objective(config) == "classification":
        return torch.nn.functional.cross_entropy(
            predictions, _require_labels(labels).long(), reduction="none"
        )
    return mse_per_sample(inputs, predictions)


def _primary_metric_name(config: dict[str, Any]) -> str:
    return "accuracy" if _objective(config) == "classification" else "mse"


def _primary_metric_per_sample(
    config: dict[str, Any], inputs: Tensor, labels: Tensor | None, predictions: Tensor
) -> Tensor:
    if _objective(config) == "classification":
        return accuracy_per_sample(_require_labels(labels).long(), predictions)
    return mse_per_sample(inputs, predictions)


def _evaluate_dataset(
    model: SemanticModel,
    channel: nn.Module,
    loader: Iterable[Any],
    config: dict[str, Any],
    snr_db: float,
    device: torch.device,
) -> dict[str, float]:
    distortions: list[Tensor] = []
    qualities: list[Tensor] = []
    losses: list[Tensor] = []
    margins: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)
            predictions = model(inputs, channel, snr_db)
            target = _require_labels(labels) if _objective(config) == "classification" else inputs
            distortions.append(_distortion(config, target, predictions).cpu())
            qualities.append(_quality(config, target, predictions).cpu())
            losses.append(_loss_per_sample(config, inputs, labels, predictions).cpu())
            if _objective(config) == "classification":
                margins.append(classification_margin(target.long(), predictions).cpu())
    all_distortions = torch.cat(distortions)
    all_qualities = torch.cat(qualities)
    all_losses = torch.cat(losses)
    result = {
        "samples": int(all_distortions.numel()),
        "mean_distortion": float(all_distortions.mean()),
        "mean_loss": float(all_losses.mean()),
    }
    if _objective(config) == "classification":
        result.update(
            {
                "mean_cross_entropy": float(all_losses.mean()),
                "mean_accuracy": float(all_qualities.mean()),
                "std_accuracy": float(all_qualities.std(unbiased=False)),
                "mean_logit_margin": float(torch.cat(margins).mean()),
            }
        )
    else:
        result.update(
            {
                "mean_mse": float(all_losses.mean()),
                f"mean_{quality_name(config['task'])}": float(all_qualities.mean()),
                f"std_{quality_name(config['task'])}": float(
                    all_qualities.std(unbiased=False)
                ),
            }
        )
    return result


def evaluate_clean(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    output_override: str | Path | None = None,
    device_name: str = "auto",
) -> Path:
    seed = int(config.get("seed", 2026))
    set_seed(seed, bool(config.get("deterministic", True)))
    device = choose_device(device_name)
    model, _ = load_checkpoint(config, checkpoint_path, device)
    dataset = build_dataset(config, "test")
    evaluation = config["evaluation"]
    dataset = limited_dataset(dataset, int(evaluation.get("max_samples", 0)) or None)
    loader = make_loader(
        dataset,
        batch_size=int(config["data"].get("evaluation_batch_size", 256)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        seed=seed,
    )
    channel = build_channel(config, training=False).to(device)
    rows: list[dict[str, Any]] = []
    repeats = int(evaluation.get("channel_repeats", 1))
    for snr_db in map(float, evaluation["snr_db"]):
        repeat_rows = []
        for repeat in range(repeats):
            set_seed(seed + repeat + int(round(10 * snr_db)))
            repeat_rows.append(_evaluate_dataset(model, channel, loader, config, snr_db, device))
        key = "mean_accuracy" if _objective(config) == "classification" else f"mean_{quality_name(config['task'])}"
        row = {
            "snr_db": snr_db,
            "samples": repeat_rows[0]["samples"],
            "channel_repeats": repeats,
            "mean_distortion": sum(item["mean_distortion"] for item in repeat_rows) / repeats,
            "mean_loss": sum(item["mean_loss"] for item in repeat_rows) / repeats,
            key: sum(item[key] for item in repeat_rows) / repeats,
            f"repeat_std_{key.removeprefix('mean_')}": float(
                np.std([item[key] for item in repeat_rows])
            ),
        }
        if _objective(config) == "classification":
            row["mean_cross_entropy"] = sum(item["mean_cross_entropy"] for item in repeat_rows) / repeats
            row["mean_logit_margin"] = sum(item["mean_logit_margin"] for item in repeat_rows) / repeats
        else:
            row["mean_mse"] = sum(item["mean_mse"] for item in repeat_rows) / repeats
        rows.append(row)
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    output = _output_dir(config, output_override)
    path = output / "clean_metrics.csv"
    _write_csv(path, rows)
    _write_json(
        output / "clean_manifest.json",
        {
            "scope": "cifar10-image-only clean evaluation",
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "task": config["task"],
            "objective": _objective(config),
            "rows": rows,
        },
    )
    return path


def _attack_instance(name: str, config: dict[str, Any]) -> Any:
    attack = config["attacks"][name]
    if name == "pga":
        return ProgressiveGradientAscent(
            step_size=float(attack.get("step_size", 0.1)),
            max_steps=int(attack.get("max_steps", 2000)),
            eps=float(attack.get("eps", 1e-8)),
            refine_steps=int(attack.get("refine_steps", 0)),
        )
    if name == "cw":
        return CWRegressionAttack(
            learning_rate=float(attack.get("learning_rate", 0.01)),
            initial_c=float(attack.get("initial_c", 1.0)),
            c_min=float(attack.get("c_min", 1e-6)),
            c_max=float(attack.get("c_max", 100.0)),
            binary_search_steps=int(attack.get("binary_search_steps", 9)),
            max_steps=int(attack.get("max_steps", 2000)),
            kappa=float(attack.get("kappa", 0.0)),
            early_stop_on_success=bool(attack.get("early_stop_on_success", True)),
        )
    raise ValueError(f"Unsupported semantic attack: {name}")


def evaluate_attacks(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    attack_names: list[str],
    *,
    output_override: str | Path | None = None,
    device_name: str = "auto",
) -> tuple[Path, Path]:
    seed = int(config.get("seed", 2026))
    set_seed(seed, bool(config.get("deterministic", True)))
    device = choose_device(device_name)
    model, _ = load_checkpoint(config, checkpoint_path, device)
    dataset = build_dataset(config, "test")
    evaluation = config["evaluation"]
    max_samples = int(config.get("attacks", {}).get("max_samples", evaluation.get("max_samples", 0)))
    dataset = limited_dataset(dataset, max_samples or None)
    channel = build_channel(config, training=False).to(device)
    task = config["task"]
    objective = _objective(config)
    target_quality = (
        None
        if objective == "classification"
        else float(config["attacks"].get("target_quality_db", 15.0))
    )
    threshold = 0.0 if objective == "classification" else target_distortion(task, target_quality)
    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for attack_name in attack_names:
        attack = _attack_instance(attack_name, config)
        batch_size = int(config["attacks"][attack_name].get("batch_size", 32 if attack_name == "pga" else 1))
        loader = make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config["data"].get("num_workers", 0)),
            seed=seed,
        )
        for snr_db in map(float, evaluation["snr_db"]):
            set_seed(seed + int(round(10 * snr_db)))
            current_rows: list[dict[str, Any]] = []
            for batch in loader:
                inputs, labels = unpack_batch(batch)
                inputs = inputs.to(device, non_blocking=True)
                if labels is not None:
                    labels = labels.to(device, non_blocking=True)
                target = _require_labels(labels) if objective == "classification" else inputs
                with torch.no_grad():
                    symbols = model.encode(inputs)
                    received = channel(symbols, snr_db)
                    clean_predictions = model.decode(received)
                    clean_distortion = _distortion(config, target, clean_predictions)
                    clean_quality = _quality(config, target, clean_predictions)
                result = attack(
                    model.decoder,
                    target,
                    received,
                    lambda expected, predictions: _distortion(config, expected, predictions),
                    threshold,
                )
                adversarial_quality = _quality(config, target, result.reconstruction)
                for local_index in range(inputs.shape[0]):
                    row = {
                        "attack": attack_name,
                        "objective_variant": result.objective_variant,
                        "task_objective": objective,
                        "snr_db": snr_db,
                        "sample_index": len(current_rows),
                        "target_distortion": threshold,
                        "clean_distortion": float(clean_distortion[local_index]),
                        "attacked_distortion": float(result.distortion[local_index]),
                        "success": int(result.success[local_index]),
                        "steps": int(result.steps[local_index]),
                        "attack_power_total_l2_sq": float(result.total_power[local_index]),
                        "attack_power_per_channel_use": float(
                            result.power_per_channel_use[local_index]
                        ),
                    }
                    if objective == "classification":
                        row.update(
                            {
                                "class_label": int(target[local_index]),
                                "target_decision_margin": 0.0,
                                "clean_correct": int(clean_quality[local_index]),
                                "clean_logit_margin": float(-clean_distortion[local_index]),
                                "attacked_correct": int(adversarial_quality[local_index]),
                                "attacked_logit_margin": float(-result.distortion[local_index]),
                            }
                        )
                    else:
                        row.update(
                            {
                                "target_quality_db": target_quality,
                                f"clean_{quality_name(task)}": float(clean_quality[local_index]),
                                f"attacked_{quality_name(task)}": float(adversarial_quality[local_index]),
                            }
                        )
                    current_rows.append(row)
                    sample_rows.append(row)
            eligible = (
                [row for row in current_rows if row["clean_correct"]]
                if objective == "classification"
                else current_rows
            )
            successes = [row for row in eligible if row["success"]]
            mean_success_power = (
                sum(row["attack_power_total_l2_sq"] for row in successes) / len(successes)
                if successes
                else None
            )
            summary = {
                "attack": attack_name,
                "snr_db": snr_db,
                "samples": len(eligible),
                "total_samples": len(current_rows),
                "successes": len(successes),
                "success_rate": len(successes) / max(len(eligible), 1),
                "mean_attack_power_total_l2_sq_successes": mean_success_power,
                "mean_attack_power_total_l2_sq_all": (
                    mean_success_power if len(successes) == len(eligible) else None
                ),
                "paper_power_convention": "total_l2_squared",
            }
            summary_rows.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    output = _output_dir(config, output_override)
    sample_path = output / "attack_samples.csv"
    summary_path = output / "attack_summary.csv"
    _write_csv(sample_path, sample_rows)
    _write_csv(summary_path, summary_rows)
    _write_json(
        output / "attack_manifest.json",
        {
            "scope": "cifar10-image-only adversarial evaluation",
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "attacks": attack_names,
            "task": task,
            "objective": objective,
            "target_quality_db": target_quality,
            "target_distortion": threshold,
            "classification_success_rule": (
                "max_other_logit - true_logit >= 0, reported among clean-correct samples"
                if objective == "classification"
                else None
            ),
            "power_fields": {
                "paper_default": "sum_i s_i^2",
                "diagnostic": "mean_i s_i^2",
            },
            "cw_note": "Uses corrected hinge relu(D* + kappa - D); Eq. (38) has the opposite sign.",
        },
    )
    return sample_path, summary_path


def plot_results(
    config: dict[str, Any],
    clean_csv: str | Path | None,
    attack_csv: str | Path | None,
    output_path: str | Path,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib.") from exc
    panels = int(clean_csv is not None) + int(attack_csv is not None)
    if panels == 0:
        raise ValueError("At least one of clean_csv or attack_csv is required.")
    figure, axes = plt.subplots(1, panels, figsize=(6 * panels, 4.5), squeeze=False)
    axis_index = 0
    if clean_csv is not None:
        with Path(clean_csv).open("r", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        axis = axes[0, axis_index]
        classification = _objective(config) == "classification"
        key = "mean_accuracy" if classification else f"mean_{quality_name(config['task'])}"
        axis.plot([float(row["snr_db"]) for row in rows], [float(row[key]) for row in rows], "*-", label="DeepJSCC")
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("Accuracy" if classification else "PSNR (dB)")
        axis.set_title(f"{_objective(config).title()} clean performance")
        axis.grid(True, alpha=0.3)
        axis.legend()
        axis_index += 1
    if attack_csv is not None:
        with Path(attack_csv).open("r", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        axis = axes[0, axis_index]
        names = sorted({row["attack"] for row in rows})
        for name in names:
            selected = [row for row in rows if row["attack"] == name]
            x_values = [float(row["snr_db"]) for row in selected]
            y_values = [
                float(row["mean_attack_power_total_l2_sq_successes"])
                if row["mean_attack_power_total_l2_sq_successes"] not in {"", "None"}
                else math.nan
                for row in selected
            ]
            axis.plot(x_values, y_values, "*-", label=name.upper())
        axis.set_yscale("log")
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("rho* (total squared L2 power)")
        axis.set_title("Semantic minimum attack power")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.tight_layout()
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
