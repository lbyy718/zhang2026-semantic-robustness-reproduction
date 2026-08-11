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
from torch.utils.data import Dataset

from .attacks import CWRegressionAttack, ProgressiveGradientAscent
from .channel import AWGNChannel
from .config import resolve_relative, without_runtime_fields
from .data import (
    cifar10_datasets,
    csi_dataset,
    limited_dataset,
    make_loader,
    unwrap_batch,
)
from .metrics import (
    distortion_per_sample,
    mse_per_sample,
    nmse_db,
    psnr,
    quality_name,
    target_distortion,
)
from .model import DeepJSCC


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


def build_model(config: dict[str, Any]) -> DeepJSCC:
    model = config["model"]
    return DeepJSCC(
        in_channels=int(model["in_channels"]),
        channel_multiplier=int(model["channel_multiplier"]),
        spatial_size=tuple(model.get("spatial_size", [32, 32])),
        kernel_size=int(model.get("kernel_size", 3)),
        residual_kernel_size=int(model.get("residual_kernel_size", 3)),
        intermediate_sigmoid=bool(model.get("intermediate_sigmoid", True)),
        zero_mean_symbols=bool(model.get("zero_mean_symbols", False)),
        global_mixing=bool(model.get("global_mixing", False)),
        complex_symbols=bool(model.get("complex_symbols", False)),
    )


def _csi_split(config: dict[str, Any], split: str) -> tuple[Path, str | None]:
    data = config["data"]
    path_value = data.get(f"{split}_path", data.get("path"))
    if not path_value:
        raise ValueError(f"CSI configuration is missing data.{split}_path or data.path.")
    key = data.get(f"{split}_key", split if str(path_value).lower().endswith(".npz") else None)
    return resolve_relative(config, path_value), key


def build_dataset(
    config: dict[str, Any],
    split: str,
    *,
    csi_normalization: dict[str, float] | None = None,
) -> tuple[Dataset[Any], dict[str, float] | None]:
    data = config["data"]
    if config["task"] == "image":
        root = resolve_relative(config, data.get("root", "../data/cifar10"))
        train, test = cifar10_datasets(root, download=bool(data.get("download", False)))
        return (train if split == "train" else test), None

    path, key = _csi_split(config, split)
    minimum = None if csi_normalization is None else csi_normalization["minimum"]
    maximum = None if csi_normalization is None else csi_normalization["maximum"]
    dataset, normalization = csi_dataset(
        path,
        key=key,
        representation=data.get("representation", "magnitude"),
        already_angular_delay=bool(data.get("already_angular_delay", True)),
        normalization_min=minimum,
        normalization_max=maximum,
    )
    return dataset, normalization


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
    model: DeepJSCC,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    csi_normalization: dict[str, float] | None,
) -> dict[str, Any]:
    return {
        "paper": "Zhang et al. 2026, Unanticipated Adversarial Robustness of Semantic Communication",
        "scope": "semantic-only reproduction",
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": without_runtime_fields(config),
        "csi_normalization": csi_normalization,
    }


def load_checkpoint(
    config: dict[str, Any], checkpoint_path: str | Path, device: torch.device
) -> tuple[DeepJSCC, dict[str, Any]]:
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
    train_data, csi_normalization = build_dataset(config, "train")
    validation_split = "validation" if config["task"] == "csi" else "test"
    validation_data, _ = build_dataset(
        config, validation_split, csi_normalization=csi_normalization
    )

    data_config = config["data"]
    train_loader = make_loader(
        train_data,
        batch_size=int(data_config.get("batch_size", 512)),
        shuffle=True,
        num_workers=int(data_config.get("num_workers", 0)),
        seed=seed,
    )
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
    channel = AWGNChannel(float(config.get("channel", {}).get("fading_gain", 1.0))).to(device)
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
        train_mse_sum = 0.0
        train_count = 0
        for batch in train_loader:
            inputs = unwrap_batch(batch).to(device, non_blocking=True)
            snr_db = _sample_train_snr(training.get("snr_db", 10.0), device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(inputs, channel, snr_db)
            losses = _loss_per_sample(config, inputs, reconstruction)
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * inputs.shape[0]
            train_mse_sum += float(
                mse_per_sample(inputs, reconstruction).detach().sum()
            )
            train_count += inputs.shape[0]

        model.eval()
        validation_loss_sum = 0.0
        validation_mse_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                inputs = unwrap_batch(batch).to(device, non_blocking=True)
                reconstruction = model(inputs, channel, validation_snr)
                losses = _loss_per_sample(config, inputs, reconstruction)
                validation_loss_sum += float(losses.sum())
                validation_mse_sum += float(
                    mse_per_sample(inputs, reconstruction).sum()
                )
                validation_count += inputs.shape[0]
        train_loss = train_loss_sum / max(train_count, 1)
        validation_loss = validation_loss_sum / max(validation_count, 1)
        train_mse = train_mse_sum / max(train_count, 1)
        validation_mse = validation_mse_sum / max(validation_count, 1)
        row = {
            "epoch": epoch,
            "loss_name": loss_name,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "validation_snr_db": validation_snr,
            "elapsed_seconds": time.time() - start_time,
        }
        if loss_name == "nmse":
            row["train_nmse"] = train_loss
            row["validation_nmse"] = validation_loss
        log_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        payload = _checkpoint_payload(model, optimizer, epoch, config, csi_normalization)
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
            "scope": "semantic-only",
            "device": str(device),
            "seed": seed,
            "epochs": epochs,
            "loss_name": loss_name,
            "selection_metric": f"validation_{loss_name}",
            "best_validation_loss": best_loss,
            f"best_validation_{loss_name}": best_loss,
            "csi_normalization": csi_normalization,
            "config": without_runtime_fields(config),
        },
    )
    return output


def _metric_tensors(
    config: dict[str, Any], target: Tensor, reconstruction: Tensor
) -> tuple[Tensor, Tensor]:
    if config["task"] != "csi":
        return target, reconstruction
    center = float(config.get("data", {}).get("metric_center", 0.0))
    return target - center, reconstruction - center


def _distortion(config: dict[str, Any], target: Tensor, reconstruction: Tensor) -> Tensor:
    metric_target, metric_reconstruction = _metric_tensors(
        config, target, reconstruction
    )
    return distortion_per_sample(
        config["task"], metric_target, metric_reconstruction
    )


def _quality(config: dict[str, Any], target: Tensor, reconstruction: Tensor) -> Tensor:
    metric_target, metric_reconstruction = _metric_tensors(
        config, target, reconstruction
    )
    return (
        psnr(metric_target, metric_reconstruction)
        if config["task"] == "image"
        else nmse_db(metric_target, metric_reconstruction)
    )


def _loss_name(config: dict[str, Any]) -> str:
    name = str(config.get("training", {}).get("loss", "mse")).lower()
    if name not in {"mse", "nmse"}:
        raise ValueError("training.loss must be 'mse' or 'nmse'.")
    if name == "nmse" and config["task"] != "csi":
        raise ValueError("training.loss='nmse' is only supported for CSI.")
    return name


def _loss_per_sample(
    config: dict[str, Any], target: Tensor, reconstruction: Tensor
) -> Tensor:
    return (
        mse_per_sample(target, reconstruction)
        if _loss_name(config) == "mse"
        else _distortion(config, target, reconstruction)
    )


def _evaluate_dataset(
    model: DeepJSCC,
    channel: AWGNChannel,
    loader: Iterable[Any],
    config: dict[str, Any],
    snr_db: float,
    device: torch.device,
) -> dict[str, float]:
    distortions: list[Tensor] = []
    qualities: list[Tensor] = []
    mses: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            inputs = unwrap_batch(batch).to(device, non_blocking=True)
            reconstruction = model(inputs, channel, snr_db)
            distortions.append(_distortion(config, inputs, reconstruction).cpu())
            qualities.append(_quality(config, inputs, reconstruction).cpu())
            mses.append(mse_per_sample(inputs, reconstruction).cpu())
    all_distortions = torch.cat(distortions)
    all_qualities = torch.cat(qualities)
    all_mses = torch.cat(mses)
    return {
        "samples": int(all_distortions.numel()),
        "mean_distortion": float(all_distortions.mean()),
        "mean_mse": float(all_mses.mean()),
        f"mean_{quality_name(config['task'])}": float(all_qualities.mean()),
        f"std_{quality_name(config['task'])}": float(all_qualities.std(unbiased=False)),
    }


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
    model, checkpoint = load_checkpoint(config, checkpoint_path, device)
    dataset, _ = build_dataset(
        config, "test", csi_normalization=checkpoint.get("csi_normalization")
    )
    evaluation = config["evaluation"]
    dataset = limited_dataset(dataset, int(evaluation.get("max_samples", 0)) or None)
    loader = make_loader(
        dataset,
        batch_size=int(config["data"].get("evaluation_batch_size", 256)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        seed=seed,
    )
    channel = AWGNChannel(float(config.get("channel", {}).get("fading_gain", 1.0))).to(device)
    rows: list[dict[str, Any]] = []
    repeats = int(evaluation.get("channel_repeats", 1))
    for snr_db in map(float, evaluation["snr_db"]):
        repeat_rows = []
        for repeat in range(repeats):
            set_seed(seed + repeat + int(round(10 * snr_db)))
            repeat_rows.append(_evaluate_dataset(model, channel, loader, config, snr_db, device))
        key = f"mean_{quality_name(config['task'])}"
        rows.append(
            {
                "snr_db": snr_db,
                "samples": repeat_rows[0]["samples"],
                "channel_repeats": repeats,
                "mean_distortion": sum(row["mean_distortion"] for row in repeat_rows) / repeats,
                "mean_mse": sum(row["mean_mse"] for row in repeat_rows) / repeats,
                key: sum(row[key] for row in repeat_rows) / repeats,
                f"repeat_std_{quality_name(config['task'])}": float(
                    np.std([row[key] for row in repeat_rows])
                ),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    output = _output_dir(config, output_override)
    path = output / "clean_metrics.csv"
    _write_csv(path, rows)
    _write_json(
        output / "clean_manifest.json",
        {
            "scope": "semantic-only clean evaluation",
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "task": config["task"],
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
    model, checkpoint = load_checkpoint(config, checkpoint_path, device)
    dataset, _ = build_dataset(
        config, "test", csi_normalization=checkpoint.get("csi_normalization")
    )
    evaluation = config["evaluation"]
    max_samples = int(config.get("attacks", {}).get("max_samples", evaluation.get("max_samples", 0)))
    dataset = limited_dataset(dataset, max_samples or None)
    channel = AWGNChannel(float(config.get("channel", {}).get("fading_gain", 1.0))).to(device)
    task = config["task"]
    target_quality = float(config["attacks"].get("target_quality_db", 15.0 if task == "image" else -16.0))
    threshold = target_distortion(task, target_quality)
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
                inputs = unwrap_batch(batch).to(device, non_blocking=True)
                with torch.no_grad():
                    symbols = model.encode(inputs)
                    received = channel(symbols, snr_db)
                    clean_reconstruction = model.decode(received)
                    clean_distortion = _distortion(config, inputs, clean_reconstruction)
                    clean_quality = _quality(config, inputs, clean_reconstruction)
                result = attack(
                    model.decoder,
                    inputs,
                    received,
                    lambda target, reconstruction: _distortion(
                        config, target, reconstruction
                    ),
                    threshold,
                )
                adversarial_quality = _quality(config, inputs, result.reconstruction)
                for local_index in range(inputs.shape[0]):
                    row = {
                        "attack": attack_name,
                        "objective_variant": result.objective_variant,
                        "snr_db": snr_db,
                        "sample_index": len(current_rows),
                        "target_quality_db": target_quality,
                        "target_distortion": threshold,
                        "clean_distortion": float(clean_distortion[local_index]),
                        f"clean_{quality_name(task)}": float(clean_quality[local_index]),
                        "attacked_distortion": float(result.distortion[local_index]),
                        f"attacked_{quality_name(task)}": float(adversarial_quality[local_index]),
                        "success": int(result.success[local_index]),
                        "steps": int(result.steps[local_index]),
                        "attack_power_total_l2_sq": float(result.total_power[local_index]),
                        "attack_power_per_channel_use": float(
                            result.power_per_channel_use[local_index]
                        ),
                    }
                    current_rows.append(row)
                    sample_rows.append(row)
            successes = [row for row in current_rows if row["success"]]
            mean_success_power = (
                sum(row["attack_power_total_l2_sq"] for row in successes) / len(successes)
                if successes
                else None
            )
            summary = {
                "attack": attack_name,
                "snr_db": snr_db,
                "samples": len(current_rows),
                "successes": len(successes),
                "success_rate": len(successes) / max(len(current_rows), 1),
                "mean_attack_power_total_l2_sq_successes": mean_success_power,
                "mean_attack_power_total_l2_sq_all": (
                    mean_success_power if len(successes) == len(current_rows) else None
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
            "scope": "semantic-only adversarial evaluation",
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "attacks": attack_names,
            "task": task,
            "target_quality_db": target_quality,
            "target_distortion": threshold,
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
        key = f"mean_{quality_name(config['task'])}"
        axis.plot([float(row["snr_db"]) for row in rows], [float(row[key]) for row in rows], "*-", label="DeepJSCC")
        axis.set_xlabel("SNR η (dB)")
        axis.set_ylabel("PSNR (dB)" if config["task"] == "image" else "NMSE (dB)")
        axis.set_title("Semantic clean performance")
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
        axis.set_xlabel("SNR η (dB)")
        axis.set_ylabel("ρ* (total squared L2 power)")
        axis.set_title("Semantic minimum attack power")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.tight_layout()
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
