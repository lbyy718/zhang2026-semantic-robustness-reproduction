"""JSON configuration loading and CIFAR-10 experiment validation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    config["_config_path"] = str(config_path.resolve())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    task = config.get("task")
    if task != "image":
        raise ValueError("This minimal repository only supports config.task='image'.")
    objective = str(config.get("objective", "reconstruction")).lower()
    if objective not in {"reconstruction", "classification"}:
        raise ValueError("config.objective must be 'reconstruction' or 'classification'.")
    model = config.get("model", {})
    data = config.get("data", {})
    architecture = str(model.get("architecture", "small")).lower()
    if architecture not in {"small", "resnet18_bottleneck"}:
        raise ValueError(
            "model.architecture must be 'small' or 'resnet18_bottleneck'."
        )
    channels = int(model.get("in_channels", 0))
    spatial = tuple(model.get("spatial_size", []))
    if channels < 1 or len(spatial) != 2:
        raise ValueError("model requires positive in_channels and spatial_size.")
    if spatial[0] % 4 or spatial[1] % 4:
        raise ValueError("model.spatial_size must be divisible by four.")
    if architecture == "small":
        multiplier = int(model.get("channel_multiplier", 0))
        if multiplier < 1:
            raise ValueError("small models require model.channel_multiplier >= 1.")
        channel_uses = 2 * multiplier * (spatial[0] // 4) * (spatial[1] // 4)
    else:
        if objective != "classification":
            raise ValueError(
                "resnet18_bottleneck is a classification-only architecture."
            )
        if channels != 3 or spatial != (32, 32):
            raise ValueError(
                "resnet18_bottleneck requires 3-channel 32x32 CIFAR-10 inputs."
            )
        channel_uses = int(model.get("latent_dim", 0))
        if int(model.get("feature_dim", 512)) != 512:
            raise ValueError("resnet18_bottleneck requires model.feature_dim=512.")
    source_dimension = channels * spatial[0] * spatial[1]
    expected_ratio = float(data.get("bandwidth_ratio", 0.25))
    actual_ratio = channel_uses / source_dimension
    if abs(actual_ratio - expected_ratio) > 1e-9:
        raise ValueError(
            f"Model implies bandwidth ratio {actual_ratio:g}, not configured {expected_ratio:g}."
        )
    if channel_uses != 768:
        raise ValueError(
            f"The paper's CIFAR-10 setup requires 768 channel uses, got {channel_uses}."
        )
    if objective == "classification":
        if int(model.get("num_classes", 10)) < 2:
            raise ValueError("classification requires model.num_classes >= 2.")
        if int(model.get("classifier_hidden", 53)) < 1:
            raise ValueError("classification requires model.classifier_hidden >= 1.")
    training = config.get("training", {})
    regularizer = str(training.get("regularizer", "none")).lower()
    if regularizer not in {"none", "latent_failure_jacobian"}:
        raise ValueError(
            "training.regularizer must be 'none' or 'latent_failure_jacobian'."
        )
    if regularizer == "latent_failure_jacobian":
        if objective != "classification":
            raise ValueError("latent_failure_jacobian requires classification.")
        if bool(training.get("channel_noise", True)):
            raise ValueError(
                "The CSJ control must set training.channel_noise=false."
            )
        target_ratio = float(training.get("jacobian_target_ratio", 0.1))
        if not 0.0 < target_ratio < 1.0:
            raise ValueError("training.jacobian_target_ratio must be in (0, 1).")
    optimizer = str(training.get("optimizer", "adamw")).lower()
    if optimizer not in {"adamw", "sgd"}:
        raise ValueError("training.optimizer must be 'adamw' or 'sgd'.")
    scheduler = str(training.get("scheduler", "none")).lower()
    if scheduler not in {"none", "cosine"}:
        raise ValueError("training.scheduler must be 'none' or 'cosine'.")
    augmentation = str(data.get("augmentation", "none")).lower()
    if augmentation not in {"none", "cifar_standard"}:
        raise ValueError("data.augmentation must be 'none' or 'cifar_standard'.")
    validation_source = str(data.get("validation_source", "test"))
    if validation_source not in {"test", "train_holdout"}:
        raise ValueError("data.validation_source must be 'test' or 'train_holdout'.")


def without_runtime_fields(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = deepcopy(config)
    for key in list(cleaned):
        if key.startswith("_"):
            cleaned.pop(key)
    return cleaned


def resolve_relative(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_path = Path(config.get("_config_path", Path.cwd() / "config.json"))
    return (config_path.parent / path).resolve()
