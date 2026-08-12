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
    model = config.get("model", {})
    data = config.get("data", {})
    channels = int(model.get("in_channels", 0))
    multiplier = int(model.get("channel_multiplier", 0))
    spatial = tuple(model.get("spatial_size", []))
    if channels < 1 or multiplier < 1 or len(spatial) != 2:
        raise ValueError("model requires positive in_channels/channel_multiplier and spatial_size.")
    if spatial[0] % 4 or spatial[1] % 4:
        raise ValueError("model.spatial_size must be divisible by four.")
    channel_uses = 2 * multiplier * (spatial[0] // 4) * (spatial[1] // 4)
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
