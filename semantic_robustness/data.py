"""CIFAR-10 data loading helpers for image transmission experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


def cifar10_datasets(
    root: str | Path, download: bool = False
) -> tuple[Dataset[Any], Dataset[Any]]:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError("CIFAR-10 loading requires torchvision.") from exc

    transform = transforms.ToTensor()
    train = datasets.CIFAR10(
        root=str(root), train=True, transform=transform, download=download
    )
    test = datasets.CIFAR10(
        root=str(root), train=False, transform=transform, download=download
    )
    return train, test


def unwrap_batch(batch: Any) -> Tensor:
    return unpack_batch(batch)[0]


def unpack_batch(batch: Any) -> tuple[Tensor, Tensor | None]:
    """Return inputs and optional labels without breaking reconstruction callers."""
    if isinstance(batch, Tensor):
        return batch, None
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], Tensor):
        labels = batch[1] if len(batch) > 1 and isinstance(batch[1], Tensor) else None
        return batch[0], labels
    raise TypeError(f"Cannot extract tensors from batch type {type(batch)!r}.")


def limited_dataset(dataset: Dataset[Any], max_samples: int | None) -> Dataset[Any]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, range(max_samples))


def make_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
