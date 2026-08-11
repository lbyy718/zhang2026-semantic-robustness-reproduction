"""CIFAR-10 and COST2100-derived CSI data interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


class TensorOnlyDataset(Dataset[Tensor]):
    def __init__(self, values: Tensor) -> None:
        if values.ndim != 4:
            raise ValueError("Dataset values must be [samples, channels, height, width].")
        self.values = values.float()

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return self.values[index]


def _load_array(path: Path, key: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if suffix == ".npz":
        archive = np.load(path, mmap_mode="r")
        selected = key or (archive.files[0] if len(archive.files) == 1 else None)
        if selected is None or selected not in archive:
            raise KeyError(f"Choose one of the NPZ keys: {archive.files}")
        return archive[selected]
    if suffix == ".mat":
        try:
            from scipy.io import loadmat

            values = loadmat(path)
            if key is None:
                candidates = [name for name in values if not name.startswith("__")]
                if len(candidates) != 1:
                    raise KeyError(f"Choose one MAT key from: {candidates}")
                key = candidates[0]
            return values[key]
        except NotImplementedError:
            pass
        except ImportError as exc:
            raise ImportError("Reading MAT files requires scipy or h5py.") from exc
        try:
            import h5py

            with h5py.File(path, "r") as stream:
                if key is None:
                    candidates = list(stream.keys())
                    if len(candidates) != 1:
                        raise KeyError(f"Choose one HDF5 key from: {candidates}")
                    key = candidates[0]
                return np.asarray(stream[key])
        except ImportError as exc:
            raise ImportError("MAT v7.3 files require h5py.") from exc
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Reading HDF5 files requires h5py.") from exc
        with h5py.File(path, "r") as stream:
            if key is None:
                candidates = list(stream.keys())
                if len(candidates) != 1:
                    raise KeyError(f"Choose one HDF5 key from: {candidates}")
                key = candidates[0]
            return np.asarray(stream[key])
    raise ValueError(f"Unsupported data format: {suffix}")


def _select_complex_representation(values: np.ndarray, representation: str) -> np.ndarray:
    if not np.iscomplexobj(values):
        return values
    if representation == "magnitude":
        return np.abs(values)
    if representation == "real":
        return values.real
    if representation == "imaginary":
        return values.imag
    if representation == "real_imag":
        return np.stack((values.real, values.imag), axis=1)
    raise ValueError(
        "Complex CSI representation must be magnitude, real, imaginary, or real_imag."
    )


def prepare_cost2100_array(
    values: np.ndarray,
    *,
    delay_taps: int = 32,
    antennas: int = 32,
    representation: str = "magnitude",
    already_angular_delay: bool = False,
) -> np.ndarray:
    """Convert raw frequency-spatial CSI to the paper's 32x32 input matrices.

    For raw H[sample, subcarrier, antenna], Eq. (40) is inverted with unitary
    FFTs to obtain the angular-delay representation, after which the first 32
    delay taps are retained.  Preprocessed 32x32 arrays can skip the transform.
    """
    array = np.asarray(values)
    if array.ndim == 2:
        if not already_angular_delay:
            raise ValueError(
                "Flattened COST2100 arrays are already in angular-delay form; "
                "set already_angular_delay=true."
            )
        features = array.shape[1]
        spatial_features = delay_taps * antennas
        if features == 2 * spatial_features:
            # The CsiNet authors distribute HT as [real, imaginary], with each
            # component shifted by +0.5 and flattened from 2x32x32.  The Zhang
            # paper uses a single-channel 32x32 input and source dimension 1024,
            # so the magnitude representation restores the complex coefficients
            # and reduces them to the paper's stated one-channel input.
            channels = array.reshape(array.shape[0], 2, delay_taps, antennas)
            output_shape = (
                (array.shape[0], 2, delay_taps, antennas)
                if representation == "real_imag"
                else (array.shape[0], delay_taps, antennas)
            )
            output = np.empty(output_shape, dtype=np.float32)
            chunk_size = 4096
            for start in range(0, array.shape[0], chunk_size):
                stop = min(start + chunk_size, array.shape[0])
                if representation == "magnitude":
                    real = channels[start:stop, 0].astype(np.float32) - 0.5
                    imaginary = channels[start:stop, 1].astype(np.float32) - 0.5
                    np.hypot(real, imaginary, out=output[start:stop])
                elif representation == "real":
                    output[start:stop] = channels[start:stop, 0]
                elif representation == "imaginary":
                    output[start:stop] = channels[start:stop, 1]
                elif representation == "real_imag":
                    # Keep the CsiNet authors' [0,1] representation.  Its
                    # physical zero is 0.5 and is accounted for by
                    # data.metric_center during NMSE evaluation.
                    output[start:stop] = channels[start:stop].astype(np.float32)
                else:
                    raise ValueError(
                        "CSI representation must be magnitude, real, imaginary, "
                        "or real_imag."
                    )
            return output
        if features == spatial_features:
            return np.asarray(
                array.reshape(array.shape[0], delay_taps, antennas), dtype=np.float32
            )
        raise ValueError(
            f"Flattened CSI has {features} features; expected {spatial_features} "
            f"or {2 * spatial_features}."
        )
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError("Expected CSI shaped [samples, subcarriers/delay, antennas].")
    if already_angular_delay:
        transformed = array
    else:
        transformed = np.fft.ifft(array, axis=1, norm="ortho")
        transformed = np.fft.fft(transformed, axis=2, norm="ortho")
    transformed = transformed[:, :delay_taps, :antennas]
    if transformed.shape[1:] != (delay_taps, antennas):
        raise ValueError(
            f"Not enough CSI dimensions for {(delay_taps, antennas)}: {transformed.shape}."
        )
    selected = _select_complex_representation(transformed, representation)
    return np.asarray(selected, dtype=np.float32)


def normalize_csi(
    values: np.ndarray,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[np.ndarray, float, float]:
    minimum = float(np.min(values)) if minimum is None else float(minimum)
    maximum = float(np.max(values)) if maximum is None else float(maximum)
    if maximum <= minimum:
        raise ValueError("CSI normalization maximum must exceed minimum.")
    normalized = (values - minimum) / (maximum - minimum)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32), minimum, maximum


def cifar10_datasets(root: str | Path, download: bool = False) -> tuple[Dataset[Any], Dataset[Any]]:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError("CIFAR-10 loading requires torchvision.") from exc
    transform = transforms.ToTensor()
    train = datasets.CIFAR10(root=str(root), train=True, transform=transform, download=download)
    test = datasets.CIFAR10(root=str(root), train=False, transform=transform, download=download)
    return train, test


def csi_dataset(
    path: str | Path,
    *,
    key: str | None,
    representation: str,
    already_angular_delay: bool,
    normalization_min: float | None,
    normalization_max: float | None,
) -> tuple[TensorOnlyDataset, dict[str, float]]:
    raw = _load_array(Path(path), key)
    prepared = prepare_cost2100_array(
        raw,
        representation=representation,
        already_angular_delay=already_angular_delay,
    )
    normalized, minimum, maximum = normalize_csi(
        prepared, minimum=normalization_min, maximum=normalization_max
    )
    normalized_array = np.asarray(normalized)
    if normalized_array.ndim == 3:
        normalized_array = normalized_array[:, None, :, :]
    elif normalized_array.ndim != 4:
        raise ValueError(
            "Prepared CSI must be [samples,height,width] or "
            "[samples,channels,height,width]."
        )
    tensor = torch.from_numpy(normalized_array)
    return TensorOnlyDataset(tensor), {"minimum": minimum, "maximum": maximum}


def unwrap_batch(batch: Any) -> Tensor:
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], Tensor):
        return batch[0]
    raise TypeError(f"Cannot extract input tensor from batch type {type(batch)!r}.")


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
