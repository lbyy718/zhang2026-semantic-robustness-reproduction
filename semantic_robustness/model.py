"""DeepJSCC encoder/decoder reconstructed from Table I of the paper.

The paper specifies tensor sizes but omits kernel sizes and the internals of a
residual block.  This implementation uses 3x3 convolutions and a two-convolution
residual block.  Those choices are deliberately exposed as constructor options.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """A shape-preserving two-layer residual block."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Residual-block kernel_size must be odd.")
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.body(inputs))


class PowerNormalizer(nn.Module):
    """Normalize each codeword so that ||z||_2^2 equals its channel uses N."""

    def __init__(
        self,
        eps: float = 1e-8,
        zero_mean: bool = False,
        complex_symbols: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.zero_mean = zero_mean
        self.complex_symbols = complex_symbols

    def forward(self, symbols: Tensor) -> Tensor:
        flat = symbols.flatten(start_dim=1)
        if self.zero_mean:
            flat = flat - flat.mean(dim=1, keepdim=True)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)
        target_energy = flat.shape[1] / (2 if self.complex_symbols else 1)
        return flat * (math.sqrt(target_energy) / norm)


class GlobalResidualMixer(nn.Module):
    """Globally mix a flattened codeword while preserving a residual path."""

    def __init__(self, features: int, hidden_multiplier: int = 2) -> None:
        super().__init__()
        hidden = features * hidden_multiplier
        self.network = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, features),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.network(inputs)


class DeepJSCCEncoder(nn.Module):
    """Table-I encoder with two stride-2 convolutions."""

    def __init__(
        self,
        in_channels: int,
        channel_multiplier: int,
        spatial_size: tuple[int, int] = (32, 32),
        kernel_size: int = 3,
        residual_kernel_size: int = 3,
        zero_mean_symbols: bool = False,
        global_mixing: bool = False,
        complex_symbols: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Encoder kernel_size must be odd.")
        if spatial_size[0] % 4 or spatial_size[1] % 4:
            raise ValueError("Encoder spatial_size must be divisible by four.")
        latent_channels = 2 * channel_multiplier
        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size, stride=2, padding=padding),
            nn.ReLU(inplace=True),
            ResidualBlock(16, residual_kernel_size),
            nn.Conv2d(16, latent_channels, kernel_size, stride=2, padding=padding),
            nn.ReLU(inplace=True),
            ResidualBlock(latent_channels, residual_kernel_size),
        )
        features = latent_channels * (spatial_size[0] // 4) * (spatial_size[1] // 4)
        self.global_mixer = GlobalResidualMixer(features) if global_mixing else nn.Identity()
        self.normalizer = PowerNormalizer(
            zero_mean=zero_mean_symbols, complex_symbols=complex_symbols
        )

    def forward(self, inputs: Tensor) -> Tensor:
        latent = self.network(inputs).flatten(start_dim=1)
        return self.normalizer(self.global_mixer(latent))


class DeepJSCCDecoder(nn.Module):
    """Table-I decoder accepting flattened real channel symbols."""

    def __init__(
        self,
        out_channels: int,
        channel_multiplier: int,
        spatial_size: tuple[int, int] = (32, 32),
        kernel_size: int = 3,
        residual_kernel_size: int = 3,
        intermediate_sigmoid: bool = True,
        global_mixing: bool = False,
    ) -> None:
        super().__init__()
        height, width = spatial_size
        if height % 4 or width % 4:
            raise ValueError("spatial_size must be divisible by four.")
        if kernel_size % 2 == 0:
            raise ValueError("Decoder kernel_size must be odd.")
        self.latent_shape = (2 * channel_multiplier, height // 4, width // 4)
        self.global_mixer = (
            GlobalResidualMixer(math.prod(self.latent_shape)) if global_mixing else nn.Identity()
        )
        padding = kernel_size // 2
        output_padding = 1

        # Deconv1 and Deconv2 preserve H/4 x W/4, as specified in Table I.
        # The intermediate sigmoid after Deconv2 is unusual but is retained by
        # default to match the published architecture description.  It can be
        # disabled for the explicitly configured structure ablation.
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(self.latent_shape[0], 32, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size, padding=padding),
        ]
        if intermediate_sigmoid:
            layers.append(nn.Sigmoid())
        layers.extend([
            ResidualBlock(32, residual_kernel_size),
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
            ),
            ResidualBlock(16, residual_kernel_size),
            nn.ConvTranspose2d(
                16,
                out_channels,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.Sigmoid(),
        ])
        self.network = nn.Sequential(*layers)

    @property
    def channel_uses(self) -> int:
        return math.prod(self.latent_shape)

    def forward(self, received: Tensor) -> Tensor:
        if received.ndim == 2:
            expected = self.channel_uses
            if received.shape[1] != expected:
                raise ValueError(
                    f"Expected {expected} flattened channel symbols, "
                    f"received {received.shape[1]}."
                )
            received = self.global_mixer(received)
            received = received.reshape(received.shape[0], *self.latent_shape)
        elif tuple(received.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"Expected latent shape {self.latent_shape}, got {tuple(received.shape[1:])}."
            )
        elif not isinstance(self.global_mixer, nn.Identity):
            received = self.global_mixer(received.flatten(start_dim=1)).reshape(
                received.shape[0], *self.latent_shape
            )
        return self.network(received)


class DeepJSCC(nn.Module):
    """End-to-end DeepJSCC model for image or one-channel CSI transmission."""

    def __init__(
        self,
        in_channels: int,
        channel_multiplier: int,
        spatial_size: tuple[int, int] = (32, 32),
        kernel_size: int = 3,
        residual_kernel_size: int = 3,
        intermediate_sigmoid: bool = True,
        zero_mean_symbols: bool = False,
        global_mixing: bool = False,
        complex_symbols: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.complex_symbols = complex_symbols
        self.encoder = DeepJSCCEncoder(
            in_channels,
            channel_multiplier,
            spatial_size=spatial_size,
            kernel_size=kernel_size,
            residual_kernel_size=residual_kernel_size,
            zero_mean_symbols=zero_mean_symbols,
            global_mixing=global_mixing,
            complex_symbols=complex_symbols,
        )
        self.decoder = DeepJSCCDecoder(
            in_channels,
            channel_multiplier,
            spatial_size=spatial_size,
            kernel_size=kernel_size,
            residual_kernel_size=residual_kernel_size,
            intermediate_sigmoid=intermediate_sigmoid,
            global_mixing=global_mixing,
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def channel_uses(self) -> int:
        real_dimensions = self.decoder.channel_uses
        return real_dimensions // 2 if self.complex_symbols else real_dimensions

    @property
    def real_channel_dimensions(self) -> int:
        return self.decoder.channel_uses

    @property
    def source_dimension(self) -> int:
        return self.in_channels * math.prod(self.spatial_size)

    @property
    def bandwidth_ratio(self) -> float:
        return self.channel_uses / self.source_dimension

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, received: Tensor) -> Tensor:
        return self.decoder(received)

    def forward(self, inputs: Tensor, channel: nn.Module, snr_db: float | Tensor) -> Tensor:
        symbols = self.encode(inputs)
        received = channel(symbols, snr_db)
        return self.decode(received)
